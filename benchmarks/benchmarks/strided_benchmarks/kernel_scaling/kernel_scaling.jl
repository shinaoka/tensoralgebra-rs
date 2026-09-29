# Julia counterpart of kernel_scaling.rs (strided-rs#269).
#
# Same case names and shapes as the Rust binary. Every variant writes into a
# preallocated output except `julia_alloc`, which times the allocating Base
# call (for example `reverse(A; dims=1)` or `vcat(A, B)`) because that is the
# idiomatic API. Each variant is checked against a plain loop before timing.
#
# Usage: JULIA_NUM_THREADS=N julia --project=. kernel_scaling.jl [--time]
# Environment: BENCH_RUNS, BENCH_WARMUP, KERNEL_SCALING_SHRINK, BENCH_FILTER
# (same meaning as for the Rust binary).

using Strided

const TIMING = "--time" in ARGS
const RUNS = parse(Int, get(ENV, "BENCH_RUNS", "11"))
const WARMUP = parse(Int, get(ENV, "BENCH_WARMUP", "2"))
const SHRINK = max(1, parse(Int, get(ENV, "KERNEL_SCALING_SHRINK", "1")))
const FILTER = get(ENV, "BENCH_FILTER", "")
const NT = Threads.nthreads()

requested = parse(Int, get(ENV, "JULIA_NUM_THREADS", string(NT)))
@assert NT == requested "Threads.nthreads()=$NT but JULIA_NUM_THREADS=$requested"
Strided.set_num_threads(NT)
@assert Strided.get_num_threads() == NT

d(x) = max(1, x ÷ SHRINK)
n1(x) = max(1, x ÷ (SHRINK * SHRINK))
enabled(case) = isempty(FILTER) || occursin(FILTER, case)

println(stderr, "# kernel_scaling.jl julia=$(VERSION) threads=$NT strided_threads=$(Strided.get_num_threads()) timing=$TIMING runs=$RUNS shrink=$SHRINK")
TIMING && println("case,variant,threads,median_ns,samples")

function measure(f::F, case, variant) where {F}
    if !TIMING
        f()
        return
    end
    for _ in 1:WARMUP
        f()
    end
    GC.gc()
    ts = Vector{UInt64}(undef, RUNS)
    for k in 1:RUNS
        t0 = time_ns()
        f()
        ts[k] = time_ns() - t0
    end
    sort!(ts)
    med = isodd(RUNS) ? ts[(RUNS + 1) ÷ 2] : (ts[RUNS ÷ 2] + ts[RUNS ÷ 2 + 1]) ÷ 2
    println("$case,$variant,$NT,$med,$RUNS")
    flush(stdout)
end

function check(case, variant, got, expected; rtol=0.0)
    size(got) == size(expected) || error("$case $variant: size $(size(got)) != $(size(expected))")
    if rtol == 0
        all(isequal.(got, expected)) || error("$case $variant: mismatch")
    else
        all(isapprox.(got, expected; rtol=rtol)) || error("$case $variant: mismatch")
    end
    println(stderr, "CHECK $case $variant threads=$NT ok")
end

# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------

genl(i) = ((i % 97) - 48.0) / 32.0          # i is 0-based like the Rust side
genr(i) = 0.5 + (i % 89) / 64.0
genc(i, w) = w == 0 ? complex(genl(i), genr(i + 13)) : complex(genr(i), genl(i + 13))

nanmax(a, b) = (isnan(a) | isnan(b)) ? NaN : (a >= b ? a : b)
nanmin(a, b) = (isnan(a) | isnan(b)) ? NaN : (a <= b ? a : b)

# Operands for a layout. `contig`: 1D vectors. `trans`: lhs is the transpose of
# a row-major buffer, so lhs[i, j] reads buffer offset j + i*C like Rust.
function ew_operands(layout, gen_l, gen_r)
    if layout == "contig"
        n = n1(33_554_432)
        a = [gen_l(i) for i in 0:n-1]
        b = [gen_r(i) for i in 0:n-1]
        return a, b
    else
        R, C = d(8192), d(4096)
        buf = [gen_l(i) for i in 0:R*C-1]
        a = transpose(reshape(buf, C, R))   # R x C lazy transpose
        b = reshape([gen_r(i) for i in 0:R*C-1], R, C)
        return a, b
    end
end

function bench_binary(opname, T, layout, f)
    case = "ew_$(opname)_$(T == Float64 ? "f64" : "c64")_$layout"
    enabled(case) || return
    gl = T == Float64 ? genl : (i -> genc(i, 0))
    gr = T == Float64 ? genr : (i -> genc(i, 1))
    a, b = ew_operands(layout, gl, gr)
    expected = similar(b)
    for I in eachindex(IndexCartesian(), b)
        expected[I] = f(a[I], b[I])
    end
    out = similar(b)
    fill!(out, T(NaN))
    measure(case, "julia_base") do
        out .= f.(a, b)
    end
    check(case, "julia_base", out, expected)
    fill!(out, T(NaN))
    sa, sb, so = StridedView(a), StridedView(b), StridedView(out)
    measure(case, "julia_strided") do
        @strided so .= f.(sa, sb)
    end
    check(case, "julia_strided", out, expected)
end

function bench_unary(opname, T, layout, f)
    case = "ew_$(opname)_$(T == Float64 ? "f64" : "c64")_$layout"
    enabled(case) || return
    gl = T == Float64 ? genl : (i -> genc(i, 0))
    a, _ = ew_operands(layout, gl, gl)
    U = typeof(f(one(T)))
    expected = Array{U}(undef, size(a))
    for I in eachindex(IndexCartesian(), a)
        expected[I] = f(a[I])
    end
    out = similar(expected)
    fill!(out, U(NaN))
    measure(case, "julia_base") do
        out .= f.(a)
    end
    check(case, "julia_base", out, expected)
    fill!(out, U(NaN))
    sa, so = StridedView(a), StridedView(out)
    measure(case, "julia_strided") do
        @strided so .= f.(sa)
    end
    check(case, "julia_strided", out, expected)
end

for layout in ("contig", "trans")
    bench_binary("add", Float64, layout, +)
    bench_binary("sub", Float64, layout, -)
    bench_binary("mul", Float64, layout, *)
    bench_binary("div", Float64, layout, /)
    bench_binary("max", Float64, layout, nanmax)
    bench_binary("min", Float64, layout, nanmin)
    bench_unary("neg", Float64, layout, -)
    bench_unary("abs", Float64, layout, abs)
end
bench_binary("add", ComplexF64, "contig", +)
bench_binary("mul", ComplexF64, "contig", *)
bench_binary("div", ComplexF64, "contig", /)
bench_unary("neg", ComplexF64, "contig", -)
bench_unary("conj", ComplexF64, "contig", conj)
bench_unary("abs", ComplexF64, "contig", abs)
bench_unary("conj", ComplexF64, "trans", conj)

# ---------------------------------------------------------------------------
# Ternary elementwise: select and clamp
# ---------------------------------------------------------------------------

genp(i) = (i * 7919) % 13 < 6                     # irregular predicate, 0-based i
genx_clamp(i) = i % 1021 == 0 ? NaN : genl(i)     # sparse NaN exercises propagation

# Same semantics as strided: any NaN operand gives NaN, ties return the bound.
function nanclamp(x, lo, hi)
    raised = lo >= x ? lo : x
    lowered = hi <= raised ? hi : raised
    return (isnan(x) | isnan(lo) | isnan(hi)) ? NaN : lowered
end

# The first operand uses the lhs layout (transposed for `trans`), the others
# and the destination are column major, like the Rust side.
function ter_operands(layout, gen_p, gen_b, gen_c)
    if layout == "contig"
        n = n1(33_554_432)
        return [gen_p(i) for i in 0:n-1], [gen_b(i) for i in 0:n-1], [gen_c(i) for i in 0:n-1]
    else
        R, C = d(8192), d(4096)
        p = transpose(reshape([gen_p(i) for i in 0:R*C-1], C, R))
        b = reshape([gen_b(i) for i in 0:R*C-1], R, C)
        c = reshape([gen_c(i) for i in 0:R*C-1], R, C)
        return p, b, c
    end
end

function bench_ternary(opname, layout, gen_p, gen_b, gen_c, f, fref)
    case = "ter_$(opname)_f64_$layout"
    enabled(case) || return
    p, b, c = ter_operands(layout, gen_p, gen_b, gen_c)
    expected = similar(b)
    for I in eachindex(IndexCartesian(), b)
        expected[I] = fref(p[I], b[I], c[I])
    end
    out = similar(b)
    fill!(out, -12345.0)
    measure(case, "julia_base") do
        out .= f.(p, b, c)
    end
    check(case, "julia_base", out, expected)
    fill!(out, -12345.0)
    sp, sb, sc, so = StridedView(p), StridedView(b), StridedView(c), StridedView(out)
    measure(case, "julia_strided") do
        @strided so .= f.(sp, sb, sc)
    end
    check(case, "julia_strided", out, expected)
end

for layout in ("contig", "trans")
    bench_ternary("select", layout, genp, genl, genr, ifelse, (p, a, b) -> p ? a : b)
    bench_ternary("clamp", layout, genx_clamp, _ -> -0.25, _ -> 0.25, clamp, nanclamp)
end

# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

red_gen(op, i) = op == "prod" ? 1.0 + ((i % 1013) - 506.0) * 1e-9 :
                 0.5 + (i % 1013) / 1013.0 + (i % 7) * 1e-3

const RED_BASE = Dict("sum" => sum, "prod" => prod, "max" => maximum, "min" => minimum)
const RED_INPLACE = Dict("sum" => sum!, "prod" => prod!, "max" => maximum!, "min" => minimum!)
const RED_FOLD = Dict("sum" => (+, 0.0), "prod" => (*, 1.0), "max" => (nanmax, -Inf), "min" => (nanmin, Inf))

function bench_reduce(R0, C0, axes, op)
    case = "red_$(op)_$(axes)_$(R0)x$(C0)"
    enabled(case) || return
    R, C = d(R0), d(C0)
    A = reshape([red_gen(op, i) for i in 0:R*C-1], R, C)
    g, init = RED_FOLD[op]
    rtol = op in ("sum", "prod") ? 1e-9 : 0.0
    fbase = RED_BASE[op]
    finplace = RED_INPLACE[op]
    sA = StridedView(A)
    if axes == "all"
        expected = fill(foldl(g, A; init=init))
        res = Ref(0.0)
        measure(case, "julia_base") do
            res[] = fbase(A)
        end
        check(case, "julia_base", fill(res[]), expected; rtol=rtol)
        measure(case, "julia_strided") do
            res[] = fbase(sA)
        end
        check(case, "julia_strided", fill(res[]), expected; rtol=rtol)
    else
        dim = axes == "axis0" ? 1 : 2
        expected = vec(mapslices(x -> foldl(g, x; init=init), A; dims=dim))
        out = dim == 1 ? zeros(1, C) : zeros(R, 1)
        measure(case, "julia_base") do
            finplace(out, A)
        end
        check(case, "julia_base", vec(out), expected; rtol=rtol)
        fill!(out, NaN)
        so = StridedView(out)
        measure(case, "julia_strided") do
            finplace(so, sA)
        end
        check(case, "julia_strided", vec(out), expected; rtol=rtol)
        local r
        measure(case, "julia_alloc") do
            r = fbase(A; dims=dim)
        end
        TIMING || check(case, "julia_alloc", vec(r), expected; rtol=rtol)
    end
end

for (R0, C0) in ((8192, 4096), (2048, 2048)), axes in ("all", "axis0", "axis1"), op in ("sum", "prod", "max", "min")
    bench_reduce(R0, C0, axes, op)
end

# ---------------------------------------------------------------------------
# Structural copies
# ---------------------------------------------------------------------------

gen_copy(i) = i * 0.5 + 1.0
matrix(R, C, f=gen_copy) = reshape([f(i) for i in 0:R*C-1], R, C)

# Run the three standard variants for a copy described by a view `v` of the
# source and an allocating Base expression `alloc`.
function copy_variants(case, v, alloc, expected)
    out = similar(expected)
    fill!(out, NaN)
    measure(case, "julia_copyto") do
        copyto!(out, v)
    end
    check(case, "julia_copyto", out, expected)
    fill!(out, NaN)
    so, sv = StridedView(out), StridedView(v)
    measure(case, "julia_strided") do
        @strided so .= sv
    end
    check(case, "julia_strided", out, expected)
    if alloc !== nothing
        local r
        measure(case, "julia_alloc") do
            r = alloc()
        end
        check(case, "julia_alloc", r, expected)
    end
end

function copy_slice_step2()
    enabled("copy_slice_step2") || return
    R, C = d(8192), d(4096)
    A = matrix(R, C)
    v = view(A, 1:2:2*(R÷2), :)
    copy_variants("copy_slice_step2", v, () -> A[1:2:2*(R÷2), :], collect(v))
end

function copy_reverse(axis)
    case = "copy_reverse_axis$axis"
    enabled(case) || return
    R, C = d(4096), d(4096)
    A = matrix(R, C)
    v = axis == 0 ? view(A, R:-1:1, :) : view(A, :, C:-1:1)
    copy_variants(case, v, () -> reverse(A; dims=axis + 1), collect(v))
end

function copy_concat(axis)
    case = "copy_concat_axis$axis"
    enabled(case) || return
    R, C = d(4096), d(4096)
    hr, hc = axis == 0 ? (R ÷ 2, C) : (R, C ÷ 2)
    A1 = matrix(hr, hc)
    A2 = matrix(hr, hc, i -> -gen_copy(i))
    expected = axis == 0 ? vcat(A1, A2) : hcat(A1, A2)
    out = similar(expected)
    # Preallocated two-block copy (the natural in-place form of concatenate).
    blk1 = axis == 0 ? (1:hr, :) : (:, 1:hc)
    blk2 = axis == 0 ? (hr+1:2hr, :) : (:, hc+1:2hc)
    fill!(out, NaN)
    measure(case, "julia_copyto") do
        copyto!(view(out, blk1...), A1)
        copyto!(view(out, blk2...), A2)
    end
    check(case, "julia_copyto", out, expected)
    fill!(out, NaN)
    s1, s2 = StridedView(view(out, blk1...)), StridedView(view(out, blk2...))
    sA1, sA2 = StridedView(A1), StridedView(A2)
    measure(case, "julia_strided") do
        @strided s1 .= sA1
        @strided s2 .= sA2
    end
    check(case, "julia_strided", out, expected)
    local r
    measure(case, "julia_alloc") do
        r = axis == 0 ? vcat(A1, A2) : hcat(A1, A2)
    end
    check(case, "julia_alloc", r, expected)
end

function copy_dynslice_rank1()
    enabled("copy_dynslice_rank1") || return
    W = n1(16_777_216)
    A = [gen_copy(i) for i in 0:2W-1]
    s = W ÷ 3
    v = view(A, s+1:s+W)
    copy_variants("copy_dynslice_rank1", v, () -> A[s+1:s+W], collect(v))
end

function copy_dynslice_rank2()
    enabled("copy_dynslice_rank2") || return
    O, W = d(4608), d(4096)
    A = matrix(O, O)
    s0, s1 = d(100), d(200)
    v = view(A, s0+1:s0+W, s1+1:s1+W)
    copy_variants("copy_dynslice_rank2", v, () -> A[s0+1:s0+W, s1+1:s1+W], collect(v))
end

function copy_pad()
    enabled("copy_pad") || return
    DR = d(4096)
    p = d(48)
    r = DR - 2p
    A = matrix(r, r)
    expected = zeros(DR, DR)
    expected[p+1:p+r, p+1:p+r] .= A
    out = similar(expected)
    # Preallocated pad: zero the border strips, copy the interior.
    function pad_base!(out, A)
        out[1:p, :] .= 0.0
        out[p+r+1:end, :] .= 0.0
        out[p+1:p+r, 1:p] .= 0.0
        out[p+1:p+r, p+r+1:end] .= 0.0
        copyto!(view(out, p+1:p+r, p+1:p+r), A)
    end
    fill!(out, NaN)
    measure("copy_pad", "julia_copyto") do
        pad_base!(out, A)
    end
    check("copy_pad", "julia_copyto", out, expected)
    fill!(out, NaN)
    so, sA = StridedView(out), StridedView(A)
    measure("copy_pad", "julia_strided") do
        # Strided has no pad primitive: fill then strided interior copy.
        @strided so .= 0.0
        sv = StridedView(view(out, p+1:p+r, p+1:p+r))
        @strided sv .= sA
    end
    check("copy_pad", "julia_strided", out, expected)
end

copy_slice_step2()
copy_reverse(0)
copy_reverse(1)
copy_concat(0)
copy_concat(1)
copy_dynslice_rank1()
copy_dynslice_rank2()
copy_pad()

println(stderr, "# done")
