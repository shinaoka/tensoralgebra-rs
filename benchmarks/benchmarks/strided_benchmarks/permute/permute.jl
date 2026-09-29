using JSON
using Strided

const PATTERN_PATH = joinpath(@__DIR__, "patterns.json")

function col_major_strides(shape::Vector{Int})
    strides = ones(Int, length(shape))
    for i in 2:length(shape)
        strides[i] = strides[i - 1] * shape[i - 1]
    end
    return strides
end

function deterministic_data(total::Int)
    return Float64.(1:total)
end

function median_duration(samples::Vector{Float64})
    sort!(samples)
    n = length(samples)
    if isodd(n)
        return samples[(n + 1) ÷ 2]
    else
        return (samples[n ÷ 2] + samples[n ÷ 2 + 1]) / 2
    end
end

function bench(f, label::String, warmup::Int, iters::Int, bytes::Int)
    for _ in 1:warmup
        f()
    end
    samples = Float64[]
    sizehint!(samples, iters)
    for _ in 1:iters
        t0 = time_ns()
        f()
        push!(samples, (time_ns() - t0) / 1e9)
    end
    med = median_duration(samples)
    p25 = sort(samples)[max(1, length(samples) ÷ 4)]
    p75 = sort(samples)[max(1, (length(samples) * 3) ÷ 4)]
    ms = med * 1e3
    gbps = bytes / med / 1e9
    println("  ", rpad(label, 30), " ", lpad(round(ms, digits=3), 8),
            " ms  (", round(p25 * 1e3, digits=3), " / ",
            round(p75 * 1e3, digits=3), ")  ", round(gbps, digits=2), " GB/s")
end

function timing_counts(total::Int)
    return total >= (1 << 23) ? (3, 15) : (5, 40)
end

function load_patterns()
    suite = JSON.parsefile(PATTERN_PATH)
    suite["version"] == 1 || error("unsupported pattern schema version")
    suite["index_base"] == 0 || error("patterns must use index_base = 0")
    suite["semantics"] == "out[i0,...,ik] = src[i_perm0,...,i_permk]" ||
        error("unsupported semantics")
    suite["data"] == "deterministic_index_value" || error("unsupported data mode")
    return suite["patterns"]
end

function source_view(pattern)
    shape = Int.(pattern["shape"])
    total = prod(shape)
    parent = deterministic_data(total)
    layout = pattern["src_layout"]
    strides = if layout["kind"] == "col_major"
        col_major_strides(shape)
    elseif layout["kind"] == "explicit_strides"
        Int.(layout["strides"])
    else
        kind = layout["kind"]
        error("unsupported src_layout kind $kind")
    end
    return parent, StridedView(parent, Tuple(shape), Tuple(strides), 0)
end

function output_shape(pattern)
    shape = Int.(pattern["shape"])
    perm = Int.(pattern["perm"]) .+ 1
    return shape[perm]
end

function verify_output(label::String, actual, expected)
    vec(actual) == vec(expected) || error("$label mismatch")
end

function run_pattern(pattern)
    id = pattern["id"]
    label = pattern["label"]
    participants = Set(String.(pattern["participants"]))
    if !("julia_base" in participants || "strided_jl" in participants)
        return false
    end
    perm = Int.(pattern["perm"]) .+ 1
    out_shape = output_shape(pattern)
    total = prod(out_shape)
    bytes = total * sizeof(Float64) * 2
    warmup, iters = timing_counts(total)

    println("=== $label ===")
    println("  id=$id elems=$total bytes(r+w)=$bytes")

    parent_data, src = source_view(pattern)
    src_perm = permutedims(src, Tuple(perm))
    reference = Array(src_perm)

    if "julia_base" in participants
        dst = Array{Float64}(undef, Tuple(out_shape))
        if pattern["src_layout"]["kind"] == "col_major"
            src_array = reshape(parent_data, Tuple(Int.(pattern["shape"])))
            permutedims!(dst, src_array, Tuple(perm))
            verify_output("Julia Base permutedims!", dst, reference)
            bench("Julia Base permutedims!", warmup, iters, bytes) do
                permutedims!(dst, src_array, Tuple(perm))
            end
        else
            src_base_perm = PermutedDimsArray(src, Tuple(perm))
            copyto!(dst, src_base_perm)
            verify_output("Julia Base copyto!", dst, reference)
            bench("Julia Base copyto!", warmup, iters, bytes) do
                copyto!(dst, src_base_perm)
            end
        end
    end

    if "strided_jl" in participants
        dst_parent = Vector{Float64}(undef, total)
        dst = StridedView(dst_parent, Tuple(out_shape), Tuple(col_major_strides(out_shape)), 0)
        @strided dst .= src_perm
        verify_output("Strided.jl @strided", Array(dst), reference)
        bench("Strided.jl @strided", warmup, iters, bytes) do
            @strided dst .= src_perm
        end
    end

    println()
    return true
end

function main()
    println("Strided.jl permutation benchmarks")
    println("=================================")
    println("Patterns: $PATTERN_PATH")
    if haskey(ENV, "PATTERN_ID")
        println("Pattern filter: ", ENV["PATTERN_ID"])
    end
    println("Julia threads: ", Threads.nthreads(), " Strided.jl threads: ", Strided.get_num_threads())
    println("Format: label  median_ms  (p25 / p75)  bandwidth_GB/s")
    println()

    patterns = load_patterns()
    if haskey(ENV, "PATTERN_ID")
        patterns = filter(p -> p["id"] == ENV["PATTERN_ID"], patterns)
    end
    isempty(patterns) && error("PATTERN_ID did not match any pattern")

    ran = false
    for pattern in patterns
        ran |= run_pattern(pattern)
    end
    if !ran
        println("No Julia participants for the selected pattern set.")
    end
end

main()
