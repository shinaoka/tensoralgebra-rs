"""
Fair comparison: use the SAME scrambled label ordering as Rust's TN light 415 late-step contraction.

In Rust, TN light 415 late-step contraction receives tensors with scrambled labels from previous contraction steps.
The einsum2 plan reorders these scattered-stride dims → expensive copy.

This test reproduces the same scrambled ordering in Julia's OMEinsum.
"""

using OMEinsum
using LinearAlgebra

# --- Exact labels from Rust's TN light 415 late-step contraction (reconstructed from captured permutations) ---
# ia (Rust) = "caxydefghizjb" → use integer labels
# lo={a,b}→{1,2}, sum={c..j}→{3..10}, ro={k..w}→{11..23}, batch={x,y,z}→{24,25,26}
# Rust ia mapping: c=3, a=1, x=24, y=25, d=4, e=5, f=6, g=7, h=8, i=9, z=26, j=10, b=2
const IA_SCRAMBLED = [3, 1, 24, 25, 4, 5, 6, 7, 8, 9, 26, 10, 2]

# ib (Rust) = "hklicxmnopdqfrstyjuzgvwe"
# h=8, k=11, l=12, i=9, c=3, x=24, m=13, n=14, o=15, p=16, d=4, q=17, f=6,
# r=18, s=19, t=20, y=25, j=10, u=21, z=26, g=7, v=22, w=23, e=5
const IB_SCRAMBLED = [8, 11, 12, 9, 3, 24, 13, 14, 15, 16, 4, 17, 6, 18, 19, 20, 25, 10, 21, 26, 7, 22, 23, 5]

# ic (Rust) = "abklwmnopqrstuvxyz"
# a=1, b=2, k=11, l=12, w=23, m=13, n=14, o=15, p=16, q=17, r=18, s=19, t=20, u=21, v=22, x=24, y=25, z=26
const IC_SCRAMBLED = [1, 2, 11, 12, 23, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26]

# Natural (OMEinsum-style) labels (indices in natural order)
const IA_NATURAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 24, 25, 26]
const IB_NATURAL = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
const IC_NATURAL = [1, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

function bench(f, warmup=3, nruns=15)
    for _ in 1:warmup; f(); end
    times = Float64[]
    for _ in 1:nruns
        t = @elapsed f()
        push!(times, t)
    end
    sort!(times)
    med = times[div(length(times)+1, 2)]
    q1 = times[max(1, div(length(times)+1, 4))]
    q3 = times[min(length(times), div(3*(length(times)+1), 4))]
    return med, q3 - q1
end

function main()
    A = rand(Float64, ntuple(_ -> 2, 13)...)
    B = rand(Float64, ntuple(_ -> 2, 24)...)

    println("TN light 415 late-step fair comparison (scrambled vs natural labels)")
    println("A: $(size(A)) = $(length(A)) elements")
    println("B: $(size(B)) = $(length(B)) elements")
    println("=" ^ 70)

    # --- 1) OMEinsum with SCRAMBLED labels (same as Rust) ---
    code_scram = DynamicEinCode([IA_SCRAMBLED, IB_SCRAMBLED], IC_SCRAMBLED)
    med_scram, iqr_scram = bench(() -> einsum(code_scram, (A, B)))
    println("OMEinsum (scrambled labels):   $(round(med_scram*1000, digits=3)) ms (IQR $(round(iqr_scram*1000, digits=3)) ms)")

    # --- 2) OMEinsum with NATURAL labels (favorable ordering) ---
    code_nat = DynamicEinCode([IA_NATURAL, IB_NATURAL], IC_NATURAL)
    med_nat, iqr_nat = bench(() -> einsum(code_nat, (A, B)))
    println("OMEinsum (natural labels):     $(round(med_nat*1000, digits=3)) ms (IQR $(round(iqr_nat*1000, digits=3)) ms)")

    # --- 3) Isolate: permutedims! for B with Rust's right_perm ---
    # Rust's right_perm (0-indexed) = [4, 10, 23, 12, 20, 0, 3, 17, 1, 2, 6, 7, 8, 9, 11, 13, 14, 15, 18, 21, 22, 5, 16, 19]
    # Julia 1-indexed:
    RIGHT_PERM_JULIA = (5, 11, 24, 13, 21, 1, 4, 18, 2, 3, 7, 8, 9, 10, 12, 14, 15, 16, 19, 22, 23, 6, 17, 20)
    B_buf = similar(B)
    med_perm, iqr_perm = bench(() -> permutedims!(B_buf, B, RIGHT_PERM_JULIA))
    println("permutedims!(B, Rust perm):    $(round(med_perm*1000, digits=3)) ms (IQR $(round(iqr_perm*1000, digits=3)) ms)")

    # --- 4) permutedims! for B with near-identity perm (batch to front) ---
    # OMEinsum-style: move batch (positions 22,23,24) to front
    BATCH_FRONT_PERM = (22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    B_buf2 = similar(B)
    med_perm2, iqr_perm2 = bench(() -> permutedims!(B_buf2, B, BATCH_FRONT_PERM))
    println("permutedims!(B, batch→front):  $(round(med_perm2*1000, digits=3)) ms (IQR $(round(iqr_perm2*1000, digits=3)) ms)")

    # --- 5) Just BLAS GEMM (pre-contiguous, batch-last) ---
    m, k, n, nb = 4, 256, 8192, 8
    A_mat = reshape(A, m, k, nb)  # A happens to be [lo, sum, batch] already for natural labels
    B_mat = reshape(B, k, n, nb)
    C_mat = Array{Float64}(undef, m, n, nb)
    med_gemm, iqr_gemm = bench(() -> begin
        for bi in 1:nb
            mul!(view(C_mat, :, :, bi), view(A_mat, :, :, bi), view(B_mat, :, :, bi))
        end
    end)
    println("BLAS gemm only (8 batches):    $(round(med_gemm*1000, digits=3)) ms (IQR $(round(iqr_gemm*1000, digits=3)) ms)")

    # --- Summary ---
    println()
    println("--- Key comparison ---")
    println("OMEinsum scrambled:  $(round(med_scram*1000, digits=1)) ms  (Rust equivalent labels)")
    println("OMEinsum natural:    $(round(med_nat*1000, digits=1)) ms  (favorable labels)")
    println("permutedims!(Rust):  $(round(med_perm*1000, digits=1)) ms  (just B copy with scattered perm)")
    println("permutedims!(easy):  $(round(med_perm2*1000, digits=1)) ms  (just B copy with batch→front)")
    println("BLAS GEMM:           $(round(med_gemm*1000, digits=1)) ms  (pre-contiguous)")
end

main()
