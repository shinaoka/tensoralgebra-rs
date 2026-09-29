"""
Investigate what OMEinsum actually does for TN light 415 late-step contraction.
Specifically: how does it avoid the 16M-element permutation?
"""

using OMEinsum
using LinearAlgebra

const NDIMS_A = 13
const NDIMS_B = 24

const IA = (1,2,3,4,5,6,7,8,9,10,24,25,26)
const IB = (3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)
const IC = (1,2,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)

A = rand(Float64, ntuple(_ -> 2, NDIMS_A)...)
B = rand(Float64, ntuple(_ -> 2, NDIMS_B)...)

code = DynamicEinCode(collect.([IA, IB]), collect(IC))
println("DynamicEinCode: ", code)
println()

# What does OMEinsum's internal implementation do?
# Let's look at the method dispatch
println("=== OMEinsum internal approach ===")

# OMEinsum uses einsum_lowering to convert to a sequence of operations
# Let's check what operations it generates
# For a pairwise contraction (DynamicEinCode), it should directly call
# the binary einsum implementation.

# The key is in OMEinsum's `einsum` method for EinCode/DynamicEinCode.
# It permutes, reshapes to 2D, calls GEMM, then reshapes/permutes back.
# But it's smart about which dims to fuse.

# Let's trace the internal steps manually:
# classify indices:
a_indices = collect(IA)
b_indices = collect(IB)
c_indices = collect(IC)

a_set = Set(a_indices)
b_set = Set(b_indices)
c_set = Set(c_indices)

batch = intersect(a_set, b_set, c_set)
sum_idx = setdiff(intersect(a_set, b_set), c_set)
lo = setdiff(a_set, b_set)
ro = setdiff(b_set, a_set)

println("lo = $lo (m = $(2^length(lo)))")
println("sum = $sum_idx (k = $(2^length(sum_idx)))")
println("ro = $ro (n = $(2^length(ro)))")
println("batch = $batch (nb = $(2^length(batch)))")

println()
println("=== Timing individual operations ===")

# Step 1: permute A so batch dims are first, then lo, then sum
# OMEinsum convention: result is (batch..., lo..., sum...) for left operand
# and (batch..., sum..., ro...) for right operand

# For A: current order = IA = (1,2,3,4,5,6,7,8,9,10,24,25,26)
# Need: batch first, then lo, then sum
# batch indices in A: {24,25,26} at positions 11,12,13 (1-indexed)
# lo indices in A: {1,2} at positions 1,2
# sum indices in A: {3,4,5,6,7,8,9,10} at positions 3..10

# Target: (24,25,26, 1,2, 3,4,5,6,7,8,9,10)
# Permutation from current to target:
# Current:  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 24, 25, 26)
# Position:  1  2  3  4  5  6  7  8  9  10  11  12  13
# Target:   (24, 25, 26, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
# Need:     pos 11, 12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
a_perm_omeinsum = (11, 12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# For B: current order = IB = (3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)
# batch indices in B: {24,25,26} at positions 22,23,24 (1-indexed)
# sum indices in B: {3,4,5,6,7,8,9,10} at positions 1..8
# ro indices in B: {11,12,...,23} at positions 9..21
# Target: (24,25,26, 3,4,5,6,7,8,9,10, 11,12,...,23)
# = positions 22,23,24, 1,2,3,4,5,6,7,8, 9,10,...,21
b_perm_omeinsum = (22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)

println("A permutation (OMEinsum-style): $a_perm_omeinsum")
println("B permutation (OMEinsum-style): $b_perm_omeinsum")

# Verify by timing just the permutations
A_perm_buf = similar(A, ntuple(_ -> 2, NDIMS_A)...)
B_perm_buf = similar(B, ntuple(_ -> 2, NDIMS_B)...)

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

# Time permutedims! with OMEinsum's permutation order for B
med_b_perm, _ = bench(() -> permutedims!(B_perm_buf, B, b_perm_omeinsum))
println("permutedims!(B, OMEinsum perm): $(round(med_b_perm*1000, digits=3)) ms")

# Time the OMEinsum approach (batch=first convention)
function omeinsum_style_manual()
    nb = 8
    m = 4
    k = 256
    n = 8192

    # Permute
    permutedims!(A_perm_buf, A, a_perm_omeinsum)
    permutedims!(B_perm_buf, B, b_perm_omeinsum)

    # Reshape to 3D: (batch, m, k) and (batch, k, n)
    A_3d = reshape(A_perm_buf, nb, m, k)
    B_3d = reshape(B_perm_buf, nb, k, n)
    C_3d = Array{Float64}(undef, nb, m, n)

    # Batched GEMM
    for bi in 1:nb
        mul!(view(C_3d, bi, :, :), view(A_3d, bi, :, :), view(B_3d, bi, :, :))
    end
    return C_3d
end

med_manual, _ = bench(omeinsum_style_manual)
println("OMEinsum-style manual (perm+GEMM): $(round(med_manual*1000, digits=3)) ms")

# Now check: does OMEinsum actually use permutedims! or something else?
println()
println("=== What OMEinsum actually calls ===")
# Let's use @code_lowered or similar to see
# Actually, let's just time with profiling

# The key insight might be that OMEinsum uses a DIFFERENT approach:
# Instead of permuting the full 24-dim tensor, it might:
# 1. Group contiguous dims (dims 3-10 for sum in B are already at positions 1-8)
# 2. Only permute batch dims (3 dims) to the front, which is cheap
# 3. Use stride tricks for the GEMM

# Let's verify: is B's memory layout favorable for OMEinsum?
println("B memory layout analysis:")
println("B is col-major with shape $(size(B))")
println("B strides: $(strides(B))")
println()

# For OMEinsum, B needs [batch, sum, ro] = [x,y,z, c..j, k..w]
# In the original B (IB order), sum dims (3..10) are at positions 1..8 (contiguous in memory)
# and batch dims (24,25,26) are at positions 22,23,24 (outermost)
# So the permutation only needs to move batch dims from end to front!

# This is a "circular shift" type permutation which is very cache-friendly
# for col-major data: the innermost dims (sum+ro = 21 dims) stay together,
# and only 3 batch dims move.

# Let's test: what if we just reshape B without permuting?
# B is [sum(8), ro(13), batch(3)] in col-major → already [inner, outer]
# If we reshape to (k*n, nb) = (256*8192, 8) = (2097152, 8), the memory is correct!
println("Key insight: B's col-major layout = [sum(8 dims), ro(13 dims), batch(3 dims)]")
println("Reshape to (k*n, nb) = (2097152, 8) works WITHOUT permutation!")

# Verify
B_no_perm = reshape(B, 256*8192, 8)
B_with_perm = reshape(permutedims(B, b_perm_omeinsum), 8, 256, 8192)

println()
println("=== Simplified test: reshape-only approach ===")
function reshape_only_gemm()
    nb = 8
    m = 4
    k = 256
    n = 8192

    # A: need [batch, lo, sum] → permute batch dims to front
    permutedims!(A_perm_buf, A, a_perm_omeinsum)
    A_3d = reshape(A_perm_buf, nb, m, k)

    # B: already [sum, ro, batch] in memory → just reshape
    # But wait, we need [batch, sum, ro] for batched GEMM...
    # Unless we do GEMM differently: for each batch, extract the right slice

    # Actually, B is [sum, ro, batch] in col-major
    # A view B[:, :, bi] gives a contiguous (sum*ro) slice for batch bi
    B_3d = reshape(B, k, n, nb)  # this is a no-copy reshape!
    C_3d = Array{Float64}(undef, m, n, nb)

    for bi in 1:nb
        mul!(view(C_3d, :, :, bi), view(A_3d, bi, :, :), view(B_3d, :, :, bi))
    end
    return C_3d
end

med_reshape, _ = bench(reshape_only_gemm)
println("reshape-only B + GEMM:  $(round(med_reshape*1000, digits=3)) ms")

# Compare: what does the actual DynamicEinCode do?
med_omeinsum, _ = bench(() -> einsum(code, (A, B)))
println("OMEinsum DynamicEinCode: $(round(med_omeinsum*1000, digits=3)) ms")
