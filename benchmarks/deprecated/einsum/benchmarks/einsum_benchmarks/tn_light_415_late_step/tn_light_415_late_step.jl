"""
Minimal benchmark reproducing TN light 415 late-step contraction of tensornetwork_permutation_light_415.

Binary einsum: abcdefghijxyz,cdefghijklmnopqrstuvwxyz->abklmnopqrstuvwxyz
All dims = 2 (binary tensor network).
m=4, k=256, n=8192, batch=8.

Compares:
  1) OMEinsum pairwise contraction (what the full benchmark uses)
  2) Manual: permutedims! + BLAS gemm (to isolate permutation cost)
  3) Just the permutedims! cost alone
"""

using OMEinsum
using LinearAlgebra

# --- Tensor setup ---
# All dims = 2, column-major. A has 13 dims, B has 24 dims.
# After lazy permutation (metadata-only reorder), strides become scattered.
# This is the situation strided-rs faces.

const NDIMS_A = 13
const NDIMS_B = 24
const NDIMS_C = 18

function make_tensors()
    A = rand(Float64, ntuple(_ -> 2, NDIMS_A)...)
    B = rand(Float64, ntuple(_ -> 2, NDIMS_B)...)
    return A, B
end

# The canonical einsum expression
# A indices:   a b c d e f g h i j x y z    (lo=ab, sum=cdefghij, batch=xyz)
# B indices:   c d e f g h i j k l m n o p q r s t u v w x y z  (sum=cdefghij, ro=klmnopqrstuvw, batch=xyz)
# C indices:   a b k l m n o p q r s t u v w x y z  (lo=ab, ro=klmnopqrstuvw, batch=xyz)

# einsum notation (1-indexed labels for OMEinsum)
# A: (1,2,3,4,5,6,7,8,9,10,24,25,26)
# B: (3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)
# C: (1,2,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)

const IA = (1,2,3,4,5,6,7,8,9,10,24,25,26)
const IB = (3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)
const IC = (1,2,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)

# --- The permutations that strided-rs applies (0-indexed -> 1-indexed) ---
# After lazy permutation of A from previous steps, A's dims are reordered by left_perm.
# left_perm (0-indexed) = [1, 12, 0, 4, 5, 6, 7, 8, 9, 11, 2, 3, 10]
# This means: canonical dim j uses original dim left_perm[j].
# Julia permutedims(A, perm): result[i1,i2,...] = A[i_{perm[1]}, i_{perm[2]}, ...]
# So Julia perm = left_perm .+ 1
const LEFT_PERM = (2, 13, 1, 5, 6, 7, 8, 9, 10, 12, 3, 4, 11)
const RIGHT_PERM = (5, 11, 24, 13, 21, 1, 4, 18, 2, 3, 7, 8, 9, 10, 12, 14, 15, 16, 19, 22, 23, 6, 17, 20)

# --- Benchmark helpers ---
function bench(f, warmup=3, nruns=15)
    for _ in 1:warmup
        f()
    end
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
    A_orig, B_orig = make_tensors()

    println("TN light 415 late-step benchmark (all dims=2)")
    println("A: $(size(A_orig)) = $(length(A_orig)) elements")
    println("B: $(size(B_orig)) = $(length(B_orig)) elements")
    println("=" ^ 70)

    # --- 1) OMEinsum pairwise contraction (DynamicEinCode) ---
    code = DynamicEinCode(collect.([IA, IB]), collect(IC))
    f_omeinsum() = einsum(code, (A_orig, B_orig))
    med, iqr = bench(f_omeinsum)
    C_ref = f_omeinsum()
    println("OMEinsum DynamicEinCode:         $(round(med*1000, digits=3)) ms (IQR $(round(iqr*1000, digits=3)) ms)")

    # --- 2) Simulate strided-rs's situation ---
    # strided-rs has lazy-permuted tensors (scattered strides).
    # Equivalent in Julia: create a PermutedDimsArray (a view with reordered dims)
    # then permutedims! to make contiguous, then BLAS GEMM.

    # Create the "lazy permuted" view (like strided-rs after metadata permutation)
    # Note: PermutedDimsArray(A, perm) creates a view where
    # result[i1,...] = A[i_{perm[1]}, ...] -- same semantics as permutedims
    A_lazy = PermutedDimsArray(A_orig, LEFT_PERM)
    B_lazy = PermutedDimsArray(B_orig, RIGHT_PERM)

    # The "eager copy" path: permutedims! to materialize contiguous data
    A_buf = similar(A_lazy)
    B_buf = similar(B_lazy)

    function copy_and_gemm()
        permutedims!(A_buf, A_orig, LEFT_PERM)
        permutedims!(B_buf, B_orig, RIGHT_PERM)
        # Now reshape for GEMM: A_buf is [lo(2), sum(8), batch(3)] = [4, 256, 8]
        # B_buf is [sum(8), ro(13), batch(3)] = [256, 8192, 8]
        m, k, n, nb = 4, 256, 8192, 8
        A_mat = reshape(A_buf, m, k, nb)
        B_mat = reshape(B_buf, k, n, nb)
        C_mat = Array{Float64}(undef, m, n, nb)
        for bi in 1:nb
            mul!(view(C_mat, :, :, bi), view(A_mat, :, :, bi), view(B_mat, :, :, bi))
        end
        return C_mat
    end
    med2, iqr2 = bench(copy_and_gemm)
    println("permutedims! + BLAS gemm:        $(round(med2*1000, digits=3)) ms (IQR $(round(iqr2*1000, digits=3)) ms)")

    # --- 3) Just the permutedims! cost (B only, since it's 16M elements) ---
    function just_perm_B()
        permutedims!(B_buf, B_orig, RIGHT_PERM)
    end
    med3, iqr3 = bench(just_perm_B)
    println("permutedims!(B) only (16M f64):  $(round(med3*1000, digits=3)) ms (IQR $(round(iqr3*1000, digits=3)) ms)")

    # --- 4) Just the permutedims! cost (A only, 8K elements) ---
    function just_perm_A()
        permutedims!(A_buf, A_orig, LEFT_PERM)
    end
    med4, iqr4 = bench(just_perm_A)
    println("permutedims!(A) only (8K f64):   $(round(med4*1000, digits=3)) ms (IQR $(round(iqr4*1000, digits=3)) ms)")

    # --- 5) Just BLAS gemm (with pre-materialized contiguous data) ---
    permutedims!(A_buf, A_orig, LEFT_PERM)
    permutedims!(B_buf, B_orig, RIGHT_PERM)
    m, k, n, nb = 4, 256, 8192, 8
    A_mat = reshape(A_buf, m, k, nb)
    B_mat = reshape(B_buf, k, n, nb)
    C_mat = Array{Float64}(undef, m, n, nb)
    function just_gemm()
        for bi in 1:nb
            mul!(view(C_mat, :, :, bi), view(A_mat, :, :, bi), view(B_mat, :, :, bi))
        end
    end
    med5, iqr5 = bench(just_gemm)
    println("BLAS gemm only (8 batches):      $(round(med5*1000, digits=3)) ms (IQR $(round(iqr5*1000, digits=3)) ms)")

    # --- Verify correctness ---
    C_manual = copy_and_gemm()
    # C_ref has shape (2,2,...,2) 18 dims, C_manual is (4, 8192, 8)
    # Reshape for comparison
    C_ref_r = reshape(C_ref, m, n, nb)
    err = maximum(abs.(C_ref_r .- C_manual))
    println("\nMax error (OMEinsum vs manual): $err")
end

main()
