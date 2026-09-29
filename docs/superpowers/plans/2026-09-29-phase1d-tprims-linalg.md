# Phase 1d: tprims-linalg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dense linear algebra on explicit `Exec` contexts, faer per item, covering tenferro's CPU linear algebra (Cholesky, LDLᴴ, partial- and full-pivot LU with solves, QR with and without column pivoting, SVD, eigh, nonsymmetric eig; solve, lstsq, inv, det, logdet), plus a `batched` module with scratch sized once per call; correctness by reconstruction and residuals; 1T/4T benchmarks including the small-matrix large-batch regression case.

**Architecture:** `crates/tprims-linalg`, depending on `tprims-blas` (for `Scalar`, `Conj`, GEMM in reconstructions) and faer. Inputs are rank-2 `StridedView`s copied once into library-owned column-major factor storage (an explicit, documented materialization: factors are outputs the library owns). Factor objects keep faer's representation (permutations as `Vec<usize>` forward/inverse, Householder bases plus block factors) and expose solves on caller `StridedViewMut` right-hand sides. Every call computes one `StackReq`, allocates one `MemBuffer`, and takes `Par` from `Exec::install` with a width from a flop estimate. The `batched` module runs equal-shape batches (batch axis last) with one scratch buffer and one work matrix reused across items, a per-item status, items partitioned over the pool with `Par::Seq` when items are small.

**Tech Stack:** faer 0.24.4 (`faer::linalg::{cholesky, lu, qr, svd, evd, householder, triangular_solve}`, `faer::dyn_stack`), tprims-blas, tprims-exec, strided-view.

**Spec:** `docs/superpowers/specs/2026-09-29-phase1-cpu-backend-design.md` (section 1d).

## Global Constraints

- Branch `phase1d-linalg` from `main` after 1b merges (else from `phase1b-blas`).
- `-j 16`; clippy with `cargo +1.98.0`.
- Scalars: extend `tprims_blas::Scalar` with `faer::traits::ComplexField<Real = Self::Re>` and `Re: faer::traits::RealField` so real parts are nameable as `T::Re`.
- Errors: `#[non_exhaustive] enum Error { Rank, Shape(String), NotSquare, NotPositiveDefinite { index }, Singular { index }, NoConvergence, NonFinite, AliasedOutput }`. Factorizations never fail on singular input (LAPACK `getrf` semantics); solves apply one rule: `Singular { index }` when `|u_ii| <= n * eps * max_j |u_jj|`.
- Conventions (documented contracts, not re-sorted): `svd` values real, non-increasing, returns `U`, `S`, `V` (not Vᴴ); `eigh` values real, non-decreasing, reads the lower triangle; `eig` values and right eigenvectors complex; real-input pairs with `|im| <= eps * max(|re|, 1)` are real, else `v = U[:, j] ± i U[:, j + 1]` (reimplementing faer's private conversion).
- Batched loops: no per-item allocation, no per-item workspace query; one `MemBuffer` from `StackReq::any_of` of all item needs.
- Tests reimplemented (not copied) from tenferro's residual/reconstruction ideas; tolerances relative to `n * eps * ||A||`.

## Review Focus

- Non-finite input (NaN) to svd/eigh/eig returns `NoConvergence` or `NonFinite`, never panics.
- Empty matrices (n = 0, m = 0) for every factorization and solve.
- Complex Hermitian inputs whose upper triangle disagrees with the lower (eigh/cholesky read only the lower triangle).
- Rank-deficient `lstsq` (column-pivoted QR with a rank tolerance) and singular `solve` (typed error, index of the failing pivot).
- Batched call where one item is singular / not positive definite: other items still solved, status reports the failing item.

---

### Task 1: Crate, errors, `Scalar` extension, owned column-major matrix
Files: `crates/tprims-linalg/{Cargo.toml,src/lib.rs,src/error.rs,src/mat.rs}`, `crates/tprims-blas/src/scalar.rs` (add the faer `Real` equality bound and `Re: RealField`), tests `tests/mat.rs`.
Produces: `pub struct Matrix<T> { rows, cols, data: Vec<T> }` (column-major, `view()`, `view_mut()`, `from_view(&StridedView) -> Result<Self>` copying once, `get(i, j)`), `pub(crate) fn faer_ref/faer_mut`, `pub(crate) fn par(exec, flops) -> impl FnOnce(...)` helper wrapping `Exec::install` + width from `GemmPolicy`.

### Task 2: Cholesky and LDLᴴ
`cholesky(exec, a) -> Result<Cholesky<T>>` (lower; `NotPositiveDefinite { index }` from `LltError::NonPositivePivot`), `Cholesky::{l() -> &Matrix, solve(exec, b: &mut StridedViewMut)}`; `ldlt(exec, a) -> Result<Ldlt<T>>` via Bunch–Kaufman `lblt` (`subdiag`, `perm`), `Ldlt::solve`. Tests: reconstruction `L Lᴴ = A`, solve residual, non-PD error index, indefinite matrix through LDLᴴ, complex Hermitian with garbage upper triangle, n = 0.

### Task 3: LU (partial and full pivoting), solve, det, logdet, inv
`lu(exec, a) -> Result<Lu<T>>` (rectangular allowed; `perm`, `perm_inv`, `transpositions`), `Lu::{solve(exec, op: Op, b), det(), logdet() -> (T, T::Re), l(), u(), p()}` (square-only methods return `NotSquare`); `lu_full(exec, a)` with row/col perms and `solve`; free functions `solve(exec, a, b)`, `inv(exec, a) -> Matrix`, `det(exec, a)`, `logdet(exec, a)`. Tests: `P A = L U` reconstruction (rectangular 7x5, 5x7), solve residual for `Op::{N, T, C}`, singular → `Singular { index }`, det of known matrices (permutation parity), logdet sign/magnitude for complex, inv·A = I.

### Task 4: QR and least squares
`qr(exec, a) -> Qr<T>` (block size `recommended_block_size`), `Qr::{r(), q_thin(), q_full(), apply_q(exec, b), apply_qh(exec, b), solve_lstsq(exec, b)}`; `qr_col_piv(exec, a) -> ColPivQr<T>` with `perm()` and `rank(tol)`; `lstsq(exec, a, b, rcond) -> Lstsq { x: Matrix, rank }` (col-pivoted QR, rank from `|r_ii| > rcond * |r_00|`, minimum-norm not promised — documented). Tests: `Q R = A`, `Qᴴ Q = I`, lstsq residual orthogonality `Aᴴ (A x − b) ≈ 0` for full rank, rank detection on a rank-2 6x4 matrix, wide and tall shapes.

### Task 5: SVD, eigh, eig
`svd(exec, a, vectors: Vectors::{None, Thin, Full}) -> Result<Svd<T>>`, `eigh(exec, a, vectors: bool) -> Result<Eigh<T>>`, `eig(exec, a, vectors: bool) -> Result<Eig<T>>`. Non-finite input checked first → `NonFinite`. Tests: `U S Vᴴ = A` and orthonormality, ordering, `A V = V diag(w)` for eigh (Hermitian complex), `A v = λ v` for eig on a real matrix with complex pairs (rotation block) and a complex matrix, NaN input error.

### Task 6: `batched` module
`batched::{solve, cholesky, cholesky_solve, eigh, svd, qr}` over rank-3 inputs `[m, n, batch]`, outputs written into caller rank-3 `StridedViewMut` (factors in place of the input copy for `cholesky`/`qr`, vectors/values for `eigh`/`svd`), returning `Vec<Status>` (`Ok`, `Singular { index }`, `NotPositiveDefinite { index }`, `NoConvergence`). Scratch: one `MemBuffer` for the maximum item requirement, one work `Matrix`, reused. Schedule: the 1b rule (item width 1 and total width > 1 → `for_each_partition` over items with per-lane scratch allocated once per lane, `Par::Seq`). Tests: every op vs the single-matrix API per item (bitwise for serial vs pool), one singular item in a batch of 16, batch 0, 2x2 with batch 1024.

### Task 7: Benchmarks and docs
`benchmarks/benchmarks/tprims/linalg/linalg.rs` (bin `linalg`): each op (cholesky, lu, solve, qr, svd, eigh, eig) at n ∈ {2, 4, 8, 32, 128, 512}, single matrix and batched (batch 1024 for n ≤ 8, 64 for n = 32), f64 and c64, 1T and 4T; `CHECK` lines with residuals. Record `README.md` like blas. Decision log row for batched linalg; README crate table. Gate, fresh review, PR, merge after green CI.
