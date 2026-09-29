# Phase 1b: tprims-blas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tprims-blas` with GEMM, strided batched GEMM (two strategies: faer plus a loop over items, and TBLIS-style via tensorcontract) and TRSM, all on an explicit `Exec`, over strided views with per-operand conjugation — with a shared correctness suite and 1T/4T benchmarks for every operation and strategy.

**Architecture:** One crate `crates/tprims-blas`. A sealed `Scalar` trait ties `faer::ComplexField` and `tensorcontract::Element` for f32/f64/c32/c64. Operands are `strided_view::StridedView` (rank 2 for GEMM/TRSM, rank 3 with the batch axis last for batched GEMM) plus a `Conj` flag; outputs are `StridedViewMut`. faer is called through `MatRef/MatMut::from_raw_parts` on the views' pointers and signed strides; parallelism comes from `Exec::install` with a width from a flop estimate. The TBLIS strategy builds one `tensorcontract::Plan` per call and runs items through `for_each_partition` (items serial, batch axis parallel) or, for fewer items than the budget, each item through the `Spmd` seam.

**Tech Stack:** faer 0.24, tensorcontract (in-repo), strided-view 0.4.4, tprims-exec, thiserror, num-complex 0.4.

**Spec:** `docs/superpowers/specs/2026-09-29-phase1-cpu-backend-design.md` (section 1b).

## Global Constraints

- Branch `phase1b-blas` from `phase1a-exec` (or from `main` once 1a is merged); PR base accordingly.
- `-j 16` on every cargo call; clippy with the newest installed stable (`cargo +1.98.0 clippy`), since CI uses stable.
- Crate metadata as in 1a (`version = "0.1.0"`, `publish = false`, `license = "MIT OR Apache-2.0"`, `[lints] workspace = true`).
- faer parallelism only from `Exec::install`'s `Par`; never `Par::rayon(0)`; tensorcontract only through `run_raw_with` (never `run`/`run_raw`, which read `TENSORCONTRACT_THREADS` and may spawn).
- Validation before any pointer use: ranks, shapes, output injectivity (a zero or overlapping output stride with extent > 1 is `Error::AliasedOutput`); faer treats aliasing `MatMut` as UB.
- β = 0 never reads C (C may hold NaN); β ∉ {0, 1} scales C first then accumulates.
- Unit tests under `crates/tprims-blas/tests/` (integration) — no inline test modules.
- Credit: the TBLIS strategy's rustdoc and the crate README credit Lukas Devos's tensorprimitives-rs (`tensorcontract`) and Matthews, *High-Performance Tensor Contraction without Transposition* (arXiv:1607.00291). Any code moved out of `tensorcontract` follows REPOSITORY_RULES "Imported Code" (authors kept, `Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>`).

## Review Focus

- β = 0 with C containing NaN must give a finite result (C not read) for GEMM, both batched strategies.
- Negative strides (reversed views) and transposed views on every operand, both strategies.
- `k == 0`: result is β·C (and exactly 0 for β = 0), no provider call with empty K that could read garbage.
- Complex conjugation on A only, B only, both — against a reference that conjugates explicitly.
- Batched output whose batch stride overlaps items (e.g. batch stride 0) is rejected before any write.

---

### Task 1: Crate skeleton, `Scalar`, errors, validated matrix operands

**Files:** Create `crates/tprims-blas/{Cargo.toml,src/lib.rs,src/scalar.rs,src/error.rs,src/operand.rs}`, `crates/tprims-blas/tests/common/mod.rs`, `crates/tprims-blas/tests/operand.rs`; modify root `Cargo.toml` (member + `tprims-blas` workspace dep).

**Interfaces (Produces):**
```rust
pub trait Scalar: sealed::Sealed + faer::traits::ComplexField + tensorcontract::Element<Real = <Self as Scalar>::Re> + Copy + Send + Sync + 'static {
    type Re: tensorcontract::KernelSet;
    const IS_COMPLEX: bool;
    fn zero() -> Self; fn one() -> Self;
    fn is_zero(self) -> bool; fn is_one(self) -> bool;
    fn mul(self, o: Self) -> Self;
}   // impl for f32, f64, Complex<f32>, Complex<f64>
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Conj { No, Yes }
#[derive(Debug, thiserror::Error, PartialEq, Eq)] #[non_exhaustive]
pub enum Error {
    #[error("{operand}: expected rank {expected}, got {got}")] Rank { operand: &'static str, expected: usize, got: usize },
    #[error("shape mismatch: {0}")] Shape(String),
    #[error("output has zero or overlapping strides")] AliasedOutput,
    #[error(transparent)] Contract(#[from] tensorcontract::Error),
}
pub type Result<T> = core::result::Result<T, Error>;
pub struct MatIn<'v, 'a, T> { pub view: &'v StridedView<'a, T>, pub conj: Conj }
impl<'v, 'a, T> MatIn<'v, 'a, T> { pub fn new(view: &'v StridedView<'a, T>) -> Self; pub fn conj(self) -> Self; }
pub(crate) struct Mat2 { pub rows: usize, pub cols: usize, pub rs: isize, pub cs: isize }
pub(crate) fn mat2<T>(name: &'static str, dims: &[usize], strides: &[isize]) -> Result<Mat2>;
pub(crate) fn check_injective_2d(m: &Mat2) -> Result<()>;   // sufficient test: sort |rs|,|cs|; small*(ext_small-1) < large, or an extent <= 1
pub(crate) unsafe fn scale_in_place<T: Scalar>(ptr: *mut T, m: &Mat2, beta: T);   // beta==0 writes zero, never reads
```
Tests (`tests/operand.rs`): rank errors; `check_injective_2d` accepts col-major, row-major, negative strides, a padded leading dimension, extent-1 zero stride; rejects stride-0 with extent 2 and `rs = cs = 1` with 2x2. `tests/common/mod.rs` provides `naive_gemm<T>(alpha, a, conj_a, b, conj_b, beta, c) -> Vec<T>` over `StridedView` element access (`get`), dense column-major result, and `max_rel_err`.

Steps: write tests → fail → implement → pass → commit "Add tprims-blas skeleton: Scalar, errors, validated operands".

### Task 2: `gemm` on faer

**Interfaces:**
```rust
pub fn gemm<T: Scalar>(exec: &Exec<'_>, alpha: T, a: MatIn<'_, '_, T>, b: MatIn<'_, '_, T>, beta: T, c: &mut StridedViewMut<'_, T>) -> Result<()>;
pub struct GemmPolicy { pub ns_per_flop: f64, pub width: WidthPolicy }   // Default: 0.05 ns/flop (provisional), WidthPolicy::default()
```
Semantics: `C = alpha * op(A) * op(B) + beta * C`, `op` = conjugation only (transposes are expressed by the views). Shapes `(m,k) (k,n) (m,n)`. Empty `m` or `n`: Ok, nothing touched. `k == 0`: `scale_in_place(c, beta)`. Otherwise: if β ∉ {0,1} scale C; `Accum::Replace` when β = 0 else `Add`; width `exec.width_for(2·m·n·k·(4 if complex)·ns_per_flop, …)`; `exec.install(k, |par| faer::linalg::matmul::matmul_with_conj(c_mat, accum, a_mat, conj_a, b_mat, conj_b, alpha, to_faer(par)))`. Normalize size-1 extents' strides to 1 before building faer views (faer asserts on some degenerate strides).

Tests (`tests/gemm.rs`), each for f64 and c64, against `naive_gemm` with `max_rel_err < 1e-12`: square and rectangular (7x5x3, 64x33x17); A and B transposed views (`permute(&[1,0])`); negative strides via reversed views (build with `StridedView::new(data, dims, &[-1*.., ..], offset)`); C with padded leading dimension; conj A / conj B / both (c64); β = 0 with C prefilled with NaN; β = 2.5; k = 0 with β = 0 and β = 3; m = 0; `AliasedOutput` for a stride-0 C; the 4-worker `Exec` gives the same result as serial within 1e-12 and `pool.stats().entries == 0` for an 8x8x8 GEMM (serial-size work never enters).

Commit "tprims-blas: gemm on faer with explicit Exec".

### Task 3: `gemm_batched`, strategy FaerLoop

**Interfaces:**
```rust
pub struct BatchIn<'v, 'a, T> { pub view: &'v StridedView<'a, T>, pub conj: Conj }   // rank 3: [rows, cols, batch]
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum BatchStrategy { Auto, FaerLoop, Tblis }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Selected { FaerLoop { outer_parallel: bool }, Tblis { outer_parallel: bool } }
pub fn gemm_batched<T: Scalar>(exec: &Exec<'_>, alpha: T, a: BatchIn<'_, '_, T>, b: BatchIn<'_, '_, T>, beta: T, c: &mut StridedViewMut<'_, T>, strategy: BatchStrategy) -> Result<Selected>;
```
Validation: equal batch extents; per-item shapes as GEMM; output injective across items (check the 3-D layout: sort the three `(extent, |stride|)` pairs and require each stride ≥ previous stride × previous extent, a sufficient condition; else `AliasedOutput`). Scheduling: `outer_parallel = batch >= 2 && exec.budget() > 1 && per_item_ns < policy.width.serial_below_ns * 4` — then `exec.for_each_partition(batch, &|i| gemm_item(Par::Seq, i))`; otherwise loop items on the caller with each item's own `install` width. Items are addressed by raw pointer + `i * batch_stride`; the closure captures a `SendPtr` wrapper (`struct SendPtr<T>(*mut T); unsafe impl Send/Sync`) with an `// INVARIANT:` stating injectivity was checked. `Auto` maps to `FaerLoop` until a measured rule exists.

Tests (`tests/batched.rs`): batch 1, 3, 64 of 4x3x5 and 16x16x16, f64/c64, batch axis last and a permuted batch layout (batch stride smallest), conj, β = 0 with NaN, reference per item; outer-parallel run on a 4-worker pool equals serial exactly (same per-item code path) and reports `Selected::FaerLoop { outer_parallel: true }`; batch stride 0 on C → `AliasedOutput`.

Commit "tprims-blas: batched gemm, faer plus a loop over items".

### Task 4: `gemm_batched`, strategy Tblis

Build `tensorcontract::Layout::new(vec![rows, cols], vec![rs, cs])` for A, B, C per item (strides as i64) with labels A `[0, 2]`, B `[2, 1]`, D `[0, 1]`; `Operand::conj()` per `Conj::Yes`; `Plan::new(a, b, None /* or C == D */, d)` once. β handling: pass `c = d` pointer with `beta` (tensorcontract allows `c == d` through `run_raw_with`), `beta = 0` passes `None`-equivalent zero so C is not read (verify with the NaN test). Items: if `outer_parallel` (same rule as Task 3), `for_each_partition(batch, …)` with a width-1 `Spmd` (`struct Serial; width() = 1; broadcast(1, f) runs f(0)`); else loop items with `ExecSpmd { exec, width: exec.budget() }` (the 1a adapter, moved here as `pub(crate) struct ExecSpmd`). Map `tensorcontract::Error` via `Error::Contract`.

Tests: the Task 3 suite parametrized over both strategies (a `for strategy in [FaerLoop, Tblis]` loop inside each test) with identical expectations (compare to the reference with 1e-12 relative tolerance, not bitwise, across strategies).

Commit "tprims-blas: TBLIS-style batched gemm through tensorcontract".

### Task 5: `trsm`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Side { Left, Right }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Uplo { Lower, Upper }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Op { N, T, C }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Diag { NonUnit, Unit }
pub fn trsm<T: Scalar>(exec: &Exec<'_>, side: Side, uplo: Uplo, op: Op, diag: Diag, alpha: T, a: &StridedView<'_, T>, b: &mut StridedViewMut<'_, T>) -> Result<()>;
```
Solves `op(A) X = alpha B` (Left) or `X op(A) = alpha B` (Right) in place in B. Construction: `a' = A` for N, `Aᵀ` for T and C (swap rs/cs, flip uplo), conj for C. Right side: `X op(A) = B  ⇔  op(A)ᵀ Xᵀ = Bᵀ`: transpose `a'` again (flip uplo again) and solve on `Bᵀ` (swap B's strides). Then `faer::linalg::triangular_solve::solve_{lower,unit_lower,upper,unit_upper}_triangular_in_place_with_conj(a_mat, conj, b_mat, par)`. alpha by `scale_in_place` first. Width from `n²·nrhs` flops. Singular diagonals are not checked (BLAS semantics; documented).

Tests (`tests/trsm.rs`): all 2×2×3×2 combinations of side/uplo/op/diag for f64 and c64 at n = 7, nrhs = 5: build a well-conditioned triangular A (diagonal 4 + i, off-diagonal small), random X, B = op(A)·X (or X·op(A)) via `naive_gemm`, solve with alpha = 1, compare X within 1e-10; alpha = 2 scales; strided/transposed B view; shape errors.

Commit "tprims-blas: trsm on faer".

### Task 6: Benchmarks (1T/4T)

`benchmarks/benchmarks/tprims/blas/blas.rs` (bin `blas`), reusing `tprims_bench::{threads, timing}`. Cases (f64 and c64): `gemm` n ∈ {8, 32, 128, 512, 1024} square col-major and one A-transposed variant at 512; `gemm_batched` × {FaerLoop, Tblis} for (size, batch) ∈ {(2,1024), (4,1024), (8,1024), (32,256), (128,16), (512,1)}; `trsm` Left/Lower/N at n ∈ {32, 256, 1024} with nrhs = n. Each row: `case,variant,threads,median_ns,samples`; a `CHECK` line per case comparing 4T and 1T results (rel err < 1e-12) and TBLIS vs FaerLoop. Run `taskset` 1T then 4T on idle cores of one L3 domain, record in `benchmarks/benchmarks/tprims/blas/README.md` with commit, CPU, idle check, and a derived table (1T, 4T, 4T/1T, TBLIS/faer ratio). Any 4T row not faster than 1T for a tensor-sized case is listed as a finding.

Commit "tprims-bench: blas benchmark (gemm, batched both strategies, trsm) at 1T and 4T".

### Task 7: Docs, gate, PR

Decision log: batched GEMM row gains the measured observation and the current `Auto = FaerLoop` rule; README crate table marks `tprims-blas` implemented. Gate as in 1a (fmt, clippy 1.98 default + parallel lanes, workspace tests, strided-alone lane, MSRV 1.89, docs). Whole-branch review by a fresh reviewer, one fix pass, push, PR, merge after green CI (maintainer approved merges).
