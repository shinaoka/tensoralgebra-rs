# Phase 1: tprims CPU parts (1a, 1b, 1d, 1c, 1f)

Date: 2026-09-29. Status: design derived from `README.md`, `docs/architecture.md`
and `docs/decision-log.md` (maintainer decisions of 2026-09-29) plus source
surveys of the imported code; pending maintainer review. Step 1e (tenferro
integration) is out of scope here.

Each step is one branch and one PR. Phase 0 is not merged when Phase 1
starts, so branches stack: `phase1a-exec` branches from
`phase0-consolidation`, `phase1b-blas` from `phase1a-exec`, and so on; each
PR's base is the previous step's branch. Plans are written per step, just
before the step starts, against the code the previous step produced.

## Common rules for every step

- Rules: `AGENTS.md`, `REPOSITORY_RULES.md`, `PERFORMANCE_TIPS.md` (ported in
  Phase 0). No ambient pool; serial work never enters a pool; faer `Par` only
  from `Exec`.
- Scalars: `f32`, `f64`, `Complex<f32>`, `Complex<f64>` (num-complex 0.4,
  identical to `faer::c32`/`c64`). Generic over a sealed `tprims` scalar
  trait, dtype dispatch only at outer boundaries (C ABI).
- Operands: `strided_view::StridedView` / `StridedViewMut` (signed element
  strides, offset), plus a per-operand `Conj` flag where the operation
  supports conjugation.
- Errors: one `thiserror` enum per crate, `#[non_exhaustive]`, no panics on
  user input.
- Benchmarks: every public operation and every alternative implementation
  gets rows in `tprims-bench`, measured at 1T (`Exec::Serial`) and 4T (a
  bounded 4-worker pool borrowed through `Exec`) in the same run, effective
  width asserted at startup, conflicting thread environment variables
  rejected. Results recorded under `docs/experiments.md` protocol with
  commit, CPU, `taskset` set and profile; numbers are not claims until
  recorded.
- Correctness before timing: reference or reconstruction/residual checks,
  noncontiguous and negative-stride views, empty and rectangular shapes,
  complex values.

## 1a: `tprims-exec` and explicit execution in strided

**Crate `crates/tprims-exec`** (depends on `rayon` only).

```rust
pub enum Exec<'a> {
    Serial,
    Rayon { pool: &'a rayon::ThreadPool, budget: NonZeroUsize },
    Host(&'a dyn BroadcastExecutor),
}
pub trait BroadcastExecutor: Sync {
    /// Runs `f(i)` for every i in 0..width concurrently (guaranteed
    /// co-scheduled, so barriers are allowed) and returns when all finish.
    fn broadcast(&self, width: usize, f: &(dyn Fn(usize) + Sync));
    fn max_width(&self) -> usize;
}
impl Exec<'_> {
    pub fn serial() -> Exec<'static>;
    pub fn rayon(pool: &rayon::ThreadPool) -> Exec<'_>;            // budget = pool size
    pub fn with_budget(self, max_threads: usize) -> Result<Self, ExecError>; // clamps, 0 is an error
    pub fn budget(&self) -> usize;                                   // 1 for Serial
    pub fn width_for(&self, work: Work, policy: &WidthPolicy) -> usize;
    /// Barrier-free partition: runs `f(i)` for i in 0..k, k <= budget.
    /// k == 1 runs inline on the caller. Enters the pool only if the caller
    /// is not already one of its workers.
    pub fn for_each_partition(&self, k: usize, f: &(dyn Fn(usize) + Sync));
    /// Runs `op` inside the pool (for faer `Par::rayon(k)`), inline when the
    /// caller is already a worker; `k == 1` never enters.
    pub fn install<R: Send>(&self, k: usize, op: impl FnOnce(Par) -> R + Send) -> R;
    /// SPMD with barriers: full dispatch width; returns Err(Unavailable)
    /// when the context cannot guarantee co-scheduling (nested inside the
    /// same pool, Serial) so the caller picks its barrier-free variant.
    pub fn broadcast(&self, width: usize, f: &(dyn Fn(usize) + Sync)) -> Result<(), ExecError>;
}
pub enum Par { Seq, Threads(NonZeroUsize) }   // mapped to faer::Par by consumers
pub struct Work { pub flops: f64, pub bytes: f64 }
pub struct WidthPolicy { pub serial_below_ns: f64, pub ns_per_flop: f64, pub entry_ns: f64 }
```

- Width rule: `k = 1` when estimated serial time < `serial_below_ns`
  (default 50 µs, from the rayon-entry measurement); otherwise the smallest k
  with `entry_ns(k) + T/k` minimal, capped by the budget. Constants live in
  `WidthPolicy::default()` and are replaced by measured values per kernel.
- `broadcast` on `Rayon` uses `ThreadPool::broadcast`; workers with index ≥
  `width` return immediately; `width` must be ≤ pool size (else
  repartition error). Concurrent broadcasts on one pool from different host
  threads are serialized by a mutex held by the `Exec` owner type
  (`RayonPool` wrapper) — the `Exec::Rayon` variant carries a reference to
  that wrapper rather than to a bare `ThreadPool` if the mutex is needed;
  the plan decides after the nested/concurrent SPMD tests.
- No thread-local ambient policy in `tprims-exec`.

**strided changes (in-repo, `strided/`):**
- `strided-basic::ExecContext` gains a borrowed-pool kind:
  `ExecContext::from_exec(&Exec)`; its decision point `rayon_threads()` reads
  the budget from the active context instead of `rayon::current_num_threads()`.
  The thread-local `ACTIVE_POLICY` stays as strided's internal carrier of the
  context down its recursion, but its root is always set from an explicit
  `ExecContext`; the `AmbientRayon` policy is kept for existing strided
  callers and documented as the one ambient mode, never used by tprims.
- Pool entry happens only in the parallel branches (`threading.rs`
  `join_with_policy`, `strided-perm` `hptt/execute.rs` parallel loops), via
  `Exec::install`, skipped when already on a worker. `ExecContext::run` does
  not enter a pool.
- `strided-perm` parallel entry points take the context
  (`copy_into_par_with(ctx, dst, src)`); the existing `_par` functions keep
  ambient behaviour for compatibility with published strided users.

**tensorcontract seam (in-repo, `tensorprimitives/`):** add a public
`Executor` hook: `Plan::run_with(exec: &dyn tensorcontract::Spmd, ...)` where
`Spmd::broadcast(p, f) -> bool` (false = cannot co-schedule p → fall back to
`execute_capped(.., 1)`, never to `thread::scope`, when called through the
tprims path). The existing `thread::scope` default stays for upstream users.
`tprims-exec` implements the trait for `Exec`. `TENSORCONTRACT_THREADS` does
not influence the tprims path; the width comes from `Plan::with_threads`
set from `Exec`.

**Done when:** tests for active width < pool width, partition above budget
(repartitioned, no new threads — count OS threads), nested SPMD (runs
barrier-free/serial, no deadlock, with a timeout), concurrent SPMD from two
host threads; a faer GEMM and a tensorcontract contraction run on a borrowed
pool with zero pool entries for serial-size work (entry counter);
`tprims-bench` bench `exec_entry`: entry cost vs width on a fixed 4-worker
pool, 1T/4T strided `copy`/`map` rows through `ExecContext::from_exec`.

## 1b: `tprims-blas`

```rust
pub fn gemm<T: Scalar>(exec: &Exec, alpha: T, a: MatIn<T>, b: MatIn<T>, beta: T, c: StridedViewMut<T>) -> Result<()>;
pub fn gemm_batched<T: Scalar>(exec: &Exec, alpha: T, a: BatchIn<T>, b: BatchIn<T>, beta: T, c: BatchOut<T>, strategy: BatchStrategy) -> Result<Selected>;
pub fn trsm<T: Scalar>(exec: &Exec, side: Side, uplo: Uplo, op: Op, diag: Diag, alpha: T, a: MatIn<T>, b: StridedViewMut<T>) -> Result<()>;
pub enum BatchStrategy { Auto, FaerLoop, Tblis }
```
- `MatIn` = rank-2 `StridedView` + `Conj`; `BatchIn` = rank-3 view, batch
  axis last (strided batch); a pointer-array/grouped variant follows only if
  1c needs it.
- faer GEMM via `matmul_with_conj` over `MatRef::from_raw_parts`; β=0 uses
  `Accum::Replace` and never reads C; β∉{0,1} scales C first, then
  `Accum::Add`; singleton strides normalized; zero/overlapping output strides
  rejected (faer UB).
- Batched faer-loop: items partitioned over `for_each_partition` when the
  batch is large, each item `Par::Seq`; few large items use one item at a
  time with inner `Par::Threads(k)`.
- Batched TBLIS: `tensorcontract::contract_batched` items sharing one
  `Plan` (H axis is serial inside a plan, so the batch axis is the item axis),
  driven through the 1a seam.
- TRSM on faer `solve_*_triangular_in_place_with_conj`; alpha by pre-scaling
  B; right side by transposed views; `Op::{N, T, C}`.
- `Auto` selection rule comes from the benchmark; until measured it is
  `FaerLoop`.
- **Bench:** `gemm` square/rectangular sizes 8..2048, f64/c64, contiguous
  and transposed strides; `gemm_batched` both strategies, batch 1..4096 at
  sizes 2..128; `trsm`. All 1T and 4T.

## 1d: `tprims-linalg`

Factor objects own their storage (column-major, compute dims left) and keep
faer's representation.

| Op | Outputs / object | faer |
| --- | --- | --- |
| `cholesky` | `Cholesky<T>` (L) + `solve` | `llt::factor::cholesky_in_place` |
| `ldlt` (Bunch–Kaufman) | `Ldl<T>` + `solve` | `lblt::factor::cholesky_in_place` |
| `lu` (partial pivot) | `Lu<T>` (packed LU, `perm/perm_inv: Vec<usize>`, parity) + `solve(op)` | `lu::partial_pivoting` |
| `lu_full` | `FullPivLu<T>` (row/col perms) + `solve(op)` | `lu::full_pivoting` |
| `qr`, `qr_col_piv` | `Qr<T>` (packed reflectors + block T factors) + `q_thin/q_full/r/apply_q/apply_qh/solve_lstsq` | `qr::{no,col}_pivoting`, `householder::apply_block_householder_*` |
| `svd`, `svd_values` | `U, S (real, nonincreasing), V` (V, not Vᴴ) | `svd::svd` |
| `eigh`, `eigh_values` | `w (real, nondecreasing), V` | `evd::self_adjoint_evd` (lower triangle) |
| `eig`, `eig_values` | complex `w`, right eigenvectors | `evd::evd_real` + tprims pair conversion, `evd_cplx` |
| `solve`, `lstsq`, `inv`, `det`, `logdet` | via the factor objects | |

- Error enum: `NotPositiveDefinite { index }`, `Singular { index }`,
  `NoConvergence`, `InvalidShape`, `NonFinite`. One singularity rule for all
  solves: `|u_ii| <= eps * max_j |u_jj| * n` reports `Singular`. Factorization
  itself never errors on singular input (LAPACK `getrf` semantics).
- Real eig pairs: `|im| <= eps * max(|re|, 1)` is real; otherwise
  `v = U[:,j] ± i U[:,j+1]`.
- Ordering contracts are faer's and are documented, not re-sorted.
- `batched` module: equal-shape batches (batch axis last), per-item status
  array, one `StackReq::any_of` scratch sized once per call and reused, no
  per-item allocation, batch partitioned over `Exec` with `Par::Seq` per item
  for large batches.
- Tests adapted from tenferro's residual/reconstruction tests (reimplemented,
  no copying; same maintainers, MIT OR Apache-2.0).
- **Bench:** each op at n = 2, 4, 8, 32, 128, 512, batch 1 and 1024
  (small-matrix large-batch regression case), f64/c64, 1T/4T.

## 1c: `tprims-contract`

```rust
pub struct DotGeneral { pub lhs_contract: Vec<usize>, pub rhs_contract: Vec<usize>, pub lhs_batch: Vec<usize>, pub rhs_batch: Vec<usize> }
// output axis order: [lhs_free..., rhs_free..., batch...]  (tenferro convention)
pub struct ContractPlan<T> { ... }   // validated shapes/strides, chosen strategy, owned tensorcontract::Plan when TBLIS
impl<T: Scalar> ContractPlan<T> {
    pub fn new(cfg: &DotGeneral, a: &Layout, b: &Layout, c: &Layout, conj: (Conj, Conj), strategy: Strategy, flags: Flags) -> Result<Self>;
    pub fn selected(&self) -> Selected;          // PermuteGemm { materialized: [bool; 2] } | Tblis
    pub fn execute(&self, exec: &Exec, alpha: T, a: StridedView<T>, b: StridedView<T>, beta: T, c: StridedViewMut<T>) -> Result<()>;
}
pub enum Strategy { Auto, PermuteGemm, Tblis }
pub struct Flags { pub no_materialize: bool }
```
- PermuteGemm port of tenferro `dot_runtime.rs` + `gemm/mod.rs` at
  `5a4e7fd`: `try_fuse_dims`, `analyse_gemm` (copy-free batched matrix view),
  otherwise `canonical_gemm_layout` and materialization through
  `strided-perm`/`strided-basic` copy (conjugating copy when needed), then
  `tprims-blas::gemm_batched`. All-batch contractions use an elementwise
  path. `no_materialize` turns a needed copy into an error.
- Tblis: label mapping — lhs axes get labels `0..ra`, rhs contracted axes
  reuse lhs labels, batch axes shared by all three, output labels in
  `[lhs_free, rhs_free, batch]` order; the `tensorcontract::Plan` is built
  once in `ContractPlan::new` and run through the 1a seam.
- Plan cost is kept out of `execute`; scratch reuse across calls is Phase 3.
- Thin wrappers: `permute`, `add`, `trace`.
- **Bench corpus** (predeclared, in `benchmarks/.../contract/cases.json`):
  matmul-like, batched, permuted, irregular strides, tiny (2^3) to large
  (2^24 elements), real and complex; both strategies, plan and execute timed
  separately, 1T/4T.

## 1f: C ABI slice

- `crates/tprims-core`: DLPack 1.x `#[repr(C)]` mirror (`DLTensor`,
  `DLManagedTensorVersioned`, flags), `tprims_tensor`, `tprims_status`,
  thread-local `tprims_last_error`, `tprims_abi_version`, `tprims_has_part`,
  executor handles (`serial`, `rayon_create` with joined close via
  `spawn_handler`, `set_budget`, `retain/release`, `close` with
  `TPRIMS_BUSY` / `TPRIMS_ERR_WOULD_DEADLOCK` / `TPRIMS_ERR_CLOSED`).
- `crates/tprims-blas-capi`: `tprims_blas_gemm`, `tprims_blas_gemm_batched`.
- `crates/tprims-contract-capi`: plan create/execute/destroy.
- `crates/tprims-bundle`: `cdylib` + `staticlib`, features `blas`,
  `contract`; generated `tprims/*.h` checked by a C compile test.
- C benchmark harness `benchmarks/c/`: empty call, small GEMM, small
  contraction, serial and 4-thread pool, against the same Rust calls;
  pointer-identity zero-copy checks for strided and column-major inputs.
- Every entry catches panics; read-only outputs rejected before writes.

## Order and dependencies

1a → 1b → 1d → 1c → 1f, each stacked on the previous branch. 1c depends on
1b's `gemm_batched`; 1d depends on 1a/1b only. A plan may reorder tasks
inside a step but not across steps.
