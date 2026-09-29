# Architecture design notes

[Back to the visual overview](../README.md). These notes retain the detailed rationale, numerical requirements, and primary sources.

An experimental design space for a CPU algebra stack: explicit execution, strided kernels, matrix multiplication, dense linear algebra including batches, and binary tensor contraction. The stack is meant for any Rust, C, Julia or Python host, not for a single consumer. AI-assisted design and implementation are welcome; contributors remain responsible for correctness, measurements, and code provenance. There is no stable API, ABI or performance claim yet.

## Working hypotheses

- **Large GEMM:** Evaluate a BLIS-style packed-panel design and existing BLIS, `gemm`, and gemmkit providers before writing architecture-specific kernels.
- **Tensor contraction:** Port two strategies behind one plan API and compare them: tenferro-rs's permute plus batched GEMM, and the TBLIS-style direct contraction of tensorprimitives-rs. Include irregular strides and complex values.
- **Small matrices and batches:** Start from faer with a loop over items (batched GEMM and batched linear algebra), scheduled on the outer batch axis. Test the crossover to per-matrix parallelism, and to the TBLIS-style kernel for batched GEMM, rather than assuming one rule fits every size.
- **Execution ownership:** Make the effective thread budget, pool, and scratch lifetime explicit. A Rust caller supplies an execution context. A C/Julia/Python caller can create, use and close a pool through the C ABI, or inject its own scheduler, without the library silently taking over the host's threads.

These are questions, not settled design decisions. [The research map](research-map.md) distinguishes published evidence from project-specific hypotheses. [The experiment plan](experiments.md) defines the first three independent prototypes and how to compare them.

## Design principles

The full statement and rationale are in [design principles](design-principles.md). In short:

1. **Parts, no facade.** Every crate is independently usable and publishable. No crate re-exports the stack as a whole.
2. **Short names under one prefix.** Crates are named `tprims-<part>`; the part name says what it does. Crate, header and C symbol map one to one: `tprims-blas`, `tprims/blas.h`, `tprims_blas_*`. The prefix avoids claiming generic crates.io names and identifies the family for users outside the original project.
3. **C ABI per part, one library per build.** Each part owns its C ABI crate, where its semantics are tested. C ABI crates are `rlib` only. `tprims-bundle` links the selected parts into one `cdylib`/`staticlib`, `libtprims`.
4. **Zero copy at the boundary.** Data crosses the C ABI as DLPack descriptors. The library never copies an input or output merely to cross the boundary.
5. **Caller-owned execution.** Every expensive operation receives an explicit execution context. No part uses an ambient global pool.

## Crates

| Crate | Responsibility | Depends on |
| --- | --- | --- |
| `tprims-exec` | Execution context: serial, a Rayon pool borrowed from the host (or created by a C host), host scheduling callbacks with guaranteed width; thread budget for nested parallelism; scratch-size queries and reusable per-worker scratch. | none in the stack |
| `strided-traits`, `strided-view` (existing) | Checked borrowed strided views, scalar and conjugation contracts, rank-2 views of the same storage. | none |
| `strided-perm`, `strided-kernel` (existing) | Copy and permutation (HPTT-inspired); map, reduce, broadcast and fused elementwise kernels. Execution and scratch become explicit through `tprims-exec`. | `strided-view`, `tprims-exec` |
| `tprims-gemm-kernel` | Packed A/B panel format, scalar fallback, ISA dispatch, microkernels computing bounded result tiles, initially ported from tensorprimitives-rs `tensorcontract`. No scheduler, no full-matrix GEMM, no tensor labels. | `strided-view` |
| `tprims-blas` | GEMM and batched GEMM with two implementations to compare (faer plus a loop over items; TBLIS-style packing on `tprims-gemm-kernel`), TRSM, then SYRK/HERK and further BLAS-like operations as needed (GEMV is not excluded by the name). | faer, `tprims-gemm-kernel`, `tprims-exec` |
| `tprims-linalg` | Dense linear algebra: factorizations, direct solves, least squares, symmetric/Hermitian eigendecomposition, factor objects, errors, workspace plans; `batched` module. Initially faer per item. | faer, `tprims-blas`, `strided-kernel`, `tprims-exec` |
| `tprims-contract` | Binary contraction with free, contracted and batch indices (`dot_general` semantics): one plan API over two strategies, permute plus batched GEMM and TBLIS-style direct packing with bounded scatter. Thin permute / add / trace wrappers and matricized factorization wrappers. | `tprims-gemm-kernel`, `tprims-blas`, `tprims-linalg`, `strided-perm`, `strided-kernel`, `tprims-exec` |

The name `contract` was chosen over `tensordot` because NumPy, PyTorch and JAX `tensordot` has no batch indices; the operation here does, as in cuTENSOR's `cutensorContract`.

### C ABI crates

| Crate | Exposes | Header |
| --- | --- | --- |
| `tprims-core` | DLPack types (mirroring the upstream `dlpack.h`), `tprims_status`, thread-local last-error message, execution handles, ABI version queries | `tprims/core.h` |
| `strided-capi` | Permute, copy, elementwise and reduce over DLPack operands | `tprims/strided.h` |
| `tprims-blas-capi` | GEMM, batched GEMM, TRSM, SYRK/HERK | `tprims/blas.h` |
| `tprims-linalg-capi` | Factor objects as opaque handles, solves, `lstsq`, `eigh`, batched entries | `tprims/linalg.h` |
| `tprims-contract-capi` | Contraction plans and execution | `tprims/contract.h` |
| `tprims-bundle` | No API. `cdylib` + `staticlib` with one feature per part; installs the selected headers, a generated umbrella `tprims/tprims.h`, and a pkg-config file | |

`tprims-gemm-kernel` has no C ABI: its packed format is an internal contract between drivers.

### Dependency rules

- `tprims-exec` and `tprims-core` are the bottom of the graph. `strided-*` crates may depend on them and on nothing else in the stack.
- `tprims-gemm-kernel` depends only on `strided-view` / `strided-traits`. It takes no execution context.
- `tprims-blas`, `tprims-linalg` and `tprims-contract` take an explicit `tprims-exec` context for every expensive operation.
- A C ABI crate depends on its Rust part and `tprims-core` only. It contains no algorithm.
- No cycle between crates. See the decision log for the repository placement of `tprims-exec` and `tprims-core`.

### What is excluded

- **N-ary einsum.** Index notation and contraction-order planning stay in `strided-opteinsum` or in the consumer and call `tprims-contract` for each binary step. The stack exposes no index-string API.
- **Iterative (Krylov) solvers.** CG, GMRES, Lanczos and Davidson need a linear-operator callback, convergence control and preconditioning, a contract distinct from dense linear algebra. Many consumers already carry their own. Deferred; a later `tprims-krylov` would depend on `tprims-blas` without changing this layout.
- **AD, traced execution, device transfer, GPU backends**, and adapters for `ndarray` / `mdarray`. These sit above the stack.
- **Tensor-level numerical algorithms.** `tprims-contract` does not implement pivoting, convergence, or scaling. A tensor SVD is a matricized call into `tprims-linalg`.

## One shared library

### Why not one shared library per part

- An opaque handle is a Rust type. Two shared libraries each contain their own copy of `tprims-exec`, so a pool created in one is not a valid object in the other.
- Each copy would carry its own Rayon runtime, defeating a single thread budget.
- Two Rust static libraries linked into one C program duplicate Rust standard library symbols.

Each part therefore provides its `extern "C"` functions in an `rlib`, and `tprims-bundle` selects parts through Cargo features:

```toml
[lib]
name = "tprims"
crate-type = ["cdylib", "staticlib"]

[dependencies]
tprims-core = { path = "../tprims-core" }
tprims-blas-capi = { path = "../tprims-blas-capi", optional = true }

[features]
blas = ["dep:tprims-blas-capi"]
```

The bundle's `lib.rs` names each selected crate (`pub use tprims_blas_capi;`) so that it is linked. A local prototype on 2026-09-29 (macOS, stable Rust) confirmed that only the selected parts' `#[no_mangle]` symbols are exported and that a handle created by the core crate is accepted by two different parts. Linux and Windows export behaviour must be checked in CI before the design relies on it.

### ABI conventions

- Symbols: `tprims_<part>_<operation>`; core symbols are `tprims_<object>_<operation>` (for example `tprims_exec_close`).
- Every entry point catches panics and returns `tprims_status`; `tprims_last_error()` returns a thread-local message.
- `tprims_abi_version()` and `tprims_has_part("blas")` let a host check at run time what the loaded library contains. Headers carry the matching version macro.
- One bundle pins one version of each part. Symbol versioning policy is decided before the first ABI release.

## Zero-copy data exchange

The C ABI uses [DLPack](https://dmlc.github.io/dlpack/latest/) so that NumPy, PyTorch, JAX, CuPy and Julia arrays pass without copying.

- **Operand descriptor.** Every tensor argument is a borrowed `tprims_tensor`, a view plus the DLPack flags that `DLTensor` alone does not carry (`DLPACK_FLAG_BITMASK_READ_ONLY` lives in `DLManagedTensorVersioned.flags`):

  ```c
  typedef struct {
      DLTensor *view;      /* borrowed; never freed by tprims */
      uint64_t  flags;     /* DLPACK_FLAG_BITMASK_* */
  } tprims_tensor;

  /* Borrow a versioned tensor: copies view pointer and flags; takes no ownership, never calls the deleter. */
  tprims_tensor tprims_tensor_borrow_versioned(DLManagedTensorVersioned *m);
  /* Raw DLTensor (DLPack 0.x producers): the caller asserts the memory is writable if used as an output. */
  tprims_tensor tprims_tensor_borrow_raw(DLTensor *t, uint64_t flags);
  ```

  The descriptor is valid only for the duration of the call; ownership and lifetime stay with the caller.
- **Inputs** are `tprims_tensor` read through `view`; the READ_ONLY flag is permitted and no copy is made. The library honours `byte_offset`, arbitrary element strides (column-major and negative strides included) and a NULL `strides` meaning compact row-major. `lanes` must be 1. Accepted devices are `kDLCPU` and `kDLCUDAHost`.
- **Caller-provided outputs** are `tprims_tensor` written in place with `C = alpha * op(A, B) + beta * C` semantics and defined zero-size behaviour. Outputs must not overlap inputs unless an operation documents in-place support; overlap is checked where affordable.
- **Library-allocated outputs** are returned as `DLManagedTensorVersioned` (DLPack 1.x) with a deleter, for results whose shape or storage the library determines (factors, `eigh` results). The receiver owns them and can hand them to NumPy or Julia without a copy.
- **Conjugation** is a per-operand argument, since DLPack has no conjugation flag. It maps to the lazy conjugation already present in `strided-view`.
- **Read-only outputs** are rejected with `TPRIMS_ERR_READ_ONLY` during validation, before any write. For a descriptor made with `tprims_tensor_borrow_raw`, the check sees only the flags the caller passed; writability of the memory is the caller's precondition.
- **Materialization** is never hidden. When a stride layout forces an internal copy, the call reports the selected strategy; the flag `TPRIMS_NO_MATERIALIZE` turns that case into an error.
- **Dtypes** in ABI v1: `f32`, `f64`, `complex64`, `complex128`. Tag space for `f16` / `bf16` exists in DLPack; whether v1 accepts them is an open decision.

The DLPack header is Apache-2.0; `tprims-core` mirrors its `#[repr(C)]` layout and records the upstream version.

## Execution context

`tprims-exec` defines the Rust contract and `tprims-core` its C face. A context is one of:

- **Serial:** work runs on the calling thread.
- **Rayon pool:** a pool borrowed from the host (for example the pool of tenferro-rs) for the duration of a call, or created through `tprims_exec_rayon_create` by a C host that has none. The global Rayon pool is never used implicitly.
- **Host callbacks:** a vtable through which a Julia, Python or C host schedules tasks on its own threads.

```c
tprims_exec *tprims_exec_serial(void);
tprims_exec *tprims_exec_rayon_create(size_t nthreads, const tprims_rayon_opts *opts); /* thread name, stack size, pinning hint */
tprims_exec *tprims_exec_from_callbacks(const tprims_exec_vtable *host);
size_t       tprims_exec_num_threads(const tprims_exec *exec);
tprims_status tprims_exec_set_budget(tprims_exec *exec, size_t max_threads);
tprims_status tprims_exec_close(tprims_exec *exec);   /* owned pool: joins workers; borrowed: detaches */
void         tprims_exec_retain(tprims_exec *exec);
void         tprims_exec_release(tprims_exec *exec);
```

- **Lifetime:** handles are reference counted so that a host binding and several library objects can share one pool.
- **Close of an owned pool is synchronous and joins the threads.** A pool created by `tprims_exec_rayon_create` is built with `ThreadPoolBuilder::spawn_handler`, which spawns each worker with `std::thread::Builder` and keeps its `JoinHandle`. `tprims_exec_close` marks the context closed, returns `TPRIMS_BUSY` if work is in flight, otherwise drops the pool to start shutdown and joins every handle. An exit-handler notification is not sufficient: in rayon-core 1.13 it runs inside the worker's main loop, before thread-local destructors ([probe](../experiments/pool-close/README.md)). After a successful close no tprims worker code, including TLS teardown, is running.
- **Close of a borrowed context only detaches.** A context wrapping a host pool (the Rust `Exec::Rayon` case, or a C host's callback executor) never terminates the host's threads; close waits for tprims calls in flight on that context (or returns `TPRIMS_BUSY`) and then invalidates the handle.
- **Edge cases.** Close called from a worker of the same pool returns `TPRIMS_ERR_WOULD_DEADLOCK` without side effects (detected with `ThreadPool::current_thread_index`). A second close is a no-op returning `TPRIMS_OK`. Calls on a closed context return `TPRIMS_ERR_CLOSED`. `release` of the last reference to an owned pool that was never closed performs the close; if that happens on a worker of the pool, the join is handed to a detached reaper thread and reported through a debug status, since joining is impossible there.
- **Tests when implemented:** a TLS-destructor handshake with a bounded wait (as in the probe) proving close does not return early; busy, repeated and self-worker close.
- **Thread budget:** one budget controls batch-level and inner-matrix parallelism so nested parallelism does not oversubscribe.
- **Scratch:** operations expose scratch-size queries; a context can own reusable per-worker scratch.

**Entry happens inside kernels, only for parallel work.** The calling thread drives every call. A kernel chooses its width from the amount of work; at width one it runs on the calling thread and never touches the pool. Only a parallel kernel enters the pool, and not at all if the calling thread is already one of its workers. With this rule an entry cost of about 10 µs is acceptable ([measurement](../experiments/rayon-entry/README.md), [decision](decision-log.md#execution)), and faer can serve as the initial backend: `Par::Seq` for serial work, `install` followed by `Par::rayon(n)` for parallel work.

```rust
pub enum Exec<'a> {
    Serial,
    Rayon(&'a rayon::ThreadPool),       // borrowed from the host
    Host(&'a dyn BroadcastExecutor),    // host scheduler with guaranteed width
}
```

**Three widths, kept distinct.**

- *Budget* `b`: the most threads a kernel may occupy, set by the host (`tprims_exec_set_budget`, never above the pool size). tprims never creates threads beyond the pool; there is no scoped-thread fallback.
- *Active width* `k`: the number of workers doing arithmetic, chosen from the work, `k ≤ b`.
- *Dispatch width* `d`: the number of workers woken. It sets the entry cost. For Rayon `ThreadPool::broadcast`, `d` is always the whole pool, whatever `k` is: on a fixed 18-worker pool, a broadcast with one active worker costs as much as one with 18 (about 165 µs from idle, [measurement](../experiments/rayon-entry/README.md#active-width-on-a-fixed-pool)).

Two execution shapes follow:

- **Barrier-free partition** (batches, independent output tiles, faer's own parallel loops): `install` plus `k` tasks, so `d ≈ k`. Entry scales with `k` (about 17 µs for one task, 50 µs for four, from idle). Tasks are not guaranteed to run concurrently, so no barrier may be used.
- **SPMD with barriers** (the TBLIS inner driver): needs `k` workers guaranteed to run at once. On Rayon this is `broadcast`, so `k = d =` pool width, and it is chosen only when the kernel is large enough to amortize the full-pool entry. A narrower SPMD team on a borrowed Rayon pool would need a subset-broadcast primitive that Rayon lacks; that is a separate prototype, not an assumption.

**Insufficient width.** If a plan's partition needs more workers than `b`, the planner repartitions to at most `b` (down to serial) before execution. It never spawns extra threads and never enters a barrier with fewer participants than the barrier counts.

**Nesting.** An SPMD kernel called from a worker that is already inside an SPMD region, or inside any job of the same pool, runs its barrier-free variant or serially: a worker blocked at an outer barrier could not take its share of an inner broadcast. Concurrent SPMD kernels on one pool from different host threads are serialized by the context. Both rules are tested before the adapter is used as a general borrowed-pool solution.

### Cost of parallel execution

A parallel kernel pays a fixed entry cost before any work is shared. A simple model for work `T` split over `n` threads is

```text
T_par(n) ≈ L(n, state) + T / n
```

where `L` depends on the width and on whether the workers are awake. The [rayon-entry measurement](../experiments/rayon-entry/README.md) (M5 Max, 2026-09-29) splits `L` into three parts:

| Part | Cost | Cause |
| --- | --- | --- |
| Caller wake-up | about 3 µs | The caller outside the pool blocks on a lock latch (a `Condvar`) and must be woken when the job finishes. Same as a raw two-thread `Condvar` round trip. |
| Worker wake-up | 6 to 8 µs | Idle workers sleep after 32 `yield_now` rounds, within 100 µs; Rayon wakes one sleeping worker per injected job. |
| Fan-out | 55 to 200 µs at 18 threads | Waking every worker; the heterogeneous efficiency cores likely contribute. About 8 to 17 µs at 4 threads. Grows with the dispatch width, not the active width. |
| Already inside the pool | 8 to 33 ns | No handoff: the job runs on the current worker. |

Consequences for the design:

- **Break-even.** Parallelism gains only if `T - T/n > L`. With `L` about 10 µs at small width, work below roughly 50 to 100 µs should run serially on the calling thread; an `N = 100` matrix-vector product (about 1 µs) is always serial. Full width pays off only for work of the order of milliseconds.
- **Width from work.** A kernel picks `n` from its flop and byte count, not from the pool size, so a medium kernel wakes a few workers rather than all of them. This holds for barrier-free partitions only; an SPMD kernel on Rayon always pays full-pool dispatch (see *Three widths* above), so its threshold is higher. The thresholds are per kernel and per machine and must be measured; they are not fixed constants of the API.
- **No per-call handoff for serial work.** This is the difference from tenferro-rs today, which enters its pool once per session and so pays 8 to 14 µs on every FFI call, including serial ones ([tenferro #1945](https://github.com/tensor4all/tenferro-rs/issues/1945)).
- **OpenMP is not intrinsically cheaper.** A default libomp parallel region costs 10 to 12 µs at 4 threads and 53 to 94 µs at 18 threads, the same range as Rayon. Sub-microsecond OpenMP entry comes from active waiting (`KMP_BLOCKTIME`), which burns cores; at 18 threads (all cores) its median rose to 59 µs and p90 to 1.1 ms after a 10 ms idle gap. A spin policy is therefore deferred ([decision](decision-log.md#execution)).
- **Chains of small parallel kernels** are the case the model penalizes most: each pays `L`. The remedy is fusing them into one parallel region (a batch, or an SPMD kernel with barriers), not lowering `L`.

Not yet measured: real GEMM and contraction break-even against width, barrier cost inside one SPMD call, a subset-broadcast primitive, a homogeneous x86 CPU, and the fixed cost through the C ABI ([Prototype 4](experiments.md#prototype-4-c-abi-slice)).

TBLIS also uses cooperating threads and barriers inside a blocked contraction. An arbitrary task-submission interface, including host callbacks, does not automatically provide that contract. Start with outer-batch parallelism and serial inner contractions, then prototype an explicitly synchronized inner driver on a Rayon context if large contractions need it.

## Two contraction strategies

`tprims-contract` ports two existing implementations behind one plan API and compares them on a predeclared corpus. Neither is assumed to win.

| Strategy | Source | Idea |
| --- | --- | --- |
| Permute plus batched GEMM | tenferro-rs `dot_general` (`tenferro-cpu/src/dot_runtime.rs`, `gemm/`), MIT OR Apache-2.0 | Fold compatible strides into a batched matrix view without copying when possible; otherwise materialize operands through `strided-perm`, then call `tprims-blas` batched GEMM. |
| TBLIS-style direct | tensorprimitives-rs `tensorcontract` by Lukas Devos, MIT OR Apache-2.0 | Pack tensor panels with general strides directly into the `tprims-gemm-kernel` format, run the microkernels, and scatter bounded output tiles. No full operand transpose. [Matthews, TBLIS](https://arxiv.org/abs/1607.00291). |

A plan validates free-left, contracted, free-right and batch indices, output shape and aliasing, then selects a strategy. Planning may fold contiguous dimensions and reorder logical traversal without changing user-visible index order. The comparison reports end-to-end time, bytes moved and scratch, for tiny contractions where entry dominates and for large ones where packing and cache behavior dominate.

The TBLIS strategy runs cooperating workers with barriers inside one contraction. It therefore needs guaranteed concurrent width from `tprims-exec`, not arbitrary task submission. The Rayon `ThreadPool::broadcast` adapter proposed in [tensorprimitives-rs #1](https://github.com/lkdvos/tensorprimitives-rs/issues/1) provides it at full pool width only: workers with index at or above the active width return immediately but are still dispatched and awaited. tprims therefore uses it only for contractions large enough to amortize full-pool entry, repartitions to the budget instead of that proposal's scoped-thread fallback, and otherwise runs a barrier-free partition (independent output tiles, each worker packing its own panels). See [three widths](#execution-context).

The ported source keeps its authorship, commit history and license notices; the import mechanism (for example `git subtree` without squashing) is chosen when the code is brought in, after the design is settled. See the [provenance policy](provenance.md).

## Dense linear algebra

`tprims-linalg` owns pivoting, scaling, factor storage, convergence, and solve semantics. It uses `tprims-blas` for large trailing updates and small direct/panel kernels where GEMM is a poor fit.

### Factorizations

| Operation | Starting algorithm | Batched baseline |
| --- | --- | --- |
| Cholesky | Small unblocked factorization with a positive-definiteness check; larger blocked panels with TRSM and SYRK/HERK or GEMM updates. [LAPACK `POTRF`](https://netlib.org/lapack/explore-html/d2/d09/group__potrf_ga84e90859b02139934b166e579dd211d4.html). | One serial factorization per matrix; report the failing matrix and pivot. |
| LU | Partial-pivot panel, row swaps, TRSM, GEMM trailing update; retain pivots and singularity status. [LAPACK `GETRF`](https://www.netlib.org/lapack/explore-html/db/d04/group__getrf_gaea332d65e208d833716b405ea2a1ab69.html). | Outer-batch parallelism first, with independent pivot arrays. |
| LDLᴴ | Symmetric-indefinite factorization with Bunch-Kaufman pivoting (LAPACK `SYTRF`/`HETRF`). | Reuse the serial routine first. |
| QR | Householder reflectors, small unblocked path, compact blocked application and trailing update; column pivoting as a variant. Store reflectors/`tau`; generate Q only on request. [LAPACK `GEQRF`](https://www.netlib.org/lapack/explore-html/d3/d69/dgeqrf_8f_source.html). | Reuse scalar QR first; compare a specialized small-matrix route later. |
| SVD | Scaling, bidiagonal reduction, a robust bidiagonal solver, and singular-vector back-transformation. GEMM assists updates but does not provide accuracy or convergence. [LAPACK SVD overview](https://www.netlib.org/lapack/lug/node53.html). | Begin with a validated provider or correct serial implementation per item; defer specialized batched SVD until justified. |
| Symmetric / Hermitian eigen | Tridiagonal reduction and a tridiagonal solver (LAPACK `SYEVD`/`HEEVD` family). | Per-item serial first. |
| Nonsymmetric eigen | Phase 1: a thin wrapper over faer's nonsymmetric eigendecomposition (complex eigenvalues and eigenvectors), because tenferro's CPU backend exposes `eig`. A native solver (Hessenberg reduction, shifted QR, LAPACK `GEEV`) is deferred. | Per-item faer. |

### Solves

Solves live in the same crate as the factorizations because they operate on each factorization's internal representation (pivot arrays, reflectors and `tau`, blocked storage). Splitting them out would turn that representation into a public API and, through the C ABI, into a frozen contract. In C, a factor object is an opaque handle.

- Factor-object solves: `lu.solve`, `cholesky.solve`, `ldl.solve` (LAPACK `GETRS`/`POTRS`/`SYTRS` equivalents), with transpose and conjugate-transpose variants.
- One-shot `solve(A, B)`, least squares `lstsq` (QR for full rank, SVD or pivoted QR for rank-deficient input with an explicit rank tolerance), `inv`, `det` and `logdet`.
- Triangular solve with a matrix right-hand side is TRSM in `tprims-blas`.

Start with `f32`/`f64`, then complex arithmetic with explicit conjugation behavior. Require reconstruction/solve residuals, QR orthogonality, rank-deficient, indefinite and non-positive-definite inputs, extreme scales, and convergence status before timing. SVD and eigensolvers are separate numerical workstreams, not straightforward GEMM extensions.

Tensor-level factorizations in `tprims-contract` reshape a strided tensor into a rank-2 view over the same storage when the split is stride-compatible, or materialize through `strided-perm` otherwise, then call `tprims-linalg`. The wrapper reports which of the two it did and adds no numerical logic.

### Batched execution

`tprims-linalg` contains a `batched` module with factorization and solve entry points; `tprims-blas` owns batched GEMM. The batch module owns batch descriptors, output/status arrays, scratch planning, and the choice of batch versus inner-matrix parallelism. It executes the schedule on the caller's `tprims-exec` context and reuses per-matrix routines as its first implementation. Specialized small-matrix or interleaved batch kernels can later replace the per-item implementation under the same batch contract; they require their own correctness and performance evidence.

**Phase 1 baseline: faer plus a loop.** Every batched operation first runs the faer per-matrix routine in a loop over items: serial per item on the calling thread for a small batch, the items distributed over the borrowed pool for a large batch, and faer's inner `Par::rayon(n)` only for a few large matrices. Batched GEMM additionally has the TBLIS-style implementation, compared against the faer loop.

The first API targets equally shaped strided matrices, expressed in C as a DLPack tensor with a leading batch axis; grouped heterogeneous shapes can follow. A prepared batch plan can reuse validated descriptors and one scratch region per worker. Schedule **one serial matrix operation per independent task** initially. For a small batch of large matrices, measure inner-matrix threading instead. A single C ABI batch call amortizes call and executor-entry costs.

Return a per-item status and define the contents of failed outputs. Batched GEMM should compare arrays of matrices, interleaved/structure-of-arrays layouts, and dedicated small-matrix SIMD kernels. A common scheduler does not imply one data layout or microkernel. [Haidar et al. §4.1](https://www.netlib.org/utk/people/JackDongarra/PAPERS/batched-matrix-comp.pdf) motivates serial-per-matrix CPU execution; [Deshmukh et al.](https://arxiv.org/abs/2311.07602) motivates cache- and shape-specific batch GEMM. Their results do not establish a universal crossover or a CPU batched-SVD algorithm.

## Relationship to `strided-rs`

The [current `strided-rs` workspace](https://github.com/tensor4all/strided-rs/blob/main/Cargo.toml) provides views, basic and fused kernels, HPTT-inspired permutation, binary einsum, N-ary planning, adapters, and a facade. Its crates keep their names and become the strided part of this stack. Proposed changes:

| Current component | Proposed role and review |
| --- | --- |
| `strided-traits`, `strided-view` | Reused unchanged as the shared view contract. Preserve checked borrowing and lazy conjugation. |
| `strided-perm`, `strided-kernel` | Gain an explicit `tprims-exec` context where they currently rely on ambient threading. Preserve [HPTT provenance and license](https://github.com/tensor4all/strided-rs/blob/main/docs/PROVENANCE_AND_CITATION_POLICY.md). |
| `strided-capi` (planned in [strided-rs #234](https://github.com/tensor4all/strided-rs/issues/234)) | Becomes an `rlib` over `tprims-core` types (DLPack operands, `tprims_exec`) and joins `tprims-bundle`, instead of shipping its own `cdylib`/`staticlib`. |
| `strided-einsum2` | Its binary contraction role is taken over by `tprims-contract`; semantic tests and provider comparisons move there once equivalent. |
| `strided-opteinsum` | Stays as the N-ary planner; its binary step is retargeted to `tprims-contract`. |
| `mdarray-opteinsum`, `ndarray-opteinsum`, `strided-rs` facade | Thin adapters above the stack. Keep current consumers, including tenferro, pinned until equivalent correctness and performance are verified. |

Current binary einsum [passes operands through a contiguous-preparation path before GEMM](https://github.com/tensor4all/strided-rs/blob/main/strided-einsum2/src/lib.rs); compatible views may avoid a copy, but a direct tensor packer is a different architecture. Moving Julia/HPTT-derived code also requires preserving its source attribution and file-level licenses under the [provenance policy](provenance.md). No `strided-rs` code or tests have been copied into this repository.

## Implementation order

Phase 1 puts being usable as the tenferro-rs CPU backend first. It also builds a thin C ABI slice and benchmarks it, to find out early whether the design holds across the C boundary; full C ABI coverage follows in Phase 2.

| Phase | Content | Done when |
| --- | --- | --- |
| 1a | `tprims-exec`: borrowed Rayon pool, width chosen from work, kernel-level entry, `broadcast(n, f)`. | tenferro's pool runs a faer kernel and a TBLIS SPMD kernel through `tprims-exec`, with no entry for serial work. Tests: active width below pool width, partition above budget (repartitioned, no new threads, no barrier deadlock), nested and concurrent SPMD. Benchmark: fixed large pool, several active widths. |
| 1b | `tprims-blas`: GEMM and batched GEMM, faer plus loop and TBLIS-style; TRSM on faer (heavily used by AD rules). | Both batched GEMM implementations pass the same correctness suite; a measured selection rule by shape. |
| 1c | `tprims-contract`: permute plus batched GEMM (ported from tenferro) and TBLIS-style direct (ported from tensorprimitives-rs); batch dimensions, conjugation, `alpha`/`beta`. | Both strategies agree with a reference; comparison recorded on a predeclared corpus. |
| 1d | `tprims-linalg`: faer per item plus batched loops, covering the tenferro CPU linear algebra operations (Cholesky, triangular solve, LU and full-pivot LU families with solves, QR and Householder operations, SVD, eigh, and nonsymmetric eig as a faer wrapper). | tenferro's linear algebra and AD rule tests pass, including nonsymmetric `eig` cases. |
| 1e | tenferro-rs integration behind a `cpu-tprims` feature: `dot_general`, grouped / batched GEMM, then linear algebra. An operation table maps each tenferro CPU op to tprims or to the existing tenferro backend as an explicit fallback. | A/B correctness against the current backend for every op routed to tprims; fallback ops pass tenferro's suite unchanged with the feature on; a same-run performance gate. |
| 1f | C ABI slice: `tprims-core` (DLPack types, status, `tprims_exec` create / borrow / close), `tprims-blas-capi` (GEMM, batched GEMM), `tprims-contract-capi`, and `tprims-bundle` with those features. A C benchmark harness. | From C: per-call fixed cost of a small GEMM and contraction against direct Rust, zero copy verified for strided and column-major DLPack inputs, pool create / use / close, cross-part handles, recorded under the experiment protocol. |
| 2 | Full C ABI: `tprims-linalg-capi` with factor handles, `strided-capi`, host-callback executors, capability queries. Build and call from a real C program on Linux, macOS and Windows. | ABI conventions and pool close verified on all three platforms. |
| 3 | Replace faer paths or add specialized small-batch kernels only where measured; SVD and eigensolver accuracy requirements before any native solver. | Each replacement passes the correctness suite and a performance gate. |

Crates are published only after an interface and a consumer exist, consistent with [tenferro #1927](https://github.com/tensor4all/tenferro-rs/issues/1927).

AI-assisted contributions may include algorithms, implementations, benchmarks, counterexamples, and design proposals. Acceptance rests on attributable sources, numerical tests, reproducible performance evidence for optimization claims, and maintainer review.

## Why this project exists

[tenferro-rs #1945](https://github.com/tensor4all/tenferro-rs/issues/1945) documents a concrete FFI problem with an ambient Rayon pool: entering a CPU session costs roughly 8 to 14 µs on one measured AMD EPYC configuration, against about 1 µs for one small GEMM inside the session. The numbers are machine- and configuration-specific. [faer #319](https://codeberg.org/sarah-quinones/faer/issues/319) requests an explicit caller-owned Rayon pool. [tenferro-rs #1927](https://github.com/tensor4all/tenferro-rs/issues/1927) says to keep existing batched-linalg ownership in tenferro until another consumer and a stable buffer/scratch/provider/threading interface justify extraction. This repository explores that interface independently and for any host; it does not imply that tenferro will adopt a new backend.

## Sources and provenance

[Research map](research-map.md) links primary papers, official API documentation, project decisions, and upstream licenses. [Provenance policy](provenance.md) describes how to record an independently implemented algorithm, a code port, or a reused test. No upstream source or tests have been copied into this repository.

## Status

Research setup, revised 2026-09-29. No production design, ABI or package publication has been approved. The repository license is pending a maintainer choice; no third-party code is included.
