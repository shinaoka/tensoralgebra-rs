# Architecture design notes

[Back to the visual overview](../README.md). These notes retain the detailed rationale, numerical requirements, and primary sources.

An experimental design space for a CPU algebra stack: explicit execution, strided kernels, matrix multiplication, dense linear algebra including batches, and binary tensor contraction. The stack is meant for any Rust, C, Julia or Python host, not for a single consumer. AI-assisted design and implementation are welcome; contributors remain responsible for correctness, measurements, and code provenance. There is no stable API, ABI or performance claim yet.

## Working hypotheses

- **Large GEMM:** Evaluate a BLIS-style packed-panel design and existing BLIS, `gemm`, and gemmkit providers before writing architecture-specific kernels.
- **Tensor contraction:** Evaluate a TBLIS-style direct contraction against transpose/reshape-then-GEMM, including irregular strides and complex values.
- **Small matrices and batches:** Evaluate dedicated batched GEMM, LU, Cholesky, and QR kernels. Schedule independent matrices on the outer batch axis; test the crossover to per-matrix parallelism rather than assuming one rule fits every size.
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
| `tprims-exec` | Execution context: serial, caller-owned Rayon pool, host scheduling callbacks; thread budget for nested parallelism; scratch-size queries and reusable per-worker scratch. | none in the stack |
| `strided-traits`, `strided-view` (existing) | Checked borrowed strided views, scalar and conjugation contracts, rank-2 views of the same storage. | none |
| `strided-perm`, `strided-kernel` (existing) | Copy and permutation (HPTT-inspired); map, reduce, broadcast and fused elementwise kernels. Execution and scratch become explicit through `tprims-exec`. | `strided-view`, `tprims-exec` |
| `tprims-gemm-kernel` | Packed A/B panel format, scalar fallback, ISA dispatch, microkernels computing bounded result tiles. No scheduler, no full-matrix GEMM, no tensor labels. | `strided-view` |
| `tprims-blas` | Matrix GEMM driver, matrix packers, blocking, batched GEMM, TRSM, SYRK/HERK and further BLAS-like operations as needed (GEMV is not excluded by the name). | `tprims-gemm-kernel`, `tprims-exec` |
| `tprims-linalg` | Dense linear algebra: factorizations, direct solves, least squares, symmetric/Hermitian eigendecomposition, factor objects, errors, workspace plans; `batched` module. | `tprims-blas`, `strided-kernel`, `tprims-exec` |
| `tprims-contract` | Binary contraction with free, contracted and batch indices (`dot_general` semantics): plans, tensor panel packing, bounded scatter, output tile updates. Thin permute / add / trace wrappers and matricized factorization wrappers. | `tprims-gemm-kernel`, `tprims-blas`, `tprims-linalg`, `strided-perm`, `strided-kernel`, `tprims-exec` |

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

- **Inputs** are borrowed `const DLTensor*`. The library honours `byte_offset`, arbitrary element strides (column-major and negative strides included) and a NULL `strides` meaning compact row-major. `lanes` must be 1. Accepted devices are `kDLCPU` and `kDLCUDAHost`.
- **Caller-provided outputs** are `DLTensor*` written in place with `C = alpha * op(A, B) + beta * C` semantics and defined zero-size behaviour. Outputs must not overlap inputs unless an operation documents in-place support; overlap is checked where affordable.
- **Library-allocated outputs** are returned as `DLManagedTensorVersioned` (DLPack 1.x) with a deleter, for results whose shape or storage the library determines (factors, `eigh` results). The receiver owns them and can hand them to NumPy or Julia without a copy.
- **Conjugation** is a per-operand argument, since DLPack has no conjugation flag. It maps to the lazy conjugation already present in `strided-view`.
- **Read-only** versioned tensors are rejected as outputs.
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
tprims_status tprims_exec_close(tprims_exec *exec);
void         tprims_exec_retain(tprims_exec *exec);
void         tprims_exec_release(tprims_exec *exec);
```

- **Lifetime:** handles are reference counted so that a host binding and several library objects can share one pool.
- **Close is synchronous.** Dropping a Rayon `ThreadPool` terminates its threads asynchronously. `tprims_exec_close` marks the context closed, returns `TPRIMS_BUSY` if work is in flight, and otherwise waits until every worker has exited, observed through the pool's exit handler. Calls on a closed context return an error.
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

TBLIS also uses cooperating threads and barriers inside a blocked contraction. An arbitrary task-submission interface, including host callbacks, does not automatically provide that contract. Start with outer-batch parallelism and serial inner contractions, then prototype an explicitly synchronized inner driver on a Rayon context if large contractions need it.

## Why tensor contraction uses the GEMM kernel, not only `gemm()`

[BLIS](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf) separates blocking, packing, and a register-tile microkernel. [TBLIS §6 and §7](https://arxiv.org/html/1607.00291v4) reuses BLIS microkernels but needs its own tensor-aware packing and inner driver. An irregular output tile may require a small temporary tile followed by scatter; it does **not** require a full output transpose. TBLIS explicitly reports that the ordinary BLIS framework did not expose enough flexibility in its inner loops. Accordingly, `tprims-contract` depends on the `tprims-gemm-kernel` packed-kernel contract; calling public `tprims-blas` GEMM is one selected fast path, not the entire implementation.

A binary contraction plan validates free-left, contracted, free-right, and batch indices, output shape and aliasing, then chooses a measured path: (1) collapse compatible strides to a matrix view and call `tprims-blas` GEMM without copying; (2) pack bounded tensor panels directly into the `tprims-gemm-kernel` format and scatter irregular output tiles; or (3) explicitly materialize operands through `strided-perm` when extra bytes are worth the GEMM speed. Planning may fold contiguous dimensions and reorder logical traversal without changing user-visible index order.

The first kernel-contract prototype can use a scalar tile kernel. `gemm`, gemmkit, and BLIS are candidates or baselines; whether any exposes a usable low-level panel/microkernel seam must be checked. Their public GEMM calls alone cannot implement the direct TBLIS-style path. Kernel-specific panel packing and output scatter stay with the `tprims-contract` driver, which knows the packed format, index plan, and tile update semantics.

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
| Symmetric / Hermitian eigen | Tridiagonal reduction and a tridiagonal solver (LAPACK `SYEVD`/`HEEVD` family). Nonsymmetric `GEEV` is deferred. | Per-item serial first. |

### Solves

Solves live in the same crate as the factorizations because they operate on each factorization's internal representation (pivot arrays, reflectors and `tau`, blocked storage). Splitting them out would turn that representation into a public API and, through the C ABI, into a frozen contract. In C, a factor object is an opaque handle.

- Factor-object solves: `lu.solve`, `cholesky.solve`, `ldl.solve` (LAPACK `GETRS`/`POTRS`/`SYTRS` equivalents), with transpose and conjugate-transpose variants.
- One-shot `solve(A, B)`, least squares `lstsq` (QR for full rank, SVD or pivoted QR for rank-deficient input with an explicit rank tolerance), `inv`, `det` and `logdet`.
- Triangular solve with a matrix right-hand side is TRSM in `tprims-blas`.

Start with `f32`/`f64`, then complex arithmetic with explicit conjugation behavior. Require reconstruction/solve residuals, QR orthogonality, rank-deficient, indefinite and non-positive-definite inputs, extreme scales, and convergence status before timing. SVD and eigensolvers are separate numerical workstreams, not straightforward GEMM extensions.

Tensor-level factorizations in `tprims-contract` reshape a strided tensor into a rank-2 view over the same storage when the split is stride-compatible, or materialize through `strided-perm` otherwise, then call `tprims-linalg`. The wrapper reports which of the two it did and adds no numerical logic.

### Batched execution

`tprims-linalg` contains a `batched` module with factorization and solve entry points; `tprims-blas` owns batched GEMM. The batch module owns batch descriptors, output/status arrays, scratch planning, and the choice of batch versus inner-matrix parallelism. It executes the schedule on the caller's `tprims-exec` context and reuses per-matrix routines as its first implementation. Specialized small-matrix or interleaved batch kernels can later replace the per-item implementation under the same batch contract; they require their own correctness and performance evidence.

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

1. Define `tprims-exec` and `tprims-core`: serial and Rayon contexts with synchronous close, DLPack operand validation, status and error model. Build a minimal `tprims-bundle` with two stub parts and call it from a real C program on Linux, macOS and Windows.
2. Test the `tprims-gemm-kernel` packed tile seam with ordinary matrix GEMM and several irregular contractions. Compare providers; keep a scalar fallback. Add ISA kernels only when measurement warrants it.
3. Implement or integrate `tprims-blas` GEMM, TRSM, SYRK/HERK and small/panel routines; validate Cholesky, LU, LDLᴴ, QR and solves in `tprims-linalg` against reconstruction, residual and provider oracles.
4. Add homogeneous batched calls and the `tprims-contract` direct pack/scatter path; compare with materialization and provider-backed paths. Then consider interleaved small batches, grouped batches, and retargeting `strided-opteinsum`.
5. Specify SVD and eigensolver accuracy and convergence requirements and benchmark a provider baseline before committing to native solvers. Publish crates only after an interface and a consumer exist, consistent with [tenferro #1927](https://github.com/tensor4all/tenferro-rs/issues/1927).

AI-assisted contributions may include algorithms, implementations, benchmarks, counterexamples, and design proposals. Acceptance rests on attributable sources, numerical tests, reproducible performance evidence for optimization claims, and maintainer review.

## Why this project exists

[tenferro-rs #1945](https://github.com/tensor4all/tenferro-rs/issues/1945) documents a concrete FFI problem with an ambient Rayon pool: entering a CPU session costs roughly 8 to 14 µs on one measured AMD EPYC configuration, against about 1 µs for one small GEMM inside the session. The numbers are machine- and configuration-specific. [faer #319](https://codeberg.org/sarah-quinones/faer/issues/319) requests an explicit caller-owned Rayon pool. [tenferro-rs #1927](https://github.com/tensor4all/tenferro-rs/issues/1927) says to keep existing batched-linalg ownership in tenferro until another consumer and a stable buffer/scratch/provider/threading interface justify extraction. This repository explores that interface independently and for any host; it does not imply that tenferro will adopt a new backend.

## Sources and provenance

[Research map](research-map.md) links primary papers, official API documentation, project decisions, and upstream licenses. [Provenance policy](provenance.md) describes how to record an independently implemented algorithm, a code port, or a reused test. No upstream source or tests have been copied into this repository.

## Status

Research setup, revised 2026-09-29. No production design, ABI or package publication has been approved. The repository license is pending a maintainer choice; no third-party code is included.
