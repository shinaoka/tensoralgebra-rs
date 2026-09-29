# Architecture design notes

[Back to the visual overview](../README.md). These notes retain the detailed rationale, numerical requirements, and primary sources.

An experimental design space for a tensor4all CPU algebra stack: strided kernels, matrix multiplication, dense factorizations including batches, and binary tensor contraction. AI-assisted design and implementation are welcome; contributors remain responsible for correctness, measurements, and code provenance. There is no stable API or performance claim yet.

## Working hypotheses

- **Large GEMM:** Evaluate a BLIS-style packed-panel design and existing BLIS, `gemm`, and gemmkit providers before writing architecture-specific kernels.
- **Tensor contraction:** Evaluate a TBLIS-style direct contraction against transpose/reshape-then-GEMM, including irregular strides and complex values.
- **Small matrices and batches:** Evaluate dedicated batched GEMM, LU, Cholesky, and QR kernels. Schedule independent matrices on the outer batch axis; test the crossover to per-matrix parallelism rather than assuming one rule fits every size.
- **Execution ownership:** Make the effective thread budget, pool, and scratch lifetime explicit. A Rust caller should be able to supply an executor. A C/Julia/Python caller needs an FFI contract that does not silently take over the host's threads.

These are questions, not settled design decisions. [The research map](research-map.md) distinguishes published evidence from project-specific hypotheses. [The experiment plan](experiments.md) defines the first three independent prototypes and how to compare them.

## Three layers

The stack is three independent workspaces with one direction of dependency: tensor primitives depend on matrix algebra, matrix algebra depends on strided kernels, and strided kernels depend on nothing in the stack. Repository placement follows this graph; no cycle is allowed.

| Layer | Workspace | Crate | Responsibility |
| --- | --- | --- | --- |
| Strided kernels | `strided-rs` | `strided-traits`, `strided-view` | Checked borrowed strided views and metadata, scalar/conjugation contracts, rank-2 matrix views of the same storage. No owning tensor type from tenferro, no thread pool. Existing crates, reused. |
| | | `strided-exec` (new) | Caller-owned executor/session contract, thread budget, scratch-size queries, reusable per-worker scratch, serial fallback. No ambient global pool requirement. |
| | | `strided-perm` | Explicit copy and permutation (HPTT-inspired). Existing crate. |
| | | `strided-kernel` | Map, reduce, broadcast and fused elementwise kernels on borrowed views. Existing crate; execution and scratch to be made explicit. |
| Matrix algebra | `matalg-rs` | `matalg-kernel` | Packed A/B panel format, scalar fallback, ISA dispatch, and GEMM microkernels computing bounded result tiles. No public full-matrix GEMM, no scheduler, no tensor labels. |
| | | `matalg` | Matrix GEMM driver, matrix packers, blocking, batched GEMM, and necessary BLAS-like operations such as GEMV, TRSM, and SYRK/HERK. |
| | | `matalg-linalg` | LU, Cholesky, QR, SVD, factor objects, solves, errors, workspace plans. Batched entry points are a `batched` module of this crate. |
| Tensor primitives | `tprims-rs` | `tprims` | Binary `tensordot`/`dot_general`: plans, tensor panel packing, bounded scatter metadata, output tile updates. Thin wrappers for permute, add and trace over `strided-perm` / `strided-kernel`. Thin tensor-level SVD/QR/LU wrappers over `matalg-linalg` (matricize, call, reshape back). |

The central boundary is the **packed GEMM tile kernel** in `matalg-kernel`, shared by matrix GEMM and direct tensor contraction. `matalg` owns ordinary matrix packing and the blocked driver. `tprims` owns index planning, tensor packing, and irregular output scatter. `matalg-linalg` uses `matalg` operations and its own numerical algorithms. Batched APIs compose per-matrix algorithms with explicit scheduling and reusable scratch.

Each layer is deliberately small. `tprims` is a single crate: contraction is its only substantial algorithm, and everything else in it is a wrapper that names a tensor operation and delegates. `matalg-linalg` is a single crate: batched scheduling is a module, not a separate numerical library. Additional crates are created only when a separate consumer and a stable interface justify the split.

### What is excluded

- **N-ary einsum.** Index notation and contraction-order planning are a frontend concern. They stay in `strided-opteinsum` or in the consumer (tenferro, Julia/Python frontends) and call `tprims` for each binary step. The stack exposes no index-string API.
- **AD, traced execution, device transfer, GPU backends.** Tenferro owns these. Adapters for `ndarray` / `mdarray` and any facade sit above the stack.
- **Tensor-level numerical algorithms.** `tprims` does not implement pivoting, convergence, or scaling. A tensor SVD is a matricized call into `matalg-linalg`.

### Dependency rules

- `strided-*` crates never depend on `matalg-*` or `tprims`.
- `matalg-kernel` depends only on `strided-view` / `strided-traits`. It contains no scheduler and takes no executor.
- `matalg` and `matalg-linalg` take an explicit `strided-exec` context for every expensive operation. `matalg-linalg` may reuse generic copy/scale from `strided-kernel`; numerically specialized reductions and panel algorithms remain its responsibility.
- `tprims` depends on `matalg-kernel` (direct path), `matalg` (matrix-view fast path), `matalg-linalg` (matricized factorizations), `strided-perm` and `strided-kernel` (materialize path and wrappers), and `strided-exec`.
- Every dependency on the strided layer uses the caller's execution and scratch context.

### Why tensor contraction uses the GEMM kernel, not only `gemm()`

[BLIS](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf) separates blocking, packing, and a register-tile microkernel. [TBLIS §6–7](https://arxiv.org/html/1607.00291v4) reuses BLIS microkernels but needs its own tensor-aware packing and inner driver. An irregular output tile may require a small temporary tile followed by scatter; it does **not** require a full output transpose. TBLIS explicitly reports that the ordinary BLIS framework did not expose enough flexibility in its inner loops. Accordingly, `tprims` depends on the `matalg-kernel` packed-kernel contract; calling public `matalg` GEMM is one selected fast path, not the entire implementation.

A binary contraction plan validates free-left, reduction, free-right, and shared batch indices, output shape and aliasing, then chooses a measured path: (1) collapse compatible strides to a matrix view and call `matalg` GEMM without copying; (2) pack bounded tensor panels directly into the `matalg-kernel` format and scatter irregular output tiles; or (3) explicitly materialize operands through `strided-perm` when extra bytes are worth the GEMM speed. The public semantics should be `C = alpha * contract(A, B) + beta * C`, with a checked caller-output form and defined zero-size behavior. Planning may fold contiguous dimensions and reorder logical traversal without changing user-visible index order. Full-tensor materialization must be reported as a selected strategy.

The first kernel-contract prototype can use a scalar tile kernel. `gemm`, gemmkit, and BLIS are candidates or baselines; whether any exposes a usable low-level panel/microkernel seam must be checked. Their public GEMM calls alone cannot implement the direct TBLIS-style path.

Full copy/permutation is implemented by `strided-perm`. Kernel-specific panel packing and output scatter stay with the `tprims` driver, which knows the packed format, index plan, and tile update semantics. A direct path therefore need not call a standalone permutation kernel, and specialized packers need not be expressed through a generic elementwise API.

### Matrix decompositions

`matalg-linalg` owns pivoting, scaling, factor storage, convergence, and solve semantics. It uses `matalg` for large trailing updates and small direct/panel kernels where GEMM is a poor fit. The operations share the matrix view and executor, not one universal decomposition algorithm.

| Operation | Starting decomposition | Batched baseline |
| --- | --- | --- |
| Cholesky | Small unblocked factorization with a positive-definiteness check; larger blocked panels with TRSM and SYRK/HERK or GEMM updates. [LAPACK `POTRF`](https://netlib.org/lapack/explore-html/d2/d09/group__potrf_ga84e90859b02139934b166e579dd211d4.html). | One serial factorization per matrix; report the failing matrix and pivot. |
| LU | Partial-pivot panel, row swaps, TRSM, GEMM trailing update; retain pivots and singularity status. [LAPACK `GETRF`](https://www.netlib.org/lapack/explore-html/db/d04/group__getrf_gaea332d65e208d833716b405ea2a1ab69.html). | Outer-batch parallelism first, with independent pivot arrays. |
| QR | Householder reflectors, small unblocked path, compact blocked application and trailing update. Store reflectors/`tau`; generate Q only on request. [LAPACK `GEQRF`](https://www.netlib.org/lapack/explore-html/d3/d69/dgeqrf_8f_source.html). | Reuse scalar QR first; compare a specialized small-matrix route later. |
| SVD | Scaling, bidiagonal reduction, a robust bidiagonal solver, and singular-vector back-transformation. GEMM assists updates but does not provide accuracy or convergence. [LAPACK SVD overview](https://www.netlib.org/lapack/lug/node53.html). | Begin with a validated provider or correct serial implementation per item; defer specialized batched SVD until justified. |

Start with `f32`/`f64`, then complex arithmetic with explicit conjugation behavior. Require reconstruction/solve residuals, QR orthogonality, rank-deficient and non-positive-definite inputs, extreme scales, and convergence status before timing. SVD is a separate numerical workstream, not a straightforward GEMM extension.

Tensor-level factorizations in `tprims` reshape a strided tensor into a rank-2 view over the same storage when the split is stride-compatible, or materialize through `strided-perm` otherwise, then call `matalg-linalg`. The wrapper reports which of the two it did and adds no numerical logic.

### Batched execution

`matalg-linalg` contains both per-matrix factorization algorithms and a `batched` module with LU, Cholesky, QR, and SVD entry points. The batch module owns batch descriptors, output/status arrays, scratch planning, and the choice of batch versus inner-matrix parallelism. It uses `strided-exec` to execute that schedule and reuses per-matrix routines as its first implementation. `matalg` similarly owns batched GEMM. Specialized small-matrix or interleaved batch kernels can later replace the per-item implementation under the same batch contract; they require their own correctness and performance evidence. Batched SVD follows the staged solver plan described above.

The first API targets equally shaped strided matrices; grouped heterogeneous shapes can follow. A prepared batch plan can reuse validated descriptors and one scratch region per worker. Schedule **one serial matrix operation per independent task** on the caller's executor initially. For a small batch of large matrices, measure inner-matrix threading instead. One thread budget controls both levels so nested parallelism does not oversubscribe. A single FFI batch call amortizes call and executor-entry costs.

Return a per-item status and define the contents of failed outputs. Batched GEMM should compare arrays of matrices, interleaved/structure-of-arrays layouts, and dedicated small-matrix SIMD kernels. A common scheduler does not imply one data layout or microkernel. [Haidar et al. §4.1](https://www.netlib.org/utk/people/JackDongarra/PAPERS/batched-matrix-comp.pdf) motivates serial-per-matrix CPU execution; [Deshmukh et al.](https://arxiv.org/abs/2311.07602) motivates cache- and shape-specific batch GEMM. Their results do not establish a universal crossover or a CPU batched-SVD algorithm.

### Executor and FFI contract

`strided-exec` defines the contract. Every expensive operation accepts an explicit context or prepared handle referring to caller-owned execution and scratch. Serial work runs directly on the calling thread. A C/Julia/Python host needs a retainable opaque handle or equivalent callback contract; a Rust `&rayon::ThreadPool` is only a Rust adapter. Specify task submission/completion, thread budget, scratch lifetime, and caller-thread participation. Direct pool passage and low-cost FFI entry are goals to prototype and measure, not present capabilities of `gemm` or gemmkit.

TBLIS also uses cooperating threads and barriers inside a blocked contraction. An arbitrary task-submission pool does not automatically provide that contract. Start with outer-batch parallelism and serial inner contractions, then prototype an explicitly synchronized inner driver if large contractions need it. Measure it with the caller's actual executor.

## Relationship to `strided-rs`

The [current `strided-rs` workspace](https://github.com/tensor4all/strided-rs/blob/main/Cargo.toml) already has views, basic and fused kernels, HPTT-inspired permutation, binary einsum, N-ary planning, adapters, and a facade. It is not migrated into this stack; it *is* the lowest layer. The proposed changes are limited:

| Current component | Proposed role and review |
| --- | --- |
| `strided-traits`, `strided-view` | Reused unchanged as the shared view contract. Preserve checked borrowing and lazy conjugation. |
| `strided-exec` (new) | Add the executor, thread budget and scratch contract. Existing kernels gain an explicit context parameter where they currently rely on ambient threading. |
| `strided-perm` | Reused as the copy/permutation kernel for `tprims` materialize paths and permute wrappers. Preserve [HPTT provenance and license](https://github.com/tensor4all/strided-rs/blob/main/docs/PROVENANCE_AND_CITATION_POLICY.md). |
| `strided-kernel` | Reused for map/reduce/fused paths and for generic copy/scale in `matalg-linalg`. Review repeated metadata/index work; make execution and scratch explicit. |
| `strided-einsum2` | Superseded by `tprims`. Its semantic tests and provider comparisons move to `tprims`; its default contiguous-preparation path is replaced by the matrix-view / direct-pack / materialize plan. |
| `strided-opteinsum` | Stays in `strided-rs` as the N-ary planner. Its binary step is retargeted to `tprims`. |
| `mdarray-opteinsum`, `ndarray-opteinsum`, `strided-rs` facade | Thin adapters above the stack. Keep current consumers, including tenferro, pinned until equivalent correctness and performance are verified. |

Current binary einsum [passes operands through a contiguous-preparation path before GEMM](https://github.com/tensor4all/strided-rs/blob/main/strided-einsum2/src/lib.rs); compatible views may avoid a copy, but a direct tensor packer is a different architecture. Moving Julia/HPTT-derived code also requires preserving its source attribution and file-level licenses under the [provenance policy](provenance.md). An independently implemented published algorithm and a translated source file have different obligations. No `strided-rs` code or tests have been copied into this repository.

## Implementation order

1. Define `strided-exec`: executor and scratch ownership, reference scalar kernels, and a real C caller. Validate results before timing.
2. Test the `matalg-kernel` packed tile seam with ordinary matrix GEMM and several irregular contractions. Compare providers; keep a scalar fallback. Add ISA kernels only when measurement warrants it.
3. Implement or integrate `matalg` GEMM, TRSM, SYRK/HERK and small/panel routines; validate Cholesky, LU, and QR in `matalg-linalg` against reconstruction and provider oracles.
4. Add homogeneous batched calls and the `tprims` direct pack/scatter path; compare with materialization and provider-backed paths. Then consider interleaved small batches, grouped batches, and retargeting `strided-opteinsum`.
5. Specify SVD accuracy and convergence requirements and benchmark a provider baseline before committing to a native solver. Create the `matalg-rs` repository and production crates only after an interface and another consumer exist, consistent with [tenferro #1927](https://github.com/tensor4all/tenferro-rs/issues/1927).

AI-assisted contributions may include algorithms, implementations, benchmarks, counterexamples, and design proposals. Acceptance rests on attributable sources, numerical tests, reproducible performance evidence for optimization claims, and maintainer review. [Research map](research-map.md), [experiment protocol](experiments.md), [decision log](decision-log.md), and [provenance policy](provenance.md) record the evidence and open questions.

## Why this project exists

[tenferro-rs #1945](https://github.com/tensor4all/tenferro-rs/issues/1945) documents a concrete FFI problem with an ambient Rayon pool: entering a CPU session costs roughly 8–14 µs on one measured AMD EPYC configuration, against about 1 µs for one small GEMM inside the session. The numbers are machine- and configuration-specific. [faer #319](https://codeberg.org/sarah-quinones/faer/issues/319) requests an explicit caller-owned Rayon pool. [tenferro-rs #1927](https://github.com/tensor4all/tenferro-rs/issues/1927) says to keep existing batched-linalg ownership in tenferro until another consumer and a stable buffer/scratch/provider/threading interface justify extraction. This repository explores that interface independently; it does not imply that tenferro will adopt a new backend.

The linked [faer #316](https://codeberg.org/sarah-quinones/faer/issues/316) is a GPU roadmap discussion, separate from the thread-pool request in #319.

## Experiments

1. **Executor and FFI cost:** Compare direct serial execution, a caller-owned Rayon pool, and a scoped-thread batch driver. Measure entry, per-item, and batch costs separately, including a C caller.
2. **Small GEMM and factorizations:** Establish correct scalar baselines, then compare batch layouts, scheduling, scratch reuse, and GEMM providers. Include LU, Cholesky, and QR only after reconstruction tests pass.
3. **Tensor contraction:** Compare direct contraction with TBLIS and transpose/reshape-then-GEMM, covering general strides, complex arithmetic, and different cache topologies.

See [experiment protocol](experiments.md). The initial repository contains research notes and the protocol; measured implementations and results will be added as separate, reviewable experiments.

## Sources and provenance

[Research map](research-map.md) links primary papers, official API documentation, project decisions, and upstream licenses. [Provenance policy](provenance.md) describes how to record an independently implemented algorithm, a code port, or a reused test. No upstream source or tests have been copied into this repository.

## Status

Research setup, 2026-09-29. No production design or package publication has been approved. The repository license is pending a maintainer choice; no third-party code is included. Discussion and reproducible counterexamples are welcome.
