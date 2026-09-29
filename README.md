# tensoralgebra-rs

An experimental design space for a tensor4all CPU algebra stack: strided arrays, matrix multiplication, tensor contraction, and dense factorizations, including batches. AI-assisted design and implementation are welcome; contributors remain responsible for correctness, measurements, and code provenance. There is no stable API or performance claim yet.

## Working hypotheses

- **Large GEMM:** Evaluate a BLIS-style packed-panel design and existing BLIS, `gemm`, and gemmkit providers before writing architecture-specific kernels.
- **Tensor contraction:** Evaluate a TBLIS-style direct contraction against transpose/reshape-then-GEMM, including irregular strides and complex values.
- **Small matrices and batches:** Evaluate dedicated batched GEMM, LU, Cholesky, and QR kernels. Schedule independent matrices on the outer batch axis; test the crossover to per-matrix parallelism rather than assuming one rule fits every size.
- **Execution ownership:** Make the effective thread budget, pool, and scratch lifetime explicit. A Rust caller should be able to supply an executor. A C/Julia/Python caller needs an FFI contract that does not silently take over the host's threads.

These are questions, not settled design decisions. [The research map](docs/research-map.md) distinguishes published evidence from project-specific hypotheses. [The experiment plan](docs/experiments.md) defines the first three independent prototypes and how to compare them.

## Proposed crate boundaries

The central boundary is a **packed GEMM tile kernel** shared by matrix GEMM and direct tensor contraction. Matrix GEMM owns its ordinary matrix packing and blocked driver. Tensor contraction owns index planning, tensor packing, and irregular output scatter. Matrix factorizations use matrix operations and their own numerical algorithms. Batched APIs compose per-matrix algorithms with explicit scheduling and reusable scratch.

This is a provisional map, not a request to create every crate now. Prototypes can start in this workspace. If the interfaces stabilize, `matrixalgebra-rs` can be the lower-level backend repository and `tensoralgebra-rs` the higher-level repository under tensor4all. Dependencies must point from tensoralgebra to matrixalgebra, with no repository cycle. Neither a new backend repository nor a transfer of this one has been performed.

| Proposed crate | Responsibility |
| --- | --- |
| `algebra-view` (lower level) | Checked borrowed strided views and metadata, scalar/conjugation contracts, rank-2 matrix views of the same storage. No owning tenferro tensor or thread pool. |
| `algebra-exec` (lower level) | Caller-owned executor/session contract, thread budget, scratch-size queries, reusable per-worker scratch, serial fallback. No ambient global pool requirement. |
| `matrixalgebra-kernel` | Packed A/B panel format, scalar fallback, ISA dispatch, and GEMM microkernels computing bounded result tiles. No public full-matrix GEMM or tensor labels. |
| `matrixalgebra` | Matrix GEMM driver, matrix packers, blocking, and necessary BLAS-like operations such as GEMV, TRSM, and SYRK/HERK. |
| `matrixalgebra-linalg` | LU, Cholesky, QR, SVD, factor objects, solves, errors, workspace plans, and batched entry points. |
| `tensoralgebra-strided` | Map/reduce/broadcast, fused elementwise work, and explicit copy/permutation, redesigned from `strided-rs`. |
| `tensoralgebra-contract` | Binary `tensordot`/`dot_general`: plans, tensor panel packing, bounded scatter metadata, output tile updates. |
| `tensoralgebra-einsum` | N-ary index syntax and contraction-order planning, calling the binary engine. |

These are *responsibilities* before they are package names. Keep related modules in one crate until a separate consumer and stable interface justify a split. Batched scheduling is shared execution machinery; `batched_gemm` and batched factorizations remain beside their scalar operations. Thin `ndarray`/`mdarray` adapters and an optional facade sit above the core. Tenferro continues to own AD, traced execution, device transfer, and GPU backends.

`algebra-view` and `algebra-exec` provide the shared contracts. `matrixalgebra-kernel` contains no scheduler; `matrixalgebra` uses the kernel and shared contracts; `matrixalgebra-linalg` uses matrix operations. `tensoralgebra-contract` uses the shared contracts, the tile kernel, and matrix GEMM for compatible fast paths; `tensoralgebra-einsum` uses the binary contraction engine. `tensoralgebra-strided` uses the shared contracts. No matrix crate depends on a tensor crate.

### Why tensor contraction uses the GEMM kernel, not only `gemm()`

[BLIS](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf) separates blocking, packing, and a register-tile microkernel. [TBLIS §6–7](https://arxiv.org/html/1607.00291v4) reuses BLIS microkernels but needs its own tensor-aware packing and inner driver. An irregular output tile may require a small temporary tile followed by scatter; it does **not** require a full output transpose. TBLIS explicitly reports that the ordinary BLIS framework did not expose enough flexibility in its inner loops. Accordingly, `tensoralgebra-contract` depends on the packed-kernel contract; calling public matrix GEMM is one selected fast path, not the entire implementation.

A binary contraction plan validates free-left, reduction, free-right, and shared batch indices, output shape and aliasing, then chooses a measured path: (1) collapse compatible strides to a matrix view and call GEMM without copying; (2) pack bounded tensor panels directly into the shared kernel format and scatter irregular output tiles; or (3) explicitly materialize operands when extra bytes are worth the GEMM speed. The public semantics should be `C = alpha * contract(A, B) + beta * C`, with a checked caller-output form and defined zero-size behavior. Planning may fold contiguous dimensions and reorder logical traversal without changing user-visible index order. Full-tensor materialization must be reported as a selected strategy.

The first kernel-contract prototype can use a scalar tile kernel. `gemm`, gemmkit, and BLIS are candidates or baselines; whether any exposes a usable low-level panel/microkernel seam must be checked. Their public GEMM calls alone cannot implement the direct TBLIS-style path.

### Matrix decompositions

`matrixalgebra-linalg` owns pivoting, scaling, factor storage, convergence, and solve semantics. It uses `matrixalgebra` for large trailing updates and small direct/panel kernels where GEMM is a poor fit. The operations share the matrix view and executor, not one universal decomposition algorithm.

| Operation | Starting decomposition | Batched baseline |
| --- | --- | --- |
| Cholesky | Small unblocked factorization with a positive-definiteness check; larger blocked panels with TRSM and SYRK/HERK or GEMM updates. [LAPACK `POTRF`](https://netlib.org/lapack/explore-html/d2/d09/group__potrf_ga84e90859b02139934b166e579dd211d4.html). | One serial factorization per matrix; report the failing matrix and pivot. |
| LU | Partial-pivot panel, row swaps, TRSM, GEMM trailing update; retain pivots and singularity status. [LAPACK `GETRF`](https://www.netlib.org/lapack/explore-html/db/d04/group__getrf_gaea332d65e208d833716b405ea2a1ab69.html). | Outer-batch parallelism first, with independent pivot arrays. |
| QR | Householder reflectors, small unblocked path, compact blocked application and trailing update. Store reflectors/`tau`; generate Q only on request. [LAPACK `GEQRF`](https://www.netlib.org/lapack/explore-html/d3/d69/dgeqrf_8f_source.html). | Reuse scalar QR first; compare a specialized small-matrix route later. |
| SVD | Scaling, bidiagonal reduction, a robust bidiagonal solver, and singular-vector back-transformation. GEMM assists updates but does not provide accuracy or convergence. [LAPACK SVD overview](https://www.netlib.org/lapack/lug/node53.html). | Begin with a validated provider or correct serial implementation per item; defer specialized batched SVD until justified. |

Start with `f32`/`f64`, then complex arithmetic with explicit conjugation behavior. Require reconstruction/solve residuals, QR orthogonality, rank-deficient and non-positive-definite inputs, extreme scales, and convergence status before timing. SVD is a separate numerical workstream, not a straightforward GEMM extension.

### Batched execution

The first API targets equally shaped strided matrices; grouped heterogeneous shapes can follow. A prepared batch plan can reuse validated descriptors and one scratch region per worker. Schedule **one serial matrix operation per independent task** on the caller's executor initially. For a small batch of large matrices, measure inner-matrix threading instead. One thread budget controls both levels so nested parallelism does not oversubscribe. A single FFI batch call amortizes call and executor-entry costs.

Return a per-item status and define the contents of failed outputs. Batched GEMM should compare arrays of matrices, interleaved/structure-of-arrays layouts, and dedicated small-matrix SIMD kernels. A common scheduler does not imply one data layout or microkernel. [Haidar et al. §4.1](https://www.netlib.org/utk/people/JackDongarra/PAPERS/batched-matrix-comp.pdf) motivates serial-per-matrix CPU execution; [Deshmukh et al.](https://arxiv.org/abs/2311.07602) motivates cache- and shape-specific batch GEMM. Their results do not establish a universal crossover or a CPU batched-SVD algorithm.

### Executor and FFI contract

Every expensive operation accepts an explicit context or prepared handle referring to caller-owned execution and scratch. Serial work runs directly on the calling thread. A C/Julia/Python host needs a retainable opaque handle or equivalent callback contract; a Rust `&rayon::ThreadPool` is only a Rust adapter. Specify task submission/completion, thread budget, scratch lifetime, and caller-thread participation. Direct pool passage and low-cost FFI entry are goals to prototype and measure, not present capabilities of `gemm` or gemmkit.

TBLIS also uses cooperating threads and barriers inside a blocked contraction. An arbitrary task-submission pool does not automatically provide that contract. Start with outer-batch parallelism and serial inner contractions, then prototype an explicitly synchronized inner driver if large contractions need it. Measure it with the caller's actual executor.

## Integrating and redesigning `strided-rs`

The [current `strided-rs` workspace](https://github.com/tensor4all/strided-rs/blob/main/Cargo.toml) already has views, basic and fused kernels, HPTT-inspired permutation, binary einsum, N-ary planning, adapters, and a facade. The integration should preserve tested semantics while replacing boundaries that force ambient threading or hidden materialization:

| Current component | Proposed destination and review |
| --- | --- |
| `strided-traits`, `strided-view` | `algebra-view`: preserve checked borrowing and lazy conjugation; separate borrowed metadata from owned `StridedArray`, which belongs with `tensoralgebra-strided`. Place the shared view crate in the lower-level workspace to avoid a dependency cycle. |
| `strided-basic`, `strided-fused`, `strided-kernel` | `tensoralgebra-strided`: keep useful map/reduce/fused paths, make execution and scratch explicit, and review repeated metadata/index work. Basic and fused can remain modules until separate crates earn their cost. |
| `strided-perm` | Explicit copy/transpose under `tensoralgebra-strided`, or its own crate if independently needed. Preserve [HPTT provenance and license](https://github.com/tensor4all/strided-rs/blob/main/docs/PROVENANCE_AND_CITATION_POLICY.md). |
| `strided-einsum2` | `tensoralgebra-contract`: preserve semantic tests and provider comparisons; replace default contiguous preparation with matrix-view/direct-pack/materialize choices. |
| `strided-opteinsum` | `tensoralgebra-einsum`: preserve the N-ary planner while using one binary contraction engine. |
| `mdarray-opteinsum`, `ndarray-opteinsum`, `strided-rs` facade | Thin adapters or transition facade. Keep current consumers, including tenferro, pinned until equivalent correctness and performance are verified. |

Current binary einsum [passes operands through a contiguous-preparation path before GEMM](https://github.com/tensor4all/strided-rs/blob/main/strided-einsum2/src/lib.rs); compatible views may avoid a copy, but a direct tensor packer is a different architecture. Moving Julia/HPTT-derived code also requires preserving its source attribution and file-level licenses under the [provenance policy](docs/provenance.md). An independently implemented published algorithm and a translated source file have different obligations. No `strided-rs` code or tests have been copied into this repository.

## Implementation order

1. Define checked views, scalar/output semantics, executor and scratch ownership, reference scalar kernels, and a real C caller. Validate results before timing.
2. Test the packed tile seam with ordinary matrix GEMM and several irregular contractions. Compare providers; keep a scalar fallback. Add ISA kernels only when measurement warrants it.
3. Implement or integrate matrix GEMM, TRSM, SYRK/HERK and small/panel routines; validate Cholesky, LU, and QR against reconstruction and provider oracles.
4. Add homogeneous batched calls and direct tensor pack/scatter; compare with materialization and provider-backed paths. Then consider interleaved small batches, grouped batches, and N-ary planning.
5. Specify SVD accuracy and convergence requirements and benchmark a provider baseline before committing to a native solver. Extract production crates only after an interface and another consumer exist, consistent with [tenferro #1927](https://github.com/tensor4all/tenferro-rs/issues/1927).

AI-assisted contributions may include algorithms, implementations, benchmarks, counterexamples, and design proposals. Acceptance rests on attributable sources, numerical tests, reproducible performance evidence for optimization claims, and maintainer review. [Research map](docs/research-map.md), [experiment protocol](docs/experiments.md), [decision log](docs/decision-log.md), and [provenance policy](docs/provenance.md) record the evidence and open questions.

## Why this project exists

[tenferro-rs #1945](https://github.com/tensor4all/tenferro-rs/issues/1945) documents a concrete FFI problem with an ambient Rayon pool: entering a CPU session costs roughly 8–14 µs on one measured AMD EPYC configuration, against about 1 µs for one small GEMM inside the session. The numbers are machine- and configuration-specific. [faer #319](https://codeberg.org/sarah-quinones/faer/issues/319) requests an explicit caller-owned Rayon pool. [tenferro-rs #1927](https://github.com/tensor4all/tenferro-rs/issues/1927) says to keep existing batched-linalg ownership in tenferro until another consumer and a stable buffer/scratch/provider/threading interface justify extraction. This repository explores that interface independently; it does not imply that tenferro will adopt a new backend.

The linked [faer #316](https://codeberg.org/sarah-quinones/faer/issues/316) is a GPU roadmap discussion, separate from the thread-pool request in #319.

## Experiments

1. **Executor and FFI cost:** Compare direct serial execution, a caller-owned Rayon pool, and a scoped-thread batch driver. Measure entry, per-item, and batch costs separately, including a C caller.
2. **Small GEMM and factorizations:** Establish correct scalar baselines, then compare batch layouts, scheduling, scratch reuse, and GEMM providers. Include LU, Cholesky, and QR only after reconstruction tests pass.
3. **Tensor contraction:** Compare direct contraction with TBLIS and transpose/reshape-then-GEMM, covering general strides, complex arithmetic, and different cache topologies.

See [experiment protocol](docs/experiments.md). The initial repository contains research notes and the protocol; measured implementations and results will be added as separate, reviewable experiments.

## Sources and provenance

[Research map](docs/research-map.md) links primary papers, official API documentation, project decisions, and upstream licenses. [Provenance policy](docs/provenance.md) describes how to record an independently implemented algorithm, a code port, or a reused test. No upstream source or tests have been copied into this repository.

## Status

Research setup, 2026-09-29. No production design or package publication has been approved. The repository license is pending a maintainer choice; no third-party code is included. Discussion and reproducible counterexamples are welcome.
