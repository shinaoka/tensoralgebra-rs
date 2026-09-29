# Provisional decisions

Dated 2026-09-29. A source link supports the observation; the proposed response remains a hypothesis until an experiment records evidence. Structural decisions (naming, packaging, ABI shape) are marked **Decided** when the maintainer has chosen them; they can still be revisited before the first release.

## Structure and ABI

| Question | Present position | Evidence or next check |
| --- | --- | --- |
| Facade crate? | **Decided: no facade.** Every part is an independently usable crate, as in tenferro-rs. | Maintainer decision, 2026-09-29 |
| Crate naming? | **Decided:** `tprims-<part>` with a short descriptive part name (`tprims-exec`, `tprims-gemm-kernel`, `tprims-blas`, `tprims-linalg`, `tprims-contract`). Crate, header and C symbol map one to one. Existing `strided-*` crates keep their names. No `tensor4all` or `t4a` naming: the stack targets hosts beyond tensor4all, and `t4a_` is already the tensor4all-rs C prefix. | Maintainer decision, 2026-09-29; `tprims` unused on crates.io at that date |
| `contract` or `tensordot`? | **Decided: `tprims-contract`.** NumPy / PyTorch / JAX `tensordot` has no batch indices; this operation does, as in cuTENSOR `cutensorContract`. | [cuTENSOR API](https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html) |
| Where does the C ABI live? | **Decided:** each part owns an `rlib` C ABI crate; only `tprims-bundle` builds `libtprims` (`cdylib` + `staticlib`) from Cargo-feature-selected parts. | Local prototype 2026-09-29 (macOS): selected parts' `#[no_mangle]` symbols exported, cross-part handle accepted. Next: CI on Linux and Windows. |
| Separate shared library per part? | Rejected. Handles, Rayon runtimes and (for static libraries) Rust std symbols would be duplicated. | Design reasoning; see [architecture](architecture.md#one-shared-library) |
| Data exchange? | **Decided: DLPack.** Borrowed `DLTensor` inputs and caller-provided outputs; library-allocated outputs as `DLManagedTensorVersioned`. Conjugation as a per-operand argument. No hidden materialization. | [DLPack specification](https://dmlc.github.io/dlpack/latest/) |
| Pool control from C? | **Decided:** create, query, budget, retain/release and synchronous close of a Rayon pool; serial and host-callback contexts as alternatives. | [Rayon ThreadPool drop semantics](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html) |
| `f16` / `bf16` in ABI v1? | Open. DLPack has the tags; kernels do not exist yet. | Decide before the first ABI release |
| Repository placement of `tprims-exec` / `tprims-core`? | Open. If they live here, `strided-rs` depends on them while this repository depends on `strided-*`: acyclic at crate level but cyclic at repository level. Alternatives: a small separate repository, or hosting them in `strided-rs` under the `tprims-` name. | Decide before `strided-capi` work starts ([strided-rs #234](https://github.com/tensor4all/strided-rs/issues/234)) |
| Solves and eigendecomposition? | **Decided:** in `tprims-linalg` with the factorizations, because solves use each factorization's internal representation. Nonsymmetric eigen deferred. | Design reasoning |
| Iterative Krylov solvers? | **Deferred, out of scope for now.** They need a linear-operator callback, convergence control and preconditioning, a different contract from dense linear algebra, and many consumers carry their own. A later `tprims-krylov` would depend on `tprims-blas`. | Maintainer decision, 2026-09-29 |

## Execution

| Question | Present position | Evidence or next check |
| --- | --- | --- |
| How much pool-entry cost is acceptable? | **Decided:** about 10 µs per parallel kernel is acceptable. Serial operations never enter a pool; they run on the calling thread. Parallel width is chosen from the amount of work, so full-width fan-out is reserved for large kernels. | [Rayon entry measurement](../experiments/rayon-entry/README.md), 2026-09-29, M5 Max: `install` 4 to 10 µs at 1 to 4 threads; 18-thread fan-out 55 to 200 µs. Maintainer decision, 2026-09-29. |
| Who owns the pool when a host library such as tenferro-rs also uses Rayon? | **Decided: borrow.** `tprims-exec` borrows the host's pool for the duration of a call (or a host session) and never creates a second pool beside it. `tprims_exec_rayon_create` exists only for C hosts that have no pool. | Two pools would oversubscribe cores and turn every call into a cross-pool handoff. Sharing a pool by reference requires one Rayon build in one binary. |
| Where is the pool entered? | **Decided:** inside a kernel, and only when it runs in parallel. If the calling thread is already a worker of the borrowed pool, the kernel runs in place without `install`. The caller's thread stays the driving thread for everything else, so host thread-local state needs no propagation. | Contrast: tenferro enters once per session and runs the whole session on a worker ([tenferro #1945](https://github.com/tensor4all/tenferro-rs/issues/1945)). |
| Initial backend? | **Decided (provisional):** faer for GEMM and dense linear algebra. `Par::Seq` on the calling thread for serial work; `install` on the borrowed pool, then `Par::rayon(n)`, for parallel work. Replace kernels only with measured justification. | Works with faer's ambient-pool API ([faer #319](https://codeberg.org/sarah-quinones/faer/issues/319) still open) because entry happens only at parallel kernels. |
| C ABI for execution: explicit handle per call, or a C-level "enter pool and run callback" command? | Open; leaning to an explicit `tprims_exec*` argument on every call with kernel-level entry inside. A C callback run on a pool worker would reintroduce the tenferro session problems and requires foreign runtimes (Python GIL, Julia thread adoption) to accept calls on threads they did not create. | Decide with the first `tprims-core` prototype |
| Spin-wait policy (OpenMP `KMP_BLOCKTIME` style)? | **Deferred.** Not needed while serial calls skip the pool and parallel calls are large. Revisit if a measured workload needs sub-microsecond chaining of parallel kernels. | OpenMP with active wait: 0.5 µs at 4 threads, but p90 spikes of 80 to 190 µs when spinning threads exceed cores ([measurement](../experiments/rayon-entry/README.md)). |

## Phase 1 scope

| Question | Present position | Evidence or next check |
| --- | --- | --- |
| What comes first? | **Decided:** being usable as the tenferro-rs CPU backend, behind a feature, with A/B correctness and a same-run performance gate. | Maintainer decision, 2026-09-29; [implementation order](architecture.md#implementation-order) |
| Contraction implementation? | **Decided:** port two strategies and compare them under one plan API: tenferro-rs permute plus batched GEMM, and the TBLIS-style direct contraction of tensorprimitives-rs `tensorcontract`. | Maintainer decision, 2026-09-29. Both sources are MIT OR Apache-2.0. Comparison in [Prototype 3](experiments.md#prototype-3-direct-tensor-contraction). |
| Batched GEMM? | **Decided:** two implementations to compare, faer plus a loop over items and the TBLIS-style kernel. | Maintainer decision, 2026-09-29; [Prototype 2](experiments.md#prototype-2-batched-gemm-and-factorizations) |
| Batched linear algebra? | **Decided:** faer per item with batched loops first; specialized small-batch kernels only where measured (Phase 3). | Maintainer decision, 2026-09-29 |
| C ABI in Phase 1? | **Decided:** a thin slice (`tprims-core`, `tprims-blas-capi`, `tprims-contract-capi`, `tprims-bundle`) plus a C benchmark harness, to test early whether the design holds across the C boundary. Full coverage is Phase 2. | Maintainer decision, 2026-09-29; [Prototype 4](experiments.md#prototype-4-c-abi-slice) |
| How is tensorprimitives-rs code brought in? | **Decided:** `git subtree add` without `--squash`, keeping Lukas Devos's authorship and commit history, after the overall design is settled. Lukas has been contacted. | Maintainer decision, 2026-09-29; [provenance](provenance.md) |

## Numerical and performance questions

| Question | Present position | Evidence or next check |
| --- | --- | --- |
| Should this become a production library? | Undecided. This repository is an experiment; extraction from tenferro needs another consumer and a stable interface. | [tenferro #1927](https://github.com/tensor4all/tenferro-rs/issues/1927) |
| Is Rayon unsuitable? | No blanket conclusion. It fits a barrier-free batch. Naive task scheduling is unsuitable for the specific barrier-driven inner contraction described by tensorprimitives-rs. | [tensorprimitives-rs D49/D50](https://github.com/lkdvos/tensorprimitives-rs/blob/main/docs/decisions.md); [Rayon ThreadPool API](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html) |
| Who owns threads? | Explicit caller-owned execution; serial execution stays available to a host that owns scheduling. | [tenferro #1945](https://github.com/tensor4all/tenferro-rs/issues/1945); [faer #319](https://codeberg.org/sarah-quinones/faer/issues/319) |
| Large GEMM provider? | Compare BLIS, Sarah Quiñones's `gemm`, and gemmkit; custom kernels require measured justification. | [BLIS paper](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf); [gemm](https://github.com/sarah-quinones/gemm); [gemmkit](https://github.com/SomeB1oody/gemmkit); [tenferro #1660](https://github.com/tensor4all/tenferro-rs/issues/1660) |
| Upstream pool API or fork? | Prefer a focused upstream request for the selected provider. Consider a narrow fork only if support is unavailable; neither provider accepts a pool reference today. | [`gemm` parallelism](https://github.com/sarah-quinones/gemm/blob/main/gemm-common/src/lib.rs); [`gemmkit` parallelism](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/src/parallel.rs); [faer #319](https://codeberg.org/sarah-quinones/faer/issues/319) |
| Tensor contraction provider? | Compare TBLIS-style direct contraction with materializing GEMM, not just kernel throughput. Both are ported in Phase 1 (see above). | [TBLIS paper](https://arxiv.org/abs/1607.00291) |
| Small batched linalg? | Start with serial-per-matrix (faer), outer-batch parallelism; measure the crossover. | [Haidar et al. §4.1](https://www.netlib.org/utk/people/JackDongarra/PAPERS/batched-matrix-comp.pdf) |
| Can we use published algorithms? | Yes, implement independently and cite sources. Record exact provenance if translating code or importing tests. | [Provenance policy](provenance.md) |
| Can we reuse faer/OpenBLAS tests? | Run as external oracles first. Review each file's license before importing test code or fixtures. | [faer license](https://github.com/sarah-quinones/faer-rs/blob/main/LICENSE); [OpenBLAS license](https://github.com/OpenMathLib/OpenBLAS/blob/develop/LICENSE) |

Update a row only after a recorded result or maintainer decision changes the evidence or conclusion. Keep rejected hypotheses visible.
