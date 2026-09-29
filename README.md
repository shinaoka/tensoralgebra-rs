# tensoralgebra-rs

An experimental Rust project for CPU tensor contractions, GEMM, and batched dense linear algebra. Its purpose is to test a few small designs before choosing a production library architecture. There is no stable API or performance claim yet.

## Working hypotheses

- **Large GEMM:** Evaluate a BLIS-style packed-panel design and existing BLIS, `gemm`, and gemmkit providers before writing architecture-specific kernels.
- **Tensor contraction:** Evaluate a TBLIS-style direct contraction against transpose/reshape-then-GEMM, including irregular strides and complex values.
- **Small matrices and batches:** Evaluate dedicated batched GEMM, LU, Cholesky, and QR kernels. Schedule independent matrices on the outer batch axis; test the crossover to per-matrix parallelism rather than assuming one rule fits every size.
- **Execution ownership:** Make the effective thread budget, pool, and scratch lifetime explicit. A Rust caller should be able to supply an executor. A C/Julia/Python caller needs an FFI contract that does not silently take over the host's threads.

These are questions, not settled design decisions. [The research map](docs/research-map.md) distinguishes published evidence from project-specific hypotheses. [The experiment plan](docs/experiments.md) defines the first three independent prototypes and how to compare them.

`matrixalgebra-rs` is a possible future CPU backend boundary. It will only be created if experiments identify a useful stable interface; it is not part of this repository setup.

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
