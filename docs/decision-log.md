# Provisional decisions

Dated 2026-09-29. A source link supports the observation; the proposed response remains a hypothesis until an experiment records evidence.

| Question | Present position | Evidence or next check |
| --- | --- | --- |
| Should this become a production library? | Undecided. This repository is an experiment; extraction from tenferro needs another consumer and a stable interface. | [tenferro #1927](https://github.com/tensor4all/tenferro-rs/issues/1927) |
| Is Rayon unsuitable? | No blanket conclusion. It fits a barrier-free batch. Naive task scheduling is unsuitable for the specific barrier-driven inner contraction described by tensorprimitives-rs. | [tensorprimitives-rs D49/D50](https://github.com/lkdvos/tensorprimitives-rs/blob/main/docs/decisions.md); [Rayon ThreadPool API](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html) |
| Who owns threads? | Prototype explicit caller-owned execution; keep serial execution available to an FFI host that owns scheduling. | [tenferro #1945](https://github.com/tensor4all/tenferro-rs/issues/1945); [faer #319](https://codeberg.org/sarah-quinones/faer/issues/319) |
| Large GEMM provider? | Compare BLIS, Sarah Quiñones's `gemm`, and gemmkit; custom kernels require measured justification. | [BLIS paper](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf); [gemm](https://github.com/sarah-quinones/gemm); [gemmkit](https://github.com/SomeB1oody/gemmkit); [tenferro #1660](https://github.com/tensor4all/tenferro-rs/issues/1660) |
| Should a `matalg-rs` backend exist? | Possible future boundary, not a repository to create now. Decide after provider and executor experiments establish a stable interface. | [tenferro #1927](https://github.com/tensor4all/tenferro-rs/issues/1927); [experiment protocol](experiments.md) |
| Upstream pool API or fork? | Prefer a focused upstream request for the selected provider. Consider a narrow fork only if support is unavailable; neither provider accepts a pool reference today. | [`gemm` parallelism](https://github.com/sarah-quinones/gemm/blob/main/gemm-common/src/lib.rs); [`gemmkit` parallelism](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/src/parallel.rs); [faer #319](https://codeberg.org/sarah-quinones/faer/issues/319) |
| Tensor contraction provider? | Compare TBLIS-style direct contraction with materializing GEMM, not just kernel throughput. | [TBLIS paper](https://arxiv.org/abs/1607.00291) |
| Small batched linalg? | Start with serial-per-matrix, outer-batch parallelism; measure the crossover. | [Haidar et al. §4.1](https://www.netlib.org/utk/people/JackDongarra/PAPERS/batched-matrix-comp.pdf) |
| Can we use published algorithms? | Yes, implement independently and cite sources. Record exact provenance if translating code or importing tests. | [Provenance policy](provenance.md) |
| Can we reuse faer/OpenBLAS tests? | Run as external oracles first. Review each file's license before importing test code or fixtures. | [faer license](https://github.com/sarah-quinones/faer-rs/blob/main/LICENSE); [OpenBLAS license](https://github.com/OpenMathLib/OpenBLAS/blob/develop/LICENSE) |

Update a row only after a recorded result changes the evidence or conclusion. Keep rejected hypotheses visible.
