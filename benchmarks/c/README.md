# C ABI benchmark

Per-call cost of `libtprims` from C against the same calls made directly
from Rust (`tprims-bench` binary `capi_rust`): an empty call
(`tprims_abi_version`), an 8x8 f64 GEMM, and a 2x2x2 contraction through a
prebuilt plan, each on a serial executor and on a 4-thread pool. Each sample
is the mean of 1000 calls; the median of 101 samples is reported.

```bash
benchmarks/c/run.sh 57 57-60 /tmp/capi   # 1T cores, 4T cores, output
```

## Observation, 2026-09-30 (single run, not a claim)

- tprims-rs `509f30f` (clean tree), release profile, rustc 1.97.1, gcc -O2.
- AMD EPYC 7713P, cores 57-60 idle before the run. Raw output:
  [results/](results/).

| Case | C 1T | Rust 1T | C 4T | Rust 4T |
| --- | ---: | ---: | ---: | ---: |
| empty call | 2.9 ns | 0.3 ns | 3.0 ns | 0.3 ns |
| 8x8 GEMM | 430 ns | 145 ns | 427 ns | 120 ns |
| 2x2x2 contraction (plan) | 661 ns | 264 ns | 716 ns | 266 ns |

Closing the 4-thread pool (joining its workers) took 99 us.

### Findings

1. **The C boundary adds about 300-450 ns per small call**, 3x the Rust call
   for an 8x8 GEMM. This fails Prototype 4's gate ("C-side fixed cost within
   a small margin of the Rust call") for tiny operations. *Hypothesis, not
   profiled:* per-operand metadata allocations (the DLPack layout's `Vec`s and
   the `Arc` dims/strides inside `strided_view::StridedView`, about ten per
   GEMM call). `strided-view`'s borrowed-metadata `RawStridedRef` avoids
   them; accepting it in `tprims-blas`/`tprims-contract` is the follow-up.
2. Small calls never enter the pool (4T rows equal 1T rows), as designed.
3. Zero copy holds: the C test checks that results land in the caller's
   buffers for row-major (NULL strides), column-major and strided inputs.
