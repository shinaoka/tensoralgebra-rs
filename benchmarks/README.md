# tprims-bench

Benchmarks for the tprims stack. The package was imported from
[strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite)
(upstream `0550611`) with its history; its einsum half (strided-opteinsum and
OMEinsum.jl runners, Python dataset pipeline, fixtures) is frozen under
[`deprecated/`](deprecated/README.md).

## Live benchmarks

[`benchmarks/strided_benchmarks/`](benchmarks/strided_benchmarks/README.md):
kernel-level comparisons of naive loops, strided-rs and HPTT.

| Binary | Page |
| --- | --- |
| `permute` | [permute](benchmarks/strided_benchmarks/permute/) |
| `scale_transpose` | [transpose_scale](benchmarks/strided_benchmarks/transpose_scale/) |
| `fused_elementwise` | [fused_elementwise](benchmarks/strided_benchmarks/fused_elementwise/) |
| `dense_kernels` | [dense_kernels](benchmarks/strided_benchmarks/dense_kernels/README.md) |
| `kernel_scaling` (feature `parallel`) | [kernel_scaling](benchmarks/strided_benchmarks/kernel_scaling/) |

New tprims benchmarks (Phase 1 onward) live under `benchmarks/tprims/`.

## Build and run

The package is a member of the root workspace, so binaries land in the
repository's `target/`:

```bash
cargo build -j 16 --release -p tprims-bench --features parallel --bins
taskset -c 0   ../target/release/kernel_scaling --threads 1 --time
taskset -c 0-3 ../target/release/kernel_scaling --threads 4 --time
```

Julia baselines (`permute.jl`, `kernel_scaling.jl`) use this directory's
Julia environment (`Project.toml`: JSON, Strided):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Rules

Measure every tensor-sized case at 1 and 4 threads in the same run, verify the
effective thread count at startup, pin with `taskset` inside one L3 domain,
never run two benchmarks at once, and record the tprims-rs commit, CPU, core
set and profile beside every published table. See the root
[`PERFORMANCE_TIPS.md`](../PERFORMANCE_TIPS.md) and [`AGENTS.md`](AGENTS.md).
