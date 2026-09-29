# TN Light 415 Late-Step Einsum Benchmark

Reproduces the bottleneck step of `tensornetwork_permutation_light_415`.
This page is a focused einsum case study for one late contraction step. It is
not intended to summarize the full benchmark suite or make a broad
Rust-vs-Julia claim.

Binary einsum with all dims = 2: A(13 dims, 8K) × B(24 dims, 16M) → C(18 dims, 262K).
m=4, k=256, n=8192, batch=8.

## Run

```bash
# Rust (blas backend)
cargo run --release --no-default-features --features blas --bin tn_light_415_late_step

# Julia
julia --project=. benchmarks/einsum_benchmarks/tn_light_415_late_step/tn_light_415_late_step.jl

# Fair comparison (scrambled vs natural labels)
julia --project=. benchmarks/einsum_benchmarks/tn_light_415_late_step/tn_light_415_late_step_fair.jl
```

## Results (Apple M5 Max MacBook Pro)

Measured on macOS 26.5.1, 2026-07-02. macOS runs are not CPU-pinned.

- `strided-rs`: `7cdc813`
- Rust compiler flags: `RUSTFLAGS="-C target-cpu=native"`
- Julia environment: Manifest-instantiated `OMEinsum v0.9.3`

### Rust BLAS

Median milliseconds. Run with
`cargo run --release --no-default-features --features blas --bin tn_light_415_late_step`.

| Threads | full scattered | copy B | copy A | contiguous GEMM |
|---:|---:|---:|---:|---:|
| 1 | 17.087 | 12.465 | 0.005 | 5.261 |
| 4 | 17.020 | 12.305 | 0.005 | 5.202 |

### Julia Decomposition

Median milliseconds. Run with `julia --project=. benchmarks/einsum_benchmarks/tn_light_415_late_step/tn_light_415_late_step.jl`.
The manual `permutedims! + BLAS gemm` path in this script currently reports a
large max error, so treat this table as component timing, not correctness
validation.

| Threads | OMEinsum DynamicEinCode | permutedims! + BLAS | permutedims! B | permutedims! A | BLAS GEMM |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.010 | 28.326 | 22.269 | 0.008 | 5.778 |
| 4 | 2.147 | 24.771 | 22.356 | 0.008 | 2.235 |

### Julia Fair Comparison

Median milliseconds. Run with `julia --project=. benchmarks/einsum_benchmarks/tn_light_415_late_step/tn_light_415_late_step_fair.jl`.

| Threads | OMEinsum scrambled | OMEinsum natural | permutedims! B Rust perm | permutedims! B batch-front | BLAS GEMM |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.796 | 5.931 | 21.986 | 16.883 | 5.760 |
| 4 | 25.066 | 2.219 | 21.999 | 16.957 | 2.114 |
