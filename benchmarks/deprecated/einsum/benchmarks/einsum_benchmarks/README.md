# Einsum Benchmarks

This directory contains the full-suite einsum benchmark result tables and
focused einsum case studies. The repository root README intentionally links
here instead of embedding large result tables.

## Pages

| Page | Scope |
|---|---|
| [Full suite results](#benchmark-instances) | Repository-level `strided-opteinsum` versus OMEinsum.jl results. |
| [TN light 415 late-step case study](tn_light_415_late_step/README.md) | One late binary contraction from `tensornetwork_permutation_light_415`. |

## Benchmark Instances

Instances are from the [einsum benchmark](https://benchmark.einsum.org/) suite.
Selection is per-category (see
[Export benchmark metadata](../../README.md#1-export-benchmark-metadata)); dtype is
float64 or complex128; tensors are zero-filled at runtime.

| Instance | Category | Tensors | Dims | Typical shapes | Steps | log10(FLOPS) | log2(SIZE) |
|----------|----------|--------:|------|----------------|------:|-------------:|------------:|
| `gm_queen5_5_3.wcsp` | Graphical model | 160 | 2D | 3×3 | 159 | 9.75 | 26.94 |
| `lm_batch_likelihood_brackets_4_4d` | Language model | 84 | 2–4D | 4×4, 4×4×4, 7×1996 | 83 | 8.37 | 18.96 |
| `lm_batch_likelihood_sentence_3_12d` | Language model | 38 | 2–4D | 11×11, 11×11×11, 100×1100 | 37 | 9.20 | 20.86 |
| `lm_batch_likelihood_sentence_4_4d` | Language model | 84 | 2–4D | 4×4, 4×4×4, 7×1900 | 83 | 8.46 | 18.89 |
| `str_matrix_chain_multiplication_100` | Structured | 100 | 2D | 21×478 to 511×507 | 99 | 8.48 | 17.26 |
| `str_mps_varying_inner_product_200` | Structured (MPS) | 200 | 2D | varying | 199 | 8.31 | 15.48 |
| `str_nw_mera_closed_120` | Structured (MERA) | 120 | 2D | 3×3, etc. | 119 | 10.66 | 25.02 |
| `str_nw_mera_open_26` | Structured (MERA) | 26 | 2D | 3×3, etc. | 25 | 10.49 | 25.36 |
| `tensornetwork_permutation_light_415` | Tensor network | 415 | 2D | 2×2 (uniform) | 414 | 9.65 | 24.0 |
| `tensornetwork_permutation_focus_step409_316` | Tensor network (focused) | 316 | 2D | 2×2 (uniform) | 315 | 9.65 | 24.0 |

- **Graphical model (gm_*)**: e.g. WCSP / constraint networks; many small 2D factors (e.g. 3×3), full contraction to scalar.
- **Language model (lm_*)**: many small multi-dimensional tensors (3D/4D) with large batch dimensions; many steps with small GEMM kernels.
- **Structured — matrix chain (str_matrix_chain_*)**: large 2D matrices; each step is one large GEMM.
- **Structured — MPS (str_mps_*)**: matrix product state–style networks; varying inner dimensions, many 2D contractions.
- **Structured — MERA (str_nw_mera_*)**: tensor networks from multi-scale entanglement renormalization; many small 3×3-like tensors, heavy contraction.
- **Tensor network (tensornetwork_permutation_light_415)**: lightweight variant (~5 s vs ~40 s); 415 tensors extracted from the full instance via BFS-connected subgraph. Create via `scripts/create_lightweight_instance.py` (see [Optional: Create a lightweight tensor network instance](../../README.md#optional-create-a-lightweight-tensor-network-instance)).
- **Tensor network focused (tensornetwork_permutation_focus_step409_316)**: focused subtree instance for profiling original late bottleneck steps (408/409). Contains 316 tensors and keeps the high-cost intermediate (`log2_size = 24`) while reducing total contractions.

## Benchmark Results

### Apple M5 Max MacBook Pro

Environment: Apple M5 Max, macOS 26.5.1. Median ± IQR (ms) of 15 timed runs after 3 warmup runs. Run date: 2026-07-02. macOS CPU pinning was unavailable.

- `strided-rs`: `7cdc813`
- Rust: `rustc 1.96.0`, `RUSTFLAGS="-C target-cpu=native"`
- Julia: 1.12.5, BLAS via libblastrampoline/OpenBLAS
- Allocation is outside timed regions for Rust and Julia runners.
- `-` means the implementation skipped the instance because the runner does not support that index pattern.

#### 1 Thread

`JULIA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`.

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.865 ± 0.269 | **0.390 ± 0.245** | 0.786 ± 0.155 |
| bin_elementwise_mul_2048x2048 | 1.119 ± 0.068 | 0.872 ± 0.216 | **0.794 ± 0.107** |
| bin_matmul_256 | 0.676 ± 0.060 | **0.100 ± 0.028** | 0.575 ± 0.005 |
| bin_outer_product_4096 | 1.949 ± 0.108 | **1.384 ± 0.188** | - |
| gm_queen5_5_3.wcsp | **1627.338 ± 45.372** | 1925.129 ± 60.404 | - |
| lm_batch_likelihood_brackets_4_4d | 14.047 ± 0.288 | **9.883 ± 0.363** | 12.327 ± 1.057 |
| lm_batch_likelihood_sentence_3_12d | 43.693 ± 2.054 | **18.716 ± 0.841** | 44.673 ± 48.256 |
| lm_batch_likelihood_sentence_4_4d | 17.045 ± 1.394 | **11.564 ± 0.432** | 12.950 ± 7.627 |
| str_matrix_chain_multiplication_100 | 9.322 ± 0.566 | **1.916 ± 0.151** | 9.425 ± 0.169 |
| str_mps_varying_inner_product_200 | 9.829 ± 1.400 | **5.598 ± 0.149** | 10.864 ± 7.377 |
| str_nw_mera_closed_120 | 1025.524 ± 9.837 | **316.247 ± 2.923** | 980.052 ± 6.455 |
| str_nw_mera_open_26 | 621.918 ± 10.468 | **194.218 ± 6.322** | 738.852 ± 49.426 |
| tensornetwork_permutation_focus_step409_316 | 138.453 ± 1.805 | **91.690 ± 2.227** | 181.847 ± 7.825 |
| tensornetwork_permutation_light_415 | 139.193 ± 1.526 | **92.167 ± 1.065** | 183.142 ± 8.495 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.559 ± 0.014 | **0.362 ± 0.005** | 0.918 ± 0.034 |
| bin_elementwise_mul_2048x2048 | 0.810 ± 0.025 | 0.834 ± 0.028 | **0.805 ± 0.098** |
| bin_matmul_256 | 0.579 ± 0.010 | **0.084 ± 0.004** | 0.579 ± 0.005 |
| bin_outer_product_4096 | 1.937 ± 0.045 | **1.329 ± 0.080** | - |
| gm_queen5_5_3.wcsp | 653.771 ± 9.441 | **349.005 ± 15.687** | - |
| lm_batch_likelihood_brackets_4_4d | 12.604 ± 0.157 | **10.260 ± 0.763** | 13.028 ± 7.622 |
| lm_batch_likelihood_sentence_3_12d | 43.617 ± 0.817 | **22.962 ± 1.385** | 43.820 ± 44.961 |
| lm_batch_likelihood_sentence_4_4d | 15.992 ± 0.318 | **12.054 ± 0.424** | 19.725 ± 29.786 |
| str_matrix_chain_multiplication_100 | 9.053 ± 0.173 | **2.023 ± 0.151** | 9.499 ± 0.286 |
| str_mps_varying_inner_product_200 | 9.087 ± 0.303 | **6.111 ± 0.476** | 10.014 ± 0.199 |
| str_nw_mera_closed_120 | 902.288 ± 20.622 | **189.883 ± 16.869** | 958.599 ± 4.204 |
| str_nw_mera_open_26 | 637.112 ± 16.323 | **194.204 ± 2.064** | 732.565 ± 57.115 |
| tensornetwork_permutation_focus_step409_316 | 143.506 ± 2.982 | **90.244 ± 0.778** | 182.389 ± 31.008 |
| tensornetwork_permutation_light_415 | 145.214 ± 2.893 | **92.389 ± 4.466** | 183.992 ± 7.073 |

#### 4 Threads

`JULIA_NUM_THREADS=4`, `OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`.

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.951 ± 0.192 | **0.368 ± 0.007** | 0.920 ± 0.037 |
| bin_elementwise_mul_2048x2048 | 0.603 ± 0.039 | **0.582 ± 0.036** | 0.874 ± 3.953 |
| bin_matmul_256 | 0.200 ± 0.015 | **0.085 ± 0.007** | 0.159 ± 0.002 |
| bin_outer_product_4096 | **1.124 ± 0.036** | 1.349 ± 0.100 | - |
| gm_queen5_5_3.wcsp | **1670.391 ± 5.061** | 1809.209 ± 11.255 | - |
| lm_batch_likelihood_brackets_4_4d | 11.683 ± 0.253 | **9.996 ± 0.169** | 10.274 ± 19.283 |
| lm_batch_likelihood_sentence_3_12d | 24.367 ± 0.679 | **18.416 ± 0.349** | 25.318 ± 20.926 |
| lm_batch_likelihood_sentence_4_4d | 12.865 ± 0.263 | 11.502 ± 0.308 | **10.520 ± 11.452** |
| str_matrix_chain_multiplication_100 | 5.803 ± 1.014 | **1.941 ± 0.099** | 5.919 ± 0.207 |
| str_mps_varying_inner_product_200 | 9.545 ± 0.378 | **5.574 ± 0.130** | 10.664 ± 4.825 |
| str_nw_mera_closed_120 | 421.289 ± 9.195 | 320.002 ± 1.460 | **316.507 ± 4.096** |
| str_nw_mera_open_26 | 221.605 ± 1.008 | **191.901 ± 1.786** | 282.868 ± 13.125 |
| tensornetwork_permutation_focus_step409_316 | **81.662 ± 0.663** | 90.169 ± 0.556 | 94.264 ± 15.430 |
| tensornetwork_permutation_light_415 | **82.397 ± 0.563** | 92.312 ± 0.584 | 93.484 ± 16.341 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.579 ± 0.017 | **0.346 ± 0.009** | 0.723 ± 0.049 |
| bin_elementwise_mul_2048x2048 | 0.551 ± 0.073 | **0.549 ± 0.048** | 0.855 ± 0.031 |
| bin_matmul_256 | 0.171 ± 0.026 | **0.082 ± 0.005** | 0.161 ± 0.015 |
| bin_outer_product_4096 | **1.111 ± 0.057** | 1.213 ± 0.077 | - |
| gm_queen5_5_3.wcsp | 435.898 ± 5.110 | **350.812 ± 3.639** | - |
| lm_batch_likelihood_brackets_4_4d | 10.696 ± 0.411 | **9.781 ± 0.180** | 11.020 ± 10.921 |
| lm_batch_likelihood_sentence_3_12d | 25.407 ± 0.365 | **19.795 ± 0.466** | 36.518 ± 15.941 |
| lm_batch_likelihood_sentence_4_4d | 12.456 ± 0.444 | 10.967 ± 0.226 | **9.849 ± 12.378** |
| str_matrix_chain_multiplication_100 | 4.865 ± 0.444 | **1.976 ± 0.133** | 5.950 ± 0.240 |
| str_mps_varying_inner_product_200 | 9.295 ± 0.216 | **5.812 ± 0.268** | 9.446 ± 10.516 |
| str_nw_mera_closed_120 | 301.255 ± 14.386 | **182.871 ± 1.203** | 302.706 ± 20.720 |
| str_nw_mera_open_26 | 229.060 ± 5.339 | **195.579 ± 1.133** | 275.761 ± 3.891 |
| tensornetwork_permutation_focus_step409_316 | **82.418 ± 0.694** | 91.955 ± 1.435 | 89.494 ± 9.445 |
| tensornetwork_permutation_light_415 | **83.234 ± 0.680** | 93.968 ± 1.035 | 90.422 ± 7.534 |

**Notes:**

- Rust OpenBLAS is the fastest path for most BLAS-heavy einsum cases on this M5 Max run.
- The faer backend is competitive on some permutation-heavy tensor-network cases at 4 threads.
- OMEinsum.jl has high IQR on several language-model and tensor-network cases even after rerunning on AC power; the medians are reported with the observed IQR rather than filtered.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) — Rust tensor library
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) — Julia einsum library
- [TensorNetworkBenchmarks](https://github.com/TensorBFS/TensorNetworkBenchmarks) — tensor network contraction benchmarks (PyTorch, OMEinsum.jl)
