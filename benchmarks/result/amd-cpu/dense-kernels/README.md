# Dense CPU kernel instruction comparison

**Instruction counts, not elapsed times, bandwidth, or predicted speedups.**

- Baseline strided-rs: `ec585b8a4bf0af96863a6136f0b1f8e9c1aeadba` (migration completed, before optimization).
- Optimized strided-rs: `17e05ffb168d0f826529ea2c3aceeb4ec851448b`.
- Benchmark source: `8ee3945ae90a1be77354c19011e99c648b7a5dcc`.
- AMD EPYC 7713P; CPU 16 pinned (L3 domain 16–23); explicit Sequential/1T; no BLAS.
- Rust 1.97.1, release; Valgrind 3.22.0; three collected calls after one excluded warmup.
- Both multiply implementations ran pulp V3/AVX2; this is not a scalar-versus-SIMD comparison.
- Other users' Julia jobs were active; **no native timing comparison was performed**.

| Case | Baseline Ir/call | Optimized Ir/call | Ir reduction |
|---|---:|---:|---:|
| mul_2048 | 34,604,461 | 5,244,343 | 84.84% |
| mul_odd | 8,669,139 | 1,314,777 | 84.83% |
| mul_strided | 6,555,925 | 6,555,925 | 0.00% |
| axpby_1m | 3,932,305 | 3,932,305 | 0.00% |
| tril_1024 | 2,587,856 | 1,705,902 | 34.08% |
| triu_rect | 2,327,763 | 1,955,919 | 15.97% |
| diag_rank2 | 1,367,481 | 1,367,509 | -0.00% |

Contiguous multiply now uses full-vector loads/stores, reserving partial accesses for its tail.
Triangle masks copy only retained intervals and fill the rest, rather than copying and then masking.
AXPBY and noncontiguous multiply are unchanged controls. Diagonal embedding differs by 28 Ir/call
(0.002%, displayed as -0.00%); no improvement or meaningful regression is claimed for it.

The baseline already includes migration. These numbers do not measure migration versus old tenferro,
and do not include tenferro dispatch, input materialization, allocation or buffer-pool overhead.

Every case checks all output values against nonzero scalar references outside the collection window.
Memcheck reports zero errors for the seven benchmark cases and separate real/complex uninitialized
SIMD output tests. Leak checking was disabled; this is not a leak-free claim.

See raw/environment.txt for image/container IDs, compiler information and binary/source/lockfile hashes,
raw/features.txt for effective crate features, and raw/*.callgrind plus *.log for primary evidence.
Regenerate with `python3 result/amd-cpu/dense-kernels/summarize.py`.

## Validation and remaining work

- strided-optimized-suite: 359 passed (including doctests), zero failures.
- tenferro-optimized-suite: 818 passed (including doctests), zero failures.
- Focused release SIMD/dense tests and SIMD-disabled dense tests also passed; logs retained.
- Tenferro tests used its default CPU feature set (faer), with explicit local strided path overrides.
- Strict Clippy failed on pre-existing lint errors in unchanged dependency/basic code under Rust 1.97.1; both logs are retained. No passing strict-lint gate is claimed.
- Quiet-host native measurements remain pending.
- Tenferro consumer base is `d8759f4320a337d2399f4a87dfec55af51d2ebf1`; its exact uncommitted migration diff is raw/tenferro-migration.patch and the tested override is raw/cpu-kernel-migration-cargo.toml (local absolute paths).
- At measurement time the tenferro git dependency pin was unchanged; these historical local-override tests alone did not establish a ready-to-merge dependency chain. Subsequent integration is tracked by strided-rs PR259 (merged commit `5bc5ab75`) and tenferro-rs PR1807.
