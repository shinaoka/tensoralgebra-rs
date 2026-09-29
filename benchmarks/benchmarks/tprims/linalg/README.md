# linalg

`tprims-linalg` at an enforced thread count: single-matrix factorizations
(cholesky, lu, solve with one right-hand side, qr, thin svd, eigh with
vectors, eig with vectors) at n = 2 to 512, and the `batched` module
(solve, cholesky, eigh with vectors, svd values) including the small-matrix,
large-batch regression case (n <= 8, batch 1024). f64 and c64. The timed
boundary is one public call including the input copy into factor storage.

```bash
cargo build -j 16 --release -p tprims-bench --bin linalg
benchmarks/benchmarks/tprims/linalg/run.sh 19 19-22 /tmp/linalg   # 1T cores, 4T cores, output
```

`run.sh` measures every case in its own process (`BENCH_CASE`) with the 1T
and 4T runs of a case back to back. Both matter: parallel cases leave the pool
in a state that slows later serial rows in the same process, and slow host
drift biased a run that measured all 1T cases before all 4T cases.

## Observation, 2026-09-30 (single run, not a claim)

- tprims-rs `2b0e3b5` (clean tree), release profile (thin LTO,
  codegen-units 1), rustc 1.97.1, faer 0.24.4.
- AMD EPYC 7713P, shared host (load average about 5); cores 19-22 (one L3
  domain) busy fraction <= 0.01 over 3 s before and after.
- Noise floor (A/A, same 1T binary twice): up to 20% for sub-microsecond
  cases (cholesky n = 2: 200 ns vs 160 ns), about 1% above 1 us. The 1T/4T
  ratios of rows that stay serial (width one by the policy) are within this
  floor.
- Raw output: [results/2026-09-30-1t.csv](results/2026-09-30-1t.csv),
  [results/2026-09-30-4t.csv](results/2026-09-30-4t.csv).

| Case | Variant | 1T | 4T | 1T/4T |
| --- | --- | ---: | ---: | ---: |
| `cholesky_f64` | n2 | 160 ns | 210 ns | 0.76 |
| `cholesky_f64` | n4 | 230 ns | 290 ns | 0.79 |
| `cholesky_f64` | n8 | 380 ns | 480 ns | 0.79 |
| `cholesky_f64` | n32 | 3.7 µs | 4.3 µs | 0.87 |
| `cholesky_f64` | n128 | 65.9 µs | 55.6 µs | 1.18 |
| `cholesky_f64` | n512 | 1.62 ms | 1.50 ms | 1.08 |
| `cholesky_c64` | n2 | 170 ns | 170 ns | 1.00 |
| `cholesky_c64` | n4 | 230 ns | 240 ns | 0.96 |
| `cholesky_c64` | n8 | 410 ns | 440 ns | 0.93 |
| `cholesky_c64` | n32 | 5.9 µs | 5.9 µs | 1.00 |
| `cholesky_c64` | n128 | 155.2 µs | 173.2 µs | 0.90 |
| `cholesky_c64` | n512 | 4.81 ms | 3.14 ms | 1.53 |
| `lu_f64` | n2 | 190 ns | 200 ns | 0.95 |
| `lu_f64` | n4 | 330 ns | 330 ns | 1.00 |
| `lu_f64` | n8 | 690 ns | 700 ns | 0.99 |
| `lu_f64` | n32 | 5.9 µs | 5.7 µs | 1.04 |
| `lu_f64` | n128 | 104.6 µs | 103.4 µs | 1.01 |
| `lu_f64` | n512 | 3.78 ms | 2.94 ms | 1.29 |
| `lu_c64` | n2 | 210 ns | 220 ns | 0.95 |
| `lu_c64` | n4 | 390 ns | 410 ns | 0.95 |
| `lu_c64` | n8 | 760 ns | 770 ns | 0.99 |
| `lu_c64` | n32 | 8.6 µs | 8.7 µs | 0.98 |
| `lu_c64` | n128 | 243.5 µs | 200.8 µs | 1.21 |
| `lu_c64` | n512 | 8.38 ms | 5.04 ms | 1.66 |
| `solve_f64` | n2 | 400 ns | 400 ns | 1.00 |
| `solve_f64` | n4 | 610 ns | 610 ns | 1.00 |
| `solve_f64` | n8 | 1.1 µs | 1.3 µs | 0.87 |
| `solve_f64` | n32 | 7.3 µs | 7.9 µs | 0.93 |
| `solve_f64` | n128 | 113.4 µs | 113.5 µs | 1.00 |
| `solve_f64` | n512 | 4.11 ms | 3.09 ms | 1.33 |
| `solve_c64` | n2 | 440 ns | 440 ns | 1.00 |
| `solve_c64` | n4 | 710 ns | 710 ns | 1.00 |
| `solve_c64` | n8 | 1.3 µs | 1.4 µs | 0.96 |
| `solve_c64` | n32 | 11.1 µs | 10.9 µs | 1.02 |
| `solve_c64` | n128 | 253.6 µs | 213.4 µs | 1.19 |
| `solve_c64` | n512 | 10.27 ms | 5.21 ms | 1.97 |
| `qr_f64` | n2 | 250 ns | 310 ns | 0.81 |
| `qr_f64` | n4 | 460 ns | 580 ns | 0.79 |
| `qr_f64` | n8 | 1.1 µs | 1.4 µs | 0.80 |
| `qr_f64` | n32 | 17.8 µs | 17.6 µs | 1.01 |
| `qr_f64` | n128 | 277.5 µs | 284.2 µs | 0.98 |
| `qr_f64` | n512 | 5.98 ms | 5.20 ms | 1.15 |
| `qr_c64` | n2 | 310 ns | 310 ns | 1.00 |
| `qr_c64` | n4 | 580 ns | 580 ns | 1.00 |
| `qr_c64` | n8 | 1.5 µs | 1.5 µs | 1.00 |
| `qr_c64` | n32 | 29.3 µs | 29.6 µs | 0.99 |
| `qr_c64` | n128 | 645.5 µs | 541.2 µs | 1.19 |
| `qr_c64` | n512 | 21.61 ms | 10.07 ms | 2.15 |
| `svd_f64` | n2 | 1.2 µs | 1.3 µs | 0.99 |
| `svd_f64` | n4 | 3.3 µs | 3.3 µs | 0.99 |
| `svd_f64` | n8 | 11.2 µs | 11.1 µs | 1.01 |
| `svd_f64` | n32 | 149.4 µs | 149.2 µs | 1.00 |
| `svd_f64` | n128 | 2.73 ms | 2.29 ms | 1.19 |
| `svd_f64` | n512 | 72.41 ms | 49.18 ms | 1.47 |
| `svd_c64` | n2 | 1.4 µs | 1.4 µs | 0.99 |
| `svd_c64` | n4 | 3.9 µs | 3.7 µs | 1.05 |
| `svd_c64` | n8 | 12.8 µs | 11.7 µs | 1.09 |
| `svd_c64` | n32 | 197.7 µs | 207.8 µs | 0.95 |
| `svd_c64` | n128 | 4.74 ms | 3.73 ms | 1.27 |
| `svd_c64` | n512 | 169.35 ms | 74.57 ms | 2.27 |
| `eigh_f64` | n2 | 560 ns | 560 ns | 1.00 |
| `eigh_f64` | n4 | 2.1 µs | 2.2 µs | 0.98 |
| `eigh_f64` | n8 | 6.4 µs | 6.5 µs | 1.00 |
| `eigh_f64` | n32 | 89.3 µs | 90.8 µs | 0.98 |
| `eigh_f64` | n128 | 1.82 ms | 1.77 ms | 1.03 |
| `eigh_f64` | n512 | 36.50 ms | 19.99 ms | 1.83 |
| `eigh_c64` | n2 | 610 ns | 630 ns | 0.97 |
| `eigh_c64` | n4 | 2.4 µs | 2.3 µs | 1.02 |
| `eigh_c64` | n8 | 6.8 µs | 6.8 µs | 1.01 |
| `eigh_c64` | n32 | 112.5 µs | 121.8 µs | 0.92 |
| `eigh_c64` | n128 | 2.71 ms | 2.57 ms | 1.06 |
| `eigh_c64` | n512 | 81.15 ms | 37.36 ms | 2.17 |
| `eig_f64` | n2 | 1.3 µs | 1.7 µs | 0.79 |
| `eig_f64` | n4 | 5.8 µs | 7.2 µs | 0.80 |
| `eig_f64` | n8 | 21.8 µs | 27.0 µs | 0.81 |
| `eig_f64` | n32 | 308.8 µs | 308.0 µs | 1.00 |
| `eig_f64` | n128 | 6.53 ms | 7.29 ms | 0.90 |
| `eig_f64` | n512 | 208.40 ms | 177.02 ms | 1.18 |
| `eig_c64` | n2 | 1.5 µs | 1.5 µs | 1.01 |
| `eig_c64` | n4 | 6.8 µs | 6.8 µs | 1.00 |
| `eig_c64` | n8 | 25.0 µs | 25.6 µs | 0.98 |
| `eig_c64` | n32 | 421.9 µs | 434.8 µs | 0.97 |
| `eig_c64` | n128 | 15.15 ms | 15.13 ms | 1.00 |
| `eig_c64` | n512 | 381.72 ms | 389.13 ms | 0.98 |
| `batched_solve_f64` | n2_b1024 | 181.7 µs | 181.5 µs | 1.00 |
| `batched_solve_f64` | n4_b1024 | 353.6 µs | 355.9 µs | 0.99 |
| `batched_solve_f64` | n8_b1024 | 883.9 µs | 915.8 µs | 0.97 |
| `batched_solve_f64` | n32_b64 | 408.0 µs | 151.6 µs | 2.69 |
| `batched_solve_c64` | n2_b1024 | 260.9 µs | 262.0 µs | 1.00 |
| `batched_solve_c64` | n4_b1024 | 524.0 µs | 522.7 µs | 1.00 |
| `batched_solve_c64` | n8_b1024 | 1.36 ms | 447.3 µs | 3.04 |
| `batched_solve_c64` | n32_b64 | 703.2 µs | 226.6 µs | 3.10 |
| `batched_cholesky_f64` | n2_b1024 | 61.6 µs | 61.6 µs | 1.00 |
| `batched_cholesky_f64` | n4_b1024 | 114.3 µs | 111.4 µs | 1.03 |
| `batched_cholesky_f64` | n8_b1024 | 243.1 µs | 239.9 µs | 1.01 |
| `batched_cholesky_f64` | n32_b64 | 196.2 µs | 196.7 µs | 1.00 |
| `batched_cholesky_c64` | n2_b1024 | 68.8 µs | 67.0 µs | 1.03 |
| `batched_cholesky_c64` | n4_b1024 | 128.5 µs | 128.4 µs | 1.00 |
| `batched_cholesky_c64` | n8_b1024 | 321.8 µs | 320.2 µs | 1.00 |
| `batched_cholesky_c64` | n32_b64 | 368.6 µs | 196.6 µs | 1.87 |
| `batched_eigh_f64` | n2_b1024 | 383.0 µs | 382.8 µs | 1.00 |
| `batched_eigh_f64` | n4_b1024 | 1.93 ms | 1.94 ms | 1.00 |
| `batched_eigh_f64` | n8_b1024 | 6.56 ms | 1.69 ms | 3.87 |
| `batched_eigh_f64` | n32_b64 | 5.89 ms | 1.51 ms | 3.91 |
| `batched_eigh_c64` | n2_b1024 | 481.7 µs | 481.4 µs | 1.00 |
| `batched_eigh_c64` | n4_b1024 | 2.28 ms | 729.8 µs | 3.13 |
| `batched_eigh_c64` | n8_b1024 | 7.34 ms | 1.90 ms | 3.87 |
| `batched_eigh_c64` | n32_b64 | 7.51 ms | 7.57 ms | 0.99 |
| `batched_svd_f64` | n2_b1024 | 640.7 µs | 640.1 µs | 1.00 |
| `batched_svd_f64` | n4_b1024 | 2.42 ms | 2.39 ms | 1.01 |
| `batched_svd_f64` | n8_b1024 | 8.23 ms | 2.08 ms | 3.95 |
| `batched_svd_f64` | n32_b64 | 7.01 ms | 1.77 ms | 3.96 |
| `batched_svd_c64` | n2_b1024 | 846.8 µs | 844.0 µs | 1.00 |
| `batched_svd_c64` | n4_b1024 | 3.20 ms | 815.1 µs | 3.93 |
| `batched_svd_c64` | n8_b1024 | 8.78 ms | 2.63 ms | 3.33 |
| `batched_svd_c64` | n32_b64 | 7.46 ms | 7.34 ms | 1.02 |

### Findings

1. **faer's parallel LU and QR lose to serial at n = 128** (measured before
   the fix: 0.6x LU, 0.4x QR at 4 threads, isolated runs) and gain only
   1.15-1.3x at n = 512 f64. They now use their own width policy (serial below
   an estimated 1 ms; `tprims-linalg/src/util.rs`), which is why the n = 128
   rows equal their 1T rows.
2. **The batched module scales where items are serial-sized and the batch is
   large**: eigh and svd at n = 8, batch 1024 reach 3.2-4.0x; solve at n = 32,
   batch 64 2.4-3.5x.
3. **Tiny batches stay serial** (n = 2 and 4 at batch 1024 for most ops), as
   in `blas`: the schedule's flop estimate ignores per-item overhead. The same
   refinement applies here.
4. **The nonsymmetric eigensolver does not scale** at n = 512 (1.18x f64,
   0.98x c64 at 4 threads); eigh and svd reach 1.5-2.3x there. This is faer's
   parallel efficiency, not investigated further here; `eig` may deserve the
   same kind of kernel-specific width policy as LU/QR once measured more
   widely.
