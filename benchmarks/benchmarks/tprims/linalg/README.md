# linalg

`tprims-linalg` at an enforced thread count: single-matrix factorizations
(cholesky, lu, solve with one right-hand side, qr, thin svd, eigh with
vectors, eig with vectors) at n = 2 to 512, and the `batched` module
(solve, cholesky, eigh with vectors, svd values) including the small-matrix,
large-batch regression case (n <= 8, batch 1024). f64 and c64. The timed
boundary is one public call including the input copy into factor storage.

```bash
cargo build --release -p tprims-bench --bin linalg
benchmarks/benchmarks/tprims/linalg/run.sh 19 19-22 /tmp/linalg   # 1T cores, 4T cores, output
```

`run.sh` measures every case in its own process (`BENCH_CASE`) with the 1T
and 4T runs of a case back to back. Both matter: parallel cases leave the pool
in a state that slows later serial rows in the same process, and slow host
drift biased a run that measured all 1T cases before all 4T cases.

## Observation, 2026-09-30 (single run, not a claim)

- tprims-rs `65dabc1` (clean tree; after the Phase 1d review fixes), release
  profile (thin LTO, codegen-units 1), rustc 1.97.1, faer 0.24.4.
- AMD EPYC 7713P, shared host (load average about 5.4); cores 19-22 (one L3
  domain) idle (busy fraction 0.00) over 3 s before and after.
- Raw output: [results/2026-09-30-1t.csv](results/2026-09-30-1t.csv),
  [results/2026-09-30-4t.csv](results/2026-09-30-4t.csv); LU/QR before their
  width policy: [results/2026-09-30-lu-qr-before-policy.txt](results/2026-09-30-lu-qr-before-policy.txt).

**Serial-path rows are not a clean comparison.** Rows whose width is one run
the identical code in the 1T and 4T processes, yet a full `run.sh` pass shows
some of them 10-30% slower in the 4T process. Which rows varies from pass to
pass (an earlier pass: `eig_f64`, `qr_f64`, `cholesky_f64` at n <= 32; this
pass: `solve_f64`, `svd_f64` at n <= 32), and re-running such a case alone
gives equal times (`eig_f64` n8: 21.7 us at 4T on cores 19-22 vs 21.8 us at 1T
on core 19, twice, in both pinning variants). The A/A noise floor of the same
1T binary is up to 20% below 1 us. The effect is attributed to run ordering
(each case's 4T run follows its heavy 1T run) and host state, not to the
serial code path, and is not fully explained.

| Case | Variant | 1T | 4T | 1T/4T |
| --- | --- | ---: | ---: | ---: |
| `cholesky_f64` | n2 | 210 ns | 170 ns | 1.24 |
| `cholesky_f64` | n4 | 290 ns | 230 ns | 1.26 |
| `cholesky_f64` | n8 | 481 ns | 400 ns | 1.20 |
| `cholesky_f64` | n32 | 4.6 µs | 3.7 µs | 1.24 |
| `cholesky_f64` | n128 | 65.9 µs | 66.9 µs | 0.99 |
| `cholesky_f64` | n512 | 1.94 ms | 1.53 ms | 1.26 |
| `cholesky_c64` | n2 | 170 ns | 190 ns | 0.89 |
| `cholesky_c64` | n4 | 240 ns | 240 ns | 1.00 |
| `cholesky_c64` | n8 | 420 ns | 420 ns | 1.00 |
| `cholesky_c64` | n32 | 5.4 µs | 5.4 µs | 1.01 |
| `cholesky_c64` | n128 | 148.9 µs | 145.4 µs | 1.02 |
| `cholesky_c64` | n512 | 5.70 ms | 3.46 ms | 1.65 |
| `lu_f64` | n2 | 200 ns | 200 ns | 1.00 |
| `lu_f64` | n4 | 330 ns | 330 ns | 1.00 |
| `lu_f64` | n8 | 700 ns | 690 ns | 1.01 |
| `lu_f64` | n32 | 5.9 µs | 5.8 µs | 1.01 |
| `lu_f64` | n128 | 104.8 µs | 102.5 µs | 1.02 |
| `lu_f64` | n512 | 3.76 ms | 2.97 ms | 1.26 |
| `lu_c64` | n2 | 220 ns | 220 ns | 1.00 |
| `lu_c64` | n4 | 380 ns | 390 ns | 0.97 |
| `lu_c64` | n8 | 780 ns | 760 ns | 1.03 |
| `lu_c64` | n32 | 8.8 µs | 8.9 µs | 0.99 |
| `lu_c64` | n128 | 245.3 µs | 203.2 µs | 1.21 |
| `lu_c64` | n512 | 8.39 ms | 4.26 ms | 1.97 |
| `solve_f64` | n2 | 410 ns | 530 ns | 0.77 |
| `solve_f64` | n4 | 630 ns | 790 ns | 0.80 |
| `solve_f64` | n8 | 1.2 µs | 1.4 µs | 0.81 |
| `solve_f64` | n32 | 7.5 µs | 9.5 µs | 0.79 |
| `solve_f64` | n128 | 113.5 µs | 95.7 µs | 1.19 |
| `solve_f64` | n512 | 3.67 ms | 3.14 ms | 1.17 |
| `solve_c64` | n2 | 450 ns | 460 ns | 0.98 |
| `solve_c64` | n4 | 730 ns | 740 ns | 0.99 |
| `solve_c64` | n8 | 1.4 µs | 1.4 µs | 0.97 |
| `solve_c64` | n32 | 11.2 µs | 11.2 µs | 1.00 |
| `solve_c64` | n128 | 222.3 µs | 255.0 µs | 0.87 |
| `solve_c64` | n512 | 10.21 ms | 5.20 ms | 1.96 |
| `qr_f64` | n2 | 250 ns | 250 ns | 1.00 |
| `qr_f64` | n4 | 470 ns | 470 ns | 1.00 |
| `qr_f64` | n8 | 1.1 µs | 1.1 µs | 1.00 |
| `qr_f64` | n32 | 17.5 µs | 14.8 µs | 1.18 |
| `qr_f64` | n128 | 273.5 µs | 232.7 µs | 1.18 |
| `qr_f64` | n512 | 6.67 ms | 4.50 ms | 1.48 |
| `qr_c64` | n2 | 310 ns | 310 ns | 1.00 |
| `qr_c64` | n4 | 580 ns | 580 ns | 1.00 |
| `qr_c64` | n8 | 1.5 µs | 1.5 µs | 0.99 |
| `qr_c64` | n32 | 24.8 µs | 24.7 µs | 1.00 |
| `qr_c64` | n128 | 613.9 µs | 648.9 µs | 0.95 |
| `qr_c64` | n512 | 18.13 ms | 10.00 ms | 1.81 |
| `svd_f64` | n2 | 1.2 µs | 1.6 µs | 0.80 |
| `svd_f64` | n4 | 3.3 µs | 4.8 µs | 0.68 |
| `svd_f64` | n8 | 11.2 µs | 9.5 µs | 1.18 |
| `svd_f64` | n32 | 152.1 µs | 151.5 µs | 1.00 |
| `svd_f64` | n128 | 2.73 ms | 1.91 ms | 1.43 |
| `svd_f64` | n512 | 64.22 ms | 40.58 ms | 1.58 |
| `svd_c64` | n2 | 1.3 µs | 1.3 µs | 0.99 |
| `svd_c64` | n4 | 3.7 µs | 3.7 µs | 0.99 |
| `svd_c64` | n8 | 11.7 µs | 11.8 µs | 0.99 |
| `svd_c64` | n32 | 196.6 µs | 206.9 µs | 0.95 |
| `svd_c64` | n128 | 4.48 ms | 3.34 ms | 1.34 |
| `svd_c64` | n512 | 141.60 ms | 66.18 ms | 2.14 |
| `eigh_f64` | n2 | 550 ns | 580 ns | 0.95 |
| `eigh_f64` | n4 | 2.2 µs | 2.2 µs | 1.00 |
| `eigh_f64` | n8 | 6.6 µs | 6.5 µs | 1.01 |
| `eigh_f64` | n32 | 75.9 µs | 75.0 µs | 1.01 |
| `eigh_f64` | n128 | 1.55 ms | 1.76 ms | 0.88 |
| `eigh_f64` | n512 | 31.10 ms | 18.21 ms | 1.71 |
| `eigh_c64` | n2 | 610 ns | 610 ns | 1.00 |
| `eigh_c64` | n4 | 2.3 µs | 2.3 µs | 1.01 |
| `eigh_c64` | n8 | 6.8 µs | 6.8 µs | 1.00 |
| `eigh_c64` | n32 | 112.1 µs | 102.4 µs | 1.09 |
| `eigh_c64` | n128 | 2.71 ms | 2.50 ms | 1.08 |
| `eigh_c64` | n512 | 68.21 ms | 32.35 ms | 2.11 |
| `eig_f64` | n2 | 1.3 µs | 1.6 µs | 0.84 |
| `eig_f64` | n4 | 5.7 µs | 5.8 µs | 0.99 |
| `eig_f64` | n8 | 21.5 µs | 21.7 µs | 0.99 |
| `eig_f64` | n32 | 258.9 µs | 257.0 µs | 1.01 |
| `eig_f64` | n128 | 5.45 ms | 6.12 ms | 0.89 |
| `eig_f64` | n512 | 172.62 ms | 160.43 ms | 1.08 |
| `eig_c64` | n2 | 1.5 µs | 1.5 µs | 1.01 |
| `eig_c64` | n4 | 6.9 µs | 6.9 µs | 1.00 |
| `eig_c64` | n8 | 25.1 µs | 25.1 µs | 1.00 |
| `eig_c64` | n32 | 362.3 µs | 363.0 µs | 1.00 |
| `eig_c64` | n128 | 12.94 ms | 13.73 ms | 0.94 |
| `eig_c64` | n512 | 356.83 ms | 243.44 ms | 1.47 |
| `batched_solve_f64` | n2_b1024 | 195.1 µs | 194.9 µs | 1.00 |
| `batched_solve_f64` | n4_b1024 | 376.2 µs | 377.8 µs | 1.00 |
| `batched_solve_f64` | n8_b1024 | 927.0 µs | 767.5 µs | 1.21 |
| `batched_solve_f64` | n32_b64 | 354.0 µs | 188.0 µs | 1.88 |
| `batched_solve_f64` | n128_b16 | 1.36 ms | 451.6 µs | 3.02 |
| `batched_solve_c64` | n2_b1024 | 258.6 µs | 214.1 µs | 1.21 |
| `batched_solve_c64` | n4_b1024 | 526.0 µs | 440.9 µs | 1.19 |
| `batched_solve_c64` | n8_b1024 | 1.14 ms | 374.5 µs | 3.04 |
| `batched_solve_c64` | n32_b64 | 618.7 µs | 195.2 µs | 3.17 |
| `batched_solve_c64` | n128_b16 | 3.39 ms | 1.07 ms | 3.16 |
| `batched_cholesky_f64` | n2_b1024 | 60.9 µs | 60.7 µs | 1.00 |
| `batched_cholesky_f64` | n4_b1024 | 112.8 µs | 112.5 µs | 1.00 |
| `batched_cholesky_f64` | n8_b1024 | 242.7 µs | 238.3 µs | 1.02 |
| `batched_cholesky_f64` | n32_b64 | 165.0 µs | 194.4 µs | 0.85 |
| `batched_cholesky_f64` | n128_b16 | 790.7 µs | 407.6 µs | 1.94 |
| `batched_cholesky_c64` | n2_b1024 | 68.5 µs | 56.9 µs | 1.20 |
| `batched_cholesky_c64` | n4_b1024 | 132.0 µs | 108.3 µs | 1.22 |
| `batched_cholesky_c64` | n8_b1024 | 262.7 µs | 256.9 µs | 1.02 |
| `batched_cholesky_c64` | n32_b64 | 282.9 µs | 163.5 µs | 1.73 |
| `batched_cholesky_c64` | n128_b16 | 1.99 ms | 844.1 µs | 2.36 |
| `batched_eigh_f64` | n2_b1024 | 356.4 µs | 357.8 µs | 1.00 |
| `batched_eigh_f64` | n4_b1024 | 1.62 ms | 1.62 ms | 1.00 |
| `batched_eigh_f64` | n8_b1024 | 5.48 ms | 1.69 ms | 3.24 |
| `batched_eigh_f64` | n32_b64 | 4.96 ms | 1.52 ms | 3.27 |
| `batched_eigh_f64` | n128_b16 | 23.99 ms | 6.68 ms | 3.59 |
| `batched_eigh_c64` | n2_b1024 | 456.9 µs | 384.6 µs | 1.19 |
| `batched_eigh_c64` | n4_b1024 | 1.92 ms | 593.0 µs | 3.25 |
| `batched_eigh_c64` | n8_b1024 | 6.14 ms | 1.61 ms | 3.81 |
| `batched_eigh_c64` | n32_b64 | 6.34 ms | 1.64 ms | 3.85 |
| `batched_eigh_c64` | n128_b16 | 36.13 ms | 9.32 ms | 3.88 |
| `batched_svd_f64` | n2_b1024 | 625.0 µs | 629.4 µs | 0.99 |
| `batched_svd_f64` | n4_b1024 | 1.98 ms | 1.98 ms | 1.00 |
| `batched_svd_f64` | n8_b1024 | 6.88 ms | 1.81 ms | 3.80 |
| `batched_svd_f64` | n32_b64 | 5.97 ms | 1.51 ms | 3.94 |
| `batched_svd_f64` | n128_b16 | 25.80 ms | 6.56 ms | 3.94 |
| `batched_svd_c64` | n2_b1024 | 827.7 µs | 822.9 µs | 1.01 |
| `batched_svd_c64` | n4_b1024 | 2.68 ms | 819.6 µs | 3.27 |
| `batched_svd_c64` | n8_b1024 | 9.21 ms | 2.65 ms | 3.48 |
| `batched_svd_c64` | n32_b64 | 7.29 ms | 2.14 ms | 3.41 |
| `batched_svd_c64` | n128_b16 | 39.12 ms | 11.17 ms | 3.50 |

### Findings

1. **faer's parallel LU, QR and Cholesky lose to serial at n = 128** (before
   the policy, isolated runs: LU 0.63x, QR 0.43x, c64 Cholesky 0.76x at 4
   threads) and gain only 1.2-1.5x at n = 512. They use their own width
   policy (serial below an estimated 1 ms; `tprims-linalg/src/util.rs`); the
   n = 128 rows now match their 1T rows (c64 Cholesky 1.02).
2. **The batched module scales when the batch is spread over the pool**:
   eigh and svd at n >= 8 (c64 from n = 4) reach 3.2-3.9x, solve and cholesky
   at n = 128, batch 16 1.9-3.2x. The review fix (items split whenever the
   batch is at least as long as the budget) turned c64 eigh/svd at n = 32,
   batch 64 from 1.0x into 3.4-3.9x and removed a 2.5x slowdown of solve at
   n = 128, batch 16.
3. **Tiny batches stay serial** (n = 2 at batch 1024, and n = 4 f64): the
   schedule's flop estimate ignores per-item overhead, as in `blas`.
4. **Large single-matrix eigensolvers scale modestly**: eig 1.1x (f64) and
   1.5x (c64) at n = 512; eigh and svd 1.6-2.1x. This is faer's parallel
   efficiency, not investigated further here.
