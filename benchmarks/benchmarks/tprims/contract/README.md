# contract

`tprims-contract` at an enforced thread count on a predeclared corpus, both
strategies: permute plus batched GEMM (`pg`) and TBLIS-style direct through
Lukas Devos's `tensorcontract` (`tblis`). Planning (`ContractPlan::new`) and
execution (`execute`, alpha = 1, beta = 0) are timed separately.

| Case | A (storage order) | B (storage order) | Contraction |
| --- | --- | --- | --- |
| `tiny_matmul` | 2 x 2 | 2 x 2 | `ij,jk->ik` |
| `matmul_256` | 256 x 256 | 256 x 256 | `ij,jk->ik` |
| `batched_64_b32` | 64 x 64 x 32 | 64 x 64 x 32 | `ijb,jkb->ikb` |
| `permuted_fusable` | 64 x 32 x 32, stored (c, b, a) | 32 x 32 x 64, stored (d, b, c) | `abc,cbd->ad` |
| `permuted_nonfusable` | 64 x 32 x 32, stored (b, a, c) | 32 x 32 x 64, stored (b, a, d) | `abc,cbd->ad` (A must be copied) |
| `network_ijkl_klmn` | 16^4 | 16^4 | `ijkl,klmn->ijmn` |
| `large_ijk_jkl` | 256 x 64 x 64 | 64 x 64 x 256 | `ijk,jkl->il` |

```bash
cargo build -j 16 --release -p tprims-bench --bin contract
benchmarks/benchmarks/tprims/contract/run.sh 57 57-60 /tmp/contract   # 1T cores, 4T cores, output
```

`run.sh` runs every case in its own process, 1T and 4T back to back, and
keeps the `# selected` (materialized operands) and `CHECK` (pg vs tblis)
lines in a log.

## Observation, 2026-09-30 (single run, not a claim)

- tprims-rs `0e68641` (clean tree), release profile (thin LTO,
  codegen-units 1), rustc 1.97.1, faer 0.24.4.
- AMD EPYC 7713P, shared host (load average about 4); cores 57-60 (one L3
  domain) busy fraction <= 0.01 over 3 s before and after. Sub-microsecond
  rows are within the ~20% A/A noise floor measured for `linalg`.
- Raw output: [results/](results/). All `CHECK` lines ok (relative
  difference <= 1.5e-15); every case ran copy-free under `pg` except
  `permuted_nonfusable`, where A was materialized.

| Case | pg plan | pg 1T | pg 4T | pg 1T/4T | tblis plan | tblis 1T | tblis 4T | tblis 1T/4T | tblis/pg 1T | tblis/pg 4T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_matmul_f64` | 1.1 µs | 360 ns | 460 ns | 0.78 | 1.6 µs | 730 ns | 730 ns | 1.00 | 2.03 | 1.59 |
| `tiny_matmul_c64` | 1.3 µs | 500 ns | 380 ns | 1.32 | 1.8 µs | 870 ns | 720 ns | 1.21 | 1.74 | 1.89 |
| `matmul_256_f64` | 1.2 µs | 634.0 µs | 206.1 µs | 3.08 | 5.3 µs | 1.14 ms | 322.1 µs | 3.54 | 1.80 | 1.56 |
| `matmul_256_c64` | 1.2 µs | 2.48 ms | 854.0 µs | 2.90 | 5.3 µs | 2.86 ms | 999.8 µs | 2.86 | 1.15 | 1.17 |
| `batched_64_b32_f64` | 1.3 µs | 429.3 µs | 127.5 µs | 3.37 | 3.2 µs | 684.8 µs | 518.5 µs | 1.32 | 1.59 | 4.07 |
| `batched_64_b32_c64` | 1.2 µs | 1.54 ms | 675.2 µs | 2.28 | 3.2 µs | 2.23 ms | 964.3 µs | 2.31 | 1.45 | 1.43 |
| `permuted_fusable_f64` | 1.2 µs | 286.9 µs | 144.5 µs | 1.99 | 5.5 µs | 304.7 µs | 161.1 µs | 1.89 | 1.06 | 1.11 |
| `permuted_fusable_c64` | 1.2 µs | 837.9 µs | 260.4 µs | 3.22 | 4.6 µs | 1.01 ms | 317.2 µs | 3.17 | 1.20 | 1.22 |
| `permuted_nonfusable_f64` | 2.1 µs | 248.4 µs | 175.5 µs | 1.41 | 5.6 µs | 303.0 µs | 161.7 µs | 1.87 | 1.22 | 0.92 |
| `permuted_nonfusable_c64` | 1.9 µs | 719.3 µs | 399.1 µs | 1.80 | 4.7 µs | 825.7 µs | 350.2 µs | 2.36 | 1.15 | 0.88 |
| `network_ijkl_klmn_f64` | 1.5 µs | 737.6 µs | 239.2 µs | 3.08 | 6.8 µs | 1.36 ms | 277.2 µs | 4.90 | 1.84 | 1.16 |
| `network_ijkl_klmn_c64` | 1.4 µs | 2.47 ms | 812.3 µs | 3.04 | 5.8 µs | 2.85 ms | 995.7 µs | 2.86 | 1.15 | 1.23 |
| `large_ijk_jkl_f64` | 1.3 µs | 10.68 ms | 3.16 ms | 3.38 | 14.2 µs | 12.46 ms | 4.03 ms | 3.09 | 1.17 | 1.28 |
| `large_ijk_jkl_c64` | 1.2 µs | 44.03 ms | 12.77 ms | 3.45 | 14.2 µs | 47.82 ms | 14.97 ms | 3.19 | 1.09 | 1.17 |

### Findings

1. **Permute plus batched GEMM wins wherever the operands fuse**: TBLIS-style
   is 1.1-2.0x slower at 1T and 1.1-1.9x at 4T (4.1x for `batched_64_b32` at
   4T, where the pg path spreads the batch over the pool and tensorcontract's
   batch axis is serial inside each SPMD team).
2. **When an operand must be copied, TBLIS-style is competitive and wins at
   4T** (`permuted_nonfusable`: tblis/pg 0.92 f64, 0.88 c64 at 4T), because
   the pg path's copy of A is serial-bandwidth work that the direct kernel
   avoids. This is the first case where the direct strategy is preferable;
   an `Auto` rule "TBLIS when pg would materialize a large operand" is a
   candidate, to be decided on a wider corpus.
3. Planning costs 1-2 us for pg and 3-14 us for tblis (scatter vectors of
   length M + N + K); both are outside `execute`.
