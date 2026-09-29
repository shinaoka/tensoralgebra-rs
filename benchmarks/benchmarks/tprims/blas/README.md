# blas

`tprims-blas` at an enforced thread count: `gemm`, `gemm_batched` with both
strategies (faer plus a loop over items; TBLIS-style through Lukas Devos's
`tensorcontract`), and `trsm`, for f64 and c64. Column-major operands;
`gemm_batched` batch axis last. `# selected` lines show the schedule each
batched call chose; `CHECK` lines compare the two strategies' outputs.

```bash
cargo build -j 16 --release -p tprims-bench --bin blas
taskset -c 57    target/release/blas --threads 1
taskset -c 57-60 target/release/blas --threads 4
```

Environment: `BENCH_RUNS` (default 50; capped at 15 above 5e8 flops and 5
above 5e9), `BENCH_WARMUP` (5), `BENCH_FILTER`. Timed boundary: one public
call, including validation and (for TBLIS) plan construction.

## Observation, 2026-09-29 (single run, not a claim)

- tprims-rs `07fe918` (clean tree), release profile (thin LTO,
  codegen-units 1), rustc 1.97.1, faer 0.24.4.
- AMD EPYC 7713P, shared host (load average about 7); cores 57-60 (one L3
  domain) busy fraction ≤ 0.07 over 3 s before and after.
- Raw output: [results/2026-09-29-1t.csv](results/2026-09-29-1t.csv),
  [results/2026-09-29-4t.csv](results/2026-09-29-4t.csv). All `CHECK` lines ok.

| Case | Variant | 1T | 4T | 1T/4T |
| --- | --- | ---: | ---: | ---: |
| `gemm_f64` | n8 | 170 ns | 170 ns | 1.00 |
| `gemm_f64` | n32 | 2.09 µs | 2.40 µs | **0.87** |
| `gemm_f64` | n128 | 78.8 µs | 43.5 µs | 1.81 |
| `gemm_f64` | n512 | 5.17 ms | 1.59 ms | 3.26 |
| `gemm_f64` | n512, Aᵀ | 5.16 ms | 1.52 ms | 3.39 |
| `gemm_f64` | n1024 | 42.9 ms | 12.3 ms | 3.48 |
| `gemm_c64` | n8 | 330 ns | 330 ns | 1.00 |
| `gemm_c64` | n32 | 6.91 µs | 6.92 µs | 1.00 |
| `gemm_c64` | n128 | 377 µs | 112 µs | 3.36 |
| `gemm_c64` | n512 | 23.3 ms | 5.90 ms | 3.96 |
| `gemm_c64` | n1024 | 173 ms | 48.1 ms | 3.59 |
| `trsm_f64` | n = nrhs = 256 | 571 µs | 238 µs | 2.40 |
| `trsm_f64` | n = nrhs = 1024 | 32.0 ms | 8.82 ms | 3.63 |
| `trsm_c64` | n = nrhs = 1024 | 108 ms | 28.5 ms | 3.78 |

Batched GEMM, both strategies (TBLIS/faer is the time ratio):

| dtype | n, batch | faer 1T | faer 4T | TBLIS 1T | TBLIS 4T | TBLIS/faer 1T | 4T |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 2, 1024 | 27.9 µs | 34.2 µs | 495 µs | 588 µs | 17.8 | 17.2 |
| f64 | 4, 1024 | 39.6 µs | 47.1 µs | 542 µs | 643 µs | 13.7 | 13.6 |
| f64 | 8, 1024 | 66.3 µs | 42.4 µs | 710 µs | 329 µs | 10.7 | 7.8 |
| f64 | 32, 256 | 440 µs | 136 µs | 1.03 ms | 335 µs | 2.3 | 2.5 |
| f64 | 128, 16 | 1.28 ms | 659 µs | 1.77 ms | 967 µs | 1.4 | 1.5 |
| f64 | 512, 1 | 5.16 ms | 1.54 ms | 5.59 ms | 1.89 ms | 1.1 | 1.2 |
| c64 | 2, 1024 | 38.4 µs | 44.6 µs | 498 µs | 601 µs | 13.0 | 13.5 |
| c64 | 8, 1024 | 211 µs | 72.7 µs | 918 µs | 301 µs | 4.4 | 4.1 |
| c64 | 32, 256 | 1.50 ms | 456 µs | 2.76 ms | 846 µs | 1.8 | 1.9 |
| c64 | 128, 16 | 5.29 ms | 1.84 ms | 6.70 ms | 2.39 ms | 1.3 | 1.3 |
| c64 | 512, 1 | 21.8 ms | 5.90 ms | 26.5 ms | 7.01 ms | 1.2 | 1.2 |

### Findings

1. **faer loop wins at every measured shape.** TBLIS-style is 10-18x slower
   for tiny items (per-call scatter construction and panel allocation inside
   tensorcontract dominate) and 1.1-1.5x slower from n = 128. `Auto` stays
   `FaerLoop`.
2. **Serial-path rows are 13-18% slower in the 4T process** (gemm f64 n32;
   batched n2/n4 at batch 1024, which the schedule keeps serial: total work
   below the 50 µs threshold). The same code path runs in both processes, so
   the difference is environmental (4-core affinity, idle pool workers,
   frequency) or a harness effect, not parallel overhead; it is recorded as a
   finding to investigate, not explained.
3. **Batch schedule threshold is based on a flop estimate** (0.05 ns/flop)
   that ignores per-item call overhead (~27 ns/item for faer at n = 2), so the
   tiny batches stay serial even at batch 1024 where outer parallelism might
   pay. A per-item overhead term is the next policy refinement, to be decided
   by measurement.
