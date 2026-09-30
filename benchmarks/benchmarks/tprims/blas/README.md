# blas

`tprims-blas` at an enforced thread count: `gemm`, `gemm_batched` with both
strategies (faer plus a loop over items; TBLIS-style through Lukas Devos's
`tensorcontract`), `gemm_grouped` (variable-size jobs in shared buffers,
against a loop of `gemm` calls on the same blocks), and `trsm`, for f64 and
c64. Column-major operands;
`gemm_batched` batch axis last. `# selected` lines show the schedule each
batched call chose; `CHECK` lines compare the two strategies' outputs.

```bash
cargo build --release -p tprims-bench --bin blas
# recorded batched-GEMM shapes, faer loop vs TBLIS-style, paired thread counts:
CORPUS=path/to/corpus.json benchmarks/benchmarks/tprims/blas/run.sh /tmp/blas "$(python3 benchmarks/scripts/idle_cpus.py pick 8)" 1 4 8
taskset -c 57    target/release/blas --threads 1
taskset -c 57-60 target/release/blas --threads 4
```

Environment: `BENCH_RUNS` (default 50; capped at 15 above 5e8 flops and 5
above 5e9), `BENCH_WARMUP` (5), `BENCH_FILTER`. Timed boundary: one public
call, including validation and (for TBLIS) plan construction.

## Observation, 2026-09-30 (single run, not a claim)

- tprims-rs `c181402` (clean tree; after the Phase 1b review fixes),
  release profile (thin LTO, codegen-units 1), rustc 1.97.1, faer 0.24.4.
- AMD EPYC 7713P, shared host (load average about 5.6); cores 1-4 (one L3
  domain) busy fraction ≤ 0.05 over 3 s before and after.
- Raw output: [results/2026-09-30-1t.csv](results/2026-09-30-1t.csv),
  [results/2026-09-30-4t.csv](results/2026-09-30-4t.csv). All `CHECK` lines
  ok. An earlier run on cores 57-60 at `07fe918` is kept in
  `results/2026-09-29-*.csv`; the two runs are on different cores and are not
  a paired comparison.

| Case | Variant | 1T | 4T | 1T/4T |
| --- | --- | ---: | ---: | ---: |
| `gemm_f64` | n8 | 170 ns | 140 ns | 1.21 |
| `gemm_f64` | n32 | 2.2 µs | 1.7 µs | 1.30 |
| `gemm_f64` | n128 | 93.5 µs | 43.2 µs | 2.16 |
| `gemm_f64` | n512 | 6.18 ms | 1.52 ms | 4.06 |
| `gemm_f64` | n512, Aᵀ | 6.16 ms | 1.50 ms | 4.10 |
| `gemm_f64` | n1024 | 50.3 ms | 11.3 ms | 4.46 |
| `gemm_f64` | 256 x 1024 x 64 | 750 µs | 238 µs | 3.15 |
| `gemm_f64` | 1024 x 64 x 256 | 793 µs | 212 µs | 3.74 |
| `gemm_f64` | n2048 | 382 ms | 97.3 ms | 3.93 |
| `gemm_c64` | n8 | 330 ns | 330 ns | 1.00 |
| `gemm_c64` | n32 | 7.0 µs | 6.9 µs | 1.01 |
| `gemm_c64` | n128 | 375 µs | 113 µs | 3.31 |
| `gemm_c64` | n512 | 20.8 ms | 5.92 ms | 3.52 |
| `gemm_c64` | n1024 | 201 ms | 45.5 ms | 4.41 |
| `gemm_c64` | 256 x 1024 x 64 | 2.96 ms | 841 µs | 3.52 |
| `trsm_f64` | n = nrhs = 256 | 534 µs | 251 µs | 2.13 |
| `trsm_f64` | n = nrhs = 1024 | 30.1 ms | 9.34 ms | 3.22 |
| `trsm_c64` | n = nrhs = 256 | 2.00 ms | 664 µs | 3.02 |
| `trsm_c64` | n = nrhs = 1024 | 91.3 ms | 26.7 ms | 3.42 |

(`n = 32` trsm and `n <= 32` c64 gemm stay serial by the width rule; their
1T and 4T rows agree.)

Batched GEMM, both strategies (TBLIS/faer is the time ratio):

| dtype | n, batch | faer 1T | faer 4T | TBLIS 1T | TBLIS 4T | TBLIS/faer 1T | 4T |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 2, 1 | 160 ns | 160 ns | 2.2 µs | 2.1 µs | 13.6 | 13.1 |
| f64 | 2, 1024 | 32.6 µs | 32.3 µs | 576 µs | 582 µs | 17.7 | 18.0 |
| f64 | 2, 4096 | 130 µs | 129 µs | 2.31 ms | 2.01 ms | 17.8 | 15.6 |
| f64 | 4, 1024 | 46.4 µs | 46.7 µs | 621 µs | 624 µs | 13.4 | 13.3 |
| f64 | 8, 1024 | 76.3 µs | 42.9 µs | 829 µs | 364 µs | 10.9 | 8.5 |
| f64 | 32, 256 | 495 µs | 148 µs | 1.22 ms | 279 µs | 2.5 | 1.9 |
| f64 | 128, 16 | 1.52 ms | 578 µs | 2.09 ms | 968 µs | 1.4 | 1.7 |
| f64 | 512, 1 | 6.17 ms | 1.52 ms | 6.73 ms | 1.86 ms | 1.1 | 1.2 |
| c64 | 2, 1024 | 44.5 µs | 45.3 µs | 599 µs | 504 µs | 13.5 | 11.1 |
| c64 | 2, 4096 | 179 µs | 153 µs | 2.40 ms | 2.18 ms | 13.4 | 14.2 |
| c64 | 4, 1024 | 66.3 µs | 67.1 µs | 661 µs | 564 µs | 10.0 | 8.4 |
| c64 | 8, 1024 | 248 µs | 74.2 µs | 1.03 ms | 324 µs | 4.2 | 4.4 |
| c64 | 32, 256 | 1.76 ms | 400 µs | 3.24 ms | 766 µs | 1.8 | 1.9 |
| c64 | 128, 16 | 6.08 ms | 1.72 ms | 7.60 ms | 2.37 ms | 1.2 | 1.4 |
| c64 | 512, 1 | 24.8 ms | 5.93 ms | 27.4 ms | 7.05 ms | 1.1 | 1.2 |

### Findings

1. **The faer loop wins at every measured shape.** TBLIS-style is 10-18x
   slower for n ≤ 4, 4-11x at n = 8, and 1.1-1.7x from n = 128.
   *Hypothesis, not profiled:* per-call scatter construction and panel
   allocation inside tensorcontract dominate tiny items. `Auto` stays
   `FaerLoop`.
2. **Tiny batches stay serial even when outer parallelism could pay.** The
   schedule uses a flop estimate (0.05 ns/flop) that ignores per-item call
   overhead: 2x2 at batch 4096 takes 130 µs but is estimated at about 3 µs,
   below the 50 µs serial threshold, so the 4T row equals the 1T row. A
   per-item overhead term in the policy is the next refinement, to be decided
   by a paired measurement.
3. The earlier run on cores 57-60 showed serial-path rows 15-23% slower in
   the 4T process; this run on cores 1-4 does not (n = 2, batch 1024: 32.6 µs
   vs 32.3 µs). It was not reproduced and is treated as host noise.

## Phase 1e P2: batched GEMM corpora (2026-09-30)

faer loop (`faer`) against TBLIS-style (`tblis`) on two corpora, tprims-rs
`d8e565a`, release profile, EPYC 7713P, one CCD per session (CPUs 0-7; the
`tenferro-p1-gemm` session 2 ran on CPUs 8-15), 1T/4T/8T paired per case,
three sessions ([`results/2026-09-30-p2/`](results/2026-09-30-p2/), manifests
and `decision.txt` inside). All `CHECK` lines ok.

- [`tenferro-p1-gemm.json`](../corpus/tenferro-p1-gemm.json): 53 GEMM shape
  groups of tenferro-benchmark's einsum and `cpu/public_api` suites with only
  the GEMM slot on tprims (`TPRIMS_ROUTES=gemm`), weighted by call count —
  mostly single large GEMMs (for example 990x2187x4620, 4782969x27x9) and a
  few batched ones (batch 4-2187).
- [`large-batched-gemm.json`](../corpus/large-batched-gemm.json): synthetic
  large batched GEMM, 128^3 x 512 to 4096^3 x 1, f64 and c64, equal weights.

| corpus | 1T faer/tblis | 4T | 8T | noise |
| --- | --- | --- | --- | --- |
| tenferro-p1-gemm (workload time) | 0.82 | 0.73-0.75 | 0.68-0.69 | 0.4-2.3% |
| large-batched-gemm (workload time) | 0.93-0.94 | 0.86-0.88 | 0.85 | 0.8-3.1% |

faer is faster at every thread count in every session; `BatchStrategy::Auto`
stays the faer loop. Per case (session 3) faer/tblis ranges 0.47-0.98 at 1T
and 0.65-0.99 at 8T; the gap is widest for many small items (128^3 x 512:
0.47 f64 at 1T) and narrowest for single large GEMMs (4096^3: 0.94 at 1T,
0.87 at 8T). Since no operand needs a copy here, this is the GEMM engine
itself; see the source study
[`docs/worklogs/2026-09-30-gemm-strategy-faer-vs-tensorcontract.md`](../../../../docs/worklogs/2026-09-30-gemm-strategy-faer-vs-tensorcontract.md).
