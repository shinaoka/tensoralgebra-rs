# exec_entry

Entry cost of the `tprims-exec` primitives, and two kernels driven through
`Exec`: a transposing strided map (`strided_basic::run_with_exec`) and a
512³ f64 GEMM through tensorcontract's `Spmd` seam (`Plan::run_with` with
`Exec::broadcast`). `# stats` lines give pool entry and broadcast counts per
case, so "serial work never enters the pool" is counted, not inferred.

```bash
cargo build -j 16 --release -p tprims-bench --bin exec_entry
taskset -c 19    target/release/exec_entry --threads 1
taskset -c 19-22 target/release/exec_entry --threads 4
```

Environment: `BENCH_RUNS` (default 200; large cases cap at 30 / 20),
`BENCH_WARMUP` (default 20), `BENCH_FILTER`. Timed boundary: one call of the
primitive or kernel; the GEMM plan is built outside the timed region, and the
GEMM result is checked bitwise against the serial run (`CHECK ... ok`).

## Observation, 2026-09-29 (single run, not a claim)

- tprims-rs `96a5dc1` (phase1a-exec) plus the harness sample-count fix;
  release profile (thin LTO, codegen-units 1); rustc 1.97.1.
- AMD EPYC 7713P (64 cores, 1 thread/core, 8 L3 domains of 8 cores), shared
  host with load average about 7.3; cores 19-22 (one L3 domain) measured idle
  (busy fraction ≤ 0.01) over 3 s before and after the runs.
- Raw output: [results/2026-09-29-1t.csv](results/2026-09-29-1t.csv),
  [results/2026-09-29-4t.csv](results/2026-09-29-4t.csv).

| Case | Variant | 1T median | 4T median | 4T entries / broadcasts |
| --- | --- | ---: | ---: | --- |
| `install_empty` | k1 | 30 ns | 30 ns | 0 / 0 |
| `install_empty` | k2 | — | 6.74 µs | 1 per call |
| `install_empty` | k4 | — | 6.81 µs | 1 per call |
| `partition_empty` | k2 | — | 6.91 µs | 1 per call |
| `partition_empty` | k4 | — | 6.85 µs | 1 per call |
| `broadcast_empty` | w4 | — | 7.59 µs | 1 broadcast per call |
| `strided_map_f64` | transpose 2^12 (below threshold) | 3.58 µs | 3.16 µs | 0 / 0 |
| `strided_map_f64` | transpose 2^22 | 14.77 ms | 4.09 ms (3.6x) | 1 per call |
| `tensorcontract_gemm_f64` | 512³ | 5.96 ms | 1.96 ms (3.0x) | 1 broadcast per call |

Width-one requests cost about 30 ns and never touch the pool; a small map
below strided's `MINTHREADLENGTH` stays on the caller at 4T. Entering the
pool costs about 7 µs at width 2 to 4 on this machine, in line with the
`experiments/rayon-entry` measurement on the M5 Max.
