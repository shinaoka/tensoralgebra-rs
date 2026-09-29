@CLAUDE.md

## Critical Rules

- **NEVER run benchmarks in parallel.** Benchmarks must be run sequentially (1-thread first, then 4-thread, etc.). Running them concurrently causes interference between benchmark processes, producing unreliable and misleading results.
- **Record the exact `strided-rs` git hash for every published benchmark result table.** Before running benchmarks, commit all changes in `strided-rs` and record the hash used to produce the numbers. Include that hash next to every updated result table, not just in ad-hoc notes, so each published result can be traced to the implementation that produced it.
- **Always pin CPU cores with `taskset` for ALL Linux benchmarks (Rust and Julia, including 1T).** Use `taskset -c 0` (1T), `taskset -c 0-3` (4T), `taskset -c 0-7` (8T) to bind benchmark processes to specific cores within the same L3/CCD domain. Without pinning, threads migrate across CCDs on AMD EPYC, causing L3 cache misses and up to 24% performance degradation. Before running, check `top` or `ps -eo pid,psr,%cpu,comm --sort=-%cpu | head` for CPU-intensive processes from other users, and avoid their CCD (use `lscpu -e` to map core→L3 domain). On macOS, state that CPU pinning was unavailable.
