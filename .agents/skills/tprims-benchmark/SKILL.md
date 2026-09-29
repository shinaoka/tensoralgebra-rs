---
name: tprims-benchmark
description: Use when running, adding or reporting a tprims-rs benchmark — the tprims-bench binaries (exec_entry, blas, linalg, contract, capi_rust), the C benchmark, or tenferro-benchmark runs that compare tprims against tenferro's default backend. Covers building, choosing idle cores in one L3 domain, pinning, paired thread counts, the A/A noise floor and recording results; the rules themselves live in PERFORMANCE_TIPS.md.
---

# tprims benchmark

The procedure for measuring. The rules it applies are in `PERFORMANCE_TIPS.md`
(`Performance-Sensitive Tests And Benchmarks`, `Performance-Gated Experiment
Protocol`, `CPU Threading Contract`); read those sections first.

1. **Quiet host.** Never run two benchmarks at once, and do not build or test
   while measuring. Look for other users' load:
   `ps -eo pid,user,psr,%cpu,comm --sort=-%cpu | head`.
2. **Build once, release profile.** The build job count depends on the host:
   set `CARGO_BUILD_JOBS` for it (for example 16 on the shared 64-core EPYC)
   instead of hardcoding `-j`; cargo and the runners read it.
   `cargo build --release -p tprims-bench --bins`
   (the C benchmark builds itself in `benchmarks/c/run.sh`). Build
   parallelism does not change the measured thread count.
3. **Choose idle cores in one L3 domain:**
   `python3 benchmarks/scripts/idle_cpus.py pick N` prints N idle CPUs of one
   L3 domain (one hardware thread per core), for example `8,9,10,11`. Pick for
   the largest thread count of the run and use a prefix for the smaller ones
   (1T = the first CPU). A full-domain count (8 on the EPYC 7713P) needs a
   whole idle L3 domain; if `pick` exits 2, wait, do not widen across domains.
4. **Run each measurement through `benchmarks/scripts/pinned.sh CPUS -- CMD`.**
   It pins with `taskset`, checks the cores are idle before and after, keeps
   only valid runs and retries spoiled ones. The per-benchmark runners already
   use it and pair thread counts per case, one process per case:
   `benchmarks/benchmarks/tprims/{linalg,contract}/run.sh CPUS1 CPUS4 OUT`,
   `benchmarks/c/run.sh CPUS1 CPUS4 OUT`. For a single binary:
   `BENCH_CASE=<case> benchmarks/scripts/pinned.sh 8 -- target/release/blas --threads 1`.
5. **Thread counts:** every tensor-sized case at 1T and 4T in the same run;
   add 8T (one full L3 domain) where the experiment calls for it. The binary
   asserts its effective width at startup and rejects conflicting
   `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
   `TENSORCONTRACT_THREADS`; do not set them.
6. **Noise floor:** run the same binary twice on the same cores minutes apart
   (A/A) and report the spread; differences below it are not findings.
7. **Record** beside every published table: tprims-rs commit (and tenferro-rs /
   tenferro-benchmark commits for tenferro runs), CPU model, core set, profile,
   thread counts, `pinned.sh` attempts, and the A/A spread. Raw CSVs go under
   the benchmark's `results/`, summaries in its `README.md`. Report negative
   and inconclusive results as such.
8. **tenferro-benchmark runs** (tprims provider vs default backend) use the
   same core choice and `pinned.sh`, with tenferro's paired ABBA runner
   (`scripts/run_paired_timing.sh`) as the command and its idle-host guard
   left enabled. tenferro-benchmark's devcontainer suites do not pin by
   default; these runs do, and say so in the result.
9. **CPU affinity exists only on Linux.** On other hosts (macOS, Windows)
   `idle_cpus.py` exits 3 and `pinned.sh` runs the command unpinned with a
   note; state in the result that pinning was unavailable.
