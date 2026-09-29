# Dense CPU kernels

Nonzero correctness checks and focused 1T measurements for tenferro's migrated
AXPBY, triangular masks and diagonal embedding, plus the existing shared
multiply SIMD path and uninitialized permutation copies. Requires strided-rs
`78d519013e8b44bd80f77eb01d31039f0be9aae6` or compatible descendants.
The merged integration revision is `5bc5ab75a20277f0c8820cb288b23f6bb6dfbd91`
(strided-rs PR259, on `umbrella/issue-burndown-2026-08`, not yet `main`).
Check out that revision in the existing sibling `../strided-rs` repository;
the historical result tables retain the exact revisions actually measured.

```sh
cargo build -j 16 --locked --release --no-default-features \
  --features parallel,strided-kernel/simd --bin dense_kernels
RAYON_NUM_THREADS=1 taskset -c 16 target/release/dense_kernels
```

The default mode checks every result; it does not print elapsed times.
`BENCH_INSTANCE` selects one of `mul_2048`, `mul_odd`, `mul_strided`, `axpby_1m`,
`tril_1024`, `triu_rect`, `diag_rank2`, `copy_contiguous`, `copy_transpose`,
`copy_lm`, `copy_small`, `copy_negative`, or `copy_rank6`. For copy cases,
`COPY_MAP_BASELINE=1` selects the previous generic-map implementation in the
same binary. The default selects the uninitialized copy API.
`BENCH_RUNS` defaults to 3, after one
excluded warmup. Kernels always run under `ExecutionPolicy::Sequential`;
thread count is independent of affinity and Cargo build jobs. No BLAS is used.

## Instruction collection

Build each revision in a clean sibling worktree using the committed Cargo.lock,
and save separately named binaries. Run these sequentially, never during a build:

```sh
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  BENCH_INSTANCE=mul_2048 BENCH_RUNS=3 taskset -c 16 \
  valgrind --tool=callgrind --collect-atstart=no \
    --toggle-collect=profile_dense --callgrind-out-file=mul.callgrind \
    target/release/dense_kernels
```

`profile_dense` runs on the caller thread. Its windows exclude allocation,
input preparation, view construction, restoration, warmup, timer calls and
numerical verification. Read `summary:` and divide by BENCH_RUNS. Do not collect
an entire process and label it kernel execution. Raw profiles, environment,
checksums and the generated comparison are under
[`result/amd-cpu/dense-kernels`](../../../../result/amd-cpu/dense-kernels/).

## Native timing

Only on a quiet host, check the selected core AND its complete L3 domain before
running the same executable with `--time`. Choose a suitable CPU with taskset;
16 is the recorded instruction-run affinity, not a universal recommendation.
Use more samples, e.g. `BENCH_RUNS=21`. CSV output is case, threads, samples,
median nanoseconds. Restoration and correctness checks remain outside timing.
The current report contains instructions only, not wall-clock speedups or
bandwidth measurements. Other users' jobs must not be stopped for this test.
