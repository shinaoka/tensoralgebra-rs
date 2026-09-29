# Uninitialized copy: kernel instruction experiment

Final strided revision: **78d519013e8b44bd80f77eb01d31039f0be9aae6**.
Initial experiment: **20240fba2e67246594ae11abc0f88118c5dde8e8**.
Both modes in each stage use the same binary: `COPY_MAP_BASELINE=1` selects the
previous generic `map_into(..., MaybeUninit::new)`; 0 selects `copy_into_uninit`.
No BLAS routine is called. These are not native timing or bandwidth results.

| f64 case | map Ir/call | final copy Ir/call | reduction |
|---|---:|---:|---:|
| contiguous 1024×1025 | 722,523 | 723,246 | −0.100% |
| transpose 1024×1025 | 6,454,177 | 3,783,376 | 41.381% |
| LM permutation 12×12×1100 | 1,199,862 | 572,749 | 52.265% |
| small transpose 3×5 | 4,972 | 4,984.667 | −0.255% |
| reversed axis 33×65 | 28,812 | 11,963 | 58.479% |
| row-to-column-major rank 6 | 319,959 | 75,095 | 76.530% |

One process per mode/case, one excluded warmup and three collected calls;
these are instruction totals divided by three, not three independent process
pairs or statistical native-speedup estimates. Setup, allocation, view creation,
output checks and output destruction are outside the collection window. Internal
copy planning/validation and temporary view construction remain inside it.
Every run checks all nonzero output values against a separate coordinate oracle.

The initial version regressed contiguous copying by 28.04% in Ir. The final
version retains generic map for contiguous layouts, non-f32/f64 types, and
workloads selected for bounded parallel execution. Only sequential strided
native-float copies use the existing permutation engine. The exact-type guard
avoids its float reinterpretation for padded or more weakly aligned Copy types.
No new traversal, thread pool, or size threshold was added.

## Reproduction

Inside the authorized Linux Docker environment, build the harness with:

```sh
RUSTC_WRAPPER= CARGO_PROFILE_RELEASE_DEBUG=1 cargo build -j 16 \
  --release --features parallel --bin dense_kernels
```

Run sequentially for each table case and mode (0/1):

```sh
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
COPY_MAP_BASELINE=$mode BENCH_INSTANCE=$case BENCH_RUNS=3 taskset -c 16 \
valgrind --tool=callgrind --collect-atstart=no --toggle-collect=profile_dense \
  --callgrind-out-file=$file.callgrind /tmp/uninit-copy-kernels-final \
  > $file.log 2>&1
```

The harness explicitly installs `ExecutionPolicy::Sequential`; affinity is not
used as a substitute for a 1T execution policy. Rust1.98.1/LLVM22.1.8,
Valgrind3.22.0, AMD EPYC7713P, CPU16, L3 CPUs16–23. Shared-host Julia jobs remain
active; no native timing clearance was claimed. Stage binary hashes, raw
profiles/logs and build/test logs are retained here. `summarize.py` regenerates
`summary.json` from plain or gzip artifacts.

Final focused tests: five Docker release tests pass, including bounded-two-worker
policy inside a four-worker pool, holes/offsets/negative/broadcast layouts,
f32/f64 NaN-bit-preserving tiles, complex and padded-type fallbacks. All five
also pass Memcheck with zero errors (leak checking disabled). Earlier full
strided-basic verification passed 364 tests/doctests; final broader validation
and whole-eager LM/native acceptance remain separate requirements.
