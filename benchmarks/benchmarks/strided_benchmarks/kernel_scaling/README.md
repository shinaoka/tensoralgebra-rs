# Kernel scaling

Coverage for [tensor4all/strided-rs#269](https://github.com/tensor4all/strided-rs/issues/269):
elementwise, ternary select and clamp, reduction, and structural copy kernels measured through the erased
entries, the typed entries, and a raw pointer baseline, at 1 and 4 threads,
next to Julia Base and Strided.jl. A gate script turns the four defect classes
of the issue into threshold checks:

| Defect in #269 | Cases that expose it | Gate |
|---|---|---|
| Erased elementwise entries dispatch the op per element | `ew_*` erased and erased_uninit versus typed and raw | (b), (c) |
| Parallel reduce paths use a scalar single accumulator leaf | `red_*_all_*`, `red_*_axis0_*` versus raw 8 lane accumulators | (a), (c) |
| Contiguous axis reductions with a strided output take the scalar general path | `red_*_axis1_*` (and axis0 with its compact output) | (b), (c), (d) |
| CopyPlan structural ops have no parallel branch | `copy_*` at 4T versus 1T | (a) |
| Erased select validated the bool predicate with a byte by byte `find` (2/3 of its time) and erased clamp tested NaN at each of the maximum and minimum steps; both only visible past the LLC ([strided-rs#277](https://github.com/tensor4all/strided-rs/pull/277)) | `ter_select_*`, `ter_clamp_*` erased_uninit versus typed and raw | (b), (c) |

## Files

- `kernel_scaling.rs`: Rust binary (`[[bin]] kernel_scaling`, feature `parallel`).
- `kernel_scaling.jl`: Julia counterpart with the same case names and shapes.
- `run.sh`: builds against any strided-rs tree and writes the CSVs.
- `gate.py`: reads the CSVs, prints a table, and exits nonzero on flags (Python stdlib only).

## Building against a strided-rs tree

The suite `Cargo.toml` points at the sibling `../strided-rs`. To measure any
other checkout without editing tracked files, `run.sh` writes a standalone
manifest under `target/kernel-scaling/<rev>/` (gitignored) whose only path
dependency is `$STRIDED_RS_DIR/strided-kernel` with features `parallel,simd`.
The strided-rs revision is taken from `git rev-parse` in that tree, suffixed
with `-dirty` when it has uncommitted tracked changes, compiled into the binary,
printed in its header, and used in the output directory name
`data/results/kernel-scaling-<rev>/`. `RUSTFLAGS` defaults to
`-C target-cpu=native`.

```sh
page=benchmarks/strided_benchmarks/kernel_scaling
export STRIDED_RS_DIR=/path/to/strided-rs

$page/run.sh build            # compile only
$page/run.sh check 4          # correctness of every case at 4T, no timing
$page/run.sh rust 1           # data/results/kernel-scaling-<rev>/rust_1t.csv
$page/run.sh rust 4           # data/results/kernel-scaling-<rev>/rust_4t.csv
$page/run.sh julia 1          # data/results/kernel-scaling-<rev>/julia_1t.csv
$page/run.sh julia 4          # data/results/kernel-scaling-<rev>/julia_4t.csv
python3 $page/gate.py data/results/kernel-scaling-<rev>
```

Run the four timing commands one after another, never concurrently, on a quiet
host. On Linux the runs are pinned with `taskset -c ${CPUS:-0-(T-1)}`. On macOS
`taskset` does not exist and the script says that pinning was not applied.
`OUTPUT_DIR` overrides the result directory. Julia uses `julia --project=.`
at the suite root (Strided 2.6.1).

With the sibling checkout the binary can also be built in tree:
`cargo build --release --features parallel --bin kernel_scaling`, then
`target/release/kernel_scaling --threads 4 [--time]`.

### Binary options

| Setting | Default | Meaning |
|---|---|---|
| `--threads N` | required | thread budget, enforced (see below) |
| `--time` | off | without it every case runs once and is checked; with it the CSV is printed |
| `BENCH_RUNS` | 11 | timed samples per row; the median is reported |
| `BENCH_WARMUP` | 2 | untimed warmup calls per row |
| `KERNEL_SCALING_SHRINK` | 1 | divides every 2D extent by S and every 1D length by S squared (smoke runs, for example 16) |
| `BENCH_FILTER` | empty | only cases whose name contains this substring |

The Julia script reads the same environment variables and `--time`.

### Thread enforcement

The Rust binary builds the global Rayon pool with exactly N threads and runs
every case inside `rayon::scope`, so all work, including the raw baselines,
runs on that pool. Strided entries receive `ExecContext::serial()` at 1T and
`ExecContext::max_threads(N)` otherwise; typed entries run inside
`ExecContext::run`. Before any case the binary asserts that
`rayon::current_num_threads() == N`, that the body runs on a pool worker, that a
4M element typed `map_into` probe touched at most N distinct workers and none
outside the pool, and that the process has at most N + 1 OS threads
(`/proc/self/status` on Linux, `ps -M` on macOS). The header line on stderr
prints all of these, for example:

```
# threads: requested=4 rayon_pool=4 exec_ctx=ExecContext { kind: MaxThreads(4) } probe_workers=4 os_threads=5
```

The Julia script asserts `Threads.nthreads() == JULIA_NUM_THREADS` and sets
`Strided.set_num_threads(N)`.

## Protocol

Inputs, outputs, erased descriptors, compiled plans, and views are created
before timing. Each row is `BENCH_WARMUP` untimed calls followed by
`BENCH_RUNS` timed calls; the median is reported. Inputs and outputs pass
through `black_box`. After each variant is timed its output is compared on
every element with a plain sequential reference (exact for elementwise f64,
copies and max/min, relative 1e-12 for complex elementwise, relative 1e-9 for
sum and prod). Check mode runs the same comparisons with a single call.
Exception: typed `reduce_axis` returns a freshly allocated `StridedArray`, so
that allocation is inside its timed region. It is the only typed per axis API.

CSV columns: `case,variant,threads,median_ns,samples`.

## Cases

Sizes are nominal (`KERNEL_SCALING_SHRINK=1`). All arrays are column major f64
unless stated.

| Family | Case | Shape |
|---|---|---|
| Elementwise | `ew_{add,sub,mul,div,max,min,neg,abs}_f64_contig` | 1D, 33554432 elements |
| Elementwise | `ew_{add,sub,mul,div,max,min,neg,abs}_f64_trans` | 8192 x 4096; lhs is row major (strides `[4096, 1]`), rhs and destination column major |
| Elementwise | `ew_{add,mul,div,neg,conj,abs}_c64_contig` | 1D, 33554432 Complex64 (abs writes f64) |
| Elementwise | `ew_conj_c64_trans` | 8192 x 4096 Complex64, transposed source |
| Ternary | `ter_{select,clamp}_f64_contig` | 1D, 33554432 elements |
| Ternary | `ter_{select,clamp}_f64_trans` | 8192 x 4096; the first operand (predicate or x) is row major, the other operands and the destination column major |
| Reduction | `red_{sum,prod,max,min}_{all,axis0,axis1}_{8192x4096,2048x2048}` | column major source; axis0 reduces the contiguous axis, axis1 the strided axis |
| Structural | `copy_slice_step2` | 8192 x 4096 to 4096 x 4096, step `[2, 1]` |
| Structural | `copy_reverse_axis0`, `copy_reverse_axis1` | 4096 x 4096 |
| Structural | `copy_concat_axis0`, `copy_concat_axis1` | two halves into 4096 x 4096 |
| Structural | `copy_dynslice_rank1` | length 33554432 operand, 16777216 window, start 5592405 |
| Structural | `copy_dynslice_rank2` | 4608 x 4608 operand, 4096 x 4096 window, starts `[100, 200]` (i64) |
| Structural | `copy_pad` | 4000 x 4000 padded by 48 on every side to 4096 x 4096, fill 0 |

Binary max and min are NaN propagating, matching `ErasedZipOp::Maximum` and
`Minimum`; the typed and raw closures use the same function.

Select reads a bool predicate with the irregular pattern `(i*7919)%13 < 6`
(0 based `i`), picking the lhs generator where true and the rhs generator
otherwise. Clamp reads x from the lhs generator with a NaN at every 1021st
element and full arrays `lo = -0.25`, `hi = 0.25` (the lhs generator hits both
bounds exactly, so ties are exercised). The typed and raw closures use the
strided semantics: `raised = lo >= x ? lo : x`, `lowered = hi <= raised ? hi :
raised`, and NaN when any operand is NaN. Julia `clamp` agrees on these inputs
because only x carries NaN.

## Variants

| Variant | Rust entry | Julia API |
|---|---|---|
| `raw` | slice loops the compiler vectorizes (three input slices for select and clamp), split into contiguous chunks over the same Rayon pool; 64 x 64 tiles for transposed reads; 8 lane accumulators (with a NaN flag for max/min) for reductions; `copy_nonoverlapping` or per column loops for copies | |
| `typed` | `zip_map2_into`, `map_into`, `zip_map3_into` (select, clamp), `reduce`, `reduce_axis`, `SlicePlan`, `ReversePlan`, `ConcatenatePlan`, `DynamicSlicePlan`, `PadPlan` inside `ExecContext::run` | |
| `erased` | `erased_zip_into`, `erased_map_into` (select and clamp have no initialized erased entry), `ErasedReducePlan::compile` or `compile_axes`, `Erased{Slice,Reverse,Concatenate,DynamicSlice,Pad}Plan::execute` | |
| `erased_uninit` | `erased_zip_into_uninit`, `erased_map_into_uninit`, `erased_select_into_uninit`, `erased_clamp_into_uninit` (elementwise and ternary only) | |
| `julia_base` | | elementwise: `out .= f.(a, b)` into a preallocated array; ternary: `out .= ifelse.(p, a, b)`, `out .= clamp.(x, lo, hi)`; reductions: `sum(A)` etc. for all, `sum!(out, A)` etc. for dims |
| `julia_strided` | | elementwise, ternary, and copies: `@strided so .= ...` on `StridedView`s; reductions: `sum(StridedView(A))` for all, `sum!(StridedView(out), StridedView(A))` for dims |
| `julia_alloc` | | allocating Base call: `sum(A; dims=d)`, `A[1:2:end, :]`, `reverse(A; dims=d)`, `vcat` or `hcat`, `A[s+1:s+W, ...]` |
| `julia_copyto` | | copies: `copyto!(out, view(A, ...))`; concat: two `copyto!` into views of `out`; pad: zero the four border strips, then `copyto!` the interior |

The Julia reductions use `sum`, `prod`, `maximum`, `minimum` and their
in place forms. Julia has no pad or dynamic slice primitive, so those rows use
the preallocated copies listed above.

## Gate

```sh
python3 benchmarks/strided_benchmarks/kernel_scaling/gate.py DIR_OR_CSVS... [--report-only]
```

Strided variants are `typed`, `erased`, and `erased_uninit`. Flags:

| Flag | Condition | Default |
|---|---|---|
| (a) | a strided variant is not at least 1.3x faster at 4T than at 1T, for cases whose 1T median is at least 1 ms (tensor sized); the raw scaling of the same case is printed for context | `--scaling 1.3 --threads 4 --min-scaling-ns 1e6` |
| (b) | an erased variant is more than 1.25x slower than typed at the same thread count | `--erased 1.25` |
| (c) | a strided variant is more than 1.5x slower than `raw` at the same thread count | `--raw 1.5` |
| (d) | a strided variant is more than 2x slower than the fastest Julia variant at the same thread count | `--julia 2.0` |

`--min-ns` skips (b), (c) and (d) when the strided side is below that many ns
(default 0). `--filter` restricts to matching cases. The script exits 1 when
anything is flagged unless `--report-only` is given.

## Not covered

- Complex elementwise is limited to add, mul, div, neg, conj, abs on the
  contiguous layout plus conj on the transposed layout. Complex sub is
  omitted, and erased max and min do not accept complex inputs.
- Uninitialized destination entries of the structural plans are not measured.
- Typed reductions have no preallocated per axis entry, so `typed` axis rows
  include the output allocation.
- Only f64 and Complex64 are exercised; f32 and integer dtypes are not.
- Select and clamp are f64 only, and clamp bounds are full arrays rather
  than stride 0 broadcasts. The results below predate the ternary cases;
  their timings are pending.

## Results

Host: Apple M5 Max, macOS (Darwin 25.5.0). CPU pinning is unavailable on
macOS, so no run was pinned. `RUSTFLAGS=-C target-cpu=native`,
`BENCH_RUNS=11`, runs executed sequentially on an otherwise idle host.

- Before: strided-rs `c1c14a6` (v0.4.2 release commit).
- After: strided-rs `2b66201` (main with #271, #272, #273).
- Julia: Strided 2.6.1, best of the Julia variants; Julia does not depend on
  strided-rs, so the same Julia CSVs serve both columns.

The gate (`gate.py --report-only`) reports 341 flags at `c1c14a6` and 31
at `2b66201`. The remaining flags are typed `reduce_axis` along axis 1 at 1T
(1.5x to 2.1x of raw), `copy_pad` at 4T (1.7x of raw), and transposed
elementwise cases at 4T, where medians vary between runs (for example erased
`ew_add_f64_trans` at 4T measured 30.2 ms against 20.7 ms before, with raw at
20.3 ms). They are tracked in tensor4all/strided-rs#274.

1 thread (median ms):

| case | typed before | typed after | erased before | erased after | raw | Julia |
|---|---|---|---|---|---|---|
| `ew_add_f64_contig` | 6.40 | 6.86 | 21.17 | 6.88 | 6.67 | 6.47 |
| `ew_add_f64_trans` | 57.87 | 54.36 | 61.06 | 55.62 | 61.95 | 46.28 |
| `ew_max_f64_contig` | 6.95 | 6.56 | 24.28 | 6.73 | 6.53 | 6.47 |
| `red_sum_all_8192x4096` | 17.30 | 2.96 | 3.21 | 3.09 | 3.05 | 2.99 |
| `red_max_all_8192x4096` | 31.27 | 5.23 | 8.50 | 4.10 | 4.45 | 2.91 |
| `red_sum_axis0_8192x4096` | 19.10 | 2.95 | 32.69 | 3.08 | 3.06 | 2.99 |
| `red_sum_axis1_8192x4096` | 9.23 | 5.38 | 133.42 | 3.20 | 4.48 | 4.46 |
| `red_max_axis0_8192x4096` | 46.77 | 5.00 | 46.88 | 4.13 | 4.46 | 2.91 |
| `red_max_axis1_8192x4096` | 15.75 | 9.06 | 138.29 | 4.40 | 4.31 | 4.46 |
| `copy_concat_axis0` | 2.16 | 2.12 | 2.17 | 2.16 | 1.76 | 3.90 |
| `copy_reverse_axis0` | 4.79 | 3.24 | 4.77 | 3.22 | 3.30 | 4.57 |
| `copy_slice_step2` | 4.20 | 4.03 | 5.86 | 4.00 | 3.44 | 3.98 |
| `copy_pad` | 2.80 | 2.77 | 2.80 | 2.78 | 2.12 | 6.04 |
| `copy_dynslice_rank2` | 26.49 | 2.22 | 28.35 | 2.23 | 2.04 | 3.87 |

4 threads (median ms):

| case | typed before | typed after | erased before | erased after | raw | Julia |
|---|---|---|---|---|---|---|
| `ew_add_f64_contig` | 2.62 | 2.60 | 5.90 | 2.49 | 2.56 | 3.05 |
| `ew_add_f64_trans` | 22.02 | 21.51 | 20.68 | 30.19 | 20.30 | 13.55 |
| `ew_max_f64_contig` | 2.50 | 2.52 | 6.68 | 2.49 | 2.47 | 3.43 |
| `red_sum_all_8192x4096` | 5.20 | 0.94 | 5.14 | 0.96 | 1.10 | 2.63 |
| `red_max_all_8192x4096` | 7.96 | 1.65 | 11.78 | 1.34 | 1.51 | 2.04 |
| `red_sum_axis0_8192x4096` | 18.89 | 1.13 | 32.91 | 0.99 | 1.10 | 2.45 |
| `red_sum_axis1_8192x4096` | 10.83 | 2.17 | 126.12 | 1.72 | 2.64 | 2.84 |
| `red_max_axis0_8192x4096` | 45.49 | 1.73 | 46.55 | 1.36 | 1.51 | 2.05 |
| `red_max_axis1_8192x4096` | 15.71 | 2.68 | 136.12 | 1.82 | 2.84 | 2.92 |
| `copy_concat_axis0` | 2.19 | 0.81 | 2.17 | 0.81 | 0.77 | 1.32 |
| `copy_reverse_axis0` | 5.07 | 1.13 | 4.73 | 1.11 | 1.14 | 1.60 |
| `copy_slice_step2` | 5.41 | 1.34 | 4.53 | 1.32 | 1.31 | 1.63 |
| `copy_pad` | 2.85 | 1.29 | 2.89 | 1.30 | 0.77 | 2.09 |
| `copy_dynslice_rank2` | 10.12 | 0.85 | 10.52 | 0.84 | 0.80 | 1.30 |

### Ternary cases (2026-09-23)

Same host and settings, `BENCH_FILTER=ter_`. Before: strided-rs `b93cb6b`
(main after v0.4.3). After: strided-rs `2570533` (v0.4.4 release commit, with
tensor4all/strided-rs#277). Julia is the better of `julia_base` and
`julia_strided`.

| case | T | typed before | typed after | erased_uninit before | erased_uninit after | raw | Julia |
|---|---|---|---|---|---|---|---|
| `ter_select_f64_contig` | 1 | 7.82 | 7.80 | 15.68 | 8.21 | 7.83 | 7.81 |
| `ter_select_f64_contig` | 4 | 3.40 | 3.37 | 11.55 | 3.78 | 3.30 | 3.63 |
| `ter_clamp_f64_contig` | 1 | 11.06 | 10.96 | 20.63 | 10.98 | 10.98 | 8.76 |
| `ter_clamp_f64_contig` | 4 | 3.75 | 3.71 | 5.37 | 3.72 | 3.74 | 4.50 |
| `ter_select_f64_trans` | 1 | 32.16 | 30.55 | 39.82 | 30.80 | 40.71 | 36.67 |
| `ter_select_f64_trans` | 4 | 14.51 | 13.16 | 24.44 | 25.05 | 19.59 | 12.00 |
| `ter_clamp_f64_trans` | 1 | 62.68 | 63.37 | 58.85 | 61.58 | 76.00 | 54.92 |
| `ter_clamp_f64_trans` | 4 | 20.00 | 18.93 | 19.99 | 20.00 | 22.78 | 15.59 |

The gate flags only `ter_select_f64_trans` at 4T in the after run (erased
1.90x of typed). That median does not reproduce: three reruns of the case alone
gave erased/typed 1.22, 1.09, and 1.11, a rerun of all `ter_` cases 1.05, and
80 samples 1.00 (15.27 ms against 15.30 ms). Transposed cases at 4T vary
between runs on this unpinned host, as noted above for `ew_add_f64_trans`.
