# CLAUDE.md

Persistent project context. Re-read this at the start of every session and
treat it as the source of truth for how to operate here.

**Read next, in order:** `DECISIONS.md` (top section is "Resume here"), then
`DESIGN.md`. `DECISIONS.md` carries the decision log, the standing
assumptions, and the phase reports with all measured data.

---

## What this project is

A native-Rust, transpose-free dense tensor contraction engine, plus a
systematic benchmark of complex tensor contraction. Core algorithm:
block-scatter-matrix tensor contraction (BSMTC, Matthews arXiv:1607.00291) —
treat a contraction as a GEMM over a scatter / block-scatter memory layout,
inside BLIS's five-loop, two-level-packing structure, so no explicit
transposition or temp workspace is needed.

Its distinguishing feature is **three interchangeable complex methods**
(planar, 1m, 3m) behind one switch, sharing every other line of the engine, so
they can be benchmarked against each other and against TBLIS on equal footing.

## Current state (2026-08-02)

* **Phase 1 complete.** Design doc, repo, harness, baselines, premise check.
* **Phase 2 complete.** Engine is correct and framework-complete.
* **Phase 3 complete.** AVX-512 micro-kernels for `f32`/`f64` and all three
  complex methods, runtime-dispatched; register blocks chosen by measurement.
  Absolute performance is meaningful from here on. The three-way comparison is
  done — see below and the Phase 3 report.
* **Phase 4 in progress.** Item 1 (the write-back) is **done**: the profiled
  2x defect is closed in all four dtypes for +11–12% corpus geometric mean,
  with no case left slower beyond noise. Items 1c, 2, 3 and the rest remain.
* Phase 5 not started.

Workspace MSRV is **1.89** (AVX-512 intrinsics stabilised there).

Everything builds warning-free, `cargo clippy --workspace --all-targets` is
clean, and `cargo test --workspace --release` is green.

## The thesis, and what happened to it

The project began from the hypothesis that TBLIS underperforms on complex
contractions — worst in memory-bound / awkward-stride cases — because
interleaved-complex storage and scatter packing compound and force work onto
the slow gather path. **Phase 1 measured this and the picture is now settled:**

* Against **TBLIS v1.3.0** (the latest *stable* release) the weakness is real
  and large: mean complex-over-real efficiency ratio **0.215**.
* Against **TBLIS 2.0-dev** it is gone: **1.06** (and 1.03 with irregular
  strides forced, 1.15 in single precision).
* The cause is **not** the proposed mechanism. TBLIS 1.x simply has **no
  complex micro-kernel** for any post-Sandy-Bridge x86 config; complex falls
  back to the generic templated kernel while real gets BLIS assembly. Its
  complex throughput is flat across shapes (4.1–9.1 GF/s) while real spans
  6.2–48.4. Block scatter is fully regular (`regA = 1.00`) throughout, so the
  gather path is not involved at all. TBLIS 2.0 fixed it by adopting BLIS as
  its core framework, which brings 1m.
* Complex is not intrinsically disadvantaged: it does 4x the flops on 2x the
  bytes, i.e. **twice the arithmetic intensity**, so overheads amortise better.

So: the planar-complex idea is not a research win over 1m, but the three-way
comparison is worth having and is now built. **Do not re-litigate this.** If
new data changes it, record that in `DECISIONS.md`.

**Where the measured headroom actually is:** low arithmetic intensity in
*either* domain. TBLIS 2.0's efficiency against a same-shape GEMM ceiling runs
0.85 on large compute-bound contractions down to **0.34** on small-`k` skinny
ones, and is worse in f32 than f64.

## Operating mode

Run **autonomously**. Do not wait for approval at routine phase boundaries; at
each boundary write a report to `DECISIONS.md` and continue. Prefer stating an
assumption and proceeding over asking. Escalate only on a kill/pivot condition
or a genuine blocker you cannot resolve after a documented attempt.

## Constraints

- Core is native Rust, **no FFI in the hot path**. FFI only for benchmark
  baselines.
- Prefer maintained crates over reinvention where they fit; justify every
  build-vs-reuse call in `DECISIONS.md`.
- **Primary integration surface: TAPP.** Implemented in
  `crates/tensorcontract-tapp` and verified against the real headers from
  `TAPPorg/reference-implementation`. Note: despite the TAPP paper's claim,
  TBLIS has **no in-tree TAPP support**, so the benchmark drives TBLIS through
  `tblis_tensor_mult` directly.
- RSTSR / REST integration is an optional secondary deliverable, not a design
  driver.
- Benchmark baselines: TBLIS (both v1.3.0 and 2.0-dev), TTGT via OpenBLAS, and
  a same-shape vendor GEMM as the roofline ceiling.
- Benchmark corpus: TCCG. **It is 49 cases, not 48** — see `DESIGN.md` §5.2.

## The three complex methods

Selected per plan with `Plan::with_complex_method`, or globally with
`TENSORCONTRACT_COMPLEX=planar|1m|3m`. All three share the index analysis,
scatter machinery, five-loop driver and write-back scatter; they differ only in
`PackFormat`, the micro-kernel, and `TileFormat`.

| method | A / B reals per complex elt | FMAs per k per tile | accumulator planes |
|---|---|---|---|
| `Planar` (default) | 2 / 2 | `4*MR*NR` | 2 |
| `OneM` | 4 / 2 | `4*MR*NR` | 2 (as `2*MR x NR` real) |
| `ThreeM` | 3 / 3 | `3*MR*NR` | 3 |

`Blocking::derive` takes the per-element real counts, so 1m automatically gets
a smaller `MC` and every method sees the same L2 budget. Getting that wrong
would silently rig the comparison.

**Measured ranking, with real AVX-512 kernels (Phase 3).** Full 49-case
corpus, single core, geometric-mean throughput relative to planar:

| method | c64 | c32 |
|---|---|---|
| planar | **1.000** | **1.000** |
| 1m | 0.967 | 0.979 |
| 3m | 0.956 | 0.921 |

**Planar wins — but not for the reason the project assumed, and not
everywhere.** Three things to carry forward and not re-derive:

1. The deciding quantity is **bytes moved per useful flop**, not flop count and
   not shuffles. Planar packs 2 reals per complex element in both operands; 1m
   packs 4 in `A`; 3m packs 3 in both while doing only 3/4 the products.
2. **3m's 25% flop saving is real** and shows up whenever the kernel is
   FMA-issue-bound — with L1-resident panels 3m is the *fastest* of the three.
   At the `kc` the engine uses, the `A` sliver is an L2 stream and the saving
   is consumed by the extra plane traffic.
3. **The ranking inverts by shape.** On memory-bound corpus cases
   (`min(n,k) <= 64`) 3m wins; on compute-bound ones planar wins by ~11% over
   3m. Exploiting this is a Phase 4 item.

Do not re-derive the register blocks — they are measured, recorded in
`kernel::x86`, and re-derivable with `examples/kernel_shapes` if the machine
changes.

## Phases

Each phase ends with a self-check against its gate, a report to
`DECISIONS.md`, and automatic progression.

**Phase 1 — Exploration & design.** *Complete.*

**Phase 2 — Correct, framework-complete implementation.** *Complete.*

**Phase 3 — Micro-kernels.** *Complete.* AVX-512 kernels for `f32`/`f64` and
all three complex methods in `kernel/x86.rs`, macro-generated over `(MV, NR)`
const generics, runtime-dispatched, scalar path retained. Nothing outside that
file changed — the `Ukr` contract carried them unmodified. Gate met on all
three counts. No AVX2 path yet; deferred to Phase 5's multi-arch work.

**Phase 4 — Profiling & improvement.** *In progress.*

1. **Fix the write-back.** ***Done.*** The 2x defect was *not* the write-back's
   own L2 traffic, as Phase 3 concluded — it was the row/column **orientation**
   of the matrix view. Computing `D^T = B^T A^T` when `D`'s column direction is
   the contiguous one recovers all of it; a regular-block write-back on top
   adds 6–13% more, but only in single precision. See the Phase 4 report.
   Still open under this item: make the micro-tile aspect ratio follow the
   output's stride pattern. Phase 3 sized it at ±11%; it is now also the way to
   move whole block families onto the write-back's unit-stride path.
2. **Sweep `MC`/`KC`/`NC`**, still the untouched Phase 2 heuristic. `KC` is now
   known to be first-order: it decides whether the `A` sliver is an L1 resident
   or an L2 stream, which is what the whole method ranking turns on.
3. **Dispatch the complex method by shape** — 3m on memory-bound shapes,
   planar otherwise. The inversion is measured and large enough to exploit.
4. Then: threading (BLIS-style, `std::thread::scope`, static partitioning with
   a shared packed-B panel), small-`k` handling, prefetch, block-scatter
   regularity exploitation, and the rest of the low-arithmetic-intensity work
   from Phase 1 (fusing the `pc` loop so `C` is touched once rather than
   `K/KC` times; a pack-free fast path when block scatter is already
   unit-stride).

Gate: performance targets met, or a clear evidence-based account of the gap.

**Phase 5 — Packaging.** API polish, docs, examples, feature flags, multi-arch
CI (including the AVX2 kernels Phase 3 deferred), crate publication, TAPP
conformance, reproducible benchmark artifact, paper-shaped writeup covering
both the negative result and the three-way comparison.

## Kill / pivot conditions (only reasons to escalate)

Escalate with data and a recommendation if: TAPP cannot express a required
operation; a gate is unmet and root-cause shows the shortfall is structural,
not incremental; a counted-on dependency is unmaintained with no substitute.
The original complex-weakness premise has already been resolved — that
question is closed.

## Standing rules

- Keep the test suite green at every stage; never trade correctness for speed.
- Maintain `DECISIONS.md`. Every material decision, assumption and phase report
  goes there.
- **Name the TBLIS version** in any statement about TBLIS performance. v1.3.0
  and 2.0-dev differ by ~5x on complex, and they swap the `TYPE_DOUBLE` /
  `TYPE_SCOMPLEX` ABI enumerators — a silent mismatch. Use the `tblis13` cargo
  feature; the harness self-checks at startup.
- Any claim about awkward strides must say which `--stress` mode produced it
  and quote the observed `regA`. The unperturbed TCCG corpus is fully regular.
- Hold shapes fixed across dtypes when comparing real and complex.
- Verify library/spec status via search, not training data; cite sources.
- Negative results are valid and publishable. Report honestly.
- Commit at meaningful milestones; keep the working tree clean between tasks.

## Environment and build recipes

Reference machine: `ccqlin038` (Flatiron CCQ), Xeon Gold 6244, Cascade Lake,
AVX-512, 2x8 cores, 1 MiB L2/core, 25 MiB L3. Full details in `DECISIONS.md`.

```bash
module load gcc/13.3.0 openblas          # cmake/3.31.6 for TBLIS 2.x
export TBLIS_ROOT=/path/to/tblis-install
source scripts/env.sh                    # pins everything single-threaded

cargo test --workspace --release
TENSORCONTRACT_KERNEL=scalar cargo test --workspace --release
cargo build --release -p tensorcontract-bench --features tblis,blas
```

`scripts/env.sh` documents how to build TBLIS 2.x. For TBLIS 1.3.0 use
autotools (`./configure --prefix=... --enable-shared && make && make install`)
and build the harness with `--features tblis13,blas`.

Both baselines currently live outside the repo at
`../baselines/tblis-{1.3.0,2.0}-install` (built 2026-08-02, same commits as
Phase 1). If they are gone, rebuild them — it is ~30 min unattended.

Two reproducible measurement entry points:

```bash
# The Phase 3 measurement set: verify, GEMM-roofline premise, full corpus
# sweeps, ragged stress, and the TBLIS 1.3.0 comparison. ~4 h, single core.
TBLIS_ROOT_2X=../baselines/tblis-2.0-install \
TBLIS_ROOT_13=../baselines/tblis-1.3.0-install \
  scripts/phase3-bench.sh 64 3

# Micro-kernel register-block sweep. Needs no baselines. ~8 min.
cargo run --release -p tensorcontract --example kernel_shapes
```

Benchmarks are single-core measurements: **do not compile, build a baseline or
run anything else on the machine while one is in flight.**

Useful environment variables:

| variable | effect |
|---|---|
| `TENSORCONTRACT_COMPLEX` | `planar` \| `1m` \| `3m` |
| `TENSORCONTRACT_KERNEL` | `scalar` forces the portable kernels |
| `TENSORCONTRACT_MC/_KC/_NC` | override cache blocking |
| `TENSORCONTRACT_ORIENT` | `none` \| `swap`: pin the row/column orientation |

## Starting references

Matthews, "High-Performance Tensor Contraction without Transposition"
(arXiv:1607.00291); Van Zee, "1m method" (SIAM J. Sci. Comput. 2020) and Van
Zee & Smith "3m/4m methods" (ACM TOMS 2017); Springer & Bientinesi GETT
(arXiv:1607.00145) + the `HPAC/tccg` repo; "Strassen's Algorithm for Tensor
Contraction" (arXiv:1704.03092); TAPP (arXiv:2601.07827) +
`TAPPorg/reference-implementation`; BLIS papers.
