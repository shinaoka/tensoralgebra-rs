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

## Current state (2026-08-03)

* **Phase 1 complete.** Design doc, repo, harness, baselines, premise check.
* **Phase 2 complete.** Engine is correct and framework-complete.
* **Phase 3 complete.** AVX-512 micro-kernels for `f32`/`f64` and all three
  complex methods, runtime-dispatched; register blocks chosen by measurement.
  Absolute performance is meaningful from here on. The three-way comparison is
  done — see below and the Phase 3 report. **AVX2 kernels now exist too** (one
  macro body per method across both ISAs, dispatch `avx512f` → `avx2+fma` →
  scalar), correct but with *provisional, unmeasured* register blocks — Phase 5's
  multi-arch item is no longer empty. The AVX-512 instruction stream is
  byte-identical to before, so all Phase 3/4 numbers stand.
* **Phase 4 in progress.** Item 1 (the write-back) is **done**: the profiled
  2x defect is closed in all four dtypes for +12–17% corpus geometric mean.
  **Item 1c (the micro-tile row block) is done** — a menu of register blocks
  per method plus a rule needing three measured guards; worth 1.026 corpus
  geomean in `c64` planar, and its more valuable output was pricing the
  orientation at ~2x. **Item 1d (the orientation rule) is done** — the
  discriminant A14 had recorded as unknown was found, worth +1.0–2.8% corpus
  geomean in every 32-bit column and up to 1.48x per case. **Item 2
  (`MC`/`KC`/`NC`) is built but unmeasured** — the grid is written and
  smoke-tested and needs ~7.5 h of an idle machine, and it now carries a
  **cache-model arm** (part 9) as well as the parameter arms. **Item 4's
  threading is implemented, correct and off by default**, now a 2-D `pm x pn`
  partition, also unmeasured; **K-parallelism is ruled out on evidence** (A21).
  See `DECISIONS.md` Phase 4 report parts 5–9.
* **Phase 5 in progress.** The multi-arch item (AVX2, above); **the C/C++
  consumption surface** — a shipped header
  (`crates/tensorprimitives-tapp/include/tapp.h`), a CMake/corrosion consumer
  under `examples/c-consumer` that CI compiles and runs in *three* link modes, a
  panic boundary at the ABI, and a vendored-offline build job (D36–D38, Phase 5
  interlude); and **the distribution surface** — a version symbol and header
  macros checked against each other, link-time SONAME/install-name, prefix install
  plus pkg-config, eleven cross-compile targets in CI, a BinaryBuilder recipe under
  `packaging/yggdrasil`, and a Julia package under `julia/TensorPrimitives` with a
  `TensorOperations.jl` backend. See D39, D40 and the Phase 5 report part 2. The
  engine was not touched by any of it.

  **Nothing is published and nothing is tagged**, on purpose. `RELEASING.md` is the
  order; `.github/workflows/release.yml` is the gate and has no credentials. Three
  things block an actual release and all three are human decisions: the repository
  is still **private** (Yggdrasil needs public source), the version is untagged,
  and the two Apple targets are unverified because they need the Xcode SDK licence
  accepted.

**Four things now exist that no benchmark has seen** — AVX2 kernels, the
analytical blocking model, 2-D threading, and the parallel-width analysis. All
are off or inert by default and green across ten switch combinations, so the
engine still measures as it did. **Two measurements are pending and both need an
exclusive machine**; `DECISIONS.md`'s "Resume here" says which script and what to
do with the output. Do not start either while anything else runs — a compile
counts.

**Those measurements are going to a Rusty node, and that is a design decision,
not a change of venue — read `DECISIONS.md` part 10 before running anything.**
Rusty has no Cascade Lake, so the partition picks the question; `rome` is chosen,
which means the **AVX2** kernels and their *unmeasured* register blocks (D26), so
**nothing measured there is comparable to any number in this repo** and every
ratio must be computed within the session, including a re-derived noise floor.
Part 10 also fixes, in advance, the accept/reject rule for running grid arms
concurrently one per L3 domain. Drive it with `scripts/node-session.sh <stage>`,
which stages the work in descending value and refuses to start a measurement
while anything of yours is compiling.

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
  `crates/tensorprimitives-tapp` and verified against the real headers from
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

**These margins are Phase 3 and are no longer current.** Phase 4.1 helped 3m
most — `c32` 3m improved 1.167 over Phase 3 against planar's 1.116 — so the
gaps have narrowed and the `c32` ordering may have flipped. The *mechanism*
below is unchanged and still holds. Re-measure before quoting any ranking
number; do not copy this table forward.

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
   adds 2–15% more, most of it in single precision. See the Phase 4 report.

   **1c, the micro-tile row block: done.** Each method now carries a *menu* of
   register blocks and `Plan::row_block` picks from it. The naive rule — take
   the shape that keeps the most output row blocks off the gather path — is a
   **loss** (0.936 in `f32`); it needs three measured guards (shallow `k`, a
   default that is substantially broken, and no change of orientation), after
   which it fires on 20 of 392 case-dtype-methods for 1.11–1.12x on those.
   Its more important result is a negative one: it does *not* dissolve the
   orientation misses as Phase 4.1 predicted, it prices them — those cases gain
   1.17–1.39x while paying ~30% in kernel shape, so the orientation is worth
   ~2x and `MR` is the wrong way to buy it.

   **1d, the orientation rule: done.** The missing discriminant (A14) was
   found by forcing both arms over the whole corpus and noticing that the
   corpus families are **mirror images**, so a correct rule must be
   *antisymmetric* under exchanging the two directions — which neither previous
   version, both phrased about the column direction alone, could be. Condition
   2 became a preference with a fallback: prefer the arm whose row block fits
   inside a run; failing that, put the shorter-run direction in the row role.
   12 better / 0 worse over both arms of all 392 case-dtype-methods, and `f32`
   is now within 0.4% of a hindsight oracle. Two hypotheses from part 4 were
   tested and refuted on the way (longer column run; maximise write-back
   regularity). 21 cases still take the slower arm — a different family, worth
   up to 1.36x, with both arms already measured in `bench-results/phase4d`.
2. **Sweep `MC`/`KC`/`NC`**, still the untouched Phase 2 heuristic. `KC` is now
   known to be first-order: it decides whether the `A` sliver is an L1 resident
   or an L2 stream, which is what the whole method ranking turns on. 1c added a
   second input: the *register block* is depth-conditional too (`c64` 3m wants
   a wider `MR` at `k <= 24`, worth 1.088), so vary the shape alongside `kc`
   rather than holding the Phase 3 table fixed.
3. **Dispatch the complex method by shape** — 3m on memory-bound shapes,
   planar otherwise. The inversion is measured and large enough to exploit.
4. **Threading — implemented, unmeasured.** BLIS-style, a 2-D `pm x pn`
   partition of the *output*: row strips of whole `MR` panels by column groups of
   whole `NR` blocks, per-thread packed `A`, one shared L3-sized packed `B`,
   `std::thread::scope`, static partitioning. `pn > 1` only when the row axis
   cannot fill the threads. Results are **bitwise identical to serial at every
   thread count and every partition** (no reduction is parallelised), which is the
   invariant the tests assert. Off by default (`TENSORCONTRACT_THREADS` or
   `Plan::with_threads` opts in) so every committed single-core number stays
   reproducible. **K-parallelism is ruled out on evidence, not skipped** — see
   A21; do not build per-thread accumulators without a shape that demands them.
   Remaining limits: threads are spawned per call rather than pooled, and `NC`'s
   L3 budget is charged per core in the legacy blocking (the model arm fixes it,
   A23). Measure with `scripts/phase4f-threads.sh` — it wants a whole socket.
   Then: small-`k` handling, prefetch, block-scatter
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
cargo build --release -p tensorprimitives-bench --features tblis,blas
```

`scripts/env.sh` documents how to build TBLIS 2.x. For TBLIS 1.3.0 use
autotools (`./configure --prefix=... --enable-shared && make && make install`)
and build the harness with `--features tblis13,blas`.

Both baselines currently live outside the repo at
`../baselines/tblis-{1.3.0,2.0}-install` (built 2026-08-02, same commits as
Phase 1). If they are gone, rebuild them — it is ~30 min unattended.

Three reproducible measurement entry points:

```bash
# The Phase 4 A/B pattern: A, B, A' with a repeat bracketing the treatment,
# plus sibling-CPU occupancy. Copy this shape for any new comparison. ~2 h.
scripts/phase4-remeasure.sh 4 bench-results/phase4

# Turn two sweep CSVs into the ratio tables used throughout DECISIONS.md.
# Needs no CPU; run it on committed data to check the tooling still agrees.
scripts/compare-sweeps.py BASE_CSVS NEW_CSVS

# The Phase 4.1c pattern, and the better one when the choice is discrete: sweep
# the *whole grid* of options once (~2 h), then score candidate rules against it
# offline, as many as you like, for free. This is how the row-block rule was
# derived and how the orientation rule should be.
scripts/phase4c-rowblock.sh  4 bench-results/phase4c   # every register block
scripts/phase4d-orient.sh    4 bench-results/phase4d   # both orientation arms
scripts/phase4e-blocking.sh  4 bench-results/phase4e   # the MC/KC/NC grid, ~7 h
scripts/phase4f-threads.sh auto bench-results/phase4f  # thread scaling, ~1 h,
                                                       # wants a whole socket
scripts/rowblock-score-rules.py bench-results/phase4c/shapes.csv bench-results/phase4c
scripts/orient-score-rules.py   bench-results/phase4d/features.csv bench-results/phase4d
scripts/blocking-score-rules.py bench-results/phase4e/features.csv bench-results/phase4e

# The two analyses: no CPU cost, no data touched, exactly reproducible. Safe to
# run while a benchmark is in flight, and the right way to decide which
# measurements are worth making.
./target/release/tcbench shapes --csv out.csv   # what each MR does to write-back
./target/release/tcbench orient --csv out.csv   # both arms' structural features
```

```bash
# The Phase 3 measurement set: verify, GEMM-roofline premise, full corpus
# sweeps, ragged stress, and the TBLIS 1.3.0 comparison. ~4 h, single core.
TBLIS_ROOT_2X=../baselines/tblis-2.0-install \
TBLIS_ROOT_13=../baselines/tblis-1.3.0-install \
  scripts/phase3-bench.sh 64 3

# Micro-kernel register-block sweep. Needs no baselines. ~8 min.
cargo run --release -p tensorcontract --example kernel_shapes
```

**A rule validated with the other levers pinned is validated only there.**
Phase 4.1c and 4.1d each pinned the other's lever to isolate their own — correct
experimental design, and exactly why neither could see that a 3% orientation
error was blocking a 20% shape change in the shipped combination. Always finish
with an end-to-end A/B in the configuration that actually ships (A20).

On a cluster node, submit the whole session unattended:

```bash
sbatch scripts/rusty-phase4.sbatch      # rome, exclusive, 1 node, 12 h
```

It runs `prep, shapes, threads, validate, grid`, decides the grid's placement with
`scripts/placement-verdict.py` (the rule pre-registered in `DECISIONS.md` part 10),
and falls back to a scoped sequential grid inside the remaining wall time if the
placement is rejected. Or drive the same stages by hand:

```bash
scripts/node-session.sh prep     bench-results/<node>-<arch>   # build, describe, predict
scripts/node-session.sh threads  bench-results/<node>-<arch>   # scaling + partition arms
scripts/node-session.sh shapes   bench-results/<node>-<arch>   # register-block calibration
scripts/node-session.sh validate bench-results/<node>-<arch>   # may arms run concurrently?
PLACEMENT=auto scripts/node-session.sh grid bench-results/<node>-<arch>
```

Stages run one at a time, only `prep` compiles, and `topology.py` /
`run-arms.py` record which cores were busy for every arm so exclusivity is
evidence rather than assumption. See `DECISIONS.md` part 10.

Benchmarks are single-core measurements: **do not compile, build a baseline or
run anything else on the machine while one is in flight.** Pinning is not
enough — the pinned core's **hyperthread sibling** shares L1d and L2, which is
what every cache-blocking measurement here turns on. Two Phase 4 conclusions
had to be corrected after re-measuring on an exclusive machine.

Use `scripts/phase4-remeasure.sh` as the pattern for any A/B: it runs
`A, B, A'` so a repeat brackets the treatment, prefers a runtime switch over a
rebuild so the arms interleave, and records sibling-CPU occupancy alongside the
results. **Noise floor when the machine is exclusive: ±1.3% on a 49-case
geometric mean, ±6% per case.** Anything smaller is not a result.

Useful environment variables:

| variable | effect |
|---|---|
| `TENSORCONTRACT_COMPLEX` | `planar` \| `1m` \| `3m` |
| `TENSORCONTRACT_KERNEL` | `scalar` \| `avx2` \| `avx512` \| `auto`: pin the instruction set. A pinned ISA the CPU lacks falls back to scalar, so `avx2` is how the AVX2 kernels get exercised on this AVX-512 machine |
| `TENSORCONTRACT_BLOCKMODEL` | `legacy` (default) \| `model`: the analytical cache model instead of the hardcoded constants |
| `TENSORCONTRACT_MC/_KC/_NC` | override cache blocking absolutely |
| `TENSORCONTRACT_MC_PCT/_NC_PCT` | scale the *derived* `mc`/`nc`, so each dtype and method keeps its budget share |
| `TENSORCONTRACT_KC_COUPLE` | set `kc` *and* re-derive `mc`/`nc` at that depth; the item 2 grid's second arm |
| `TENSORCONTRACT_PARTITION` | `m` \| `n` \| `<pm>x<pn>`: pin the thread partition instead of using `Plan::partition`'s rule |
| `TENSORCONTRACT_THREADS` | thread count, default **1**. Results are bitwise identical at any value, so this is never a correctness or accuracy decision. Also runs the whole test suite through the threaded driver, which is worth doing after any driver change |
| `TENSORCONTRACT_ORIENT` | `none` \| `swap`: pin the row/column orientation; `legacy`: the Phase 4.1 rule |
| `TENSORCONTRACT_WRITEBACK` | `gather` forces the general scatter write-back |
| `TENSORCONTRACT_ROWBLOCK` | `base` \| `auto` \| `mr=<n>` \| `idx=<i>`: pin the micro-tile row block |

The last three exist to make a change an A/B switch at run time rather than a
rebuild, so both arms can be measured interleaved in one session. Add one
whenever you introduce a fast path — a build-to-build diff already produced one
wrong sign in Phase 4. `idx=<i>` names a menu position rather than an `MR`,
because `mr=16` names different shapes in `f32` and `f64` while `idx=1` means
"the first alternate" in both; it is what makes a whole-grid sweep possible.

## Starting references

Matthews, "High-Performance Tensor Contraction without Transposition"
(arXiv:1607.00291); Van Zee, "1m method" (SIAM J. Sci. Comput. 2020) and Van
Zee & Smith "3m/4m methods" (ACM TOMS 2017); Springer & Bientinesi GETT
(arXiv:1607.00145) + the `HPAC/tccg` repo; "Strassen's Algorithm for Tensor
Contraction" (arXiv:1704.03092); TAPP (arXiv:2601.07827) +
`TAPPorg/reference-implementation`; BLIS papers.
