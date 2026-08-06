# CLAUDE.md

How to operate in this repository. Re-read at the start of every session.

**This file is state and rules, not history.** Everything historical lives in
`docs/`. If you find yourself appending a paragraph here about something
that was measured, it belongs there instead.

## Read next

| when | read |
|---|---|
| **every session, in this order** | `docs/open-questions.md` (what is unmeasured and what would settle it), then `docs/README.md` |
| **before proposing any performance idea** | `docs/refuted.md`. Roughly half the obvious ideas here have already been measured and lost, and each entry says what would reopen it |
| before quoting any number | `docs/results.md`, and the `PROVENANCE.txt` next to its CSV in `bench-results/` |
| before designing a measurement | `docs/measurement-rules.md` |
| for the full account behind any `A<n>` or `D<n>` | `docs/decisions.md`, then the `docs/notebook/` chapter its last column names |
| the architecture | `docs/design.md` |
| before running anything on a cluster | `scripts/README.md` |


## What this project is

A native-Rust, transpose-free dense tensor contraction engine, plus a systematic
benchmark of complex tensor contraction. The algorithm is block-scatter-matrix
tensor contraction (BSMTC, Matthews arXiv:1607.00291): treat a contraction as a
GEMM over a scatter / block-scatter memory layout, inside BLIS's five-loop,
two-level-packing structure, so no explicit transposition and no temporary
workspace are needed.

Its distinguishing feature is **three interchangeable complex methods** (planar,
1m, 3m) behind one switch, sharing every other line of the engine, so they can be
compared with each other and with TBLIS on equal footing.

## Current state

**Deliberately not recorded here.** It used to be, and it rotted weekly: a phase
table, plus five bullets that duplicated the results, the open questions and the
CHANGELOG. `docs/open-questions.md` is the live version and is the first thing to
read each session.

The one thing worth repeating, because it governs what you may *say*: **anything
touching threads or caches is a per-microarchitecture claim until shown
otherwise** (A56). Four have now failed to transfer — register blocks, the thread
partition, the complex-method ranking, and the thread pool. Two machine classes
is the minimum for a recommendation, and this project has been wrong on one class
twice.


## Operating mode

Run **autonomously.** Do not wait for approval at routine phase boundaries; at
each boundary write a report to the right `docs/notebook/` chapter and continue. Prefer stating an
assumption and proceeding over asking. Escalate only on a kill/pivot condition or
a genuine blocker you cannot resolve after a documented attempt.

**Kill / pivot conditions — the only reasons to escalate.** TAPP cannot express a
required operation; a gate is unmet and root-cause shows the shortfall is
structural rather than incremental; a counted-on dependency is unmaintained with
no substitute. The original complex-weakness premise is already resolved and that
question is closed.

## Standing rules

* **Keep the test suite green at every stage.** Never trade correctness for speed.
* **Maintain `docs/`.** A report goes in the `docs/notebook/` chapter it belongs
  to; a decision or an assumption goes in `docs/decisions.md` and nowhere else;
  a number quoted anywhere goes in `docs/results.md` and is cited from there.
* **A refutation gets a `docs/refuted.md` entry in the same commit as the
  measurement**, with a confidence level, evidence, and what would reopen it.
* **Name the TBLIS version** in any statement about TBLIS performance. v1.3.0 and
  2.0-dev differ by ~5x on complex and swap the `TYPE_DOUBLE` / `TYPE_SCOMPLEX`
  ABI enumerators — a silent mismatch. Use the `tblis13` cargo feature; the
  harness self-checks at startup.
* **The corpus is *not* "fully regular".** That shorthand is wrong and steered
  conclusions for three phases. TCCG rounds stride-1 extents to multiples of 24,
  which is regular only at a register block that *divides* 24 — and the shipped
  `f32`/`c32` blocks are `MR` 16, 32 and 48, none of which does. On the arm the
  rule picks, `reg_a < 1.0` on **42.9%** of the 392 case-dtype-methods at
  `--size 64` — 45 of them at `reg_a = 0.0`, i.e. *entirely* on the gather path,
  and the rest at 0.667 / 0.889 / 0.963. Both qualifiers are part of the claim:
  it is 11.7% on an AVX2 node, where `MR = 8` for `f64` does divide 24, and 40.6%
  at `tcbench orient`'s default size, because the extents scale with it.
  Any awkward-stride claim must name its `--stress` mode and quote the observed
  `reg_a`. What the corpus cannot produce is *aperiodic* irregularity; that is
  what `--stress ragged` is for.
* **Hold shapes fixed across dtypes** when comparing real and complex.
* **Every noise floor names its session and its thread count.** There is no
  project-wide floor; see `docs/measurement-rules.md`. A per-case ratio
  at 64 threads is not readable at all (A39).
* **Every new fast path gets a runtime switch**, so both arms interleave in one
  process-restart A/B. A build-to-build diff has already produced a wrong sign
  here (A15).
* **Do not re-derive the register blocks.** They are measured and recorded in
  `kernel::x86`; `examples/kernel_shapes` re-derives them if the machine changes.
* Verify library and spec status by search, not from training data; cite sources.
* **Negative results are valid and publishable.** Report honestly.
* Commit at meaningful milestones; keep the working tree clean between tasks.
* **The CHANGELOG's confidence table and the README's "what is tuned" section are
  the two places a stale claim reaches a user rather than a maintainer.** Update
  them in the same commit as any future measurement — nothing in CI catches this.

## Constraints

* Core is native Rust, **no FFI in the hot path.** FFI only for benchmark
  baselines.
* Prefer maintained crates over reinvention where they fit; justify every
  build-vs-reuse call in `docs/decisions.md`.
* **Primary integration surface: TAPP**, in `crates/tensorprimitives-tapp`,
  verified against the real headers from `TAPPorg/reference-implementation`.
  Despite the TAPP paper's claim, TBLIS has **no in-tree TAPP support**, so the
  benchmark drives TBLIS through `tblis_tensor_mult` directly.
* RSTSR / REST integration is an optional secondary deliverable, not a design
  driver.
* Benchmark baselines: TBLIS (both v1.3.0 and 2.0-dev), TTGT via OpenBLAS, and a
  same-shape vendor GEMM as the roofline ceiling.
* Benchmark corpus: TCCG. **49 cases, not 48** — see `docs/design.md` §5.2.

## The three complex methods

Selected per plan with `Plan::with_complex_method`, or globally with
`TENSORCONTRACT_COMPLEX=planar|1m|3m`. All three share the index analysis,
scatter machinery, five-loop driver and write-back scatter; they differ only in
`PackFormat`, the micro-kernel and `TileFormat`.

| method | A / B reals per complex elt | FMAs per k per tile | accumulator planes |
|---|---|---|---|
| `Planar` (default) | 2 / 2 | `4*MR*NR` | 2 |
| `OneM` | 4 / 2 | `4*MR*NR` | 2 (as `2*MR x NR` real) |
| `ThreeM` | 3 / 3 | `3*MR*NR` | 3 |

`Blocking::derive` takes the per-element **real** counts, not
`size_of::<Element>()`, so 1m automatically gets a smaller `MC` and every method
sees the same L2 budget. Getting that wrong would silently rig the comparison.

**What is settled:** the deciding quantity is **bytes moved per useful flop**, not
flop count and not shuffles, and the accounting behind it — 3m does 3 products
where planar does 4, and moves 3 planes of both operands where planar moves 2. That
much is arithmetic and holds everywhere.

**What is not settled, and was wrongly recorded here as if it were:** that 3m's
flop saving *pays* in a nameable regime. This file used to say "with L1-resident
panels 3m is the fastest of the three". At `kc = 16`, which is that regime, 3m
leads planar by 1.105 (`c64`) and 1.155 (`c32`) on Cascade Lake — and trails at
0.567 and 0.662 on Ice Lake. **On Ice Lake 3m does not win at any depth, at any
shape, in either precision**, and it already runs its own Ice Lake-optimal shape,
so this is not a shape miss (A44, part 3). Treat 3m as the method that makes the
comparison honest, not as a candidate default.

**No ranking table is reproduced here on purpose.** Planar wins the corpus on
both AVX-512 machines measured; everything below that is machine-specific (A44)
and the AVX2 kernel-level ordering is different again (A24). Re-measure before
quoting any ranking number.

**On new hardware, run `examples/kernel_shapes` as a matter of course** — eight
minutes, no baselines. It is what would tell you whether Cascade Lake's 3m
behaviour is a family trait or one machine's, and this project has twice described
a question as unmeasured while its answer sat in committed output.

## Environment and build recipes

Reference machine: `ccqlin038` (Flatiron CCQ), Xeon Gold 6244, Cascade Lake,
AVX-512, 2x8 cores, 1 MiB L2/core, 25 MiB L3. **Shared during working hours** —
run long benchmarks off-hours and do the zero-CPU analyses during the day. Full
details, and the cluster nodes, in `docs/measurement-rules.md`.

```bash
module load gcc/13.3.0 openblas          # cmake/3.31.6 for TBLIS 2.x
export TBLIS_ROOT=/path/to/tblis-install
source scripts/env.sh                    # pins everything single-threaded

cargo test --workspace --release
TENSORCONTRACT_KERNEL=scalar cargo test --workspace --release
cargo build --release -p tensorprimitives-bench --features tblis,blas
```

Both TBLIS baselines live outside the repo at
`../baselines/tblis-{1.3.0,2.0}-install` (~30 min to rebuild unattended;
`scripts/env.sh` has the recipe). **TBLIS 2.0's install is ISA-specific and this
is a trap** — the `auto`-configured build is skx-only and SIGILLs without
AVX-512; `../baselines/tblis-2.0-x86_64-install` is the multi-config sibling a
non-AVX-512 node needs. Keep both, and see `docs/measurement-rules.md` for why.

`scripts/README.md` is the index of all 32 scripts, including which are
superseded. The entry points that matter:

```bash
# The engine against its baselines. Warm-up arm discarded, both TBLIS ABIs from
# prebuilt binaries so nothing compiles mid-run, two repeat arms that derive the
# session's own floor. ~3.5 h, exclusive machine.
TBLIS_ROOT_2X=../baselines/tblis-2.0-install \
TBLIS_ROOT_13=../baselines/tblis-1.3.0-install \
  scripts/compare-bench.sh prep                       # the only compile
  scripts/compare-bench.sh bench-results/$(hostname -s)-$(scripts/arch-label.sh)

# A/B one runtime switch, end to end: warm-up, A, B, A'. ~2 h.
scripts/ab.sh bench-results/ab-<name> "TENSORCONTRACT_<SWITCH>=<value>"

# When the choice is discrete, sweep the whole grid once and score candidate
# rules against it offline, as often as you like, for free.
scripts/phase4c-rowblock.sh 4 bench-results/phase4c    # every register block
scripts/phase4d-orient.sh   4 bench-results/phase4d    # both orientation arms
scripts/phase4f-threads.sh auto bench-results/phase4f  # thread scaling, ~1 h,
                                                       # wants a whole socket
scripts/rowblock-score-rules.py  bench-results/phase4c/shapes.csv bench-results/phase4c
scripts/orient-score-rules.py    bench-results/phase4d/features.csv bench-results/phase4d
scripts/blocking-score-rules.py  bench-results/ccqlin038-blocking/features.csv \
                                 bench-results/ccqlin038-blocking
scripts/partition-score-rule.py -p 64 -d 16 bench-results/worker5137-zen2

# No CPU cost, no data touched, exactly reproducible. Safe while a benchmark is
# in flight, and the right way to decide which measurements are worth making.
scripts/compare-sweeps.py BASE_CSVS NEW_CSVS    # the ratio tables in docs/notebook/
./target/release/tcbench shapes --csv out.csv   # what each MR does to write-back
./target/release/tcbench orient --csv out.csv   # both arms' structural features

# Micro-kernel register-block sweep. Needs no baselines. ~8 min.
cargo run --release -p tensorcontract --example kernel_shapes
```

On a cluster node, submit the whole session unattended:

```bash
sbatch --constraint=icelake scripts/rusty-compare.sbatch  # engine vs baselines
sbatch scripts/rusty-phase4.sbatch                        # rome, exclusive, 12 h
```

**Submitting is a human step — never automate it.** Only `squeue`, `sacct`,
`sinfo` and `scontrol show` may be executed here at all, scoped and one-shot.
Write out the `sbatch` line and ask.

`rusty-phase4.sbatch` runs `prep, shapes, threads, validate, grid`, decides the
grid's placement with `scripts/placement-verdict.py` (the rule pre-registered in
part 10), and falls back to a scoped sequential grid inside the remaining wall
time if the placement is rejected. `scripts/node-session.sh <stage> <outdir>`
drives the same stages by hand. Only `prep` compiles; `topology.py` and
`run-arms.py` record which cores were busy for every arm, so exclusivity is
evidence rather than assumption.

### Two rules that have each cost a measurement

* **Benchmarks are single-core measurements: do not compile, build a baseline, or
  run anything else on the machine while one is in flight.** Pinning is not
  enough — the pinned core's **hyperthread sibling** shares L1d and L2, which is
  what every cache-blocking measurement here turns on. Two Phase 4 conclusions
  had to be corrected after re-measuring on an exclusive machine.
* **A cluster job runs in the submit directory and uses its `target/`.** So
  "do not compile while a benchmark is in flight" applies to *this* machine even
  when the benchmark is on a compute node: `cargo build` — and `cargo test`,
  which relinks the same artefacts — replaces the very binary the running job
  invokes for each arm, and the job will not notice. `rusty-phase4.sbatch` and
  `rusty-compare.sbatch` set a per-job `CARGO_TARGET_DIR` for this reason;
  anything driven by hand still needs the submit directory left alone.

## Runtime switches

Every performance-relevant choice is reachable at run time, so an A/B is a
process restart rather than a rebuild. **None of them affects correctness**, and
results are bitwise identical across all of them.

| variable | effect |
|---|---|
| `TENSORCONTRACT_COMPLEX` | `planar` (default) \| `1m` \| `3m` |
| `TENSORCONTRACT_KERNEL` | `auto` (default) \| `scalar` \| `avx2` \| `avx512`: pin the instruction set. A pinned ISA the CPU lacks falls back to scalar, so `avx2` is how the AVX2 kernels get exercised on an AVX-512 machine |
| `TENSORCONTRACT_THREADS` | thread count, default **1**. Bitwise identical at any value, so never a correctness or accuracy decision. Also runs the whole test suite through the threaded driver — worth doing after any driver change |
| `TENSORCONTRACT_PARTITION` | `domain` (**default** since D44) \| `legacy`: which partition *rule*; or `m` \| `n` \| `<pm>x<pn>` to pin it outright. `domain` gates `Plan::partition`'s `panels >= p` early return on the L3 domain count — a measured no-op on 392 of 392 cases where one L3 serves the thread set, 1.133 corpus geomean at 64 Zen2 threads where sixteen do. `legacy` is the ungated rule, which is what every threaded number committed before 2026-08-04 was measured with |
| `TENSORCONTRACT_POOL` | `on`: reuse parked threads instead of spawning per `execute` call. **Measured on two machine classes and it does not transfer**: up to 11.6x on Zen2 (part 18), up to **2.5x slower** on Ice Lake (part 19). **Off, and D53's recommendation to default it on is withdrawn.** Suspected cause is reused buffer addresses contending in one shared L3 (A55) |
| `TENSORCONTRACT_L3_DOMAINS` | override how many L3 domains the thread set is taken to span. The derivation assumes compact placement (A37); this exercises the other case without a rebuild |
| `TENSORCONTRACT_BLOCKMODEL` | `legacy` (default) \| `model`: the analytical cache model instead of the hardcoded constants. `legacy` is the default **on evidence** — the model loses in 11 of 12 columns on a foreign machine (A33) |
| `TENSORCONTRACT_MC` / `_KC` / `_NC` | override cache blocking absolutely |
| `TENSORCONTRACT_MC_PCT` / `_NC_PCT` | scale the *derived* `mc`/`nc`, so each dtype and method keeps its budget share |
| `TENSORCONTRACT_KC_COUPLE` | set `kc` *and* re-derive `mc`/`nc` at that depth |
| `TENSORCONTRACT_ORIENT` | `none` \| `swap`: pin the row/column orientation; `legacy`: the Phase 4.1 rule |
| `TENSORCONTRACT_WRITEBACK` | `gather` forces the general scatter write-back |
| `TENSORCONTRACT_ROWBLOCK` | `base` \| `auto` \| `mr=<n>` \| `idx=<i>`: pin the micro-tile row block. **Use `idx=`** — the menu is keyed by *position* (D43), so two entries may share an `MR` and differ only in `NR`, and `mr=<n>` resolves to the first entry of that height. `idx=3` reaches the `32x5` shape A35 is about |

`Plan::with_complex_method`, `Plan::with_threads` and `Plan::with_blocking` are
the programmatic equivalents and take precedence.

## Starting references

Matthews, "High-Performance Tensor Contraction without Transposition"
(arXiv:1607.00291); Van Zee, "1m method" (SIAM J. Sci. Comput. 2020) and Van Zee
& Smith, "3m/4m methods" (ACM TOMS 2017); Springer & Bientinesi GETT
(arXiv:1607.00145) + the `HPAC/tccg` repo; "Strassen's Algorithm for Tensor
Contraction" (arXiv:1704.03092); TAPP (arXiv:2601.07827) +
`TAPPorg/reference-implementation`; BLIS papers.
