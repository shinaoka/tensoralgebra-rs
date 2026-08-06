## The baseline comparison

The engine against TBLIS 2.0-dev, TBLIS v1.3.0 and TTGT, re-measured on the tightest floor in this file — and the finding that the method ranking does not travel.

### Phase 5 part 3: the comparison re-measured, and the method ranking does not travel

Job **6753260**, `worker6156` (Ice Lake-SP, AVX-512, 2 x 32 cores, SMT off),
2026-08-04, 231 min, `--constraint=icelake --exclusive`,
`scripts/rusty-compare.sbatch` → `scripts/compare-bench.sh`. Raw data in
`bench-results/worker6156-icelake/compare-tblis-skx/`.

**Why this run happened at all.** The newest engine-vs-baseline data in the repo
was `bench-results/phase3-sweep-*` from 2026-08-02, and every Phase 4 gain landed
after it, so the README's comparison understated the engine. This is the
re-measurement. It is **not** on the reference machine — that was a deliberate
choice, taken because `ccqlin038` is shared and the last A/B run there had 6–14
co-tenants per arm — and the cost of that choice is that **nothing here can be
differenced against the Phase 3 table.** Different microarchitecture, different
cache hierarchy. The improvement attributable to Phase 4 remains unmeasured; only
a `ccqlin038` run can supply it.

#### The measurement is the tightest in this file

| floor | arms | geomean spread | outside ±6% |
|---|---|---|---|
| near (`A2` vs `A`) | minutes apart | 0.995–1.003 | 1 of 980 |
| **session-span (`A3` vs `A`)** | **2.5 h apart** | **0.998–1.001** | **0 of 980** |

Identical arms repeated to within 0.4 s on 1424 s; every arm reports `self 100%,
co-tenants: none`. That is roughly **5x tighter than `ccqlin038`'s published
±1.3% geomean / ±6% per case**, which is the argument for A32 — derive the floor
in session — stated as a number.

**A32 is refined by this run, and in the direction of "it is not universal".**
The session-span floor is *no worse* than the near floor, so drift does not grow
with the distance between arms here. On Zen2 it did (0.02% at a minute, 1–2% at
an hour, 4.4% across a cold start). The 35-minute discarded warm-up arm bought
confirmation rather than a correction. A31 and A32 look like properties of boost
headroom and co-tenancy, not of measurement in general — keep the warm-up arm
(it is cheap and it is how you learn which case you are in), but do not expect it
to move anything on an exclusive SMT-off node.

#### The engine, this build, this machine

49-case corpus at 64 MiB, single core, planar, GF/s. **These columns are
independent of the TBLIS question below**, which cannot touch them.

| dtype | min | median | geomean | max |
|---|---|---|---|---|
| `f32` | 11.0 | 74.8 | 77.2 | 168.8 |
| `f64` | 7.1 | 44.7 | 43.7 | 71.6 |
| `c32` | 25.5 | 121.0 | 113.9 | 179.3 |
| `c64` | 15.7 | 62.0 | 63.3 | 91.1 |

Correctness verified on the node in all three methods before any timing.
Against TTGT (OpenBLAS, unaffected by the TBLIS question): **2.15x `f64`, 2.12x
`c64`, 2.20x `f32`, 1.95x `c32`** on corpus geometric mean.

**Complex beats real again, on a second microarchitecture.** `c64/f64` = 1.449,
`c32/f32` = 1.474, against Cascade Lake's 1.416 and 1.432. This is a within-run
ratio, so it is valid where the absolute numbers are not comparable, and it is
the project's central technical result replicating on new hardware to within
about two points. Twice the arithmetic intensity amortises overhead better; the
weak spot is low arithmetic intensity in either domain, not complex.

#### The method ranking does not transfer, and item 3's premise fails here

This is the result worth carrying forward, and it contradicts something the
Resume-here block lists as settled.

| relative to planar | Cascade Lake (Phase 3) | **Ice Lake (here)** |
|---|---|---|
| `c64` 1m | 0.967 | 0.868 |
| `c64` 3m | 0.956 | **0.694** |
| `c32` 1m | 0.979 | 1.004 |
| `c32` 3m | 0.921 | **0.744** |

And the inversion that Phase 4 item 3 was going to exploit is **absent**:

| subset | `c64` | `c32` |
|---|---|---|
| memory-bound (`min(n,k) <= 64`, n=24) | planar 49.3 > 1m 44.7 > 3m 35.9 | 1m 87.0 > planar 81.4 > 3m 67.3 |
| compute-bound (n=25) | planar 80.5 > 1m 67.0 > 3m 53.4 | planar 157.2 > 1m 148.6 > 3m 105.6 |

**3m is last in every column and wins 0 of 49 cases**, its per-case ratio against
planar running 0.63–0.84 — uniform, not a few catastrophic shapes. On Cascade
Lake 3m was the *fastest* of the three on the memory-bound subset. So:

* **Item 3 ("dispatch 3m on memory-bound shapes") is a pessimisation on Ice
  Lake** and must not be built as an unconditional rule. If it ships at all it has
  to be conditioned on the microarchitecture, which is the same conclusion A34
  reached for register blocks and A36 for the thread partition. That is three
  independent findings pointing one way.
* **This looked confounded with A34, and it is not.** See the next subsection: the
  deconfounding data was already committed and the confound does not exist.
* Practical consequence for quoting: **the ranking, and the inversion, are Cascade
  Lake results.** The README must say so rather than stating them as properties of
  the engine.

#### The A34 confound does not exist, and the data was already on disk

*Added 2026-08-05, from committed data, at no machine cost.* The subsection above
recorded A44 as confounded with A34 and named "extend the Ice Lake register-block
sweep to 3m" as the experiment that would separate them. **That experiment had
already run.** `bench-results/worker6016-icelake/kernel-shapes.txt` (job 6746817)
covers all four methods — 35 lines of 3m — at three depths in both AVX-512 dtype
pairs. This is a second instance of A35's lesson: the answer sat in committed raw
output for two days while the file above described it as unmeasured.

**The confound requires 3m to be running a shape that is wrong for Ice Lake. It is
not.** 3m ships `(1, 10)` in both precisions — `8x10` in `c64`, `16x10` in `c32` —
and the Ice Lake sweep names `MV=1 NR=10` as 3m's *best* shape in both. Shipped and
optimal coincide, on both machines, in both precisions. A34's shape error is real
and it lands on `real` (Ice Lake wants `MV=3 NR=9`) and on planar `f32` (A35's
`32x5`); it does not touch 3m at all, because there is no better shape to give it.

So the comparison can be made at each method's shipped shape, which is what the
corpus actually runs, at the operating `kc` for each dtype:

| | Cascade Lake | **Ice Lake** |
|---|---|---|
| `c64`, `kc = 256`: planar `16x6` / 3m `8x10` | 102.8 / 87.8 → **0.854** | 108.4 / 63.8 → **0.589** |
| `c32`, `kc = 384`: planar `32x6` / 3m `16x10` | 195.7 / 194.9 → **0.996** | 221.8 / 134.5 → **0.606** |

Those kernel-level ratios bracket the corpus-level 0.694 / 0.744 that A44 measured,
which is the consistency check. And the collapse is **uniform across 3m's whole
shape space** rather than a shape choice: every 3m shape is slower on Ice Lake at
every depth (`8x10` 87.8 → 63.8, `16x4` 82.5 → 59.5, `24x3` 76.1 → 56.2) while
planar `16x6` gets *faster* (102.8 → 108.4). Nor is it the register budget: 3m's
`16x10` is flagged `live = 32!` on Ice Lake, but the best *unflagged* 3m shape there
is slower still (`32x4` at 125.8 against `16x10`'s 134.5), so avoiding the spill
does not rescue it either.

**The bigger casualty is the mechanism, not the rule.** This project has carried
"3m's 25% flop saving is real, and with L1-resident panels 3m is the *fastest* of
the three" as settled mechanism rather than as a machine-specific result. At
`kc = 16`, which is that regime, at shipped shapes:

| | Cascade Lake | **Ice Lake** |
|---|---|---|
| `c64` 3m / planar | **1.105** | **0.567** |
| `c32` 3m / planar | **1.155** | **0.662** |

3m leads by 10–16% on Cascade Lake at L1-resident depth and trails by 34–43% on Ice
Lake at the same depth. **On Ice Lake 3m does not win at any depth, at any shape, in
either precision.** So the L1-resident advantage is a Cascade Lake property too, and
the sentence above must be qualified wherever it appears — it was carried into
`CLAUDE.md` unchecked during the 2026-08-05 clean-up, which is the same failure mode
as "the corpus is fully regular".

What survives is the *accounting*: 3m really does 3 products where planar does 4,
and it really moves 3 planes of both operands where planar moves 2. What does not
survive is the claim that the saving pays in a nameable regime. The one candidate
explanation visible in the committed columns is load-port pressure — 3m runs at
`f/l` 0.89–1.71 against planar's 1.78–3.75, i.e. it is load-bound where planar is
FMA-bound — and Ice Lake evidently punishes that. That is a **hypothesis from one
column of one sweep**, not a measurement.

**Consequence: item 3 is dead rather than blocked**, and no node time is owed to it.
The Ice Lake shape sweep that was on the to-do list is deleted from it.

#### The TBLIS 2.0 columns are under review, and why

This run used `../baselines/tblis-2.0-install` — the build every previously
committed TBLIS 2.0 number used. Its BLIS was configured with
`BLIS_CONFIG_FAMILY=auto`, which resolves against the *build* machine, so it is
**skx-only**: `nm` shows one `bli_cntx_init_skx`, 1364 `bli_` symbols, and no
skinny-GEMM (`sup`) kernel set. Rebuilding the same source at the same commit
with `BLIS_CONFIG_FAMILY=x86_64` gives thirteen contexts and 3809 symbols, and a
one-shot look on `ccqlin038` put it **up to 1.68x ahead** on the premise shapes,
clustered on the low-arithmetic-intensity cases.

Two things follow, and the first is uncomfortable:

1. **A mis-configured baseline flatters this engine on exactly the shapes the
   project identifies as the real headroom.** The README's "roughly parity with
   TBLIS 2.0 on complex" rests on it.
2. The Phase 1 headline is probably safe, because it is a ratio *within* TBLIS 2.0
   (complex efficiency ÷ real efficiency) and both halves moved together in every
   pair inspected. It should still be re-derived.

That 1.68x is `--size 8 --reps 2`, one shot, on a shared machine, with no warm-up
and no repeat arm — a signal, not a result, by this file's own standards.
`scripts/ab-tblis.sh` prices it properly (job 6754877): warm-up, A (skx),
B (multi-config), A′ (skx), the treatment being `LD_LIBRARY_PATH` on one binary,
with **`planar` in every arm as a control no TBLIS library can move**. It covers
the sweep corpus and then the premise shapes, the latter because a dry run showed
the two libraries indistinguishable over the sweep corpus at trivial size and the
original signal came from `premise`.

**Resolved by the next subsection: the TBLIS 2.0 columns from this run are
sound after all.** The A/B found the two builds identical at both measured sizes,
so the skx build understated nothing here, and the 1.68x was an artifact of the
8 MiB size it was taken at. Read this paragraph together with that retraction.

#### The TBLIS A/B lands, and it retracts the alarm — the 1.68x was a size artifact

Job **6754877**, `worker6156` (the same node, which is the best case for
comparability), 2026-08-04, stage 1 = 90 min. Data in
`bench-results/worker6156-icelake/ab-tblis/`.

**The two TBLIS 2.0 builds are indistinguishable at the sizes this project
publishes at.** Treatment is multi-config ÷ skx; `planar` is the in-arm control,
which no TBLIS library can move.

| | `f64` | `c64` | `f32` | `c32` |
|---|---|---|---|---|
| sweep corpus, 64 MiB — floor (`A2` vs `A`) | 1.001 | 0.998 | 1.001 | 0.999 |
| sweep corpus, 64 MiB — **treatment** | 1.000 | 0.997 | 0.998 | 0.997 |
| premise, 200 MiB — floor | 1.000 | 1.000 | — | — |
| premise, 200 MiB — **treatment** | 0.998 | 0.998 | — | — |

Every treatment ratio is inside its floor. Per case on the premise shapes the
spread is 0.979–1.002 against a floor spread of 0.992–1.003, and **the specific
shapes that produced the original 1.5–1.7x read 0.993–1.001**: `ajbc-ckba-jk`
0.993/1.000, `abrs-qb-aqrs` 0.997/0.999, `ijkl-imkn-jnlm` 1.000/1.000,
`ij-ikl-ljk` 0.999/0.999, `abjcd-dkbac-jk` 0.999/1.001.

**So the alarm is withdrawn.** The claim that the skx-only build understated
TBLIS 2.0 by up to 1.68x on low-arithmetic-intensity shapes, and that the
README's near-parity-with-TBLIS-2.0 statement therefore rested on a
mis-configured baseline, **was wrong.** Nothing in the README or the CHANGELOG
needed to change on that account, and the Phase 1 headline is untouched.

**What actually happened, because the process failure is the useful part.** The
1.68x came from one `premise --size 8 --reps 2` run: a problem size **8 to 25x
smaller** than anything this project publishes (sweeps at 64 MiB, premise at
200 MiB), on a shared machine, no warm-up, no repeat, no control. At 8 MiB the
operands are small enough that BLIS's skinny-GEMM (`sup`) kernels — present in the
multi-config build and absent from the skx-only one — decide the result. At 64 and
200 MiB they are irrelevant, so the two libraries converge exactly. The symbol
count that looked like a smoking gun (1364 against 3809 `bli_` symbols) was real
and had no consequence at the sizes that matter.

This is a **new failure mode for this file's methodology rules**, which are all
about time: discard a warm-up arm (A31), derive the floor in session (A32), price
co-tenancy per arm (A27). None of them would have caught this, because the
confound was **problem size**, not timing or contention. Hence A46.

It also vindicates a decision that was not mine: rewriting the docs on the
strength of the signal was considered and rejected in favour of measuring first.
Had the signal been acted on, a false retraction would have gone into the
release-facing documents — the exact failure mode the CHANGELOG's confidence
table exists to prevent, arriving through the person maintaining it.

**What survives.** The portability half, entirely: the skx-only build SIGILLs on
any machine without AVX-512, which killed job 6753261, and
`../baselines/tblis-2.0-x86_64-install` is now the build a non-AVX-512 node needs.
And a genuine positive result, cheaply held: the TBLIS 2.0 baseline is **measured
to be build-insensitive at publication sizes**, where before it was an
assumption nobody had tested.

#### Replicated, and the baselines placed

Job 6754877 stage 2 ran the **whole comparison set a second time** on the same
node, against the multi-config TBLIS 2.0 (231 min, rc=0). Data in
`compare-tblis-x86_64/`. So there are now two independent full runs, in separate
allocations three hours apart, and they agree across **all twenty dtype × engine
columns to 0.997–1.003**. The second run's own near and session floors are both
1.000–1.002. The `tblis` column agrees to 0.998–1.001, which confirms the A/B
verdict a second way — a full run does not see the build difference either.

That is the strongest evidence base this repo has for any comparison number: two
runs, each internally bracketed, agreeing at the level of their own floors.

**The Phase 1 headline replicates, and one number is identical to three digits.**
Efficiency against a same-shape GEMM ceiling, mean over the 12 premise cases at
200 MiB, complex efficiency ÷ real efficiency *within each engine*:

| engine | `f64`→`c64` | `f32`→`c32` | Cascade Lake, Phase 1 |
|---|---|---|---|
| **TBLIS v1.3.0** (latest stable) | **0.215** | — | **0.215** |
| TBLIS 2.0-dev | 1.413 | 1.306 | 1.06 / 1.15 |
| TTGT (OpenBLAS) | 1.307 | 1.348 | 1.17 |
| this engine, planar | 1.316 | 1.232 | — |
| this engine, 3m | 0.934 | 0.950 | — |

**TBLIS v1.3.0's 0.215 on Ice Lake is the same 0.215 measured on Cascade Lake**,
and that is not a coincidence to be marvelled at — it is the mechanism confirming
itself. 1.x has no complex micro-kernel for any post-Sandy-Bridge x86
configuration, so complex runs the generic template while real gets tuned
assembly, on *both* machines. The ratio is a property of the software, not of the
hardware, which is exactly what Phase 1 concluded from the flat-across-shapes
diagnostic. Nothing in this project has replicated so cleanly.

The refutation against 2.0-dev is *stronger* here than on the reference machine
(1.413 against 1.06). 3m is the only engine below 1.0, consistent with A44.

**Where the engine sits, median GF/s over those 12 cases** — same run, same core,
single-threaded throughout:

| dtype | this engine | TBLIS 2.0-dev | TTGT | engine ÷ TBLIS 2.0 |
|---|---|---|---|---|
| `f64` | 46.5 | 30.8 | 14.0 | **1.51x** |
| `c64` | 64.0 | 48.8 | 25.9 | **1.31x** |
| `f32` | 81.1 | 45.6 | 25.9 | **1.78x** |
| `c32` | 123.2 | 88.3 | 49.2 | **1.40x** |

This is a materially better position than the README's current claim of "roughly
parity with TBLIS 2.0 on complex", which was a Cascade Lake Phase 1 number
(42.5 against 42.4 in `c64`) taken before every Phase 4 gain. **Two caveats keep
it honest.** It is a different machine, so it is not evidence of how much Phase 4
bought — that remains unmeasured and needs a `ccqlin038` run. And the engine is
*handicapped* here: A34 says three of the eight shipped register blocks are 9–13%
off on Ice Lake, and A44 says the method ranking does not transfer. It wins these
columns while running shapes chosen for a different microarchitecture.

#### Assumptions added

| # | Assumption | Status |
|---|---|---|
| A44 | The complex-method ranking, the memory-bound inversion, **and 3m's L1-resident advantage** are properties of the engine and the shape. | **Refuted, and no longer confounded.** On Ice Lake 3m is last in every column and wins 0 of 49 cases (0.694 `c64`, 0.744 `c32` against planar, against Cascade Lake's 0.956 / 0.921), and the inversion is absent. The A34 confound was *assumed* and does not exist: 3m ships `(1, 10)` in both precisions and the Ice Lake sweep names that as 3m's own best shape, so there is no better shape to give it. At shipped shapes the kernel-level ratio goes 0.854 → 0.589 (`c64`) and 0.996 → 0.606 (`c32`), the collapse is uniform across 3m's whole shape space, and **at the L1-resident `kc = 16` 3m leads by 10–16% on Cascade Lake and trails by 34–43% on Ice Lake** — so the mechanism this project called settled is machine-specific too. Item 3 is **dead**, not blocked. |
| A45 | A baseline install built from the right source at the right commit is the right baseline. | **Refuted, but on portability only** — the performance half was measured and withdrawn, see the A/B subsection. `BLIS_CONFIG_FAMILY=auto` silently fits BLIS to the *build host*, so this TBLIS 2.0 is skx-only and **SIGILLs on any machine without AVX-512** (job 6753261). It also lacks the skinny-GEMM kernels, which looked like a large performance defect and turns out to cost **nothing at 64 or 200 MiB** (0.997–1.000 against a 0.998–1.003 floor). So: record a baseline's *configuration*, not just its version and commit; verify a multi-ISA claim with `nm` rather than `strings`, because BLIS compiles its config name table in whether or not the kernels are there; and do not infer a performance consequence from a symbol count. |
| A46 | This file's measurement rules (A27, A31, A32) cover the ways a comparison can mislead. | **Refuted: they are all about time, and problem size is a separate axis.** A one-shot `premise --size 8` run reported TBLIS 2.0's two builds differing by up to 1.68x. At the sizes actually published — 64 MiB sweeps, 200 MiB premise, i.e. 8–25x larger — they are identical to within the floor, because the kernels that differ only matter while the operands are small. A warm-up arm, an in-session floor and per-arm occupancy would each have passed the bad measurement through unchanged. **Measure at the size you publish at, and treat a result taken at a smaller size as being about that size.** |

---
