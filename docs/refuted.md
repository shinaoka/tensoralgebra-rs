# Refuted — things that were tried and did not work

**This file is not part of the per-session reading list.** It is a lookup table.
**Consult it before proposing any performance idea, any measurement design, or
any claim about what this engine or this corpus does.** Roughly half the ideas
that look obvious here have already been measured and lost, and the other half
lost for a reason that tells you where to look instead.

Negative results are the most valuable content this project produced, and the
easiest to mistake for clutter. They live here so they stop competing for space
in `CLAUDE.md` and in the live sections of `notebook/`, and so each one
carries the field that makes it useful rather than discouraging: **what would
reopen it.**

The full accounts stay in [`notebook/`](notebook/README.md); every entry below
points at one. For what *did* survive measurement, see
[`results.md`](results.md).

Thirty entries, each with a confidence level, an evidence field naming a job id,
a CSV path or a `file:line`, and a reopening condition. **An entry without
evidence is a rumour and does not belong here.**

## The confidence ladder

| level | means | how to treat it |
|---|---|---|
| `settled` | Replicated on ≥2 machines or ≥2 independent sessions, **with a control column that did not move.** | Reopen only with a new machine or a new mechanism. Re-deriving it is waste. |
| `measured once` | One session, one machine, floor derived in-session. | The **direction** is trustworthy, the **magnitude** is not. Do not quote the number off its machine. |
| `structural` | Follows from the code or the shape of the problem; no measurement needed. | Holds until the structure changes. Check the structure, not the number. |
| `single observation` | One code read, one document, one data point. | Enough to stop a wrong assumption. Not enough to build on. |

Every entry has an **evidence** field: a job id, a CSV path, or a `file:line`.
An entry without one is a rumour and does not belong here.

---

# Engine and performance

## The complex-weakness thesis

**Tried.** The project's founding hypothesis: existing engines underperform on
complex contraction — worst on memory-bound and awkward-stride shapes — because
interleaved-complex storage compounds with scatter/gather packing and forces work
onto the slow path.

**Expected / what happened.** Expected a large, mechanism-attributable gap.
Found a large gap against **TBLIS v1.3.0** (complex-over-real efficiency ratio
**0.215**, i.e. ~5x) and **none** against TBLIS 2.0-dev (**1.06**; 1.03 with
irregular strides forced, 1.15 in single precision). The mechanism is wrong:
TBLIS 1.x has **no complex micro-kernel** for any post-Sandy-Bridge x86
configuration, so complex runs the generic templated fallback while real gets
BLIS assembly. Its complex throughput is flat across shapes (4.1–9.1 GF/s) where
real spans 6.2–48.4 — a fixed kernel ceiling, not a memory effect. Block scatter
was fully regular (`regA = 1.00`) in every case, so the gather path was not
involved at all. Complex is not intrinsically disadvantaged: 4x the flops on 2x
the bytes is **twice the arithmetic intensity**, so overheads amortise *better*.

**Confidence.** `settled` — and replicated on a second microarchitecture: TBLIS
v1.3.0's 0.215 came back *identically* on Ice Lake, because the ratio is a
property of the software.

**Evidence.** `bench-results/phase3-premise-{f64c64,f32c32}.csv`,
`phase3-premise-tblis130-f64c64.csv`; replication in
`bench-results/worker6156-icelake/compare-tblis-skx/` (job 6753260). Phase 1
report; A2b.

**What would reopen it.** Nothing about TBLIS. The *displaced* finding is where
the headroom actually is — **low arithmetic intensity in either domain** (TBLIS
2.0 runs 0.85 of a same-shape GEMM ceiling on large compute-bound contractions
and **0.34** on small-`k` skinny ones). A mechanism specific to complex
*storage*, demonstrated on a shape where arithmetic intensity is held fixed,
would be a new claim rather than this one.

## Pinned `kc = 512`

**Tried.** Shipping a deeper `kc` for 8-byte real geometry, on the strength of a
Zen2 sweep.

**Expected / what happened.** Expected a portable few-percent win: **1.050 on
Zen2, 0.961 on Cascade Lake.** The sign flips between machines. `kc = 384` is
the measured optimum on the reference machine and the shipped 256 is close to it.

**Confidence.** `settled` — two machines, opposite signs.

**Evidence.** `bench-results/worker5040-zen2/` (job 6745376),
`bench-results/ccqlin038-blocking/`. Blocking report, part 7.

**What would reopen it.** A third machine class where the optimum is deeper
*and* an account of why the L2/L1 geometry puts it there — the mechanism section
of part 7 gives the shape of that argument. Not another sweep on these two.

## Coupled deepening (`TENSORCONTRACT_DEEPEN`)

**Tried.** Setting `kc = 512` *and* re-deriving `mc`/`nc` at that depth, which
looked like +3% on two machines — the one blocking result that appeared to
replicate.

**Expected / what happened.** Expected +3% end to end. **Failed its end-to-end
A/B.** Both grids' `base` arm was 1–2% slow, which inflated every `arm/base`
ratio in both grids *identically*, so the two-machine agreement was a shared
artefact rather than a replication. Control-corrected, the treatment is ≈0.977.

**The switch is gone.** It survived for a while as an off-by-default record of
the experiment; it was **removed before 0.1.0**, because a refuted branch in a
published crate is a maintenance cost and a reader's false lead, and this entry
plus the committed data is the record. The last commit that contains it is the
parent of the one that removed it — `git log -S TENSORCONTRACT_DEEPEN` finds
both. Nothing about the result changes; the code implementing it does not need
to ship for the finding to stand.

**Confidence.** `measured once` for the refutation (one end-to-end A/B on the
reference machine, floor derived in session), `settled` for the conclusion that
the original +3% was an artefact — the shared-`base` mechanism is arithmetic, not
a measurement.

**Evidence.** `bench-results/ab-deepen/` (`scripts/ab.sh`);
`bench-results/ccqlin038-blocking/`'s slow `base` arm is documented in its own
`PROVENANCE.txt`. Part 7's A/B subsection.

**What would reopen it.** Nothing on this machine. This entry's real value is
the **methodology**: a grid whose `base` arm is off contaminates every ratio in
it in the same direction, which looks exactly like a replication. See A20 below.

## `MC` as a tuning lever

**Tried.** Finding a few percent in the L2 blocking factor.

**Expected / what happened.** `MC` is a **wide plateau**: a sixteenfold range
moves the corpus geomean by at most 3%. There is nothing to win.

**Confidence.** `settled` — swept on Zen2 and on the reference machine.

**Evidence.** `bench-results/worker5040-zen2/`,
`bench-results/ccqlin038-blocking/`; score offline with
`scripts/blocking-score-rules.py`. Part 7.

**What would reopen it.** A machine whose L2 is small enough that the plateau
has an edge inside the useful range, or a change that makes the packed `A` block
compete with something new (a thread pool's per-thread buffers, say).

## The analytical cache-blocking model as a portability fix (A33)

**Tried.** Deriving `mc`/`kc`/`nc` from probed cache descriptors via BLIS's
analytical model, instead of three hand-fitted constants — built specifically to
solve the "the constants belong to one workstation" problem.

**Expected / what happened.** Expected it to be *safe* on an unseen machine.
On the first unseen machine it is **worse in 11 of 12 columns, by up to 7.2% in
the complex methods**, against constants fitted to a completely different
hierarchy. On the reference machine it costs 14–34%. The whole Zen2 loss is
attributable to its `kc`; on the reference machine its `mc` is what sinks it —
so *both halves* of the derivation are wrong, in different places. "Analytical"
bought traceability and cost throughput, and the two were assumed to come
together.

**Confidence.** `settled` — two machines, and the loss localises to different
terms on each.

**Evidence.** `bench-results/worker5040-zen2/blocking-model.txt` and the model
arms in `bench-results/worker5175-zen2/` (job 6745978); part 9's measured
subsection; A33.

**What would reopen it.** A corrected `kc` term with an independent argument
(the model's `kc` assumes an L1-resident `A` sliver, which is the regime the
method ranking turns on — see part 7's mechanism section). The probing and the
cache descriptors it introduced are kept and used regardless; only the default
is refuted.

## The `C`-traffic account of depth-adaptive `MC`

**Tried.** Shrinking `MC` at shallow `kc` on the theory that the binding cost is
the `D` strip a `jr` pass revisits, so the `C`/`D` traffic per unit of kernel
work is what `MC` should be chosen against.

**Expected / what happened.** The mechanism is real as a *bound* (A13: `MC` also
bounds the revisited `D` strip, which binds whenever the output's rows are
strided) and **worthless as a rule**. Depth-adaptive `MC` measured flat to
negative, and part 7 then falsified the traffic account outright as the thing
that decides `MC`.

**Confidence.** `measured once` for the rule, `structural` for A13's bound
(it follows from the loop order).

**Evidence.** Part 1's "a negative result on depth-adaptive `MC`" subsection;
part 7's `MC` plateau; A13.

**What would reopen it.** An output geometry where the `D` strip genuinely
dominates — the corpus's strided-output cases were priced by the orientation work
at ~2x and the *orientation* is the instrument that buys them, not `MC`.

## A cross-domain traffic term in the partition cost model (A38)

**Tried.** Extending `Plan::partition`'s cost model with a term for traffic
across L3 domains, so the column axis gets chosen by a continuous model rather
than by a binary gate. **Four variants.**

**Expected / what happened.** Expected calibration against the committed grids.
**Cannot be calibrated.** The weights that separate the two `abcijk`/`ijkl`
family pairs are 1% apart and want **opposite** answers — in element units and in
real units alike — and the symmetric two-sided model ranks the two families
backwards. The shipped answer is therefore a binary gate (D41) over the two arms
that were actually measured.

**Confidence.** `measured once` — scored offline against grids from two Zen2
nodes, so the refutation is as good as those grids and no better.

**Evidence.** `scripts/partition-score-rule.py` against
`bench-results/worker5137-zen2/` and `worker5479-zen2/`; part 12's negative-result
subsection; A38.

**What would reopen it.** A new mechanism — not a new weight. A machine where
the two families separate under *some* weight would also do it, and the grids to
check that against are free.

## Write-back regularity as an objective (wrong three times)

**Tried.** Choosing things so as to maximise the fraction of output row blocks
that avoid the gather path. Once as a row-block rule (A16), once as an
orientation rule (part 6), and once as the naive form of the row-block rule
before its guards.

**Expected / what happened.** Scores **0.936 in `f32`** as a row-block rule and
**0.936** as an orientation rule. It has now pointed the wrong way three times.
It explains the write-back path and nothing else; the kernel shape it gives up to
get there costs more than the gather path it avoids.

**It is a gate on a change, never an objective.** As a gate it is load-bearing:
the row-block rule needs it plus two more guards before it stops being a loss.

**Confidence.** `settled` — three independent derivations, all scored against
whole-corpus grids.

**Evidence.** `bench-results/phase4c/` and `bench-results/phase4d/` (both grids,
all 392 case-dtype-methods); `scripts/rowblock-score-rules.py`,
`scripts/orient-score-rules.py`. A16; parts 5 and 6.

**What would reopen it.** A machine or a kernel where the gather path is much
more expensive relative to kernel shape than it is here. Note that the corpus
*does* exercise the gather path — `reg_a < 1.0` on 42.9% of case-dtype-methods at
the shipped `f32`/`c32` blocks and `--size 64` — so this is not a "the corpus
can't show it" result. See A4 below.

## Removing the `panels >= p` early return (A36)

**Tried.** `Plan::partition` returns `pm = p, pn = 1` as soon as the row axis has
enough `MR` panels to fill the threads. On Zen2 at 64 threads that costs up to
**4.3x** on 18 memory-bound cases, so: delete it.

**Expected / what happened.** Deleting it is **wrong**. On Ice Lake, where one
L3 serves the whole 32-core socket, the same forced column split is worth
**1.014** — the effect is simply absent, and the early return is *correct* there.
The discriminator is **how many L3 domains the thread set spans**, because `NC`
sizes the shared packed-`B` panel for *an* L3: spread over sixteen 16 MiB L3s, a
panel every thread reads whole is effectively replicated sixteen times and
re-streamed from memory.

So the fix is to **gate** the early return on the domain count, not to remove it
(D41, now the default per D44).

**Confidence.** `settled` — refuted by a second topology, then separated from
thread count *within* one machine: 16 threads packed into 4 domains reads
1.19–1.29x where 16 threads spread over 16 domains reads **1.49–3.05x**.

**Evidence.** `bench-results/worker6016-icelake/phase4f/` (job 6746817),
`worker5137-zen2/` (job 6751550), `worker5479-zen2/` and `worker6150-icelake/`
(jobs 6753208/6753209). Part 8's Ice Lake subsection; part 12; A36.

**What would reopen it.** Nothing — this one is built. The *residual* question is
whether the gate's other two conditions (`blocks >= p`, `k <= 64`) are the right
ones; D42 makes the whole truth table unit-testable on any machine.

## The thread pool as a default (D53, withdrawn the day it was written)

**Tried.** Replace per-call `std::thread::scope` with parked workers, removing the
~20–36 µs/thread A43 priced. Measured on Zen2 at up to **11.6x** (0.25 MiB, 64
threads), 2.0–2.8x at 16 MiB, with no per-family cell below 0.99 at any size or width
and a per-case sub-0.90 tail of 0.7–1.4% indistinguishable from A39's floor. On that
basis it was recommended as the default (D53), with the flip left to the user pending a
second machine class.

**Expected, and what happened.** The second machine class **reversed it**. On Ice Lake
— 32 cores sharing one 48 MiB L3 — `pool / base` is **46 of 60 per-family cells below
0.97, worst 0.403**, i.e. up to 2.5x *slower* than per-call spawn, with 36–48% of
individual points below 0.90 at 64 MiB. At the pre-registered decision cell (64 MiB,
`t32`) it reads **0.80** against the `< 0.97` threshold that had been written down in
advance as "a real surprise".

So: 11.6x better on one machine class, 2.5x worse on another, with the same code and a
clean `t1` control (0.996–1.007) on both.

**The a-priori argument that this could not happen is the useful part.** It was recorded
before the run: *"the pool has no reason to be topology-dependent — it touches where
threads come from, not what they read."* False. **Reusing a thread reuses its allocator
arena**, so every worker gets the same packed-`A` buffer address back on each call where
a freshly spawned thread gets a fresh, well-spread one. A pool is an
**allocation-locality change** as much as a thread-lifetime one (A55) — and 32 buffers
at repeating addresses contending in one shared L3 is exactly the configuration that
loses, where Zen2 spread the same buffers over sixteen private L3s and won.

**Confidence.** `settled` that it does not transfer — two machine classes, opposite
signs, four dtypes each, both with clean controls. `measured once` for each magnitude.
`single observation` for the mechanism, which is a candidate and not a finding: the cost
grows with width and with size then plateaus, which the implementation's fixed costs
(~100–200 µs of serial wakes and one contended completion mutex) are an order of
magnitude too small to explain; a 32-participant barrier is the other candidate, and
`t16` being worse than `t32` fits neither cleanly.

**Evidence.** `bench-results/worker5086-zen2/phase4g/` (job 6760092, the win) and
`bench-results/worker6194-icelake/phase4g/` (job 6762432, the loss). Parts 18 and 19;
D49, D52, D53, A55, A56.

**What would reopen it.** A named, cheap experiment: **offset each worker's packed-`A`
buffer by a per-worker amount** so the addresses cannot alias, then re-run both nodes'
arms (~1.8 h each). If the Ice Lake loss disappears, the pool becomes a default
candidate again and A55's mechanism is confirmed; if it does not, the barrier is next
and the pool stays conditional. Note what is *not* in doubt: the pool is worth up to
11.6x on Zen2, so this is an implementation defect to find rather than a dead idea —
which is why D49 keeps the switch rather than deleting the code.

**And a rule that came out of it.** Anything touching threads or caches is a
**per-microarchitecture** claim until shown otherwise. Four now fail to transfer:
register blocks (A34), the thread partition (A36), the complex-method ranking (A44),
and this (A56). One machine class is not evidence for a default, and this project has
now been wrong on one machine class twice in the same week.

## Cost-weighted and dynamic partitioning for the dense path (D47)

**Tried.** Block-scatter blocks are *not* equal-cost — 123 of 392
case-dtype-methods have mixed regularity within one contraction, at 1/3, 1/9 or
1/27 of blocks on the gather path, which the orientation work priced at roughly
2x. A static cut by panel *count* must therefore be leaving load imbalance on the
table. Ladder: cost-weighted static cut → dynamic `ic` claiming → a
dependency-counted task DAG.

**Expected / what happened.** Built the perturbation instead of the fix.
`--stress ragged` takes heterogeneous cases from **11.7% to 87.2%** of
case-dtype-methods and from periodic quantisation (0.667 / 0.889 / 0.963) to
aperiodic fractions down to 0.348. Parallel efficiency, each mode against its
own `t1`, moves **0.976 to 1.124** — i.e. not at all, and if anything ragged
scales slightly *better*. Seven and a half times the heterogeneity, no cost.

Why, most likely: at 64 threads occupancy is 32–39%, so threads wait on memory
rather than on each other and compute-side imbalance hides inside that; and
static strips are 12+ panels wide, which averages a good deal even when the
pattern is aperiodic.

**Confidence.** `measured once` — one Zen2 node, six dtype × width points, each
against its own `t1`.

**Evidence.** `bench-results/worker5178-zen2/phase4f/rg-*` (job 6755009);
`RAGGED=1 scripts/phase4f-threads.sh`. Part 16's second experiment; D47; A41.

**What would reopen it.** Two openings, both real and both named in D47:
**block-sparse with wildly varying block sizes**, where the imbalance is block
*size* rather than block *regularity* and is far larger; and a
**compute-bound** regime — this was measured where the machine is bandwidth-bound,
so a faster kernel or a smaller working set could expose the imbalance the
perturbation could not.

Corroboration worth knowing: TBLIS computes the same per-block regularity
sentinel and **feeds it to no scheduling decision** in either version, and its
author published the diagnosis without a fix (A42). Reaching the same conclusion
by inaction is weak evidence, but it is evidence.

## The thread-count amortisation guard, in the form that was built (D48 → D52)

**Tried.** Cap the thread count so per-call spawn stays a bounded fraction of the
work: `p <= work_fmas / 3e6`, holding the spawn fraction at ~0.2. Built as the
smaller of the two answers to A43 — it steers around the spawn cost rather than
removing it — and calibrated offline against a committed grid before being shipped
off-by-default.

**Expected, and what happened.** Expected "no worse than serial, everywhere". Got a
**trade**. Measured on `worker5086` at 64 Zen2 cores, drift-corrected per family:

| 0.25 MiB | t2 | t4 | t8 | t32 | t64 |
|---|---|---|---|---|---|
| `f64` | 0.76 | **0.61** | 0.83 | 2.81 | 7.86 |
| `f32` | 0.82 | **0.73** | 1.07 | 4.10 | **10.84** |

It rescues an over-threaded caller by up to 10.8x and costs a correctly-threaded one
10–39% at 2–8 threads, at the one size where it acts at all (inert at 4 and 16 MiB).
Per case it leaves **12–14% of all points more than 10% slower** at those widths,
worst 0.327 — far outside the ±11% a 64-thread measurement produces on identical
configurations (A39), so systematic rather than noise. **And on top of the thread
pool it is pure loss**, reaching 0.32, because it rations a cost the pool has already
removed.

Which side a caller lands on depends on whether their thread count was already well
chosen — which a library cannot know.

**Why the offline calibration missed it, and this is the transferable part.** The
scoring simulated `requested = 64` and nothing else, so it measured only the winning
half of the trade and reported "0 points slower than serial at all four sizes" —
true, at 64 threads. **A rule that consumes a caller's parameter must be scored across
that parameter** (A54). `partition-score-rule.py` does this correctly, sweeping `-p`
and `-d`; `amortise-score-rule.py` did not, and now documents that it does not.

**The form is wrong, not the constant.** A fixed spawn-*fraction* threshold says
"4 threads on a 0.32 ms contraction is 46% overhead, cap it" while measurement says 4
threads is the optimum. The real trade is *marginal* — does the next thread return
more than it costs — and answering that needs a scaling model, which is where the
analytical blocking model died (A33). Fitting a second constant would be fitting the
same wrong shape.

**Confidence.** `measured once` for the magnitudes (one Zen2 node, one session, floor
derived in-session from per-row `t1` controls); `settled` for the direction, because
the losing and winning halves are both large and appear in all four dtypes; and
`structural` for the form argument, which does not depend on the measurement.

**Evidence.** `bench-results/worker5086-zen2/phase4g/` (job 6760092), arms `sm-guard-*`
and `sm-both-*` against `sm-*`. Re-derive the offline half with
`scripts/amortise-score-rule.py bench-results/worker5086-zen2/phase4g`. Part 18; D48,
D52, A50, A54.

**What would reopen it.** Not a new constant. A **marginal** criterion — compare the
predicted gain of thread `p+1` against its cost, rather than a fraction of total work
— would be a different rule and would need its own scaling model to be worth trying;
A33 is the warning about what that costs. Alternatively, a caller-supplied hint ("this
thread count is a ceiling, not a request") moves the decision to where the knowledge
is. Note also that the guard's whole justification was D46's sub-megabyte loss, and
**the pool removes that loss directly** (D53), so the guard now has little left to
protect.

## Barriers as the reason to restructure threading

**Tried.** The intuition that the two barriers per `(jc, pc)` are what caps
parallel efficiency, so a barrier-free design (a task DAG) is the way up.

**Expected / what happened.** **Backwards.** Barriers cost *skew* per iteration;
imbalance costs *total*, because a systematically unlucky strip does more work
every iteration and the barrier merely exposes it. Removing barriers does not fix
imbalance. And the engine already *has* a barrier-free configuration — `pm == 1`,
which the domain-aware gate now selects on chiplet machines for exactly the
memory-bound family.

**Confidence.** `structural` — it is an argument about what a barrier does, and
`crates/tensorcontract/src/driver.rs:443` is the code it is about.

**Evidence.** Part 14, "What is *not* a reason to do any of this". Earlier drafts
of that argument had the emphasis backwards, which is why it is recorded.

**What would reopen it.** A measurement of barrier wait time that is large after
the load is known to be balanced. Nothing has measured that.

## `pc` fusion as a general enabler

**Tried.** Fusing the `pc` loop so `C` is touched once rather than `K/KC` times
was described in earlier drafts of this project's own notes as the enabler for a
general task graph.

**Expected / what happened.** It is not. Fusing makes the packed `A` block
`mc x K`; at `K = 3744` and `mc = 256` that is **7.7 MB in `f64`**, far past any
L2. It is viable only for **small `K`** — which is the memory-bound family, where
it is most wanted, so it stays on the list as a *targeted* optimisation. But the
road to a general task graph is a dependency-counted DAG, not fusion.

**Confidence.** `structural` — arithmetic on the footprint.

**Evidence.** Part 14's design ladder, item 3 and the paragraph after it.

**What would reopen it.** Nothing about the general claim. The small-`K` case is
unrefuted and unbuilt.

## `rayon` for the intra-contraction path

**Tried.** Using `rayon`'s `scope`/`spawn` instead of `std::thread::scope`, to
get a thread pool for free and remove the ~20–36 µs/thread per-call spawn cost.

**Expected / what happened.** **Deadlocks by construction.** The parallelism here
is SPMD-with-barriers: `pn` barriers, each of width `pm`, and every thread
rendezvous twice per `(jc, pc)` iteration
(`crates/tensorcontract/src/driver.rs:443,569,586`). A rayon task that blocks on
a barrier occupies a worker thread; if the pool has fewer workers than barrier
participants — which a caller's pool routinely will — the remaining participants
are never scheduled and the barrier never opens. Work-stealing also fights the
design directly: D21/D27 give every output element one owning thread precisely so
no reduction is parallelised.

`rayon::ThreadPool::broadcast` *does* fit the shape — it runs one closure per
worker, which is what SPMD wants — but it ties the parallel degree to the pool
size, where `Plan::partition` chooses `(pm, pn)` per contraction. And a global
pool inside a library fights the host runtime, which for this project is a
concrete problem rather than a hypothetical: the Julia package and the C
consumers both bring their own.

**Confidence.** `structural` — the barrier structure is in the code, and the
deadlock is a property of blocking inside a bounded pool.

**Evidence.** `crates/tensorcontract/src/driver.rs:107,443,569,586`; the
build-vs-reuse row for `rayon`; part 15's note that TBLIS also spawns its
parallel region per call (`external/tci/src/tci/parallel.c:24`) and so has no
advantage here either.

**What would reopen it.** Nothing for the intra-contraction path. Our own
parked-thread pool is built instead (D49,
`crates/tensorcontract/src/pool.rs`), and building it surfaced two things worth
carrying: a pool must decline **all-or-nothing** rather than folding surplus indices
onto one thread, which would deadlock on the shared barrier (A51); and it must
recover from mutex poisoning, or one panicking contraction stops it pooling for the
life of the process (A52).

**rayon does fit the batched API** — the batch axis has no barriers and plain
fork-join is right. It is still not used there, but for a much weaker reason: ten
lines of `std::thread::scope` do the same job and the crate already owns a pool that
can be extended to that axis (D50). The objection on the batch axis is
*unnecessary*, not *unsound*, and that distinction is the point of this entry.

## `K`-parallelism (A21)

**Tried.** Parallelising the `pc` loop, with per-thread accumulators and a
reduction, for compute-bound contractions too narrow to fill the threads from
`M` alone.

**Expected / what happened.** Not needed, and the argument is structural rather
than empirical. 20 of 392 case-dtype-methods cannot fill 8 threads from `M`;
**all 20 can from `M x N`.** And needing `K` at all requires fewer than `p`
micro-tiles in the whole output, which bounds arithmetic intensity at
`~2MN/((M+N)*bytes)` and so bounds the case *away* from compute-bound. The
premise is self-defeating.

**Confidence.** `structural`, and confirmed empirically at every thread count to
128 in both instruction sets — 16x the width it was argued at. Independently
corroborated: TBLIS parallelises four of its five loops and **never `PC`**, for
the same stated reason.

**Evidence.** A21; part 10's replication; part 15's reading of TBLIS. Note that
D45 (bitwise identity is no longer a design constraint) does **not** reopen this:
A21 was argued from shape, not from float ordering, so it survives independently.

**What would reopen it.** A real shape with fewer than `p` micro-tiles in the
output *and* enough arithmetic intensity to be worth parallelising — which the
bound above says cannot be a corpus case, but could be a caller's.

## The complex-method ranking, and the memory-bound inversion, as facts about the engine (A44)

**Tried.** Phase 3 measured, on Cascade Lake, that planar wins the corpus and
that the ranking **inverts on memory-bound shapes** where 3m's 25% flop saving
pays. Phase 4 item 3 was to exploit that: dispatch 3m when
`min(n, k) <= 64`, planar otherwise.

**Expected / what happened.** On Ice Lake the ranking does not transfer and the
inversion is **absent**:

| relative to planar | Cascade Lake | Ice Lake |
|---|---|---|
| `c64` 1m | 0.967 | 0.868 |
| `c64` 3m | 0.956 | **0.694** |
| `c32` 1m | 0.979 | 1.004 |
| `c32` 3m | 0.921 | **0.744** |

3m is **last in every column and wins 0 of 49 cases**, per-case ratios running
0.63–0.84 — uniform, not a few catastrophic shapes. On the memory-bound subset,
where Cascade Lake had 3m *fastest*, Ice Lake reads planar 49.3 > 1m 44.7 > 3m
35.9 (`c64`). **Item 3 is a pessimisation there.**

**Item 3 is dropped as a rule** (user's decision, 2026-08-05). An unconditional
rule is refuted; a per-microarchitecture rule would need per-microarchitecture
shape tables that do not exist. Neither the ranking nor the inversion may be
quoted without naming Cascade Lake.

**Confidence.** `settled` for the ranking, and **the confound is gone**. It was
recorded as confounded with A34 (three of eight shipped blocks are 9–13% wrong on
Ice Lake) and the deconfounding sweep turned out to be **already committed** —
`bench-results/worker6016-icelake/kernel-shapes.txt` covers all four methods. It
says the confound never existed: **3m ships `(1, 10)` in both precisions and that is
the Ice Lake sweep's own best 3m shape**, so there is no better shape to give it.
A34 lands on `real` and on planar `f32`, not on 3m.

At shipped shapes, kernel-level, at each dtype's operating `kc`: `c64` goes
102.8/87.8 = 0.854 on Cascade Lake to 108.4/63.8 = **0.589** on Ice Lake; `c32` goes
0.996 to **0.606**. Those bracket the corpus ratios, and the collapse is uniform
across 3m's whole shape space — every 3m shape is slower on Ice Lake at every depth
while planar gets faster — so it is not a shape choice. Nor is it the register
budget: 3m's `16x10` is over budget on Ice Lake (`live = 32!`), but the best
unflagged 3m shape there is slower still.

**And the mechanism does not survive either, which is the bigger casualty.** This
project carried "3m's 25% flop saving is real, and with L1-resident panels 3m is the
*fastest* of the three" as settled. At `kc = 16`, which *is* that regime, 3m leads
planar by **1.105** (`c64`) and **1.155** (`c32`) on Cascade Lake and trails at
**0.567** and **0.662** on Ice Lake. On Ice Lake 3m does not win at any depth, at
any shape, in either precision. What survives is only the accounting — 3 products
against 4, 3 planes against 2 — not the claim that the saving pays anywhere
nameable. The one visible candidate is load-port pressure (3m runs `f/l` 0.89–1.71
against planar's 1.78–3.75), and that is a hypothesis from one column, not a
measurement.

**Evidence.** `bench-results/worker6156-icelake/compare-tblis-skx/` (job
6753260); Cascade Lake baseline in `bench-results/phase3-sweep-*.csv`. Phase 5
report part 3; A44, A34.

**What would reopen it.** Not the sweep — that is done, and it closed the question
rather than opening it. Item 3 is **dead**: an unconditional rule is a pessimisation
on one of the two AVX-512 machines measured, and a per-microarchitecture rule would
have to be built on a mechanism that does not reproduce.

What *would* reopen it is a **third microarchitecture on which 3m wins at some
depth** — that would make the Cascade Lake behaviour a family trait rather than one
machine's, and it costs eight minutes of `examples/kernel_shapes` on any new machine,
no baselines needed. Run it as a matter of course when this project first touches new
hardware. Until then, treat 3m as the method that exists to make the comparison
honest rather than as a candidate default.

**Second lesson, and it is the reason this entry is long:** the answer was in
committed raw output for two days while `notebook/` described it as unmeasured and
a to-do list asked for node time to obtain it. That is A35's lesson recurring — check
what is already on disk before booking a machine.

---

# Measurement methodology

These are the rules the hard way. Every one of them was learned by publishing a
number that later turned out to be an artefact.

## `A, B, A'` bracketing establishes a session's noise floor (A31)

**Tried.** Bracketing a treatment between two measurements of the same thing, and
reading the repeat's disagreement as the session's precision.

**Expected / what happened.** The bracket **cannot tell a cold-start transient
from a noise floor.** The session's opening arm ran at single-core boost on a cold
package and nothing else did, so its repeat came back **3.4–4.7% slower
uniformly** — reported by the bracket as a ±4.7% floor, while every ratio measured
against that opening arm was inflated by the same amount. The second dtype pair,
measured entirely hot, drifts 1.1–1.9%. Where the true repeat precision was
**0.02%**, the bracket reported 4.4%.

Invisible on `ccqlin038`, because a shared workstation is never cold.

**Confidence.** `settled` — reproduced independently on three Zen2 nodes, always
on the session's opening arm.

**Evidence.** `bench-results/worker5040-zen2/phase4f/` (job 6745376),
`worker5175-zen2/` (6745978), `worker5137-zen2/` (6751550). A31.

**Rule.** **Run and discard a warm-up arm.** Every current script does
(`scripts/ab.sh`, `compare-bench.sh`, `phase4f-threads.sh`, `phase4g-small.sh`);
`phase3-bench.sh` predates it, which is one reason it is superseded.

**What would reopen it.** Nothing. But see A40 — the warm-up is a partial fix.

## A session's drift is a single number (A32)

**Tried.** Quoting one noise floor per session.

**Expected / what happened.** Drift is a **function of how far apart the two arms
are.** Same node, same core, same corpus: **0.02% at 69 s, 1.1–1.9% at ~1 h,
4.4% across the cold-start transient.** Quoting one floor for a whole session is
exactly what let a 4.4% artefact be mistaken for the precision of a repeat.

**Refined, and in the direction of "not universal".** On an exclusive SMT-off Ice
Lake node the session-span floor (2.5 h apart) is **no worse** than the near
floor: 0.998–1.001 against 0.995–1.003. Drift does not grow with separation
there at all. A31 and A32 look like properties of **boost headroom and
co-tenancy**, not of measurement in general.

**Confidence.** `settled` — three Zen2 nodes for the growth, one exclusive Ice
Lake node for the counterexample.

**Evidence.** `bench-results/worker5040-zen2/`;
`bench-results/worker6156-icelake/compare-tblis-skx/` (job 6753260, the two-floor
design is in `scripts/compare-bench.sh`). A32.

**Rule.** **Derive the floor in session, from repeats adjacent to the arms being
compared.** Identical-configuration arms already in a grid are free repeats —
find them before booking machine time.

**What would reopen it.** Nothing. Keep the warm-up arm even where it moves
nothing: it is cheap, and it is how you learn which case you are in.

## Per-case ratios are readable at ±6% (A39)

**Tried.** Quoting per-case ratios from threaded runs at the ±6% per-case floor
the reference machine reports for single-core work.

**Expected / what happened.** At **64 threads**, repeats of an **identical**
partition run **p10 0.885 / p90 1.107 with tails to 0.80–1.55.** Per-case ratios
are unreadable at that width. Only **per-family geomeans** (1–2%) are quotable.

**Confidence.** `settled` — confirmed on a second machine and width: Ice Lake at
32 threads reads p10 0.915 / p90 1.108 over 392 identical-partition repeats. It
is a property of threaded measurement here, not of one machine.

**Evidence.** `bench-results/worker5479-zen2/phase4f/`,
`worker6150-icelake/phase4f/` (jobs 6753208/6753209); the control comes out of the
same data at no cost — `scripts/partition-score-rule.py` prints it.

**Rule.** **Name the thread count with every floor.** A floor without one is not
a floor. `notebook/`'s measurement-rules section carries the table.

**What would reopen it.** A pooled implementation might tighten it, since some of
the spread is per-call spawn variance. Re-derive it after task 3 item 3 rather
than assuming either way.

## A discarded warm-up arm removes the opening-arm artefact (A40)

**Tried.** Treating the warm-up arm as the fix A31 asked for, and therefore
treating the first bracket after it as a floor.

**Expected / what happened.** **Partly.** Three of four `t1`/`t1b` brackets came
back inside 0.3%, against 0.954–0.973 on three nodes without a warm-up. The
exception is the **first dtype pair of the longer session**, at 0.972–0.977 —
whose *second* pair is clean. So the residual tracks **position in the session**,
not package temperature, which makes it A32 rather than A31 and means a hotter or
longer warm-up would not remove it.

**Confidence.** `measured once` — one pair of nodes, four brackets.

**Evidence.** `bench-results/worker5479-zen2/`, `worker6150-icelake/` (jobs
6753208/6753209). A40.

**Rule.** Keep the warm-up; **do not treat it as a floor.** Derive the floor from
a bracket adjacent to the arms being compared — which is what a
"columns the change cannot touch" control does for free.

**What would reopen it.** A session design where the first *measured* pair is
itself preceded by a full second pair. Cheap to try, not yet tried.

## This project's measurement rules cover the ways a comparison can mislead (A46)

**Tried.** Believing that A27 (co-tenancy), A31 (warm-up) and A32 (in-session
floor) between them cover it.

**Expected / what happened.** **They are all about time. Problem size is a
separate axis.** A one-shot `premise --size 8` run reported TBLIS 2.0's two builds
differing by up to **1.68x**, clustered on exactly the low-arithmetic-intensity
cases this project calls its headroom. At the sizes actually published — 64 MiB
sweeps, 200 MiB premise, i.e. **8–25x larger** — the two builds are identical to
within the floor (0.997–1.000 against a 0.998–1.003 floor), because the kernels
that differ only matter while the operands are small. A warm-up arm, an
in-session floor and per-arm occupancy would each have passed the bad measurement
through unchanged.

The 1.68x claim was published in this file and is **withdrawn**.

**Confidence.** `settled` for the retraction (a full A/B at both published sizes,
with a control), `single observation` for the 1.68x that caused it.

**Evidence.** `bench-results/worker6156-icelake/ab-tblis/` (`scripts/ab-tblis.sh`
via `scripts/rusty-tblis-ab.sbatch`). A45, A46.

**Rule.** **Measure at the size you publish at**, and treat a result taken at a
smaller size as being about that size.

**What would reopen it.** Nothing about these builds. The general lesson is
open-ended: the next confound will not be timing either.

## A sequential build-to-build A/B is good enough for a few-percent effect (A15)

**Tried.** Measuring a change by building twice and comparing the two runs.

**Expected / what happened.** It reported `c64` 3m at **0.973** where a paired
runtime A/B gives **1.020** — a wrong *sign* on a real effect.

**Confidence.** `settled` in the sense that the practice was abandoned and never
recovered a correct answer; `measured once` as a number.

**Evidence.** Part 1's re-measurement; A15.

**Rule.** **Put every new fast path behind an environment switch** so both arms
interleave in one process-restart A/B. That is why
`TENSORCONTRACT_{ORIENT,WRITEBACK,ROWBLOCK,PARTITION,POOL,BLOCKMODEL,KERNEL}`
exist, and adding one is part of adding a fast path.

**What would reopen it.** Nothing.

## A rule validated with the other levers pinned is validated (A20)

**Tried.** Deriving the row-block rule with the orientation pinned, and the
orientation rule with the shape pinned. Correct experimental design for isolating
each one.

**Expected / what happened.** The orientation rule was **12 better / 0 worse**
with the shape pinned, and still **cost 20% on three cases in the shipped
configuration** — because a 3% orientation error was blocking a 20% shape change.
Neither experiment could see it, by construction.

**Confidence.** `settled`.

**Evidence.** Parts 5 and 6, and the joint end-to-end result in part 6; A20.

**Rule.** Isolate to derive; then **always finish with an end-to-end A/B in the
configuration that actually ships.**

**What would reopen it.** Nothing.

## Concurrent arms, one per L3 domain, measure what a solo arm measures (A27)

**Tried.** Running many benchmark arms concurrently, one per L3 domain, to turn a
20-hour grid into a 1-hour one. Pre-registered as a hypothesis with its
accept/reject rule written down first.

**Expected / what happened.** **Confirmed on the corpus (+0.3%), refuted on the
memory-bound half (−3.2%, up to −10%).** Private per-CCX L3 is enough for
compute-bound work and irrelevant to bandwidth: `abcijk` at `k = 24` is
bandwidth-bound and 24 arms contend for the same memory controllers. It returned
20.4x on the grid it was accepted for.

**Confidence.** `settled` — the accept and the reject came from the same
pre-registered test.

**Evidence.** `bench-results/worker5040-zen2/` (placement validation, job
6745376); the sequential re-run of the traffic-changing arms is
`worker5175-zen2/` (job 6745978). `scripts/validate-placement.sh`,
`scripts/placement-verdict.py`. A27.

**Rule.** Use the placement for arms that **do not change memory traffic**; give
traffic-changing arms their own sequential run. `scripts/rusty-phase4.sbatch`
decides this with `placement-verdict.py` rather than by hand.

**What would reopen it.** A machine with per-domain memory controllers.

---

# Facts that were true when written and expired when the code moved

This class is the reason the whole file exists: "the corpus is fully regular" was
false for three phases and steered conclusions the whole time. Treat every claim
about the corpus, the register blocks or the defaults as *dated*.

## "The TCCG corpus is fully regular" (A4)

**Was true.** TCCG rounds every stride-1 extent up to a multiple of 24, which
divides every register block that was in use when it was written. `regA = 1.00`
everywhere, and that is still the right reading of the **TBLIS** comparison,
where the blocks do divide 24.

**Expired.** The moment Phase 3 shipped `f32`/`c32` blocks of `MR` **16, 32 and
48** — none of which divides 24. On the arm the orientation rule actually picks,
`reg_a < 1.0` on **42.9%** of the 392 case-dtype-methods, `reg_b` on **38.3%**,
the write-back fraction on **35.7%**. `reg_a` and the write-back fraction are
quantised — 0.667 / 0.889 / 0.963, i.e. one block in three, nine or twenty-seven
straddles — plus **45 case-dtype-methods at 0.0, entirely on the gather path**;
`reg_b` is not quantised that way and runs down to 0.501. **The corpus does
exercise the gather path here.** Irregularity is produced by the interaction of the layout
with `MR`, not by the tensors, which is why the row-block and orientation rules
exist at all.

What the corpus still cannot produce is **aperiodic** irregularity: its
straddling is periodic, so a static partition self-averages. That is what
`--stress ragged` is for.

**Confidence.** `settled` — computed for all 392 case-dtype-methods, no CPU cost,
reproducible with `./target/release/tcbench orient --csv`.

**Evidence.** `bench-results/phase4d/features.csv`; part 14's correction; A4. The
figure is **11.7%** on an AVX2 node, because `MR = 8` for `f64` *does* divide 24 —
so quote the ISA with it (part 16).

**What would reopen it.** A register block that divides 24 in every dtype would
make it true again. Nothing plans to ship one.

## Register blocks are a property of the instruction set (A34)

**Was assumed.** One measurement per ISA is enough — the premise of D19 and of
the `cfg_avx512_*` / `cfg_avx2_*` split, and of how dispatch still selects them.

**Expired / refuted.** Cascade Lake and Ice Lake — **same ISA, same 32
registers** — disagree by up to **13% on three of eight shapes**, each machine
preferring the other's loser by about 9%. `real` wants `NR+1` in both precisions
on Ice Lake, whose 48 KiB 12-way L1d accommodates an accumulator footprint
Cascade Lake's 32 KiB 8-way does not. **Shapes are per-microarchitecture.**
Probed L1 geometry separates these two with no CPUID table.

Consequence that is live: on any Ice Lake machine this engine currently runs a
`real` kernel **9.2% off its own optimum**, and dispatch has no way to know.

**Confidence.** `settled` for the refutation (two microarchitectures, same ISA);
the *fix* is unbuilt.

**Evidence.** `bench-results/worker6016-icelake/kernel-shapes.txt` (job 6746817)
against `bench-results/phase3-kernel-shapes.txt`. A34.

**What would reopen it.** Nothing — but it opens something: selecting register
blocks on probed L1 geometry rather than on the ISA bit. That is unbuilt and is
the most concrete unclaimed win in the kernel layer.

## The shipped register blocks are the ones the sweep selected (A35)

**Was assumed.** Eight shipped configurations, all chosen by measurement.

**Expired / refuted for one of eight.** `planar` `f32`/`c32` ships **`32x6`**
where Phase 3's own sweep output names **`32x5`** at 210.9 GF/s against 195.7 —
**7.8% faster** at the operating `kc`, and also ahead at `kc = 64`. The bolded
bytes-per-flop in the config's doc comment suggests it was chosen by model over
the measurement sitting next to it, which is the same error A16 records for
write-back regularity.

**Not fixed on purpose**, and no longer untestable: the menu was keyed by `MR`,
and `32x5` shares its `MR` with the shipped `32x6`, so no runtime switch could
reach it. Re-keying the menu by **position** (D43) fixed that — `32x5` is the
last planar `f32` entry and `TENSORCONTRACT_ROWBLOCK=idx=3` selects it end to
end. A kernel margin is not a corpus margin (`NR` also moves the `jr` loop count
and the packed-`B` sliver geometry), so it is an arm and not a retune.

**Confidence.** `measured once` for the 7.8% (a kernel sweep, one machine);
`structural` for the reachability fix.

**Evidence.** `bench-results/phase3-kernel-shapes.txt` — found inside committed
raw output months after the fact, at no machine cost, which is the return on
committing raw output. A35; D43; part 13.

**What would reopen it.** It is open: run the A/B.
`scripts/ab.sh bench-results/ab-rowblock-idx3 "TENSORCONTRACT_ROWBLOCK=idx=3"`,
~2 h on an exclusive machine.

## Per-call thread spawn is a second-order cost (A43)

**Was assumed.** Worth fixing after the partition — the partition was the
first-order term.

**Expired / refuted at small sizes.** **~20–36 µs per thread.** That is the
entire story below 1 MiB: a 0.22 ms `f32` contraction takes **2.2 ms on 64
threads** (a speedup of 0.10), and the optimal thread count walks 4 → 8 → 16 →
32 → 64 across 0.25 → 64 MiB. It is first-order for exactly the workload Phase 1
identified as the real headroom.

**Two regimes, and only one is dangerous.** Below ~1 MiB spawn dominates and
threading is a *loss*. Above it the curve saturates early (best at `t8`–`t32`) —
that is a **bandwidth ceiling**, a different mechanism, and benign. **A guard
fitted to the first must not be fitted to the second.**

**Confidence.** `measured once` — one Zen2 node, seven size rungs, speedups taken
against `t1` at the same size so the fixed cost is inside the measurement. **Worse
per case than the corpus geomean showed**: re-scored per point from the same data,
the 0.25 MiB worst case runs at **0.022** (45x), and at 1 MiB — where the corpus
geomean reads 1.10 and looks safe — **229 of 588 points are still slower than
serial**.

**Evidence.** `bench-results/worker5139-zen2/phase4g/` (job 6754849);
`scripts/phase4g-small.sh`. Part 16; A43; D46.

**Measured, and the answer is the pool.** Part 18 removed the cost and it is worth up
to **11.6x** at 0.25 MiB / 64 threads and **2.0–2.8x even at 16 MiB** — so it was not
confined below a megabyte, and part 16's "benign bandwidth ceiling" above 1 MiB was
substantially this. The guard is refuted in the form built (above, D52); the batched
API remains unmeasured.

**What would reopen it.** One thing, and it is a question rather than a task: **why
the win scales with size is unknown** (A53). A fixed ~37 µs/thread is solid and
confirms this entry's own 20–36 µs independently, but the pooled-vs-unpooled difference
also has a proportional component, so the per-thread figure overstates it. The
per-thread packed-`A` buffer was the named candidate and is **refuted**: `ap_len` is
identical at all four sizes, because `mc` never binds against `m` at these shapes — so
hoisting those allocations has no measured basis. Untested candidates: scheduler
placement of freshly created threads, whose penalty accrues over the thread's life;
and barrier skew, whose count grows with `N/NC` x `K/KC`. A microbenchmark separating
spawn from placement from barrier count would settle it, and needs no corpus and no
exclusive node.

## A baseline built from the right source at the right commit is the right baseline (A45)

**Was assumed.** Version plus commit identifies a baseline.

**Expired / refuted, on portability.** `BLIS_CONFIG_FAMILY=auto` silently fits
BLIS to the **build host**, so `../baselines/tblis-2.0-install` is **skx-only**
and **SIGILLs on any machine without AVX-512** — it killed a `rome` job two
seconds in (job 6753261; the wreckage is committed at
`bench-results/worker5479-zen2/failed-6753261-sigill/`).
`../baselines/tblis-2.0-x86_64-install` is the same source with
`BLIS_CONFIG_FAMILY=x86_64`: thirteen contexts, 3809 `bli_` symbols against the
skx build's one context and 1364. TBLIS **1.3.0** needs none of this care — it is
genuinely multi-config already.

**And the performance half is withdrawn.** The skx build also lacks the
skinny-GEMM (`sup`) kernels, which *looked* like a large defect and costs
**nothing** at 64 or 200 MiB. See A46 above.

**Confidence.** `settled` for the portability half (a SIGILL is not ambiguous),
`settled` for the withdrawal (full A/B with a control).

**Evidence.** job 6753261; `bench-results/worker6156-icelake/ab-tblis/`;
`scripts/rusty-compare.sbatch` picks the build by grepping `avx512f` out of
`/proc/cpuinfo` and records the choice in each run's `PROVENANCE.txt`. A45.

**Rule.** Record a baseline's **configuration**, not just its version and commit.
Verify a multi-ISA claim with `nm`, not `strings` — BLIS compiles its config
*name* table in whether or not the kernels are there. Do not infer a performance
consequence from a symbol count.

**What would reopen it.** Nothing. Keep both installs: the `auto` build is what
every committed number used, so prefer it wherever it runs.
