## Running a measurement session

How a cluster session is designed and driven, and the pre-registration parts 7–9 and 11 consume.

### Part 10: taking the two pending measurements to a cluster node

**Status: designed, tooled and pre-registered. Nothing is measured yet.** This
section is written *before* the run so that the node choice, the placement
hypothesis and its accept/reject rule are on the record and cannot be adjusted to
fit whatever comes back. Results go into parts 7, 8 and 8b, and a new part for
the AVX2 calibration.

#### The node choice *is* the experiment design

The two pending measurements want an exclusive machine, and `ccqlin038` is a
shared workstation. Rusty has one, but **no Cascade Lake node**, so the choice of
partition picks which question gets answered:

| feature | S:C:T | cores | ISA consequence |
|---|---|---|---|
| `rome` (Zen2) | 2:64:1 | 128 | **no AVX-512** — runs the AVX2 kernels |
| `icelake` | 2:32:1 | 64 | AVX-512, Intel, closest to the reference machine |
| `genoa` (Zen4) | 2:48:1 | 96 | AVX-512 (double-pumped), a third cache topology |

`T:1` in all three: **SMT is off on Rusty's CPU nodes**, so the hyperthread
sibling that forced two Phase 4 retractions is absent rather than merely handled.
That is checked on the node by `scripts/topology.py` rather than believed from
`sinfo`, and `phase4f-threads.sh` keeps its sibling guard regardless — a guard
that trivially passes costs nothing and stops being load-bearing only when
someone proves it is.

**Chosen: `rome`.** Three reasons, in order of weight:

1. **It converts a documented guess into a measurement.** The AVX2 register
   blocks are provisional and explicitly unmeasured (D26): chosen from a register
   budget and a uop model on a machine that cannot execute them competitively.
   `rome` is the largest partition on the cluster and AVX2 is what most users
   get, so this is the widest remaining gap between what this file measures and
   what a user experiences.
2. **Its topology is the one where concurrent placement can work.** Zen2 shares
   one L3 between four cores, so one arm per L3 domain gets a *private* L3 —
   better isolation than `ccqlin038` ever had, where eight cores share 25 MiB. On
   `icelake` the L3 is a whole 32-core socket, which leaves about two usable
   domains and no way to run the 40-job grid concurrently at all.
3. **It tests the analytical model harder than `icelake` would.** The point of
   D23 was that the legacy constants are `ccqlin038`'s cache sizes written down
   by hand. Zen2's are further away than Ice Lake's in the direction that
   matters: a 512 KiB L2 against a legacy `A`-block budget of 512 KiB, i.e. the
   constants ask for the entire L2 where the model would reserve ways for the
   streaming operand. If the model is going to win anywhere, it should win here,
   and if it does not, that is a finding about the model rather than about the
   node.

What this deliberately gives up, stated so it is not discovered later as a
surprise: `rome` cannot say anything about AVX-512 blocking on a second Intel
hierarchy, which is the `icelake` question and remains open. And the blocking
grid measured on `rome` sits on top of register blocks that are themselves
unmeasured until the `shapes` stage runs — so the calibration must be read
before the grid, and the grid's conclusions are conditional on the shapes the
engine actually shipped that day, not on the calibrated ones. Recalibrating and
re-running the grid is a second session, not this one.

#### Nothing measured there is comparable to anything above this line

Different machine, different cache hierarchy, different instruction set. Every
ratio must be computed *within* the session, and **the ±1.3% geomean / ±6%
per-case noise floor is `ccqlin038`'s and does not transfer.** The floor is
re-derived on the node from bracketing repeats the scripts already run:
`phase4f-threads.sh` now compares `t1` against its bracketing `t1b` and prints
that first, `validate-placement.sh` runs the same arm solo twice for the same
purpose, and the grid keeps its three `base` repeats. No effect gets quoted
before the floor it is measured against.

#### The placement problem, and the hypothesis

An exclusive 128-core node makes both obvious answers wrong. One arm at a time
leaves 127 cores idle, and 40 grid jobs at ~22 min each (on the *reference*
machine; AVX2 will be slower per core) is a day's work. Forty arms at once
corrupts the quantity being measured, because arms sharing an L3 or a memory
controller perturb each other and `NC` is *sized for L3*.

So the placement is treated as a hypothesis: **one measurement thread per L3
domain, every other core in that domain idle, SMT siblings idle, and concurrency
held below the domain count** — 24 of 32 by default on `rome` — because memory
bandwidth and the interconnect stay shared however threads are placed.

The accept/reject rule, fixed now:

* **Accept** if placed-vs-solo lies inside the solo-vs-solo floor on the
  geometric mean, and no case moves by more than the per-case floor.
* **Reject** otherwise, run the grid sequentially, and record the rejection.

Both arms are measured on the full corpus *and* on the `abcijk` family, which is
exactly the 18 cases that contract over `k = 24` — the memory-bound half, i.e.
where bandwidth contention should show up first, bandwidth being the one resource
no placement can privatise. A rejected placement is a result worth writing down:
it tells the next person on a different machine what to expect, and the mechanism
(a private L3 per CCX against a socket-wide L3 on Intel) predicts that the answer
differs by machine.

#### What was built for it, all of it zero-CPU

* `scripts/topology.py` — L3 domains, SMT siblings, NUMA and cache descriptors
  from the *allocation's* affinity mask, so it describes the job and not the
  node, plus the placement plan. Emitted as JSON so the runner and this file read
  the same facts.
* `scripts/run-arms.py` — runs independent arms one per L3 domain, and records
  `/proc/stat` occupancy for **every core in each arm's own L3 domain** across
  exactly that arm's window, plus the observed overlap with other arms. The
  reference scripts sample the pinned core and its sibling; this widens that to
  the domain, because on this placement the domain is the unit that has to be
  clean. That recording is what made the earlier retractions detectable.
* `scripts/validate-placement.sh` + `scripts/placement-spread.py` — the
  hypothesis test above, including the spread *between* slots, which prices
  position within the node. A uniform slowdown cancels in the ratios the grid is
  scored on; a position-dependent one does not, and would turn slot assignment
  into a per-arm bias.
* `phase4e-blocking.sh` now runs both regimes through one code path
  (`<cpu>` sequential, `auto` placed) from a single arm list, so the two cannot
  drift apart. `phase4f-threads.sh` derives its cpuset from the allocation
  instead of a hardcoded `0-7`, scales its thread counts to the core count, gives
  each arm exactly as many cores as it has threads (so L3 sharing is a known
  function of `nt` rather than the scheduler's choice), records occupancy over the
  whole allocation as an exclusivity check, and labels the cross-socket arm as the
  separate question it is.
* `scripts/node-session.sh` — stages the session so the ordering constraints are
  enforced rather than remembered: one stage at a time, only `prep` compiles, and
  a guard refuses to start a measurement while any `cargo`/`rustc`/`tcbench` of
  the user's is alive.
* `scripts/placement-verdict.py` — the accept/reject rule **as code**, so an
  unattended run can act on it and, more to the point, so the threshold cannot
  drift once the numbers are visible. Strict reading of the pre-registered rule:
  the per-case bound is the solo pair's *worst* case, not a percentile, because a
  percentile is a knob and a knob chosen after seeing the data is how a
  pre-registered rule stops being one. Every placed replicate must pass, not just
  the one on the reference core, since the grid assigns arms to slots arbitrarily.
* `scripts/rusty-phase4.sbatch` — the whole session unattended. Stage order is
  `prep, shapes, threads, validate, grid`: `shapes` moves ahead of the
  higher-priority `threads` because on an AVX2 node it is the highest-value single
  deliverable and costs twenty minutes, so putting it behind a four-hour stage
  risks the cheap irreplaceable thing for nothing. Stages are independent and a
  failure does not abort the rest.

**The rejection path returns data rather than nothing.** A rejected placement
means the full grid cannot run — sequential is ~23 h on this node and does not fit
a 12 h allocation — so the batch script runs a *scoped* grid sequentially inside
the wall time that is left, in the order `base model base2` then the `kc` family:
the model arm is a whole different derivation rather than a point in the grid and
needs its brackets to mean anything, and `kc` is the first-order parameter. Arms
that do not fit are named in `skipped-arms.txt` and in the log, because a bounded
run that does not say what it dropped reads as complete coverage. A scoped grid is
reported as scoped — offline rule scoring against a partial grid is not the asset
that scoring against a whole one is.

#### The prediction, made first

`scripts/thread-width.py` no longer hardcodes `P = 8` — it takes a list, and it
now replays `Plan::partition` (`PACK_WEIGHT` included) instead of only reporting
raw widths, so it predicts what the *rule* will ask for rather than what the shape
allows. Replayed over all 392 case-dtype-methods:

| P | no fill from `M` | rule goes 2-D | leaves threads idle | of which int-div waste | of which a genuine limit | needs `K` |
|---|---|---|---|---|---|---|
| **AVX-512 blocks (the reference machine's)** ||||||
| 8 | 16 | 12 | 4 | 0 | 4 | 0 |
| 16 | 26 | 22 | 12 | 4 | 8 | 0 |
| 32 | 32 | 30 | 16 | 4 | 12 | 0 |
| 64 | 33 | 32 | 18 | 12 | 6 | 0 |
| 128 | 72 | 68 | 49 | 46 | 3 | 0 |
| **AVX2 blocks (what `rome` will run)** ||||||
| 8 | 1 | 1 | 0 | 0 | 0 | 0 |
| 16 | 8 | 6 | 2 | 0 | 2 | 0 |
| 32 | 23 | 15 | 10 | 0 | 10 | 0 |
| 64 | 32 | 26 | 16 | 0 | 16 | 0 |
| 128 | 32 | 32 | 20 | 6 | 14 | 0 |

Four things fall out of it, none of which needed the machine:

1. **`K`-parallelism is never needed, at any thread count up to a whole 128-core
   node, in either ISA.** A21 was argued at 8 threads and the structural argument
   was claimed to generalise; this is that claim checked to 16x the width. A21 is
   now confirmed far outside the regime it was made in.
2. **The 2-D rule will be exercised 5x harder than it was designed against.** It
   fires on 12 of 392 at 8 threads and 32–68 at 128. `PACK_WEIGHT` was priced by
   replay at 2–32 threads and every weight in `[4, 64]` gave the same partition;
   that replay does not cover 64 or 128, so the `=m` / `=n` arms at the top thread
   count are the load-bearing part of the threading run, not the scaling curve.
3. **Most of the apparent shortfall at large `P` is arithmetic, not judgement.**
   `pn = min(p / pm, blocks)` is integer division, so a rule that wants `9 x 14`
   out of 128 threads gets 126 and idles two. Separating that from a genuine
   refusal matters, because only the second kind can show up as a visibly flat
   scaling curve: at `P = 128` on AVX-512 blocks, 46 of the 49 are integer-division
   waste of ≤2%, and just **three** are real.
4. **AVX2's smaller register blocks buy parallel width**, and by a lot: at 8
   threads 16 of 392 case-dtype-methods cannot fill the threads from `M` on
   AVX-512 blocks against **1** on AVX2, because `MR` is 3–6x smaller and the row
   panel count rises accordingly. This is a fresh instance of A25 — smaller
   register blocks are not purely a cost — in a dimension A25 did not consider,
   and it means the `rome` scaling curve should look *better* than the reference
   machine's would, for a reason that has nothing to do with the cores.

The genuinely partition-limited population on AVX2 is one family: `ij-ikl-ljk`
and `ij-kil-lkj`, which cap at `27x1` of 32 threads and `54x2` of 128 (84%).
Those are the curves that must flatten. If any *other* case flattens, that is
contention or a bug, not the partition — which is what makes this prediction
worth having.

Caveat on the table: the `frac of best` column in the full output ranks cases by
throughput measured on `ccqlin038`, used only as a compute-bound proxy for
sorting. It is not a cross-machine performance claim and is regenerated from the
node's own `t1` arm once that exists.

#### Result: the placement is clean on the corpus and *not* clean on the memory-bound half

Measured on `worker5040`, 24 concurrent arms one per L3 domain against a solo arm.
The honest comparison is against the **hot** solo arm (`t1b`, same core), because
the reused solo pair straddles the cold-start transient A31 describes:

| | geomean vs hot solo | range over slots |
|---|---|---|
| full corpus, socket 0 slots (12) | **1.0029** | 1.0017–1.0050 |
| full corpus, socket 1 slots (12) | 1.0215 | 1.0181–1.0255 |
| **memory-bound `abcijk` half, all 24** | **0.9684** | 0.9616–0.9769 |

Read in order:

1. **On the corpus as a whole the placement costs 0.3% — nothing.** Twenty-four
   arms running at once, each with a private 16 MiB L3 and three idle cores in its
   domain, measure what a solo arm measures. A27 is confirmed, and it bought a
   **23.7x** speedup: 533 minutes of arm-time in 22.5 minutes of wall clock, which
   is the difference between the grid being affordable and not.
2. **The 2.2% socket split is thermal, not spatial.** The `threads` stage had run
   for two hours entirely on socket 0, so socket 1 was cold when the placed round
   started; its slots read 2.2% *faster*. `placement-spread.py` was built expecting
   memory-system position to be the variable, and on this node it is temperature
   history instead. Confirmed by the memory-bound round twenty minutes later, when
   both sockets had been loaded and the split had shrunk to 0.6%.
3. **On the memory-bound half the placement is rejected, decisively.** 3.2%
   geomean and up to 10% per case, against a solo-pair floor of **0.02%** — the
   cleanest floor in this whole project, from two arms 69 seconds apart. Every one
   of the 24 slots fails. Bandwidth is the resource no placement can privatise, the
   `abcijk` family at `k = 24` is where that bites first, and it does.

**A gap between the rule as written and the rule as coded, which is mine.** Part 10
pre-registered that both the full corpus *and* the `abcijk` family would be
measured, but `rusty-phase4.sbatch` called `placement-verdict.py` with its default
prefix, so only the full-corpus arms were scored and the grid launched placed on an
ACCEPT that had not consulted the memory-bound arms. Scored after the fact with
`--prefix mbp --solo mbsolo`, they REJECT. The verdict script now takes `--solo` so
the rule can be applied to the arm set it was written for; the sequencing error had
already happened.

**What that does to the grid, and what it does not.** The grid is scored on
*ratios between arms*, and every arm ran under the identical 24-way placement, so a
uniform 3.2% penalty on memory-bound cases cancels in `arm / base`. The exposure is
not uniformity but **interaction**: an arm that changes memory traffic suffers
different contention, and that is exactly what the `nc` arms and the `model` arm do
— `model` raises `nc` about 4.7x here. So:

* `kc` and `mc` arms: the penalty is close to uniform and the ratios stand.
* `nc25`, `nc400`, `model`: **confounded at the same order as the effect**, and
  `model` is the single most interesting arm in the grid.

The fix is cheap and does not need the whole grid re-run: measure `base`, `nc25`,
`nc400`, `model`, `base2` sequentially on one core — 10 jobs, ~3.7 h — and compare
those against each other in the clean regime, using `base`/`base2` to tie them back
to the placed run. `scripts/rusty-phase4-seq.sbatch` does exactly that and nothing
else. **Until it has run, do not quote the `nc` or `model` arms from the placed
grid.**

| # | Assumption | Status |
|---|---|---|
| A27 | Concurrent arms placed one per L3 domain measure the same thing as a solo arm. | **Confirmed on the corpus (+0.3%), refuted on the memory-bound half (−3.2%, up to −10%).** Private per-CCX L3 is enough for compute-bound work and irrelevant to bandwidth: `abcijk` at `k = 24` is bandwidth-bound and 24 arms contend. Use the placement for arms that do not change memory traffic; measure the `nc` and `model` arms sequentially. |
| A32 | A session's drift is a single number. | **Refuted; it is a function of how far apart the two arms are.** Same node, same core, same corpus: 0.02% at 69 s apart, 1.1–1.9% at ~1 h, 4.4% across the cold-start transient. Quoting one floor for a whole session is what let a 4.4% cold-start artifact be mistaken for the precision of a repeat. |

#### What would invalidate the run

Checked before anything is believed, and each one recorded rather than assumed:
the allocation is genuinely exclusive (`scontrol show job`, once, plus the
per-core occupancy every arm now records for the whole allocation); no compile
overlapped a measurement (`node-session.sh` refuses); the bracketing repeats
agree within the floor derived in the same session; and no number is compared
across machines.

| # | Assumption | Status |
|---|---|---|
| A27 | Concurrent arms placed one per L3 domain measure the same thing as a solo arm. | **Hypothesis, with the test and the accept/reject rule pre-registered above.** Zen2's private-per-CCX L3 is what makes it plausible; shared memory bandwidth is what makes it doubtful, which is why the memory-bound `abcijk` half is measured separately. |
| A28 | The 2-D partition rule and `PACK_WEIGHT` behave at node scale as they do at 8 threads. | **Open, and the reason the threading run is worth more than a scaling curve.** The rule fires on 12 of 392 at 8 threads and 32–68 at 128; `PACK_WEIGHT`'s indifference was only ever replayed to 32. |
| A29 | Parallel width is a property of the contraction. | **Refuted at no CPU cost — it is a property of the contraction *and the ISA*.** Row panels scale as `M / MR`, so AVX2's smaller register blocks give the same corpus 3–6x more of them: 1 of 392 case-dtype-methods short of 8 threads against 16 on AVX-512. |
