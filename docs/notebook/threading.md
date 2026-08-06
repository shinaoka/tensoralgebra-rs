## Threading

Phase 4 item 4, end to end: the scheme, the 2-D partition, the domain-aware gate that is now the default, load imbalance, what TBLIS does, and the two experiments that decide the default.

### Part 8 (item 4): threading, implemented and measured

**Status when written: built, correct, and not yet measured.** It shipped **off**
(D22) so nothing else in this file changed and the item 2 grid still measured the
engine it was designed against.

> **Overtaken, 2026-08-04.** It is now measured on seven nodes (five Zen2, two Ice Lake). The measured
> subsections below are the ones to read; the forward-looking parts of this
> report were written before any of them. Threads are **still off by default**,
> now for a measured reason (D46) rather than for want of data, and the partition
> gate *is* on by default (D44).

#### The scheme

`M` is cut once into `p` contiguous strips of whole `MR` panels. Each thread
runs loops 3, 2 and 1 over its own strip with its own packed `A` block; the
packed `B` panel is shared, packed cooperatively (each thread takes a slice of
its `NR` slivers), and bracketed by two barriers per `(jc, pc)` iteration — one
so nobody is still reading the previous panel, one so the new one is complete.
Loops 5 and 4 are identical across threads, which is what makes the barrier
counts agree without tracking them.

Consequences, all deliberate (D21):

* **No reduction anywhere.** Every output element has one owning thread which
  accumulates over the full `K` in the original order.
* **Bitwise identical to serial at any thread count**, therefore. This is the
  strongest available correctness invariant and it is asserted directly, not
  approximated by a tolerance: a strip that dropped a row, one that
  double-counted, or a thread reading `B` across a barrier would each break it,
  and the last of those is exactly the kind of bug a tolerance check waves
  through.
* **Strips are whole `MR` panels**, so each thread's row blocks stay aligned
  with the block scatter. The write-back fast path, the row-block rule and the
  orientation rule are all untouched — threading does not re-open any of
  Phase 4.1.
* **The serial path is unchanged**: at `p == 1` the difference from the
  pre-threading driver is two `Option` checks per `(jc, pc)`, nowhere near the
  hot loops.

#### Correctness

`cargo test --workspace --release` green, and green again under
`TENSORCONTRACT_KERNEL=scalar`, at `TENSORCONTRACT_THREADS` of 1, 2, 4 and 8 —
i.e. the *entire* existing suite, oracle comparisons included, also runs through
the threaded driver, which was free coverage worth taking. Five new tests cover
bitwise agreement with serial across all four dtypes and all three methods, a
blocking that forces hundreds of barrier round-trips, a batch axis (whose
failure mode is a deadlock rather than a wrong answer, so it is worth isolating),
and thread counts far exceeding the panel count.

`Plan::strips` reports how many strips a plan will really use, and the tests
assert it — following the precedent of the orientation tests, which assert the
heuristic actually fires so that a test cannot quietly stop testing anything.
That assertion earned itself immediately: it caught that the strips run along
the **oriented** row direction, so a `1 x 33` output parallelises into two
strips along 33. Skinny-`M` is therefore not automatically serial; skinny in
*both* directions is.

#### `K`-parallelism is not needed, and the reason is structural

Worth recording so it is not re-opened: parallelising the `pc` loop — the one
axis that would require per-thread accumulators and a reduction — is **not
needed by anything in this corpus**, and the argument generalises beyond it.

Needing it means the output has fewer than `p` micro-tiles in total, i.e. an
output of order 1–3 thousand elements while `K` is large. But arithmetic
intensity at the matrix level is bounded by roughly `2MN / ((M + N) * bytes)`,
so small `M` *and* small `N` cap it from above. Compute-bound requires a large
output, and a large output has plenty of tiles: the two conditions are in
tension, which is why the intersection is empty here rather than merely
unpopulated.

Measured confirmation is in `scripts/thread-width.py` above: every case short of
width 8 in `M` has `ceil(N / NR) >= 8`. And if a user ever does bring such a
shape, the feature is *cheap* precisely where it is needed — the output being
tiny is what makes per-thread accumulator tiles L1-resident and the final
reduction negligible. So this is a demand-driven feature, not a Phase 4 item.

#### Measured, on `worker5040` (Zen2, 64 cores of one socket, AVX2)

**Everything in this subsection is within-session.** Nothing in it is comparable
to the single-core numbers elsewhere in this file. Each arm ran on exactly as many
cores as it had threads, taken in order so that `t4` is one 4-core L3 domain and
`t8` is two; occupancy over all 128 cores was recorded per arm and the 64–127
cores outside the cpuset were idle at **0.0% mean, 1.1% max** throughout, so the
node was genuinely exclusive rather than nominally so.

##### The noise floor, and a defect in the A/B/A' pattern itself

| pair | `t1` vs `t1b` | when both ran |
|---|---|---|
| `f64`/`c64` | **0.953–0.966** | `t1` first thing, `t1b` after ~50 min of 64-core load |
| `f32`/`c32` | **0.981–0.989** | both after the node was already hot |

The first pair's repeat is 3.4–4.7% slower than its opening arm, uniformly across
every case, dtype and method — a direction, not a spread, so it is drift and not
noise. The second pair, measured entirely on a hot node, drifts only 1.1–1.9%.
That difference is the diagnosis: **the opening arm of the session was measured at
single-core boost on a cold package and nothing else was.**

This is a real limitation of the `A, B, A'` pattern this project standardised on
(A15, and the header of `phase4-remeasure.sh`): bracketing detects drift, but when
the drift is a *cold-start* transient the bracket reports it as a floor of ±4.7%
and silently inflates every ratio measured against `A`. On `ccqlin038` the effect
was invisible because a shared workstation is never cold. **Future sessions on a
boost-happy machine need a warm-up arm that is run and discarded**, and until one
exists the honest floor for the `f64`/`c64` column below is asymmetric: its
scaling ratios are understated by roughly 4%.

A second floor, and the one that matters more, comes free from the partition arms:
on the 270 case-dtype-methods where `TENSORCONTRACT_PARTITION=m` and the rule
chose *bit-for-bit the same partition*, two adjacent hot arms differ by

| | geomean | median case | p95 case |
|---|---|---|---|
| `f64`/`c64` | 0.980 | 6% | **31%** |
| `f32`/`c32` | 0.994 | 6% | **26%** |

So at 64 threads the per-case median floor is ±6% — the same as `ccqlin038`'s
single-core figure — but the **tail explodes to ±26–31%**. Per-case claims at high
thread counts are close to worthless here; geomeans over the corpus are good to
about ±2%.

##### Scaling saturates at 16–32 threads and then declines

Per-case geometric mean against the same session's `t1`:

| threads | 2 | 4 | 8 | 16 | 32 | 64 | cores busy at 64 |
|---|---|---|---|---|---|---|---|
| `f64` | 1.90 | 3.41 | 4.72 | **5.76** | 5.68 | 5.37 | 36% |
| `c64` planar | 1.96 | 3.73 | 6.22 | 7.80 | **8.39** | 7.65 | 36% |
| `f32` | 1.90 | 3.51 | 5.53 | 6.36 | **6.56** | 6.01 | 29% |
| `c32` planar | 1.95 | 3.75 | 6.14 | 7.72 | **8.37** | 7.55 | 29% |

Four things follow, and the occupancy column is doing most of the work:

1. **Scaling is clean to 8 threads** (4.7–6.2x) and saturates by 16–32. Beyond the
   peak it *declines*: 64 cores are slower than 32 in every dtype.
2. **The cores are idle, not saturated.** Busy fraction falls 98% → 95% → 88% →
   73% → 49% → 36% as threads double. At `t64` roughly two thirds of the wall
   clock is not compute, which rules out memory bandwidth as the primary limit and
   points at the two structural limits part 8 listed in advance: threads spawned
   per `execute` call rather than pooled, and two barriers per `(jc, pc)`
   iteration now crossing 16 separate L3 domains. **Limit #2 was named before this
   run and is now the leading suspect, priced.**
3. **Complex scales better than real** at every thread count (7.65 against 5.37 at
   64 in 64-bit). That is the twice-arithmetic-intensity argument from Phase 1
   appearing somewhere it was never claimed — complex is less exposed to the
   memory system, so it holds up longer as cores are added. A *new* instance of a
   settled result, not a re-derivation of it.
4. **Per case at 64 threads: median 5.89x, zero cases above 32x, 193 of 294 below
   8x.** The best case reaches 17.8x. Nothing in this corpus scales well at 64
   threads.

##### The prediction, scored

Part 10 predicted, before the run, that the genuinely partition-limited family at
64 threads would be `ij-ikl-ljk` and `ij-kil-lkj`, and that if anything else
flattened it would be contention or a bug rather than the partition. Scored:

* **Right about those two.** They are the two worst cases in the whole corpus at
  `t64` — 2.13x and 2.29x — and the CSV's notes column confirms they ran `27x2`,
  the clamped partition the prediction named.
* **Wrong about the rest, and the miss is the more valuable half.** The next worst
  are the `abcijk` family at 2.43–2.47x running `64x1` — full parallel width, no
  partition limit at all. They are the memory-bound `k = 24` cases, and the reason
  they flatten turns out not to be contention either. See part 8b.

##### Ice Lake: the same code scales twice as well, and it identifies the mechanism

A second session (job 6746817, `worker6016`, Ice Lake-SP, `STAGES="shapes threads"`,
one 48 MiB L3 per 32-core socket) ran the same arms. Its `t1`/`t1b` floor is
**0.4–0.6%**. Against the same session's `t1`:

| threads | Zen2 `f64` | **Ice Lake `f64`** | Zen2 `c64` 3m | **Ice Lake `c64` 3m** |
|---|---|---|---|---|
| 8 | 4.72 | 6.17 | 6.23 | 6.96 |
| 16 | 5.76 | **9.67** | 7.65 | **12.26** |
| 32 | 5.68 | **10.59** | 8.06 | **15.33** |

At 32 threads Ice Lake reaches 48% of linear in `c64` 3m and 33% in `f64`, against
Zen2's 25% and 18% — **roughly twice the parallel efficiency from identical code**,
and still rising at 32 where Zen2 had already turned over.

And the partition sweep, on the `abcijk` family, forced partition against the rule:

| threads | `pn/rule` `f64` | `pn/rule` `c64` | `pm/rule` (control) |
|---|---|---|---|
| 4 | 0.988 | 0.944 | 1.003–1.005 |
| 8 | 0.962 | 0.843 | 1.001–1.003 |
| 16 | 0.983 | 0.806 | 1.002 |
| 32 | **1.014** | 0.934 | 0.987–0.988 |

**The 4.3x that a 1-D `N` split won on Zen2 is simply absent here.** So the effect
is not a property of the partition — it is a property of the *topology*, and both
findings now have one mechanism:

`NC` sizes the shared packed-`B` panel for **an L3**. On Ice Lake there is one L3 per
socket, so "shared" means what the design assumed and the panel is genuinely shared.
On Zen2, 64 threads span **sixteen** separate 16 MiB L3s, so a panel every thread
must read whole is effectively replicated across sixteen caches and re-streamed from
memory. Under `64x1` every thread needs all of `B`, which maximises that traffic;
under `1x64` each thread owns a narrow column group and touches only its own slice,
which fits its local L3. That explains the 4.3x *and* the halved scaling efficiency
with a single cause, which neither a barrier-count nor a false-sharing story does.

**This retargets the fix.** Removing `Plan::partition`'s `panels >= p` early return
would be wrong: on a one-L3-per-socket machine the early return is *correct* and `pn`
buys nothing. What the rule is missing is **how many L3 domains the thread set
spans** — a quantity the engine already probes for D23 (`cores_sharing`). Keep
`pm = p` when the threads share one L3; split `N` so each thread's `B` slice fits its
local L3 when they do not.

**Confirmed** (job 6751550, `worker5137`). The sweep prices the axis choice against
thread count on the `abcijk` family, and on Zen2 four cores share an L3, so thread
count fixes the domain count:

| machine | threads | L3 domains spanned | `pn/rule` `f64` | `pn/rule` `c64` |
|---|---|---|---|---|
| Zen2 | 4 | **1** | 1.031 | 0.981 |
| **Ice Lake** | **32** | **1** | **1.014** | 0.934 |
| Zen2 | 16 | 4 | 1.173 | 1.236 |
| Zen2 | 64 | 16 | **2.004** | 1.242 |

`pm/rule` is 0.959–1.003 across all of it, which is the control (the rule already
chooses `pm = p` for these cases).

**Replicated in all four dtypes on that node**, single-domain point null every time:

| threads | domains | `f64` | `c64` | `f32` | `c32` |
|---|---|---|---|---|---|
| 4 | **1** | 1.031 | 0.981 | 0.988 | 0.966 |
| 16 | 4 | 1.173 | 1.236 | 1.252 | 1.270 |
| 64 | 16 | 2.004 | 1.242 | **2.380** | 1.269 |

`f32` reaches **2.38x**. `worker5137`'s scaling curve also reproduces `worker5040`'s
(`t64`: `f64` 5.54 against 5.37, `c64` 7.69 against 7.65), and its `t1`/`t1b` floor
came back 0.954–0.973 — **A31's cold-start artefact reproducing independently on a
third node**, since `t1` was again the session's opening arm.

**Ice Lake's 32-thread row is what makes this an answer rather than a correlation.**
Within Zen2 alone, threads and domains move together and cannot be separated. But Ice
Lake runs *32* threads across *one* domain and shows nothing (1.014), while Zen2 runs
*4* threads across one domain and also shows nothing (1.031). Two single-domain points
an order of magnitude apart in thread count, both null; three multi-domain points with
a monotone effect. **The driver is the number of L3 domains the thread set spans, not
the thread count.**

So the fix is now specified rather than guessed: `Plan::partition` should split `N`
when the thread set spans more than one L3 domain, and keep `pm = p` when it does not.
The input is `cores_sharing` on the L3, which `CacheHierarchy` already probes for D23.

**Residual gap, nameable and cheap.** The perfect within-machine separation would place
a *fixed* thread count either packed into few domains or spread one-per-domain across
many — 16 threads on 4 whole CCXs against 16 threads on 16 CCXs. `phase4f-threads.sh`
packs by construction (`cpuset_for` takes the first `nt` cores in domain order), so
that arm does not exist yet; it wants a `--spread` cpuset mode, which is a few lines.
The cross-machine control above makes it confirmatory rather than load-bearing.

##### The default: revised by the second machine

Threading is off by default (D22) purely because it was unmeasured. It is now
measured on two machines, and **the first machine alone would have given the wrong
recommendation.**

On Zen2 the honest reading was "off": scaling turned over at 16–32 threads and
reached 5.7–8.1x on 64 cores. On Ice Lake the same code reaches **10.6x (`f64`) and
15.3x (`c64` 3m) on 32 cores and is still climbing**, with a 0.4–0.6% floor. Those
are numbers worth having on by default.

So the recommendation is **conditional, and the condition is the topology**: enable
it where the threads share an L3 (one L3 per socket, i.e. Intel here), and treat
chiplet machines as the limited case until the partition is domain-aware. That is
not a satisfying default to ship — `TENSORCONTRACT_THREADS` still defaults to 1 —
but it is the shape of the answer, and the domain-aware partition is a small change
to `Plan::partition` over a quantity the engine already probes.

Two limits remain and are independent of topology: threads are spawned per
`execute` call rather than pooled (first-order for the small repeated contractions
Phase 1 named as the real headroom), and Zen2's occupancy of 29–36% at 64 threads
says most of the wall clock there is not compute. Turning the default on is the
user's decision per D22 and is deliberately not taken in the same commit as the
measurement.

| # | Assumption | Status |
|---|---|---|
| A21 | Some compute-bound contractions will need `K`-parallelism, hence per-thread accumulators and a reduction. | **Refuted for this corpus, and argued structurally.** 20 of 392 case-dtype-methods cannot fill 8 threads from `M`, all 20 can from `M x N`, and needing `K` requires fewer than `p` micro-tiles in the whole output — which bounds arithmetic intensity at `~2MN/((M+N)*bytes)` and so bounds the case away from compute-bound. Build the 2-D partition; leave `K` unbuilt until a real shape demands it. *Confirmed again in part 10 at every thread count to 128 in both instruction sets, i.e. 16x the width it was argued at.* |
| A31 | `A, B, A'` bracketing is enough to establish a session's noise floor. | **Refuted on a machine with boost headroom.** The session's opening arm ran at single-core boost on a cold package and nothing else did, so its repeat came back 3.4–4.7% slower *uniformly* — reported by the bracket as a ±4.7% floor, while every ratio measured against that opening arm was inflated by the same amount. The second dtype pair, measured entirely hot, drifts 1.1–1.9%. Invisible on `ccqlin038` because a shared workstation is never cold. **Run and discard a warm-up arm.** |

### Part 8b (item 4): the partition becomes 2-D

**Status when written: built, correct, unmeasured.** Measured in part 12; the
partition rule it introduced is now the default (D44), while the *thread count*
still is not (D46).

Part 8 named the 1-D partition's first limit and sized it before the fix was
built: 16 of 392 case-dtype-methods cannot fill 8 threads from `M` alone, over
four cases (`aqrs-pa-pqrs` at 2–6 row panels; `ij-ikl-ljk`, `ij-kil-lkj` and
`ijk-il-jlk` at 5–7), and every one of them is 26–23435 `NR` blocks wide. (That
count is at the *default* register block, which is what `tcbench orient` emits;
the shipped row-block rule changes `MR` on some cases, so a case at the boundary
can move either way.) The output is now cut into a `pm x pn` grid (D27–D29).
Three things are worth carrying forward.

**The two axes are cut in different places, and that is the design.** Row strips
are cut once, outside everything; column groups are cut inside loop 5, per `NC`
block. So every thread walks the same `(h, jc, pc)` sequence and a thread's cell
changes only how much work happens inside an iteration, never how many
iterations there are. That preserves both properties the 1-D scheme rested on —
one L3-sized shared `B` panel, and barrier counts that agree because loops 5 and
4 are identical across threads — while making the `N` axis available. A thread
whose column group is empty in a tail `jc` block still takes both of that
block's barriers and *then* skips loop 3.

**A negative result on indexing, and the most valuable thing this produced.**
Indexing the shared `B` panel by *absolute sliver*, which is what the 1-D scheme
did, is wrong the moment `N` is split: a group's sliver range **moves** between
`jc` blocks (a tail block has fewer slivers to divide) and its stride changes
with `pc_len`, so one group's next block lands on another group's current one —
and by design nothing orders them. Each group now owns a fixed slice, strided by
the worst-case sliver size so groups on different `pc` blocks cannot overlap
either. It surfaced only under test-level oversubscription and never in a
standalone repro; the barrier-stress case, which has one sliver per `jc` block
and therefore ranges that never move, passed throughout. **Worth remembering as
the shape of bug this parallelisation produces: not a missing barrier, but a
buffer whose ownership map is not constant.**

**The one modelled decision, priced without the machine.** `PACK_WEIGHT = 8` was
replayed offline over all 392 case-dtype-methods at five thread counts, and every
weight in `[4, 64]` gives an identical partition everywhere; the only two cases
that move at all choose between candidates 1.4% apart in modelled cost. At 8
threads the rule spills most of the narrow cases onto the column axis and
deliberately leaves `ij-ikl-ljk` / `ij-kil-lkj` 1-D **on 7 of 8 threads by
choice** — 37 column blocks split eight ways is five per thread against a whole
packed `A` block each. A hand estimate puts 1-D ahead by ~2.6%, which is an
estimate, which is why `TENSORCONTRACT_PARTITION` exists and why
`scripts/phase4f-threads.sh` now runs `=m` and `=n` arms at 8 threads beside the
scaling curve. Verified behaviourally at merge: `aqrs-pa-pqrs` runs `2x4` in
`f64` and `1x8` in `c64`, a wide case stays `8x1`, and the sweep CSV records
`t<threads>/<pm>x<pn>` so the partition is recoverable from the data.

#### Measured, on `worker5040` at 64 threads: the 2-D split is right and the rule's first line is wrong

Two arms, both against the rule at the same thread count, on the cases where they
actually change the partition (everything else is a control group and is what the
±2% geomean / ±6% median-case floor in part 8 was derived from).

**`TENSORCONTRACT_PARTITION=m` — forcing the old 1-D `M` split — loses everywhere
the 2-D rule fires:**

| dtype | cases changed | geomean |
|---|---|---|
| `f64` | 12 | **0.607** |
| `f32` | 12 | **0.607** |
| `c64` | 6 | **0.297** |
| `c32` | 12 | **0.521** |

So the 2-D extension is worth **1.6x to 3.4x** on exactly the population it was
built for, in all four dtypes, far outside any floor. D27–D29 are vindicated, and
part 8b's hand estimate that leaving two cases 1-D on 7 of 8 threads was worth
~2.6% turns out to have been the wrong thing to worry about — at 64 threads the
axis choice is worth factors, not percent.

**`TENSORCONTRACT_PARTITION=n` — forcing 1-D `N` — beats the rule in the real
dtypes and loses badly in the complex ones:**

| dtype | cases changed | geomean | max |
|---|---|---|---|
| `f64` | 141 | **1.055** | 4.32 |
| `f32` | 144 | **1.100** | 3.14 |
| `c64` | 147 | 0.678 | 1.82 |
| `c32` | 141 | 0.748 | 2.34 |

And the real-dtype gain is not spread thinly — it is concentrated on the
memory-bound `abcijk` family (18 cases, `k = 24`), which gains a **geomean 2.25–2.26x
and up to 4.32x**:

```
4.32x  abcijk-jkmc-miab  f64   rule t64/64x1 -> t64/1x64
4.23x  abcijk-ijmc-mkab  f64   rule t64/64x1 -> t64/1x64
```

**The diagnosis is sharper than A28 guessed, and it is not `PACK_WEIGHT`.** These
cases have 768 row panels against 64 threads, so `Plan::partition` never reaches
the cost model at all — it returns from the first line:

```rust
if panels >= p { return (p, 1); }
```

That early return encodes "if `M` alone can fill the threads, `M` is the right
axis". At 8 threads on a socket-wide L3 that was true and cheap. At 64 threads on
Zen2's sixteen 4-core L3 domains it costs **up to 4.3x on 18 of 49 corpus cases**,
and it is the same population — the `k = 24` memory-bound family — that part 8's
scaling curve showed flattening at 2.43x for no visible reason. The two findings
are one finding.

**`PACK_WEIGHT`'s direction is right, which is why the fix is not "always consult
the cost model".** The complex methods *lose* 25–32% from the same `1x64` switch,
and that is exactly what the packing term predicts: a complex packed `A` carries
two to four reals per element, so duplicating it across 64 column groups costs more
than the real path pays. So the rule is right for complex and wrong for real on
this family, and a corrected rule needs the early return removed *and* a term that
captures whatever makes the real path prefer `N` here — which this run has not
identified. Two candidates, both cheap to test with the switch that already exists:
write-back locality (under `64x1` sixty-four threads write interleaved row strips
of `D`, and false sharing across separate L3s is expensive on this topology, where
under `1x64` each thread owns a contiguous column group), and the cost of the two
barriers per `(jc, pc)` iteration now spanning 16 L3 domains. **Do not change the
rule until one of those is measured** — this is the third time in Phase 4 that a
plausible mechanism for a real effect turned out to be the wrong one.

| # | Assumption | Status |
|---|---|---|
| A36 | The 4.3x that a 1-D `N` partition wins on the memory-bound family is a property of the partition rule. | **Refuted by a second topology.** Absent on Ice Lake (`pn/rule` 1.014 at 32 threads against Zen2's 2.26x geomean), where one L3 serves the whole socket. It is a property of how many L3 domains the shared packed-`B` panel is spread across, which also explains Ice Lake scaling twice as well on identical code. So the fix is a domain-aware partition, **not** removing the `panels >= p` early return — on a one-L3-per-socket machine that early return is correct. |
| A28 | The 2-D partition rule and `PACK_WEIGHT` behave at node scale as they do at 8 threads. | **Refuted, but not where predicted.** `PACK_WEIGHT` is fine — it correctly keeps the complex methods off the `N` axis, which costs them 25–32% when forced. What fails is the `panels >= p` early return that precedes it: worth up to 4.3x on the 18 memory-bound `abcijk` cases in the real dtypes at 64 threads. The 2-D machinery itself is vindicated at 1.6–3.4x on the cases it fires on. |

Correctness is unchanged in kind and stronger in coverage: bitwise identity with
serial at every thread count *and* every partition, plus a `Split` expectation on
every threaded case pinning whether it is meant to be 1-D, 2-D, a genuine grid or
clamped to serial — written in units of `MR`/`NR`, since a shape two row panels
deep in `f32` is fourteen with the portable kernels. Green in release and debug,
and across all ten combinations of the three features merged today.

### Part 12: the partition becomes L3-domain-aware

**Status: built, measured, and it met its pre-registered prediction. Still off by
default — that is D22 and it is the user's call.** The prediction was made from
another node's committed grid before this one was booked; the result is at the end
of this section.

Part 8b diagnosed `Plan::partition`'s first line as costing up to 4.3x at 64
threads, and part 8 then refuted the obvious fix: removing the early return is
wrong, because on a machine with one L3 per socket the early return is *correct*
(Ice Lake, 32 threads on one domain, `pn/rule` 1.014). A36 is settled on three
nodes and in all four dtypes — the driver is the number of L3 domains the thread
set spans, not the thread count. This part turns that into code.

#### The change

`CacheHierarchy::l3_domains(threads)` is new and is three lines: `threads`
divided by the cores that share an L3, which the probe already reports for D23.
`tcbench info` prints it as a function of thread count, because a session that
says "16 domains" without saying at what width has said nothing.

`Plan::partition`'s early return is now **gated, not removed** (D41). In the
regime where the row axis alone fills the threads, the column axis takes them
instead when all three of these hold:

| condition | why it is there |
|---|---|
| `domains > 1` | The mechanism. `NC` sizes the shared packed-`B` panel for *an* L3; under `p x 1` every thread reads the whole panel, so it is replicated across every domain the thread set covers. Two single-domain points an order of magnitude apart in thread count (Zen2 t4, Ice Lake t32) are both null. |
| `blocks >= p` | The column axis must be able to fill the threads by itself, or the swap buys locality by giving up cores. The corpus's narrow half loses 2–5x on a forced `1 x p` for exactly this reason, and it is the largest population in the `n` arm. |
| `k <= 64` (`BANDWIDTH_BOUND_K`) | The penalty being dodged is **bandwidth**, so it cannot dominate a compute-bound case. `k` is this corpus's knob for that, and the guard confines the rule to the population the evidence covers — see the negative result below. |

It is a **binary** choice between `p x 1` and `1 x min(p, blocks)`, which are the
only two arms any session has measured, and it is reached only from the early
return: the cost model below it is untouched, so the narrow-`M` regime the 2-D
partition was built for behaves exactly as it did.

`TENSORCONTRACT_PARTITION` gains `domain` (and `legacy`, the default, spelled
out). It is a *rule*, not a pin, so unlike `m`/`n`/`<pm>x<pn>` the test suite
keeps asserting the rule's invariants under it — which is how "bitwise identical
to serial at every thread count **and every partition**" now covers the new path
as well. The correctness suite exercises the swapped branch on 60-odd shapes and
is green in every combination of `{default, domain} x {avx512, scalar} x
{1, 4, forced-16 domains}`.

#### The prediction, made before the node is booked

`scripts/partition-score-rule.py` replays the gate over a committed grid and
scores it against the arms that grid already measured. Both arms exist for every
case: the gate only ever selects the rule's own partition or the forced `1 x p`
one, and part 8b measured both on the whole corpus. So this is not a model of the
answer, it *is* the answer, up to the session-to-session drift of running it
again.

On `worker5137` (Zen2, 4 cores per 16 MiB L3), whole corpus at 64 threads — 16
domains — the gate moves **144 of 392 case-dtype-methods**, all of them the
`abcijk` family, and nothing else:

| dtype | n | predicted arm/rule |
|---|---|---|
| `f64` | 18 | **1.947** |
| `f32` | 18 | **2.647** |
| `c64` | 54 | 1.172 |
| `c32` | 54 | 1.266 |
| all changed | 144 | **1.423** |

which is **1.138 over the whole 392** with the unchanged cases counted at 1.000,
or per dtype 1.277 / 1.430 / 1.060 / 1.091. At 16 threads (4 domains) it moves
the same population for 1.242. At **4 threads — one domain — it changes nothing**,
and on **Ice Lake at any thread count up to the socket it changes nothing at
all**: the gate is bit-identical to the shipped rule there, which makes that
machine a free null control rather than a second experiment.

Put beside part 8's scaling curve, that is the difference between 5.97x and
~7.6x on 64 Zen2 cores in `f64`, and it closes most of the gap between Zen2's
parallel efficiency and Ice Lake's on identical code — which is what the
mechanism predicted it should do.

#### The negative result: a traffic term cannot be calibrated, and here is why not

The first design was the principled one — add a cross-domain `B`-replication term
to the cost model and let it decide — and it does not work. Four attempts, each
scored offline against the grid:

* A term in `min(pm, domains)` large enough to move the `k = 24` family moves
  **263 of 392 case-dtype-methods onto intermediate grids** like `4 x 16` and
  `8 x 8` that no session has ever run. The term saturates at `pm >= domains`, so
  its optimum is an intermediate, not `1 x p`.
* Restricted to the two extremes, the switch-over weight is **33.6** for `f64`
  `abcijk`, **56.7** for `f64` `ijkl`, **50.6** for `c64` `abcijk` and **56.2**
  for `c64` `ijkl`. The measurement wants the first three to switch and the
  fourth not to. There is no weight that does that: the two `ijkl` thresholds are
  1% apart and want opposite answers.
* Making the term count *reals* rather than elements — so 1m's four-reals-per-`A`
  and 3m's three-plane traffic enter, and the `D == 1` behaviour is provably
  unchanged for every method — moves the pair to 66.0 and 67.6 and leaves them
  still inverted.
* Adding the `A`-side traffic term that makes the model symmetric (`min(pn, d)`
  domains read each row strip) ranks `ijkl` as *more* inclined to the column axis
  than `abcijk`, which is backwards by a factor of two in the measurement.

So the mechanism that explains the domain count does **not** explain which wide
families want the swap, and four plausible accounts of it are refuted. The
threshold that does separate them is `k`, by a factor of 40 — and a threshold on
a bandwidth-boundness proxy is at least the right *kind* of quantity for a
bandwidth effect. It is stated as a confinement of scope, not as a mechanism.

**What it costs to be that conservative, stated rather than hidden.** The same
session says a forced column split is worth 1.168–1.801 on the wide
*compute-bound* families (`ijkl-*`, `ij-ik-kj`) in the real dtypes, and 0.715–0.876
in the complex ones. Flipping those too is a **0.916 geomean over the 96
case-dtype-methods involved** — a loss, because the complex populations are three
times the real ones — so leaving them alone is the right call on this evidence
even before the one-machine, one-session caveat. The `n` arm already prices that
variant in every future run; nothing new is needed to revisit it.

#### A methodological result: per-case ratios at 64 threads are not readable

The `m` arm at 64 threads produces the **same partition as the rule** on 366 of
392 case-dtype-methods, so those ratios are pure repeats and measure this
session's per-case precision directly. They come back geomean 0.997 with **p10
0.885 and p90 1.107, and tails to 0.80 and 1.55** — on the `abcijk` family in
`f64`, 0.801 to 1.545.

Per-family geomeans are good to 1–2% and everything quoted above is one. **No
per-case number at 64 threads on this corpus means anything**, which is why the
gate was not fitted to one, and why `partition-score-rule.py` prints that control
first and labels the per-case column as spread rather than as a result. This is
the same lesson as A31 and A32 in a third form: the floor is a property of the
measurement's shape, and it has to be re-derived in-session every time.

#### The confirmation run, and what would falsify it

```bash
PARTITION_SWEEP=1 STAGES=threads sbatch -C rome scripts/rusty-phase4.sbatch
```

~1.5–2 h. `phase4f-threads.sh` now runs a `domain` arm beside `m` and `n`, over
the whole corpus at the top thread count and across the thread-count sweep on the
memory-bound family, and prints the prediction from this section next to the
result. Pre-registered, so it cannot be reinterpreted afterwards:

1. **`dom/rule` on the `abcijk` family must reproduce `n/rule`** to within the
   session's own floor, at 64 threads and at 16, in all four dtypes. It is the
   same partition, so anything else is contention.
2. **`dom/rule` on every other case must be 1.000** — the same code path runs.
   This is the column the change cannot touch, and it is the check that caught a
   contaminated A/B in part 7. If it moves, nothing else in the run counts.
3. **At 4 threads (one domain) `dom` must equal `rule` exactly.**
4. `t1` against `t1b` should now come back near 1.000 rather than 0.95–0.97,
   because a discarded warm-up arm runs first (A31). If it does not, the warm-up
   is at the wrong thread count and the floor is still a cold-start artefact.

A run on `-C icelake` is a stronger null than a repeat here: the gate must be
bit-identical to the rule at every thread count up to the socket.

#### A36's residual gap, now closable in the same run

Every separation of domain count from thread count so far has been
*cross-machine*: Zen2 at 4 threads on one domain and Ice Lake at 32 threads on
one domain are both null, an order of magnitude apart in width. That is a strong
control, but the within-machine version was missing because `phase4f-threads.sh`
packs its cpusets by construction, which ties the domain count to the thread
count and makes the two inseparable inside one node.

`SPREAD=1` supplies it, in about eight family-restricted arms per dtype pair. It
holds the thread count fixed and varies only the packing: on this node 16 threads
on cpus `0-15` (**4 domains**) against 16 threads on `0,4,8,…,60` (**16
domains**), verified against the committed `topology.json`. Both placements run
`rule`, `m`, `n` and `domain`, so the comparison is two `n/rule` ratios at one
thread count with the control beside them.

The engine cannot infer the spread placement — `l3_domains` assumes compact (A37)
and would under-count — so those arms declare the true span through
`TENSORCONTRACT_L3_DOMAINS`, which is what that override was added for. The
prediction, if A36's mechanism is right: `n/rule` near 1.17 packed (4 domains,
the measured `t16` figure) and near 2.0 spread (16 domains, the measured `t64`
figure), at the *same* 16 threads. If instead both read ~1.17, the driver is
thread count after all and D41 is wrong in a way three machines have not shown.

```bash
SPREAD=1 PARTITION_SWEEP=1 STAGES=threads sbatch -C rome scripts/rusty-phase4.sbatch
```

#### Result: all four checks pass, and the prediction was right to 1%

Run 2026-08-04, jobs 6753208 (`worker5479`, rome, 64 cores of one socket, 4 per
16 MiB L3) and 6753209 (`worker6150`, Ice Lake-SP, 2x32 cores, **one 48 MiB L3
per socket**, so one domain over the 32-core cpuset). 108 arms, **none flagged
non-exclusive**, 141 min and 70 min.

**Check 2 first, because it decides whether the rest counts.** The 248
case-dtype-methods the gate cannot touch read **1.0116** — a uniform drift
between the rule arm and the `dom` arm an hour later, which is A32's
drift-with-separation and not contamination. Every number below is quoted raw and
deflated by it.

**Zen2, 64 threads, 16 domains**, against the prediction made from `worker5137`'s
grid before this node existed:

| | predicted | measured | drift-corrected |
|---|---|---|---|
| case-dtype-methods moved | 144 of 392 | **144 of 392** | — |
| geomean on those | 1.423 | 1.449 | **1.433** |
| whole corpus | 1.138 | 1.146 | **1.133** |

Per dtype, measured: `f64` 2.207, `f32` 2.379, `c64` 1.217, `c32` 1.272. Check 1
(`dom` against the `n` arm, which is the same partition) is 1.019 raw and **1.007
corrected** — inside per-case noise, as it must be.

**Check 3, Ice Lake: the gate moved 0 of 392.** Not approximately a no-op — the
identical partition on every case at every thread count up to the socket. A
one-L3-per-socket machine is untouched by this change, which was the whole reason
for gating the early return rather than deleting it.

**The domain sweep is monotone in domains and flat in threads**, `abcijk` family,
gate against rule:

| threads | domains | `f64` | `c64` | `f32` | `c32` |
|---|---|---|---|---|---|
| 2 | **1** | 0.999 | 1.001 | 1.000 | 1.001 |
| 4 | **1** | 0.999 | 1.003 | 1.001 | 1.001 |
| 8 | 2 | 1.145 | 1.025 | 1.129 | 1.043 |
| 16 | 4 | 1.181 | 1.179 | 1.221 | 1.272 |
| 32 | 8 | 1.645 | 1.267 | 1.806 | 1.226 |
| 64 | 16 | 2.100 | 1.237 | 2.300 | 1.272 |

**And the spread arms close A36's residual gap within one machine.** Sixteen
threads throughout, same node, same cases, only the packing different:

| placement | domains | `f64` | `c64` | `f32` | `c32` |
|---|---|---|---|---|---|
| packed | 4 | 1.189 | 1.218 | 1.219 | 1.293 |
| **spread** | **16** | **2.736** | **1.487** | **3.045** | **1.744** |

Predicted in this section: "near 1.17 packed and near 2.0 spread". Packed landed
at 1.19. Spread landed *higher* than the `t64` figure, and the mechanism says it
should — 16 threads on 16 domains is one thread per 16 MiB L3, so the replicated
`B` panel under `16 x 1` is at its worst against the private cache each thread
could have had instead. Every prior separation of domain count from thread count
was cross-machine; this one holds the thread count fixed on one node.

**What it does to scaling**, corpus geomean over each session's own `t1`:

| machine | dtype | t8 | t16 | t32 | t64 | **t_top + gate** |
|---|---|---|---|---|---|---|
| Zen2 | `f64` | 4.62 | 5.77 | 5.76 | 5.81 | **7.54** |
| Zen2 | `f32` | 5.21 | 6.14 | 5.98 | 5.51 | **7.88** |
| Zen2 | `c64` | 6.11 | 7.83 | 8.15 | 7.65 | **8.24** |
| Zen2 | `c32` | 6.01 | 7.63 | 8.04 | 7.28 | **8.06** |
| Ice Lake | `f64` | 6.13 | 9.61 | **10.54** | — | 10.40 |
| Ice Lake | `c64` | 6.82 | 11.58 | **14.23** | — | 14.31 |

The gate turns Zen2's *decline* past 16 threads into a rise, and the Ice Lake
column moves only by session drift, since the partitions there were identical.

**Check 4, the warm-up arm: it works, but not completely, and the residual names
its own mechanism.** `t1` against `t1b`:

| node | `f64c64` | `f32c32` |
|---|---|---|
| Ice Lake | 0.997–0.999 | 0.998 |
| Zen2 | **0.972–0.977** | 1.001–1.003 |

Three of four brackets are now inside 0.3%, against 0.954–0.973 on three nodes
without a warm-up. The exception is the *first* dtype pair of the *longer*
session, and its second pair is clean — so the residual tracks **position in the
session**, not package temperature, which makes it A32 rather than A31 and means
a hotter warm-up would not fix it. Derive the floor from a bracket adjacent to the
arms being compared, which is exactly what check 2 does above.

*Decisions introduced here: D41, D42, D44 — stated in [Design decisions](#design-decisions).*

| # | Assumption | Status |
|---|---|---|
| A36 | The 4.3x that a 1-D `N` partition wins on the memory-bound family is a property of the partition rule. | **Refuted; now implemented as a property of the topology.** See part 8. D41 is the code. |
| A37 | `l3_domains(p)` may assume **compact placement**: `p` threads occupy `p` consecutive physical cores, filling one L3 domain before starting the next. | **Assumed, and true of every measurement in this file.** `phase4f-threads.sh` builds its cpusets that way by construction (`cpuset_for` takes the first `nt` cores in domain order) and a whole-node run leaves nothing to spread over. A *scattered* placement spans more domains than this counts, and the error is in the safe direction: it under-counts, so the rule falls back to the behaviour every committed number was measured with. `TENSORCONTRACT_L3_DOMAINS` overrides it, and `SPREAD=1` in `phase4f-threads.sh` is the arm that uses it. |
| A38 | A cross-domain traffic term in the partition cost model can be calibrated to select the column axis where measurement wants it. | **Refuted, four ways** — see the negative result above. The weights that separate the two `abcijk`/`ijkl` pairs are 1% apart and want opposite answers, in element units and in real units alike, and the symmetric two-sided model ranks the two families backwards. Do not re-derive it without a new mechanism. |
| A39 | Per-case ratios at 64 threads are readable at the ±6% the reference machine reports. | **Refuted on this corpus and this width.** Repeats of an identical partition run p10 0.885 / p90 1.107 with tails to 0.80–1.55. Only per-family geomeans (1–2%) are quotable at this thread count, and the control that shows it comes out of the same data at no cost. *Confirmed on two further nodes: Ice Lake at 32 threads reads p10 0.915 / p90 1.108 over 392 identical-partition repeats, so this is a property of threaded measurement here and not of one machine.* |
| A40 | A discarded warm-up arm at the top thread count removes the opening-arm artefact (the fix A31 asked for). | **Partly.** Three of four `t1`/`t1b` brackets came back inside 0.3%, against 0.954–0.973 on three nodes without one. The exception is the first dtype pair of the longer session, at 0.972–0.977, whose *second* pair is clean — so the residual tracks position in the session rather than package temperature, and a hotter or longer warm-up would not remove it. Keep the warm-up; do not treat it as a floor. Derive the floor from a bracket adjacent to the arms being compared, which is what the "columns the change cannot touch" control does for free. |

### Part 14: load imbalance, and a claim of ours that expired

Not a measurement — a correction, an argument, and the experiment that settles
it. Prompted by the observation that a block-scatter contraction has a load
imbalance a dense GEMM does not: some blocks sit on the regular fast path and
some on the gather path, so **equal block counts are not equal work**.

#### The correction: "the corpus is fully regular" is false, and it steered things

A4 recorded, in Phase 1, that TCCG rounds stride-1 extents to multiples of 24
"which divides every register block in use", so `regA = 1.00` everywhere. That
was true *then* and is still the right reading of the TBLIS comparison, which was
measured at TBLIS's register blocks. **It stopped being true of this engine in
Phase 3**, when `f32`/`c32` shipped `MR` of 16, 32 and 48 — none of which divides
24. Nobody re-checked the sentence, and it has been repeated in `CLAUDE.md`,
`README.md` and a memory ever since.

Measured, on the arm the orientation rule actually picks, over all 392 corpus
case-dtype-methods (`bench-results/worker6150-icelake/features.csv`).

**These fractions are `--size 64` figures and the size is part of the claim** — the
extents scale with it, so which runs straddle an `MR` boundary does too. At
`tcbench orient`'s default size the same AVX-512 engine reads 40.6 / 26.0 / 34.9,
and `bench-results/phase4d/features.csv` (generated at that default) reads
36.7 / 26.0 / 33.2. Reproduce the row below with
`./target/release/tcbench orient --size 64 --csv -` on any AVX-512 machine; it
costs no CPU and touches no data.

| quantity | fraction below 1.0 | values it takes |
|---|---|---|
| `reg_a` | **42.9%** | 0.0, 0.667, 0.889, 0.963 |
| `reg_b` | **38.3%** | down to 0.501 |
| `wb` (output row blocks off the gather path) | **35.7%** | 0.0, 0.667, 0.889, 0.963 |

The quantised values *are* the imbalance: `reg_a = 0.667` means one row block in
three is on the gather path and two are not, **within a single contraction**. 123
of 392 are genuinely mixed rather than uniformly good or uniformly bad.

The mechanism is worth stating because it is not what the old sentence implies:
irregularity here is produced by the interaction of the layout with `MR`, not by
the tensors. A 24-run straddles a 16-wide block. That is the same effect the
row-block and orientation rules were built for; we simply never connected it to
*threading*.

#### What the corpus still cannot show, and the experiment for it

Its straddling is **periodic** — every third block — so a static strip of many
blocks self-averages, and at 12 panels per strip (768 panels over 64 threads) the
averaging is imperfect but real. The case where a static partition is genuinely
bound by its unluckiest strip is *aperiodic* irregularity: ragged extents, real
user tensors, block-sparse blocks. The corpus cannot produce that, which is what
`--stress ragged` exists for (it subtracts 1 from every extent, destroying the
multiple-of-24 property).

So the question is decided by a comparison the harness can already make, and
`RAGGED=1` in `phase4f-threads.sh` now makes it: the scaling curve under ragged
against the scaling curve unperturbed, **each against its own `t1`**, because
ragged changes the shapes and only within-mode ratios mean anything. If ragged
scales visibly worse, static partitioning is losing to imbalance.

#### The design ladder, if it is

1. **Cost-weighted static cut.** Strips are currently cut by panel *count*. The
   per-block cost is already known before any thread starts — `a_m_bs` and
   `d_m_bs` carry the `IRREGULAR` sentinel — so cut for equal estimated cost
   instead. No scheduler, no new memory, no semantics changed.
2. **Dynamic `ic` claiming within a `(jc, pc)` iteration**, and this is the one
   worth knowing about: **it preserves bitwise identity for free.** The barriers
   already serialise `pc` iterations across a column group — a thread entering
   `pc+1` waits at "nobody is still reading the previous panel" — so `D`
   accumulates in `pc` order no matter *which* thread does which row block, and
   each block is touched once per `pc`. Full load balancing at no cost to the
   invariant. What it does cost is locality: today a thread owns the same rows for
   the whole run, so its `A` and `D` regions stay put. Mitigation is the standard
   one, claim your home range first and steal only when starved.
3. **A dependency-counted task DAG.** Replace the collective rendezvous with nodes
   — `pack-B(jc,pc)` → many `compute(ic,jc,pc)` → `pack-B(jc,pc+1)` gated on a
   completion counter. Same memory, same packing work, no rendezvous, automatic
   balancing. Node counts are coarse enough for the overhead to vanish: `abcijk`
   at 64 threads is roughly `24 x 4 x 1` nodes.

**`pc` fusion is *not* the general enabler this file previously implied.** Fusing
makes the packed `A` block `mc x K`; at `K = 3744` and `mc = 256` that is 7.7 MB
in `f64`, far past L2. It is viable only for small `K` — the memory-bound family,
where it is most wanted — but the road to a general task graph is 3, not fusion.

#### What is *not* a reason to do any of this

Barriers. They cost skew per iteration; imbalance costs *total*, because a
systematically unlucky strip does more work every iteration and the barrier
merely exposes it. Removing barriers does not fix imbalance, and the engine
already has a barrier-free configuration — `pm == 1`, which the domain-aware gate
now selects on chiplet machines for exactly the memory-bound family. Earlier
drafts of this argument had that emphasis backwards.

*Decisions introduced here: D45 — stated in [Design decisions](#design-decisions).*

| # | Assumption | Status |
|---|---|---|
| A41 | Blocks of a block-scatter contraction are equal-cost, so partitioning by block count balances the load. | **False, and measured false on the corpus we call regular** — 123 of 392 case-dtype-methods have mixed regularity within one contraction, at ratios of 1/3, 1/9 or 1/27 of blocks on the gather path, which the orientation work priced at roughly 2x. Whether it *costs* at thread scale is what `RAGGED=1` answers; on the unperturbed corpus the straddling is periodic and a static cut partly self-averages. |

### Part 15: what TBLIS actually does about threading

Prior art for part 14, read from source rather than assumed. Both trees are on
disk — `baselines/tblis-1.3.0` (git `c4f81e0`) and `baselines/tblis-2.0` (git
`555320c`, which vendors BLIS `358e689c`). Nothing was built. The four
load-bearing claims below were spot-checked directly against the source after the
survey, and all four hold.

#### The three findings that matter to us

**1. `PC` is never parallelised, in either version, and for our reason.** 1.3.0
gives the `kc` level exactly one gang — `communicator comm_kc = comm_nc.gang(
TCI_EVENLY, 1);` (`src/nodes/gemm.hpp:127`) — and 2.0 inherits BLIS's
`bli_rntm_set_pc_ways_only(1, rntm); // Disable pc_nt values.`
(`frame/base/bli_rntm.c:292`). Matthews states the reason in the BSMTC paper
itself (arXiv:1607.00291 §7.2): the `p` loop *"is not parallelized since this
would require additional synchronization and/or temporary buffers with
reduction."* **This is independent confirmation of A21** from a mature engine
that had every opportunity to do otherwise.

**2. TBLIS does nothing whatsoever about block-scatter load imbalance.** It
computes the same irregularity sentinel we do — `fill_block_stride` sets a
block's stride to `0` when it is not uniform (1.3.0
`src/matrix/block_scatter_matrix.hpp:232-247`; 2.0
`tblis/frame/base/block_scatter.cxx:219-236`) — and forks on it at pack and
write-back time, so its per-block cost varies as much as ours. **No partitioning
site consults it.** Every split is by count or length: `nodes/partm.hpp:40` (the
single decision point for all five levels in 1.3.0),
`gemm_ker_bsmtc.cxx:158-159` and `packm_blk_bsmtc.cxx:75,78,115` in 2.0. A
case-insensitive grep for `cost|imbalance|load.?balan` over `frame/3t` and
`frame/3m` returns **zero hits**. The decisive detail is ordering: in
`gemm_ker_bsmtc.cxx` the block-stride arrays are *built* by a count-split (180-183),
barriered (213), and then read by threads whose ranges were already fixed —
**the regularity information is produced after the scheduling decision and never
fed back into it.**

BLIS's one cost-weighted partitioner, `bli_thread_range_weighted_sub`, is gated
on triangular structure (`bli_thread_range.c:790-824`) and cannot fire on the
dense objects TBLIS builds. Its TLB partitioner disclaims this exact class in its
own comment: *"It makes no effort, however, to account for differences in
threads' workload that is attributable to differences in the number of edge-case
microtiles"* (`bli_thread_range_tlb.c:650-657`).

And the author measured the consequence and shipped anyway (§9.2): BSMTC's weak
scalability is *"only slightly less"* than BLIS's ~90%, **"possibly due to load
imbalance stemming from edge cases which must use the full scatter vector."**
Diagnosed, quantified as small on TCCG, never addressed.

**So a cost-aware partition is open ground rather than catching up.** Two
qualifiers, and the first corrects the survey itself: it is *not* true that our
corpus is fully regular and would measure nothing (see part 14 — 42.9% of
case-dtype-methods have `reg_a < 1.0` at our register blocks), so the unperturbed
corpus can show the periodic case and `--stress ragged` is for the aperiodic one.
Second, TBLIS's real escape hatch is upstream of scheduling: `sort_by_stride`
(`frame/3t/dense/mult.cxx:380-382`) reorders tensor *dimensions* to manufacture
regular blocks so the imbalance rarely bites. **We already have that** —
`plan.rs:373-376` orders `M`/`N`/`H` by increasing `|stride|` in `D` and `K` by
`|stride|` in `A`, then folds. So we are not missing their mitigation; we are
both left with the residue it does not remove.

**3. TBLIS has a dynamic, atomic-claim task scheduler — and uses it only for the
sparse formats.** `comm.do_tasks_deferred`, backed by a CAS on a slot
(`external/tci/src/tci/task_set.c:35-45`), appears in `3t/indexed/mult.cxx`,
`3t/indexed_dpd/mult.cxx` and the `1t/indexed*` family, and **never** in
`3t/dense`, `3m` or `1m` — verified by listing every call site. That is precisely
the split part 14 proposes and the user's instinct predicted: **static
partitioning for one dense contraction, dynamic fork-join for block-sparse.** The
most experienced implementation of this algorithm made the same division.

#### The structural divergence, which is a real trade

| | this engine | TBLIS |
|---|---|---|
| decomposition | flat `pm x pn` grid, one cell per thread | five nested gangs, `jc → kc(x1) → ic → jr → ir` |
| packed `A` | per thread, `pm * pn` copies | **shared** within an `ic`/`ir` gang, packed cooperatively; footprint scales with `jc_nt` |
| packed `B` | one panel per column group | one panel |
| barriers | 2 per `(jc, pc)` | ~3 per `PC` iteration + ~4 per `IC` iteration (2.0) |
| per-loop thread counts | `pm`/`pn` from the cost model | 1.3.0 honours `BLIS_{JC,IC,JR,IR}_NT`; **2.0 wipes them** — `bli_rntm_set_num_threads` clears the ways so only the total is honoured (`bli_rntm.c:257-266`) |

They buy memory and cache sharing with barrier depth; we buy barrier shallowness
with duplicated packing. The one thing they can do that our grid structurally
cannot is put several threads on a *single* `MC x NC` block (the `jr`/`ir` ways),
which keeps threads cache-coherent by construction — where our answer to the same
pressure is the topology-aware gate of D41. Worth knowing when the `jr`/`ir` axis
is next considered.

Two smaller notes: TBLIS also spawns its parallel region per call with no pool of
its own (`external/tci/src/tci/parallel.c:24` — `#pragma omp parallel
num_threads(nthread)`), so it has no advantage over our `std::thread::scope`
there; and `tblis_tensor_mult` takes a **communicator** as its first argument
rather than a thread count, which is a more composable API than
`Plan::with_threads` if this engine is ever nested inside a caller's parallel
region. Both baselines here are configured `TCI_USE_OPENMP_THREADS 1` with
`TCI_USE_SPIN_BARRIER 1`.

| # | Assumption | Status |
|---|---|---|
| A42 | Block-scatter load imbalance is a solved problem in mature implementations, so a cost-aware partition would be reinventing something. | **False.** TBLIS computes the same per-block regularity sentinel and feeds it to no scheduling decision in either version, and its author published the diagnosis without a fix. Its mitigation is upstream — reorder dimensions to manufacture regular blocks — and we already do the equivalent. |

### Part 16: the two experiments that decide the threading default

Jobs 6754849 (`worker5139`) and 6755009 (`worker5178`), both rome, both rc=0,
2026-08-04. One settles the default; the other kills a design direction.

#### Small contractions: the default cannot be a fixed thread count

Speedup against `t1` **at the same size**, so the fixed cost is inside the
measurement — `timed()` takes the best of `reps` with one `execute` per rep, and
threads are spawned per call.

| MiB | dtype | t2 | t4 | t8 | t16 | t32 | t64 | serial ms |
|---|---|---|---|---|---|---|---|---|
| 0.25 | `f64` | 1.56 | **2.12** | 1.50 | 0.85 | 0.41 | **0.14** | 0.32 |
| 0.25 | `f32` | 1.40 | **1.73** | 1.10 | 0.56 | 0.26 | **0.10** | 0.22 |
| 1 | `f64` | 1.81 | 3.24 | **4.16** | 3.60 | 2.25 | 1.10 | 4.55 |
| 1 | `f32` | 1.76 | 2.91 | **3.14** | 2.43 | 1.42 | 0.66 | 2.37 |
| 4 | `f64` | 1.86 | 3.68 | 5.38 | **5.70** | 4.18 | 2.61 | 6.31 |
| 16 | `f64` | 1.85 | 3.20 | 4.44 | 4.99 | **6.19** | 5.03 | 26.07 |
| 64 | `f64` | 1.90 | 3.41 | 4.91 | 6.10 | 6.28 | **7.50** | — |

**At 64 threads, threading costs up to 10x below a megabyte.** `f32` at 0.25 MiB
runs at **0.10** — a 0.22 ms contraction takes 2.2 ms. And the best thread count
walks monotonically with size: 4, 8, 16, 32, 64 at 0.25, 1, 4, 16, 64 MiB. **A
fixed default is wrong at every size but one.**

The fixed cost reads straight off the smallest rung, where parallel work is
negligible: `t64` on a 0.32 ms job takes 2.29 ms, so the overhead is ~2.3 ms, or
**~36 µs per thread**; at `t8` the same arithmetic gives ~21 µs. That is
per-call `std::thread` spawn, and it is what makes the sub-MiB region
catastrophic rather than merely inefficient.

**Two regimes, and only one is dangerous.** Below ~1 MiB the spawn cost dominates
and threading is a *loss*. Above it, the curve saturates early (best at `t8`–`t32`
rather than `t64`) — that is a bandwidth ceiling, not spawn, and it is benign: you
get 4x instead of 6x. A guard only has to fix the first.

> **Corrected by part 18, and this paragraph is the thing it corrects.** The
> saturation above ~1 MiB is *substantially fixed cost, not bandwidth*: pooling the
> threads makes `t64` beat unpooled best-at-any-width in all eight size x dtype rows,
> and at 16 MiB `f64` the peak goes 6.20 at `t32` to 13.19. What survives is that the
> optimum still sits below 64 threads, so a fixed default is still wrong at most
> sizes — D46 stands as a decision, but not for this reason. And "a guard only has to
> fix the first" turned out to be the sentence that mattered: the guard was scored
> only at `requested = 64` and loses 39% at `requested = 4` (A54, D52).

**So `TENSORCONTRACT_THREADS` must not simply be flipped on.** Two ways forward,
and they compose:

* **Amortisation guard** — cap the thread count so the spawn cost stays a bounded
  fraction of the estimated serial work. Fitting the 0.25 MiB rung, "spawn ≤ 50%
  of serial" gives `p <= 5`, and `t4` is indeed the measured optimum there. Small,
  needs only a flop estimate, and converts a 10x regression into "no worse than
  serial". It does *not* predict the saturation above 1 MiB, which is a different
  mechanism — do not fit it to that.
* **Pool the threads**, which removes the per-call cost outright and shrinks the
  dangerous region rather than steering around it. *Measured in part 18: it removes
  more than spawn, and the "dangerous region" is wider than this part thought.*

And it sharpens the batched-API argument: for many small contractions the right
parallel axis is *the batch*, giving one spawn per batch instead of one per
contraction, which is exactly the cost measured here.

#### Load imbalance from block-scatter irregularity: refuted, and cleanly

`--stress ragged` subtracts 1 from every extent, destroying TCCG's multiple-of-24
property. It did what it was meant to, and the perturbation is characterised
rather than assumed:

| | case-dtype-methods with `reg_a < 1.0` | values |
|---|---|---|
| unperturbed | **11.7%** | 0.667, 0.889, 0.963 — the periodic quantisation |
| ragged | **87.2%** | 0.348, 0.349, 0.696, 0.697, 0.789, 0.793, 0.87, 0.901 — aperiodic |

(The unperturbed figure is lower here than the 42.9% in part 14 because this is an
AVX2 node: `MR = 8` for `f64` *does* divide 24. The contrast is therefore cleaner,
not weaker.)

Scaling, each mode against **its own** `t1`:

| dtype | width | unperturbed | ragged | ragged/unperturbed |
|---|---|---|---|---|
| `f64` | 8 | 4.93 | 5.23 | 1.062 |
| `c64` | 8 | 5.96 | 6.13 | 1.028 |
| `f64` | 64 | 7.64 | 7.92 | 1.037 |
| `c64` | 64 | 8.21 | 8.60 | 1.048 |
| `f32` | 64 | 8.04 | 8.36 | 1.040 |
| `c32` | 64 | 7.91 | 8.89 | 1.124 |

**No degradation — 0.976 to 1.124, and if anything ragged scales slightly
better.** Seven and a half times as many heterogeneous cases, at aperiodic
fractions down to 35% of blocks irregular, and parallel efficiency does not move.

Why, most likely: at 64 threads occupancy is 32–39%, so threads are waiting on
memory rather than on each other, and compute-side imbalance hides inside that.
Static strips are also 12+ panels wide at 64 threads, which averages a good deal
even when the pattern is aperiodic.

**So the design ladder in part 14 is not worth building** — cost-weighted cuts and
dynamic `ic` claiming are solutions to a problem that does not measurably exist
here. Two limits on that conclusion, both real: it says nothing about **block-sparse
with wildly varying block sizes**, which is a far larger imbalance of a different
kind, and it was measured where the machine is bandwidth-bound. If a future kernel
or a smaller working set makes the engine compute-bound at scale, re-run it.

*Decisions introduced here: D46, D47 — stated in [Design decisions](#design-decisions).*

| # | Assumption | Status |
|---|---|---|
| A41 | Blocks of a block-scatter contraction are equal-cost, so partitioning by block count balances the load. | **False in the premise, true in the consequence.** Blocks genuinely differ in cost — part 14 measured the heterogeneity and it is large. But making it 7.5x more prevalent and aperiodic changes parallel efficiency by less than the noise floor, so the imbalance does not *cost* at these thread counts on this machine class. Stated this way because the premise may matter again where the consequence does not follow — a compute-bound machine, or block-sparse. |
| A43 | Per-call thread spawn is a second-order cost, worth fixing after the partition. | **Refuted at small sizes.** ~20–36 µs per thread, which is the entire story below 1 MiB: a 0.22 ms `f32` contraction takes 2.2 ms on 64 threads. It is first-order for exactly the workload Phase 1 identified as the headroom. *Part 18 removed it and found it worth up to 11.6x — and also that it is not the whole cost, nor confined to below 1 MiB (A53).* |

### Part 17: three answers to the spawn cost, and the batch axis

**Status: built, correct, and measured only offline.** All three are **off or
inert by default** and each is a run-time switch, so the next session can A/B them
against the shipped behaviour in one process-restart rather than as a diff between
two builds (A15). Written after part 16 established that per-call spawn is
first-order below a megabyte (A43) and that no *fixed* thread count can be the
default (D46).

Part 16 left one problem with three distinct answers, and they compose rather than
compete:

| answer | what it does to the ~20–36 µs/thread | switch |
|---|---|---|
| amortisation guard | **steers around it** — caps the thread count so it stays a bounded fraction of the work | `TENSORCONTRACT_AMORTISE=on` |
| thread pool | **removes it** — parked workers instead of `std::thread::scope`. Measured in part 18, and it removes *more* than spawn: the cost is size-dependent, so the wording "removes the ~20–36 µs/thread" throughout this part is an undercount (A53) | `TENSORCONTRACT_POOL=on` |
| batched API | **removes the count** — one spawn set per batch instead of one per contraction | `tensorcontract::batch` |

#### The amortisation guard, calibrated against a committed grid

The rule is one line: each thread must be given at least `MIN_FMAS_PER_THREAD`
real FMAs, so `p <= work_fmas / C` clamped to `[1, requested]`. Written that way
it holds the spawn **fraction** constant rather than the thread count — with spawn
`S` and machine rate `R` FMAs/s, spawn over serial time is
`(W/C) * S / (W/R) = S * R / C`, independent of the problem. At `S = 30 µs`,
`R ≈ 20 G` FMAs/s and `C = 3e6` that fraction is **0.2**.

`work_fmas` is `m * n * k` weighted by real FMAs per logical MAC — 1 real, 4
planar/1m, **3 for 3m**, which is exactly the flop saving that method exists for.
Unweighted `m * n * k` misjudges the two domains in opposite directions, because at
a fixed byte size a complex contraction has half the elements and four times the
arithmetic per element.

`C` was **not guessed**: `bench-results/worker5139-zen2/phase4g/` already measured
all 588 case-dtype-method points at 1, 2, 4, 8, 16, 32 and 64 threads at four
sizes, so candidate constants score exactly, offline, for free — the same method
that settled the row-block rule, the orientation rule and the partition gate.
**Reproduce every row below with `scripts/amortise-score-rule.py
bench-results/worker5139-zen2/phase4g`**, which also cross-checks the constant it
scores against `plan.rs` so the script and the engine cannot drift apart in silence.
(The table was first produced by an ad-hoc script and was therefore un-rederivable
from the repository for a day, which is the defect the rest of this file exists to
prevent.)
Corpus geometric mean against serial with 64 threads requested throughout, and the
count of points left slower than serial:

| nominal size | unguarded | `C = 2e6` | **`C = 3e6`** | `C = 4e6` | `C = 8e6` | per-case oracle |
|---|---|---|---|---|---|---|
| 0.25 MiB | **0.20**, 565 slower, worst **0.022** | 1.33, 0 | **1.66, 0** | 1.18, 0 | 1.04, 0 | 2.40 |
| 1 MiB | 1.32, 229 slower, worst 0.077 | 3.98, 0 | **3.50, 0** | 3.31, 0 | 2.50, 0 | 4.94 |
| 4 MiB | 2.91, 57 slower, worst 0.363 | 5.38, 0 | **4.60, 0** | 5.11, 0 | 3.81, 0 | 6.37 |
| 16 MiB | 5.48, 3 slower, worst 0.800 | 5.50, **3** | **5.67, 0** | 5.76, 0 | 6.65, 0 | 7.42 |

Three things this table says that the corpus-geomean table in part 16 could not:

* **The damage is far worse per case than per corpus.** Part 16 reported the 0.25
  MiB geomean at 0.14–0.10; per case the worst point runs at **0.022**, i.e. 45x
  slower on 64 threads than on one.
* **1 MiB is not safe either.** Part 16's corpus geomean at 1 MiB is 1.10 for
  `t64`, which reads as "no longer a loss". Per case, **229 of 588 points are still
  slower than serial there**, worst 0.077. "Below ~1 MiB" understates where the
  guard is needed.
* **The guard beats no guard at every size, not only at the small end.** At 16 MiB
  it also removes the three points that were slower unguarded.

`C = 3e6` is chosen and the choice is a plateau, not a peak: every value in
`[3e6, 8e6]` leaves 0 points slower at all four sizes, and the spread between them
is inside the ±11% per-case floor a 64-thread measurement has (A39). The choice
between *guard* and *no guard* is not inside any floor. Weighting by method is
worth having: unweighted, the same scoring gives 1.23 at 0.25 MiB against 1.66.

**What it deliberately does not do** is predict the saturation above ~1 MiB, where
the best count is `t8`–`t32` rather than `t64`. That is a bandwidth ceiling — a
different mechanism, and benign — and fitting one constant to two mechanisms is how
the analytical blocking model lost (A33). At 16 MiB this rule caps only 60 of 588
points and never below 36 threads, which is the intended near-inertness.

`C` bundles the machine's rate, so it is a fitted constant with `kc`'s caveat and
the opposite sign: **a faster machine wants a larger one.** A single unrepeated
observation on the shared 16-core reference machine — 8 threads on the 0.25 MiB
`ij-ik-kj` case — has the guard capping to 2 and *costing* throughput there, which
is consistent with a lower thread count making the total spawn cost smaller. That
is one rep on a shared machine and is not a measurement; it is recorded because it
is the direction the caveat predicts, and because it is why the guard is off by
default rather than on.

#### The thread pool, and the deadlock that rules out the obvious reuse

`crates/tensorcontract/src/pool.rs`, ~200 lines including its tests, no dependency.
One `Mutex`/`Condvar` mailbox per worker, a shared completion counter, and one
operation: `try_broadcast(n, f)` runs `f(0)` … `f(n-1)` and returns when all have
finished. The submitting thread takes index 0, so width `n` needs `n-1` workers and
**a serial caller never causes a thread to exist**.

Three design points that are not free choices:

1. **`rayon::scope`/`spawn` deadlocks by construction here.** Our parallelism is
   SPMD-with-barriers — `pn` barriers of width `pm`, two rendezvous per `(jc, pc)`
   iteration. A task that blocks on a barrier occupies a worker thread, so a pool
   with fewer workers than barrier participants never schedules the rest and the
   barrier never opens. `ThreadPool::broadcast` *does* fit the shape but ties the
   parallel degree to the pool size, where `Plan::partition` chooses `(pm, pn)` per
   contraction; and a global pool inside a library fights the host runtime, which
   for this project means the Julia package and the C consumers concretely rather
   than hypothetically. See `REFUTED.md`.
2. **`try_broadcast` declines all-or-nothing.** Two cases: another broadcast is in
   flight (one completion counter serves the pool), or fewer than `n-1` workers
   could be spawned. The tempting graceful degradation — fold the surplus indices
   onto the submitting thread — is a **deadlock**, because indices `t` and `t + pn`
   share a barrier and running them sequentially waits forever for a participant
   that has already left. Declining costs the spawn saving for that call and
   nothing else: the driver answers a decline with `std::thread::scope`, which is
   the shipped path anyway. (A51.)
3. **Poisoning must not brick the pool.** This was found by the test, not by
   reading: a worker's panic propagates through `try_broadcast` while it holds the
   submission mutex, so the *next* broadcast finds it poisoned. Handled with
   `TryLockError::Poisoned(e) => e.into_inner()`, so `WouldBlock` is the only
   decline. Without it, the pool would silently stop pooling for the life of any
   process in which one contraction ever panicked — and `Panel::new` asserts on an
   allocation it cannot satisfy, so that is reachable (D38, A52).

The driver now writes one thread's whole job as a closure of its grid index and
reaches it two ways, pooled or spawned, so the arms cannot drift apart. The
partition, the strips and therefore the arithmetic are identical, which is why the
result stays bitwise identical to serial either way — asserted by running the whole
suite at 2, 4 and 8 threads with the pool on.

**Directional only, and stated as such**: on the shared reference machine at 8
threads, 0.25 MiB `ij-ik-kj` reads 24.4 GF/s unpooled against 43.0 and 80.1 GF/s
pooled on two runs, and 16 MiB reads 328.3 against 329.7 — large at the small end,
a no-op at the large end, which is what the mechanism predicts. The 2x spread
between the two pooled runs is why this is not a number: it is a shared machine and
five reps. The measurement is `scripts/ab.sh` on an exclusive node.

#### The batched API

`crates/tensorcontract/src/batch.rs`: `BatchItem` plus `contract_batched` /
`contract_batched_with_threads`. Many independent contractions, **the batch as the
only parallel axis**, each item running serially — nesting the two axes would spend
the spawn saving again inside every item, which is the whole point of the batch.
`driver::execute_capped` is how "serially" is expressed, and it is additive:
`execute` is that with no cap.

Three things worth recording:

* **Soundness is the borrow checker's, not a comment's.** Items are held in a
  `&mut [BatchItem]` and each item's `d` is a `&mut` borrow, so the outputs are
  *proved* disjoint; `chunks_mut` hands each thread an exclusive slice. There is no
  unsafe block on the caller's side and none in the fan-out.
* **All or nothing on validation.** Every item's bounds are checked before any item
  runs, so a batch containing one bad item writes to no output at all. A loop over
  `Plan::run` cannot give that, and a partially executed batch leaves the caller
  unable to say which outputs are valid. This is the reason to prefer the batched
  entry point even at one thread.
* **`rayon` fits here and is still not used**, for a much weaker reason than the
  inner path's: fork-join over disjoint `&mut` chunks is ten lines of
  `std::thread::scope`, and the crate already owns a pool that can be extended to
  this axis. Recorded so the next person does not re-derive it — the objection here
  is *unnecessary*, not *unsound*.

The split is static and contiguous by item count. Balancing it by estimated work is
the obvious refinement and is deliberately not guessed at: D47 measured that
block-scatter load imbalance does not cost on the dense path, and the case where it
plausibly does is **block-sparse, where items differ in size rather than in
regularity**, which is not built. That is the same division TBLIS makes — a dynamic
atomic-claim scheduler for its sparse formats, static partitioning for dense (part
15) — reached independently.

**Block-sparse is not started**, and that is the honest state of the second half of
this item.

#### Two hygiene fixes that belong here

* **A per-job `CARGO_TARGET_DIR`.** A cluster job runs in the submit directory and
  used its `target/`, so a `cargo build` on the workstation — or a `cargo test`,
  which relinks the same artefacts — replaced the very executable each arm invokes,
  mid-session, with nothing reporting it. It happened on 2026-08-04 (part 13) and
  was inert only by luck. Every script now resolves its binary from `TC_TARGET`
  (default `target/`, so hand runs are unchanged) and the `.sbatch` wrappers set it
  to `target-job-$SLURM_JOB_ID`. The two prebuilt-binary jobs *snapshot* into it
  instead, because they never compile — which also closes the failure that killed
  job 6753197, where a concurrent `prep` left a 0-byte executable that every arm
  then ran for 0.0 s with `rc=0`.
* **The harness reports the thread count it *used*.** `sweep`'s `notes` column read
  `t{requested}`; under the guard that would claim `t64` for a contraction that ran
  on two threads. It now reports `t{used}/of{requested}` when they differ, because a
  CSV that misstates its own configuration is the provenance error this project has
  had to retract twice.

### Part 18: the pool wins on Zen2, the guard does not, and the joint arm is why

> **Overtaken, 2026-08-05.** This part measured one machine class and recommended
> the pool as a default on that evidence. Part 19 measured the other and found the
> same switch costs up to 2.5x on Ice Lake, so **D53's recommendation is withdrawn**
> and the pool stays off. The Zen2 numbers below stand; the recommendation drawn
> from them does not. Read part 19 before quoting anything here.

Job **6760092**, `worker5086` (Zen2 `rome`, 64 cores of one socket, 16 L3 domains),
2026-08-05, 107 min, `--exclusive`, `OverSubscribe=NO`, engine at commit `a2f425f`.
Raw data in `bench-results/worker5086-zen2/phase4g/` — 225 CSVs, four arms over
4 sizes x 7 widths x 2 dtype pairs.

Part 17 built three answers to the per-call spawn cost and measured none of them.
This measures them, and the result is not what part 17 predicted.

**The design is a 2x2, and that is the whole reason this run is conclusive.** Arms:
`base`, `pool` (`TENSORCONTRACT_POOL=on`), `guard` (`TENSORCONTRACT_AMORTISE=on`),
`both`. One discarded warm-up serves all four (A31). Each arm's `t1` column is a free
control — no switch here can touch single-threaded work — so every row is deflated by
its own `t1` before being read, which is the "columns the change cannot touch" rule
applied per row rather than per table.

#### The result

Drift-corrected treatment/base, per-family geomeans (the only granularity a
64-thread measurement supports, A39):

| | `pool` t4 | `pool` t64 | `guard` t4 | `guard` t64 | `both` t4 | `both` t64 |
|---|---|---|---|---|---|---|
| 0.25 MiB `f64` | **1.53** | **10.30** | **0.61** | 7.86 | 0.63 | 8.48 |
| 0.25 MiB `f32` | **1.77** | **11.64** | **0.73** | 10.84 | 0.79 | 12.50 |
| 0.25 MiB `c64` | 1.20 | 7.65 | 0.86 | 4.51 | 0.96 | 6.22 |
| 0.25 MiB `c32` | 1.36 | 10.44 | 0.90 | 6.66 | 1.10 | 10.48 |
| 1 MiB `f64` | 1.13 | 5.18 | 0.89 | 2.91 | 0.98 | 4.88 |
| 4 MiB `f64` | 1.12 | 4.08 | 1.01 | 1.80 | 1.10 | 4.03 |
| 16 MiB `f64` | 0.99 | 2.38 | 0.99 | 1.08 | 1.01 | 2.41 |
| 16 MiB `c32` | 1.02 | 2.27 | 1.00 | 1.02 | 1.01 | 2.40 |

**The pool is a uniform win**: no per-family cell below 0.99, up to 11.6x, monotone
down with size and up with thread count — which is what a fixed cost divided by
growing work has to look like.

**The guard is a trade, not a win.** It rescues the over-threaded case (1.0–10.8x at
`t64`) and *penalises the correctly-threaded case* by 10–39% at `t2`–`t8` on the
smallest size, where part 16 measured `t4` to be the optimum. Which side a caller
lands on depends on whether their thread count was already well chosen — something a
library cannot know.

**And on top of the pool the guard is pure loss.** `both / pool`, drift-corrected:
0.32 at 0.25 MiB `f64` `t16`, 0.41 `f32`, 0.46 `c64`, 0.51 `c32`; 0.70–0.94 at 1 MiB;
inert (0.99–1.07) at 4 and 16 MiB. Never better than 1.07. The mechanism is plain:
the cap was fitted against a ~37 µs/thread spawn cost that pooling has removed, so it
rations a resource that is now cheap.

**Per case is where the two really separate**, and it is the reading that matters
because the geomeans understate the guard's damage by 3x. Control-corrected, over all
2352 points at each width:

| width | `pool` p10 | `pool` worst | `pool` < 0.90 | `both` p10 | `both` worst | `both` < 0.90 |
|---|---|---|---|---|---|---|
| 2 | 0.987 | 0.711 | 16 | 0.849 | 0.548 | **274** |
| 4 | 0.995 | 0.742 | 23 | 0.681 | 0.327 | **323** |
| 8 | 1.011 | 0.578 | 33 | 0.861 | 0.302 | **286** |
| 16 | 1.134 | 0.688 | 16 | 1.079 | 0.466 | 59 |
| 32 | 1.712 | 0.959 | 0 | 1.595 | 0.711 | 8 |
| 64 | 1.782 | 0.928 | 0 | 1.785 | 0.850 | 1 |

The pool's sub-0.90 tail is **0.7–1.4% of points**, which is what A39's floor already
produces on *identical* configurations — so it is not evidence of a systematic loss,
and it is not proof of none. The guard's is **12–14% of points**, far outside that
floor, which is.

**So, on this machine class: `TENSORCONTRACT_POOL` wins and `TENSORCONTRACT_AMORTISE`
does not ship (D52).** The recommendation to default the pool on (D53) was drawn here
and **withdrawn in part 19**, which measured the other machine class; both switches
stay off.

#### Two controls that were not the point, and both replicate

* **The base arm reproduces part 16 on a fourth Zen2 node.** 16 MiB `f64` reads best
  6.20 at `t32` and 5.00 at `t64` here, against `worker5139`'s 6.19 and 5.03. Two
  within-session ratios agreeing to 0.6%, which is what makes the treatment arm beside
  it worth reading — and it is the same form of replication part 8 used.
* **D48's offline calibration reproduces**, simulated from this node's own base grid:
  no guard 0.20 with 564 of 588 points slower, `C = 3e6` 1.65 with 0 slower, oracle
  2.37 — against `worker5139`'s 0.20/565, 1.66/0, 2.40
  (`scripts/amortise-score-rule.py bench-results/worker5086-zen2/phase4g`). The
  *simulation* is sound. What it could not see is the next subsection.

Predicted against actual for the guard, `f64c64`, requesting 64 threads: simulated
1.72 / 3.84 / 5.04 / 5.99 with 0 points slower at each size, measured 1.56 / 3.43 /
4.60 / 6.06 with 3 / 0 / 0 / 0. The three are at 0.967, i.e. at the 0.97 threshold and
inside A39's floor. The gap between predicted and actual tracks the arm's own `t1`
drift (0.934–0.994; the guard arm ran ~50 min after base).

#### The calibration was validated at one setting of another lever — A20 again

D48's table reports "0 points slower than serial at all four sizes". That is true
**at `requested = 64`**, which is the only case the offline scoring simulated, and it
is the winning half of a trade. The losing half — 0.61 at `t4` — was invisible to it
by construction, and D48 was committed quoting the table without that qualifier.

This is the third instance of A20 in this project, and the first where the author of
the rule and the author of the check were the same. The general form: **a rule that
takes a caller's parameter must be scored across that parameter, not at one value of
it.** `partition-score-rule.py` gets this right — it sweeps `-p` and `-d` — and the
amortisation scoring did not.

**The deeper problem is the functional form, not the constant.** A fixed
spawn-*fraction* threshold says "4 threads on a 0.32 ms contraction is 46% overhead,
cap it", while measurement says 4 threads is the optimum there. The real trade is
marginal — does the *next* thread return more than it costs — and answering that needs
a scaling model. That is exactly where the analytical blocking model died (A33), so
the honest move is to record the form as refuted rather than to fit a second constant.

#### Part 16's "benign bandwidth ceiling" is substantially wrong

Part 16 split the size range into two regimes and called the one above ~1 MiB benign:
"the curve saturates early — that is a bandwidth ceiling, not spawn". Within this one
session, pooled `t64` beats unpooled *best-at-any-width* in **all eight** size x dtype
rows:

| 16 MiB | unpooled best | pooled `t64` | pooled best |
|---|---|---|---|
| `f64` | 6.20 @ t32 | 11.89 | 13.19 @ t32 |
| `c64` | 7.34 @ t16 | 13.29 | 13.29 @ t64 |
| `f32` | 6.73 @ t16 | 12.83 | 15.59 @ t32 |
| `c32` | 7.29 @ t32 | 13.46 | 15.94 @ t32 |

At 4 MiB the same pattern: `f64` 5.07 @ t16 unpooled against 11.93 @ t32 pooled. So a
large part of what part 16 attributed to bandwidth was fixed cost, and D46's second
regime is narrower than it claimed. What survives is that the *optimal thread count*
still saturates below 64 — pooled, the walk compresses toward `t32` rather than
reaching `t64` — so a fixed default remains wrong at most sizes even with a pool. D46
stands as a decision; its explanation does not.

#### And the cost the pool removes is not thread spawn

Median per-call time removed at `t64`, divided by 64:

| nominal size | µs/thread, `f64c64` | µs/thread, `f32c32` |
|---|---|---|
| 0.25 MiB | 37.2 | 37.2 |
| 1 MiB | 43.4 | 41.5 |
| 4 MiB | 54.0 | 49.8 |
| 16 MiB | **93.6** | **66.2** |

A thread-creation cost is size-independent. At 0.25 MiB this lands at 37.2 µs in
*both* dtype pairs — the top of A43's 20–36 µs, and a nice independent confirmation of
it. By 16 MiB it is 2.5x that. **So "the pool removes the ~20–36 µs/thread spawn",
which is how part 17 words it, is an incomplete account** (A53).

**A hypothesis was offered here and is now refuted, from committed data and at no
machine cost.** The candidate was the per-thread packed-`A` buffer: `Panel::new(ap_len)`
is called *inside* each thread's closure on every call, so 64 fresh threads mean 64
fresh allocator arenas faulting in fresh pages, and if `ap_len` grew with the problem
that would scale with size where spawn does not.

**`ap_len` does not grow.** It is `panel_len(min(mc, ceil(m/mr)*mr), mr, kc, ..)`, and
across all four sizes the footprint is *identical* — 65536 reals in `f64`, 22528 in
`c64`, 135168 in `f32`, 46080 in `c32` — because even at 0.25 MiB the median oriented
`m` is 384, comfortably above every `mc` in play, so the cap never binds. The buffer
at 0.25 MiB is the same buffer as at 16 MiB. Its allocation cannot explain a cost that
grows 2.5x between them.

**A trap found on the way, worth more than the hypothesis was.** The first attempt at
this check read the `mc` recorded in each CSV's `notes` column — and `notes` comes from
`plan_config`, which returns the *derived* blocking, while the driver caps it at
`driver.rs:404` and allocates from the capped value at `:445`. The two agree here only
because `m` is large; on a narrow case they would not, and any analysis of buffer
footprints from these CSVs has to apply the cap itself.

**So the mechanism of the size-scaling component is unidentified**, and the honest
reading is that the "µs/thread" framing above is partly an artefact of my own
arithmetic: it assumes the whole pooled-vs-unpooled difference is a fixed per-call cost
and divides by the thread count, so any proportional component appears as growth. The
difference is neither purely fixed (it grows 2.4 → 6.0 ms) nor purely proportional (the
ratio falls 9.6 → 2.1), and fitting two parameters to four points on one node would be
over-fitting.

What is solid: **a fixed component of ~37 µs/thread**, visible at 0.25 MiB where
parallel work is negligible, identical in both dtype pairs, and an independent
confirmation of A43's 20–36 µs. What is not: everything about the rest. Two candidates
remain, neither testable from this data — scheduler placement of freshly created
threads, whose penalty accrues over the thread's life and so scales with runtime; and
barrier skew, whose count grows with `N/NC` x `K/KC` and hence with size.

**Consequently the "third, smaller fix" this section originally proposed — hoisting the
`Panel` allocations out of the per-call closure — is not justified.** Its premise was
that the buffer grows, and the buffer does not. It might still help by removing a
constant allocation, but that is a different and much weaker claim, and nothing here
supports building it.

#### What is not measured

* **One node, one session.** The direction is trustworthy on this machine class; the
  magnitudes are not portable, and nothing here may be differenced against part 16's
  absolute speedups (different node).
* **Nothing at 64 MiB.** The corpus size the rest of this file uses is outside the
  sweep. The pool's win decays with size (2.0–2.8x at 16 MiB) and may be small there,
  but "may be" is the honest word. `THREADS="1 16 64" STAGES=threads` is ~55 min.
* **Nothing on an Intel hierarchy.** Every partition and threading finding so far has
  been topology-dependent (A36), and the pool has no reason to be — it touches where
  threads come from, not what they read — but that is an argument, not a measurement.
* **The batched API is unmeasured entirely.** It was built in part 17 and this run did
  not exercise it.

| # | Assumption | Status |
|---|---|---|
| A43 | Per-call thread spawn is a second-order cost. | **Refuted, and now measured directly rather than inferred.** Removing it is worth up to 11.6x at 0.25 MiB and 64 threads, and 2.0–2.8x even at 16 MiB. The 0.25 MiB figure of 37.2 µs/thread, identical in both dtype pairs, independently confirms the 20–36 µs part 16 inferred from a different node. |
| A50 | A guard fitted to the sub-megabyte regime will misjudge the saturation above it. | **Confirmed, and the misjudgement is the other way round from the worry.** The guard is inert above 4 MiB (0.98–1.09), so it does no damage there. What was misjudged is the *saturation itself*: it is largely fixed cost, not bandwidth. |
| A53 | The cost a thread pool removes is thread creation, and the part that scales with size is the per-thread packed-`A` allocation. | **First half refuted, second half refuted too.** The pooled-vs-unpooled difference is not a pure fixed cost: it grows 2.4 → 6.0 ms with size while the ratio falls 9.6 → 2.1, so it has a proportional component and the "µs/thread" figure manufactures growth by dividing a mixed quantity by the thread count. The named candidate is *also* wrong: `ap_len` is byte-identical at all four sizes (65536 reals in `f64`, 22528 in `c64`), because `mc` never binds against `m` at these shapes. **A fixed ~37 µs/thread is solid** and confirms A43 independently; the scaling component's mechanism is **unknown**, with scheduler placement and barrier skew as untested candidates. Nothing supports hoisting the `Panel` allocations. |
| A54 | A rule that consumes a caller's parameter can be validated at one value of it. | **Refuted, and it is A20 in a third form.** D48's guard scores 0 points slower than serial at `requested = 64` and loses 39% at `requested = 4` on the same size. The offline scoring simulated only 64, so the losing half of the trade was invisible by construction. Score across the caller's parameter, as `partition-score-rule.py` does with `-p` and `-d`. |

### Part 19: the pool loses on Ice Lake, and D53 is withdrawn

Job **6762432**, `worker6194` (Ice Lake-SP, 32 cores of one socket, **one 48 MiB L3**),
2026-08-05, 107 min, `--exclusive`, engine at `ea10301`. Two arms, `base` and `pool`,
at 1 / 16 / 64 MiB. Raw data in `bench-results/worker6194-icelake/phase4g/` — 84
arm-runs, 21168 rows, 0 missing, 0 zero-second.

This was the strengthening evidence part 18 asked for before flipping D53: the same
arms on an Intel hierarchy, plus a 64 MiB point at the corpus size. **The reading rule
was fixed before the data existed** — at 64 MiB and `t32`, `>= 1.10` meant "matters
everywhere", `1.00–1.10` "small contractions only", `< 0.97` "the pool costs at corpus
size, which would be a real surprise".

It came in at **0.80**.

#### The result

`pool / base`, drift-corrected by each row's own `t1` (0.996–1.007 throughout, so the
control is clean and this is not session drift):

| MiB | dtype | t2 | t4 | t8 | t16 | t32 |
|---|---|---|---|---|---|---|
| 1 | `f64` | 1.054 | 1.006 | 0.591 | **0.457** | 0.527 |
| 1 | `c32` | 1.028 | 0.928 | 0.546 | **0.403** | 0.478 |
| 16 | `f64` | 0.994 | 0.875 | 0.740 | 0.505 | 0.587 |
| 16 | `c32` | 0.993 | 0.895 | 0.773 | 0.592 | 0.650 |
| 64 | `f64` | 0.993 | 0.932 | 0.872 | 0.791 | **0.802** |
| 64 | `c32` | 0.997 | 0.961 | 0.893 | 0.779 | **0.781** |

**46 of 60 cells below 0.97, worst 0.403** — the pool is up to **2.5x slower** than
per-call spawn on this machine. All four dtypes agree. Per case at 64 MiB, **210–284 of
588 points fall below 0.90** (36–48%), worst 0.094, which is systematic under any
reading of A39.

The shape: neutral at `t2` (1.01–1.08, where the pool has a single worker), degrading
through `t4`–`t8`, **worst at `t16`**, partial recovery at `t32`. Worst at `t16` rather
than at the top width is not explained and does not fit a simple monotone story.

So on Zen2 the pool is worth up to 11.6x and on Ice Lake it costs up to 2.5x. **D53's
recommendation cannot stand**, and the switch is topology-conditional at best.

#### The a-priori argument was wrong, and that is the transferable part

Recorded here before the run, as the reason a null result was expected rather than a
reversal: *"the pool has no reason to be topology-dependent — it touches where threads
come from, not what they read."*

**That is false, and it is false for a reason worth carrying.** Reusing a thread also
reuses its **allocator arena**, so each worker gets the *same* packed-`A` buffer address
back on every call, where a freshly spawned thread gets a fresh, well-spread one. **A
thread pool is an allocation-locality change as much as a thread-lifetime change** — it
changes what the threads read, not just where they come from (A55).

That reframing also makes the topology dependence unsurprising in hindsight, which is
the uncomfortable part: 32 long-lived buffers at repeating addresses all contend in
**one** 48 MiB 12-way L3 here, where Zen2 spread the same buffers across **sixteen**
private 16 MiB L3s — and Zen2 is exactly where the pool won.

#### The mechanism is a candidate, not a finding

The extra time the pool costs is neither a fixed per-call cost nor proportional to
runtime:

| MiB | width | base ms | pool ms | delta ms | delta/base |
|---|---|---|---|---|---|
| 1 | 32 | 0.609 | 1.046 | 0.431 | 0.71 |
| 16 | 32 | 2.304 | 3.944 | 1.671 | 0.73 |
| 64 | 32 | 5.355 | 7.484 | 1.781 | 0.33 |
| 64 | 8 | 8.293 | 9.284 | 0.939 | 0.11 |
| 64 | 16 | 6.061 | 7.857 | 1.459 | 0.24 |

It grows with width and with size, then plateaus. The implementation's *fixed* costs —
31 serial `notify_one` wakes and 31 threads contending on one completion mutex — are of
order 100–200 µs, an order of magnitude too small to be this.

Two candidates, and this data cannot separate them:

1. **Shared-L3 conflict between reused buffers**, per A55 above. Scales with width (more
   aliasing buffers) and with traffic (size). A cheap test: pad each worker's buffer by
   a per-worker offset so the addresses cannot alias, and re-run this arm.
2. **Barrier cost at 32 participants**, or the wake pattern interacting with it. `t16`
   being worse than `t32` is unexplained under (1) and might belong here.

Both locate the problem in **our implementation rather than in pooling as a concept**,
which is the reason D49 keeps the pool as a switch rather than deleting it.

#### What this run says about the process, twice over

* **The pre-registered reading rule earned its keep.** Writing `< 0.97 → a real
  surprise` down *before* the data meant the reversal could not be narrated as
  "roughly neutral" afterwards. It cost two minutes.
* **Two of my own a-priori arguments were wrong in one day** — that the pool could not
  be topology-dependent (this part) and that the buffer grows with problem size (part
  18's A53). Both were stated as reasoning rather than measurement, and both were
  checked because they were written down as claims. The lesson is not "argue less"; it
  is that an argument recorded as an argument gets tested, and one folded into prose
  does not.

*Decisions introduced here: none. D53 is revised in place — see
[Design decisions](#design-decisions).*

| # | Assumption | Status |
|---|---|---|
| A55 | A thread pool changes thread lifetime, not memory locality — it touches where threads come from, not what they read. | **Refuted, and it was the reason a reversal was thought impossible.** Reusing a thread reuses its allocator arena, so every worker gets the same packed-`A` buffer address back each call where a fresh thread gets a well-spread one. On a one-L3-per-socket machine 32 such buffers contend in one 48 MiB cache; on Zen2 they spread over sixteen private L3s, which is where the pool won. A pool is an allocation-locality change. |
| A56 | The pool's benefit is topology-independent, so one machine class is enough to recommend it. | **Refuted — 11.6x on Zen2, 0.40–0.80 on Ice Lake.** This is the **fourth** threading or kernel choice here that fails to transfer between microarchitectures, after A34 (register blocks), A36 (the partition) and A44 (the method ranking). Treat "measured on one machine class" as a statement about that class until shown otherwise, for anything touching threads or caches. |
