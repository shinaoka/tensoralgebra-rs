# Open questions

What is not known, and what would settle it. Everything here is a *question*;
none of it is work in progress.

Answered questions live in [`decisions.md`](decisions.md) and
[`refuted.md`](refuted.md); the reports are in
[`notebook/`](notebook/README.md).

## 1. What Phase 4 bought, end to end

**The one substantive open engine question.** The current engine-vs-baseline
numbers are Ice Lake (jobs 6753260 and 6754877, `worker6156`) and the Phase 3
table they would be differenced against is Cascade Lake. Two things changed at
once, so the difference is not attributable to either.

Nothing else answers it: a within-session ratio on one machine cannot be
subtracted from one on another (see
[`measurement-rules.md`](measurement-rules.md)). Only a `ccqlin038` run supplies
it:

```bash
TBLIS_ROOT_2X=../baselines/tblis-2.0-install \
TBLIS_ROOT_13=../baselines/tblis-1.3.0-install \
  scripts/compare-bench.sh prep                       # the only compile
  scripts/compare-bench.sh bench-results/$(hostname -s)-$(scripts/arch-label.sh)
```

~3.5 h, and it wants the machine to itself. **0.1.0 ships without it**, and the
README makes no throughput claim as a result.

## 2. Why the thread pool does not transfer

`TENSORCONTRACT_POOL=on` is worth up to **11.6x** on 64 Zen2 cores at 0.25 MiB
and up to **2.5x slower** on 32 Ice Lake cores sharing one 48 MiB L3 (jobs
6760092, 6762432). D53 recommended it as a default on the Zen2 evidence alone
and **that recommendation is withdrawn**.

So it is a conditional switch with an implementation defect to find, not a dead
end. The suspected cause is A55: a pool is an *allocation-locality* change, not
just a thread-lifetime one. Reusing a thread reuses its allocator arena, so every
worker gets the same packed-`A` buffer address back on each call where a fresh
thread would get a well-spread one — which is exactly the situation that hurts
where one L3 serves every thread.

The cheap test is to offset each worker's buffer by a per-worker amount and
re-run both nodes' arms:

```bash
ARMS="base;pool:TENSORCONTRACT_POOL=on" STAGES=small SIZES="1 16 64" \
  sbatch --constraint=icelake scripts/rusty-phase4.sbatch   # ~1.8 h
```

## 3. Why the pool's win scales with size

Solid: a fixed ~37 µs per thread. Unexplained: the component that grows with the
problem. **Do not build the fix that was first proposed for it** — the per-thread
`Panel` allocation was offered as the candidate and refuted in the same report,
because `ap_len` is identical at all four sizes, so hoisting those allocations
has no measured basis (A53).

The two untested candidates are scheduler placement of freshly created threads,
and barrier skew growing with `N/NC` x `K/KC`. The cheapest form of the question
is a microbenchmark separating spawn from placement from barrier count — no
corpus and no exclusive node needed.

## 4. The remaining orientation gap

21 case-dtype-methods still take the slower arm, worth up to 1.36x, and they are
a different population from the `abcijk` family the rule was derived on. **Both
arms are already committed** in `bench-results/phase4d/or-*`, so a candidate rule
costs no machine time at all: score it offline with
`scripts/orient-score-rules.py`.

## 5. What the corpus does on aarch64

Everything known about the M3 Max is the **12 premise shapes** (§4 of
[`results.md`](results.md)). The 49-case corpus was never run there: the Stage A
session was cut short after `info`, `orient`, the three `verify` arms and both
`premise` arms, and the five corpus `sweep` arms, the `ragged` arm and the TBLIS
sweep were never written. Part 22 has since supplied the floor and a 64 MiB TBLIS
number that session lacked, so what remains missing is specifically corpus-wide:

* **No `ragged` arm** — and on this machine `--stress ragged` is very nearly the only
  source of irregularity there is, because the measured NEON `f64` `MR = 16` is
  demoted away on exactly the 12 cases where it would cost write-back regularity
  (A61). So the irregular path is unmeasured on aarch64, not lightly measured.
* **No per-case corpus spread**, so nothing per-case over the 49 cases is quotable.

```bash
scripts/macos-session.sh prep  bench-results/CKF6QCDVPD-m3max   # REQUIRED, see below
scripts/macos-session.sh bench bench-results/CKF6QCDVPD-m3max 64 3   # ~5 h
```

It is a **more valuable** run than when it was planned, because its default arm now
exercises the NEON kernel over all 49 cases rather than the portable path. `SIZE=32`,
or dropping `1m`/`3m` from `ENGINES`, is a fine way to shorten it as long as the
write-up says which.

**`prep` is not optional.** The binaries in that directory's `bin/` predate the NEON
kernel — `strings` finds no `neon-real` in them — so running `bench` against them
would measure the portable path and label it the current engine. That is D51's hazard
from the other side: not a build racing a running job, but a stale build outliving
its source. After `prep`, do not compile while the arms run.

**The one open engine item on this machine is the blocking constants**
(`kernel/mod.rs`). They are the Cascade Lake values on aarch64, and until Stage B
there was no point looking: the portable path was instruction-bound, so the cache
effects those constants exist for were invisible. With a real FMA kernel they are
finally measurable. Note that touching them re-opens `f64` real `16x3`, whose 4.9%
win is measured at `kc = 256` and which is **bimodal at `kc = 16`**;
`legacy_blocking_is_unchanged` pins the current values.

**Three things not to do here**, each for a recorded reason: do not port the other
drivers (17 of them depend on `taskset`, `/proc`, sysfs or `sched_getaffinity`), do
not thread anything on this machine (nothing measured here is pinned), and do not
make the analytical block model the default (A33, and A57 — it budgets a whole
cluster-shared 16 MiB L2 to one thread, a 6x over-allocation).

## 6. Smaller, and each needing a machine

* **Re-run the row-block grid at the chosen `kc`.** `kc` decides the regime and
  the shape is chosen inside it; the Phase 3 shapes were chosen at a depth the
  engine may no longer use.
* **A35's A/B**, reachable since D43 and never run:
  `scripts/ab.sh bench-results/ab-rowblock-idx3 "TENSORCONTRACT_ROWBLOCK=idx=3"`.
* **The batched API is unmeasured**, and block-sparse — its second half — is not
  built. Block-sparse is where items differ in *size* rather than in regularity,
  which is the one place D47's null result does not reach.
* Route the batch axis through the thread pool, removing its one remaining spawn
  set.
* The rest of Phase 4: small-`k` handling, `pc`-loop fusion **for small `K` only**
  (it is not the general enabler earlier drafts implied — see
  [`refuted.md`](refuted.md)), a pack-free fast path for already-unit-stride
  block scatter, prefetch.
