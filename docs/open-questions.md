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

## 5. Smaller, and each needing a machine

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
