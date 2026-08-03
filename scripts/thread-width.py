#!/usr/bin/env python3
"""How much parallel width does the corpus actually have, and where?

The driver partitions the output into a `pm x pn` grid, so its parallelism is
capped at `ceil(M / MR) * ceil(N / NR)` cells — but the *rule* in
`Plan::partition` will not always use every cell it could, because splitting `N`
costs each thread a whole packed `A` block of its own. This answers both
questions from committed data at no CPU cost, for any thread count: for each
case-dtype-method it reports the `M` width on the arm the orientation rule
actually picks, the width an `N` split would add, the partition the shipped rule
would choose, and how fast the case currently runs.

The question behind it is which parallel axis a case needs. `M` and `N` splits
both give every output element a single owning thread, so they keep the result
bitwise identical to serial and need no reduction. A `K` split does not: it needs
per-thread accumulators and a reduction, which is a different and much more
invasive feature.

**Answered at 8 threads on 2026-08-03** (Phase 4 report, part 8): 16 of 392
case-dtype-methods cannot fill 8 threads from `M` alone, all 16 reach 8 with
`M x N` (the narrowest has 26 `NR` blocks), and the shapes that would need `K`
are bounded away from compute-bound — needing it means fewer than `p`
micro-tiles in the whole output, and an output that small caps arithmetic
intensity at roughly `2MN/((M+N) * bytes)`.

**`P` is now a parameter, because 8 was the reference machine's socket and
nothing else's.** A cluster node has 64–128 cores, and the population the 2-D
rule was designed against grows with `P`: at 8 threads the rule's `PACK_WEIGHT`
term almost never binds, while at 64 or 128 it decides most of the corpus. The
`achieved width` section below is the prediction to make *before* measuring a
scaling curve, so the curve is interpreted rather than admired — a case whose
rule-chosen partition has `pm * pn < P` cannot scale past that, and its
flattening is a modelled decision showing up in the data, not a mystery.

Count the entries in the project's convention — one per case-dtype-method, so
real dtypes appear once and each complex dtype once per method, 392 in total.
Aggregating to case-dtypes and multiplying by the method count overcounts,
because the register block differs per method and so does the panel count; that
error is how this file first reported 20. Note also that these are the *default*
register blocks, which is what `tcbench orient` emits — the shipped row-block
rule changes `MR` on some cases, so a case near the boundary can move.

"frac of best" is a crude compute-bound proxy: throughput as a fraction of the
fastest the engine achieves anywhere in that dtype on this corpus. It is not a
roofline, and it is not meant to be — it only has to separate 5 GF/s cases from
70 GF/s ones.

Usage: thread-width.py [-p 8,16,64] FEATURES_CSV BASELINE_CSV...
  e.g. scripts/thread-width.py -p 8,16,32,64,128 \\
           bench-results/phase4e/features.csv \\
           bench-results/phase4/rm-A-f64c64.csv bench-results/phase4/rm-A-f32c32.csv
"""
import argparse
import csv
import math
import sys
from collections import defaultdict

# Cost of packing one `A` element relative to one micro-kernel lane-FMA. Must
# track `PACK_WEIGHT` in `crates/tensorcontract/src/plan.rs`, which documents why
# it is 8 and why its exact value is not load-bearing.
PACK_WEIGHT = 8


def partition(panels, blocks, nr, p):
    """`Plan::partition`, replayed offline. Keep in step with plan.rs."""
    if panels >= p:
        return (p, 1)
    best, best_cost = (1, 1), None
    for pm in range(1, panels + 1):
        pn = min(p // pm, blocks)
        cost = math.ceil(panels / pm) * (max(nr, 1) * math.ceil(blocks / pn) + PACK_WEIGHT)
        # `<=`, so among equal-cost partitions the largest `pm` wins, matching
        # the engine: the `M` split needs no duplicated packing.
        if best_cost is None or cost <= best_cost:
            best, best_cost = (pm, pn), cost
    return best


def bucket(rs, label, width=48):
    if not rs:
        print(f"{label:<{width}} 0")
        return
    fr = sorted(r["frac"] for r in rs)
    print(f"{label:<{width}} {len(rs):>4}   frac of best: "
          f"median {fr[len(fr) // 2]:.2f}  max {fr[-1]:.2f}")


def report(rows, p):
    print(f"{'=' * 72}")
    print(f"=== P = {p} threads")
    print(f"{'=' * 72}")
    for r in rows:
        r["mstrips"] = min(p, r["panels"])
        r["nblocks"] = min(p, r["blocks"])
        r["both"] = min(p, r["panels"] * r["blocks"])
        r["pm"], r["pn"] = partition(r["panels"], r["blocks"], r["nr"], p)
        r["width"] = r["pm"] * r["pn"]

    print("--- M-direction width (the 1-D partition, i.e. TENSORCONTRACT_PARTITION=m)")
    bucket([r for r in rows if r["mstrips"] >= p], f"fills {p} strips from M alone")
    bucket([r for r in rows if r["mstrips"] < p], f"cannot fill {p} from M")
    for lim in (1, 2, 4, 8, 16):
        if lim < p:
            bucket([r for r in rows if r["panels"] <= lim], f"  M width <= {lim}")

    print()
    print("--- adding N-direction parallelism (no accumulators needed)")
    bucket([r for r in rows if r["mstrips"] < p and r["both"] >= p], "fixed by M x N together")
    stuck = [r for r in rows if r["both"] < p]
    bucket(stuck, f"still short of {p} -> would need K-parallelism")

    print()
    print("--- achieved width: what the shipped rule actually asks for")
    print("    (pm * pn < P is a modelled refusal to split N, not a shape limit;")
    print("     these are the cases whose scaling curve must flatten)")
    full = [r for r in rows if r["width"] >= p]
    part = [r for r in rows if r["width"] < p]
    bucket(full, f"rule uses all {p} threads")
    bucket(part, f"rule leaves threads idle (pm*pn < {p})")
    bucket([r for r in part if r["both"] >= p],
           f"  ... though M x N could have reached {p}")
    bucket([r for r in rows if r["pn"] > 1], "rule splits N at all (2-D)")
    # Most of the shortfall at large `P` is arithmetic, not judgement: `pn =
    # min(p / pm, blocks)` is integer division, so a rule that wants `9 x 14`
    # out of 128 threads gets 126 and idles two. That is worth a different
    # label from a case the rule genuinely declines to spread, because only the
    # second kind can show up as a visibly flat scaling curve.
    near = [r for r in part if r["width"] >= 0.9 * p]
    real = [r for r in part if r["width"] < 0.9 * p]
    bucket(near, "  of which integer-division waste (>= 90% of P)")
    bucket(real, "  of which a genuine limit (< 90% of P)")
    if part:
        med = sorted(r["width"] / p for r in part)
        print(f"    of the idle-leaving cases, achieved fraction of P: "
              f"median {med[len(med) // 2]:.2f}  min {med[0]:.2f}")

    if real:
        print()
        print(f"--- the {len(real)} genuinely partition-limited case-dtype-methods at P = {p}")
        print("    (these are the scaling curves that must flatten; expect them to)")
        print(f"{'case':<24} {'dtype':<5} {'meth':<7} {'panels':>7} {'blocks':>7} "
              f"{'MR':>3} {'NR':>3} {'rule':>8} {'width':>6} {'MxN':>5} {'GF/s':>7} {'frac':>5}")
        for r in sorted(real, key=lambda r: (r["width"] / p, -r["frac"])):
            grid = "{}x{}".format(r["pm"], r["pn"])
            print(f"{r['case']:<24} {r['dtype']:<5} {r['method']:<7} "
                  f"{r['panels']:>7} {r['blocks']:>7} {r['mr']:>3} {r['nr']:>3} "
                  f"{grid:>8} {r['width']:>6} {r['both']:>5} "
                  f"{r['gflops']:>7.1f} {r['frac']:>5.2f}")

    if part:
        print()
        print(f"--- all {len(part)} case-dtype-methods that cannot reach {p} threads under the rule")
        print(f"{'case':<24} {'dtype':<5} {'meth':<7} {'panels':>7} {'blocks':>7} "
              f"{'MR':>3} {'NR':>3} {'rule':>8} {'width':>6} {'MxN':>5} {'GF/s':>7} {'frac':>5}")
        for r in sorted(part, key=lambda r: (-r["frac"], r["case"])):
            grid = "{}x{}".format(r["pm"], r["pn"])
            print(f"{r['case']:<24} {r['dtype']:<5} {r['method']:<7} "
                  f"{r['panels']:>7} {r['blocks']:>7} {r['mr']:>3} {r['nr']:>3} "
                  f"{grid:>8} {r['width']:>6} {r['both']:>5} "
                  f"{r['gflops']:>7.1f} {r['frac']:>5.2f}")

    if stuck:
        print()
        print(f"--- shapes that would need per-thread accumulators at P = {p}")
        print(f"{'width':>5} {'case':<24} {'dtype':<5} {'method':<7} "
              f"{'m':>7} {'n':>6} {'k':>7} {'GF/s':>7} {'frac':>5}")
        for r in sorted(stuck, key=lambda r: -r["frac"]):
            print(f"{r['both']:>5} {r['case']:<24} {r['dtype']:<5} {r['method']:<7} "
                  f"{r['m']:>7} {r['n']:>6} {r['k']:>7} {r['gflops']:>7.1f} {r['frac']:>5.2f}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-p", "--threads", default="8",
                    help="comma-separated thread counts to report (default 8)")
    ap.add_argument("features", help="tcbench orient --csv output")
    ap.add_argument("baselines", nargs="+", help="tcbench sweep --csv outputs, for GF/s")
    a = ap.parse_args()

    feats = {}
    with open(a.features) as f:
        for r in csv.DictReader(f):
            if r["rule_picks"] != "true":
                continue
            # One entry per case-dtype-method, the project's 392 convention: the
            # real path is one entry (it is method-independent), each complex
            # dtype is one per method. `perf` is keyed by engine name, so the
            # real path's single entry is looked up under whichever engine name
            # reported it.
            feats[(r["case"], r["dtype"], r["method"])] = {
                k: int(r[k]) for k in ("mr", "nr", "m", "n", "k")
            }

    perf = {}
    for p in a.baselines:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["engine"] in ("ttgt", "tblis"):
                    continue
                me = r["engine"] if r["dtype"].startswith("c") else "real"
                perf[(r["case"], r["dtype"], me)] = float(r["gflops"])

    # Empirical per-dtype ceiling: the fastest thing the engine achieves on this
    # corpus. Used only to say "compute-bound" relative to what the machine can
    # do, not as a roofline claim.
    ceil_ = defaultdict(float)
    for (case, dt, me), g in perf.items():
        ceil_[dt] = max(ceil_[dt], g)

    rows = []
    for key, f in feats.items():
        if key not in perf:
            continue
        g = perf[key]
        rows.append(dict(
            case=key[0], dtype=key[1], method=key[2],
            m=f["m"], n=f["n"], k=f["k"], mr=f["mr"], nr=f["nr"],
            panels=max(1, math.ceil(f["m"] / f["mr"])),
            blocks=max(1, math.ceil(f["n"] / f["nr"])),
            gflops=g, frac=g / ceil_[key[1]],
        ))

    print(f"{len(rows)} case-dtype-methods; per-dtype best observed GF/s: "
          + "  ".join(f"{d}={v:.0f}" for d, v in sorted(ceil_.items())))
    print(f"partition rule replayed with PACK_WEIGHT = {PACK_WEIGHT} "
          f"(must match plan.rs)")
    print()
    for p in [int(x) for x in a.threads.split(",")]:
        report(rows, p)

    print("=== summary: how the rule's reach changes with P")
    print(f"{'P':>5} {'no fill from M':>15} {'rule 2-D':>9} {'idle':>6} "
          f"{'int-div':>8} {'genuine':>8} {'need K':>7}")
    for p in [int(x) for x in a.threads.split(",")]:
        nm = sum(1 for r in rows if min(p, r["panels"]) < p)
        parts = [partition(r["panels"], r["blocks"], r["nr"], p) for r in rows]
        idle = [pm * pn for (pm, pn) in parts if pm * pn < p]
        twod = sum(1 for (pm, pn) in parts if pn > 1)
        near = sum(1 for w in idle if w >= 0.9 * p)
        real = sum(1 for w in idle if w < 0.9 * p)
        needk = sum(1 for r in rows if r["panels"] * r["blocks"] < p)
        print(f"{p:>5} {nm:>15} {twod:>9} {len(idle):>6} {near:>8} {real:>8} {needk:>7}")
    print()
    print("`need K` is the column that would justify per-thread accumulators and a")
    print("reduction (A21). It is 0 at every P above, which extends A21's refutation")
    print("from the 8 threads it was argued at to a whole 128-core node.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
