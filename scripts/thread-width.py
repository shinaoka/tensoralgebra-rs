#!/usr/bin/env python3
"""How much parallel width does the corpus actually have, and where?

The driver threads over `M` only, so its parallelism is capped at `ceil(M / MR)`
strips. This answers, from committed data and with no CPU cost, whether that cap
bites and what it would take to lift it: for each case-dtype-method it reports
the M width on the arm the orientation rule actually picks, the width an
N-direction partition would add (`ceil(N / NR)`), and how fast the case currently
runs.

The question behind it is which parallel axis a case needs. `M` and `N` splits
both give every output element a single owning thread, so they keep the result
bitwise identical to serial and need no reduction. A `K` split does not: it needs
per-thread accumulators and a reduction, which is a different and much more
invasive feature. So the thing worth knowing is whether any case is left short of
the target width once both `M` and `N` are available — and, if so, whether those
cases are compute-bound, since that is the only regime where the extra machinery
would pay.

Answered on 2026-08-03 (see the Phase 4 report, part 8): **none are**. 20 of 392
case-dtype-methods cannot fill 8 threads from `M` alone, all 20 reach 8 with `M x
N`, and the shapes that would need `K` are bounded away from compute-bound —
needing it means fewer than `p` micro-tiles in the whole output, and an output
that small caps arithmetic intensity at roughly `2MN/((M+N) * bytes)`.

"frac of best" is a crude compute-bound proxy: throughput as a fraction of the
fastest the engine achieves anywhere in that dtype on this corpus. It is not a
roofline, and it is not meant to be — it only has to separate 5 GF/s cases from
70 GF/s ones.

Usage: thread-width.py FEATURES_CSV BASELINE_CSV...
  e.g. scripts/thread-width.py bench-results/phase4e/features.csv \\
           bench-results/phase4/rm-A-f64c64.csv bench-results/phase4/rm-A-f32c32.csv
"""
import csv
import math
import sys
from collections import defaultdict

P = 8  # target thread count: physical cores per socket on the reference machine

feats = {}
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if r["rule_picks"] != "true":
            continue
        methods = ["planar", "1m", "3m"] if r["method"] == "real" else [r["method"]]
        for me in methods:
            feats[(r["case"], r["dtype"], me)] = {
                k: int(r[k]) for k in ("mr", "nr", "m", "n", "k")
            }

perf = {}
for p in sys.argv[2:]:
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["engine"] in ("ttgt", "tblis"):
                continue
            perf[(r["case"], r["dtype"], r["engine"])] = float(r["gflops"])

# Empirical per-dtype ceiling: the fastest thing the engine achieves on this
# corpus. Used only to say "compute-bound" relative to what the machine can do,
# not as a roofline claim.
ceil_ = defaultdict(float)
for (case, dt, me), g in perf.items():
    ceil_[dt] = max(ceil_[dt], g)

rows = []
for key, f in feats.items():
    if key not in perf:
        continue
    mstrips = min(P, math.ceil(f["m"] / f["mr"]))
    nblocks = min(P, math.ceil(f["n"] / f["nr"]))
    both = min(P, math.ceil(f["m"] / f["mr"]) * math.ceil(f["n"] / f["nr"]))
    g = perf[key]
    rows.append(
        dict(
            case=key[0], dtype=key[1], method=key[2],
            m=f["m"], n=f["n"], k=f["k"], mr=f["mr"], nr=f["nr"],
            mstrips=mstrips, nblocks=nblocks, both=both,
            gflops=g, frac=g / ceil_[key[1]],
        )
    )

print(f"{len(rows)} case-dtype-methods; per-dtype best observed GF/s: "
      + "  ".join(f"{d}={v:.0f}" for d, v in sorted(ceil_.items())))
print()

def bucket(rs, label):
    if not rs:
        print(f"{label:<44} 0")
        return
    fr = sorted(r["frac"] for rs_ in [rs] for r in rs_)
    print(f"{label:<44} {len(rs):>4}   frac of best: "
          f"median {fr[len(fr)//2]:.2f}  max {fr[-1]:.2f}")

print("=== M-direction width (what ships today) ===")
bucket([r for r in rows if r["mstrips"] >= P], f"fills {P} strips from M alone")
bucket([r for r in rows if r["mstrips"] < P], f"cannot fill {P} from M")
for lim in (1, 2, 4):
    bucket([r for r in rows if r["mstrips"] <= lim], f"  M width <= {lim}")

print()
print("=== adding N-direction parallelism (no accumulators needed) ===")
bucket([r for r in rows if r["mstrips"] < P and r["both"] >= P],
       f"fixed by M x N together")
stuck = [r for r in rows if r["both"] < P]
bucket(stuck, f"still short of {P} -> would need K-parallelism")

print()
print("=== the cases that would need per-thread accumulators ===")
if stuck:
    print(f"{'gain':>5} {'case':<24} {'dtype':<5} {'method':<7} "
          f"{'m':>7} {'n':>6} {'k':>7} {'MxN':>5} {'GF/s':>7} {'frac':>5}")
    for r in sorted(stuck, key=lambda r: -r["frac"]):
        print(f"{r['both']:>5} {r['case']:<24} {r['dtype']:<5} {r['method']:<7} "
              f"{r['m']:>7} {r['n']:>6} {r['k']:>7} {r['both']:>5} "
              f"{r['gflops']:>7.1f} {r['frac']:>5.2f}")
else:
    print("none: every corpus case reaches the target width from M x N alone.")

print()
print("=== arithmetic intensity of the M-limited cases ===")
print("flop/byte at the matrix level, 2MNK / ((MK + KN + MN) * bytes)")
print(f"{'case':<24} {'dtype':<5} {'Mw':>3} {'Nw':>3} {'flop/byte':>9} {'GF/s':>7} {'frac':>5}")
seen = set()
for r in sorted([r for r in rows if r["mstrips"] < P], key=lambda r: -r["frac"]):
    if (r["case"], r["dtype"]) in seen:
        continue
    seen.add((r["case"], r["dtype"]))
    b = 4 if r["dtype"].endswith("32") else 8
    if r["dtype"].startswith("c"):
        b *= 2
    m, n, k = r["m"], r["n"], r["k"]
    ai = 2 * m * n * k / ((m * k + k * n + m * n) * b)
    if r["dtype"].startswith("c"):
        ai *= 4  # complex MAC is 8 flops, not 2
    print(f"{r['case']:<24} {r['dtype']:<5} {r['mstrips']:>3} {r['nblocks']:>3} "
          f"{ai:>9.1f} {r['gflops']:>7.1f} {r['frac']:>5.2f}")
