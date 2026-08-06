#!/usr/bin/env python3
"""Score the thread-count amortisation guard against a committed `phase4g` grid.

No CPU cost, no data touched, exactly reproducible — safe to run while a benchmark
is in flight, and the right way to decide whether a candidate constant is worth
machine time. Same pattern as `rowblock-score-rules.py`, `orient-score-rules.py`,
`blocking-score-rules.py` and `partition-score-rule.py`: the grid was measured once,
so a rule can be scored against it offline as often as you like.

    scripts/amortise-score-rule.py bench-results/worker5139-zen2/phase4g

It reproduces the table in `docs/notebook/` part 17 (D48), which is the point: that
table was originally produced by an ad-hoc script and so could not be re-derived
from the repository. If the numbers here stop matching the ones in that part, one of
the two has drifted and the file is wrong.

# The rule

`p <= work_fmas / C`, clamped to `[1, requested]`, where `work_fmas` is `m * n * k`
weighted by real FMAs per logical MAC — 1 for a real dtype, 4 for planar and 1m, and
**3 for 3m**, which is exactly the flop saving that method exists for. Weighting
matters: at a fixed byte size a complex contraction has half the elements and four
times the arithmetic per element, so an unweighted `m * n * k` misjudges the two
domains in opposite directions. `--unweighted` shows that.

# What to read

**The `slower` column, not the geomean.** The guard's job is to make a threading
default *safe*, not optimal: it converts an order-of-magnitude regression into "no
worse than serial". A candidate that leaves any point below serial has not done its
job, however good its mean looks.

And read it knowing what a threaded measurement can support: per-case ratios at 64
threads run p10 0.885 / p90 1.107 with tails to 1.55 on repeats of an *identical*
partition (A39). So the difference between two candidates that both reach zero
slower points is inside the floor; the difference between guard and no guard is not.

# What it deliberately cannot tell you

Whether the constant is right for *this* machine. `C` bundles the machine's FMA rate
(the spawn fraction is `S * R / C`), so it is a fitted constant like `kc`, with the
opposite sign to the usual intuition: a **faster** machine wants a **larger** one.
Scoring against a Zen2 grid says what the guard does on Zen2.

It also says nothing about the saturation above ~1 MiB, where the best thread count
is `t8`-`t32` rather than `t64`. That is a bandwidth ceiling, a different mechanism,
and larger constants score better there for a reason that has nothing to do with
spawn (A50) — which is why `--pick` reports the small sizes separately.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

# The constant the guard was calibrated to (D48). It is **not** in the engine: the
# guard was refuted (D52) and removed before 0.1.0. Kept so this script still
# reproduces the calibration table that `docs/refuted.md` cites.
SHIPPED_C = 3_000_000

FMAS_PER_MAC = {"real": 1, "planar": 4, "1m": 4, "3m": 3}


def load(directory, prefix="sm"):
    """-> {(size, case, dtype, engine): {threads: seconds}}, {key: work_fmas}."""
    times = defaultdict(dict)
    work = {}
    pat = re.compile(rf"^{re.escape(prefix)}-s([\d.]+)-t(\d+)-")
    files = sorted(glob.glob(os.path.join(directory, f"{prefix}-s*-t*-*.csv")))
    if not files:
        sys.exit(f"no {prefix}-s*-t*-*.csv under {directory} -- is that a phase4g dir?")
    for path in files:
        m = pat.match(os.path.basename(path))
        if not m:
            continue
        size, threads = float(m.group(1)), int(m.group(2))
        for row in csv.DictReader(open(path)):
            if row["engine"] in ("ttgt", "tblis"):
                continue
            key = (size, row["case"], row["dtype"], row["engine"])
            times[key][threads] = float(row["seconds"])
            macs = int(row["macs"])
            weight = 1 if not row["dtype"].startswith("c") else FMAS_PER_MAC[row["engine"]]
            work[key] = macs * weight
    return times, work


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def score(times, work, c, requested, widths, weighted=True):
    """-> {size: (geomean speedup, points slower than serial, worst, capped_below)}."""
    out = {}
    for size in sorted({k[0] for k in times}):
        speedups, slower, worst, capped = [], 0, math.inf, 0
        for key in (k for k in times if k[0] == size):
            per = times[key]
            serial = per.get(1)
            if not serial:
                continue
            w = work[key] if weighted else work[key] // FMAS_PER_MAC.get(key[3], 1)
            cap = max(1, min(requested, w // c)) if c else requested
            usable = [t for t in widths if t <= cap and t in per]
            chosen = max(usable) if usable else 1
            if chosen < requested:
                capped += 1
            ratio = serial / per[chosen]
            speedups.append(ratio)
            worst = min(worst, ratio)
            # 0.97 rather than 1.0: a point one noise-width under serial is not
            # evidence of a loss at these thread counts (A39).
            if ratio < 0.97:
                slower += 1
        if speedups:
            out[size] = (geomean(speedups), slower, worst, capped, len(speedups))
    return out


def oracle(times):
    out = {}
    for size in sorted({k[0] for k in times}):
        best = []
        for key in (k for k in times if k[0] == size):
            per = times[key]
            if 1 in per:
                best.append(max(per[1] / per[t] for t in per))
        if best:
            out[size] = geomean(best)
    return out


def check_shipped_constant():
    """The engine no longer carries this constant, so there is nothing to check.

    The guard was measured (D52) and does not ship; `TENSORCONTRACT_AMORTISE` and
    `MIN_FMAS_PER_THREAD` were removed from `plan.rs` before 0.1.0. This script is
    kept because it re-derives D48's calibration table from committed data, which
    is what makes the refutation in `docs/refuted.md` checkable. `SHIPPED_C` below is
    the value the guard *would* have shipped with, not a value in the engine.
    """
    return


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", help="a phase4g output directory")
    ap.add_argument("--prefix", default="sm", help="arm tag prefix (default sm, i.e. `base`)")
    ap.add_argument("-p", "--requested", type=int, default=64, help="threads asked for")
    ap.add_argument(
        "--candidates",
        default="0,1e6,2e6,3e6,4e6,6e6,8e6,16e6",
        help="comma-separated C values; 0 means no guard",
    )
    ap.add_argument("--unweighted", action="store_true", help="score on m*n*k, not real FMAs")
    args = ap.parse_args()

    check_shipped_constant()
    times, work = load(args.directory, args.prefix)
    sizes = sorted({k[0] for k in times})
    widths = sorted({t for per in times.values() for t in per})
    baseline = score(times, work, 0, args.requested, widths)
    npoints = max((v[4] for v in baseline.values()), default=0)

    print(f"grid      : {args.directory} (prefix {args.prefix!r})")
    print(f"sizes     : {', '.join(str(s) for s in sizes)} MiB")
    print(f"widths    : {', '.join(str(t) for t in widths)}")
    print(f"points    : {npoints} case-dtype-methods per size")
    print(f"requested : {args.requested} threads")
    print(f"work      : {'m*n*k (UNWEIGHTED)' if args.unweighted else 'real FMAs (1 real, 4 planar/1m, 3 for 3m)'}")
    print()
    print("geomean speedup against serial / points slower than serial / worst point")
    print()

    header = f"{'C':>10}" + "".join(f"{f'{s} MiB':>26}" for s in sizes)
    print(header)
    print("-" * len(header))
    for spec in args.candidates.split(","):
        c = int(float(spec))
        res = score(times, work, c, args.requested, widths, not args.unweighted)
        label = "no guard" if c == 0 else f"{c / 1e6:g}e6"
        if c == SHIPPED_C:
            label += " *"
        row = f"{label:>10}"
        for s in sizes:
            g, slower, worst, _capped, _n = res[s]
            row += f"{f'{g:.2f} / {slower:>3} / {worst:.3f}':>26}"
        print(row)
    orc = oracle(times)
    print(f"{'oracle':>10}" + "".join(f"{f'{orc[s]:.2f} /   - /     -':>26}" for s in sizes))
    print()
    print("* = the constant the engine ships (plan.rs MIN_FMAS_PER_THREAD).")
    print("`oracle` picks the best measured width per point in hindsight: the ceiling.")
    print()
    print("Read the middle number first. A candidate that leaves any point slower than")
    print("serial has not done the guard's job, whatever its mean looks like. Candidates")
    print("that all reach 0 differ by less than a 64-thread measurement can resolve (A39).")
    print()
    res = score(times, work, SHIPPED_C, args.requested, widths, not args.unweighted)
    for s in sizes:
        _g, _sl, _w, capped, n = res[s]
        print(f"  at C = {SHIPPED_C/1e6:g}e6, {s:>5} MiB: capped below {args.requested} on {capped} of {n} points")
    print()
    print("Near-inertness at the large sizes is intended, not a shortfall: above ~1 MiB")
    print("the limit is a bandwidth ceiling, not spawn, and fitting one constant to two")
    print("mechanisms is how the analytical blocking model lost (A50, A33).")


if __name__ == "__main__":
    main()
