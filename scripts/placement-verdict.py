#!/usr/bin/env python3
"""Apply the pre-registered accept/reject rule for concurrent arm placement.

The rule is fixed in `docs/notebook/` part 10 and in the header of
`scripts/validate-placement.sh`, *before* any data existed:

  * **Accept** if placed-vs-solo lies inside the solo-vs-solo floor on the
    geometric mean, and no case moves by more than the per-case floor.
  * **Reject** otherwise.

This implements it as code so an unattended batch job can act on it and — more
importantly — so the threshold cannot drift once the numbers are visible. Both
floors are derived from *this session's* solo pair. The published ±1.3% / ±6% is
`ccqlin038`'s and is deliberately not used.

Definitions, chosen to be the strict reading rather than the convenient one:

  floor_geo  = |geomean(solo2 / solo1) - 1|
  floor_case = max over cases of |solo2/solo1 - 1|
  accept     iff |geomean(placed / solo1) - 1| <= floor_geo
             and max over cases of |placed/solo1 - 1| <= floor_case

`floor_geo` is a single number, so a placement that shifts the mean by less than
the session's own repeat drift cannot be distinguished from nothing. `floor_case`
uses the max rather than a percentile on purpose: a percentile would be a knob,
and a knob chosen after seeing the data is how a pre-registered rule stops being
one. The cost is that one flaky case can reject a placement — which is the
conservative direction, and the whole point is that contaminated data is worse
than slow data.

Every placed replicate is checked, not just the one on the reference core: the
grid assigns arms to slots arbitrarily, so a placement is only usable if *every*
slot is usable.

Exit status: 0 accept, 1 reject, 2 could not decide (missing data).

Usage: placement-verdict.py PLACEMENT_DIR [--prefix p] [--json out.json]
"""
import argparse
import csv
import json
import math
import os
import sys


def load(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["engine"] in ("ttgt", "tblis"):
                continue
            g = float(r["gflops"])
            if g > 0:
                out[(r["case"], r["dtype"], r["engine"])] = g
    return out


def ratios(base, other):
    return {k: other[k] / base[k] for k in base if k in other}


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", help="the validate-placement.sh output directory")
    ap.add_argument("--prefix", default="p", help="placed-replicate tag prefix (default p)")
    ap.add_argument("--solo", default="solo",
                    help="solo-pair tag stem, i.e. <stem>1.csv and <stem>2.csv "
                         "(default `solo`; use `mbsolo` for the memory-bound half)")
    ap.add_argument("--json", help="also write the verdict here")
    a = ap.parse_args()

    solo1 = os.path.join(a.dir, f"{a.solo}1.csv")
    solo2 = os.path.join(a.dir, f"{a.solo}2.csv")
    for p in (solo1, solo2):
        if not os.path.exists(p):
            print(f"UNDECIDED: no {p}")
            return 2
    base, rep = load(solo1), load(solo2)
    floor = ratios(base, rep)
    if not floor:
        print("UNDECIDED: the solo pair has no cases in common")
        return 2
    floor_geo = abs(geomean(list(floor.values())) - 1.0)
    floor_case = max(abs(v - 1.0) for v in floor.values())

    reps = sorted(
        t for t in (os.path.splitext(f)[0] for f in os.listdir(a.dir) if f.endswith(".csv"))
        if t.startswith(a.prefix) and t[len(a.prefix):].isdigit()
    )
    if not reps:
        print(f"UNDECIDED: no placed replicates matching {a.prefix}NN in {a.dir}")
        return 2

    print(f"this session's floor, from {len(floor)} cases of the solo pair:")
    print(f"  geomean drift   {100 * floor_geo:.2f}%")
    print(f"  worst case      {100 * floor_case:.1f}%")
    print()
    print(f"{'slot':<8} {'n':>4} {'geomean':>8} {'dev':>7} {'worst case dev':>15} {'verdict':>9}")

    worst_geo, worst_case, offenders, per_slot = 0.0, 0.0, [], {}
    for tag in reps:
        placed = load(os.path.join(a.dir, f"{tag}.csv"))
        r = ratios(base, placed)
        if not r:
            print(f"{tag:<8} no cases in common with the solo baseline")
            return 2
        gm = geomean(list(r.values()))
        dev = abs(gm - 1.0)
        cdev = max(abs(v - 1.0) for v in r.values())
        ok = dev <= floor_geo and cdev <= floor_case
        worst_geo, worst_case = max(worst_geo, dev), max(worst_case, cdev)
        per_slot[tag] = {"geomean": gm, "geo_dev": dev, "worst_case_dev": cdev, "pass": ok}
        if not ok:
            for k, v in r.items():
                if abs(v - 1.0) > floor_case:
                    offenders.append((abs(v - 1.0), tag, k, v))
        print(f"{tag:<8} {len(r):>4} {gm:>8.3f} {100 * dev:>6.2f}% "
              f"{100 * cdev:>14.1f}% {'pass' if ok else 'FAIL':>9}")

    accept = worst_geo <= floor_geo and worst_case <= floor_case
    print()
    print(f"worst slot: geomean deviation {100 * worst_geo:.2f}% against a "
          f"{100 * floor_geo:.2f}% floor; "
          f"worst case {100 * worst_case:.1f}% against {100 * floor_case:.1f}%")
    if offenders:
        print()
        print(f"cases outside the per-case floor ({len(offenders)}), worst first:")
        for dev, tag, (case, dtype, engine), v in sorted(offenders, reverse=True)[:10]:
            print(f"  {v:.3f}  {case:<24} {dtype:<5} {engine:<7} on {tag}")
    print()
    print("VERDICT: " + ("ACCEPT -- run the grid placed (`auto`)" if accept else
                         "REJECT -- the placement perturbs the measurement; "
                         "run the grid sequentially"))

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"accept": accept, "floor_geo": floor_geo,
                       "floor_case": floor_case, "worst_geo": worst_geo,
                       "worst_case": worst_case, "slots": per_slot,
                       "n_offenders": len(offenders)}, f, indent=2)
    return 0 if accept else 1


if __name__ == "__main__":
    sys.exit(main())
