#!/usr/bin/env python3
"""Does it matter *where* in the node a concurrent measurement arm ran?

`scripts/validate-placement.sh` runs the same arm simultaneously on every
placement slot. Every replicate computes exactly the same thing, so the spread
between them is not a property of the contraction — it is a property of position
within the node: NUMA distance, share of the memory controllers, which half of
the interconnect. This reports each replicate against the solo baseline, ordered
by slot, together with the L3 domain and NUMA node it ran on.

Two ways to read it, and both matter:

  * **The best replicate** bounds how good the placement can be. If even the best
    is well below the solo baseline, the contention is global (bandwidth,
    interconnect) and no placement fixes it.
  * **The spread** says whether the arms are interchangeable. A grid that assigns
    arms to slots arbitrarily turns any position-dependent penalty into a
    per-arm bias, which is worse than a uniform slowdown: a uniform slowdown
    cancels in the ratios the grid is scored on, a positional one does not.

Usage: placement-spread.py OUTDIR PREFIX SOLO_CSV
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict


def load(path):
    """(case, dtype, engine) -> gflops."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["engine"] in ("ttgt", "tblis"):
                continue
            out[(r["case"], r["dtype"], r["engine"])] = float(r["gflops"])
    return out


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__.strip().splitlines()[-1])
    outdir, prefix, solo_csv = sys.argv[1:4]

    solo = load(solo_csv)
    # Prefer the cumulative record: an output directory holds several
    # invocations and `arms.json` describes only the last one.
    placement = {}
    jsonl = os.path.join(outdir, "arms.jsonl")
    single = os.path.join(outdir, "arms.json")
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    a = json.loads(line)
                    placement[a["tag"]] = a
    elif os.path.exists(single):
        with open(single) as f:
            for a in json.load(f)["arms"]:
                placement[a["tag"]] = a

    reps = sorted(
        t for t in (os.path.splitext(f)[0] for f in os.listdir(outdir) if f.endswith(".csv"))
        if t.startswith(prefix) and t[len(prefix):].isdigit()
    )
    if not reps:
        print(f"no replicates matching {prefix}NN in {outdir}")
        return 1

    print(f"=== {len(reps)} concurrent replicates of the same arm, "
          f"each against {os.path.basename(solo_csv)}")
    print(f"{'slot':<8} {'cpu':>4} {'dom':>4} {'numa':>5} {'n':>4} "
          f"{'geomean':>8} {'min case':>9} {'max case':>9} {'secs':>7} {'self%':>6}")
    per_rep = {}
    for tag in reps:
        new = load(os.path.join(outdir, f"{tag}.csv"))
        keys = [k for k in solo if k in new and solo[k] > 0]
        ratios = [new[k] / solo[k] for k in keys]
        if not ratios:
            print(f"{tag:<8} no overlapping cases with the solo baseline")
            continue
        per_rep[tag] = ratios
        a = placement.get(tag, {})
        print(f"{tag:<8} {a.get('cpu', '?'):>4} {a.get('domain', '?'):>4} "
              f"{str(a.get('numa', '?')):>5} {len(ratios):>4} "
              f"{geomean(ratios):>8.3f} {min(ratios):>9.3f} {max(ratios):>9.3f} "
              f"{a.get('seconds', float('nan')):>7.0f} "
              f"{(a.get('own_busy') or float('nan')):>6.0f}")

    if per_rep:
        gms = sorted(geomean(r) for r in per_rep.values())
        print()
        print(f"across replicates: geomean-of-geomeans {geomean(gms):.3f}, "
              f"best {gms[-1]:.3f}, worst {gms[0]:.3f}, "
              f"spread {100 * (gms[-1] / gms[0] - 1):.1f}%")

        # Per-case worst placement penalty, which is the number the decision rule
        # in validate-placement.sh is stated against.
        by_case = defaultdict(list)
        for tag in per_rep:
            new = load(os.path.join(outdir, f"{tag}.csv"))
            for k in solo:
                if k in new and solo[k] > 0:
                    by_case[k].append(new[k] / solo[k])
        worst = sorted(((min(v), k) for k, v in by_case.items()))[:8]
        print()
        print("worst-hit cases (min over replicates):")
        for ratio, (case, dtype, engine) in worst:
            print(f"  {ratio:.3f}  {case:<24} {dtype:<5} {engine}")

        # NUMA is the one positional variable the placement chooses deliberately,
        # so aggregate by it rather than leaving it in the per-slot rows.
        by_numa = defaultdict(list)
        for tag, ratios in per_rep.items():
            by_numa[placement.get(tag, {}).get("numa")].append(geomean(ratios))
        if len(by_numa) > 1:
            print()
            print("by numa node: " + "  ".join(
                f"{k}:{geomean(v):.3f}(n={len(v)})"
                for k, v in sorted(by_numa.items(), key=lambda kv: (kv[0] is None, kv[0]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
