#!/usr/bin/env python3
"""Compare two `tcbench sweep` runs case by case.

Every ratio quoted in the Phase 4 report came out of this. It exists because a
bare "geomean 1.09" is not interpretable without a noise floor, and because the
cases a change *cannot* touch are the cheapest noise floor available.

Usage:
    scripts/compare-sweeps.py BASE NEW [ORIENT_SRC]

BASE and NEW are comma-separated lists of sweep CSVs (pass the f64c64 and
f32c32 halves of one run together). ORIENT_SRC defaults to NEW and only
supplies the AB/BA tag the harness writes into the notes column; point it at a
run whose notes carry that tag if BASE/NEW predate it.

Reading the output:

  * The **AB/BA split** separates cases by whether the engine transposed the
    GEMM. For a change that only fires on one of them — the orientation rule,
    say — the other group is a control: the change cannot affect it, so its
    spread is drift plus contention. For a change that fires on both, such as
    the write-back fast path, there is no control and you need an explicit
    A-vs-A' run instead (see `scripts/phase4-remeasure.sh`).
  * **Noise floor on an exclusive machine is ±1.3% on the geomean and ±6% per
    case** (measured, Phase 4 part 4). A per-case ratio inside ±6% is not a
    result. A geomean inside ±1.3% is not a result.
  * Run it on an A-vs-A' pair first if you have any doubt; it should print
    1.000 and it is the only way to know what the machine was doing.
"""
import csv
import math
import sys
from collections import defaultdict


def load(paths):
    """Map (case, dtype, engine) -> (gflops, notes) over comma-separated CSVs."""
    out = {}
    for p in paths.split(","):
        with open(p) as f:
            for r in csv.DictReader(f):
                out[(r["case"], r["dtype"], r["engine"])] = (
                    float(r["gflops"]),
                    r.get("notes", ""),
                )
    return out


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    base, new = load(argv[1]), load(argv[2])
    orient = {}
    for (case, _, _), (_, notes) in load(argv[3] if len(argv) > 3 else argv[2]).items():
        for tok in notes.split():
            if tok in ("AB", "BA"):
                orient[case] = tok

    groups, rows = defaultdict(list), []
    for k in sorted(set(base) & set(new)):
        b, n = base[k][0], new[k][0]
        if b <= 0 or n <= 0:
            continue
        o = orient.get(k[0], "?")
        groups[(k[1], k[2], o)].append(n / b)
        rows.append((n / b, k[0], k[1], k[2], o, b, n))

    if not rows:
        sys.exit("no overlapping (case, dtype, engine) rows -- wrong file pair?")

    print(f"{'dtype':>5} {'engine':>7} {'or':>3} {'n':>3} {'geomean':>8} {'min':>7} {'max':>7}")
    for (d, e, o), rs in sorted(groups.items()):
        print(f"{d:>5} {e:>7} {o:>3} {len(rs):>3} {geo(rs):>8.3f} {min(rs):>7.3f} {max(rs):>7.3f}")

    rows.sort()
    outliers = [r for r in rows if r[0] < 0.94 or r[0] > 1.06]
    print(f"\n--- outside the +/-6% per-case noise floor ({len(outliers)} of {len(rows)}) ---")
    for r in outliers[:10] + ([("...",)] if len(outliers) > 20 else []) + outliers[-10:]:
        if r[0] == "...":
            print("  ...")
            continue
        print(f"{r[0]:>7.3f}  {r[1]:>18} {r[2]:>4} {r[3]:>7} {r[4]:>3}  {r[5]:>7.1f} -> {r[6]:>7.1f}")

    print("\n--- overall, all cases (what a user sees) ---")
    for d, e in sorted({(k[0], k[1]) for k in groups}):
        rs = [r[0] for r in rows if r[2] == d and r[3] == e]
        print(f"{d:>5} {e:>7}  {geo(rs):.3f}")


if __name__ == "__main__":
    main(sys.argv)
