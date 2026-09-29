#!/usr/bin/env python3
"""Decompose a row-block grid into "what the shape costs" and "what it buys".

The pinned arms of `scripts/phase4c-rowblock.sh` force *every* case onto a
given shape, not only the cases a rule would move. That is what makes the grid
worth more than an A/B: for each (dtype, method, shape) it splits the corpus
into

  * cases where the shape changes nothing the write-back can see (`wb` equal to
    the default's) — their ratio is the shape's own cost, and
  * cases where it does (`wb` strictly better) — their ratio is that cost plus
    whatever the code-path change buys.

A shape is worth selecting exactly when the second group beats the first by
more than the noise floor, and a *rule* is worth shipping only if it fires on
that group and not the other.

Usage:
    scripts/rowblock-decompose.py SHAPES_CSV BASE_CSVS ARM_CSVS ARM_NAME
"""
import csv
import math
import sys
from collections import defaultdict

NOISE_GEOMEAN = 0.013  # measured, Phase 4 part 4


def load_times(paths):
    out = {}
    for p in paths.split(","):
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["engine"] in ("ttgt", "tblis"):
                    continue
                out[(r["case"], r["dtype"], r["engine"])] = (
                    float(r["gflops"]),
                    r.get("notes", ""),
                )
    return out


def load_shapes(path):
    """(case, dtype, method) -> list of shape dicts, menu order."""
    out = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["case"], r["dtype"], r["method"])].append(r)
    return out


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    shapes_path, base_paths, arm_paths, arm = sys.argv[1:5]
    shapes = load_shapes(shapes_path)
    base = load_times(base_paths)
    new = load_times(arm_paths)

    # Which menu position this arm pins to; `auto` follows the rule instead.
    idx = None
    if arm.startswith("idx="):
        idx = int(arm[4:])

    print(f"{'dtype':>5} {'method':>7} {'shape':>8} {'group':>9} {'n':>4} "
          f"{'geomean':>8} {'min':>7} {'max':>7}")
    rows = defaultdict(list)
    for (case, dtype, engine), (g0, _) in base.items():
        if (case, dtype, engine) not in new:
            continue
        g1 = new[(case, dtype, engine)][0]
        method = engine if dtype.startswith("c") else "real"
        menu = shapes.get((case, dtype, method))
        if not menu:
            continue
        default = menu[0]
        if idx is not None:
            want = menu[idx] if idx < len(menu) else default
        else:
            want = next((s for s in menu if s["chosen"] == "true"), default)
        moved = want["mr"] != default["mr"]
        better_wb = float(want["wb"]) > float(default["wb"]) + 1e-9
        if not moved:
            group = "unmoved"
        elif better_wb:
            group = "wb-better"
        else:
            group = "wb-same"
        label = f"{want['mr']}x{want['nr']}" if moved else "-"
        rows[(dtype, method, label, group)].append((g1 / g0, case))

    for key in sorted(rows):
        dtype, method, label, group = key
        rs = [r for r, _ in rows[key]]
        print(f"{dtype:>5} {method:>7} {label:>8} {group:>9} {len(rs):>4} "
              f"{geomean(rs):>8.3f} {min(rs):>7.3f} {max(rs):>7.3f}")

    # The headline: for every (dtype, method), does the moved group beat the
    # unmoved one by more than the noise floor? That is the whole question.
    print("\nselective verdict (moved-group geomean / same-shape cost):")
    seen = set()
    for (dtype, method, label, group) in sorted(rows):
        if group != "wb-better" or (dtype, method, label) in seen:
            continue
        seen.add((dtype, method, label))
        gain = geomean([r for r, _ in rows[(dtype, method, label, group)]])
        cost_key = (dtype, method, label, "wb-same")
        unmoved_key = (dtype, method, "-", "unmoved")
        cost_rs = rows.get(cost_key) or rows.get(unmoved_key) or []
        cost = geomean([r for r, _ in cost_rs]) if cost_rs else float("nan")
        verdict = "TAKE" if gain > 1 + NOISE_GEOMEAN else (
            "reject" if gain < 1 - NOISE_GEOMEAN else "wash")
        print(f"  {dtype:>4} {method:>7} -> {label:>7}: moved {gain:.3f} "
              f"(cost elsewhere {cost:.3f})  {verdict}")


if __name__ == "__main__":
    main()
