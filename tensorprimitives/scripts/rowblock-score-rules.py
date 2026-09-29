#!/usr/bin/env python3
"""Score candidate row-block rules offline against the measured grid.

`scripts/phase4c-rowblock.sh` pins every shape on every menu, so for each
(case, dtype, method) the throughput of *every* choice is known. A rule is then
just a function from the shape metadata to a menu index, and its corpus effect
can be evaluated without running anything — which is the only honest way to
compare rules once it is clear that the obvious one does not work.

Ratios are against the `base` arm; `base` vs `base'` bounds what "no effect"
looks like in the same session.

Usage:
    scripts/rowblock-score-rules.py SHAPES_CSV RESULT_DIR
"""
import csv
import math
import sys
from collections import defaultdict

SHALLOW_K = 32  # see the Phase 4.1c report on how weakly the corpus resolves this


def load(paths):
    out = {}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["engine"] in ("ttgt", "tblis"):
                    continue
                out[(r["case"], r["dtype"], r["engine"])] = (float(r["gflops"]), int(r["k"]))
    return out


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


# ---- candidate rules -------------------------------------------------------
#
# Each takes (menu, default, k) and returns the index it would select.
# `menu` entries carry mr, nr, wb, reg_a, orient, in menu order.


def rule_base(menu, k):
    return 0


def rule_max_wb(menu, k):
    """The first thing anyone would write: maximise write-back regularity."""
    best = 0
    for i, s in enumerate(menu):
        if s["wb"] > menu[best]["wb"] + 1e-9:
            best = i
    return best


def rule_max_wb_shallow(menu, k):
    """...but only where the write-back is not amortised by depth."""
    return rule_max_wb(menu, k) if k <= SHALLOW_K else 0


def rule_full_regularity(menu, k):
    """Only accept a shape that reaches *full* regularity from less."""
    if k > SHALLOW_K or menu[0]["wb"] >= 1.0 - 1e-9:
        return 0
    for i, s in enumerate(menu):
        if s["wb"] >= 1.0 - 1e-9:
            return i
    return 0


def rule_full_regularity_same_orient(menu, k):
    """As above, and refuse to change the orientation as a side effect.

    Changing `MR` can flip `Plan::transposes_gemm`, and the orientation is worth
    far more than the write-back path is — so a shape change that moves it is
    making a decision it has no evidence for.
    """
    if k > SHALLOW_K or menu[0]["wb"] >= 1.0 - 1e-9:
        return 0
    for i, s in enumerate(menu):
        if s["wb"] >= 1.0 - 1e-9 and s["orient"] == menu[0]["orient"]:
            return i
    return 0


def rule_full_reg_and_rega(menu, k):
    """Full regularity, same orientation, and no loss of packing regularity."""
    if k > SHALLOW_K or menu[0]["wb"] >= 1.0 - 1e-9:
        return 0
    for i, s in enumerate(menu):
        if (
            s["wb"] >= 1.0 - 1e-9
            and s["orient"] == menu[0]["orient"]
            and s["reg_a"] >= menu[0]["reg_a"] - 1e-9
        ):
            return i
    return 0


def rule_oracle(menu, k, times=None):
    """Upper bound: the fastest shape, chosen with hindsight."""
    return max(range(len(menu)), key=lambda i: times[i])


RULES = [
    ("base (committed)", rule_base),
    ("max wb", rule_max_wb),
    ("max wb, k<=32", rule_max_wb_shallow),
    ("wb->1.0, k<=32", rule_full_regularity),
    ("wb->1.0, k<=32, same orient", rule_full_regularity_same_orient),
    ("wb->1.0, k<=32, same orient, regA kept", rule_full_reg_and_rega),
]


def main():
    shapes_path, d = sys.argv[1], sys.argv[2].rstrip("/")
    shapes = defaultdict(list)
    with open(shapes_path) as f:
        for r in csv.DictReader(f):
            shapes[(r["case"], r["dtype"], r["method"])].append(
                {
                    "mr": int(r["mr"]),
                    "nr": int(r["nr"]),
                    "wb": float(r["wb"]),
                    "reg_a": float(r["reg_a"]),
                    "orient": r["orient"],
                }
            )
    arms = {
        a: load([f"{d}/rb-{a}-f64c64.csv", f"{d}/rb-{a}-f32c32.csv"])
        for a in ("base", "idx1", "idx2", "base2")
    }
    # Menu index -> arm. idx>=3 was not measured; treat it as unavailable.
    arm_of = {0: "base", 1: "idx1", 2: "idx2"}

    dtypes = ["f32", "f64", "c32", "c64"]
    methods = ["planar", "1m", "3m"]

    header = f"{'rule':<40}" + "".join(f"{d + '/' + m:>13}" for d in dtypes for m in methods)
    print(header)
    print("-" * len(header))

    for name, rule in RULES + [("oracle (hindsight)", None)]:
        cells = []
        for dt in dtypes:
            for me in methods:
                ratios = []
                for (case, d2, m2), menu in shapes.items():
                    if d2 != dt:
                        continue
                    if (m2 == "real") != (not dt.startswith("c")):
                        continue
                    if dt.startswith("c") and m2 != me:
                        continue
                    key = (case, dt, me)
                    if key not in arms["base"]:
                        continue
                    b, k = arms["base"][key]
                    avail = [i for i in range(len(menu)) if i in arm_of]
                    times = [arms[arm_of[i]][key][0] for i in avail]
                    i = (
                        max(avail, key=lambda j: arms[arm_of[j]][key][0])
                        if rule is None
                        else rule([menu[j] for j in avail], k)
                    )
                    i = i if i in arm_of else 0
                    ratios.append(arms[arm_of[i]][key][0] / b)
                cells.append(geomean(ratios))
        print(f"{name:<40}" + "".join(f"{c:>13.3f}" for c in cells))

    # The same-session no-effect band.
    cells = []
    for dt in dtypes:
        for me in methods:
            ratios = [
                arms["base2"][(c, d2, me)][0] / arms["base"][(c, d2, me)][0]
                for (c, d2, m2) in shapes
                if d2 == dt
                and ((m2 == "real") == (not dt.startswith("c")))
                and (not dt.startswith("c") or m2 == me)
                and (c, d2, me) in arms["base"]
            ]
            cells.append(geomean(ratios))
    print(f"{'base vs base2 (noise floor)':<40}" + "".join(f"{c:>13.3f}" for c in cells))


if __name__ == "__main__":
    main()
