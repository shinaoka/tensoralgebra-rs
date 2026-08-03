#!/usr/bin/env python3
"""Score candidate `MC`/`KC`/`NC` rules against the item 2 grid.

`scripts/phase4e-blocking.sh` measures every corpus case, in every dtype and
method, under every arm of a blocking grid: the `kc` axis with `mc`/`nc` pinned,
the same axis with them re-derived at the new depth, and `mc`/`nc` scaled around
their derived values. So for each case the best available blocking is known, and
a candidate rule — a predicate over the structural features `tcbench orient`
emits, mapping a case to an arm — can be scored without running anything.

Same pattern as `rowblock-score-rules.py` and `orient-score-rules.py`, and for
the same reason: guessing a rule and spending hours measuring it has failed
repeatedly in this phase, while scoring a hundred candidates against a grid that
already exists is free. The A/B is then spent on a rule that survived all 392
case-dtype-methods rather than on the first plausible one.

What is reported:

  * `arms`   — geomean of each arm against `base`, per dtype and method. This is
               what changing that parameter *globally* would be worth, which is
               the first question and possibly the only one that matters.
  * `rules`  — the same for per-case rules, which can beat every global arm.
  * `oracle` — per-case best arm chosen with hindsight, overall and per family.
               It bounds every rule; the gap to it is what is unexplained.
  * `noise`  — two independent estimates from inside the run itself: the three
               `base` repeats against each other, and, for the third of the
               corpus that contracts over k <= 64, the spread across pinned-`kc`
               arms that are bit-for-bit the same computation.

Usage:
    scripts/blocking-score-rules.py FEATURES_CSV RESULT_DIR
"""
import csv
import math
import sys

NOISE_CASE = 0.06  # measured, Phase 4 part 4

# Arm tags in the order `phase4e-blocking.sh` runs them, grouped by family.
KC = ["kc64", "kc128", "kc256", "kc384", "kc512"]
CK = ["ck64", "ck128", "ck256", "ck384", "ck512"]
MC = ["mc25", "mc50", "mc200", "mc400"]
NC = ["nc25", "nc400"]
BASES = ["base", "basem", "base2"]
ARMS = BASES + KC + CK + MC + NC

DTYPES = ["f32", "f64", "c32", "c64"]
METHODS = ["planar", "1m", "3m"]
COLS = [(dt, me) for dt in DTYPES for me in METHODS]


def load(paths):
    """(case, dtype, engine) -> gflops, over comma-separated CSVs."""
    out = {}
    for p in paths:
        try:
            f = open(p)
        except FileNotFoundError:
            continue
        with f:
            for r in csv.DictReader(f):
                if r["engine"] in ("ttgt", "tblis"):
                    continue
                out[(r["case"], r["dtype"], r["engine"])] = float(r["gflops"])
    return out


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def load_features(path):
    """(case, dtype, engine) -> features of the arm the shipped rule picks.

    The orientation rule decides which tensor plays the row role, so the row/
    column features of the *executed* arm are the ones a blocking rule would
    see. `tcbench orient` records which arm the rule picks, so this cannot
    drift from the engine."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["rule_picks"] != "true":
                continue
            row = {
                k: (float(v) if k in ("wb", "reg_a", "reg_b") else int(v))
                for k, v in r.items()
                if k not in ("case", "dtype", "method", "arm", "rule_picks")
            }
            if r["dtype"].startswith("c"):
                engines = [r["method"]]
            elif r["method"] == "real":
                # The real path is measured once and reported under each method
                # name, so it needs a feature row under each.
                engines = METHODS
            else:
                continue
            for me in engines:
                out[(r["case"], r["dtype"], me)] = row
    return out


# ---- candidate rules -------------------------------------------------------
#
# Each takes the feature dict of one case-dtype-method and returns the arm tag
# to use for it. Returning "base" means "leave this case alone".


def unit_rows(f):
    """Every output row block is unit-stride, so the strip of `D` a `jr` pass
    revisits is `NR` sequential runs rather than `MC` scattered lines. This is
    the gate Phase 4 part 3 added to the depth-adaptive rule, and the quantity
    that makes `MC`'s upper bound bite."""
    return f["row_stride"] == 1


def rule_base(f):
    return "base"


def rule_shallow_couple_64(f):
    return "ck64" if f["k"] <= 32 else "base"


def rule_shallow_couple_128(f):
    return "ck128" if f["k"] <= 64 else "base"


def rule_shallow_couple_unit(f):
    """Part 3's gated rule, as close as the grid can express it: widen `mc` on
    shallow contractions, but only where the output strip is streamed."""
    return "ck64" if f["k"] <= 32 and unit_rows(f) else "base"


def rule_deep_kc512(f):
    return "kc512" if f["k"] >= 256 else "base"


def rule_deep_kc128(f):
    return "kc128" if f["k"] >= 256 else "base"


def rule_wide_mc_unit(f):
    return "mc200" if unit_rows(f) else "base"


def rule_narrow_mc_strided(f):
    return "mc50" if not unit_rows(f) else "base"


def rule_wide_nc_deep(f):
    return "nc400" if f["k"] >= 256 else "base"


RULES = [
    ("base (control)", rule_base),
    ("couple kc at k<=32 -> ck64", rule_shallow_couple_64),
    ("couple kc at k<=64 -> ck128", rule_shallow_couple_128),
    ("couple, gated on unit rows", rule_shallow_couple_unit),
    ("deep k: kc512", rule_deep_kc512),
    ("deep k: kc128", rule_deep_kc128),
    ("unit rows: mc x2", rule_wide_mc_unit),
    ("strided rows: mc /2", rule_narrow_mc_strided),
    ("deep k: nc x4", rule_wide_nc_deep),
]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    feats_path, d = sys.argv[1], sys.argv[2].rstrip("/")
    feats = load_features(feats_path)
    arms = {
        a: load([f"{d}/bl-{a}-f64c64.csv", f"{d}/bl-{a}-f32c32.csv"]) for a in ARMS
    }
    present = [a for a in ARMS if arms[a]]
    missing = [a for a in ARMS if not arms[a]]
    if "base" not in present:
        sys.exit(f"no base arm in {d}")
    if missing:
        print(f"note: {len(missing)} arm(s) absent, skipped: {' '.join(missing)}\n")

    keys = [k for k in arms["base"] if k in feats]

    def col_keys(dt, me):
        return [k for k in keys if k[1] == dt and k[2] == me]

    def ratio(arm, k):
        """Arm against base for one case, or None where either is missing."""
        b = arms["base"].get(k)
        v = arms[arm].get(k)
        if not b or not v or b <= 0 or v <= 0:
            return None
        return v / b

    def score_arm(arm, dt, me):
        rs = [ratio(arm, k) for k in col_keys(dt, me)]
        return geomean([r for r in rs if r])

    header = f"{'arm':<30}" + "".join(f"{dt + '/' + me:>13}" for dt, me in COLS)
    print("=== arms, geomean against base ===\n")
    print(header)
    print("-" * len(header))
    for arm in present:
        if arm == "base":
            continue
        line = "".join(f"{score_arm(arm, dt, me):>13.3f}" for dt, me in COLS)
        print(f"{arm:<30}{line}")

    print("\n=== per-case rules, geomean against base ===\n")
    print(header)
    print("-" * len(header))
    for name, rule in RULES:
        cells = []
        for dt, me in COLS:
            rs = []
            for k in col_keys(dt, me):
                arm = rule(feats[k])
                if arm not in arms or not arms[arm]:
                    arm = "base"
                r = ratio(arm, k)
                if r:
                    rs.append(r)
            cells.append(geomean(rs))
        print(f"{name:<30}" + "".join(f"{c:>13.3f}" for c in cells))

    print("\n=== oracles (hindsight per case) ===\n")
    print(header)
    print("-" * len(header))
    for label, family in (
        ("all arms", present),
        ("kc pinned only", ["base"] + [a for a in KC if arms[a]]),
        ("kc coupled only", ["base"] + [a for a in CK if arms[a]]),
        ("mc only", ["base"] + [a for a in MC if arms[a]]),
        ("nc only", ["base"] + [a for a in NC if arms[a]]),
    ):
        cells = []
        for dt, me in COLS:
            rs = []
            for k in col_keys(dt, me):
                vs = [ratio(a, k) for a in family]
                vs = [v for v in vs if v]
                if vs:
                    rs.append(max(vs))
            cells.append(geomean(rs))
        print(f"{'oracle: ' + label:<30}" + "".join(f"{c:>13.3f}" for c in cells))

    print("\n=== noise, from inside this run ===\n")
    print(header)
    print("-" * len(header))
    for b in [a for a in BASES if arms[a] and a != "base"]:
        print(f"{b + ' vs base':<30}" + "".join(f"{score_arm(b, dt, me):>13.3f}" for dt, me in COLS))
    # Cases with k <= the shallowest pinned kc run identical computations in
    # every pinned-kc arm, so their spread is noise measured in situ.
    shallow_arms = [a for a in KC if arms[a]]
    if shallow_arms:
        cells = []
        for dt, me in COLS:
            spread = []
            for k in col_keys(dt, me):
                if feats[k]["k"] > 64:
                    continue
                vs = [ratio(a, k) for a in shallow_arms]
                vs = [v for v in vs if v]
                if len(vs) > 1:
                    spread.append(max(vs) / min(vs))
            cells.append(geomean(spread))
        print(f"{'k<=64 identical-arm spread':<30}" + "".join(f"{c:>13.3f}" for c in cells))
        print("  (max/min over the pinned-kc arms, which are the same computation there)")

    print("\n=== where the best arm is not base ===\n")
    wins = []
    for k in keys:
        b = arms["base"].get(k)
        if not b:
            continue
        best, best_arm = b, "base"
        for a in present:
            v = arms[a].get(k)
            if v and v > best:
                best, best_arm = v, a
        if best / b > 1 + NOISE_CASE:
            wins.append((best / b, k, best_arm, feats[k]["k"], feats[k]["row_stride"]))
    wins.sort(reverse=True)
    print(f"{len(wins)} of {len(keys)} case-dtype-methods gain more than the per-case noise floor")
    print(f"{'gain':>6} {'case':<24} {'dtype':<5} {'engine':<7} {'arm':<7} {'k':>6} {'row_stride':>11}")
    for r, k, arm, kk, rs in wins[:40]:
        print(f"{r:6.3f} {k[0]:<24} {k[1]:<5} {k[2]:<7} {arm:<7} {kk:>6} {rs:>11}")
    if len(wins) > 40:
        print(f"    ... and {len(wins) - 40} more")


if __name__ == "__main__":
    main()
