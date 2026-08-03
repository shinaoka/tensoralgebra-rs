#!/usr/bin/env python3
"""Score candidate row/column orientation rules against the forced-arm grid.

`scripts/phase4d-orient.sh` measures both arms of every corpus case in every
dtype and method, so the *right* answer is known for all of them. A candidate
rule is then a predicate over the structural features `tcbench orient` emits,
and its corpus effect can be evaluated without running anything.

This exists because the shipped rule is right on 63 of 72 measured `abcijk`
case-dtypes and nobody has found what the nine misses have in common — and
because guessing a rule and spending two hours measuring it has now failed
twice. Fitting against a table is not more rigorous than an A/B, but it is much
cheaper per candidate, so the eventual A/B is spent on a rule that already
survived every case rather than on the first plausible one.

Reported per dtype and method:

  * `rule`    — geomean over the corpus of (arm the rule picks) / (arm AB),
                i.e. what shipping that rule would be worth against never
                swapping. The shipped rule is one of the candidates.
  * `oracle`  — the same with the faster arm chosen by hindsight. This bounds
                every rule, and the gap to it is what is still unexplained.
  * `misses`  — cases where the rule picks the slower arm by more than the
                per-case noise floor, and what it leaves on the table.

Usage:
    scripts/orient-score-rules.py FEATURES_CSV RESULT_DIR
"""
import csv
import math
import sys
from collections import defaultdict

NOISE_CASE = 0.06  # measured, Phase 4 part 4


def load(paths):
    out = {}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["engine"] in ("ttgt", "tblis"):
                    continue
                out[(r["case"], r["dtype"], r["engine"])] = float(r["gflops"])
    return out


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


# ---- candidate rules -------------------------------------------------------
#
# Each takes the AB and BA feature dicts and returns True to swap.


def never(ab, ba):
    return False


def shipped(ab, ba):
    """`Plan::transposes_gemm`, read out of the engine rather than
    reimplemented — `tcbench orient` records which arm it picks, so this cannot
    drift from the code it is meant to represent."""
    return bool(ba["rule_picks"])


def abs_or_inf(s):
    return abs(s) if s != 0 else float("inf")


def more_contiguous_rows(ab, ba):
    """Condition 1 alone: swap whenever the columns are more contiguous."""
    return abs_or_inf(ba["row_stride"]) < abs_or_inf(ab["row_stride"])


def unit_rows(ab, ba):
    """Swap iff it makes the rows unit-stride and they were not."""
    return ba["row_stride"] == 1 and ab["row_stride"] != 1


def longer_col_run(ab, ba):
    """The part-4 observation: prefer the arm whose *column* direction folds
    into the longer run, so one `jr` pass walks fewer discontiguous strips."""
    return ba["col_run"] > ab["col_run"]


def unit_rows_or_longer_cols(ab, ba):
    return unit_rows(ab, ba) or (ba["row_stride"] == ab["row_stride"] and longer_col_run(ab, ba))


def max_wb(ab, ba):
    """Prefer the arm with more output row blocks off the gather path."""
    return ba["wb"] > ab["wb"] + 1e-9


def wb_then_colrun(ab, ba):
    if abs(ba["wb"] - ab["wb"]) > 1e-9:
        return ba["wb"] > ab["wb"]
    return longer_col_run(ab, ba)


def rows_unit_and_cols_long(ab, ba):
    """Both at once: the rows must be unit-stride and the columns must not get
    shorter runs by swapping."""
    return unit_rows(ab, ba) and ba["col_run"] >= ab["col_run"]


def _row_block_fits(f):
    """The micro-tile row block lands inside one run of the output's rows."""
    return f["row_stride"] == 1 and f["row_run"] >= f["mr"]


def shorter_run_rows(ab, ba):
    """Put the direction with the *shorter* run in the row role.

    Suggested by the four `abcijk` families that are exact mirror images of one
    another: for each, the faster arm is the one whose row direction has the
    shorter run, whichever of `A` and `B` that happens to be."""
    return ba["row_run"] < ab["row_run"]


def fits_else_shorter_run(ab, ba):
    """Take an arm whose row block fits inside a run; failing that, put the
    shorter-run direction in the row role.

    This is condition 2 of the shipped rule promoted from a veto to a
    *preference*, with a defined fallback for when neither arm satisfies it —
    which is the case the shipped rule handles by giving up and never swapping,
    and where all nine of its known misses live."""
    a, b = _row_block_fits(ab), _row_block_fits(ba)
    if a != b:
        return b
    if a and b:
        return False  # both work; no reason to move
    return shorter_run_rows(ab, ba)


def fits_else_unit_rows(ab, ba):
    """Same, but falling back to the shipped rule's condition 1."""
    a, b = _row_block_fits(ab), _row_block_fits(ba)
    if a != b:
        return b
    if a and b:
        return False
    return unit_rows(ab, ba)


RULES = [
    ("never swap", never),
    ("shipped (transposes_gemm)", shipped),
    ("cond 1 only: rows more contiguous", more_contiguous_rows),
    ("rows become unit-stride", unit_rows),
    ("longer column run", longer_col_run),
    ("unit rows, else longer cols", unit_rows_or_longer_cols),
    ("max wb", max_wb),
    ("wb, tie-broken by column run", wb_then_colrun),
    ("unit rows and cols no worse", rows_unit_and_cols_long),
    ("shorter run in the row role", shorter_run_rows),
    ("row block fits, else shorter run", fits_else_shorter_run),
    ("row block fits, else unit rows", fits_else_unit_rows),
]


def main():
    feats_path, d = sys.argv[1], sys.argv[2].rstrip("/")
    feats = {}
    with open(feats_path) as f:
        for r in csv.DictReader(f):
            row = {
                k: (float(v) if k in ("wb", "reg_a", "reg_b") else int(v))
                for k, v in r.items()
                if k not in ("case", "dtype", "method", "arm", "rule_picks")
            }
            row["rule_picks"] = r["rule_picks"] == "true"
            feats.setdefault((r["case"], r["dtype"], r["method"]), {})[r["arm"]] = row

    arms = {
        a: load([f"{d}/or-{a}-f64c64.csv", f"{d}/or-{a}-f32c32.csv"])
        for a in ("ab", "ba", "ab2")
    }

    dtypes = ["f32", "f64", "c32", "c64"]
    methods = ["planar", "1m", "3m"]
    cols = [(dt, me) for dt in dtypes for me in methods]

    header = f"{'rule':<36}" + "".join(f"{dt + '/' + me:>13}" for dt, me in cols)
    print(header)
    print("-" * len(header))

    def score(pred):
        cells = []
        for dt, me in cols:
            rs = []
            for (case, d2, m2), arms_f in feats.items():
                if d2 != dt:
                    continue
                if d2.startswith("c"):
                    if m2 != me:
                        continue
                elif m2 != "real":
                    continue
                key = (case, dt, me)
                if key not in arms["ab"] or key not in arms["ba"]:
                    continue
                swap = pred(arms_f["AB"], arms_f["BA"]) if pred else (
                    arms["ba"][key] > arms["ab"][key]
                )
                rs.append((arms["ba"][key] if swap else arms["ab"][key]) / arms["ab"][key])
            cells.append(geomean(rs))
        return cells

    for name, pred in RULES:
        print(f"{name:<36}" + "".join(f"{c:>13.3f}" for c in score(pred)))
    print(f"{'oracle (hindsight)':<36}" + "".join(f"{c:>13.3f}" for c in score(None)))
    print(
        f"{'ab vs ab2 (noise floor)':<36}"
        + "".join(
            f"{geomean([arms['ab2'][k] / arms['ab'][k] for k in arms['ab'] if k[1] == dt and k[2] == me and k in arms['ab2']]):>13.3f}"
            for dt, me in cols
        )
    )

    # Where each rule is wrong, so the misses can be inspected rather than only
    # counted. The shipped rule's nine known misses should appear here.
    for name, pred in RULES:
        if name == "never swap":
            continue
        misses = []
        for (case, d2, m2), arms_f in feats.items():
            me = m2 if d2.startswith("c") else "planar"
            key = (case, d2, me)
            if key not in arms["ab"] or key not in arms["ba"]:
                continue
            if not d2.startswith("c") and m2 != "real":
                continue
            swap = pred(arms_f["AB"], arms_f["BA"])
            got = arms["ba"][key] if swap else arms["ab"][key]
            best = max(arms["ab"][key], arms["ba"][key])
            if best / got > 1 + NOISE_CASE:
                misses.append((best / got, case, d2, m2, "BA" if swap else "AB"))
        misses.sort(reverse=True)
        tot = sum(1 for _ in feats)
        print(f"\n{name}: {len(misses)} misses beyond noise (of {tot})")
        for r, case, dt, me, picked in misses[:8]:
            print(f"    {r:5.3f}x left on the table  {case:22} {dt} {me:6} picked {picked}")


if __name__ == "__main__":
    main()
