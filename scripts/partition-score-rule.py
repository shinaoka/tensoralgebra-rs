#!/usr/bin/env python3
"""Score the domain-aware partition gate against a committed threading grid.

`Plan::partition`'s `panels >= p` early return is right where the threads share
one L3 and wrong where they span many (A36). The gate that fixes it is a pure
function of four numbers, and the sessions in `bench-results/` already measured
both of the arms it chooses between — the rule's own partition and a forced 1-D
`N` — on every case, at several thread counts, on two topologies. So the gate can
be scored exactly, offline, for free, before anyone books a node.

This is the `rowblock-score-rules.py` / `orient-score-rules.py` pattern applied to
the partition: sweep the grid once, score candidate rules against it as many times
as you like. What it reports per (family, dtype) is the geomean of the *measured*
arm the gate would have selected against the arm the shipped rule selected — so a
number above 1 is a predicted win and the whole table is a prediction that the
confirmation run can be held to.

Two things it cannot do, both worth knowing before quoting it:

  * It scores only the cases where the gate changes the answer, plus the corpus
    total. Everything else is identical by construction, not measured to be equal.
  * Per-case ratios at 64 threads on this family have a repeat precision of about
    +-13% (p10-p90) with tails past +-50% — check `identical-partition repeats`
    in the output, which measures exactly that from the same data. **Only the
    per-family geomeans are worth reading**; the per-case column is there to show
    the spread, not to be quoted.

Usage: partition-score-rule.py [-p N] [-d N] [--prefix TAG] DIR
  e.g. scripts/partition-score-rule.py -p 64 -d 16 bench-results/worker5137-zen2
       scripts/partition-score-rule.py -p 32 -d 1  bench-results/worker6016-icelake

`DIR` holds `features.csv` (from `tcbench orient --csv`, generated on that machine
— the register blocks differ by ISA) and a `phase4f/` subdirectory of sweep CSVs.
`-d` is the number of L3 domains the thread set spans, i.e. what
`CacheHierarchy::l3_domains` returns there; `scripts/topology.py` prints the cores
per domain it comes from.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

# Must track `SHALLOW_K` and `columns_beat_rows` in
# `crates/tensorcontract/src/plan.rs`, which document why each term is there.
SHALLOW_K = 64


def columns_beat_rows(panels, blocks, k, p, domains):
    """The gate, replayed. Keep in step with plan.rs."""
    return panels >= p and domains > 1 and blocks >= p and k <= SHALLOW_K


def load_features(path):
    feats = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["rule_picks"] != "true":
                continue
            # The arm the orientation rule actually picks, so `m` and `n` are the
            # oriented extents and the panel/block counts are the ones the driver
            # will use. One entry per case-dtype-method (the project's 392).
            feats[(r["case"], r["dtype"], r["method"])] = {
                k: int(r[k]) for k in ("mr", "nr", "m", "n", "k")
            }
    return feats


def load_sweep(path):
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["engine"] in ("ttgt", "tblis"):
                continue
            me = r["engine"] if r["dtype"].startswith("c") else "real"
            out[(r["case"], r["dtype"], me)] = (
                float(r["gflops"]),
                r["notes"].split()[-1],
            )
    return out


def first(*paths):
    for p in paths:
        got = load_sweep(p)
        if got is not None:
            return got
    return None


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", help="a bench-results/<node>-<arch> directory")
    ap.add_argument("-p", "--threads", type=int, default=64)
    ap.add_argument("-d", "--domains", type=int, default=1,
                    help="L3 domains the thread set spans (CacheHierarchy::l3_domains)")
    ap.add_argument("--prefix", default=None,
                    help="sweep tag, default th-t<P> (whole corpus); ps-t<P> is the "
                         "family-restricted partition sweep")
    a = ap.parse_args()

    base = a.dir.rstrip("/") + "/"
    pfx = a.prefix or f"th-t{a.threads}"
    feats = load_features(base + "features.csv")

    changed = defaultdict(list)
    control = []
    n_seen = n_flip = 0
    for pair in ("f64c64", "f32c32"):
        rule = first(f"{base}phase4f/{pfx}-rule-{pair}.csv",
                     f"{base}phase4f/{pfx}-{pair}.csv")
        cols = first(f"{base}phase4f/{pfx}-n-{pair}.csv",
                     f"{base}phase4f/{pfx}-pn-{pair}.csv")
        rows = first(f"{base}phase4f/{pfx}-m-{pair}.csv",
                     f"{base}phase4f/{pfx}-pm-{pair}.csv")
        if rule is None or cols is None:
            print(f"no {pfx} rule/n arms for {pair} in {base}phase4f", file=sys.stderr)
            continue
        for key, (gf, part) in rule.items():
            f = feats.get(key)
            if f is None:
                continue
            n_seen += 1
            panels = max(1, math.ceil(f["m"] / f["mr"]))
            blocks = max(1, math.ceil(f["n"] / f["nr"]))
            # The control: an arm that produced the *same* partition as the rule
            # is a pure repeat, so its spread is this session's per-case precision
            # at this thread count. Nothing else here reports that.
            if rows is not None and key in rows and rows[key][1] == part:
                control.append(rows[key][0] / gf)
            if not columns_beat_rows(panels, blocks, f["k"], a.threads, a.domains):
                continue
            if key not in cols:
                continue
            n_flip += 1
            changed[(key[0].split("-")[0], key[1])].append((cols[key][0] / gf, key))

    print(f"{pfx}: P = {a.threads} threads over {a.domains} L3 domain(s), "
          f"{n_seen} case-dtype-methods measured")
    if control:
        c = sorted(control)
        print(f"identical-partition repeats: n={len(c)} geomean {gm(c):.3f} "
              f"p10 {c[len(c) // 10]:.3f} median {c[len(c) // 2]:.3f} "
              f"p90 {c[-1 - len(c) // 10]:.3f}   <- per-case precision, read it first")
    if not n_flip:
        print("the gate changes nothing here: it is bit-identical to the shipped rule")
        return 0

    print(f"the gate moves {n_flip} of {n_seen} onto the column axis "
          f"(1 x min(P, blocks)); predicted ratio against the shipped rule:")
    print(f"  {'family':<10}{'dtype':<6}{'n':>4}{'geomean':>9}{'min':>8}{'max':>8}")
    allv = []
    for k in sorted(changed):
        v = [x for x, _ in changed[k]]
        allv += v
        print(f"  {k[0]:<10}{k[1]:<6}{len(v):>4}{gm(v):>9.3f}{min(v):>8.3f}{max(v):>8.3f}")
    print(f"  {'ALL':<10}{'':<6}{len(allv):>4}{gm(allv):>9.3f}")
    print(f"  whole corpus, unchanged cases counted at 1.000: "
          f"{math.exp(sum(math.log(x) for x in allv) / n_seen):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
