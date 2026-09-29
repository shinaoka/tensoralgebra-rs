#!/usr/bin/env python3
"""Compare two or more `examples/kernel_shapes` runs, cell by cell.

`kernel_shapes` prints a table per ISA and a `best per method` footer, and that
footer is what gets copied into a `cfg_*` menu. A single run cannot say whether
its winner leads by more than the machine's own spread, and this repository has
already published a one-shot number that did not survive re-measurement (A46).
So: run the sweep two or three times, point this at all of them, and read three
things.

  * **The floor.** The spread of each cell across the arms. Reported as a
    median and a p90; the p90 is what a lead is scored against, because the
    median flatters a distribution with a bimodal tail.
  * **Unstable shapes.** A cell whose arms disagree by more than `--unstable`
    (default 5%) is not a noisy measurement of one number, it is a shape that
    does two different things — spilling that flips, most often, and every one
    seen so far was flagged `!` over the register budget. Such a shape is
    excluded from the floor and **must not be a menu default at any mean**,
    which is a stronger statement than "it measured slower".
  * **The margin.** How far each method's winner leads its best stable rival,
    in units of the floor. Inside the floor is a tie, and a tie is where a
    secondary criterion may pick instead of the stopwatch — a smaller `live`
    count, or an `MR` that divides the corpus's 24.

Cells are scored on the **median** across arms, so one wild arm cannot crown a
shape. Pure stdlib, no CPU cost, safe to run while a benchmark is in flight.

    scripts/kernel-shapes-compare.py A.txt B.txt C.txt
    scripts/kernel-shapes-compare.py A.txt B.txt --column 0   # the kc=16 column
"""

import argparse
import re
import statistics
import sys

# "real        4x8     16   19     10   16   1.60  1.500   52.4   55.4   55.6"
ROW = re.compile(
    r"^(real|planar|1m|3m)\s+(\d+x\d+)\s+(\d+)\s+(\d+)(!?)\s+\d+\s+\d+\s+"
    r"[\d.]+\s+[\d.]+\s+(.*)$"
)
SECTION = re.compile(r"^=== (.+?) ===")
HEADER = re.compile(r"^method\s+MRxNR.*?(GF/s.*)$")


def parse(path):
    """{section: {(method, shape): (over_budget, [gf/s per kc column])}}."""
    sections, labels, cur = {}, {}, None
    with open(path) as fh:
        for line in fh:
            m = SECTION.match(line)
            if m:
                cur = m.group(1)
                sections[cur] = {}
                continue
            m = HEADER.match(line)
            if m and cur:
                labels[cur] = m.group(1).split()
                continue
            m = ROW.match(line)
            if m and cur:
                sections[cur][(m.group(1), m.group(2))] = (
                    m.group(5) == "!",
                    [float(x) for x in m.group(6).split()],
                )
    return sections, labels


def pct(xs, q):
    """Nearest-rank percentile; `xs` need not be sorted."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * len(xs)) - 1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("arms", nargs="+", help="two or more kernel_shapes outputs")
    ap.add_argument(
        "--column",
        type=int,
        default=-1,
        help="which kc column to score; default -1, the deepest, which is what "
        "the driver actually runs and what `best per method` uses",
    )
    ap.add_argument(
        "--unstable",
        type=float,
        default=5.0,
        help="percent arm-to-arm spread above which a shape is called unstable "
        "rather than noisy, and excluded from the floor (default 5)",
    )
    args = ap.parse_args()
    if len(args.arms) < 2:
        sys.exit("need at least two arms: one run has no floor")

    parsed = [parse(p) for p in args.arms]
    base, labels = parsed[0]
    if not base:
        sys.exit(f"no tables parsed from {args.arms[0]}")

    for section in base:
        tables = [p[0][section] for p in parsed if section in p[0]]
        if len(tables) < len(parsed):
            print(f"!! {section}: absent from one arm, skipped\n")
            continue
        col = args.column
        label = labels.get(section, [])
        name = label[col] if label and -len(label) <= col < len(label) else f"col{col}"
        print(f"=== {section} — {name} ===")
        print(f"{len(parsed)} arms, scored on the per-cell median\n")

        shared = [k for k in tables[0] if all(k in t for t in tables[1:])]
        series = {k: [t[k][1][col] for t in tables] for k in shared}
        spread = {k: max(v) / min(v) - 1 for k, v in series.items()}
        med = {k: statistics.median(v) for k, v in series.items()}

        unstable = sorted(
            (k for k in shared if spread[k] * 100 > args.unstable),
            key=lambda k: -spread[k],
        )
        stable = [k for k in shared if k not in unstable]
        floor = pct([spread[k] for k in stable], 0.90)
        print(
            f"floor over {len(stable)} stable shapes: median "
            f"{statistics.median([spread[k] for k in stable]) * 100:.2f}%, "
            f"p90 {floor * 100:.2f}%   <- leads are scored against the p90"
        )
        if unstable:
            print(f"unstable (>{args.unstable:g}% between arms, NOT shippable):")
            for k in unstable:
                arms = " / ".join(f"{x:.1f}" for x in series[k])
                budget = " over budget" if tables[0][k][0] else ""
                print(f"  {k[0]:<7}{k[1]:>6}  {arms}   spread {spread[k] * 100:5.1f}%{budget}")
        print()

        for method in ("real", "planar", "1m", "3m"):
            rows = [k for k in stable if k[0] == method]
            if not rows:
                continue
            rank = sorted(rows, key=lambda k: -med[k])
            win = rank[0]
            print(f"  {method:<7} best {win[1]:>6}  {med[win]:6.1f} GF/s")
            if len(rank) > 1:
                runner = rank[1]
                lead = med[win] / med[runner] - 1
                mult = lead / floor if floor else float("inf")
                verdict = "MEASURED" if mult >= 2 else "inside the floor — a tie"
                print(
                    f"          next {runner[1]:>6}  {med[runner]:6.1f} GF/s   "
                    f"lead {lead * 100:5.2f}%  ({mult:.1f}x floor) -> {verdict}"
                )
            ties = [k for k in rank[1:] if floor and med[win] / med[k] - 1 <= floor]
            if ties:
                print("          tied within the floor: " + ", ".join(k[1] for k in ties))
            print()


if __name__ == "__main__":
    main()
