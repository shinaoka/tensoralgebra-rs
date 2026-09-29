#!/usr/bin/env python3
"""Performance gate for the kernel scaling page (strided-rs#269).

Reads the CSVs written by run.sh (columns case,variant,threads,median_ns,samples),
prints one table row per (case, threads) and flags:

  (a) scaling:  a strided variant at the high thread count is not at least
                SCALING x faster than at 1 thread, for cases whose 1 thread
                median is at least MIN_SCALING_NS ("tensor sized");
  (b) erased:   an erased variant is more than ERASED x slower than typed;
  (c) raw:      a strided variant is more than RAW x slower than the raw
                baseline at the same thread count;
  (d) julia:    a strided variant is more than JULIA x slower than the
                fastest Julia variant at the same thread count.

The checks are keyed only on case and variant names, so every family,
including the ternary `ter_select_*` and `ter_clamp_*` cases (which have
only an `erased_uninit` erased row), goes through the same flags.

Exits 1 when anything is flagged, unless --report-only is given.
Standard library only.
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

STRIDED = ("typed", "erased", "erased_uninit")
RAW = "raw"


def load(paths):
    rows = {}
    for path in paths:
        with open(path, newline="") as fh:
            for rec in csv.DictReader(fh):
                key = (rec["case"], rec["variant"], int(rec["threads"]))
                rows[key] = int(rec["median_ns"])
    return rows


def fmt(ns):
    if ns is None:
        return "-"
    if ns >= 1e9:
        return f"{ns / 1e9:.2f}s"
    if ns >= 1e6:
        return f"{ns / 1e6:.2f}ms"
    return f"{ns / 1e3:.1f}us"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="CSV files or directories containing *.csv")
    ap.add_argument("--report-only", action="store_true", help="print flags but always exit 0")
    ap.add_argument("--threads", type=int, default=4, help="high thread count for the scaling check (default 4)")
    ap.add_argument("--scaling", type=float, default=1.3, help="(a) required 1T/NT speedup (default 1.3)")
    ap.add_argument("--min-scaling-ns", type=float, default=1e6,
                    help="(a) only cases whose 1T median is at least this many ns (default 1e6)")
    ap.add_argument("--erased", type=float, default=1.25, help="(b) max erased/typed ratio (default 1.25)")
    ap.add_argument("--raw", type=float, default=1.5, help="(c) max strided/raw ratio (default 1.5)")
    ap.add_argument("--julia", type=float, default=2.0, help="(d) max strided/fastest Julia ratio (default 2.0)")
    ap.add_argument("--min-ns", type=float, default=0.0,
                    help="skip (b), (c), (d) when the slower side is below this many ns (default 0)")
    ap.add_argument("--filter", default="", help="only cases containing this substring")
    args = ap.parse_args()

    paths = []
    for p in args.inputs:
        paths += sorted(glob.glob(os.path.join(p, "*.csv"))) if os.path.isdir(p) else [p]
    if not paths:
        sys.exit("no CSV files found")
    rows = load(paths)

    by = defaultdict(dict)  # (case, threads) -> variant -> ns
    for (case, variant, threads), ns in rows.items():
        if args.filter in case:
            by[(case, threads)][variant] = ns

    flags = []

    def flag(kind, case, threads, variant, msg):
        flags.append((kind, case, threads, variant, msg))

    for (case, threads), v in sorted(by.items()):
        julia = {k: ns for k, ns in v.items() if k.startswith("julia")}
        fastest_julia = min(julia.items(), key=lambda kv: kv[1]) if julia else None
        for s in STRIDED:
            if s not in v:
                continue
            ns = v[s]
            if s != "typed" and "typed" in v and ns >= args.min_ns:
                r = ns / v["typed"]
                if r > args.erased:
                    flag("b", case, threads, s, f"{s}/typed = {r:.2f} > {args.erased}")
            if RAW in v and ns >= args.min_ns:
                r = ns / v[RAW]
                if r > args.raw:
                    flag("c", case, threads, s, f"{s}/raw = {r:.2f} > {args.raw}")
            if fastest_julia and ns >= args.min_ns:
                r = ns / fastest_julia[1]
                if r > args.julia:
                    flag("d", case, threads, s, f"{s}/{fastest_julia[0]} = {r:.2f} > {args.julia}")
        if threads == args.threads:
            one = by.get((case, 1), {})
            for s in STRIDED:
                if s in v and s in one and one[s] >= args.min_scaling_ns:
                    sp = one[s] / v[s]
                    if sp < args.scaling:
                        raw_sp = (one[RAW] / v[RAW]) if RAW in one and RAW in v else None
                        ctx = f", raw scales {raw_sp:.2f}x" if raw_sp else ""
                        flag("a", case, threads, s,
                             f"{s} 1T/{threads}T = {sp:.2f} < {args.scaling}{ctx}")

    variants = sorted({var for v in by.values() for var in v},
                      key=lambda x: (not x == RAW, x not in STRIDED, x))
    flagged_keys = defaultdict(set)
    for kind, case, threads, _, _ in flags:
        flagged_keys[(case, threads)].add(kind)

    header = ["case", "T"] + variants + ["flags"]
    table = [header]
    for (case, threads), v in sorted(by.items()):
        table.append([case, str(threads)] + [fmt(v.get(var)) for var in variants]
                     + ["".join(sorted(flagged_keys.get((case, threads), ())))])
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    for i, r in enumerate(table):
        print("  ".join(c.ljust(w) if j < 1 else c.rjust(w) for j, (c, w) in enumerate(zip(r, widths))))
        if i == 0:
            print("  ".join("-" * w for w in widths))

    print()
    print(f"thresholds: (a) scaling >= {args.scaling}x at {args.threads}T when 1T >= {fmt(args.min_scaling_ns)}; "
          f"(b) erased/typed <= {args.erased}; (c) strided/raw <= {args.raw}; (d) strided/julia <= {args.julia}")
    if flags:
        print(f"{len(flags)} flag(s):")
        for kind, case, threads, _, msg in sorted(flags):
            print(f"  ({kind}) {case} {threads}T: {msg}")
    else:
        print("no flags")
    if flags and not args.report_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
