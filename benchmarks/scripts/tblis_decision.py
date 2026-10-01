#!/usr/bin/env python3
"""Apply the Phase 1e TBLIS rule to paired corpus sessions.

Primary metric (maintainer decision 2026-09-30): the workload time of each
strategy, the sum over corpus entries of `calls x execution median`
(`<pg>_exec` / `<tblis>_exec` for contract, `faer` / `tblis` for blas). The
noise of a thread count is the mean |log(total / median over sessions)| of
both strategies' totals across sessions. Switch the default to TBLIS only if
`total_pg / total_tblis` exceeds 1 + noise at every gating thread count in
every session. The unweighted geometric mean of the per-case ratios is
reported alongside; it does not decide.

    tblis_decision.py --corpus CORPUS.json SESSION_DIR... [--bin contract|blas] [--gate 1,4,8]

Non-regression mode: two *builds* instead of two strategies, one group per
variant. A group fails when its workload time is more than `--limit` (default
5%) slower than the other build's, and not explained by either side's
session-to-session noise.

    tblis_decision.py --corpus CORPUS.json --old OLD_DIR... --new NEW_DIR... \
        [--bin contract|blas] [--gate 1,4,8] [--limit 0.05]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

VARIANTS = {"contract": ("pg_exec", "tblis_exec"), "blas": ("faer", "tblis")}


def load(session: Path, name: str, threads: int) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = defaultdict(dict)
    with open(session / f"{name}-{threads}t.csv") as f:
        for r in csv.DictReader(f):
            rows[r["case"]][r["variant"]] = float(r["median_ns"])
    return rows


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def workload(rows: dict[str, dict[str, float]], calls: dict[str, int], cases: list[str], variant: str) -> float:
    """`calls x median` of one variant over `cases` -- the gating metric."""
    return sum(calls[c] * rows[c][variant] for c in cases)


def noise_of(totals: list[float]) -> float:
    """Mean |log(total / median over sessions)| of one side's sessions."""
    if len(totals) < 2:
        return 0.0
    med = statistics.median(totals)
    return math.exp(sum(abs(math.log(t / med)) for t in totals) / len(totals)) - 1


def pair_report(
    old_dirs: list[Path],
    new_dirs: list[Path],
    corpus: Path,
    binary: str,
    threads: list[int],
    limit: float,
) -> int:
    """Non-regression: no variant's workload more than `limit` slower, beyond noise."""
    calls = {e["name"]: e.get("calls") or 1 for e in json.loads(corpus.read_text())["entries"]}
    failed = False
    print(f"{'T':>3} {'group':>11} {'old':>13} {'new':>13} {'ratio':>6} {'noise':>6}  verdict")
    for t in threads:
        old = [load(d, binary, t) for d in old_dirs]
        new = [load(d, binary, t) for d in new_dirs]
        cases = sorted(set.intersection(*(set(d) for d in old + new)))
        cases = [c for c in cases if all(v in d[c] for d in old + new for v in VARIANTS[binary])]
        missing = set(calls) - set(cases)
        if missing:
            print(
                f"{t}T: {len(missing)} corpus entries missing from some session; sessions must be complete",
                file=sys.stderr,
            )
            return 2
        for variant in VARIANTS[binary]:
            old_totals = [workload(d, calls, cases, variant) for d in old]
            new_totals = [workload(d, calls, cases, variant) for d in new]
            ratio = statistics.median(new_totals) / statistics.median(old_totals)
            noise = max(noise_of(old_totals), noise_of(new_totals))
            ok = ratio <= max(1.0 + limit, 1.0 + noise)
            failed |= not ok
            print(
                f"{t:>3} {variant:>11} {statistics.median(old_totals) / 1e9:>12.3f}s "
                f"{statistics.median(new_totals) / 1e9:>12.3f}s {ratio:>6.3f} {noise:>6.3f}  "
                f"{'ok' if ok else 'SLOWER'}"
            )
    print("gate:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", type=Path)
    ap.add_argument("--corpus", type=Path, required=True, help="corpus with per-entry `calls`")
    ap.add_argument("--bin", default="contract", choices=sorted(VARIANTS))
    ap.add_argument("--gate", default="1,4,8")
    ap.add_argument("--old", nargs="+", type=Path, help="one build's result directories (pair mode)")
    ap.add_argument("--new", nargs="+", type=Path, help="the other build's result directories (pair mode)")
    ap.add_argument("--limit", type=float, default=0.05, help="allowed slowdown in pair mode")
    a = ap.parse_args(argv)
    threads = [int(t) for t in a.gate.split(",")]
    if a.old or a.new:
        if not (a.old and a.new):
            print("pair mode needs both --old and --new", file=sys.stderr)
            return 2
        if a.sessions:
            print("pair mode takes no positional session directories", file=sys.stderr)
            return 2
        return pair_report(a.old, a.new, a.corpus, a.bin, threads, a.limit)
    if not a.sessions:
        print("give at least one session directory, or --old and --new", file=sys.stderr)
        return 2
    calls = {e["name"]: e.get("calls") or 1 for e in json.loads(a.corpus.read_text())["entries"]}
    base, alt = VARIANTS[a.bin]
    switch = True
    print(f"{'T':>3} {'session':>8} {'n':>3} {'workload base':>14} {'workload tblis':>15} {'ratio':>6} {'noise':>6} {'geomean':>8}")
    for t in threads:
        data = [load(s, a.bin, t) for s in a.sessions]
        cases = sorted(set.intersection(*(set(d) for d in data)))
        cases = [c for c in cases if all(base in d[c] and alt in d[c] for d in data)]
        missing = set(calls) - set(cases)
        if missing:
            print(f"{t}T: {len(missing)} corpus entries missing from some session; sessions must be complete", file=sys.stderr)
            return 2
        totals = [(sum(calls[c] * d[c][base] for c in cases), sum(calls[c] * d[c][alt] for c in cases)) for d in data]
        devs = []
        for k in (0, 1):
            med = statistics.median(x[k] for x in totals)
            devs += [abs(math.log(x[k] / med)) for x in totals]
        noise = math.exp(sum(devs) / len(devs)) - 1 if len(data) > 1 else 0.0
        for s, d, (tb, ta) in zip(a.sessions, data, totals):
            gm = geomean([d[c][base] / d[c][alt] for c in cases])
            ratio = tb / ta
            ok = ratio > 1 + noise
            switch &= ok
            print(f"{t:>3} {s.name:>8} {len(cases):>3} {tb / 1e9:>13.3f}s {ta / 1e9:>14.3f}s {ratio:>6.2f} {noise:>6.3f} {gm:>8.3f}{'' if ok else '  (keep)'}")
    print("decision:", "switch the default to TBLIS" if switch else "keep the current default")
    return 0


if __name__ == "__main__":
    sys.exit(main())
