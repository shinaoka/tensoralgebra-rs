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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--corpus", type=Path, required=True, help="corpus with per-entry `calls`")
    ap.add_argument("--bin", default="contract", choices=sorted(VARIANTS))
    ap.add_argument("--gate", default="1,4,8")
    a = ap.parse_args(argv)
    calls = {e["name"]: e.get("calls") or 1 for e in json.loads(a.corpus.read_text())["entries"]}
    base, alt = VARIANTS[a.bin]
    threads = [int(t) for t in a.gate.split(",")]
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
