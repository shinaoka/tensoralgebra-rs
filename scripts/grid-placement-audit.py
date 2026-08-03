#!/usr/bin/env python3
"""Did *where* each grid arm ran contaminate the grid's arm-vs-base ratios?

The blocking grid is scored on `arm / base`, and under concurrent placement each
arm ran on whichever L3 domain happened to be free. That is fine if position in
the node is neutral, and it is not: on `worker5040` the two sockets differed by
2.2% purely from thermal history (part 10), which is the same order as the effects
the grid is trying to resolve. So before any arm ratio is quoted, this reports:

  * **The base repeats against each other.** `base`, `basem` and `base2` are the
    same computation on different slots at different times. Their spread is the
    only floor that applies to this grid, and it bounds every arm ratio in it.
  * **Where each arm ran** — cpu, L3 domain, socket, NUMA node, and its wall-clock
    window — so an arm ratio can be checked for a socket mismatch rather than
    assumed innocent.
  * **Which arm-vs-base comparisons cross a socket boundary**, since those carry
    the offset and the same-socket ones do not.

This does not correct anything. Correcting a positional bias with a measured
offset is worse than measuring the arm again in the clean regime, and for the
traffic-sensitive arms that is what `rusty-phase4-seq.sbatch` is for. This only
says which numbers are safe to read.

Usage: grid-placement-audit.py GRID_DIR [BASE_TAG]
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

PAIRS = ("f64c64", "f32c32")
BASES = ("base", "basem", "base2")


def load(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["engine"] in ("ttgt", "tblis"):
                continue
            g = float(r["gflops"])
            if g > 0:
                out[(r["case"], r["dtype"], r["engine"])] = g
    return out


def gm(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def ratio(a, b):
    ks = [k for k in a if k in b and a[k] > 0]
    return gm([b[k] / a[k] for k in ks]), len(ks)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[-1])
    d = sys.argv[1].rstrip("/")
    base_tag = sys.argv[2] if len(sys.argv) > 2 else "base"

    arms = {}
    jsonl = os.path.join(d, "arms.jsonl")
    if os.path.exists(jsonl):
        for line in open(jsonl):
            if line.strip():
                a = json.loads(line)
                arms[a["tag"]] = a
    if not arms:
        sys.exit(f"no arms.jsonl in {d}: cannot audit placement")

    def sock(cpu, per_socket=64):
        return cpu // per_socket

    print(f"=== where each arm ran ({len(arms)} arms)")
    print(f"{'arm':<20} {'cpu':>4} {'dom':>4} {'sock':>5} {'numa':>5} {'secs':>7} {'overlap':>8}")
    for tag in sorted(arms):
        a = arms[tag]
        print(f"{tag:<20} {a['cpu']:>4} {a['domain']:>4} {sock(a['cpu']):>5} "
              f"{str(a['numa']):>5} {a['seconds']:>7.0f} {a['overlap']:>8}")

    print()
    print("=== the base repeats: the only noise floor that applies to this grid")
    for pair in PAIRS:
        present = [b for b in BASES if os.path.exists(os.path.join(d, f"bl-{b}-{pair}.csv"))]
        if len(present) < 2:
            print(f"  {pair}: only {present} present -- no floor available")
            continue
        ref = load(os.path.join(d, f"bl-{present[0]}-{pair}.csv"))
        print(f"  {pair}, against bl-{present[0]}:")
        for b in present[1:]:
            r, n = ratio(ref, load(os.path.join(d, f"bl-{b}-{pair}.csv")))
            t0, t1 = arms.get(f"bl-{present[0]}-{pair}", {}), arms.get(f"bl-{b}-{pair}", {})
            s0, s1 = sock(t0.get("cpu", 0)), sock(t1.get("cpu", 0))
            flag = "" if s0 == s1 else f"  <- crosses socket {s0}->{s1}"
            print(f"    {b:<8} {r:.4f}  (n={n}){flag}")

    print()
    print("=== arm-vs-base comparisons that cross a socket boundary")
    crossing, same = [], []
    for pair in PAIRS:
        bt = f"bl-{base_tag}-{pair}"
        if bt not in arms:
            continue
        bs = sock(arms[bt]["cpu"])
        for tag in sorted(arms):
            if not tag.endswith(pair) or tag == bt:
                continue
            (crossing if sock(arms[tag]["cpu"]) != bs else same).append(tag)
    print(f"  same socket as base: {len(same)}")
    print(f"  crossing:            {len(crossing)}")
    if crossing:
        for t in crossing:
            print(f"    {t}")
        print()
        print("  Those ratios carry the socket offset. Read them against the base-repeat")
        print("  floor above, and prefer the sequential re-run for any arm whose effect is")
        print("  the same order as that offset.")

    per_sock = defaultdict(int)
    for a in arms.values():
        per_sock[sock(a["cpu"])] += 1
    print()
    print("arms per socket: " + "  ".join(f"{k}:{v}" for k, v in sorted(per_sock.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
