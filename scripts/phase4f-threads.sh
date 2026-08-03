#!/bin/bash
# Phase 4 item 4: thread scaling.
#
# The driver parallelises the `M` direction into strips of whole `MR` panels,
# with a shared packed-`B` panel (see the driver's module docs). This measures
# what that is worth on the corpus, and — more usefully — *where it stops*,
# since the interesting output is not the mean speedup but the population of
# cases that fail to scale and why.
#
# Design notes, and why this is not just the sweep with more cores:
#
#   * Threads run on **physical cores of one socket only** (`0-7` here: cpu16-23
#     are their hyperthread siblings and cpu8-15 the other socket). Two threads
#     sharing a core would share the L1d and L2 that the packed `A` block is
#     sized for, which measures a different question; crossing sockets would add
#     a second L3 and a NUMA hop to a run whose packed `B` panel is sized for one
#     L3. Both are worth measuring eventually, and neither is this measurement.
#   * Every arm runs on the same 8-CPU set, so the 1-thread arm is *not* the
#     committed single-core number (that one is pinned to cpu4) — it is the
#     denominator of a self-consistent ratio measured in the same session.
#   * `t1` runs first and last, bracketing the treatments, as everywhere else in
#     Phase 4.
#   * Ratios come from `scripts/compare-sweeps.py`, which already reports per-case
#     and per-dtype/method geomeans of one CSV against another. No new analysis.
#
# Expect scaling to be capped by `ceil(M / MR)` strips, so the corpus cases with
# a short row direction cannot use eight cores at all. That is a known,
# structural limit of this partition rather than a tuning failure: the fix is
# `N`-direction parallelism, and the point of this run is to size how much of the
# corpus needs it before building it.
#
# Cost: the `t1` arms dominate, so ~1 h for both dtype pairs if scaling works at
# all. Nothing else may run on the machine, and this one wants the *whole*
# socket, not one core.
#
# Usage: scripts/phase4f-threads.sh [cpuset] [outdir] [size_mib] [reps] [filter]
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CPUS=${1:-0-7}
OUT=${2:-bench-results/phase4f}
SIZE=${3:-64}
REPS=${4:-3}
FILTER=${5:-}
BIN=./target/release/tcbench
[ -x "$BIN" ] || { echo "no $BIN in $PWD -- cargo build --release -p tensorcontract-bench" >&2; exit 1; }
FILT=()
[ -n "$FILTER" ] && FILT=(--case "$FILTER")
mkdir -p "$OUT"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# Verify the cpuset is one thread per physical core: two logical CPUs sharing a
# core would silently measure hyperthread contention instead of scaling.
python3 - "$CPUS" <<'PY'
import sys, os
def expand(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
cpus = expand(sys.argv[1])
seen = {}
for c in cpus:
    with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
        key = f.read().strip()
    seen.setdefault(key, []).append(c)
bad = {k: v for k, v in seen.items() if len(v) > 1}
print(f"cpuset {sys.argv[1]}: {len(cpus)} logical CPUs on {len(seen)} physical cores")
if bad:
    print(f"REFUSING: these share a core: {bad}")
    sys.exit(1)
PY

echo "threads will run on cpus $CPUS"

run() {  # run <tag> <threads> <dtypes>
    local tag=$1 nt=$2 dt=$3
    env TENSORCONTRACT_THREADS="$nt" \
        taskset -c "$CPUS" $BIN sweep \
        --size $SIZE --reps $REPS "${FILT[@]}" \
        --engines planar,1m,3m --dtype "$dt" --csv "$OUT/$tag.csv" \
        > "$OUT/$tag.txt" 2>&1
    echo "  $(date +%H:%M:%S) $tag done"
}

for pair in f64,c64 f32,c32; do
    t=${pair/,/}
    for nt in 1 2 4 8; do
        run "th-t$nt-$t" "$nt" "$pair"
    done
    run "th-t1b-$t" 1 "$pair"   # the bracketing repeat
    echo "done $t"
done

echo
echo "scaling, each arm against the 1-thread arm:"
for pair in f64c64 f32c32; do
    for nt in 2 4 8 1b; do
        echo "== $pair t$nt"
        scripts/compare-sweeps.py "$OUT/th-t1-$pair.csv" "$OUT/th-t$nt-$pair.csv" \
            | head -20
    done
done
