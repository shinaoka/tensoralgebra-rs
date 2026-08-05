#!/bin/bash
# Phase 4 re-measurement on an exclusive workstation.
#
# The Phase 4.1 numbers were taken while other work shared the machine. The
# benchmark pins to one logical CPU, but nothing stopped a co-tenant landing on
# its **hyperthread sibling**, which shares the 32 KiB L1d and 1 MiB L2 — the
# exact resources every one of these measurements is about. This script exists
# to redo the two claims that were inside the resulting error bar, and to
# replace the inferred noise floor with a measured one.
#
# Design:
#   * Runs are ordered A, B, A' so the identical repeat *brackets* the treatment
#     in time. A vs A' is the noise floor; A vs B is the effect; if the two are
#     comparable the effect is not real.
#   * reps stays at 3, matching the recorded Phase 3 sweeps, so the noise floor
#     published here applies to the numbers already quoted in DECISIONS.md.
#   * Per-CPU occupancy of the pinned core and its sibling is sampled across
#     every sweep and written next to the results, so a contended run is a
#     recorded fact rather than a silent one.
#
# Usage: scripts/phase4-remeasure.sh [cpu] [outdir] [size_mib] [reps] [filter]
#
# The last three exist so the whole thing can be smoke-tested in place
# (`scripts/phase4-remeasure.sh 4 /tmp/smoke 8 1 ijkl-imjn-lnkm`) rather than
# through a doctored copy, which is how the first version of this script came
# to be run from the wrong directory.
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CPU=${1:-4}
OUT=${2:-bench-results/phase4}
SIZE=${3:-64}
REPS=${4:-3}
FILTER=${5:-}
SIB=$(cat /sys/devices/system/cpu/cpu$CPU/topology/thread_siblings_list)
BIN=${TC_TARGET:-target}/release/tcbench
[ -x "$BIN" ] || { echo "no $BIN in $PWD -- cargo build --release -p tensorprimitives-bench" >&2; exit 1; }
B="taskset -c $CPU $BIN"
FILT=()
[ -n "$FILTER" ] && FILT=(--case "$FILTER")
mkdir -p "$OUT"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

echo "pinned to cpu$CPU; hyperthread siblings: $SIB"

# Busy fraction of the pinned core's sibling set, sampled across a command.
cpu_busy() {
    python3 - "$1" <<'PY'
import sys
cpus = sys.argv[1].split(",")
def snap():
    out = {}
    for l in open("/proc/stat"):
        f = l.split()
        if f[0].startswith("cpu") and f[0][3:].isdigit():
            out[f[0][3:]] = [int(x) for x in f[1:]]
    return out
print(repr(snap()))
PY
}

run() {  # run <tag> <env-assignments...> -- <sweep args...>
    local tag=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    local before after
    before=$(cpu_busy "$SIB")
    env "${envs[@]}" $B sweep "$@" > "$OUT/$tag.txt" 2>&1
    after=$(cpu_busy "$SIB")
    python3 - "$SIB" "$before" "$after" > "$OUT/$tag.cpu" <<'PY'
import sys, ast
cpus = sys.argv[1].split(",")
a, b = ast.literal_eval(sys.argv[2]), ast.literal_eval(sys.argv[3])
for c in cpus:
    d = [y - x for x, y in zip(a[c], b[c])]
    tot, idle = sum(d), d[3] + d[4]
    print(f"cpu{c} busy {100*(tot-idle)/tot if tot else 0:.1f}%")
PY
    echo "  $tag: $(tr '\n' ' ' < "$OUT/$tag.cpu")"
}

for pair in f64,c64 f32,c32; do
    t=${pair/,/}
    # A: the committed build.
    run "rm-A-$t"  -- --size $SIZE --reps $REPS "${FILT[@]}" \
        --engines planar,1m,3m --dtype "$pair" --csv "$OUT/rm-A-$t.csv"
    # B: write-back fast path disabled, everything else identical.
    run "rm-B-$t"  TENSORCONTRACT_WRITEBACK=gather -- \
        --size $SIZE --reps $REPS "${FILT[@]}" \
        --engines planar,1m,3m --dtype "$pair" --csv "$OUT/rm-B-$t.csv"
    # A': identical repeat of A, bracketing B.
    run "rm-A2-$t" -- --size $SIZE --reps $REPS "${FILT[@]}" \
        --engines planar,1m,3m --dtype "$pair" --csv "$OUT/rm-A2-$t.csv"
    echo "done $t"
done

# The orientation guard: the one per-case claim that was inside the old error
# bar. Targeted, so it can afford many more reps.
for mode in none swap; do
    run "rm-orient-$mode" "TENSORCONTRACT_ORIENT=$mode" -- \
        --size $SIZE --reps $((REPS * 8)) --case "${FILTER:-abcijk}" --dtype f32,f64,c32,c64 \
        --engines planar --csv "$OUT/rm-orient-$mode.csv"
done
echo "done orient"
