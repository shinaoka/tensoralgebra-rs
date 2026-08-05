#!/bin/bash
# Phase 4 item 1c: the micro-tile row block, measured as a grid.
#
# `MR` is not only a kernel constant. It is the granularity at which the
# output's row scatter is blocked, so it decides which of the write-back's three
# paths each block takes, and — through `Plan::transposes_gemm` — it is also an
# input to the row/column orientation. `tcbench shapes` says 106 of 392
# case-dtype-methods would run a different shape under the rule, so the question
# is not whether the mechanism fires but whether firing helps.
#
# Design, following scripts/phase4-remeasure.sh:
#   * Every arm is a runtime environment switch, never a rebuild, so the arms
#     interleave inside one session (A15).
#   * The committed behaviour (`base`) runs first and last, so a repeat brackets
#     the treatments and gives the noise floor for *this* session.
#   * `idx=N` pins the Nth shape on each kernel menu rather than a bare `MR`,
#     because `mr=16` names different shapes in `f32` and `f64` while `idx=1`
#     means "the first alternate" in both. Which shape that is per dtype and
#     method is recorded in bench-results/phase4c/shapes.csv and in each row's
#     `notes` column, so the grid can be re-read later without guessing.
#   * `f64` is a built-in control group: no shape on any `f64` menu differs from
#     the default in what the rule scores, so `auto` cannot move it. Any `f64`
#     movement between arms is noise by construction.
#   * Per-CPU occupancy of the pinned core and its hyperthread sibling is
#     recorded next to every result. The sibling shares L1d and L2, which is
#     what all of this turns on.
#
# Usage: scripts/phase4c-rowblock.sh [cpu] [outdir] [size_mib] [reps] [filter]
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CPU=${1:-4}
OUT=${2:-bench-results/phase4c}
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

snap() {
    python3 - <<'PY'
out = {}
for l in open("/proc/stat"):
    f = l.split()
    if f[0].startswith("cpu") and f[0][3:].isdigit():
        out[f[0][3:]] = [int(x) for x in f[1:]]
print(repr(out))
PY
}

run() {  # run <tag> <rowblock> <dtypes>
    local tag=$1 rb=$2 dt=$3
    local before after
    before=$(snap)
    env TENSORCONTRACT_ROWBLOCK="$rb" $B sweep \
        --size $SIZE --reps $REPS "${FILT[@]}" \
        --engines planar,1m,3m --dtype "$dt" --csv "$OUT/$tag.csv" \
        > "$OUT/$tag.txt" 2>&1
    after=$(snap)
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

# The whole menu, plus the rule, bracketed by the committed behaviour.
for pair in f64,c64 f32,c32; do
    t=${pair/,/}
    run "rb-base-$t"  base  "$pair"
    run "rb-idx1-$t"  idx=1 "$pair"
    run "rb-idx2-$t"  idx=2 "$pair"
    run "rb-auto-$t"  auto  "$pair"
    run "rb-base2-$t" base  "$pair"
    echo "done $t"
done
echo "grid complete; compare with scripts/compare-sweeps.py"
