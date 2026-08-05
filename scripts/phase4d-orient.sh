#!/bin/bash
# Phase 4 item 1d: the row/column orientation, measured as a grid.
#
# The orientation rule is right on 63 of 72 `abcijk` case-dtypes and its nine
# misses are worth 1.18-1.47x (Phase 4 part 4); item 1c then priced them at
# about 2x and showed `MR` cannot buy them affordably (part 5). Nobody has
# found the discriminant, and the existing ground truth — rm-orient-*.csv — is
# 72 points from a single case family, which is a thin basis for fitting one.
#
# So: force both arms over the *whole* corpus in every dtype and method, the
# same way the row-block grid forced every shape. Both arms of all 392
# case-dtype-methods is a real basis, and candidate discriminants can then be
# scored against it offline for free, as many as one likes.
#
#   * TENSORCONTRACT_ORIENT=none|swap forces the arm; the rule is not consulted.
#   * TENSORCONTRACT_ROWBLOCK=base pins the micro-tile shape, because the
#     row-block rule takes the orientation as an input and would otherwise
#     confound the two. This measures the orientation alone.
#   * The `none` arm runs first and last, so a repeat brackets the treatment.
#
# Usage: scripts/phase4d-orient.sh [cpu] [outdir] [size_mib] [reps] [filter]
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CPU=${1:-4}
OUT=${2:-bench-results/phase4d}
SIZE=${3:-64}
REPS=${4:-3}
FILTER=${5:-}
SIB=$(cat /sys/devices/system/cpu/cpu$CPU/topology/thread_siblings_list)
BIN=${TC_TARGET:-target}/release/tcbench
[ -x "$BIN" ] || { echo "no $BIN in $PWD -- cargo build --release -p tensorprimitives-bench" >&2; exit 1; }
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

run() {  # run <tag> <orient> <dtypes>
    local tag=$1 or=$2 dt=$3 before after
    before=$(snap)
    env TENSORCONTRACT_ORIENT="$or" TENSORCONTRACT_ROWBLOCK=base \
        taskset -c $CPU $BIN sweep \
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

for pair in f64,c64 f32,c32; do
    t=${pair/,/}
    run "or-ab-$t"   none "$pair"
    run "or-ba-$t"   swap "$pair"
    run "or-ab2-$t"  none "$pair"
    echo "done $t"
done
echo "grid complete; score rules with scripts/orient-score-rules.py"
