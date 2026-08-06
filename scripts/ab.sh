#!/bin/bash
# A generic end-to-end A/B: `warm-up, A, B, A'`, one runtime switch as the treatment.
#
#   scripts/ab.sh OUTDIR "ENV=VAL [ENV=VAL ...]" [cpu] [size_mib] [reps] [filter]
#
# e.g. the row-block A/B that D43 made reachable:
#   scripts/ab.sh bench-results/ab-rowblock-idx3 "TENSORCONTRACT_ROWBLOCK=idx=3"
#
# `scripts/phase4-remeasure.sh` is the same pattern with its B arm hardcoded to the
# write-back experiment; its output is committed and cited, so it is left alone and
# this is the reusable form.
#
# Three things it does that the older script does not, each a lesson from the Rusty
# sessions:
#
#   * **A discarded warm-up arm first (A31).** `A, B, A'` cannot tell a cold-start
#     transient from a noise floor: on an idle machine the opening arm runs at
#     single-core boost and nothing else does, which came back as a uniform 4.4%
#     "floor" while inflating every ratio measured against A. One arm, thrown away.
#   * **A' is compared to A explicitly**, and that ratio is the floor every claim
#     about B is judged against. Do not import ±1.3%/±6% from `DECISIONS.md`; those
#     are `ccqlin038`'s from one session and drift is a function of how far apart two
#     arms are (A32) — 0.02% at a minute, 1–2% at an hour, 4.4% across a cold start.
#   * **Occupancy for the pinned core, its SMT sibling, and its whole L3 domain** is
#     recorded per arm, because a co-tenant in the domain is what invalidated two
#     earlier Phase 4 conclusions. A *constant* co-tenant cancels in A-vs-B ratios; an
#     intermittent one does not, so check the per-arm files, not just the mean.
#
# Cost: four arms x two dtype pairs. About 2 h on `ccqlin038` at the defaults.
# Nothing else may run on the machine, including a compile or an analysis script.
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT=${1:?usage: scripts/ab.sh OUTDIR \"ENV=VAL ...\" [cpu] [size] [reps] [filter]}
BENV=${2:?give the B arm as environment assignments, e.g. \"TENSORCONTRACT_POOL=on\"}
CPU=${3:-4}
SIZE=${4:-64}
REPS=${5:-3}
FILTER=${6:-}
BIN=${TC_TARGET:-target}/release/tcbench
[ -x "$BIN" ] || { echo "no $BIN -- cargo build --release -p tensorprimitives-bench" >&2; exit 1; }
mkdir -p "$OUT"

others=$(pgrep -a -u "$USER" -f 'cargo|rustc|tcbench|kernel_shapes' | grep -v "ab\.sh\|pgrep" || true)
[ -n "$others" ] && { echo "REFUSING: something of yours is running:" >&2; echo "$others" >&2; exit 1; }

echo "A/B on cpu$CPU; treatment: $BENV" | tee "$OUT/ab.log"
scripts/topology.py --json "$OUT/topology.json" >/dev/null

# One job per (tag, dtypes, env) through the shared runner, which pins to `--cpus`
# and records the whole L3 domain's occupancy per arm.
jobs="$OUT/arms.jobs"
{
    echo "# warm-up first, and discarded (A31)"
    for pair in f64,c64 f32,c32; do
        t=${pair/,/}
        echo "warm-$t   $pair"
        echo "A-$t      $pair"
        echo "B-$t      $pair  $BENV"
        echo "A2-$t     $pair"
    done
} > "$jobs"

FILT=()
[ -n "$FILTER" ] && FILT=(--case "$FILTER")
scripts/run-arms.py "$jobs" --outdir "$OUT" --bin "$BIN" \
    --size "$SIZE" --reps "$REPS" --engines planar,1m,3m \
    "${FILT[@]}" --cpus "$CPU" --sequential | tee -a "$OUT/ab.log"

echo | tee -a "$OUT/ab.log"
echo "###### this session's floor: A' against A ######" | tee -a "$OUT/ab.log"
scripts/compare-sweeps.py "$OUT/A-f64c64.csv,$OUT/A-f32c32.csv" \
    "$OUT/A2-f64c64.csv,$OUT/A2-f32c32.csv" | tee "$OUT/floor.txt"

echo | tee -a "$OUT/ab.log"
echo "###### the treatment: B against A ######" | tee -a "$OUT/ab.log"
scripts/compare-sweeps.py "$OUT/A-f64c64.csv,$OUT/A-f32c32.csv" \
    "$OUT/B-f64c64.csv,$OUT/B-f32c32.csv" | tee "$OUT/treatment.txt"

echo
echo "Judge the treatment against floor.txt, not against any number in DECISIONS.md."
echo "The warm-* arms are deliberately not analysed."
