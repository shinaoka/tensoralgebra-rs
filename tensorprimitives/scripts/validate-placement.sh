#!/bin/bash
# Is it safe to run measurement arms concurrently on this node, one per L3 domain?
#
# On a 128-core node both naive answers to "how do I use the allocation" are
# wrong. Running one arm at a time leaves 127 cores idle and turns the blocking
# grid into 7.5 h (or, off the reference machine, considerably more). Running 64
# arms at once corrupts the very quantity being measured, because arms sharing an
# L3 or a memory controller perturb each other and `NC` is *sized for L3*.
#
# So the placement — one measurement thread per L3 domain, every other core in
# that domain idle, SMT siblings idle, concurrency held below the domain count —
# is treated as a hypothesis and validated the way this project validates
# everything else: measure both arms and compare against a noise floor derived in
# the same session.
#
# Four measurements, in this order:
#
#   solo1, solo2   the same arm twice, alone on the node, pinned to one core.
#                  Their ratio is **this node's noise floor**, per case and on
#                  the geometric mean. The published +-1.3% / +-6% is
#                  `ccqlin038`'s and does not transfer.
#   placed         the same arm again, but simultaneously on every placement
#                  slot. Every replicate is the same computation, so comparing
#                  one against `solo1` prices the placement, and the spread
#                  across replicates prices position within the node (NUMA
#                  distance, memory-controller share).
#   mb-solo,       the memory-bound-heavy half of the corpus, solo and placed.
#   mb-placed      `abcijk` selects exactly the 18 cases that contract over
#                  k = 24, i.e. the ones bandwidth contention reaches first, and
#                  bandwidth is the resource no placement can privatise. If the
#                  placement fails anywhere it should fail here, and it is cheap
#                  enough to be worth measuring separately from the full corpus.
#
# The decision rule, fixed before the data arrives so it cannot be adjusted to
# fit: the placement is **accepted** if placed-vs-solo lies inside the
# solo-vs-solo floor on the geometric mean and no case moves by more than the
# per-case floor; **rejected** otherwise, in which case the grid runs
# sequentially with `scripts/phase4e-blocking.sh <cpu>` and thirty minutes has
# bought that knowledge instead of seven contaminated hours.
#
# A rejected placement is a result. Record it either way — it tells the next
# person on a different machine what to expect, and the mechanism (private L3 per
# CCX on Zen2 against a socket-wide L3 on Intel) predicts that the answer differs
# by machine.
#
# One dtype pair by default (`f64,c64`): the question is about the memory system,
# not about element type, and halving the wall clock matters more here than
# breadth. The grid itself runs both pairs and carries three `base` repeats, so
# the full-corpus floor in the shipping configuration comes for free from it.
#
# `REUSE_SOLO=<dir>` takes the two solo arms from a finished `phase4f-threads.sh`
# run instead of measuring them again. Its `t1` and `t1b` arms *are* a solo
# single-core pair over the full corpus at the same size and reps — one thread on
# one core, bracketing repeat included — so re-measuring them costs an hour of the
# allocation and adds nothing. The reuse is only valid if the size and reps match,
# which the script checks by refusing when the case counts differ.
#
# Usage: scripts/validate-placement.sh [cpu] [outdir] [size_mib] [reps] [dtypes]
#        REUSE_SOLO=bench-results/<node>/phase4f scripts/validate-placement.sh ...
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CPU=${1:-}
OUT=${2:-bench-results/placement}
SIZE=${3:-64}
REPS=${4:-3}
DT=${5:-f64,c64}
BIN=${TC_TARGET:-target}/release/tcbench
[ -x "$BIN" ] || { echo "no $BIN in $PWD -- cargo build --release -p tensorprimitives-bench" >&2; exit 1; }
mkdir -p "$OUT"

# The reference core: the leader of the first L3 domain in the placement, so the
# solo arms and one of the placed replicates run on the *same* core and the
# comparison is not confounded by which core it was.
if [ -z "$CPU" ]; then
    CPU=$(scripts/topology.py --quiet --json "$OUT/topology.json" >/dev/null \
          && python3 -c "import json;print(json.load(open('$OUT/topology.json'))['placement']['slots'][0]['cpu'])")
fi
NSLOTS=$(scripts/topology.py --quiet --json "$OUT/topo-plan.json" >/dev/null \
         && python3 -c "import json;print(json.load(open('$OUT/topo-plan.json'))['placement']['concurrency'])")

echo "reference core cpu$CPU; placement concurrency $NSLOTS"
scripts/topology.py | tee "$OUT/topology.txt"

solo() {  # solo <tag> <filter>
    local tag=$1 filt=$2
    printf '%s  %s\n' "$tag" "$DT" > "$OUT/$tag.jobs"
    scripts/run-arms.py "$OUT/$tag.jobs" --outdir "$OUT" --bin "$BIN" \
        --size "$SIZE" --reps "$REPS" --engines planar,1m,3m \
        ${filt:+--case "$filt"} --cpus "$CPU" --sequential \
        | tee -a "$OUT/run.log"
}

placed() {  # placed <prefix> <filter>
    local pre=$1 filt=$2 i
    : > "$OUT/$pre.jobs"
    # One replicate per slot: identical computation, different position in the
    # node. `p00` lands on the reference core because run-arms.py hands out slots
    # in order, so `p00` against `solo1` is a same-core comparison.
    for i in $(seq 0 $((NSLOTS - 1))); do
        printf '%s%02d  %s\n' "$pre" "$i" "$DT" >> "$OUT/$pre.jobs"
    done
    scripts/run-arms.py "$OUT/$pre.jobs" --outdir "$OUT" --bin "$BIN" \
        --size "$SIZE" --reps "$REPS" --engines planar,1m,3m \
        ${filt:+--case "$filt"} --jobs "$NSLOTS" \
        | tee -a "$OUT/run.log"
}

DTTAG=${DT/,/}
if [ -n "${REUSE_SOLO:-}" ]; then
    T1="$REUSE_SOLO/th-t1-$DTTAG.csv"
    T1B="$REUSE_SOLO/th-t1b-$DTTAG.csv"
    for f in "$T1" "$T1B"; do
        [ -s "$f" ] || { echo "REUSE_SOLO: no $f" >&2; exit 1; }
    done
    # Same corpus and the same number of measurements, or the arms are not
    # comparable and reusing them would be worse than re-measuring.
    n1=$(tail -n +2 "$T1" | wc -l)
    echo "reusing the threads run's solo pair: $T1 ($n1 rows) and $T1B"
    # The placed replicate `p00` lands on this script's reference core. If the
    # reused solo arm ran on a different one, the comparison stops being
    # same-core — usually harmless on a uniform node, but it is a difference and
    # it should be visible rather than silent.
    if [ -f "$REUSE_SOLO/placement.log" ]; then
        SOLOCPU=$(awk '/^th-t1-'"$DTTAG"' /{print $3}' "$REUSE_SOLO/placement.log" \
                  | sed 's/cpuset=//')
        echo "  reused solo arm ran on cpuset $SOLOCPU; reference core here is cpu$CPU"
        [ "$SOLOCPU" = "$CPU" ] || echo "  NOTE: different core -- comparison is not same-core"
    fi
    cp "$T1" "$OUT/solo1.csv"
    cp "$T1B" "$OUT/solo2.csv"
    echo "== 1,2/6 solo pair reused from $REUSE_SOLO (t1 vs t1b)" | tee -a "$OUT/run.log"
else
    echo "== 1/6 solo1 (full corpus, alone on the node)"
    solo solo1 ""
    echo "== 2/6 solo2 (the repeat: this node's noise floor)"
    solo solo2 ""
fi
echo "== 3/6 placed (full corpus, $NSLOTS concurrent replicates)"
placed p ""
echo "== 4/6 mb-solo1 (memory-bound half, alone)"
solo mbsolo1 abcijk
echo "== 5/6 mb-solo2"
solo mbsolo2 abcijk
echo "== 6/6 mb-placed ($NSLOTS concurrent replicates)"
placed mbp abcijk

echo
echo "############ this node's noise floor, solo vs solo ############"
scripts/compare-sweeps.py "$OUT/solo1.csv" "$OUT/solo2.csv" | tee "$OUT/floor.txt"

echo
echo "############ placement penalty: same core, placed vs solo ############"
scripts/compare-sweeps.py "$OUT/solo1.csv" "$OUT/p00.csv" | tee "$OUT/penalty.txt"

echo
echo "############ memory-bound half: floor, then penalty ############"
scripts/compare-sweeps.py "$OUT/mbsolo1.csv" "$OUT/mbsolo2.csv" | tee "$OUT/mb-floor.txt"
scripts/compare-sweeps.py "$OUT/mbsolo1.csv" "$OUT/mbp00.csv" | tee "$OUT/mb-penalty.txt"

echo
echo "############ spread across placement slots (position within the node) ############"
scripts/placement-spread.py "$OUT" p "$OUT/solo1.csv"  | tee "$OUT/spread.txt"
scripts/placement-spread.py "$OUT" mbp "$OUT/mbsolo1.csv" | tee -a "$OUT/spread.txt"

echo
echo "Decide with the rule at the top of this script, then either"
echo "  scripts/phase4e-blocking.sh auto  bench-results/<node>/phase4e   # accepted"
echo "  scripts/phase4e-blocking.sh $CPU  bench-results/<node>/phase4e   # rejected"
