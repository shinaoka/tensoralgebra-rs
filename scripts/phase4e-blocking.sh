#!/bin/bash
# Phase 4 item 2: the MC/KC/NC grid.
#
# The cache blocking is still the untouched Phase 2 heuristic: `kc` is 384 for
# 4-byte reals and 256 otherwise, and `mc`/`nc` follow from fixed L2/L3 budgets
# and the packed footprint each method produces. Phase 3 showed the whole
# complex-method ranking turns on whether the `A` sliver is an L1 resident or an
# L2 stream, which makes `kc` first-order rather than a tuning knob.
#
# Two things about the design, both learned the hard way:
#
#   * `MC` is bounded from *both* sides (A13). Below by keeping the packed `A`
#     block in L2, above by the strip of `D` that one `jr` pass revisits. A
#     single "best MC" does not exist and a sweep that looks for one cannot see
#     the second bound. So the `kc` axis is measured twice: once with `mc`/`nc`
#     pinned at the derived default (`_KC`), which moves the panel depth and
#     leaves the `D` strip alone, and once re-derived against the same cache
#     budgets at the new depth (`_KC_COUPLE`), which moves both. The pair
#     separates the two effects; either alone confounds them. Phase 4 part 3
#     measured the coupled form on three cases, got one sign each way and backed
#     it out — this is that experiment done on all 392.
#   * The `mc`/`nc` axes are swept as *percentages* of the derived value, not as
#     absolute numbers. An absolute `mc` is a different fraction of the L2
#     budget in every dtype and method (1m derives half of planar's, by design),
#     so pinning one number across an arm would rig the method comparison the
#     same way a wrong `Blocking::derive` would.
#
# This is the "sweep the whole grid once, score rules offline" pattern from
# items 1c and 1d (see `scripts/phase4c-rowblock.sh`), which is what makes a
# per-case rule — `kc = min(k, KC)` gated on something — cost nothing to
# evaluate. Every arm is a runtime switch, so no arm is a rebuild, and the
# shipped row-block and orientation rules stay *on*: neither takes the blocking
# as an input, so there is no confound to pin away, and leaving them on means
# the grid is measured in the configuration that ships (A20).
#
# Free bonus, worth knowing when reading the output: a third of the corpus
# contracts over k <= 24, so for those cases every pinned-`kc` arm is bit-for-bit
# the same computation. Their spread across arms is therefore a direct per-case
# noise measurement inside this very run, not one imported from another session.
#
# Cost: ~22 min per arm for both dtype pairs, 20 arms, so ~7.5 h. Single core.
# Nothing else may run on the machine (not even a compile) while it is in
# flight: the pinned core's hyperthread sibling shares the L1d and L2 that all
# of this is about.
#
# Usage: scripts/phase4e-blocking.sh [cpu] [outdir] [size_mib] [reps] [filter]
#
# Smoke-test the whole thing in place before committing 7 h to it:
#   scripts/phase4e-blocking.sh 4 /tmp/smoke 8 1 ijkl-imjn-lnkm
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CPU=${1:-4}
OUT=${2:-bench-results/phase4e}
SIZE=${3:-64}
REPS=${4:-3}
FILTER=${5:-}
SIB=$(cat /sys/devices/system/cpu/cpu$CPU/topology/thread_siblings_list)
BIN=./target/release/tcbench
[ -x "$BIN" ] || { echo "no $BIN in $PWD -- cargo build --release -p tensorcontract-bench" >&2; exit 1; }
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

# run <tag> <dtypes> [env assignments...]
run() {
    local tag=$1 dt=$2 before after
    shift 2
    before=$(snap)
    env "$@" taskset -c $CPU $BIN sweep \
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
    echo "  $(date +%H:%M:%S) $tag: $(tr '\n' ' ' < "$OUT/$tag.cpu")"
}

# The arms. `base` appears three times — first, middle and last — so drift over
# a seven-hour run is a measured quantity and every treatment is bracketed.
#
#   kc<n>  pinned mc/nc, kc forced: panel depth alone.
#   ck<n>  kc forced and mc/nc re-derived at that depth: depth plus footprint.
#   mc<p>  mc scaled to p% of derived, kc default: the two-sided MC question.
#   nc<p>  nc likewise.
ARMS=(
    "base"
    "kc64   TENSORCONTRACT_KC=64"
    "kc128  TENSORCONTRACT_KC=128"
    "kc256  TENSORCONTRACT_KC=256"
    "kc384  TENSORCONTRACT_KC=384"
    "kc512  TENSORCONTRACT_KC=512"
    "ck64   TENSORCONTRACT_KC_COUPLE=64"
    "ck128  TENSORCONTRACT_KC_COUPLE=128"
    "basem"
    "ck256  TENSORCONTRACT_KC_COUPLE=256"
    "ck384  TENSORCONTRACT_KC_COUPLE=384"
    "ck512  TENSORCONTRACT_KC_COUPLE=512"
    "mc25   TENSORCONTRACT_MC_PCT=25"
    "mc50   TENSORCONTRACT_MC_PCT=50"
    "mc200  TENSORCONTRACT_MC_PCT=200"
    "mc400  TENSORCONTRACT_MC_PCT=400"
    "nc25   TENSORCONTRACT_NC_PCT=25"
    "nc400  TENSORCONTRACT_NC_PCT=400"
    # The analytical model (part 9), which is the arm that matters most now: it
    # is not a point in this grid but a whole different derivation, and it moves
    # all three parameters at once — `kc` down 2.4-8x, `mc` up 4-6x, `nc` up
    # ~20x. Its `kc` makes the `A` sliver an L1 resident where the constants make
    # it an L2 stream, which Phase 3 found the method ranking to turn on; its
    # `mc` is exactly what A13's missing upper bound would punish. Both effects
    # land in one arm, which is why the single-parameter arms above are still
    # needed to attribute whatever it does.
    "model  TENSORCONTRACT_BLOCKMODEL=model"
    "base2"
)

for pair in f64,c64 f32,c32; do
    t=${pair/,/}
    for arm in "${ARMS[@]}"; do
        read -r tag envs <<<"$arm"
        # shellcheck disable=SC2086  # envs is a deliberate word list
        run "bl-$tag-$t" "$pair" $envs
    done
    echo "done $t"
done
echo "grid complete; score rules with scripts/blocking-score-rules.py"
