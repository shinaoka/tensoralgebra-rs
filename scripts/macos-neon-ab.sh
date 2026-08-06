#!/usr/bin/env bash
# The scalar-vs-NEON A/B on Apple Silicon, end to end, in one process family.
#
# `TENSORCONTRACT_KERNEL` switches the two arms at run time, so this is a
# within-session comparison and not a build-to-build diff — which is the whole
# reason the switch exists (A15), and which the Phase 4 kernel work could not
# do because AVX-512 hardware cannot un-have its own kernels.
#
# Four arms in this order, and the order is the method:
#
#   warm        discarded. The first arm on a laptop pays for cold caches, a
#               cold branch predictor and a clock still ramping onto AC.
#   A-scalar    the control.
#   B-neon      the treatment.
#   A2-scalar   the control again. A/A2 is THIS SESSION'S FLOOR, and without it
#               the A/B number has no denominator. Every other floor in this
#               repository names its session and thread count; so does this one.
#
# `ttgt` and `tblis` ride along in every arm as a second control: no setting of
# `TENSORCONTRACT_KERNEL` can reach either baseline, so any movement in those
# columns is drift, and it bounds the engine columns' credibility from outside.
#
# There is NO CPU PINNING on Darwin and this script cannot invent one. See
# `DECISIONS.md` part 20. The compensations are the warm-up arm, the repeat
# control, AC power, a quiesced machine, single-threaded everything, and the
# thermal level recorded at both ends.
#
#   scripts/macos-neon-ab.sh bench-results/$(hostname -s)-$(scripts/arch-label.sh)/neon-ab
#
# Expects `$OUT/bin/tcbench-tblis` to exist already — build it once, before the
# run, and never during: the arms invoke that binary and `cargo build` would
# replace it underneath them (D51).

set -euo pipefail

OUT=${1:?usage: macos-neon-ab.sh <outdir>}
SIZE=${SIZE:-64}
REPS=${REPS:-3}
ENGINES=${ENGINES:-planar,1m,3m,ttgt,tblis}
DTYPES=${DTYPES:-f64,c64}
BIN=$OUT/bin/tcbench-tblis
LOG=$OUT/session.log

[[ -x $BIN ]] || { echo "no $BIN -- build it first, outside this script"; exit 1; }

export TENSORCONTRACT_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TBLIS_NUM_THREADS=1
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH:-}

{
    echo "session   : $(date -Iseconds)"   # -Iseconds; BSD date rejects -Is
    echo "host      : $(hostname -s)  $(sysctl -n machdep.cpu.brand_string)"
    echo "size      : $SIZE MiB, reps $REPS, engines $ENGINES, dtypes $DTYPES"
    echo "power     : $(pmset -g batt | head -1)"
    echo "thermal   : $(pmset -g therm | tr '\n' ' ')"
    "$BIN" info
} | tee -a "$LOG"

for arm in warm A-scalar B-neon A2-scalar; do
    k=scalar
    case $arm in *neon*) k=neon ;; esac
    echo -e "\n\n########## $arm  (TENSORCONTRACT_KERNEL=$k) ##########\n" | tee -a "$LOG"
    TENSORCONTRACT_KERNEL=$k "$BIN" premise \
        --size "$SIZE" --reps "$REPS" \
        --engines "$ENGINES" --dtype "$DTYPES" \
        --csv "$OUT/$arm.csv" 2>&1 | tee -a "$LOG" | tail -20
done

{
    echo -e "\n\nended     : $(date -Iseconds)"
    echo "thermal   : $(pmset -g therm | tr '\n' ' ')"
} | tee -a "$LOG"

echo -e "\n=== the floor: A against A2, nothing changed between them ==="
scripts/compare-sweeps.py "$OUT/A-scalar.csv" "$OUT/A2-scalar.csv" | tee "$OUT/floor.txt"
echo -e "\n=== the treatment: scalar against NEON ==="
scripts/compare-sweeps.py "$OUT/A-scalar.csv" "$OUT/B-neon.csv" | tee "$OUT/treatment.txt"
