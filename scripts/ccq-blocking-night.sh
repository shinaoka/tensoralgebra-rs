#!/bin/bash
# The blocking grid on the *reference* machine, overnight.
#
#   nohup scripts/ccq-blocking-night.sh > /dev/null 2>&1 &
#
# Run it on `ccqlin038` when the workstation is free, and then leave the machine
# alone: it is single-core by requirement, and the pinned core's hyperthread
# sibling shares the L1d and L2 that all of this is about.
#
# ---------------------------------------------------------------------------
# Why the reference machine still needs this after the Zen2 run
# ---------------------------------------------------------------------------
#
# The `rome` session answered item 2 on Zen2 with AVX2 register blocks. Three
# reasons that does not settle it here:
#
#   1. **`kc = 512` is a lower bound, not an optimum.** It was the largest arm the
#      Zen2 grid swept and it won. `kc768` and `kc1024` now exist so the optimum
#      gets bracketed rather than clipped.
#   2. **This is the machine the default affects.** Every committed number in
#      `DECISIONS.md` is AVX-512 on this box, and the grid was never run here — it
#      was started 2026-08-03 08:11 and stopped after two arms. Changing
#      `Blocking::derive` on Zen2 evidence alone would be changing it for a machine
#      class that was never measured.
#   3. **`kc` is where the analytical model failed** (part 9, A33), and the repair
#      has to be judged on more than one hierarchy.
#
# ---------------------------------------------------------------------------
# What the Zen2 run lets this one drop, and what it adds
# ---------------------------------------------------------------------------
#
# Evidence-based scoping, which is the return on having run the full grid once:
#
#   dropped  `ck64`, `ck128`, `ck384` — coupling was measured to add nothing over
#            pinning (`ck512` 1.031 against `kc512` 1.050 in `f64`). `ck256` and
#            `ck512` stay as a check that this still holds on a different
#            hierarchy, since one machine is not a general result.
#   added    `kc768`, `kc1024` — see above.
#   kept     all four `mc` arms. `MC` was a plateau on Zen2, but Zen2 has a 512 KiB
#            private L2 against this machine's 1 MiB, and "the plateau is wide" is
#            exactly the kind of claim that is allowed to be machine-specific.
#
# And a **warm-up arm that is discarded** (A31): this project's `A, B, A'` bracket
# cannot distinguish a cold-start transient from a noise floor, and on the `rome`
# session it reported 4.4% where the true repeat precision was 0.02%. An idle
# workstation is exactly the cold package that produces that.
#
# Cost: 19 arms x 2 dtype pairs plus the warm-up, so 40 jobs at roughly 11 min
# (`f64c64`) and 6 min (`f32c32`) each on this machine — about 7 h. Overnight, not
# an evening.
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CPU=${1:-4}
OUT=${2:-bench-results/$(hostname -s)-blocking}
SIZE=${3:-64}
REPS=${4:-3}
mkdir -p "$OUT"
LOG="$OUT/night.log"

say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Refuse to start if anything of ours is alive. The whole measurement rests on the
# machine being empty, and "I thought it was free" is how two Phase 4 conclusions
# had to be retracted.
others=$(pgrep -a -u "$USER" -f 'cargo|rustc|tcbench|kernel_shapes' \
         | grep -v "ccq-blocking-night\|pgrep" || true)
if [ -n "$others" ]; then
    echo "REFUSING: something of yours is already running:" >&2
    echo "$others" >&2
    exit 1
fi

[ -x ./target/release/tcbench ] || { echo "build first: cargo build --release -p tensorprimitives-bench" >&2; exit 1; }

say "reference-machine blocking grid; cpu$CPU; out $OUT"
./target/release/tcbench info 2>&1 | tee "$OUT/tcbench-info.txt" | tee -a "$LOG"
scripts/topology.py --json "$OUT/topology.json" | tee "$OUT/topology.txt"
./target/release/tcbench orient --size "$SIZE" --csv "$OUT/features.csv" > /dev/null

# Discarded warm-up (A31). One `base` arm, into a directory nothing analyses.
say "warm-up arm, discarded"
ARMS_ONLY="base" scripts/phase4e-blocking.sh "$CPU" "$OUT/warmup" "$SIZE" "$REPS" \
    >> "$LOG" 2>&1
say "warm-up done; the package is now at the clock the rest of the run will see"

say "grid: 19 arms x 2 dtype pairs = 38 jobs"
ARMS_ONLY="base kc64 kc128 kc256 kc384 kc512 kc768 kc1024 basem ck256 ck512 mc25 mc50 mc200 mc400 nc25 nc400 model base2" \
    scripts/phase4e-blocking.sh "$CPU" "$OUT" "$SIZE" "$REPS" >> "$LOG" 2>&1
say "grid complete"

say "scoring (per-case floor is re-derived below; do not import ccqlin038's 0.06 blindly)"
scripts/grid-placement-audit.py "$OUT" 2>&1 | tee "$OUT/placement-audit.txt" | tail -20
scripts/blocking-score-rules.py "$OUT/features.csv" "$OUT" 2>&1 | tee "$OUT/score.txt" | tail -40
say "done. Read $OUT/score.txt; the noise section gives this machine's own floor."
