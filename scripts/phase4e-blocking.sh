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
# ---------------------------------------------------------------------------
# Placement: one core, or one core per L3 domain
# ---------------------------------------------------------------------------
#
# The arms are independent, so on a many-core node they can run concurrently —
# 20 arms x 2 dtype pairs is 40 jobs, and 7.5 h of sequential work fits in under
# an hour if 24 of them run at once. That is a *hypothesis about placement*, not
# a free lunch, because arms sharing an L3 perturb each other and `NC` is sized
# for L3. So both regimes are the same code path here, selected by the first
# argument, and the parallel one must be validated before it is trusted:
#
#   scripts/phase4e-blocking.sh 4      OUT   # one pinned core, the reference regime
#   scripts/phase4e-blocking.sh auto   OUT   # one core per L3 domain, rest idle
#
# `scripts/validate-placement.sh` is what decides which of those to use: it runs
# one arm solo and then the same arm under the placement and compares them
# against the noise floor measured in the same session. Run it first. A rejected
# placement is a result worth recording, and costs half an hour rather than
# seven contaminated hours.
#
# Either way `scripts/run-arms.py` records `/proc/stat` occupancy for **every
# core in each arm's own L3 domain** across exactly that arm's window, plus the
# observed overlap with other arms, so co-tenancy is a recorded fact per arm.
# That recording is what made two earlier Phase 4 retractions detectable.
#
# Cost: ~22 min per arm for both dtype pairs on the reference machine, 20 arms,
# so ~7.5 h pinned to one core. Nothing else may run on the machine (not even a
# compile) while it is in flight.
#
# Usage: scripts/phase4e-blocking.sh [cpu|auto] [outdir] [size_mib] [reps] [filter]
#        JOBS=<n> scripts/phase4e-blocking.sh auto ...   # cap the concurrency
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
BIN=./target/release/tcbench
[ -x "$BIN" ] || { echo "no $BIN in $PWD -- cargo build --release -p tensorprimitives-bench" >&2; exit 1; }
mkdir -p "$OUT"

# The arms. `base` appears three times — first, middle and last — so drift over
# a long run is a measured quantity and every treatment is bracketed. Under
# concurrent placement "first, middle, last" becomes "three independent
# repeats", which brackets contention rather than drift; both are what the
# repeats are for.
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
    # ~20x on the reference machine. Its `kc` makes the `A` sliver an L1 resident
    # where the constants make it an L2 stream, which Phase 3 found the method
    # ranking to turn on; its `mc` is exactly what A13's missing upper bound
    # would punish. Both effects land in one arm, which is why the
    # single-parameter arms above are still needed to attribute whatever it does.
    #
    # On any machine that is not `ccqlin038` this arm carries a second and larger
    # claim: the legacy constants *are* `ccqlin038`'s cache sizes written down by
    # hand, so elsewhere they are wrong by construction and the model should win.
    # If it does not, that is a finding about the model, not about the node.
    "model  TENSORCONTRACT_BLOCKMODEL=model"
    "base2"
)

# One jobfile, generated from the list above, so the arm definitions have a
# single source of truth whichever regime runs them. The dtype pairs are
# separate jobs rather than an inner loop: they are independent measurements, so
# under concurrent placement they are 40 jobs and not 20.
#
# `ARMS_ONLY` keeps a subset, by tag. It exists for the case where the placement
# was *rejected* and the grid has to run sequentially inside whatever wall time
# is left: then the arms are not all equally worth having, and the order to keep
# them in is `base model base2` first — the model arm is a whole different
# derivation rather than a point in the grid, and it needs its two brackets to
# mean anything — then the `kc` family, `kc` being the first-order parameter.
# A subset is a *scoped* grid and must be reported as one; offline rule scoring
# against a partial grid is not the same asset as against the whole one.
JOBS_FILE="$OUT/arms.jobs"
: > "$JOBS_FILE"
{
    echo "# generated by $0 on $(date -Iseconds)"
    echo "# tag  dtypes  env..."
    for pair in f64,c64 f32,c32; do
        t=${pair/,/}
        for arm in "${ARMS[@]}"; do
            read -r tag envs <<<"$arm"
            if [ -n "${ARMS_ONLY:-}" ]; then
                case " $ARMS_ONLY " in *" $tag "*) ;; *) continue ;; esac
            fi
            echo "bl-$tag-$t  $pair  $envs"
        done
    done
} >> "$JOBS_FILE"
if [ -n "${ARMS_ONLY:-}" ]; then
    echo "SCOPED GRID: only arms [$ARMS_ONLY] -- this is not the full grid" \
        | tee -a "$OUT/run.log"
fi

PLACE=(--cpus "$CPU" --sequential)
if [ "$CPU" = auto ]; then
    PLACE=()
    [ -n "${JOBS:-}" ] && PLACE=(--jobs "$JOBS")
fi

FILT=()
[ -n "$FILTER" ] && FILT=(--case "$FILTER")
DEAD=()
[ -n "${DEADLINE:-}" ] && DEAD=(--deadline "$DEADLINE")

scripts/run-arms.py "$JOBS_FILE" --outdir "$OUT" --bin "$BIN" \
    --size "$SIZE" --reps "$REPS" --engines planar,1m,3m \
    "${FILT[@]}" "${PLACE[@]}" "${DEAD[@]}" | tee -a "$OUT/run.log"

echo "grid complete; score rules with scripts/blocking-score-rules.py" \
    | tee -a "$OUT/run.log"
