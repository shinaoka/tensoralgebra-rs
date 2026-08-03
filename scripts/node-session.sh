#!/bin/bash
# Run the pending Phase 4 measurements on an exclusive cluster node.
#
# Everything in `DECISIONS.md` is single-core on `ccqlin038`, a Cascade Lake
# workstation. **Nothing measured here is comparable to any of it** — different
# machine, different cache hierarchy, and on a Zen2 node a different instruction
# set as well. Every ratio must be computed *within* this session, which is why
# each stage below either brackets its own treatments or derives its own noise
# floor, and why the published +-1.3% / +-6% floor is quoted nowhere in it.
#
# Stages, in descending value, so a short allocation still buys the top of the
# list. Run them one at a time or `all`:
#
#   prep      build once, then describe the machine and *predict* the results.
#             Costs no measurement time and must happen before anything else,
#             because a compile during a measurement invalidates it.
#   threads   thread scaling (phase4f). Unblocks a default: threading is
#             implemented, correct and off only because it was never measured.
#   shapes    the micro-kernel register-block sweep. On an AVX2 node this is the
#             highest-value item in the file: it turns D26's provisional,
#             explicitly unmeasured AVX2 register blocks into measured ones.
#   validate  is it safe to run grid arms concurrently, one per L3 domain? Half
#             an hour that either unlocks the grid or saves seven contaminated
#             hours. Read scripts/validate-placement.sh for the decision rule.
#   grid      the MC/KC/NC grid (phase4e), including the analytical-model arm.
#
# Isolation rules this enforces rather than trusts: exactly one stage runs at a
# time, `prep` is the only stage that compiles, and every stage records
# `/proc/stat` occupancy for the cores it used *and* for the rest of the
# allocation, so "the node was exclusive" is a recorded fact per arm rather than
# an assumption. Two Phase 4 conclusions had to be retracted for want of exactly
# that record.
#
# Usage: scripts/node-session.sh <stage> [outdir]
#        scripts/node-session.sh all bench-results/worker5144-zen2
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE=${1:-help}
OUT=${2:-}
SIZE=${SIZE:-64}
REPS=${REPS:-3}

if [ -z "$OUT" ]; then
    LABEL=$(python3 -c "
import re
m = ''
for line in open('/proc/cpuinfo'):
    if line.startswith('model name'):
        m = line.split(':', 1)[1].strip()
        break
low = m.lower()
if 'epyc' in low and '7' in low:
    fam = 'zen'
else:
    fam = 'x86'
print(re.sub(r'[^a-z0-9]+', '-', low)[:24] or fam)")
    OUT="bench-results/$(hostname -s)-$LABEL"
fi
mkdir -p "$OUT"
LOG="$OUT/session.log"

say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

guard_nothing_else() {
    # A measurement is only as good as the machine's emptiness. Refuse to start
    # one while a compile, another arm, or a stray baseline is alive.
    local others
    others=$(pgrep -a -u "$USER" -f 'cargo|rustc|tcbench|kernel_shapes' \
             | grep -v "node-session\|pgrep" || true)
    if [ -n "$others" ]; then
        echo "REFUSING: something of yours is already running:" >&2
        echo "$others" >&2
        exit 1
    fi
}

stage_prep() {
    say "prep: build, describe, predict. This is the only stage that compiles."
    guard_nothing_else
    say "building (release)"
    cargo build --release -p tensorprimitives-bench 2>&1 | tail -3 | tee -a "$LOG"
    cargo build --release -p tensorcontract --examples 2>&1 | tail -3 | tee -a "$LOG"

    say "machine description (the engine's own probes, not sinfo)"
    ./target/release/tcbench info 2>&1 | tee "$OUT/tcbench-info.txt" | tee -a "$LOG"
    scripts/topology.py --json "$OUT/topology.json" | tee "$OUT/topology.txt"

    # Exclusivity, recorded once from Slurm's own view. Read-only and scoped to
    # this job: organisation policy permits `scontrol show` and forbids anything
    # that changes state, and forbids polling in a loop.
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        say "slurm job $SLURM_JOB_ID"
        scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm-job.txt" 2>&1 || true
        grep -E 'JobId|NumNodes|NumCPUs|OverSubscribe|Shared|NodeList|TimeLimit' \
            "$OUT/slurm-job.txt" | tee -a "$LOG" || true
    else
        say "WARNING: no SLURM_JOB_ID — cannot record that the node is exclusive"
    fi

    # The structural analyses: no data touched, no CPU cost, exactly
    # reproducible. Generated *here* because the register blocks are per-ISA, so
    # the reference machine's copies describe a different engine.
    say "structural features for this ISA (free: touches no data)"
    ./target/release/tcbench orient --size "$SIZE" --csv "$OUT/features.csv" \
        > "$OUT/orient.txt" 2>&1
    ./target/release/tcbench shapes --size "$SIZE" --csv "$OUT/shapes.csv" \
        > "$OUT/shapes-analysis.txt" 2>&1
    cargo run --release -q -p tensorcontract --example blocking_model \
        > "$OUT/blocking-model.txt" 2>&1 || true
    say "what the analytical model predicts here:"
    cat "$OUT/blocking-model.txt" | tee -a "$LOG"

    # The prediction, made before the measurement so the scaling curve is
    # interpreted rather than admired. Uses the reference machine's throughputs
    # only as a compute-bound proxy for ranking cases; that is labelled in the
    # output and is not a cross-machine performance claim.
    local ncore
    ncore=$(python3 -c "import json;print(json.load(open('$OUT/topology.json'))['physical_cores'])")
    local plist="1,2,4,8,16,32,64,128"
    say "predicted parallel width up to $ncore cores"
    scripts/thread-width.py -p "$plist" "$OUT/features.csv" \
        bench-results/phase4/rm-A-f64c64.csv bench-results/phase4/rm-A-f32c32.csv \
        > "$OUT/thread-width-prediction.txt" 2>&1 || true
    tail -14 "$OUT/thread-width-prediction.txt" | tee -a "$LOG"
    say "prep done. Nothing measured yet."
}

stage_threads() {
    say "threads: scaling curve plus the two partition arms"
    guard_nothing_else
    SIZE=$SIZE REPS=$REPS scripts/phase4f-threads.sh auto "$OUT/phase4f" \
        "$SIZE" "$REPS" 2>&1 | tee -a "$OUT/phase4f.log"
    say "threads done"
}

stage_shapes() {
    say "shapes: micro-kernel register-block sweep (single core, ~8-20 min)"
    guard_nothing_else
    local cpu
    cpu=$(python3 -c "import json;print(json.load(open('$OUT/topology.json'))['placement']['slots'][0]['cpu'])")
    say "pinned to cpu$cpu"
    taskset -c "$cpu" ./target/release/examples/kernel_shapes \
        > "$OUT/kernel-shapes.txt" 2>&1
    say "best per method (this is what belongs in cfg_avx2_f64 / cfg_avx2_f32):"
    grep -A20 -i 'best per method' "$OUT/kernel-shapes.txt" | tee -a "$LOG" || \
        tail -30 "$OUT/kernel-shapes.txt" | tee -a "$LOG"
    say "shapes done"
}

stage_validate() {
    say "validate: is concurrent placement safe on this node?"
    guard_nothing_else
    scripts/validate-placement.sh "" "$OUT/placement" "$SIZE" "$REPS" \
        2>&1 | tee -a "$OUT/placement.log"
    say "validate done -- read the decision rule in scripts/validate-placement.sh"
}

stage_grid() {
    # `PLACEMENT=auto` only after `validate` has accepted it. Default to the
    # sequential regime, because that is the one that cannot be wrong.
    local mode=${PLACEMENT:-}
    if [ -z "$mode" ]; then
        echo "grid needs an explicit placement decision:" >&2
        echo "  PLACEMENT=auto  scripts/node-session.sh grid $OUT   # validate accepted it" >&2
        echo "  PLACEMENT=<cpu> scripts/node-session.sh grid $OUT   # sequential, one core" >&2
        exit 1
    fi
    say "grid: MC/KC/NC, placement=$mode"
    guard_nothing_else
    scripts/phase4e-blocking.sh "$mode" "$OUT/phase4e" "$SIZE" "$REPS" \
        2>&1 | tee -a "$OUT/phase4e.log"
    say "grid done"
}

case "$STAGE" in
    prep)     stage_prep ;;
    threads)  stage_threads ;;
    shapes)   stage_shapes ;;
    validate) stage_validate ;;
    grid)     stage_grid ;;
    all)
        stage_prep
        stage_threads
        stage_shapes
        stage_validate
        say "stopping before the grid: it needs the placement decision from validate."
        say "then: PLACEMENT=auto scripts/node-session.sh grid $OUT"
        ;;
    *)
        sed -n '2,40p' "$0"
        exit 1 ;;
esac

say "outdir: $OUT"
