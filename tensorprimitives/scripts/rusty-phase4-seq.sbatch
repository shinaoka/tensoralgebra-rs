#!/bin/bash
# The blocking arms that concurrent placement cannot measure, run sequentially.
#
#   sbatch scripts/rusty-phase4-seq.sbatch
#
# Submit from the repo root, after `rusty-phase4.sbatch` has finished. Pass the
# same OUT if you want the results beside that run:
#   OUT=bench-results/worker5040-zen2 sbatch scripts/rusty-phase4-seq.sbatch
#
# ---------------------------------------------------------------------------
# Why this exists
# ---------------------------------------------------------------------------
#
# The concurrent placement (one arm per L3 domain, 24 at once) was validated at
# +0.3% on the corpus as a whole and **rejected at −3.2%, up to −10% per case, on
# the memory-bound `abcijk` family** — see part 10. The grid is scored on ratios
# between arms measured under identical placement, so a uniform penalty cancels;
# what does not cancel is an arm that *changes memory traffic*, because then the
# contention it suffers differs from `base`'s.
#
# Three arms do that: `nc25`, `nc400`, and `model` — which raises `nc` about 4.7x
# on this machine and is the most interesting single arm in the grid. This run
# measures those three plus the `base`/`base2` brackets in the clean single-core
# regime, so they can be compared against each other honestly and tied back to the
# placed grid through `base`.
#
# 5 arms x 2 dtype pairs = 10 jobs, ~22 min each on Zen2, so ~3.7 h. It is
# deliberately *not* the whole grid: 40 jobs sequentially is ~15 h.
#
# A warm-up arm runs first and is discarded (A31). The session's opening arm
# otherwise runs at single-core boost on a cold package while nothing else does,
# which the standard A/B/A' bracket reports as a ±4.4% floor while inflating every
# ratio measured against it. That cost this project a misread floor once already.
# ---------------------------------------------------------------------------

#SBATCH --job-name=tp-phase4-seq
#SBATCH --partition=gen
#SBATCH --constraint=rome
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=slurm-tp-phase4-seq-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "job ${SLURM_JOB_ID:-<none>} on $(hostname -s), started $(date -Iseconds)"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol show job "$SLURM_JOB_ID" 2>&1 \
        | grep -E 'JobId|NumCPUs|OverSubscribe|Shared|NodeList|TimeLimit' || true
fi
command -v cargo >/dev/null || export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || { echo "FATAL: no cargo on PATH" >&2; exit 1; }

# A build tree private to this job, so an editing session on the submit host
# cannot relink the binary a running arm invokes. See `rusty-phase4.sbatch` for
# the incident this prevents.
export TC_TARGET=${TC_TARGET:-$PWD/target-job-${SLURM_JOB_ID:-manual}}
echo "build tree: $TC_TARGET  (private to this job)"

OUT=${OUT:-bench-results/$(hostname -s)-zen2}
SEQ="$OUT/phase4e-seq"
SIZE=${SIZE:-64}
REPS=${REPS:-3}
mkdir -p "$SEQ"

# Build first and alone; nothing else may compile once a measurement starts.
cargo build --release -p tensorprimitives-bench 2>&1 | tail -2
scripts/topology.py --json "$SEQ/topology.json" | tee "$SEQ/topology.txt"
CPU=$(python3 -c "import json;print(json.load(open('$SEQ/topology.json'))['placement']['slots'][0]['cpu'])")
echo "pinned to cpu$CPU"

# The warm-up, discarded. One `base` arm is enough to bring the package to the
# clock the rest of the session will run at; its output goes to a directory that
# is deliberately not analysed.
echo "########## warm-up (discarded, see A31)  $(date -Iseconds) ##########"
ARMS_ONLY="base" scripts/phase4e-blocking.sh "$CPU" "$SEQ/warmup" "$SIZE" "$REPS" \
    2>&1 | tail -4

echo "########## traffic-sensitive arms, sequential  $(date -Iseconds) ##########"
END_EPOCH=${SLURM_JOB_END_TIME:-}
if [ -z "$END_EPOCH" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    END=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
          | tr ' ' '\n' | awk -F= '/^EndTime=/{print $2}')
    [ -n "${END:-}" ] && END_EPOCH=$(date -d "$END" +%s 2>/dev/null || true)
fi
[ -n "$END_EPOCH" ] && export DEADLINE=$((END_EPOCH - 900))

ARMS_ONLY="base nc25 nc400 model base2" \
    scripts/phase4e-blocking.sh "$CPU" "$SEQ" "$SIZE" "$REPS" 2>&1 | tee -a "$SEQ/run.log"

echo
echo "finished $(date -Iseconds) after $((SECONDS / 60)) min"
echo "outdir $SEQ"
echo
echo "These are single-core arms. Compare them against each other, and against the"
echo "placed grid only through the base arms -- the placed nc/model arms are"
echo "confounded by bandwidth contention (part 10), which is why this run exists."
