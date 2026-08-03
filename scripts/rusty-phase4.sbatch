#!/bin/bash
# The two pending Phase 4 measurements, unattended, on one exclusive Rusty node.
#
#   sbatch scripts/rusty-phase4.sbatch
#
# Submit it from the repo root. Everything lands in
# `bench-results/<node>-<arch>/` and the job log in `slurm-tp-phase4-<jobid>.out`.
#
# Read `DECISIONS.md` part 10 before submitting: the node choice is the experiment
# design, and nothing measured here is comparable to any number committed in this
# repo (different machine, different cache hierarchy, and on `rome` a different
# instruction set).
#
# ---------------------------------------------------------------------------
# Why the stage order is not the priority order
# ---------------------------------------------------------------------------
#
# `shapes` runs second, before the much longer `threads`, even though thread
# scaling is the higher-priority *item*. On an AVX2 node the register-block
# calibration is the highest-value single deliverable in the file — it turns
# D26's provisional, explicitly unmeasured blocks into measurements — and it
# costs twenty minutes. Spending those twenty minutes before a four-hour stage is
# risk management, not a change of priorities: if the job dies early, the cheap
# irreplaceable thing is already on disk.
#
# ---------------------------------------------------------------------------
# What happens if something fails
# ---------------------------------------------------------------------------
#
# Stages are independent and a failure does not abort the rest, because losing
# the 20-minute calibration to a 4-hour stage's failure would be absurd. Each
# stage's status is recorded and summarised at the end.
#
# The grid is the exception: it runs only if `validate` **accepts** the concurrent
# placement, and that decision is made by `scripts/placement-verdict.py`, which
# implements the rule pre-registered in part 10 before any data existed.
#
# On **rejection** the full grid cannot run — sequential is ~23 h here and does not
# fit the wall time — so instead of returning nothing it runs a *scoped* grid
# sequentially inside whatever wall time is left: `base model base2` first,
# because the analytical-model arm is the most interesting single arm and needs
# its brackets to mean anything, then the `kc` family, `kc` being the first-order
# parameter. Arms that do not fit are named in `skipped-arms.txt` and in the log.
# A scoped grid is reported as scoped: offline rule scoring against a partial grid
# is not the asset a whole grid is.
#
# ---------------------------------------------------------------------------

#SBATCH --job-name=tp-phase4
#SBATCH --partition=gen
#SBATCH --constraint=rome
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-tp-phase4-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "=========================================================="
echo "job      : ${SLURM_JOB_ID:-<none>} on $(hostname -s)"
echo "submitted: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "started  : $(date -Iseconds)"
echo "=========================================================="

# `--exclusive` is requested above, but requested is not the same as granted.
# Record Slurm's own view once, scoped to this job: read-only, one shot.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol show job "$SLURM_JOB_ID" 2>&1 \
        | grep -E 'JobId|NumNodes|NumCPUs|OverSubscribe|Shared|NodeList|TimeLimit' || true
fi

# Slurm exports the submitting environment, so a rustup toolchain in $HOME comes
# along. Fail early and legibly if it did not, rather than in the middle of a
# stage.
command -v cargo >/dev/null || export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || {
    echo "FATAL: no cargo on PATH. Submit from a shell that can build the workspace." >&2
    exit 1
}
cargo --version
echo

OUT=${OUT:-bench-results/$(hostname -s)-zen2}
export SIZE=${SIZE:-64}
export REPS=${REPS:-3}
mkdir -p "$OUT"
SUMMARY="$OUT/stage-status.txt"
: > "$SUMMARY"

stage() {  # stage <name> [env assignments passed through by the caller]
    local name=$1 start rc
    start=$SECONDS
    echo
    echo "########## stage $name  $(date -Iseconds) ##########"
    if scripts/node-session.sh "$name" "$OUT"; then rc=0; else rc=$?; fi
    printf '%-10s rc=%-3s %6d min\n' "$name" "$rc" "$(((SECONDS - start) / 60))" \
        | tee -a "$SUMMARY"
    return $rc
}

# prep must succeed: it is the build, and every later stage needs the binary.
if ! stage prep; then
    echo "FATAL: prep failed, so there is nothing to measure with." >&2
    cat "$SUMMARY"
    exit 1
fi

# Cheap and irreplaceable first, then the long one. Neither aborts the other.
stage shapes  || echo "WARNING: shapes stage failed; continuing" >&2
stage threads || echo "WARNING: threads stage failed; continuing" >&2
stage validate || echo "WARNING: validate stage failed; continuing" >&2

echo
echo "########## placement verdict  $(date -Iseconds) ##########"
VERDICT=2
if [ -d "$OUT/placement" ]; then
    scripts/placement-verdict.py "$OUT/placement" --json "$OUT/placement/verdict.json"
    VERDICT=$?
fi
# When the wall clock, not the science, is what bounds the grid, the bound has to
# be explicit. Take the job's own end time from Slurm and keep a margin so the
# analysis at the end of the grid script still runs.
END_EPOCH=${SLURM_JOB_END_TIME:-}
if [ -z "$END_EPOCH" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
     END=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
           | tr ' ' '\n' | awk -F= '/^EndTime=/{print $2}')
    [ -n "${END:-}" ] && END_EPOCH=$(date -d "$END" +%s 2>/dev/null || true)
fi
if [ -n "$END_EPOCH" ]; then
    DEADLINE=$((END_EPOCH - 900))   # 15 min margin
    echo "grid deadline: $(date -d "@$DEADLINE" -Iseconds) (job ends $(date -d "@$END_EPOCH" -Iseconds))"
else
    DEADLINE=""
    echo "WARNING: could not determine the job's end time; the grid runs unbounded"
fi
export DEADLINE

case $VERDICT in
    0)  echo "placement ACCEPTED -- running the full grid concurrently"
        printf '%-10s %s\n' verdict accepted >> "$SUMMARY"
        export PLACEMENT=auto
        stage grid || echo "WARNING: grid stage failed" >&2 ;;
    1)  echo "placement REJECTED -- this is a result, not a failure."
        echo "Running a SCOPED grid sequentially in the wall time that is left."
        printf '%-10s %s\n' verdict rejected >> "$SUMMARY"
        export PLACEMENT
        PLACEMENT=$(python3 -c "import json;print(json.load(open('$OUT/topology.json'))['placement']['slots'][0]['cpu'])")
        export ARMS_ONLY="base model base2 kc64 kc128 kc256 kc384 kc512"
        stage grid || echo "WARNING: scoped grid failed" >&2 ;;
    *)  echo "placement UNDECIDED (validate did not produce comparable arms)."
        echo "Skipping the grid: an undecided placement is not an accepted one."
        printf '%-10s %s\n' verdict undecided >> "$SUMMARY" ;;
esac

echo
echo "=========================================================="
echo "finished : $(date -Iseconds)   after $((SECONDS / 60)) min"
echo "outdir   : $OUT"
echo "----------------------------------------------------------"
cat "$SUMMARY"
echo "=========================================================="
echo
echo "Nothing here is comparable to any number in DECISIONS.md. Every ratio is"
echo "computed within this session, and the noise floor is the one this node's"
echo "own bracketing repeats produced -- not the committed +-1.3% / +-6%."
