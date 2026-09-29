#!/bin/bash
# The two pending Phase 4 measurements, unattended, on one exclusive Rusty node.
#
#   sbatch scripts/rusty-phase4.sbatch
#
# Submit it from the repo root. Everything lands in
# `bench-results/<node>-<arch>/` and the job log in `slurm-tp-phase4-<jobid>.out`.
#
# Read `docs/notebook/` part 10 before submitting: the node choice is the experiment
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

# ---------------------------------------------------------------------------
# A per-job build tree, so the submit host cannot pull the binary out from under
# a running arm
# ---------------------------------------------------------------------------
#
# This job runs in the submit directory and, until this existed, used its
# `target/`. So a `cargo build` on the workstation while the job was in flight --
# or a `cargo test`, which relinks the same artefacts -- replaced the very
# executable each arm invokes, mid-session, and nothing reported it. That happened
# on 2026-08-04 (part 13) and was inert only by luck; the fix is five lines and
# removes the whole class rather than relying on remembering.
#
# `node-session.sh` exports this as `CARGO_TARGET_DIR` and every stage script and
# `run-arms.py` resolves its binary from it, so the build and the arms agree by
# construction. Unset, they all fall back to `target/` and a hand run is unchanged.
export TC_TARGET=${TC_TARGET:-$PWD/target-job-${SLURM_JOB_ID:-manual}}
echo "build tree: $TC_TARGET  (private to this job)"
echo
cargo --version
echo

# Label from what the machine reports, not from the partition this script was first
# written for. Hardcoding `-zen2` here filed an Ice Lake run under a Zen2 name once
# already; a directory name is a provenance claim.
OUT=${OUT:-bench-results/$(hostname -s)-$(
    python3 -c "
import re
m=''
for l in open('/proc/cpuinfo'):
    if l.startswith('model name'): m=l.split(':',1)[1].strip().lower(); break
for pat, tag in (('epyc 7[0-9]*2','zen2'), ('epyc 9','zen4'), ('epyc','zen'),
                 ('platinum 83','icelake'), ('gold 63','skylake'),
                 ('gold 62','cascadelake'), ('xeon','intel')):
    if re.search(pat, m): print(tag); break
else: print(re.sub(r'[^a-z0-9]+','-',m)[:16] or 'x86')
")}
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

# `STAGES` selects what runs after prep. The default is everything, but a machine
# whose L3 is a whole socket can host at most one or two concurrent arms, which
# makes the 36-job grid a ten-hour sequential run and the placement validation
# nearly meaningless — on such a node `STAGES="shapes threads"` is the whole value.
# `PARTITION_SWEEP=1`, `SPREAD=1` and `SP_T` are passed through to the threads
# stage; see phase4f-threads.sh. `SPREAD` is the within-machine separation of
# domain count from thread count (A36's residual gap) and is cheap.
# `small` is opt-in, and `ARMS` is how the spawn-cost switches get measured — it
# sweeps several configurations against a `base` arm in one session, at ~30 min per
# arm. `ab.sh` cannot do this job: it never sets a thread count, so
# `TENSORCONTRACT_POOL` and `_AMORTISE` are no-ops in every one of its arms.
#
#   ARMS="base;pool:TENSORCONTRACT_POOL=on" STAGES=small sbatch scripts/rusty-phase4.sbatch
STAGES=${STAGES:-"shapes threads validate grid"}
export PARTITION_SWEEP=${PARTITION_SWEEP:-}
export SPREAD=${SPREAD:-}
export RAGGED=${RAGGED:-}
export RAGGED_THREADS=${RAGGED_THREADS:-}
export SP_T=${SP_T:-}
echo "stages after prep: $STAGES"

# Cheap and irreplaceable first, then the long one. Neither aborts the other.
for s in shapes threads small validate; do
    case " $STAGES " in *" $s "*) ;; *) echo "skipping stage $s"; continue ;; esac
    stage "$s" || echo "WARNING: $s stage failed; continuing" >&2
done

case " $STAGES " in
    *" grid "*) ;;
    *)  echo "grid not in STAGES; stopping here."
        printf '%-10s %s\n' grid skipped >> "$SUMMARY"
        echo "outdir $OUT"; cat "$SUMMARY"; exit 0 ;;
esac

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
echo "Nothing here is comparable to any number in docs/notebook/. Every ratio is"
echo "computed within this session, and the noise floor is the one this node's"
echo "own bracketing repeats produced -- not the committed +-1.3% / +-6%."
