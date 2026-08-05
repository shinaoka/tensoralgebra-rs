#!/bin/bash
# Price the TBLIS 2.0 build difference, then re-run the comparison against the
# build that turns out to be right. One exclusive AVX-512 node, one queue wait.
#
#   sbatch --constraint=icelake scripts/rusty-tblis-ab.sbatch
#
# **The constraint must name an AVX-512 node.** The skx-only build SIGILLs
# anywhere else, and it is one of the two arms, so `rome` cannot run stage 1 at
# all -- that failure is what started this (job 6753261).
#
# Two stages, ~5 h total inside a 12 h wall:
#
#   1. `ab-tblis.sh`      ~1.5 h -- warm-up, A (skx), B (x86_64), A' (skx), with
#                                   `planar` in every arm as a control no TBLIS
#                                   library can move.
#   2. `compare-bench.sh` ~3.5 h -- the full comparison set against the
#                                   multi-config build, so there is a like-for-like
#                                   counterpart to the skx run from job 6753260.
#
# Stage 2 runs even if stage 1 fails, because they answer different questions and
# losing the 3.5 h set to a 1.5 h stage's failure would be absurd. Each stage's
# status is reported at the end.
#
# Nothing here compiles: run `scripts/compare-bench.sh prep` before submitting.

#SBATCH --job-name=tp-tblis-ab
#SBATCH --partition=gen
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-tp-tblis-ab-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "=========================================================="
echo "job      : ${SLURM_JOB_ID:-<none>} on $(hostname -s)"
echo "started  : $(date -Iseconds)"
echo "=========================================================="
if [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol show job "$SLURM_JOB_ID" 2>&1 \
        | grep -E 'JobId|NumNodes|NumCPUs|OverSubscribe|NodeList|TimeLimit|Features' || true
fi

if ! grep -qw avx512f /proc/cpuinfo; then
    echo "FATAL: this node has no AVX-512, so the skx arm cannot run." >&2
    echo "Resubmit with --constraint=icelake." >&2
    exit 1
fi

export TBLIS_ROOT_2X_A=${TBLIS_ROOT_2X_A:-$PWD/../baselines/tblis-2.0-install}
export TBLIS_ROOT_2X_B=${TBLIS_ROOT_2X_B:-$PWD/../baselines/tblis-2.0-x86_64-install}
export TBLIS_ROOT_13=${TBLIS_ROOT_13:-$PWD/../baselines/tblis-1.3.0-install}
for r in "$TBLIS_ROOT_2X_A" "$TBLIS_ROOT_2X_B" "$TBLIS_ROOT_13"; do
    [ -s "$r/lib/libtblis.so" ] || { echo "FATAL: no TBLIS install at $r" >&2; exit 1; }
done

if [ ! -s target/tblis2/release/tcbench ] || [ ! -s target/tblis13/release/tcbench ]; then
    echo "FATAL: binaries missing. Before submitting, run:" >&2
    echo "  TBLIS_ROOT_2X=$TBLIS_ROOT_2X_A TBLIS_ROOT_13=$TBLIS_ROOT_13 \\" >&2
    echo "    scripts/compare-bench.sh prep" >&2
    exit 1
fi

# Snapshot into a build tree private to this job, so a `prep` or an editing
# session on the submit host cannot relink or truncate the binary a running arm
# invokes. See `rusty-compare.sbatch` for the job this killed.
export TC_TARGET=${TC_TARGET:-$PWD/target-job-${SLURM_JOB_ID:-manual}}
mkdir -p "$TC_TARGET/tblis2/release" "$TC_TARGET/tblis13/release"
cp -f target/tblis2/release/tcbench  "$TC_TARGET/tblis2/release/tcbench"
cp -f target/tblis13/release/tcbench "$TC_TARGET/tblis13/release/tcbench"
BIN=$TC_TARGET/tblis2/release/tcbench
echo "binaries snapshotted into $TC_TARGET; this job will not compile"

OUT=${OUT:-bench-results/$(hostname -s)-$(scripts/arch-label.sh)}
mkdir -p "$OUT"

# One core that owns its whole L3 domain, recorded rather than assumed.
CPU=${CPU:-}
if [ -z "$CPU" ]; then
    scripts/topology.py --quiet --jobs 1 --json "$OUT/topology-plan.json" || true
    CPU=$(python3 -c "
import json
try:
    print(json.load(open('$OUT/topology-plan.json'))['placement']['slots'][0]['cpu'])
except Exception:
    print(4)")
fi
echo "outdir   : $OUT"
echo "measurement core: cpu$CPU"
SUMMARY="$OUT/stage-status.txt"
: > "$SUMMARY"

stage() {  # stage <name> <command...>
    local name=$1; shift
    local start=$SECONDS rc
    echo
    echo "########## stage $name  $(date -Iseconds) ##########"
    if "$@"; then rc=0; else rc=$?; fi
    printf '%-22s rc=%-4s %6d min\n' "$name" "$rc" "$(((SECONDS - start) / 60))" \
        | tee -a "$SUMMARY"
    return $rc
}

# Stage 1: which TBLIS 2.0 build is the honest baseline, and by how much.
stage ab-tblis scripts/ab-tblis.sh "$OUT/ab-tblis" "$CPU" || true

# Stage 2: the full comparison set against the multi-config build. Same shape as
# job 6753260's run, which used the skx build, so the two are directly comparable
# within this machine class.
# Exported, not prefixed: `VAR=x func` leaks the assignment past the call in
# bash, and this one decides which baseline a 3.5 h measurement uses.
export TBLIS_ROOT_2X=$TBLIS_ROOT_2X_B
stage compare-tblis-x86_64 scripts/compare-bench.sh \
    "$OUT/compare-tblis-x86_64" "$CPU" || true

echo
echo "=========================================================="
cat "$SUMMARY"
echo "finished : $(date -Iseconds)"
echo "=========================================================="
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "${SLURM_JOB_ID} $(hostname -s) $(date -Iseconds)" > "$OUT/slurm-job.txt"
    cp -f "slurm-tp-tblis-ab-${SLURM_JOB_ID}.out" "$OUT/" 2>/dev/null || true
fi
