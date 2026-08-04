#!/bin/bash
# The engine against its baselines, unattended, on one exclusive Rusty node.
#
#   sbatch --constraint=icelake scripts/rusty-compare.sbatch
#   sbatch --constraint=rome    scripts/rusty-compare.sbatch
#
# Submit from the repo root. Everything lands in `bench-results/<node>-<arch>/`
# and the job log in `slurm-tp-compare-<jobid>.out`.
#
# The constraint is deliberately *not* set here, because the node choice is the
# experiment and the two answer different questions:
#
#   icelake (2:32:1, 64 cores, AVX-512) -- the same AVX-512 kernels every
#     committed number in this repo used, and the cache hierarchy closest to the
#     reference machine. The catch is A34: three of eight shipped register blocks
#     are 9-13% off on Ice Lake, so the engine is measured handicapped and every
#     engine-versus-baseline ratio from this node *understates* it. Say so
#     wherever the numbers are quoted.
#
#   rome (2:64:1, AVX2, no AVX-512) -- the engine at its measured best, since all
#     eight AVX2 register blocks were confirmed as the winners on Zen2 (part 11,
#     D26). It also closes the open end of A24: the AVX2 method ranking is
#     currently a kernel-level result only, and this is the corpus-level
#     measurement that would settle it.
#
# Running both is worth more than either alone, and is what the README needs to
# say plainly what is and is not tuned.
#
# ---------------------------------------------------------------------------
# What this is NOT comparable to
# ---------------------------------------------------------------------------
#
# Nothing measured here may be compared with a single-core number measured on
# `ccqlin038`: different machine, different cache hierarchy, and on `rome` a
# different instruction set. That is the CHANGELOG's own rule and it is not
# negotiable by convenience. Every ratio this job produces is within-session,
# against a floor derived in-session by the `A2` and `A3` arms.
#
# ---------------------------------------------------------------------------
# Why 12 hours for a 3.5 hour job
# ---------------------------------------------------------------------------
#
# The measurement is sequential by design -- the baselines change memory traffic,
# and A27 rejected concurrent one-arm-per-L3-domain placement on exactly the
# memory-bound half of the corpus. There is no concurrency to fall back on, so
# the wall time carries the margin instead.
#
# `prep` (the only compile) is expected to be a no-op: build the two binaries on
# the submitting host first, since neither carries `target-cpu=native` and the
# engine dispatches its ISA at run time. If they are missing, this job builds
# them, which costs a few minutes and is why cargo has to be reachable.

#SBATCH --job-name=tp-compare
#SBATCH --partition=gen
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-tp-compare-%j.out

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
        | grep -E 'JobId|NumNodes|NumCPUs|OverSubscribe|Shared|NodeList|TimeLimit|Features' || true
fi

# Slurm exports the submitting environment, so a rustup toolchain in $HOME comes
# along. Fail early and legibly if it did not.
command -v cargo >/dev/null || export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || {
    echo "FATAL: no cargo on PATH. Submit from a shell that can build the workspace." >&2
    exit 1
}
cargo --version

# The baselines are outside the repo and are not rebuilt here. Both were checked
# to be multi-configuration libraries carrying AVX-512 and Zen kernels, so the
# same install serves an `icelake` and a `rome` node without a rebuild.
export TBLIS_ROOT_2X=${TBLIS_ROOT_2X:-$PWD/../baselines/tblis-2.0-install}
export TBLIS_ROOT_13=${TBLIS_ROOT_13:-$PWD/../baselines/tblis-1.3.0-install}
for r in "$TBLIS_ROOT_2X" "$TBLIS_ROOT_13"; do
    [ -d "$r/lib" ] || { echo "FATAL: no TBLIS install at $r" >&2; exit 1; }
done
echo "tblis 2.0-dev : $TBLIS_ROOT_2X"
echo "tblis v1.3.0  : $TBLIS_ROOT_13"

OUT=${OUT:-bench-results/$(hostname -s)-$(scripts/arch-label.sh)}
CPU=${CPU:-}
export PREMISE_SIZE=${PREMISE_SIZE:-200}
echo "outdir   : $OUT"
echo

# Pick a measurement core that owns its whole L3 domain, rather than assuming
# cpu4 as the reference machine's scripts do. `topology.py` chooses domain
# leaders and leaves SMT siblings idle; the first leader is as good as any on an
# exclusive node, and recording the choice is the point.
if [ -z "$CPU" ]; then
    mkdir -p "$OUT"
    scripts/topology.py --quiet --jobs 1 --json "$OUT/topology-plan.json" || true
    CPU=$(python3 -c "
import json, sys
try:
    d = json.load(open('$OUT/topology-plan.json'))
    print(d['placement']['slots'][0]['cpu'])
except Exception:
    print(4)")
fi
echo "measurement core: cpu$CPU"

# prep is the only stage that compiles, and it runs before anything is timed.
if ! scripts/compare-bench.sh prep; then
    echo "FATAL: prep failed, so there is nothing to measure with." >&2
    exit 1
fi

start=$SECONDS
if scripts/compare-bench.sh "$OUT" "$CPU"; then rc=0; else rc=$?; fi
echo
echo "=========================================================="
echo "compare-bench rc=$rc after $(((SECONDS - start) / 60)) min"
echo "finished : $(date -Iseconds)"
echo "=========================================================="

# Keep the job log with the data it describes; the log is the only record of
# what Slurm actually granted.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "$OUT" ]; then
    echo "${SLURM_JOB_ID} $(hostname -s) $(date -Iseconds)" > "$OUT/slurm-job.txt"
    cp -f "slurm-tp-compare-${SLURM_JOB_ID}.out" "$OUT/" 2>/dev/null || true
fi
exit $rc
