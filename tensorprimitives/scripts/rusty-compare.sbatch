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
# ---------------------------------------------------------------------------
# This job does not compile, and that is a hard precondition
# ---------------------------------------------------------------------------
#
# Run `scripts/compare-bench.sh prep` on the submitting host before submitting.
# Neither binary carries `target-cpu=native` and the engine dispatches its ISA at
# run time, so a binary built anywhere runs correctly everywhere.
#
# The reason this is a precondition rather than a convenience: two jobs submitted
# together share one checkout on GPFS, so two concurrent `prep`s race in the same
# cargo target directory. That is not hypothetical -- it is how jobs 6753197 and
# 6753198 died, one on a `cp` from a file cargo had not relinked and the other on
# a 0-byte binary the race left behind, which then produced seven stages of arms
# that each took 0.0 seconds and returned success.

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


# The baselines are outside the repo and are not rebuilt here. Which TBLIS 2.0
# install is usable depends on the node, and getting this wrong is not a subtle
# failure -- it is a SIGILL two seconds in (job 6753261):
#
#   * `tblis-2.0-install` was configured with BLIS_CONFIG_FAMILY=auto on
#     ccqlin038, so its BLIS is **skx-only** and issues AVX-512 unconditionally.
#     Every number committed in this repo used it, so it is the one to prefer
#     wherever it runs at all.
#   * `tblis-2.0-x86_64-install` is the same source with BLIS_CONFIG_FAMILY=x86_64,
#     i.e. multi-configuration with runtime dispatch. It is what a machine without
#     AVX-512 needs.
#
# TBLIS v1.3.0 needs no such care: it is genuinely multi-config already
# (zen/skx/haswell/sandybridge, each with a runtime check), which is why the
# failing job got past `info` and died inside `verify`.
if [ -z "${TBLIS_ROOT_2X:-}" ]; then
    if grep -qw avx512f /proc/cpuinfo; then
        TBLIS_ROOT_2X=$PWD/../baselines/tblis-2.0-install
        echo "node has AVX-512: using the skx TBLIS 2.0 build, as every committed number did"
    else
        TBLIS_ROOT_2X=$PWD/../baselines/tblis-2.0-x86_64-install
        echo "node has no AVX-512: using the multi-config TBLIS 2.0 build"
    fi
fi
export TBLIS_ROOT_2X
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

# The binaries must already exist and must actually run. This job never builds.
for b in target/tblis2/release/tcbench target/tblis13/release/tcbench; do
    if [ ! -s "$b" ] || [ ! -x "$b" ]; then
        echo "FATAL: $b is missing or empty. Before submitting, on the login or" >&2
        echo "submitting host, run:" >&2
        echo "  TBLIS_ROOT_2X=$TBLIS_ROOT_2X TBLIS_ROOT_13=$TBLIS_ROOT_13 \\" >&2
        echo "    scripts/compare-bench.sh prep" >&2
        exit 1
    fi
done

# A build tree private to this job -- here a *snapshot*, because this job never
# compiles and `prep` ran on the submit host. Without it the arms would keep
# executing `target/tblis{2,13}/release/tcbench` in the submit directory, which a
# later `prep` or an editing session can relink or truncate underneath them. That
# is not hypothetical: job 6753197 died because a concurrent copy left a 0-byte
# executable that every arm then ran happily for 0.0 seconds. Copying costs a few
# seconds once and removes the whole class.
export TC_TARGET=${TC_TARGET:-$PWD/target-job-${SLURM_JOB_ID:-manual}}
mkdir -p "$TC_TARGET/tblis2/release" "$TC_TARGET/tblis13/release"
cp -f target/tblis2/release/tcbench  "$TC_TARGET/tblis2/release/tcbench"
cp -f target/tblis13/release/tcbench "$TC_TARGET/tblis13/release/tcbench"

BIN2=$TC_TARGET/tblis2/release/tcbench
BIN13=$TC_TARGET/tblis13/release/tcbench
echo "binaries snapshotted into $TC_TARGET; this job will not compile"

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
