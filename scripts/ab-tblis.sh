#!/bin/bash
# A/B two TBLIS 2.0 builds against each other: `warm-up, A, B, A'`.
#
#   scripts/ab-tblis.sh OUTDIR [cpu] [size] [reps]
#
# Why this exists. `../baselines/tblis-2.0-install` was configured with
# BLIS_CONFIG_FAMILY=auto, which detects the *build* machine, so its BLIS is
# skx-only: 1364 `bli_` symbols, one `bli_cntx_init_skx` and no other context
# initialiser. `../baselines/tblis-2.0-x86_64-install` is the same source at the
# same commit with BLIS_CONFIG_FAMILY=x86_64: 3809 symbols, thirteen contexts
# including zen2, and BLIS's skinny-GEMM (`sup`) kernel set. The only differing
# cmake option is the config family; everything else in both CMakeCaches matches.
#
# A one-shot look on a shared machine put the difference at up to 1.68x, clustered
# on the low-arithmetic-intensity cases -- which is precisely where this project
# says the headroom is, and precisely where its README claims near-parity with
# TBLIS 2.0. That look was `--size 8 --reps 2` with no warm-up and no repeat, so
# by this project's own rules it is a signal and not a result. This script is the
# result.
#
# **The engine is the control, and it is a good one.** Every arm measures
# `planar` as well as `tblis`, and which TBLIS shared object is loaded cannot
# affect the engine's own throughput. So the `planar` column prices the
# environment and the `tblis` column carries the treatment, in the same arm, on
# the same core, minutes apart. `ab-deepen` had to reconstruct that after the
# fact from three columns the switch could not touch; here it is by construction.
#
# The treatment is `LD_LIBRARY_PATH`, i.e. a runtime switch on one binary rather
# than two builds (A15). Verified: the binary carries no TBLIS RPATH, and `ldd`
# resolves whichever install LD_LIBRARY_PATH names.
#
# Cost: about 2.7 h -- four sweep arms x two dtype pairs (~1.5 h), then three
# premise arms at 200 MiB in f64/c64 (~1.2 h). Needs an exclusive machine, and
# needs one **with AVX-512** -- the skx build SIGILLs anywhere else, which is how
# this whole thing was found (job 6753261).
set -e -o pipefail
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT=${1:?usage: scripts/ab-tblis.sh OUTDIR [cpu] [size] [reps]}
CPU=${2:-4}
SIZE=${3:-64}
REPS=${4:-3}
# The premise stage keeps Phase 1's 200 MiB so its shapes are the ones the
# efficiency-against-a-GEMM-ceiling metric was defined on.
PREMISE_SIZE=${PREMISE_SIZE:-200}
BIN=./target/tblis2/release/tcbench

A_ROOT=${TBLIS_ROOT_2X_A:-$PWD/../baselines/tblis-2.0-install}
B_ROOT=${TBLIS_ROOT_2X_B:-$PWD/../baselines/tblis-2.0-x86_64-install}

module load gcc/13.3.0 openblas >/dev/null 2>&1 || true
grep -qw avx512f /proc/cpuinfo || {
    echo "REFUSING: the skx build needs AVX-512; this machine has none." >&2
    echo "That is the failure this A/B is about -- see the header." >&2
    exit 1
}
[ -s "$BIN" ] || { echo "no $BIN -- scripts/compare-bench.sh prep" >&2; exit 1; }
for r in "$A_ROOT" "$B_ROOT"; do
    [ -s "$r/lib/libtblis.so" ] || { echo "no TBLIS 2.0 install at $r" >&2; exit 1; }
done

others=$(pgrep -a -u "$USER" -f 'cargo|rustc|tcbench|kernel_shapes' \
         | grep -v "ab-tblis\.sh\|pgrep" || true)
if [ -n "$others" ]; then
    echo "REFUSING: something of yours is running:" >&2; echo "$others" >&2; exit 1
fi

mkdir -p "$OUT"
LOG=$OUT/ab-tblis.log
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }

export TBLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
BASE_LD="${OPENBLAS_ROOT:+$OPENBLAS_ROOT/lib:}${LD_LIBRARY_PATH:-}"
A_LD="$A_ROOT/lib:$BASE_LD"
B_LD="$B_ROOT/lib:$BASE_LD"

say "TBLIS 2.0 build A/B on cpu$CPU"
say "  A (skx, as every committed number used) : $A_ROOT"
say "  B (x86_64 multi-config)                 : $B_ROOT"
say "  control: the planar column, which no TBLIS library can move"
say "started $(date -Is) on $(hostname -s)"

# Record what each library actually is, so the claim is in the output directory
# rather than in a commit message.
{
    for tag in A B; do
        [ $tag = A ] && r=$A_ROOT || r=$B_ROOT
        echo "=== $tag: $r ==="
        echo -n "bli_ symbols: "; nm -D "$r/lib/libtblis.so" | grep -c 'bli_' || true
        echo -n "contexts    : "
        nm -D "$r/lib/libtblis.so" | grep -oE 'bli_cntx_init_[a-z0-9]+' \
            | sort -u | sed 's/bli_cntx_init_//' | tr '\n' ' '; echo
    done
} | tee "$OUT/libraries.txt" | tee -a "$LOG"

scripts/topology.py --json "$OUT/topology.json" >/dev/null

# The warm-up is discarded (A31). A' brackets B so the treatment sits between two
# measurements of the same thing (A32).
jobs="$OUT/arms.jobs"
{
    echo "# treatment is which libtblis.so is loaded; planar is the in-arm control"
    for pair in f64,c64 f32,c32; do
        t=${pair/,/}
        echo "warm-$t $pair LD_LIBRARY_PATH=$A_LD"
        echo "A-$t    $pair LD_LIBRARY_PATH=$A_LD"
        echo "B-$t    $pair LD_LIBRARY_PATH=$B_LD"
        echo "A2-$t   $pair LD_LIBRARY_PATH=$A_LD"
    done
} > "$jobs"

scripts/run-arms.py "$jobs" --outdir "$OUT" --bin "$BIN" \
    --size "$SIZE" --reps "$REPS" --engines planar,tblis \
    --cpus "$CPU" --sequential | tee -a "$LOG"

for tag in A-f64c64 A-f32c32 B-f64c64 B-f32c32 A2-f64c64 A2-f32c32; do
    [ -s "$OUT/$tag.csv" ] && [ "$(wc -l < "$OUT/$tag.csv")" -ge 2 ] || {
        echo "FATAL: arm '$tag' produced no measurements" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# The same A/B on the premise shapes, because that is where the signal was
# ---------------------------------------------------------------------------
# The one-shot look that started this used `premise`, not `sweep`: a different,
# hand-picked 12-case set at 200 MiB, much of it deliberately skinny. A tiny
# `sweep` dry run showed the two libraries indistinguishable, which is exactly the
# result that would make skipping this stage a mistake -- an A/B that cannot see
# the effect it was built to price is not evidence of absence.
#
# `f64,c64` only, and no warm-up: the package is hot by now, and three arms at
# 200 MiB is already ~70 min. A' still brackets B.
say ""
say "###### the same A/B on the premise shapes, at ${PREMISE_SIZE} MiB ######"
pjobs="$OUT/arms-premise.jobs"
{
    echo "pA-f64c64  f64,c64 LD_LIBRARY_PATH=$A_LD"
    echo "pB-f64c64  f64,c64 LD_LIBRARY_PATH=$B_LD"
    echo "pA2-f64c64 f64,c64 LD_LIBRARY_PATH=$A_LD"
} > "$pjobs"

scripts/run-arms.py "$pjobs" --outdir "$OUT" --bin "$BIN" \
    --subcommand premise --size "$PREMISE_SIZE" --reps "$REPS" \
    --engines planar,tblis --cpus "$CPU" --sequential | tee -a "$LOG"

for tag in pA-f64c64 pB-f64c64 pA2-f64c64; do
    [ -s "$OUT/$tag.csv" ] && [ "$(wc -l < "$OUT/$tag.csv")" -ge 2 ] || {
        echo "FATAL: arm '$tag' produced no measurements" >&2; exit 1; }
done

say ""
say "###### premise floor: pA' against pA ######"
scripts/compare-sweeps.py "$OUT/pA-f64c64.csv" "$OUT/pA2-f64c64.csv" \
    | tee "$OUT/floor-premise.txt" | tee -a "$LOG"
say ""
say "###### premise treatment: multi-config against skx ######"
scripts/compare-sweeps.py "$OUT/pA-f64c64.csv" "$OUT/pB-f64c64.csv" \
    | tee "$OUT/treatment-premise.txt" | tee -a "$LOG"

say ""
say "###### this session's floor: A' against A ######"
scripts/compare-sweeps.py "$OUT/A-f64c64.csv,$OUT/A-f32c32.csv" \
    "$OUT/A2-f64c64.csv,$OUT/A2-f32c32.csv" | tee "$OUT/floor.txt" | tee -a "$LOG"
say ""
say "###### the treatment: multi-config against skx ######"
scripts/compare-sweeps.py "$OUT/A-f64c64.csv,$OUT/A-f32c32.csv" \
    "$OUT/B-f64c64.csv,$OUT/B-f32c32.csv" | tee "$OUT/treatment.txt" | tee -a "$LOG"

{
    echo "Produced by scripts/ab-tblis.sh on $(hostname -f)."
    echo "  finished  : $(date -Is)"
    echo "  slurm job : ${SLURM_JOB_ID:-<not under slurm>}"
    echo "  commit    : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "  A (skx)   : $A_ROOT"
    echo "  B (x86_64): $B_ROOT"
    echo "  size ${SIZE} MiB, ${REPS} reps, cpu$CPU, arms sequential"
    echo
    echo "Both libraries are TBLIS 2.0-dev at the same commit (555320c). They"
    echo "differ only in BLIS_CONFIG_FAMILY: auto (which resolved to skx-only on"
    echo "the Cascade Lake build host) against x86_64 (multi-config). See"
    echo "libraries.txt for each one's symbol count and context list."
    echo
    echo "HOW TO READ IT. treatment.txt has two engine columns per dtype:"
    echo "  * planar -- the CONTROL. No TBLIS library can change it. Whatever it"
    echo "    reads is this session's environment, and the tblis column has to be"
    echo "    judged against it, not against 1.000."
    echo "  * tblis  -- the treatment."
    echo "floor.txt is A' against A, the same library twice, minutes apart."
    echo
    echo "Nothing here is comparable to a number measured on another machine."
} > "$OUT/PROVENANCE.txt"

say ""
say "done $(date -Is). Read $OUT/PROVENANCE.txt, then treatment.txt."
say "Judge the tblis column against the planar control in the same arm."
