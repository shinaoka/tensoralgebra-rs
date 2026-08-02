#!/bin/bash
# Phase 3 measurement set: vectorised micro-kernels vs baselines.
#
# Runs everything single-threaded, one measurement at a time, with nothing else
# on the machine. Writes CSVs to bench-results/ and a transcript to
# bench-results/phase3-log.txt.
#
#   scripts/phase3-bench.sh [sweep_size_mib] [reps]
#
# `premise` runs at 200 MiB, the size Phase 1 used, so that the rebuilt TBLIS
# baselines reproduce the Phase 1 numbers and the efficiency-vs-GEMM-ceiling
# metric stays comparable across phases. The full 49-case sweeps run at a
# smaller size because at 200 MiB a single case is ~5 minutes of wall clock per
# engine and the corpus would take a day.
#
# Expects TBLIS_ROOT_2X and TBLIS_ROOT_13 to point at the two TBLIS installs;
# see DECISIONS.md for how they were built.

set -euo pipefail
cd "$(dirname "$0")/.."

SWEEP_SIZE=${1:-64}
PREMISE_SIZE=200
REPS=${2:-3}
OUT=bench-results
LOG=$OUT/phase3-log.txt

module load gcc/13.3.0 openblas >/dev/null 2>&1 || true
: "${TBLIS_ROOT_2X:?point at the TBLIS 2.0-dev install}"
: "${TBLIS_ROOT_13:?point at the TBLIS v1.3.0 install}"

export TBLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $OUT
: > $LOG

run() { echo -e "\n\n########## $* ##########\n" | tee -a $LOG; "$@" 2>&1 | tee -a $LOG; }

# ---- against TBLIS 2.0-dev ------------------------------------------------
export TBLIS_ROOT=$TBLIS_ROOT_2X
export LD_LIBRARY_PATH="$TBLIS_ROOT/lib:${OPENBLAS_ROOT:+$OPENBLAS_ROOT/lib:}${LD_LIBRARY_PATH:-}"
cargo build --release -p tensorcontract-bench --features tblis,blas
B=./target/release/tcbench

# 1. Correctness of the new kernels against both baselines, all methods.
for m in planar 1m 3m; do
  TENSORCONTRACT_COMPLEX=$m run $B verify --size 32
done

# 2. Efficiency against a same-shape GEMM ceiling.
run $B premise --size "$PREMISE_SIZE" --reps "$REPS" \
    --engines planar,1m,3m,ttgt,tblis --dtype f64,c64 --csv $OUT/phase3-premise-f64c64.csv
run $B premise --size "$PREMISE_SIZE" --reps "$REPS" \
    --engines planar,1m,3m,ttgt,tblis --dtype f32,c32 --csv $OUT/phase3-premise-f32c32.csv

# 3. The headline: full 49-case corpus, three methods plus baselines.
run $B sweep --size "$SWEEP_SIZE" --reps "$REPS" \
    --engines planar,1m,3m,ttgt,tblis --dtype f64,c64 --csv $OUT/phase3-sweep-f64c64.csv
run $B sweep --size "$SWEEP_SIZE" --reps "$REPS" \
    --engines planar,1m,3m,ttgt,tblis --dtype f32,c32 --csv $OUT/phase3-sweep-f32c32.csv

# 4. Irregular strides: the only mode that exercises the gather path.
run $B sweep --size "$SWEEP_SIZE" --reps "$REPS" --stress ragged \
    --engines planar,1m,3m,tblis --dtype c64 --csv $OUT/phase3-sweep-ragged-c64.csv

# ---- against TBLIS v1.3.0 -------------------------------------------------
export TBLIS_ROOT=$TBLIS_ROOT_13
export LD_LIBRARY_PATH="$TBLIS_ROOT/lib:${OPENBLAS_ROOT:+$OPENBLAS_ROOT/lib:}${LD_LIBRARY_PATH:-}"
cargo build --release -p tensorcontract-bench --features tblis13,blas
run $B premise --size "$PREMISE_SIZE" --reps "$REPS" \
    --engines planar,tblis --dtype f64,c64 --csv $OUT/phase3-premise-tblis130-f64c64.csv

echo -e "\nphase 3 measurement set complete; transcript in $LOG"
