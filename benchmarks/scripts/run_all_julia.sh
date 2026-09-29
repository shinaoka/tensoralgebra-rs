#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Julia benchmark runner (OMEinsum.jl + TensorOperations.jl)
#
# Usage: ./scripts/run_all_julia.sh [NUM_THREADS]
#
# NUM_THREADS (default: 1) controls:
#   - OMP_NUM_THREADS   (OpenBLAS internal threading)
#   - JULIA_NUM_THREADS  (Julia task parallelism)
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

export OMP_NUM_THREADS="$NUM_THREADS"
export JULIA_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"

mkdir -p "$RESULTS_DIR"

TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

echo "============================================"
echo " Julia benchmark (threads=${NUM_THREADS})"
echo "============================================"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  JULIA_NUM_THREADS=$JULIA_NUM_THREADS"
echo ""

JULIA_LOG="$RESULTS_DIR/julia_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Running Julia benchmark..."
julia --startup-file=no --project="$PROJECT_DIR" "$PROJECT_DIR/src/main.jl" 2>&1 | tee "$JULIA_LOG"

echo ""
echo "============================================"
echo " Julia benchmark complete"
echo "============================================"
echo "Results:"
echo "  $JULIA_LOG"
