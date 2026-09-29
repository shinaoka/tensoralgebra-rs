#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Full benchmark runner: Rust (faer + blas) + Julia
#
# Usage: ./scripts/run_all.sh [NUM_THREADS]
#
# Delegates to run_all_rust.sh and run_all_julia.sh, then formats results.
#
# IMPORTANT: On Linux, set OPENBLAS_LIB_DIR and LD_LIBRARY_PATH to a
# recent OpenBLAS (>= 0.3.29). See docs/environment-setup.md for details.
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"

export BENCHMARK_TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

echo "============================================"
echo " strided-rs benchmark suite"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Threads:      $NUM_THREADS"
echo "Timestamp:    $BENCHMARK_TIMESTAMP"
echo ""

# ---------------------------------------------------------------------------
# Rust benchmarks (faer + blas)
# ---------------------------------------------------------------------------
"$SCRIPT_DIR/run_all_rust.sh" "$NUM_THREADS"

# ---------------------------------------------------------------------------
# Julia benchmarks
# ---------------------------------------------------------------------------
"$SCRIPT_DIR/run_all_julia.sh" "$NUM_THREADS"

# ---------------------------------------------------------------------------
# Format results as markdown table
# ---------------------------------------------------------------------------
RUST_FAER_LOG="$RESULTS_DIR/rust_faer_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
RUST_BLAS_LOG="$RESULTS_DIR/rust_blas_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
JULIA_LOG="$RESULTS_DIR/julia_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
MARKDOWN_OUT="$RESULTS_DIR/results_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"

LOGS=()
[ -f "$RUST_FAER_LOG" ] && LOGS+=("$RUST_FAER_LOG")
[ -f "$RUST_BLAS_LOG" ] && LOGS+=("$RUST_BLAS_LOG")
[ -f "$JULIA_LOG" ] && LOGS+=("$JULIA_LOG")

if [ ${#LOGS[@]} -gt 0 ]; then
    echo "Formatting results as markdown..."
    uv run python "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_OUT"
    echo ""
fi

echo "============================================"
echo " Benchmark complete"
echo "============================================"
echo "Results:"
for log in "${LOGS[@]}"; do
    echo "  $log"
done
[ -f "$MARKDOWN_OUT" ] && echo "  Markdown: $MARKDOWN_OUT"
