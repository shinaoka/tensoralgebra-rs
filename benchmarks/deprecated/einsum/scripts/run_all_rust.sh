#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Rust benchmark runner (faer + blas backends)
#
# Usage: ./scripts/run_all_rust.sh [NUM_THREADS]
#
# NUM_THREADS (default: 1) controls:
#   - OMP_NUM_THREADS   (OpenBLAS internal threading)
#   - RAYON_NUM_THREADS  (Rust rayon parallelism)
#
# IMPORTANT: On Linux, set OPENBLAS_LIB_DIR and LD_LIBRARY_PATH to a
# recent OpenBLAS (>= 0.3.29). The system libopenblas-dev on Ubuntu 20.04
# is 0.3.8, which is ~2x slower than Julia's bundled 0.3.29 on AMD EPYC.
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

export OMP_NUM_THREADS="$NUM_THREADS"
export RAYON_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"

# Auto-detect custom OpenBLAS at $HOME/opt/openblas-0.3.29
CUSTOM_OPENBLAS="$HOME/opt/openblas-0.3.29/lib"
if [ -z "${OPENBLAS_LIB_DIR:-}" ] && [ -d "$CUSTOM_OPENBLAS" ]; then
    export OPENBLAS_LIB_DIR="$CUSTOM_OPENBLAS"
    export LD_LIBRARY_PATH="${CUSTOM_OPENBLAS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Auto-detected OpenBLAS 0.3.29 at $CUSTOM_OPENBLAS"
fi

mkdir -p "$RESULTS_DIR"

TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

echo "============================================"
echo " Rust benchmark (threads=${NUM_THREADS})"
echo "============================================"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo ""

RUST_LOGS=()

# ---------------------------------------------------------------------------
# faer backend
# ---------------------------------------------------------------------------
echo "============================================"
echo " Rust: strided-opteinsum (faer)"
echo "============================================"

RUST_FAER_LOG="$RESULTS_DIR/rust_faer_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Building Rust (release, faer)..."
cargo build --release --no-default-features --features faer,parallel --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1

echo "Running Rust benchmark (faer)..."
cargo run --release --no-default-features --features faer,parallel --bin strided-rs-benchmark-suite --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1 | tee "$RUST_FAER_LOG"

echo ""
echo "Rust (faer) results saved to: $RUST_FAER_LOG"
echo ""
RUST_LOGS+=("$RUST_FAER_LOG")

# ---------------------------------------------------------------------------
# blas (OpenBLAS) backend
# ---------------------------------------------------------------------------
echo "============================================"
echo " Rust: strided-opteinsum (blas)"
echo "============================================"

RUST_BLAS_LOG="$RESULTS_DIR/rust_blas_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Building Rust (release, blas)..."
if cargo build --release --no-default-features --features blas,parallel --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1; then
    echo "Running Rust benchmark (blas)..."
    cargo run --release --no-default-features --features blas,parallel --bin strided-rs-benchmark-suite --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1 | tee "$RUST_BLAS_LOG"
    echo ""
    echo "Rust (blas) results saved to: $RUST_BLAS_LOG"
    RUST_LOGS+=("$RUST_BLAS_LOG")
else
    echo "WARNING: blas build failed (OpenBLAS not found?). Skipping blas benchmark."
    echo "  Install OpenBLAS: brew install openblas (macOS) / apt install libopenblas-dev (Ubuntu)"
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Rust benchmark complete"
echo "============================================"
echo "Results:"
for log in "${RUST_LOGS[@]}"; do
    echo "  $log"
done
