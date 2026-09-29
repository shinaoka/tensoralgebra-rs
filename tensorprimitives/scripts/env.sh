#!/bin/bash
# Environment for building the benchmark harness with its C baselines.
#
# Usage:  source scripts/env.sh
#
# Expects a TBLIS install prefix in TBLIS_ROOT. Build one with:
#   git clone --branch develop https://github.com/devinamatthews/tblis
#   cd tblis && git submodule update --init --recursive
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_INSTALL_PREFIX=$PWD/../tblis-install -DBUILD_SHARED_LIBS=ON
#   cmake --build build -j && cmake --install build

module load gcc/13.3.0 openblas 2>/dev/null

: "${TBLIS_ROOT:?set TBLIS_ROOT to a TBLIS install prefix}"
export TBLIS_ROOT
export LD_LIBRARY_PATH="$TBLIS_ROOT/lib:${OPENBLAS_ROOT:+$OPENBLAS_ROOT/lib:}$LD_LIBRARY_PATH"

# Keep every baseline single-threaded; this project's comparisons are
# single-core unless stated otherwise.
export TBLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "TBLIS_ROOT=$TBLIS_ROOT"
echo "OPENBLAS_ROOT=${OPENBLAS_ROOT:-<unset>}"
