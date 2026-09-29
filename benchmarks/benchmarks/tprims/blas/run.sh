#!/usr/bin/env bash
# Batched GEMM entries of a corpus, each in its own process, thread counts
# paired per case.
#   CORPUS=file run.sh OUTDIR CPUS THREADS...
set -euo pipefail
: "${CORPUS:?set CORPUS to a corpus file with gemm_batched entries}"
root=$(cd "$(dirname "$0")/../../../.." && pwd)
exec "$root/benchmarks/scripts/paired.sh" "$root/target/release/blas" "$@"
