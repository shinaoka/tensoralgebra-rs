#!/usr/bin/env bash
# Measure every linalg case in its own process, 1T and 4T paired per case.
#   run.sh CPUS1 CPUS4 OUTDIR     e.g. run.sh 1 1-4 /tmp/out
set -euo pipefail
cpus1=$1; cpus4=$2; out=$3
root=$(cd "$(dirname "$0")/../../../.." && pwd)
bin=$root/target/release/linalg
mkdir -p "$out"
cases=()
for op in cholesky lu solve qr svd eigh eig batched_solve batched_cholesky batched_eigh batched_svd; do
    for t in f64 c64; do cases+=("${op}_${t}"); done
done
# Paired: each case runs at 1T and then 4T back to back, so slow drift of the
# host state (boost frequency, neighbours) cannot bias one side.
for t in 1 4; do echo "case,variant,threads,median_ns,samples" > "$out/linalg-${t}t.csv"; done
for c in "${cases[@]}"; do
    BENCH_CASE=$c taskset -c "$cpus1" "$bin" --threads 1 | grep -v -E '^(#|case,)' >> "$out/linalg-1t.csv"
    BENCH_CASE=$c taskset -c "$cpus4" "$bin" --threads 4 | grep -v -E '^(#|case,)' >> "$out/linalg-4t.csv"
done
