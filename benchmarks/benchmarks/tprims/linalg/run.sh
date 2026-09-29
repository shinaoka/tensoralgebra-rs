#!/usr/bin/env bash
# Measure every linalg case in its own process at 1T then 4T.
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
for threads in 1 4; do
    cpus=$cpus1; [[ $threads == 4 ]] && cpus=$cpus4
    f="$out/linalg-${threads}t.csv"
    echo "case,variant,threads,median_ns,samples" > "$f"
    for c in "${cases[@]}"; do
        BENCH_CASE=$c taskset -c "$cpus" "$bin" --threads "$threads" | grep -v -E '^(#|case,)' >> "$f"
    done
done
