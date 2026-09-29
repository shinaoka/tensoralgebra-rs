#!/usr/bin/env bash
# Every contract case in its own process, 1T and 4T paired per case.
#   run.sh CPUS1 CPUS4 OUTDIR
set -euo pipefail
cpus1=$1; cpus4=$2; out=$3
root=$(cd "$(dirname "$0")/../../../.." && pwd)
bin=$root/target/release/contract
mkdir -p "$out"
for t in 1 4; do echo "case,variant,threads,median_ns,samples" > "$out/contract-${t}t.csv"; : > "$out/contract-${t}t.log"; done
for c in tiny_matmul matmul_256 batched_64_b32 permuted_fusable permuted_nonfusable network_ijkl_klmn large_ijk_jkl; do
    for ty in f64 c64; do
        for t in 1 4; do
            cpus=$cpus1; [[ $t == 4 ]] && cpus=$cpus4
            BENCH_CASE=${c}_${ty} taskset -c "$cpus" "$bin" --threads "$t" > "$out/tmp.txt"
            grep -v -E '^(#|case,|CHECK)' "$out/tmp.txt" >> "$out/contract-${t}t.csv"
            grep -E '^(# selected|CHECK)' "$out/tmp.txt" >> "$out/contract-${t}t.log" || true
        done
    done
done
