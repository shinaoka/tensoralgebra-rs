#!/usr/bin/env bash
# Phase 2 S3 before/after driver. Per case: pick 8 idle CPUs of one L3 domain,
# then for T in 1 4 8 run BEFORE and AFTER back to back on a prefix of that set
# (each through pinned.sh); a case whose runs are not all valid is discarded
# and retried on a new pick. Same retry scheme as the S0 driver.
#   drive.sh OUT CORPUS_NAME BEFORE_BIN AFTER_BIN CASE...
set -uo pipefail
out=$1; name=$2; bb=$3; ab=$4; shift 4
root=$(cd "$(dirname "$0")/../../../../../../.." && pwd)
corpus=$root/benchmarks/benchmarks/tprims/corpus/$name.json
mkdir -p "$out"
for v in before after; do for t in 1 4 8; do
  echo "case,variant,threads,median_ns,samples" > "$out/$name-$v-${t}t.csv"; : > "$out/$name-$v-${t}t.log"
done; done
echo "case,cpus,attempt" > "$out/$name-cpus.csv"
export BENCH_RUNS=${BENCH_RUNS:-5}
for c in "$@"; do
  ok=0
  for attempt in $(seq 1 60); do
    if ! cpus=$(python3 "$root/benchmarks/scripts/idle_cpus.py" pick 8 --seconds 2 2>/dev/null); then sleep 20; continue; fi
    IFS=, read -ra all <<< "$cpus"
    tmpd=$(mktemp -d); good=1
    for t in 1 4 8; do
      set_cpus=$(IFS=,; echo "${all[*]:0:$t}")
      for v in before after; do
        bin=$bb; [[ $v == after ]] && bin=$ab
        if ! PINNED_RETRIES=1 BENCH_CASE=$c "$root/benchmarks/scripts/pinned.sh" "$set_cpus" -- "$bin" --threads "$t" --corpus "$corpus" > "$tmpd/$v-$t" 2>>"$out/pinned.err"; then good=0; break 2; fi
      done
    done
    if (( good )); then
      for v in before after; do for t in 1 4 8; do
        grep -v -E '^(#|case,|CHECK)' "$tmpd/$v-$t" >> "$out/$name-$v-${t}t.csv"
        grep -E '^(#|CHECK)' "$tmpd/$v-$t" | sed "s/^/[$c] /" >> "$out/$name-$v-${t}t.log"
      done; done
      echo "$c,\"$cpus\",$attempt" >> "$out/$name-cpus.csv"; ok=1; rm -rf "$tmpd"; break
    fi
    rm -rf "$tmpd"
  done
  (( ok )) || echo "SKIPPED $c" >> "$out/$name-skipped.txt"
done
echo done > "$out/$name-done.flag"
