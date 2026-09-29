#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
strided_rs=${STRIDED_RS_DIR:-"$root/../strided-rs"}
expected=${STRIDED_RS_REF:-b40cd2f6d83c35ca23b24a8fb371ca061495729c}
actual=$(git -C "$strided_rs" rev-parse HEAD)
if [[ "$actual" != "$expected" ]]; then
    echo "strided-rs is $actual; expected $expected" >&2
    exit 1
fi

cpus=${CPUS:-0-3}
out=${OUTPUT_DIR:-"$root/data/results/erased-replay-$actual"}
target=${CARGO_TARGET_DIR:-"$root/target/erased-replay-$actual"}
mkdir -p "$out" "$target"

groups=(
    erased_gather_generic_rank_layout
    erased_dynamic_slice_generic_rank_layout
    erased_dynamic_update_generic_rank_layout
    erased_axis_reduce_generic_rank_layout
    erased_pad_generic_rank_layout
    erased_integer_zip_preflight
    erased_scatter_generic_rank_layout
)

for group in "${groups[@]}"; do
    echo "==> $group (CPUs $cpus, strided-rs $actual)"
    STRIDED_KERNEL_ERASED_POLICY_BENCH_PROFILE=threshold \
    STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS=4 \
    RAYON_NUM_THREADS=4 CARGO_TARGET_DIR="$target" \
        taskset -c "$cpus" cargo bench \
        --manifest-path "$strided_rs/Cargo.toml" \
        -p strided-kernel --bench erased_policy_thresholds \
        --features parallel -- "$group" 2>&1 | tee "$out/$group.log"
done
