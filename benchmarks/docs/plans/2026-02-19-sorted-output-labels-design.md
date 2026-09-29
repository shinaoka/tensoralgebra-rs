# Sorted Output Labels for Intermediate Tensors

## Problem

`tensornetwork_permutation_light_415` takes ~455ms in strided-rs vs ~388ms in OMEinsum.jl (1T, AMD EPYC 7713P).

Phase profiling reveals one step (Step 408: m=4, k=256, n=8192, batch=[2,2,2]) accounts for 56% of total runtime. Of that, 96% is `prepare_input_owned` for B — a 128MB scattered-stride copy (`copy_into_col_major`) that takes 238ms due to cold L3 cache.

Root cause: strided-rs uses lazy permutation (metadata-only dim/stride reorder) between contraction steps. Scattered strides accumulate through the tree. When a downstream step needs contiguous data, it faces an expensive cold-cache copy of high-rank scattered data.

OMEinsum.jl avoids this by storing each intermediate with sorted labels in contiguous layout. Subsequent steps receive contiguous inputs and need minimal permutation.

## Solution

Change `compute_binary_output_ids` in `strided-opteinsum/src/expr.rs` to return sorted output labels instead of `[lo, ro, batch]` order.

### How it works

1. `compute_binary_output_ids` returns sorted labels (add `out.sort()`)
2. `eval_pair_alloc` passes sorted labels as `ic` to `einsum2_into`
3. einsum2's `c_to_internal_perm` maps sorted → `[lo, ro, batch]` internally
4. `prepare_output_view` detects non-contiguous output strides → allocates temp
5. GEMM writes to temp (contiguous, fast)
6. `finalize_into` copies temp → output buffer (warm cache, fast)
7. Result: col-major buffer with sorted labels returned to parent node
8. Next step receives contiguous input → `prepare_input_owned` takes zero-copy path

### Why this works

- `finalize_into` runs immediately after GEMM — data is warm in L3 cache
- Warm-cache copy of 128MB: ~16ms (vs 238ms cold-cache scattered copy)
- Downstream steps see contiguous inputs → skip `prepare_input_owned` copy entirely
- Net effect: trade one 238ms cold copy for many small warm copies

## Changes

### File: `strided-opteinsum/src/expr.rs`

**Function: `compute_binary_output_ids` (line 238-260)**

Add `out.sort()` before return. No other changes needed.

### No changes to:

- strided-einsum2 (einsum2's existing `c_to_internal_perm` + `prepare_output_view` + `finalize_into` handles everything)
- Buffer pools (both opteinsum BufferPool and einsum2 contiguous.rs pool remain as-is)
- Zero initialization (already eliminated in production path — `pool_acquire` and `alloc_col_major_uninit` use uninit)

## Expected Performance

| Phase | Before | After |
|---|---|---|
| Step 408 prep_b | 238 ms (cold scattered copy) | ~0 ms (zero-copy) or few ms |
| Step 408 finalize | ~0 ms | few ms (warm copy) |
| Other steps finalize | ~0 ms | small overhead (warm copies) |
| **Total** | **~455 ms** | **~200-250 ms** (estimated) |

## Verification

1. `cargo test -p strided-opteinsum` — all 82 tests pass
2. `cargo test -p strided-einsum2` — all 84 tests pass (no changes)
3. Benchmark: `BENCH_INSTANCE=tensornetwork_permutation_light_415 RAYON_NUM_THREADS=1 cargo run --release`
