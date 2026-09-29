# Strided Benchmarks

This directory contains kernel-level benchmarks for `strided-rs` components.
Each benchmark page keeps the measurement code and the published results
together.

## Pages

| Page | Compares | HPTT coverage |
|---|---|---|
| [Permutation kernels](permute/README.md) | JSON-defined permutation patterns comparing naive, `strided_perm`, HPTT, Julia Base, and Strided.jl where applicable | Included in the unified Rust runner for directly matching contiguous-source transpose cases |
| [Transpose-scale kernels](transpose_scale/README.md) | raw pointer naive loops versus `copy_transpose_scale_into`, `map_into`, and `strided_perm::copy_into` where applicable | Not included yet; add only if the HPTT runner covers the same scale/update semantics |
| [Fused elementwise kernels](fused_elementwise/README.md) | per-op reused buffers versus `fused_elementwise_into` static runtime-DAG specializations and interpreter fallback | Not applicable |
| [Erased replay rank/layout scaling](erased_replay/README.md) | before/after generic gather, dynamic slice/update, axis reduction, pad, and integer preflight across ranks/layouts/threads | Not applicable |
| [Kernel scaling](kernel_scaling/README.md) | erased versus typed versus raw baselines for elementwise, reduction, and structural copy plans at 1T and 4T, plus Julia Base and Strided.jl, with a threshold gate (strided-rs#269) | Not applicable |

## Result Policy

- Keep setup/allocation outside timed regions unless the page says otherwise.
- Record the `strided-rs` git hash beside measured results.
- Do not publish HPTT rows unless the case is directly comparable and checked
  for correctness.
- Run thread-count variants sequentially.
