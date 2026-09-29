# Phase 1c: tprims-contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Binary tensor contraction with batch indices (`dot_general` semantics) behind one plan API with two strategies — permute plus batched GEMM (the tenferro-rs approach, reimplemented on `tprims-blas`) and TBLIS-style direct (Lukas Devos's `tensorcontract`) — on explicit `Exec`, with a reference-checked corpus, no hidden materialization (`no_materialize`), and 1T/4T benchmarks that time planning and execution separately.

**Architecture:** `crates/tprims-contract`. `DotGeneral` names lhs/rhs contracting and batch axes; free axes keep their order; the output is `[lhs_free..., rhs_free..., batch...]` (tenferro convention, needed by 1e). `ContractPlan::new` validates shapes and strides against the config and chooses a strategy:
- **PermuteGemm:** the four index groups M (A, C), N (B, C), K (A, B) and H (A, B, C) are each ordered by one operand's strides and fused per operand (`s[i+1] == s[i] * e[i]` after dropping extent-1 axes). If every group fuses, the contraction is one strided batched GEMM on views of the caller's memory (no copy). Otherwise each non-fusable operand is copied once into a compact column-major buffer in canonical `[M, K, H]` / `[K, N, H]` / `[M, N, H]` order (strided-basic copy through `run_with_exec`), reported as `materialized`, or refused with `Error::WouldMaterialize` under `no_materialize`. A materialized C is filled from C when `beta != 0` and copied back.
- **Tblis:** labels `0..ra` for A; B's contracting and batch axes reuse the matching A labels, B's free axes get new labels; C's labels follow the output convention. One `tensorcontract::Plan` is built in `new` and executed through the `Spmd` seam (in place, `C = D`).
`Auto` selects PermuteGemm (faster in every 1b measurement). Empty problems and `alpha == 0` never touch A/B pointers (the 1b review class).

**Tech Stack:** tprims-blas (`gemm_batched`), tprims-exec, strided-view, strided-basic (`copy_into`, `run_with_exec`), tensorcontract.

**Spec:** `docs/superpowers/specs/2026-09-29-phase1-cpu-backend-design.md` (section 1c). Provenance: the PermuteGemm strategy follows tenferro-rs `crates/tenferro-cpu/src/{dot_runtime.rs,gemm/mod.rs}` at `5a4e7fd` (same maintainers, MIT OR Apache-2.0) as a reimplementation of its approach, not a code copy; say so in the module header.

## Global Constraints

- Branch `phase1c-contract` from `main` (1b merged), worktree `../tprims-rs-1c`.
- `-j 16`; clippy `cargo +1.98.0`; crate metadata as for other tprims crates.
- Credit Lukas Devos / tensorcontract and Matthews (arXiv:1607.00291) in the Tblis module and README (REPOSITORY_RULES "Imported Code").
- No pointer arithmetic on empty operands; alpha == 0 or an empty K scales C only.

## Review Focus

- Output with zero or overlapping strides → `AliasedOutput` before any write (both strategies).
- Contracting axes listed in different orders for A and B (`"abc,cba"`-style), batch axes permuted between operands.
- Rank-0 output (full contraction) and outer products (no K).
- Negative strides on every operand; conjugation on A and on B.
- `no_materialize` with a non-fusable operand is an error, with a fusable one succeeds without copies (pointer identity via `Selected::PermuteGemm { materialized: [false, false, false] }`).

---

### Task 1: Crate, config, validation, reference
`crates/tprims-contract/{Cargo.toml,src/lib.rs,src/error.rs,src/config.rs}`; `tests/common/mod.rs` with a naive reference (odometer over output indices and contracted indices, explicit conj) and helpers to build strided views (permuted, reversed). `DotGeneral::new(lhs_contract, rhs_contract, lhs_batch, rhs_batch)`; `validate(&self, a_dims, b_dims) -> Result<Shape>` (distinct axes, in range, matching extents) where `Shape { m_axes, n_axes, k_pairs, h_pairs, out_dims }`. Tests: typed errors for repeated axes, out of range, extent mismatch; `out_dims` order.

### Task 2: PermuteGemm strategy
`src/permute_gemm.rs`: group fusion, batched-matrix views, materialization of non-fusable operands (and C round trip), `Selected::PermuteGemm { materialized: [bool; 3] }`; `src/plan.rs`: `ContractPlan<T>` (`new`, `selected`, `execute`), `Strategy::{Auto, PermuteGemm, Tblis}`, `Flags { no_materialize }`. Tests: corpus against the reference (matmul, batched, contraction over two axes in different orders, outer product, rank-0 result, batch-only Hadamard, permuted operands, negative strides, conj A/B, beta = 0 with NaN C, k = 0, empty free extent, f64 and c64), copy-free detection, `no_materialize` error, aliased output error, 4-worker pool equals serial.

### Task 3: Tblis strategy
`src/tblis.rs`: label mapping, `tensorcontract::Plan` built at plan time, execution through an `Spmd` adapter on `Exec`, in-place C. The Task 2 corpus runs for both strategies.

### Task 4: Wrappers
`permute(exec, a, perm, c)` (strided copy with axis permutation into `c`) and `add(exec, alpha, a, beta, c)` (`C = alpha A + beta C`, strided), with tests. `trace` is deferred (ruling: no consumer yet in 1e's first slice).

### Task 5: Benchmarks and docs
`benchmarks/benchmarks/tprims/contract/contract.rs` (bin `contract`) with the predeclared corpus in the source and README: tiny (2x2x2 matmul, `ij,jk->ik`), matmul 256³, batched `bij,bjk->bik` (64³, batch 32), permuted `abc,cbd->ad` (a,b,c,d = 64,32,32,64 with A and B stored transposed), 4-index network `ijkl,klmn->ijmn` (16⁴), large `ijk,jkl->il` (256, 64, 64, 256), f64 and c64, both strategies, plan and execute timed separately, 1T/4T paired per case through a `run.sh` like linalg's. Record README; decision-log row; gate; fresh review; PR; merge after green CI.
