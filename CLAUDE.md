# CLAUDE.md

Persistent project context for this repository. Re-read this at the start of every session and treat it as the source of truth for how to operate here.

## What this project is

A native-Rust, transpose-free tensor contraction engine in the spirit of TBLIS, whose distinguishing feature is complex-number handling. Core algorithm: block-scatter-matrix tensor contraction (BSMTC) — treat a contraction as a GEMM over a scatter / block-scatter memory layout, using BLIS's structure (five loops around a micro-kernel, two-level cache packing), so no explicit transposition or temp workspace is needed.

## Operating mode

You run **autonomously** across all phases. Do not wait for human approval at routine phase boundaries. At each boundary, write a concise report to `DECISIONS.md` and continue. Escalate to the human **only** on a kill/pivot condition (see below) or a genuine blocker you cannot resolve after a documented, honest attempt. Prefer stating an assumption and proceeding over asking a question. Record every material assumption and decision in `DECISIONS.md` so the run is auditable after the fact.

## Working thesis — verify, do not assume

Treat every claim here as a hypothesis to confirm, quantify, or refute in Phase 1.

- TBLIS reuses BLIS's micro-kernels/blocking params and inherits BLIS's complex path, the **1m** induced method, which encodes complex multiply into an interleaved packing format so a real micro-kernel produces the complex product.
- Hypothesis: TBLIS underperforms on complex contractions, worst in memory-bound / awkward-stride cases, because interleaved-complex storage and the scatter/block-scatter packing compound and force more work onto the slow full-scatter (gather) path. This is currently folklore; there is no known systematic public complex tensor-contraction benchmark across libraries. Establishing one is a deliverable; validating/refuting the claim is a Phase 1 decision point.
- Proposed distinction: pack to **planar / split-complex (SoA)** layout, fused into the scatter pass; run a real micro-kernel over the real and imaginary planes; reassemble interleaved complex only at write-back. Why this helps for tensors specifically (unlike matrix GEMM, where Van Zee's 1m avoids a separate planarization pass): tensor packing is already a full scatter/gather pass over every element, so planarization rides on it at near-zero marginal cost, and it improves both the micro-kernel (real SIMD lanes, no shuffles) and the block-scatter fast path (clean real strides). Should also compose cleanly with mixed real×complex operands and arbitrary real scalar types (extended precision, `Complex<bf16>`, dual numbers for forward-mode AD).
- Cost model to test: the only planar-specific overhead is the write-back interleave, ~`c / (4 · min(K, KC))` of compute per output tile, `c` = interleave-ops/element (~0.5 vectorized `unpacklo`/`unpackhi` / `zip1`/`zip2`, ~2 scalar). Negligible for `K ≥ KC`; rises ~`1/(8K)` for small `K`; material only at `K` in the low single digits → handle by dispatch (planar vs 1m vs TTGT). Complex needs 2× accumulator register state regardless of layout, shrinking the complex micro-tile — inherent, not planar-specific.

## Constraints

- Core is native Rust with **no FFI in the hot path**. FFI (`tblis-rs`, `rstsr-blis-ffi`) is allowed only as benchmark baselines.
- Prefer reuse of maintained crates over reinvention where they fit; justify every build-vs-reuse call in `DECISIONS.md`.
- **Primary integration and benchmarking surface: TAPP** (Tensor Algebra Processing Primitives). Make the engine callable through TAPP (C-ABI export + idiomatic internal Rust bindings). TBLIS supports TAPP and the TAPP repo carries cuTENSOR bindings, so building against TAPP makes this engine, TBLIS, and cuTENSOR swappable for apples-to-apples benchmarking. Verify the current TAPP spec first: operations, dtypes (incl. complex), stability, gaps.
- RSTSR / REST integration is an **optional secondary** deliverable (thin downstream adapter), not a design driver.
- Benchmark baselines: TBLIS (via `tblis-rs` and/or its TAPP support), TTGT (permute + ZGEMM), a native TTGT path (RSTSR / `ndarray`-einsum), cuTENSOR via TAPP where a GPU reference helps.
- Benchmark corpus: the TCCG set of 48 contractions in `f32`/`f64`/`c32`/`c64`. Structures are precision-agnostic, so the **per-contraction real-vs-complex ratio is the headline metric**.

## Ecosystem notes

Strong, maintained kernel/SIMD layer: `faer`, the `gemm` crate, `matrixmultiply`, `microgemm`; SIMD via `pulp` / `std::simd` / `macerator`. Path optimization exists: `opt-einsum-path`. N-d arrays: `ndarray`, `RSTSR`. The gap this project fills is the native transpose-free engine; today the only option is FFI to C++ TBLIS (`tblis-rs`). Verify current crate/spec status via web search rather than training data, and cite sources.

## Starting references (verify currency)

Matthews, "High-Performance Tensor Contraction without Transposition" (arXiv:1607.00291); Van Zee "1m method" and Van Zee & Smith "3m/4m methods"; Springer & Bientinesi GETT (arXiv:1607.00145) + the `tccg` repo; "Strassen's Algorithm for Tensor Contraction" (arXiv:1704.03092, flags packing/scatter-copy as a bottleneck); TAPP draft (arXiv:2601.07827) + its repo; BLIS papers.

## Phases (self-gated, autonomous)

Each phase ends with a self-check against its gate, a report to `DECISIONS.md`, and automatic progression unless a kill/pivot condition fires.

**Phase 1 — Exploration & design (no core implementation).** Design doc covering: literature review + summary (BSMTC; BLIS five-loop+packing; 1m/3m/4m; GETT/cuTENSOR; TAPP surface & maturity; complex/scatter-copy evidence — cited); Rust ecosystem survey with per-layer build-vs-reuse decisions; design discussion (tensor + scatter/block-scatter data model; packing kernels; planar-complex representation & write-back; micro-kernel interface; loop/blocking structure & params; threading; dispatch planar/1m/TTGT by shape and `K`; element-type genericity & AD; TAPP export/ABI + optional RSTSR adapter); **empirical premise check** (measure representative complex contractions via `tblis-rs` + TTGT to confirm/refute the weakness before committing); repo setup (workspace, CI, lint/format, feature flags, MSRV, license, contrib docs); benchmark & testing framework (reference oracle; property tests; TAPP-based swappable harnesses vs TTGT/`ndarray`/TBLIS; real/complex TCCG harness; `criterion`; real-vs-complex ratio metric; roofline annotation); implementation plan for Phases 2–5 with milestones/risks/kill criteria; **self-scrutiny** (critique the plan and thesis; list assumptions, unknowns, alternatives — 3M vs 1m vs planar, TTGT-with-good-permute, wrapping TBLIS, contributing upstream; state the single most likely failure mode and how Phase 1 detects it early).
Gate: self-approved design doc; premise resolved with data; green scaffolded repo; working harness with baselines wired in.

**Phase 2 — Correct, framework-complete initial implementation (scatter/gather focus).** Correctness over speed; may be slow but architecturally right, especially in scatter/gather. Tensor data model; scatter/block-scatter vector computation; packing kernels that gather tensor → packed panels via scatter/block-scatter, **including the planar-complex split during packing**; handle arbitrary index permutations, repeated indices / partial traces, remainder/edge blocks (full-scatter fallback); reference scalar micro-kernel + five-loop driver; correct end-to-end real and complex; expose through TAPP even if slow.
Gate: numerically correct across the full matrix (shapes, permutations, dtypes, traces, degenerate cases) vs oracle, TTGT, TBLIS. Perf measured as baseline, not a goal.

**Phase 3 — Micro-kernels.** Vectorized real micro-kernels (register-blocked MR×NR) via the chosen SIMD path for `f32`/`f64`; complex via planar real micro-kernels over the two planes with vectorized write-back interleave; optional 3M variant (opt-in); blocking params seeded from known-good BLIS values, arch-detected; verify both the regular-block fast path and gather fallback feed the kernel correctly.
Gate: correctness unchanged; single-core throughput vs baselines and a GEMM roofline; within a defined band of TBLIS on real cases and demonstrably better on targeted complex / awkward-stride cases, or a documented evidence-based reason why not.

**Phase 4 — Profiling & improvement.** Profile (cache misses, vectorization, bandwidth); locate bottlenecks (packing bandwidth, scatter edge cases, write-back, kernel efficiency); implement threading (parallelize appropriate loops, BLIS-style, NUMA-aware), planar/1m/TTGT dispatch, small-`K` handling, prefetch, block-scatter regularity exploitation; re-run full suite; produce the real-vs-complex ratio comparison vs TBLIS and TTGT; write up findings. The central thesis is answered here.
Gate: performance targets met, or a clear evidence-based account of the gap. Complex claim resolved either way.

**Phase 5 — Packaging.** API polish, docs, examples, error handling, feature flags, multi-arch CI, crate publication, TAPP conformance verification (+ optional RSTSR adapter), reproducible benchmark artifact, paper-shaped writeup of the complex results.
Gate: released, documented, reproducible, TAPP-integrated.

## Kill / pivot conditions (only reasons to escalate)

Escalate with data and a recommendation if: the Phase 1 premise check refutes the complex-weakness hypothesis (recommend pivot: improve TTGT / wrap TBLIS / contribute upstream / stop); TAPP cannot express the required ops/dtypes (esp. complex) or is too unstable (recommend an alternative surface); a gate is unmet and root-cause shows the shortfall is structural, not incremental; a counted-on build-vs-reuse dependency is unmaintained/broken with no viable substitute. Absent these, run autonomously and do not pause for approval.

## Standing rules

- Keep the test suite green at every stage; never trade correctness for speed.
- Maintain `DECISIONS.md` (decisions + assumptions + risks); note immediately if the thesis weakens.
- At each phase boundary, write a report to `DECISIONS.md` and continue — no approval wait.
- Verify current library/spec status via search, not training data; cite sources.
- Negative results are valid and publishable: a well-characterized "planar does not beat 1m in regime X" is an acceptable outcome. Report honestly.
- Commit at meaningful milestones with clear messages; keep the working tree clean between tasks.
