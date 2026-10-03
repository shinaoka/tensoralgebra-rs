# Phase 2: route GEMM-fusable work to faer, optimize the packed driver for the rest

Status: **accepted**, revised 2026-10-03. Issue: [#50](https://github.com/tensor4all/tprims-rs/issues/50).
Baseline: `origin/main` after Phase 1 (#37).

Revision history:
- 2026-10-03: the first version planned to delete faer. After the S0
  baseline, the maintainer decided to **keep faer for every GEMM-fusable
  contraction, large and small**. This version supersedes the deletion plan
  (former S4) and the "one route" goal.

## 1. Decision and end state

S0 (`benchmarks/benchmarks/tprims/contract/results/2026-10-03-phase2-s0/`)
measured the packed driver against the default route on every case the
default sends to faer, at 1T, 4T and 8T. faer is faster in every size class
of copy-free GEMM:
- tiny MNK ≤ 4096 problems, through nano-gemm;
- n ≤ 4, rank-1 and matvec shapes, through dedicated paths;
- batches of small items, which it distributes barrier-free;
- small-K/small-M shapes, where it reads B in place;
- large square GEMMs, by 10–15%.

The packed driver's micro-kernel matches faer's. The losses are structural
fixed costs: packing, scalar write-back and per-item barriers. Matching faer
everywhere would mean re-implementing its size-specific dispatch.

**Routing (amends Phase 1 D5):**

| Problem | Route |
|---|---|
| Fuses to a strided (batched) GEMM over the caller's memory without a copy, any size | faer (`strategy/faer.rs`) |
| Would need a copy to become a GEMM (permuted/non-fusable layouts) | packed driver (TBLIS-style, kernel families) |
| All-batch (no M, N or K) | strided-rs (`strategy/elementwise.rs`) |

faer stays a permanent dependency. An explicit kernel, selector or partition
still forces the packed driver, as today.

**Phase 2 delivers:**
- faer's coverage widened to every GEMM-fusable problem it can serve;
- the packed driver made faster on the problems it owns;
- removal of the packed driver's pathological scaling.

## 2. Work items

Each bullet is its own PR. It is kept only if its focused check (§3) shows a
gain on its target cases and no regression beyond noise elsewhere. A
neutral or negative result is recorded and reverted.

**W1. Batch-axis job claiming in the packed driver.**
- **Defect.** S0 measured it: per-item team barriers make batched tiny
  items 10–77× slower at 4T/8T than at 1T.
- **Change.** Run the in-plan batch loop barrier-free across items when
  the items are small or H ≥ width. Each item stays serial and bitwise
  identical. Large items keep team SPMD.
- **Why it still matters.** It applies to packed-route batched problems
  and to forced-packed configurations.

**W2. Widen faer coverage.**
- Today faer declines some GEMM-fusable problems, for example a separately
  described C (`strategy/faer.rs:17`).
- For each declined class, measure faer against packed:
  - separate C (initialize D from C, then accumulate);
  - other `plan()` refusals.
- Route a class to faer only where it wins.
- Keep the C/D/op semantics, and check them with the oracle tests.
- **Done in W2 (separate C).** faer computes `D := op_D(beta * op_C(C))` in
  one parallel strided pass (strided-basic, type-level conj; the in-place
  update uses the same pass instead of faer's former serial scale) and then
  accumulates `op_D(alpha * op_A(A) * op_B(B))` with `Accum::Add`; `beta == 0`
  overwrites D and never reads C. The pass is output-sized, not an operand
  normalization: A, B and C are not copied, packed or reordered, so
  `no_materialize` semantics and `PlanReport::materialized` are unchanged.
  A separate C goes to faer when K >= 512 or the output has at most 2^20
  elements (measured; results under `results/2026-10-03-phase2-w2/`); a large
  output with small K stays packed because the pass costs 20-50% of faer's
  time there. `beta` is an execution argument, so the rule cannot depend on it.
  An isolated one-input K axis stays on packed.

**W3. Packed-driver codegen for its own domain.** These are causes 1 and
4 of the source study.
- **Pack and write-back.** Move the runtime `conj` flag in packing
  (`pack/pack.rs:654-693`) and the `plain`/`beta_is_zero`/`conj_c`/`conj_d`
  flags in `writeback_rows` (`pack/writeback.rs:324-348`) into const
  generics (tensor4all-agent-rules#16). Add per-ISA `#[target_feature]`
  variants selected with the family. Use SIMD for regular-block packing.
- **Blocking**, as one arm: kc = 512 with an A budget of about 256 KiB, or
  the analytical model. kc and blocking must not depend on width.

**W4. Optional.** These are kept only if W1–W3 leave a measured gap on
packed-route cases:
- a bounded spin-then-park team barrier (a stated `tprims-exec` API change,
  since `TeamSet::barriers` is public);
- DynamicTiles or direct-B as defaults where they win;
- the decided medium-width rule.

Out of Phase 2: AVX2 Direct kernels and the small-problem family. Their
targets, large and small GEMM-fusable shapes, now go to faer. They come
back only with evidence on packed-route shapes.

## 3. Measurement

**Setup and recording.**
- Use the `tprims-benchmark` skill on pinned idle cores of one CCD, with
  the `contract` harness.
- Metric and noise come from `benchmarks/scripts/tblis_decision.py`.
- Record commits, CPU, cores, threads, dtype, shape and timed boundary.

**Per item.** Compare before and after at 1T, 4T and 8T, on:
- the item's target cases;
- the packed-route cases of `tenferro-p1` (the harness prints
  `# selected … plan: packed`);
- a no-regression control set.

8T must not regress (maintainer decision).

**W2 decision.** Measure faer against packed on the newly routed class
before the routing changes.

**No deletion gate.** faer stays.

## 4. Constraints

- **Hot-loop rules.** tensor4all-agent-rules#16 applies.
- **Tests.** All existing kernel-contract, oracle, direct-path,
  DynamicTiles and TAPP tests stay green on Linux x86_64 and macOS arm64.
- **Determinism.** It holds across widths for the same family and fixed
  blocking.
- **API.** No new public API, except the W4 barrier if it is ever done.

## 5. Maintainer decisions (2026-10-03)

1. `kernel.perf_event_paranoid` is relaxed (to 1) for profiling.
2. GEMM-fusable contractions, large and small, go to faer. faer is not
   deleted.
3. Items merge when their own check passes. 8T must not regress.
