# Phase 2: optimize the packed driver and remove faer

Status: **accepted** on 2026-10-03, after the Fable 5.1 review. Baseline: `origin/main` after
Phase 1 (#37, closed). Prerequisite reading:
- the [Phase 1 spec](2026-10-02-source-integration-design.md), in particular D5 and §10;
- the source study [`docs/worklogs/2026-09-30-gemm-strategy-faer-vs-tensorcontract.md`](../../worklogs/2026-09-30-gemm-strategy-faer-vs-tensorcontract.md);
- `PERFORMANCE_TIPS.md`.

## 1. Goal and end state

Contractions with a K role run **one route**, the packed driver. Cases differ
only in the selected kernel family, which is chosen at plan time. All-batch
problems stay on strided-rs (D5, amended 2026-10-02).

At the end of Phase 2:
- `strategy/faer.rs` and every `faer` dependency are deleted;
- the default plan for a copy-free contraction is the packed driver;
- the acceptance gate (§5) has passed at 1T, 4T and 8T.

Phase 2 is performance work. Unlike Phase 1 (D10), its decisions need
measured evidence. The protocol stays proportionate: one paired baseline and
profile up front, a focused check for each change, and one paired gate
before faer is deleted (§5).

## 2. What faer wins, and why

**Recorded measurements.** These come from the EPYC 7713P (Zen 3, AVX2, no
AVX-512):
- **Copy-free contractions.** The faer route is faster. The packed/faer
  workload ratio is 0.94 at 1T and 0.40 at 8T (decision log, P2
  2026-09-30). 8T is the largest gap.
- **Historical figures.** The batched-GEMM figures (the faer loop being
  1.2–1.5× faster, and 10–18× faster for n ≤ 4) and the #27 blas-corpus
  8T numbers come from the blas harness, which was deleted in #37. They
  are **historical** and are replaced by S0's paired run.

On AVX2 x86, faer's large-GEMM work is private-gemm-x86's hand-scheduled
assembly. faer uses nano-gemm for MNK ≤ 4096 and special paths for matvec
and rank-1 problems.

**Suspected causes.** These come from the source study and are inferred,
not measured. S0 attributes them.

| # | Cause in the packed driver | Expected effect |
|---|---|---|
| 1 | Scratch tile, scalar write-back with no `target_feature`, a separate scalar A/B pack, and a write-back pass every kc = 256 | the 1.1–1.2× gap at 1T |
| 2 | Two futex `std::sync::Barrier` waits per (jc, pc), a B pack that does not overlap compute, and a static 1-D M split | the widening gap at 8T |
| 3 | Per-call pack allocation | mostly fixed since #27 (owner-scoped arenas); S0 verifies |
| 4 | Legacy blocking: A_c = 512 KiB, all of Zen 3's L2 (`abi/types.rs:376-392`) | 1T for m ≥ 256 |
| 5 | Provisional AVX2 tiles | the base gap |
| 6 | Packed-B traffic, where faer leaves B unpacked | second order |

The indirect call per micro-tile is inherent to the fn-pointer family ABI.
It is accepted and not a work item.

**Facts the work items depend on** (checked in source):
- **Direct kernels.** No optimized family implements `UkrFn::Direct`. The
  only Direct family is the portable reference
  (`kernels/reference/portable.rs:228`), and Direct is real-only
  (`abi/family.rs:446`).
- **Direct-B.** It requires a Direct family (`family.rs:449`) and forces
  `pm = 1, pn = p` (`driver/mod.rs:546-549`).
- **`DirectUkrFn` input format.** It takes A as a packed panel; only B
  carries strides (`family.rs:188-203`).
- **Remaining flags in hot loops.** The pack/write-back format dispatch is
  already const-generic. What remains is the runtime `conj` flag in packing
  (`pack/pack.rs:654-693`) and the `plain`/`beta_is_zero`/`conj_c`/`conj_d`
  flags in `writeback_rows` (`pack/writeback.rs:324-348`).

## 3. Work items

Each bullet below is its own PR, with its own focused check (§5). The PR is
kept only if the check shows a gain or removes a cost. A neutral or negative
result is recorded and reverted, not merged.

**S0. Baseline and attribution.** No code change.
- **Paired run.** Run `contract --corpus` with the `plan` (current
  default) and `packed` rows on the gate corpora (§5) at 1T, 4T and 8T,
  with an A/A noise run. This refreshes the §2 baseline, gives the
  per-case targets, and fixes the noise floor.
- **Profile.** Run `perf record` (`kernel.perf_event_paranoid` relaxed to
  1) on one 1T and one 8T case per dtype: f64 and c64, 1024³, plus the
  top-weighted faer-selected corpus cases. Attribute the cycles to
  kernel, pack, write-back, barrier and page faults.
- **Ordering.** The attribution fixes the order of S1–S3.

**S1. The 1T base gap** (causes 1, 4, 5):
- **AVX2 Direct kernels for f64/f32** (for example 12×4 or 8×6). They do
  alpha/beta in the epilogue and handle partial m/n in-kernel, since
  `direct_tile` passes edge tiles with `m = mrem` (`driver/tile.rs:223-226`).
  Then make direct-C the default for those families. c64 stays on
  scratch + write-back by contract, so its 1T gap is addressed by the next
  two bullets and the #30 native families.
- **Pack/write-back codegen.** Move the runtime `conj` flag in packing and
  the write-back flags into const generics, so they are dispatched once,
  outside the loop. Add per-ISA `#[target_feature]` variants of pack and
  write-back, selected together with the family. Use SIMD for
  regular-block packing.
- **Blocking**, measured as **one arm**: kc = 512 together with an A
  budget of about 256 KiB (half of L2), or the analytical `BlockModel`.
  Invariant: kc and blocking never depend on width, because the
  across-width bitwise promise rests on that.
- **AVX2 tile retuning** toward 12 accumulators. Promote the #30 native c64
  families into `Auto` where they win.

**S2. Parallel scaling** (causes 2, 6):
- **Barrier.** Replace `std::sync::Barrier` in the team lease with a
  bounded spin-then-park barrier; the spin must be bounded because a
  participant can be delayed by another host job. `TeamSet::barriers` is a
  public `tprims-exec` field, so this is a stated `tprims-exec` API change.
- **Direct-B as a default** where it wins. It is ordered after the S1
  Direct kernels. Its check must include a tall-M/narrow-N case, and it
  must account for the A duplication that `pm = 1` causes.
- **DynamicTiles (#29) as a default** for the shapes where faer wins at 8T.
- **The decided medium-width rule** (decision log, "SPMD width on a
  borrowed Rayon pool"):
  - `width == pool size` → barrier `broadcast`;
  - `1 < width < pool size` → a `pm = 1`, `pn`-split grid run through
    `for_each_partition`, with no barriers.

  The check adds one pool-8/budget-4 case, because the harness otherwise
  builds pool size == thread count and never exercises this.

**S3. Small problems and batched small GEMMs:**
- **A small-problem family**, selected at plan time (D5; never a
  per-call switch). B is read in place when its stride allows, and A is
  packed into the worker's leased scratch (MR × k). Fixed cost close to
  nano-gemm's is the target. `DirectUkrFn` is unchanged.
- **Batch-axis job claiming** for the in-plan batch loop (`h` in
  `run_strip`/`run_dynamic`, `plan.stats.batch`), so a batch of tiny GEMMs
  fills the team. This is the cross-batch claiming #29 deferred.
  `contract_batched` is already partitioned and is not affected.
- **Matvec-like and rank-1 shapes** become families or driver cases, not
  strategies.

**S4. Switch and delete.** This is a separate PR, revertible on its own,
merged only after the §5 gate passes. It deletes:
- `Algorithm::Faer` (`plan/report.rs:16`, public);
- `strategy/faer.rs` and its `strategy/mod.rs` entry;
- `tests/faer_schedule.rs`;
- the assertions of faer selection in `tests/kernel_select.rs:72`,
  `packed_explicit_config.rs:39`, `contract.rs:240`, `c_modes.rs:300` and
  `dynamic_partition.rs:19`;
- the faer oracle in `tprims-exec/tests/entry.rs:82`, replaced by a
  closure;
- the faer mention in `tprims-capi/src/handle.rs:145`;
- the workspace `faer`/`faer-traits` dependencies (`Cargo.toml:35-36`) and
  the `tprims-exec` dev-dependency.

It also folds the bench `plan`/`packed` rows, which coincide after S4, and
updates the README, the architecture doc, the migration guide and
decision-log line 22. That line still says Phase 2 deletes the elementwise
path, which is stale after D5.

## 4. Constraints

- **Hot-loop rules.** tensor4all-agent-rules#16 applies: no runtime mode
  flags and no per-element dynamic calls in tensor-sized loops. Reusable
  SIMD goes into strided-rs where practical; GEMM microkernels and packing
  stay in `tprims-kernel`.
- **Correctness.** Every change keeps the kernel-contract, oracle,
  direct-path, DynamicTiles and TAPP tests green on Linux x86_64 and macOS
  arm64. Bitwise determinism across widths holds for the same family and
  fixed blocking, as it does today.
- **API.** No new public API unless an item needs it. The S2 barrier type
  is the one expected change. Changing a default is not an API change.
- **Hardware.** Optimization targets the Zen 3 (AVX2) host. AVX-512 and
  NEON are kept correct (NEON through macOS CI) but are not tuned. Claims
  are made only for hardware that was measured.

## 5. Measurement protocol and acceptance gate

**Recording and setup.** Record commits, CPU, pinned cores, threads, dtype,
shape and timed boundary. Use the `tprims-benchmark` skill on pinned idle
cores of one CCD. Its blas step is stale after #37; use the `contract`
harness. Metric and noise come from `benchmarks/scripts/tblis_decision.py`,
as in P2.

**Per item.** Run a before/after check on the cases the item targets. It is
1T for S1 and adds 8T for S2. The medium-width item also runs the
pool-8/budget-4 case. There is no full-corpus campaign per PR.

**Acceptance gate.** Run it once before S4.
- **Comparison.** A paired comparison of the harness's `packed` row
  (`PlanConfig::packed()`) against the `plan` row (the current default,
  faer where it applies). No internal switch is needed: the gate corpora
  contain no all-batch entries.
- **Corpora:**
  - `tenferro-p1`;
  - `tenferro-p1-gemm`, which contains 12 entries with min(M, N) ≤ 4;
  - `large-batched-gemm`;
  - plus one added rank-1 (K = 1) `dot_general` entry and a few n ≤ 4 /
    large-H entries.
- **Widths.** 1T, 4T and **8T**. All three gate, so 8T, where faer leads
  most, must not regress.
- **Runs.** `BENCH_RUNS=5`, with an A/A noise run.
- **Pass, per corpus and width:**
  1. **Faer-selected subset.** The calls-weighted workload ratio over the
     cases the default route sends to faer (the harness prints
     `# selected … plan: faer`) is ≤ max(1.05, 1 + noise). The
     whole-corpus ratio is reported beside it.
  2. **Per-case cap.** No case whose default time is above the 50 µs
     serial threshold exceeds 1.5×. Cases at or below it are covered by
     the weighted ratio, because their time share is negligible and A/A
     noise there is around 20%.
- **hadamard.** It runs on strided-rs and is unaffected. It is checked
  once for no regression.

## 6. Out of scope

- AVX-512 and NEON performance tuning.
- New public APIs beyond the S2 barrier.
- C ABI changes.
- The tenferro migration.
- GPU.

## 7. Maintainer decisions (2026-10-03)

1. **Profiling.** `kernel.perf_event_paranoid` is relaxed (to 1) on the
   EPYC host for `perf record`.
2. **Gate.** As in §5: the faer-selected subset ratio is at most
   max(1.05, 1 + noise), and cases above 50 µs are capped at 1.5×. The
   gate runs at **1T, 4T and 8T**, so 8T must not regress.
3. **Pace.** Each S1–S3 item merges when its own check passes. S4 merges
   only after the gate passes.
