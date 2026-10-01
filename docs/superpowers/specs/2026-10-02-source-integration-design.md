# Source integration and clean-slate crate redesign

Status: **draft for review**. Date: 2026-10-02. Baseline: `origin/main` `fbc83f5`.

This spec records the maintainer decisions taken on 2026-10-01/02. It covers
**Phase 1**: dissolve the `tensorprimitives/` subtree into a newly designed
crate set and remove the parts that leave the repository. **Phase 2**
removes faer. It is summarised in §10 and gets its own spec later.

## 1. Goals and decisions

The goal is one coherent library: each responsibility exists in exactly one
place, every existing numerical test still passes or is ported, and nothing
regresses in a smoke run.

| # | Decision |
|---|---|
| D1 | `tensorprimitives/`, imported from lkdvos/tensorprimitives-rs (Lukas Devos), is dissolved into tprims. No upstream sync or separate product continues: no crates.io `tensorcontract`, no JLL/Yggdrasil, no Julia wrapper. Authorship and git history are preserved (§9). |
| D2 | The public Rust API may break freely. tenferro gets a migration guide; the `ext/tenferro-cpu-tprims` update is separate work. |
| D3 | Removed: the public BLAS API (`gemm`, `gemm_batched`, `gemm_grouped`, `trsm`), the BLAS C ABI (`tprims-blas-capi`), the engine layer (`Engine`, `EngineChoice`, `TPRIMS_GEMM_ENGINE`), the private-gemm-x86 engine (`tprims-kernel-pgx86`), the gemm-f64/f32 adapter (`tprims-kernel-gemm`) and `tprims-linalg`. BLAS and linalg may return later as separate repositories. |
| D4 | Phase 1 keeps faer only as an internal contraction strategy, for copy-free permute+GEMM and the batched faer loop. It lives in one module. Phase 2 deletes it after the packed driver is optimized. |
| D5 | Phase 2 target: **one route**. Every contraction runs the packed driver, and the only difference between cases is which kernel family is called. Small problems, matvec-like and Hadamard products are covered by families and driver handling, not by separate paths. |
| D6 | The central contraction representation is a lowered, role-grouped problem (§4.1). Label-based (TAPP) and `DotGeneral` (Rust/tenferro) inputs are thin front ends lowered at plan time. Execution cost is identical: the driver already runs on precomputed offset tables (`tensorcontract/src/plan.rs:275`), not on labels. |
| D7 | Kernel family IDs are renamed to one scheme (§6.2). |
| D8 | The library reads **no environment variables**. Every tuning knob becomes explicit configuration; benchmarks translate env/flags into it. |
| D9 | Benchmarks are trimmed to the remaining surface, but the Hadamard corpus (`hadamard.json`) and the GEMM-route corpora are kept, rewritten as contractions where needed. |
| D10 | Heavy performance protocols are not part of Phase 1. Verification is correctness tests, the local gate and one 1T smoke run (§11). |

## 2. Current state (facts at `fbc83f5`)

- 19 workspace members plus `benchmarks`. Lukas Devos's crates are `tensorprimitives/crates/{tensorcontract, tensorprimitives-tapp, tensorprimitives-bench}`. The kernel layer was already moved out of `tensorcontract` in #27.
- Duplicated responsibilities:
  - **Executors and SPMD.** tensorcontract `pool.rs`/`spmd.rs`, `tprims-exec`, the `tprims-core` executor and `tprims-contract-traits::HostExecution`. `Par` is defined twice, and an identical `ExecSpmd` adapter exists three times (blas/tblis.rs, contract/tblis.rs, tapp lib.rs).
  - **Validation.** tensorcontract plan/resolve validate labels, while `DotGeneral::validate`/`validate_layouts` validate axis lists. `is_injective_layout` exists twice, and there are three overlap checks (`core::tensor::spans_overlap`, tapp `overlap`, blas-capi `check_no_overlap`).
  - **TBLIS bridge.** It is re-implemented in tprims-blas and in tprims-contract.
  - **Errors and oracles.** There are seven or more error types and five or more reference oracles.
  - **Thread spawning in tensorcontract.** It still spawns its own threads: a driver fallback using `std::thread::scope`, `batch.rs`, and `pool.rs`.
- Environment variables read in source:
  - In gemm-kernel: `TENSORCONTRACT_{THREADS,KERNEL,COMPLEX,MC,KC,NC,MC_PCT,NC_PCT,KC_COUPLE,BLOCKMODEL,L3_DOMAINS,WRITEBACK}`.
  - In tensorcontract plan/pool: `TENSORCONTRACT_{ORIENT,ROWBLOCK,PARTITION,POOL}`.
  - `TPRIMS_GEMM_ENGINE` and `TPRIMS_GEMM_KERNEL`.
- Files over 800 lines:
  - `tensorcontract/src/plan.rs` (1850)
  - `driver.rs` (1548)
  - `tprims-gemm-kernel/src/cache.rs` (1622)
  - `tensorprimitives-tapp/src/lib.rs` (1239)
- External consumer: tenferro `ext/tenferro-cpu-tprims` pins rev `d8e565a`. It uses:
  - `tprims_exec::{Exec, Pool}`;
  - `tprims_blas::{gemm, gemm_batched, gemm_grouped, trsm, ...}`;
  - `tprims_contract::{ContractPlan, DotGeneral, Flags, Strategy}`;
  - `tprims_linalg` decompositions.

## 3. Target crate set

```
tprims-exec ← tprims-kernel ← tprims-contract ← tprims-capi
                                   ↑
              tprims-testkit (dev-only), tprims-bench (bench-only)
```

The dependency direction is strict. `tprims-kernel` does not depend on `tprims-exec`.

| Crate | Contents (from) | Public surface |
|---|---|---|
| `tprims-exec` | `tprims-exec`; tensorcontract `spmd.rs`; the workspace from gemm-kernel `workspace.rs` | `Pool`, `Exec`, `install`, `for_each_partition`, `broadcast`, `WidthPolicy`, workspace providers, `strided::run_with_exec` (feature `strided`) |
| `tprims-kernel` | gemm-kernel; kernel-tensorcontract; kernel-cplx; portable | `KernelFamily`, `KernelCatalog`/`KernelHandle`, `KernelChoice`, `list_kernels`, `unsafe register`, `SelectError` |
| `tprims-contract` | contract-traits; contract; tensorcontract plan/driver/dynamic/batch/select/resolve/layout/buffer | `Problem`, `Labels`, `DotGeneral`, `Plan`, `PlanConfig`, `ContractionBackend`/`BoxedPlan`, `Error`, `add`, `permute` |
| `tprims-capi` | tprims-core; tensorprimitives-tapp; tprims-bundle | TAPP standard ABI, `tprims_tapp_executor_*`, DLPack, status; builds `libtprims` (cdylib/staticlib/rlib) |
| `tprims-testkit` | tensorcontract `reference.rs`; contract-testkit; the per-test naive helpers | the reference oracle, seeded input generators, a naive second backend (`publish = false`) |
| `tprims-bench` | `benchmarks/`; tensorprimitives-bench | contract, exec_entry, capi_rust, tcbench (TCCG corpus; TBLIS FFI behind a feature) |

Deleted crates:
- `tprims-blas`
- `tprims-blas-capi`
- `tprims-linalg`
- `tprims-kernel-pgx86`
- `tprims-kernel-gemm`
- `tprims-bundle`
- `tprims-custom-kernel-test` (its tests move into kernel/contract)
- `tprims-contract-testkit` (moves into testkit)
- `tprims-contract-traits` (moves into the `tprims-contract::api` module)

The unused workspace dependencies `strided-perm` and `strided-kernel` are removed. The `strided-view`/`strided-basic` pin stays at the tenferro rev.

## 4. `tprims-contract`

### 4.1 Central representation

`Problem` is the single validated description of
`D = alpha * op(A) op(B) + beta * op(C)`:

- dtype T. Extents, signed element strides and offsets for A, B, D, and optionally C. When C is absent, D is updated in place.
- Axes grouped by role, exactly as the driver uses them today:

  | Role | Present in |
  |---|---|
  | batch | A, B, D (and C) |
  | M | A, C, D |
  | N | B, C, D |
  | K | A, B |

  Label patterns the current driver does not support stay typed errors.
- Per-operand conjugation for A, B, C and D, plus alpha and beta.

All validation happens once, when the `Problem` is built:

- repeated and out-of-range axes;
- extent agreement;
- checked size and byte products;
- output injectivity;
- C/D mapping agreement and partial overlap;
- D overlapping A or B.

This replaces the duplicated injectivity and overlap checks listed in §2.

### 4.2 Front ends

- **`Labels`.** Per-operand i64 labels, with free output order and separate C and D. Used by TAPP.
- **`DotGeneral`.** Paired contracting and batch axes. Output order is `[lhs_free, rhs_free, batch]` and C = D. Used by Rust and tenferro.

Both lower to `Problem` at plan time. No front end is visible to the driver.

### 4.3 Plan

`Plan::<T>::new(&problem, &PlanConfig) -> Result<Plan<T>>`.

`PlanConfig` holds:

- `kernel`: `Auto`, an id, or a selector with a catalog (#28);
- `partition`: StaticGrid or DynamicTiles (#29);
- `no_materialize`;
- explicit tuning (§6.4).

The family is resolved once, at plan time. Execution never reselects, reads the environment or registers anything.

In Phase 1 the plan picks one internal strategy, using today's rule:

1. **Elementwise** when every axis is batch.
2. **faer permute+GEMM** only when it fuses every operand without a copy and neither a kernel nor a partition was requested.
3. **Packed driver** otherwise. This is the default.

Strategies live in `src/strategy/{faer.rs, elementwise.rs}`. No public enum names them; `plan.report()` only displays the choice, together with the family, blocking, partition and materialization. `strategy/faer.rs` is the only file that depends on faer. Phase 2 deletes both files.

Execution:

- `plan.execute(&exec, alpha, a, b, beta, c, d)` over `StridedView`/`StridedViewMut`;
- an `unsafe` raw-pointer variant for the C ABI;
- `contract_batched` for many items (§5.2).

### 4.4 Backend trait

`ContractionBackend<T>` / `PreparedContraction<T>` / `BoxedPlan<T>` (#31) move into the `tprims_contract::api` module.

They take `&Exec` directly. This removes the following #31 types:

- `HostExecution`
- `SerialHost`
- `NativeHost`
- `ExecHost`
- the `TypeId` downcast

`Plan<T>` implements the trait itself. The naive second backend lives in testkit.

### 4.5 Errors

There is one `tprims_contract::Error`, with these variants:

- `Config`
- `Shape`
- `Layout`
- `Alias`
- `Unsupported`
- `Select(tprims_kernel::SelectError)`
- `Exec(tprims_exec::ExecError)`

Sources are typed, never stringified. The `tensorcontract::Error`, `contract-traits::Error`, `blas::Error` and `linalg::Error` types disappear. `tprims-capi` maps `Error` to TAPP status codes in one table.

### 4.6 File layout

```
src/
  lib.rs
  api/{problem.rs, labels.rs, dot_general.rs, validate.rs, backend.rs, error.rs}
  plan/{mod.rs, analysis.rs, orientation.rs, report.rs}   ← tensorcontract plan.rs (1850) split
  driver/{mod.rs, static_grid.rs, dynamic.rs, tile.rs}    ← tensorcontract driver.rs (1548) split
  select.rs, resolve.rs, layout.rs, buffer.rs, batch.rs
  strategy/{faer.rs, elementwise.rs}                      ← Phase 1 only
  wrappers.rs                                             ← add, permute (strided-basic)
```

## 5. `tprims-exec` and the parallel strategy

### 5.1 Primitives

- `Pool<'p>` wraps a borrowed `&rayon::ThreadPool`. There is one wrapper per pool, and it owns the SPMD gate and the entry counters.
- `Exec` is either Serial or Rayon { pool, budget }.
- `install(k, op)`: width one runs inline; a wider region enters the pool once.
- `for_each_partition(k, f)` is barrier-free.
- `broadcast(width, f)` runs the whole team concurrently or nothing. It is serialized per pool and refused from a worker of the same pool.
- Workspace: `ArenaProvider`, `TeamLease` and the rest move here from `tprims-kernel`. There is one arena per pool plus a serial arena, and no process-global state. These types are byte arenas, so `tprims-kernel` needs no executor type.
- `strided::run_with_exec` is kept for `add`/`permute` and for the copies the Phase 1 faer strategy makes.

Deleted:

- tensorcontract `pool.rs` and `TENSORCONTRACT_POOL`;
- the `Spmd` trait;
- every `std::thread::scope` path;
- the three `ExecSpmd` adapters;
- the duplicate `Par`;
- `TENSORCONTRACT_THREADS`.

### 5.2 Parallel strategy

1. **One source of threads.** Threads come only from `Exec`; without one, execution is serial.
2. **Width.** It is chosen once per execution: `width = WidthPolicy(exec.budget, plan estimate)`. The estimate uses the selected family's cost per flop. The hard-coded `NS_PER_FLOP` constants (tapp and blas) are removed. Work estimated below the serial threshold never enters the pool.
3. **Inside one contraction.** SPMD via `broadcast`, with StaticGrid by default (MR/NR-aligned pm×pn, L3-domain aware) or DynamicTiles as opt-in. In-plan batch axes stay serial inside the team in Phase 1.
4. **Many items** (`contract_batched`, `TAPP_execute_batched_product`):
   - when there are at least as many items as the width, items are split barrier-free with `for_each_partition` and each item runs serially;
   - otherwise items run in sequence, each with inner SPMD.

   One budget governs both levels.
5. **Nesting.** A broadcast from a worker of the same pool is refused, and execution falls back to serial with the same family.
6. **faer strategy (Phase 1).** It enters with `install` and runs `faer::Par::rayon(n)`. The batched faer loop from the deleted `tprims-blas` moves into `strategy/faer.rs`, so the batched GEMM corpora do not regress.

## 6. `tprims-kernel`

### 6.1 Layout

```
src/
  lib.rs
  abi/{family.rs, types.rs, element.rs, cpu.rs}
  select/{registry.rs, resolve.rs, catalog.rs, partition.rs}
  pack/{pack.rs, scatter.rs, writeback.rs}
  blocking/{probe.rs, model.rs}                 ← cache.rs (1622) split
  kernels/
    mod.rs          built-in registration and priorities
    macros.rs       ISA-generic tile bodies (was simd.rs)
    reference/{portable.rs, scalar.rs, induced.rs}
    x86/{mod.rs, avx2.rs, avx512.rs, avx2_complex.rs}
    aarch64/neon.rs
```

There is one crate for all project-owned kernels. Lukas Devos's code and the project's own share MIT OR Apache-2.0, so per-origin crates are no longer needed. Only external-dependency adapters needed separate crates, and those are deleted (D3).

### 6.2 Family IDs

The scheme is `{isa}.{dtype}.{scheme}.{MR}x{NR}`:

| Field | Values |
|---|---|
| isa | `ref`, `avx2`, `avx512`, `neon` |
| dtype | storage dtype `f32`/`f64`/`c32`/`c64` |
| scheme | `real` for real families; for complex, `planar`, `native`, `1m`, `3m`, `4m`; `i1m`/`i4m` for induced families; `direct` for direct-C |

The complex field changes from today's real-type name: `tc.*` currently names complex families with `f64`.

| Old | New |
|---|---|
| `tc.avx2.f64.8x6` | `avx2.f64.real.8x6` |
| `tc.avx2.f64.planar.4x5` (c64) | `avx2.c64.planar.4x5` |
| `cplx.avx2.c64.native.4x4` | `avx2.c64.native.4x4` |
| `portable.c64.native.4x4` | `ref.c64.native.4x4` |

Provenance is carried by the `Origin` metadata, not by the ID. The full ID list is pinned by a snapshot test, and resolution rejects duplicates.

### 6.3 Selection

`Auto` is unchanged in Phase 1: among CPU-available families in descending priority, it takes the first with `allow_auto`. ID selection, the catalog/selector (#28) and DynamicTiles (#29) are kept.

### 6.4 Configuration instead of environment variables

Every `TENSORCONTRACT_*` and `TPRIMS_GEMM_*` variable is deleted. The knobs become `PlanConfig` fields:

- kernel
- complex method
- blocking override (mc/kc/nc and percentages)
- orientation
- row block
- partition
- cache-model override (block model, L3 domains)
- write-back mode

Benchmarks parse env/flags into `PlanConfig`.

## 7. `tprims-capi`, `tprims-testkit`, `tprims-bench`

- **`tprims-capi`.** The ABI is unchanged from #26. The `blas` feature is gone.
  - Headers in `include/`: the pinned upstream TAPP headers (BSD-3) and `tprims/{tprims,core,tapp_ext}.h`. Lukas Devos's second `tapp.h` is deleted.
  - `examples/c-consumer`, the pkg-config template and `install.sh` are adapted to `libtprims`.
  - The tapp `lib.rs` (1239 lines) is split into `tensor_info.rs`, `product.rs`, `validate.rs` and `executor.rs`.
- **`tprims-testkit`.**
  - One label-based reference oracle (from `tensorcontract::reference`), used by every test that needs an oracle.
  - Seeded generators (rand_chacha).
  - The naive `ContractionBackend`.
- **`tprims-bench`.**
  - Contract benchmarks driven by the corpora: `tenferro-p1`, `hadamard`, `example`, and `tenferro-p1-gemm`/`large-batched-gemm` rewritten as contractions.
  - `exec_entry`, `capi_rust` and `tcbench`.
  - The blas and linalg benchmarks are deleted.

## 8. Documents and assets

- `tensorprimitives/docs/` (design, decisions, refuted, measurement-rules, notebook) moves to `docs/archive/tensorprimitives/`.
- Root `docs/` (architecture, decision-log, provenance, design-principles, research-map) is rewritten for the new structure. A new `docs/migration-2026-10.md` maps old paths and names to new ones for tenferro.
- Deleted: `RELEASING.md` (crates.io/JLL/Julia procedure, obsolete by D1), `CHANGELOG.md`, `CONTRIBUTING.md`, `tensorprimitives/CLAUDE.md`.
- Removed from the tree: `bench-results/` (about 2400 files) and `scripts/` (sbatch and analysis). Both stay in history. `docs/provenance.md` records the last commit that contains them, `2155bb63e1a87a1f993d50c439b98a9bc7298ad5`. Scripts that tcbench still needs are kept under `tprims-bench`.
- Add root `LICENSE-MIT` / `LICENSE-APACHE`, with copyright for Lukas Devos and tensor4all contributors.

## 9. History and authorship

- Moves use `git mv`, in commits separate from content edits, so `git log --follow` works.
- Files derived from tensorprimitives-rs keep their source/author headers.
- The `authors` of `tprims-kernel`, `tprims-contract` and `tprims-capi` include Lukas Devos.
- Commits that port his code carry `Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>`.

## 10. Phase 2 (separate spec, summary only)

**Goal.** Optimize the packed driver, then delete `strategy/faer.rs` and `strategy/elementwise.rs` and the faer dependency.

**End state.** One route, where cases differ only in the selected family.

**Work items:**

- direct-C write-back by default;
- tuned AVX2 tiles;
- `target_feature` on packing and write-back;
- a small-problem path with no fixed packing cost;
- parallel-partition review;
- driver-side batch-axis job claiming, the cross-batch claiming deferred in #29, needed for tiny batched GEMMs and Hadamard;
- reusable SIMD placed in strided-rs where practical.

**Acceptance.** No regression on `tenferro-p1`, `hadamard` and the GEMM corpora.

## 11. Delivery and verification (Phase 1)

There are four PRs in dependency order. Each is green on the local gate before it is pushed.

1. **exec consolidation.** Workspace move, `Spmd` replaced by `Exec`, thread-spawning paths deleted.
2. **kernel consolidation.** `tprims-kernel` layout, ID rename, removal of environment reads.
3. **contract consolidation.** `Problem`, front ends, plan, single error, driver move, strategies. Deletes blas, linalg, pgx86, kernel-gemm and contract-traits.
4. **capi, testkit and bench consolidation.** Docs archive, licenses, and removal of `tensorprimitives/`.

Implementation is mostly one sequential lane. testkit/bench and the docs archive can run in parallel. PR 3 is internally sequential: problem and front ends, then the driver move, then the strategies.

Verification:

- every numerical test of a kept feature is ported and passes;
- tests of deleted features are removed together with the feature;
- the local gate passes: fmt, clippy `-D warnings`, workspace tests, release tests of the packed driver, docs `-D warnings`, MSRV 1.89, C/C++ ABI consumers, the `nm` symbol check and the scripts job;
- one 1T smoke run of `tenferro-p1` and `hadamard` against `fbc83f5`, checking only for gross regressions, with no paired protocol.

## 12. Issues

- **#22.** Add a note that the linalg split is obsolete (linalg is deleted); the repository rename remains.
- **#31.** Slice 4 (the tenferro bridge) stays open and is rebased onto the new API.
- **New tracking issue.** One issue for this spec. Phase 2 gets its own issue when its spec is written.

## 13. Out of scope

- The tenferro update.
- Phase 2 optimization.
- New kernels.
- Repository rename.
- Publishing.
