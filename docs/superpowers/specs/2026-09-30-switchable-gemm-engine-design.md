# Switchable GEMM engine: design

- Date: 2026-09-30
- Issue: #23 (this spec revises its provenance rule; see §9)
- Status: draft v3 for maintainer review (v1 reviewed by an independent
  reviewer; findings and resolutions in Appendix A)
- Scope: the **mechanism only**. Optimization, kernel ports and tuning are a
  separate, later session. This work stops when the mechanism is in place,
  tested, and shown not to slow down the current paths.

## 1. Goal

Make every choice inside tprims' GEMM path an explicit, queryable value,
selected once and carried into execution:

- which engine runs a matrix GEMM: faer, private-gemm-x86 called directly, or
  the packed (TBLIS-style) driver;
- inside the packed driver, which kernel family: microkernel, packers, C
  update, MR/NR and blocksizes (MC/KC/NC);
- for complex types, the arithmetic method (native, 1m, 3m, 4m) and packed
  complex layout (interleaved, planar, 1e/1r);
- SIMD-optimized versus portable generic kernels;
- how the work is split across threads (partition policy);
- how packing buffers are sized, owned and first touched (workspace).

The same mechanism must host kernels ported later from BLIS, OpenBLAS and
faer, each in its own crate under its own license, and optimized native
complex kernels when they exist.

### Non-goals (this phase)

- No new optimized kernels and no ports of BLIS/OpenBLAS/faer source.
- No tuning of tile shapes or blocksizes; current values carry over.
- No implementation of a dynamic partition policy: it is designed (§6.1) and
  implemented in the optimization session.
- No C ABI for selection (shape only, §8).
- No NUMA-aware team pools, huge pages or buffer trimming policy beyond
  `trim()` (§6.2).

## 2. Background (research summary)

Sources: `docs/worklogs/2026-09-30-gemm-strategy-faer-vs-tensorcontract.md`
and source studies made for this spec (BLIS `913242c` and 0.9.0 `14c86f6`;
TBLIS `eb719e7` and 1.x `c4f81e0`; faer, gemm-common 0.19, private-gemm-x86
0.1.20, nano-gemm 0.2; the in-tree `tensorcontract`). Only structure was
studied; no code is taken from them in this phase.

- **BLIS** splits state into two levels:
  - an immutable *context* (`cntx_t`), one per CPU configuration, chosen once
    at startup (CPUID newest first, env override, `generic` fallback) and
    holding per-dtype kernel pointers, packers, row/column preference and
    {default, max} blocksizes;
  - a per-call *control* object that fixes the pack schemas, the scaled
    blocksizes and the kernel pointer.

  The complex method is data: 1m changes only pack schemas, blocksize scales
  and which kernel pointer is called. BLIS removed 3m/4m in 2021. Threading
  ("ways" of jc/ic/jr/ir) is a separate runtime object.
- **TBLIS**:
  - 1.x has per-dtype function-pointer tables registered with priorities and
    a check function; scatter support is extra pack kernels plus a C-update
    wrapper around an unchanged microkernel;
  - 2.x is a BLIS plugin and inherits BLIS's selection.
- **faer** on this EPYC 7713P routes f64/c64 matrix GEMM to private-gemm-x86:
  - build-time generated assembly, with a native interleaved c64 6×4 kernel;
  - B read in place in most cases, and A packed inside the kernel;
  - dynamic 2-D job claiming with spin-then-sleep workers;
  - persistent page-aligned thread-local buffers.

  gemm-common compiles the whole driver per ISA and caches one function
  pointer; `gemm-f64` exposes its microkernels publicly as
  `gemm_common::microkernel::MicroKernelFn` (alpha/beta/dst owned by the
  kernel); `gemm-c64`'s are private.
- **tensorcontract** already has most of a descriptor:
  `KernelConfig<R> = { Ukr<R>, Blocking }` (`kernel/mod.rs:291-465`) and
  per-ISA tables `IsaConfigs<R>` (`x86.rs:517-524`). What it lacks:
  - packers and write-back are generic, with a per-element format `match`
    (`pack.rs:124-157`, `writeback.rs:66-93`);
  - there is no ISA or method tag and no validation;
  - there is no native-interleaved or 4m variant;
  - `config_for_plan` runs on every execute (`driver.rs:357`), and
    `Blocking::derive` reads `TENSORCONTRACT_KC_COUPLE` each time
    (`kernel/mod.rs:393`);
  - `model` blocking sizes NC with `plan.threads()` (`kernel/mod.rs:903`)
    instead of the Spmd width (`driver.rs:461-466`).

  `Plan` is dtype-agnostic (`plan.rs:298-302`).
- **Threading today**:
  - tensorcontract splits output M (whole MR panels) and N (whole NR
    slivers, re-cut per NC block) statically; K is serial and batch is never
    parallel; there are two futex barriers per (batch, NC, KC) step; strip
    edges are not cache-line aligned.
  - private-gemm-x86 runs KC slabs serially and claims (2–4)·mr × 32-column
    jobs dynamically inside a slab.
- **Buffers today**:
  - tensorcontract allocates every packing buffer per call (64-B aligned).
  - The shared B panel is allocated by the caller before `broadcast`
    (`driver.rs:503-545`) but first touched by the packing workers.
  - A is packed and consumed by one worker, but duplicated `pn` times and
    never reused.
  - The serial fallback allocates B twice (`driver.rs:586-595`).
  - Barriers (`driver.rs:554`) and block-scatter vectors
    (`driver.rs:427-437`) are also allocated per call.
  - BLIS keeps page-aligned team-owned pool buffers reused across calls.
    Neither BLIS nor TBLIS handles NUMA.
- **This host**: 1 NUMA node, 8 L3 domains (CCDs) of 8 cores. First touch
  matters for L3 locality here and for DRAM locality on multi-node hosts.

## 3. Architecture

Two selection levels, as in BLIS, plus orthogonal runtime policies:

```
GemmEngine                                  per plan (default resolved once per process)
 ├─ Faer               faer matmul               (current matrix-GEMM default)
 ├─ PrivateGemmX86     private_gemm_x86::gemm    (called directly)
 └─ Packed(&'static KernelFamily<T>)             (tensorcontract driver)
        KernelFamily<T> = typed immutable descriptor from a per-dtype registry:
          microkernel fn + MR/NR + C preference, packers, C update,
          complex scheme, {default, max} MC/KC/NC, capabilities
   + PartitionPolicy   (split of the (batch, M, N) space across threads)
   + Workspace         (sizes, ownership, first touch, reuse)
```

- The **typed** tprims plans (`tprims-blas` plans, `ContractPlan<T>` at
  `tprims-contract/src/plan.rs:84`) resolve all choices once at plan creation
  into `ResolvedGemm<T>`: engine, family, concrete scaled blocksizes for the
  effective thread width, partition policy, and `WorkspaceReq`.
- The tensorcontract driver gains `execute_with(…, &ResolvedGemm<T>, spmd)`.
  Its existing upstream entry points `Plan::run` / `run_raw` build a
  `ResolvedGemm<T>` lazily through per-dtype `OnceLock` slots, so upstream
  users keep working.
- Execution reads only `ResolvedGemm<T>`: it never consults environment
  variables or rebuilds configs.
- `Exec` owns threads and pools; the kernel layer owns none.

### 3.1 Crates

The kernel layer is split out of `tensorcontract` into its own crates, so
the driver, the kernel contract and each origin's kernels are separate
crates and licenses never mix. Lukas Devos's files move with `git mv`: their
history stays reachable with `git log --follow`, file headers and
authorship are kept, and `authors` in the new crates' `Cargo.toml` lists
him. Moving files out of the `tensorprimitives/` subtree ends plain
`git subtree pull` from upstream for those files; this is accepted and is
recorded in `docs/provenance.md`.

| Crate | Contents | License / authorship |
| --- | --- | --- |
| `tprims-gemm-kernel` (new) | The contract. Moved from tensorcontract: `element.rs` (`Element`, `Real`), `scatter.rs`, `pack.rs`, `writeback.rs`, `kernel/cache.rs`, and the type half of `kernel/mod.rs` (`PackFormat`, `TileFormat`, `Ukr`, `Blocking`, `ComplexMethod`). New: `KernelFamily<T>`, the fn-pointer types, the per-dtype registry, selection and validation, `CpuFeatures`, `WorkspaceReq` and `WorkspaceProvider`, induced 1m/4m generation, and the portable family (§4.5). It has no threading and depends on no other tprims crate. | MIT OR Apache-2.0; moved files keep Lukas Devos's authorship; commits that rework them carry `Co-authored-by: Lukas Devos` |
| `tprims-kernel-tensorcontract` (new) | Lukas Devos's SIMD and scalar microkernels (`kernel/x86.rs`, `aarch64.rs`, `simd.rs`, `scalar.rs`), moved and registered as families | MIT OR Apache-2.0, Lukas Devos |
| `tprims-kernel-gemm` (new, optional) | Wraps the public `gemm-f64`/`gemm-f32` microkernels (`MicroKernelFn`) as `CUpdate::Direct` families. Calls only; no source copied. | adapter MIT OR Apache-2.0; depends on `gemm-*` (MIT, Sarah Quiñones) |
| `tprims-kernel-pgx86` (new, optional) | Calls `private_gemm_x86::gemm` as the `PrivateGemmX86` engine. Calls only. | adapter MIT OR Apache-2.0; depends on private-gemm-x86 (MIT, Sarah Quiñones) |
| `tprims-kernel-blis`, `-openblas`, `-faer` (future) | Ported families, one crate per origin. Not created now. | BSD-3-Clause / BSD-3-Clause / MIT with original notices |
| `tensorcontract` (existing) | The packed (TBLIS-style) driver: `plan.rs`, `driver.rs`, `batch.rs`, `layout.rs`, `buffer.rs`, `pool.rs`, `spmd.rs`, `reference.rs` (the test oracle), `error.rs`, plus the plan glue that stays from `kernel/mod.rs` (`config_for_plan`, `plan_config`, env handling). It re-exports the moved types (`Element`, `Real`, `C32`, `C64`, `Blocking`, `ComplexMethod`, `KernelSet`), so its public API does not change. | MIT OR Apache-2.0, Lukas Devos |
| `tprims-blas`, `tprims-contract` | Typed plans that resolve engine, family and policies, and expose selection and query | unchanged |

Dependency direction:

```
tprims-gemm-kernel  <-  tprims-kernel-*  <-  tensorcontract  <-  tprims-blas  <-  tprims-contract
        ^                                        |
        +----------------------------------------+   (the driver uses the contract directly)
```

- Kernel crates depend only on `tprims-gemm-kernel` and their own upstream
  crate. They never depend on the driver.
- `tensorcontract` depends on `tprims-gemm-kernel` and
  `tprims-kernel-tensorcontract`, which provide its default families.
- `tprims-blas` features choose which optional kernel crates are built
  (`kernel-gemm`, `kernel-pgx86`, later `kernel-blis`, …).
- The portable family and the tensorcontract families are always built.

## 4. Kernel families

### 4.1 Descriptor

`KernelFamily<T>` is a plain `'static` struct, built by `const`/`fn`
constructors; no trait objects, no registration macros:

- `id: &'static str` — stable identifier (`tc.avx2.f64.8x6`,
  `tc.avx512.c64.planar.8x6`, `portable.c64.native.4x4`, `gemm.avx2.f64.…`).
- `origin: Origin` (with crate name and license) for `list_kernels()`.
- `isa: Isa`, `required: CpuFeatures` — a family is never selectable on a
  CPU lacking `required`.
- `imp: KernelImpl` — `Reference`, `Optimized`, `Induced` (§4.3).
- `complex: Option<ComplexScheme>` — `{ method, a_layout, b_layout,
  tile_layout }`, `None` for real `T`.
- `mr, nr, pack_mr, pack_nr`; `tile_bound` (scratch tile size in reals).
- `c_pref: CPref` — `Row`, `Col`, `Any`. It feeds the existing orientation
  rule (`plan.transposes_gemm(mr)`, `plan.rs:808`, and
  `TENSORCONTRACT_ORIENT`); no second transposition rule is added.
- `ukr: UkrFn<T>` (§4.2).
- `pack_a: PackFn<T>`, `pack_b: PackFn<T>` — by default the existing
  tensorcontract packers, monomorphized per layout; families with their own
  format (OpenBLAS-style copy routines) supply their own.
- `b_access: BAccess` — `Packed`, or `Direct { unit_stride: Axis }` (B read
  in place, faer style; §6.2).
- `c_update: CUpdate` — `ScratchTile` (current contract) or `Direct` (§4.2).
- `blocks: Blocksizes` — `{default, max}` for MC/KC/NC with the multiples
  they must respect. `max` is descriptor data for the future tail-merge rule
  and has no consumer in this phase.
- `caps` — `scatter_pack`, `conj_a`, `conj_b`.

The registry validates every descriptor in a unit test over all registered
families, and in debug builds at first use:

- MR and NR are non-zero, and even where 1m is allowed;
- the blocksize multiples hold;
- `pack_mr` ≥ MR and `pack_nr` ≥ NR;
- `tile_bound` covers the tile layout;
- the layouts are consistent with the method (§4.3 table).

The registry is per dtype: `families::<T>() -> &'static [&'static
KernelFamily<T>]`, via a sealed trait implemented for f32/f64/c32/c64.
`KernelSet` is kept only as a thin compatibility shim over the registry and
removed once no caller needs it.

### 4.2 Microkernel contract

```rust
pub type UkrFn<T> = unsafe fn(k: usize, a: *const <T as Element>::Real,
    b: *const <T as Element>::Real, c: *mut <T as Element>::Real,
    rs_c: isize, cs_c: isize, alpha: T, beta: T, aux: &UkrAux<T>);
```

Typed per dtype, not erased: erasing buys nothing until the C ABI exists.

- **`CUpdate::ScratchTile`** (default; today's tensorcontract contract,
  `kernel/mod.rs:313-317`). The driver passes the plan-owned scratch tile in
  the family's tile layout with `alpha = 1`, `beta = 0`, and the kernel
  overwrites it. Edges come from zero-filled packing. Write-back applies
  alpha, beta, `op_C`, `op_D` and scatter, computing `D = op_D(alpha·AB +
  beta·op_C(C))` with distinct `c` and `d` as today (`driver.rs:772-816`).
- **`CUpdate::Direct`** (BLIS/OpenBLAS/gemm-f64 style). The kernel applies
  alpha and beta and stores a full tile into the output with general
  strides. The plan uses it only when all of these hold:
  - D is written in place: D aliases C exactly (same base and the same
    scatter), or beta = 0, in which case the first KC block uses beta = 0
    and later blocks beta = 1 on D;
  - `conj_c == conj_d == false`;
  - the tile is full and has constant strides (no block scatter).

  Every other tile, and every plan that fails the first two conditions, goes
  through a scratch tile and the family's reference write-back. This is
  decided per plan, and per tile for edges and scatter; it is not an error.
- `UkrAux<T>` carries:
  - `a_next`/`b_next` (prefetch);
  - B strides for direct-B;
  - imaginary strides for planar layouts;
  - `inner: Option<&'static KernelFamily<T::Real>>`, which induced complex
    families use to reach their real kernel.

### 4.3 Complex schemes

External storage is always interleaved (`Complex<T>`). Packed layout and
arithmetic are per family:

| method | a / b layout | tile layout | kernel | status |
| --- | --- | --- | --- | --- |
| `Native` | interleaved / interleaved | interleaved | native complex ukr | new: portable family (§4.5); optimized native kernels later |
| `Native` | planar / planar | planar | complex ukr on split planes | existing tensorcontract "planar" |
| `OneM` | 1e / 1r (per `c_pref`) | real 2MR×NR or MR×2NR | real ukr | existing tensorcontract "1m" |
| `ThreeM` | 3-plane | 3-plane | fused 3m ukr | existing; accuracy trade-off, never chosen by `Auto` unless allowed |
| `FourM` | planar / planar | **4-plane** (Ar·Br, Ai·Bi, Ar·Bi, Ai·Br) | real ukr, 4 overwrite calls per tile | new: induced from any real family; recombined in write-back like 3m (`writeback.rs:84-92`); `tile_bound` = 4·MR·NR |

- Tensorcontract's `ComplexMethod::{Planar, OneM, ThreeM}` maps to
  `(Native, planar)`, `OneM` and `ThreeM`. The new enums are
  `#[non_exhaustive]`, and the `TENSORCONTRACT_COMPLEX` values keep their
  meaning.
- Induced families (`OneM` from any real family with even MR/NR, `FourM`
  from any real family) are generated by one function from a real
  `KernelFamily<R>`, as BLIS derives 1m from the real context. That one
  function supplies:
  - the blocksize scaling (KC/2 for 1m; MR or NR ×2 per `c_pref`);
  - the choice of packers;
  - the wrapper ukr.

  A newly ported real kernel therefore gets 1m and 4m with no extra code.
- Induced families require `CUpdate::ScratchTile` on the wrapper; the inner
  real kernel may be `Direct` or `ScratchTile`, since the wrapper always
  calls it into the scratch planes.

### 4.4 Selection

- `list_kernels()` returns, per built family: `(id, origin, license, isa,
  dtype, complex, mr, nr)`, plus `available_on_this_cpu` as a separate
  field that is not part of the id.
- Plan-config fields:
  - `engine: EngineChoice::{Auto, Faer, PrivateGemmX86, Packed}`;
  - `kernel: KernelChoice::{Auto, Id(Cow<'static, str>)}`, which is used by
    `Packed` and by the TBLIS-style contraction branch.
- **Process default** per dtype:
  - resolved once through `OnceLock` from CPU features;
  - an optional env override is read once (`TPRIMS_GEMM_ENGINE`,
    `TPRIMS_GEMM_KERNEL`); existing `TENSORCONTRACT_*` variables are also
    read once, into the default;
  - `Auto` picks the highest-priority family available on this CPU and never
    fails, because the portable family is always last.
- **Plan override** wins over the default. Two plans in one process may use
  different families concurrently; nothing mutable is global.
- **Errors at plan creation, never at execute**:
  - an unknown id;
  - a family that was not built (the error names the cargo feature);
  - a CPU that lacks the family's required features;
  - a dtype mismatch;
  - an operand the family cannot handle (direct-B without the needed unit
    stride, or scatter operands without `scatter_pack`);
  - an engine that cannot express the operation (`PrivateGemmX86` on
    non-x86 or for block-scatter contraction);
  - a partition policy that is not implemented (§6.1).

  There is no silent substitution: a forced choice runs as chosen or fails.
  The per-tile Direct→ScratchTile fallback of §4.2 is part of the family's
  contract, not a substitution.
- **Query**: `plan.selected() -> SelectedGemm` returns `{ engine, family_id,
  complex, mr, nr, mc, kc, nc, partition, workspace, batched }`. The
  `batched` field keeps today's `Selected::{FaerLoop, Tblis} {
  outer_parallel }` information (`batched.rs:70`). The value is printable
  and serializable to JSON for benchmark manifests.

### 4.5 Portable family

Project-owned, in `tprims-gemm-kernel` (`portable` module):

- `fn portable_ukr<T, const MR: usize, const NR: usize>` for real and native
  interleaved complex, written with plain loops for LLVM to vectorize (no
  intrinsics).
- Variants that exercise every contract branch in this phase:
  `ScratchTile`, `Direct`, and `BAccess::Direct`.
- Its roles: the always-available fallback, the first native interleaved
  complex family, and the base for induced 1m/4m tests.

The existing `reference.rs` stays the independent test oracle and shares no
code with this family (`reference.rs:1-6`).

## 5. Engines

- **`Faer`**: the current `tprims-blas` matrix GEMM path, unchanged. faer's
  own small-size routes (nano-gemm, matvec, `matmul/mod.rs:1214-1331`) stay
  inside it.
- **`PrivateGemmX86`**: `tprims-kernel-pgx86` calls
  `private_gemm_x86::gemm(dtype, itype, instr, …, n_threads)`
  (`lib.rs:1154-1187`) for dense strided f32/f64/c32/c64 matrix GEMM, with
  `InstrSet::Avx256`/`Avx512` taken from CPU features.
  - **Threading**: spindle's `with_lock` opens a `rayon::scope` in the
    current pool and caps it at `n_threads` (`spindle/src/lib.rs:214,
    274`). The adapter calls it inside `Exec`'s pool install with `n_threads`
    equal to the plan's width, so it is confined exactly as today's Faer path
    is. It also inherits that path's traits: n−1 workers spin (up to about
    64K iterations) for the whole call, and `Exec::budget` is seen only
    through `n_threads`. The engine inherits this; it does not newly
    guarantee anything.
  - **Bypassed routes**: calling private-gemm-x86 directly bypasses faer's
    small-size routes; that difference is part of what the engine switch
    measures.
  - **Scope**: it cannot pack block-scatter tensors, so `tprims-contract`
    reaches it only through permute+GEMM.
  - **Microkernel API**: its public but undocumented
    `call_microkernel`/millikernel API uses a custom register ABI. Wrapping
    it as a packed-driver family is recorded as a later candidate.
- **`Packed(family)`**: the tensorcontract driver with the chosen family.

For `tprims-contract`, `Strategy::Auto` keeps its copy-aware rule
(TBLIS-style only when permute+GEMM would copy):
- the permute+GEMM branch uses the plan's matrix engine;
- the TBLIS-style branch uses the packed driver with the plan's family.

## 6. Threading and buffers

### 6.1 Partition policy

`PartitionPolicy` is resolved per plan from the family's blocksizes and the
**effective** thread width, which fixes the `model` blocking bug.

**Implemented: `StaticGrid { pm, pn }`.** This is today's behaviour:
- whole MR panels × whole NR slivers;
- K serial;
- the batch axis inside each thread;
- two barriers per KC step.

**Designed, not implemented: `DynamicTiles { job_m, job_n }`.** Selecting it
returns a "not implemented" error at plan creation. Its design, recorded for
the optimization session:
- KC slabs run serially; within a slab, jobs of `job_m`×`job_n` (multiples
  of MR/NR) are claimed from an atomic counter.
- Sync per slab: one barrier after cooperative B packing, and one before B
  is overwritten by the next slab. With a single persistent B buffer that
  is two points, not one.
- A packing: either each job packs its own A strip (A packing cost ×
  NC/job_n, first touch by the consumer), or rows are claimed as bands so
  each A block is packed once by the worker that claims its band. The second
  keeps consumer first touch and is the preferred design.
- Determinism: K stays serial, every tile boundary is an MR/NR multiple, and
  every tile uses the same kernel and the same KC-ordered accumulation.
  Results are therefore bitwise identical to serial, provided A-packing does
  not reorder any sum (it does not: packing only moves values).

**Invariants (tested for StaticGrid, required of any future policy):**
- K is never split; results are bitwise identical to width 1 for every
  width.
- All threads of one execute use the same MC/KC/NC, taken from the resolved
  plan.
- M and N split boundaries are multiples of MR and NR; the last part takes
  the ragged tail.
- The index space passed to a policy includes the batch axis, so a later
  policy may split batch items. StaticGrid keeps batch inside each thread.
- `align_c_lines` (off by default) rounds M/N split boundaries so C strips do
  not share cache lines; it exists so the optimization session can measure
  it.

### 6.2 Workspace

At plan creation the resolved plan computes a `WorkspaceReq`. Any entry may
be zero.
- **Per worker**: the packed A block (MC×KC in the A layout), the scratch
  tile(s) (`tile_bound`), and that worker's block-scatter vectors.
- **Per team**: the packed B panel (KC×NC in the B layout), the barriers,
  and the shared scatter vectors.
- **B not packed ⇒ no B buffer.** When the family reads B in place, or the
  plan decides B needs no packing, the B entry is zero and no B buffer is
  allocated. The decision is made per plan from the family and the actual
  operand strides, not per family, because a direct-B family may still have
  to pack B when its strides are unfriendly.

Buffers come from a **workspace arena reached through the `Spmd` seam**:
- `Spmd` gains `fn workspace(&self) -> Option<&dyn WorkspaceProvider>`.
  `ExecSpmd` (tprims-exec) owns a provider for its pool; tensorcontract's own
  `std::thread::scope` fallback and opt-in pool (`driver.rs:598-608`,
  `pool.rs`) return `None` and keep today's per-call allocation. Those paths
  remain for upstream users; tprims never reaches them.
- **Per-worker buffers**:
  - held in the worker thread's TLS, keyed by the pool;
  - allocated and first touched by the worker that consumes them, inside the
    broadcast closure;
  - page-aligned (4096 B), grow-only, reused across calls.
- **Team buffers (B panel, barriers, shared scatter vectors)**:
  - the provider owns them per pool, not a worker's TLS;
  - the caller, which is not a pool worker (`exec.rs:224-225`), obtains the
    pointer before `broadcast` and places it in `Ctx`;
  - the provider allocates without touching pages, so each packing worker
    first touches only the slice it packs, before the barrier;
  - one team buffer set per concurrent execute on the pool (a small
    lock-free free-list), so concurrent plans on one pool never share one.
- **Re-entrancy**: TLS buffers are taken with `RefCell::try_borrow_mut`, and
  a failed borrow falls back to a fresh `Vec` for that call, as
  private-gemm-x86 does (`lib.rs:1300-1310`). The width-1 serial fallback
  uses the caller's buffers and no longer allocates B twice.
- **Bound and release**: per worker, one A block plus the tile(s) at the
  largest size seen; per team set, one B panel. `WorkspaceProvider::trim()`
  releases everything, and dropping a pool drops its provider.
- **Steady state**: after a first call of a given size, an execute performs
  no heap allocation for packing, barriers or scatter vectors. This is
  tested with a counting allocator.

Block-scatter vectors are filled cooperatively, before the barrier, into the
arena rather than by the caller.

## 7. Changes to existing code

- **Crate split** (new crates under `crates/`, per the repository layout):
  - `tprims-gemm-kernel` receives `element.rs`, `scatter.rs`, `pack.rs`,
    `writeback.rs`, `kernel/cache.rs` and the type half of `kernel/mod.rs`;
  - `tprims-kernel-tensorcontract` receives `kernel/x86.rs`, `aarch64.rs`,
    `simd.rs`, `scalar.rs`; their `KernelConfig`/`IsaConfigs` menus become
    `KernelFamily<T>` registrations, each MR×NR menu entry a family with an
    id;
  - their unit tests move with them; doc links to `crate::Plan` in moved
    files become links to `tensorcontract`, or plain text where a link would
    create a dependency cycle.
- **In `tprims-gemm-kernel`**:
  - packing and write-back become family pointers, and the per-element
    format `match` becomes per-layout monomorphized functions;
  - a four-plane tile format and write-back recombination for 4m;
  - the test `x86_isas_agree_on_the_pack_contract` (moved to
    `tprims-kernel-tensorcontract`) becomes "families with the same complex
    scheme agree on the pack contract".
- **`tensorcontract`**:
  - re-exports the moved public types;
  - `config_for_plan` leaves the execute path;
  - the orientation swap swaps `pack_a`/`pack_b`;
  - env reads happen once;
  - `Spmd::workspace`.
- **`tprims-blas`**:
  - plans carry `EngineChoice`/`KernelChoice`;
  - the `Scalar::Re: KernelSet` bound (`scalar.rs:34`) moves to the
    registry trait;
  - `tblis.rs`, which today builds a `Plan` per call, uses `ResolvedGemm`;
  - `Selected` becomes `SelectedGemm`.
- **`tprims-contract`**: `Config` gains `engine` and `kernel`; plans expose
  `selected()`.
- **`tprims-linalg`** uses `tensorcontract::Element` (`cholesky.rs:55`,
  `batched.rs:16`, `mat.rs:34`). `Element` does not move, so nothing breaks;
  it is listed so the shim removal checks these callers.
- **`tprims-exec`**: `ExecSpmd` gets the `WorkspaceProvider`.
- **Docs, in the same PR** (docs-vs-code check):
  - `README.md` crate table and the `tprims-gemm-kernel` mention;
  - `docs/architecture.md:32-55`, where the planned `tprims-gemm-kernel`
    becomes real and the kernel crates are added, plus the scalar-trait
    dependency at line 34;
  - `docs/provenance.md`: per-origin crates, and `gemm-*`/private-gemm-x86
    as called dependencies;
  - `docs/decision-log.md`.

## 8. Future C ABI (shape only)

The C ABI is not implemented in this phase. Its intended shape:
- `tprims_gemm_kernels_list`, returning ids and attributes;
- `kernel_id` and `engine` string fields on the blas and contract plan
  configs;
- `tprims_*_plan_selected`, returning the JSON of `SelectedGemm`.

## 9. Provenance rule (revision of #23)

Issue #23 said not to port or closely translate faer, gemm, nano-gemm or
private-gemm-x86 source. The maintainer revised this on 2026-09-30:

- **Porting is allowed.** Kernels from BLIS, OpenBLAS and faer may be ported,
  preserving the original copyright and license notices and recording
  upstream, version, path and relationship per `docs/provenance.md` ("Code
  port or close translation").
- **One crate per origin.** Licenses do not mix within a crate. faer is MIT
  but bundles MPL-2.0 (Eigen-derived) and BSD (LAPACK/SuiteSparse-derived)
  parts, so each ported faer file needs per-file provenance, and MPL-2.0
  code is not ported without a separate decision.
- **Call, don't reimplement.** Where private-gemm-x86 or gemm-common/gemm-*
  can be called directly, they are called (§3.1: `tprims-kernel-gemm`,
  `tprims-kernel-pgx86`).

This phase ports nothing; the rule is recorded so the crates are laid out
for it.

## 10. Testing

- **Registry**: every family passes validation. A snapshot of the
  `list_kernels()` ids is taken, excluding `available_on_this_cpu`.
- **Parity**: every built family × dtype × complex scheme available on the
  test CPU must match the `reference.rs` oracle within the dtype's
  tolerance. 3m and 4m have their own residual bounds. The cases:
  - edge shapes: m, n, k ∈ {1, MR−1, MR, MR+1, MC+1, …};
  - transposed and general-stride operands;
  - conj flags, and alpha/beta ∈ {0, 1, other};
  - D ≠ C and D = C;
  - block-scatter operands, for families with `scatter_pack`.

  Direct-C families are tested in both the Direct and the fallback regimes.
- **Contract branches**: the portable ScratchTile, Direct and direct-B
  variants and the `gemm-f64` Direct family, the last only when
  `kernel-gemm` is built.
- **Selection errors**: each case in §4.4 returns its specific error at
  plan creation. The "CPU lacks features" case uses a masked `CpuFeatures`.
- **Concurrency**: two plans with different families, executed concurrently
  on one pool and on two pools, give correct results and never share a team
  buffer.
- **Partition**: for widths {1, 2, 3, 4, 8}, results are bitwise equal to
  width 1; boundaries are MR/NR multiples; all threads see the same
  blocksizes.
- **Workspace**:
  - a direct-B plan's `WorkspaceReq` has B = 0, and a counting allocator
    sees no B-sized allocation;
  - steady-state executes allocate nothing;
  - per-worker buffers are allocated on their consuming worker (debug
    instrumentation records the allocating thread);
  - re-entrant calls work.
- **No env reads on execute**: changing a `TENSORCONTRACT_*`/`TPRIMS_*`
  variable after planning does not change the plan's result or
  `selected()`.
- **Performance non-regression** (acceptance), with the tprims-benchmark
  skill: ABBA paired runs at 1T/4T/8T of old main against this branch, on:
  - `tenferro-p1.json` (contraction, default selection: this exercises the
    TBLIS-style branch with the reworked driver);
  - `tenferro-p1-gemm.json` with `engine = Packed` forced, and batched with
    `Tblis` forced, so the reworked packed path is measured on GEMM shapes;
  - the same GEMM corpus at default selection (Faer), as a control.

  Gate: calls-weighted workload time within session-to-session noise (the
  `tblis_decision.py` noise rule), and no group slower by more than 5%.
  Selection and dispatch overhead is measured separately on the smallest
  corpus shapes. `PrivateGemmX86` against `Faer` is recorded, not gated: it
  is an optimization-session input.

## 11. Delivery

One PR on tprims-rs, in reviewable commits:
1. the crate split: `git mv` of the contract files into `tprims-gemm-kernel`
   and the microkernels into `tprims-kernel-tensorcontract`, with re-exports
   so `tensorcontract`'s API and all tests are unchanged (a pure move, so it
   is reviewable on its own);
2. the contract (`KernelFamily<T>`), the registry, and the portable
   family with its three variants;
3. the tensorcontract families, and the driver on `ResolvedGemm`
   (typed-plan path plus the `OnceLock` shim for `Plan::run`);
4. 1m/4m induced-family generation, and the four-plane tile;
5. the partition policy (StaticGrid, effective-width blocking fix,
   DynamicTiles designed only);
6. the workspace arena through `Spmd`;
7. `tprims-kernel-gemm`, and `tprims-kernel-pgx86`;
8. tprims-blas / tprims-contract selection and query;
9. docs.

Merge after green CI and the non-regression benchmark, as for earlier phases;
then stop. Optimization starts in a separate session.

## Appendix A. Review of v1 and resolutions

| # | Finding | Resolution |
| --- | --- | --- |
| C1 | Crate layout contradicted itself: generic packers claimed for a new crate below `tensorcontract` while also "staying in place" | v2 kept the contract in `tensorcontract`; the maintainer chose the split instead (v3): the contract files move into `tprims-gemm-kernel` and the microkernels into `tprims-kernel-tensorcontract`, with authorship and history preserved (§3.1) |
| C2 | `CUpdate::Direct` ignored distinct C/D and conj_c/conj_d | Plan-level guard (aliasing or beta = 0, no C/D conj, full constant-stride tile), else scratch tile (§4.2) |
| C3 | `ResolvedGemm` had no home in dtype-agnostic `Plan` | Typed tprims plans own `ResolvedGemm<T>`; `execute_with` takes it; `OnceLock` shim for `Plan::run` (§3) |
| C4 | Team B buffer could not be published from a worker TLS; barriers/scatter vectors unaccounted | Arena through `Spmd::workspace`, team buffers owned by the provider, try_borrow fallback, barriers and scatter in arena (§6.2) |
| I1 | 4m cannot produce a planar tile with overwrite calls | Four-plane tile + write-back recombination (§4.3) |
| I2 | Direct-C and direct-B had no exerciser; gemm-common not called | Portable family variants; `tprims-kernel-gemm` wraps gemm-f64/f32 microkernels (§3.1, §4.5) |
| I3 | DynamicTiles under-specified, over-scoped | Designed, not implemented; sync and A-packing choices recorded (§6.1) |
| I4 | Gate did not measure the changed path | Forced `Packed`/`Tblis` corpora added; Faer as control (§10) |
| I5 | `u8` pointers unnecessary | Typed `KernelFamily<T>` and `UkrFn<T>` (§4.1–4.2) |
| I6 | `KernelSet`/`Element` ripple understated | Listed; `Element` does not move (§7) |
| Minor | `c_pref` duplicate rule; spindle claim; `max` unused; `Id(&str)`; TLS release; `Selected` info; fallback paths; snapshot field | All adopted (§4.1, §4.4, §5, §6.2, §10) |
