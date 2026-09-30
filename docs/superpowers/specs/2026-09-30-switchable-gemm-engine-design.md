# Switchable GEMM engine: design

- Date: 2026-09-30
- Issue: #23 (this spec revises its provenance rule; see §9)
- Status: draft for review
- Scope: the **mechanism only**. Optimization, kernel ports and tuning are a
  separate, later session. This work stops when the mechanism is in place,
  tested, and shown not to slow down the current default path.

## 1. Goal

Make every choice inside tprims' GEMM path an explicit, queryable value that
is selected once and carried into execution:

- which engine runs a matrix GEMM (faer, private-gemm-x86 called directly, or
  tprims' own packed driver);
- inside the packed driver, which kernel family: microkernel, packers, C
  update, MR/NR, and blocksizes (MC/KC/NC);
- for complex types, the arithmetic method (native, 1m, 3m, 4m) and packed
  complex layout (interleaved, planar, 1e/1r);
- SIMD-optimized versus portable generic kernels;
- how the work is split across threads (partition policy);
- how packing buffers are owned and first touched (workspace policy).

The same mechanism must be able to host kernels ported later from BLIS,
OpenBLAS and faer, each in its own crate under its own license, and optimized
native complex kernels when they exist.

### Non-goals (this phase)

- No new optimized kernels, no ports of BLIS/OpenBLAS/faer kernel source.
- No tuning of tile shapes or blocksizes; current values are carried over.
- No C ABI for kernel selection. The Rust API is shaped so a C ABI can list,
  select and query later (§8), but that ABI is a follow-up.
- No NUMA-aware team pools or huge pages; the workspace policy leaves room for
  them (§6).

## 2. Background (research summary)

Sources: `docs/worklogs/2026-09-30-gemm-strategy-faer-vs-tensorcontract.md`
and two source studies made for this spec (BLIS `913242c`, BLIS 0.9.0
`14c86f6`, TBLIS `eb719e7` / 1.x `c4f81e0`; faer, gemm-common 0.19,
private-gemm-x86 0.1.20, nano-gemm 0.2; the in-tree `tensorcontract`).
Only structure was studied; no code is taken from them in this phase.

- **BLIS** splits state into two levels. An immutable *context* (`cntx_t`) per
  CPU configuration is chosen once at startup (CPUID, newest first, env
  override, `generic` fallback) and holds per-dtype kernel pointers, packers,
  row/column preference and {default, max} blocksizes. A per-call *control*
  object then fixes the pack schemas, scaled blocksizes and kernel pointer.
  The complex method is data: 1m changes only the pack schemas, the blocksize
  scales and which kernel pointer is called. BLIS removed 3m and 4m in 2021.
  Threading ("ways" of jc/ic/jr/ir) lives in a separate runtime object.
- **TBLIS 1.x** has a config struct with per-dtype function-pointer tables,
  registered with priorities and a check function; scatter support is extra
  pack kernels plus a C-update wrapper around an unchanged microkernel.
  TBLIS 2.x is a BLIS plugin and inherits BLIS's selection.
- **faer** (on this EPYC 7713P) routes f64/c64 matrix GEMM to
  private-gemm-x86: build-time generated assembly, a native interleaved c64
  6×4 kernel, B read directly (not packed) in most cases, A packed inside the
  kernel, dynamic 2-D job claiming with spin-then-sleep workers, persistent
  page-aligned thread-local buffers. gemm-common compiles the whole driver per
  ISA and caches one function pointer. nano-gemm's `Plan<T>` (copyable value:
  kernel table + driver pointer + mr/nr) is the closest analogue to the
  descriptor proposed here.
- **tensorcontract** already has most of a descriptor:
  `KernelConfig<R> = { Ukr<R>, Blocking }` (`kernel/mod.rs:291-465`) and
  per-ISA tables `IsaConfigs<R>` (`x86.rs:517-524`). Missing:
  - packers and write-back are generic functions with a per-element format
    `match` (`pack.rs:124-157`, `writeback.rs:66-93`), not pointers;
  - no ISA or method tag, no validation;
  - no native/interleaved or 4m variant;
  - `config_for_plan` rebuilds the config on every execute (`driver.rs:357`)
    and `Blocking::derive` reads `TENSORCONTRACT_KC_COUPLE` each time;
  - the `model` blocking sizes NC with `plan.threads()` (1 in tprims) instead
    of the effective Spmd width.
- **Threading today.** tensorcontract splits output M (whole MR panels) and N
  (whole NR slivers, re-cut per NC block) statically; K is serial; the batch
  axis is never parallel; two futex barriers per (batch, NC, KC) step; strip
  edges are not cache-line aligned, so C lines at strip edges can be
  write-shared. faer/private-gemm-x86 runs KC slabs serially and claims
  (2–4)·mr × 32-column jobs dynamically with no barrier inside a slab.
- **Buffers today.** tensorcontract allocates every packing buffer per call
  (64-B aligned) and frees it on return; the shared B panel is allocated by
  the caller but first touched by the packing workers; A is packed and
  consumed by the same worker but duplicated `pn` times and never reused; the
  serial fallback allocates B twice. BLIS keeps page-aligned team-owned pool
  buffers reused across calls; neither BLIS nor TBLIS handles NUMA.
- **This host**: 1 NUMA node, 8 L3 domains (CCDs) of 8 cores. First-touch
  matters for L3/CCD locality here and for DRAM locality on multi-node hosts.

## 3. Architecture

Two selection levels, as in BLIS, and two orthogonal runtime policies:

```
GemmEngine (matrix GEMM route)             selected per plan (default: once/process)
 ├─ Faer              faer::linalg::matmul       (current default for matrix GEMM)
 ├─ PrivateGemmX86    private_gemm_x86::gemm     (called directly, no reimplementation)
 └─ Packed(&KernelFamily)  tprims packed driver  (current TBLIS-style path)
        KernelFamily = immutable descriptor, &'static, from a registry
          ├─ microkernel fn ptr + MR/NR + C preference
          ├─ packers (A, B) fn ptrs per packed layout
          ├─ C update: scratch tile + write-back fn ptr, or direct
          ├─ complex: method, layouts
          └─ blocksizes {default, max} for MC/KC/NC
   + PartitionPolicy   (how the index space is split across threads)
   + WorkspacePolicy   (buffer sizes, ownership, first touch, reuse)
```

- A **plan** (`tprims-blas` GEMM plan, `tprims-contract` plan) resolves all of
  these once at plan creation into a `ResolvedGemm` value: engine, family,
  concrete scaled blocksizes for the effective thread width, partition and
  workspace requirement. Execution reads only that value; it never consults
  environment variables or rebuilds configs.
- `Exec` stays responsible for threads and pools. Nothing in the kernel layer
  owns threads.

### 3.1 Crates

| Crate | Role | License / authorship |
| --- | --- | --- |
| `tprims-gemm-kernel` (new) | The contract: enums, `KernelFamily`, function-pointer types, the registry and selection, CPU feature detection, workspace requirement types, and the portable reference family (§4.5). No threading, no `Exec`, no dependency on any other tprims crate. | MIT OR Apache-2.0, project-owned |
| `tensorcontract` (existing, in `tensorprimitives/`) | The packed driver and its existing kernels. Its kernel tables become `KernelFamily` registrations; its driver consumes a family instead of building a `KernelConfig`. Code stays in place so history and authorship are kept. | MIT OR Apache-2.0, Lukas Devos; changes carry `Co-authored-by: Lukas Devos` where they rework his code |
| `tprims-kernel-pgx86` (new, optional) | Adapter that calls `private_gemm_x86::gemm` as the `PrivateGemmX86` engine. Adapter code only; no source copied. | Adapter MIT OR Apache-2.0; depends on private-gemm-x86 (MIT, Sarah Quiñones) |
| `tprims-kernel-blis`, `tprims-kernel-openblas`, `tprims-kernel-faer` (future) | Ported kernel families, one crate per origin so licenses never mix in one crate. Not created in this phase. | BSD-3-Clause (BLIS, OpenBLAS), MIT (faer) with original notices |
| `tprims-blas`, `tprims-contract` | Plans that resolve engine + family + policies; expose selection and query. | unchanged |

Dependency direction: `tprims-gemm-kernel` ← kernel crates ←
`tensorcontract` ← `tprims-blas` ← `tprims-contract`. Kernel crates depend
only on `tprims-gemm-kernel` (and their upstream crate, if any). Which kernel
crates are built is chosen with cargo features on `tprims-blas`
(`kernel-pgx86`, later `kernel-blis`, …); the portable reference family and
the tensorcontract families are always built.

## 4. Kernel families

### 4.1 Descriptor

`KernelFamily` is a plain `'static` struct (no trait objects, no macros for
registration) built by `const` or `fn` constructors in each kernel crate:

- `id: &'static str` — stable identifier, e.g. `tc.avx2.f64.8x6`,
  `tc.avx512.c64.planar.8x6`, `ref.c64.native.4x4`. Stable across releases;
  used for selection, diagnostics and benchmarks.
- `origin: Origin` — `Tensorcontract`, `Reference`, later `Blis`, `OpenBlas`,
  `Faer`; with the crate name and license, for `list_kernels()` output.
- `dtype: Dtype` — the element type the family computes (`F32`, `F64`, `C32`,
  `C64`).
- `isa: Isa` and `required: CpuFeatures` — checked at selection; a family is
  never selectable on a CPU that lacks `required`.
- `imp: KernelImpl` — `Reference`, `Optimized`, or `Induced` (a complex
  family built from a real one, §4.3).
- `complex: Option<ComplexScheme>` — `{ method, a_layout, b_layout,
  tile_layout }` for complex families; `None` for real ones.
- `mr, nr` — the logical tile; `pack_mr, pack_nr` — the packed panel leading
  dimensions (may exceed mr/nr for alignment).
- `c_pref: CPref` — `Row`, `Col`, or `Any`; the planner may transpose the
  problem to match, as BLIS does.
- `ukr: UkrFn` — the microkernel pointer (§4.2).
- `pack_a: PackFn`, `pack_b: PackFn` — packer pointers for the family's
  layouts. Families that use the standard layouts point at the generic
  packers in `tprims-gemm-kernel`; families with their own format
  (OpenBLAS-style copy routines) supply their own.
- `b_access: BAccess` — `Packed`, or `Direct { needs_unit_stride: Axis }` for
  faer-style families that read B in place (§6.2).
- `c_update: CUpdate` — `ScratchTile { write_back: WriteBackFn }` (current
  tensorcontract contract) or `Direct` (BLIS-style: the kernel applies alpha,
  beta and stores into C with general strides).
- `blocks: Blocksizes` — `{default, max}` for MC, KC, NC, and the multiples
  they must respect (MC % MR == 0, NC % NR == 0, KC % kr == 0).
- `caps: Caps` — capability flags: `scatter_pack` (packers accept
  block-scatter sources, needed by direct tensor contraction), `conj_a`,
  `conj_b` (conjugation during packing), `beta_nonzero_direct`.

The registry validates every descriptor when it is built (in a unit test
that iterates all registered families, and in debug builds at first use):
MR/NR non-zero and even where 1m is allowed; blocksize multiples; packed
leading dimensions ≥ mr/nr; the scratch tile fits the declared bound; layouts
consistent with `complex.method` (§4.3 table).

### 4.2 Microkernel contract

One signature for all packed-driver families:

```rust
pub type UkrFn = unsafe fn(k: usize, a: *const u8, b: *const u8,
                           c: *mut u8, rs_c: isize, cs_c: isize,
                           alpha: *const u8, beta: *const u8,
                           aux: &UkrAux);
```

- `a` and `b` are packed micropanels in the family's layouts (`b` may be a
  direct pointer into B when `b_access` is `Direct`; strides then come in
  `aux`).
- With `CUpdate::ScratchTile`, the driver passes the plan-owned scratch tile
  as `c` with the tile layout, `alpha = 1`, `beta = 0`, and the kernel
  overwrites it; edges are handled by zero-filled packing; the write-back
  function applies alpha, beta, conjugation of C/D and scatter. This is the
  current tensorcontract behaviour and stays the default.
- With `CUpdate::Direct`, the kernel receives C with general strides, alpha
  and beta, and handles full tiles only. The driver still routes edge tiles
  and scatter C through a scratch tile and the generic write-back, as TBLIS
  does. This is how BLIS/OpenBLAS kernels will plug in unchanged.
- `UkrAux` carries `a_next`/`b_next` (prefetch), the B strides for direct-B,
  imaginary strides for planar layouts, and `inner: Option<&'static
  KernelFamily>` so induced families reach their real kernel.

Pointers are `u8`-typed only at this boundary; each kernel crate exposes a
typed safe wrapper for tests, and the registry check guarantees the dtype
matches before any call.

### 4.3 Complex schemes

External storage is always interleaved (`Complex<T>`). The packed layout and
the arithmetic are chosen per family:

| method | a / b layout | tile | kernel | status |
| --- | --- | --- | --- | --- |
| `Native` | interleaved / interleaved | interleaved | native complex ukr | new: reference family (§4.5); optimized native kernels later |
| `Native` | planar / planar | planar | complex ukr on split planes | existing tensorcontract "planar" |
| `OneM` | 1e / 1r (swapped by `c_pref`) | real 2MR×NR or MR×2NR | **real** ukr | existing tensorcontract "1m" |
| `ThreeM` | 3-plane | 3-plane | fused 3m ukr | existing tensorcontract "3m"; accuracy trade-off, never chosen by `Auto` unless allowed |
| `FourM` | planar / planar | planar | **real** ukr, 4 calls per tile | new: induced family over any real family |

The tensorcontract enum `ComplexMethod::{Planar, OneM, ThreeM}` becomes
`(Native, planar)`, `OneM`, `ThreeM`; the new enums in `tprims-gemm-kernel`
are `#[non_exhaustive]` so later additions are not breaking. The existing
`TENSORCONTRACT_COMPLEX` values keep their meaning.

Induced families (`OneM`, `FourM`) are generated from a real family by a
function in `tprims-gemm-kernel`, as BLIS derives 1m from the real context:
blocksize scaling (KC/2 for 1m, MR or NR ×2 per `c_pref`), pack function
choice and the wrapper ukr come from one table, so a newly ported real kernel
gets 1m and 4m without further code.

### 4.4 Selection

- `Registry::families(dtype) -> &[&'static KernelFamily]` lists families that
  were built. `list_kernels()` returns `(id, origin, license, isa, dtype,
  complex, mr, nr, available_on_this_cpu)` for diagnostics.
- `KernelChoice::{Auto, Id(&str)}` and `EngineChoice::{Auto, Faer,
  PrivateGemmX86, Packed(KernelChoice)}` are plan-config fields.
- **Process default**: resolved once per dtype through `OnceLock` from CPU
  features, with an optional env override read once
  (`TPRIMS_GEMM_ENGINE`, `TPRIMS_GEMM_KERNEL`; existing `TENSORCONTRACT_*`
  variables are also read once, into the default). `Auto` picks the
  highest-priority family available on this CPU and never fails: the
  portable reference family is always last.
- **Plan override**: an explicit choice on a plan wins over the process
  default. Two plans in one process can use different families concurrently;
  nothing mutable is global.
- **Errors, at plan creation** (never at execute): unknown id; family not
  built (feature off: the error names the feature); CPU lacks required
  features; dtype mismatch; operand incompatible with the family (e.g.
  direct-B without the needed unit stride, scatter operands without
  `scatter_pack`); engine cannot express the operation (e.g.
  `PrivateGemmX86` for block-scatter tensor contraction, or on non-x86). No
  silent substitution: a forced choice either runs as chosen or fails.
- **Query**: `plan.selected() -> SelectedGemm { engine, family_id,
  complex, mr, nr, mc, kc, nc, partition, workspace }`, printable and
  serializable to JSON for benchmark manifests.

`Selected` (the enum `gemm_grouped` and friends return today) is extended
into, or replaced by, `SelectedGemm`.

### 4.5 Portable reference family

Project-owned, in `tprims-gemm-kernel`: `fn ref_ukr<T, const MR: usize,
const NR: usize>` for real and native-interleaved complex, with generic
packers and write-back. Purpose: the always-available fallback, the
correctness oracle for every other family, and the first native interleaved
complex family (so the interleaved layout path is exercised before any
optimized native kernel exists). Written with plain loops for LLVM to
vectorize; no intrinsics.

## 5. Engines

- `Faer`: the current `tprims-blas` matrix GEMM path, unchanged.
- `PrivateGemmX86`: `tprims-kernel-pgx86` calls
  `private_gemm_x86::gemm(dtype, itype, instr, …, n_threads)` directly for
  dense strided matrix GEMM (f32/f64/c32/c64), selecting `InstrSet::Avx256`
  or `Avx512` from CPU features. Its threading goes through spindle, which
  uses `rayon::current_num_threads()` of the current pool; the adapter runs
  it inside `Exec`'s pool install with `n_threads` = the plan's width, so
  `Exec` still owns the threads. It is not usable for block-scatter tensor
  packing (no scatter packers): `tprims-contract` can use it only through
  permute+GEMM. Its lower-level `call_microkernel` / millikernel API is
  public but undocumented and uses a custom register ABI; wrapping it as a
  packed-driver family is recorded as a later candidate, not done here.
- `Packed(family)`: the tensorcontract driver with the chosen family.

Engine choice for tprims-contract: `Strategy::Auto` keeps its copy-aware rule
(TBLIS-style when permute+GEMM would copy); the permute+GEMM branch uses the
plan's matrix engine; the TBLIS-style branch always uses `Packed` with the
plan's family.

## 6. Threading and buffers

### 6.1 Partition policy

A `PartitionPolicy` value, resolved per plan from the family's blocksizes and
the effective thread width (not `plan.threads()`; this fixes the current
`model` blocking bug):

- `StaticGrid { pm, pn }` — current tensorcontract behaviour (whole MR
  panels × whole NR slivers, K serial, two barriers per KC step).
- `DynamicTiles { job_m, job_n }` — faer-style: KC slabs serial; within a
  slab, jobs of `job_m`×`job_n` (multiples of MR/NR) claimed from an atomic
  counter; one barrier per slab (after packing B), none between tiles.

Invariants, for all policies:

- K is never split, so results are bitwise identical to serial for every
  policy and width (tested).
- All threads of one execute share the same MC/KC/NC (they come from the
  resolved plan, not per thread).
- Split boundaries in M and N are multiples of MR and NR; the last part takes
  the ragged tail.
- The index space includes the batch axis, so a policy may split batch items
  across threads; the existing static policy keeps batch inside each thread.
  (Splitting batch is a later optimization; the interface allows it now.)
- Optional `align_c_lines`: round M-split boundaries up so C strips do not
  share cache lines when C is column-major (and N for row-major). Off by
  default in this phase; present so the optimization session can measure it.

### 6.2 Workspace policy

At plan creation the resolved plan computes a `WorkspaceReq`:

- per worker: packed A block (MC×KC in the A layout), scratch tile(s), and
  scatter vectors for its ranges;
- per team: packed B panel (KC×NC in the B layout);
- each entry may be zero. **When the family reads B directly
  (`BAccess::Direct`), or the plan otherwise decides not to pack B, the B
  requirement is zero and no B buffer is allocated.** The decision is per
  plan (from the family and the actual operand strides), not per family,
  because a direct-B family may still need to pack B for unfriendly strides.

Buffers come from a `WorkspaceArena`:

- **Per-worker buffers** live in thread-local storage of the worker that uses
  them, keyed by that worker (stable within an `Exec` pool). They are
  allocated and first touched by that worker, grow only, are page-aligned
  (4096 B), and are reused across calls. A requirement of zero never
  allocates.
- **Team buffers (B)**: the team's first worker allocates (or reuses) the
  buffer; each packing worker first touches only the slice it packs, before
  the barrier. Reused across calls on the same pool. On a single NUMA node
  this keeps pages hot in the right L3s; per-node teams for multi-node hosts
  are future work that this layout allows.
- **Re-entrant and serial fallback paths** (a call from inside a pool
  worker, width 1) use the calling thread's own per-worker buffers; the
  serial fallback no longer allocates B twice.
- Allocation happens outside the hot loop; after the first call of a given
  size a steady-state execute performs no heap allocation for packing
  buffers (tested with a counting allocator in a test binary).

Block-scatter vectors keep being generated per block but into the arena, and
are filled cooperatively before the barrier rather than by the caller.

## 7. Changes to existing code

- `tensorcontract`:
  - `KernelConfig`/`IsaConfigs` become `KernelFamily` registrations (their
    current menus of MR×NR shapes become separate families with ids);
    `KernelSet` is kept as a thin compatibility trait over the registry until
    `tprims-blas` no longer uses it, then removed.
  - `config_for_plan` is removed from the execute path; the driver takes a
    `&ResolvedGemm`.
  - Packing and write-back are called through the family's pointers; the
    per-element format `match` moves into per-layout monomorphized functions.
    The orientation swap swaps `pack_a`/`pack_b` too.
  - Env reads (`Blocking::derive`, `TENSORCONTRACT_*`) happen once, into the
    process default.
  - The test `x86_isas_agree_on_the_pack_contract` becomes "families with the
    same complex scheme agree on the pack contract".
- `tprims-blas`: plans carry `EngineChoice`; `Scalar::Re: KernelSet` bound is
  replaced by the registry lookup; `Selected` → `SelectedGemm`.
- `tprims-contract`: `Config` gains `engine` and `kernel` fields; plan
  exposes `selected()`.
- Docs updated in the same PR (docs-vs-code check): `README.md` crate table,
  `docs/architecture.md` (the planned `tprims-gemm-kernel` becomes real),
  `docs/provenance.md` (per-origin crate policy, private-gemm-x86 as a
  called dependency), `docs/decision-log.md`.

## 8. Future C ABI (shape only)

`tprims_gemm_kernels_list` (ids + attributes), a `kernel_id` / `engine`
string field on the blas and contract plan configs, and
`tprims_*_plan_selected` returning the JSON of `SelectedGemm`. Not
implemented in this phase.

## 9. Provenance rule (revision of #23)

Issue #23 said not to port or closely translate faer, gemm, nano-gemm or
private-gemm-x86 source. The maintainer revised this on 2026-09-30:

- Porting kernels from BLIS, OpenBLAS and faer is allowed, preserving the
  original copyright and license notices, recording upstream, version, path
  and relationship per `docs/provenance.md` §"Code port or close
  translation".
- Each origin gets its own kernel crate so licenses do not mix within a
  crate. faer is MIT but bundles MPL-2.0 (Eigen-derived) and BSD
  (LAPACK/SuiteSparse-derived) parts: each ported faer file needs per-file
  provenance, and MPL-2.0-derived code is not ported without a separate
  decision.
- Where private-gemm-x86 or gemm-common can be **called** directly, it is
  called, not reimplemented or ported.

This phase ports nothing; the rule is recorded now so the kernel crates are
laid out for it.

## 10. Testing

- **Registry**: every registered family passes descriptor validation; ids are
  unique and stable (snapshot test of `list_kernels()` ids).
- **Parity**: for every built family × dtype × complex scheme available on
  the test CPU, GEMM and direct contraction match the reference family
  within the dtype's tolerance on edge shapes (m, n, k ∈ {1, MR−1, MR, MR+1,
  MC+1, …}), transposed and general-stride operands, conj flags, alpha/beta
  ∈ {0, 1, other}, and block-scatter operands for families with
  `scatter_pack`. 3m uses its own residual bound.
- **Selection errors**: unknown id, feature not built, CPU unsupported
  (simulated by a masked `CpuFeatures`), dtype mismatch, incompatible
  operand, engine unable to express the op — each returns its specific error
  at plan creation.
- **Concurrency**: two plans with different families executed concurrently
  from different threads produce correct results.
- **Partition**: for each policy and widths {1, 2, 3, 4, 8}, results are
  bitwise equal to width 1; boundaries are MR/NR multiples; all threads see
  the same blocksizes.
- **Workspace**: direct-B plans allocate no B buffer; steady-state executes
  allocate nothing (counting allocator); per-worker buffers are allocated on
  the worker that uses them (debug instrumentation records the allocating
  thread).
- **No env reads on execute**: a test that counts env reads (or sets a
  changed env var after planning and checks the plan is unaffected).
- **Performance non-regression** (acceptance of this phase): with the
  tprims-benchmark skill, ABBA paired runs of the P1 corpora
  (`tenferro-p1.json`, `tenferro-p1-gemm.json`) at 1T/4T/8T, old main vs
  this branch with default selection. Gate: calls-weighted workload time
  within session-to-session noise (the `tblis_decision.py` noise rule), and
  no group slower by more than 5%. Selection overhead per execute is
  measured separately on the smallest corpus shapes.

## 11. Delivery

One PR on tprims-rs, in reviewable commits: (1) `tprims-gemm-kernel`
contract + reference family; (2) tensorcontract families + driver on
`ResolvedGemm`; (3) induced 1m/4m generation; (4) partition policy; (5)
workspace arena; (6) `tprims-kernel-pgx86` engine; (7) tprims-blas /
tprims-contract selection and query; (8) docs. Merge after green CI and the
non-regression benchmark, as for earlier phases. Then stop; optimization
starts in a separate session.
