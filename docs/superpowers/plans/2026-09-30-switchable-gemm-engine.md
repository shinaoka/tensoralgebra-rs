# Switchable GEMM Engine Implementation Plan

> **Execution:** proceed task-by-task in the main session, with tests first
> and a final integrated self-review. Delegate only on explicit request.
> The original superpowers skills are not installed here. Steps use
> checkbox (`- [ ]`) syntax for tracking.
>
> **2026-09-30 ownership correction:** Task 8 uses pool-owned team storage
> and owner-keyed worker TLS, not a process-global arena.

**Goal:** Turn every choice in tprims' GEMM path into an immutable, queryable
value: engine, kernel family, complex method and layout, partition policy,
and workspace. Select it once, carry it into execution, and prove the current
paths did not get slower.

**Architecture:** The kernel layer is split out of `tensorcontract`:
- `tprims-gemm-kernel` holds the contract, the packers and write-back, the
  registry, and the portable family;
- `tprims-kernel-tensorcontract` holds Lukas Devos's microkernels;
- `tprims-kernel-gemm` and `tprims-kernel-pgx86` hold adapters that call
  upstream crates.

The `tensorcontract` driver executes a `ResolvedGemm<R>` resolved once per
plan. `tprims-blas` and `tprims-contract` resolve and expose it.

**Tech Stack:** Rust 1.89 (workspace MSRV), cargo workspace, rayon via
`tprims-exec`, `gemm-f64`/`gemm-f32`/`gemm-common` 0.19.0,
`private-gemm-x86` 0.1.20, `num-complex`.

**Spec:** `docs/superpowers/specs/2026-09-30-switchable-gemm-engine-design.md`
(v3, approved 2026-09-30). The spec is the binding authority; this plan
argues from it.

## Global Constraints

- Scope is the mechanism only. No new optimized kernels, no ports of
  BLIS/OpenBLAS/faer source, and no retuning of tiles or blocksizes; current
  values carry over (spec §1 Non-goals).
- The kernel layer owns no threads; `Exec`/`Pool` owns threads (spec §3).
- Execution never reads environment variables and never rebuilds configs;
  everything is resolved at plan creation (spec §3).
- A forced choice runs as chosen or fails at plan creation with a specific
  error. There is no silent substitution; the per-tile Direct→ScratchTile
  fallback is part of the family contract (spec §4.4).
- K is never split, so results are bitwise identical to width 1 for every
  width (spec §6.1).
- B not packed ⇒ no B buffer is allocated (spec §6.2).
- Per-worker buffers are allocated and first touched by the consuming worker.
  Team buffers are first touched slice by slice by their packing workers
  (spec §6.2).
- Lukas Devos's files move with `git mv`: headers and authorship are kept,
  `authors` lists him, and commits that rework his code carry
  `Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>` (spec §3.1).
- Adapters call `gemm-*` and private-gemm-x86; nothing from them is copied
  (spec §9).
- Builds: set `CARGO_BUILD_JOBS=16` on this host (AGENTS.md); do not
  hardcode `-j` in scripts. While a benchmark measures on CCD 0, run builds
  with `taskset -c 16-63`.
- Local gate before the PR (AGENTS.md):
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo build -p tprims-bundle`
  - `cargo test --workspace`
  - `cargo test -p tprims-exec`
  - `cargo test -p tensorcontract --release`
  - `python3 scripts/check-agent-skills.py`
  - the README/docs-vs-code check.
- Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- One PR on branch `gemm-engine-spec` (it already carries the spec, this plan and HANDOFF.md) for the whole plan. It merges after green CI
  and the non-regression benchmark (Task 12). Then stop: optimization is a
  separate session.

## Review Focus

1. **Orientation swap with asymmetric families.** When
   `plan.transposes_gemm(mr)` swaps A and B, the family's `pack_a`/`pack_b`,
   `b_access` and the 1m 1e/1r roles must swap consistently. Otherwise the
   swapped problem packs B with A's layout, or tries to read "direct B" from
   the original A. Tests: `swapped_orientation_matches_oracle_for_every_family`
   (Task 4) and `direct_b_disabled_when_orientation_swaps_to_strided_b`
   (Task 5).
2. **Direct C update with D ≠ C, `conj_d`, or beta ≠ 0.** Direct must fall
   back to the scratch tile on each of these. A wrong guard silently writes
   `alpha·AB + beta·C` into C instead of D. Test:
   `direct_c_guard_matrix` (Task 5).
3. **Concurrent executes sharing a team buffer.** Two plans on one pool, or
   one plan executed from two threads, must never receive the same team B
   buffer. Test: `concurrent_executes_never_share_team_buffers` (Task 8).
4. **Re-entrant execute inside a pool worker.** A borrowed per-worker TLS
   buffer must not be taken twice. Test:
   `reentrant_execute_uses_fresh_buffers` (Task 8).
5. **Env vars changed after planning.** A plan must keep its selection and
   blocking. Test: `env_change_after_planning_has_no_effect` (Task 4), run in
   its own test binary because it mutates process env.

---

## File Structure

New crates (under `crates/`, per the AGENTS.md layout):

```
crates/tprims-gemm-kernel/
  Cargo.toml                 authors: Lukas Devos + tprims contributors; MIT OR Apache-2.0
  LICENSE-MIT, LICENSE-APACHE  copied from tensorprimitives/crates/tensorcontract
  src/lib.rs                 module list, re-exports, env_once! macro (moved)
  src/element.rs             moved (Element, Real, C32, C64)
  src/scatter.rs             moved
  src/pack.rs                moved; + per-layout monomorphized packers, Interleaved layout
  src/writeback.rs           moved; + FourM and Interleaved tile formats, per-format write-back fns
  src/cache.rs               moved from kernel/cache.rs
  src/types.rs               moved type half of kernel/mod.rs: ParseError, ComplexMethod,
                             PackFormat, TileFormat, Ukr, Blocking, KernelConfig, env blocking
  src/cpu.rs                 new: CpuFeatures, Isa, detect()
  src/family.rs              new: KernelFamily<R>, enums, UkrFn/PackFn/WriteBackFn, validate()
  src/registry.rs            new: Registry, KernelInfo, list_kernels, select, SelectError
  src/induced.rs             new: one_m(), four_m() generation from a real family
  src/partition.rs           new: PartitionPolicy, split helpers, invariants
  src/workspace.rs           new: WorkspaceReq, WorkspaceProvider, ArenaProvider, TLS buffers
  src/resolved.rs            new: ResolvedGemm<R>, resolve()
  src/portable.rs            new: portable family (ScratchTile, Direct, direct-B variants)
crates/tprims-kernel-tensorcontract/
  Cargo.toml                 authors: Lukas Devos; MIT OR Apache-2.0
  LICENSE-MIT, LICENSE-APACHE
  src/lib.rs                 families::<R>() registration, KernelSet shim, kernel_force
  src/x86.rs, aarch64.rs, simd.rs, scalar.rs   moved from tensorcontract/src/kernel/
crates/tprims-kernel-gemm/
  Cargo.toml                 MIT OR Apache-2.0; deps gemm-f64, gemm-f32, gemm-common = "=0.19.0"
  src/lib.rs                 Direct families wrapping gemm-f64/f32 UKR tables
crates/tprims-kernel-pgx86/
  Cargo.toml                 MIT OR Apache-2.0; dep private-gemm-x86 = "=0.1.20"
  src/lib.rs                 PrivateGemmX86 engine adapter
```

Modified:

- `tensorprimitives/crates/tensorcontract/src/lib.rs`: re-exports, and
  `env_once!` imported from `tprims-gemm-kernel`.
- `tensorprimitives/crates/tensorcontract/src/kernel/mod.rs`: reduced to
  plan glue (`config_for_plan` → `resolve_for_plan`, `plan_config`,
  `selected_config`, `selected_kernel_name`).
- `tensorprimitives/crates/tensorcontract/src/driver.rs`: consumes
  `ResolvedGemm`; Direct/direct-B paths; workspace; partition.
- `tensorprimitives/crates/tensorcontract/src/spmd.rs`:
  `Spmd::workspace()`.
- `tensorprimitives/crates/tensorcontract/src/plan.rs`: `Plan::resolved::<T>()`
  cache, and `env_threads` read once.
- `crates/tprims-blas/src/{lib.rs, scalar.rs, gemm.rs, batched.rs,
  grouped.rs, tblis.rs}`, plus new `engine.rs` and `select.rs`.
- `crates/tprims-contract/src/{plan.rs, tblis.rs, permute_gemm.rs, lib.rs}`.
- `crates/tprims-linalg`: `Element` imports stay (`tensorcontract`
  re-exports it); verify only.
- `benchmarks/benchmarks/tprims/blas/blas.rs`: `--engine` option.
- `Cargo.toml` (workspace members and deps), `README.md`,
  `docs/architecture.md`, `docs/provenance.md`, `docs/decision-log.md`.

### Naming ruling carried from the spec

The spec writes `KernelFamily<T>`. Kernels in this codebase operate on the
**real** scalar `R` (`Ukr<T>` is instantiated with `T::Real`, e.g.
`kernel/mod.rs` `Ukr<T>` and `driver.rs` `Ctx.ukr: Ukr<T::Real>`).
Therefore:
- the descriptor is `KernelFamily<R: Real>`, whose `complex:
  Option<ComplexScheme>` says whether it serves `R` or `Complex<R>`;
- `families::<T: Element>()` filters by `T::IS_COMPLEX`;
- `ResolvedGemm<R>` is keyed the same way.

This is a representation detail, not a spec change.

---

### Task 1: Split the kernel layer into crates (pure move)

**Files:**
- Create: `crates/tprims-gemm-kernel/{Cargo.toml,LICENSE-MIT,LICENSE-APACHE,src/lib.rs,src/types.rs}`
- Create: `crates/tprims-kernel-tensorcontract/{Cargo.toml,LICENSE-MIT,LICENSE-APACHE,src/lib.rs}`
- Move (`git mv`): `tensorcontract/src/{element.rs,scatter.rs,pack.rs,writeback.rs}` → `crates/tprims-gemm-kernel/src/`; `tensorcontract/src/kernel/cache.rs` → `crates/tprims-gemm-kernel/src/cache.rs`; `tensorcontract/src/kernel/{x86.rs,aarch64.rs,simd.rs,scalar.rs}` → `crates/tprims-kernel-tensorcontract/src/`
- Modify: `tensorcontract/src/kernel/mod.rs`, `tensorcontract/src/lib.rs`, `tensorcontract/Cargo.toml`, `tensorcontract/src/{driver.rs,plan.rs,batch.rs,reference.rs,buffer.rs}` (imports only), root `Cargo.toml`

**Interfaces:**
- Produces:
  - `tprims_gemm_kernel::{Element, Real, C32, C64, ParseError, ComplexMethod, PackFormat, TileFormat, Ukr, Blocking, KernelConfig, IRREGULAR}`;
  - the modules `tprims_gemm_kernel::{scatter, cache}`;
  - `tprims_gemm_kernel::pack::{panel_len, pack_panel}`;
  - `tprims_gemm_kernel::writeback::{writeback, scale_only}`;
  - the `tprims_gemm_kernel::env_once!` macro, `#[macro_export]`, `#[doc(hidden)]`;
  - `tprims_gemm_kernel::env_threads() -> usize`, moved from `plan.rs`, `TENSORCONTRACT_THREADS`;
  - `tprims_kernel_tensorcontract::{KernelSet, KernelForce, kernel_force, scalar, x86, aarch64}`.
- The previously `pub(crate)` items that crossed the new boundary become
  `pub` with `#[doc(hidden)]`: `pack_panel`, `panel_len`, `writeback`,
  `scale_only`, `KernelConfig::{normalise, normalise_for, retarget_threads}`,
  `Blocking::derive_at_depth`, `ParseError::new`, `KernelForce`,
  `kernel_force`.
- `tensorcontract` public API unchanged: `tensorcontract::{Element, Real,
  C32, C64, Blocking, ComplexMethod, KernelSet}`,
  `tensorcontract::element`, `tensorcontract::scatter` and
  `tensorcontract::kernel::{cache, scalar, x86, aarch64, selected_config,
  selected_kernel_name, plan_config}` resolve as before via `pub use`.

- [x] **Step 1: Create the branch and record the baseline**

```bash
cd ~/tensor4all/tprims-rs && git fetch && git checkout gemm-engine-spec && git pull
git rebase origin/main   # spec/plan/HANDOFF branch; implementation continues on it (one PR)
export CARGO_BUILD_JOBS=16
cargo test -p tensorcontract --release 2>&1 | tail -3 > /tmp/claude-2000/-home-shinaoka-tensor4all/9e93717e-bbdb-4bfe-bcc4-cb6d49651a25/scratchpad/baseline-tc.txt
cargo test --workspace 2>&1 | grep -E "^test result" | awk '{s+=$4} END {print s " passed"}'
```
Expected: all green; note the pass count. It must be identical after
Step 7, because a pure move adds and removes no tests.

- [x] **Step 2: Create the crates and move files**

```bash
mkdir -p crates/tprims-gemm-kernel/src crates/tprims-kernel-tensorcontract/src
TC=tensorprimitives/crates/tensorcontract
for f in element scatter pack writeback; do git mv $TC/src/$f.rs crates/tprims-gemm-kernel/src/$f.rs; done
git mv $TC/src/kernel/cache.rs crates/tprims-gemm-kernel/src/cache.rs
for f in x86 aarch64 simd scalar; do git mv $TC/src/kernel/$f.rs crates/tprims-kernel-tensorcontract/src/$f.rs; done
cp $TC/../../LICENSE-MIT $TC/../../LICENSE-APACHE crates/tprims-gemm-kernel/ 2>/dev/null || cp tensorprimitives/LICENSE-* crates/tprims-gemm-kernel/
cp crates/tprims-gemm-kernel/LICENSE-* crates/tprims-kernel-tensorcontract/
git commit -m "Move kernel-layer files out of tensorcontract (git mv only)

Pure rename commit so history follows the files; the next commit makes
them compile.

Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>
Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
Keeping the rename in its own commit makes `git log --follow` reliable.
Expected: the commit contains renames only (`git show --stat HEAD` lists
`rename`).

- [x] **Step 3: Write `crates/tprims-gemm-kernel/Cargo.toml`**

```toml
[package]
name = "tprims-gemm-kernel"
description = "GEMM kernel contract for tprims: packed formats, packers, write-back, kernel-family registry"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license = "MIT OR Apache-2.0"
authors = ["Lukas Devos <ldevos@flatironinstitute.org>", "tensor4all contributors"]
publish = false

[dependencies]
num-complex.workspace = true

[features]
default = ["std"]
std = []

[lints]
workspace = true
```
Check whether `tensorcontract/Cargo.toml` has a `[lints]` table. If it does
not, omit `[lints]` here too, so moved code is not newly linted into
failure. Then write `crates/tprims-kernel-tensorcontract/Cargo.toml` the
same way, with `name = "tprims-kernel-tensorcontract"`, `authors = ["Lukas
Devos <ldevos@flatironinstitute.org>"]`, and dependencies `num-complex` and
`tprims-gemm-kernel = { workspace = true }`.

Root `Cargo.toml`: add both paths to `members` and to
`[workspace.dependencies]`:

```toml
tprims-gemm-kernel = { version = "0.1.0", path = "crates/tprims-gemm-kernel" }
tprims-kernel-tensorcontract = { version = "0.1.0", path = "crates/tprims-kernel-tensorcontract" }
```
In `tensorcontract/Cargo.toml` add `tprims-gemm-kernel = { workspace = true
}` and `tprims-kernel-tensorcontract = { workspace = true }`. Forward the
`std` feature: `std = ["tprims-gemm-kernel/std",
"tprims-kernel-tensorcontract/std"]`, and give
`tprims-kernel-tensorcontract` a `std = ["tprims-gemm-kernel/std"]` feature.

- [x] **Step 4: Split `kernel/mod.rs`**

Move these items, with their doc comments, verbatim into
`crates/tprims-gemm-kernel/src/types.rs`:
- `ParseError` and its impls;
- `ComplexMethod`, `PackFormat`, `TileFormat` and `Ukr`, with their impls;
- `Blocking` and `KernelConfig`, with their impls;
- `BlockingOverride`, `env_blocking` and `env_usize`.

`Blocking::model` references `cache::…`; write it as `crate::cache::…`.
`KernelConfig::normalise` calls `crate::plan::env_threads()`; it now calls
`crate::env_threads()`. Move `env_threads` from `tensorcontract/src/plan.rs`
into `tprims-gemm-kernel/src/lib.rs` verbatim, and make `plan.rs` call
`tprims_gemm_kernel::env_threads()`.

Move `KernelSet`, `KernelForce`, `kernel_force`, `force_scalar`,
`impl_kernel_set!` and its four invocations, and the whole `#[cfg(test)] mod
tests`, into `crates/tprims-kernel-tensorcontract/src/lib.rs`. They test the
kernels, so they belong with them. Add at the top of that file:

```rust
//! Lukas Devos's SIMD and scalar microkernels for the tprims GEMM kernel
//! contract (moved from `tensorcontract::kernel`, tensorprimitives-rs).
#![allow(clippy::missing_safety_doc)]
use tprims_gemm_kernel::{Blocking, ComplexMethod, KernelConfig, PackFormat, Real, TileFormat, Ukr};

pub mod scalar;
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[macro_use]
mod simd;
#[doc(hidden)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64 as simd_isa;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use x86 as simd_isa;
```
Only use the `#![allow]` if the moved code had equivalent crate-level
allows in `tensorcontract/src/lib.rs`; copy the exact crate attributes from
there.

In the moved kernel files, replace `use super::{…}` with `use crate::{…}`,
and `crate::element::Real` with `tprims_gemm_kernel::Real`. The `use
super::*` inside nested modules stays, since it refers to the parent file.
`super::kernel_force()` becomes `crate::kernel_force()`, and `use
super::KernelForce` becomes `use crate::KernelForce`.

`tensorcontract/src/kernel/mod.rs` keeps only:

```rust
//! Kernel selection glue between `Plan` and the kernel families.
pub use tprims_gemm_kernel::cache;
pub use tprims_gemm_kernel::{Blocking, ComplexMethod, KernelConfig, PackFormat, ParseError, TileFormat, Ukr};
pub use tprims_kernel_tensorcontract::{scalar, KernelSet};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[doc(hidden)]
pub use tprims_kernel_tensorcontract::x86;
#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub use tprims_kernel_tensorcontract::aarch64;
```
Below that, the existing `selected_config`, `selected_kernel_name`,
`config_for`, `config_for_plan` and `plan_config`, unchanged.

- [x] **Step 5: Rewire imports**

- In `tprims-gemm-kernel/src/lib.rs`:

  ```rust
  //! GEMM kernel contract for tprims. `element`, `scatter`, `pack`,
  //! `writeback`, `cache` and `types` were moved from tensorprimitives-rs
  //! `tensorcontract` (Lukas Devos); see docs/provenance.md.
  #[doc(hidden)]
  #[macro_export]
  macro_rules! env_once { /* body copied verbatim from tensorcontract/src/lib.rs:313-329 */ }
  pub mod cache;
  pub mod element;
  #[doc(hidden)] pub mod pack;
  pub mod scatter;
  mod types;
  #[doc(hidden)] pub mod writeback;
  pub use element::{Element, Real, C32, C64};
  pub use scatter::IRREGULAR;
  pub use types::*;
  /// `TENSORCONTRACT_THREADS`, read once. (Moved from tensorcontract `plan.rs`.)
  pub fn env_threads() -> usize { /* body moved verbatim */ }
  ```

  Copy the macro body; it must not be a placeholder. Inside
  `tprims-gemm-kernel`, the macro is invoked as `crate::env_once!` or
  `env_once!`. In the other crates it is `tprims_gemm_kernel::env_once!`.
  Because the macro expands to `std::sync::OnceLock` under
  `#[cfg(feature = "std")]`, and a `cfg` in an exported macro is evaluated
  in the calling crate, each calling crate needs its own `std` feature.
  `tensorcontract` and `tprims-kernel-tensorcontract` both have one after
  Step 3.
- In moved files: `crate::kernel::PackFormat` → `crate::PackFormat`, and
  `crate::kernel::TileFormat` → `crate::TileFormat`.
- Intra-doc links in moved files that point at `crate::Plan…`,
  `crate::buffer::Panel` or `crate::TensorViewMut` become plain code spans,
  e.g. `` `tensorcontract::Plan::row_block` ``. The link would otherwise
  need a dependency cycle.
- `tensorcontract/src/lib.rs`: delete the `env_once!` definition, and add
  `use tprims_gemm_kernel::env_once;` before the module list. Replace `mod
  pack; mod writeback; pub mod element; pub mod scatter;` with:

  ```rust
  pub use tprims_gemm_kernel::{element, scatter};
  use tprims_gemm_kernel::{pack, writeback};
  pub use tprims_gemm_kernel::{Element, Real, C32, C64};
  ```
- `driver.rs`, `plan.rs`, `batch.rs`, `reference.rs`, `buffer.rs`: `use
  crate::pack::…`, `crate::writeback::…` and `crate::element::…` keep
  working through these `use` aliases. Fix any path the compiler reports.
- `tprims-blas/src/scalar.rs` bounds on `tensorcontract::KernelSet` and
  `tensorcontract::Element` keep working through the re-exports.

- [x] **Step 6: Build and fix until clean**

Run: `cargo build --workspace --all-targets 2>&1 | grep -E "^(error|warning)" | sort | uniq -c`
Expected: no output. Fix only import paths and visibility; change no logic.

- [x] **Step 7: Run all tests; counts must match Step 1**

Run:
```bash
cargo test -p tensorcontract --release 2>&1 | tail -3
cargo test -p tprims-gemm-kernel -p tprims-kernel-tensorcontract --release 2>&1 | grep "test result"
cargo test --workspace 2>&1 | grep -E "^test result" | awk '{s+=$4} END {print s " passed"}'
cargo clippy --workspace --all-targets -- -D warnings
```
Expected: the same total number of passed tests as Step 1, now split across
crates. Clippy is clean.

- [x] **Step 8: Commit**

```bash
git add -A && git commit -m "Split the kernel layer into tprims-gemm-kernel and tprims-kernel-tensorcontract

Imports and visibility only; tensorcontract re-exports the moved items so its
public API is unchanged. Test count unchanged.

Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>
Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Kernel-family contract, CPU features, validation, registry

**Files:**
- Create: `crates/tprims-gemm-kernel/src/{cpu.rs,family.rs,registry.rs}`
- Modify: `crates/tprims-gemm-kernel/src/lib.rs`
- Test: `crates/tprims-gemm-kernel/tests/family_contract.rs`

**Interfaces:**
- Produces (all `pub`, re-exported at the crate root):

```rust
// cpu.rs
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CpuFeatures { pub avx2: bool, pub fma: bool, pub avx512f: bool, pub neon: bool }
impl CpuFeatures {
    pub fn detect() -> CpuFeatures;                       // cached in OnceLock (std) / all-false (no std)
    pub const NONE: CpuFeatures;                          // all false
    pub fn contains(self, required: CpuFeatures) -> bool; // every true in `required` is true in self
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Isa { Portable, Avx2, Avx512, Neon }

// family.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] #[non_exhaustive]
pub enum Origin { Tensorcontract, Portable, Gemm, PrivateGemmX86 }
impl Origin { pub fn crate_name(self) -> &'static str; pub fn license(self) -> &'static str; }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] #[non_exhaustive]
pub enum KernelImpl { Reference, Optimized, Induced }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] #[non_exhaustive]
pub enum Method { Native, OneM, ThreeM, FourM }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] #[non_exhaustive]
pub enum Layout { Real, Interleaved, Planar, OneE, OneR, ThreeM }   // packed operand layout
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComplexScheme { pub method: Method, pub a: Layout, pub b: Layout, pub tile: TileFormat }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum CPref { Row, Col, Any }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Axis { Row, Col }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum BAccess { Packed, Direct { unit_stride: Axis } }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum CUpdate { ScratchTile, Direct }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocksizes { pub mc: (usize, usize), pub kc: (usize, usize), pub nc: (usize, usize) } // (default, max)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps { pub scatter_pack: bool, pub conj_a: bool, pub conj_b: bool }

/// Scratch-tile kernel: overwrite `tile` with A·B for `kc` steps (existing contract).
pub type TileUkrFn<R> = unsafe fn(kc: usize, a: *const R, b: *const R, tile: *mut R);
/// Direct kernel: `d = alpha_d·d + beta_ab·(A·B)` on an m×n (≤ MR×NR) tile with
/// general strides; `b` may be a packed panel (b_rs = NR, b_cs = 1) or B in place.
pub type DirectUkrFn<R> = unsafe fn(m: usize, n: usize, k: usize, d: *mut R, rs_d: isize, cs_d: isize,
    a: *const R, a_cs: isize, b: *const R, b_rs: isize, b_cs: isize, alpha_d: R, beta_ab: R, aux: &UkrAux<R>);
#[derive(Clone, Copy)]
pub enum UkrFn<R: 'static> { Tile(TileUkrFn<R>), Direct(DirectUkrFn<R>) }
#[derive(Clone, Copy)]
pub struct UkrAux<R: 'static> { pub a_next: *const R, pub b_next: *const R, pub inner: Option<&'static KernelFamily<R>> }
pub type PackFn<T> = unsafe fn(base: *const T, vscat: &[i64], vbs: &[i64], kscat: &[i64], vr: usize,
    conj: bool, out: *mut <T as Element>::Real);

#[derive(Clone, Copy)]
pub struct KernelFamily<R: 'static> {
    pub id: &'static str,
    pub origin: Origin,
    pub isa: Isa,
    pub required: CpuFeatures,
    pub imp: KernelImpl,
    pub priority: u16,               // higher wins under Auto; portable = 0
    pub complex: Option<ComplexScheme>,
    pub mr: usize, pub nr: usize,
    pub a_per_k: usize, pub b_per_k: usize, // reals per k-step of one packed sliver (as Ukr)
    pub tile_bound: usize,           // reals in the scratch tile
    pub c_pref: CPref,
    pub ukr: UkrFn<R>,
    pub b_access: BAccess,
    pub c_update: CUpdate,
    pub blocks: Blocksizes,
    pub caps: Caps,
    pub allow_auto: bool,            // false for ThreeM (accuracy trade-off)
}
impl<R: Real> KernelFamily<R> {
    pub fn validate(&self) -> Result<(), FamilyError>;
    /// Bridge to the existing driver type; `None` for Direct families.
    pub fn as_ukr(&self) -> Option<Ukr<R>>;
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyError { pub id: &'static str, pub reason: &'static str }

// registry.rs
pub trait Families: Element { fn registered() -> &'static [&'static KernelFamily<Self::Real>]; }
pub struct Registry;  // see below
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelInfo { pub id: &'static str, pub origin: Origin, pub crate_name: &'static str,
    pub license: &'static str, pub isa: Isa, pub dtype: &'static str, pub complex: Option<ComplexScheme>,
    pub mr: usize, pub nr: usize, pub available_on_this_cpu: bool }
#[derive(Clone, Debug, PartialEq, Eq)] #[non_exhaustive]
pub enum SelectError {
    UnknownId { id: String },
    NotBuilt { id: String, feature: &'static str },
    CpuUnsupported { id: String, missing: CpuFeatures },
    DtypeMismatch { id: String, dtype: &'static str },
    Incompatible { id: String, reason: &'static str },
    EngineUnsupported { engine: &'static str, reason: &'static str },
    NotImplemented { what: &'static str },
}
impl core::fmt::Display for SelectError { /* one line per variant naming id/feature/reason */ }
impl core::error::Error for SelectError {}
```

- The registry is **extensible without dependency cycles**:
  - `tprims-gemm-kernel` cannot name the kernel crates, so `Registry` holds
    a process-wide list of registration functions, one list per real type.
  - Registration: `pub fn register<R: Real>(f: fn() -> &'static [&'static
    KernelFamily<R>])`, which appends to a `OnceLock<Mutex<Vec<fn() ->
    …>>>` per `R` (a `TypeId`-keyed map is avoided: use four statics
    through a sealed trait `RealSlot` implemented for f32/f64).
  - Families supply their list as `static` descriptors, so the functions
    are free.
  - Lookup: `Registry::families::<T: Element>(cpu: CpuFeatures, include_unavailable: bool) -> Vec<&'static KernelFamily<T::Real>>`
    returns the families for `T`'s complexity, sorted by descending
    priority. It **always includes the portable families**, registered by
    `tprims-gemm-kernel` itself.
  - The `NotBuilt` error needs a map from id prefix to cargo feature:
    `pub fn register_known_prefix(prefix: &'static str, feature: &'static
    str)`. `tprims-blas` registers `gemm.` → `kernel-gemm` and `pgx86.` →
    `kernel-pgx86` even when those features are off, so an unknown id with
    a known prefix is reported as `NotBuilt`.
- Validation rules, in `validate`:
  - `mr > 0 && nr > 0`;
  - `a_per_k >= mr` and `b_per_k >= nr` for real families, and `>= 2*mr`
    (planar) for complex ones;
  - `tile_bound >= mr*nr*tile_planes(tile)`, where `Real` = 1, `Planar` = 2,
    `OneM` = 2, `ThreeM` = 3, `FourM` = 4, `Interleaved` = 2;
  - `blocks.mc.0 % mr == 0`, `blocks.nc.0 % nr == 0`, `default <= max`;
  - `Method::OneM ⇒ mr % 2 == 0 && nr % 2 == 0`;
  - `complex.is_some() ⇔ (a, b) ≠ (Real, Real)`;
  - `(method, a, b, tile)` must be one of the rows of spec §4.3;
  - `Direct ukr ⇔ c_update == Direct`;
  - `b_access == Direct ⇒ c_update == Direct` (the only in-place-B kernels
    in scope are Direct kernels).

- [ ] **Step 1: Write the failing tests**

`crates/tprims-gemm-kernel/tests/family_contract.rs`:

```rust
use tprims_gemm_kernel::*;

unsafe fn nop_tile(_: usize, _: *const f64, _: *const f64, _: *mut f64) {}

fn fam(mr: usize, nr: usize) -> KernelFamily<f64> {
    KernelFamily {
        id: "test.f64.4x4", origin: Origin::Portable, isa: Isa::Portable, required: CpuFeatures::NONE,
        imp: KernelImpl::Reference, priority: 0, complex: None, mr, nr, a_per_k: mr, b_per_k: nr,
        tile_bound: mr * nr, c_pref: CPref::Any, ukr: UkrFn::Tile(nop_tile), b_access: BAccess::Packed,
        c_update: CUpdate::ScratchTile, blocks: Blocksizes { mc: (mr * 8, mr * 16), kc: (256, 512), nc: (nr * 64, nr * 128) },
        caps: Caps { scatter_pack: true, conj_a: true, conj_b: true }, allow_auto: true,
    }
}

#[test]
fn valid_family_passes() { assert_eq!(fam(4, 4).validate(), Ok(())); }

#[test]
fn zero_mr_rejected() { assert!(fam(0, 4).validate().is_err()); }

#[test]
fn mc_not_multiple_of_mr_rejected() {
    let mut f = fam(4, 4); f.blocks.mc = (6, 16);
    assert_eq!(f.validate().unwrap_err().reason, "mc default is not a multiple of mr");
}

#[test]
fn tile_bound_too_small_rejected() {
    let mut f = fam(4, 4); f.tile_bound = 15;
    assert!(f.validate().is_err());
}

#[test]
fn direct_b_requires_direct_c() {
    let mut f = fam(4, 4); f.b_access = BAccess::Direct { unit_stride: Axis::Col };
    assert!(f.validate().is_err());
}

#[test]
fn cpu_contains() {
    let have = CpuFeatures { avx2: true, fma: true, ..CpuFeatures::NONE };
    assert!(have.contains(CpuFeatures { avx2: true, ..CpuFeatures::NONE }));
    assert!(!have.contains(CpuFeatures { avx512f: true, ..CpuFeatures::NONE }));
}

#[test]
fn select_error_messages_name_the_cause() {
    let e = SelectError::NotBuilt { id: "gemm.avx2.f64.8x6".into(), feature: "kernel-gemm" };
    assert!(e.to_string().contains("kernel-gemm"));
}
```

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p tprims-gemm-kernel --test family_contract`
Expected: compile errors (`KernelFamily` and friends not found).

- [ ] **Step 3: Implement `cpu.rs`, `family.rs`, `registry.rs`**

`CpuFeatures::detect`:
- on x86: `is_x86_feature_detected!("avx2")`, `("fma")`, `("avx512f")`;
- on aarch64: `std::arch::is_aarch64_feature_detected!("neon")`;
- cached in a `static OnceLock<CpuFeatures>` under `std`.

The ISA detection in `tprims-kernel-tensorcontract/src/x86.rs`
(`available_isas`) stays as is; `CpuFeatures` is the contract's view. In
`validate`, return the first failing rule with the exact reason strings
the tests use:
- `"mr is zero"`
- `"nr is zero"`
- `"mc default is not a multiple of mr"`
- `"nc default is not a multiple of nr"`
- `"default blocksize exceeds max"`
- `"tile_bound below tile size"`
- `"packed k-step shorter than the tile"`
- `"1m needs even mr and nr"`
- `"complex scheme not in the supported table"`
- `"ukr kind disagrees with c_update"`
- `"direct B needs a Direct kernel"`

Registry statics:

```rust
use std::sync::{Mutex, OnceLock};
type List<R> = fn() -> &'static [&'static KernelFamily<R>];
pub trait RealSlot: Real { fn slot() -> &'static OnceLock<Mutex<Vec<List<Self>>>>; }
impl RealSlot for f32 { fn slot() -> &'static OnceLock<Mutex<Vec<List<f32>>>> { static S: OnceLock<Mutex<Vec<List<f32>>>> = OnceLock::new(); &S } }
impl RealSlot for f64 { fn slot() -> &'static OnceLock<Mutex<Vec<List<f64>>>> { static S: OnceLock<Mutex<Vec<List<f64>>>> = OnceLock::new(); &S } }
pub fn register<R: RealSlot>(f: List<R>) {
    let v = R::slot().get_or_init(|| Mutex::new(Vec::new()));
    let mut v = v.lock().unwrap();
    if !v.iter().any(|g| core::ptr::fn_addr_eq(*g, f)) { v.push(f); }
}
```
If `fn_addr_eq` is not stable at MSRV 1.89, compare `*g as usize == f as
usize`. It is stable since 1.85.

`Registry::families` concatenates the portable list (Task 3) with every
registered list, filters by complexity, by `cpu.contains(required)` (unless
`include_unavailable`) and by `id` uniqueness (`debug_assert!` on
duplicates), then sorts stably by `priority` descending.

`list_kernels::<T>()` maps to `KernelInfo`. It uses `Registry::families(…,
true)` and sets `available_on_this_cpu = CpuFeatures::detect().contains(f.required)`.
`dtype` is `"f32"`, `"f64"`, `"c32"` or `"c64"`, from `T::IS_COMPLEX` and
`size_of::<T::Real>()`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p tprims-gemm-kernel --test family_contract`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "tprims-gemm-kernel: kernel-family descriptor, CPU features, validation, registry

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Portable family, interleaved layout, tensorcontract families

**Files:**
- Create: `crates/tprims-gemm-kernel/src/portable.rs`
- Modify: `crates/tprims-gemm-kernel/src/{pack.rs,writeback.rs,types.rs,registry.rs}`
- Modify: `crates/tprims-kernel-tensorcontract/src/lib.rs` (families registration)
- Test: `crates/tprims-gemm-kernel/tests/portable.rs`, `crates/tprims-kernel-tensorcontract/tests/families.rs`

**Interfaces:**
- Consumes: Task 2 types.
- Produces:
  - `PackFormat::Interleaved` (2 reals per element, stored `re, im` per
    lane: lane `t` at `2*t`, `2*t+1`);
  - `TileFormat::Interleaved` (column-major complex tile, `(i, j)` at
    `2*(j*mr+i)`);
  - `TileFormat::FourM` (declared here and used in Task 6: four planes of
    `mr*nr`);
  - `tprims_gemm_kernel::portable::{families_f32, families_f64}`
    (`&'static [&'static KernelFamily<R>]`), registered implicitly by
    `Registry`;
  - portable ids:
    - `portable.f64.4x4`, `portable.f32.4x4`;
    - `portable.c64.native.4x4`, `portable.c32.native.4x4` (interleaved);
    - `portable.f64.4x4.direct`, `portable.f64.4x4.direct-b` (Task 5 fills
      their ukr; declared here with `allow_auto: false`, and absent from
      the list until Task 5, to keep this task's list consistent).
  - `tprims_kernel_tensorcontract::families_f32()` / `families_f64()`,
    covering every `IsaConfigs` menu entry and scalar kernel;
  - `pub fn register()`, calling `tprims_gemm_kernel::register` for f32 and
    f64 once (guarded by a `std::sync::Once`).
- **tensorcontract family ids**:
  - form: `tc.{isa}.{dtype}[.{method}].{mr}x{nr}`;
  - `isa` is one of `scalar`, `avx2`, `avx512`, `neon`;
  - `method` is one of `planar`, `1m`, `3m`, for complex only;
  - examples: `tc.avx2.f64.8x6`, `tc.avx512.c64.planar.8x6`,
    `tc.scalar.f32.4x4`.
- **Priorities**:
  - the menu head (`config_real()` / `config_cplx(method)` of the best ISA)
    gets 300 for avx512, 200 for avx2 and 200 for neon;
  - other menu entries of the same ISA get the head's priority minus
    1..=n, in menu order;
  - scalar tensorcontract gets 10, portable gets 0.
- **Auto selection among complex methods**:
  - `planar` gets the table priority;
  - `1m` gets priority − 50;
  - `3m` has `allow_auto: false`.

  This reproduces today's default `ComplexMethod::Planar`.
- **Blocksizes**: `blocks` = `(cfg.blk.{mc,kc,nc}, same)` from the
  normalised `KernelConfig`. `max` equals `default`; the spec says `max` has
  no consumer yet.
- Because `KernelConfig` values come from functions
  (`config_real()`, `isa_configs_f64(isa).config_at(...)`), the lists are
  built once into a `static OnceLock<Vec<&'static KernelFamily<R>>>`, with
  `Box::leak` per descriptor. `families_*()` returns `&'static [...]` from
  it.

- [ ] **Step 1: Write the failing tests**

`crates/tprims-gemm-kernel/tests/portable.rs`:

```rust
use num_complex::Complex;
use tprims_gemm_kernel::*;

#[test]
fn portable_families_validate_and_have_ids() {
    for f in Registry::families::<f64>(CpuFeatures::NONE, true) {
        f.validate().unwrap();
        assert!(f.id.starts_with("portable.") || f.id.starts_with("tc."), "{}", f.id);
    }
    let ids: Vec<_> = Registry::families::<Complex<f64>>(CpuFeatures::NONE, false).iter().map(|f| f.id).collect();
    assert!(ids.contains(&"portable.c64.native.4x4"), "{ids:?}");
}

#[test]
fn portable_native_complex_tile_matches_naive() {
    let f = Registry::families::<Complex<f64>>(CpuFeatures::NONE, false)
        .into_iter().find(|f| f.id == "portable.c64.native.4x4").unwrap();
    let UkrFn::Tile(ukr) = f.ukr else { panic!() };
    let kc = 7;
    // interleaved packed A: k-step p, lane i at [p*8 + 2i], [p*8 + 2i + 1]
    let a: Vec<f64> = (0..kc * 8).map(|x| ((x * 37 % 19) as f64 - 9.0) / 7.0).collect();
    let b: Vec<f64> = (0..kc * 8).map(|x| ((x * 53 % 23) as f64 - 11.0) / 5.0).collect();
    let mut tile = vec![0.0; f.tile_bound];
    unsafe { ukr(kc, a.as_ptr(), b.as_ptr(), tile.as_mut_ptr()) };
    for j in 0..4 { for i in 0..4 {
        let mut w = Complex::new(0.0, 0.0);
        for p in 0..kc {
            w += Complex::new(a[p * 8 + 2 * i], a[p * 8 + 2 * i + 1]) * Complex::new(b[p * 8 + 2 * j], b[p * 8 + 2 * j + 1]);
        }
        let g = Complex::new(tile[2 * (j * 4 + i)], tile[2 * (j * 4 + i) + 1]);
        assert!((g - w).norm() <= 1e-12 * w.norm().max(1.0), "({i},{j}) {g} vs {w}");
    }}
}

#[test]
fn interleaved_pack_roundtrip() {
    // pack a 3x2 complex block (vr = 4, zero-filled lane 3) and read lanes back
    let src: Vec<Complex<f64>> = (0..6).map(|x| Complex::new(x as f64, -(x as f64))).collect();
    let vscat = [0i64, 1, 2]; let kscat = [0i64, 3];
    let vbs = tprims_gemm_kernel::scatter::build_block_scatter(&vscat, 4);
    let mut out = vec![f64::NAN; tprims_gemm_kernel::pack::panel_len(3, 4, 2, PackFormat::Interleaved)];
    unsafe { tprims_gemm_kernel::pack::pack_panel::<Complex<f64>>(src.as_ptr(), &vscat, &vbs, &kscat, 4, false, PackFormat::Interleaved, out.as_mut_ptr()) };
    assert_eq!(&out[0..8], &[0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 0.0, 0.0]);
    assert_eq!(&out[8..16], &[3.0, -3.0, 4.0, -4.0, 5.0, -5.0, 0.0, 0.0]);
}
```

`crates/tprims-kernel-tensorcontract/tests/families.rs`:

```rust
use tprims_gemm_kernel::*;
use num_complex::Complex;

#[test]
fn every_tc_family_validates_and_ids_are_unique() {
    tprims_kernel_tensorcontract::register();
    let mut ids = std::collections::BTreeSet::new();
    for f in Registry::families::<f64>(CpuFeatures::NONE, true).iter()
        .chain(Registry::families::<Complex<f64>>(CpuFeatures::NONE, true).iter()) {
        f.validate().unwrap_or_else(|e| panic!("{e:?}"));
        assert!(ids.insert(f.id), "duplicate id {}", f.id);
    }
}

#[test]
fn auto_head_matches_legacy_default() {
    tprims_kernel_tensorcontract::register();
    let cpu = CpuFeatures::detect();
    let head = Registry::families::<f64>(cpu, false).into_iter().find(|f| f.allow_auto).unwrap();
    let legacy = <f64 as tprims_kernel_tensorcontract::KernelSet>::config_real();
    assert_eq!((head.mr, head.nr), (legacy.ukr.mr, legacy.ukr.nr), "{}", head.id);
    let chead = Registry::families::<Complex<f64>>(cpu, false).into_iter().find(|f| f.allow_auto).unwrap();
    let clegacy = <f64 as tprims_kernel_tensorcontract::KernelSet>::config_cplx(ComplexMethod::Planar);
    assert_eq!((chead.mr, chead.nr, chead.complex.unwrap().method), (clegacy.ukr.mr, clegacy.ukr.nr, Method::Native));
}

#[test]
fn id_snapshot() {
    tprims_kernel_tensorcontract::register();
    let ids: Vec<_> = list_kernels::<f64>().into_iter().map(|k| k.id).collect();
    // First run: print and paste into tests/snapshots/f64_ids.txt; later runs compare.
    let want = include_str!("snapshots/f64_ids.txt");
    assert_eq!(ids.join("\n"), want.trim_end(), "update tests/snapshots/f64_ids.txt deliberately");
}
```

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p tprims-gemm-kernel --test portable; cargo test -p tprims-kernel-tensorcontract --test families`
Expected: compile errors (`portable`, `PackFormat::Interleaved` and
`register` are missing).

- [ ] **Step 3: Implement**

- `types.rs`:
  - add `PackFormat::Interleaved` (`reals_per_element` = 2);
  - add `TileFormat::Interleaved` and `TileFormat::FourM`;
  - add `tile_planes(TileFormat) -> usize` as defined in Task 2.
- `pack.rs` `emit`: add the arm `PackFormat::Interleaved => { *o.add(2*t) =
  re; *o.add(2*t+1) = if conj { -im } else { im }; }`, following the
  existing arms' conj handling.
- `writeback.rs` `tile_value`:
  - add `TileFormat::Interleaved => (ab[2*(j*mr+i)], ab[2*(j*mr+i)+1])`;
  - add `TileFormat::FourM`, computing `(p0 - p1, p2 + p3)` over the planes
    `(Ar·Br, Ai·Bi, Ar·Bi, Ai·Br)` at offsets `q*mr*nr + j*mr + i`;
  - add the same two arms to the `tile_value` test helper in
    `tprims-kernel-tensorcontract/src/lib.rs`.
- `portable.rs`, whose header is:

  ```rust
  //! Portable kernel family: plain loops for LLVM to vectorize, no intrinsics.
  //! Always available; the fallback for Auto and the first native interleaved
  //! complex family. Project-owned (tprims), MIT OR Apache-2.0.
  ```

  Then:

  ```rust
  pub unsafe fn real_tile<R: Real, const MR: usize, const NR: usize>(kc: usize, a: *const R, b: *const R, t: *mut R) {
      let mut acc = [[R::ZERO; MR]; NR];
      for p in 0..kc { for j in 0..NR { let bj = *b.add(p * NR + j); for i in 0..MR {
          acc[j][i] = acc[j][i] + *a.add(p * MR + i) * bj; } } }
      for j in 0..NR { for i in 0..MR { *t.add(j * MR + i) = acc[j][i]; } }
  }
  pub unsafe fn cplx_tile<R: Real, const MR: usize, const NR: usize>(kc: usize, a: *const R, b: *const R, t: *mut R) {
      let mut re = [[R::ZERO; MR]; NR]; let mut im = [[R::ZERO; MR]; NR];
      for p in 0..kc { for j in 0..NR {
          let (br, bi) = (*b.add(p * 2 * NR + 2 * j), *b.add(p * 2 * NR + 2 * j + 1));
          for i in 0..MR {
              let (ar, ai) = (*a.add(p * 2 * MR + 2 * i), *a.add(p * 2 * MR + 2 * i + 1));
              re[j][i] = re[j][i] + ar * br - ai * bi;
              im[j][i] = im[j][i] + ar * bi + ai * br;
      } } }
      for j in 0..NR { for i in 0..MR { *t.add(2 * (j * MR + i)) = re[j][i]; *t.add(2 * (j * MR + i) + 1) = im[j][i]; } }
  }
  ```

  Use the arithmetic trait methods that `Real` actually provides: the
  kernels in `scalar.rs` show the spelling (`R::ZERO`, `+`, `*`, or
  `mul_add`). Match them.

  Descriptors are `static`s. For the real portable 4×4: `a_per_k: 4,
  b_per_k: 4, tile_bound: 16, blocks: (mc 64, kc 256, nc 1024)`, taken from
  `Blocking::derive_at_depth(8, 1, 1, 256)` rounded down to multiples of 4,
  and hard-coded. For the complex portable 4×4: `a_per_k: 8, b_per_k: 8,
  tile_bound: 32`, complex `(Native, Interleaved, Interleaved,
  TileFormat::Interleaved)`. `caps` is all true, `c_pref: Any`, `priority:
  0`.
- `tprims-kernel-tensorcontract`:
  - `families_f64()` iterates `scalar::config_real::<f64,4,4>()` and
    `config_cplx` for each method;
  - on x86 it iterates `x86::available_isas()`, but must register **all
    compiled** ISAs, using `include_unavailable` semantics. So use
    `x86::isa_configs_f64(isa)` for every `isa` in the ISA enum, not only
    the available ones, and mark `required` accordingly (avx2+fma, or
    avx512f). Build `KernelFamily` from `KernelConfig` via `fn
    from_config(id, isa, required, prio, cfg, scheme) -> KernelFamily<R>`;
  - `a_per_k`/`b_per_k`/`tile_bound` come from `cfg.ukr.{a_per_k, b_per_k,
    tile}`;
  - `ukr` is `UkrFn::Tile(cfg.ukr.func)`;
  - the complex scheme maps `(a_pack, b_pack, tile_fmt)`:
    - `Planar, Planar, Planar` → `Method::Native`, `Layout::Planar`;
    - `OneE, Real, OneM` → `OneM`, `(OneE, OneR)`; the packed B of 1m is
      `PackFormat::Real` today, and `Layout::OneR` names that role;
    - `ThreeM` → `ThreeM`.
  - check each against the §4.3 table in `validate`.
- Create `tests/snapshots/f64_ids.txt` by running `id_snapshot` once with a
  temporary `println!`. Review the list by eye: every menu entry of every
  compiled ISA appears once. Then commit it.

- [ ] **Step 4: Run tests**

Run: `cargo test -p tprims-gemm-kernel && cargo test -p tprims-kernel-tensorcontract --release`
Expected: all pass, including the moved kernel tests from Task 1.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Portable family, interleaved complex layout, tensorcontract kernels as families

Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>
Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `ResolvedGemm` and the driver on it

**Files:**
- Create: `crates/tprims-gemm-kernel/src/resolved.rs`
- Modify: `tensorcontract/src/{driver.rs,kernel/mod.rs,lib.rs,plan.rs}`, `crates/tprims-gemm-kernel/src/pack.rs` (per-layout monomorphized packers)
- Test: `tensorcontract/tests/resolved.rs`, `tensorcontract/tests/env_after_plan.rs` (own binary)

**Interfaces:**
- Consumes:
  - `KernelFamily<R>`, `Registry` and `SelectError` (Task 2);
  - the tc and portable families (Task 3).
- Produces:

```rust
// tprims-gemm-kernel/src/resolved.rs
#[derive(Clone, Copy, Debug)]
pub struct ResolvedGemm<R: 'static> {
    pub family: &'static KernelFamily<R>,
    pub mc: usize, pub kc: usize, pub nc: usize,   // scaled, multiples enforced, for `width`
    pub width: usize,                               // effective thread width used for blocking
    pub a_layout: PackFormat,                       // packer chosen by layout, see below
    pub b_layout: PackFormat,
    // Task 7 adds `partition` and `opts`; per-call decisions (Task 5) live in ResolvedCall.
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelChoice { Auto, Id(std::borrow::Cow<'static, str>) }
pub fn resolve<T: Element + Families>(choice: &KernelChoice, method: Option<Method>, cpu: CpuFeatures,
    width: usize, blocking_override: Option<Blocking>) -> Result<ResolvedGemm<T::Real>, SelectError>;
pub fn process_default<T: Element + Families>() -> &'static ResolvedGemm<T::Real>;  // OnceLock per dtype, width = env_threads()
```

- **Packers**. `PackFn<T>` is generic over the element `T`, while
  `ResolvedGemm` is keyed by `R`, so `ResolvedGemm` stores the layouts
  (`a_layout`, `b_layout`) and the driver calls
  `pack::pack_fn::<T>(fmt) -> PackFn<T>`, a `match` done **once per
  execute** that returns a monomorphized
  `pack_panel_fmt::<T, const FMT: u8>`. The per-element `match` in `emit`
  therefore disappears from the hot loop, because `FMT` is a const
  parameter. This satisfies spec §7 ("per-layout monomorphized
  functions").
- **Custom packers**. For custom-format families (future ports),
  `KernelFamily` gains `pack_override: Option<(PackFnReal<R>, PackFnCplx<R>)>`.
  It is declared now, always `None`, and ignored by the driver. The ruling
  is recorded here: custom packers exist in the contract without an
  exerciser until a port needs one; the spec's pack pointers are satisfied
  by the layout-selected monomorphized packers.
- **`Plan` side**:
  - add `pub fn resolved<T>(&self) -> ResolvedGemm<T::Real>`; it returns
    the plan's cached value, or `process_default::<T>()` adjusted by the
    plan's `complex_method` and `row_block` overrides, cached in the plan;
  - add `pub fn with_kernel(self, choice: KernelChoice) -> Result<Plan>`,
    which resolves eagerly for all four dtypes lazily. Resolution needs
    `T`, so `with_kernel` stores the choice, and `resolved::<T>()` resolves
    and caches per dtype in four `OnceLock<Result<ResolvedGemm<_>,
    SelectError>>` fields;
  - errors surface through `Plan::try_resolved::<T>() -> Result<…>`, which
    typed callers (tprims) call at plan creation.
- **Driver**:
  - `execute_capped` gains a `rg: &ResolvedGemm<T::Real>` parameter;
  - `execute`/`execute_with` pass `&plan.resolved::<T>()`;
  - a new `execute_resolved(plan, rg, spmd, alpha, a, b, beta, c, d)` is
    the tprims entry;
  - `config_for_plan` is no longer called from the driver: it becomes the
    implementation detail of `resolved()` for the legacy menu (row_block),
    via the family whose `(mr, nr)` matches the legacy config;
  - `Blocking::derive` is no longer called on execute: blocking comes from
    `rg.{mc,kc,nc}`.
- **Effective-width blocking fix**. `resolve` computes the Analytical
  blocking with the `width` argument; the tprims callers pass the Spmd
  width. `plan.threads()` is used only by the legacy `execute` path.

- [ ] **Step 1: Write the failing tests**

`tensorcontract/tests/resolved.rs`:

```rust
use tensorcontract::*;
use tprims_gemm_kernel::{CpuFeatures, KernelChoice, Registry};

fn small_plan() -> (Layout, Layout, Layout) {
    (Layout::col_major(&[37, 29]), Layout::col_major(&[29, 41]), Layout::col_major(&[37, 41]))
}

/// Same numbers for every family: each family vs the reference oracle, both orientations.
#[test]
fn swapped_orientation_matches_oracle_for_every_family() {
    tprims_kernel_tensorcontract::register();
    for f in Registry::families::<f64>(CpuFeatures::detect(), false) {
        for (sa, sb) in [((1, 37), (1, 29)), ((29, 1), (41, 1))] { // col-major and row-major A/B
            check_family_vs_oracle::<f64>(f.id, sa, sb, 1e-12);
        }
    }
    for f in Registry::families::<num_complex::Complex<f64>>(CpuFeatures::detect(), false) {
        check_family_vs_oracle::<num_complex::Complex<f64>>(f.id, (1, 37), (41, 1), 1e-11);
    }
}
```
`check_family_vs_oracle` is a helper in the same file. It:
- builds A (37×29) and B (29×41) with the given (row, col) strides and
  deterministic values;
- builds `Plan::new(...)`, then `.with_kernel(KernelChoice::Id(id.into()))`;
- runs `plan.run_raw(1.5, a, b, 0.5, c, d)`;
- runs `reference::contract` on the same operands;
- asserts the max abs error ≤ `tol × max(|ref|)`.

Use the exact `Plan::new` / `Operand` / `reference` signatures already used
by `tensorcontract/tests/*.rs`, and copy the operand-building helper from
the existing test that does the closest thing (grep for
`reference::contract` in `tensorcontract/tests`).

```rust
#[test]
fn execute_does_not_rebuild_config() {
    // resolved() is cached: two calls return the same family pointer and blocking.
    let (a, b, d) = small_plan();
    let p = Plan::new(Operand::new(&a, &[0, 2]), Operand::new(&b, &[2, 1]), None, Operand::new(&d, &[0, 1])).unwrap();
    let r1 = p.resolved::<f64>(); let r2 = p.resolved::<f64>();
    assert!(core::ptr::eq(r1.family, r2.family));
    assert_eq!((r1.mc, r1.kc, r1.nc), (r2.mc, r2.kc, r2.nc));
}

#[test]
fn unknown_kernel_id_is_a_plan_error() {
    let (a, b, d) = small_plan();
    let p = Plan::new(Operand::new(&a, &[0, 2]), Operand::new(&b, &[2, 1]), None, Operand::new(&d, &[0, 1])).unwrap()
        .with_kernel(KernelChoice::Id("tc.nope.f64.1x1".into())).unwrap();
    assert!(matches!(p.try_resolved::<f64>(), Err(tprims_gemm_kernel::SelectError::UnknownId { .. })));
}

#[test]
fn blocking_uses_effective_width() {
    let r1 = tprims_gemm_kernel::resolve::<f64>(&KernelChoice::Auto, None, CpuFeatures::detect(), 1, None).unwrap();
    let r8 = tprims_gemm_kernel::resolve::<f64>(&KernelChoice::Auto, None, CpuFeatures::detect(), 8, None).unwrap();
    // Legacy model: width-independent; Analytical model: nc may shrink with width. Either way width is recorded.
    assert_eq!((r1.width, r8.width), (1, 8));
}
```
`tensorcontract/tests/env_after_plan.rs`, a separate binary because it
mutates the environment:

```rust
#[test]
fn env_change_after_planning_has_no_effect() {
    let before = tprims_gemm_kernel::process_default::<f64>().kc;
    unsafe { std::env::set_var("TENSORCONTRACT_KC_COUPLE", "7"); std::env::set_var("TENSORCONTRACT_KERNEL", "scalar"); }
    let after = tprims_gemm_kernel::process_default::<f64>();
    assert_eq!(after.kc, before);
    assert!(!after.family.id.starts_with("tc.scalar") || before == after.kc);
}
```

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p tensorcontract --test resolved --test env_after_plan`
Expected: compile errors (`resolved`, `with_kernel`, `KernelChoice` and
`resolve` are missing).

- [ ] **Step 3: Implement**

1. **`resolve`**:
   - candidates = `Registry::families::<T>(cpu, true)`;
   - `Id(s)`:
     - find by id, else `UnknownId`, or `NotBuilt` when the prefix is
       known;
     - check `cpu.contains(required)`, else `CpuUnsupported { missing }`;
     - check complexity, else `DtypeMismatch`;
     - check that `method` (if `Some`) equals the family's method, else
       `Incompatible { reason: "family implements a different complex method" }`;
   - `Auto`: the first family with `allow_auto`, `cpu.contains`, and
     `method` matching when `Some`.
2. **Blocking**:
   - start from `family.blocks` defaults;
   - if `cache::block_model()` is Analytical, recompute with
     `Blocking::model` using `width`;
   - apply `env_blocking()` overrides, which are already read once;
   - apply `blocking_override` last;
   - enforce multiples as `KernelConfig::with_blocking` does.

   This is exactly `normalise_for(width)` plus `with_blocking`, so reuse
   them through `KernelConfig { ukr: family.as_ukr()?, blk }`; Direct
   families have no `Ukr`, so compute blocking directly for them.
3. **`process_default`**:
   - four `OnceLock` statics, one per dtype;
   - `TENSORCONTRACT_KERNEL`, read once, maps to a restriction on `isa`:
     `scalar` → only `tc.scalar.*`/portable, `avx2`, `avx512`, `neon`;
   - `TENSORCONTRACT_COMPLEX` → `method`;
   - `TPRIMS_GEMM_KERNEL` → `Id`;
   - all through `env_once!`.
4. **Driver**:
   - replace the `config_for_plan` and `plan.blocking` lines at the top of
     `execute_capped` with `let ukr = rg.family.as_ukr().expect("Direct
     families are handled in Task 5");` and `let (mc0, kc, nc0) = (rg.mc,
     rg.kc, rg.nc);`;
   - keep the rest as is;
   - replace every `pack_panel::<T>(…, ukr.a_pack, …)` call with
     `(pack_a)(…)`, where `let pack_a = pack_fn::<T>(if swap { rg.b_layout
     } else { rg.a_layout });` and likewise for `pack_b`. This is the
     orientation swap of the packers (Review Focus 1).
5. **`pack_fn::<T>`** in `pack.rs`:

   ```rust
   pub fn pack_fn<T: Element>(fmt: PackFormat) -> PackFn<T> {
       match fmt {
           PackFormat::Real => pack_panel_fmt::<T, 0>,
           PackFormat::Planar => pack_panel_fmt::<T, 1>,
           PackFormat::OneE => pack_panel_fmt::<T, 2>,
           PackFormat::ThreeM => pack_panel_fmt::<T, 3>,
           PackFormat::Interleaved => pack_panel_fmt::<T, 4>,
       }
   }
   unsafe fn pack_panel_fmt<T: Element, const F: u8>(base: *const T, vscat: &[i64], vbs: &[i64], kscat: &[i64], vr: usize, conj: bool, out: *mut T::Real) {
       unsafe { pack_panel::<T>(base, vscat, vbs, kscat, vr, conj, fmt_of(F), out) }
   }
   const fn fmt_of(f: u8) -> PackFormat { match f { 0 => PackFormat::Real, 1 => PackFormat::Planar, 2 => PackFormat::OneE, 3 => PackFormat::ThreeM, _ => PackFormat::Interleaved } }
   ```

   Mark `pack_panel` and `emit` `#[inline(always)]` so the constant
   `fmt_of(F)` folds. Keep `pack_panel`'s existing signature for its unit
   tests. `PackFn<T>` is defined in `family.rs` (Task 2).
6. **`Plan`**:
   - add the fields `kernel: KernelChoice` (default `Auto`) and `resolved:
     [OnceLock<Result<ResolvedGemm<f32>, SelectError>>; 2]` plus the same
     for f64. Index 0 is real and 1 is complex, so there are four slots in
     total. `Plan` must stay `Clone`: implement `Clone` manually and clone
     the choices, not the caches;
   - `resolved::<T>()` panics with the `SelectError` message if resolution
     fails. Upstream callers get a clear panic; tprims calls
     `try_resolved` first;
   - the legacy `row_block` menu: if `plan.row_block(menu)` returns `Some(i)`,
     choose the family whose `(mr, nr)` equals `menu[i]` among the same
     ISA's families.
7. **`KernelSet`** stays in `tprims-kernel-tensorcontract` as the
   compatibility shim, now unused by the driver. `tprims-blas` still names
   it in `Scalar::Re` until Task 11.

- [ ] **Step 4: Run tests**

Run: `cargo test -p tensorcontract --release && cargo test -p tensorcontract --test env_after_plan && cargo test --workspace`
Expected: all pass; the counts are those of Task 3 plus the new tests.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "tensorcontract: execute a plan-resolved ResolvedGemm; no config rebuild or env read on execute

Blocking is resolved with the effective thread width (fixes the Analytical
model sizing NC with plan.threads()). Packers are layout-monomorphized and
swap with the orientation.

Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>
Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Direct C update and direct-B in the driver

**Files:**
- Modify: `tensorcontract/src/driver.rs`, `crates/tprims-gemm-kernel/src/{portable.rs,resolved.rs,family.rs}`
- Test: `tensorcontract/tests/direct.rs`

**Interfaces:**
- Consumes: `DirectUkrFn<R>`, `BAccess`, `CUpdate` (Task 2);
  `ResolvedGemm` (Task 4).
- Produces:
  - `portable.f64.4x4.direct` (`CUpdate::Direct`, `BAccess::Packed`);
  - `portable.f64.4x4.direct-b` (`CUpdate::Direct`, `BAccess::Direct {
    unit_stride: Axis::Col }`, meaning B's column stride may be anything
    but its row stride along k must be 1). Both have `allow_auto: false`;
  - `ResolvedGemm` gains `pub pack_b_needed: bool` and `pub
    direct_c_allowed: bool`, decided in `execute_resolved` per call from
    the operands; see below.
- **Semantics of `DirectUkrFn`** (gemm-common convention, so Task 9 wraps
  it without adapters):
  - `d = alpha_d·d + beta_ab·(A·B)`;
  - with `alpha_d == 0`, `d` is not read;
  - the kernel handles `m ≤ MR`, `n ≤ NR`;
  - `a` is packed column-major MR panels with `a_cs` reals between k-steps;
  - `b` is either packed (`b_rs = NR`, `b_cs = 1`) or B in place (`b_rs` =
    B's k stride, `b_cs` = B's n stride).
- **Guard, decided once per execute** (spec §4.2):
  - `direct_c_allowed = !plan.conj_c && !plan.conj_d && (beta == 0 || c
    aliases d exactly)`;
  - "aliases exactly" means `c == d as *const T` and `plan.c_m == plan.d_m`
    and `plan.c_n == plan.d_n` and `plan.h_c == plan.h_d`;
  - plus `!T::IS_COMPLEX`: the Direct families in scope are real. A complex
    Direct family is future work. Its absence is enforced by `validate`
    rejecting `Direct` + `complex.is_some()`, with reason `"complex Direct
    kernels are not supported yet"`.
- **Per tile**:
  - with `direct_c_allowed`, a tile is written directly when `d_m_bs[i0/mr]
    != IRREGULAR` (constant row stride) and the column offsets `dn[j0..j0+nrem]`
    form an arithmetic progression. Precompute `d_n_bs =
    build_block_scatter(dn, nr)` and use `d_n_bs[j0/nr] != IRREGULAR`;
  - the first KC block calls with `alpha_d = if beta == 0 { 0 } else { beta
    }` and `beta_ab = alpha`; later KC blocks call with `alpha_d = 1`;
  - otherwise the tile goes to the scratch tile: call the Direct kernel with
    `d = tile`, `rs = 1`, `cs = mr`, `alpha_d = 0`, `beta_ab = 1`, then call
    the existing `writeback` with `TileFormat::Real`.
- **Direct B**:
  - `pack_b_needed = !(family.b_access == Direct{..} && B's k-stride is 1
    along the packed axis && B's block scatter is regular for every NR
    block)`;
  - when the orientation swap is active, the "B" is the user's A, so the
    check uses the swapped operands (Review Focus 1);
  - when `!pack_b_needed`, the driver skips the B packing and its barriers
    for this group, and passes `bh.offset(bn[j0] + bk[pc])` with `b_rs =
    bk[1] - bk[0]` (or 1 if `k == 1`) and `b_cs = bn stride` to the kernel;
  - direct-B requires `pm == 1` in this phase (no shared panel to publish),
    so `resolve` forces `pn = width, pm = 1` when direct-B is active, and
    records this in `partition`.

- [ ] **Step 1: Write the failing tests**

`tensorcontract/tests/direct.rs`. Reuse `check_family_vs_oracle` by moving
it into `tensorcontract/tests/common/mod.rs`, and `mod common;` in both
files.

```rust
mod common;
use common::*;

#[test]
fn direct_c_guard_matrix() {
    // (beta, d_aliases_c, conj_c, conj_d) → must match the oracle in every case
    for &beta in &[0.0, 1.0, 0.5] {
        for &alias in &[false, true] {
            for &(cc, cd) in &[(false, false), (true, false), (false, true)] {
                check_family_vs_oracle_opts::<f64>("portable.f64.4x4.direct", Opts { beta, alias, conj_c: cc, conj_d: cd, ..Opts::default() }, 1e-12);
            }
        }
    }
}

#[test]
fn direct_family_scatter_output_falls_back_per_tile() {
    check_family_vs_oracle_opts::<f64>("portable.f64.4x4.direct", Opts { scattered_d: true, ..Opts::default() }, 1e-12);
}

#[test]
fn direct_b_reads_b_in_place_and_matches_oracle() {
    check_family_vs_oracle_opts::<f64>("portable.f64.4x4.direct-b", Opts::default(), 1e-12);
}

#[test]
fn direct_b_disabled_when_orientation_swaps_to_strided_b() {
    // Row-major output forces the swap; the swapped "B" (user's A) has k-stride != 1 → must pack.
    let r = plan_for_opts::<f64>("portable.f64.4x4.direct-b", Opts { row_major_d: true, a_k_stride_not_one: true, ..Opts::default() });
    assert!(r.pack_b_needed);
    check_family_vs_oracle_opts::<f64>("portable.f64.4x4.direct-b", Opts { row_major_d: true, a_k_stride_not_one: true, ..Opts::default() }, 1e-12);
}

#[test]
fn direct_b_plan_has_zero_b_workspace() {
    let r = plan_for_opts::<f64>("portable.f64.4x4.direct-b", Opts::default());
    assert!(!r.pack_b_needed);
}
```
In `common/mod.rs`, define `Opts { beta: f64, alias: bool, conj_c: bool,
conj_d: bool, scattered_d: bool, row_major_d: bool, a_k_stride_not_one:
bool }` with a `Default` (beta 0, all false), and:
- `plan_for_opts::<T>(id, opts) -> ResolvedCall`, which returns the
  per-execute decisions through a new `#[doc(hidden)] pub fn
  tensorcontract::driver_decisions::<T>(plan, rg, c, d, beta) ->
  ResolvedCall { pack_b_needed, direct_c_allowed }`. That function is the
  same code the driver calls, so the tests pin the real decision;
- `check_family_vs_oracle_opts`, which builds the operands and applies
  `Operand::conj()` for the conj flags. For real `T` conjugation is a
  no-op, so the guard's conj branch is reachable only with complex `T`,
  and `validate` guarantees no complex Direct family exists in this phase.
  The conj columns therefore pin that the flags do not break the real
  path. The guard's conj condition is unit-tested directly on
  `driver_decisions::<Complex<f64>>` with a hand-built `ResolvedGemm`
  whose family is `portable.f64.4x4.direct` (the decision function never
  calls the kernel), asserting `direct_c_allowed == false` for `conj_c`
  or `conj_d`.

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p tensorcontract --test direct`
Expected: FAIL. The ids are not registered, and `driver_decisions` is
missing.

- [ ] **Step 3: Implement the portable Direct kernel**

```rust
pub unsafe fn real_direct<R: Real, const MR: usize, const NR: usize>(m: usize, n: usize, k: usize, d: *mut R, rs_d: isize, cs_d: isize,
    a: *const R, a_cs: isize, b: *const R, b_rs: isize, b_cs: isize, alpha_d: R, beta_ab: R, _aux: &UkrAux<R>) {
    let mut acc = [[R::ZERO; MR]; NR];
    for p in 0..k { for j in 0..n { let bj = *b.offset(p as isize * b_rs + j as isize * b_cs);
        for i in 0..m { acc[j][i] = acc[j][i] + *a.offset(p as isize * a_cs + i as isize) * bj; } } }
    for j in 0..n { for i in 0..m {
        let dij = d.offset(i as isize * rs_d + j as isize * cs_d);
        *dij = if alpha_d == R::ZERO { beta_ab * acc[j][i] } else { alpha_d * *dij + beta_ab * acc[j][i] };
    } }
}
```
Register the two descriptors in `portable::families_f64()` (and the f32
equivalents with ids `portable.f32.4x4.direct` / `.direct-b`).

Driver: add a `match rg.family.ukr` at the ukr call site, with the
`UkrFn::Tile` arm unchanged and the `UkrFn::Direct` arm following the
guard and per-tile rules above. Also add `driver_decisions`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p tensorcontract --release && cargo test -p tprims-gemm-kernel`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Driver: Direct C update with plan-level guard and per-tile fallback; direct-B without packing

Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>
Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Induced 1m and 4m families

**Files:**
- Create: `crates/tprims-gemm-kernel/src/induced.rs`
- Modify: `crates/tprims-gemm-kernel/src/{registry.rs,writeback.rs}`, `tensorcontract/src/driver.rs` (4m: four calls per tile)
- Test: `crates/tprims-gemm-kernel/tests/induced.rs`, extend `tensorcontract/tests/resolved.rs`

**Interfaces:**
- Consumes: real `KernelFamily<R>` values with `UkrFn::Tile`; the Task 3
  tile formats.
- Produces:

```rust
pub fn one_m<R: Real>(real: &'static KernelFamily<R>) -> Option<&'static KernelFamily<R>>;  // None if mr or nr odd, or Direct
pub fn four_m<R: Real>(real: &'static KernelFamily<R>) -> Option<&'static KernelFamily<R>>; // None if Direct
```
Ids: `{real_id}.1m-induced` and `{real_id}.4m-induced`; `imp: Induced`;
`inner: Some(real)` via `UkrAux`; `priority = real.priority − 100`;
`allow_auto: false`, because the tc families already provide their own
native 1m, and induced variants are for switching and testing. `Registry`
adds induced variants of **every** registered real family on lookup,
cached in a `OnceLock<Vec<…>>` per dtype built on first lookup.

- **1m induced**:
  - scheme `(OneM, OneE, OneR, TileFormat::OneM)`;
  - `mr' = mr/2`, `nr' = nr`;
  - `kc' = kc/2`, `a_per_k' = 2*2*mr'`, `b_per_k' = 2*nr`,
    `tile_bound = mr*nr`;
  - the ukr is the real ukr itself, called with `2*kc` real k-steps;
  - OneE packing of A gives a real `2mr'×2k` panel, and OneR packing of B
    (`PackFormat::Real` on complex data: `[re; im]` per lane, interleaved
    in k) gives `2k×nr`.

  Before wiring this, verify it against the tensorcontract 1m kernel's
  packed layout. The existing `OneE`/`Real` formats are exactly BLIS 1m
  with the real ukr (issue #23 feasibility note), so the induced 1m ukr
  is a thin wrapper `unsafe fn(kc, a, b, t) { (inner)(2*kc, a, b, t) }`.
  Because a plain `fn` pointer cannot capture `inner`, the wrapper is
  generated per real family as a `static` closure-free fn. To do that,
  store the real kernel in `UkrAux.inner`, and make `TileUkrFn` calls for
  induced families go through `induced::call(fam, kc, a, b, t)`, which
  dispatches on `fam.imp == Induced` and calls `fam.inner_ukr` with the
  scaled k. Implement this by adding `pub inner: Option<&'static
  KernelFamily<R>>` to `KernelFamily`, `None` except for induced families.
  The driver's tile call becomes `induced::tile_call(fam, kc, a, b, t)`:

  ```rust
  #[inline(always)]
  pub unsafe fn tile_call<R: Real>(f: &KernelFamily<R>, kc: usize, a: *const R, b: *const R, t: *mut R) {
      match (f.imp, f.inner, f.complex.map(|c| c.method)) {
          (KernelImpl::Induced, Some(inner), Some(Method::OneM)) => { let UkrFn::Tile(u) = inner.ukr else { unreachable!() }; u(2 * kc, a, b, t) }
          (KernelImpl::Induced, Some(inner), Some(Method::FourM)) => four_m_tile(inner, f, kc, a, b, t),
          _ => { let UkrFn::Tile(u) = f.ukr else { unreachable!() }; u(kc, a, b, t) }
      }
  }
  ```
  For induced families, `f.ukr` is set to the inner ukr, only so the field
  is populated.
- **4m induced**:
  - scheme `(FourM, Planar, Planar, TileFormat::FourM)`;
  - `mr' = mr`, `nr' = nr`;
  - `a_per_k' = 2*mr`, `b_per_k' = 2*nr`, `tile_bound = 4*mr*nr`;
  - `four_m_tile` needs contiguous real panels for Ar, Ai, Br and Bi, but
    planar packing interleaves the planes per k-step (`p*2*vr + {0, vr}`).
    Pack 4m operands with `PackFormat::Planar`. `four_m_tile` then copies
    the four real sub-panels (Ar, Ai with stride `2*mr` per k-step) into a
    plan-owned `4*kc*(mr+nr)` scratch as contiguous real panels, and makes
    four overwrite calls into the four tile planes:
    `(Ar,Br) → plane 0`, `(Ai,Bi) → 1`, `(Ar,Bi) → 2`, `(Ai,Br) → 3`;
  - write-back recombines with the Task 3 `FourM` arm;
  - the extra copy is acceptable: 4m is a switchable correctness variant,
    and its performance is optimization-session material. Record it in the
    induced.rs module doc;
  - the scratch lives in the per-worker workspace (Task 8). Until then,
    allocate it once per `run_strip` call, outside the loops.

- [ ] **Step 1: Write the failing tests**

`crates/tprims-gemm-kernel/tests/induced.rs`:

```rust
use tprims_gemm_kernel::*;
use num_complex::Complex;

#[test]
fn induced_variants_exist_for_the_portable_real_family() {
    let ids: Vec<_> = Registry::families::<Complex<f64>>(CpuFeatures::NONE, true).iter().map(|f| f.id).collect();
    assert!(ids.contains(&"portable.f64.4x4.1m-induced"), "{ids:?}");
    assert!(ids.contains(&"portable.f64.4x4.4m-induced"), "{ids:?}");
}

#[test]
fn induced_families_validate() {
    for f in Registry::families::<Complex<f64>>(CpuFeatures::NONE, true) {
        if f.imp == KernelImpl::Induced { f.validate().unwrap(); assert!(!f.allow_auto); }
    }
}

#[test]
fn one_m_rejects_odd_tiles_and_direct() {
    let direct = Registry::families::<f64>(CpuFeatures::NONE, true).into_iter().find(|f| f.id == "portable.f64.4x4.direct").unwrap();
    assert!(induced::one_m(direct).is_none());
    assert!(induced::four_m(direct).is_none());
}
```
Extend `swapped_orientation_matches_oracle_for_every_family` in
`resolved.rs`: it already iterates all complex families, so the induced
ones are covered at tolerances of 1e-11 for 1m and 1e-10 for 4m. Add a
per-id tolerance map: 3m and 4m get 1e-10.

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p tprims-gemm-kernel --test induced`
Expected: FAIL (ids missing).

- [ ] **Step 3: Implement `induced.rs`**, the registry hook, and the driver
  `tile_call`. Build descriptors with `Box::leak` inside the per-dtype
  `OnceLock`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p tprims-gemm-kernel && cargo test -p tensorcontract --release --test resolved`
Expected: all pass, including every tc real family's induced 1m/4m against
the oracle.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Induced 1m and 4m complex families generated from any real family

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Partition policy

**Files:**
- Create: `crates/tprims-gemm-kernel/src/partition.rs`
- Modify: `crates/tprims-gemm-kernel/src/resolved.rs`, `tensorcontract/src/driver.rs`
- Test: `crates/tprims-gemm-kernel/tests/partition.rs`, `tensorcontract/tests/partition_bitwise.rs`

**Interfaces:**
- Produces:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)] #[non_exhaustive]
pub enum PartitionPolicy {
    /// Today's grid: pm row strips × pn column groups; 0 = decided by the driver's cost model.
    StaticGrid { pm: usize, pn: usize },
    /// Designed (spec §6.1), not implemented: resolution returns SelectError::NotImplemented.
    DynamicTiles { job_m: usize, job_n: usize },
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartitionOpts { pub align_c_lines: bool }
/// Row range of strip r of pm over m rows, in whole MR panels (last strip takes the tail).
pub fn strip(r: usize, pm: usize, m: usize, mr: usize, align: usize) -> (usize, usize);
```
- `strip` reproduces `driver.rs` exactly: `lo = (r*npanels/pm)*mr`, `hi =
  min(((r+1)*npanels/pm)*mr, m)`.
- With `align > mr` (for example `align = lcm(mr, 64/size_of::<T>())`
  elements), boundaries round **down** to multiples of `align`, and the
  last strip keeps the tail.
- The driver's M split calls `strip`. The N split keeps `BPart::ranges`,
  which already works in whole slivers.
- `ResolvedGemm` gains `partition` and `opts: PartitionOpts`.
- `resolve` accepts a `PartitionPolicy` argument; `DynamicTiles` returns
  `Err(SelectError::NotImplemented { what: "DynamicTiles partition" })`.

- [ ] **Step 1: Write the failing tests**

`crates/tprims-gemm-kernel/tests/partition.rs`:

```rust
use tprims_gemm_kernel::partition::*;

#[test]
fn strips_tile_the_rows_on_mr_multiples() {
    for &(m, mr, pm) in &[(1, 8, 1), (37, 8, 3), (1000, 6, 7), (8, 8, 8)] {
        let mut next = 0;
        for r in 0..pm {
            let (lo, hi) = strip(r, pm, m, mr, 0);
            assert_eq!(lo, next); assert!(lo % mr == 0 || lo == m); next = hi;
        }
        assert_eq!(next, m);
    }
}

#[test]
fn aligned_strips_round_to_the_alignment() {
    for r in 1..4 { let (lo, _) = strip(r, 4, 1000, 6, 24); assert_eq!(lo % 24, 0); }
}

#[test]
fn dynamic_tiles_is_not_implemented() {
    let e = tprims_gemm_kernel::resolve_with::<f64>(&tprims_gemm_kernel::KernelChoice::Auto, None,
        tprims_gemm_kernel::CpuFeatures::detect(), 4, None, PartitionPolicy::DynamicTiles { job_m: 16, job_n: 32 }, PartitionOpts::default());
    assert!(matches!(e, Err(tprims_gemm_kernel::SelectError::NotImplemented { .. })));
}
```
(`resolve_with` is the full-argument form; `resolve` from Task 4 calls it
with `StaticGrid { pm: 0, pn: 0 }` and the default opts.)

`tensorcontract/tests/partition_bitwise.rs`. It uses a test-only `Spmd`
that runs `broadcast` on `std::thread::scope` threads, which also exercises
the workspace-`None` path:

```rust
mod common;
use common::*;

#[test]
fn every_width_is_bitwise_identical_to_width_one() {
    tprims_kernel_tensorcontract::register();
    for id in ["tc.scalar.f64.4x4", "portable.f64.4x4", "portable.f64.4x4.direct"] {
        let base = run_with_width::<f64>(id, 1, Shape { m: 203, n: 157, k: 311 });
        for w in [2, 3, 4, 8] {
            let got = run_with_width::<f64>(id, w, Shape { m: 203, n: 157, k: 311 });
            assert!(base.iter().zip(&got).all(|(x, y)| x.to_bits() == y.to_bits()), "{id} width {w}");
        }
    }
}

#[test]
fn aligned_c_lines_are_bitwise_identical_too() {
    let base = run_with_width_opts::<f64>("portable.f64.4x4", 1, Shape { m: 203, n: 157, k: 311 }, true);
    let got = run_with_width_opts::<f64>("portable.f64.4x4", 4, Shape { m: 203, n: 157, k: 311 }, true);
    assert!(base.iter().zip(&got).all(|(x, y)| x.to_bits() == y.to_bits()));
}
```
Define `run_with_width` in `common/mod.rs` using `ScopeSpmd { width }`,
where `broadcast` spawns `p` scoped threads and returns `true`. It calls
`tensorcontract::execute_resolved`.

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p tprims-gemm-kernel --test partition; cargo test -p tensorcontract --release --test partition_bitwise`
Expected: compile errors (`partition`, `resolve_with` and
`execute_resolved` are not public yet).

- [ ] **Step 3: Implement** `partition.rs` and `resolve_with`. In the
  driver, replace the `lo`/`hi` computation in `cell` with
  `partition::strip(r, pm, m, mr, align)`, where `align = if
  rg.opts.align_c_lines { lcm(mr, 64 / size_of::<T>()) } else { 0 }`. Honor
  an explicit `StaticGrid { pm, pn }` (non-zero) by skipping
  `plan.partition_with` and clamping `pm*pn ≤ want`. `pm = pn = 0` keeps
  today's cost model.

- [ ] **Step 4: Run tests**

Run: `cargo test -p tprims-gemm-kernel && cargo test -p tensorcontract --release`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "PartitionPolicy: StaticGrid via one strip() rule, optional C cache-line alignment, DynamicTiles designed only

Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>
Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Workspace arena through `Spmd`

**Files:**
- Create: `crates/tprims-gemm-kernel/src/workspace.rs`
- Modify: `tensorcontract/src/{spmd.rs,driver.rs,buffer.rs}`, `crates/tprims-exec/{Cargo.toml,src/pool.rs,src/exec.rs}`, typed blas/contract plan storage and their `tblis.rs` adapters. Add only the thread-free `tprims-gemm-kernel` dependency to tprims-exec (no driver dependency).
- Test: `crates/tprims-gemm-kernel/tests/workspace.rs`, `tensorcontract/tests/workspace_alloc.rs` (own binary, counting allocator)

**Interfaces:**
- Produces:

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceReq { pub a_reals: usize, pub tile_reals: usize, pub worker_scatter: usize,
                          pub b_reals: usize, pub team_scatter: usize, pub barriers: usize }
/// Page-aligned (4096 B), grow-only, not zeroed, first touched by whoever writes it.
pub struct PageBuf { /* ptr, cap_bytes */ }
impl PageBuf { pub fn ensure(&mut self, bytes: usize) -> *mut u8; pub fn trim(&mut self); }
pub struct TeamSet { pub b: PageBuf, pub scatter: Vec<i64>, pub barriers: Vec<std::sync::Barrier> }
pub trait WorkspaceProvider: Sync {
    /// Run `f` with this thread's per-worker buffers (A block, tile, scatter); falls back to
    /// fresh buffers if already borrowed (re-entrant call).
    fn with_worker(&self, req: &WorkspaceReq, f: &mut dyn FnMut(*mut u8, *mut u8, &mut Vec<i64>));
    /// Take an exclusive team set sized for `req` (never shared by two concurrent executes).
    fn take_team(&self, req: &WorkspaceReq, pm: usize, pn: usize) -> TeamLease<'_>;
    fn trim(&self);
}
pub struct TeamLease<'a> { /* returns the set to the free-list on drop */ }
impl std::ops::DerefMut for TeamLease<'_> { type Target = TeamSet; }
// No process-global ARENA. Construct one ArenaProvider per owner.
```
- `Spmd` gains `fn workspace(&self) -> Option<&dyn WorkspaceProvider>
  { None }`, borrowing the context rather than requiring `'static`.
- Each `tprims_exec::Pool` owns one `ArenaProvider` and exposes a borrowed
  provider accessor. Both blas/contract `ExecSpmd` adapters use that same
  accessor, so two operations on one Pool reuse its storage. Multiple
  wrappers around a borrowed host Rayon pool remain separate owners.
- Serial typed plans retain an owner-local provider and lend it to their
  inline adapter. No ambient serial arena or per-execute construction.
- Worker A/tile/scatter payloads are allocated by their consuming thread.
  TLS caches non-owning handles keyed by provider identity, not merely by
  ThreadId or a recyclable pool address. The provider owns retained worker
  slots; stale TLS handles are discarded. Taking an already-borrowed slot
  uses fresh call-local storage, never blocks on itself.
- Team sets are exclusively leased and returned only to the originating
  provider. No cross-pool free-list. Pool drop releases idle worker/team
  storage even when borrowed host threads stay alive; `trim()` releases
  idle payloads without invalidating active leases. Serial-owner drop has
  the same rule.
- This corrects the earlier process-global ruling. The actual location of
  `ExecSpmd` does not prevent Pool ownership: tprims-exec can depend on the
  thread-free kernel contract without depending on tensorcontract.
  Existing SPMD serialization and re-entrant serial fallback remain intact.
  Pool ownership alone does not guarantee NUMA or L3 placement.
- **`WorkspaceReq`** is computed at `execute_resolved` from `(mc, kc, nc,
  pm, pn, family formats, pack_b_needed)`:
  - `a_reals = panel_len(mc, mr, kc, a_layout)`;
  - `tile_reals = family.tile_bound`, plus the 4m scratch `4*kc*(mr+nr)`
    for FourM;
  - `b_reals = if pack_b_needed { pn * b_group } else { 0 }`;
  - `barriers = if pm > 1 { pn } else { 0 }`.

  A zero entry never allocates. `PageBuf::ensure(0)` returns a dangling,
  aligned pointer.
- **Driver**:
  - with `spmd.workspace() == Some(ws)`:
    - `let lease = ws.take_team(&req, pm, pn);` on the calling thread,
      which also fills `lease.barriers` (a `Barrier` cannot be reset, so
      the free-list keeps sets per `(pm, pn)` and rebuilds barriers only
      when `(pm, pn)` changes);
    - the block-scatter vectors `a_m_bs`, `b_n_bs`, `d_m_bs`, `c_m_bs` and
      `d_n_bs` are built into `lease.scatter`, which stays capacity-reused.
      The caller fills them before broadcast. They are small (`m/mr` and
      `n/nr` entries), and the spec's "cooperative fill" is ruled
      **deferred**: moving the fill into workers needs a barrier the
      StaticGrid path does not have before the first pack. The cost if
      wrong is a caller-side O(m/mr + n/nr) loop, which is negligible;
    - `bp = lease.b.ensure(b_reals * size_of::<R>())`, **not touched**
      here;
    - in `cell(t)`, `ws.with_worker(&req, &mut |ap, tile, _| run_strip(…,
      ap, tile, …))`;
  - with `None`: today's `Panel::new` per call, unchanged (the upstream
    std/pool fallbacks);
  - the serial `p == 1` path uses `ws.with_worker` too when `Some`, and
    `bp` comes from the lease. That removes the double B allocation, since
    the recursive `execute_capped(..., 1, Some(&Inline))` fallback also
    reuses the same lease: pass it down instead of re-deriving it. Change
    the recursive call to a direct `run_strip` on the caller with the
    caller's worker buffers.
- **Instrumentation** (for the first-touch test): `#[cfg(any(test,
  feature = "workspace-trace"))]` records `(thread_id, bytes)` per
  `PageBuf` growth into a `Mutex<Vec<…>>`, readable through
  `workspace::trace_take()`. Feature `workspace-trace` is off by default;
  tests enable it via `dev-dependencies` features.

- [ ] **Step 1: Write the failing tests**

`crates/tprims-gemm-kernel/tests/workspace.rs`:

```rust
use tprims_gemm_kernel::*;

#[test]
fn zero_requirement_never_allocates() {
    let arena = ArenaProvider::default();
    let _ = workspace::trace_take();
    arena.with_worker(&WorkspaceReq::default(), &mut |_, _, _| {});
    let _lease = arena.take_team(&WorkspaceReq::default(), 1, 1);
    assert!(workspace::trace_take().is_empty());
}

#[test]
fn page_aligned_and_reused() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq { a_reals: 1000, tile_reals: 64, ..Default::default() };
    let mut p1 = 0usize; let mut p2 = 0usize;
    arena.with_worker(&req, &mut |a, _, _| p1 = a as usize);
    arena.with_worker(&req, &mut |a, _, _| p2 = a as usize);
    assert_eq!(p1 % 4096, 0); assert_eq!(p1, p2);
}

#[test]
fn reentrant_execute_uses_fresh_buffers() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq { a_reals: 64, ..Default::default() };
    arena.with_worker(&req, &mut |outer, _, _| {
        arena.with_worker(&req, &mut |inner, _, _| assert_ne!(outer, inner));
    });
}

#[test]
fn concurrent_executes_never_share_team_buffers() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq { b_reals: 4096, ..Default::default() };
    let a = arena.take_team(&req, 2, 2);
    let b = arena.take_team(&req, 2, 2);
    assert_ne!(a.b.as_ptr(), b.b.as_ptr());
}

#[test]
fn worker_buffers_are_allocated_on_the_worker() {
    let arena = ArenaProvider::default();
    let _ = workspace::trace_take();
    let req = WorkspaceReq { a_reals: 1 << 16, ..Default::default() };
    let h = std::thread::spawn(move || { arena.with_worker(&req, &mut |_, _, _| {}); std::thread::current().id() });
    let tid = h.join().unwrap();
    assert!(workspace::trace_take().iter().all(|(t, _)| *t == tid));
}
```
`tensorcontract/tests/workspace_alloc.rs`, its own binary with
`#[global_allocator]`:

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
struct Count; static N: AtomicUsize = AtomicUsize::new(0); static BIG: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { N.fetch_add(1, Relaxed); if l.size() >= 1 << 14 { BIG.fetch_add(1, Relaxed); } unsafe { System.alloc(l) } }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { unsafe { System.dealloc(p, l) } }
}
#[global_allocator] static A: Count = Count;
mod common; use common::*;

#[test]
fn steady_state_execute_allocates_nothing_and_direct_b_allocates_no_b() {
    // warm up, then count; Spmd borrows a persistent owner-local provider
    let mut run = prepared_run::<f64>("tc.scalar.f64.4x4", 2, Shape { m: 300, n: 300, k: 300 });
    run(); let before = N.load(Relaxed); run(); run();
    assert_eq!(N.load(Relaxed), before, "steady-state execute allocated");
    let mut run_b = prepared_run::<f64>("portable.f64.4x4.direct-b", 1, Shape { m: 300, n: 300, k: 300 });
    let big0 = BIG.load(Relaxed); run_b();
    // the only big allocation allowed on the first run is the A block (mc*kc*8 B); B would be kc*nc*8
    let r = plan_for_opts::<f64>("portable.f64.4x4.direct-b", Opts::default());
    assert!(!r.pack_b_needed);
    assert!(BIG.load(Relaxed) - big0 <= 1, "direct-B plan allocated a B-sized buffer");
}
```
`prepared_run` builds the plan, resolution and operands once, and returns a
closure that runs `execute_resolved` with an `ArenaSpmd { width, arena }`. That
test `Spmd` returns `Some(&self.arena)` and broadcasts on persistent threads, so
no thread spawns allocate: start `width − 1` parked threads once in
`prepared_run` and hand them work via a `Mutex`/`Condvar`.

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p tprims-gemm-kernel --test workspace; cargo test -p tensorcontract --release --test workspace_alloc`
Expected: compile errors.

- [ ] **Step 3: Implement** `workspace.rs`:
- `PageBuf` via `std::alloc::alloc` with `Layout::from_size_align(bytes,
  4096)`, grown by doubling to the requested size;
- provider-owned worker slots with non-owning TLS lookup, keyed by owner;
  a failed exclusive borrow uses fresh call-local buffers;
- provider-local `Mutex<Vec<TeamSet>>` free-list, exclusive `TeamLease`,
  return-on-drop to the same provider;
- owner drop and `trim()` free idle payloads, not live leases; stale TLS
  lookup handles do not retain storage.

Then add the Pool-owned provider, serial-plan workspace ownership,
`Spmd::workspace`, driver changes and both `ExecSpmd` overrides.

Additional failing tests before implementation:
- same-provider B identity is reused after return;
- different pools never borrow the same retained live B/team set;
- blas and contract adapters on the same Pool borrow the same provider;
- owner drop frees idle team/worker payloads while borrowed host workers
  remain alive (allocation accounting, not pointer reuse after free);
- trim frees idle payloads and cannot free a currently borrowed buffer;
- two owners used on one caller do not share a worker slot;
- serial steady-state allocation and re-entrant checks remain enabled.

- [ ] **Step 4: Run tests**

Run: `cargo test -p tprims-gemm-kernel && cargo test -p tensorcontract --release && cargo test --workspace`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Workspace arena through Spmd: page-aligned reused per-worker buffers first touched by their worker, leased team B sets, no B buffer when B is read in place

Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>
Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `tprims-kernel-gemm` (gemm-f64/f32 Direct families)

**Files:**
- Create: `crates/tprims-kernel-gemm/{Cargo.toml,src/lib.rs}`
- Modify: root `Cargo.toml`, `crates/tprims-blas/Cargo.toml` (feature `kernel-gemm`)
- Test: `crates/tprims-kernel-gemm/tests/parity.rs`

**Interfaces:**
- Consumes: `DirectUkrFn<R>` semantics (Task 5), which match
  `gemm_common::microkernel::MicroKernelFn<R>` up to argument order.
- Produces:
  - `tprims_kernel_gemm::register()`;
  - ids `gemm.{isa}.{f64|f32}.{MR}x{NR}` for `isa ∈ {avx2 (gemm "fma"),
    avx512 (gemm "avx512f", only with gemm's `x86-v4` feature), scalar}`;
  - `MR = MR_DIV_N·N`, `NR` taken from the module constants.
- **Adapter**: one shim per dtype. The gemm function pointer travels
  through a new `opaque: *const ()` field: `KernelFamily.opaque` holds it
  (null by default), and the driver copies it into `UkrAux.opaque` for each
  call. The f64 shim is:

  ```rust
  unsafe fn shim_f64(m: usize, n: usize, k: usize, d: *mut f64, rs_d: isize, cs_d: isize, a: *const f64, a_cs: isize,
      b: *const f64, b_rs: isize, b_cs: isize, alpha_d: f64, beta_ab: f64, aux: &UkrAux<f64>) {
      let f: MicroKernelFn<f64> = unsafe { core::mem::transmute(aux.opaque) };
      let status = if alpha_d == 0.0 { 0 } else if alpha_d == 1.0 { 1 } else { 2 };
      // gemm-common argument order: (m, n, k, dst, lhs, rhs, dst_cs, dst_rs, lhs_cs, rhs_rs, rhs_cs, alpha, beta, alpha_status, conj_dst, conj_lhs, conj_rhs, next_lhs)
      unsafe { f(m, n, k, d, a, b, cs_d, rs_d, a_cs, b_rs, b_cs, alpha_d, beta_ab, status, false, false, false, a) }
  }
  ```
  `KernelFamily` and `UkrAux` gain `opaque: *const ()` (`KernelFamily`
  keeps `Sync` via an `unsafe impl Sync`, justified in a comment: it points
  to a `fn`). Update Task 2's tests' `fam()` helper with `opaque:
  core::ptr::null()`.
- Packed A layout for gemm kernels: column-major MR panel, `a_cs = MR` per
  k-step, i.e. `PackFormat::Real` with `vr = MR`, which is the existing
  packer. gemm's kernels require `dst_rs == 1` for their fast path and
  handle other strides in a slow path, so no restriction is needed.
- `b_access`:
  - `BAccess::Direct { unit_stride: Axis::Col }`: gemm kernels read B
    through `rhs_rs`/`rhs_cs` with `splat`, so they read B in place for any
    stride and are fastest with `rhs_cs == 1`;
  - `b_per_k = NR` for the packed case;
  - `pack_b_needed` is decided as in Task 5.
- `blocks`: `mc = 4·MR·?`. Use `Blocking::derive_at_depth(size_of::<R>(),
  1, 1, 256)` rounded to MR/NR multiples. This carries over the project
  default and is not retuned.
- `Cargo.toml` deps, pinned exactly because the microkernel module is
  undocumented (issue #23): `gemm-common = "=0.19.0"`, `gemm-f64 =
  "=0.19.0"`, `gemm-f32 = "=0.19.0"`.
- ISA gating:
  - `required: avx2+fma` for the fma module;
  - the scalar module has `required: NONE`;
  - avx512 is compiled only with `cfg(feature = "x86-v4")` forwarded to
    gemm-f64.
- `tprims-blas` feature `kernel-gemm = ["dep:tprims-kernel-gemm"]`, which
  calls `tprims_kernel_gemm::register()` in the blas registration
  function (Task 11). This task registers from the test only.

- [ ] **Step 1: Write the failing test**

`crates/tprims-kernel-gemm/tests/parity.rs`:

```rust
use tprims_gemm_kernel::*;

#[test]
fn gemm_families_validate_and_match_portable() {
    tprims_kernel_gemm::register();
    let fams: Vec<_> = Registry::families::<f64>(CpuFeatures::detect(), false).into_iter().filter(|f| f.id.starts_with("gemm.")).collect();
    assert!(!fams.is_empty());
    for f in fams {
        f.validate().unwrap();
        assert_eq!(f.c_update, CUpdate::Direct);
        // Direct call on a full tile and an edge tile vs the portable direct kernel.
        for &(m, n) in &[(f.mr, f.nr), (f.mr - 1, 1)] {
            let k = 17;
            let a: Vec<f64> = (0..f.mr * k).map(|x| ((x * 31 % 17) as f64 - 8.0) / 6.0).collect();
            let b: Vec<f64> = (0..k * f.nr).map(|x| ((x * 41 % 13) as f64 - 6.0) / 4.0).collect();
            let mut d1 = vec![1.0; f.mr * f.nr]; let mut d2 = d1.clone();
            let aux = UkrAux { a_next: a.as_ptr(), b_next: b.as_ptr(), inner: None, opaque: f.opaque };
            let UkrFn::Direct(u) = f.ukr else { panic!() };
            unsafe { u(m, n, k, d1.as_mut_ptr(), 1, f.mr as isize, a.as_ptr(), f.mr as isize, b.as_ptr(), f.nr as isize, 1, 0.5, 2.0, &aux) };
            // naive
            for j in 0..n { for i in 0..m {
                let mut s = 0.0; for p in 0..k { s += a[p * f.mr + i] * b[p * f.nr + j]; }
                d2[j * f.mr + i] = 0.5 * d2[j * f.mr + i] + 2.0 * s;
            } }
            for (x, y) in d1.iter().zip(&d2) { assert!((x - y).abs() <= 1e-12 * y.abs().max(1.0), "{}: {x} vs {y}", f.id); }
        }
    }
}
```
It also runs the full contraction through
`tensorcontract/tests/common::check_family_vs_oracle` for each `gemm.*` id,
with `tprims-kernel-gemm` added as a dev-dependency of `tensorcontract`.
`tensorcontract` must not normally depend on it; dev-dep only is fine, with
no cycle, because `tprims-kernel-gemm` depends only on
`tprims-gemm-kernel`.

- [ ] **Step 2: Run to confirm it fails**

Run: `cargo test -p tprims-kernel-gemm`
Expected: FAIL (crate missing).

- [ ] **Step 3: Implement** the crate. Its `lib.rs` header:

  ```rust
  //! Adapter exposing the public microkernels of `gemm-f64`/`gemm-f32`
  //! (sarah-quinones/gemm, MIT) as tprims `CUpdate::Direct` kernel families.
  //! Calls only: no gemm source is copied (docs/provenance.md). The
  //! microkernel module is undocumented upstream, so the versions are
  //! pinned exactly.
  ```
  Iterate `gemm_f64::microkernel::fma::f64::UKR[i][j]` for all `i`, `j`
  (MR = (i+1)·N, NR = j+1), and keep the largest-area entry as the family
  head with priority 250. Other entries get priority 240 − index and
  `allow_auto: false`. Only the head is Auto-eligible, and even then only
  when the `kernel-gemm` feature enables registration.

- [ ] **Step 4: Run tests**

Run: `cargo test -p tprims-kernel-gemm && cargo test -p tensorcontract --release`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "tprims-kernel-gemm: gemm-f64/f32 public microkernels as Direct families (called, not copied)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: `tprims-kernel-pgx86` engine

**Files:**
- Create: `crates/tprims-kernel-pgx86/{Cargo.toml,src/lib.rs}`
- Test: `crates/tprims-kernel-pgx86/tests/parity.rs`

**Interfaces:**
- Produces:

```rust
pub fn available() -> bool;   // x86_64 && avx2+fma
/// Dense strided GEMM D = alpha·op(A)·op(B) + beta·D via private_gemm_x86::gemm.
/// # Safety: pointers valid for the given shapes/strides; D not aliasing A/B.
pub unsafe fn gemm<T: PgScalar>(m: usize, n: usize, k: usize, d: *mut T, d_rs: isize, d_cs: isize, beta_zero: bool,
    a: *const T, a_rs: isize, a_cs: isize, conj_a: bool, b: *const T, b_rs: isize, b_cs: isize, conj_b: bool,
    alpha: T, n_threads: usize);
pub trait PgScalar: Copy { const DTYPE: private_gemm_x86::DType; }  // f32, f64, Complex<f32>, Complex<f64>
```
- Call:

  ```rust
  private_gemm_x86::gemm(T::DTYPE, IType::U64, instr, m, n, k, d as *mut (), d_rs, d_cs,
      null(), null(), DstKind::Full, if beta_zero { Accum::Replace } else { Accum::Add },
      a as *const (), a_rs, a_cs, conj_a, null(), 0, b as *const (), b_rs, b_cs, conj_b,
      &alpha as *const T as *const (), n_threads)
  ```
  `instr = if avx512f { Avx512 } else { Avx256 }`. Check the argument order
  against `private-gemm-x86-0.1.20/src/lib.rs:1154-1187` while writing: the
  signature is `(dtype, itype, instr, nrows, ncols, depth, dst, dst_rs,
  dst_cs, dst_row_idx, dst_col_idx, dst_kind, beta: Accum, lhs, lhs_rs,
  lhs_cs, conj_lhs, real_diag, diag_stride, rhs, rhs_rs, rhs_cs, conj_rhs,
  alpha, n_threads)`.
- `Accum::Add` computes `D += alpha·A·B`. tprims-blas therefore scales D by
  beta first when `beta ∉ {0, 1}`, exactly as `gemm_raw` does for faer.
- Threading: the caller runs it inside `exec.install(width, |_| …)`, and
  spindle uses `rayon::current_num_threads()` of the installed pool
  (spec §5).
- `Cargo.toml`: `private-gemm-x86 = { version = "=0.1.20",
  default-features = false, features = ["std", "rayon"] }`, under
  `[target.'cfg(target_arch = "x86_64")'.dependencies]`. On other targets,
  `available()` returns false and `gemm` is `unreachable!()` behind a
  `debug_assert!(available())`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn pgx86_matches_naive_for_every_dtype() {
    if !tprims_kernel_pgx86::available() { return; }
    for &(m, n, k) in &[(1, 1, 1), (7, 5, 3), (64, 33, 129)] {
        let a: Vec<f64> = (0..m * k).map(|x| (x % 7) as f64 - 3.0).collect();
        let b: Vec<f64> = (0..k * n).map(|x| (x % 5) as f64 - 2.0).collect();
        let mut d = vec![0.0; m * n];
        unsafe { tprims_kernel_pgx86::gemm(m, n, k, d.as_mut_ptr(), 1, m as isize, true, a.as_ptr(), 1, m as isize, false, b.as_ptr(), 1, k as isize, false, 1.0, 1) };
        for j in 0..n { for i in 0..m { let mut s = 0.0; for p in 0..k { s += a[p * m + i] * b[j * k + p]; } assert_eq!(d[j * m + i], s); } }
    }
}

#[test]
fn pgx86_complex_conj() {
    use num_complex::Complex64 as C;
    if !tprims_kernel_pgx86::available() { return; }
    let (m, n, k) = (5, 4, 3);
    let a: Vec<C> = (0..m * k).map(|x| C::new(x as f64, 1.0 - x as f64)).collect();
    let b: Vec<C> = (0..k * n).map(|x| C::new(2.0 - x as f64, x as f64)).collect();
    let mut d = vec![C::new(0.0, 0.0); m * n];
    unsafe { tprims_kernel_pgx86::gemm(m, n, k, d.as_mut_ptr(), 1, m as isize, true, a.as_ptr(), 1, m as isize, true, b.as_ptr(), 1, k as isize, false, C::new(1.0, 0.0), 1) };
    for j in 0..n { for i in 0..m { let mut s = C::new(0.0, 0.0); for p in 0..k { s += a[p * m + i].conj() * b[j * k + p]; } assert!((d[j * m + i] - s).norm() < 1e-12); } }
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cargo test -p tprims-kernel-pgx86`
Expected: FAIL (crate missing).

- [ ] **Step 3: Implement** it. Its header:

  ```rust
  //! Calls private-gemm-x86 (sarah-quinones, MIT) directly as the tprims
  //! `PrivateGemmX86` matrix engine. Calls only; no source copied. Threads
  //! come from the rayon pool installed by the caller (spindle).
  ```

- [ ] **Step 4: Run tests**

Run: `cargo test -p tprims-kernel-pgx86`
Expected: 2 passed on this host.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "tprims-kernel-pgx86: private_gemm_x86::gemm as a callable matrix engine

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Selection and query in tprims-blas and tprims-contract

**Files:**
- Create: `crates/tprims-blas/src/{engine.rs,select.rs}`
- Modify:
  - `crates/tprims-blas/src/{lib.rs,scalar.rs,gemm.rs,batched.rs,grouped.rs,tblis.rs}`
  - `crates/tprims-blas/Cargo.toml`: add `tprims-gemm-kernel` and
    `tprims-kernel-tensorcontract`, and the optional `tprims-kernel-gemm`
    and `tprims-kernel-pgx86`
  - `crates/tprims-contract/src/{lib.rs,plan.rs,tblis.rs,permute_gemm.rs}`
  - `crates/tprims-blas-capi` and `crates/tprims-contract-capi`: none, since
    the ABI is out of scope; verify they compile
- Test:
  - `crates/tprims-blas/tests/engine_select.rs`
  - `crates/tprims-contract/tests/kernel_select.rs`
  - `crates/tprims-blas/tests/concurrent_families.rs`

**Interfaces:**
- Produces:

```rust
// tprims-blas/src/engine.rs
#[derive(Clone, Debug, PartialEq, Eq, Default)] #[non_exhaustive]
pub enum EngineChoice { #[default] Auto, Faer, PrivateGemmX86, Packed }
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GemmConfig { pub engine: EngineChoice, pub kernel: tprims_gemm_kernel::KernelChoice,
                        pub method: Option<tprims_gemm_kernel::Method> }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] #[non_exhaustive]
pub enum Engine { Faer, PrivateGemmX86, Packed }
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedGemm { pub engine: Engine, pub family_id: Option<&'static str>,
    pub complex: Option<tprims_gemm_kernel::ComplexScheme>, pub mr: usize, pub nr: usize,
    pub mc: usize, pub kc: usize, pub nc: usize, pub partition: Option<tprims_gemm_kernel::PartitionPolicy>,
    pub workspace: Option<tprims_gemm_kernel::WorkspaceReq>, pub batched: Option<Selected> }
impl SelectedGemm { pub fn to_json(&self) -> String; }   // hand-written, no serde dep
pub fn gemm_with<T: Scalar>(exec: &Exec<'_>, cfg: &GemmConfig, alpha: T, a: MatIn<'_, T>, b: MatIn<'_, T>,
    beta: T, c: &mut StridedViewMut<'_, T>) -> Result<SelectedGemm>;
pub fn list_kernels<T: Scalar>() -> Vec<tprims_gemm_kernel::KernelInfo>;  // registers built kernel crates first
```
- `gemm(...)` keeps its signature and return type, and calls
  `gemm_with(&GemmConfig::default(), …)`.
- `gemm_batched`/`gemm_grouped` keep `Selected`. New `*_with(cfg)` variants
  return `SelectedGemm` with `batched: Some(selected)`.
- `SelectedGemm` wraps `Selected` instead of replacing it. **Ruling:**
  changing `Selected`, the public return type used by
  tenferro-cpu-tprims (pinned at d8e565a), would break the pinned
  consumer. The spec's "Selected becomes SelectedGemm" is met additively.
  The cost if wrong is one extra type.
- **Process default engine**:
  - `EngineChoice::Auto` → `Faer` for matrix GEMM, today's behaviour;
  - the env `TPRIMS_GEMM_ENGINE` ∈ {`faer`, `pgx86`, `packed`} is read once
    through `env_once!` in `engine.rs` (the `std` feature exists in
    tprims-blas? If not, add `std = []` as the default feature to
    tprims-blas, required by the macro's cfg);
  - `TPRIMS_GEMM_KERNEL` is handled in `process_default` (Task 4).
- **Errors** (`crate::Error::Select(tprims_gemm_kernel::SelectError)`, a new
  variant):
  - `PrivateGemmX86` when `!tprims_kernel_pgx86::available()` or the
    feature is off → `EngineUnsupported { engine: "pgx86", reason }`;
  - an unknown or unbuilt kernel id → as returned by `resolve`;
  - `DynamicTiles` → `NotImplemented`.
- **Registration**: `pub(crate) fn register_built()`, run once through
  `std::sync::Once`, calls `tprims_kernel_tensorcontract::register()`,
  `#[cfg(feature = "kernel-gemm")] tprims_kernel_gemm::register()`, and
  `tprims_gemm_kernel::register_known_prefix("gemm.", "kernel-gemm")` and
  `("pgx86.", "kernel-pgx86")`.
- **`Scalar`**: the bound `type Re: tensorcontract::KernelSet + …` becomes
  `type Re: tprims_gemm_kernel::Real + tprims_gemm_kernel::RealSlot +
  faer::traits::RealField + Copy`. `Scalar` gains the supertrait
  `tprims_gemm_kernel::Families`.
- **Packed engine for matrix GEMM**: `gemm_with` with `Packed` builds a
  tensorcontract `Plan` for the 2-D problem, exactly as `tblis::run` does
  for one item. It calls `plan.with_kernel(cfg.kernel)`, resolves with
  `width = gemm_width(...)`, and runs `execute_resolved` with `ExecSpmd`.
- **tprims-contract**:
  - `DotGeneral` stays;
  - `ContractPlan::new_with(cfg, a, b, c, conj, strategy, flags, gemm:
    &GemmConfig)` is new, and `ContractPlan::new` passes the default;
  - `TbPlan` stores `ResolvedGemm<T::Real>` resolved at plan creation, so
    errors surface there;
  - `PgPlan` stores the `GemmConfig` for its matrix engine;
  - `ContractPlan::selected_gemm() -> Option<SelectedGemm>` is new;
    `selected()` is unchanged.

- [ ] **Step 1: Write the failing tests**

`crates/tprims-blas/tests/engine_select.rs`:

```rust
use tprims_blas::*;
use tprims_exec::Exec;
use tprims_gemm_kernel::{KernelChoice, SelectError};

fn run(cfg: &GemmConfig) -> Result<(SelectedGemm, Vec<f64>)> {
    let (m, n, k) = (37, 29, 41);
    let a: Vec<f64> = (0..m * k).map(|x| (x % 11) as f64 - 5.0).collect();
    let b: Vec<f64> = (0..k * n).map(|x| (x % 7) as f64 - 3.0).collect();
    let mut c = vec![0.0; m * n];
    let sel = {
        let mut cv = strided_view::StridedViewMut::new(&mut c, &[m, n], &[1, m as isize]).unwrap();
        gemm_with(&Exec::serial(), cfg, 1.0, MatIn::col_major(&a, m, k), MatIn::col_major(&b, k, n), 0.0, &mut cv)?
    };
    Ok((sel, c))
}
```
Match `MatIn`/`StridedViewMut` construction to the existing
`tprims-blas/tests` files, since the spelling may differ: copy it from
`tests/gemm.rs`.

```rust
#[test]
fn every_engine_agrees() {
    let (s0, c0) = run(&GemmConfig::default()).unwrap();
    assert_eq!(s0.engine, Engine::Faer);
    let (s1, c1) = run(&GemmConfig { engine: EngineChoice::Packed, ..Default::default() }).unwrap();
    assert_eq!(s1.engine, Engine::Packed); assert!(s1.family_id.is_some());
    assert_eq!(c0, c1); // integers: exact
    if cfg!(feature = "kernel-pgx86") && tprims_kernel_pgx86_available() {
        let (s2, c2) = run(&GemmConfig { engine: EngineChoice::PrivateGemmX86, ..Default::default() }).unwrap();
        assert_eq!(s2.engine, Engine::PrivateGemmX86); assert_eq!(c0, c2);
    }
}

#[test]
fn forced_kernel_is_used_and_reported() {
    let (s, _) = run(&GemmConfig { engine: EngineChoice::Packed, kernel: KernelChoice::Id("portable.f64.4x4".into()), ..Default::default() }).unwrap();
    assert_eq!(s.family_id, Some("portable.f64.4x4"));
    assert!(s.to_json().contains("\"family_id\":\"portable.f64.4x4\""));
}

#[test]
fn unbuilt_kernel_names_the_feature() {
    if cfg!(feature = "kernel-gemm") { return; }
    let e = run(&GemmConfig { engine: EngineChoice::Packed, kernel: KernelChoice::Id("gemm.avx2.f64.8x6".into()), ..Default::default() }).unwrap_err();
    assert!(matches!(e, Error::Select(SelectError::NotBuilt { feature: "kernel-gemm", .. })), "{e:?}");
}

#[test]
fn cpu_unsupported_is_reported() {
    // force an avx512 tc family on a CPU mask without avx512
    let e = tprims_gemm_kernel::resolve::<f64>(&KernelChoice::Id("tc.avx512.f64.16x6".into()), None,
        tprims_gemm_kernel::CpuFeatures::NONE, 1, None);
    assert!(matches!(e, Err(SelectError::CpuUnsupported { .. }) | Err(SelectError::UnknownId { .. })));
}
```
The id `tc.avx512.f64.16x6` must be taken from the snapshot file created in
Task 3; replace it with the real avx512 f64 head id from
`tests/snapshots/f64_ids.txt`. Then the assertion is `CpuUnsupported` only,
without the `UnknownId` alternative.

`tprims_kernel_pgx86_available()` is a small helper in the test, cfg'd on
the feature.

`crates/tprims-blas/tests/concurrent_families.rs`:

```rust
#[test]
fn two_families_concurrently_on_one_pool() {
    let pool = tprims_exec::Pool::new(4).unwrap();
    let exec = tprims_exec::Exec::rayon(&pool);
    std::thread::scope(|s| {
        for id in ["portable.f64.4x4", "tc.scalar.f64.4x4"] {
            let exec = &exec;
            s.spawn(move || for _ in 0..20 { /* gemm_with Packed + Id(id) on 300x300x300, compare to faer result */ });
        }
    });
}
```
Write the body like `every_engine_agrees`, with `m = n = k = 300` and a
tolerance of `1e-12·‖ref‖`. Check the `Pool::new` constructor name in
`tprims-exec/src/pool.rs` and use its real signature.

`crates/tprims-contract/tests/kernel_select.rs`:

```rust
#[test]
fn contract_plan_resolves_kernel_at_creation_and_reports_it() {
    // A contraction whose permute+GEMM would copy → Auto picks Tblis (P2 rule).
    // new_with(... GemmConfig { kernel: Id("portable.f64.4x4") }) → selected_gemm().family_id == Some("portable.f64.4x4")
    // new_with(... Id("nope")) → Err at creation
}
```
Write it with the same operand setup as the existing test
`auto_uses_tblis_exactly_when_permute_gemm_would_copy` in
`tprims-contract/tests/contract.rs`: copy its layouts verbatim, then make
the two assertions above.

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p tprims-blas --test engine_select --test concurrent_families; cargo test -p tprims-contract --test kernel_select`
Expected: compile errors.

- [ ] **Step 3: Implement** as described in Interfaces. Keep `gemm`,
  `gemm_batched`, `gemm_grouped`, `ContractPlan::new` and `selected()`
  behaviour identical at default config: the existing tests must pass
  unchanged.

- [ ] **Step 4: Run tests with every feature combination**

Run:
```bash
cargo test -p tprims-blas && cargo test -p tprims-blas --features kernel-gemm,kernel-pgx86
cargo test -p tprims-contract && cargo test --workspace
cargo build -p tprims-bundle && cargo test --workspace   # C ABI test links libtprims.so
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "tprims-blas/contract: engine + kernel selection at plan creation, SelectedGemm query, kernel-gemm/kernel-pgx86 features

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Docs, benchmark option, non-regression gate, PR

**Files:**
- Modify: `README.md`, `docs/architecture.md`, `docs/provenance.md`, `docs/decision-log.md`, `benchmarks/benchmarks/tprims/blas/blas.rs`, `benchmarks/Cargo.toml`
- Results: `benchmarks/benchmarks/tprims/{contract,blas}/results/2026-10-XX-gemm-engine-*` (date of the run)

- [ ] **Step 1: Benchmark option**

In `blas.rs`, add `--engine faer|packed|pgx86`, default `faer`, parsed like
`--corpus`. When set, the single-GEMM cases use `gemm_with(&GemmConfig {
engine, .. })`. The CSV gains an `engine` column, and the `family_id` is
printed once per case to stderr. `benchmarks/Cargo.toml` gets the feature
`pgx86 = ["tprims-blas/kernel-pgx86"]`.

Run: `cargo run --release -p tprims-bench --bin blas -- --threads 1 --engine packed --list`
Expected: it lists the cases without error.

- [ ] **Step 2: Docs** (docs-vs-code check, AGENTS.md):
- **`README.md`**:
  - the crate table gets `tprims-gemm-kernel`, `tprims-kernel-tensorcontract`,
    `tprims-kernel-gemm` and `tprims-kernel-pgx86`, with one line each and
    their licenses;
  - the "A separate `tprims-gemm-kernel` … is planned" sentences (lines
    66 and 73) become present tense;
  - the dependency diagram must match `cargo tree -p tprims-contract -e
    normal --depth 2`.
- **`docs/architecture.md:32-55`**:
  - the kernel layer is now real;
  - selection has two levels;
  - `Spmd::workspace`;
  - line 34: the scalar trait now comes from `tprims-gemm-kernel`.
- **`docs/provenance.md`**:
  - a row for the moved files (from tensorcontract at the `git mv`
    commit, Lukas Devos, MIT OR Apache-2.0, `git log --follow`);
  - a note that moved files no longer `git subtree pull` from upstream;
  - `gemm-*` and private-gemm-x86 as called dependencies (MIT, Sarah
    Quiñones, pinned exactly);
  - the per-origin crate rule and the faer MPL-2.0 caveat (spec §9);
  - the revision of the issue #23 no-port rule.
- **`docs/decision-log.md`**: add an entry "2026-09-30 switchable GEMM
  engine: two-level selection, crate split, DynamicTiles designed only,
  pool-owned team workspace and owner-keyed worker TLS (ownership
  correction), SelectedGemm additive (ruling)".

Run: `grep -n "planned" README.md docs/architecture.md | grep -i kernel`
Expected: no stale "planned" mention of `tprims-gemm-kernel`.

- [ ] **Step 3: Full local gate**

Run the Global Constraints gate list in order.
Expected: every step passes. Record the output tails in the ledger.

- [ ] **Step 4: Non-regression benchmark** (spec §10)

Use the `tprims-benchmark` skill (`.agents/skills/tprims-benchmark/SKILL.md`).
Follow it exactly: idle-core pinning, `paired.sh` ABBA, and builds pinned
away from the measured CCD. The baseline is `main` at the merge-base, and
the candidate is this branch.
- `contract --corpus benchmarks/benchmarks/tprims/corpus/tenferro-p1.json`
  at `--threads 1/4/8` (pg and tblis rows; the tblis rows exercise the
  reworked driver);
- `blas --corpus benchmarks/benchmarks/tprims/corpus/tenferro-p1-gemm.json`
  at `1/4/8` (faer and tblis batched rows);
- the gate is calls-weighted workload time within session-to-session
  noise, via `benchmarks/scripts/tblis_decision.py`'s noise rule applied
  to old vs new, with no group more than 5% slower. The script compares
  strategies. For old-vs-new, pass the two result directories as the two
  "strategies", or add a `--pair old new` mode if the script lacks one; if
  added, commit the script change in this task with a test in
  `benchmarks/scripts/`;
- recorded, not gated: `blas --engine packed` and `--engine pgx86` on the
  built-in GEMM cases at 1/4/8.

Expected: the gate passes. If it fails, stop: do not push. Report the
failing groups and the measured ratios to the maintainer. A perf fix is
optimization-session work unless the regression comes from dispatch
overhead introduced here, such as a per-execute lookup or an extra
allocation. Fix that class here with a benchmark rerun.

- [ ] **Step 5: Commit results and docs, push, and open the PR**

```bash
git add -A && git commit -m "Docs, blas --engine, non-regression results for the switchable GEMM engine

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push origin gemm-engine-spec
gh pr create --title "Switchable GEMM engine: kernel-family contract, crate split, engine selection (#23)" --body "$(cat <<'EOF'
Implements docs/superpowers/specs/2026-09-30-switchable-gemm-engine-design.md (v3).

- Kernel layer split out of tensorcontract (history kept with git mv): tprims-gemm-kernel (contract), tprims-kernel-tensorcontract (Lukas Devos's kernels), tprims-kernel-gemm and tprims-kernel-pgx86 (call-only adapters).
- Two-level selection: engine (faer / private-gemm-x86 / packed) and kernel family, resolved once per plan; no env reads or config rebuilds on execute.
- Complex: native interleaved (portable), planar, 1m, 3m, induced 1m/4m.
- Direct C update with plan-level guard, direct-B without a B buffer.
- PartitionPolicy (StaticGrid; DynamicTiles designed only), workspace arena (page-aligned, reused, worker first touch).
- Non-regression: <paste the gate summary table>.

Rulings: <paste the ledger's Ruling: lines>.

Closes nothing yet: optimization continues in a separate session (#23 stays open).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Replace both `<paste …>` markers with the actual gate table and ruling
lines before running; they are filled from the ledger, not left as
placeholders.

- [ ] **Step 6: Watch CI, then merge on green**

Merges were approved for this work after green CI. Watch with `gh run watch`,
merge with `gh pr merge <N> --merge --delete-branch`, then update issue #23
with a summary comment and the list of optimization follow-ups:
- implementing DynamicTiles;
- per-node team pools;
- native SIMD complex kernels;
- ports;
- the C ABI.

Stop there.
