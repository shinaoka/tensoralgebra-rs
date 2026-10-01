# Migration guide, October 2026 source integration

Tracking issue: [#37](https://github.com/tensor4all/tprims-rs/issues/37). This file records what a
consumer of the old crates, family ids, environment variables and C symbols has to change. Each
integration PR extends it; this version covers PR 1 (removals) and PR 2 (kernel consolidation).

## Crates

| Old | New |
|---|---|
| `tprims-gemm-kernel`, `tprims-kernel-tensorcontract`, `tprims-kernel-cplx` | `tprims-kernel` (one crate; Lukas Devos's files keep their history and headers) |
| `tprims-custom-kernel-test` | `tprims_contract_testkit::custom_kernels` plus tests in `tprims-blas` and `tprims-contract` |
| `tprims-linalg`, `tprims-blas-capi`, `tprims-kernel-gemm`, `tprims-kernel-pgx86` | removed in PR 1 (no retained consumer) |

Module paths: `tprims_gemm_kernel::cache` is `tprims_kernel::blocking`; `tprims_kernel_tensorcontract::{scalar, x86, aarch64, KernelSet}`
are `tprims_kernel::kernels::{reference::scalar, x86, aarch64, KernelSet}`; the workspace provider moved to `tprims_exec`.
The built-in families need no registration call: `register()` of the old provider crates is gone and
`tprims_kernel::register` remains only for explicit, unsafe external manifests.

## Family ids

Built-in ids are `{isa}.{dtype}.{scheme}.{MR}x{NR}`, from the descriptor's storage dtype and **logical**
tile geometry. Provenance is carried by `Origin`, not by the id.

| Field | Values |
|---|---|
| isa | `ref`, `avx2`, `avx512`, `neon` (the full CPU-feature mask stays in the descriptor) |
| dtype | storage dtype `f32` `f64` `c32` `c64` |
| scheme | real: `real` (SIMD and portable), `real-scalar` (the scalar menu's real family, distinct from the portable one of the same geometry); direct update: `direct`, `direct-b`; complex: `planar`, `native`, `1m`, `3m`, `4m`; induced: `i1m`, `i4m`, with a `-scalar` suffix when the base is the scalar-menu family |

Induced families take the logical geometry of the induced tile: 1m halves the base real `MR`, 4m keeps it.
Induced families of a **caller's** base family keep the old `<base>.1m-induced` / `<base>.4m-induced` suffix, and
external catalog ids stay provider-owned opaque names. Priorities, tie order and `allow_auto` are unchanged;
the order of the tables below is the Auto order. The scheme is injective per dtype (a test pins this and
the whole x86-64 manifest, `crates/tprims-kernel/tests/snapshots/x86_64_manifest.txt`).

### x86-64 (complete; unavailable ISAs included)

**f32**

| Old | New |
|---|---|
| `tc.avx512.f32.48x8` | `avx512.f32.real.48x8` |
| `tc.avx512.f32.32x8` | `avx512.f32.real.32x8` |
| `tc.avx512.f32.16x10` | `avx512.f32.real.16x10` |
| `tc.avx2.f32.16x6` | `avx2.f32.real.16x6` |
| `tc.avx2.f32.24x4` | `avx2.f32.real.24x4` |
| `tc.avx2.f32.8x8` | `avx2.f32.real.8x8` |
| `tc.scalar.f32.4x4` | `ref.f32.real-scalar.4x4` |
| `portable.f32.4x4` | `ref.f32.real.4x4` |
| `portable.f32.4x4.direct` | `ref.f32.direct.4x4` |
| `portable.f32.4x4.direct-b` | `ref.f32.direct-b.4x4` |

**f64**

| Old | New |
|---|---|
| `tc.avx512.f64.24x8` | `avx512.f64.real.24x8` |
| `tc.avx512.f64.16x8` | `avx512.f64.real.16x8` |
| `tc.avx512.f64.8x8` | `avx512.f64.real.8x8` |
| `tc.avx2.f64.8x6` | `avx2.f64.real.8x6` |
| `tc.avx2.f64.12x4` | `avx2.f64.real.12x4` |
| `tc.avx2.f64.4x8` | `avx2.f64.real.4x8` |
| `tc.scalar.f64.4x4` | `ref.f64.real-scalar.4x4` |
| `portable.f64.4x4` | `ref.f64.real.4x4` |
| `portable.f64.4x4.direct` | `ref.f64.direct.4x4` |
| `portable.f64.4x4.direct-b` | `ref.f64.direct-b.4x4` |

**c32**

| Old | New |
|---|---|
| `tc.avx512.c32.planar.32x6` | `avx512.c32.planar.32x6` |
| `tc.avx512.c32.3m.16x10` | `avx512.c32.3m.16x10` |
| `tc.avx512.c32.planar.48x4` | `avx512.c32.planar.48x4` |
| `tc.avx512.c32.3m.32x4` | `avx512.c32.3m.32x4` |
| `tc.avx512.c32.planar.16x12` | `avx512.c32.planar.16x12` |
| `tc.avx512.c32.3m.48x3` | `avx512.c32.3m.48x3` |
| `tc.avx512.c32.planar.32x5` | `avx512.c32.planar.32x5` |
| `tc.avx512.c32.1m.32x6` | `avx512.c32.1m.32x6` |
| `tc.avx512.c32.1m.24x8` | `avx512.c32.1m.24x8` |
| `tc.avx512.c32.1m.16x8` | `avx512.c32.1m.16x8` |
| `tc.avx512.c32.1m.8x12` | `avx512.c32.1m.8x12` |
| `tc.avx2.c32.planar.8x5` | `avx2.c32.planar.8x5` |
| `tc.avx2.c32.3m.8x4` | `avx2.c32.3m.8x4` |
| `tc.avx512.f32.48x8.1m-induced` | `avx512.c32.i1m.24x8` |
| `tc.avx512.f32.48x8.4m-induced` | `avx512.c32.i4m.48x8` |
| `tc.avx2.c32.planar.16x2` | `avx2.c32.planar.16x2` |
| `tc.avx2.c32.3m.16x2` | `avx2.c32.3m.16x2` |
| `tc.avx512.f32.32x8.1m-induced` | `avx512.c32.i1m.16x8` |
| `tc.avx512.f32.32x8.4m-induced` | `avx512.c32.i4m.32x8` |
| `tc.avx512.f32.16x10.1m-induced` | `avx512.c32.i1m.8x10` |
| `tc.avx512.f32.16x10.4m-induced` | `avx512.c32.i4m.16x10` |
| `tc.avx2.c32.1m.8x6` | `avx2.c32.1m.8x6` |
| `tc.avx2.c32.1m.12x4` | `avx2.c32.1m.12x4` |
| `tc.avx2.c32.1m.4x8` | `avx2.c32.1m.4x8` |
| `cplx.avx2.c32.native.8x4` | `avx2.c32.native.8x4` |
| `tc.avx2.f32.16x6.1m-induced` | `avx2.c32.i1m.8x6` |
| `tc.avx2.f32.16x6.4m-induced` | `avx2.c32.i4m.16x6` |
| `tc.avx2.f32.24x4.1m-induced` | `avx2.c32.i1m.12x4` |
| `tc.avx2.f32.24x4.4m-induced` | `avx2.c32.i4m.24x4` |
| `tc.avx2.f32.8x8.1m-induced` | `avx2.c32.i1m.4x8` |
| `tc.avx2.f32.8x8.4m-induced` | `avx2.c32.i4m.8x8` |
| `tc.scalar.c32.planar.2x4` | `ref.c32.planar.2x4` |
| `tc.scalar.c32.3m.2x4` | `ref.c32.3m.2x4` |
| `portable.c32.native.4x4` | `ref.c32.native.4x4` |
| `tc.scalar.c32.1m.2x4` | `ref.c32.1m.2x4` |
| `portable.f32.4x4.1m-induced` | `ref.c32.i1m.2x4` |
| `portable.f32.4x4.4m-induced` | `ref.c32.i4m.4x4` |
| `tc.scalar.f32.4x4.1m-induced` | `ref.c32.i1m-scalar.2x4` |
| `tc.scalar.f32.4x4.4m-induced` | `ref.c32.i4m-scalar.4x4` |

**c64**

| Old | New |
|---|---|
| `tc.avx512.c64.planar.16x6` | `avx512.c64.planar.16x6` |
| `tc.avx512.c64.3m.8x10` | `avx512.c64.3m.8x10` |
| `tc.avx512.c64.planar.24x3` | `avx512.c64.planar.24x3` |
| `tc.avx512.c64.3m.16x4` | `avx512.c64.3m.16x4` |
| `tc.avx512.c64.planar.8x8` | `avx512.c64.planar.8x8` |
| `tc.avx512.c64.3m.24x3` | `avx512.c64.3m.24x3` |
| `tc.avx512.c64.1m.12x8` | `avx512.c64.1m.12x8` |
| `tc.avx512.c64.1m.16x6` | `avx512.c64.1m.16x6` |
| `tc.avx512.c64.1m.8x8` | `avx512.c64.1m.8x8` |
| `tc.avx2.c64.planar.4x5` | `avx2.c64.planar.4x5` |
| `tc.avx2.c64.3m.4x4` | `avx2.c64.3m.4x4` |
| `tc.avx512.f64.24x8.1m-induced` | `avx512.c64.i1m.12x8` |
| `tc.avx512.f64.24x8.4m-induced` | `avx512.c64.i4m.24x8` |
| `tc.avx2.c64.planar.8x2` | `avx2.c64.planar.8x2` |
| `tc.avx2.c64.3m.8x2` | `avx2.c64.3m.8x2` |
| `tc.avx512.f64.16x8.1m-induced` | `avx512.c64.i1m.8x8` |
| `tc.avx512.f64.16x8.4m-induced` | `avx512.c64.i4m.16x8` |
| `tc.avx512.f64.8x8.1m-induced` | `avx512.c64.i1m.4x8` |
| `tc.avx512.f64.8x8.4m-induced` | `avx512.c64.i4m.8x8` |
| `tc.avx2.c64.1m.4x6` | `avx2.c64.1m.4x6` |
| `tc.avx2.c64.1m.6x4` | `avx2.c64.1m.6x4` |
| `tc.avx2.c64.1m.2x8` | `avx2.c64.1m.2x8` |
| `cplx.avx2.c64.native.4x4` | `avx2.c64.native.4x4` |
| `tc.avx2.f64.8x6.1m-induced` | `avx2.c64.i1m.4x6` |
| `tc.avx2.f64.8x6.4m-induced` | `avx2.c64.i4m.8x6` |
| `tc.avx2.f64.12x4.1m-induced` | `avx2.c64.i1m.6x4` |
| `tc.avx2.f64.12x4.4m-induced` | `avx2.c64.i4m.12x4` |
| `tc.avx2.f64.4x8.1m-induced` | `avx2.c64.i1m.2x8` |
| `tc.avx2.f64.4x8.4m-induced` | `avx2.c64.i4m.4x8` |
| `tc.scalar.c64.planar.2x4` | `ref.c64.planar.2x4` |
| `tc.scalar.c64.3m.2x4` | `ref.c64.3m.2x4` |
| `portable.c64.native.4x4` | `ref.c64.native.4x4` |
| `tc.scalar.c64.1m.2x4` | `ref.c64.1m.2x4` |
| `portable.f64.4x4.1m-induced` | `ref.c64.i1m.2x4` |
| `portable.f64.4x4.4m-induced` | `ref.c64.i4m.4x4` |
| `tc.scalar.f64.4x4.1m-induced` | `ref.c64.i1m-scalar.2x4` |
| `tc.scalar.f64.4x4.4m-induced` | `ref.c64.i4m-scalar.4x4` |

### AArch64 (NEON)

The NEON menus (`cfg_neon_f64`, `cfg_neon_f32`) are compiled only on aarch64, so this table is derived from the
menu definitions and the same id function (unit-tested with NEON inputs); CI on macOS arm64 pins the
`f64` real list. The portable `ref.*` families of the x86-64 tables (`ref.f64.real.4x4`, `ref.f64.direct.4x4`,
`ref.c64.native.4x4`, ...) and the scalar menu are the same on every target.

**f64 / c64 (menu order; Auto order interleaves them by priority)**

| Old | New |
|---|---|
| `tc.neon.f64.16x3` | `neon.f64.real.16x3` |
| `tc.neon.f64.4x8` | `neon.f64.real.4x8` |
| `tc.neon.f64.8x6` | `neon.f64.real.8x6` |
| `tc.neon.f64.6x8` | `neon.f64.real.6x8` |
| `tc.neon.c64.planar.4x6` | `neon.c64.planar.4x6` |
| `tc.neon.c64.planar.4x5` | `neon.c64.planar.4x5` |
| `tc.neon.c64.planar.2x12` | `neon.c64.planar.2x12` |
| `tc.neon.c64.1m.2x8` | `neon.c64.1m.2x8` |
| `tc.neon.c64.1m.4x6` | `neon.c64.1m.4x6` |
| `tc.neon.c64.1m.3x8` | `neon.c64.1m.3x8` |
| `tc.neon.c64.3m.2x8` | `neon.c64.3m.2x8` |
| `tc.neon.c64.3m.4x3` | `neon.c64.3m.4x3` |
| `tc.neon.c64.3m.4x4` | `neon.c64.3m.4x4` |
| `tc.neon.f64.16x3.1m-induced` | `neon.c64.i1m.8x3` |
| `tc.neon.f64.16x3.4m-induced` | `neon.c64.i4m.16x3` |
| `tc.neon.f64.4x8.1m-induced` | `neon.c64.i1m.2x8` |
| `tc.neon.f64.4x8.4m-induced` | `neon.c64.i4m.4x8` |
| `tc.neon.f64.8x6.1m-induced` | `neon.c64.i1m.4x6` |
| `tc.neon.f64.8x6.4m-induced` | `neon.c64.i4m.8x6` |
| `tc.neon.f64.6x8.1m-induced` | `neon.c64.i1m.3x8` |
| `tc.neon.f64.6x8.4m-induced` | `neon.c64.i4m.6x8` |

**f32 / c32 (menu order; Auto order interleaves them by priority)**

| Old | New |
|---|---|
| `tc.neon.f32.8x8` | `neon.f32.real.8x8` |
| `tc.neon.f32.16x6` | `neon.f32.real.16x6` |
| `tc.neon.f32.24x4` | `neon.f32.real.24x4` |
| `tc.neon.f32.12x8` | `neon.f32.real.12x8` |
| `tc.neon.c32.planar.8x6` | `neon.c32.planar.8x6` |
| `tc.neon.c32.planar.8x5` | `neon.c32.planar.8x5` |
| `tc.neon.c32.planar.4x12` | `neon.c32.planar.4x12` |
| `tc.neon.c32.1m.8x6` | `neon.c32.1m.8x6` |
| `tc.neon.c32.1m.6x8` | `neon.c32.1m.6x8` |
| `tc.neon.c32.1m.12x4` | `neon.c32.1m.12x4` |
| `tc.neon.c32.3m.4x8` | `neon.c32.3m.4x8` |
| `tc.neon.c32.3m.8x3` | `neon.c32.3m.8x3` |
| `tc.neon.c32.3m.8x4` | `neon.c32.3m.8x4` |
| `tc.neon.f32.8x8.1m-induced` | `neon.c32.i1m.4x8` |
| `tc.neon.f32.8x8.4m-induced` | `neon.c32.i4m.8x8` |
| `tc.neon.f32.16x6.1m-induced` | `neon.c32.i1m.8x6` |
| `tc.neon.f32.16x6.4m-induced` | `neon.c32.i4m.16x6` |
| `tc.neon.f32.24x4.1m-induced` | `neon.c32.i1m.12x4` |
| `tc.neon.f32.24x4.4m-induced` | `neon.c32.i4m.24x4` |
| `tc.neon.f32.12x8.1m-induced` | `neon.c32.i1m.6x8` |
| `tc.neon.f32.12x8.4m-induced` | `neon.c32.i4m.12x8` |

## Environment variables and engine selection (PR 2)

No library reads the environment any more. Every knob is an explicit input:

| Removed | Replacement |
|---|---|
| `TENSORCONTRACT_THREADS` | none: threads come only from the `tprims_exec::Exec` passed to `run_with` (plan width is `Plan::with_threads`, a planning hint) |
| `TENSORCONTRACT_POOL` | none (the parked-thread pool is gone) |
| `TENSORCONTRACT_KERNEL` | `tprims_kernel::Tuning::kernel_force` (`Plan::with_tuning`, `GemmConfig::tuning`) |
| `TENSORCONTRACT_COMPLEX` | `Plan::with_complex_method` (`GemmConfig::method`) |
| `TENSORCONTRACT_MC` `_KC` `_NC` `_MC_PCT` `_NC_PCT` | `Tuning::blocking` (`BlockingOverride`) |
| `TENSORCONTRACT_KC_COUPLE` | `Tuning::kc_couple` |
| `TENSORCONTRACT_BLOCKMODEL` | `Tuning::block_model` |
| `TENSORCONTRACT_WRITEBACK` | `Tuning::writeback_gather` |
| `TENSORCONTRACT_L3_DOMAINS` | `Plan::with_l3_domains` (`GemmConfig::l3_domains`) |
| `TENSORCONTRACT_ORIENT` | `Plan::with_orientation(Orient)` (`GemmConfig::orientation`) |
| `TENSORCONTRACT_ROWBLOCK` | `Plan::with_row_block(RowBlock)` (`GemmConfig::row_block`) |
| `TENSORCONTRACT_PARTITION` | `Plan::with_partition_mode(PartitionMode)` (`GemmConfig::partition_mode`); the grid/dynamic request stays `Plan::with_partition` |
| `TPRIMS_GEMM_KERNEL` | `Plan::with_kernel(KernelChoice::Id(..))` / `GemmConfig::kernel` |
| `TPRIMS_GEMM_ENGINE` | `GemmConfig::engine` (`default_engine()` is gone; `Auto` is faer) |

`tcbench` still accepts the `TENSORCONTRACT_*` spellings of the table (except `THREADS` and `POOL`) and parses
them into plan configuration; an unparseable value or a removed variable is an error. The `tprims-bench`
binaries refuse to run with any removed variable set.

## Removed C symbols (PR 1)

`tprims_blas_gemm`, `tprims_blas_gemm_batched` and `tprims/blas.h` are gone; `tprims_has_part("blas")` returns 0.
