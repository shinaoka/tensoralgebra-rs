//! Native interleaved complex microkernel families for the packed driver
//! (issue #30). Opt-in only: no family here is `allow_auto`, so the default
//! selection is unchanged; select one by id (`cplx.avx2.c64.native.4x4`,
//! `cplx.avx2.c32.native.8x4`) through `KernelChoice::Id`.
//!
//! Project-owned code, MIT OR Apache-2.0, written from the arithmetic
//! `Cr += Ar*Br - Ai*Bi; Ci += Ai*Br + Ar*Bi` with FMA (the interleaved
//! "swap and sign once per A vector" form discussed in issue #30). It is not a
//! port of `gemm-c64`/`gemm-c32` (their `microkernel` module is private) nor
//! of `private-gemm-x86` (its low-level entry points have a custom register
//! ABI); see `docs/provenance.md`. Packing, scatter, write-back, blocking and
//! scheduling are the existing driver's (Lukas Devos's tensorcontract driver
//! and the project's `tprims-gemm-kernel`); this crate only supplies tile
//! kernels and their descriptors.
//!
//! # Tile contract
//!
//! `A` per k step: `[ar0, ai0, ar1, ai1, ...]` (`2*MR` reals), `B` per k step
//! `[br0, bi0, ...]` (`2*NR` reals), `kc` is the logical complex depth. The
//! kernel **overwrites the whole** `2*MR*NR`-real column-major interleaved
//! tile (also for `kc == 0`), never reads it, and reads exactly
//! `2*MR*kc`/`2*NR*kc` reals of the panels. All loads and stores are
//! unaligned.
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

#[cfg(target_arch = "x86_64")]
mod avx2;

use tprims_gemm_kernel::KernelFamily;

/// Native-interleaved complex families for `f32` storage (`c32`), including
/// those whose CPU requirements this machine does not meet. Empty on targets
/// without an implementation.
///
/// # Examples
/// ```
/// for f in tprims_kernel_cplx::families_f32() {
///     assert!(f.validate().is_ok());
///     assert!(!f.allow_auto);
/// }
/// ```
pub fn families_f32() -> &'static [&'static KernelFamily<f32>] {
    #[cfg(target_arch = "x86_64")]
    {
        avx2::families_f32()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        &[]
    }
}

/// Native-interleaved complex families for `f64` storage (`c64`).
///
/// # Examples
/// ```
/// assert!(tprims_kernel_cplx::families_f64().iter().all(|f| f.validate().is_ok()));
/// ```
pub fn families_f64() -> &'static [&'static KernelFamily<f64>] {
    #[cfg(target_arch = "x86_64")]
    {
        avx2::families_f64()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        &[]
    }
}

/// Register the families once. Reads no environment and probes no CPU: ISA
/// availability is checked when a plan selects a family.
///
/// # Examples
/// ```
/// tprims_kernel_cplx::register();
/// tprims_kernel_cplx::register();
/// ```
pub fn register() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        // SAFETY: the manifests are immutable statics; each kernel implements
        // its declared interleaved arithmetic, panel/tile footprints and full
        // tile overwrite, and requires AVX2+FMA as declared (checked at
        // selection). The ISA body is tested against a packed-product oracle.
        unsafe {
            tprims_gemm_kernel::register::<f32>(families_f32);
            tprims_gemm_kernel::register::<f64>(families_f64);
        }
    });
}
