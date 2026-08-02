//! Micro-kernels and cache blocking parameters.
//!
//! # Panel formats
//!
//! Both the real and the complex path use the *same* packed panel shape, with
//! the complex path simply carrying two real planes per k-step:
//!
//! ```text
//! real A sliver:     [ a_0 .. a_{MR-1} ]                              per k
//! complex A sliver:  [ re_0 .. re_{MR-1} | im_0 .. im_{MR-1} ]        per k
//! real B sliver:     [ b_0 .. b_{NR-1} ]                              per k
//! complex B sliver:  [ re_0 .. re_{NR-1} | im_0 .. im_{NR-1} ]        per k
//! ```
//!
//! This is the planar (split-complex) packing that the project is built
//! around. In BLIS terminology it is "1r" format for *both* operands, driven
//! by a genuinely complex micro-kernel, as opposed to BLIS/TBLIS's 1m method
//! which uses "1e" for A (four reals per complex element, a 2x2 block
//! expansion) and "1r" for B, driven by a real micro-kernel. The packed A
//! panel is therefore half the size here, and the kernel needs no in-register
//! shuffles to separate real and imaginary parts.
//!
//! # Kernel contract
//!
//! A micro-kernel *overwrites* an `MR x NR` accumulator tile with the panel
//! product; `alpha`, `beta` and the scattered write-back to `C`/`D` are
//! applied afterwards by [`crate::writeback`]. Keeping them separate is what
//! lets one kernel serve the regular fast path, the gather path and every edge
//! block without duplication.
//!
//! Accumulator tile layout is column-major within the tile: `ab[j * MR + i]`.
//! For the complex path the imaginary plane follows the real one, at offset
//! `MR * NR`.

use crate::element::Real;

pub mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

/// A micro-kernel: register-blocked panel-panel product.
#[derive(Clone, Copy)]
pub struct Ukr<T> {
    pub mr: usize,
    pub nr: usize,
    /// # Safety
    /// `a` must address `mr * kc` (real) or `2 * mr * kc` (complex) values,
    /// `b` likewise with `nr`, and `ab` must address `mr * nr` (real) or
    /// `2 * mr * nr` (complex) values.
    pub func: unsafe fn(kc: usize, a: *const T, b: *const T, ab: *mut T),
    pub name: &'static str,
}

impl<T> core::fmt::Debug for Ukr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Ukr({} {}x{})", self.name, self.mr, self.nr)
    }
}

/// Cache blocking parameters, in elements of the *storage* type.
///
/// * `mc x kc` is the packed A block, sized for L2.
/// * `kc x nc` is the packed B block, sized for L3.
/// * `kc x nr` is the B sliver streamed through L1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocking {
    pub mc: usize,
    pub kc: usize,
    pub nc: usize,
}

/// Everything the driver needs for one element type and domain.
#[derive(Clone, Copy, Debug)]
pub struct KernelConfig<T> {
    pub ukr: Ukr<T>,
    pub blk: Blocking,
}

impl<T> KernelConfig<T> {
    /// Apply any environment override, then round `mc`/`nc` to whole multiples
    /// of the register block (the driver's loop arithmetic relies on that).
    fn normalise(mut self) -> Self {
        if let Some(o) = env_blocking() {
            if let Some(v) = o.mc {
                self.blk.mc = v;
            }
            if let Some(v) = o.kc {
                self.blk.kc = v;
            }
            if let Some(v) = o.nc {
                self.blk.nc = v;
            }
        }
        let blk = self.blk;
        self.with_blocking(blk)
    }

    /// Replace the blocking parameters, re-imposing the register-block
    /// alignment invariant.
    pub fn with_blocking(mut self, blk: Blocking) -> Self {
        self.blk.mc = blk.mc.next_multiple_of(self.ukr.mr).max(self.ukr.mr);
        self.blk.nc = blk.nc.next_multiple_of(self.ukr.nr).max(self.ukr.nr);
        self.blk.kc = blk.kc.max(1);
        self
    }
}

#[derive(Clone, Copy)]
struct BlockingOverride {
    mc: Option<usize>,
    kc: Option<usize>,
    nc: Option<usize>,
}

/// `TENSORCONTRACT_MC` / `_KC` / `_NC` override the cache blocking. Read once
/// per process; used for Phase 4 parameter sweeps and to exercise every level
/// of the loop nest on small test problems.
fn env_blocking() -> Option<BlockingOverride> {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static ENV: OnceLock<Option<BlockingOverride>> = OnceLock::new();
        *ENV.get_or_init(|| {
            let get = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
            let o = BlockingOverride {
                mc: get("TENSORCONTRACT_MC"),
                kc: get("TENSORCONTRACT_KC"),
                nc: get("TENSORCONTRACT_NC"),
            };
            (o.mc.is_some() || o.kc.is_some() || o.nc.is_some()).then_some(o)
        })
    }
    #[cfg(not(feature = "std"))]
    {
        None
    }
}

/// Real scalar types for which the engine has a micro-kernel.
///
/// `f32` and `f64` get runtime-dispatched vectorised kernels. Any other
/// [`Real`] type can opt in with the generic scalar kernels, e.g.
///
/// ```ignore
/// impl KernelSet for MyDual {
///     fn config_real() -> KernelConfig<Self> { scalar::config_real::<Self, 4, 4>() }
///     fn config_cplx() -> KernelConfig<Self> { scalar::config_cplx::<Self, 4, 4>() }
/// }
/// ```
pub trait KernelSet: Real + Sized {
    fn config_real() -> KernelConfig<Self>;
    fn config_cplx() -> KernelConfig<Self>;
}

/// The register block `(MR, NR)` and cache blocking the engine will use for
/// element type `T`. Exposed for diagnostics and for harnesses that want to
/// report block-scatter regularity at the same granularity the engine sees.
pub fn selected_config<T>() -> (usize, usize, Blocking)
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    let cfg = if T::IS_COMPLEX {
        <T::Real as KernelSet>::config_cplx()
    } else {
        <T::Real as KernelSet>::config_real()
    };
    (cfg.ukr.mr, cfg.ukr.nr, cfg.blk)
}

/// Name of the micro-kernel selected for `T`.
pub fn selected_kernel_name<T>() -> &'static str
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    if T::IS_COMPLEX {
        <T::Real as KernelSet>::config_cplx().ukr.name
    } else {
        <T::Real as KernelSet>::config_real().ukr.name
    }
}

/// Force the portable scalar kernels regardless of CPU features.
/// Set `TENSORCONTRACT_KERNEL=scalar` to compare against the reference path.
fn force_scalar() -> bool {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static FORCE: OnceLock<bool> = OnceLock::new();
        *FORCE.get_or_init(|| {
            std::env::var("TENSORCONTRACT_KERNEL")
                .map(|v| v.eq_ignore_ascii_case("scalar"))
                .unwrap_or(false)
        })
    }
    #[cfg(not(feature = "std"))]
    {
        false
    }
}

impl KernelSet for f64 {
    fn config_real() -> KernelConfig<Self> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !force_scalar() {
            if let Some(c) = x86::config_real_f64() {
                return c.normalise();
            }
        }
        scalar::config_real::<f64, 4, 4>().normalise()
    }

    fn config_cplx() -> KernelConfig<Self> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !force_scalar() {
            if let Some(c) = x86::config_cplx_f64() {
                return c.normalise();
            }
        }
        scalar::config_cplx::<f64, 4, 4>().normalise()
    }
}

impl KernelSet for f32 {
    fn config_real() -> KernelConfig<Self> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !force_scalar() {
            if let Some(c) = x86::config_real_f32() {
                return c.normalise();
            }
        }
        scalar::config_real::<f32, 4, 4>().normalise()
    }

    fn config_cplx() -> KernelConfig<Self> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !force_scalar() {
            if let Some(c) = x86::config_cplx_f32() {
                return c.normalise();
            }
        }
        scalar::config_cplx::<f32, 4, 4>().normalise()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel must agree with the portable scalar reference.
    fn check_real<T: KernelSet>(tol: f64) {
        let cfg = T::config_real();
        let (mr, nr) = (cfg.ukr.mr, cfg.ukr.nr);
        let kc = 37usize;
        let a: Vec<T> = (0..mr * kc)
            .map(|i| T::from_f64(((i * 37 % 19) as f64 - 9.0) / 7.0))
            .collect();
        let b: Vec<T> = (0..nr * kc)
            .map(|i| T::from_f64(((i * 53 % 23) as f64 - 11.0) / 5.0))
            .collect();
        let mut got = vec![T::ZERO; mr * nr];
        unsafe { (cfg.ukr.func)(kc, a.as_ptr(), b.as_ptr(), got.as_mut_ptr()) };

        for j in 0..nr {
            for i in 0..mr {
                let mut want = 0.0f64;
                for p in 0..kc {
                    want += a[p * mr + i].to_f64() * b[p * nr + j].to_f64();
                }
                let g = got[j * mr + i].to_f64();
                assert!(
                    (g - want).abs() <= tol * want.abs().max(1.0),
                    "{} real mismatch at ({i},{j}): {g} vs {want}",
                    cfg.ukr.name
                );
            }
        }
    }

    fn check_cplx<T: KernelSet>(tol: f64) {
        let cfg = T::config_cplx();
        let (mr, nr) = (cfg.ukr.mr, cfg.ukr.nr);
        let kc = 29usize;
        let a: Vec<T> = (0..2 * mr * kc)
            .map(|i| T::from_f64(((i * 31 % 17) as f64 - 8.0) / 6.0))
            .collect();
        let b: Vec<T> = (0..2 * nr * kc)
            .map(|i| T::from_f64(((i * 41 % 13) as f64 - 6.0) / 4.0))
            .collect();
        let mut got = vec![T::ZERO; 2 * mr * nr];
        unsafe { (cfg.ukr.func)(kc, a.as_ptr(), b.as_ptr(), got.as_mut_ptr()) };

        for j in 0..nr {
            for i in 0..mr {
                let (mut wr, mut wi) = (0.0f64, 0.0f64);
                for p in 0..kc {
                    let ar = a[p * 2 * mr + i].to_f64();
                    let ai = a[p * 2 * mr + mr + i].to_f64();
                    let br = b[p * 2 * nr + j].to_f64();
                    let bi = b[p * 2 * nr + nr + j].to_f64();
                    wr += ar * br - ai * bi;
                    wi += ar * bi + ai * br;
                }
                let gr = got[j * mr + i].to_f64();
                let gi = got[mr * nr + j * mr + i].to_f64();
                assert!(
                    (gr - wr).abs() <= tol * wr.abs().max(1.0),
                    "{} cplx re mismatch at ({i},{j}): {gr} vs {wr}",
                    cfg.ukr.name
                );
                assert!(
                    (gi - wi).abs() <= tol * wi.abs().max(1.0),
                    "{} cplx im mismatch at ({i},{j}): {gi} vs {wi}",
                    cfg.ukr.name
                );
            }
        }
    }

    #[test]
    fn kernels_match_reference_f64() {
        check_real::<f64>(1e-12);
        check_cplx::<f64>(1e-12);
    }

    #[test]
    fn kernels_match_reference_f32() {
        check_real::<f32>(1e-4);
        check_cplx::<f32>(1e-4);
    }
}
