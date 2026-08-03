//! Micro-kernels, complex methods, and cache blocking parameters.
//!
//! # The three complex methods
//!
//! Complex contraction can be induced from real arithmetic in several ways.
//! This crate implements three and lets the caller choose, because which one
//! wins depends on the shape, the element type and the machine — and because
//! the whole point of the project is to be able to measure them against each
//! other on equal footing. Select with [`ComplexMethod`], per plan via
//! [`crate::Plan::with_complex_method`] or globally via the
//! `TENSORCONTRACT_COMPLEX` environment variable (`planar` | `1m` | `3m`).
//!
//! All three share the *same* index analysis, scatter machinery, five-loop
//! driver and write-back path. They differ only in what packing emits, what
//! the micro-kernel computes, and how the accumulator tile is read back.
//!
//! ## Packed panel formats
//!
//! With `VR` the register block along the packed axis (`MR` for A, `NR` for
//! B), one *logical* k-step of a sliver holds:
//!
//! ```text
//! real            [ x_0 .. x_{VR-1} ]                                   VR
//!
//! planar          [ re_0..re_{VR-1} | im_0..im_{VR-1} ]                2*VR
//!
//! 1m, operand A   [ re_0, im_0, re_1, im_1, ...                        4*MR
//!   ("1e")          | -im_0, re_0, -im_1, re_1, ... ]
//! 1m, operand B   [ re_0..re_{NR-1} | im_0..im_{NR-1} ]                2*NR
//!   ("1r")          (bit-identical to planar)
//!
//! 3m              [ re_0..re_{VR-1} | im_0..im_{VR-1}                  3*VR
//!                   | (re+im)_0..(re+im)_{VR-1} ]
//! ```
//!
//! ## What each method costs
//!
//! | | reals packed per complex elt (A / B) | real FMAs per k per tile | accumulator planes |
//! |---|---|---|---|
//! | planar | 2 / 2 | `4*MR*NR` | 2 |
//! | 1m | **4** / 2 | `4*MR*NR` | 2 (as `2*MR x NR` real) |
//! | 3m | 3 / 3 | **`3*MR*NR`** | 3 |
//!
//! * **planar** — real and imaginary parts separated during the scatter pass
//!   that packing has to make anyway, then a fused complex micro-kernel over
//!   the two planes. No in-register shuffles, and the smallest packed A panel.
//! * **1m** — Van Zee's induced method, what BLIS and TBLIS 2.x use. Each
//!   complex element of A becomes the real `2x2` block `[[re, -im], [im, re]]`
//!   and each of B a `2x1` column, so a *single real* micro-kernel of shape
//!   `2*MR x NR` over `2*KC` steps produces the complex product. Costs 2x the
//!   packed-A footprint and 2x the packing store traffic; needs no complex
//!   kernel at all.
//! * **3m** — Karatsuba. `M1 = Ar*Br`, `M2 = Ai*Bi`, `M3 = (Ar+Ai)*(Br+Bi)`,
//!   then `Cr = M1 - M2` and `Ci = M3 - M1 - M2`. **25% fewer flops**, and the
//!   sum planes are formed for free during packing. The error bound is weaker
//!   than the other two (it is relative to `|Ar||Br| + |Ai||Bi|` rather than to
//!   the complex magnitudes, so it can lose relative accuracy under
//!   cancellation), which is why it is opt-in rather than the default.
//!
//! # Kernel contract
//!
//! A micro-kernel *overwrites* an accumulator tile with the panel product;
//! `alpha`, `beta`, the scattered write-back to `C`/`D` and any recombination
//! of planes are applied afterwards by [`crate::writeback`]. Keeping them
//! separate is what lets one kernel serve the regular fast path, the gather
//! path and every edge block without duplication.
//!
//! Kernels take the *logical* `kc` (in complex elements for complex methods)
//! and know their own panel layout, so the driver does not have to.

use crate::element::Real;

pub mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

/// How to induce complex arithmetic from real micro-kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ComplexMethod {
    /// Planar / split-complex packing with a fused complex micro-kernel.
    #[default]
    Planar,
    /// Van Zee's 1m induced method: "1e" packing for A, "1r" for B, and a
    /// single real micro-kernel. This is what BLIS and TBLIS 2.x do.
    OneM,
    /// Karatsuba 3m: three real products instead of four. 25% fewer flops,
    /// weaker error bound.
    ThreeM,
}

impl ComplexMethod {
    pub const ALL: [ComplexMethod; 3] = [
        ComplexMethod::Planar,
        ComplexMethod::OneM,
        ComplexMethod::ThreeM,
    ];

    pub fn parse(s: &str) -> Option<ComplexMethod> {
        match s.trim().to_ascii_lowercase().as_str() {
            "planar" | "split" => Some(ComplexMethod::Planar),
            "1m" | "onem" => Some(ComplexMethod::OneM),
            "3m" | "threem" | "karatsuba" => Some(ComplexMethod::ThreeM),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ComplexMethod::Planar => "planar",
            ComplexMethod::OneM => "1m",
            ComplexMethod::ThreeM => "3m",
        }
    }

    /// The default method, from `TENSORCONTRACT_COMPLEX` or [`Self::Planar`].
    pub fn from_env() -> ComplexMethod {
        #[cfg(feature = "std")]
        {
            use std::sync::OnceLock;
            static M: OnceLock<ComplexMethod> = OnceLock::new();
            *M.get_or_init(|| {
                std::env::var("TENSORCONTRACT_COMPLEX")
                    .ok()
                    .and_then(|v| ComplexMethod::parse(&v))
                    .unwrap_or_default()
            })
        }
        #[cfg(not(feature = "std"))]
        {
            ComplexMethod::Planar
        }
    }
}

/// What [`crate::pack`] should emit for one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackFormat {
    /// One real value per element.
    Real,
    /// Two planes per k-step: all real parts, then all imaginary parts. Used
    /// for both operands under [`ComplexMethod::Planar`], and for operand B
    /// under [`ComplexMethod::OneM`] (BLIS's "1r").
    Planar,
    /// BLIS's "1e": operand A under [`ComplexMethod::OneM`]. Four reals per
    /// complex element, laid out as two real k-steps of `2*VR`.
    OneE,
    /// Three planes per k-step: real, imaginary, and their sum.
    ThreeM,
}

impl PackFormat {
    /// Reals emitted per element per logical k-step.
    #[inline]
    pub const fn reals_per_element(self) -> usize {
        match self {
            PackFormat::Real => 1,
            PackFormat::Planar => 2,
            PackFormat::ThreeM => 3,
            PackFormat::OneE => 4,
        }
    }
}

/// How [`crate::writeback`] should read a micro-kernel's accumulator tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileFormat {
    /// `ab[j*MR + i]`, imaginary part zero.
    Real,
    /// Real plane then imaginary plane, each `MR x NR` column-major.
    Planar,
    /// One real `2*MR x NR` column-major tile; row `2i` is the real part of
    /// complex row `i` and row `2i+1` the imaginary part.
    OneM,
    /// Three `MR x NR` planes `M1`, `M2`, `M3`; the complex value is
    /// `(M1 - M2) + i*(M3 - M1 - M2)`.
    ThreeM,
}

/// A micro-kernel: register-blocked panel-panel product.
#[derive(Clone, Copy)]
pub struct Ukr<T> {
    /// Rows of the micro-tile, in *logical* (complex for complex methods)
    /// elements.
    pub mr: usize,
    /// Columns of the micro-tile.
    pub nr: usize,
    /// Reals in an A sliver per logical k-step (`mr * a_pack.reals_per_element()`).
    pub a_per_k: usize,
    /// Reals in a B sliver per logical k-step.
    pub b_per_k: usize,
    /// Reals in the accumulator tile.
    pub tile: usize,
    pub a_pack: PackFormat,
    pub b_pack: PackFormat,
    pub tile_fmt: TileFormat,
    /// # Safety
    /// `a` must address `a_per_k * kc` values, `b` must address `b_per_k * kc`
    /// values, and `ab` must address `tile` values.
    pub func: unsafe fn(kc: usize, a: *const T, b: *const T, ab: *mut T),
    pub name: &'static str,
}

impl<T> core::fmt::Debug for Ukr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Ukr({} {}x{})", self.name, self.mr, self.nr)
    }
}

/// Cache blocking parameters, in logical elements.
///
/// * `mc x kc` is the packed A block, sized for L2.
/// * `kc x nc` is the packed B block, sized for L3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocking {
    pub mc: usize,
    pub kc: usize,
    pub nc: usize,
}

impl Blocking {
    /// Derive blocking from the packed footprint the kernel actually produces.
    ///
    /// Doing it this way rather than from `size_of::<Element>()` is what makes
    /// 1m automatically get a smaller `MC`: its packed A carries four reals per
    /// complex element instead of two, so the same L2 budget buys half as many
    /// rows. Getting that wrong would hand 1m an unfair L2 overflow and make
    /// the comparison meaningless.
    pub fn derive(real_bytes: usize, a_reals: usize, b_reals: usize) -> Blocking {
        let kc = if real_bytes <= 4 { 384 } else { 256 };
        // A *coupled* `kc` override re-derives `mc`/`nc` against the same cache
        // budgets at the new depth; the plain `TENSORCONTRACT_KC` override
        // changes `kc` alone and leaves the `D` strip a `jr` pass revisits
        // exactly as wide as it was. The item 2 grid needs both arms, because
        // `MC` is bounded from two sides and only the pair separates them (A13).
        let kc = env_usize("TENSORCONTRACT_KC_COUPLE").unwrap_or(kc);
        Blocking::derive_at_depth(real_bytes, a_reals, b_reals, kc)
    }

    /// [`Blocking::derive`] with `kc` supplied rather than chosen.
    ///
    /// Exists for the `MC`/`KC`/`NC` sweep, which needs to vary `kc` and get
    /// the cache-budget-consistent `mc`/`nc` that go with it rather than a
    /// grid of independently chosen numbers.
    ///
    /// Tempting use it takes some care with: re-deriving against `min(k, KC)`
    /// once the contraction depth is known. A contraction shallower than `KC`
    /// — `k = 24` against a `KC` of 256 describes the entire `abcijk` third of
    /// the corpus — otherwise packs into an `A` block sized for a panel ten
    /// times deeper, using a tenth of its L2 budget. Doing that measured
    /// **+13% on one case and −18% on another**, because `MC` also bounds the
    /// strip of `D` that a `jr` pass revisits, which this budget does not
    /// model. See the Phase 4 report before reaching for it.
    pub fn derive_at_depth(
        real_bytes: usize,
        a_reals: usize,
        b_reals: usize,
        kc: usize,
    ) -> Blocking {
        // Roughly half of a 1 MiB L2 for the packed A block, 3 MiB of L3 for B.
        const A_BUDGET: usize = 512 * 1024;
        const B_BUDGET: usize = 3 * 1024 * 1024;
        let kc = kc.max(1);
        let mc = (A_BUDGET / (kc * a_reals * real_bytes)).max(1);
        let nc = (B_BUDGET / (kc * b_reals * real_bytes)).max(1);
        Blocking { mc, kc, nc }
    }
}

/// Everything the driver needs for one element type and complex method.
#[derive(Clone, Copy, Debug)]
pub struct KernelConfig<T> {
    pub ukr: Ukr<T>,
    pub blk: Blocking,
}

impl<T> KernelConfig<T> {
    /// Apply any environment override, then round `mc`/`nc` to whole multiples
    /// of the register block (the driver's loop arithmetic relies on that).
    pub(crate) fn normalise(self) -> Self {
        let mut blk = self.blk;
        if let Some(o) = env_blocking() {
            if let Some(v) = o.mc {
                blk.mc = v;
            }
            if let Some(v) = o.kc {
                blk.kc = v;
            }
            if let Some(v) = o.nc {
                blk.nc = v;
            }
            // Percentage forms scale whatever the derivation produced. An
            // absolute `MC` means different fractions of the L2 budget in each
            // dtype and method — 1m derives half the `mc` planar does, by
            // design — so pinning one number across a sweep arm would rig the
            // comparison the same way getting `Blocking::derive` wrong would.
            // The percentage stays budget-proportional, so one arm is one
            // question.
            if let Some(p) = o.mc_pct {
                blk.mc = (blk.mc * p / 100).max(1);
            }
            if let Some(p) = o.nc_pct {
                blk.nc = (blk.nc * p / 100).max(1);
            }
        }
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
    mc_pct: Option<usize>,
    nc_pct: Option<usize>,
}

/// `TENSORCONTRACT_MC` / `_KC` / `_NC` override the cache blocking absolutely;
/// `_MC_PCT` / `_NC_PCT` scale the derived value instead, and
/// `_KC_COUPLE` (read in [`Blocking::derive`]) sets `kc` *and* re-derives
/// `mc`/`nc` against the cache budgets at that depth. Read once per process;
/// used for the Phase 4 parameter sweeps and to exercise every level of the
/// loop nest on small test problems.
fn env_blocking() -> Option<BlockingOverride> {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static ENV: OnceLock<Option<BlockingOverride>> = OnceLock::new();
        *ENV.get_or_init(|| {
            let o = BlockingOverride {
                mc: env_usize("TENSORCONTRACT_MC"),
                kc: env_usize("TENSORCONTRACT_KC"),
                nc: env_usize("TENSORCONTRACT_NC"),
                mc_pct: env_usize("TENSORCONTRACT_MC_PCT"),
                nc_pct: env_usize("TENSORCONTRACT_NC_PCT"),
            };
            (o.mc.is_some()
                || o.kc.is_some()
                || o.nc.is_some()
                || o.mc_pct.is_some()
                || o.nc_pct.is_some())
            .then_some(o)
        })
    }
    #[cfg(not(feature = "std"))]
    {
        None
    }
}

/// One `usize`-valued environment variable, or `None` if unset or unparseable.
/// Not cached: the callers that use it directly run once per configuration.
fn env_usize(_key: &str) -> Option<usize> {
    #[cfg(feature = "std")]
    {
        std::env::var(_key).ok().and_then(|v| v.parse().ok())
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
///     fn config_cplx(m: ComplexMethod) -> KernelConfig<Self> {
///         scalar::config_cplx::<Self, 4, 4>(m)
///     }
/// }
/// ```
pub trait KernelSet: Real + Sized {
    fn config_real() -> KernelConfig<Self>;
    fn config_cplx(method: ComplexMethod) -> KernelConfig<Self>;

    /// Logical row blocks `MR` this kernel set can run, default first.
    ///
    /// More than one entry is an invitation to [`crate::Plan::row_block`] to
    /// pick a shape that suits the output's stride pattern rather than the
    /// kernel's own peak. An empty menu means "no choice", which is what the
    /// portable path returns.
    fn row_blocks(complex: bool, method: ComplexMethod) -> &'static [usize] {
        let _ = (complex, method);
        &[]
    }

    /// The configuration at a chosen row block, or `None` if there is no such
    /// kernel. Only ever called with an `mr` from [`KernelSet::row_blocks`].
    fn config_at(complex: bool, method: ComplexMethod, mr: usize) -> Option<KernelConfig<Self>> {
        let _ = (complex, method, mr);
        None
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

macro_rules! impl_kernel_set {
    ($t:ty, $real_x86:path, $cplx_x86:path, $rows_x86:path,
     $real_at_x86:path, $cplx_at_x86:path, $mr:literal, $nr:literal) => {
        impl KernelSet for $t {
            fn config_real() -> KernelConfig<Self> {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if !force_scalar() {
                    if let Some(c) = $real_x86() {
                        return c.normalise();
                    }
                }
                scalar::config_real::<$t, $mr, $nr>().normalise()
            }

            fn config_cplx(method: ComplexMethod) -> KernelConfig<Self> {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if !force_scalar() {
                    if let Some(c) = $cplx_x86(method) {
                        return c.normalise();
                    }
                }
                scalar::config_cplx::<$t, $mr, $nr>(method).normalise()
            }

            fn row_blocks(complex: bool, method: ComplexMethod) -> &'static [usize] {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if !force_scalar() {
                    return $rows_x86(complex, method);
                }
                // The portable path has one shape per method and no menu.
                #[allow(unreachable_code)]
                {
                    let _ = (complex, method);
                    &[]
                }
            }

            fn config_at(
                complex: bool,
                method: ComplexMethod,
                mr: usize,
            ) -> Option<KernelConfig<Self>> {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if !force_scalar() {
                    let c = if complex {
                        $cplx_at_x86(method, mr)
                    } else {
                        $real_at_x86(mr)
                    };
                    return c.map(KernelConfig::normalise);
                }
                #[allow(unreachable_code)]
                {
                    let _ = (complex, method, mr);
                    None
                }
            }
        }
    };
}

impl_kernel_set!(
    f64,
    x86::config_real_f64,
    x86::config_cplx_f64,
    x86::row_blocks_f64,
    x86::config_real_f64_at,
    x86::config_cplx_f64_at,
    4,
    4
);
impl_kernel_set!(
    f32,
    x86::config_real_f32,
    x86::config_cplx_f32,
    x86::row_blocks_f32,
    x86::config_real_f32_at,
    x86::config_cplx_f32_at,
    4,
    4
);

/// The register block `(MR, NR)` and cache blocking the engine will use for
/// element type `T`. Exposed for diagnostics and for harnesses that want to
/// report block-scatter regularity at the same granularity the engine sees.
pub fn selected_config<T>(method: ComplexMethod) -> (usize, usize, Blocking)
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    let cfg = config_for::<T>(method);
    (cfg.ukr.mr, cfg.ukr.nr, cfg.blk)
}

/// Name of the micro-kernel selected for `T`.
pub fn selected_kernel_name<T>(method: ComplexMethod) -> &'static str
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    config_for::<T>(method).ukr.name
}

/// Pick the kernel configuration for an element type and complex method.
/// `method` is ignored for real element types.
pub(crate) fn config_for<T>(method: ComplexMethod) -> KernelConfig<T::Real>
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    if T::IS_COMPLEX {
        <T::Real as KernelSet>::config_cplx(method)
    } else {
        <T::Real as KernelSet>::config_real()
    }
}

/// [`config_for`] plus the plan's choice of micro-tile row block.
///
/// This is what execution uses. It differs from [`config_for`] only when the
/// output's stride pattern makes a shape other than the kernel set's default
/// worth having — see [`crate::Plan::row_block`] — and falls back to the
/// default whenever the requested shape does not exist.
pub(crate) fn config_for_plan<T>(plan: &crate::plan::Plan) -> KernelConfig<T::Real>
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    let method = plan.complex_method();
    let menu = <T::Real as KernelSet>::row_blocks(T::IS_COMPLEX, method);
    plan.row_block(menu)
        .and_then(|mr| <T::Real as KernelSet>::config_at(T::IS_COMPLEX, method, mr))
        .unwrap_or_else(|| config_for::<T>(method))
}

/// The register block `(MR, NR)` and cache blocking a *plan* will execute with.
///
/// [`selected_config`] reports the kernel set's default; this reports what the
/// plan actually gets, which can differ. Harnesses reporting block-scatter
/// regularity must use this one, or they describe a traversal that does not
/// happen.
pub fn plan_config<T>(plan: &crate::plan::Plan) -> (usize, usize, Blocking)
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    let cfg = config_for_plan::<T>(plan);
    (cfg.ukr.mr, cfg.ukr.nr, cfg.blk)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel must agree with the mathematical definition, whatever
    /// packing and tile format it uses. This is the contract the driver relies
    /// on, so it is checked directly rather than only end to end.
    fn check_real<T: KernelSet>(tol: f64) {
        check_real_cfg::<T>(T::config_real(), tol);
        for &mr in T::row_blocks(false, ComplexMethod::default()) {
            let cfg = T::config_at(false, ComplexMethod::default(), mr)
                .unwrap_or_else(|| panic!("real menu offers MR={mr} with no kernel"));
            assert_eq!(cfg.ukr.mr, mr);
            check_real_cfg::<T>(cfg, tol);
        }
    }

    fn check_real_cfg<T: KernelSet>(cfg: KernelConfig<T>, tol: f64) {
        let (mr, nr) = (cfg.ukr.mr, cfg.ukr.nr);
        let kc = 37usize;
        let a: Vec<T> = (0..cfg.ukr.a_per_k * kc)
            .map(|i| T::from_f64(((i * 37 % 19) as f64 - 9.0) / 7.0))
            .collect();
        let b: Vec<T> = (0..cfg.ukr.b_per_k * kc)
            .map(|i| T::from_f64(((i * 53 % 23) as f64 - 11.0) / 5.0))
            .collect();
        let mut got = vec![T::ZERO; cfg.ukr.tile];
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

    /// Read a complex value out of an accumulator tile.
    fn tile_value<T: Real>(
        ab: &[T],
        fmt: TileFormat,
        mr: usize,
        nr: usize,
        i: usize,
        j: usize,
    ) -> (f64, f64) {
        match fmt {
            TileFormat::Real => (ab[j * mr + i].to_f64(), 0.0),
            TileFormat::Planar => (ab[j * mr + i].to_f64(), ab[mr * nr + j * mr + i].to_f64()),
            TileFormat::OneM => (
                ab[j * (2 * mr) + 2 * i].to_f64(),
                ab[j * (2 * mr) + 2 * i + 1].to_f64(),
            ),
            TileFormat::ThreeM => {
                let m1 = ab[j * mr + i].to_f64();
                let m2 = ab[mr * nr + j * mr + i].to_f64();
                let m3 = ab[2 * mr * nr + j * mr + i].to_f64();
                (m1 - m2, m3 - m1 - m2)
            }
        }
    }

    /// Build a packed sliver in the format the kernel expects, from complex
    /// values supplied as `(re, im)` pairs indexed `[k][lane]`.
    fn pack_for<T: Real>(fmt: PackFormat, vr: usize, vals: &[Vec<(f64, f64)>]) -> Vec<T> {
        let per_k = vr * fmt.reals_per_element();
        let mut out = vec![T::ZERO; per_k * vals.len()];
        for (p, row) in vals.iter().enumerate() {
            let base = p * per_k;
            for (t, &(re, im)) in row.iter().enumerate() {
                match fmt {
                    PackFormat::Real => out[base + t] = T::from_f64(re),
                    PackFormat::Planar => {
                        out[base + t] = T::from_f64(re);
                        out[base + vr + t] = T::from_f64(im);
                    }
                    PackFormat::ThreeM => {
                        out[base + t] = T::from_f64(re);
                        out[base + vr + t] = T::from_f64(im);
                        out[base + 2 * vr + t] = T::from_f64(re + im);
                    }
                    PackFormat::OneE => {
                        out[base + 2 * t] = T::from_f64(re);
                        out[base + 2 * t + 1] = T::from_f64(im);
                        out[base + 2 * vr + 2 * t] = T::from_f64(-im);
                        out[base + 2 * vr + 2 * t + 1] = T::from_f64(re);
                    }
                }
            }
        }
        out
    }

    fn check_cplx<T: KernelSet>(method: ComplexMethod, tol: f64) {
        check_cplx_cfg::<T>(T::config_cplx(method), method, tol);
        for &mr in T::row_blocks(true, method) {
            let cfg = T::config_at(true, method, mr)
                .unwrap_or_else(|| panic!("{} menu offers MR={mr} with no kernel", method.name()));
            assert_eq!(cfg.ukr.mr, mr);
            check_cplx_cfg::<T>(cfg, method, tol);
        }
    }

    fn check_cplx_cfg<T: KernelSet>(cfg: KernelConfig<T>, method: ComplexMethod, tol: f64) {
        let (mr, nr) = (cfg.ukr.mr, cfg.ukr.nr);
        let kc = 29usize;

        let av: Vec<Vec<(f64, f64)>> = (0..kc)
            .map(|p| {
                (0..mr)
                    .map(|i| {
                        (
                            (((p * mr + i) * 31 % 17) as f64 - 8.0) / 6.0,
                            (((p * mr + i) * 13 % 11) as f64 - 5.0) / 3.0,
                        )
                    })
                    .collect()
            })
            .collect();
        let bv: Vec<Vec<(f64, f64)>> = (0..kc)
            .map(|p| {
                (0..nr)
                    .map(|j| {
                        (
                            (((p * nr + j) * 41 % 13) as f64 - 6.0) / 4.0,
                            (((p * nr + j) * 7 % 19) as f64 - 9.0) / 5.0,
                        )
                    })
                    .collect()
            })
            .collect();

        let a = pack_for::<T>(cfg.ukr.a_pack, mr, &av);
        let b = pack_for::<T>(cfg.ukr.b_pack, nr, &bv);
        assert_eq!(a.len(), cfg.ukr.a_per_k * kc);
        assert_eq!(b.len(), cfg.ukr.b_per_k * kc);

        let mut got = vec![T::ZERO; cfg.ukr.tile];
        unsafe { (cfg.ukr.func)(kc, a.as_ptr(), b.as_ptr(), got.as_mut_ptr()) };

        #[allow(clippy::needless_range_loop)]
        for j in 0..nr {
            for i in 0..mr {
                let (mut wr, mut wi) = (0.0f64, 0.0f64);
                for p in 0..kc {
                    let (ar, ai) = av[p][i];
                    let (br, bi) = bv[p][j];
                    wr += ar * br - ai * bi;
                    wi += ar * bi + ai * br;
                }
                let (gr, gi) = tile_value(&got, cfg.ukr.tile_fmt, mr, nr, i, j);
                assert!(
                    (gr - wr).abs() <= tol * wr.abs().max(1.0),
                    "{} [{}] re mismatch at ({i},{j}): {gr} vs {wr}",
                    cfg.ukr.name,
                    method.name()
                );
                assert!(
                    (gi - wi).abs() <= tol * wi.abs().max(1.0),
                    "{} [{}] im mismatch at ({i},{j}): {gi} vs {wi}",
                    cfg.ukr.name,
                    method.name()
                );
            }
        }
    }

    #[test]
    fn kernels_match_reference_f64() {
        check_real::<f64>(1e-12);
        for m in ComplexMethod::ALL {
            check_cplx::<f64>(m, 1e-12);
        }
    }

    #[test]
    fn kernels_match_reference_f32() {
        check_real::<f32>(1e-4);
        for m in ComplexMethod::ALL {
            check_cplx::<f32>(m, 1e-4);
        }
    }

    /// A menu with a repeated `MR` would make `config_at` unreachable for the
    /// later entry, silently pinning to the wrong shape; and the head of the
    /// menu must be exactly what the default builders return, or `base` and
    /// `idx=0` would mean different things.
    #[test]
    fn row_block_menus_are_well_formed() {
        fn check<T: KernelSet>(complex: bool, method: ComplexMethod, default_mr: usize) {
            let menu = T::row_blocks(complex, method);
            if menu.is_empty() {
                return; // no vectorised kernels on this CPU
            }
            assert_eq!(menu[0], default_mr, "menu head is not the default shape");
            for (i, &mr) in menu.iter().enumerate() {
                assert!(mr > 0);
                assert!(!menu[..i].contains(&mr), "MR={mr} appears twice on a menu");
            }
        }
        check::<f64>(false, ComplexMethod::default(), f64::config_real().ukr.mr);
        check::<f32>(false, ComplexMethod::default(), f32::config_real().ukr.mr);
        for m in ComplexMethod::ALL {
            check::<f64>(true, m, f64::config_cplx(m).ukr.mr);
            check::<f32>(true, m, f32::config_cplx(m).ukr.mr);
        }
    }

    #[test]
    fn panel_sizes_are_self_consistent() {
        for m in ComplexMethod::ALL {
            let c = f64::config_cplx(m);
            assert_eq!(c.ukr.a_per_k, c.ukr.mr * c.ukr.a_pack.reals_per_element());
            assert_eq!(c.ukr.b_per_k, c.ukr.nr * c.ukr.b_pack.reals_per_element());
            assert_eq!(c.blk.mc % c.ukr.mr, 0);
            assert_eq!(c.blk.nc % c.ukr.nr, 0);
        }
    }

    #[test]
    fn one_m_packs_a_twice_as_large_as_planar() {
        // The structural cost of 1m, and the reason its MC must be smaller.
        let planar = f64::config_cplx(ComplexMethod::Planar);
        let onem = f64::config_cplx(ComplexMethod::OneM);
        assert_eq!(planar.ukr.a_pack.reals_per_element(), 2);
        assert_eq!(onem.ukr.a_pack.reals_per_element(), 4);
        assert_eq!(onem.ukr.b_pack, planar.ukr.b_pack, "B is 1r either way");
        // Same L2 budget, so twice the reals per element buys half the rows.
        let bytes =
            |c: &KernelConfig<f64>| c.blk.mc * c.blk.kc * c.ukr.a_pack.reals_per_element() * 8;
        assert!(
            bytes(&onem) <= bytes(&planar) * 3 / 2,
            "1m packed A block {} vs planar {}",
            bytes(&onem),
            bytes(&planar)
        );
    }

    #[test]
    fn three_m_does_fewer_flops() {
        let planar = f64::config_cplx(ComplexMethod::Planar);
        let threem = f64::config_cplx(ComplexMethod::ThreeM);
        // 3 planes of accumulator instead of 2, but 3 products instead of 4.
        assert_eq!(threem.ukr.tile, 3 * threem.ukr.mr * threem.ukr.nr);
        assert_eq!(planar.ukr.tile, 2 * planar.ukr.mr * planar.ukr.nr);
    }

    #[test]
    fn method_parsing_roundtrips() {
        for m in ComplexMethod::ALL {
            assert_eq!(ComplexMethod::parse(m.name()), Some(m));
        }
        assert_eq!(ComplexMethod::parse("nope"), None);
    }
}
