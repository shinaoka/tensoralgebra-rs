//! Typed kernel-family contracts. Follows the context/control separation in
//! BLIS (Van Zee & van de Geijn, TOMS 2015) and TBLIS (Matthews, SISC 2018).
//! New descriptor code; existing pack/tile semantics come from tensorcontract.

use crate::{tile_planes, CpuFeatures, Element, Isa, PackFormat, Real, TileFormat, Ukr};

/// Kernel provenance (adapters call their upstream; they do not copy it).
///
/// # Examples
/// ```
/// use tprims_gemm_kernel::Origin;
/// assert_eq!(Origin::Tensorcontract.crate_name(), "tprims-kernel-tensorcontract");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Origin {
    /// Lukas Devos's tensorcontract kernels.
    Tensorcontract,
    /// Project-owned portable kernels.
    Portable,
    /// Called gemm-f32/f64 microkernels.
    Gemm,
    /// Called private-gemm-x86 engine.
    PrivateGemmX86,
}
impl Origin {
    /// Crate providing this implementation.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::Origin;
    /// assert_eq!(Origin::Portable.crate_name(), "tprims-gemm-kernel");
    /// ```
    pub const fn crate_name(self) -> &'static str {
        match self {
            Self::Tensorcontract => "tprims-kernel-tensorcontract",
            Self::Portable => "tprims-gemm-kernel",
            Self::Gemm => "tprims-kernel-gemm",
            Self::PrivateGemmX86 => "tprims-kernel-pgx86",
        }
    }
    /// License of the underlying kernel implementation (not the adapter).
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::Origin;
    /// assert_eq!(Origin::Gemm.license(), "MIT");
    /// ```
    pub const fn license(self) -> &'static str {
        match self {
            Self::Tensorcontract | Self::Portable => "MIT OR Apache-2.0",
            Self::Gemm | Self::PrivateGemmX86 => "MIT",
        }
    }
}

/// Implementation category reported by a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KernelImpl {
    /// Portable/reference implementation.
    Reference,
    /// ISA-optimized implementation.
    Optimized,
    /// Complex arithmetic induced from a real kernel.
    Induced,
}

/// Complex arithmetic method, separate from the packed layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Method {
    /// Native complex arithmetic.
    Native,
    /// One real product on expanded/reordered panels.
    OneM,
    /// Three products, with an opt-in cancellation/accuracy trade-off.
    ThreeM,
    /// Four real products recombined by write-back.
    FourM,
}

/// Packed operand layout; external complex storage stays interleaved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Layout {
    /// Real values.
    Real,
    /// Adjacent real/imaginary values.
    Interleaved,
    /// Separate real/imaginary planes per k-step.
    Planar,
    /// 1m expanded operand.
    OneE,
    /// 1m reordered operand.
    OneR,
    /// Real, imaginary and their sum.
    ThreeM,
}

/// The operand and scratch-tile formats of one complex family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComplexScheme {
    /// Arithmetic method.
    pub method: Method,
    /// A packing layout.
    pub a: Layout,
    /// B packing layout.
    pub b: Layout,
    /// Scratch tile layout.
    pub tile: TileFormat,
}

/// Preferred output-contiguous axis (a performance hint, not a restriction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CPref {
    /// Contiguous columns within a row.
    Row,
    /// Contiguous rows within a column.
    Col,
    /// No preference.
    Any,
}
/// Axis that must have unit stride for a direct-B path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Row axis.
    Row,
    /// Column axis.
    Col,
}
/// How a family reads B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BAccess {
    /// Driver-packed panels.
    Packed,
    /// B may be read in place when the indicated axis has unit stride.
    Direct {
        /// Required contiguous axis.
        unit_stride: Axis,
    },
}
/// Who applies alpha/beta and stores output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CUpdate {
    /// Kernel overwrites a scratch tile, driver writes back.
    ScratchTile,
    /// Kernel applies scaling and writes a strided output tile.
    Direct,
}
/// Default and maximum cache blocksizes; max is reserved for tail merging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocksizes {
    /// Row block (default, max), multiples of MR.
    pub mc: (usize, usize),
    /// Depth block (default, max).
    pub kc: (usize, usize),
    /// Column block (default, max), multiples of NR.
    pub nc: (usize, usize),
}
/// Operand capabilities checked at plan construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps {
    /// Scatter operands can be packed.
    pub scatter_pack: bool,
    /// A conjugation is supported.
    pub conj_a: bool,
    /// B conjugation is supported.
    pub conj_b: bool,
}

/// Scratch-tile kernel: overwrite the entire declared tile with A·B.
/// Pointers must cover the family's packed panel and tile sizes.
pub type TileUkrFn<R> = unsafe fn(kc: usize, a: *const R, b: *const R, tile: *mut R);
/// Direct kernel: `d = alpha_d*d + beta_ab*(A*B)` on an m×n tile.
/// A is packed with unit row stride; B and D use the supplied strides.
/// With alpha_d zero the kernel must not read D. All addressed values must
/// be valid and output must not alias input panels.
pub type DirectUkrFn<R> = unsafe fn(
    m: usize,
    n: usize,
    k: usize,
    d: *mut R,
    rs_d: isize,
    cs_d: isize,
    a: *const R,
    a_cs: isize,
    b: *const R,
    b_rs: isize,
    b_cs: isize,
    alpha_d: R,
    beta_ab: R,
    aux: &UkrAux<R>,
);
/// Typed kernel pointer; the variant must agree with `CUpdate`.
#[derive(Clone, Copy, Debug)]
pub enum UkrFn<R: 'static> {
    /// Overwrites scratch (no alpha/beta or output access).
    Tile(TileUkrFn<R>),
    /// Writes a strided output tile with alpha/beta.
    Direct(DirectUkrFn<R>),
}
/// Auxiliary state passed to direct and induced kernels.
#[derive(Clone, Copy, Debug)]
pub struct UkrAux<R: 'static> {
    /// Next A sliver for optional prefetching.
    pub a_next: *const R,
    /// Next B sliver for optional prefetching.
    pub b_next: *const R,
    /// Real kernel underlying an induced family.
    pub inner: Option<&'static KernelFamily<R>>,
}
/// Monomorphized packer for one element type and layout.
/// Caller supplies valid scatter offsets and a sufficiently sized output;
/// every packed lane, including edge padding, is overwritten.
pub type PackFn<T> = unsafe fn(
    base: *const T,
    vscat: &[i64],
    vbs: &[i64],
    kscat: &[i64],
    vr: usize,
    conj: bool,
    out: *mut <T as Element>::Real,
);

/// Monomorphized write-back for one storage type and frozen tile format.
/// Caller supplies a fully overwritten tile, valid live scatters and disjoint
/// output. C is not read when beta is zero. All other raw writeback obligations
/// apply, including positive live extents no larger than MR/NR.
pub type EmitFn<T> = unsafe fn(
    *const <T as Element>::Real,
    usize,
    usize,
    usize,
    usize,
    T,
    T,
    *const T,
    &[i64],
    &[i64],
    i64,
    bool,
    *mut T,
    &[i64],
    &[i64],
    i64,
    bool,
);

/// Immutable descriptor keyed by the arithmetic real type.
/// Complex families are distinguished by `complex`, not by `R`.
/// The pointer's implementation must satisfy the described unsafe contract.
#[derive(Clone, Copy, Debug)]
pub struct KernelFamily<R: 'static> {
    /// Stable kernel identifier.
    pub id: &'static str,
    /// Implementation provenance.
    pub origin: Origin,
    /// Diagnostic ISA label.
    pub isa: Isa,
    /// CPU capabilities necessary to call the kernel.
    pub required: CpuFeatures,
    /// Reference, optimized or induced implementation.
    pub imp: KernelImpl,
    /// Auto priority; higher is preferred.
    pub priority: u16,
    /// Complex formats, or none for a real family.
    pub complex: Option<ComplexScheme>,
    /// Logical tile rows.
    pub mr: usize,
    /// Logical tile columns.
    pub nr: usize,
    /// Packed reals per A k-step.
    pub a_per_k: usize,
    /// Packed reals per B k-step.
    pub b_per_k: usize,
    /// Scratch tile capacity in reals.
    pub tile_bound: usize,
    /// Preferred output layout.
    pub c_pref: CPref,
    /// Kernel pointer.
    pub ukr: UkrFn<R>,
    /// B access policy.
    pub b_access: BAccess,
    /// Output update contract.
    pub c_update: CUpdate,
    /// Default/max cache block dimensions.
    pub blocks: Blocksizes,
    /// Supported operand operations.
    pub caps: Caps,
    /// Whether Auto may choose this family.
    pub allow_auto: bool,
}

/// Invalid descriptor geometry or format combination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyError {
    /// Invalid family's identifier.
    pub id: &'static str,
    /// Specific violated invariant.
    pub reason: &'static str,
}
impl core::fmt::Display for FamilyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "kernel {}: {}", self.id, self.reason)
    }
}
impl core::error::Error for FamilyError {}

impl<R: Real> KernelFamily<R> {
    /// Validate descriptor geometry and supported format combinations.
    ///
    /// # Errors
    /// Returns `FamilyError` naming the invalid size, packing/tile contract,
    /// complex scheme, or direct-update combination. Does not invoke kernels.
    pub fn validate(&self) -> Result<(), FamilyError> {
        let fail = |reason| {
            Err(FamilyError {
                id: self.id,
                reason,
            })
        };
        if self.mr == 0 {
            return fail("mr is zero");
        }
        if self.nr == 0 {
            return fail("nr is zero");
        }
        if !self.blocks.mc.0.is_multiple_of(self.mr) {
            return fail("mc default is not a multiple of mr");
        }
        if !self.blocks.nc.0.is_multiple_of(self.nr) {
            return fail("nc default is not a multiple of nr");
        }
        for (default, max) in [self.blocks.mc, self.blocks.kc, self.blocks.nc] {
            if default == 0 {
                return fail("default blocksize is zero");
            }
            if default > max {
                return fail("default blocksize exceeds max");
            }
        }
        if !self.blocks.mc.1.is_multiple_of(self.mr) || !self.blocks.nc.1.is_multiple_of(self.nr) {
            return fail("max blocksize is not a tile multiple");
        }
        let (a, b, tile) = match self.complex {
            None => (Layout::Real, Layout::Real, TileFormat::Real),
            Some(s) if supported(s) => (s.a, s.b, s.tile),
            Some(_) => return fail("complex scheme not in the supported table"),
        };
        let Some(size) = self
            .mr
            .checked_mul(self.nr)
            .and_then(|n| n.checked_mul(tile_planes(tile)))
        else {
            return fail("tile size overflow");
        };
        if self.tile_bound < size {
            return fail("tile_bound below tile size");
        }
        let a_len = self.mr.checked_mul(pack_format(a).reals_per_element());
        let b_len = self.nr.checked_mul(pack_format(b).reals_per_element());
        if a_len.is_none_or(|n| self.a_per_k < n) || b_len.is_none_or(|n| self.b_per_k < n) {
            return fail("packed k-step shorter than the tile");
        }
        if matches!(self.ukr, UkrFn::Direct(_)) != (self.c_update == CUpdate::Direct) {
            return fail("ukr kind disagrees with c_update");
        }
        if self.complex.is_some() && self.c_update == CUpdate::Direct {
            return fail("complex Direct kernels are not supported");
        }
        if matches!(self.b_access, BAccess::Direct { .. }) && self.c_update != CUpdate::Direct {
            return fail("direct B needs a Direct kernel");
        }
        Ok(())
    }

    /// Re-express a validated scratch family in the legacy driver contract.
    /// Returns none for direct kernels or unsupported complex schemes.
    pub fn as_ukr(&self) -> Option<Ukr<R>> {
        let UkrFn::Tile(func) = self.ukr else {
            return None;
        };
        let (a_pack, b_pack, tile_fmt) = match self.complex {
            None => (PackFormat::Real, PackFormat::Real, TileFormat::Real),
            Some(s) if supported(s) => (pack_format(s.a), pack_format(s.b), s.tile),
            Some(_) => return None,
        };
        Some(Ukr {
            mr: self.mr,
            nr: self.nr,
            a_per_k: self.a_per_k,
            b_per_k: self.b_per_k,
            tile: self.tile_bound,
            a_pack,
            b_pack,
            tile_fmt,
            func,
            name: self.id,
        })
    }
}

fn supported(s: ComplexScheme) -> bool {
    matches!(
        (s.method, s.a, s.b, s.tile),
        (
            Method::Native,
            Layout::Planar,
            Layout::Planar,
            TileFormat::Planar
        ) | (
            Method::Native,
            Layout::Interleaved,
            Layout::Interleaved,
            TileFormat::Interleaved
        ) | (
            Method::FourM,
            Layout::Planar,
            Layout::Planar,
            TileFormat::FourM
        ) | (Method::OneM, Layout::OneE, Layout::OneR, TileFormat::OneM)
            | (Method::OneM, Layout::OneR, Layout::OneE, TileFormat::OneM)
            | (
                Method::ThreeM,
                Layout::ThreeM,
                Layout::ThreeM,
                TileFormat::ThreeM
            )
    )
}
pub(crate) fn pack_format(layout: Layout) -> PackFormat {
    match layout {
        Layout::Real => PackFormat::Real,
        Layout::Planar | Layout::OneR => PackFormat::Planar,
        Layout::OneE => PackFormat::OneE,
        Layout::ThreeM => PackFormat::ThreeM,
        Layout::Interleaved => PackFormat::Interleaved,
    }
}
