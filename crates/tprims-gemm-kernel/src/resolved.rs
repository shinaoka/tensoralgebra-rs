//! Planning-time family selection and frozen blocking policy.
//! Reuses Lukas Devos's tensorcontract blocking geometry and the existing
//! cache model; see `cache` for the Low et al. model and its provenance.

use crate::{
    cache::{self, BlockModel, CacheHierarchy, PanelGeom},
    partition::{PartitionOpts, PartitionPolicy},
    types::{env_blocking, BlockingOverride},
    Blocking, CpuFeatures, Families, KernelFamily, Layout, Real, Registry, SelectError, TileFormat,
};

/// Family choice, validated for a concrete dtype during resolution.
///
/// # Examples
/// ```
/// use tprims_gemm_kernel::{KernelChoice, ResolvedGemm};
/// let choice = KernelChoice::Id("portable.f64.4x4".into());
/// let rg = ResolvedGemm::<f64>::resolve::<f64>(&choice, 1)?;
/// assert_eq!(rg.family().id, "portable.f64.4x4");
/// # Ok::<(), tprims_gemm_kernel::SelectError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelChoice {
    /// Highest-priority available, Auto-eligible family.
    #[default]
    Auto,
    /// Exact stable identifier; no silent fallback.
    Id(String),
}

impl KernelChoice {
    /// `TPRIMS_GEMM_KERNEL`, captured once; absent/auto means Auto.
    ///
    /// # Examples
    /// ```
    /// assert!(core::ptr::eq(tprims_gemm_kernel::KernelChoice::from_env(),
    ///     tprims_gemm_kernel::KernelChoice::from_env()));
    /// ```
    pub fn from_env() -> &'static Self {
        static CHOICE: std::sync::OnceLock<KernelChoice> = std::sync::OnceLock::new();
        CHOICE.get_or_init(|| {
            #[cfg(feature = "std")]
            if let Ok(id) = std::env::var("TPRIMS_GEMM_KERNEL") {
                if !id.eq_ignore_ascii_case("auto") {
                    return Self::Id(id);
                }
            }
            Self::Auto
        })
    }
}

/// Frozen process default for one storage dtype. Register providers first;
/// later registration does not invalidate this immutable metadata cache.
/// Selection failures are cached too. It owns no execution workspace.
///
/// # Examples
/// ```
/// let a = tprims_gemm_kernel::process_default::<f64>()?;
/// let b = tprims_gemm_kernel::process_default::<f64>()?;
/// assert!(core::ptr::eq(a, b));
/// # Ok::<(), tprims_gemm_kernel::SelectError>(())
/// ```
pub fn process_default<T: Families>() -> Result<&'static ResolvedGemm<T::Real>, SelectError> {
    T::process_default()
}

pub(crate) fn resolve_default<T: Families>() -> Result<ResolvedGemm<T::Real>, SelectError>
where
    T::Real: crate::RealSlot,
{
    let width = crate::env_threads();
    if let KernelChoice::Id(_) = KernelChoice::from_env() {
        return ResolvedGemm::<T::Real>::resolve::<T>(KernelChoice::from_env(), width);
    }
    let cpu = CpuFeatures::detect();
    let method = crate::ComplexMethod::from_env();
    let isa = match crate::kernel_force() {
        crate::KernelForce::Auto => None,
        crate::KernelForce::Avx2 if cpu.avx2 && cpu.fma => Some(crate::Isa::Avx2),
        crate::KernelForce::Avx512 if cpu.avx512f && cpu.fma => Some(crate::Isa::Avx512),
        crate::KernelForce::Neon if cpu.neon => Some(crate::Isa::Neon),
        _ => Some(crate::Isa::Portable),
    };
    let candidates = Registry::families::<T>(cpu, false);
    let family = candidates
        .into_iter()
        .find(|f| {
            let method_matches = !T::IS_COMPLEX
                || f.complex.is_some_and(|s| match method {
                    crate::ComplexMethod::Planar => {
                        matches!(s.method, crate::Method::FourM | crate::Method::Native)
                    }
                    crate::ComplexMethod::OneM => s.method == crate::Method::OneM,
                    crate::ComplexMethod::ThreeM => s.method == crate::Method::ThreeM,
                });
            // ThreeM is excluded from unqualified Auto, but the legacy explicit
            // COMPLEX=3m control requests it intentionally.
            let eligible =
                f.allow_auto || (T::IS_COMPLEX && method == crate::ComplexMethod::ThreeM);
            method_matches
                && eligible
                && isa.is_none_or(|i| f.isa == i || f.isa == crate::Isa::Portable)
        })
        .ok_or(SelectError::Incompatible {
            id: "auto".into(),
            reason: "no available family for legacy ISA/complex preference",
        })?;
    ResolvedGemm::<T::Real>::resolve::<T>(&KernelChoice::Id(family.id.into()), width)
}

#[derive(Clone, Copy, Debug)]
struct BlockingPolicy {
    model: BlockModel,
    hierarchy: CacheHierarchy,
    overrides: Option<BlockingOverride>,
    explicit: Option<Blocking>,
}
impl BlockingPolicy {
    fn snapshot() -> Self {
        Self {
            model: cache::block_model(),
            hierarchy: cache::hierarchy(),
            overrides: env_blocking(),
            explicit: None,
        }
    }

    fn blocking<R: Real>(self, family: &KernelFamily<R>, threads: usize) -> Option<Blocking> {
        let mut blk = match self.model {
            BlockModel::Legacy => Blocking {
                mc: family.blocks.mc.0,
                kc: family.blocks.kc.0,
                nc: family.blocks.nc.0,
            },
            BlockModel::Analytical => cache::analytical(
                PanelGeom {
                    real_bytes: core::mem::size_of::<R>(),
                    a_reals: family.a_per_k / family.mr,
                    b_reals: family.b_per_k / family.nr,
                    mr: family.mr,
                    nr: family.nr,
                },
                threads,
                &self.hierarchy,
            ),
        };
        if let Some(explicit) = self.explicit {
            blk = explicit;
        } else if let Some(o) = self.overrides {
            blk = o.apply(blk, usize::checked_mul)?;
        }
        blk.mc = blk.mc.checked_next_multiple_of(family.mr)?.max(family.mr);
        blk.nc = blk.nc.checked_next_multiple_of(family.nr)?.max(family.nr);
        blk.kc = blk.kc.max(1);
        Some(blk)
    }
}

/// Resolved immutable family and execution geometry, with no workspace payload.
/// Blocking environment/cache facts are captured at resolution; retargeting
/// only recomputes against this snapshot, never reads environment variables.
///
/// # Examples
/// ```
/// use tprims_gemm_kernel::{KernelChoice, ResolvedGemm};
/// let rg = ResolvedGemm::<f32>::resolve::<f32>(&KernelChoice::Auto, 4)?;
/// let serial = rg.with_threads(1)?;
/// assert!(core::ptr::eq(rg.family(), serial.family()));
/// assert_eq!(serial.effective_threads, 1);
/// # Ok::<(), tprims_gemm_kernel::SelectError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ResolvedGemm<R: Real> {
    /// Selected trusted, registered family.
    family: &'static KernelFamily<R>,
    /// Logical register tile rows.
    pub mr: usize,
    /// Logical register tile columns.
    pub nr: usize,
    /// Packed A reals per k-step.
    pub a_per_k: usize,
    /// Packed B reals per k-step.
    pub b_per_k: usize,
    /// Scratch tile capacity in reals.
    pub tile_bound: usize,
    /// Kernel-row operand packing layout (fixed when user operands swap).
    pub a_layout: Layout,
    /// Kernel-column operand packing layout.
    pub b_layout: Layout,
    /// Scratch tile format.
    pub tile_fmt: TileFormat,
    /// Cache block rows.
    pub mc: usize,
    /// Cache block columns.
    pub nc: usize,
    /// Cache block reduction depth.
    pub kc: usize,
    /// Width against which blocking was derived.
    pub effective_threads: usize,
    /// How the driver cuts the output into worker cells.
    pub partition: PartitionPolicy,
    /// Options that change which cells exist without changing the grid.
    pub opts: PartitionOpts,
    policy: BlockingPolicy,
    gather: bool,
}
impl<R: Real> ResolvedGemm<R> {
    /// Selected immutable, registered descriptor. The reference cannot be
    /// replaced, so safe retargeting retains validated geometry and policy.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::{KernelChoice, ResolvedGemm};
    /// let rg = ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Auto, 1)?;
    /// assert!(rg.family().validate().is_ok());
    /// # Ok::<(), tprims_gemm_kernel::SelectError>(())
    /// ```
    pub fn family(&self) -> &'static KernelFamily<R> {
        self.family
    }

    /// Bind packers once for a raw GEMM driver using storage type `T`.
    /// Returned unsafe functions require the declared panel footprints. Packing
    /// roles stay fixed when the driver exchanges user operands.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::{KernelChoice, ResolvedGemm};
    /// let rg = ResolvedGemm::<f64>::resolve::<f64>(
    ///     &KernelChoice::Id("portable.f64.4x4".into()), 1)?;
    /// let (pack_a, _) = rg.packers::<f64>();
    /// let input = [1., 2., 3., 4.];
    /// let mut output = [0.; 4];
    /// // SAFETY: one full 4-row real panel, valid scatters and write capacity.
    /// unsafe { pack_a(input.as_ptr(), &[0,1,2,3], &[1], &[0], 4, false, output.as_mut_ptr()); }
    /// assert_eq!(output, input);
    /// # Ok::<(), tprims_gemm_kernel::SelectError>(())
    /// ```
    pub fn packers<T: crate::Element<Real = R>>(&self) -> (crate::PackFn<T>, crate::PackFn<T>) {
        (
            crate::pack::pack_fn::<T>(self.a_layout),
            crate::pack::pack_fn::<T>(self.b_layout),
        )
    }

    /// Bind format-specialized write-back with the planning-time gather policy.
    /// Returned raw function obeys [`crate::EmitFn`]'s tile/scatter contract.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::{KernelChoice, ResolvedGemm};
    /// let rg = ResolvedGemm::<f64>::resolve::<f64>(
    ///     &KernelChoice::Id("portable.f64.4x4".into()), 1)?;
    /// let emit = rg.emitter::<f64>();
    /// let tile = [2.; 16];
    /// let mut output = [99.];
    /// // SAFETY: fully initialized 4x4 tile, one valid live element. Beta=0
    /// // permits null C; output/scatters cover the complete live extent.
    /// unsafe { emit(tile.as_ptr(),4,4,1,1,1.,0.,std::ptr::null(),&[0],&[0],1,false,
    ///     output.as_mut_ptr(),&[0],&[0],1,false); }
    /// assert_eq!(output, [2.]);
    /// # Ok::<(), tprims_gemm_kernel::SelectError>(())
    /// ```
    pub fn emitter<T: crate::Element<Real = R>>(&self) -> crate::EmitFn<T> {
        crate::writeback::emit_fn::<T>(self.tile_fmt, self.gather)
    }

    /// Resolve an exact choice or Auto for the storage dtype `T` on this CPU.
    /// Providers must register before the first resolution that needs them.
    ///
    /// # Errors
    /// Returns `UnknownId`, `NotBuilt`, `CpuUnsupported`, `DtypeMismatch`, or
    /// `Incompatible` from registry selection. `Incompatible` also reports
    /// zero width and overflow in blocking overrides/alignment.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::{KernelChoice, ResolvedGemm};
    /// let rg = ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Auto, 1)?;
    /// assert_eq!(rg.a_per_k, rg.mr);
    /// # Ok::<(), tprims_gemm_kernel::SelectError>(())
    /// ```
    pub fn resolve<T: Families<Real = R>>(
        choice: &KernelChoice,
        effective_threads: usize,
    ) -> Result<Self, SelectError> {
        Self::resolve_with::<T>(
            choice,
            effective_threads,
            PartitionPolicy::default(),
            PartitionOpts::default(),
        )
    }

    /// [`resolve`](Self::resolve), with an explicit partition policy and the
    /// execution options that go with it.
    ///
    /// # Errors
    /// Everything [`resolve`](Self::resolve) returns, plus
    /// [`SelectError::NotImplemented`] for a designed-but-unbuilt policy and
    /// [`SelectError::Incompatible`] for a half-specified grid.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::{KernelChoice, PartitionOpts, PartitionPolicy, ResolvedGemm};
    /// let grid = PartitionPolicy::StaticGrid { pm: 2, pn: 2 };
    /// let rg = ResolvedGemm::<f64>::resolve_with::<f64>(
    ///     &KernelChoice::Auto, 4, grid, PartitionOpts { align_c_lines: true })?;
    /// assert_eq!(rg.partition, grid);
    /// assert!(rg.opts.align_c_lines);
    /// # Ok::<(), tprims_gemm_kernel::SelectError>(())
    /// ```
    pub fn resolve_with<T: Families<Real = R>>(
        choice: &KernelChoice,
        effective_threads: usize,
        partition: PartitionPolicy,
        opts: PartitionOpts,
    ) -> Result<Self, SelectError> {
        match partition {
            PartitionPolicy::DynamicTiles { .. } => {
                return Err(SelectError::NotImplemented {
                    what: "DynamicTiles partition",
                })
            }
            PartitionPolicy::StaticGrid { pm, pn } if (pm == 0) != (pn == 0) => {
                return Err(SelectError::Incompatible {
                    id: "StaticGrid".into(),
                    reason: "pm and pn must both be zero (driver cost model) or both nonzero",
                })
            }
            PartitionPolicy::StaticGrid { .. } => {}
        }
        let mut rg = Self::resolve_selected::<T>(choice, effective_threads)?;
        rg.partition = partition;
        rg.opts = opts;
        Ok(rg)
    }

    fn resolve_selected<T: Families<Real = R>>(
        choice: &KernelChoice,
        effective_threads: usize,
    ) -> Result<Self, SelectError> {
        if effective_threads == 0 {
            return Err(SelectError::Incompatible {
                id: match choice {
                    KernelChoice::Auto => "Auto".into(),
                    KernelChoice::Id(id) => id.clone(),
                },
                reason: "zero effective thread width",
            });
        }
        let cpu = CpuFeatures::detect();
        let id = match choice {
            KernelChoice::Id(id) => id.as_str(),
            KernelChoice::Auto => {
                // INVARIANT: portable families exist for each sealed dtype.
                // Ambiguous registration ids are still rejected by select.
                let families = Registry::families::<T>(cpu, false);
                families
                    .iter()
                    .find(|f| f.allow_auto)
                    .map(|f| f.id)
                    .ok_or_else(|| SelectError::Incompatible {
                        id: "Auto".into(),
                        reason: "no available Auto family",
                    })?
            }
        };
        let family = Registry::select::<T>(id, cpu)?;
        let (a_layout, b_layout, tile_fmt) = family
            .complex
            .map(|s| (s.a, s.b, s.tile))
            .unwrap_or((Layout::Real, Layout::Real, TileFormat::Real));
        let rg = Self {
            family,
            mr: family.mr,
            nr: family.nr,
            a_per_k: family.a_per_k,
            b_per_k: family.b_per_k,
            tile_bound: family.tile_bound,
            a_layout,
            b_layout,
            tile_fmt,
            mc: 0,
            nc: 0,
            kc: 0,
            effective_threads: 0,
            partition: PartitionPolicy::default(),
            opts: PartitionOpts::default(),
            policy: BlockingPolicy::snapshot(),
            gather: crate::writeback::force_gather(),
        };
        rg.with_threads(effective_threads)
    }

    /// Override cache blocking, retaining it when the execution width changes.
    /// Register alignment and the legacy minimum KC of one are preserved.
    ///
    /// # Errors
    /// Returns `Incompatible` if register alignment overflows.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::{Blocking, KernelChoice, ResolvedGemm};
    /// let rg = ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Auto, 1)?;
    /// let rg = rg.with_blocking(Blocking { mc: 5, kc: 3, nc: 7 })?;
    /// assert_eq!(rg.with_threads(4)?.kc, 3);
    /// # Ok::<(), tprims_gemm_kernel::SelectError>(())
    /// ```
    pub fn with_blocking(mut self, blocking: Blocking) -> Result<Self, SelectError> {
        self.policy.explicit = Some(blocking);
        let width = self.effective_threads;
        self.with_threads(width)
    }

    /// Recompute blocking using the effective execution width, not the budget.
    /// Percentage overrides apply once to the original family/model, not to an
    /// already scaled resolution. No selection or environment lookup occurs.
    ///
    /// # Errors
    /// Returns `Incompatible` for zero width or blocking arithmetic overflow.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::{KernelChoice, ResolvedGemm};
    /// let rg = ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Auto, 8)?;
    /// assert_eq!(rg.with_threads(1)?.effective_threads, 1);
    /// # Ok::<(), tprims_gemm_kernel::SelectError>(())
    /// ```
    pub fn with_threads(mut self, threads: usize) -> Result<Self, SelectError> {
        let fail = |reason| SelectError::Incompatible {
            id: self.family.id.into(),
            reason,
        };
        if threads == 0 {
            return Err(fail("zero effective thread width"));
        }
        let blk = self
            .policy
            .blocking(self.family, threads)
            .ok_or_else(|| fail("blocking arithmetic overflow"))?;
        self.mc = blk.mc;
        self.kc = blk.kc;
        self.nc = blk.nc;
        self.effective_threads = threads;
        Ok(self)
    }
}

#[cfg(test)]
mod tests;
