//! Planning-time family selection and frozen blocking policy.
//! Reuses Lukas Devos's tensorcontract blocking geometry and the existing
//! cache model; see `cache` for the Low et al. model and its provenance.

use crate::{
    cache::{self, BlockModel, CacheHierarchy, PanelGeom},
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

#[derive(Clone, Copy, Debug)]
struct BlockingPolicy {
    model: BlockModel,
    hierarchy: CacheHierarchy,
    overrides: Option<BlockingOverride>,
}
impl BlockingPolicy {
    fn snapshot() -> Self {
        Self {
            model: cache::block_model(),
            hierarchy: cache::hierarchy(),
            overrides: env_blocking(),
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
        if let Some(o) = self.overrides {
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
    policy: BlockingPolicy,
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
            policy: BlockingPolicy::snapshot(),
        };
        rg.with_threads(effective_threads)
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
