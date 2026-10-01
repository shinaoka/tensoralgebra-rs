use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_blas::{Conj, Scalar};
use tprims_exec::Exec;

use crate::permute_gemm::{self, PgPlan};
use crate::tblis::{self, TbPlan};
use crate::util::select_err;
use crate::{DotGeneral, Error, Result};

/// Which implementation a plan uses.
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_contract::Strategy::PermuteGemm, tprims_contract::Strategy::Tblis);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Strategy {
    /// Library choice: an elementwise pass for all-batch (Hadamard)
    /// problems; otherwise [`Strategy::PermuteGemm`] when it fuses every
    /// operand without a copy, and [`Strategy::Tblis`] when permute+GEMM
    /// would copy any operand. Measured on the tenferro-benchmark shape
    /// corpus (Phase 1e P2, `docs/decision-log.md`): permute+GEMM wins the
    /// copy-free problems, TBLIS-style wins the copying ones, which carry
    /// most of the workload time. With [`Flags::no_materialize`], a copying
    /// problem therefore plans TBLIS-style instead of failing.
    Auto,
    /// Fuse to strided batched GEMM, copying non-fusable operands once.
    PermuteGemm,
    /// TBLIS-style direct contraction (tensorcontract, by Lukas Devos).
    Tblis,
}

/// Planning options.
///
/// # Examples
///
/// ```
/// assert!(!tprims_contract::Flags::default().no_materialize);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// Refuse a plan that would copy any operand ([`Error::WouldMaterialize`]).
    pub no_materialize: bool,
}

/// What a plan runs.
///
/// # Examples
///
/// ```
/// let s = tprims_contract::Selected::PermuteGemm { materialized: [false; 3] };
/// assert!(matches!(s, tprims_contract::Selected::PermuteGemm { .. }));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Selected {
    /// Batched GEMM; which of A, B, C are copied into compact buffers.
    PermuteGemm {
        /// A, B, C copied.
        materialized: [bool; 3],
    },
    /// Direct contraction; nothing is copied.
    Tblis,
    /// All axes are batch axes (Hadamard product): one elementwise pass.
    Elementwise,
}

#[derive(Debug)]
enum Inner {
    Pg(Box<PgPlan>),
    Tb(Box<TbPlan>),
    /// A and B axis of each output (batch) axis.
    Elementwise {
        a_axes: Vec<usize>,
        b_axes: Vec<usize>,
    },
}

/// A validated contraction for fixed layouts, reusable across calls.
#[derive(Debug)]
pub struct ContractPlan<T> {
    layouts: [(Vec<usize>, Vec<isize>); 3],
    conj: (Conj, Conj),
    k_empty: bool,
    inner: Inner,
    _t: std::marker::PhantomData<fn() -> T>,
}

type Lay<'a> = (&'a [usize], &'a [isize]);

impl<T: Scalar> ContractPlan<T> {
    /// Validate `cfg` against the layouts (extents and element strides of A,
    /// B and C) and choose a strategy.
    ///
    /// # Errors
    ///
    /// [`Error::Config`], [`Error::Shape`] (including C's extents),
    /// [`Error::AliasedOutput`], [`Error::WouldMaterialize`] under
    /// `no_materialize`, [`Error::Backend`] when tensorcontract rejects the
    /// problem.
    pub fn new(
        cfg: &DotGeneral,
        a: Lay<'_>,
        b: Lay<'_>,
        c: Lay<'_>,
        conj: (Conj, Conj),
        strategy: Strategy,
        flags: Flags,
    ) -> Result<Self> {
        Self::new_with(
            &tprims_blas::GemmConfig::default(),
            cfg,
            a,
            b,
            c,
            conj,
            strategy,
            flags,
        )
    }

    /// [`ContractPlan::new`], choosing the matrix engine and kernel.
    ///
    /// The choice is resolved *here*, so an unknown or unusable kernel id is an
    /// error from this call rather than from the first contraction that runs.
    ///
    /// # Errors
    ///
    /// As [`ContractPlan::new`], plus [`Error::Backend`] when the GEMM
    /// configuration cannot be used.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    pub fn new_with(
        gemm: &tprims_blas::GemmConfig,
        cfg: &DotGeneral,
        a: Lay<'_>,
        b: Lay<'_>,
        c: Lay<'_>,
        conj: (Conj, Conj),
        strategy: Strategy,
        flags: Flags,
    ) -> Result<Self> {
        Self::build(gemm, cfg, a, b, c, conj, strategy, flags, None)
    }

    /// [`ContractPlan::new_with`] on the TBLIS-style packed driver, choosing the
    /// kernel with a caller-supplied selector over a caller-supplied
    /// [`KernelCatalog`](tprims_blas::KernelCatalog) (issue #28).
    ///
    /// The selector sees metadata only — the folded problem and each admissible
    /// candidate, with the thread budget `exec` grants this problem — and
    /// returns an opaque handle. It is called once, here, outside every lock
    /// and worker broadcast; [`execute`](Self::execute) never calls it, and the
    /// plan keeps only the chosen trusted handle, so the selector and catalog
    /// may be dropped. The performance protocol of the issue was deferred.
    ///
    /// The selector is a requirement, not a hint, so strategies that cannot
    /// honour it are errors rather than silent fallbacks:
    /// [`Strategy::PermuteGemm`] (computes with faer, and copies), and
    /// [`Strategy::Auto`] on an all-batch (elementwise) problem. `Auto` and
    /// `Tblis` otherwise plan the packed driver; nothing is copied.
    ///
    /// # Errors
    ///
    /// As [`ContractPlan::new_with`], plus a typed `SelectError` source in [`Error::Backend`] for an
    /// engine other than `Auto`/`Packed`, a forced kernel id alongside the
    /// selector, an incompatible strategy, no admissible candidate, the
    /// selector's own `Err` (unchanged) or a handle it should not have
    /// returned. Zero-size problems are selected and validated too.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    pub fn new_with_selector<F>(
        exec: &Exec<'_>,
        gemm: &tprims_blas::GemmConfig,
        catalog: &tprims_blas::KernelCatalog<T>,
        selector: F,
        cfg: &DotGeneral,
        a: Lay<'_>,
        b: Lay<'_>,
        c: Lay<'_>,
        conj: (Conj, Conj),
        strategy: Strategy,
        flags: Flags,
    ) -> Result<Self>
    where
        F: FnOnce(
            &tprims_blas::SelectionContext<'_>,
            &[tprims_blas::KernelCandidate<T>],
        )
            -> std::result::Result<tprims_blas::KernelHandle<T>, tprims_blas::SelectError>,
    {
        use tprims_blas::SelectError;
        match gemm.engine {
            tprims_blas::EngineChoice::Auto | tprims_blas::EngineChoice::Packed => {}
            _ => {
                return Err(select_err(SelectError::EngineUnsupported {
                    engine: "contract",
                    reason: "a custom kernel selector needs the packed engine",
                }));
            }
        }
        if let tprims_blas::KernelChoice::Id(id) = &gemm.kernel {
            return Err(select_err(SelectError::Incompatible {
                id: id.clone(),
                reason: "a forced kernel id and a custom selector are ambiguous",
            }));
        }
        if strategy == Strategy::PermuteGemm {
            return Err(select_err(SelectError::EngineUnsupported {
                engine: "permute+GEMM",
                reason: "the permute+GEMM strategy computes with faer; use Strategy::Tblis",
            }));
        }
        let mut selector = Some(selector);
        let mut chooser = |ctx: &tprims_blas::SelectionContext<'_>,
                           cands: &[tprims_blas::KernelCandidate<T>]| {
            (selector.take().expect("a single-plan selector runs once"))(ctx, cands)
        };
        // The width the executor will grant this problem, as `execute` will
        // compute it, so the selector sees the real budget.
        let k: usize = cfg
            .lhs_contract
            .iter()
            .map(|&x| a.0.get(x).copied().unwrap_or(0))
            .product();
        let out: usize = c.0.iter().product();
        let flops = 2.0 * out as f64 * k as f64 * if T::IS_COMPLEX_SCALAR { 4.0 } else { 1.0 };
        let threads = exec
            .width_for(
                flops * tprims_blas::GemmPolicy::default().ns_per_flop,
                &tprims_exec::WidthPolicy::default(),
            )
            .max(1);
        let custom = crate::tblis::Custom {
            catalog,
            chooser: &mut chooser,
            threads,
            method: gemm.method,
        };
        Self::build(gemm, cfg, a, b, c, conj, strategy, flags, Some(custom))
    }

    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    fn build(
        gemm: &tprims_blas::GemmConfig,
        cfg: &DotGeneral,
        a: Lay<'_>,
        b: Lay<'_>,
        c: Lay<'_>,
        conj: (Conj, Conj),
        strategy: Strategy,
        flags: Flags,
        custom: Option<crate::tblis::Custom<'_, T>>,
    ) -> Result<Self> {
        let tprims_contract_traits::Validated {
            shape,
            k_empty,
            all_batch,
        } = tprims_contract_traits::validate_layouts::<T>(cfg, a, b, c)?;
        let dims = [a.0, b.0, c.0];
        let strides = [a.1, b.1, c.1];
        let inner = match strategy {
            // A custom selector is a requirement: an elementwise pass has no
            // kernel to select, so it is refused rather than ignoring it.
            Strategy::Auto if all_batch && custom.is_some() => {
                return Err(select_err(tprims_blas::SelectError::EngineUnsupported {
                    engine: "elementwise",
                    reason: "an all-batch problem has no kernel to select; use Strategy::Tblis",
                }));
            }
            _ if custom.is_some() => Inner::Tb(Box::new(tblis::plan::<T>(
                cfg, &shape, dims, strides, conj, gemm, custom,
            )?)),
            // A partition policy is a requirement of the packed driver; the
            // elementwise pass has no output grid to assign.
            Strategy::Auto | Strategy::PermuteGemm if all_batch && gemm.has_partition_request() => {
                return Err(select_err(tprims_blas::SelectError::EngineUnsupported {
                    engine: "elementwise",
                    reason: "an all-batch problem has no partition to choose; use Strategy::Tblis",
                }));
            }
            Strategy::Auto | Strategy::PermuteGemm if all_batch => Inner::Elementwise {
                a_axes: cfg.lhs_batch.clone(),
                b_axes: cfg.rhs_batch.clone(),
            },
            Strategy::Tblis => Inner::Tb(Box::new(tblis::plan::<T>(
                cfg, &shape, dims, strides, conj, gemm, None,
            )?)),
            Strategy::PermuteGemm => Inner::Pg(Box::new(permute_gemm::plan(
                cfg,
                &shape,
                dims,
                strides,
                flags.no_materialize,
                gemm,
            )?)),
            Strategy::Auto => {
                // An explicit engine or kernel is a requirement, not a hint:
                // it is met by the packed driver or it is an error. The
                // copying permute+GEMM plan is only a fallback for the default
                // configuration.
                let wants_packed = gemm.kernel != tprims_blas::KernelChoice::Auto
                    || gemm.has_partition_request()
                    || matches!(gemm.engine, tprims_blas::EngineChoice::Packed);
                let pg = permute_gemm::plan(cfg, &shape, dims, strides, false, gemm)?;
                if pg.materialized.iter().any(|&m| m) || wants_packed {
                    match tblis::plan::<T>(cfg, &shape, dims, strides, conj, gemm, None) {
                        Ok(tb) => Inner::Tb(Box::new(tb)),
                        // tensorcontract declined: keep the copying plan
                        // unless copies were refused or the driver was asked
                        // for.
                        Err(e) if flags.no_materialize || wants_packed => return Err(e),
                        Err(_) => Inner::Pg(Box::new(pg)),
                    }
                } else {
                    Inner::Pg(Box::new(pg))
                }
            }
        };
        let plan = Self {
            layouts: [0, 1, 2].map(|o| (dims[o].to_vec(), strides[o].to_vec())),
            conj,
            k_empty,
            inner,
            _t: std::marker::PhantomData,
        };
        // Resolve now: a kernel this problem cannot use is a configuration
        // error, not something to discover inside a contraction.
        match &plan.inner {
            Inner::Tb(_) => {
                plan.resolved_gemm()?;
            }
            Inner::Pg(_) => {
                // The permute+GEMM arm computes with faer, so a configuration
                // that asks for another engine or a named kernel has no arm
                // here; saying so now beats a surprise when it runs.
                let unsupported = gemm.kernel != tprims_blas::KernelChoice::Auto
                    || gemm.has_partition_request()
                    || matches!(
                        gemm.engine,
                        tprims_blas::EngineChoice::PrivateGemmX86
                            | tprims_blas::EngineChoice::Packed
                    );
                if unsupported {
                    return Err(Error::Unsupported(
                        "the permute+GEMM strategy computes with faer; \
                         select Strategy::Tblis for a packed engine or a named kernel"
                            .into(),
                    ));
                }
            }
            Inner::Elementwise { .. } => {}
        }
        Ok(plan)
    }

    /// What the packed plan resolved to: the family, its geometry and the grid.
    ///
    /// `None` for the strategies that do not use the packed driver. The
    /// resolution is the plan's own cached one, so this is a lookup, not a
    /// re-selection.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] when a forced kernel cannot serve this contraction.
    pub fn selected_gemm(&self) -> Result<Option<tprims_blas::SelectedGemm>> {
        match self.inner {
            Inner::Tb(_) => Ok(Some(self.resolved_gemm()?)),
            _ => Ok(None),
        }
    }

    /// The packed plan's resolution, as a report.
    fn resolved_gemm(&self) -> Result<tprims_blas::SelectedGemm> {
        let Inner::Tb(tb) = &self.inner else {
            return Err(Error::backend("not a packed plan"));
        };
        let rg = tb.resolved::<T>().map_err(Error::backend)?;
        Ok(tprims_blas::SelectedGemm {
            engine: tprims_blas::Engine::Packed,
            family_id: Some(rg.family().id),
            complex: rg.family().complex,
            mr: rg.mr,
            nr: rg.nr,
            mc: rg.mc,
            nc: rg.nc,
            kc: rg.kc,
            partition: rg.partition,
            batched: None,
            origin: Some(rg.family().origin),
            dynamic: tb.dynamic(&rg),
        })
    }

    /// The implementation this plan runs.
    pub fn selected(&self) -> Selected {
        match &self.inner {
            Inner::Pg(p) => Selected::PermuteGemm {
                materialized: p.materialized,
            },
            Inner::Tb(_) => Selected::Tblis,
            Inner::Elementwise { .. } => Selected::Elementwise,
        }
    }

    /// `C = alpha * contract(op(A), op(B)) + beta * C` on views with exactly
    /// the planned layouts. `beta == 0` never reads C; `alpha == 0` or an
    /// empty contraction never reads A or B.
    ///
    /// # Errors
    ///
    /// [`Error::LayoutMismatch`] when a view differs from the plan (nothing is
    /// written); [`Error::Backend`] from a lower layer.
    pub fn execute(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        c: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        let views = [
            (a.dims(), a.strides()),
            (b.dims(), b.strides()),
            (c.dims(), c.strides()),
        ];
        for (o, (name, v)) in ["A", "B", "C"].iter().zip(views).enumerate() {
            if v.0 != self.layouts[o].0.as_slice() || v.1 != self.layouts[o].1.as_slice() {
                return Err(Error::LayoutMismatch(format!(
                    "{name}: {:?} / {:?}",
                    v.0, v.1
                )));
            }
        }
        if c.dims().contains(&0) {
            return Ok(());
        }
        if self.k_empty || alpha == <T as Element>::zero() || a.is_empty() || b.is_empty() {
            let (dims, strides) = (c.dims().to_vec(), c.strides().to_vec());
            // SAFETY: C is non-empty and bounds-checked; exclusive borrow.
            unsafe { crate::util::scale(c.as_mut_ptr(), &dims, &strides, beta) };
            return Ok(());
        }
        match &self.inner {
            Inner::Pg(p) => {
                permute_gemm::execute(p, exec, alpha, a, self.conj.0, b, self.conj.1, beta, c)
            }
            Inner::Tb(p) => tblis::execute(p, exec, alpha, a, b, beta, c),
            Inner::Elementwise { a_axes, b_axes } => {
                let (dims, cs) = (c.dims().to_vec(), c.strides().to_vec());
                let sa: Vec<isize> = a_axes.iter().map(|&x| a.strides()[x]).collect();
                let sb: Vec<isize> = b_axes.iter().map(|&x| b.strides()[x]).collect();
                let (ca, cb) = self.conj;
                let op = |x: T, cj: Conj| if cj == Conj::Yes { Element::conj(x) } else { x };
                let read = beta != <T as Element>::zero();
                // SAFETY: all three views are non-empty and bounds-checked;
                // A and B axes are permuted onto C's (batch) axes, whose
                // extents they share; C is exclusive and injective.
                unsafe {
                    crate::util::zip_update(
                        exec,
                        &dims,
                        (c.as_mut_ptr(), &cs),
                        [(a.ptr(), &sa), (b.ptr(), &sb)],
                        read,
                        &move |y, [x, z]| {
                            Element::add(
                                Element::mul(alpha, Element::mul(op(x, ca), op(z, cb))),
                                Element::mul(beta, y),
                            )
                        },
                    )
                };
                Ok(())
            }
        }
    }
}
