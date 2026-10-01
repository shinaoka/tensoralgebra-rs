use strided_view::{StridedView, StridedViewMut};
use tprims_exec::{Exec, Par};

use crate::gemm::{check_gemm, gemm_raw, gemm_width, to_faer, GemmShape, SendConst, SendMut};
use crate::operand::{check_injective, mat2};
use crate::{Conj, Error, Result, Scalar};

/// A rank-3 batched input `[rows, cols, batch]` plus a conjugation flag.
///
/// # Examples
///
/// ```
/// let d = [0.0f64; 8];
/// let v = strided_view::StridedView::new(&d, &[2, 2, 2], &[1, 2, 4], 0).unwrap();
/// assert_eq!(tprims_blas::BatchIn::new(&v).conj, tprims_blas::Conj::No);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BatchIn<'v, 'a, T> {
    /// The borrowed view, batch axis last.
    pub view: &'v StridedView<'a, T>,
    /// Conjugation applied on read.
    pub conj: Conj,
}

impl<'v, 'a, T> BatchIn<'v, 'a, T> {
    /// Use `view` as stored.
    pub fn new(view: &'v StridedView<'a, T>) -> Self {
        Self {
            view,
            conj: Conj::No,
        }
    }

    /// Conjugate on read.
    pub fn conj(mut self) -> Self {
        self.conj = Conj::Yes;
        self
    }
}

/// Which implementation a batched GEMM uses.
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_blas::BatchStrategy::FaerLoop, tprims_blas::BatchStrategy::Tblis);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchStrategy {
    /// Let the library choose (currently [`BatchStrategy::FaerLoop`]; the rule
    /// is replaced once the Phase 1b comparison is recorded).
    Auto,
    /// faer per item, in a loop over items.
    FaerLoop,
    /// TBLIS-style direct kernel (tensorcontract, by Lukas Devos).
    Tblis,
}

/// The implementation and schedule a batched GEMM actually ran.
///
/// # Examples
///
/// ```
/// let s = tprims_blas::Selected::FaerLoop { outer_parallel: false };
/// assert!(matches!(s, tprims_blas::Selected::FaerLoop { .. }));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Selected {
    /// faer per item; `outer_parallel` when items were spread over the pool.
    FaerLoop {
        /// Items ran in parallel, each serially.
        outer_parallel: bool,
    },
    /// TBLIS-style; `outer_parallel` as above.
    Tblis {
        /// Items ran in parallel, each serially.
        outer_parallel: bool,
    },
}

/// Validated batched problem: per-item shape plus batch extent and strides.
pub(crate) struct Batch {
    pub item: GemmShape,
    pub count: usize,
    pub sa: isize,
    pub sb: isize,
    pub sc: isize,
}

fn split3(name: &'static str, v: &[usize], s: &[isize]) -> Result<(usize, isize)> {
    if v.len() != 3 {
        return Err(Error::Rank {
            operand: name,
            expected: 3,
            got: v.len(),
        });
    }
    Ok((v[2], s[2]))
}

pub(crate) fn check_batch<T>(
    a: &StridedView<'_, T>,
    b: &StridedView<'_, T>,
    c: &StridedViewMut<'_, T>,
) -> Result<Batch> {
    let (na, sa) = split3("A", a.dims(), a.strides())?;
    let (nb, sb) = split3("B", b.dims(), b.strides())?;
    let (nc, sc) = split3("C", c.dims(), c.strides())?;
    if na != nb || na != nc {
        return Err(Error::Shape(format!("batch extents {na}, {nb}, {nc}")));
    }
    let item = check_gemm(
        mat2("A", &a.dims()[..2], &a.strides()[..2])?,
        mat2("B", &b.dims()[..2], &b.strides()[..2])?,
        mat2("C", &c.dims()[..2], &c.strides()[..2])?,
    )?;
    let (cd, cs) = (c.dims(), c.strides());
    check_injective(&[(cd[0], cs[0]), (cd[1], cs[1]), (cd[2], cs[2])])?;
    Ok(Batch {
        item,
        count: na,
        sa,
        sb,
        sc,
    })
}

/// How to schedule a batch: `None` = serial loop on the caller;
/// `Some(k)` with `outer` = partition items over `k` workers, each serial;
/// otherwise items one after another with inner width `inner`.
pub(crate) struct Schedule {
    pub outer: Option<usize>,
    pub inner: usize,
}

pub(crate) fn schedule<T: Scalar>(exec: &Exec<'_>, s: &Batch) -> Schedule {
    let (m, n, k) = (s.item.c.rows, s.item.c.cols, s.item.a.cols);
    let inner = gemm_width::<T>(exec, m, n, k);
    let total = gemm_width::<T>(exec, m, n, k.saturating_mul(s.count));
    if inner == 1 && total > 1 && s.count > 1 {
        Schedule {
            outer: Some(total.min(s.count)),
            inner: 1,
        }
    } else {
        Schedule { outer: None, inner }
    }
}

/// Batched `C[:, :, i] = alpha * op(A[:, :, i]) * op(B[:, :, i]) + beta * C[:, :, i]`.
///
/// `strategy` selects faer-per-item or the TBLIS-style kernel; the returned
/// [`Selected`] reports what ran. Small items over a large batch are spread
/// over the pool with each item serial; large items use inner parallelism.
///
/// # Errors
///
/// As [`crate::gemm`], plus [`Error::Shape`] for unequal batch extents and
/// [`Error::AliasedOutput`] when output items overlap.
pub fn gemm_batched<T: Scalar>(
    exec: &Exec<'_>,
    alpha: T,
    a: BatchIn<'_, '_, T>,
    b: BatchIn<'_, '_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
    strategy: BatchStrategy,
) -> Result<Selected> {
    gemm_batched_with(
        exec,
        &crate::GemmConfig::default(),
        alpha,
        a,
        b,
        beta,
        c,
        strategy,
    )
    .map(|sel| sel.batched.expect("the batched report is always set"))
}

/// [`gemm_batched`], with a configuration and a full report.
///
/// The TBLIS-style strategy is the packed driver, so it honours
/// [`crate::GemmConfig::kernel`]; the faer-loop strategies compute with faer and so
/// refuse a configuration that asks for anything else, rather than quietly
/// ignoring it.
///
/// # Errors
///
/// As [`gemm_batched`], plus [`Error::Select`] for an unusable engine, kernel
/// or feature.
#[allow(clippy::too_many_arguments)] // INVARIANT: batched GEMM argument set.
pub fn gemm_batched_with<T: Scalar>(
    exec: &Exec<'_>,
    cfg: &crate::GemmConfig,
    alpha: T,
    a: BatchIn<'_, '_, T>,
    b: BatchIn<'_, '_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
    strategy: BatchStrategy,
) -> Result<crate::SelectedGemm> {
    batched_impl(exec, cfg, alpha, a, b, beta, c, strategy, None)
}

/// Shared body of [`gemm_batched_with`] and the selector entrypoint. With a
/// `custom` selection, the kernel is chosen (once, for the whole homogeneous
/// batch) before any empty-problem shortcut, so a bad selection is reported
/// for zero-size inputs too.
#[allow(clippy::too_many_arguments)] // INVARIANT: batched GEMM argument set.
pub(crate) fn batched_impl<T: Scalar>(
    exec: &Exec<'_>,
    cfg: &crate::GemmConfig,
    alpha: T,
    a: BatchIn<'_, '_, T>,
    b: BatchIn<'_, '_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
    strategy: BatchStrategy,
    custom: Option<crate::tblis::Custom<'_, T>>,
) -> Result<crate::SelectedGemm> {
    if strategy != BatchStrategy::Tblis {
        require_faer_loop_config(cfg)?;
    }
    let s = check_batch(a.view, b.view, c)?;
    let selected = |outer_parallel| match strategy {
        BatchStrategy::Tblis => Selected::Tblis { outer_parallel },
        _ => Selected::FaerLoop { outer_parallel },
    };
    // A custom selection reports the family it chose; the built-in strategies
    // keep their historical report.
    let report = |engine: crate::Engine,
                  rg: Option<&tprims_gemm_kernel::ResolvedGemm<T::Re>>,
                  dynamic: Option<tensorcontract::DynamicReport>,
                  selected: Selected| crate::SelectedGemm {
        engine,
        family_id: rg.map(|rg| rg.family().id),
        complex: rg.and_then(|rg| rg.family().complex),
        mr: rg.map_or(0, |rg| rg.mr),
        nr: rg.map_or(0, |rg| rg.nr),
        mc: rg.map_or(0, |rg| rg.mc),
        nc: rg.map_or(0, |rg| rg.nc),
        kc: rg.map_or(0, |rg| rg.kc),
        partition: rg.map_or(tprims_gemm_kernel::PartitionPolicy::default(), |rg| {
            rg.partition
        }),
        batched: Some(selected),
        origin: rg.map(|rg| rg.family().origin),
        dynamic,
    };
    let (m, n, k) = (s.item.c.rows, s.item.c.cols, s.item.a.cols);
    let trivial = m == 0 || n == 0 || s.count == 0 || k == 0 || alpha == crate::scalar::zero();
    // The selector sees the width the batch will run at.
    let sched = if trivial {
        Schedule {
            outer: None,
            inner: 1,
        }
    } else {
        schedule::<T>(exec, &s)
    };
    // A custom selection, or a partition request, is resolved before any
    // empty-problem shortcut so an invalid choice is reported for empty input.
    let custom_used = custom.is_some();
    let prepared = match (custom, strategy) {
        (Some(custom), _) => Some(crate::tblis::prepare::<T>(
            cfg,
            &s,
            a.conj,
            b.conj,
            sched.inner,
            Some(custom),
        )?),
        (None, BatchStrategy::Tblis) if cfg.has_partition_request() => Some(
            crate::tblis::prepare::<T>(cfg, &s, a.conj, b.conj, sched.inner, None)?,
        ),
        _ => None,
    };
    let dyn_report = prepared
        .as_ref()
        .and_then(|(plan, rg)| tensorcontract::dynamic_report(plan, rg, sched.inner));
    let chosen = prepared.as_ref().filter(|_| custom_used).map(|(_, rg)| rg);
    // Empty problems touch no pointer: an empty view's pointer and batch
    // stride are not validated by strided-view.
    if m == 0 || n == 0 || s.count == 0 {
        return Ok(report(
            engine_of(strategy),
            chosen,
            dyn_report,
            selected(false),
        ));
    }
    if k == 0 || alpha == crate::scalar::zero() {
        // C = beta * C per item; A and B are not referenced. C is non-empty,
        // so its pointer and batch stride were bounds-checked.
        let cp = c.as_mut_ptr();
        for i in 0..s.count {
            // SAFETY: i < count inside C's validated layout.
            unsafe {
                crate::operand::scale_in_place(cp.offset(i as isize * s.sc), &s.item.c, beta)
            };
        }
        return Ok(report(
            engine_of(strategy),
            chosen,
            dyn_report,
            selected(false),
        ));
    }
    // All three operands are non-empty here, so their pointers and batch
    // strides were validated at view construction.
    let (ap, bp, cp) = (
        SendConst(a.view.ptr()),
        SendConst(b.view.ptr()),
        SendMut(c.as_mut_ptr()),
    );
    match strategy {
        BatchStrategy::Tblis => {
            let (plan, rg) = match prepared {
                Some(prepared) => prepared,
                None => crate::tblis::prepare::<T>(cfg, &s, a.conj, b.conj, sched.inner, None)?,
            };
            let (selected, rg) =
                crate::tblis::run(exec, &s, &sched, alpha, ap, bp, beta, cp, plan, rg)?;
            // The built-in strategy keeps its historical report: the family id
            // only. A custom selection reports the whole resolution.
            Ok(report(
                crate::Engine::Packed,
                custom_used.then_some(&rg),
                dyn_report,
                selected,
            ))
        }
        BatchStrategy::Auto | BatchStrategy::FaerLoop => {
            let (ca, cb) = (a.conj, b.conj);
            let item = |i: usize, par: faer::Par| {
                let (ap, bp, cp) = (ap, bp, cp);
                let i = i as isize;
                // SAFETY: offsets `i * stride` stay inside the bounds-checked
                // views (i < count); output items are disjoint (check_batch).
                unsafe {
                    gemm_raw(
                        &s.item,
                        alpha,
                        ap.0.offset(i * s.sa),
                        ca,
                        bp.0.offset(i * s.sb),
                        cb,
                        beta,
                        cp.0.offset(i * s.sc),
                        par,
                    )
                }
            };
            match sched.outer {
                Some(k) => {
                    let lanes = exec.with_budget(k).unwrap_or(*exec);
                    lanes.for_each_partition(s.count, &|i| item(i, faer::Par::Seq));
                }
                None => {
                    // One pool entry for the whole batch.
                    exec.install(sched.inner, |par: Par| {
                        for i in 0..s.count {
                            item(i, to_faer(par));
                        }
                    });
                }
            }
            Ok(report(
                crate::Engine::Faer,
                None,
                None,
                Selected::FaerLoop {
                    outer_parallel: sched.outer.is_some(),
                },
            ))
        }
    }
}

/// The engine a strategy uses, for the report.
fn engine_of(strategy: BatchStrategy) -> crate::Engine {
    match strategy {
        BatchStrategy::Tblis => crate::Engine::Packed,
        _ => crate::Engine::Faer,
    }
}

/// A faer-loop strategy computes with faer, so a configuration that asks for a
/// different engine or a specific kernel family is refused rather than ignored.
fn require_faer_loop_config(cfg: &crate::GemmConfig) -> Result<()> {
    let unsupported = match cfg.engine {
        crate::EngineChoice::Auto | crate::EngineChoice::Faer => {
            cfg.kernel != tprims_gemm_kernel::KernelChoice::Auto
                || cfg.method.is_some()
                || cfg.has_partition_request()
        }
        crate::EngineChoice::Packed => true,
    };
    if unsupported {
        return Err(Error::Select(
            tprims_gemm_kernel::SelectError::EngineUnsupported {
                engine: "batched faer loop",
                reason: "select BatchStrategy::Tblis for a packed engine or a named kernel",
            },
        ));
    }
    Ok(())
}
