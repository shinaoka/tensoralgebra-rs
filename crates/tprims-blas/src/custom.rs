//! GEMM with a caller-supplied kernel catalog and selector (issue #28).
//!
//! The caller admits its own static packed kernels once
//! ([`KernelCatalog::from_static_families`], `unsafe`) and passes the catalog
//! with a selection callback. The callback sees metadata only — a
//! [`SelectionContext`] and the admissible [`KernelCandidate`]s — and returns
//! an opaque [`KernelHandle`]. The choice is made once when the call is
//! planned and the frozen choice executes: the callback is never invoked while
//! computing, and nothing is registered or cached process-wide.
//!
//! Every entry point here runs the packed driver ([`Engine::Packed`]); a
//! configuration that asks for another engine, or for a forced kernel id as
//! well, is refused rather than ignored. Falling back to a built-in family is
//! explicit: put [`builtin_catalog`] in the catalog
//! ([`KernelCatalog::union`]) and return one of its handles.
//!
//! The performance protocol of the issue (paired 1T/4T tensor-sized runs with
//! A/A noise) was deferred; nothing here claims a speed-up.
use strided_view::StridedViewMut;
pub use tensorcontract::{KernelCandidate, OperandMeta, SelectionContext};
use tprims_exec::Exec;
pub use tprims_gemm_kernel::{KernelCatalog, KernelHandle, SelectError};

use crate::batched::{batched_impl, BatchIn, BatchStrategy};
use crate::tblis::Custom;
use crate::{
    Conj, EngineChoice, Error, GemmConfig, GroupedJob, MatIn, Result, Scalar, SelectedGemm,
};

/// The built-in and registered families for `T` that this CPU can run, as a
/// caller-scoped catalog. Registers the workspace's kernel providers first.
///
/// # Examples
/// ```
/// let catalog = tprims_blas::builtin_catalog::<f64>();
/// assert!(catalog.get("portable.f64.4x4").is_some());
/// ```
pub fn builtin_catalog<T: Scalar>() -> KernelCatalog<T> {
    crate::engine::register_built();
    KernelCatalog::builtin()
}

/// A selector entry point is the packed driver with the caller's policy. An
/// explicit faer/pgx86 request, a forced kernel id, or a configuration that
/// cannot be honoured is refused.
fn check_config(cfg: &GemmConfig) -> Result<()> {
    match cfg.engine {
        // `Auto` is overridden by the explicit selector, even when the process
        // default engine is another one.
        EngineChoice::Auto | EngineChoice::Packed => {}
        EngineChoice::Faer => {
            return Err(Error::Select(SelectError::EngineUnsupported {
                engine: "faer",
                reason: "a custom kernel selector needs the packed engine",
            }))
        }
        EngineChoice::PrivateGemmX86 => {
            return Err(Error::Select(SelectError::EngineUnsupported {
                engine: "pgx86",
                reason: "a custom kernel selector needs the packed engine",
            }))
        }
    }
    if let tprims_gemm_kernel::KernelChoice::Id(id) = &cfg.kernel {
        return Err(Error::Select(SelectError::Incompatible {
            id: id.clone(),
            reason: "a forced kernel id and a custom selector are ambiguous",
        }));
    }
    Ok(())
}

/// [`gemm_with`](crate::gemm_with), choosing the kernel with `selector` over
/// `catalog`.
///
/// `cfg.engine` must be `Auto` or `Packed`; `cfg.kernel` must be `Auto`;
/// `cfg.method` constrains the complex method as usual. The selector is called
/// once, on this thread, outside every lock and worker broadcast, before any
/// element is touched — including for empty operands and `alpha == 0`, so an
/// invalid selection is reported for those too. The report names the chosen
/// family and its provenance ([`SelectedGemm::origin`]).
///
/// # Errors
///
/// Everything [`gemm_with`](crate::gemm_with) returns, plus [`Error::Select`]
/// for an unusable configuration, an empty or non-admitting catalog
/// (`NoCandidates`), the selector's own `Err` (returned unchanged), or a
/// handle it should not have returned (`ForeignHandle`, `CpuUnsupported`,
/// `NotACandidate`). Nothing is written then.
///
/// # Examples
/// ```
/// use strided_view::{StridedView, StridedViewMut};
/// use tprims_blas::{builtin_catalog, gemm_with_selector, GemmConfig, MatIn};
/// let (a, b) = ([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 1.0]);
/// let mut c = [0.0; 4];
/// let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
/// let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
/// let mut cv = StridedViewMut::new(&mut c, &[2, 2], &[1, 2], 0).unwrap();
/// let catalog = builtin_catalog::<f64>();
/// let report = gemm_with_selector(
///     &tprims_exec::Exec::serial(), &GemmConfig::default(), &catalog,
///     |_ctx, candidates| Ok(candidates[0].handle),
///     1.0, MatIn::new(&av), MatIn::new(&bv), 0.0, &mut cv,
/// ).unwrap();
/// assert_eq!(c, a);
/// assert!(report.family_id.is_some());
/// ```
#[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set.
pub fn gemm_with_selector<T, F>(
    exec: &Exec<'_>,
    cfg: &GemmConfig,
    catalog: &KernelCatalog<T>,
    selector: F,
    alpha: T,
    a: MatIn<'_, '_, T>,
    b: MatIn<'_, '_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
) -> Result<SelectedGemm>
where
    T: Scalar,
    F: FnOnce(
        &SelectionContext<'_>,
        &[KernelCandidate<T>],
    ) -> core::result::Result<KernelHandle<T>, SelectError>,
{
    check_config(cfg)?;
    let s = crate::gemm::check_gemm(
        crate::operand::mat2("A", a.view.dims(), a.view.strides())?,
        crate::operand::mat2("B", b.view.dims(), b.view.strides())?,
        crate::operand::mat2("C", c.dims(), c.strides())?,
    )?;
    let (m, n, k) = (s.c.rows, s.c.cols, s.a.cols);
    let trivial = m == 0 || n == 0 || k == 0 || alpha == crate::scalar::zero();
    let mut selector = Some(selector);
    let mut chooser = |ctx: &SelectionContext<'_>, cands: &[KernelCandidate<T>]| {
        (selector.take().expect("a single-plan selector runs once"))(ctx, cands)
    };
    // SAFETY: the shapes and strides were validated against the views, which
    // were bounds-checked at construction; `c` is an exclusive, injective
    // borrow aliasing neither operand. With `compute == false` no operand
    // pointer is dereferenced.
    let report = unsafe {
        crate::tblis::run_one_plan(
            exec,
            cfg,
            alpha,
            a.view.ptr(),
            a.conj,
            a.view.dims(),
            a.view.strides(),
            b.view.ptr(),
            b.conj,
            b.view.dims(),
            b.view.strides(),
            beta,
            c,
            Some(Custom {
                catalog,
                chooser: &mut chooser,
            }),
            !trivial,
        )?
    };
    if trivial {
        // Every engine agrees that an empty K or a zero alpha only scales C,
        // and none of them may read A or B then.
        // SAFETY: shapes validated; `c` is exclusive and injective.
        unsafe { crate::operand::scale_in_place(c.as_mut_ptr(), &s.c, beta) };
    }
    Ok(report)
}

/// [`gemm_batched_with`](crate::gemm_batched_with) on the packed driver
/// ([`BatchStrategy::Tblis`]), choosing the kernel with `selector` over
/// `catalog`.
///
/// One homogeneous batch is one problem: the selector is called once, for the
/// per-item shape, not once per item. The same configuration rules as
/// [`gemm_with_selector`] apply.
///
/// # Errors
///
/// As [`gemm_batched_with`](crate::gemm_batched_with) and
/// [`gemm_with_selector`].
#[allow(clippy::too_many_arguments)] // INVARIANT: batched GEMM argument set.
pub fn gemm_batched_with_selector<T, F>(
    exec: &Exec<'_>,
    cfg: &GemmConfig,
    catalog: &KernelCatalog<T>,
    selector: F,
    alpha: T,
    a: BatchIn<'_, '_, T>,
    b: BatchIn<'_, '_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
) -> Result<SelectedGemm>
where
    T: Scalar,
    F: FnOnce(
        &SelectionContext<'_>,
        &[KernelCandidate<T>],
    ) -> core::result::Result<KernelHandle<T>, SelectError>,
{
    check_config(cfg)?;
    let mut selector = Some(selector);
    let mut chooser = |ctx: &SelectionContext<'_>, cands: &[KernelCandidate<T>]| {
        (selector.take().expect("a single-plan selector runs once"))(ctx, cands)
    };
    batched_impl(
        exec,
        cfg,
        alpha,
        a,
        b,
        beta,
        c,
        BatchStrategy::Tblis,
        Some(Custom {
            catalog,
            chooser: &mut chooser,
        }),
    )
}

/// [`gemm_grouped_with`](crate::gemm_grouped_with) on the packed driver:
/// every job is its own group plan, so `selector` is called once per non-empty
/// job — with that job's shape — on the calling thread, before any job runs.
/// Jobs then run one after another, each with the width its size warrants.
///
/// The returned report is the first non-empty job's resolution (the jobs may
/// differ; use the selector to log the others). Jobs with no output element
/// need no kernel and never call the selector.
///
/// # Errors
///
/// As [`gemm_grouped_with`](crate::gemm_grouped_with) and
/// [`gemm_with_selector`]. Nothing is written when any job's selection fails.
#[allow(clippy::too_many_arguments)] // INVARIANT: the grouped GEMM argument set.
pub fn gemm_grouped_with_selector<T, F>(
    exec: &Exec<'_>,
    cfg: &GemmConfig,
    catalog: &KernelCatalog<T>,
    mut selector: F,
    alpha: T,
    a: &[T],
    ca: Conj,
    b: &[T],
    cb: Conj,
    beta: T,
    c: &mut [T],
    jobs: &[GroupedJob],
) -> Result<SelectedGemm>
where
    T: Scalar,
    F: FnMut(
        &SelectionContext<'_>,
        &[KernelCandidate<T>],
    ) -> core::result::Result<KernelHandle<T>, SelectError>,
{
    check_config(cfg)?;
    crate::grouped::check_jobs(a, b, c, jobs)?;
    let alpha_zero = alpha == crate::scalar::zero();
    // Plan (and select for) every group before computing any of them, so a
    // failing selection leaves C untouched. Each live job is its own group
    // plan; the selector is not called again once execution starts.
    let mut planned = Vec::new();
    for j in jobs.iter().filter(|j| j.rows > 0 && j.cols > 0) {
        let shape = j.shape()?;
        let mut chooser =
            |ctx: &SelectionContext<'_>, cands: &[KernelCandidate<T>]| selector(ctx, cands);
        let one = crate::tblis::plan_one::<T>(
            exec,
            cfg,
            ca,
            &[shape.a.rows, shape.a.cols],
            &[shape.a.rs, shape.a.cs],
            cb,
            &[shape.b.rows, shape.b.cols],
            &[shape.b.rs, shape.b.cs],
            &[shape.c.rows, shape.c.cols],
            &[shape.c.rs, shape.c.cs],
            Some(Custom {
                catalog,
                chooser: &mut chooser,
            }),
        )?;
        planned.push((*j, shape, one));
    }
    let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    for (j, shape, one) in &planned {
        // SAFETY: `check_jobs` proved every referenced block lies inside its
        // buffer and the non-empty output blocks are pairwise disjoint; `c` is
        // borrowed exclusively for the call. A job with no inner extent or a
        // zero alpha reads neither A nor B.
        unsafe {
            let cj = cp.add(j.c_offset);
            if j.inner == 0 || alpha_zero {
                crate::operand::scale_in_place(cj, &shape.c, beta);
            } else {
                one.execute(
                    exec,
                    alpha,
                    ap.add(j.a_offset),
                    bp.add(j.b_offset),
                    beta,
                    cj,
                );
            }
        }
    }
    // With no output element there is nothing to select for.
    Ok(planned.first().map_or(
        SelectedGemm {
            engine: crate::Engine::Packed,
            family_id: None,
            complex: None,
            mr: 0,
            nr: 0,
            mc: 0,
            nc: 0,
            kc: 0,
            partition: tprims_gemm_kernel::PartitionPolicy::default(),
            batched: None,
            origin: None,
            dynamic: None,
        },
        |(_, _, one)| one.report(),
    ))
}
