//! TBLIS-style batched GEMM through `tensorcontract`, the direct tensor
//! contraction engine of Lukas Devos's tensorprimitives-rs (after D. A.
//! Matthews, *High-Performance Tensor Contraction without Transposition*,
//! arXiv:1607.00291).
//!
//! One `Plan` is built per call for the item shape; items run either spread
//! over the pool (each serial, width-one seam) or one after another, each on
//! the pool's co-scheduled threads through `Exec::broadcast`.
use strided_view::StridedViewMut;
use tensorcontract::{Layout, Operand, Plan};
use tprims_exec::Exec;

use crate::batched::{Batch, Schedule, Selected};
use crate::gemm::{SendConst, SendMut};
use crate::operand::Mat2;
use crate::{Conj, Error, Result, Scalar};

/// A selection failure is the caller's engine choice and keeps its own type;
/// anything else is a contraction error.
fn as_select(e: tensorcontract::Error) -> Error {
    match e {
        tensorcontract::Error::KernelSelection(s) => Error::Select(s),
        other => Error::Contract(other),
    }
}

/// A caller's catalog and selector, as one planning input.
///
/// The selector is called at most once per plan built from it and is never
/// stored in the plan: only the chosen trusted handle is.
pub(crate) struct Custom<'a, T: Scalar> {
    pub catalog: &'a tprims_gemm_kernel::KernelCatalog<T>,
    pub chooser: &'a mut tensorcontract::Chooser<'a, T>,
}

/// Apply the caller's selector to a plan built from `ops` (A, B, C, D).
fn select<T: Scalar>(plan: Plan, ops: [Operand<'_>; 4], custom: Custom<'_, T>) -> Result<Plan> {
    plan.with_selector::<T, _>(ops, custom.catalog, custom.chooser)
        .map_err(as_select)
}

fn op(o: Operand<'_>, c: Conj) -> Operand<'_> {
    if c == Conj::Yes {
        o.conj()
    } else {
        o
    }
}

fn layout(m: &Mat2) -> Result<Layout> {
    let i = |x: usize| i64::try_from(x).map_err(|_| Error::Shape("extent exceeds i64".into()));
    Ok(Layout::new(
        vec![i(m.rows)?, i(m.cols)?],
        vec![m.rs as i64, m.cs as i64],
    )?)
}

/// Build the per-item plan once and resolve its kernel (by the configured id
/// or by the caller's selector), so a selection failure surfaces before any
/// item runs.
pub(crate) fn prepare<T: Scalar>(
    cfg: &crate::GemmConfig,
    s: &Batch,
    ca: Conj,
    cb: Conj,
    threads: usize,
    custom: Option<Custom<'_, T>>,
) -> Result<(Plan, tprims_gemm_kernel::ResolvedGemm<T::Re>)> {
    // Every id-resolving packed entry point registers the providers first, so
    // a forced id never depends on another call having done it.
    crate::engine::register_built();
    let (la, lb, ld) = (layout(&s.item.a)?, layout(&s.item.b)?, layout(&s.item.c)?);
    let (ia, ib, id) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    let (oa, ob) = (
        op(Operand::new(&la, &ia), ca),
        op(Operand::new(&lb, &ib), cb),
    );
    let (oc, od) = (Operand::new(&ld, &id), Operand::new(&ld, &id));
    // C is D itself (in place); beta == 0 never reads it.
    let plan = Plan::new(oa, ob, Some(oc), od)?;
    let plan = if cfg.has_partition_request() {
        plan.with_partition(cfg.partition, cfg.partition_opts)
    } else {
        plan
    };
    let plan = match custom {
        Some(custom) => select(plan.with_threads(threads), [oa, ob, oc, od], custom)?,
        None => plan.with_kernel(cfg.kernel.clone()).map_err(as_select)?,
    };
    // Resolve once, here: a selection failure is the caller's configuration
    // and must surface before any item runs, not as a panic inside the first
    // one.
    let rg = plan.resolved::<T>().map_err(Error::Select)?;
    Ok((plan, rg))
}

/// Run a prepared batch: one co-scheduled broadcast per item, or the items
/// spread over the pool, each serial.
#[allow(clippy::too_many_arguments)] // INVARIANT: batched GEMM argument set.
pub(crate) fn run<T: Scalar>(
    exec: &Exec<'_>,
    s: &Batch,
    sched: &Schedule,
    alpha: T,
    a: SendConst<T>,
    b: SendConst<T>,
    beta: T,
    c: SendMut<T>,
    plan: Plan,
    rg: tprims_gemm_kernel::ResolvedGemm<T::Re>,
) -> Result<(Selected, tprims_gemm_kernel::ResolvedGemm<T::Re>)> {
    let item = |i: usize, exec: &Exec<'_>| {
        let (a, b, c) = (a, b, c);
        let i = i as isize;
        // SAFETY: the plan's scatters address exactly the validated item
        // layout; offsets `i * stride` stay inside the bounds-checked views;
        // output items are disjoint (check_batch); `c == d` is tensorcontract's
        // in-place form.
        unsafe {
            let d = c.0.offset(i * s.sc);
            plan.run_raw_with(
                exec,
                None,
                alpha,
                a.0.offset(i * s.sa),
                b.0.offset(i * s.sb),
                beta,
                d,
                d,
            );
        }
    };
    match sched.outer {
        Some(k) => {
            let lanes = exec.with_budget(k).unwrap_or(*exec);
            lanes.for_each_partition(s.count, &|i| item(i, &Exec::Serial));
        }
        None => {
            // Each item is one co-scheduled SPMD broadcast of its own.
            let inner = exec.with_budget(sched.inner).unwrap_or(*exec);
            for i in 0..s.count {
                item(i, &inner);
            }
        }
    }
    Ok((
        Selected::Tblis {
            outer_parallel: sched.outer.is_some(),
        },
        rg,
    ))
}

/// A planned single GEMM: the plan, its frozen resolution and the width it
/// was planned for. Nothing here refers to the caller's selector.
pub(crate) struct OnePlan<T: Scalar> {
    plan: Plan,
    rg: tprims_gemm_kernel::ResolvedGemm<T::Re>,
    width: usize,
}

impl<T: Scalar> OnePlan<T> {
    /// What the planner resolved, as the public report.
    pub(crate) fn report(&self) -> crate::SelectedGemm {
        let rg = &self.rg;
        crate::SelectedGemm {
            origin: Some(rg.family().origin),
            engine: crate::Engine::Packed,
            family_id: Some(rg.family().id),
            complex: rg.family().complex,
            mr: rg.mr,
            nr: rg.nr,
            mc: rg.mc,
            nc: rg.nc,
            kc: rg.kc,
            partition: rg.partition,
            batched: None,
            dynamic: tensorcontract::dynamic_report(&self.plan, rg, self.width),
        }
    }

    /// Execute the frozen plan: no selection, lookup or registration.
    ///
    /// # Safety
    /// The GEMM contract of [`run_one_plan`] for the layouts planned.
    pub(crate) unsafe fn execute(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        d: *mut T,
    ) {
        let exec = exec.with_budget(self.width).unwrap_or(*exec);
        // SAFETY: the caller's contract; `C` is `D`, which is this driver's
        // in-place form, and the plan was built from the same layouts.
        unsafe {
            tensorcontract::execute_resolved(
                &self.plan, &self.rg, &exec, None, alpha, a, b, beta, d, d,
            )
        };
    }
}

/// Plan one packed-driver GEMM for the given layouts, with the configured
/// kernel or the caller's selector. Reads no operand data.
#[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set.
pub(crate) fn plan_one<T: Scalar>(
    exec: &Exec<'_>,
    cfg: &crate::GemmConfig,
    ca: Conj,
    a_dims: &[usize],
    a_strides: &[isize],
    cb: Conj,
    b_dims: &[usize],
    b_strides: &[isize],
    c_dims: &[usize],
    c_strides: &[isize],
    custom: Option<Custom<'_, T>>,
) -> Result<OnePlan<T>> {
    let i = |x: usize| i64::try_from(x).map_err(|_| Error::Shape("extent exceeds i64".into()));
    let la = Layout::new(
        a_dims.iter().map(|&d| i(d)).collect::<Result<Vec<_>>>()?,
        a_strides.iter().map(|&s| s as i64).collect(),
    )?;
    let lb = Layout::new(
        b_dims.iter().map(|&d| i(d)).collect::<Result<Vec<_>>>()?,
        b_strides.iter().map(|&s| s as i64).collect(),
    )?;
    let ld = Layout::new(
        c_dims.iter().map(|&d| i(d)).collect::<Result<Vec<_>>>()?,
        c_strides.iter().map(|&s| s as i64).collect(),
    )?;
    // Registering before the plan is built is what makes an id from an absent
    // provider report its feature instead of looking unknown.
    crate::engine::register_built();
    let (ia, ib, idd) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    let (oa, ob) = (
        op(Operand::new(&la, &ia), ca),
        op(Operand::new(&lb, &ib), cb),
    );
    let (oc, od) = (Operand::new(&ld, &idd), Operand::new(&ld, &idd));
    let mut plan = Plan::new(oa, ob, Some(oc), od)?;
    if custom.is_none() {
        plan = plan.with_kernel(cfg.kernel.clone()).map_err(as_select)?;
    }
    if cfg.has_partition_request() {
        plan = plan.with_partition(cfg.partition, cfg.partition_opts);
    }
    if let Some(method) = cfg.method {
        // A method the driver's planner can express goes into the plan; the
        // rest is checked against the family below, so a mismatch is an error
        // rather than a silent substitution.
        match method {
            tprims_gemm_kernel::Method::OneM => {
                plan = plan.with_complex_method(tensorcontract::ComplexMethod::OneM);
            }
            tprims_gemm_kernel::Method::ThreeM => {
                plan = plan.with_complex_method(tensorcontract::ComplexMethod::ThreeM);
            }
            _ => {}
        }
    }
    let width = crate::gemm::gemm_width::<T>(exec, c_dims[0], c_dims[1], a_dims[1]);
    if let Some(custom) = custom {
        // Last, so the selector sees the width and method this call will use.
        plan = select(plan.with_threads(width), [oa, ob, oc, od], custom)?;
    }
    let rg = plan
        .resolved::<T>()
        .and_then(|rg| rg.with_threads(width))
        .map_err(Error::Select)?;
    if let Some(method) = cfg.method {
        let family_method = rg.family().complex.map(|s| s.method);
        let matches = match method {
            tprims_gemm_kernel::Method::Native => {
                family_method.is_none_or(|m| m == tprims_gemm_kernel::Method::Native)
            }
            other => family_method == Some(other),
        };
        if !matches {
            return Err(Error::Select(
                tprims_gemm_kernel::SelectError::Incompatible {
                    id: rg.family().id.into(),
                    reason: "family implements a different complex method",
                },
            ));
        }
    }
    Ok(OnePlan { plan, rg, width })
}

/// One packed-driver GEMM for the whole matrix, with the configured kernel.
///
/// The batched path builds a plan per item; a plain matrix GEMM has exactly one
/// item, so this is the same construction without the item loop, plus the
/// report of what the planner resolved. With `compute == false` the plan is
/// built and validated but no operand is read or written.
///
/// # Safety
/// The caller's GEMM contract: the operands' shapes and strides describe valid
/// readable `m x k` and `k x n` regions, and `c` is an exclusive, injective
/// `m x n` region that aliases neither.
#[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set.
pub(crate) unsafe fn run_one_plan<T: Scalar>(
    exec: &Exec<'_>,
    cfg: &crate::GemmConfig,
    alpha: T,
    a: *const T,
    ca: Conj,
    a_dims: &[usize],
    a_strides: &[isize],
    b: *const T,
    cb: Conj,
    b_dims: &[usize],
    b_strides: &[isize],
    beta: T,
    c: &mut StridedViewMut<'_, T>,
    custom: Option<Custom<'_, T>>,
    compute: bool,
) -> Result<crate::SelectedGemm> {
    let one = plan_one::<T>(
        exec,
        cfg,
        ca,
        a_dims,
        a_strides,
        cb,
        b_dims,
        b_strides,
        c.dims(),
        c.strides(),
        custom,
    )?;
    if compute {
        let dp = c.as_mut_ptr();
        // SAFETY: the caller's contract, forwarded.
        unsafe { one.execute(exec, alpha, a, b, beta, dp) };
    }
    Ok(one.report())
}
