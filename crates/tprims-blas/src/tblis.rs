//! TBLIS-style batched GEMM through `tensorcontract`, the direct tensor
//! contraction engine of Lukas Devos's tensorprimitives-rs (after D. A.
//! Matthews, *High-Performance Tensor Contraction without Transposition*,
//! arXiv:1607.00291).
//!
//! One `Plan` is built per call for the item shape; items run either spread
//! over the pool (each serial, width-one seam) or one after another, each on
//! the pool's co-scheduled threads through the `Spmd` seam.
use strided_view::StridedViewMut;
use tensorcontract::spmd::Spmd;
use tensorcontract::{Layout, Operand, Plan};
use tprims_exec::Exec;

use crate::batched::{Batch, Schedule, Selected};
use crate::gemm::{SendConst, SendMut};
use crate::operand::Mat2;
use crate::{Conj, Error, Result, Scalar};

/// `Spmd` on an `Exec`: co-scheduled threads come from the borrowed pool.
pub(crate) struct ExecSpmd<'a> {
    pub exec: &'a Exec<'a>,
    pub width: usize,
    /// Storage this operation's threads share: the pool's own when it has one.
    pub workspace: Option<&'a dyn tprims_gemm_kernel::WorkspaceProvider>,
}

impl Spmd for ExecSpmd<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
        self.exec.broadcast(p, f).is_ok()
    }
    fn workspace(&self) -> Option<&dyn tprims_gemm_kernel::WorkspaceProvider> {
        self.workspace
    }
}

/// A selection failure is the caller's engine choice and keeps its own type;
/// anything else is a contraction error.
fn as_select(e: tensorcontract::Error) -> Error {
    match e {
        tensorcontract::Error::KernelSelection(s) => Error::Select(s),
        other => Error::Contract(other),
    }
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

#[allow(clippy::too_many_arguments)] // INVARIANT: batched GEMM argument set.
pub(crate) fn run<T: Scalar>(
    exec: &Exec<'_>,
    cfg: &crate::GemmConfig,
    s: &Batch,
    sched: &Schedule,
    alpha: T,
    ca: Conj,
    a: SendConst<T>,
    cb: Conj,
    b: SendConst<T>,
    beta: T,
    c: SendMut<T>,
) -> Result<(Selected, Option<&'static str>)> {
    let (la, lb, ld) = (layout(&s.item.a)?, layout(&s.item.b)?, layout(&s.item.c)?);
    let (ia, ib, id) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    // C is D itself (in place); beta == 0 never reads it.
    let plan = Plan::new(
        op(Operand::new(&la, &ia), ca),
        op(Operand::new(&lb, &ib), cb),
        Some(Operand::new(&ld, &id)),
        Operand::new(&ld, &id),
    )?
    .with_kernel(cfg.kernel.clone())
    .map_err(as_select)?;
    // Resolve once, here: a selection failure is the caller's configuration
    // and must surface before any item runs, not as a panic inside the first
    // one.
    let family_id = Some(plan.resolved::<T>().map_err(Error::Select)?.family().id);
    let item = |i: usize, spmd: &dyn Spmd| {
        let (a, b, c) = (a, b, c);
        let i = i as isize;
        // SAFETY: the plan's scatters address exactly the validated item
        // layout; offsets `i * stride` stay inside the bounds-checked views;
        // output items are disjoint (check_batch); `c == d` is tensorcontract's
        // in-place form.
        unsafe {
            let d = c.0.offset(i * s.sc);
            plan.run_raw_with(
                spmd,
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
            let serial = ExecSpmd {
                exec: &Exec::Serial,
                width: 1,
                workspace: None,
            };
            lanes.for_each_partition(s.count, &|i| item(i, &serial));
        }
        None => {
            // Each item is one co-scheduled SPMD broadcast of its own.
            let spmd = ExecSpmd {
                exec,
                width: sched.inner,
                workspace: exec.workspace(),
            };
            for i in 0..s.count {
                item(i, &spmd);
            }
        }
    }
    Ok((
        Selected::Tblis {
            outer_parallel: sched.outer.is_some(),
        },
        family_id,
    ))
}

/// One packed-driver GEMM for the whole matrix, with the configured kernel.
///
/// The batched path builds a plan per item; a plain matrix GEMM has exactly one
/// item, so this is the same construction without the item loop, plus the
/// report of what the planner resolved.
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
) -> Result<crate::SelectedGemm> {
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
        c.dims().iter().map(|&d| i(d)).collect::<Result<Vec<_>>>()?,
        c.strides().iter().map(|&s| s as i64).collect(),
    )?;
    // Registering before the plan is built is what makes an id from an absent
    // provider report its feature instead of looking unknown.
    crate::engine::register_built();
    let (ia, ib, idd) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    let mut plan = Plan::new(
        op(Operand::new(&la, &ia), ca),
        op(Operand::new(&lb, &ib), cb),
        Some(Operand::new(&ld, &idd)),
        Operand::new(&ld, &idd),
    )?
    .with_kernel(cfg.kernel.clone())
    .map_err(as_select)?;
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
    let width = crate::gemm::gemm_width::<T>(exec, c.dims()[0], c.dims()[1], a_dims[1]);
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
    let spmd = ExecSpmd {
        exec,
        width,
        workspace: exec.workspace(),
    };
    let dp = c.as_mut_ptr();
    // SAFETY: the caller's contract; `C` is `D`, which is this driver's
    // in-place form, and the plan was built from the same layouts.
    unsafe { tensorcontract::execute_resolved(&plan, &rg, Some(&spmd), alpha, a, b, beta, dp, dp) };
    Ok(crate::SelectedGemm {
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
    })
}
