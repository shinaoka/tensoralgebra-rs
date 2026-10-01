//! TBLIS-style direct contraction through `tensorcontract`, the engine of
//! Lukas Devos's tensorprimitives-rs (after D. A. Matthews,
//! *High-Performance Tensor Contraction without Transposition*,
//! arXiv:1607.00291): general strides are packed straight into the
//! micro-kernel panels, so no operand is transposed in memory.
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::{Layout, Operand, Plan};
use tprims_blas::{Conj, Scalar};
use tprims_exec::{Exec, WidthPolicy};

use crate::util::select_err;
use crate::{DotGeneral, Shape};
use crate::{Error, Result};

/// A tensorcontract plan with its labels, built once.
///
/// It owns a workspace, so a serial operation on this plan reuses its worker
/// buffers and B panel across calls instead of allocating per call.
#[derive(Debug)]
pub(crate) struct TbPlan {
    plan: Plan,
    work: f64,
    workspace: tprims_exec::ArenaProvider,
}

impl TbPlan {
    /// The plan's cached resolution, which is what `selected_gemm` reports.
    pub(crate) fn resolved<T: tprims_blas::Scalar>(
        &self,
    ) -> std::result::Result<
        tprims_gemm_kernel::ResolvedGemm<<T as tprims_blas::Scalar>::Re>,
        tprims_gemm_kernel::SelectError,
    > {
        self.plan.resolved::<T>()
    }

    /// The dynamic assignment this plan resolved to, when its policy is
    /// `DynamicTiles`. A contraction plan is built before an executor is
    /// chosen, so the active width reported is the job-count cap.
    pub(crate) fn dynamic<R: tprims_gemm_kernel::Real>(
        &self,
        rg: &tprims_gemm_kernel::ResolvedGemm<R>,
    ) -> Option<tensorcontract::DynamicReport> {
        tensorcontract::dynamic_report(&self.plan, rg, usize::MAX)
    }
}

fn layout(dims: &[usize], strides: &[isize]) -> Result<Layout> {
    let e = |x: usize| i64::try_from(x).map_err(|_| Error::Shape("extent exceeds i64".into()));
    let ext = dims.iter().map(|&d| e(d)).collect::<Result<Vec<_>>>()?;
    let st = strides.iter().map(|&s| s as i64).collect();
    Layout::new(ext, st).map_err(Error::backend)
}

fn op(o: Operand<'_>, c: Conj) -> Operand<'_> {
    if c == Conj::Yes {
        o.conj()
    } else {
        o
    }
}

/// A caller's catalog and selector for one plan, with the host thread budget
/// the plan will run at. The selector is called at most once and not kept.
pub(crate) struct Custom<'a, T: Scalar> {
    pub catalog: &'a tprims_gemm_kernel::KernelCatalog<T>,
    pub chooser: &'a mut tensorcontract::Chooser<'a, T>,
    pub threads: usize,
    pub method: Option<tprims_gemm_kernel::Method>,
}

fn backend_or_select(e: tensorcontract::Error) -> Error {
    match e {
        tensorcontract::Error::KernelSelection(s) => select_err(s),
        other => Error::backend(other),
    }
}

pub(crate) fn plan<T: Scalar>(
    cfg: &DotGeneral,
    s: &Shape,
    dims: [&[usize]; 3],
    strides: [&[isize]; 3],
    conj: (Conj, Conj),
    gemm: &tprims_blas::GemmConfig,
    custom: Option<Custom<'_, T>>,
) -> Result<TbPlan> {
    let ra = dims[0].len();
    let la: Vec<i64> = (0..ra as i64).collect();
    let mut next = ra as i64;
    let mut lb = vec![0i64; dims[1].len()];
    for (&l, &r) in cfg
        .lhs_contract
        .iter()
        .zip(&cfg.rhs_contract)
        .chain(cfg.lhs_batch.iter().zip(&cfg.rhs_batch))
    {
        lb[r] = l as i64;
    }
    for &r in &s.rhs_free {
        lb[r] = next;
        next += 1;
    }
    let lc: Vec<i64> = s
        .lhs_free
        .iter()
        .map(|&x| la[x])
        .chain(s.rhs_free.iter().map(|&x| lb[x]))
        .chain(cfg.lhs_batch.iter().map(|&x| la[x]))
        .collect();
    let (ya, yb, yc) = (
        layout(dims[0], strides[0])?,
        layout(dims[1], strides[1])?,
        layout(dims[2], strides[2])?,
    );
    // C is D itself (in place); beta == 0 never reads it.
    let (oa, ob) = (
        op(Operand::new(&ya, &la), conj.0),
        op(Operand::new(&yb, &lb), conj.1),
    );
    let (oc, od) = (Operand::new(&yc, &lc), Operand::new(&yc, &lc));
    // Registering first lets a forced family id name any built provider.
    tprims_blas::register_kernels();
    let plan = Plan::new(oa, ob, Some(oc), od).map_err(Error::backend)?;
    let plan = if gemm.has_partition_request() {
        plan.with_partition(gemm.partition, gemm.partition_opts)
    } else {
        plan
    };
    let plan = match custom {
        None => plan
            .with_kernel(gemm.kernel.clone())
            .map_err(Error::backend)?,
        Some(custom) => {
            let mut plan = plan;
            match custom.method {
                Some(tprims_gemm_kernel::Method::OneM) => {
                    plan = plan.with_complex_method(tensorcontract::ComplexMethod::OneM);
                }
                Some(tprims_gemm_kernel::Method::ThreeM) => {
                    plan = plan.with_complex_method(tensorcontract::ComplexMethod::ThreeM);
                }
                _ => {}
            }
            // Last, so the selector sees the width and method this plan uses.
            let plan = plan
                .with_threads(custom.threads)
                .with_selector::<T, _>([oa, ob, oc, od], custom.catalog, custom.chooser)
                .map_err(backend_or_select)?;
            if let Some(method) = custom.method {
                let rg = plan.resolved::<T>().map_err(select_err)?;
                let family = rg.family().complex.map(|s| s.method);
                let ok = match method {
                    tprims_gemm_kernel::Method::Native => {
                        family.is_none_or(|m| m == tprims_gemm_kernel::Method::Native)
                    }
                    other => family == Some(other),
                };
                if !ok {
                    return Err(select_err(tprims_gemm_kernel::SelectError::Incompatible {
                        id: rg.family().id.into(),
                        reason: "family implements a different complex method",
                    }));
                }
            }
            plan
        }
    };
    let k: usize = cfg.lhs_contract.iter().map(|&x| dims[0][x]).product();
    let out: usize = dims[2].iter().product();
    Ok(TbPlan {
        plan,
        work: 2.0 * (out as f64) * (k as f64),
        workspace: tprims_exec::ArenaProvider::new(),
    })
}

/// Execute a validated, non-empty, alpha != 0 problem.
pub(crate) fn execute<T: Scalar>(
    p: &TbPlan,
    exec: &Exec<'_>,
    alpha: T,
    a: &StridedView<'_, T>,
    b: &StridedView<'_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    let flops = p.work * if T::IS_COMPLEX_SCALAR { 4.0 } else { 1.0 };
    let width = exec.width_for(
        flops * tprims_blas::GemmPolicy::default().ns_per_flop,
        &WidthPolicy::default(),
    );
    // A borrowed pool lends its own arena; a serial context uses the plan's,
    // which is why a serial plan's steady state allocates nothing either.
    let exec = exec.with_budget(width).unwrap_or(*exec);
    let workspace: &dyn tprims_exec::WorkspaceProvider = exec.workspace().unwrap_or(&p.workspace);
    let d = c.as_mut_ptr();
    // SAFETY: the plan was built from these views' validated layouts (checked
    // again in `ContractPlan::execute`); all three are non-empty so their
    // origins are in bounds; C is exclusive and injective; `c == d` is
    // tensorcontract's in-place form.
    unsafe {
        p.plan
            .run_raw_with(&exec, Some(workspace), alpha, a.ptr(), b.ptr(), beta, d, d)
    };
    Ok(())
}
