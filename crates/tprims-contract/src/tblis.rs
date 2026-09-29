//! TBLIS-style direct contraction through `tensorcontract`, the engine of
//! Lukas Devos's tensorprimitives-rs (after D. A. Matthews,
//! *High-Performance Tensor Contraction without Transposition*,
//! arXiv:1607.00291): general strides are packed straight into the
//! micro-kernel panels, so no operand is transposed in memory.
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::spmd::Spmd;
use tensorcontract::{Layout, Operand, Plan};
use tprims_blas::{Conj, Scalar};
use tprims_exec::{Exec, WidthPolicy};

use crate::config::{DotGeneral, Shape};
use crate::{Error, Result};

struct ExecSpmd<'a> {
    exec: &'a Exec<'a>,
    width: usize,
}

impl Spmd for ExecSpmd<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
        self.exec.broadcast(p, f).is_ok()
    }
}

/// A tensorcontract plan with its labels, built once.
#[derive(Debug)]
pub(crate) struct TbPlan {
    plan: Plan,
    work: f64,
}

fn layout(dims: &[usize], strides: &[isize]) -> Result<Layout> {
    let e = |x: usize| i64::try_from(x).map_err(|_| Error::Shape("extent exceeds i64".into()));
    let ext = dims.iter().map(|&d| e(d)).collect::<Result<Vec<_>>>()?;
    let st = strides.iter().map(|&s| s as i64).collect();
    Layout::new(ext, st).map_err(|e| Error::Backend(e.to_string()))
}

fn op(o: Operand<'_>, c: Conj) -> Operand<'_> {
    if c == Conj::Yes {
        o.conj()
    } else {
        o
    }
}

pub(crate) fn plan(
    cfg: &DotGeneral,
    s: &Shape,
    dims: [&[usize]; 3],
    strides: [&[isize]; 3],
    conj: (Conj, Conj),
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
    let plan = Plan::new(
        op(Operand::new(&ya, &la), conj.0),
        op(Operand::new(&yb, &lb), conj.1),
        Some(Operand::new(&yc, &lc)),
        Operand::new(&yc, &lc),
    )
    .map_err(|e| Error::Backend(e.to_string()))?;
    let k: usize = cfg.lhs_contract.iter().map(|&x| dims[0][x]).product();
    let out: usize = dims[2].iter().product();
    Ok(TbPlan {
        plan,
        work: 2.0 * (out as f64) * (k as f64),
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
    let spmd = ExecSpmd { exec, width };
    let d = c.as_mut_ptr();
    // SAFETY: the plan was built from these views' validated layouts (checked
    // again in `ContractPlan::execute`); all three are non-empty so their
    // origins are in bounds; C is exclusive and injective; `c == d` is
    // tensorcontract's in-place form.
    unsafe {
        p.plan
            .run_raw_with(&spmd, alpha, a.ptr(), b.ptr(), beta, d, d)
    };
    Ok(())
}
