//! TBLIS-style batched GEMM through `tensorcontract`, the direct tensor
//! contraction engine of Lukas Devos's tensorprimitives-rs (after D. A.
//! Matthews, *High-Performance Tensor Contraction without Transposition*,
//! arXiv:1607.00291).
//!
//! One `Plan` is built per call for the item shape; items run either spread
//! over the pool (each serial, width-one seam) or one after another, each on
//! the pool's co-scheduled threads through the `Spmd` seam.
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
}

impl Spmd for ExecSpmd<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
        self.exec.broadcast(p, f).is_ok()
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
    s: &Batch,
    sched: &Schedule,
    alpha: T,
    ca: Conj,
    a: SendConst<T>,
    cb: Conj,
    b: SendConst<T>,
    beta: T,
    c: SendMut<T>,
) -> Result<Selected> {
    let (la, lb, ld) = (layout(&s.item.a)?, layout(&s.item.b)?, layout(&s.item.c)?);
    let (ia, ib, id) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    // C is D itself (in place); beta == 0 never reads it.
    let plan = Plan::new(
        op(Operand::new(&la, &ia), ca),
        op(Operand::new(&lb, &ib), cb),
        Some(Operand::new(&ld, &id)),
        Operand::new(&ld, &id),
    )?;
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
            };
            lanes.for_each_partition(s.count, &|i| item(i, &serial));
        }
        None => {
            // Each item is one co-scheduled SPMD broadcast of its own.
            let spmd = ExecSpmd {
                exec,
                width: sched.inner,
            };
            for i in 0..s.count {
                item(i, &spmd);
            }
        }
    }
    Ok(Selected::Tblis {
        outer_parallel: sched.outer.is_some(),
    })
}
