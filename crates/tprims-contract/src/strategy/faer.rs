//! The faer strategy: copy-free fusion to one strided batched GEMM.
//!
//! The index groups M (A, D), N (B, D), K (A, B) and H (A, B, D) are each put
//! in one order and fused per operand. When every group fuses in every operand
//! that carries it, the contraction is one strided batched GEMM over the
//! caller's memory and faer computes it. Anything else is declined and runs on
//! the packed driver: this strategy never copies an operand.
//!
//! The approach follows tenferro-rs's CPU `dot_general`
//! (`tenferro-cpu/src/{dot_runtime.rs, gemm/mod.rs}` at `5a4e7fd`),
//! reimplemented on the validated [`Problem`] roles. The batched loop, with
//! its outer or inner parallel schedule, moved here from `tprims-blas`.
//!
//! # Semantics
//!
//! `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(D_old))` with a C mode of
//! overwrite or in-place accumulation. A separately described C is declined (it
//! would need a copy into D first). faer adds into D with coefficient one, so
//! the update is reduced to that form without a copy:
//!
//! * an output conjugation distributes over the sum, flipping the conjugation
//!   of A, B and C and conjugating `alpha` and `beta`;
//! * `beta * op_C(D_old)` is applied to D in one in-place pass when it is not
//!   the identity, then faer accumulates the product.

use faer::Accum;
use tprims_exec::{Exec, Par, WidthPolicy};
use tprims_kernel::Element;

use crate::api::{CSpec, OperandId, Problem, RoleAxis, Scalar};
use crate::plan::NS_PER_FLOP;

/// One fused group: `(extent, stride)` in each of A, B and D.
#[derive(Clone, Copy, Debug)]
struct Fused {
    extent: usize,
    a: isize,
    b: isize,
    d: isize,
}

/// The fused batched GEMM: `[M, K]`, `[K, N]`, `[M, N]` matrices over `count`
/// items.
#[derive(Clone, Debug)]
pub(crate) struct FaerPlan {
    m: Fused,
    n: Fused,
    k: Fused,
    h: Fused,
    conj_a: bool,
    conj_b: bool,
    conj_c: bool,
    conj_d: bool,
}

/// Whether `order` fuses the group in operand `o`: the strides of the
/// non-unit axes must chain (`s_next == s_prev * extent_prev`). Returns the
/// fused `(extent, stride)`.
fn fuse(group: &[RoleAxis], order: &[usize], o: OperandId) -> Option<(usize, isize)> {
    let mut ext = 1usize;
    let mut first: Option<isize> = None;
    let mut expect = 0isize;
    for &e in order {
        let d = group[e].extent();
        if d == 1 {
            continue;
        }
        let s = group[e].stride(o);
        match first {
            None => first = Some(s),
            Some(_) if s != expect => return None,
            Some(_) => {}
        }
        expect = s.checked_mul(isize::try_from(d).ok()?)?;
        ext = ext.checked_mul(d)?;
    }
    Some((ext, first.unwrap_or(1)))
}

fn sorted_by(group: &[RoleAxis], o: OperandId) -> Vec<usize> {
    let mut order: Vec<usize> = (0..group.len()).collect();
    order.sort_by_key(|&e| group[e].stride(o).unsigned_abs());
    order
}

/// Fuse one group over the operands that carry it, trying the order each
/// carrier would choose and then the natural one.
fn fuse_group(group: &[RoleAxis], carriers: &[OperandId]) -> Option<Fused> {
    let mut cands: Vec<Vec<usize>> = carriers.iter().map(|&o| sorted_by(group, o)).collect();
    cands.push((0..group.len()).collect());
    let order = cands
        .into_iter()
        .find(|ord| carriers.iter().all(|&o| fuse(group, ord, o).is_some()))?;
    let get = |o: OperandId| fuse(group, &order, o);
    let (extent, a) = get(OperandId::A).unwrap_or((1, 1));
    // Every carrier agrees on the extent; the absent operands keep stride one.
    Some(Fused {
        extent,
        a,
        b: get(OperandId::B).map_or(1, |x| x.1),
        d: get(OperandId::D).map_or(1, |x| x.1),
    })
}

/// The fusion of `p`, or `None` when this strategy cannot run it copy-free with
/// full semantics.
pub(crate) fn plan(p: &Problem) -> Option<FaerPlan> {
    // A separate C would have to be copied into D first.
    if matches!(p.c_spec(), CSpec::Separate(_)) {
        return None;
    }
    let r = p.roles();
    // A reduction over an axis only one input carries has no matrix to hand
    // to faer without a broadcast copy.
    if r.k().iter().any(|x| !(x.in_a() && x.in_b())) {
        return None;
    }
    use OperandId::{A, B, D};
    Some(FaerPlan {
        m: fuse_group(r.m(), &[A, D])?,
        n: fuse_group(r.n(), &[B, D])?,
        k: fuse_group(r.k(), &[A, B])?,
        h: fuse_group(r.h(), &[A, B, D])?,
        conj_a: p.a().op().is_conj(),
        conj_b: p.b().op().is_conj(),
        conj_c: p.op_c().is_conj(),
        conj_d: p.d().op().is_conj(),
    })
}

impl FaerPlan {
    fn flops<T: Scalar>(&self, items: usize) -> f64 {
        2.0 * self.m.extent as f64
            * self.n.extent as f64
            * self.k.extent as f64
            * items as f64
            * if T::IS_COMPLEX { 4.0 } else { 1.0 }
    }

    /// Run the batched GEMM.
    ///
    /// # Safety
    ///
    /// `a`, `b` and `d` are the elements at logical index zero of non-empty
    /// layouts matching the plan's problem (K non-empty); `d` is exclusive and
    /// injective and does not alias `a` or `b`. `beta` is zero when the problem
    /// has no C term.
    pub(crate) unsafe fn run<T: Scalar>(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        d: *mut T,
    ) {
        let (mut ca, mut cb, mut cc) = (self.conj_a, self.conj_b, self.conj_c);
        let (mut alpha, mut beta) = (alpha, beta);
        if self.conj_d {
            // conj(alpha*A*B + beta*C) = conj(alpha)*conj(A)*conj(B) + conj(beta)*conj(C)
            (ca, cb, cc) = (!ca, !cb, !cc);
            (alpha, beta) = (Element::conj(alpha), Element::conj(beta));
        }
        let count = self.h.extent;
        let policy = WidthPolicy::default();
        let width = |items: usize| exec.width_for(self.flops::<T>(items) * NS_PER_FLOP, &policy);
        let inner = width(1);
        let total = width(count);
        let (ap, bp, dp) = (SendConst(a), SendConst(b), SendMut(d));
        let item = |i: usize, par: faer::Par| {
            let i = i as isize;
            // SAFETY: offsets `i * stride` stay inside the validated layouts
            // (i < count); output items are disjoint (D is injective).
            unsafe {
                self.item(
                    alpha,
                    ap.get().offset(i * self.h.a),
                    ca,
                    bp.get().offset(i * self.h.b),
                    cb,
                    beta,
                    cc,
                    dp.get().offset(i * self.h.d),
                    par,
                )
            }
        };
        if inner == 1 && total > 1 && count > 1 {
            let lanes = exec.with_budget(total.min(count)).unwrap_or(*exec);
            lanes.for_each_partition(count, &|i| item(i, faer::Par::Seq));
        } else {
            // One pool entry for the whole batch.
            exec.install(inner, |par: Par| {
                for i in 0..count {
                    item(i, to_faer(par));
                }
            });
        }
    }

    /// One GEMM of the batch.
    ///
    /// # Safety
    ///
    /// As [`FaerPlan::run`], for one item's origins.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set.
    unsafe fn item<T: Scalar>(
        &self,
        alpha: T,
        a: *const T,
        ca: bool,
        b: *const T,
        cb: bool,
        beta: T,
        cc: bool,
        d: *mut T,
        par: faer::Par,
    ) {
        let (m, n, k) = (self.m.extent, self.n.extent, self.k.extent);
        let zero = <T as Element>::zero();
        let accum = if beta == zero {
            Accum::Replace
        } else {
            if beta != <T as Element>::one() || cc {
                // SAFETY: forwarded.
                unsafe { scale(d, self.m, self.n, beta, cc) };
            }
            Accum::Add
        };
        // SAFETY: the caller's contract; views are read-only / exclusive as faer requires.
        unsafe {
            T::matmul(
                accum == Accum::Add,
                (m, n, k),
                d,
                (self.m.d, self.n.d),
                a,
                (self.m.a, self.k.a),
                ca,
                b,
                (self.k.b, self.n.b),
                cb,
                alpha,
                par,
            )
        };
    }
}

/// `D = beta * op(D)` over one item's `m x n` matrix, in place.
///
/// # Safety
///
/// `d` with the fused strides addresses writable elements, exclusively
/// borrowed by the caller.
unsafe fn scale<T: Scalar>(d: *mut T, m: Fused, n: Fused, beta: T, conj: bool) {
    for j in 0..n.extent {
        for i in 0..m.extent {
            // SAFETY: i < m, j < n; offsets validated by the problem.
            unsafe {
                let p = d.offset(i as isize * m.d + j as isize * n.d);
                let old = if conj { Element::conj(*p) } else { *p };
                *p = Element::mul(beta, old);
            }
        }
    }
}

fn to_faer(par: Par) -> faer::Par {
    match par {
        Par::Seq => faer::Par::Seq,
        Par::Threads(n) => faer::Par::rayon(n.get()),
    }
}

/// Raw pointers moved into a pool closure.
#[derive(Clone, Copy)]
struct SendConst<T>(*const T);
impl<T> SendConst<T> {
    fn get(self) -> *const T {
        self.0
    }
}
// SAFETY: only dereferenced under the validated contracts documented at use.
unsafe impl<T> core::marker::Send for SendConst<T> {}
unsafe impl<T> Sync for SendConst<T> {}

#[derive(Clone, Copy)]
struct SendMut<T>(*mut T);
impl<T> SendMut<T> {
    fn get(self) -> *mut T {
        self.0
    }
}
// SAFETY: as `Send`; writes are to disjoint, injective regions.
unsafe impl<T> core::marker::Send for SendMut<T> {}
unsafe impl<T> Sync for SendMut<T> {}
