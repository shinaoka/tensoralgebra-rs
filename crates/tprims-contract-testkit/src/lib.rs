//! A **test-only** second implementation of the contraction interface.
//!
//! [`NaiveBackend`] is a plain loop nest over the output elements. It exists
//! to show that `tprims-contract-traits` can be implemented by a crate that
//! depends on nothing but the interface and neutral building blocks (strided
//! views, `num-complex`), and to serve as an independent backend in
//! consumer-selection tests. It is not a production fallback and is never
//! selected by default.
//!
//! It copies nothing, so it never materializes; it splits the output range
//! over the host's barrier-free partitions.

use std::ops::{Add, Mul};

use num_complex::Complex;
use strided_view::{StridedView, StridedViewMut};
use tprims_contract_traits::{
    BoxedPlan, Conj, ContractionBackend, Diagnostics, HostExecution, PlanningBudget,
    PreparedContraction, Problem, Requirements, Result, Validated,
};

/// The arithmetic the naive loop needs.
pub trait Field: tprims_contract_traits::Scalar + Add<Output = Self> + Mul<Output = Self> {
    /// Additive identity.
    const ZERO: Self;
    /// Complex conjugate (identity for real types).
    fn conj(self) -> Self;
}

macro_rules! real {
    ($($t:ty),*) => {$(
        impl Field for $t {
            const ZERO: Self = 0.0;
            fn conj(self) -> Self { self }
        }
    )*};
}
real!(f32, f64);

macro_rules! complex {
    ($($t:ty),*) => {$(
        impl Field for Complex<$t> {
            const ZERO: Self = Complex::new(0.0, 0.0);
            fn conj(self) -> Self { Complex::conj(&self) }
        }
    )*};
}
complex!(f32, f64);

/// The naive loop-nest backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct NaiveBackend;

const ID: &str = "naive-loop-nest";

struct NaivePlan {
    problem: Problem,
    valid: Validated,
}

impl<T: Field> ContractionBackend<T> for NaiveBackend {
    fn id(&self) -> &'static str {
        ID
    }

    fn prepare(
        &self,
        problem: &Problem,
        _requirements: &Requirements,
        _budget: &PlanningBudget,
    ) -> Result<BoxedPlan<T>> {
        // Never copies, so `no_materialize` is always met.
        let valid = problem.validate::<T>()?;
        Ok(Box::new(NaiveBoxed {
            plan: NaivePlan {
                problem: problem.clone(),
                valid,
            },
            diag: Diagnostics::new(ID, "loop-nest"),
        }))
    }
}

struct NaiveBoxed {
    plan: NaivePlan,
    diag: Diagnostics,
}

#[derive(Clone, Copy)]
struct Raw<T>(*mut T);
// SAFETY: dereferenced only at validated offsets; writes go to disjoint,
// injective output positions (one per output element, one lane per range).
unsafe impl<T> Send for Raw<T> {}
unsafe impl<T> Sync for Raw<T> {}

impl<T> Raw<T> {
    // A method, so closures capture the wrapper and not the bare pointer field.
    fn get(self) -> *mut T {
        self.0
    }
}

impl<T: Field> PreparedContraction<T> for NaiveBoxed {
    fn execute_into_accum(
        &self,
        host: &dyn HostExecution,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        c: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        let NaivePlan { problem, valid } = &self.plan;
        problem.check_views(
            (a.dims(), a.strides()),
            (b.dims(), b.strides()),
            (c.dims(), c.strides()),
        )?;
        let out_dims = &valid.shape.out_dims;
        let n_out: usize = out_dims.iter().product();
        if n_out == 0 {
            return Ok(());
        }
        let skip_ab = valid.k_empty || alpha == T::ZERO;
        let (nlf, nrf) = (valid.shape.lhs_free.len(), valid.shape.rhs_free.len());
        // Per output axis: stride into A and into B (zero when the axis is not theirs).
        let mut sa = Vec::new();
        let mut sb = Vec::new();
        for &x in &valid.shape.lhs_free {
            sa.push(problem.a.strides[x]);
            sb.push(0isize);
        }
        for &x in &valid.shape.rhs_free {
            sa.push(0);
            sb.push(problem.b.strides[x]);
        }
        for (&la, &lb) in problem.dot.lhs_batch.iter().zip(&problem.dot.rhs_batch) {
            sa.push(problem.a.strides[la]);
            sb.push(problem.b.strides[lb]);
        }
        debug_assert_eq!(sa.len(), nlf + nrf + problem.dot.lhs_batch.len());
        let kdims: Vec<usize> = problem
            .dot
            .lhs_contract
            .iter()
            .map(|&x| problem.a.dims[x])
            .collect();
        let ka: Vec<isize> = problem
            .dot
            .lhs_contract
            .iter()
            .map(|&x| problem.a.strides[x])
            .collect();
        let kb: Vec<isize> = problem
            .dot
            .rhs_contract
            .iter()
            .map(|&x| problem.b.strides[x])
            .collect();
        let cs = problem.c.strides.clone();
        let (ap, bp, cp) = (
            Raw(a.ptr() as *mut T),
            Raw(b.ptr() as *mut T),
            Raw(c.as_mut_ptr()),
        );
        let (ca, cb) = problem.conj;
        let lanes = host.budget().min(n_out).max(1);
        let body = |lane: usize| {
            let (lo, hi) = (lane * n_out / lanes, (lane + 1) * n_out / lanes);
            for t in lo..hi {
                // Unravel t over the output extents (first axis fastest).
                let (mut rem, mut oa, mut ob, mut oc) = (t, 0isize, 0isize, 0isize);
                for (k, &d) in out_dims.iter().enumerate() {
                    let i = (rem % d) as isize;
                    rem /= d;
                    oa += i * sa[k];
                    ob += i * sb[k];
                    oc += i * cs[k];
                }
                let mut acc = T::ZERO;
                if !skip_ab {
                    let mut kidx = vec![0usize; kdims.len()];
                    'k: loop {
                        let (mut pa, mut pb) = (oa, ob);
                        for (q, &i) in kidx.iter().enumerate() {
                            pa += i as isize * ka[q];
                            pb += i as isize * kb[q];
                        }
                        // SAFETY: offsets are inside the bounds-checked views.
                        let (x, y) = unsafe { (*ap.get().offset(pa), *bp.get().offset(pb)) };
                        let x = if ca == Conj::Yes { x.conj() } else { x };
                        let y = if cb == Conj::Yes { y.conj() } else { y };
                        acc = acc + x * y;
                        let mut q = 0;
                        loop {
                            if q == kidx.len() {
                                break 'k;
                            }
                            kidx[q] += 1;
                            if kidx[q] < kdims[q] {
                                break;
                            }
                            kidx[q] = 0;
                            q += 1;
                        }
                    }
                }
                // SAFETY: `oc` is a distinct in-bounds output position; C is
                // exclusively borrowed and injective.
                unsafe {
                    let p = cp.get().offset(oc);
                    *p = if beta == T::ZERO {
                        alpha * acc
                    } else {
                        alpha * acc + beta * *p
                    };
                }
            }
        };
        host.for_each_partition(lanes, &body);
        Ok(())
    }

    fn diagnostics(&self) -> &Diagnostics {
        &self.diag
    }
}

// Plans are shared across threads: the plan holds only owned metadata.
const _: fn() = || {
    fn assert<X: Send + Sync>() {}
    assert::<NaiveBoxed>();
};
