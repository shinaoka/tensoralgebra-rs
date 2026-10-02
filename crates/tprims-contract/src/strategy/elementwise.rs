//! The elementwise strategy: an all-batch (Hadamard) product, and the
//! output-update helper every strategy shares for `alpha == 0` and an empty
//! contraction.
//!
//! Both are one strided pass over the output's axes with the full update
//! semantics
//!
//! ```text
//! D = op_D( alpha * op_A(A) * op_B(B) + beta * op_C(C) )
//! ```
//!
//! where `C` is read from the output itself (in-place accumulation), from a
//! separately described source, or not at all (`beta == 0` reads no previous
//! value). Axes of extent one are dropped, the rest are walked with `D`'s
//! smallest-|stride| axis innermost, and the outermost axis is split into
//! barrier-free lanes on the executor.

use tprims_exec::{Exec, WidthPolicy};
use tprims_kernel::Element;

use crate::api::{OperandId, Problem, RoleAxis, Scalar};
use crate::plan::NS_PER_FLOP;

/// Where the `beta * op_C(C)` term is read from.
#[derive(Clone, Copy, Debug)]
pub(crate) enum CRead<T> {
    /// No term: nothing previous is read and `beta` is treated as zero.
    None,
    /// The output element itself.
    InPlace,
    /// A separate source with the problem's C strides; the pointer is C's
    /// element at logical index zero.
    Separate(*const T),
}

/// The input term of one update.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Inputs<T> {
    /// No input term (an output-only pass).
    None,
    /// `alpha * op_A(A) * op_B(B)`, with the two origins.
    Product(*const T, *const T),
    /// `alpha * op_A(A)`, with the origin (the axpy-like update of `add`).
    Scaled(*const T),
}

// SAFETY: the pointers are dereferenced only under the contracts of
// `ElementPlan::run`, which every caller upholds.
unsafe impl<T> Send for Inputs<T> {}
unsafe impl<T> Sync for Inputs<T> {}
unsafe impl<T> Send for CRead<T> {}
unsafe impl<T> Sync for CRead<T> {}

/// The scalars and conjugations of one update.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Expr<T> {
    pub alpha: T,
    pub beta: T,
    pub conj_a: bool,
    pub conj_b: bool,
    pub conj_c: bool,
    pub conj_d: bool,
}

#[derive(Clone, Copy)]
struct Ptr<T>(*mut T);
// SAFETY: dereferenced only at validated, disjoint (for writes) offsets.
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

impl<T> Ptr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// The axes of a pass: extent and the `[A, B, C, D]` element strides, extent-1
/// axes dropped, `D`'s smallest stride first.
#[derive(Clone, Debug)]
pub(crate) struct ElementPlan {
    axes: Vec<(usize, [isize; 4])>,
    total: usize,
}

impl ElementPlan {
    fn from_roles<'a>(roles: impl Iterator<Item = &'a RoleAxis>, with_inputs: bool) -> Self {
        let mut axes: Vec<(usize, [isize; 4])> = roles
            .filter(|r| r.extent() != 1)
            .map(|r| {
                let s = |o| r.stride(o);
                let (sa, sb) = if with_inputs {
                    (s(OperandId::A), s(OperandId::B))
                } else {
                    (0, 0)
                };
                (r.extent(), [sa, sb, s(OperandId::C), s(OperandId::D)])
            })
            .collect();
        axes.sort_by_key(|a| a.1[3].unsigned_abs());
        let total = axes.iter().map(|a| a.0).product();
        Self { axes, total }
    }

    /// The pass of an all-batch problem: every axis is a batch axis.
    pub(crate) fn product(p: &Problem) -> Self {
        Self::from_roles(p.roles().h().iter(), true)
    }

    /// The pass of an elementwise `D = alpha * A + beta * D` over equal extents
    /// with `A`'s and `D`'s strides.
    pub(crate) fn scaled(dims: &[usize], a: &[isize], d: &[isize]) -> Self {
        let mut axes: Vec<(usize, [isize; 4])> = dims
            .iter()
            .zip(a.iter().zip(d))
            .filter(|(&e, _)| e != 1)
            .map(|(&e, (&sa, &sd))| (e, [sa, 0, sd, sd]))
            .collect();
        axes.sort_by_key(|x| x.1[3].unsigned_abs());
        let total = axes.iter().map(|x| x.0).product();
        Self { axes, total }
    }

    /// The pass over the output's elements (`M`, `N` and batch axes), without
    /// inputs.
    pub(crate) fn output(p: &Problem) -> Self {
        let r = p.roles();
        Self::from_roles(r.m().iter().chain(r.n()).chain(r.h()), false)
    }

    /// Run the update.
    ///
    /// `inputs` carries the input origins when a product term is computed; an
    /// output-only pass passes [`Inputs::None`] and the term is zero.
    ///
    /// # Safety
    ///
    /// `d` and every origin are the elements at logical index zero of
    /// non-empty, bounds-checked layouts matching the plan's problem; `d` is
    /// exclusive and injective; inputs do not alias `d`; a separate `C` does not
    /// overlap `d` unless it is the same mapping at the same origin.
    pub(crate) unsafe fn run<T: Scalar>(
        &self,
        exec: &Exec<'_>,
        e: Expr<T>,
        inputs: Inputs<T>,
        c: CRead<T>,
        d: *mut T,
    ) {
        if self.total == 0 {
            return;
        }
        let zero = <T as Element>::zero();
        let read_c = !matches!(c, CRead::None) && e.beta != zero;
        let (outer, inner) = match self.axes.split_last() {
            Some((o, rest)) => (*o, rest),
            None => ((1, [0; 4]), &[][..]),
        };
        let p = WidthPolicy::default();
        let lanes = exec
            .width_for(self.total as f64 * 2.0 * NS_PER_FLOP, &p)
            .min(outer.0)
            .max(1);
        let dp = Ptr(d);
        let (ap, bp) = match inputs {
            Inputs::Product(a, b) => (Ptr(a as *mut T), Ptr(b as *mut T)),
            Inputs::Scaled(a) => (Ptr(a as *mut T), Ptr(core::ptr::null_mut())),
            Inputs::None => (Ptr(core::ptr::null_mut()), Ptr(core::ptr::null_mut())),
        };
        let cp = match c {
            CRead::Separate(c) => Ptr(c as *mut T),
            _ => Ptr(core::ptr::null_mut()),
        };
        let (has_a, has_b) = match inputs {
            Inputs::Product(..) => (true, true),
            Inputs::Scaled(_) => (true, false),
            Inputs::None => (false, false),
        };
        let in_place = matches!(c, CRead::InPlace);
        let op = |x: T, conj: bool| if conj { Element::conj(x) } else { x };
        let body = |lane: usize| {
            let mut idx = vec![0usize; inner.len()];
            for t in lane * outer.0 / lanes..(lane + 1) * outer.0 / lanes {
                idx.iter_mut().for_each(|i| *i = 0);
                let mut off: [isize; 4] = core::array::from_fn(|j| t as isize * outer.1[j]);
                loop {
                    // SAFETY: (idx, t) lies inside the validated extents, so every
                    // offset is inside its layout; lanes own disjoint t ranges and
                    // D is injective.
                    unsafe {
                        let dq = dp.get().offset(off[3]);
                        let mut val = if has_a {
                            let x = op(*ap.get().offset(off[0]), e.conj_a);
                            let prod = if has_b {
                                Element::mul(x, op(*bp.get().offset(off[1]), e.conj_b))
                            } else {
                                x
                            };
                            Element::mul(e.alpha, prod)
                        } else {
                            zero
                        };
                        if read_c {
                            let old = if in_place {
                                *dq
                            } else {
                                *cp.get().offset(off[2])
                            };
                            val = Element::add(val, Element::mul(e.beta, op(old, e.conj_c)));
                        }
                        *dq = op(val, e.conj_d);
                    }
                    let mut k = 0;
                    while k < inner.len() {
                        idx[k] += 1;
                        for j in 0..4 {
                            off[j] += inner[k].1[j];
                        }
                        if idx[k] < inner[k].0 {
                            break;
                        }
                        let back = inner[k].0 as isize;
                        for j in 0..4 {
                            off[j] -= inner[k].1[j] * back;
                        }
                        idx[k] = 0;
                        k += 1;
                    }
                    if k == inner.len() {
                        break;
                    }
                }
            }
        };
        exec.for_each_partition(lanes, &body);
    }
}
