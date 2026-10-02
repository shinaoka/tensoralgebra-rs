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
//! value).
//!
//! The traversal belongs to strided-rs (`strided-basic`): its blocked,
//! contiguous-inner-loop kernels vectorize and split across the executor's
//! pool through [`tprims_exec::strided::run_with_exec`]. This module only
//! specializes the update *outside* the per-element loop:
//!
//! In-place accumulation reads the previous `D` through the destination with
//! strided-basic's `map_update_into`, `zip_update2_into` and `zip_update3_into`
//! (no second view over `D`); `axpy` and `fma` cover the unit-scalar forms.
//!
//! 1. `op_D` is folded into the other terms, since
//!    `conj(alpha*a*b + beta*c) = conj(alpha)*conj(a)*conj(b) + conj(beta)*conj(c)`;
//!    what remains is `conj_a'`, `conj_b'`, `conj_c'` and transformed scalars.
//! 2. One dispatch on those three flags (type-level `Identity`/`Conj` views;
//!    all `false` for a real element type) and on the input and C modes picks
//!    a monomorphized closure such as `|a, b| alpha * a * b`. No flag is
//!    tested per element.

use strided_basic::{
    axpy, copy_scale, fma, map_into, map_update_into, mul_into, zip_map2_into, zip_map3_into,
    zip_update2_into, zip_update3_into, Conj, ElementOp, Identity, StridedError,
};
use strided_view::{StridedView, StridedViewMut};
use tprims_exec::strided::run_with_exec;
use tprims_exec::Exec;
use tprims_kernel::Element;

use crate::api::{Error, OperandId, Problem, Result, RoleAxis, Scalar};

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
// SAFETY: dereferenced only through strided views over validated layouts;
// strided-rs splits writes into disjoint ranges of an injective output.
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

impl<T> Ptr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// Where the previous-value term is read from, after `beta == 0` and a missing
/// `C` have been resolved to [`Mode::Overwrite`].
#[derive(Clone, Copy)]
enum Mode<T> {
    /// No term.
    Overwrite,
    /// The output itself (read through an alias of the destination).
    InPlace,
    /// A separate source.
    Separate(Ptr<T>),
}

/// The lowest and highest element offset a layout addresses, from its origin.
fn offset_span(dims: &[usize], strides: &[isize]) -> (isize, isize) {
    let (mut lo, mut hi) = (0isize, 0isize);
    for (&n, &s) in dims.iter().zip(strides) {
        let reach = (n as isize - 1) * s;
        if reach < 0 {
            lo += reach;
        } else {
            hi += reach;
        }
    }
    (lo, hi)
}

/// A read view over a validated layout whose logical origin is `origin`.
///
/// # Safety
///
/// Every offset the layout generates from `origin` is a live element, nothing
/// writes it during `'a` except through an identical in-place mapping.
unsafe fn read_view<'a, T, Op>(
    origin: *const T,
    dims: &[usize],
    strides: &[isize],
) -> StridedView<'a, T, Op> {
    let (lo, hi) = offset_span(dims, strides);
    // SAFETY: the caller's contract; `[lo, hi]` is the layout's reach.
    unsafe {
        let data = core::slice::from_raw_parts(origin.offset(lo), (hi - lo) as usize + 1);
        StridedView::new_unchecked(data, dims, strides, -lo)
    }
}

/// The write view of the destination.
///
/// # Safety
///
/// As [`read_view`], and the layout is injective and exclusively owned.
unsafe fn write_view<'a, T>(
    origin: *mut T,
    dims: &[usize],
    strides: &[isize],
) -> StridedViewMut<'a, T> {
    let (lo, hi) = offset_span(dims, strides);
    // SAFETY: the caller's contract.
    unsafe {
        let data = core::slice::from_raw_parts_mut(origin.offset(lo), (hi - lo) as usize + 1);
        StridedViewMut::new_unchecked(data, dims, strides, -lo)
    }
}

/// The axes of a pass: extents and the `[A, B, C, D]` element strides of each
/// axis, extent-1 axes dropped.
#[derive(Clone, Debug)]
pub(crate) struct ElementPlan {
    dims: Vec<usize>,
    strides: [Vec<isize>; 4],
    total: usize,
}

impl ElementPlan {
    fn from_axes(axes: impl Iterator<Item = (usize, [isize; 4])>) -> Self {
        let axes: Vec<_> = axes.filter(|a| a.0 != 1).collect();
        Self {
            dims: axes.iter().map(|a| a.0).collect(),
            strides: core::array::from_fn(|j| axes.iter().map(|a| a.1[j]).collect()),
            total: axes.iter().map(|a| a.0).product(),
        }
    }

    fn from_roles<'a>(roles: impl Iterator<Item = &'a RoleAxis>, with_inputs: bool) -> Self {
        Self::from_axes(roles.map(|r| {
            let s = |o| r.stride(o);
            let (sa, sb) = if with_inputs {
                (s(OperandId::A), s(OperandId::B))
            } else {
                (0, 0)
            };
            (r.extent(), [sa, sb, s(OperandId::C), s(OperandId::D)])
        }))
    }

    /// The pass of an all-batch problem: every axis is a batch axis.
    pub(crate) fn product(p: &Problem) -> Self {
        Self::from_roles(p.roles().h().iter(), true)
    }

    /// The pass of an elementwise `D = alpha * A + beta * D` over equal extents
    /// with `A`'s and `D`'s strides.
    pub(crate) fn scaled(dims: &[usize], a: &[isize], d: &[isize]) -> Self {
        Self::from_axes(
            dims.iter()
                .zip(a.iter().zip(d))
                .map(|(&e, (&sa, &sd))| (e, [sa, 0, sd, sd])),
        )
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
    /// # Errors
    ///
    /// [`Error::Backend`] with strided-rs's error; the plan's layouts were
    /// validated, so this is not expected.
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
    ) -> Result<()> {
        if self.total == 0 {
            return Ok(());
        }
        let zero = <T as Element>::zero();
        let mode = match c {
            CRead::None => Mode::Overwrite,
            _ if e.beta == zero => Mode::Overwrite,
            CRead::InPlace => Mode::InPlace,
            // The same mapping at the same origin is the in-place update.
            CRead::Separate(p) if core::ptr::eq(p, d) => Mode::InPlace,
            CRead::Separate(p) => Mode::Separate(Ptr(p as *mut T)),
        };
        // op_D folded into the terms: conj(alpha*a*b + beta*c) conjugates
        // every factor. A real element type has no conjugation at all.
        let flip = e.conj_d;
        let (mut alpha, mut beta) = (e.alpha, e.beta);
        let (mut ca, mut cb, mut cc) = (e.conj_a, e.conj_b, e.conj_c);
        if flip {
            alpha = Element::conj(alpha);
            beta = Element::conj(beta);
            ca = !ca;
            cb = !cb;
            cc = !cc;
        }
        if !<T as Element>::IS_COMPLEX {
            (ca, cb, cc) = (false, false, false);
        }
        let ptrs = match inputs {
            Inputs::Product(a, b) => [Ptr(a as *mut T), Ptr(b as *mut T)],
            Inputs::Scaled(a) => [Ptr(a as *mut T), Ptr(core::ptr::null_mut())],
            Inputs::None => [Ptr(core::ptr::null_mut()); 2],
        };
        let kind = match inputs {
            Inputs::Product(..) => Kind::Product,
            Inputs::Scaled(_) => Kind::Scaled,
            Inputs::None => Kind::None,
        };
        let job = Job {
            plan: self,
            alpha,
            beta,
            kind,
            ptrs,
            mode,
            d: Ptr(d),
        };
        macro_rules! dispatch {
            ($($ca:literal $cb:literal $cc:literal => $oa:ty, $ob:ty, $oc:ty;)*) => {
                match (ca, cb, cc) {
                    $(($ca, $cb, $cc) => job.go::<$oa, $ob, $oc>(exec),)*
                }
            };
        }
        dispatch! {
            false false false => Identity, Identity, Identity;
            false false true => Identity, Identity, Conj;
            false true false => Identity, Conj, Identity;
            false true true => Identity, Conj, Conj;
            true false false => Conj, Identity, Identity;
            true false true => Conj, Identity, Conj;
            true true false => Conj, Conj, Identity;
            true true true => Conj, Conj, Conj;
        }
        .map_err(Error::backend)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    /// `alpha * a * b`.
    Product,
    /// `alpha * a`.
    Scaled,
    /// No input term.
    None,
}

/// One update after folding: scalars, input kind, origins and C mode.
struct Job<'p, T> {
    plan: &'p ElementPlan,
    alpha: T,
    beta: T,
    kind: Kind,
    ptrs: [Ptr<T>; 2],
    mode: Mode<T>,
    d: Ptr<T>,
}

impl<T: Scalar> Job<'_, T> {
    /// The monomorphized pass for the conjugations `OA`, `OB` and `OC`.
    ///
    /// Each arm of the match is one strided-basic kernel with its own closure,
    /// chosen once per execution.
    fn go<OA, OB, OC>(&self, exec: &Exec<'_>) -> core::result::Result<(), StridedError>
    where
        OA: ElementOp<T>,
        OB: ElementOp<T>,
        OC: ElementOp<T>,
    {
        let one = <T as Element>::one();
        let (alpha, beta) = (self.alpha, self.beta);
        let plan = self.plan;
        run_with_exec(exec, plan.total, |_| {
            let dims = plan.dims.as_slice();
            let st = &plan.strides;
            // SAFETY: the contract of `ElementPlan::run`: every origin and
            // layout is validated; the destination is exclusive and
            // injective; a separate C is disjoint from it. In-place forms never
            // build a second view of the destination.
            unsafe {
                let dv = || write_view(self.d.get(), dims, &st[3]);
                let av = || read_view::<T, OA>(self.ptrs[0].get(), dims, &st[0]);
                let bv = || read_view::<T, OB>(self.ptrs[1].get(), dims, &st[1]);
                let cv = |p: *const T| read_view::<T, OC>(p, dims, &st[2]);
                match (self.kind, self.mode) {
                    (Kind::None, Mode::Overwrite) => {
                        // `op_D(0)`: a stride-0 broadcast of zero.
                        let z = [<T as Element>::zero()];
                        let zeros = vec![0isize; dims.len()];
                        let zv = StridedView::<T, Identity>::new_unchecked(&z, dims, &zeros, 0);
                        map_into(&mut dv(), &zv, |x| x)
                    }
                    (Kind::None, Mode::InPlace) => {
                        map_update_into::<T, OC>(&mut dv(), move |z| beta * z)
                    }
                    (Kind::None, Mode::Separate(p)) => {
                        map_into(&mut dv(), &cv(p.get()), move |x| beta * x)
                    }
                    (Kind::Scaled, Mode::Overwrite) => copy_scale(&mut dv(), &av(), alpha),
                    (Kind::Scaled, Mode::InPlace) if beta == one && OC::IS_IDENTITY => {
                        axpy(&mut dv(), &av(), alpha)
                    }
                    (Kind::Scaled, Mode::InPlace) => {
                        zip_update2_into::<T, T, OC, OA>(&mut dv(), &av(), move |z, x| {
                            alpha * x + beta * z
                        })
                    }
                    (Kind::Scaled, Mode::Separate(p)) => {
                        zip_map2_into(&mut dv(), &av(), &cv(p.get()), move |x, z| {
                            alpha * x + beta * z
                        })
                    }
                    (Kind::Product, Mode::Overwrite) if alpha == one => {
                        mul_into(&mut dv(), &av(), &bv())
                    }
                    (Kind::Product, Mode::Overwrite) => {
                        zip_map2_into(&mut dv(), &av(), &bv(), move |x, y| alpha * x * y)
                    }
                    (Kind::Product, Mode::InPlace)
                        if alpha == one && beta == one && OC::IS_IDENTITY =>
                    {
                        fma(&mut dv(), &av(), &bv())
                    }
                    (Kind::Product, Mode::InPlace) => zip_update3_into::<T, T, T, OC, OA, OB>(
                        &mut dv(),
                        &av(),
                        &bv(),
                        move |z, x, y| alpha * x * y + beta * z,
                    ),
                    (Kind::Product, Mode::Separate(p)) => {
                        zip_map3_into(&mut dv(), &av(), &bv(), &cv(p.get()), move |x, y, z| {
                            alpha * x * y + beta * z
                        })
                    }
                }
            }
        })
    }
}
