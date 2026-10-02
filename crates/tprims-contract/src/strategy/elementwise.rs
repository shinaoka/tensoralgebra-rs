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
//! Reading the previous `D` through the destination (in-place accumulation)
//! cannot be a checked strided-basic call: those take the destination as a
//! `&mut` view and the old value as a second view over the same elements, which
//! would be two live references to one allocation. The in-place forms run on
//! strided-rs's public execution contract (`strided_basic::execution`: fused
//! plan, blocked inner-block walk, threaded map-reduce) with a raw-pointer
//! inner loop that is local to this module (`in_place_pass`); the traversal and
//! threading are still strided-rs's. `axpy` and `fma` cover the unit-scalar
//! in-place forms through checked calls.
//!
//! 1. `op_D` is folded into the other terms, since
//!    `conj(alpha*a*b + beta*c) = conj(alpha)*conj(a)*conj(b) + conj(beta)*conj(c)`;
//!    what remains is `conj_a'`, `conj_b'`, `conj_c'` and transformed scalars.
//! 2. One dispatch on those three flags (type-level `Identity`/`Conj` views;
//!    all `false` for a real element type) and on the input and C modes picks
//!    a monomorphized closure such as `|a, b| alpha * a * b`. No flag is
//!    tested per element.

use strided_basic::execution::{
    build_plan_fused, build_plan_fused_small, compute_costs, for_each_inner_block_preordered,
    mapreduce_threaded, rayon_threads, MINTHREADLENGTH, SMALL_TENSOR_THRESHOLD,
};
use strided_basic::{
    axpy, copy_scale, fma, map_into, mul_into, zip_map2_into, zip_map3_into, Conj, ElementOp,
    Identity, StridedError,
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
                        let d = self.d;
                        in_place_pass::<T, _>(plan, &[3], move |o, n, s| {
                            inner1::<T, OC>(d.get().offset(o[0]), s[0], n, &move |z| beta * z)
                        })
                    }
                    (Kind::None, Mode::Separate(p)) => {
                        map_into(&mut dv(), &cv(p.get()), move |x| beta * x)
                    }
                    (Kind::Scaled, Mode::Overwrite) => copy_scale(&mut dv(), &av(), alpha),
                    (Kind::Scaled, Mode::InPlace) if beta == one && OC::IS_IDENTITY => {
                        axpy(&mut dv(), &av(), alpha)
                    }
                    (Kind::Scaled, Mode::InPlace) => {
                        let (d, a) = (self.d, self.ptrs[0]);
                        in_place_pass::<T, _>(plan, &[3, 0], move |o, n, s| {
                            inner2::<T, OA, OC>(
                                d.get().offset(o[0]),
                                s[0],
                                a.get().offset(o[1]),
                                s[1],
                                n,
                                &move |x, z| alpha * x + beta * z,
                            )
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
                    (Kind::Product, Mode::InPlace) => {
                        let (d, a, b) = (self.d, self.ptrs[0], self.ptrs[1]);
                        in_place_pass::<T, _>(plan, &[3, 0, 1], move |o, n, s| {
                            inner3::<T, OA, OB, OC>(
                                d.get().offset(o[0]),
                                s[0],
                                a.get().offset(o[1]),
                                s[1],
                                b.get().offset(o[2]),
                                s[2],
                                n,
                                &move |x, y, z| alpha * x * y + beta * z,
                            )
                        })
                    }
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

/// Walk the plan's axes with strided-rs's fused, blocked, optionally threaded
/// traversal (`lists` names the stride vectors, the destination first) and call
/// `body(offsets, len, strides)` once per inner block.
///
/// # Safety
///
/// `body` accesses only the offsets of validated layouts; the destination
/// offsets of different blocks are disjoint (the destination is injective), so
/// blocks may run on different threads.
unsafe fn in_place_pass<T, F>(
    plan: &ElementPlan,
    lists: &[usize],
    body: F,
) -> core::result::Result<(), StridedError>
where
    F: Fn(&[isize], usize, &[isize]) + Sync,
{
    let strides: Vec<&[isize]> = lists.iter().map(|&j| plan.strides[j].as_slice()).collect();
    let (fused, ordered, kp) = if plan.total <= SMALL_TENSOR_THRESHOLD {
        // SAFETY: matching ranks, bounded validated layouts.
        unsafe { build_plan_fused_small(&plan.dims, &strides) }
    } else {
        // SAFETY: as above; destination is list 0.
        unsafe { build_plan_fused(&plan.dims, &strides, Some(0), core::mem::size_of::<T>()) }
    };
    let walk = |dims: &[usize], blocks: &[usize], sl: &[Vec<isize>], offs: &[isize]| {
        // SAFETY: the blocks the plan generates stay inside the layouts.
        unsafe {
            for_each_inner_block_preordered(dims, blocks, sl, offs, |o, n, s| {
                body(o, n, s);
                Ok(())
            })
        }
    };
    let origin = vec![0isize; lists.len()];
    let threads = rayon_threads();
    if plan.total > MINTHREADLENGTH && threads > 1 {
        // SAFETY: disjoint destination regions per partition (injective D).
        unsafe {
            let costs = compute_costs(&ordered);
            return mapreduce_threaded(
                &fused, &kp.block, &ordered, &origin, &costs, threads, 0, 1, &walk,
            );
        }
    }
    walk(&fused, &kp.block, &ordered, &origin)
}

/// `d[i] = f(OC(d[i]))` over one inner block.
///
/// # Safety
///
/// `d` addresses `n` live, exclusively owned elements at stride `ds`.
unsafe fn inner1<T: Scalar, OC: ElementOp<T>>(d: *mut T, ds: isize, n: usize, f: &impl Fn(T) -> T) {
    // SAFETY: the caller's contract.
    unsafe {
        if ds == 1 {
            for z in core::slice::from_raw_parts_mut(d, n) {
                *z = f(OC::apply(*z));
            }
        } else {
            for i in 0..n as isize {
                let q = d.offset(i * ds);
                *q = f(OC::apply(*q));
            }
        }
    }
}

/// `d[i] = f(OA(a[i]), OC(d[i]))` over one inner block.
///
/// # Safety
///
/// As [`inner1`], and `a` addresses `n` live elements disjoint from `d`.
unsafe fn inner2<T: Scalar, OA: ElementOp<T>, OC: ElementOp<T>>(
    d: *mut T,
    ds: isize,
    a: *const T,
    sa: isize,
    n: usize,
    f: &impl Fn(T, T) -> T,
) {
    // SAFETY: the caller's contract.
    unsafe {
        if ds == 1 && sa == 1 {
            let a = core::slice::from_raw_parts(a, n);
            for (z, &x) in core::slice::from_raw_parts_mut(d, n).iter_mut().zip(a) {
                *z = f(OA::apply(x), OC::apply(*z));
            }
        } else {
            for i in 0..n as isize {
                let q = d.offset(i * ds);
                *q = f(OA::apply(*a.offset(i * sa)), OC::apply(*q));
            }
        }
    }
}

/// `d[i] = f(OA(a[i]), OB(b[i]), OC(d[i]))` over one inner block.
///
/// # Safety
///
/// As [`inner2`], with `b` likewise.
#[allow(clippy::too_many_arguments)] // INVARIANT: three strided operands, length and body.
unsafe fn inner3<T: Scalar, OA: ElementOp<T>, OB: ElementOp<T>, OC: ElementOp<T>>(
    d: *mut T,
    ds: isize,
    a: *const T,
    sa: isize,
    b: *const T,
    sb: isize,
    n: usize,
    f: &impl Fn(T, T, T) -> T,
) {
    // SAFETY: the caller's contract.
    unsafe {
        if ds == 1 && sa == 1 && sb == 1 {
            let a = core::slice::from_raw_parts(a, n);
            let b = core::slice::from_raw_parts(b, n);
            for ((z, &x), &y) in core::slice::from_raw_parts_mut(d, n)
                .iter_mut()
                .zip(a)
                .zip(b)
            {
                *z = f(OA::apply(x), OB::apply(y), OC::apply(*z));
            }
        } else {
            for i in 0..n as isize {
                let q = d.offset(i * ds);
                *q = f(
                    OA::apply(*a.offset(i * sa)),
                    OB::apply(*b.offset(i * sb)),
                    OC::apply(*q),
                );
            }
        }
    }
}
