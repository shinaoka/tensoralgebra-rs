//! Batched linear algebra over equal-shape stacks `[rows, cols, batch]`
//! (batch axis last), with a per-item [`Status`].
//!
//! One call sizes its scratch once per lane and reuses it and a work matrix
//! across items; nothing is allocated per item. Small items over a large
//! batch are spread over the pool (each item serial); large items use inner
//! parallelism. Failed items leave their outputs unspecified.
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt;
use faer::linalg::evd::{self, ComputeEigenvectors};
use faer::linalg::lu::partial_pivoting;
use faer::linalg::svd::{self as fsvd, ComputeSvdVectors};
use faer::perm::PermRef;
use faer::{Conj, MatMut, MatRef};
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_blas::{is_injective_layout, GemmPolicy, Scalar};
use tprims_exec::{Exec, Par};

use crate::lu::singular_pivot;
use crate::util::{abs, flops};
use crate::{Error, Matrix, Result};

/// Outcome for one batch item.
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_linalg::Status::Ok, tprims_linalg::Status::NoConvergence);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Status {
    /// Succeeded.
    Ok,
    /// Numerically singular (solve).
    Singular {
        /// Failing pivot.
        index: usize,
    },
    /// Not positive definite (Cholesky).
    NotPositiveDefinite {
        /// Failing pivot.
        index: usize,
    },
    /// An iterative decomposition did not converge.
    NoConvergence,
    /// The item contains NaN or infinity.
    NonFinite,
}

/// A validated rank-3 (or rank-2 `[len, batch]`) layout.
#[derive(Clone, Copy, Debug)]
struct Stack3 {
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
    bs: isize,
    count: usize,
}

fn stack3(name: &'static str, dims: &[usize], strides: &[isize]) -> Result<Stack3> {
    if dims.len() != 3 {
        return Err(Error::Rank {
            operand: name,
            expected: 3,
            got: dims.len(),
        });
    }
    let n1 = |e: usize, s: isize| if e <= 1 { 1 } else { s };
    Ok(Stack3 {
        rows: dims[0],
        cols: dims[1],
        rs: n1(dims[0], strides[0]),
        cs: n1(dims[1], strides[1]),
        bs: strides[2],
        count: dims[2],
    })
}

fn stack2(name: &'static str, dims: &[usize], strides: &[isize]) -> Result<Stack3> {
    if dims.len() != 2 {
        return Err(Error::Rank {
            operand: name,
            expected: 2,
            got: dims.len(),
        });
    }
    Ok(Stack3 {
        rows: dims[0],
        cols: 1,
        rs: if dims[0] <= 1 { 1 } else { strides[0] },
        cs: 1,
        bs: strides[1],
        count: dims[1],
    })
}

fn out_ok(s: &Stack3) -> Result<()> {
    if is_injective_layout(&[(s.rows, s.rs), (s.cols, s.cs), (s.count, s.bs)]) {
        Ok(())
    } else {
        Err(Error::AliasedOutput)
    }
}

fn same(what: &str, got: (usize, usize, usize), want: (usize, usize, usize)) -> Result<()> {
    if got == want {
        Ok(())
    } else {
        Err(Error::Shape(format!("{what}: {got:?}, expected {want:?}")))
    }
}

/// Raw pointer moved into lanes; items are disjoint (validated layouts).
#[derive(Clone, Copy)]
struct P<T>(*mut T);
// SAFETY: dereferenced only at disjoint, validated item offsets.
unsafe impl<T> Send for P<T> {}
unsafe impl<T> Sync for P<T> {}

impl<T> P<T> {
    fn get(self) -> *mut T {
        self.0
    }

    /// # Safety
    /// `i < count` of a non-empty validated stack starting at this pointer.
    unsafe fn item(self, s: &Stack3, i: usize) -> *mut T {
        unsafe { self.0.offset(i as isize * s.bs) }
    }
}

/// # Safety
/// `p` addresses item storage of layout `s` (validated view).
unsafe fn mat_ref<'a, T>(p: *const T, s: &Stack3) -> MatRef<'a, T> {
    unsafe { MatRef::from_raw_parts(p, s.rows, s.cols, s.rs, s.cs) }
}

/// # Safety
/// As [`mat_ref`], plus exclusive and injective.
unsafe fn mat_mut<'a, T>(p: *mut T, s: &Stack3) -> MatMut<'a, T> {
    unsafe { MatMut::from_raw_parts_mut(p, s.rows, s.cols, s.rs, s.cs) }
}

/// Run `count` items: `make(par)` builds a lane's reusable state once,
/// `f(state, i, par)` handles item `i`.
fn run_items<S>(
    exec: &Exec<'_>,
    count: usize,
    item_flops: f64,
    make: &(dyn Fn(faer::Par) -> S + Sync),
    f: &(dyn Fn(&mut S, usize, faer::Par) + Sync),
) {
    let p = GemmPolicy::default();
    let inner = exec.width_for(item_flops * p.ns_per_flop, &p.width);
    let total = exec.width_for(item_flops * count as f64 * p.ns_per_flop, &p.width);
    if inner == 1 && total > 1 && count > 1 {
        let lanes = total.min(count);
        let e = exec.with_budget(lanes).unwrap_or(*exec);
        e.for_each_partition(lanes, &|l| {
            let mut s = make(faer::Par::Seq);
            for i in l * count / lanes..(l + 1) * count / lanes {
                f(&mut s, i, faer::Par::Seq);
            }
        });
    } else {
        exec.install(inner, |par| {
            let par = match par {
                Par::Seq => faer::Par::Seq,
                Par::Threads(n) => faer::Par::rayon(n.get()),
            };
            let mut s = make(par);
            for i in 0..count {
                f(&mut s, i, par);
            }
        });
    }
}

/// Per-item statuses written by lanes at disjoint indices.
fn statuses(count: usize) -> Vec<Status> {
    vec![Status::Ok; count]
}

/// Solve `A_i X_i = B_i` for every item, in place in `b` (`[n, k, batch]`).
///
/// # Errors
///
/// Shape, rank and aliasing errors for the whole call; per-item failures are
/// statuses ([`Status::Singular`]).
pub fn solve<T: Scalar>(
    exec: &Exec<'_>,
    a: &StridedView<'_, T>,
    b: &mut StridedViewMut<'_, T>,
) -> Result<Vec<Status>> {
    let sa = stack3("A", a.dims(), a.strides())?;
    let sb = stack3("B", b.dims(), b.strides())?;
    if sa.rows != sa.cols {
        return Err(Error::NotSquare {
            rows: sa.rows,
            cols: sa.cols,
        });
    }
    same(
        "B",
        (sb.rows, sb.cols, sb.count),
        (sa.rows, sb.cols, sa.count),
    )?;
    out_ok(&sb)?;
    let (n, k, count) = (sa.rows, sb.cols, sa.count);
    let mut st = statuses(count);
    if n == 0 || count == 0 {
        return Ok(st);
    }
    let (ap, bp, sp) = (P(a.ptr() as *mut T), P(b.as_mut_ptr()), P(st.as_mut_ptr()));
    let req = |par| {
        StackReq::any_of(&[
            partial_pivoting::factor::lu_in_place_scratch::<usize, T>(
                n,
                n,
                par,
                Default::default(),
            ),
            partial_pivoting::solve::solve_in_place_scratch::<usize, T>(n, k, par),
        ])
    };
    let make = |par| {
        (
            MemBuffer::new(req(par)),
            Matrix::<T>::zeros(n, n),
            vec![0usize; n],
            vec![0usize; n],
        )
    };
    run_items(
        exec,
        count,
        flops::<T>(((n * n * n) + 2 * n * n * k) as f64),
        &make,
        &|s, i, par| {
            let (mem, work, perm, perm_inv) = s;
            // SAFETY: i < count; A and B are non-empty validated stacks; B items
            // are disjoint (out_ok); the status slot i is written by this lane only.
            unsafe {
                let ai = mat_ref(ap.item(&sa, i) as *const T, &sa);
                work.as_faer_mut().copy_from(ai);
                partial_pivoting::factor::lu_in_place(
                    work.as_faer_mut(),
                    perm,
                    perm_inv,
                    par,
                    MemStack::new(mem),
                    Default::default(),
                );
                if let Some(index) = singular_pivot(work) {
                    *sp.get().add(i) = Status::Singular { index };
                    return;
                }
                if k > 0 {
                    let perm = PermRef::new_checked(perm, perm_inv, n);
                    let bi = mat_mut(bp.item(&sb, i), &sb);
                    partial_pivoting::solve::solve_in_place_with_conj(
                        work.as_faer(),
                        work.as_faer(),
                        perm,
                        Conj::No,
                        bi,
                        par,
                        MemStack::new(mem),
                    );
                }
            }
        },
    );
    Ok(st)
}

/// Cholesky of every item in place (`[n, n, batch]`): the lower triangle
/// becomes `L`, the strict upper triangle is zeroed.
///
/// # Errors
///
/// Shape, rank and aliasing errors; per-item [`Status::NotPositiveDefinite`].
pub fn cholesky<T: Scalar>(exec: &Exec<'_>, a: &mut StridedViewMut<'_, T>) -> Result<Vec<Status>> {
    let sa = stack3("A", a.dims(), a.strides())?;
    if sa.rows != sa.cols {
        return Err(Error::NotSquare {
            rows: sa.rows,
            cols: sa.cols,
        });
    }
    out_ok(&sa)?;
    let (n, count) = (sa.rows, sa.count);
    let mut st = statuses(count);
    if n == 0 || count == 0 {
        return Ok(st);
    }
    let (ap, sp) = (P(a.as_mut_ptr()), P(st.as_mut_ptr()));
    let make = |par| {
        MemBuffer::new(llt::factor::cholesky_in_place_scratch::<T>(
            n,
            par,
            Default::default(),
        ))
    };
    run_items(
        exec,
        count,
        flops::<T>((n * n * n) as f64 / 3.0),
        &make,
        &|mem, i, par| {
            // SAFETY: i < count; disjoint validated items; status slot i is ours.
            unsafe {
                let p = ap.item(&sa, i);
                match llt::factor::cholesky_in_place(
                    mat_mut(p, &sa),
                    Default::default(),
                    par,
                    MemStack::new(mem),
                    Default::default(),
                ) {
                    Ok(_) => {
                        for j in 1..n {
                            for r in 0..j {
                                *p.offset(r as isize * sa.rs + j as isize * sa.cs) =
                                    <T as Element>::zero();
                            }
                        }
                    }
                    Err(llt::factor::LltError::NonPositivePivot { index }) => {
                        *sp.get().add(i) = Status::NotPositiveDefinite { index }
                    }
                }
            }
        },
    );
    Ok(st)
}

/// Hermitian eigendecomposition of every item (lower triangle read):
/// eigenvalues into `w` (`[n, batch]`, non-decreasing), eigenvectors into
/// `v` (`[n, n, batch]`) when given.
///
/// # Errors
///
/// Shape, rank and aliasing errors; per-item [`Status::NonFinite`] or
/// [`Status::NoConvergence`].
pub fn eigh<T: Scalar>(
    exec: &Exec<'_>,
    a: &StridedView<'_, T>,
    w: &mut StridedViewMut<'_, T::Re>,
    v: Option<&mut StridedViewMut<'_, T>>,
) -> Result<Vec<Status>> {
    let sa = stack3("A", a.dims(), a.strides())?;
    if sa.rows != sa.cols {
        return Err(Error::NotSquare {
            rows: sa.rows,
            cols: sa.cols,
        });
    }
    let sw = stack2("w", w.dims(), w.strides())?;
    same("w", (sw.rows, 1, sw.count), (sa.rows, 1, sa.count))?;
    out_ok(&sw)?;
    let sv = match &v {
        Some(v) => {
            let s = stack3("V", v.dims(), v.strides())?;
            same("V", (s.rows, s.cols, s.count), (sa.rows, sa.rows, sa.count))?;
            out_ok(&s)?;
            Some(s)
        }
        None => None,
    };
    let (n, count) = (sa.rows, sa.count);
    let mut st = statuses(count);
    if n == 0 || count == 0 {
        return Ok(st);
    }
    let (ap, wp, sp) = (P(a.ptr() as *mut T), P(w.as_mut_ptr()), P(st.as_mut_ptr()));
    let vp = v.map(|v| P(v.as_mut_ptr()));
    let cv = if vp.is_some() {
        ComputeEigenvectors::Yes
    } else {
        ComputeEigenvectors::No
    };
    let make = |par| {
        (
            MemBuffer::new(evd::self_adjoint_evd_scratch::<T>(
                n,
                cv,
                par,
                Default::default(),
            )),
            vec![<T as Element>::zero(); n],
        )
    };
    run_items(
        exec,
        count,
        flops::<T>(9.0 * (n * n * n) as f64),
        &make,
        &|s, i, par| {
            let (mem, vals) = s;
            // SAFETY: i < count; validated non-empty stacks; disjoint outputs.
            unsafe {
                let ai = mat_ref(ap.item(&sa, i) as *const T, &sa);
                if (0..n).any(|j| (j..n).any(|r| !abs(*ai.get(r, j)).is_finite())) {
                    *sp.get().add(i) = Status::NonFinite;
                    return;
                }
                let vi = match (vp, &sv) {
                    (Some(p), Some(s)) => Some(mat_mut(p.item(s, i), s)),
                    _ => None,
                };
                let d = faer::diag::DiagMut::from_slice_mut(vals);
                if evd::self_adjoint_evd(ai, d, vi, par, MemStack::new(mem), Default::default())
                    .is_err()
                {
                    *sp.get().add(i) = Status::NoConvergence;
                    return;
                }
                let wi = wp.item(&sw, i);
                for (j, &x) in vals.iter().enumerate() {
                    *wi.offset(j as isize * sw.rs) = Element::re(x);
                }
            }
        },
    );
    Ok(st)
}

/// Thin SVD of every item: singular values into `s` (`[min(m, n), batch]`,
/// non-increasing) and, when given, `U` (`[m, k, batch]`) and `V`
/// (`[n, k, batch]`).
///
/// # Errors
///
/// Shape, rank and aliasing errors; per-item [`Status::NonFinite`] or
/// [`Status::NoConvergence`].
#[allow(clippy::type_complexity)] // INVARIANT: the optional (U, V) output pair.
pub fn svd<T: Scalar>(
    exec: &Exec<'_>,
    a: &StridedView<'_, T>,
    s: &mut StridedViewMut<'_, T::Re>,
    uv: Option<(&mut StridedViewMut<'_, T>, &mut StridedViewMut<'_, T>)>,
) -> Result<Vec<Status>> {
    let sa = stack3("A", a.dims(), a.strides())?;
    let (m, n, count) = (sa.rows, sa.cols, sa.count);
    let k = m.min(n);
    let ss = stack2("s", s.dims(), s.strides())?;
    same("s", (ss.rows, 1, ss.count), (k, 1, count))?;
    out_ok(&ss)?;
    let (up, vp, su, svv) = match uv {
        Some((u, v)) => {
            let su = stack3("U", u.dims(), u.strides())?;
            let sv = stack3("V", v.dims(), v.strides())?;
            same("U", (su.rows, su.cols, su.count), (m, k, count))?;
            same("V", (sv.rows, sv.cols, sv.count), (n, k, count))?;
            out_ok(&su)?;
            out_ok(&sv)?;
            (
                Some(P(u.as_mut_ptr())),
                Some(P(v.as_mut_ptr())),
                Some(su),
                Some(sv),
            )
        }
        None => (None, None, None, None),
    };
    let mut st = statuses(count);
    if k == 0 || count == 0 {
        return Ok(st);
    }
    let (ap, sp, stp) = (P(a.ptr() as *mut T), P(s.as_mut_ptr()), P(st.as_mut_ptr()));
    let cvec = if up.is_some() {
        ComputeSvdVectors::Thin
    } else {
        ComputeSvdVectors::No
    };
    let make = |par| {
        (
            MemBuffer::new(fsvd::svd_scratch::<T>(
                m,
                n,
                cvec,
                cvec,
                par,
                Default::default(),
            )),
            vec![<T as Element>::zero(); k],
        )
    };
    run_items(
        exec,
        count,
        flops::<T>(12.0 * (m * n * k) as f64),
        &make,
        &|state, i, par| {
            let (mem, vals) = state;
            // SAFETY: i < count; validated non-empty stacks; disjoint outputs.
            unsafe {
                let ai = mat_ref(ap.item(&sa, i) as *const T, &sa);
                if (0..n).any(|j| (0..m).any(|r| !abs(*ai.get(r, j)).is_finite())) {
                    *stp.get().add(i) = Status::NonFinite;
                    return;
                }
                let (ui, vi) = match (up, vp, &su, &svv) {
                    (Some(u), Some(v), Some(su), Some(sv)) => (
                        Some(mat_mut(u.item(su, i), su)),
                        Some(mat_mut(v.item(sv, i), sv)),
                    ),
                    _ => (None, None),
                };
                let d = faer::diag::DiagMut::from_slice_mut(vals);
                if fsvd::svd(ai, d, ui, vi, par, MemStack::new(mem), Default::default()).is_err() {
                    *stp.get().add(i) = Status::NoConvergence;
                    return;
                }
                let si = sp.item(&ss, i);
                for (j, &x) in vals.iter().enumerate() {
                    *si.offset(j as isize * ss.rs) = Element::re(x);
                }
            }
        },
    );
    Ok(st)
}
