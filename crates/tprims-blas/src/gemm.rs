use faer::linalg::matmul::matmul_with_conj;
use faer::{Accum, MatMut, MatRef};
use strided_view::StridedViewMut;
use tprims_exec::{Exec, Par, WidthPolicy};

use crate::operand::{check_injective, scale_in_place, Mat2};
use crate::scalar::{one, zero};
use crate::{Error, MatIn, Result, Scalar};

/// How GEMM chooses its parallel width.
///
/// # Examples
///
/// ```
/// assert!(tprims_blas::GemmPolicy::default().ns_per_flop > 0.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemmPolicy {
    /// Estimated serial time per real flop (provisional: 20 GFLOP/s).
    pub ns_per_flop: f64,
    /// Cost model mapping the estimate to a width.
    pub width: WidthPolicy,
}

impl Default for GemmPolicy {
    fn default() -> Self {
        Self {
            ns_per_flop: 0.05,
            width: WidthPolicy::default(),
        }
    }
}

pub(crate) fn to_faer(par: Par) -> faer::Par {
    match par {
        Par::Seq => faer::Par::Seq,
        Par::Threads(n) => faer::Par::rayon(n.get()),
    }
}

pub(crate) fn gemm_width<T: Scalar>(exec: &Exec<'_>, m: usize, n: usize, k: usize) -> usize {
    let p = GemmPolicy::default();
    let flops = 2.0 * m as f64 * n as f64 * k as f64 * if T::IS_COMPLEX_SCALAR { 4.0 } else { 1.0 };
    exec.width_for(flops * p.ns_per_flop, &p.width)
}

/// Validated GEMM problem.
pub(crate) struct GemmShape {
    pub a: Mat2,
    pub b: Mat2,
    pub c: Mat2,
}

pub(crate) fn check_gemm(a: Mat2, b: Mat2, c: Mat2) -> Result<GemmShape> {
    if a.cols != b.rows || a.rows != c.rows || b.cols != c.cols {
        return Err(Error::Shape(format!(
            "A {}x{}, B {}x{}, C {}x{}",
            a.rows, a.cols, b.rows, b.cols, c.rows, c.cols
        )));
    }
    check_injective(&[(c.rows, c.rs), (c.cols, c.cs)])?;
    Ok(GemmShape { a, b, c })
}

/// One GEMM on validated raw pointers with the given faer parallelism.
///
/// # Safety
///
/// Pointers with the shapes' strides address valid elements; `c` is
/// exclusively borrowed and injective (checked by [`check_gemm`]) and does not
/// alias `a` or `b`.
#[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set.
pub(crate) unsafe fn gemm_raw<T: Scalar>(
    s: &GemmShape,
    alpha: T,
    a: *const T,
    ca: crate::Conj,
    b: *const T,
    cb: crate::Conj,
    beta: T,
    c: *mut T,
    par: faer::Par,
) {
    let (m, n, k) = (s.c.rows, s.c.cols, s.a.cols);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 || alpha == zero() {
        // alpha == 0 or an empty K: C = beta * C; A and B are not referenced.
        // SAFETY: forwarded.
        unsafe { scale_in_place(c, &s.c, beta) };
        return;
    }
    let accum = if beta == zero() {
        Accum::Replace
    } else {
        if beta != one() {
            // SAFETY: forwarded.
            unsafe { scale_in_place(c, &s.c, beta) };
        }
        Accum::Add
    };
    // SAFETY: the caller's contract; views are read-only / exclusive as faer requires.
    let (am, bm, cm) = unsafe {
        (
            MatRef::<T>::from_raw_parts(a, m, k, s.a.rs, s.a.cs),
            MatRef::<T>::from_raw_parts(b, k, n, s.b.rs, s.b.cs),
            MatMut::<T>::from_raw_parts_mut(c, m, n, s.c.rs, s.c.cs),
        )
    };
    matmul_with_conj(cm, accum, am, ca.to_faer(), bm, cb.to_faer(), alpha, par);
}

/// `C = alpha * op(A) * op(B) + beta * C`, where `op` is the operand's
/// conjugation flag and transposes are expressed by the views' strides.
///
/// `beta == 0` never reads C; `alpha == 0` never reads A or B. Serial-size products run on the caller; larger
/// ones enter the context's pool once with a width chosen from the flop count.
///
/// # Errors
///
/// [`Error::Rank`], [`Error::Shape`], or [`Error::AliasedOutput`] for a C
/// whose strides make distinct elements share memory; nothing is written
/// then.
pub fn gemm<T: Scalar>(
    exec: &Exec<'_>,
    alpha: T,
    a: MatIn<'_, '_, T>,
    b: MatIn<'_, '_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    // The default configuration is faer with the views as they are, which is
    // what this entry point has always done.
    crate::gemm_with(exec, &crate::GemmConfig::default(), alpha, a, b, beta, c).map(|_| ())
}

/// Raw pointers moved into a pool closure.
#[derive(Clone, Copy)]
pub(crate) struct SendConst<T>(pub *const T);
// SAFETY: only dereferenced under the validated contracts documented at use.
unsafe impl<T> Send for SendConst<T> {}
unsafe impl<T> Sync for SendConst<T> {}

#[derive(Clone, Copy)]
pub(crate) struct SendMut<T>(pub *mut T);
// SAFETY: as SendConst; writes are to disjoint, injective regions.
unsafe impl<T> Send for SendMut<T> {}
unsafe impl<T> Sync for SendMut<T> {}
