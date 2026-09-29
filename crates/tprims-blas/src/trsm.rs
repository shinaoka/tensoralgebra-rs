use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place_with_conj, solve_unit_lower_triangular_in_place_with_conj,
    solve_unit_upper_triangular_in_place_with_conj, solve_upper_triangular_in_place_with_conj,
};
use faer::{MatMut, MatRef};
use strided_view::{StridedView, StridedViewMut};
use tprims_exec::Exec;

use crate::gemm::{gemm_width, to_faer, SendConst, SendMut};
use crate::operand::{check_injective, mat2, scale_in_place};
use crate::scalar::one;
use crate::{Conj, Error, Result, Scalar};

/// Which side the triangular matrix multiplies.
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_blas::Side::Left, tprims_blas::Side::Right);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Solve `op(A) X = alpha B`.
    Left,
    /// Solve `X op(A) = alpha B`.
    Right,
}

/// Which triangle of A is referenced.
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_blas::Uplo::Lower, tprims_blas::Uplo::Upper);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uplo {
    /// The lower triangle.
    Lower,
    /// The upper triangle.
    Upper,
}

/// Operation applied to A.
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_blas::Op::T, tprims_blas::Op::C);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// A.
    N,
    /// Aᵀ.
    T,
    /// Aᴴ.
    C,
}

/// Whether A has an implicit unit diagonal.
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_blas::Diag::Unit, tprims_blas::Diag::NonUnit);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diag {
    /// The stored diagonal is used.
    NonUnit,
    /// The diagonal is taken as one and not read.
    Unit,
}

/// Triangular solve with multiple right-hand sides, in place in B:
/// `op(A) X = alpha B` (Left) or `X op(A) = alpha B` (Right).
///
/// Only the `uplo` triangle of A is read (and not its diagonal for
/// [`Diag::Unit`]). Singular diagonals are not detected (BLAS semantics).
///
/// # Errors
///
/// [`Error::Rank`], [`Error::Shape`] (A not square, or B's solved dimension
/// differs), [`Error::AliasedOutput`] for an aliasing B.
#[allow(clippy::too_many_arguments)] // INVARIANT: the BLAS TRSM argument set.
pub fn trsm<T: Scalar>(
    exec: &Exec<'_>,
    side: Side,
    uplo: Uplo,
    op: Op,
    diag: Diag,
    alpha: T,
    a: &StridedView<'_, T>,
    b: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    let am = mat2("A", a.dims(), a.strides())?;
    let bm = mat2("B", b.dims(), b.strides())?;
    let n = am.rows;
    let solved = match side {
        Side::Left => bm.rows,
        Side::Right => bm.cols,
    };
    if am.cols != n || solved != n {
        return Err(Error::Shape(format!(
            "A {}x{}, B {}x{} ({side:?})",
            am.rows, am.cols, bm.rows, bm.cols
        )));
    }
    check_injective(&[(bm.rows, bm.rs), (bm.cols, bm.cs)])?;
    // op(A) as a view: T and C swap strides (and so the triangle).
    let (mut rs, mut cs, mut lower, conj) = match op {
        Op::N => (am.rs, am.cs, uplo == Uplo::Lower, Conj::No),
        Op::T => (am.cs, am.rs, uplo != Uplo::Lower, Conj::No),
        Op::C => (am.cs, am.rs, uplo != Uplo::Lower, Conj::Yes),
    };
    // Right side: X op(A) = B  <=>  op(A)ᵀ Xᵀ = Bᵀ.
    let (b_rows, b_cols, b_rs, b_cs) = match side {
        Side::Left => (bm.rows, bm.cols, bm.rs, bm.cs),
        Side::Right => {
            std::mem::swap(&mut rs, &mut cs);
            lower = !lower;
            (bm.cols, bm.rows, bm.cs, bm.rs)
        }
    };
    let width = gemm_width::<T>(exec, n, b_cols, n.div_ceil(2).max(1));
    let (ap, bp) = (SendConst(a.ptr()), SendMut(b.as_mut_ptr()));
    exec.install(width, move |par| {
        let (ap, bp) = (ap, bp);
        let par = to_faer(par);
        if alpha != one() {
            let bmat = crate::operand::Mat2 {
                rows: b_rows,
                cols: b_cols,
                rs: b_rs,
                cs: b_cs,
            };
            // SAFETY: B's validated layout; exclusive borrow.
            unsafe { scale_in_place(bp.0, &bmat, alpha) };
        }
        // SAFETY: A and B views were bounds-checked at construction; the
        // transposed layouts address the same elements; B is exclusive and
        // injective and does not alias A (distinct borrows).
        let (amat, bmat) = unsafe {
            (
                MatRef::<T>::from_raw_parts(ap.0, n, n, rs, cs),
                MatMut::<T>::from_raw_parts_mut(bp.0, b_rows, b_cols, b_rs, b_cs),
            )
        };
        let c = conj.to_faer();
        match (lower, diag) {
            (true, Diag::NonUnit) => solve_lower_triangular_in_place_with_conj(amat, c, bmat, par),
            (true, Diag::Unit) => {
                solve_unit_lower_triangular_in_place_with_conj(amat, c, bmat, par)
            }
            (false, Diag::NonUnit) => solve_upper_triangular_in_place_with_conj(amat, c, bmat, par),
            (false, Diag::Unit) => {
                solve_unit_upper_triangular_in_place_with_conj(amat, c, bmat, par)
            }
        }
    });
    Ok(())
}
