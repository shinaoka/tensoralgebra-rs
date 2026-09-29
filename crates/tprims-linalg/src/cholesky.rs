use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::{lblt, llt};
use faer::{Conj, MatMut};
use strided_view::{StridedView, StridedViewMut};
use tprims_blas::Scalar;
use tprims_exec::Exec;

use crate::util::{flops, install, rhs_mut, square};
use crate::{Error, Matrix, Result};

/// Cholesky factor `A = L Lᴴ` of a Hermitian positive definite matrix.
#[derive(Clone, Debug)]
pub struct Cholesky<T> {
    l: Matrix<T>,
}

/// Factor `A = L Lᴴ`, reading only the lower triangle of `a`.
///
/// # Errors
///
/// [`Error::NotSquare`], [`Error::Rank`], [`Error::NotPositiveDefinite`]
/// with the index of the failing pivot.
pub fn cholesky<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>) -> Result<Cholesky<T>> {
    let mut l = Matrix::from_view(a)?;
    let n = square(l.rows(), l.cols())?;
    if n > 0 {
        let mat = l.as_faer_mut();
        install(exec, flops::<T>((n * n * n) as f64 / 3.0), move |par| {
            let mut mem = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<T>(
                n,
                par,
                Default::default(),
            ));
            llt::factor::cholesky_in_place(
                mat,
                Default::default(),
                par,
                MemStack::new(&mut mem),
                Default::default(),
            )
        })
        .map_err(|e| match e {
            llt::factor::LltError::NonPositivePivot { index } => {
                Error::NotPositiveDefinite { index }
            }
        })?;
    }
    for j in 0..n {
        for i in 0..j {
            l.set(i, j, <T as tensorcontract::Element>::zero());
        }
    }
    Ok(Cholesky { l })
}

impl<T: Scalar> Cholesky<T> {
    /// The lower-triangular factor (upper part zero).
    pub fn l(&self) -> &Matrix<T> {
        &self.l
    }

    /// Solve `A X = B` in place in `b`.
    ///
    /// # Errors
    ///
    /// [`Error::Rank`], [`Error::Shape`], [`Error::AliasedOutput`].
    pub fn solve(&self, exec: &Exec<'_>, b: &mut StridedViewMut<'_, T>) -> Result<()> {
        let n = self.l.rows();
        let rhs = rhs_mut(b, n)?;
        let k = rhs.ncols();
        let l = self.l.as_faer();
        install(exec, flops::<T>(2.0 * (n * n * k) as f64), move |par| {
            let mut mem = MemBuffer::new(llt::solve::solve_in_place_scratch::<T>(n, k, par));
            llt::solve::solve_in_place_with_conj(l, Conj::No, rhs, par, MemStack::new(&mut mem));
        });
        Ok(())
    }
}

/// Bunch–Kaufman `P A Pᵀ = L B Lᴴ` of a Hermitian (possibly indefinite)
/// matrix, reading the lower triangle.
#[derive(Clone, Debug)]
pub struct Ldlt<T> {
    lb: Matrix<T>,
    subdiag: Vec<T>,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
}

/// Factor a Hermitian matrix with Bunch–Kaufman pivoting (faer `lblt`).
/// Never fails on singular input; a later solve then produces non-finite
/// values rather than an error (no pivot test is defined for 2x2 blocks).
///
/// # Errors
///
/// [`Error::NotSquare`], [`Error::Rank`].
pub fn ldlt<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>) -> Result<Ldlt<T>> {
    let mut lb = Matrix::from_view(a)?;
    let n = square(lb.rows(), lb.cols())?;
    let mut subdiag = vec![<T as tensorcontract::Element>::zero(); n];
    let (mut perm, mut perm_inv) = (vec![0usize; n], vec![0usize; n]);
    if n > 0 {
        let mat: MatMut<'_, T> = lb.as_faer_mut();
        let sd = faer::diag::DiagMut::from_slice_mut(&mut subdiag);
        let (p, pi) = (&mut perm[..], &mut perm_inv[..]);
        install(exec, flops::<T>((n * n * n) as f64 / 3.0), move |par| {
            let mut mem = MemBuffer::new(lblt::factor::cholesky_in_place_scratch::<usize, T>(
                n,
                par,
                Default::default(),
            ));
            let _ = lblt::factor::cholesky_in_place(
                mat,
                sd,
                p,
                pi,
                par,
                MemStack::new(&mut mem),
                Default::default(),
            );
        });
    }
    Ok(Ldlt {
        lb,
        subdiag,
        perm,
        perm_inv,
    })
}

impl<T: Scalar> Ldlt<T> {
    /// Solve `A X = B` in place in `b`.
    ///
    /// # Errors
    ///
    /// [`Error::Rank`], [`Error::Shape`], [`Error::AliasedOutput`].
    pub fn solve(&self, exec: &Exec<'_>, b: &mut StridedViewMut<'_, T>) -> Result<()> {
        let n = self.lb.rows();
        let rhs = rhs_mut(b, n)?;
        let k = rhs.ncols();
        if n == 0 || k == 0 {
            return Ok(());
        }
        let lb = self.lb.as_faer();
        let sd = faer::diag::DiagRef::from_slice(&self.subdiag);
        // INVARIANT: perm/perm_inv come from faer's factorization of this matrix.
        let perm = faer::perm::PermRef::new_checked(&self.perm, &self.perm_inv, n);
        install(exec, flops::<T>(2.0 * (n * n * k) as f64), move |par| {
            let mut mem =
                MemBuffer::new(lblt::solve::solve_in_place_scratch::<usize, T>(n, k, par));
            lblt::solve::solve_in_place_with_conj(
                lb,
                lb.diagonal(),
                sd,
                Conj::No,
                perm,
                rhs,
                par,
                MemStack::new(&mut mem),
            );
        });
        Ok(())
    }
}
