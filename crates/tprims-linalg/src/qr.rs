use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder::{
    apply_block_householder_sequence_on_the_left_in_place_scratch,
    apply_block_householder_sequence_on_the_left_in_place_with_conj,
    apply_block_householder_sequence_transpose_on_the_left_in_place_scratch,
    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
};
use faer::linalg::qr::{col_pivoting, no_pivoting};
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::{Conj, MatRef};
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_blas::Scalar;
use tprims_exec::Exec;

use crate::util::{abs, flops, install, rhs_mut};
use crate::{Matrix, Result};

/// Householder reflectors and block factors shared by both QR kinds.
#[derive(Clone, Debug)]
struct Householder<T> {
    qr: Matrix<T>,
    coeff: Matrix<T>,
}

impl<T: Scalar> Householder<T> {
    fn k(&self) -> usize {
        self.qr.rows().min(self.qr.cols())
    }

    fn basis(&self) -> MatRef<'_, T> {
        self.qr.as_faer().get(.., ..self.k())
    }

    fn r(&self) -> Matrix<T> {
        let (k, n) = (self.k(), self.qr.cols());
        let mut r = Matrix::zeros(k, n);
        for j in 0..n {
            for i in 0..k.min(j + 1) {
                r.set(i, j, self.qr.get(i, j));
            }
        }
        r
    }

    /// Apply `Q` (`adjoint == false`) or `Qᴴ` to `b` in place.
    fn apply(&self, exec: &Exec<'_>, adjoint: bool, b: &mut StridedViewMut<'_, T>) -> Result<()> {
        let m = self.qr.rows();
        let rhs = rhs_mut(b, m)?;
        let nrhs = rhs.ncols();
        if self.k() == 0 || nrhs == 0 {
            return Ok(());
        }
        let (basis, factor) = (self.basis(), self.coeff.as_faer());
        let bs = self.coeff.rows();
        install(
            exec,
            flops::<T>(4.0 * (m * self.k() * nrhs) as f64),
            move |par| {
                let req = if adjoint {
                    apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(
                        m, bs, nrhs,
                    )
                } else {
                    apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(m, bs, nrhs)
                };
                let mut mem = MemBuffer::new(req);
                let stack = MemStack::new(&mut mem);
                if adjoint {
                    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                        basis,
                        factor,
                        Conj::Yes,
                        rhs,
                        par,
                        stack,
                    )
                } else {
                    apply_block_householder_sequence_on_the_left_in_place_with_conj(
                        basis,
                        factor,
                        Conj::No,
                        rhs,
                        par,
                        stack,
                    )
                }
            },
        );
        Ok(())
    }

    fn q(&self, exec: &Exec<'_>, cols: usize) -> Result<Matrix<T>> {
        let m = self.qr.rows();
        let mut q = Matrix::zeros(m, cols);
        for i in 0..m.min(cols) {
            q.set(i, i, <T as Element>::one());
        }
        self.apply(exec, false, &mut q.view_mut())?;
        Ok(q)
    }
}

/// `A = Q R` by Householder reflections (faer blocked QR).
#[derive(Clone, Debug)]
pub struct Qr<T> {
    h: Householder<T>,
}

fn block_size<T: Scalar>(m: usize, n: usize) -> usize {
    no_pivoting::factor::recommended_block_size::<T>(m, n).max(1)
}

/// QR without pivoting; `A` may be tall, wide or square.
///
/// # Errors
///
/// [`crate::Error::Rank`].
pub fn qr<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>) -> Result<Qr<T>> {
    let mut qrm = Matrix::from_view(a)?;
    let (m, n) = (qrm.rows(), qrm.cols());
    let k = m.min(n);
    let bs = block_size::<T>(m, n);
    let mut coeff = Matrix::zeros(bs, k);
    if k > 0 {
        let (mat, c) = (qrm.as_faer_mut(), coeff.as_faer_mut());
        install(exec, flops::<T>(2.0 * (m * n * k) as f64), move |par| {
            let mut mem = MemBuffer::new(no_pivoting::factor::qr_in_place_scratch::<T>(
                m,
                n,
                bs,
                par,
                Default::default(),
            ));
            no_pivoting::factor::qr_in_place(
                mat,
                c,
                par,
                MemStack::new(&mut mem),
                Default::default(),
            );
        });
    }
    Ok(Qr {
        h: Householder { qr: qrm, coeff },
    })
}

impl<T: Scalar> Qr<T> {
    /// Upper-triangular (trapezoidal) `R`, `min(m, n) x n`.
    pub fn r(&self) -> Matrix<T> {
        self.h.r()
    }

    /// Thin `Q`, `m x min(m, n)`.
    ///
    /// # Errors
    ///
    /// None in practice (internal shapes are consistent).
    pub fn q_thin(&self, exec: &Exec<'_>) -> Result<Matrix<T>> {
        self.h.q(exec, self.h.k())
    }

    /// Full unitary `Q`, `m x m`.
    ///
    /// # Errors
    ///
    /// None in practice.
    pub fn q_full(&self, exec: &Exec<'_>) -> Result<Matrix<T>> {
        self.h.q(exec, self.h.qr.rows())
    }

    /// `B = Q B` in place (`B` has `m` rows).
    ///
    /// # Errors
    ///
    /// [`crate::Error::Rank`], [`crate::Error::Shape`], [`crate::Error::AliasedOutput`].
    pub fn apply_q(&self, exec: &Exec<'_>, b: &mut StridedViewMut<'_, T>) -> Result<()> {
        self.h.apply(exec, false, b)
    }

    /// `B = Qᴴ B` in place.
    ///
    /// # Errors
    ///
    /// As [`Qr::apply_q`].
    pub fn apply_qh(&self, exec: &Exec<'_>, b: &mut StridedViewMut<'_, T>) -> Result<()> {
        self.h.apply(exec, true, b)
    }
}

/// `A P = Q R` with column pivoting (rank-revealing).
#[derive(Clone, Debug)]
pub struct ColPivQr<T> {
    h: Householder<T>,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
}

/// QR with column pivoting.
///
/// # Errors
///
/// [`crate::Error::Rank`].
pub fn qr_col_piv<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>) -> Result<ColPivQr<T>> {
    let mut qrm = Matrix::from_view(a)?;
    let (m, n) = (qrm.rows(), qrm.cols());
    let k = m.min(n);
    let bs = block_size::<T>(m, n);
    let mut coeff = Matrix::zeros(bs, k);
    let (mut perm, mut perm_inv) = ((0..n).collect::<Vec<_>>(), (0..n).collect::<Vec<_>>());
    if k > 0 {
        let (mat, c) = (qrm.as_faer_mut(), coeff.as_faer_mut());
        let (p, pi) = (&mut perm[..], &mut perm_inv[..]);
        install(exec, flops::<T>(2.0 * (m * n * k) as f64), move |par| {
            let mut mem = MemBuffer::new(col_pivoting::factor::qr_in_place_scratch::<usize, T>(
                m,
                n,
                bs,
                par,
                Default::default(),
            ));
            let _ = col_pivoting::factor::qr_in_place(
                mat,
                c,
                p,
                pi,
                par,
                MemStack::new(&mut mem),
                Default::default(),
            );
        });
    }
    Ok(ColPivQr {
        h: Householder { qr: qrm, coeff },
        perm,
        perm_inv,
    })
}

impl<T: Scalar> ColPivQr<T> {
    /// Column permutation: column `j` of `A P` is column `perm()[j]` of `A`.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Inverse of [`ColPivQr::perm`].
    pub fn perm_inv(&self) -> &[usize] {
        &self.perm_inv
    }

    /// Upper-triangular `R` of `A P`.
    pub fn r(&self) -> Matrix<T> {
        self.h.r()
    }

    /// Numerical rank: the number of `|r_ii| > rtol * |r_00|`.
    pub fn rank(&self, rtol: f64) -> usize {
        let k = self.h.k();
        if k == 0 {
            return 0;
        }
        let r0 = abs(self.h.qr.get(0, 0));
        (0..k)
            .take_while(|&i| abs(self.h.qr.get(i, i)) > rtol * r0)
            .count()
    }

    /// `B = Qᴴ B` in place.
    ///
    /// # Errors
    ///
    /// As [`Qr::apply_q`].
    pub fn apply_qh(&self, exec: &Exec<'_>, b: &mut StridedViewMut<'_, T>) -> Result<()> {
        self.h.apply(exec, true, b)
    }
}

/// A least-squares solution and the numerical rank used.
#[derive(Clone, Debug)]
pub struct Lstsq<T> {
    /// `n x nrhs` solution.
    pub x: Matrix<T>,
    /// Numerical rank from the pivoted QR.
    pub rank: usize,
}

/// Least squares `min ||A X - B||` by column-pivoted QR: the basic solution
/// with rank `r = rank(rtol)` (`X` has `n - r` zero rows in pivoted order).
/// It is the unique minimizer for full column rank; for rank-deficient `A`
/// it minimizes the residual but is not the minimum-norm solution.
///
/// # Errors
///
/// [`crate::Error::Rank`], [`crate::Error::Shape`] when `B` has a different
/// row count.
pub fn lstsq<T: Scalar>(
    exec: &Exec<'_>,
    a: &StridedView<'_, T>,
    b: &StridedView<'_, T>,
    rtol: f64,
) -> Result<Lstsq<T>> {
    let f = qr_col_piv(exec, a)?;
    let (m, n) = (f.h.qr.rows(), f.h.qr.cols());
    let mut y = Matrix::from_view(b)?;
    if y.rows() != m {
        return Err(crate::Error::Shape(format!(
            "B has {} rows, A has {m}",
            y.rows()
        )));
    }
    let nrhs = y.cols();
    f.apply_qh(exec, &mut y.view_mut())?;
    let r = f.rank(rtol);
    let mut xp = Matrix::zeros(n, nrhs);
    for j in 0..nrhs {
        for i in 0..r {
            xp.set(i, j, y.get(i, j));
        }
    }
    if r > 0 && nrhs > 0 {
        let r11 = f.h.qr.as_faer().get(..r, ..r);
        let rhs = xp.as_faer_mut().get_mut(..r, ..);
        install(exec, flops::<T>((r * r * nrhs) as f64), move |par| {
            solve_upper_triangular_in_place(r11, rhs, par)
        });
    }
    // Undo the pivoting: row j of xp belongs to variable perm[j].
    let mut x = Matrix::zeros(n, nrhs);
    for j in 0..nrhs {
        for (i, &p) in f.perm.iter().enumerate() {
            x.set(p, j, xp.get(i, j));
        }
    }
    Ok(Lstsq { x, rank: r })
}
