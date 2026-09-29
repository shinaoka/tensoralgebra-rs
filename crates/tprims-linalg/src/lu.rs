use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::{full_pivoting, partial_pivoting};
use faer::perm::PermRef;
use faer::Conj;
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::{Element, Real};
use tprims_blas::{Op, Scalar};
use tprims_exec::Exec;

use crate::util::{abs, eps, factor_policy, flops, install, install_with, rhs_mut, square};
use crate::{Error, Matrix, Result};

/// `P A = L U` with partial (row) pivoting; `A` may be rectangular.
#[derive(Clone, Debug)]
pub struct Lu<T> {
    lu: Matrix<T>,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    transpositions: usize,
}

/// Factor with partial pivoting. Never fails on singular input.
///
/// # Errors
///
/// [`Error::Rank`].
pub fn lu<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>) -> Result<Lu<T>> {
    let mut lu = Matrix::from_view(a)?;
    let (m, n) = (lu.rows(), lu.cols());
    let (mut perm, mut perm_inv) = ((0..m).collect::<Vec<_>>(), (0..m).collect::<Vec<_>>());
    let mut transpositions = 0;
    if m > 0 && n > 0 {
        let mat = lu.as_faer_mut();
        let (p, pi) = (&mut perm[..], &mut perm_inv[..]);
        let k = m.min(n);
        transpositions = install_with(
            exec,
            flops::<T>((m * n * k) as f64),
            factor_policy(),
            move |par| {
                let mut mem = MemBuffer::new(partial_pivoting::factor::lu_in_place_scratch::<
                    usize,
                    T,
                >(m, n, par, Default::default()));
                partial_pivoting::factor::lu_in_place(
                    mat,
                    p,
                    pi,
                    par,
                    MemStack::new(&mut mem),
                    Default::default(),
                )
                .0
                .transposition_count
            },
        );
    }
    Ok(Lu {
        lu,
        perm,
        perm_inv,
        transpositions,
    })
}

/// Index of the first pivot with `|u_ii| <= n eps max_j |u_jj|`.
pub(crate) fn singular_pivot<T: Scalar>(lu: &Matrix<T>) -> Option<usize> {
    let k = lu.rows().min(lu.cols());
    let max = (0..k).map(|i| abs(lu.get(i, i))).fold(0.0, f64::max);
    let thr = k as f64 * eps::<T>() * max;
    (0..k).find(|&i| {
        let v = abs(lu.get(i, i));
        v <= thr || !v.is_finite()
    })
}

/// The first pivot, in elimination order, that is exactly zero or
/// non-finite. An exact zero leaves 0/0 in later pivots, so only the first
/// bad pivot tells whether the matrix is singular (zero) or the input was
/// non-finite.
enum BadPivot {
    Zero,
    NonFinite,
}

fn first_bad_pivot<T: Scalar>(lu: &Matrix<T>) -> Option<BadPivot> {
    (0..lu.rows().min(lu.cols())).find_map(|i| {
        let v = abs(lu.get(i, i));
        if v == 0.0 {
            Some(BadPivot::Zero)
        } else if !v.is_finite() {
            Some(BadPivot::NonFinite)
        } else {
            None
        }
    })
}

fn op_conj(op: Op) -> Conj {
    if op == Op::C {
        Conj::Yes
    } else {
        Conj::No
    }
}

impl<T: Scalar> Lu<T> {
    /// Row permutation: row `i` of `P A` is row `p()[i]` of `A`.
    pub fn p(&self) -> &[usize] {
        &self.perm
    }

    /// Unit lower-triangular factor, `m x min(m, n)`.
    pub fn l(&self) -> Matrix<T> {
        let (m, n) = (self.lu.rows(), self.lu.cols());
        let k = m.min(n);
        // INVARIANT: m x min(m, n) is no larger than the stored m x n factor.
        let mut l = Matrix::zeros_validated(m, k);
        for j in 0..k {
            l.set(j, j, <T as Element>::one());
            for i in j + 1..m {
                l.set(i, j, self.lu.get(i, j));
            }
        }
        l
    }

    /// Upper-triangular factor, `min(m, n) x n`.
    pub fn u(&self) -> Matrix<T> {
        let (m, n) = (self.lu.rows(), self.lu.cols());
        let k = m.min(n);
        // INVARIANT: min(m, n) x n is no larger than the stored m x n factor.
        let mut u = Matrix::zeros_validated(k, n);
        for j in 0..n {
            for i in 0..=j.min(k.saturating_sub(1)) {
                if i < k {
                    u.set(i, j, self.lu.get(i, j));
                }
            }
        }
        u
    }

    /// Solve `op(A) X = B` in place in `b` (square `A` only).
    ///
    /// # Errors
    ///
    /// [`Error::NotSquare`], [`Error::Singular`] (tested before any write),
    /// [`Error::Rank`], [`Error::Shape`], [`Error::AliasedOutput`].
    pub fn solve(&self, exec: &Exec<'_>, op: Op, b: &mut StridedViewMut<'_, T>) -> Result<()> {
        let n = square(self.lu.rows(), self.lu.cols())?;
        let rhs = rhs_mut(b, n)?;
        if let Some(index) = singular_pivot(&self.lu) {
            return Err(Error::Singular { index });
        }
        let k = rhs.ncols();
        if n == 0 || k == 0 {
            return Ok(());
        }
        let lu = self.lu.as_faer();
        // INVARIANT: perm/perm_inv come from faer's factorization of this matrix.
        let perm = PermRef::new_checked(&self.perm, &self.perm_inv, n);
        install(exec, flops::<T>(2.0 * (n * n * k) as f64), move |par| {
            let mut mem = MemBuffer::new(
                partial_pivoting::solve::solve_in_place_scratch::<usize, T>(n, k, par),
            );
            let stack = MemStack::new(&mut mem);
            match op {
                Op::N => partial_pivoting::solve::solve_in_place_with_conj(
                    lu,
                    lu,
                    perm,
                    Conj::No,
                    rhs,
                    par,
                    stack,
                ),
                Op::T | Op::C => partial_pivoting::solve::solve_transpose_in_place_with_conj(
                    lu,
                    lu,
                    perm,
                    op_conj(op),
                    rhs,
                    par,
                    stack,
                ),
            }
        });
        Ok(())
    }

    /// Determinant (square only; zero for singular).
    ///
    /// # Errors
    ///
    /// [`Error::NotSquare`].
    pub fn det(&self) -> Result<T> {
        let n = square(self.lu.rows(), self.lu.cols())?;
        match first_bad_pivot(&self.lu) {
            Some(BadPivot::Zero) => return Ok(<T as Element>::zero()),
            Some(BadPivot::NonFinite) => return Ok(from_f64(f64::NAN)),
            None => {}
        }
        let mut d = <T as Element>::one();
        for i in 0..n {
            d = Element::mul(d, self.lu.get(i, i));
        }
        Ok(if self.transpositions % 2 == 1 {
            Element::mul(d, from_f64(-1.0))
        } else {
            d
        })
    }

    /// `(sign, log|det|)` with `det = sign * exp(log|det|)`; `sign` is a unit
    /// scalar (±1 for real), `(0, -inf)` for singular input.
    ///
    /// # Errors
    ///
    /// [`Error::NotSquare`].
    pub fn logdet(&self) -> Result<(T, T::Re)> {
        let n = square(self.lu.rows(), self.lu.cols())?;
        match first_bad_pivot(&self.lu) {
            Some(BadPivot::Zero) => {
                return Ok((<T as Element>::zero(), Real::from_f64(f64::NEG_INFINITY)))
            }
            Some(BadPivot::NonFinite) => return Ok((from_f64(f64::NAN), Real::from_f64(f64::NAN))),
            None => {}
        }
        let mut sign = if self.transpositions % 2 == 1 {
            from_f64::<T>(-1.0)
        } else {
            <T as Element>::one()
        };
        let mut logabs = 0.0f64;
        for i in 0..n {
            let u = self.lu.get(i, i);
            let a = abs(u);
            if a == 0.0 {
                return Ok((<T as Element>::zero(), Real::from_f64(f64::NEG_INFINITY)));
            }
            logabs += a.ln();
            sign = Element::mul(sign, Element::mul(u, from_f64(1.0 / a)));
        }
        Ok((sign, Real::from_f64(logabs)))
    }
}

fn from_f64<T: Scalar>(x: f64) -> T {
    <T as Element>::from_parts(Real::from_f64(x), Real::from_f64(0.0))
}

/// `P A Qᵀ = L U` with full pivoting (square `A`).
#[derive(Clone, Debug)]
pub struct FullPivLu<T> {
    lu: Matrix<T>,
    row: (Vec<usize>, Vec<usize>),
    col: (Vec<usize>, Vec<usize>),
}

/// Factor with full pivoting.
///
/// # Errors
///
/// [`Error::Rank`], [`Error::NotSquare`].
pub fn lu_full<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>) -> Result<FullPivLu<T>> {
    let mut lu = Matrix::from_view(a)?;
    let n = square(lu.rows(), lu.cols())?;
    let id = || (0..n).collect::<Vec<_>>();
    let (mut rp, mut rpi, mut cp, mut cpi) = (id(), id(), id(), id());
    if n > 0 {
        let mat = lu.as_faer_mut();
        let (a1, a2, a3, a4) = (&mut rp[..], &mut rpi[..], &mut cp[..], &mut cpi[..]);
        install_with(
            exec,
            flops::<T>((n * n * n) as f64),
            factor_policy(),
            move |par| {
                let mut mem =
                    MemBuffer::new(full_pivoting::factor::lu_in_place_scratch::<usize, T>(
                        n,
                        n,
                        par,
                        Default::default(),
                    ));
                let _ = full_pivoting::factor::lu_in_place(
                    mat,
                    a1,
                    a2,
                    a3,
                    a4,
                    par,
                    MemStack::new(&mut mem),
                    Default::default(),
                );
            },
        );
    }
    Ok(FullPivLu {
        lu,
        row: (rp, rpi),
        col: (cp, cpi),
    })
}

impl<T: Scalar> FullPivLu<T> {
    /// Solve `op(A) X = B` in place in `b`.
    ///
    /// # Errors
    ///
    /// [`Error::Singular`] (tested before any write), [`Error::Rank`],
    /// [`Error::Shape`], [`Error::AliasedOutput`].
    pub fn solve(&self, exec: &Exec<'_>, op: Op, b: &mut StridedViewMut<'_, T>) -> Result<()> {
        let n = self.lu.rows();
        let rhs = rhs_mut(b, n)?;
        if let Some(index) = singular_pivot(&self.lu) {
            return Err(Error::Singular { index });
        }
        let k = rhs.ncols();
        if n == 0 || k == 0 {
            return Ok(());
        }
        let lu = self.lu.as_faer();
        // INVARIANT: permutations come from faer's factorization of this matrix.
        let rp = PermRef::new_checked(&self.row.0, &self.row.1, n);
        let cp = PermRef::new_checked(&self.col.0, &self.col.1, n);
        install(exec, flops::<T>(2.0 * (n * n * k) as f64), move |par| {
            let mut mem = MemBuffer::new(full_pivoting::solve::solve_in_place_scratch::<usize, T>(
                n, k, par,
            ));
            let stack = MemStack::new(&mut mem);
            match op {
                Op::N => full_pivoting::solve::solve_in_place_with_conj(
                    lu,
                    lu,
                    rp,
                    cp,
                    Conj::No,
                    rhs,
                    par,
                    stack,
                ),
                Op::T | Op::C => full_pivoting::solve::solve_transpose_in_place_with_conj(
                    lu,
                    lu,
                    rp,
                    cp,
                    op_conj(op),
                    rhs,
                    par,
                    stack,
                ),
            }
        });
        Ok(())
    }
}

/// Solve `A X = B` in place in `b` via partial-pivot LU.
///
/// # Errors
///
/// As [`Lu::solve`].
pub fn solve<T: Scalar>(
    exec: &Exec<'_>,
    a: &StridedView<'_, T>,
    b: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    lu(exec, a)?.solve(exec, Op::N, b)
}

/// Inverse of a square matrix.
///
/// # Errors
///
/// [`Error::NotSquare`], [`Error::Singular`].
pub fn inv<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>) -> Result<Matrix<T>> {
    let f = lu(exec, a)?;
    let n = square(f.lu.rows(), f.lu.cols())?;
    let mut x = Matrix::zeros(n, n)?;
    for i in 0..n {
        x.set(i, i, <T as Element>::one());
    }
    f.solve(exec, Op::N, &mut x.view_mut())?;
    Ok(x)
}

/// Determinant.
///
/// # Errors
///
/// [`Error::NotSquare`].
pub fn det<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>) -> Result<T> {
    lu(exec, a)?.det()
}

/// `(sign, log|det|)`; see [`Lu::logdet`].
///
/// # Errors
///
/// [`Error::NotSquare`].
pub fn logdet<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>) -> Result<(T, T::Re)> {
    lu(exec, a)?.logdet()
}
