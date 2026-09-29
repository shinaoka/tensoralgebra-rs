use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self, ComputeEigenvectors};
use faer::linalg::svd::{self as fsvd, ComputeSvdVectors};
use faer::MatRef;
use num_complex::Complex;
use strided_view::StridedView;
use tensorcontract::Element;
use tprims_blas::Scalar;
use tprims_exec::Exec;

use crate::util::{abs, flops, install, square};
use crate::{Error, Matrix, Result};

/// Which singular vectors to compute.
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_linalg::Vectors::Thin, tprims_linalg::Vectors::Full);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vectors {
    /// Values only.
    None,
    /// `U` is `m x min(m, n)`, `V` is `n x min(m, n)`.
    Thin,
    /// `U` is `m x m`, `V` is `n x n`.
    Full,
}

/// `A = U diag(s) Vᴴ`; `s` real and non-increasing; `V` (not `Vᴴ`).
#[derive(Clone, Debug)]
pub struct Svd<T: Scalar> {
    /// Singular values, non-increasing.
    pub s: Vec<T::Re>,
    /// Left singular vectors, when requested.
    pub u: Option<Matrix<T>>,
    /// Right singular vectors (columns of `V`), when requested.
    pub v: Option<Matrix<T>>,
}

fn check_finite<T: Scalar>(a: &Matrix<T>) -> Result<()> {
    if a.data().iter().all(|&x| abs(x).is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite)
    }
}

/// Singular value decomposition.
///
/// # Errors
///
/// [`Error::Rank`], [`Error::NonFinite`], [`Error::NoConvergence`].
pub fn svd<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>, vectors: Vectors) -> Result<Svd<T>> {
    let a = Matrix::from_view(a)?;
    check_finite(&a)?;
    let (m, n) = (a.rows(), a.cols());
    let k = m.min(n);
    let (cu, cv, ucols, vcols) = match vectors {
        Vectors::None => (ComputeSvdVectors::No, ComputeSvdVectors::No, 0, 0),
        Vectors::Thin => (ComputeSvdVectors::Thin, ComputeSvdVectors::Thin, k, k),
        Vectors::Full => (ComputeSvdVectors::Full, ComputeSvdVectors::Full, m, n),
    };
    let mut s = vec![<T as Element>::zero(); k];
    let mut u = (vectors != Vectors::None).then(|| Matrix::<T>::zeros(m, ucols));
    let mut v = (vectors != Vectors::None).then(|| Matrix::<T>::zeros(n, vcols));
    if m > 0 && n > 0 {
        let am = a.as_faer();
        let sd = faer::diag::DiagMut::from_slice_mut(&mut s);
        let (um, vm) = (
            u.as_mut().map(Matrix::as_faer_mut),
            v.as_mut().map(Matrix::as_faer_mut),
        );
        install(exec, flops::<T>(12.0 * (m * n * k) as f64), move |par| {
            let mut mem = MemBuffer::new(fsvd::svd_scratch::<T>(
                m,
                n,
                cu,
                cv,
                par,
                Default::default(),
            ));
            fsvd::svd(
                am,
                sd,
                um,
                vm,
                par,
                MemStack::new(&mut mem),
                Default::default(),
            )
        })
        .map_err(|_| Error::NoConvergence)?;
    } else if vectors == Vectors::Full {
        // Empty A: full factors are identities.
        for (mat, d) in [(u.as_mut(), m), (v.as_mut(), n)] {
            if let Some(mat) = mat {
                for i in 0..d {
                    mat.set(i, i, <T as Element>::one());
                }
            }
        }
    }
    Ok(Svd {
        s: s.into_iter().map(Element::re).collect(),
        u,
        v,
    })
}

/// Hermitian eigendecomposition `A = V diag(w) Vᴴ`.
#[derive(Clone, Debug)]
pub struct Eigh<T: Scalar> {
    /// Eigenvalues, non-decreasing.
    pub values: Vec<T::Re>,
    /// Eigenvectors as columns, when requested.
    pub vectors: Option<Matrix<T>>,
}

/// Eigendecomposition of a Hermitian matrix, reading the lower triangle.
///
/// # Errors
///
/// [`Error::NotSquare`], [`Error::Rank`], [`Error::NonFinite`] (NaN or
/// infinity in the lower triangle), [`Error::NoConvergence`].
pub fn eigh<T: Scalar>(exec: &Exec<'_>, a: &StridedView<'_, T>, vectors: bool) -> Result<Eigh<T>> {
    let a = Matrix::from_view(a)?;
    let n = square(a.rows(), a.cols())?;
    for j in 0..n {
        for i in j..n {
            if !abs(a.get(i, j)).is_finite() {
                return Err(Error::NonFinite);
            }
        }
    }
    let mut w = vec![<T as Element>::zero(); n];
    let mut v = vectors.then(|| Matrix::<T>::zeros(n, n));
    if n > 0 {
        let am = a.as_faer();
        let wd = faer::diag::DiagMut::from_slice_mut(&mut w);
        let vm = v.as_mut().map(Matrix::as_faer_mut);
        let cv = if vectors {
            ComputeEigenvectors::Yes
        } else {
            ComputeEigenvectors::No
        };
        install(exec, flops::<T>(9.0 * (n * n * n) as f64), move |par| {
            let mut mem = MemBuffer::new(evd::self_adjoint_evd_scratch::<T>(
                n,
                cv,
                par,
                Default::default(),
            ));
            evd::self_adjoint_evd(am, wd, vm, par, MemStack::new(&mut mem), Default::default())
        })
        .map_err(|_| Error::NoConvergence)?;
    }
    Ok(Eigh {
        values: w.into_iter().map(Element::re).collect(),
        vectors: v,
    })
}

/// Eigenvalues and optional eigenvectors, as returned by [`EigScalar`].
#[doc(hidden)]
pub type EigParts<C> = (Vec<C>, Option<Matrix<C>>);

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// Scalars supporting the nonsymmetric eigendecomposition, whose results
/// are complex ([`EigScalar::C`]).
///
/// # Examples
///
/// ```
/// fn complex_of<T: tprims_linalg::EigScalar>() -> &'static str { std::any::type_name::<T::C>() }
/// assert!(complex_of::<f64>().contains("Complex<f64>"));
/// ```
pub trait EigScalar: Scalar + sealed::Sealed {
    /// The complex type of the eigenvalues and eigenvectors.
    type C: Scalar<Re = Self::Re>;
    #[doc(hidden)]
    fn eig_impl(a: MatRef<'_, Self>, vectors: bool, par: faer::Par) -> Result<EigParts<Self::C>>;
}

macro_rules! eig_impls {
    ($r:ty) => {
        impl EigScalar for $r {
            type C = Complex<$r>;
            fn eig_impl(
                a: MatRef<'_, Self>,
                vectors: bool,
                par: faer::Par,
            ) -> Result<EigParts<Self::C>> {
                let n = a.nrows();
                let (mut re, mut im) = (vec![0.0 as $r; n], vec![0.0 as $r; n]);
                let mut u = vectors.then(|| Matrix::<$r>::zeros(n, n));
                let cv = if vectors {
                    ComputeEigenvectors::Yes
                } else {
                    ComputeEigenvectors::No
                };
                {
                    let mut mem = MemBuffer::new(evd::evd_scratch::<$r>(
                        n,
                        ComputeEigenvectors::No,
                        cv,
                        par,
                        Default::default(),
                    ));
                    evd::evd_real(
                        a,
                        faer::diag::DiagMut::from_slice_mut(&mut re),
                        faer::diag::DiagMut::from_slice_mut(&mut im),
                        None,
                        u.as_mut().map(Matrix::as_faer_mut),
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| Error::NoConvergence)?;
                }
                let values: Vec<Complex<$r>> = (0..n).map(|j| Complex::new(re[j], im[j])).collect();
                // LAPACK-packed pairs: columns j, j+1 hold the real and
                // imaginary parts of the eigenvector of the eigenvalue with
                // positive imaginary part; the conjugate follows. A value is
                // real when |im| <= eps * max(|re|, 1). (tprims reimplementation
                // of faer's private real-to-complex conversion.)
                let vecs = u.map(|u| {
                    let e = <$r>::EPSILON;
                    let mut v = Matrix::<Complex<$r>>::zeros(n, n);
                    let mut j = 0;
                    while j < n {
                        if im[j].abs() <= e * re[j].abs().max(1.0) || j + 1 == n {
                            for row in 0..n {
                                v.set(row, j, Complex::new(u.get(row, j), 0.0));
                            }
                            j += 1;
                        } else {
                            let sign: $r = if im[j] > 0.0 { 1.0 } else { -1.0 };
                            for row in 0..n {
                                let (x, y) = (u.get(row, j), sign * u.get(row, j + 1));
                                v.set(row, j, Complex::new(x, y));
                                v.set(row, j + 1, Complex::new(x, -y));
                            }
                            j += 2;
                        }
                    }
                    v
                });
                Ok((values, vecs))
            }
        }

        impl EigScalar for Complex<$r> {
            type C = Complex<$r>;
            fn eig_impl(
                a: MatRef<'_, Self>,
                vectors: bool,
                par: faer::Par,
            ) -> Result<EigParts<Self::C>> {
                let n = a.nrows();
                let mut w = vec![Complex::<$r>::new(0.0, 0.0); n];
                let mut u = vectors.then(|| Matrix::<Complex<$r>>::zeros(n, n));
                let cv = if vectors {
                    ComputeEigenvectors::Yes
                } else {
                    ComputeEigenvectors::No
                };
                let mut mem = MemBuffer::new(evd::evd_scratch::<Complex<$r>>(
                    n,
                    ComputeEigenvectors::No,
                    cv,
                    par,
                    Default::default(),
                ));
                evd::evd_cplx(
                    a,
                    faer::diag::DiagMut::from_slice_mut(&mut w),
                    None,
                    u.as_mut().map(Matrix::as_faer_mut),
                    par,
                    MemStack::new(&mut mem),
                    Default::default(),
                )
                .map_err(|_| Error::NoConvergence)?;
                Ok((w, u))
            }
        }
    };
}
eig_impls!(f32);
eig_impls!(f64);

/// Nonsymmetric eigendecomposition `A V = V diag(w)` (right eigenvectors).
#[derive(Clone, Debug)]
pub struct Eig<C> {
    /// Eigenvalues (complex).
    pub values: Vec<C>,
    /// Right eigenvectors as columns (complex), when requested; faer's
    /// normalization.
    pub vectors: Option<Matrix<C>>,
}

/// Eigenvalues and right eigenvectors of a general square matrix.
///
/// # Errors
///
/// [`Error::NotSquare`], [`Error::Rank`], [`Error::NonFinite`],
/// [`Error::NoConvergence`].
pub fn eig<T: EigScalar>(
    exec: &Exec<'_>,
    a: &StridedView<'_, T>,
    vectors: bool,
) -> Result<Eig<T::C>> {
    let a = Matrix::from_view(a)?;
    let n = square(a.rows(), a.cols())?;
    check_finite(&a)?;
    if n == 0 {
        return Ok(Eig {
            values: vec![],
            vectors: vectors.then(|| Matrix::zeros(0, 0)),
        });
    }
    let am = a.as_faer();
    let (values, vectors) = install(exec, flops::<T>(25.0 * (n * n * n) as f64), move |par| {
        T::eig_impl(am, vectors, par)
    })?;
    Ok(Eig { values, vectors })
}
