//! Shared helpers: deterministic data, products and error norms on `Matrix`.
#![allow(dead_code)]

use tensorcontract::Element;
use tprims_blas::Scalar;
use tprims_linalg::Matrix;

pub fn from_f64<T: Scalar>(x: f64) -> T {
    <T as Element>::from_parts(
        tensorcontract::Real::from_f64(x),
        tensorcontract::Real::from_f64(0.0),
    )
}

pub fn cplx<T: Scalar>(re: f64, im: f64) -> T {
    <T as Element>::from_parts(
        tensorcontract::Real::from_f64(re),
        tensorcontract::Real::from_f64(im),
    )
}

pub fn random<T: Scalar>(rows: usize, cols: usize, seed: u64) -> Matrix<T> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % 10_000) as f64 / 10_000.0 - 0.5
    };
    let data = (0..rows * cols)
        .map(|_| cplx::<T>(next(), next()))
        .collect();
    Matrix::from_col_major(rows, cols, data).unwrap()
}

pub fn matmul<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    assert_eq!(a.cols(), b.rows());
    let mut out = Matrix::zeros(a.rows(), b.cols()).unwrap();
    for j in 0..b.cols() {
        for i in 0..a.rows() {
            let mut acc = <T as Element>::zero();
            for l in 0..a.cols() {
                acc = Element::add(acc, Element::mul(a.get(i, l), b.get(l, j)));
            }
            out.set(i, j, acc);
        }
    }
    out
}

pub fn adjoint<T: Scalar>(a: &Matrix<T>) -> Matrix<T> {
    let mut out = Matrix::zeros(a.cols(), a.rows()).unwrap();
    for j in 0..a.cols() {
        for i in 0..a.rows() {
            out.set(j, i, Element::conj(a.get(i, j)));
        }
    }
    out
}

pub fn abs<T: Scalar>(x: T) -> f64 {
    let (re, im): (f64, f64) = (
        tensorcontract::Real::to_f64(Element::re(x)),
        tensorcontract::Real::to_f64(Element::im(x)),
    );
    (re * re + im * im).sqrt()
}

/// max |a - b| / max(1, max |b|)
pub fn rel_err<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> f64 {
    assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
    let scale = b.data().iter().map(|&x| abs(x)).fold(1.0, f64::max);
    a.data()
        .iter()
        .zip(b.data())
        .map(|(&x, &y)| abs(Element::add(x, Element::mul(y, from_f64(-1.0)))))
        .fold(0.0, f64::max)
        / scale
}

/// Hermitian positive definite: B Bᴴ + n I.
pub fn hpd<T: Scalar>(n: usize, seed: u64) -> Matrix<T> {
    let b = random::<T>(n, n, seed);
    let mut a = matmul(&b, &adjoint(&b));
    for i in 0..n {
        a.set(i, i, Element::add(a.get(i, i), from_f64(n as f64)));
    }
    a
}

pub fn identity<T: Scalar>(n: usize) -> Matrix<T> {
    let mut a = Matrix::zeros(n, n).unwrap();
    for i in 0..n {
        a.set(i, i, <T as Element>::one());
    }
    a
}

pub fn tol<T: Scalar>(n: usize) -> f64 {
    let eps = if std::mem::size_of::<<T as Scalar>::Re>() == 4 {
        1.2e-7
    } else {
        2.3e-16
    };
    eps * 200.0 * (n.max(1) as f64)
}
