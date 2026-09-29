//! Reference implementations and helpers shared by the tprims-blas tests.
#![allow(dead_code)]

use num_complex::Complex64;
use strided_view::StridedView;
use tensorcontract::Element;
use tprims_blas::{Conj, Scalar};

/// Dense column-major `alpha * op(A) * op(B) + beta * C`, where `op` is
/// conjugation, computed element by element from the views.
pub fn naive_gemm<T: Scalar>(
    alpha: T,
    a: &StridedView<'_, T>,
    ca: Conj,
    b: &StridedView<'_, T>,
    cb: Conj,
    beta: T,
    c: &StridedView<'_, T>,
) -> Vec<T> {
    let (m, k) = (a.dims()[0], a.dims()[1]);
    let n = b.dims()[1];
    let op = |x: T, c: Conj| if c == Conj::Yes { Element::conj(x) } else { x };
    let mut out = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            let mut acc = <T as Element>::zero();
            for l in 0..k {
                acc = Element::add(
                    acc,
                    Element::mul(op(a.get(&[i, l]), ca), op(b.get(&[l, j]), cb)),
                );
            }
            let base = if beta == <T as Element>::zero() {
                <T as Element>::zero()
            } else {
                Element::mul(beta, c.get(&[i, j]))
            };
            out.push(Element::add(Element::mul(alpha, acc), base));
        }
    }
    out
}

/// Column-major dense copy of a rank-2 view.
pub fn dense<T: Scalar>(v: &StridedView<'_, T>) -> Vec<T> {
    let (m, n) = (v.dims()[0], v.dims()[1]);
    (0..n)
        .flat_map(|j| (0..m).map(move |i| v.get(&[i, j])))
        .collect()
}

/// max |x - y| / max(1, max |y|).
pub fn max_rel_err<T: Scalar>(x: &[T], y: &[T]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mag = |z: T| -> f64 {
        let (re, im) = (Element::re(z), Element::im(z));
        let (re, im): (f64, f64) = (to_f64(re), to_f64(im));
        (re * re + im * im).sqrt()
    };
    let scale = y.iter().map(|&v| mag(v)).fold(1.0, f64::max);
    x.iter()
        .zip(y)
        .map(|(&a, &b)| mag(Element::add(a, Element::mul(b, from_f64::<T>(-1.0)))))
        .fold(0.0, f64::max)
        / scale
}

fn to_f64<R: tensorcontract::Real>(r: R) -> f64 {
    r.to_f64()
}

pub fn from_f64<T: Scalar>(x: f64) -> T {
    <T as Element>::from_parts(
        tensorcontract::Real::from_f64(x),
        tensorcontract::Real::from_f64(0.0),
    )
}

/// Deterministic test data: real parts from `seed`, imaginary parts for complex.
pub fn data<T: Scalar>(len: usize, seed: u64) -> Vec<T> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    (0..len)
        .map(|_| {
            let re = next();
            let im = next();
            <T as Element>::from_parts(
                tensorcontract::Real::from_f64(re),
                tensorcontract::Real::from_f64(im),
            )
        })
        .collect()
}

pub fn c64(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}
