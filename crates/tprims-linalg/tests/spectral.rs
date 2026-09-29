use num_complex::{Complex32, Complex64};
use tensorcontract::Element;
use tprims_blas::Scalar;
use tprims_exec::Exec;
use tprims_linalg::{eig, eigh, svd, EigScalar, Error, Matrix, Vectors};

mod common;
use common::*;

fn to_c<T: Scalar>(x: T) -> Complex64 {
    Complex64::new(
        tensorcontract::Real::to_f64(Element::re(x)),
        tensorcontract::Real::to_f64(Element::im(x)),
    )
}

fn svd_case<T: Scalar>(m: usize, n: usize) {
    let e = Exec::serial();
    let a = random::<T>(m, n, 31);
    let k = m.min(n);
    for vec in [Vectors::Thin, Vectors::Full] {
        let s = svd(&e, &a.view(), vec).unwrap();
        let (u, v) = (s.u.as_ref().unwrap(), s.v.as_ref().unwrap());
        let kk = u.cols().min(v.cols());
        let mut us = Matrix::<T>::zeros(m, v.cols()).unwrap();
        for j in 0..kk.min(k) {
            let sj = <T as Element>::from_parts(s.s[j], tensorcontract::Real::from_f64(0.0));
            for i in 0..m {
                us.set(i, j, Element::mul(u.get(i, j), sj));
            }
        }
        assert!(
            rel_err(&matmul(&us, &adjoint(v)), &a) < tol::<T>(m.max(n)) * 10.0,
            "U S Vᴴ != A {m}x{n} {vec:?}"
        );
        assert!(rel_err(&matmul(&adjoint(u), u), &identity(u.cols())) < tol::<T>(m.max(n)) * 10.0);
        for w in s.s.windows(2) {
            assert!(w[0] >= w[1], "singular values not non-increasing");
        }
    }
    let s = svd(&e, &a.view(), Vectors::None).unwrap();
    assert!(s.u.is_none() && s.v.is_none() && s.s.len() == k);
}

#[test]
fn svd_reconstructs() {
    for (m, n) in [(0, 0), (1, 1), (8, 5), (5, 8), (30, 30)] {
        svd_case::<f64>(m, n);
        svd_case::<Complex64>(m, n);
    }
}

#[test]
fn eigh_diagonalizes_hermitian_matrices_reading_the_lower_triangle() {
    let e = Exec::serial();
    for n in [0, 1, 6, 40] {
        let a = hpd::<Complex64>(n, 5);
        let mut g = a.clone();
        for j in 0..n {
            for i in 0..j {
                g.set(i, j, Complex64::new(1e3, 7.0));
            }
        }
        let d = eigh(&e, &g.view(), true).unwrap();
        let v = d.vectors.as_ref().unwrap();
        let mut vw = v.clone();
        for j in 0..n {
            for i in 0..n {
                vw.set(i, j, v.get(i, j) * d.values[j]);
            }
        }
        assert!(
            rel_err(&matmul(&a, v), &vw) < tol::<Complex64>(n) * 10.0,
            "A V != V W at n={n}"
        );
        for w in d.values.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }
}

fn eig_check<T: EigScalar>(a: &Matrix<T>) {
    let n = a.rows();
    let d = eig(&Exec::serial(), &a.view(), true).unwrap();
    let v = d.vectors.as_ref().unwrap();
    for j in 0..n {
        let lam = to_c(d.values[j]);
        let mut err = 0.0f64;
        let mut vnorm = 0.0f64;
        for i in 0..n {
            let mut av = Complex64::new(0.0, 0.0);
            for l in 0..n {
                av += to_c(a.get(i, l)) * to_c(v.get(l, j));
            }
            err = err.max((av - lam * to_c(v.get(i, j))).norm());
            vnorm = vnorm.max(to_c(v.get(i, j)).norm());
        }
        assert!(
            vnorm > 0.0 && err < 1e-9 * vnorm.max(1.0) * (n as f64),
            "A v != λ v for eigenpair {j}: {err}"
        );
    }
}

#[test]
fn eig_real_with_complex_pairs_and_complex_input() {
    // Rotation block with eigenvalues 1 ± 2i, plus a real eigenvalue 3.
    let a =
        Matrix::<f64>::from_col_major(3, 3, vec![1.0, -2.0, 0.0, 2.0, 1.0, 0.0, 0.5, 0.25, 3.0])
            .unwrap();
    eig_check(&a);
    let d = eig(&Exec::serial(), &a.view(), false).unwrap();
    let mut ims: Vec<f64> = d.values.iter().map(|z| z.im).collect();
    ims.sort_by(f64::total_cmp);
    assert!((ims[0] + 2.0).abs() < 1e-12 && ims[1].abs() < 1e-12 && (ims[2] - 2.0).abs() < 1e-12);
    eig_check(&random::<f64>(20, 20, 3));
    eig_check(&random::<Complex64>(15, 15, 4));
    let small = random::<Complex32>(6, 6, 2);
    let _ = eig(&Exec::serial(), &small.view(), true).unwrap();
}

#[test]
fn non_finite_input_is_a_typed_error() {
    let mut a = identity::<f64>(3);
    a.set(1, 2, f64::NAN);
    let e = Exec::serial();
    assert_eq!(
        svd(&e, &a.view(), Vectors::Thin).unwrap_err(),
        Error::NonFinite
    );
    // eigh reads only the lower triangle: NaN above the diagonal is ignored.
    assert!(eigh(&e, &a.view(), true).is_ok());
    let mut l = identity::<f64>(3);
    l.set(2, 1, f64::NAN);
    assert_eq!(eigh(&e, &l.view(), true).unwrap_err(), Error::NonFinite);
    assert_eq!(eig(&e, &a.view(), true).unwrap_err(), Error::NonFinite);
}
