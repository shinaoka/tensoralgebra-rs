use num_complex::Complex64;
use tensorcontract::Element;
use tprims_blas::Scalar;
use tprims_exec::{Exec, Pool};
use tprims_linalg::{cholesky, ldlt, Error, Matrix};

mod common;
use common::*;

fn chol_case<T: Scalar>(exec: &Exec<'_>, n: usize) {
    let mut a = hpd::<T>(n, 3);
    let clean = a.clone();
    // Garbage in the strict upper triangle must be ignored.
    for j in 0..n {
        for i in 0..j {
            a.set(i, j, from_f64(1e3));
        }
    }
    let f = cholesky(exec, &a.view()).unwrap();
    let l = f.l();
    for j in 0..n {
        for i in 0..j {
            assert_eq!(l.get(i, j), <T as Element>::zero(), "L upper not cleared");
        }
    }
    assert!(
        rel_err(&matmul(l, &adjoint(l)), &clean) < tol::<T>(n),
        "L Lᴴ != A at n={n}"
    );
    let x = random::<T>(n, 3, 9);
    let mut b = matmul(&clean, &x);
    f.solve(exec, &mut b.view_mut()).unwrap();
    assert!(rel_err(&b, &x) < tol::<T>(n) * 10.0, "solve n={n}");
}

#[test]
fn cholesky_reconstructs_and_solves() {
    for n in [0, 1, 2, 7, 40] {
        chol_case::<f64>(&Exec::serial(), n);
        chol_case::<Complex64>(&Exec::serial(), n);
        chol_case::<f32>(&Exec::serial(), n);
    }
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    chol_case::<f64>(&Exec::rayon(&pool), 400);
}

#[test]
fn cholesky_reports_the_failing_pivot() {
    let mut a = identity::<f64>(4);
    a.set(2, 2, -1.0);
    assert_eq!(
        cholesky(&Exec::serial(), &a.view()).unwrap_err(),
        Error::NotPositiveDefinite { index: 2 }
    );
    let r = Matrix::<f64>::zeros(3, 2);
    assert_eq!(
        cholesky(&Exec::serial(), &r.view()).unwrap_err(),
        Error::NotSquare { rows: 3, cols: 2 }
    );
}

fn ldlt_case<T: Scalar>(n: usize) {
    // Hermitian indefinite: A = H with diagonal of alternating sign.
    let b = random::<T>(n, n, 5);
    let mut a = Matrix::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            let v = Element::add(b.get(i, j), Element::conj(b.get(j, i)));
            a.set(i, j, v);
        }
        let d = if j % 2 == 0 { 3.0 } else { -3.0 };
        a.set(j, j, from_f64(d));
    }
    let f = ldlt(&Exec::serial(), &a.view()).unwrap();
    let x = random::<T>(n, 2, 8);
    let mut rhs = matmul(&a, &x);
    f.solve(&Exec::serial(), &mut rhs.view_mut()).unwrap();
    assert!(rel_err(&rhs, &x) < tol::<T>(n) * 100.0, "ldlt n={n}");
}

#[test]
fn ldlt_solves_indefinite_hermitian() {
    for n in [0, 1, 2, 9, 50] {
        ldlt_case::<f64>(n);
        ldlt_case::<Complex64>(n);
    }
}
