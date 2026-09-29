use num_complex::Complex64;
use tprims_blas::Scalar;
use tprims_exec::Exec;
use tprims_linalg::{lstsq, qr, qr_col_piv, Matrix};

mod common;
use common::*;

fn qr_case<T: Scalar>(m: usize, n: usize) {
    let e = Exec::serial();
    let a = random::<T>(m, n, 21);
    let f = qr(&e, &a.view()).unwrap();
    let (q, r) = (f.q_thin(&e).unwrap(), f.r());
    let k = m.min(n);
    assert_eq!((q.rows(), q.cols(), r.rows(), r.cols()), (m, k, k, n));
    assert!(
        rel_err(&matmul(&q, &r), &a) < tol::<T>(m.max(n)) * 10.0,
        "QR != A {m}x{n}"
    );
    assert!(
        rel_err(&matmul(&adjoint(&q), &q), &identity(k)) < tol::<T>(m.max(n)) * 10.0,
        "QᴴQ != I"
    );
    let qf = f.q_full(&e).unwrap();
    assert!(
        rel_err(&matmul(&adjoint(&qf), &qf), &identity(m)) < tol::<T>(m.max(n)) * 10.0,
        "full Q not unitary"
    );
    let b = random::<T>(m, 3, 4);
    let mut y = b.clone();
    f.apply_qh(&e, &mut y.view_mut()).unwrap();
    f.apply_q(&e, &mut y.view_mut()).unwrap();
    assert!(rel_err(&y, &b) < tol::<T>(m) * 10.0, "Q Qᴴ b != b");
}

#[test]
fn qr_reconstructs_tall_wide_square_and_empty() {
    for (m, n) in [(0, 0), (1, 1), (9, 4), (4, 9), (16, 16), (70, 50)] {
        qr_case::<f64>(m, n);
        qr_case::<Complex64>(m, n);
    }
}

#[test]
fn lstsq_full_rank_satisfies_the_normal_equations() {
    let e = Exec::serial();
    for (m, n) in [(12, 5), (30, 30)] {
        let a = random::<Complex64>(m, n, 2);
        let b = random::<Complex64>(m, 2, 3);
        let s = lstsq(&e, &a.view(), &b.view(), 1e-12).unwrap();
        assert_eq!(s.rank, n);
        let res = {
            let ax = matmul(&a, &s.x);
            let mut r = ax.clone();
            for j in 0..r.cols() {
                for i in 0..r.rows() {
                    r.set(i, j, ax.get(i, j) - b.get(i, j));
                }
            }
            r
        };
        let g = matmul(&adjoint(&a), &res);
        assert!(g.data().iter().all(|z| z.norm() < 1e-10), "Aᴴ(Ax-b) != 0");
    }
}

#[test]
fn lstsq_detects_rank_and_solves_consistent_systems() {
    let e = Exec::serial();
    // rank-2 6x4
    let a = matmul(&random::<f64>(6, 2, 7), &random::<f64>(2, 4, 8));
    let x0 = random::<f64>(4, 1, 9);
    let b = matmul(&a, &x0);
    let s = lstsq(&e, &a.view(), &b.view(), 1e-10).unwrap();
    assert_eq!(s.rank, 2);
    assert!(
        rel_err(&matmul(&a, &s.x), &b) < 1e-10,
        "consistent rank-deficient system"
    );
    // wide 3x5 full row rank
    let a = random::<f64>(3, 5, 10);
    let b = random::<f64>(3, 2, 11);
    let s = lstsq(&e, &a.view(), &b.view(), 1e-12).unwrap();
    assert_eq!(s.rank, 3);
    assert!(rel_err(&matmul(&a, &s.x), &b) < 1e-10, "wide system");
    let f = qr_col_piv(&e, &a.view()).unwrap();
    assert_eq!(f.rank(1e-12), 3);
    let mut p = f.perm().to_vec();
    p.sort();
    assert_eq!(p, (0..5).collect::<Vec<_>>());
    let _ = Matrix::<f64>::zeros(0, 0).unwrap();
}
