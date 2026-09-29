use num_complex::Complex64;
use tensorcontract::Element;
use tprims_blas::{Op, Scalar};
use tprims_exec::{Exec, Pool};
use tprims_linalg::{det, inv, logdet, lu, lu_full, solve, Error, Matrix};

mod common;
use common::*;

fn permute_rows<T: Scalar>(a: &Matrix<T>, perm: &[usize]) -> Matrix<T> {
    let mut out = Matrix::zeros(a.rows(), a.cols()).unwrap();
    for (i, &p) in perm.iter().enumerate() {
        for j in 0..a.cols() {
            out.set(i, j, a.get(p, j));
        }
    }
    out
}

fn op_mat<T: Scalar>(a: &Matrix<T>, op: Op) -> Matrix<T> {
    match op {
        Op::N => a.clone(),
        Op::C => adjoint(a),
        Op::T => {
            let mut t = Matrix::zeros(a.cols(), a.rows()).unwrap();
            for j in 0..a.cols() {
                for i in 0..a.rows() {
                    t.set(j, i, a.get(i, j));
                }
            }
            t
        }
    }
}

fn case<T: Scalar>(exec: &Exec<'_>, m: usize, n: usize) {
    let a = random::<T>(m, n, 17);
    let f = lu(exec, &a.view()).unwrap();
    let pa = permute_rows(&a, f.p());
    assert!(
        rel_err(&matmul(&f.l(), &f.u()), &pa) < tol::<T>(m.max(n)),
        "P A != L U at {m}x{n}"
    );
    if m == n {
        for op in [Op::N, Op::T, Op::C] {
            let x = random::<T>(n, 3, 5);
            let mut b = matmul(&op_mat(&a, op), &x);
            f.solve(exec, op, &mut b.view_mut()).unwrap();
            assert!(rel_err(&b, &x) < tol::<T>(n) * 1e3, "solve {op:?} n={n}");
            let g = lu_full(exec, &a.view()).unwrap();
            let mut b = matmul(&op_mat(&a, op), &x);
            g.solve(exec, op, &mut b.view_mut()).unwrap();
            assert!(
                rel_err(&b, &x) < tol::<T>(n) * 1e3,
                "full-piv solve {op:?} n={n}"
            );
        }
    }
}

#[test]
fn lu_reconstructs_rectangular_and_solves_square() {
    for (m, n) in [(0, 0), (1, 1), (7, 5), (5, 7), (8, 8), (60, 60)] {
        case::<f64>(&Exec::serial(), m, n);
        case::<Complex64>(&Exec::serial(), m, n);
    }
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    case::<f64>(&Exec::rayon(&pool), 300, 300);
}

#[test]
fn singular_solve_is_a_typed_error() {
    let mut a = identity::<f64>(4);
    a.set(2, 2, 0.0);
    let mut b = Matrix::<f64>::zeros(4, 1).unwrap();
    let e = solve(&Exec::serial(), &a.view(), &mut b.view_mut()).unwrap_err();
    assert!(matches!(e, Error::Singular { .. }), "{e:?}");
    assert_eq!(det(&Exec::serial(), &a.view()).unwrap(), 0.0);
    let (sign, logabs) = logdet(&Exec::serial(), &a.view()).unwrap();
    assert_eq!(sign, 0.0);
    assert!(logabs.is_infinite() && logabs < 0.0);
    assert!(matches!(
        inv(&Exec::serial(), &a.view()),
        Err(Error::Singular { .. })
    ));
    let r = Matrix::<f64>::zeros(2, 3).unwrap();
    assert!(matches!(
        lu(&Exec::serial(), &r.view())
            .unwrap()
            .solve(&Exec::serial(), Op::N, &mut b.view_mut()),
        Err(Error::NotSquare { .. })
    ));
}

#[test]
fn det_logdet_and_inverse() {
    // det of a permutation-requiring 3x3 with known value -3.
    let a = Matrix::<f64>::from_col_major(3, 3, vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.5])
        .unwrap();
    assert!((det(&Exec::serial(), &a.view()).unwrap() - (-3.0)).abs() < 1e-14);
    let (s, l) = logdet(&Exec::serial(), &a.view()).unwrap();
    assert_eq!(s, -1.0);
    assert!((l - 3.0f64.ln()).abs() < 1e-14);
    let c = random::<Complex64>(6, 6, 3);
    let d = det(&Exec::serial(), &c.view()).unwrap();
    let (s, l) = logdet(&Exec::serial(), &c.view()).unwrap();
    let rebuilt = s * Complex64::new(l.exp(), 0.0);
    assert!((rebuilt - d).norm() < 1e-12 * d.norm().max(1.0));
    let ai = inv(&Exec::serial(), &c.view()).unwrap();
    assert!(rel_err(&matmul(&ai, &c), &identity(6)) < 1e-12);
    let _ = <Complex64 as Element>::zero();
    // Tiny but nonzero pivots keep their determinant.
    let tiny = Matrix::<f64>::from_col_major(2, 2, vec![1.0, 0.0, 0.0, 1e-20]).unwrap();
    assert_eq!(det(&Exec::serial(), &tiny.view()).unwrap(), 1e-20);
}
