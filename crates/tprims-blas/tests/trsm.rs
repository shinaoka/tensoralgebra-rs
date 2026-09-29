use num_complex::Complex64;
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_blas::{trsm, Conj, Diag, Error, Op, Scalar, Side, Uplo};
use tprims_exec::{Exec, Pool};

mod common;
use common::{data, dense, from_f64, max_rel_err, naive_gemm};

/// Well-conditioned triangular A (n x n, column-major) with the given structure.
fn tri<T: Scalar>(n: usize, uplo: Uplo, diag: Diag) -> Vec<T> {
    let r: Vec<T> = data(n * n, 5);
    let mut a = vec![<T as Element>::zero(); n * n];
    for j in 0..n {
        for i in 0..n {
            let keep = match uplo {
                Uplo::Lower => i >= j,
                Uplo::Upper => i <= j,
            };
            if i == j {
                // Unit: the stored diagonal is garbage that must be ignored.
                a[i + j * n] = if diag == Diag::Unit {
                    from_f64(1e3)
                } else {
                    Element::add(from_f64(4.0), r[i + j * n])
                };
            } else if keep {
                a[i + j * n] = Element::mul(from_f64(0.3), r[i + j * n]);
            } else {
                a[i + j * n] = from_f64(7e2); // garbage in the ignored triangle
            }
        }
    }
    a
}

/// The matrix op(A) actually used (unit diagonal, clean triangle).
fn op_a<T: Scalar>(a: &[T], n: usize, uplo: Uplo, diag: Diag, op: Op) -> Vec<T> {
    let mut clean = vec![<T as Element>::zero(); n * n];
    for j in 0..n {
        for i in 0..n {
            let keep = match uplo {
                Uplo::Lower => i >= j,
                Uplo::Upper => i <= j,
            };
            if i == j {
                clean[i + j * n] = if diag == Diag::Unit {
                    <T as Element>::one()
                } else {
                    a[i + j * n]
                };
            } else if keep {
                clean[i + j * n] = a[i + j * n];
            }
        }
    }
    let mut out = clean.clone();
    for j in 0..n {
        for i in 0..n {
            out[i + j * n] = match op {
                Op::N => clean[i + j * n],
                Op::T => clean[j + i * n],
                Op::C => Element::conj(clean[j + i * n]),
            };
        }
    }
    out
}

fn check<T: Scalar>(
    exec: &Exec<'_>,
    side: Side,
    uplo: Uplo,
    op: Op,
    diag: Diag,
    n: usize,
    nrhs: usize,
) {
    let a = tri::<T>(n, uplo, diag);
    let (br, bc) = match side {
        Side::Left => (n, nrhs),
        Side::Right => (nrhs, n),
    };
    let x: Vec<T> = data(br * bc, 9);
    let oa = op_a(&a, n, uplo, diag, op);
    let oav = StridedView::new(&oa, &[n, n], &[1, n as isize], 0).unwrap();
    let xv = StridedView::new(&x, &[br, bc], &[1, br as isize], 0).unwrap();
    let zero = vec![<T as Element>::zero(); br * bc];
    let zv = StridedView::new(&zero, &[br, bc], &[1, br as isize], 0).unwrap();
    let alpha = from_f64::<T>(2.0);
    // B = op(A) X / alpha (Left) or X op(A) / alpha (Right), so the solve returns X.
    let inv_alpha = from_f64::<T>(0.5);
    let mut b = match side {
        Side::Left => naive_gemm(
            inv_alpha,
            &oav,
            Conj::No,
            &xv,
            Conj::No,
            <T as Element>::zero(),
            &zv,
        ),
        Side::Right => naive_gemm(
            inv_alpha,
            &xv,
            Conj::No,
            &oav,
            Conj::No,
            <T as Element>::zero(),
            &zv,
        ),
    };
    let av = StridedView::new(&a, &[n, n], &[1, n as isize], 0).unwrap();
    {
        let mut bv = StridedViewMut::new(&mut b, &[br, bc], &[1, br as isize], 0).unwrap();
        trsm(exec, side, uplo, op, diag, alpha, &av, &mut bv).unwrap();
    }
    let err = max_rel_err(&b, &dense(&xv));
    assert!(err < 1e-10, "{side:?} {uplo:?} {op:?} {diag:?}: {err}");
}

#[test]
fn all_side_uplo_op_diag_combinations() {
    for side in [Side::Left, Side::Right] {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for op in [Op::N, Op::T, Op::C] {
                for diag in [Diag::NonUnit, Diag::Unit] {
                    check::<f64>(&Exec::serial(), side, uplo, op, diag, 7, 5);
                    check::<Complex64>(&Exec::serial(), side, uplo, op, diag, 7, 5);
                }
            }
        }
    }
}

#[test]
fn large_solve_on_a_pool_and_shape_errors() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    check::<f64>(
        &exec,
        Side::Left,
        Uplo::Lower,
        Op::N,
        Diag::NonUnit,
        300,
        200,
    );
    check::<f64>(&exec, Side::Right, Uplo::Upper, Op::T, Diag::Unit, 200, 150);
    let a = vec![1.0f64; 6];
    let av = StridedView::new(&a, &[2, 3], &[1, 2], 0).unwrap();
    let mut b = vec![0.0f64; 4];
    let mut bv = StridedViewMut::new(&mut b, &[2, 2], &[1, 2], 0).unwrap();
    assert!(matches!(
        trsm(
            &Exec::serial(),
            Side::Left,
            Uplo::Lower,
            Op::N,
            Diag::NonUnit,
            1.0,
            &av,
            &mut bv
        ),
        Err(Error::Shape(_))
    ));
}
