use num_complex::Complex64;
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_blas::Scalar;
use tprims_exec::{Exec, Pool};
use tprims_linalg::{batched, cholesky, eigh, lu, svd, Matrix, Status, Vectors};

mod common;
use common::*;

/// Stack `items` (all m x n) into a column-major [m, n, batch] buffer.
fn stack<T: Scalar>(items: &[Matrix<T>]) -> Vec<T> {
    items.iter().flat_map(|m| m.data().to_vec()).collect()
}

fn item<T: Scalar>(buf: &[T], m: usize, n: usize, i: usize) -> Matrix<T> {
    Matrix::from_col_major(m, n, buf[i * m * n..(i + 1) * m * n].to_vec()).unwrap()
}

fn pools() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
}

fn solve_case<T: Scalar>(exec: &Exec<'_>, n: usize, nb: usize) {
    let mut mats: Vec<Matrix<T>> = (0..nb).map(|i| random::<T>(n, n, 100 + i as u64)).collect();
    if nb > 3 {
        mats[3] = Matrix::zeros(n, n).unwrap(); // singular item
    }
    let a = stack(&mats);
    let rhs: Vec<Matrix<T>> = (0..nb).map(|i| random::<T>(n, 2, 500 + i as u64)).collect();
    let mut b = stack(&rhs);
    let st = {
        let av = StridedView::new(&a, &[n, n, nb], &[1, n as isize, (n * n) as isize], 0).unwrap();
        let mut bv =
            StridedViewMut::new(&mut b, &[n, 2, nb], &[1, n as isize, (2 * n) as isize], 0)
                .unwrap();
        batched::solve(exec, &av, &mut bv).unwrap()
    };
    assert_eq!(st.len(), nb);
    for i in 0..nb {
        if nb > 3 && i == 3 && n > 0 {
            assert!(
                matches!(st[i], Status::Singular { .. }),
                "item 3: {:?}",
                st[i]
            );
            continue;
        }
        assert_eq!(st[i], Status::Ok, "item {i}");
        let mut want = rhs[i].clone();
        lu(&Exec::serial(), &mats[i].view())
            .unwrap()
            .solve(&Exec::serial(), tprims_blas::Op::N, &mut want.view_mut())
            .unwrap();
        assert!(
            rel_err(&item(&b, n, 2, i), &want) < tol::<T>(n) * 1e3,
            "solve item {i}"
        );
    }
}

#[test]
fn batched_solve_matches_single_and_reports_the_singular_item() {
    let tp = pools();
    let pool = Pool::borrow(&tp);
    for exec in [Exec::serial(), Exec::rayon(&pool)] {
        for (n, nb) in [(2, 1024), (8, 16), (40, 3), (0, 4), (3, 0)] {
            solve_case::<f64>(&exec, n, nb);
            solve_case::<Complex64>(&exec, n, nb);
        }
    }
}

#[test]
fn batched_cholesky_and_eigh_match_single_matrix_results() {
    let tp = pools();
    let pool = Pool::borrow(&tp);
    for exec in [Exec::serial(), Exec::rayon(&pool)] {
        let (n, nb) = (6, 50);
        let mut mats: Vec<Matrix<Complex64>> = (0..nb).map(|i| hpd(n, 7 + i as u64)).collect();
        mats[5].set(0, 0, Complex64::new(-50.0, 0.0)); // not positive definite
        let orig = stack(&mats);
        let mut a = orig.clone();
        let st = {
            let mut av =
                StridedViewMut::new(&mut a, &[n, n, nb], &[1, n as isize, (n * n) as isize], 0)
                    .unwrap();
            batched::cholesky(&exec, &mut av).unwrap()
        };
        for i in 0..nb {
            if i == 5 {
                assert_eq!(st[i], Status::NotPositiveDefinite { index: 0 });
                continue;
            }
            assert_eq!(st[i], Status::Ok);
            let l = cholesky(&Exec::serial(), &mats[i].view()).unwrap();
            assert!(rel_err(&item(&a, n, n, i), l.l()) < 1e-13, "chol item {i}");
        }
        let av =
            StridedView::new(&orig, &[n, n, nb], &[1, n as isize, (n * n) as isize], 0).unwrap();
        let mut w = vec![0.0f64; n * nb];
        let mut v = vec![Complex64::new(0.0, 0.0); n * n * nb];
        let st = {
            let mut wv = StridedViewMut::new(&mut w, &[n, nb], &[1, n as isize], 0).unwrap();
            let mut vv =
                StridedViewMut::new(&mut v, &[n, n, nb], &[1, n as isize, (n * n) as isize], 0)
                    .unwrap();
            batched::eigh(&exec, &av, &mut wv, Some(&mut vv)).unwrap()
        };
        for i in 0..nb {
            assert_eq!(st[i], Status::Ok);
            let d = eigh(&Exec::serial(), &mats[i].view(), false).unwrap();
            for j in 0..n {
                assert!((w[i * n + j] - d.values[j]).abs() < 1e-11, "eigh item {i}");
            }
        }
    }
}

#[test]
fn batched_svd_values_match_and_empty_batches_work() {
    let (m, n, nb) = (5, 3, 20);
    let mats: Vec<Matrix<f64>> = (0..nb).map(|i| random(m, n, 900 + i as u64)).collect();
    let a = stack(&mats);
    let av = StridedView::new(&a, &[m, n, nb], &[1, m as isize, (m * n) as isize], 0).unwrap();
    let mut s = vec![0.0f64; n * nb];
    let mut u = vec![0.0f64; m * n * nb];
    let mut vt = vec![0.0f64; n * n * nb];
    let st = {
        let mut sv = StridedViewMut::new(&mut s, &[n, nb], &[1, n as isize], 0).unwrap();
        let mut uv =
            StridedViewMut::new(&mut u, &[m, n, nb], &[1, m as isize, (m * n) as isize], 0)
                .unwrap();
        let mut vv =
            StridedViewMut::new(&mut vt, &[n, n, nb], &[1, n as isize, (n * n) as isize], 0)
                .unwrap();
        batched::svd(&Exec::serial(), &av, &mut sv, Some((&mut uv, &mut vv))).unwrap()
    };
    for i in 0..nb {
        assert_eq!(st[i], Status::Ok);
        let d = svd(&Exec::serial(), &mats[i].view(), Vectors::None).unwrap();
        for j in 0..n {
            assert!((s[i * n + j] - d.s[j]).abs() < 1e-12);
        }
        // U S Vᴴ reconstructs the item.
        let (ui, vi) = (item(&u, m, n, i), item(&vt, n, n, i));
        let mut us = ui.clone();
        for j in 0..n {
            for r in 0..m {
                us.set(r, j, ui.get(r, j) * s[i * n + j]);
            }
        }
        assert!(rel_err(&matmul(&us, &adjoint(&vi)), &mats[i]) < 1e-12);
    }
    let e: Vec<f64> = vec![];
    let ev = StridedView::new(&e, &[2, 2, 0], &[1, 2, 4], 0).unwrap();
    let mut w: Vec<f64> = vec![];
    let mut wv = StridedViewMut::new(&mut w, &[2, 0], &[1, 2], 0).unwrap();
    assert!(batched::eigh(&Exec::serial(), &ev, &mut wv, None)
        .unwrap()
        .is_empty());
    let _ = <f64 as Element>::zero();
}
