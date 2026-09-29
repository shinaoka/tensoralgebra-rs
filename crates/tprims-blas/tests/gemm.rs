use num_complex::Complex64;
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_blas::{gemm, Conj, MatIn, Scalar};
use tprims_exec::{Exec, Pool};

mod common;
use common::{data, dense, from_f64, max_rel_err, naive_gemm};

/// Run gemm on (possibly transposed / reversed) views and compare with the reference.
#[allow(clippy::too_many_arguments)]
fn check<T: Scalar>(
    exec: &Exec<'_>,
    m: usize,
    n: usize,
    k: usize,
    a_t: bool,
    b_rev: bool,
    ca: Conj,
    cb: Conj,
    beta: T,
) {
    let ad: Vec<T> = data(m * k, 1);
    let bd: Vec<T> = data(k * n, 2);
    // A stored k x m when transposed, viewed as m x k.
    let a = if a_t {
        StridedView::new(&ad, &[m, k], &[k as isize, 1], 0).unwrap()
    } else {
        StridedView::new(&ad, &[m, k], &[1, m as isize], 0).unwrap()
    };
    // B reversed in both axes: negative strides with offset at the last element.
    let b = if b_rev && k * n > 0 {
        StridedView::new(&bd, &[k, n], &[-1, -(k as isize)], (k * n - 1) as isize).unwrap()
    } else {
        StridedView::new(&bd, &[k, n], &[1, k as isize], 0).unwrap()
    };
    // C with padded leading dimension m + 3.
    let ld = m + 3;
    let mut cd: Vec<T> = data(ld * n.max(1), 3);
    let c0 = StridedView::new(&cd, &[m, n], &[1, ld as isize], 0).unwrap();
    let want = naive_gemm(from_f64::<T>(1.5), &a, ca, &b, cb, beta, &c0);
    let (mut ma, mut mb) = (MatIn::new(&a), MatIn::new(&b));
    ma.conj = ca;
    mb.conj = cb;
    {
        let mut c = StridedViewMut::new(&mut cd, &[m, n], &[1, ld as isize], 0).unwrap();
        gemm(exec, from_f64::<T>(1.5), ma, mb, beta, &mut c).unwrap();
    }
    let got = dense(&StridedView::new(&cd, &[m, n], &[1, ld as isize], 0).unwrap());
    let err = max_rel_err(&got, &want);
    let tol = if std::mem::size_of::<<T as Scalar>::Re>() == 4 {
        2e-5
    } else {
        1e-12
    };
    assert!(
        err < tol,
        "m{m} n{n} k{k} a_t{a_t} b_rev{b_rev} {ca:?} {cb:?}: err {err}"
    );
}

fn suite<T: Scalar>(exec: &Exec<'_>) {
    let beta = from_f64::<T>(2.5);
    for &(m, n, k) in &[(1, 1, 1), (7, 5, 3), (64, 33, 17), (130, 70, 90)] {
        for a_t in [false, true] {
            for b_rev in [false, true] {
                check::<T>(exec, m, n, k, a_t, b_rev, Conj::No, Conj::No, beta);
            }
        }
    }
    check::<T>(
        exec,
        9,
        4,
        0,
        false,
        false,
        Conj::No,
        Conj::No,
        from_f64::<T>(3.0),
    );
    check::<T>(
        exec,
        9,
        4,
        0,
        false,
        false,
        Conj::No,
        Conj::No,
        <T as Element>::zero(),
    );
    check::<T>(exec, 0, 4, 5, false, false, Conj::No, Conj::No, beta);
    check::<T>(
        exec,
        6,
        6,
        6,
        false,
        false,
        Conj::No,
        Conj::No,
        <T as Element>::one(),
    );
}

#[test]
fn gemm_matches_reference_f64_and_c64() {
    suite::<f64>(&Exec::serial());
    suite::<Complex64>(&Exec::serial());
    suite::<f32>(&Exec::serial());
}

#[test]
fn complex_conjugation_on_each_operand() {
    for (ca, cb) in [
        (Conj::Yes, Conj::No),
        (Conj::No, Conj::Yes),
        (Conj::Yes, Conj::Yes),
    ] {
        check::<Complex64>(
            &Exec::serial(),
            13,
            11,
            7,
            true,
            true,
            ca,
            cb,
            from_f64(0.5),
        );
    }
}

#[test]
fn beta_zero_never_reads_c() {
    let a: Vec<f64> = data(12, 1);
    let b: Vec<f64> = data(12, 2);
    let av = StridedView::new(&a, &[3, 4], &[1, 3], 0).unwrap();
    let bv = StridedView::new(&b, &[4, 3], &[1, 4], 0).unwrap();
    let mut c = vec![f64::NAN; 9];
    let mut cv = StridedViewMut::new(&mut c, &[3, 3], &[1, 3], 0).unwrap();
    gemm(
        &Exec::serial(),
        1.0,
        MatIn::new(&av),
        MatIn::new(&bv),
        0.0,
        &mut cv,
    )
    .unwrap();
    assert!(c.iter().all(|x| x.is_finite()));
    // k == 0 with beta == 0 also clears NaN.
    let e: Vec<f64> = vec![];
    let ev = StridedView::new(&e, &[3, 0], &[1, 3], 0).unwrap();
    let fv = StridedView::new(&e, &[0, 3], &[1, 1], 0).unwrap();
    let mut c = vec![f64::NAN; 9];
    let mut cv = StridedViewMut::new(&mut c, &[3, 3], &[1, 3], 0).unwrap();
    gemm(
        &Exec::serial(),
        1.0,
        MatIn::new(&ev),
        MatIn::new(&fv),
        0.0,
        &mut cv,
    )
    .unwrap();
    assert!(c.iter().all(|&x| x == 0.0));
}

#[test]
fn pool_results_match_serial_and_small_work_never_enters() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    check::<f64>(&exec, 8, 8, 8, false, false, Conj::No, Conj::No, 0.0);
    assert_eq!(pool.stats().entries, 0, "8x8x8 must stay on the caller");
    suite::<f64>(&exec);
    check::<f64>(&exec, 400, 300, 350, true, false, Conj::No, Conj::No, 1.0);
    assert!(
        pool.stats().entries >= 1,
        "a 400x300x350 GEMM should enter the pool"
    );
}
