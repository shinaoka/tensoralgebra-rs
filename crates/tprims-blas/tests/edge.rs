//! Edge cases from the Phase 1b review: empty batches with unvalidated
//! strides, alpha == 0 (A/B not referenced), empty outputs with zero strides,
//! one pool entry per batched call, negative/noncontiguous batched strides.
use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{gemm, gemm_batched, trsm, BatchIn, BatchStrategy, Diag, MatIn, Op, Side, Uplo};
use tprims_exec::{Exec, Pool};

mod common;
use common::{data, dense, max_rel_err, naive_gemm};

const BOTH: [BatchStrategy; 2] = [BatchStrategy::FaerLoop, BatchStrategy::Tblis];

#[test]
fn empty_batches_with_huge_strides_never_touch_pointers() {
    let e: Vec<f64> = vec![];
    for strategy in BOTH {
        // k == 0: A and B empty with an absurd batch stride; C = beta * C.
        let a = StridedView::new(&e, &[2, 0, 3], &[1, 2, isize::MAX], 0).unwrap();
        let b = StridedView::new(&e, &[0, 2, 3], &[1, 1, isize::MAX], 0).unwrap();
        let mut c = vec![2.0; 12];
        {
            let mut cv = StridedViewMut::new(&mut c, &[2, 2, 3], &[1, 2, 4], 0).unwrap();
            gemm_batched(
                &Exec::serial(),
                1.0,
                BatchIn::new(&a),
                BatchIn::new(&b),
                3.0,
                &mut cv,
                strategy,
            )
            .unwrap();
        }
        assert!(c.iter().all(|&x| x == 6.0), "{strategy:?}: {c:?}");
        // m == 0: C empty (zero strides allowed) with a huge batch stride.
        let d = vec![1.0f64; 8];
        let a = StridedView::new(&e, &[0, 2, 3], &[1, 1, isize::MAX], 0).unwrap();
        let b3 = StridedView::new(&d, &[2, 1, 3], &[1, 2, 2], 0).unwrap();
        let mut cc: Vec<f64> = vec![];
        let mut cv = StridedViewMut::new(&mut cc, &[0, 1, 3], &[1, 0, 0], 0).unwrap();
        gemm_batched(
            &Exec::serial(),
            1.0,
            BatchIn::new(&a),
            BatchIn::new(&b3),
            0.0,
            &mut cv,
            strategy,
        )
        .unwrap();
    }
}

#[test]
fn empty_outputs_with_zero_strides_are_accepted() {
    let e: Vec<f64> = vec![];
    let a = StridedView::new(&e, &[0, 3], &[1, 0], 0).unwrap();
    let d = vec![1.0f64; 6];
    let b = StridedView::new(&d, &[3, 2], &[1, 3], 0).unwrap();
    let mut cc: Vec<f64> = vec![];
    let mut cv = StridedViewMut::new(&mut cc, &[0, 2], &[1, 0], 0).unwrap();
    gemm(
        &Exec::serial(),
        1.0,
        MatIn::new(&a),
        MatIn::new(&b),
        0.0,
        &mut cv,
    )
    .unwrap();
}

#[test]
fn alpha_zero_does_not_reference_a_or_b() {
    let nan = vec![f64::NAN; 9];
    let av = StridedView::new(&nan, &[3, 3], &[1, 3], 0).unwrap();
    let mut c = vec![1.5; 9];
    {
        let mut cv = StridedViewMut::new(&mut c, &[3, 3], &[1, 3], 0).unwrap();
        gemm(
            &Exec::serial(),
            0.0,
            MatIn::new(&av),
            MatIn::new(&av),
            2.0,
            &mut cv,
        )
        .unwrap();
    }
    assert!(c.iter().all(|&x| x == 3.0), "gemm: {c:?}");
    let a3 = StridedView::new(&nan, &[3, 1, 3], &[1, 3, 3], 0).unwrap();
    let b3 = StridedView::new(&nan, &[1, 3, 3], &[1, 1, 3], 0).unwrap();
    for strategy in BOTH {
        let mut c = vec![1.5; 27];
        {
            let mut cv = StridedViewMut::new(&mut c, &[3, 3, 3], &[1, 3, 9], 0).unwrap();
            gemm_batched(
                &Exec::serial(),
                0.0,
                BatchIn::new(&a3),
                BatchIn::new(&b3),
                2.0,
                &mut cv,
                strategy,
            )
            .unwrap();
        }
        assert!(c.iter().all(|&x| x == 3.0), "{strategy:?}: {c:?}");
    }
    let mut b = vec![5.0; 6];
    {
        let mut bv = StridedViewMut::new(&mut b, &[3, 2], &[1, 3], 0).unwrap();
        trsm(
            &Exec::serial(),
            Side::Left,
            Uplo::Lower,
            Op::N,
            Diag::NonUnit,
            0.0,
            &av,
            &mut bv,
        )
        .unwrap();
    }
    assert!(b.iter().all(|&x| x == 0.0), "trsm: {b:?}");
}

#[test]
fn serial_item_schedule_enters_the_pool_at_most_once_per_call() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let (n, nb) = (300, 3);
    let a: Vec<f64> = data(n * n * nb, 1);
    let st = [1, n as isize, (n * n) as isize];
    let av = StridedView::new(&a, &[n, n, nb], &st, 0).unwrap();
    let mut c = vec![0.0; n * n * nb];
    let mut cv = StridedViewMut::new(&mut c, &[n, n, nb], &st, 0).unwrap();
    gemm_batched(
        &exec,
        1.0,
        BatchIn::new(&av),
        BatchIn::new(&av),
        0.0,
        &mut cv,
        BatchStrategy::FaerLoop,
    )
    .unwrap();
    let s = pool.stats();
    assert_eq!(s.entries, 1, "{s:?}");
}

#[test]
fn batched_negative_and_padded_strides_with_conj_b() {
    use num_complex::Complex64;
    let (m, n, k, nb) = (5, 4, 3, 6);
    let ad: Vec<Complex64> = data(m * k * nb, 1);
    let bd: Vec<Complex64> = data(k * n * nb * 2, 2);
    let a = StridedView::new(
        &ad,
        &[m, k, nb],
        &[-1, -(m as isize), -((m * k) as isize)],
        (m * k * nb - 1) as isize,
    )
    .unwrap();
    // B padded: leading dimension 2k, batch stride 2kn.
    let b = StridedView::new(
        &bd,
        &[k, n, nb],
        &[1, (2 * k) as isize, (2 * k * n) as isize],
        0,
    )
    .unwrap();
    for strategy in BOTH {
        let mut cd: Vec<Complex64> = data(m * n * nb, 3);
        let c0 = cd.clone();
        {
            let mut cv = StridedViewMut::new(
                &mut cd,
                &[m, n, nb],
                &[n as isize * nb as isize, nb as isize, 1],
                0,
            )
            .unwrap();
            gemm_batched(
                &Exec::serial(),
                Complex64::new(0.5, 1.0),
                BatchIn::new(&a),
                BatchIn::new(&b).conj(),
                Complex64::new(2.0, 0.0),
                &mut cv,
                strategy,
            )
            .unwrap();
        }
        for i in 0..nb {
            let (sa, sb) = (a.strides().to_vec(), b.strides().to_vec());
            let ai = StridedView::new(
                &ad,
                &[m, k],
                &[sa[0], sa[1]],
                a.offset() + sa[2] * i as isize,
            )
            .unwrap();
            let bi = StridedView::new(
                &bd,
                &[k, n],
                &[sb[0], sb[1]],
                b.offset() + sb[2] * i as isize,
            )
            .unwrap();
            let ci0 = StridedView::new(&c0, &[m, n], &[(n * nb) as isize, nb as isize], i as isize)
                .unwrap();
            let want = naive_gemm(
                Complex64::new(0.5, 1.0),
                &ai,
                tprims_blas::Conj::No,
                &bi,
                tprims_blas::Conj::Yes,
                Complex64::new(2.0, 0.0),
                &ci0,
            );
            let got = dense(
                &StridedView::new(&cd, &[m, n], &[(n * nb) as isize, nb as isize], i as isize)
                    .unwrap(),
            );
            assert!(max_rel_err(&got, &want) < 1e-12, "{strategy:?} item {i}");
        }
    }
}
