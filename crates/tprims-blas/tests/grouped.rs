//! `gemm_grouped`: variable-size jobs over shared flat buffers.
use num_complex::Complex64;
use strided_view::StridedView;
use tprims_blas::{gemm_grouped, Conj, Error, GroupedJob, Scalar};
use tprims_exec::{Exec, Pool};

mod common;
use common::{data, from_f64, max_rel_err, naive_gemm};

/// Jobs `(m, k, n)` packed back to back in A, B and C, with a gap of one
/// element after each block so offsets are not trivially contiguous.
fn layout(shapes: &[(usize, usize, usize)]) -> (Vec<GroupedJob>, usize, usize, usize) {
    let (mut oa, mut ob, mut oc) = (0, 0, 0);
    let mut jobs = Vec::new();
    for &(m, k, n) in shapes {
        jobs.push(GroupedJob {
            a_offset: oa,
            b_offset: ob,
            c_offset: oc,
            rows: m,
            inner: k,
            cols: n,
        });
        oa += m * k + 1;
        ob += k * n + 1;
        oc += m * n + 1;
    }
    (jobs, oa, ob, oc)
}

fn reference<T: Scalar>(
    alpha: T,
    a: &[T],
    ca: Conj,
    b: &[T],
    cb: Conj,
    beta: T,
    c: &[T],
    jobs: &[GroupedJob],
) -> Vec<T> {
    let mut out = c.to_vec();
    for j in jobs {
        let (m, k, n) = (j.rows, j.inner, j.cols);
        if m == 0 || n == 0 {
            continue;
        }
        let av = StridedView::new(a, &[m, k], &[1, m as isize], j.a_offset as isize).unwrap();
        let bv =
            StridedView::new(b, &[k, n], &[1, k.max(1) as isize], j.b_offset as isize).unwrap();
        let cv = StridedView::new(c, &[m, n], &[1, m as isize], j.c_offset as isize).unwrap();
        let r = naive_gemm(alpha, &av, ca, &bv, cb, beta, &cv);
        for col in 0..n {
            for row in 0..m {
                out[j.c_offset + row + col * m] = r[row + col * m];
            }
        }
    }
    out
}

fn check<T: Scalar>(exec: &Exec<'_>, shapes: &[(usize, usize, usize)], ca: Conj, beta: f64) {
    let (jobs, la, lb, lc) = layout(shapes);
    let a: Vec<T> = data(la, 1);
    let b: Vec<T> = data(lb, 2);
    let c0: Vec<T> = data(lc, 3);
    let (alpha, beta) = (from_f64::<T>(0.75), from_f64::<T>(beta));
    let want = reference(alpha, &a, ca, &b, Conj::No, beta, &c0, &jobs);
    let mut c = c0.clone();
    gemm_grouped(exec, alpha, &a, ca, &b, Conj::No, beta, &mut c, &jobs).unwrap();
    let err = max_rel_err(&c, &want);
    let tol = if std::mem::size_of::<<T as Scalar>::Re>() == 4 {
        1e-5
    } else {
        1e-12
    };
    assert!(err < tol, "{shapes:?}: rel err {err:e}");
}

const MIXED: &[(usize, usize, usize)] = &[
    (3, 4, 5),
    (1, 1, 1),
    (17, 9, 2),
    (8, 0, 3),
    (0, 5, 4),
    (40, 33, 21),
];

#[test]
fn matches_the_reference_serially() {
    let e = Exec::serial();
    check::<f64>(&e, MIXED, Conj::No, 0.0);
    check::<f64>(&e, MIXED, Conj::No, -0.5);
    check::<Complex64>(&e, MIXED, Conj::Yes, 0.25);
    check::<f32>(&e, &[(4, 4, 4), (2, 3, 1)], Conj::No, 1.0);
}

#[test]
fn matches_the_reference_on_a_pool() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let p = Pool::borrow(&tp);
    let e = Exec::rayon(&p);
    // Many small jobs (spread over the pool) and a few large ones (inner parallelism).
    let many: Vec<_> = (0..300)
        .map(|i| (4 + i % 5, 3 + i % 7, 2 + i % 3))
        .collect();
    check::<f64>(&e, &many, Conj::No, 0.5);
    check::<Complex64>(&e, &many, Conj::Yes, 0.0);
    check::<f64>(&e, &[(300, 200, 250), (256, 256, 256)], Conj::No, 1.0);
}

#[test]
fn empty_jobs_scale_by_beta_and_zero_size_writes_nothing() {
    let (jobs, la, lb, lc) = layout(&[(3, 0, 2), (0, 4, 4)]);
    let (a, b) = (vec![1.0f64; la], vec![1.0f64; lb]);
    let mut c = vec![2.0f64; lc];
    gemm_grouped(
        &Exec::serial(),
        1.0,
        &a,
        Conj::No,
        &b,
        Conj::No,
        0.5,
        &mut c,
        &jobs,
    )
    .unwrap();
    assert!(c[..6].iter().all(|&x| x == 1.0), "{c:?}");
    assert!(
        c[6..].iter().all(|&x| x == 2.0),
        "gap and empty block untouched: {c:?}"
    );
}

#[test]
fn rejects_out_of_bounds_and_overlapping_outputs_before_writing() {
    let e = Exec::serial();
    let (a, b) = (vec![1.0f64; 16], vec![1.0f64; 16]);
    let mut c = vec![7.0f64; 16];
    let job = |c_offset, a_offset| GroupedJob {
        a_offset,
        b_offset: 0,
        c_offset,
        rows: 2,
        inner: 2,
        cols: 2,
    };
    // A block ends past the buffer.
    let r = gemm_grouped(
        &e,
        1.0,
        &a,
        Conj::No,
        &b,
        Conj::No,
        0.0,
        &mut c,
        &[job(0, 14)],
    );
    assert!(matches!(r, Err(Error::Shape(_))), "{r:?}");
    // Two output blocks share elements 2..4.
    let r = gemm_grouped(
        &e,
        1.0,
        &a,
        Conj::No,
        &b,
        Conj::No,
        0.0,
        &mut c,
        &[job(0, 0), job(2, 0), job(12, 0)],
    );
    assert_eq!(r, Err(Error::AliasedOutput));
    assert!(c.iter().all(|&x| x == 7.0), "nothing written: {c:?}");
    // Adjacent, disjoint blocks are fine.
    gemm_grouped(
        &e,
        1.0,
        &a,
        Conj::No,
        &b,
        Conj::No,
        0.0,
        &mut c,
        &[job(4, 0), job(0, 0)],
    )
    .unwrap();
    assert!(c[..8].iter().all(|&x| x == 2.0), "{c:?}");
}
