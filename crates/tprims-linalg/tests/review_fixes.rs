//! Regressions from the Phase 1d review.
use tprims_blas::Scalar;
use tprims_exec::Exec;
use tprims_linalg::{det, logdet, qr, svd, Error, Matrix, Vectors};

mod common;

#[test]
fn huge_empty_views_give_typed_errors_not_panics() {
    let e: Vec<f64> = vec![];
    // 2^33 x 0: valid (empty) view; Q_full would be 2^33 x 2^33.
    let v = strided_view::StridedView::new(&e, &[1usize << 33, 0], &[1, 1], 0).unwrap();
    let f = qr(&Exec::serial(), &v).unwrap();
    assert!(matches!(f.q_full(&Exec::serial()), Err(Error::Shape(_))));
    assert!(matches!(
        svd(&Exec::serial(), &v, Vectors::Full),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        Matrix::<f64>::zeros(1 << 40, 1 << 40),
        Err(Error::Shape(_))
    ));
}

#[test]
fn det_of_non_finite_input_is_nan_but_exact_zero_pivot_is_zero() {
    let e = Exec::serial();
    for bad in [f64::NAN, f64::INFINITY] {
        let a = Matrix::from_col_major(1, 1, vec![bad]).unwrap();
        assert!(det(&e, &a.view()).unwrap().is_nan(), "det([[{bad}]])");
        let (s, l) = logdet(&e, &a.view()).unwrap();
        assert!(s.is_nan() && l.is_nan());
    }
    // Exact zero pivot followed by 0/0 garbage: det is exactly 0.
    let z = Matrix::from_col_major(2, 2, vec![0.0, 0.0, 1.0, 2.0]).unwrap();
    assert_eq!(det(&e, &z.view()).unwrap(), 0.0);
    let _ = <f64 as Scalar>::IS_COMPLEX_SCALAR;
}

#[test]
fn from_view_of_an_oversized_broadcast_view_is_a_shape_error() {
    // #15: both strides zero, so one element backs a 2^31 x 2^31 view whose
    // copy would exceed isize::MAX bytes.
    let data = [1.0f64];
    let v =
        strided_view::StridedView::new(&data, &[1usize << 31, 1usize << 31], &[0, 0], 0).unwrap();
    let r = std::panic::catch_unwind(|| Matrix::<f64>::from_view(&v));
    assert!(
        matches!(r, Ok(Err(Error::Shape(_)))),
        "expected Err(Error::Shape), got {:?}",
        r.map(|x| x.map(|_| ()))
    );
}
