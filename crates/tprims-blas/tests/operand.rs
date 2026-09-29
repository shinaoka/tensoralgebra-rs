use strided_view::StridedView;
use tprims_blas::{gemm, Error, MatIn};
use tprims_exec::Exec;

mod common;

#[test]
fn rank_and_shape_errors_are_typed() {
    let d = vec![0.0f64; 24];
    let a3 = StridedView::new(&d, &[2, 3, 4], &[1, 2, 6], 0).unwrap();
    let a = StridedView::new(&d, &[2, 3], &[1, 2], 0).unwrap();
    let b = StridedView::new(&d, &[4, 2], &[1, 4], 0).unwrap();
    let mut cd = vec![0.0f64; 4];
    let mut c = strided_view::StridedViewMut::new(&mut cd, &[2, 2], &[1, 2], 0).unwrap();
    let e = gemm(
        &Exec::serial(),
        1.0,
        MatIn::new(&a3),
        MatIn::new(&b),
        0.0,
        &mut c,
    );
    assert!(matches!(
        e,
        Err(Error::Rank {
            operand: "A",
            expected: 2,
            got: 3
        })
    ));
    let e = gemm(
        &Exec::serial(),
        1.0,
        MatIn::new(&a),
        MatIn::new(&b),
        0.0,
        &mut c,
    );
    assert!(matches!(e, Err(Error::Shape(_))));
}

#[test]
fn aliased_outputs_are_rejected_before_any_write() {
    let d = vec![1.0f64; 4];
    let a = StridedView::new(&d, &[2, 2], &[1, 2], 0).unwrap();
    for strides in [[0isize, 1], [1, 1], [0, 0]] {
        let mut cd = vec![7.0f64; 4];
        {
            let mut c = strided_view::StridedViewMut::new(&mut cd, &[2, 2], &strides, 0).unwrap();
            let e = gemm(
                &Exec::serial(),
                1.0,
                MatIn::new(&a),
                MatIn::new(&a),
                0.0,
                &mut c,
            );
            assert_eq!(e, Err(Error::AliasedOutput), "strides {strides:?}");
        }
        assert!(cd.iter().all(|&x| x == 7.0));
    }
}

#[test]
fn padded_negative_and_degenerate_output_layouts_are_accepted() {
    let d = vec![1.0f64; 4];
    let a = StridedView::new(&d, &[2, 2], &[1, 2], 0).unwrap();
    // padded leading dimension, row-major, negative strides, extent-1 zero stride
    let cases: [(&[usize], &[isize], isize, usize); 4] = [
        (&[2, 2], &[1, 5], 0, 10),
        (&[2, 2], &[2, 1], 0, 4),
        (&[2, 2], &[-1, -2], 3, 4),
        (&[2, 1], &[1, 0], 0, 2),
    ];
    for (dims, strides, off, len) in cases {
        let mut cd = vec![0.0f64; len];
        let mut c = strided_view::StridedViewMut::new(&mut cd, dims, strides, off).unwrap();
        let b = StridedView::new(&d, &[2, dims[1]], &[1, 2], 0).unwrap();
        gemm(
            &Exec::serial(),
            1.0,
            MatIn::new(&a),
            MatIn::new(&b),
            0.0,
            &mut c,
        )
        .unwrap_or_else(|e| panic!("{dims:?} {strides:?}: {e}"));
    }
}
