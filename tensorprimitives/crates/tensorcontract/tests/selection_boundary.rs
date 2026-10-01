use tensorcontract::{
    Blocking, Error, KernelChoice, Layout, Operand, Plan, TensorView, TensorViewMut,
};
use tprims_kernel::SelectError;

#[test]
fn builders_invalidate_cached_geometry_and_preserve_explicit_blocks() {
    let l = Layout::col_major(&[2, 2]);
    let p = Plan::new(
        Operand::new(&l, &[0, 2]),
        Operand::new(&l, &[2, 1]),
        None,
        Operand::new(&l, &[0, 1]),
    )
    .unwrap()
    .with_kernel(KernelChoice::Id("portable.f64.4x4".into()))
    .unwrap();
    let first = p.resolved::<f64>().unwrap();
    let changed = p
        .clone()
        .with_blocking(Blocking {
            mc: 5,
            kc: 3,
            nc: 7,
        })
        .with_threads(8);
    let r = changed.resolved::<f64>().unwrap();
    assert_eq!((r.mc, r.kc, r.nc), (8, 3, 8));
    assert_eq!(
        (
            r.with_threads(1).unwrap().mc,
            r.with_threads(1).unwrap().kc,
            r.with_threads(1).unwrap().nc
        ),
        (8, 3, 8)
    );
    assert_eq!(p.resolved::<f64>().unwrap().kc, first.kc);
}

#[test]
fn auto_head_and_blocking_match_existing_kernelset() {
    fn check<T: tprims_kernel::Families>()
    where
        T::Real: tensorcontract::KernelSet,
    {
        let l = Layout::col_major(&[9, 9]);
        for method in tensorcontract::ComplexMethod::ALL {
            let p = Plan::new(
                Operand::new(&l, &[0, 2]),
                Operand::new(&l, &[2, 1]),
                None,
                Operand::new(&l, &[0, 1]),
            )
            .unwrap()
            .with_complex_method(method)
            .with_threads(4);
            let rg = p.resolved::<T>().unwrap();
            let (mr, nr, blk) = tensorcontract::kernel::plan_config::<T>(&p);
            assert_eq!(
                (rg.mr, rg.nr, rg.mc, rg.kc, rg.nc),
                (mr, nr, blk.mc, blk.kc, blk.nc)
            );
        }
    }
    check::<f32>();
    check::<f64>();
    check::<tensorcontract::C32>();
    check::<tensorcontract::C64>();
}

#[test]
fn safe_empty_execution_rejects_wrong_dtype_and_preserves_source() {
    let l = Layout::col_major(&[0, 0]);
    let p = Plan::new(
        Operand::new(&l, &[0, 2]),
        Operand::new(&l, &[2, 1]),
        None,
        Operand::new(&l, &[0, 1]),
    )
    .unwrap()
    .with_kernel(KernelChoice::Id("portable.c64.native.4x4".into()))
    .unwrap();
    let mut d = [];
    let err = p
        .run(
            1.0f64,
            TensorView::new(&[], &l, &[0, 2]),
            TensorView::new(&[], &l, &[2, 1]),
            0.0,
            None,
            TensorViewMut::new(&mut d, &l, &[0, 1]),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::KernelSelection(SelectError::DtypeMismatch { .. })
    ));
    let source = std::error::Error::source(&err).unwrap();
    assert!(source.downcast_ref::<SelectError>().is_some());
}
