use tprims_contract_traits::*;

fn prob(
    dot: DotGeneral,
    a: (&[usize], &[isize]),
    b: (&[usize], &[isize]),
    c: (&[usize], &[isize]),
) -> Problem {
    Problem::new(
        dot,
        Layout::new(a.0, a.1),
        Layout::new(b.0, b.1),
        Layout::new(c.0, c.1),
        (Conj::No, Conj::No),
    )
}

#[test]
fn interface_is_object_safe_at_fixed_scalar() {
    fn _backend(_: &dyn ContractionBackend<f64>) {}
    fn _plan(_: &dyn PreparedContraction<f32>) {}
    fn _host(_: &dyn HostExecution) {}
    fn _slot(_: Vec<Box<dyn ContractionBackend<num_c64::C>>>) {}
    fn send_sync<X: Send + Sync>() {}
    send_sync::<BoxedPlan<f64>>();
    send_sync::<Box<dyn ContractionBackend<f32>>>();
}

mod num_c64 {
    #![allow(dead_code)]
    /// Any `Copy + PartialEq + Send + Sync` type is a scalar of the interface.
    #[derive(Clone, Copy, PartialEq)]
    pub struct C(pub f64, pub f64);
}

#[test]
fn validation_reports_typed_errors_in_order() {
    let ok = DotGeneral::new(&[1], &[0], &[], &[]);
    let p = prob(
        ok.clone(),
        (&[2, 3], &[1, 2]),
        (&[3, 4], &[1, 3]),
        (&[2, 4], &[1, 2]),
    );
    let v = p.validate::<f64>().unwrap();
    assert_eq!(v.shape.out_dims, vec![2, 4]);
    assert!(!v.k_empty && !v.all_batch);

    let bad_cfg = DotGeneral::new(&[1, 1], &[0, 0], &[], &[]);
    let p = prob(
        bad_cfg,
        (&[2, 3], &[1, 2]),
        (&[3, 4], &[1, 3]),
        (&[2, 4], &[1, 2]),
    );
    assert!(matches!(p.validate::<f64>(), Err(Error::Config(_))));

    let p = prob(
        ok.clone(),
        (&[2, 3], &[1, 2]),
        (&[5, 4], &[1, 5]),
        (&[2, 4], &[1, 2]),
    );
    assert!(matches!(p.validate::<f64>(), Err(Error::Shape(_))));

    let p = prob(
        ok.clone(),
        (&[2, 3], &[1, 2]),
        (&[3, 4], &[1, 3]),
        (&[2, 5], &[1, 2]),
    );
    assert!(matches!(p.validate::<f64>(), Err(Error::Shape(_))));

    // rank / stride count mismatch
    let p = prob(
        ok.clone(),
        (&[2, 3], &[1]),
        (&[3, 4], &[1, 3]),
        (&[2, 4], &[1, 2]),
    );
    assert!(matches!(p.validate::<f64>(), Err(Error::Shape(_))));

    // overlapping output strides
    let p = prob(
        ok.clone(),
        (&[2, 3], &[1, 2]),
        (&[3, 4], &[1, 3]),
        (&[2, 4], &[1, 1]),
    );
    assert!(matches!(p.validate::<f64>(), Err(Error::AliasedOutput)));

    // overflowing element count is rejected even though the problem is empty-shaped
    let big = usize::MAX / 2;
    let p = prob(
        ok,
        (&[big, 0], &[1, 1]),
        (&[0, big], &[1, 1]),
        (&[big, big], &[1, 1]),
    );
    assert!(matches!(p.validate::<f64>(), Err(Error::Shape(_))));
}

#[test]
fn validation_flags_empty_contraction_and_all_batch() {
    let p = prob(
        DotGeneral::new(&[1], &[0], &[], &[]),
        (&[2, 0], &[1, 2]),
        (&[0, 3], &[1, 1]),
        (&[2, 3], &[1, 2]),
    );
    assert!(p.validate::<f32>().unwrap().k_empty);
    let p = prob(
        DotGeneral::new(&[], &[], &[0], &[0]),
        (&[4], &[1]),
        (&[4], &[1]),
        (&[4], &[1]),
    );
    assert!(p.validate::<f32>().unwrap().all_batch);
}

#[test]
fn check_views_names_the_differing_operand() {
    let p = prob(
        DotGeneral::new(&[1], &[0], &[], &[]),
        (&[2, 3], &[1, 2]),
        (&[3, 4], &[1, 3]),
        (&[2, 4], &[1, 2]),
    );
    p.check_views((&[2, 3], &[1, 2]), (&[3, 4], &[1, 3]), (&[2, 4], &[1, 2]))
        .unwrap();
    let e = p
        .check_views((&[2, 3], &[1, 2]), (&[3, 4], &[3, 1]), (&[2, 4], &[1, 2]))
        .unwrap_err();
    assert!(
        matches!(&e, Error::LayoutMismatch(m) if m.starts_with('B')),
        "{e}"
    );
}

#[test]
fn injective_layouts() {
    assert!(is_injective_layout(&[(2, 1), (3, 2)]));
    assert!(is_injective_layout(&[(2, -1), (3, 2)]));
    assert!(is_injective_layout(&[(0, 0), (3, 0)]));
    assert!(is_injective_layout(&[(1, 0), (3, 1)]));
    assert!(!is_injective_layout(&[(2, 0), (3, 1)]));
    assert!(!is_injective_layout(&[(2, 1), (3, 1)]));
}

#[test]
fn backend_errors_keep_their_source() {
    #[derive(Debug, thiserror::Error)]
    #[error("lower layer")]
    struct Lower;
    let e = Error::backend(Lower);
    let src = std::error::Error::source(&e);
    // `Backend` displays the payload; the payload stays downcastable.
    assert_eq!(e.to_string(), "lower layer");
    assert!(src.is_none());
    let Error::Backend(inner) = e else { panic!() };
    assert!(inner.downcast_ref::<Lower>().is_some());
    assert!(Error::Unsupported("x".into()).is_unsupported());
    assert!(!Error::AliasedOutput.is_unsupported());
}

#[test]
fn serial_host_contract() {
    assert_eq!(SerialHost.budget(), 1);
    let mut seen = None;
    SerialHost.install(8, &mut |p| seen = Some(p));
    assert_eq!(seen, Some(Par::Seq));
    assert_eq!(SerialHost.broadcast(0, &|_| {}), Ok(()));
    assert_eq!(
        SerialHost.broadcast(2, &|_| {}),
        Err(HostError::Unavailable)
    );
    assert!(SerialHost.native(std::any::TypeId::of::<u8>()).is_none());
}
