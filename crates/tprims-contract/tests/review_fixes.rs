//! Regressions from the Phase 1c review.
use tprims_blas::Conj;
use tprims_contract::{add, ContractPlan, DotGeneral, Error, Flags, Selected, Strategy};
use tprims_exec::Exec;

mod common;
use common::*;

#[test]
fn extent_one_axes_with_huge_strides_do_not_overflow() {
    // add: C [2, 1, 3] with an isize::MAX stride on the unit axis.
    let a = T::<f64>::new(&[2, 1, 3], 1);
    let mut c = T::<f64> {
        data: vec![1.0; 6],
        dims: vec![2, 1, 3],
        strides: vec![1, isize::MAX, 2],
        offset: 0,
    };
    add(&Exec::serial(), 2.0, &a.view(), 3.0, &mut c.view_mut()).unwrap();
    for_each_index(&[2, 1, 3], |i| assert_eq!(c.get(i), 2.0 * a.get(i) + 3.0));
    // alpha == 0 through a plan scales C, same layout.
    let cfg = DotGeneral::new(&[], &[], &[], &[]);
    let x = T::<f64>::new(&[2], 1);
    let y = T::<f64>::new(&[1, 3], 2);
    for strategy in [Strategy::PermuteGemm, Strategy::Tblis] {
        let mut c = T::<f64> {
            data: vec![1.0; 6],
            dims: vec![2, 1, 3],
            strides: vec![1, isize::MAX, 2],
            offset: 0,
        };
        let plan = ContractPlan::<f64>::new(
            &cfg,
            (&x.dims, &x.strides),
            (&y.dims, &y.strides),
            (&c.dims, &c.strides),
            (Conj::No, Conj::No),
            strategy,
            Flags::default(),
        )
        .unwrap();
        plan.execute(
            &Exec::serial(),
            0.0,
            &x.view(),
            &y.view(),
            4.0,
            &mut c.view_mut(),
        )
        .unwrap();
        assert!(c.data.iter().all(|&v| v == 4.0), "{strategy:?}");
    }
}

#[test]
fn overflowing_element_counts_are_shape_errors_for_both_strategies() {
    let d = [1usize << 33, 1 << 33];
    let s = [1isize, 1 << 33];
    let cfg = DotGeneral::new(&[0, 1], &[0, 1], &[], &[]);
    for strategy in [Strategy::PermuteGemm, Strategy::Tblis] {
        let e = ContractPlan::<f64>::new(
            &cfg,
            (&d, &s),
            (&d, &s),
            (&[], &[]),
            (Conj::No, Conj::No),
            strategy,
            Flags::default(),
        );
        assert!(matches!(e, Err(Error::Shape(_))), "{strategy:?}: {e:?}");
    }
}

#[test]
fn all_batch_contractions_run_elementwise() {
    let cfg = DotGeneral::new(&[], &[], &[0, 1], &[1, 0]);
    let a = T::<f64>::new(&[3, 4], 1);
    let b = T::<f64>::new(&[4, 3], 2);
    let c0 = T::<f64>::new(&[3, 4], 3);
    let want = reference(&cfg, 1.5, &a, Conj::No, &b, Conj::No, 0.5, &c0);
    let mut c = c0.clone();
    let plan = ContractPlan::<f64>::new(
        &cfg,
        (&a.dims, &a.strides),
        (&b.dims, &b.strides),
        (&c.dims, &c.strides),
        (Conj::No, Conj::No),
        Strategy::Auto,
        Flags::default(),
    )
    .unwrap();
    assert_eq!(plan.selected(), Selected::Elementwise);
    plan.execute(
        &Exec::serial(),
        1.5,
        &a.view(),
        &b.view(),
        0.5,
        &mut c.view_mut(),
    )
    .unwrap();
    assert!(rel_err(&c, &want) < 1e-14);
}

#[test]
fn empty_problems_are_copy_free() {
    let cfg = DotGeneral::new(&[], &[], &[], &[]);
    let a = T::<f64>::new(&[2, 0, 3], 1);
    let a = T {
        strides: vec![1, 2, 2],
        ..a
    };
    let b = T::<f64>::new(&[4], 2);
    let c = T::<f64>::new(&[2, 0, 3, 4], 3);
    let plan = ContractPlan::<f64>::new(
        &cfg,
        (&a.dims, &a.strides),
        (&b.dims, &b.strides),
        (&c.dims, &c.strides),
        (Conj::No, Conj::No),
        Strategy::PermuteGemm,
        Flags {
            no_materialize: true,
        },
    )
    .unwrap();
    assert_eq!(
        plan.selected(),
        Selected::PermuteGemm {
            materialized: [false; 3]
        }
    );
}
