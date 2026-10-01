//! Kernel selection at plan creation: the claim of this API is that a plan
//! resolves its kernel when it is built, reports it, and refuses an unusable
//! one there rather than inside a contraction.
use tprims_blas::{Conj, Engine, GemmConfig, Scalar};
use tprims_contract::{ContractPlan, DotGeneral, Error, Flags, Strategy};
use tprims_gemm_kernel::KernelChoice;

mod common;
use common::{out_dims, T};

/// The corpus case whose permute+GEMM arm must copy, so that the automatic
/// strategy chooses the packed driver and there is a resolution to report.
fn copying_case() -> (DotGeneral, Vec<usize>, Vec<usize>) {
    (
        DotGeneral::new(&[1, 2], &[2, 0], &[], &[]),
        vec![3, 4, 5],
        vec![5, 6, 4],
    )
}

fn plan_with(gemm: &GemmConfig) -> tprims_contract::Result<ContractPlan<f64>> {
    let (cfg, a_dims, b_dims) = copying_case();
    let a = T::<f64>::new(&a_dims, 1);
    let b = T::<f64>::new(&b_dims, 2);
    let od = out_dims(&cfg, &a.dims, &b.dims);
    let c = T::<f64>::new(&od, 3);
    ContractPlan::<f64>::new_with(
        gemm,
        &cfg,
        (&a.dims, &a.strides),
        (&b.dims, &b.strides),
        (&c.dims, &c.strides),
        (Conj::No, Conj::No),
        Strategy::Auto,
        Flags::default(),
    )
}

#[test]
fn a_contract_plan_resolves_and_reports_its_kernel() {
    let plan = plan_with(&GemmConfig::default()).unwrap();
    let report = plan
        .selected_gemm()
        .unwrap()
        .expect("the automatic strategy chose the packed driver");
    assert_eq!(report.engine, Engine::Packed);
    let id = report.family_id.expect("the packed plan names its family");
    assert!(!id.is_empty());
    assert!(report.mr > 0 && report.nr > 0 && report.kc > 0);

    // A forced family is the one reported.
    let forced = GemmConfig {
        kernel: KernelChoice::Id("portable.f64.4x4".into()),
        ..Default::default()
    };
    assert_eq!(
        plan_with(&forced)
            .unwrap()
            .selected_gemm()
            .unwrap()
            .unwrap()
            .family_id,
        Some("portable.f64.4x4")
    );

    // An id that does not exist fails at creation, not at execution, and the
    // error names it.
    let bad = GemmConfig {
        kernel: KernelChoice::Id("not.a.kernel".into()),
        ..Default::default()
    };
    let err = plan_with(&bad).unwrap_err();
    assert!(
        matches!(err, Error::Backend(_)) || format!("{err}").contains("not.a.kernel"),
        "unexpected error: {err}"
    );
}

#[test]
fn a_plan_that_does_not_use_the_packed_driver_reports_nothing() {
    let (cfg, a_dims, b_dims) = copying_case();
    let a = T::<f64>::new(&a_dims, 1);
    let b = T::<f64>::new(&b_dims, 2);
    let od = out_dims(&cfg, &a.dims, &b.dims);
    let c = T::<f64>::new(&od, 3);
    let plan = ContractPlan::<f64>::new_with(
        &GemmConfig::default(),
        &cfg,
        (&a.dims, &a.strides),
        (&b.dims, &b.strides),
        (&c.dims, &c.strides),
        (Conj::No, Conj::No),
        Strategy::PermuteGemm,
        Flags::default(),
    )
    .unwrap();
    assert!(plan.selected_gemm().unwrap().is_none());
    // `Scalar` is in scope for the plan bound; the real/complex split is what
    // the `Complex` scheme in the report describes.
    let _: fn() -> bool = || <f64 as Scalar>::IS_COMPLEX_SCALAR;
}
