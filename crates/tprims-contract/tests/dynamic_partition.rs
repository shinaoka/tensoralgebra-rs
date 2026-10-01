//! `PartitionPolicy::DynamicTiles` through `ContractPlan`: a plan-level option
//! of the packed driver, validated when the plan is built, honoured on a
//! non-fusable contraction, and refused by strategies that cannot use it.
use tprims_blas::{Conj, EngineChoice, GemmConfig};
use tprims_contract::{ContractPlan, DotGeneral, Error, Flags, Selected, Strategy};
use tprims_exec::{Exec, Pool};
use tprims_gemm_kernel::{KernelChoice, PartitionOpts, PartitionPolicy, SelectError};

mod common;
use common::{out_dims, T};

/// Two contracted axes in permuted order: permute+GEMM would copy.
fn copying_case() -> (DotGeneral, Vec<usize>, Vec<usize>) {
    (
        DotGeneral::new(&[1, 2], &[2, 0], &[], &[]),
        vec![24, 20, 9],
        vec![9, 30, 20],
    )
}

fn dyn_cfg(job_m: usize, job_n: usize) -> GemmConfig {
    GemmConfig {
        engine: EngineChoice::Packed,
        kernel: KernelChoice::Id("portable.f64.4x4".into()),
        partition: PartitionPolicy::DynamicTiles { job_m, job_n },
        ..Default::default()
    }
}

fn plan(cfg: &GemmConfig, strategy: Strategy) -> Result<ContractPlan<f64>, Error> {
    let (dg, ad, bd) = copying_case();
    let (a, b) = (T::<f64>::new(&ad, 1), T::<f64>::new(&bd, 2));
    let od = out_dims(&dg, &a.dims, &b.dims);
    let c = T::<f64>::new(&od, 3);
    ContractPlan::<f64>::new_with(
        cfg,
        &dg,
        (&a.dims, &a.strides),
        (&b.dims, &b.strides),
        (&c.dims, &c.strides),
        (Conj::No, Conj::No),
        strategy,
        Flags::default(),
    )
}

fn run(plan: &ContractPlan<f64>, exec: &Exec<'_>) -> Vec<f64> {
    let (_, ad, bd) = copying_case();
    let (dg, _, _) = copying_case();
    let (a, b) = (T::<f64>::new(&ad, 1), T::<f64>::new(&bd, 2));
    let od = out_dims(&dg, &a.dims, &b.dims);
    let mut c = T::<f64>::new(&od, 3);
    plan.execute(exec, 1.0, &a.view(), &b.view(), 0.5, &mut c.view_mut())
        .unwrap();
    c.data
}

fn typed(e: &Error) -> &SelectError {
    match e {
        Error::Backend(b) => b
            .downcast_ref::<SelectError>()
            .unwrap_or_else(|| panic!("not a selection error: {e:?}")),
        _ => panic!("not a typed selection error: {e:?}"),
    }
}

#[test]
fn a_non_fusable_contraction_runs_dynamically_and_reports_the_policy() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let stat = plan(
        &GemmConfig {
            engine: EngineChoice::Packed,
            kernel: KernelChoice::Id("portable.f64.4x4".into()),
            ..Default::default()
        },
        Strategy::Auto,
    )
    .unwrap();
    let dynp = plan(&dyn_cfg(8, 8), Strategy::Auto).unwrap();
    assert_eq!(dynp.selected(), Selected::Tblis);
    let report = dynp.selected_gemm().unwrap().unwrap();
    assert_eq!(
        report.partition,
        PartitionPolicy::DynamicTiles { job_m: 8, job_n: 8 }
    );
    let d = report.dynamic.expect("assignment is reported");
    assert_eq!((d.job_m, d.job_n, d.row_bands), (8, 8, 3));
    assert!(stat.selected_gemm().unwrap().unwrap().dynamic.is_none());
    for e in [Exec::serial(), exec] {
        assert_eq!(run(&dynp, &e), run(&stat, &Exec::serial()));
    }
}

#[test]
fn a_partition_request_selects_the_packed_driver_under_auto() {
    // Without an explicit engine or kernel, a partition policy alone is a
    // requirement, so the automatic strategy plans the packed driver.
    // 48 x 48 is a whole number of register blocks for every built-in family
    // (4x4, 8x6, 8x4, NEON 16x3, ...), since job extents must be multiples of
    // the selected family's logical MR and NR.
    let cfg = GemmConfig {
        partition: PartitionPolicy::DynamicTiles {
            job_m: 48,
            job_n: 48,
        },
        ..Default::default()
    };
    let p = plan(&cfg, Strategy::Auto).unwrap();
    assert_eq!(p.selected(), Selected::Tblis);
}

#[test]
fn invalid_or_unsupported_policies_are_typed_errors_at_planning() {
    for (jm, jn) in [(0, 8), (8, 0), (6, 8), (8, 6)] {
        let e = plan(&dyn_cfg(jm, jn), Strategy::Auto).expect_err("rejected");
        assert!(
            matches!(typed(&e), SelectError::Incompatible { .. }),
            "{jm}x{jn}"
        );
    }
    let aligned = GemmConfig {
        partition_opts: PartitionOpts {
            align_c_lines: true,
        },
        ..dyn_cfg(8, 8)
    };
    let e = plan(&aligned, Strategy::Tblis).expect_err("rejected");
    assert!(matches!(typed(&e), SelectError::Incompatible { .. }));
    // Permute+GEMM computes with faer: it cannot honour the policy.
    assert!(plan(&dyn_cfg(8, 8), Strategy::PermuteGemm).is_err());
    // A Hadamard-only problem is an elementwise pass with no output grid.
    let had = DotGeneral::new(&[], &[], &[0], &[0]);
    let r = ContractPlan::<f64>::new_with(
        &GemmConfig {
            partition: PartitionPolicy::DynamicTiles { job_m: 8, job_n: 8 },
            ..Default::default()
        },
        &had,
        (&[4], &[1]),
        (&[4], &[1]),
        (&[4], &[1]),
        (Conj::No, Conj::No),
        Strategy::Auto,
        Flags::default(),
    );
    assert!(matches!(
        typed(&r.expect_err("rejected")),
        SelectError::EngineUnsupported { .. }
    ));
}
