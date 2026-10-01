//! Non-fusable tensor contraction with the downstream crate's own kernels.
use std::cell::Cell;

use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{Conj, EngineChoice, GemmConfig, KernelCatalog, SelectError};
use tprims_contract::{ContractPlan, DotGeneral, Error, Flags, Selected, Strategy};
use tprims_contract_testkit::custom_kernels as own;
use tprims_exec::Exec;

fn catalog(list: &'static [&'static tprims_kernel::KernelFamily<f64>]) -> KernelCatalog<f64> {
    // SAFETY: see `selector_blas::catalog`.
    unsafe { KernelCatalog::<f64>::from_static_families(list) }.unwrap()
}

/// The typed selection error a contraction error carries as its source.
fn typed(e: &Error) -> SelectError {
    match e {
        Error::Backend(b) => b
            .downcast_ref::<SelectError>()
            .cloned()
            .unwrap_or_else(|| panic!("not a selection error: {e:?}")),
        _ => panic!("not a typed selection error: {e:?}"),
    }
}

fn ints(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|x| ((x * 5 + seed * 3) % 13) as f64 - 6.0)
        .collect()
}
fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::new();
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d.max(1) as isize;
    }
    s
}

// C[i, n] = sum_{j<4, k<5} A[i, j, k] * B[k, n, j]: two contracted axes in
// permuted order, so permute+GEMM would have to copy an operand.
const A_DIMS: [usize; 3] = [3, 4, 5];
const B_DIMS: [usize; 3] = [5, 6, 4];
const C_DIMS: [usize; 2] = [3, 6];

fn cfg() -> DotGeneral {
    DotGeneral::new(&[1, 2], &[2, 0], &[], &[])
}

fn oracle(a: &[f64], b: &[f64], alpha: f64, beta: f64, c: &mut [f64]) {
    for i in 0..3 {
        for n in 0..6 {
            let mut s = 0.0;
            for j in 0..4 {
                for k in 0..5 {
                    s += a[i + 3 * j + 12 * k] * b[k + 5 * n + 30 * j];
                }
            }
            let at = i + 3 * n;
            c[at] = alpha * s + if beta == 0.0 { 0.0 } else { beta * c[at] };
        }
    }
}

fn plan(
    cat: &KernelCatalog<f64>,
    gemm: &GemmConfig,
    strategy: Strategy,
    selector: impl FnOnce(
        &tprims_blas::SelectionContext<'_>,
        &[tprims_blas::KernelCandidate<f64>],
    ) -> Result<tprims_blas::KernelHandle<f64>, SelectError>,
) -> Result<ContractPlan<f64>, Error> {
    let (sa, sb, sc) = (col_major(&A_DIMS), col_major(&B_DIMS), col_major(&C_DIMS));
    ContractPlan::<f64>::new_with_selector(
        &Exec::serial(),
        gemm,
        cat,
        selector,
        &cfg(),
        (&A_DIMS, &sa),
        (&B_DIMS, &sb),
        (&C_DIMS, &sc),
        (Conj::No, Conj::No),
        strategy,
        Flags::default(),
    )
}

fn run(plan: &ContractPlan<f64>, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    let (a, b) = (ints(60, 1), ints(120, 2));
    let mut c = ints(18, 3);
    let mut want = c.clone();
    oracle(&a, &b, alpha, beta, &mut want);
    let (sa, sb, sc) = (col_major(&A_DIMS), col_major(&B_DIMS), col_major(&C_DIMS));
    let av = StridedView::new(&a, &A_DIMS, &sa, 0).unwrap();
    let bv = StridedView::new(&b, &B_DIMS, &sb, 0).unwrap();
    let mut cv = StridedViewMut::new(&mut c, &C_DIMS, &sc, 0).unwrap();
    plan.execute(&Exec::serial(), alpha, &av, &bv, beta, &mut cv)
        .unwrap();
    (c, want)
}

#[test]
fn a_non_fusable_contraction_runs_on_a_custom_kernel() {
    let cat = catalog(own::f64_families());
    let calls = Cell::new(0);
    let p = plan(
        &cat,
        &GemmConfig::default(),
        Strategy::Auto,
        |ctx, cands| {
            calls.set(calls.get() + 1);
            assert_eq!((ctx.stats.m, ctx.stats.n, ctx.stats.k), (3, 6, 20));
            assert_eq!(ctx.operands[0].extents, &[3, 4, 5]);
            assert_eq!(cands.len(), 2);
            Ok(cands[1].handle)
        },
    )
    .unwrap();
    assert_eq!(p.selected(), Selected::Tblis);
    let report = p.selected_gemm().unwrap().unwrap();
    assert_eq!(report.family_id, Some("custom.f64.3x4"));
    assert_eq!(report.origin, Some(own::ORIGIN));
    for (alpha, beta) in [(1.0, 0.0), (2.0, -1.0)] {
        let (got, want) = run(&p, alpha, beta);
        assert_eq!(got, want);
    }
    assert_eq!(calls.get(), 1, "execution never calls the selector");
}

#[test]
fn the_selected_plan_outlives_its_selector_and_catalog() {
    let p = {
        let cat = catalog(own::f64_families());
        let captured = [String::from("captured state"), String::from("dropped")];
        plan(&cat, &GemmConfig::default(), Strategy::Tblis, |_, cands| {
            assert_eq!(captured.len(), 2);
            Ok(cands[0].handle)
        })
        .unwrap()
        // catalog and captured state are dropped here
    };
    for _ in 0..3 {
        let (got, want) = run(&p, 1.0, 0.0);
        assert_eq!(got, want);
    }
    assert_eq!(
        p.selected_gemm().unwrap().unwrap().family_id,
        Some("custom.f64.2x2")
    );
    // The family was never registered process-wide: nothing could have looked
    // it up by id.
    assert!(matches!(
        tprims_kernel::Registry::select::<f64>(
            "custom.f64.2x2",
            tprims_kernel::CpuFeatures::detect()
        ),
        Err(tprims_kernel::SelectError::UnknownId { .. })
    ));
}

#[test]
fn incompatible_strategies_engines_and_choices_are_typed_errors() {
    let cat = catalog(own::f64_families());
    let other = catalog(own::f64_families());
    let g = GemmConfig::default();
    let sel_err = |r: Result<ContractPlan<f64>, Error>| match r {
        Err(e) => typed(&e),
        Ok(_) => panic!("accepted"),
    };
    // Permute+GEMM computes with faer; it cannot honour a selector.
    assert!(matches!(
        sel_err(plan(&cat, &g, Strategy::PermuteGemm, |_, _| unreachable!())),
        SelectError::EngineUnsupported { .. }
    ));
    let faer = GemmConfig {
        engine: EngineChoice::Faer,
        ..Default::default()
    };
    assert!(matches!(
        sel_err(plan(&cat, &faer, Strategy::Auto, |_, _| unreachable!())),
        SelectError::EngineUnsupported { .. }
    ));
    let forced = GemmConfig {
        kernel: tprims_kernel::KernelChoice::Id("ref.f64.real.4x4".into()),
        ..Default::default()
    };
    assert!(matches!(
        sel_err(plan(&cat, &forced, Strategy::Auto, |_, _| unreachable!())),
        SelectError::Incompatible { .. }
    ));
    let foreign = other.get("custom.f64.2x2").unwrap();
    assert!(matches!(
        sel_err(plan(&cat, &g, Strategy::Auto, |_, _| Ok(foreign))),
        SelectError::ForeignHandle { .. }
    ));
    assert!(matches!(
        sel_err(plan(&cat, &g, Strategy::Auto, |_, _| Err(
            SelectError::SelectorFailed { reason: "x".into() }
        ))),
        SelectError::SelectorFailed { .. }
    ));
    let masked = catalog(own::f64_impossible_families());
    assert!(matches!(
        sel_err(plan(&masked, &g, Strategy::Auto, |_, _| unreachable!())),
        SelectError::NoCandidates { .. }
    ));
}

#[test]
fn all_batch_and_zero_size_problems_still_select_or_refuse() {
    let cat = catalog(own::f64_families());
    let other = catalog(own::f64_families());
    // Hadamard product: Auto would pick the elementwise pass, which has no
    // kernel to select, so it is refused; Tblis runs it on the packed driver.
    let had = DotGeneral::new(&[], &[], &[0], &[0]);
    let dims = [4usize];
    let st = [1isize];
    let build = |strategy, sel: &mut dyn FnMut() -> bool| {
        ContractPlan::<f64>::new_with_selector(
            &Exec::serial(),
            &GemmConfig::default(),
            &cat,
            |_, cands| {
                let _ = sel();
                Ok(cands[0].handle)
            },
            &had,
            (&dims, &st),
            (&dims, &st),
            (&dims, &st),
            (Conj::No, Conj::No),
            strategy,
            Flags::default(),
        )
    };
    let mut called = false;
    assert!(matches!(
        build(Strategy::Auto, &mut || {
            called = true;
            true
        }),
        Err(e) if matches!(typed(&e), SelectError::EngineUnsupported { .. })
    ));
    assert!(!called);
    assert!(build(Strategy::Tblis, &mut || true).is_ok());

    // A zero extent: the plan is still selected and validated, and a bad
    // choice is rejected before anything runs.
    let (ad, bd, cd) = ([0usize, 4], [4usize, 3], [0usize, 3]);
    let (sa, sb, sc) = ([1isize, 1], [1isize, 4], [1isize, 1]);
    let foreign = other.get("custom.f64.2x2").unwrap();
    let zero = |h: tprims_blas::KernelHandle<f64>| {
        ContractPlan::<f64>::new_with_selector(
            &Exec::serial(),
            &GemmConfig::default(),
            &cat,
            move |_, _| Ok(h),
            &DotGeneral::new(&[1], &[0], &[], &[]),
            (&ad, &sa),
            (&bd, &sb),
            (&cd, &sc),
            (Conj::No, Conj::No),
            Strategy::Auto,
            Flags::default(),
        )
    };
    assert!(matches!(
        zero(foreign),
        Err(e) if matches!(typed(&e), SelectError::ForeignHandle { .. })
    ));
    let ok = zero(cat.get("custom.f64.2x2").unwrap()).unwrap();
    let a: Vec<f64> = vec![];
    let b = ints(12, 1);
    let mut c: Vec<f64> = vec![];
    ok.execute(
        &Exec::serial(),
        1.0,
        &StridedView::new(&a, &ad, &sa, 0).unwrap(),
        &StridedView::new(&b, &bd, &sb, 0).unwrap(),
        0.0,
        &mut StridedViewMut::new(&mut c, &cd, &sc, 0).unwrap(),
    )
    .unwrap();
}

#[test]
fn a_selected_plan_is_bound_to_its_storage_dtype() {
    use tensorcontract::{Layout, Operand, Plan};
    let cat = catalog(own::f64_families());
    let l = Layout::col_major(&[6, 6]);
    let (ia, ib, id) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    let ops = [
        Operand::new(&l, &ia),
        Operand::new(&l, &ib),
        Operand::new(&l, &id),
        Operand::new(&l, &id),
    ];
    let plan = Plan::new(ops[0], ops[1], None, ops[3])
        .unwrap()
        .with_selector::<f64, _>(ops, &cat, |_, c| Ok(c[0].handle))
        .unwrap();
    assert_eq!(
        plan.resolved::<f64>().unwrap().family().id,
        "custom.f64.2x2"
    );
    assert!(matches!(
        plan.resolved::<tprims_kernel::C64>(),
        Err(SelectError::DtypeMismatch { dtype: "c64", .. })
    ));
    // A clone keeps the choice rather than silently falling back.
    assert_eq!(
        plan.clone().resolved::<f64>().unwrap().family().id,
        "custom.f64.2x2"
    );
    // A forced id after the selector is ambiguous.
    assert!(plan
        .with_kernel(tensorcontract::KernelChoice::Id("ref.f64.real.4x4".into()))
        .is_err());
}
