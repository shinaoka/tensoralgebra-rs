use num_complex::Complex64;
use tensorcontract::Element;
use tprims_blas::{Conj, Scalar};
use tprims_contract::{ContractPlan, DotGeneral, Error, Flags, Selected, Strategy};
use tprims_exec::{Exec, Pool};

mod common;
use common::*;

pub const STRATEGIES: [Strategy; 3] = [Strategy::PermuteGemm, Strategy::Tblis, Strategy::Auto];

struct Case {
    name: &'static str,
    a: Vec<usize>,
    b: Vec<usize>,
    cfg: DotGeneral,
}

fn corpus() -> Vec<Case> {
    let dg =
        |lc: &[usize], rc: &[usize], lb: &[usize], rb: &[usize]| DotGeneral::new(lc, rc, lb, rb);
    vec![
        Case {
            name: "matmul ij,jk",
            a: vec![5, 4],
            b: vec![4, 3],
            cfg: dg(&[1], &[0], &[], &[]),
        },
        Case {
            name: "batched bij,bjk",
            a: vec![3, 5, 4],
            b: vec![3, 4, 2],
            cfg: dg(&[2], &[1], &[0], &[0]),
        },
        Case {
            name: "two contracted axes in different orders",
            a: vec![3, 4, 5],
            b: vec![5, 6, 4],
            cfg: dg(&[1, 2], &[2, 0], &[], &[]),
        },
        Case {
            name: "outer product",
            a: vec![3, 2],
            b: vec![4],
            cfg: dg(&[], &[], &[], &[]),
        },
        Case {
            name: "full contraction to rank 0",
            a: vec![3, 4],
            b: vec![4, 3],
            cfg: dg(&[0, 1], &[1, 0], &[], &[]),
        },
        Case {
            name: "hadamard batch only",
            a: vec![3, 4],
            b: vec![4, 3],
            cfg: dg(&[], &[], &[0, 1], &[1, 0]),
        },
        Case {
            name: "4-index network ijkl,klmn",
            a: vec![2, 3, 4, 2],
            b: vec![4, 2, 3, 2],
            cfg: dg(&[2, 3], &[0, 1], &[], &[]),
        },
        Case {
            name: "batch in the middle",
            a: vec![4, 3, 5],
            b: vec![5, 3, 2],
            cfg: dg(&[2], &[0], &[1], &[1]),
        },
        Case {
            name: "empty free extent",
            a: vec![0, 3],
            b: vec![3, 2],
            cfg: dg(&[1], &[0], &[], &[]),
        },
        Case {
            name: "empty contraction",
            a: vec![3, 0],
            b: vec![0, 2],
            cfg: dg(&[1], &[0], &[], &[]),
        },
    ]
}

fn out_dims(cfg: &DotGeneral, a: &[usize], b: &[usize]) -> Vec<usize> {
    cfg.validate(a, b).unwrap().out_dims
}

#[allow(clippy::too_many_arguments)]
fn run<S: Scalar>(
    exec: &Exec<'_>,
    strategy: Strategy,
    case: &Case,
    layout: u8,
    ca: Conj,
    cb: Conj,
    beta: S,
) -> Selected {
    let base_a = T::<S>::new(&case.a, 1);
    let base_b = T::<S>::new(&case.b, 2);
    // layout 0: column-major, 1: axes reversed in storage order, 2: negative strides
    let rev = |d: usize| (0..d).rev().collect::<Vec<_>>();
    let (a, b) = match layout {
        0 => (base_a, base_b),
        1 => (
            base_a.restride(&rev(case.a.len())),
            base_b.restride(&rev(case.b.len())),
        ),
        _ => (base_a.reversed(), base_b.reversed()),
    };
    let od = out_dims(&case.cfg, &case.a, &case.b);
    let c0 = if layout == 2 {
        T::<S>::new(&od, 3).reversed()
    } else {
        T::<S>::new(&od, 3)
    };
    let alpha = <S as Element>::from_parts(
        tensorcontract::Real::from_f64(0.75),
        tensorcontract::Real::from_f64(0.25),
    );
    let want = reference(&case.cfg, alpha, &a, ca, &b, cb, beta, &c0);
    let mut c = c0.clone();
    let plan = ContractPlan::<S>::new(
        &case.cfg,
        (&a.dims, &a.strides),
        (&b.dims, &b.strides),
        (&c.dims, &c.strides),
        (ca, cb),
        strategy,
        Flags::default(),
    )
    .unwrap();
    plan.execute(exec, alpha, &a.view(), &b.view(), beta, &mut c.view_mut())
        .unwrap();
    let tol = if std::mem::size_of::<<S as Scalar>::Re>() == 4 {
        1e-4
    } else {
        1e-12
    };
    let err = rel_err(&c, &want);
    assert!(
        err < tol,
        "{strategy:?} {} layout{layout} {ca:?}/{cb:?}: {err}",
        case.name
    );
    plan.selected()
}

#[test]
fn corpus_matches_the_reference() {
    for strategy in STRATEGIES {
        for case in corpus() {
            for layout in 0..3u8 {
                run::<f64>(
                    &Exec::serial(),
                    strategy,
                    &case,
                    layout,
                    Conj::No,
                    Conj::No,
                    1.5,
                );
                run::<Complex64>(
                    &Exec::serial(),
                    strategy,
                    &case,
                    layout,
                    Conj::Yes,
                    Conj::No,
                    Complex64::new(0.0, 0.0),
                );
                run::<Complex64>(
                    &Exec::serial(),
                    strategy,
                    &case,
                    layout,
                    Conj::No,
                    Conj::Yes,
                    Complex64::new(1.0, -1.0),
                );
            }
        }
    }
}

#[test]
fn larger_contractions_on_a_pool_match() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let big = Case {
        name: "big batched",
        a: vec![8, 40, 30],
        b: vec![8, 30, 50],
        cfg: DotGeneral::new(&[2], &[1], &[0], &[0]),
    };
    let net = Case {
        name: "big network",
        a: vec![6, 7, 8, 9],
        b: vec![8, 9, 5, 4],
        cfg: DotGeneral::new(&[2, 3], &[0, 1], &[], &[]),
    };
    for strategy in STRATEGIES {
        for layout in 0..3u8 {
            run::<f64>(&exec, strategy, &big, layout, Conj::No, Conj::No, 0.5);
            run::<f64>(&exec, strategy, &net, layout, Conj::No, Conj::No, 0.0);
        }
    }
}

#[test]
fn copy_free_detection_and_no_materialize() {
    // Column-major matmul: every group fuses, nothing is copied.
    let case = &corpus()[0];
    let sel = run::<f64>(
        &Exec::serial(),
        Strategy::PermuteGemm,
        case,
        0,
        Conj::No,
        Conj::No,
        0.0,
    );
    assert_eq!(
        sel,
        Selected::PermuteGemm {
            materialized: [false, false, false]
        }
    );
    // Contracted axes in different orders on A and B cannot both fuse.
    let bad = &corpus()[2];
    let a = T::<f64>::new(&bad.a, 1);
    let b = T::<f64>::new(&bad.b, 2);
    let od = out_dims(&bad.cfg, &bad.a, &bad.b);
    let c = T::<f64>::new(&od, 3);
    let strict = Flags {
        no_materialize: true,
    };
    let e = ContractPlan::<f64>::new(
        &bad.cfg,
        (&a.dims, &a.strides),
        (&b.dims, &b.strides),
        (&c.dims, &c.strides),
        (Conj::No, Conj::No),
        Strategy::PermuteGemm,
        strict,
    );
    assert!(matches!(e, Err(Error::WouldMaterialize { .. })), "{e:?}");
    let ok = ContractPlan::<f64>::new(
        &bad.cfg,
        (&a.dims, &a.strides),
        (&b.dims, &b.strides),
        (&c.dims, &c.strides),
        (Conj::No, Conj::No),
        Strategy::PermuteGemm,
        Flags::default(),
    )
    .unwrap();
    assert!(
        matches!(ok.selected(), Selected::PermuteGemm { materialized } if materialized.iter().any(|&m| m))
    );
}

#[test]
fn validation_errors_are_typed_and_nothing_is_written() {
    let a = T::<f64>::new(&[3, 4], 1);
    let b = T::<f64>::new(&[4, 2], 2);
    let cfg = DotGeneral::new(&[1], &[0], &[], &[]);
    assert!(matches!(
        DotGeneral::new(&[1, 1], &[0, 0], &[], &[]).validate(&[3, 4], &[4, 2]),
        Err(Error::Config(_))
    ));
    assert!(matches!(
        DotGeneral::new(&[2], &[0], &[], &[]).validate(&[3, 4], &[4, 2]),
        Err(Error::Config(_))
    ));
    assert!(matches!(
        DotGeneral::new(&[0], &[0], &[], &[]).validate(&[3, 4], &[4, 2]),
        Err(Error::Shape(_))
    ));
    for strategy in [Strategy::PermuteGemm, Strategy::Tblis] {
        let mut c = T::<f64> {
            data: vec![7.0; 3],
            dims: vec![3, 2],
            strides: vec![1, 0],
            offset: 0,
        };
        let e = ContractPlan::<f64>::new(
            &cfg,
            (&a.dims, &a.strides),
            (&b.dims, &b.strides),
            (&c.dims, &c.strides),
            (Conj::No, Conj::No),
            strategy,
            Flags::default(),
        );
        assert_eq!(e.err(), Some(Error::AliasedOutput), "{strategy:?}");
        let plan = ContractPlan::<f64>::new(
            &cfg,
            (&a.dims, &a.strides),
            (&b.dims, &b.strides),
            (&[3, 2], &[1, 3]),
            (Conj::No, Conj::No),
            strategy,
            Flags::default(),
        )
        .unwrap();
        let e = plan.execute(
            &Exec::serial(),
            1.0,
            &a.view(),
            &b.view(),
            0.0,
            &mut c.view_mut(),
        );
        assert!(matches!(e, Err(Error::LayoutMismatch(_))), "{e:?}");
        assert!(c.data.iter().all(|&x| x == 7.0));
    }
}

#[test]
fn auto_uses_tblis_exactly_when_permute_gemm_would_copy() {
    // Phase 1e P2 (docs/decision-log.md): permute+GEMM wins copy-free
    // problems, TBLIS-style wins the ones permute+GEMM must copy.
    let plan = |case: &Case, flags: Flags| {
        let a = T::<f64>::new(&case.a, 1);
        let b = T::<f64>::new(&case.b, 2);
        let od = out_dims(&case.cfg, &case.a, &case.b);
        let c = T::<f64>::new(&od, 3);
        ContractPlan::<f64>::new(
            &case.cfg,
            (&a.dims, &a.strides),
            (&b.dims, &b.strides),
            (&c.dims, &c.strides),
            (Conj::No, Conj::No),
            Strategy::Auto,
            flags,
        )
    };
    let fusable = &corpus()[0];
    assert_eq!(
        plan(fusable, Flags::default()).unwrap().selected(),
        Selected::PermuteGemm {
            materialized: [false, false, false]
        }
    );
    let copying = &corpus()[2];
    assert_eq!(
        plan(copying, Flags::default()).unwrap().selected(),
        Selected::Tblis
    );
    // TBLIS-style copies nothing, so Auto satisfies `no_materialize` there.
    let strict = Flags {
        no_materialize: true,
    };
    assert_eq!(plan(copying, strict).unwrap().selected(), Selected::Tblis);
    assert!(matches!(
        plan(fusable, strict).unwrap().selected(),
        Selected::PermuteGemm { .. }
    ));
}
