//! The neutral contraction interface, exercised through trait objects with two
//! independent implementations: the tprims backend and the test-only naive one.

use std::sync::atomic::{AtomicUsize, Ordering};

use num_complex::{Complex32, Complex64};
use tensorcontract::Element;
use tprims_blas::Scalar;
use tprims_contract::{ExecHost, Strategy, TprimsBackend};
use tprims_contract_testkit::{Field, NaiveBackend};
use tprims_contract_traits::{
    Conj, ContractionBackend, DotGeneral, Error, HostError, HostExecution, Layout, Par,
    PlanningBudget, Problem, Requirements,
};
use tprims_exec::{Exec, Pool};

mod common;
use common::*;

fn conj_pair(ca: tprims_blas::Conj, cb: tprims_blas::Conj) -> (Conj, Conj) {
    let f = |c| {
        if c == tprims_blas::Conj::Yes {
            Conj::Yes
        } else {
            Conj::No
        }
    };
    (f(ca), f(cb))
}

fn problem<S: Scalar>(
    cfg: &DotGeneral,
    a: &T<S>,
    b: &T<S>,
    c: &T<S>,
    conj: (Conj, Conj),
) -> Problem {
    Problem::new(
        cfg.clone(),
        Layout::new(&a.dims, &a.strides),
        Layout::new(&b.dims, &b.strides),
        Layout::new(&c.dims, &c.strides),
        conj,
    )
}

fn backends<S: Scalar + Field>() -> Vec<Box<dyn ContractionBackend<S>>> {
    let t = |strategy| TprimsBackend {
        strategy,
        ..TprimsBackend::default()
    };
    vec![
        Box::new(t(Strategy::Auto)),
        Box::new(t(Strategy::PermuteGemm)),
        Box::new(t(Strategy::Tblis)),
        Box::new(NaiveBackend),
    ]
}

fn scalar<S: Scalar>(re: f64, im: f64) -> S {
    <S as Element>::from_parts(
        tensorcontract::Real::from_f64(re),
        tensorcontract::Real::from_f64(im),
    )
}

struct Case {
    name: &'static str,
    a: Vec<usize>,
    b: Vec<usize>,
    cfg: DotGeneral,
}

fn corpus() -> Vec<Case> {
    let dg = DotGeneral::new;
    vec![
        Case {
            name: "matmul",
            a: vec![5, 4],
            b: vec![4, 3],
            cfg: dg(&[1], &[0], &[], &[]),
        },
        Case {
            name: "batched",
            a: vec![3, 5, 4],
            b: vec![3, 4, 2],
            cfg: dg(&[2], &[1], &[0], &[0]),
        },
        Case {
            name: "two contracted",
            a: vec![3, 4, 5],
            b: vec![5, 6, 4],
            cfg: dg(&[1, 2], &[2, 0], &[], &[]),
        },
        Case {
            name: "outer",
            a: vec![3, 2],
            b: vec![4],
            cfg: dg(&[], &[], &[], &[]),
        },
        Case {
            name: "scalar result",
            a: vec![3, 4],
            b: vec![4, 3],
            cfg: dg(&[0, 1], &[1, 0], &[], &[]),
        },
        Case {
            name: "hadamard",
            a: vec![3, 4],
            b: vec![4, 3],
            cfg: dg(&[], &[], &[0, 1], &[1, 0]),
        },
        Case {
            name: "singleton dims",
            a: vec![1, 4, 1],
            b: vec![4, 1, 3],
            cfg: dg(&[1], &[0], &[2], &[1]),
        },
        Case {
            name: "batch middle",
            a: vec![4, 3, 5],
            b: vec![5, 3, 2],
            cfg: dg(&[2], &[0], &[1], &[1]),
        },
        Case {
            name: "empty free",
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

/// Column-major, permuted-storage and negative-stride variants of one tensor.
fn layouts<S: Scalar>(t: &T<S>) -> Vec<T<S>> {
    let rev: Vec<usize> = (0..t.dims.len()).rev().collect();
    vec![t.clone(), t.restride(&rev), t.reversed()]
}

fn check_all<S: Scalar + Field>(tol: f64) {
    let scalars = [
        (scalar::<S>(1.0, 0.0), scalar::<S>(0.0, 0.0)),
        (scalar::<S>(0.5, 0.25), scalar::<S>(-0.75, 0.5)),
    ];
    let conjs = [
        (tprims_blas::Conj::No, tprims_blas::Conj::No),
        (tprims_blas::Conj::Yes, tprims_blas::Conj::No),
        (tprims_blas::Conj::Yes, tprims_blas::Conj::Yes),
    ];
    for case in corpus() {
        let out = out_dims(&case.cfg, &case.a, &case.b);
        let a0 = T::<S>::new(&case.a, 1);
        let b0 = T::<S>::new(&case.b, 2);
        let c0 = T::<S>::new(&out, 3);
        for (li, (la, lb)) in layouts(&a0).into_iter().zip(layouts(&b0)).enumerate() {
            let lc = layouts(&c0).swap_remove(li);
            for &(alpha, beta) in &scalars {
                for &(ca, cb) in &conjs {
                    let want = reference(&case.cfg, alpha, &la, ca, &lb, cb, beta, &lc);
                    let p = problem(&case.cfg, &la, &lb, &lc, conj_pair(ca, cb));
                    for be in backends::<S>() {
                        let plan = be
                            .prepare(&p, &Requirements::new(), &PlanningBudget::serial())
                            .unwrap_or_else(|e| panic!("{} {}: {e}", case.name, be.id()));
                        let mut got = lc.clone();
                        let host = ExecHost::new(&Exec::serial());
                        plan.execute_into_accum(
                            &host,
                            alpha,
                            &la.view(),
                            &lb.view(),
                            beta,
                            &mut got.view_mut(),
                        )
                        .unwrap();
                        let err = rel_err(&got, &want);
                        assert!(err < tol, "{} {} layout {li}: {err}", case.name, be.id());
                    }
                }
            }
        }
    }
}

#[test]
fn both_backends_match_the_reference_f64() {
    check_all::<f64>(1e-12);
}
#[test]
fn both_backends_match_the_reference_f32() {
    check_all::<f32>(2e-5);
}
#[test]
fn both_backends_match_the_reference_c64() {
    check_all::<Complex64>(1e-12);
}
#[test]
fn both_backends_match_the_reference_c32() {
    check_all::<Complex32>(2e-5);
}

fn matmul_problem(m: usize, k: usize, n: usize) -> (DotGeneral, T<f64>, T<f64>, T<f64>) {
    (
        DotGeneral::new(&[1], &[0], &[], &[]),
        T::new(&[m, k], 11),
        T::new(&[k, n], 12),
        T::new(&[m, n], 13),
    )
}

#[test]
fn beta_zero_reads_no_output_and_zero_alpha_or_empty_k_reads_no_inputs() {
    let (cfg, a, b, c) = matmul_problem(4, 3, 5);
    let p = problem(&cfg, &a, &b, &c, (Conj::No, Conj::No));
    let nan = |t: &T<f64>| T {
        data: vec![f64::NAN; t.data.len()],
        ..t.clone()
    };
    for be in backends::<f64>() {
        let plan = be
            .prepare(&p, &Requirements::new(), &PlanningBudget::serial())
            .unwrap();
        let host = ExecHost::new(&Exec::serial());
        // beta == 0: previous C values (NaN) must not leak.
        let mut out = nan(&c);
        plan.execute_into_accum(&host, 1.0, &a.view(), &b.view(), 0.0, &mut out.view_mut())
            .unwrap();
        let want = reference(
            &cfg,
            1.0,
            &a,
            tprims_blas::Conj::No,
            &b,
            tprims_blas::Conj::No,
            0.0,
            &c,
        );
        assert!(rel_err(&out, &want) < 1e-12, "{}", be.id());
        // alpha == 0: A and B (NaN) must not be read.
        let mut out = c.clone();
        plan.execute_into_accum(
            &host,
            0.0,
            &nan(&a).view(),
            &nan(&b).view(),
            2.0,
            &mut out.view_mut(),
        )
        .unwrap();
        for_each_index(&c.dims, |i| {
            assert_eq!(out.get(i), 2.0 * c.get(i), "{}", be.id())
        });
    }
    // Empty contraction: C = beta * C, inputs untouched.
    let (cfg, a, b, c) = matmul_problem(3, 0, 2);
    let p = problem(&cfg, &a, &b, &c, (Conj::No, Conj::No));
    for be in backends::<f64>() {
        let plan = be
            .prepare(&p, &Requirements::new(), &PlanningBudget::serial())
            .unwrap();
        let mut out = c.clone();
        plan.execute_into_accum(
            &ExecHost::new(&Exec::serial()),
            1.0,
            &a.view(),
            &b.view(),
            3.0,
            &mut out.view_mut(),
        )
        .unwrap();
        for_each_index(&c.dims, |i| {
            assert_eq!(out.get(i), 3.0 * c.get(i), "{}", be.id())
        });
    }
}

#[test]
fn invalid_metadata_is_rejected_before_the_no_op_shortcuts() {
    // Empty contraction plus an aliased output: prepare must fail, not shortcut.
    let cfg = DotGeneral::new(&[1], &[0], &[], &[]);
    let p = Problem::new(
        cfg.clone(),
        Layout::new(&[3, 0], &[1, 3]),
        Layout::new(&[0, 2], &[1, 1]),
        Layout::new(&[3, 2], &[1, 1]),
        (Conj::No, Conj::No),
    );
    for be in backends::<f64>() {
        let e = be
            .prepare(&p, &Requirements::new(), &PlanningBudget::serial())
            .err();
        assert!(
            matches!(e, Some(Error::AliasedOutput)),
            "{}: {e:?}",
            be.id()
        );
    }
    // Empty output with mismatching extents of B.
    let p = Problem::new(
        cfg,
        Layout::new(&[0, 3], &[1, 1]),
        Layout::new(&[4, 2], &[1, 4]),
        Layout::new(&[0, 2], &[1, 1]),
        (Conj::No, Conj::No),
    );
    for be in backends::<f64>() {
        let e = be
            .prepare(&p, &Requirements::new(), &PlanningBudget::serial())
            .err();
        assert!(matches!(e, Some(Error::Shape(_))), "{}: {e:?}", be.id());
    }
}

#[test]
fn plans_outlive_problem_and_backend_and_reject_other_layouts_before_writing() {
    let (cfg, a, b, c) = matmul_problem(4, 3, 5);
    let plans: Vec<_> = {
        let p = problem(&cfg, &a, &b, &c, (Conj::No, Conj::No));
        let bes = backends::<f64>();
        bes.iter()
            .map(|be| {
                be.prepare(&p, &Requirements::new(), &PlanningBudget::serial())
                    .unwrap()
            })
            .collect()
        // problem and backends dropped here
    };
    let host = ExecHost::new(&Exec::serial());
    for plan in &plans {
        // Different buffers with the planned layouts, run repeatedly.
        for seed in [1u64, 2, 3] {
            let a2 = T::<f64>::new(&a.dims, seed);
            let b2 = T::<f64>::new(&b.dims, seed + 10);
            let mut out = c.clone();
            plan.execute_into_accum(&host, 1.0, &a2.view(), &b2.view(), 0.5, &mut out.view_mut())
                .unwrap();
            let want = reference(
                &cfg,
                1.0,
                &a2,
                tprims_blas::Conj::No,
                &b2,
                tprims_blas::Conj::No,
                0.5,
                &c,
            );
            assert!(
                rel_err(&out, &want) < 1e-12,
                "{}",
                plan.diagnostics().backend
            );
        }
        // A transposed C layout is another problem: rejected, C untouched.
        let wrong = c.restride(&[1, 0]);
        let mut out = wrong.clone();
        let e = plan
            .execute_into_accum(&host, 1.0, &a.view(), &b.view(), 0.0, &mut out.view_mut())
            .unwrap_err();
        assert!(matches!(e, Error::LayoutMismatch(_)), "{e}");
        assert_eq!(out.data, wrong.data);
    }
}

#[test]
fn materialization_is_reported_and_refused_under_the_shared_flag() {
    // Contracted axes in different orders on A and B cannot both fuse.
    let cfg = DotGeneral::new(&[1, 2], &[2, 0], &[], &[]);
    let a = T::<f64>::new(&[3, 4, 5], 1);
    let b = T::<f64>::new(&[5, 6, 4], 2);
    let c = T::<f64>::new(&[3, 6], 3);
    let p = problem(&cfg, &a, &b, &c, (Conj::No, Conj::No));
    let pg = TprimsBackend {
        strategy: Strategy::PermuteGemm,
        ..TprimsBackend::default()
    };
    let plan = ContractionBackend::<f64>::prepare(
        &pg,
        &p,
        &Requirements::new(),
        &PlanningBudget::serial(),
    )
    .unwrap();
    assert!(
        plan.diagnostics().materialized.iter().any(|&m| m),
        "{:?}",
        plan.diagnostics()
    );
    assert_eq!(plan.diagnostics().algorithm, "permute-gemm");
    let e = ContractionBackend::<f64>::prepare(
        &pg,
        &p,
        &Requirements::new().no_materialize(true),
        &PlanningBudget::serial(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(e, Error::WouldMaterialize { .. }) && e.is_unsupported(),
        "{e}"
    );
    // The other implementations never copy for this problem.
    for be in [
        Box::new(NaiveBackend) as Box<dyn ContractionBackend<f64>>,
        Box::new(TprimsBackend::default()),
    ] {
        let plan = be
            .prepare(
                &p,
                &Requirements::new().no_materialize(true),
                &PlanningBudget::serial(),
            )
            .unwrap();
        assert_eq!(plan.diagnostics().materialized, [false; 3], "{}", be.id());
    }
}

#[test]
fn a_forced_unsupported_requirement_fails_at_preparation_without_output() {
    // permute+GEMM computes with faer: asking it for a named kernel is unsupported.
    let (cfg, a, b, c) = matmul_problem(4, 3, 5);
    let p = problem(&cfg, &a, &b, &c, (Conj::No, Conj::No));
    let be = TprimsBackend {
        strategy: Strategy::PermuteGemm,
        gemm: tprims_blas::GemmConfig {
            kernel: tprims_blas::KernelChoice::Id("not.a.kernel".into()),
            ..Default::default()
        },
    };
    let e = ContractionBackend::<f64>::prepare(
        &be,
        &p,
        &Requirements::new(),
        &PlanningBudget::serial(),
    )
    .err()
    .unwrap();
    assert!(e.is_unsupported(), "{e}");
}

/// A host that is not an `ExecHost`: scoped threads, no pool.
struct ThreadsHost {
    width: usize,
    calls: AtomicUsize,
}

impl HostExecution for ThreadsHost {
    fn budget(&self) -> usize {
        self.width
    }
    fn install(&self, _k: usize, op: &mut (dyn FnMut(Par) + Send)) {
        op(Par::Seq)
    }
    fn for_each_partition(&self, k: usize, f: &(dyn Fn(usize) + Sync)) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        std::thread::scope(|s| {
            for i in 0..k {
                s.spawn(move || f(i));
            }
        });
    }
    fn broadcast(&self, _w: usize, _f: &(dyn Fn(usize) + Sync)) -> Result<(), HostError> {
        Err(HostError::Unavailable)
    }
}

#[test]
fn a_foreign_host_serves_the_naive_backend_and_is_refused_explicitly_by_tprims() {
    let (cfg, a, b, c) = matmul_problem(8, 5, 7);
    let p = problem(&cfg, &a, &b, &c, (Conj::No, Conj::No));
    let want = reference(
        &cfg,
        1.0,
        &a,
        tprims_blas::Conj::No,
        &b,
        tprims_blas::Conj::No,
        0.0,
        &c,
    );
    let wide = ThreadsHost {
        width: 4,
        calls: AtomicUsize::new(0),
    };

    let naive = ContractionBackend::<f64>::prepare(
        &NaiveBackend,
        &p,
        &Requirements::new(),
        &PlanningBudget::new(4),
    )
    .unwrap();
    let mut out = c.clone();
    naive
        .execute_into_accum(&wide, 1.0, &a.view(), &b.view(), 0.0, &mut out.view_mut())
        .unwrap();
    assert!(rel_err(&out, &want) < 1e-12);
    assert!(wide.calls.load(Ordering::Relaxed) >= 1);

    // tprims needs an ExecHost for width > 1: explicit failure, C untouched.
    let tp = ContractionBackend::<f64>::prepare(
        &TprimsBackend::default(),
        &p,
        &Requirements::new(),
        &PlanningBudget::new(4),
    )
    .unwrap();
    let mut out = c.clone();
    let e = tp
        .execute_into_accum(&wide, 1.0, &a.view(), &b.view(), 0.0, &mut out.view_mut())
        .unwrap_err();
    assert!(
        matches!(e, Error::Host(HostError::MissingCapability(_))) && e.is_unsupported(),
        "{e}"
    );
    assert_eq!(out.data, c.data);

    // A foreign host of width one is served on the calling thread.
    let one = ThreadsHost {
        width: 1,
        calls: AtomicUsize::new(0),
    };
    let mut out = c.clone();
    tp.execute_into_accum(&one, 1.0, &a.view(), &b.view(), 0.0, &mut out.view_mut())
        .unwrap();
    assert!(rel_err(&out, &want) < 1e-12);
}

#[test]
fn one_thread_host_enters_no_pool_and_a_four_thread_host_stays_in_budget() {
    let (cfg, a, b, c) = matmul_problem(96, 80, 72);
    let p = problem(&cfg, &a, &b, &c, (Conj::No, Conj::No));
    let want = reference(
        &cfg,
        1.0,
        &a,
        tprims_blas::Conj::No,
        &b,
        tprims_blas::Conj::No,
        0.5,
        &c,
    );
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    for be in backends::<f64>() {
        let plan = be
            .prepare(&p, &Requirements::new(), &PlanningBudget::new(4))
            .unwrap();
        // 1T on a pool-backed context: no entry, no broadcast.
        let before = pool.stats();
        let host = ExecHost::new(&Exec::rayon(&pool).with_budget(1).unwrap());
        let mut out = c.clone();
        plan.execute_into_accum(&host, 1.0, &a.view(), &b.view(), 0.5, &mut out.view_mut())
            .unwrap();
        assert!(rel_err(&out, &want) < 1e-12, "{}", be.id());
        let after = pool.stats();
        assert_eq!(
            (after.entries, after.broadcasts),
            (before.entries, before.broadcasts),
            "{}",
            be.id()
        );
        // The same plan at 4T.
        let host = ExecHost::new(&Exec::rayon(&pool));
        assert_eq!(host.budget(), 4);
        let mut out = c.clone();
        plan.execute_into_accum(&host, 1.0, &a.view(), &b.view(), 0.5, &mut out.view_mut())
            .unwrap();
        assert!(rel_err(&out, &want) < 1e-12, "{}", be.id());
        // Nested entry from inside the pool completes (no deadlock) and agrees.
        let mut out = c.clone();
        tp.install(|| {
            plan.execute_into_accum(&host, 1.0, &a.view(), &b.view(), 0.5, &mut out.view_mut())
                .unwrap()
        });
        assert!(rel_err(&out, &want) < 1e-12, "{} nested", be.id());
    }
}

#[test]
fn a_shared_plan_runs_concurrently_on_independent_outputs() {
    let (cfg, a, b, c) = matmul_problem(64, 48, 40);
    let p = problem(&cfg, &a, &b, &c, (Conj::No, Conj::No));
    let want = reference(
        &cfg,
        1.0,
        &a,
        tprims_blas::Conj::No,
        &b,
        tprims_blas::Conj::No,
        0.0,
        &c,
    );
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    for be in backends::<f64>() {
        let plan = be
            .prepare(&p, &Requirements::new(), &PlanningBudget::new(4))
            .unwrap();
        std::thread::scope(|s| {
            for t in 0..4 {
                let (plan, a, b, c, want, pool) = (&plan, &a, &b, &c, &want, &pool);
                s.spawn(move || {
                    // Half the callers use the shared pool, half run serially.
                    let exec = if t % 2 == 0 {
                        Exec::rayon(pool)
                    } else {
                        Exec::serial()
                    };
                    let host = ExecHost::new(&exec);
                    for _ in 0..3 {
                        let mut out = c.clone();
                        plan.execute_into_accum(
                            &host,
                            1.0,
                            &a.view(),
                            &b.view(),
                            0.0,
                            &mut out.view_mut(),
                        )
                        .unwrap();
                        assert!(rel_err(&out, want) < 1e-12);
                    }
                });
            }
        });
    }
}
