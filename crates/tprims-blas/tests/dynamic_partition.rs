//! `PartitionPolicy::DynamicTiles` through the safe GEMM entrypoints: it is a
//! per-call option of the packed engine, validated before compute, honoured
//! by the batched TBLIS strategy, and refused where it cannot apply.
use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{
    gemm_batched_with, gemm_grouped_with, gemm_with, BatchIn, BatchStrategy, Conj, EngineChoice,
    Error, GemmConfig, GroupedJob, MatIn,
};
use tprims_exec::{Exec, Pool};
use tprims_kernel::{KernelChoice, PartitionOpts, PartitionPolicy, SelectError};

fn ints(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|x| ((x * 7 + seed * 5) % 13) as f64 - 6.0)
        .collect()
}

fn dynamic(job_m: usize, job_n: usize) -> GemmConfig {
    GemmConfig {
        engine: EngineChoice::Packed,
        kernel: KernelChoice::Id("portable.f64.4x4".into()),
        partition: PartitionPolicy::DynamicTiles { job_m, job_n },
        ..Default::default()
    }
}
fn packed() -> GemmConfig {
    GemmConfig {
        engine: EngineChoice::Packed,
        kernel: KernelChoice::Id("portable.f64.4x4".into()),
        ..Default::default()
    }
}

fn gemm(
    exec: &Exec<'_>,
    cfg: &GemmConfig,
    (m, n, k): (usize, usize, usize),
    alpha: f64,
) -> (Result<tprims_blas::SelectedGemm, Error>, Vec<f64>) {
    let (a, b) = (ints((m * k).max(1), 1), ints((k * n).max(1), 2));
    let mut c = ints((m * n).max(1), 3);
    let av = StridedView::new(&a, &[m, k], &[1, m.max(1) as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[k, n], &[1, k.max(1) as isize], 0).unwrap();
    let mut cv = StridedViewMut::new(&mut c, &[m, n], &[1, m.max(1) as isize], 0).unwrap();
    let r = gemm_with(
        exec,
        cfg,
        alpha,
        MatIn::new(&av),
        MatIn::new(&bv),
        0.5,
        &mut cv,
    );
    (r, c)
}

fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
}

#[test]
fn dynamic_gemm_matches_the_static_packed_engine_and_reports_its_policy() {
    let tp = pool();
    let p = Pool::borrow(&tp);
    for shape in [(70, 66, 130), (13, 200, 40), (200, 7, 33)] {
        let (want, c_static) = gemm(&Exec::rayon(&p), &packed(), shape, 1.5);
        assert!(want.unwrap().dynamic.is_none());
        let (got, c_dyn) = gemm(&Exec::rayon(&p), &dynamic(16, 32), shape, 1.5);
        let report = got.unwrap();
        assert_eq!(c_dyn, c_static, "{shape:?}");
        assert_eq!(
            report.partition,
            PartitionPolicy::DynamicTiles {
                job_m: 16,
                job_n: 32
            }
        );
        let d = report.dynamic.expect("dynamic reports its assignment");
        assert_eq!((d.job_m, d.job_n), (16, 32));
        assert!(d.active_width >= 1 && d.active_width <= 4);
    }
}

#[test]
fn invalid_policies_fail_before_compute_even_for_empty_problems() {
    let exec = Exec::serial();
    for shape in [(20, 20, 20), (0, 20, 20), (20, 20, 0)] {
        for (jm, jn) in [(0, 8), (8, 0), (6, 8), (8, 6)] {
            let (r, c) = gemm(&exec, &dynamic(jm, jn), shape, 1.0);
            assert!(
                matches!(r, Err(Error::Select(SelectError::Incompatible { .. }))),
                "{shape:?} {jm}x{jn}: {r:?}"
            );
            assert_eq!(c, ints((shape.0 * shape.1).max(1), 3), "nothing written");
        }
        let aligned = GemmConfig {
            partition_opts: PartitionOpts {
                align_c_lines: true,
            },
            ..dynamic(8, 8)
        };
        let (r, _) = gemm(&exec, &aligned, shape, 1.0);
        assert!(matches!(
            r,
            Err(Error::Select(SelectError::Incompatible { .. }))
        ));
        // Valid policy on an empty problem succeeds and only scales C.
        assert!(gemm(&exec, &dynamic(8, 8), shape, 1.0).0.is_ok());
    }
}

#[test]
fn other_engines_and_strategies_refuse_a_partition_policy() {
    let exec = Exec::serial();
    let policy = PartitionPolicy::DynamicTiles { job_m: 8, job_n: 8 };
    for engine in [EngineChoice::Faer, EngineChoice::Auto] {
        let cfg = GemmConfig {
            engine: engine.clone(),
            partition: policy,
            ..Default::default()
        };
        for shape in [(20, 20, 20), (0, 4, 4)] {
            let (r, _) = gemm(&exec, &cfg, shape, 1.0);
            assert!(
                matches!(r, Err(Error::Select(SelectError::EngineUnsupported { .. }))),
                "{engine:?} {shape:?}: {r:?}"
            );
        }
    }
    // The faer-loop batched strategies and the grouped path refuse it too.
    let a = ints(4 * 4 * 2, 1);
    let mut c = vec![0.0; 4 * 4 * 2];
    let av = StridedView::new(&a, &[4, 4, 2], &[1, 4, 16], 0).unwrap();
    let mut cv = StridedViewMut::new(&mut c, &[4, 4, 2], &[1, 4, 16], 0).unwrap();
    let cfg = GemmConfig {
        partition: policy,
        ..Default::default()
    };
    let r = gemm_batched_with(
        &exec,
        &cfg,
        1.0,
        BatchIn::new(&av),
        BatchIn::new(&av),
        0.0,
        &mut cv,
        BatchStrategy::FaerLoop,
    );
    assert!(matches!(r, Err(Error::Select(_))));
    let job = GroupedJob {
        a_offset: 0,
        b_offset: 0,
        c_offset: 0,
        rows: 4,
        inner: 4,
        cols: 4,
    };
    let r = gemm_grouped_with(
        &exec,
        &cfg,
        1.0,
        &a,
        Conj::No,
        &a,
        Conj::No,
        0.0,
        &mut c,
        &[job],
    );
    assert!(matches!(r, Err(Error::Select(_))));
}

#[test]
fn the_batched_tblis_strategy_honours_the_policy() {
    let tp = pool();
    let p = Pool::borrow(&tp);
    let (m, n, k, count) = (24usize, 40usize, 30usize, 3usize);
    let a = ints(m * k * count, 1);
    let b = ints(k * n * count, 2);
    let run = |cfg: &GemmConfig| {
        let mut c = vec![0.0; m * n * count];
        let av =
            StridedView::new(&a, &[m, k, count], &[1, m as isize, (m * k) as isize], 0).unwrap();
        let bv =
            StridedView::new(&b, &[k, n, count], &[1, k as isize, (k * n) as isize], 0).unwrap();
        let mut cv = StridedViewMut::new(
            &mut c,
            &[m, n, count],
            &[1, m as isize, (m * n) as isize],
            0,
        )
        .unwrap();
        let r = gemm_batched_with(
            &Exec::rayon(&p),
            cfg,
            1.0,
            BatchIn::new(&av),
            BatchIn::new(&bv),
            0.0,
            &mut cv,
            BatchStrategy::Tblis,
        );
        (r, c)
    };
    let (want, c_static) = run(&packed());
    want.unwrap();
    let (got, c_dyn) = run(&dynamic(8, 8));
    let report = got.unwrap();
    assert_eq!(c_dyn, c_static);
    assert!(report.dynamic.is_some());
    // Empty batches still validate the policy.
    let (bad, _) = run(&dynamic(6, 8));
    assert!(matches!(
        bad,
        Err(Error::Select(SelectError::Incompatible { .. }))
    ));
}

#[test]
fn pools_reuse_their_workspace_and_concurrent_pools_stay_separate() {
    let tp = pool();
    let (pa, pb) = (Pool::borrow(&tp), Pool::borrow(&tp));
    let shape = (96, 96, 96);
    let (_, want) = gemm(&Exec::serial(), &packed(), shape, 1.0);
    // Owner reuse: repeated calls do not grow the retained workspace.
    let ea = Exec::rayon(&pa);
    gemm(&ea, &dynamic(16, 16), shape, 1.0).0.unwrap();
    let retained = pa.workspace().retained_bytes();
    assert!(retained > 0);
    for _ in 0..3 {
        let (r, c) = gemm(&ea, &dynamic(16, 16), shape, 1.0);
        r.unwrap();
        assert_eq!(c, want);
    }
    assert_eq!(pa.workspace().retained_bytes(), retained);
    assert_eq!(
        pb.workspace().retained_bytes(),
        0,
        "another pool's arena is untouched"
    );

    // Two threads, one pool (serialized by the pool) and two pools at once.
    let (x, y) = std::thread::scope(|s| {
        let x = s.spawn(|| gemm(&Exec::rayon(&pa), &dynamic(16, 16), shape, 1.0).1);
        let y = s.spawn(|| gemm(&Exec::rayon(&pa), &dynamic(8, 32), shape, 1.0).1);
        (x.join().unwrap(), y.join().unwrap())
    });
    assert_eq!((x, y), (want.clone(), want.clone()));
    let (x, y) = std::thread::scope(|s| {
        let x = s.spawn(|| gemm(&Exec::rayon(&pa), &dynamic(16, 16), shape, 1.0).1);
        let y = s.spawn(|| gemm(&Exec::rayon(&pb), &dynamic(16, 16), shape, 1.0).1);
        (x.join().unwrap(), y.join().unwrap())
    });
    assert_eq!((x, y), (want.clone(), want.clone()));

    // Re-entry: a dynamic GEMM issued from inside a worker of the same pool
    // must neither deadlock nor fabricate threads; it runs and is correct.
    let nested = tp.install(|| gemm(&Exec::rayon(&pa), &dynamic(16, 16), shape, 1.0).1);
    assert_eq!(nested, want);
}
