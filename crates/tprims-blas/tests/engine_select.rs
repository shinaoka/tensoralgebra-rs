//! Engine selection: every engine must compute the same product, the packed
//! engine must report the family it used, and an unusable choice must say why.
use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{
    default_engine, gemm_with, list_kernels, Engine, EngineChoice, Error, GemmConfig, MatIn,
    SelectedGemm,
};
use tprims_exec::{Exec, Pool};
use tprims_kernel::KernelChoice;

/// Integer-valued operands, so the engines' sums agree exactly.
const M: usize = 37;
const N: usize = 29;
const K: usize = 41;

fn data(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|x| ((x * 7 + seed) % 11) as f64 - 5.0)
        .collect()
}

fn run(exec: &Exec<'_>, cfg: &GemmConfig) -> Result<(SelectedGemm, Vec<f64>), Error> {
    let a = data(M * K, 0);
    let b = data(K * N, 1);
    let mut c = vec![0.0; M * N];
    let av = StridedView::new(&a, &[M, K], &[1, M as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[K, N], &[1, K as isize], 0).unwrap();
    let mut cv = StridedViewMut::new(&mut c, &[M, N], &[1, M as isize], 0).unwrap();
    let sel = gemm_with(
        exec,
        cfg,
        1.0,
        MatIn::new(&av),
        MatIn::new(&bv),
        0.0,
        &mut cv,
    )?;
    Ok((sel, c))
}

#[test]
fn every_engine_agrees() {
    let exec = Exec::serial();
    let (faer, want) = run(&exec, &GemmConfig::default()).unwrap();
    assert_eq!(faer.engine, Engine::Faer);
    assert!(faer.family_id.is_none());

    let (packed, got) = run(
        &exec,
        &GemmConfig {
            engine: EngineChoice::Packed,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(packed.engine, Engine::Packed);
    assert!(packed.family_id.is_some());
    assert!(packed.mr > 0 && packed.nr > 0 && packed.kc > 0);
    assert_eq!(got, want, "the packed engine disagrees with faer");
}

#[test]
fn a_forced_kernel_is_used_and_reported() {
    let exec = Exec::serial();
    let cfg = GemmConfig {
        engine: EngineChoice::Packed,
        kernel: KernelChoice::Id("portable.f64.4x4".into()),
        ..Default::default()
    };
    let (sel, _) = run(&exec, &cfg).unwrap();
    assert_eq!(sel.family_id, Some("portable.f64.4x4"));
    assert!(sel.to_json().contains("\"family_id\":\"portable.f64.4x4\""));
}

#[test]
fn an_unknown_kernel_id_is_a_selection_error() {
    let exec = Exec::serial();
    let cfg = GemmConfig {
        engine: EngineChoice::Packed,
        kernel: KernelChoice::Id("not.a.kernel".into()),
        ..Default::default()
    };
    let err = run(&exec, &cfg).unwrap_err();
    assert!(
        matches!(
            err,
            Error::Select(tprims_kernel::SelectError::UnknownId { .. })
        ),
        "{err:?}"
    );
}

/// The process default is faer unless the environment chose otherwise at
/// startup, and the engine list registers what this build has.
#[test]
fn default_engine_and_kernel_list_are_available() {
    let _ = default_engine();
    let kernels = list_kernels::<f64>();
    assert!(kernels.iter().any(|k| k.id.starts_with("tc.")));
    assert!(kernels.iter().any(|k| k.id == "portable.f64.4x4"));
}

/// Two families, two threads, one pool: the workspace is leased per call, so
/// concurrent packed GEMMs must not corrupt each other.
#[test]
fn two_families_concurrently_on_one_pool() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let reference = run(&Exec::serial(), &GemmConfig::default()).unwrap().1;
    std::thread::scope(|s| {
        for id in ["portable.f64.4x4", "tc.scalar.f64.4x4"] {
            let (exec, reference) = (&exec, &reference);
            s.spawn(move || {
                for _ in 0..20 {
                    let cfg = GemmConfig {
                        engine: EngineChoice::Packed,
                        kernel: KernelChoice::Id(id.into()),
                        ..Default::default()
                    };
                    let (sel, got) = run(exec, &cfg).unwrap();
                    assert_eq!(sel.family_id, Some(id));
                    assert_eq!(&got, reference, "{id}");
                }
            });
        }
    });
}
