//! Independent catalogs, selectors and plans coexist: on one shared pool, on
//! separate pools, and next to the process defaults they never touch.
use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{gemm_with_selector, GemmConfig, KernelCatalog, MatIn};
use tprims_custom_kernel_test as own;
use tprims_exec::{Exec, Pool};

const M: usize = 150;
const N: usize = 130;
const K: usize = 90;

fn catalog() -> KernelCatalog<f64> {
    // SAFETY: see `selector_blas::catalog`.
    unsafe { KernelCatalog::<f64>::from_static_families(own::f64_families()) }.unwrap()
}

fn data(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|x| ((x * 7 + seed) % 11) as f64 - 5.0)
        .collect()
}

/// `reps` GEMMs choosing `id` from a private catalog; returns the product.
fn worker(exec: &Exec<'_>, id: &'static str, reps: usize) -> (Vec<f64>, &'static str) {
    let cat = catalog();
    let (a, b) = (data(M * K, 1), data(K * N, 2));
    let av = StridedView::new(&a, &[M, K], &[1, M as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[K, N], &[1, K as isize], 0).unwrap();
    let mut c = vec![0.0; M * N];
    let mut used = "";
    for _ in 0..reps {
        let mut cv = StridedViewMut::new(&mut c, &[M, N], &[1, M as isize], 0).unwrap();
        let report = gemm_with_selector(
            exec,
            &GemmConfig::default(),
            &cat,
            |_, _| Ok(cat.get(id).unwrap()),
            1.0,
            MatIn::new(&av),
            MatIn::new(&bv),
            0.0,
            &mut cv,
        )
        .unwrap();
        used = report.family_id.unwrap();
    }
    (c, used)
}

#[test]
fn catalogs_run_concurrently_on_one_pool_and_on_separate_pools() {
    let before = tprims_kernel::process_default::<f64>().unwrap().family().id;
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let reference = worker(&Exec::serial(), "custom.f64.2x2", 1).0;

    // Same pool, two threads, two different custom kernels.
    let shared = Pool::borrow(&tp);
    let (x, y) = std::thread::scope(|s| {
        let x = s.spawn(|| worker(&Exec::rayon(&shared), "custom.f64.2x2", 6));
        let y = s.spawn(|| worker(&Exec::rayon(&shared), "custom.f64.3x4", 6));
        (x.join().unwrap(), y.join().unwrap())
    });
    assert_eq!((x.1, y.1), ("custom.f64.2x2", "custom.f64.3x4"));
    assert_eq!(x.0, reference);
    assert_eq!(y.0, reference);

    // Separate pools (each with its own workspace) on the same worker threads.
    let (pa, pb) = (Pool::borrow(&tp), Pool::borrow(&tp));
    let (x, y) = std::thread::scope(|s| {
        let x = s.spawn(|| worker(&Exec::rayon(&pa), "custom.f64.3x4", 4));
        let y = s.spawn(|| worker(&Exec::rayon(&pb), "custom.f64.2x2", 4));
        (x.join().unwrap(), y.join().unwrap())
    });
    assert_eq!(x.0, reference);
    assert_eq!(y.0, reference);

    // The process default is exactly what it was.
    let after = tprims_kernel::process_default::<f64>().unwrap().family().id;
    assert_eq!(before, after);
    assert!(tprims_kernel::list_kernels::<f64>()
        .iter()
        .all(|k| !k.id.starts_with("custom.")));
}

#[test]
fn executing_from_inside_a_worker_of_the_same_pool_falls_back_serially_on_the_same_family() {
    use tprims_blas::Conj;
    use tprims_contract::{ContractPlan, DotGeneral, Flags, Strategy};
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let cat = catalog();
    let (dims_a, dims_b, dims_c) = ([M, K], [K, N], [M, N]);
    let (sa, sb, sc) = (
        [1isize, M as isize],
        [1isize, K as isize],
        [1isize, M as isize],
    );
    let plan = ContractPlan::<f64>::new_with_selector(
        &exec,
        &GemmConfig::default(),
        &cat,
        |ctx, cands| {
            assert!(ctx.threads > 1, "planned for the pool's width");
            Ok(cands[1].handle)
        },
        &DotGeneral::new(&[1], &[0], &[], &[]),
        (&dims_a, &sa),
        (&dims_b, &sb),
        (&dims_c, &sc),
        (Conj::No, Conj::No),
        Strategy::Tblis,
        Flags::default(),
    )
    .unwrap();
    drop(cat);
    let (a, b) = (data(M * K, 1), data(K * N, 2));
    let reference = worker(&Exec::serial(), "custom.f64.2x2", 1).0;
    let run = |exec: &Exec<'_>| {
        let av = StridedView::new(&a, &dims_a, &sa, 0).unwrap();
        let bv = StridedView::new(&b, &dims_b, &sb, 0).unwrap();
        let mut c = vec![0.0; M * N];
        let mut cv = StridedViewMut::new(&mut c, &dims_c, &sc, 0).unwrap();
        plan.execute(exec, 1.0, &av, &bv, 0.0, &mut cv).unwrap();
        c
    };
    let outside = run(&exec);
    // Inside a worker of the very pool the plan was made for, a broadcast
    // cannot be co-scheduled; the call runs serially with the frozen family.
    let inside = tp.install(|| run(&exec));
    assert_eq!(outside, reference);
    assert_eq!(inside, reference);
    assert_eq!(
        plan.selected_gemm().unwrap().unwrap().family_id,
        Some("custom.f64.3x4")
    );
}
