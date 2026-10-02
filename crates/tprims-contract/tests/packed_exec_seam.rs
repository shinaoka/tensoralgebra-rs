//! tprims addition: threads come from a `tprims_exec::Exec` alone. A wide
//! budget broadcasts on the pool, a width of one never does, and a broadcast
//! the pool refuses runs serially with the same frozen family.

use std::sync::atomic::{AtomicUsize, Ordering};

use strided_view::{StridedView, StridedViewMut};
use tprims_contract::{Plan, PlanConfig};
use tprims_exec::{Exec, Pool};
use tprims_kernel::KernelChoice;

mod common;
use common::plans::{matmul_problem, packed};

const M: usize = 256;
const N: usize = 240;
const K: usize = 200;

fn inputs() -> (Vec<f64>, Vec<f64>) {
    let a = (0..M * K).map(|x| (x % 13) as f64 - 6.0).collect();
    let b = (0..K * N).map(|x| (x % 7) as f64 * 0.5).collect();
    (a, b)
}

fn pool4() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
}

fn run_on(plan: &Plan<f64>, exec: &Exec<'_>) -> Vec<f64> {
    let (av, bv) = inputs();
    let mut out = vec![0.0; M * N];
    plan.execute_into(
        exec,
        1.0,
        &StridedView::new(&av, &[M, K], &[1, M as isize], 0).unwrap(),
        &StridedView::new(&bv, &[K, N], &[1, K as isize], 0).unwrap(),
        &mut StridedViewMut::new(&mut out, &[M, N], &[1, M as isize], 0).unwrap(),
    )
    .unwrap();
    out
}

#[test]
fn exec_matches_serial_bitwise_and_a_refused_broadcast_falls_back_serially() {
    let plan = Plan::<f64>::new(&matmul_problem(M, N, K), &packed()).unwrap();
    let serial = run_on(&plan, &Exec::serial());

    let tp = pool4();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    assert_eq!(run_on(&plan, &exec), serial);
    assert_eq!(pool.stats().broadcasts, 1);

    // Called from one of the pool's own workers, the broadcast is refused
    // before any work starts; the same contraction runs serially.
    pool.reset_stats();
    let nested = exec.install(2, |_| run_on(&plan, &exec));
    assert_eq!(nested, serial);
    assert_eq!(pool.stats().broadcasts, 0);
}

#[test]
fn width_one_never_broadcasts_whatever_the_pool_could_do() {
    let plan = Plan::<f64>::new(&matmul_problem(M, N, K), &packed()).unwrap();
    let tp = pool4();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool).with_budget(1).unwrap();
    let serial = run_on(&plan, &Exec::serial());
    assert_eq!(run_on(&plan, &exec), serial);
    assert_eq!(pool.stats().broadcasts, 0);
    assert_eq!(pool.stats().entries, 0);
}

#[test]
fn a_plan_never_reselects_its_family_for_a_different_budget() {
    use tprims_kernel::{KernelFamily, UkrFn};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    unsafe fn traced(k: usize, a: *const f64, b: *const f64, out: *mut f64) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: identical panel/tile ABI to the descriptor we copy below.
        unsafe {
            tprims_kernel::portable::real_tile::<f64, 4, 4>(k, a, b, out);
        }
    }
    fn manifest() -> &'static [&'static KernelFamily<f64>] {
        static LIST: std::sync::OnceLock<[&'static KernelFamily<f64>; 1]> =
            std::sync::OnceLock::new();
        LIST.get_or_init(|| {
            // Do not recurse through the registry while this callback's
            // OnceLock initializes: copy the immutable built-in menu directly.
            let mut f = **tprims_kernel::portable::families_f64()
                .iter()
                .find(|f| f.id == "ref.f64.real.4x4")
                .unwrap();
            f.id = "test.traced.f64.4x4";
            f.allow_auto = false;
            f.ukr = UkrFn::Tile(traced);
            [Box::leak(Box::new(f))]
        })
    }
    // SAFETY: immutable manifest copies the validated portable 4x4 footprint,
    // ISA and overwrite contract; traced only counts then calls that same ABI.
    unsafe {
        tprims_kernel::register::<f64>(manifest);
    }
    let config = PlanConfig {
        kernel: KernelChoice::Id("test.traced.f64.4x4".into()),
        ..PlanConfig::default()
    };
    let plan = Plan::<f64>::new(&matmul_problem(M, N, K), &config).unwrap();
    let expected = run_on(&plan, &Exec::serial());
    CALLS.store(0, Ordering::Relaxed);
    // Refused: the caller is a worker of the pool, so the broadcast runs
    // nothing and the contraction is retried serially, same frozen family.
    let tp = pool4();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let actual = exec.install(2, |_| run_on(&plan, &exec));
    assert_eq!(actual, expected);
    assert_eq!(pool.stats().broadcasts, 0);
    assert!(
        CALLS.load(Ordering::Relaxed) > 0,
        "fallback reselected the plan's default family"
    );
}
