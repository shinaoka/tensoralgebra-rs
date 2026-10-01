//! tprims addition: threads come from a `tprims_exec::Exec` alone. A wide
//! budget broadcasts on the pool, a width of one never does, and a broadcast
//! the pool refuses runs serially with the same frozen family.

use std::sync::atomic::{AtomicUsize, Ordering};

use tensorcontract::{Layout, Operand, Plan, TensorView, TensorViewMut};
use tprims_exec::{Exec, Pool};

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

fn plan_for(la: &Layout, lb: &Layout, ld: &Layout, threads: usize) -> Plan {
    Plan::new(
        Operand::new(la, &[0, 2]),
        Operand::new(lb, &[2, 1]),
        None,
        Operand::new(ld, &[0, 1]),
    )
    .unwrap()
    .with_threads(threads)
}

fn layouts() -> (Layout, Layout, Layout) {
    (
        Layout::col_major(&[M as i64, K as i64]),
        Layout::col_major(&[K as i64, N as i64]),
        Layout::col_major(&[M as i64, N as i64]),
    )
}

fn run_on(plan: &Plan, exec: Option<&Exec<'_>>, la: &Layout, lb: &Layout, ld: &Layout) -> Vec<f64> {
    let (av, bv) = inputs();
    let (ia, ib, id) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    let mut out = vec![0.0; M * N];
    let (a, b) = (TensorView::new(&av, la, &ia), TensorView::new(&bv, lb, &ib));
    let d = TensorViewMut::new(&mut out, ld, &id);
    match exec {
        Some(e) => plan.run_with(e, 1.0, a, b, 0.0, None, d).unwrap(),
        None => plan.run(1.0, a, b, 0.0, None, d).unwrap(),
    }
    out
}

#[test]
fn exec_matches_serial_bitwise_and_a_refused_broadcast_falls_back_serially() {
    let (la, lb, ld) = layouts();
    let plan = plan_for(&la, &lb, &ld, 1);
    let serial = run_on(&plan, None, &la, &lb, &ld);

    let tp = pool4();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    assert_eq!(run_on(&plan, Some(&exec), &la, &lb, &ld), serial);
    assert_eq!(pool.stats().broadcasts, 1);

    // Called from one of the pool's own workers, the broadcast is refused
    // before any work starts; the same contraction runs serially.
    pool.reset_stats();
    let nested = exec.install(2, |_| run_on(&plan, Some(&exec), &la, &lb, &ld));
    assert_eq!(nested, serial);
    assert_eq!(pool.stats().broadcasts, 0);
}

#[test]
fn width_one_never_broadcasts_whatever_the_plan_asks_for() {
    let (la, lb, ld) = layouts();
    let plan = plan_for(&la, &lb, &ld, 8);
    let tp = pool4();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool).with_budget(1).unwrap();
    let serial = run_on(&plan, None, &la, &lb, &ld);
    assert_eq!(run_on(&plan, Some(&exec), &la, &lb, &ld), serial);
    assert_eq!(pool.stats().broadcasts, 0);
    assert_eq!(pool.stats().entries, 0);
}

#[test]
fn explicit_resolution_survives_a_refused_broadcast_without_reselection() {
    use tprims_gemm_kernel::{KernelChoice, KernelFamily, ResolvedGemm, UkrFn};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    unsafe fn traced(k: usize, a: *const f64, b: *const f64, out: *mut f64) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: identical panel/tile ABI to the descriptor we copy below.
        unsafe {
            tprims_gemm_kernel::portable::real_tile::<f64, 4, 4>(k, a, b, out);
        }
    }
    fn manifest() -> &'static [&'static KernelFamily<f64>] {
        static LIST: std::sync::OnceLock<[&'static KernelFamily<f64>; 1]> =
            std::sync::OnceLock::new();
        LIST.get_or_init(|| {
            // Do not recurse through the registry while this callback's
            // OnceLock initializes: copy the immutable built-in menu directly.
            let mut f = **tprims_gemm_kernel::portable::families_f64()
                .iter()
                .find(|f| f.id == "portable.f64.4x4")
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
        tprims_gemm_kernel::register::<f64>(manifest);
    }
    let (a, b) = inputs();
    let la = Layout::col_major(&[M as i64, K as i64]);
    let lb = Layout::col_major(&[K as i64, N as i64]);
    let ld = Layout::col_major(&[M as i64, N as i64]);
    let p = Plan::new(
        Operand::new(&la, &[0, 2]),
        Operand::new(&lb, &[2, 1]),
        None,
        Operand::new(&ld, &[0, 1]),
    )
    .unwrap()
    .with_threads(1);
    let rg =
        ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Id("test.traced.f64.4x4".into()), 4)
            .unwrap();
    rg.with_threads(1).unwrap();
    let mut expected = vec![0.; M * N];
    let mut actual = vec![0.; M * N];
    // SAFETY: full disjoint buffers and validated matching real scratch ABI.
    unsafe {
        tensorcontract::execute_resolved(
            &p,
            &rg,
            &Exec::Serial,
            None,
            1.,
            a.as_ptr(),
            b.as_ptr(),
            0.,
            std::ptr::null(),
            expected.as_mut_ptr(),
        );
    }
    CALLS.store(0, Ordering::Relaxed);
    // Refused: the caller is a worker of the pool, so the broadcast runs
    // nothing and the contraction is retried serially, same frozen family.
    let tp = pool4();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    struct SendPtrs(*const f64, *const f64, *mut f64);
    // SAFETY: the pointers address buffers that outlive the call and are used
    // by exactly one thread, the worker the closure runs on.
    unsafe impl Send for SendPtrs {}
    impl SendPtrs {
        fn run(self, p: &Plan, rg: &ResolvedGemm<f64>, exec: &Exec<'_>) {
            // SAFETY: same valid buffers, resolution and serial bound as above.
            unsafe {
                tensorcontract::execute_resolved(
                    p,
                    rg,
                    exec,
                    None,
                    1.,
                    self.0,
                    self.1,
                    0.,
                    std::ptr::null(),
                    self.2,
                );
            }
        }
    }
    let ptrs = SendPtrs(a.as_ptr(), b.as_ptr(), actual.as_mut_ptr());
    exec.install(2, |_| ptrs.run(&p, &rg, &exec));
    assert_eq!(actual, expected);
    assert_eq!(pool.stats().broadcasts, 0);
    assert!(
        CALLS.load(Ordering::Relaxed) > 0,
        "fallback reselected the plan's default family"
    );
}
