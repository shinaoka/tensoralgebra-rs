//! tprims addition: the `Spmd` seam lets a host supply co-scheduled threads.

use std::sync::atomic::{AtomicUsize, Ordering};

use tensorcontract::spmd::Spmd;
use tensorcontract::{Layout, Operand, Plan, TensorView, TensorViewMut};

struct Scoped {
    width: usize,
    calls: AtomicUsize,
    last_p: AtomicUsize,
    refuse: bool,
}

impl Spmd for Scoped {
    fn width(&self) -> usize {
        self.width
    }
    fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.last_p.store(p, Ordering::Relaxed);
        if self.refuse {
            return false;
        }
        std::thread::scope(|s| {
            for t in 0..p {
                s.spawn(move || f(t));
            }
        });
        true
    }
}

const M: usize = 256;
const N: usize = 240;
const K: usize = 200;

fn inputs() -> (Vec<f64>, Vec<f64>) {
    let a = (0..M * K).map(|x| (x % 13) as f64 - 6.0).collect();
    let b = (0..K * N).map(|x| (x % 7) as f64 * 0.5).collect();
    (a, b)
}

#[test]
fn seam_matches_serial_bitwise_and_refusal_falls_back_serially() {
    let (av, bv) = inputs();
    let la = Layout::col_major(&[M as i64, K as i64]);
    let lb = Layout::col_major(&[K as i64, N as i64]);
    let ld = Layout::col_major(&[M as i64, N as i64]);
    let (ia, ib, id) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    let plan = Plan::new(
        Operand::new(&la, &ia),
        Operand::new(&lb, &ib),
        None,
        Operand::new(&ld, &id),
    )
    .unwrap()
    .with_threads(1);

    let mut serial = vec![0.0; M * N];
    plan.run(
        1.0,
        TensorView::new(&av, &la, &ia),
        TensorView::new(&bv, &lb, &ib),
        0.0,
        None,
        TensorViewMut::new(&mut serial, &ld, &id),
    )
    .unwrap();

    for refuse in [false, true] {
        let s = Scoped {
            width: 4,
            calls: AtomicUsize::new(0),
            last_p: AtomicUsize::new(0),
            refuse,
        };
        let mut out = vec![0.0; M * N];
        plan.run_with(
            &s,
            1.0,
            TensorView::new(&av, &la, &ia),
            TensorView::new(&bv, &lb, &ib),
            0.0,
            None,
            TensorViewMut::new(&mut out, &ld, &id),
        )
        .unwrap();
        assert_eq!(out, serial, "refuse={refuse}");
        assert_eq!(s.calls.load(Ordering::Relaxed), 1);
        let p = s.last_p.load(Ordering::Relaxed);
        assert!((2..=4).contains(&p), "p={p}");
    }
}

#[test]
fn width_one_seam_never_broadcasts() {
    let (av, bv) = inputs();
    let la = Layout::col_major(&[M as i64, K as i64]);
    let lb = Layout::col_major(&[K as i64, N as i64]);
    let ld = Layout::col_major(&[M as i64, N as i64]);
    let (ia, ib, id) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    let plan = Plan::new(
        Operand::new(&la, &ia),
        Operand::new(&lb, &ib),
        None,
        Operand::new(&ld, &id),
    )
    .unwrap()
    .with_threads(8);
    let s = Scoped {
        width: 1,
        calls: AtomicUsize::new(0),
        last_p: AtomicUsize::new(0),
        refuse: false,
    };
    let mut out = vec![0.0; M * N];
    plan.run_with(
        &s,
        1.0,
        TensorView::new(&av, &la, &ia),
        TensorView::new(&bv, &lb, &ib),
        0.0,
        None,
        TensorViewMut::new(&mut out, &ld, &id),
    )
    .unwrap();
    assert_eq!(s.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn explicit_resolution_survives_host_refusal_without_reselection() {
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
    let host = Scoped {
        width: 4,
        calls: AtomicUsize::new(0),
        last_p: AtomicUsize::new(0),
        refuse: true,
    };
    // SAFETY: same valid buffers, resolution and serial bound as above.
    unsafe {
        tensorcontract::execute_resolved(
            &p,
            &rg,
            Some(&host),
            1.,
            a.as_ptr(),
            b.as_ptr(),
            0.,
            std::ptr::null(),
            actual.as_mut_ptr(),
        );
    }
    assert_eq!(actual, expected);
    assert_eq!(host.calls.load(Ordering::Relaxed), 1);
    assert!(
        CALLS.load(Ordering::Relaxed) > 0,
        "fallback reselected the plan's default family"
    );
}
