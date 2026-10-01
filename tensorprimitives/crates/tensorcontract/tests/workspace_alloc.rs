//! A steady-state execute must not allocate.
//!
//! Its own binary, because the counter is global: another test running in
//! parallel would show up in the count. The `Exec` is serial, so the whole
//! contraction runs on the calling thread and the measurement covers the
//! driver's own reuse path — the leased team set, worker buffers and scatter
//! vectors — rather than thread plumbing. A threaded host's first touch and
//! per-thread slot reuse are pinned in `tprims-exec/tests/workspace.rs`.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use tensorcontract::{driver_decisions, KernelChoice, Layout, Operand, Plan};
use tprims_exec::{ArenaProvider, Exec};
use tprims_gemm_kernel::ResolvedGemm;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static BIG: AtomicUsize = AtomicUsize::new(0);
/// Allocations at or above this size are "a buffer", not incidental bookkeeping.
const BIG_BYTES: usize = 1 << 14;

struct Counting;

// SAFETY: every method forwards to the system allocator unchanged; the
// counters are atomics with no aliasing.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        if l.size() >= BIG_BYTES {
            BIG.fetch_add(1, Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// One case's plan, resolution and operands, built before anything is measured.
struct Prepared {
    plan: Plan,
    resolution: ResolvedGemm<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    d: Vec<f64>,
    /// The caller-lent workspace a serial `Exec` has none of.
    workspace: ArenaProvider,
}

impl Prepared {
    fn new(id: &str, m: usize, n: usize, k: usize) -> Self {
        let la = Layout::col_major(&[m as i64, k as i64]);
        let lb = Layout::col_major(&[k as i64, n as i64]);
        let ld = Layout::col_major(&[m as i64, n as i64]);
        let (ia, ib, idd) = ([0i64, 2], [2i64, 1], [0i64, 1]);
        let plan = Plan::new(
            Operand::new(&la, &ia),
            Operand::new(&lb, &ib),
            None,
            Operand::new(&ld, &idd),
        )
        .unwrap()
        .with_kernel(KernelChoice::Id(id.into()))
        .unwrap()
        .with_threads(1);
        let resolution = plan.resolved::<f64>().unwrap();
        let data = |len: usize, seed: f64| {
            (0..len)
                .map(|i| ((i as f64 * 0.37 + seed) % 3.0) - 1.0)
                .collect()
        };
        Self {
            plan,
            resolution,
            a: data(la.storage_len() as usize, 0.0),
            b: data(lb.storage_len() as usize, 1.0),
            d: vec![0.0; ld.storage_len() as usize],
            workspace: ArenaProvider::new(),
        }
    }

    fn run(&mut self) {
        // SAFETY: the buffers are sized by their layouts, `beta` is zero so `C`
        // is never read, and `D` is borrowed exclusively here.
        unsafe {
            tensorcontract::execute_resolved(
                &self.plan,
                &self.resolution,
                &Exec::Serial,
                Some(&self.workspace),
                1.5,
                self.a.as_ptr(),
                self.b.as_ptr(),
                0.0,
                std::ptr::null(),
                self.d.as_mut_ptr(),
            )
        };
    }
}

const SHAPE: (usize, usize, usize) = (300, 300, 300);

#[test]
fn steady_state_execute_allocates_nothing() {
    tprims_kernel_tensorcontract::register();
    let (m, n, k) = SHAPE;

    let mut prepared = Prepared::new("tc.scalar.f64.4x4", m, n, k);
    prepared.run();
    let before = COUNT.load(Relaxed);
    prepared.run();
    prepared.run();
    assert_eq!(
        COUNT.load(Relaxed),
        before,
        "a steady-state execute allocated"
    );

    // A direct-B family must not ask for a B-sized buffer even on its first run:
    // the only large allocation allowed there is the packed A block.
    let mut direct_b = Prepared::new("portable.f64.4x4.direct-b", m, n, k);
    let decisions = driver_decisions(
        &direct_b.plan,
        &direct_b.resolution,
        std::ptr::null(),
        direct_b.d.as_mut_ptr(),
        0.0,
    );
    assert!(
        !decisions.pack_b_needed,
        "the case is only meaningful when B is read in place"
    );
    let big = BIG.load(Relaxed);
    direct_b.run();
    assert!(
        BIG.load(Relaxed) - big <= 1,
        "direct-B allocated {} large buffers; only the A block is expected",
        BIG.load(Relaxed) - big
    );
    let count = COUNT.load(Relaxed);
    let big = BIG.load(Relaxed);
    direct_b.run();
    assert_eq!(COUNT.load(Relaxed), count, "steady state allocated");
    assert_eq!(BIG.load(Relaxed), big, "steady state allocated a buffer");
}
