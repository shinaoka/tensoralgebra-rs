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
