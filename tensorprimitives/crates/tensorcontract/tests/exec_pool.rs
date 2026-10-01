//! tensorcontract's SPMD driver on a borrowed pool through `Exec::broadcast`.

use std::time::Duration;

use tensorcontract::spmd::Spmd;
use tensorcontract::{Layout, Operand, Plan, TensorView, TensorViewMut};
use tprims_exec::{Exec, Pool};

struct ExecSpmd<'a> {
    exec: &'a Exec<'a>,
    width: usize,
}

impl Spmd for ExecSpmd<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
        self.exec.broadcast(p, f).is_ok()
    }
}

const M: usize = 256;
const N: usize = 240;
const K: usize = 200;

fn gemm(spmd: Option<&dyn Spmd>) -> Vec<f64> {
    let av: Vec<f64> = (0..M * K).map(|x| (x % 13) as f64 - 6.0).collect();
    let bv: Vec<f64> = (0..K * N).map(|x| (x % 7) as f64 * 0.5).collect();
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
    let mut out = vec![0.0; M * N];
    let (a, b) = (
        TensorView::new(&av, &la, &ia),
        TensorView::new(&bv, &lb, &ib),
    );
    let d = TensorViewMut::new(&mut out, &ld, &id);
    match spmd {
        Some(s) => plan.run_with(s, 1.0, a, b, 0.0, None, d).unwrap(),
        None => plan.run(1.0, a, b, 0.0, None, d).unwrap(),
    }
    out
}

#[test]
fn contraction_runs_spmd_on_the_borrowed_pool() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let serial = gemm(None);
    let par = gemm(Some(&ExecSpmd {
        exec: &exec,
        width: 4,
    }));
    assert_eq!(par, serial);
    assert_eq!(pool.stats().broadcasts, 1);
}

#[test]
fn nested_contraction_on_a_worker_falls_back_serially() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let pool = Pool::borrow(&tp);
        let exec = Exec::rayon(&pool);
        let serial = gemm(None);
        let nested = exec.install(2, |_| {
            gemm(Some(&ExecSpmd {
                exec: &exec,
                width: 4,
            }))
        });
        assert_eq!(nested, serial);
        assert_eq!(pool.stats().broadcasts, 0);
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("nested SPMD deadlocked");
}
