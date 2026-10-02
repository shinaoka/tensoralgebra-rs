//! The packed SPMD driver on a borrowed pool through `Exec::broadcast`.

use std::time::Duration;

use strided_view::{StridedView, StridedViewMut};
use tprims_exec::{Exec, Pool};

mod common;
use common::plans::packed_plan;

const M: usize = 256;
const N: usize = 240;
const K: usize = 200;

fn gemm(exec: &Exec<'_>) -> Vec<f64> {
    let av: Vec<f64> = (0..M * K).map(|x| (x % 13) as f64 - 6.0).collect();
    let bv: Vec<f64> = (0..K * N).map(|x| (x % 7) as f64 * 0.5).collect();
    let plan = packed_plan(M, N, K);
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
fn contraction_runs_spmd_on_the_borrowed_pool() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let serial = gemm(&Exec::serial());
    let par = gemm(&exec);
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
        let serial = gemm(&Exec::serial());
        let nested = exec.install(2, |_| gemm(&exec));
        assert_eq!(nested, serial);
        assert_eq!(pool.stats().broadcasts, 0);
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("nested SPMD deadlocked");
}
