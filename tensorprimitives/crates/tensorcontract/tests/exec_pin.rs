//! A pinned tensorcontract partition (`PartitionMode::Pin`) must not widen the
//! SPMD team beyond the `Exec`'s budget, nor make a declined broadcast spawn
//! threads. Own test binary: it counts the process's OS threads.

use std::time::Duration;

use tensorcontract::{Layout, Operand, PartitionMode, Plan, TensorView, TensorViewMut};
use tprims_exec::{Exec, Pool};

fn os_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .map(|d| d.count())
        .unwrap_or(0)
}

fn gemm(exec: &Exec<'_>) -> Vec<f64> {
    let (m, n, k) = (256usize, 240usize, 200usize);
    let av: Vec<f64> = (0..m * k).map(|x| (x % 13) as f64).collect();
    let bv: Vec<f64> = (0..k * n).map(|x| (x % 7) as f64).collect();
    let la = Layout::col_major(&[m as i64, k as i64]);
    let lb = Layout::col_major(&[k as i64, n as i64]);
    let ld = Layout::col_major(&[m as i64, n as i64]);
    let (ia, ib, id) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    let plan = Plan::new(
        Operand::new(&la, &ia),
        Operand::new(&lb, &ib),
        None,
        Operand::new(&ld, &id),
    )
    .unwrap()
    .with_partition_mode(PartitionMode::Pin(4, 2));
    let mut out = vec![0.0; m * n];
    plan.run_with(
        exec,
        1.0,
        TensorView::new(&av, &la, &ia),
        TensorView::new(&bv, &lb, &ib),
        0.0,
        None,
        TensorViewMut::new(&mut out, &ld, &id),
    )
    .unwrap();
    out
}

#[test]
fn pinned_partition_respects_host_width_and_never_spawns() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let pool = Pool::borrow(&tp);
        // Budget below the pool size: the pinned 4x2 grid must shrink to it.
        let exec = Exec::rayon(&pool).with_budget(3).unwrap();
        let wide = gemm(&exec);
        assert_eq!(pool.stats().broadcasts, 1);
        // Declined broadcast (caller is a worker): must run serially, no spawn.
        let before = os_threads();
        let nested = exec.install(2, |_| gemm(&exec));
        assert_eq!(nested, wide);
        if before > 0 {
            assert_eq!(os_threads(), before);
        }
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("pinned partition test failed or deadlocked");
}
