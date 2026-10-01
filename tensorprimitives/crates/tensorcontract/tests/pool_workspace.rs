//! One pool is one workspace owner: every operation on it shares the storage,
//! and no other pool can see it.
use tensorcontract::{Layout, Operand, Plan};
use tprims_exec::{Exec, Pool};

const M: usize = 192;
const N: usize = 160;
const K: usize = 128;

/// One `ij,jk->ik` contraction on `width` workers of the pool.
fn run(exec: &Exec<'_>, width: usize) {
    let exec = exec.with_budget(width).unwrap();
    let av: Vec<f64> = (0..M * K).map(|x| (x % 11) as f64 - 5.0).collect();
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
    .with_kernel(tensorcontract::KernelChoice::Id("tc.scalar.f64.4x4".into()))
    .unwrap()
    .with_threads(width);
    let rg = plan.resolved::<f64>().unwrap();
    let mut out = vec![0.0; M * N];
    // SAFETY: the buffers are sized by their layouts and `beta` is zero, so `C`
    // is never read.
    unsafe {
        tensorcontract::execute_resolved(
            &plan,
            &rg,
            &exec,
            None,
            1.0,
            av.as_ptr(),
            bv.as_ptr(),
            0.0,
            std::ptr::null(),
            out.as_mut_ptr(),
        )
    };
}

#[test]
fn one_pool_lends_one_workspace_and_two_pools_never_share() {
    tprims_kernel_tensorcontract::register();
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool_a = Pool::borrow(&tp);
    let pool_b = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool_a);

    // The context answers with a workspace, and the only one it can answer with
    // is this pool's: the run below is what proves it did.
    assert!(exec.workspace().is_some(), "a pool lends a workspace");
    // Both pools start empty and stay empty until something runs on them, which
    // is what makes the run below a statement about *this* pool's arena.
    assert_eq!(pool_a.workspace().retained_bytes(), 0);
    assert_eq!(pool_b.workspace().retained_bytes(), 0);
    // A serial context owns nothing; its caller lends a provider explicitly.
    assert!(Exec::serial().workspace().is_none());

    // Two operations on one pool reuse its storage: it grows on the first and
    // stops growing after that.
    run(&exec, 3);
    let panel = pool_a.workspace().retained_panel_bytes();
    assert!(panel > 0, "the pool's workspace was never used");
    assert_eq!(
        pool_b.workspace().retained_bytes(),
        0,
        "an operation on one pool touched another pool's workspace"
    );
    // The panel a shape needs does not depend on which workers took part, so a
    // second run of the same shape must reuse the very same allocation.
    run(&exec, 3);
    assert_eq!(
        pool_a.workspace().retained_panel_bytes(),
        panel,
        "the second operation grew the pool's panel again"
    );
    // Trimming releases idle storage and leaves the pool usable.
    pool_a.trim_workspace();
    assert!(pool_a.workspace().retained_panel_bytes() <= panel);
    run(&exec, 3);
    assert!(pool_a.workspace().retained_panel_bytes() > 0);
}
