//! Plan configuration is explicit: every knob that used to be a
//! `TENSORCONTRACT_*` variable is a builder, and the environment changes
//! nothing.
use tensorcontract::kernel::{ComplexMethod, Tuning};
use tensorcontract::{Layout, Operand, Orient, PartitionMode, Plan, RowBlock};
use tprims_kernel::{BlockingOverride, Isa, KernelForce};

fn plan() -> Plan {
    let la = Layout::col_major(&[96, 64]);
    let lb = Layout::col_major(&[64, 80]);
    let ld = Layout::col_major(&[96, 80]);
    Plan::new(
        Operand::new(&la, &[0, 2]),
        Operand::new(&lb, &[2, 1]),
        None,
        Operand::new(&ld, &[0, 1]),
    )
    .unwrap()
}

#[test]
fn a_pinned_scalar_kernel_and_a_blocking_override_reach_the_resolution() {
    let scalar = plan().with_tuning(Tuning {
        kernel_force: KernelForce::Scalar,
        blocking: BlockingOverride {
            kc: Some(8),
            ..Default::default()
        },
        ..Tuning::default()
    });
    let rg = scalar.resolved::<f64>().unwrap();
    assert_eq!(rg.family().isa, Isa::Portable);
    assert_eq!(rg.kc, 8);
    let c = scalar.resolved::<tensorcontract::C64>().unwrap();
    assert_eq!(c.family().isa, Isa::Portable);
    assert_eq!(scalar.complex_method(), ComplexMethod::Planar);
}

#[test]
fn orientation_row_block_and_partition_mode_are_builders() {
    let p = plan();
    assert_eq!(p.partition(4, 4), p.clone().with_threads(1).partition(4, 4));
    for swap in [false, true] {
        let forced = p.clone().with_orientation(Orient::Force(swap));
        assert_eq!(forced.transposes_gemm(4), swap);
    }
    let base = p.clone().with_row_block(RowBlock::Base);
    assert_eq!(base.row_block(&[(8, 4), (4, 4)]), None);
    let idx = p.clone().with_row_block(RowBlock::Index(1));
    assert_eq!(idx.row_block(&[(8, 4), (4, 4)]), Some(1));
    let rows = p
        .clone()
        .with_threads(4)
        .with_partition_mode(PartitionMode::Rows);
    assert_eq!(rows.partition(4, 4), (4, 1));
    let cols = p
        .clone()
        .with_threads(4)
        .with_partition_mode(PartitionMode::Cols);
    assert_eq!(cols.partition(4, 4), (1, 4));
    let pin = p.with_partition_mode(PartitionMode::Pin(3, 2));
    assert_eq!(pin.partition(4, 4), (3, 2));
}

#[test]
fn spellings_parse_and_unknown_ones_are_rejected() {
    assert_eq!(Orient::parse("swap"), Some(Orient::Force(true)));
    assert_eq!(Orient::parse("bogus"), None);
    assert_eq!(RowBlock::parse("idx=2"), Some(RowBlock::Index(2)));
    assert_eq!(RowBlock::parse("mr=16"), Some(RowBlock::Pin(16)));
    assert_eq!(RowBlock::parse("idx=x"), None);
    assert_eq!(PartitionMode::parse("4x2"), Some(PartitionMode::Pin(4, 2)));
    assert_eq!(PartitionMode::parse("legacy"), Some(PartitionMode::Rule));
    assert_eq!(PartitionMode::parse("4xq"), None);
    assert_eq!(KernelForce::parse("AVX2"), Some(KernelForce::Avx2));
    assert_eq!(KernelForce::parse("avx3"), None);
}

#[test]
fn the_process_environment_is_not_read() {
    let before = plan().resolved::<f64>().unwrap();
    // SAFETY: the only test in this binary that touches the environment.
    unsafe {
        std::env::set_var("TENSORCONTRACT_KERNEL", "scalar");
        std::env::set_var("TENSORCONTRACT_KC", "999");
        std::env::set_var("TENSORCONTRACT_THREADS", "7");
        std::env::set_var("TENSORCONTRACT_ORIENT", "swap");
        std::env::set_var("TENSORCONTRACT_PARTITION", "9x9");
        std::env::set_var("TPRIMS_GEMM_KERNEL", "does.not.exist");
    }
    let p = plan();
    let after = p.resolved::<f64>().unwrap();
    assert_eq!(before.family().id, after.family().id);
    assert_eq!(
        (before.mc, before.kc, before.nc),
        (after.mc, after.kc, after.nc)
    );
    assert_eq!(p.threads(), 1);
    assert_eq!(p.transposes_gemm(4), plan().transposes_gemm(4));
    assert_eq!(p.partition(4, 4), (1, 1));
}
