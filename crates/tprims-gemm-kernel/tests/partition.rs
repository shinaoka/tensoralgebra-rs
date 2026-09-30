//! The strip rule: strips tile the rows, land on `MR` (or alignment)
//! boundaries, and always claim the tail.
use tprims_gemm_kernel::partition::*;
use tprims_gemm_kernel::*;

#[test]
fn strips_tile_the_rows_on_mr_multiples() {
    for &(m, mr, pm) in &[
        (1usize, 8usize, 1usize),
        (37, 8, 3),
        (1000, 6, 7),
        (8, 8, 8),
    ] {
        let mut next = 0;
        for r in 0..pm {
            let (lo, hi) = strip(r, pm, m, mr, 0);
            assert_eq!(lo, next, "strip {r} of {pm} over {m} rows");
            assert!(lo % mr == 0 || lo == m);
            assert!(hi >= lo && hi <= m);
            next = hi;
        }
        assert_eq!(next, m);
    }
}

#[test]
fn aligned_strips_round_to_the_alignment() {
    for r in 1..4 {
        let (lo, hi) = strip(r, 4, 1000, 6, 24);
        assert_eq!(lo % 24, 0, "strip {r} starts at {lo}");
        assert!(hi >= lo);
    }
    // The tail is still claimed, whatever rounding does to the last boundary.
    assert_eq!(strip(3, 4, 1000, 6, 24).1, 1000);
    assert_eq!(strip(9, 10, 101, 8, 64).1, 101);
}

#[test]
fn dynamic_tiles_is_not_implemented() {
    let err = ResolvedGemm::<f64>::resolve_with::<f64>(
        &KernelChoice::Auto,
        4,
        PartitionPolicy::DynamicTiles {
            job_m: 16,
            job_n: 32,
        },
        PartitionOpts::default(),
    );
    assert!(matches!(err, Err(SelectError::NotImplemented { .. })));
}

#[test]
fn half_specified_static_grid_is_rejected() {
    let err = ResolvedGemm::<f64>::resolve_with::<f64>(
        &KernelChoice::Auto,
        4,
        PartitionPolicy::StaticGrid { pm: 2, pn: 0 },
        PartitionOpts::default(),
    );
    assert!(matches!(err, Err(SelectError::Incompatible { .. })));
}

#[test]
fn an_explicit_grid_survives_retargeting() {
    let grid = PartitionPolicy::StaticGrid { pm: 3, pn: 2 };
    let opts = PartitionOpts {
        align_c_lines: true,
    };
    let rg = ResolvedGemm::<f64>::resolve_with::<f64>(&KernelChoice::Auto, 6, grid, opts).unwrap();
    assert_eq!(rg.partition, grid);
    assert!(rg.opts.align_c_lines);
    // Re-deriving the blocking for another width must not drop the policy.
    let serial = rg.with_threads(1).unwrap();
    assert_eq!(serial.partition, grid);
    assert_eq!(serial.opts, opts);
}
