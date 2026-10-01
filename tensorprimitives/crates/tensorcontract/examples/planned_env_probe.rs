//! Standalone 1T planning/execution environment-stability check (not a benchmark).
use tensorcontract::{Blocking, Layout, Operand, Plan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let l = Layout::col_major(&[2, 2]);
    let p = Plan::new(
        Operand::new(&l, &[0, 2]),
        Operand::new(&l, &[2, 1]),
        None,
        Operand::new(&l, &[0, 1]),
    )?
    .with_threads(1)
    .with_blocking(Blocking {
        mc: 4,
        kc: 1,
        nc: 4,
    });
    let before = p.resolved::<f64>()?;
    let default_before = tprims_kernel::process_default::<f64>()?;
    assert_eq!(before.effective_threads, 1);
    let swap = p.transposes_gemm(before.mr);
    // SAFETY: standalone process creates no other threads. All environment
    // facts used by this typed plan have already been captured by planning.
    unsafe {
        for (key, value) in [
            ("TPRIMS_GEMM_KERNEL", "does.not.exist"),
            ("TENSORCONTRACT_KERNEL", "scalar"),
            ("TENSORCONTRACT_ORIENT", "swap"),
            ("TENSORCONTRACT_PARTITION", "9x9"),
            ("TENSORCONTRACT_WRITEBACK", "gather"),
            ("TENSORCONTRACT_MC", "999"),
            ("TENSORCONTRACT_NC", "999"),
            ("TENSORCONTRACT_KC", "999"),
        ] {
            std::env::set_var(key, value);
        }
    }
    let after = p.resolved::<f64>()?;
    assert!(core::ptr::eq(
        default_before,
        tprims_kernel::process_default::<f64>()?
    ));
    assert!(core::ptr::eq(before.family(), after.family()));
    assert_eq!(
        (before.mc, before.kc, before.nc),
        (after.mc, after.kc, after.nc)
    );
    assert_eq!(p.transposes_gemm(after.mr), swap);
    let a = [1., 2., 3., 4.];
    let b = [5., 6., 7., 8.];
    let mut d = [f64::NAN; 4];
    // SAFETY: complete column-major buffers; no C reads with beta=0. Resolution
    // above proves selection/blocking, including the serial width bound.
    unsafe {
        p.run_raw(
            1.,
            a.as_ptr(),
            b.as_ptr(),
            0.,
            std::ptr::null(),
            d.as_mut_ptr(),
        );
    }
    assert_eq!(d, [23., 34., 31., 46.]);
    println!(
        "width=1; {} remains selected; result={d:?}",
        after.family().id
    );
    Ok(())
}
