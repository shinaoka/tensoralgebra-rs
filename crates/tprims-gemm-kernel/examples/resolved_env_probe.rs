//! Standalone single-threaded check: retargeting uses frozen policy, not env.
use tprims_gemm_kernel::{KernelChoice, ResolvedGemm, SelectError};

fn main() -> Result<(), SelectError> {
    let rg = ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Auto, 1)?;
    assert_eq!(rg.effective_threads, 1);
    let before = (rg.mc, rg.kc, rg.nc);
    // SAFETY: this standalone process never creates another thread; its only
    // environment-reading library calls finished before these mutations.
    unsafe {
        std::env::set_var("TENSORCONTRACT_MC", "999");
        std::env::set_var("TENSORCONTRACT_KC", "999");
        std::env::set_var("TENSORCONTRACT_NC", "999");
        std::env::set_var("TENSORCONTRACT_BLOCKMODEL", "model");
        std::env::set_var("TENSORCONTRACT_MC_PCT", "999");
        std::env::set_var("TENSORCONTRACT_NC_PCT", "999");
    }
    let after = rg.with_threads(1)?;
    assert_eq!((after.mc, after.kc, after.nc), before);
    assert!(core::ptr::eq(rg.family(), after.family()));
    println!("frozen policy: width=1, blocks={before:?}; environment mutation ignored");
    Ok(())
}
