//! Separate, single-threaded check of cached process-default selection errors.
use tprims_kernel::{process_default, SelectError};
fn main() {
    // SAFETY: standalone program creates no threads; this precedes planning.
    unsafe {
        std::env::set_var("TPRIMS_GEMM_KERNEL", "does.not.exist");
    }
    let first = process_default::<f64>().unwrap_err();
    assert!(matches!(first, SelectError::UnknownId { .. }));
    // SAFETY: still single-threaded. The first selection error must be frozen.
    unsafe {
        std::env::set_var("TPRIMS_GEMM_KERNEL", "portable.f64.4x4");
    }
    assert_eq!(process_default::<f64>().unwrap_err(), first);
    println!("process-default error remains cached: {first}");
}
