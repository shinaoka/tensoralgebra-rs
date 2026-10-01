use tprims_gemm_kernel::{process_default, Families};

#[test]
fn defaults_are_cached_per_storage_dtype() {
    fn check<T: Families>() {
        let first = process_default::<T>().unwrap();
        let again = process_default::<T>().unwrap();
        assert!(core::ptr::eq(first, again));
        assert_eq!(first.family().complex.is_some(), T::IS_COMPLEX);
        assert_eq!(first.effective_threads, tprims_gemm_kernel::env_threads());
    }
    check::<f32>();
    check::<f64>();
    check::<tprims_gemm_kernel::C32>();
    check::<tprims_gemm_kernel::C64>();
}
