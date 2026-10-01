//! Regression: a forced kernel id resolves on the matrix GEMM packed engine
//! with no earlier call having registered the providers (single test, own
//! binary).
// The cplx families exist only on x86_64 (`tprims-kernel-cplx` is empty elsewhere), so
// off x86_64 there is no provider whose registration order could be tested.
#![cfg(target_arch = "x86_64")]
use num_complex::Complex;
use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{gemm_with, EngineChoice, Error, GemmConfig, MatIn};
use tprims_exec::Exec;
use tprims_gemm_kernel::{KernelChoice, SelectError};

#[test]
fn matrix_gemm_resolves_an_id_before_anything_registered_providers() {
    type C = Complex<f64>;
    let a = vec![C::new(1.0, 0.5); 16];
    let av = StridedView::new(&a, &[4, 4], &[1, 4], 0).unwrap();
    let mut c = vec![C::new(0.0, 0.0); 16];
    let mut cv = StridedViewMut::new(&mut c, &[4, 4], &[1, 4], 0).unwrap();
    let cfg = GemmConfig {
        engine: EngineChoice::Packed,
        kernel: KernelChoice::Id("cplx.avx2.c64.native.4x4".into()),
        ..Default::default()
    };
    let r = gemm_with(
        &Exec::serial(),
        &cfg,
        C::new(1.0, 0.0),
        MatIn::new(&av),
        MatIn::new(&av),
        C::new(0.0, 0.0),
        &mut cv,
    );
    match r {
        Ok(sel) => assert_eq!(sel.family_id, Some("cplx.avx2.c64.native.4x4")),
        Err(Error::Select(SelectError::CpuUnsupported { .. })) => {}
        Err(e) => panic!("the id must resolve: {e:?}"),
    }
}
