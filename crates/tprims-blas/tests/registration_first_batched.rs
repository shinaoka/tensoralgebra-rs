//! Regression: a forced kernel id must resolve on the batched TBLIS path with
//! no earlier call having registered the providers. This binary holds exactly
//! one test, so nothing else can register first.
use num_complex::Complex;
use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{gemm_batched_with, BatchIn, BatchStrategy, GemmConfig};
use tprims_exec::Exec;
use tprims_gemm_kernel::KernelChoice;

#[test]
fn batched_tblis_resolves_an_id_before_anything_registered_providers() {
    type C = Complex<f64>;
    let (m, n, k, nb) = (4usize, 4usize, 4usize, 2usize);
    let a = vec![C::new(1.0, 0.5); m * k * nb];
    let b = vec![C::new(0.5, -1.0); k * n * nb];
    let av = StridedView::new(&a, &[m, k, nb], &[1, m as isize, (m * k) as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[k, n, nb], &[1, k as isize, (k * n) as isize], 0).unwrap();
    let mut c = vec![C::new(0.0, 0.0); m * n * nb];
    let mut cv =
        StridedViewMut::new(&mut c, &[m, n, nb], &[1, m as isize, (m * n) as isize], 0).unwrap();
    // An id only `tprims-blas`'s own registration provides: UnknownId means the
    // batched path resolved it before registering. A host without AVX2+FMA
    // answers CpuUnsupported, which is also proof the id was known.
    let cfg = GemmConfig {
        kernel: KernelChoice::Id("cplx.avx2.c64.native.4x4".into()),
        ..Default::default()
    };
    let r = gemm_batched_with(
        &Exec::serial(),
        &cfg,
        C::new(1.0, 0.0),
        BatchIn::new(&av),
        BatchIn::new(&bv),
        C::new(0.0, 0.0),
        &mut cv,
        BatchStrategy::Tblis,
    );
    match r {
        Ok(_) => {
            // (1+0.5i)(0.5-i) = 1 - 0.75i per term, four terms.
            assert!(c
                .iter()
                .all(|z| (z.re - 4.0).abs() < 1e-12 && (z.im + 3.0).abs() < 1e-12));
        }
        Err(tprims_blas::Error::Select(tprims_gemm_kernel::SelectError::CpuUnsupported {
            ..
        })) => {}
        Err(e) => panic!("the id must resolve on the batched path: {e:?}"),
    }
}
