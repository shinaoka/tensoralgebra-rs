//! The native interleaved complex families through the TBLIS-style contraction
//! path (`Strategy::Tblis`): a contraction whose operands cannot fuse, forced
//! onto `cplx.avx2.c64.native.4x4`, reported, and equal to the reference.
use num_complex::Complex64;
use tprims_blas::{Conj, Engine, GemmConfig};
use tprims_contract::{ContractPlan, DotGeneral, Flags, Strategy};
use tprims_exec::Exec;
use tprims_kernel::KernelChoice;

mod common;
use common::{out_dims, reference, T};

fn have_isa() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[test]
fn a_tblis_contraction_runs_the_native_family_and_matches_the_reference() {
    // Contract A[., 1, 2] with B[2, ., 0] (the copying case of kernel_select),
    // plus a batch axis; B stored transposed and A reversed so nothing fuses.
    let cfg = DotGeneral::new(&[1, 2], &[2, 0], &[], &[]);
    let a0 = T::<Complex64>::new(&[5, 4, 6], 1);
    let b0 = T::<Complex64>::new(&[6, 7, 4], 2);
    let a = a0.reversed();
    let b = b0.restride(&[2, 0, 1]);
    let od = out_dims(&cfg, &a.dims, &b.dims);
    let c0 = T::<Complex64>::new(&od, 3);
    let gemm = GemmConfig {
        kernel: KernelChoice::Id("cplx.avx2.c64.native.4x4".into()),
        ..Default::default()
    };
    let plan = ContractPlan::<Complex64>::new_with(
        &gemm,
        &cfg,
        (&a.dims, &a.strides),
        (&b.dims, &b.strides),
        (&c0.dims, &c0.strides),
        (Conj::Yes, Conj::No),
        Strategy::Tblis,
        Flags::default(),
    );
    if !have_isa() {
        // Refused at plan creation (or execution-time resolution), never run.
        let refused = match plan {
            Err(_) => true,
            Ok(p) => p.selected_gemm().is_err(),
        };
        assert!(refused, "an unavailable family must be refused");
        return;
    }
    let plan = plan.unwrap();
    let sel = plan.selected_gemm().unwrap().expect("packed plan");
    assert_eq!(sel.engine, Engine::Packed);
    assert_eq!(sel.family_id, Some("cplx.avx2.c64.native.4x4"));
    let (alpha, beta) = (Complex64::new(0.8, -0.3), Complex64::new(0.4, 0.2));
    let mut c = c0.clone();
    plan.execute(
        &Exec::serial(),
        alpha,
        &a.view(),
        &b.view(),
        beta,
        &mut c.view_mut(),
    )
    .unwrap();
    let want = reference(&cfg, alpha, &a, Conj::Yes, &b, Conj::No, beta, &c0);
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    common::for_each_index(&c.dims, |i| {
        worst = worst.max((c.get(i) - want.get(i)).norm());
        scale = scale.max(want.get(i).norm());
    });
    // K = 4 * 6 = 24 products of O(1) values: a few K eps.
    assert!(
        worst <= 24.0 * 8.0 * f64::EPSILON * scale.max(1.0),
        "{worst:e}"
    );
}
