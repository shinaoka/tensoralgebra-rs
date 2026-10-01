//! Regression: a forced family id resolves in a contraction plan with no
//! earlier call having registered the providers (single test, own binary).
use num_complex::Complex64;
use tprims_blas::{Conj, GemmConfig};
use tprims_contract::{ContractPlan, DotGeneral, Flags, Strategy};
use tprims_gemm_kernel::KernelChoice;

#[test]
fn a_forced_id_resolves_on_the_tblis_contraction_path_first_thing() {
    let cfg = DotGeneral::new(&[1], &[0], &[], &[]);
    let (a_dims, b_dims, c_dims) = ([3usize, 4], [4usize, 5], [3usize, 5]);
    let (sa, sb, sc) = ([1isize, 3], [1isize, 4], [1isize, 3]);
    let gemm = GemmConfig {
        kernel: KernelChoice::Id("cplx.avx2.c64.native.4x4".into()),
        ..Default::default()
    };
    let plan = ContractPlan::<Complex64>::new_with(
        &gemm,
        &cfg,
        (&a_dims, &sa),
        (&b_dims, &sb),
        (&c_dims, &sc),
        (Conj::No, Conj::No),
        Strategy::Tblis,
        Flags::default(),
    );
    // UnknownId would mean the plan resolved the id before registering; a host
    // without AVX2+FMA refuses it as unsupported, which also proves it is known.
    match plan {
        Ok(p) => assert_eq!(
            p.selected_gemm().unwrap().unwrap().family_id,
            Some("cplx.avx2.c64.native.4x4")
        ),
        Err(e) => assert!(!format!("{e:?}").contains("UnknownId"), "{e:?}"),
    }
}
