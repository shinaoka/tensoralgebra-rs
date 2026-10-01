//! The native interleaved AVX2+FMA complex families through the normal
//! `Engine::Packed` entry (GEMM and TBLIS-style batched GEMM): listed, selected
//! by id, reported, and never chosen by default.
use num_complex::Complex;
use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{
    gemm_batched_with, gemm_with, list_kernels, BatchIn, BatchStrategy, Conj, Engine, EngineChoice,
    Error, GemmConfig, MatIn,
};
use tprims_exec::Exec;
use tprims_gemm_kernel::{KernelChoice, Method};

type C64 = Complex<f64>;
type C32 = Complex<f32>;

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

fn z(i: usize, s: usize) -> (f64, f64) {
    (
        ((i * 7 + s * 3) % 13) as f64 / 6.0 - 1.0,
        ((i * 5 + s) % 11) as f64 / 5.0 - 1.0,
    )
}

macro_rules! gemm_case {
    ($name:ident, $c:ty, $id:literal, $tol:expr) => {
        #[test]
        fn $name() {
            let (m, n, k) = (23usize, 19usize, 31usize);
            let mk = |len: usize, s: usize| -> Vec<$c> {
                (0..len)
                    .map(|i| {
                        let (r, im) = z(i, s);
                        <$c>::new(r as _, im as _)
                    })
                    .collect()
            };
            let a = mk(m * k, 1);
            let b = mk(k * n, 2);
            let av = StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap();
            let bv = StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap();
            let run = |cfg: &GemmConfig, ca: Conj, cb: Conj| {
                let mut c = mk(m * n, 3);
                let mut cv = StridedViewMut::new(&mut c, &[m, n], &[1, m as isize], 0).unwrap();
                let alpha = <$c>::new(0.7, -0.2);
                let beta = <$c>::new(0.5, 0.3);
                let mut ai = MatIn::new(&av);
                ai.conj = ca;
                let mut bi = MatIn::new(&bv);
                bi.conj = cb;
                let sel = gemm_with(&Exec::serial(), cfg, alpha, ai, bi, beta, &mut cv).unwrap();
                (sel, c)
            };
            let faer = GemmConfig::default();
            let packed = GemmConfig {
                engine: EngineChoice::Packed,
                kernel: KernelChoice::Id($id.into()),
                ..Default::default()
            };
            for (ca, cb) in [
                (Conj::No, Conj::No),
                (Conj::Yes, Conj::No),
                (Conj::No, Conj::Yes),
                (Conj::Yes, Conj::Yes),
            ] {
                let (_, want) = run(&faer, ca, cb);
                if !have_isa() {
                    // Refused at plan creation, never executed.
                    let mut c = mk(m * n, 3);
                    let mut cv = StridedViewMut::new(&mut c, &[m, n], &[1, m as isize], 0).unwrap();
                    let err = gemm_with(
                        &Exec::serial(),
                        &packed,
                        <$c>::new(1.0, 0.0),
                        MatIn::new(&av),
                        MatIn::new(&bv),
                        <$c>::new(0.0, 0.0),
                        &mut cv,
                    )
                    .unwrap_err();
                    assert!(matches!(
                        err,
                        Error::Select(tprims_gemm_kernel::SelectError::CpuUnsupported { .. })
                    ));
                    return;
                }
                let (sel, got) = run(&packed, ca, cb);
                assert_eq!(sel.engine, Engine::Packed);
                assert_eq!(sel.family_id, Some($id));
                let s = sel.complex.expect("complex scheme");
                assert_eq!(s.method, Method::Native);
                let err = got
                    .iter()
                    .zip(&want)
                    .map(|(g, w)| ((g.re - w.re) as f64).hypot((g.im - w.im) as f64))
                    .fold(0.0f64, f64::max);
                assert!(err < $tol, "{}: max abs error {err:e}", $id);
            }
        }
    };
}
gemm_case!(
    c64_gemm_by_id_matches_faer,
    C64,
    "cplx.avx2.c64.native.4x4",
    1e-12
);
gemm_case!(
    c32_gemm_by_id_matches_faer,
    C32,
    "cplx.avx2.c32.native.8x4",
    2e-4
);

#[test]
fn the_families_are_listed_and_the_default_is_unchanged() {
    for (id, info) in [
        ("cplx.avx2.c64.native.4x4", list_kernels::<C64>()),
        ("cplx.avx2.c32.native.8x4", list_kernels::<C32>()),
    ] {
        let k = info.iter().find(|k| k.id == id).expect("listed");
        assert_eq!(k.crate_name, "tprims-kernel-cplx");
        assert_eq!(k.license, "MIT OR Apache-2.0");
        assert_eq!(k.available_on_this_cpu, have_isa());
        let s = k.complex.unwrap();
        assert_eq!(s.method, Method::Native);
    }
    // Default packed selection never lands on them.
    if have_isa() {
        let a = vec![C64::new(1.0, 0.5); 16];
        let av = StridedView::new(&a, &[4, 4], &[1, 4], 0).unwrap();
        let mut c = vec![C64::new(0.0, 0.0); 16];
        let mut cv = StridedViewMut::new(&mut c, &[4, 4], &[1, 4], 0).unwrap();
        let sel = gemm_with(
            &Exec::serial(),
            &GemmConfig {
                engine: EngineChoice::Packed,
                ..Default::default()
            },
            C64::new(1.0, 0.0),
            MatIn::new(&av),
            MatIn::new(&av),
            C64::new(0.0, 0.0),
            &mut cv,
        )
        .unwrap();
        assert!(!sel.family_id.unwrap().starts_with("cplx."));
    }
}

#[test]
fn tblis_batched_gemm_runs_the_family_by_id() {
    if !have_isa() {
        return;
    }
    let (m, n, k, nb) = (9usize, 7usize, 13usize, 3usize);
    let mk = |len: usize, s: usize| -> Vec<C64> {
        (0..len)
            .map(|i| {
                let (r, im) = z(i, s);
                C64::new(r, im)
            })
            .collect()
    };
    let a = mk(m * k * nb, 1);
    let b = mk(k * n * nb, 2);
    let av = StridedView::new(&a, &[m, k, nb], &[1, m as isize, (m * k) as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[k, n, nb], &[1, k as isize, (k * n) as isize], 0).unwrap();
    let run = |cfg: &GemmConfig, strat| {
        let mut c = vec![C64::new(0.0, 0.0); m * n * nb];
        let mut cv =
            StridedViewMut::new(&mut c, &[m, n, nb], &[1, m as isize, (m * n) as isize], 0)
                .unwrap();
        let sel = gemm_batched_with(
            &Exec::serial(),
            cfg,
            C64::new(1.0, 0.0),
            BatchIn::new(&av),
            BatchIn::new(&bv),
            C64::new(0.0, 0.0),
            &mut cv,
            strat,
        )
        .unwrap();
        (sel, c)
    };
    let (_, want) = run(&GemmConfig::default(), BatchStrategy::FaerLoop);
    let cfg = GemmConfig {
        kernel: KernelChoice::Id("cplx.avx2.c64.native.4x4".into()),
        ..Default::default()
    };
    // The built-in TBLIS strategy keeps its historical report (no family id);
    // that the forced id is consulted is shown by a wrong-dtype id failing.
    let (_, got) = run(&cfg, BatchStrategy::Tblis);
    let wrong = GemmConfig {
        kernel: KernelChoice::Id("cplx.avx2.c32.native.8x4".into()),
        ..Default::default()
    };
    let mut c = vec![C64::new(0.0, 0.0); m * n * nb];
    let mut cv =
        StridedViewMut::new(&mut c, &[m, n, nb], &[1, m as isize, (m * n) as isize], 0).unwrap();
    let err = gemm_batched_with(
        &Exec::serial(),
        &wrong,
        C64::new(1.0, 0.0),
        BatchIn::new(&av),
        BatchIn::new(&bv),
        C64::new(0.0, 0.0),
        &mut cv,
        BatchStrategy::Tblis,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Select(tprims_gemm_kernel::SelectError::DtypeMismatch { .. })
        ),
        "{err:?}"
    );
    let err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).norm())
        .fold(0.0, f64::max);
    assert!(err < 1e-12, "max abs error {err:e}");
}
