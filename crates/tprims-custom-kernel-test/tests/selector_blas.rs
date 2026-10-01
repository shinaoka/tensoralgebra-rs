//! The downstream crate's own kernels through `tprims-blas`: matrix, batched
//! and grouped entrypoints, with a captured selector and nothing registered
//! globally.
use std::cell::Cell;

use num_complex::Complex;
use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{
    gemm_batched_with_selector, gemm_grouped_with_selector, gemm_with_selector, BatchIn, Conj,
    EngineChoice, Error, GemmConfig, GroupedJob, KernelCatalog, MatIn, SelectError,
};
use tprims_custom_kernel_test as own;
use tprims_exec::{Exec, Pool};
use tprims_gemm_kernel::KernelChoice;

fn catalog<T: tprims_gemm_kernel::Families>(
    list: &'static [&'static tprims_gemm_kernel::KernelFamily<T::Real>],
) -> KernelCatalog<T> {
    // SAFETY: this crate's kernels honour the documented tile ABI over their
    // declared geometry; they are static, stateless, thread-safe, never panic
    // and require no CPU feature.
    unsafe { KernelCatalog::<T>::from_static_families(list) }.expect("valid descriptors")
}

fn ints(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|x| ((x * 7 + seed * 3) % 11) as f64 - 5.0)
        .collect()
}

/// `C = alpha A B + beta C` with explicit strides, the oracle.
#[allow(clippy::too_many_arguments)]
fn naive(
    (m, n, k): (usize, usize, usize),
    alpha: f64,
    a: &[f64],
    sa: [isize; 2],
    b: &[f64],
    sb: [isize; 2],
    beta: f64,
    c: &mut [f64],
    sc: [isize; 2],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[(i as isize * sa[0] + p as isize * sa[1]) as usize]
                    * b[(p as isize * sb[0] + j as isize * sb[1]) as usize];
            }
            let at = (i as isize * sc[0] + j as isize * sc[1]) as usize;
            c[at] = alpha * sum + if beta == 0.0 { 0.0 } else { beta * c[at] };
        }
    }
}

type Pick<'a> = &'a dyn Fn(&tprims_blas::SelectionContext<'_>) -> &'static str;

fn pick_by<T: tprims_gemm_kernel::Families>(
    cat: &KernelCatalog<T>,
    pick: Pick<'_>,
    ctx: &tprims_blas::SelectionContext<'_>,
) -> Result<tprims_blas::KernelHandle<T>, SelectError> {
    Ok(cat.get(pick(ctx)).expect("id in catalog"))
}

#[test]
fn shapes_choose_different_custom_kernels_and_diagnostics_name_them() {
    let cat = catalog::<f64>(own::f64_families());
    let calls = Cell::new(0);
    let by_size = |ctx: &tprims_blas::SelectionContext<'_>| {
        if ctx.stats.m * ctx.stats.n <= 16 {
            "custom.f64.2x2"
        } else {
            "custom.f64.3x4"
        }
    };
    for (m, n, k, expect) in [(4, 4, 5, "custom.f64.2x2"), (30, 25, 17, "custom.f64.3x4")] {
        let (a, b) = (ints(m * k, 1), ints(k * n, 2));
        let mut c = ints(m * n, 3);
        let mut want = c.clone();
        naive(
            (m, n, k),
            2.0,
            &a,
            [1, m as isize],
            &b,
            [1, k as isize],
            -1.0,
            &mut want,
            [1, m as isize],
        );
        let av = StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap();
        let bv = StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap();
        let mut cv = StridedViewMut::new(&mut c, &[m, n], &[1, m as isize], 0).unwrap();
        let report = gemm_with_selector(
            &Exec::serial(),
            &GemmConfig::default(),
            &cat,
            |ctx, cands| {
                calls.set(calls.get() + 1);
                assert_eq!(cands.len(), 2, "both kernels are admissible");
                assert!(cands.iter().all(|c| c.handle.origin() == own::ORIGIN));
                assert_eq!((ctx.stats.m, ctx.stats.n, ctx.stats.k), (m, n, k));
                assert_eq!(ctx.dtype, "f64");
                pick_by(&cat, &by_size, ctx)
            },
            2.0,
            MatIn::new(&av),
            MatIn::new(&bv),
            -1.0,
            &mut cv,
        )
        .unwrap();
        assert_eq!(c, want);
        assert_eq!(report.family_id, Some(expect));
        assert_eq!(report.origin, Some(own::ORIGIN));
        assert!(report.to_json().contains("tprims-custom-kernel-test"));
    }
    assert_eq!(calls.get(), 2, "one selector call per call");
}

#[test]
fn arbitrary_strides_scaling_and_a_pool_stay_correct() {
    let cat = catalog::<f64>(own::f64_families());
    let (m, n, k) = (40usize, 37usize, 29usize);
    // A transposed, B column-major, C row-major with a strided leading axis.
    let a = ints(m * k, 4);
    let b = ints(k * n, 5);
    let sa = [k as isize, 1];
    let sb = [1, k as isize];
    let sc = [2 * n as isize, 2];
    let mut c = ints(2 * m * n * 2, 6);
    let mut want = c.clone();
    naive((m, n, k), 0.5, &a, sa, &b, sb, 2.0, &mut want, sc);
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let av = StridedView::new(&a, &[m, k], &sa, 0).unwrap();
    let bv = StridedView::new(&b, &[k, n], &sb, 0).unwrap();
    let mut cv = StridedViewMut::new(&mut c, &[m, n], &sc, 0).unwrap();
    gemm_with_selector(
        &Exec::rayon(&pool),
        &GemmConfig {
            engine: EngineChoice::Packed,
            ..Default::default()
        },
        &cat,
        |_, cands| Ok(cands[1].handle),
        0.5,
        MatIn::new(&av),
        MatIn::new(&bv),
        2.0,
        &mut cv,
    )
    .unwrap();
    assert_eq!(c, want);
}

#[test]
fn a_direct_b_custom_kernel_updates_c_in_place_and_reports_itself() {
    let cat = catalog::<f64>(own::f64_direct_families());
    let (m, n, k) = (23usize, 21usize, 13usize);
    let (a, b) = (ints(m * k, 7), ints(k * n, 8));
    let mut c = ints(m * n, 9);
    let mut want = c.clone();
    naive(
        (m, n, k),
        1.0,
        &a,
        [1, m as isize],
        &b,
        [1, k as isize],
        3.0,
        &mut want,
        [1, m as isize],
    );
    let av = StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap();
    let mut cv = StridedViewMut::new(&mut c, &[m, n], &[1, m as isize], 0).unwrap();
    let mut saw_direct = false;
    let report = gemm_with_selector(
        &Exec::serial(),
        &GemmConfig::default(),
        &cat,
        |_, cands| {
            saw_direct = cands[0].direct_b;
            Ok(cands[0].handle)
        },
        1.0,
        MatIn::new(&av),
        MatIn::new(&bv),
        3.0,
        &mut cv,
    )
    .unwrap();
    assert_eq!(c, want);
    assert!(saw_direct, "unit-stride B is eligible for in-place reads");
    assert_eq!(report.family_id, Some("custom.f64.4x4.direct-b"));
}

#[test]
fn custom_complex_families_need_a_dtype_correct_handle() {
    // c64 with a conjugated A.
    let cat = catalog::<Complex<f64>>(own::c64_families());
    let (m, n, k) = (7usize, 5usize, 6usize);
    let z = |len: usize, s: usize| -> Vec<Complex<f64>> {
        ints(len, s)
            .into_iter()
            .zip(ints(len, s + 50))
            .map(|(re, im)| Complex::new(re, im))
            .collect()
    };
    let (a, b) = (z(m * k, 1), z(k * n, 2));
    let mut c = vec![Complex::new(0.0, 0.0); m * n];
    let mut want = c.clone();
    for i in 0..m {
        for j in 0..n {
            let mut s = Complex::new(0.0, 0.0);
            for p in 0..k {
                s += a[i + p * m].conj() * b[p + j * k];
            }
            want[i + j * m] = s;
        }
    }
    let av = StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap();
    let mut cv = StridedViewMut::new(&mut c, &[m, n], &[1, m as isize], 0).unwrap();
    let one = Complex::new(1.0, 0.0);
    let report = gemm_with_selector(
        &Exec::serial(),
        &GemmConfig::default(),
        &cat,
        |ctx, cands| {
            assert!(ctx.is_complex);
            assert_eq!(ctx.dtype, "c64");
            assert!(ctx.operands[0].conj);
            Ok(cands[0].handle)
        },
        one,
        MatIn::new(&av).conj(),
        MatIn::new(&bv),
        Complex::new(0.0, 0.0),
        &mut cv,
    )
    .unwrap();
    assert_eq!(report.family_id, Some("custom.c64.native.2x2"));
    assert!(report.complex.is_some());
    assert_eq!(c, want);

    // c32 admits its own complex family; a real list is refused for complex
    // storage and vice versa.
    assert!(catalog_result::<Complex<f32>>(own::c32_families()).is_ok());
    assert!(matches!(
        catalog_result::<Complex<f64>>(own::f64_families()),
        Err(SelectError::DtypeMismatch { dtype: "c64", .. })
    ));
    assert!(matches!(
        catalog_result::<f64>(own::c64_families()),
        Err(SelectError::DtypeMismatch { dtype: "f64", .. })
    ));
    assert!(catalog_result::<f32>(own::f32_families()).is_ok());
}

fn catalog_result<T: tprims_gemm_kernel::Families>(
    list: &'static [&'static tprims_gemm_kernel::KernelFamily<T::Real>],
) -> Result<KernelCatalog<T>, SelectError> {
    // SAFETY: as in `catalog`.
    unsafe { KernelCatalog::<T>::from_static_families(list) }
}

/// Run a trivial 6x6x6 f64 problem; returns the call result.
fn tiny(
    cfg: &GemmConfig,
    cat: &KernelCatalog<f64>,
    dims: (usize, usize, usize),
    selector: impl FnOnce(
        &tprims_blas::SelectionContext<'_>,
        &[tprims_blas::KernelCandidate<f64>],
    ) -> Result<tprims_blas::KernelHandle<f64>, SelectError>,
    alpha: f64,
) -> (Result<tprims_blas::SelectedGemm, Error>, Vec<f64>) {
    let (m, n, k) = dims;
    let (a, b) = (ints((m * k).max(1), 1), ints((k * n).max(1), 2));
    let mut c = vec![1.0; (m * n).max(1)];
    let av = StridedView::new(&a, &[m, k], &[1, m.max(1) as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[k, n], &[1, k.max(1) as isize], 0).unwrap();
    let mut cv = StridedViewMut::new(&mut c, &[m, n], &[1, m.max(1) as isize], 0).unwrap();
    let r = gemm_with_selector(
        &Exec::serial(),
        cfg,
        cat,
        selector,
        alpha,
        MatIn::new(&av),
        MatIn::new(&bv),
        2.0,
        &mut cv,
    );
    (r, c)
}

#[test]
fn bad_choices_fail_before_compute_including_for_empty_problems() {
    let cat = catalog::<f64>(own::f64_families());
    let other = catalog::<f64>(own::f64_families());
    let masked = catalog::<f64>(own::f64_impossible_families());
    let cfg = GemmConfig::default();
    for dims in [(6, 6, 6), (0, 6, 6), (6, 6, 0)] {
        // A handle from another catalog.
        let foreign = other.get("custom.f64.2x2").unwrap();
        let (r, c) = tiny(&cfg, &cat, dims, |_, _| Ok(foreign), 1.0);
        assert!(
            matches!(r, Err(Error::Select(SelectError::ForeignHandle { .. }))),
            "{dims:?}: {r:?}"
        );
        assert!(c.iter().all(|&x| x == 1.0), "nothing is written on error");

        // A catalog whose only kernel the CPU mask excludes: no candidate.
        let (r, _) = tiny(&cfg, &masked, dims, |_, _| unreachable!(), 1.0);
        assert!(
            matches!(
                r,
                Err(Error::Select(SelectError::NoCandidates { dtype: "f64" }))
            ),
            "{dims:?}: {r:?}"
        );

        // A member that was never offered, because the CPU mask excluded it.
        let both = cat.union(&masked).unwrap();
        let impossible = both.get("custom.f64.impossible").unwrap();
        let (r, _) = tiny(&cfg, &both, dims, |_, _| Ok(impossible), 1.0);
        assert!(
            matches!(r, Err(Error::Select(SelectError::CpuUnsupported { .. }))),
            "{dims:?}: {r:?}"
        );

        // The selector's own failure surfaces unchanged, with no fallback.
        let (r, _) = tiny(
            &cfg,
            &cat,
            dims,
            |_, _| {
                Err(SelectError::SelectorFailed {
                    reason: "nope".into(),
                })
            },
            1.0,
        );
        assert_eq!(
            r.unwrap_err(),
            Error::Select(SelectError::SelectorFailed {
                reason: "nope".into()
            })
        );
    }
    // An alpha of zero still selects and validates, then only scales C.
    let (r, c) = tiny(&cfg, &cat, (6, 6, 6), |_, cands| Ok(cands[0].handle), 0.0);
    assert!(r.is_ok());
    assert!(c.iter().all(|&x| x == 2.0));
    let foreign = other.get("custom.f64.2x2").unwrap();
    let (r, _) = tiny(&cfg, &cat, (6, 6, 6), |_, _| Ok(foreign), 0.0);
    assert!(matches!(
        r,
        Err(Error::Select(SelectError::ForeignHandle { .. }))
    ));
}

#[test]
fn an_incompatible_engine_or_a_forced_id_is_refused_not_ignored() {
    let cat = catalog::<f64>(own::f64_families());
    for engine in [EngineChoice::Faer, EngineChoice::PrivateGemmX86] {
        let cfg = GemmConfig {
            engine,
            ..Default::default()
        };
        let (r, _) = tiny(&cfg, &cat, (6, 6, 6), |_, _| unreachable!(), 1.0);
        assert!(
            matches!(r, Err(Error::Select(SelectError::EngineUnsupported { .. }))),
            "{r:?}"
        );
    }
    let cfg = GemmConfig {
        kernel: KernelChoice::Id("portable.f64.4x4".into()),
        ..Default::default()
    };
    let (r, _) = tiny(&cfg, &cat, (6, 6, 6), |_, _| unreachable!(), 1.0);
    assert!(matches!(
        r,
        Err(Error::Select(SelectError::Incompatible { .. }))
    ));
}

#[test]
fn a_builtin_fallback_is_explicit() {
    let own_cat = catalog::<f64>(own::f64_families());
    let both = own_cat
        .union(&tprims_blas::builtin_catalog::<f64>())
        .unwrap();
    let (r, _) = tiny(
        &GemmConfig::default(),
        &both,
        (6, 6, 6),
        |_, cands| {
            // Prefer a built-in kernel when asked to; the report says so.
            Ok(cands
                .iter()
                .find(|c| c.handle.id() == "portable.f64.4x4")
                .unwrap()
                .handle)
        },
        1.0,
    );
    let report = r.unwrap();
    assert_eq!(report.family_id, Some("portable.f64.4x4"));
    assert_eq!(report.origin, Some(tprims_gemm_kernel::Origin::Portable));
}

#[test]
fn batched_and_grouped_select_once_per_plan_not_per_item() {
    let cat = catalog::<f64>(own::f64_families());
    let (m, n, k, count) = (9usize, 7usize, 5usize, 6usize);
    let a = ints(m * k * count, 1);
    let b = ints(k * n * count, 2);
    let mut c = vec![0.0; m * n * count];
    let mut want = c.clone();
    for i in 0..count {
        naive(
            (m, n, k),
            1.0,
            &a[i * m * k..],
            [1, m as isize],
            &b[i * k * n..],
            [1, k as isize],
            0.0,
            &mut want[i * m * n..],
            [1, m as isize],
        );
    }
    let calls = Cell::new(0);
    let av = StridedView::new(&a, &[m, k, count], &[1, m as isize, (m * k) as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[k, n, count], &[1, k as isize, (k * n) as isize], 0).unwrap();
    let mut cv = StridedViewMut::new(
        &mut c,
        &[m, n, count],
        &[1, m as isize, (m * n) as isize],
        0,
    )
    .unwrap();
    let report = gemm_batched_with_selector(
        &Exec::serial(),
        &GemmConfig::default(),
        &cat,
        |ctx, cands| {
            calls.set(calls.get() + 1);
            assert_eq!(
                ctx.stats.batch, 1,
                "the batch axis is the call's, not the plan's"
            );
            Ok(cands[0].handle)
        },
        1.0,
        BatchIn::new(&av),
        BatchIn::new(&bv),
        0.0,
        &mut cv,
    )
    .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(c, want);
    assert_eq!(report.family_id, Some("custom.f64.2x2"));
    assert_eq!(report.origin, Some(own::ORIGIN));

    // Grouped: one selection per non-empty group plan, none for an empty job,
    // all before any group computes.
    let jobs = [
        GroupedJob {
            a_offset: 0,
            b_offset: 0,
            c_offset: 0,
            rows: 4,
            inner: 3,
            cols: 5,
        },
        GroupedJob {
            a_offset: 12,
            b_offset: 15,
            c_offset: 20,
            rows: 0,
            inner: 3,
            cols: 2,
        },
        GroupedJob {
            a_offset: 12,
            b_offset: 15,
            c_offset: 20,
            rows: 6,
            inner: 2,
            cols: 3,
        },
    ];
    let ga = ints(40, 1);
    let gb = ints(40, 2);
    let mut gc = vec![0.0; 40];
    let mut gwant = gc.clone();
    for j in jobs.iter().filter(|j| j.rows > 0) {
        naive(
            (j.rows, j.cols, j.inner),
            1.0,
            &ga[j.a_offset..],
            [1, j.rows as isize],
            &gb[j.b_offset..],
            [1, j.inner as isize],
            0.0,
            &mut gwant[j.c_offset..],
            [1, j.rows as isize],
        );
    }
    let mut seen = Vec::new();
    gemm_grouped_with_selector(
        &Exec::serial(),
        &GemmConfig::default(),
        &cat,
        |ctx, cands| {
            seen.push((ctx.stats.m, ctx.stats.n, ctx.stats.k));
            Ok(cands[seen.len() % 2].handle)
        },
        1.0,
        &ga,
        Conj::No,
        &gb,
        Conj::No,
        0.0,
        &mut gc,
        &jobs,
    )
    .unwrap();
    assert_eq!(seen, vec![(4, 5, 3), (6, 3, 2)]);
    assert_eq!(gc, gwant);
}
