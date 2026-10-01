//! Induced 1m/4m variants: registered, validating, never auto-selected, and
//! numerically equal to the same complex product computed directly.
use tprims_kernel::*;

#[test]
fn induced_variants_exist_for_every_real_family() {
    let complex: Vec<_> = Registry::families::<C64>(CpuFeatures::NONE, true)
        .into_iter()
        .map(|f| f.id)
        .collect();
    for real in Registry::families::<f64>(CpuFeatures::NONE, true) {
        if real.complex.is_some() {
            continue;
        }
        let one_m_id = format!("{}.1m-induced", real.id);
        let four_m_id = format!("{}.4m-induced", real.id);
        // A Direct family has no packed arm, so neither method is derivable.
        let tile_arm = matches!(real.ukr, UkrFn::Tile(_));
        assert_eq!(
            complex.iter().any(|id| *id == one_m_id),
            induced::one_m(real).is_some(),
            "{one_m_id}"
        );
        assert_eq!(
            complex.iter().any(|id| *id == four_m_id),
            tile_arm,
            "{four_m_id}"
        );
    }
}

#[test]
fn induced_families_validate_and_are_never_auto() {
    let mut seen = 0;
    for f in Registry::families::<C64>(CpuFeatures::NONE, true) {
        if f.imp != KernelImpl::Induced {
            continue;
        }
        seen += 1;
        f.validate().unwrap_or_else(|e| panic!("{e}"));
        assert!(!f.allow_auto, "{}", f.id);
        let inner = f.inner.expect("an induced family carries its inner kernel");
        assert_eq!(inner.complex, None);
        assert!(f.id.starts_with(inner.id), "{}", f.id);
        let method = f.complex.unwrap().method;
        match method {
            // One complex k-step is two real ones, so the tile is half as tall.
            Method::OneM => assert_eq!(2 * f.mr, inner.mr, "{}", f.id),
            Method::FourM => assert_eq!(f.mr, inner.mr, "{}", f.id),
            other => panic!("unexpected induced method {other:?}"),
        }
        assert_eq!(f.nr, inner.nr, "{}", f.id);
    }
    assert!(seen > 0, "no induced families were registered");
}

/// 1m is refused where it cannot be derived: an odd tile or a Direct kernel.
#[test]
fn one_m_rejects_odd_tiles_and_direct() {
    let all = Registry::families::<f64>(CpuFeatures::NONE, true);
    let direct = all
        .iter()
        .find(|f| f.id.ends_with(".direct"))
        .expect("a direct family is registered");
    assert!(induced::one_m(direct).is_none());
    assert!(induced::four_m(direct).is_none());
    let odd = all
        .iter()
        .find(|f| f.mr % 2 != 0 || f.nr % 2 != 0)
        .copied()
        // No registered family is odd-tiled, so build one: an odd `nr` has no
        // half-tile to expand for 1m but is a perfectly good 4m source.
        .unwrap_or_else(|| {
            let mut f = *all
                .iter()
                .copied()
                .find(|f| f.complex.is_none() && matches!(f.ukr, UkrFn::Tile(_)))
                .expect("a real tile family is registered");
            f.nr = 3;
            f.b_per_k = 3;
            f.tile_bound = f.mr * 3;
            Box::leak(Box::new(f))
        });
    assert!(induced::one_m(odd).is_none(), "{}", odd.id);
    assert!(induced::four_m(odd).is_some(), "{}", odd.id);
}

/// Both induced methods reproduce a complex product through the same packers,
/// tile arm and write-back the driver binds, so a packing or plane-order
/// mistake cannot hide behind the driver's own arithmetic.
#[test]
fn induced_arithmetic_matches_the_complex_product() {
    for id in ["portable.f64.4x4.1m-induced", "portable.f64.4x4.4m-induced"] {
        let choice = KernelChoice::Id(id.into());
        let rg = ResolvedGemm::<f64>::resolve::<C64>(&choice, 1).unwrap();
        let f = rg.family();
        let (mr, nr, kc) = (f.mr, f.nr, 3);
        let a: Vec<C64> = (0..mr * kc)
            .map(|i| C64::new(0.5 + i as f64 * 0.125, -0.25 + i as f64 * 0.0625))
            .collect();
        let b: Vec<C64> = (0..nr * kc)
            .map(|i| C64::new(-0.75 + i as f64 * 0.25, 0.5 - i as f64 * 0.125))
            .collect();
        let ar: Vec<i64> = (0..mr).map(|i| i as i64).collect();
        let br: Vec<i64> = (0..nr).map(|i| i as i64).collect();
        let ak: Vec<i64> = (0..kc).map(|p| (p * mr) as i64).collect();
        let bk: Vec<i64> = (0..kc).map(|p| (p * nr) as i64).collect();
        let abs = scatter::build_block_scatter(&ar, mr);
        let bbs = scatter::build_block_scatter(&br, nr);
        let mut pa = vec![0.0_f64; f.a_per_k * kc];
        let mut pb = vec![0.0_f64; f.b_per_k * kc];
        let mut tile = vec![f64::NAN; f.tile_bound];
        let mut scratch = vec![0.0_f64; induced::four_m_scratch(mr, nr, kc)];
        let mut out = vec![C64::zero(); mr * nr];
        let cols: Vec<i64> = (0..nr).map(|j| (j * mr) as i64).collect();
        let (pack_a, pack_b) = rg.packers::<C64>();
        // SAFETY: every scat/offset pair stays inside its source, and the
        // panels, tile and scratch are fully sized for this family.
        unsafe {
            pack_a(a.as_ptr(), &ar, &abs, &ak, mr, false, pa.as_mut_ptr());
            pack_b(b.as_ptr(), &br, &bbs, &bk, nr, false, pb.as_mut_ptr());
            let ukr = f.driver_family().unwrap();
            induced::tile_call(
                &ukr,
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                tile.as_mut_ptr(),
                scratch.as_mut_ptr(),
            );
            // The family's own write-back, at the format it declares.
            let emit = rg.emitter::<C64>();
            emit(
                tile.as_ptr(),
                mr,
                nr,
                mr,
                nr,
                C64::one(),
                C64::zero(),
                core::ptr::null(),
                &ar,
                &cols,
                1,
                false,
                out.as_mut_ptr(),
                &ar,
                &cols,
                1,
                false,
            );
        }
        for j in 0..nr {
            for i in 0..mr {
                let mut want = C64::zero();
                for p in 0..kc {
                    want += a[p * mr + i] * b[p * nr + j];
                }
                let got = out[j * mr + i];
                assert!(
                    (got.re - want.re).abs() < 1e-12 && (got.im - want.im).abs() < 1e-12,
                    "{id} at ({i},{j}): {got:?} vs {want:?}"
                );
            }
        }
    }
}
