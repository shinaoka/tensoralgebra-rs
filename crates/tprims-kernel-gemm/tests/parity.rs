//! Every gemm family must compute what its own microkernel table claims: the
//! shim's argument mapping and the lane count are the adapter's only claims,
//! so a naive product is the whole oracle.
use tprims_gemm_kernel::*;

/// A packed A panel in the layout the driver produces for `PackFormat::Real`:
/// column-major `MR`, so the k stride is `MR`.
fn packed_a(mr: usize, k: usize) -> Vec<f64> {
    (0..mr * k)
        .map(|x| ((x * 31 % 17) as f64 - 8.0) / 6.0)
        .collect()
}

fn naive(
    mr: usize,
    nr: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    b: &[f64],
    b_rs: isize,
    b_cs: isize,
) -> Vec<f64> {
    let mut want = vec![1.0f64; mr * nr];
    for j in 0..n {
        for i in 0..m {
            let mut s = 0.0;
            for p in 0..k {
                let bv = b[(p as isize * b_rs + j as isize * b_cs) as usize];
                s += a[p * mr + i] * bv;
            }
            want[j * mr + i] = 0.5 * want[j * mr + i] + 2.0 * s;
        }
    }
    want
}

#[test]
fn gemm_families_validate_and_match_a_naive_product() {
    tprims_kernel_gemm::register();
    let families: Vec<_> = Registry::families::<f64>(CpuFeatures::detect(), false)
        .into_iter()
        .filter(|f| f.id.starts_with("gemm."))
        .collect();
    assert!(!families.is_empty(), "no gemm family is available");
    assert!(
        families.iter().any(|f| f.allow_auto),
        "exactly one family is the Auto head"
    );
    assert_eq!(
        families.iter().filter(|f| f.allow_auto).count(),
        1,
        "only the head may be Auto-eligible"
    );

    for f in families {
        f.validate().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(f.c_update, CUpdate::Direct);
        assert_eq!(f.origin, Origin::Gemm);
        // A full tile and an edge tile: the kernel is told the live extent.
        for &(m, n) in &[(f.mr, f.nr), (f.mr - 1, 1)] {
            let k = 17;
            let a = packed_a(f.mr, k);
            // B stored column-major for the packed case (k stride NR, columns
            // adjacent) and then row-major to check the in-place strides.
            for (b_rs, b_cs) in [(f.nr as isize, 1isize), (1, f.nr as isize)] {
                let b: Vec<f64> = (0..k * f.nr)
                    .map(|x| ((x * 41 % 13) as f64 - 6.0) / 4.0)
                    .collect();
                let mut got = vec![1.0f64; f.mr * f.nr];
                let aux = UkrAux {
                    a_next: a.as_ptr(),
                    b_next: b.as_ptr(),
                    inner: None,
                    opaque: f.opaque,
                };
                let UkrFn::Direct(u) = f.ukr else {
                    panic!("a gemm family is a Direct family");
                };
                // SAFETY: the panels cover the family's declared extents, the
                // tile is `mr * nr` writable reals, and `b` covers every
                // (p, j) the strides address.
                unsafe {
                    u(
                        m,
                        n,
                        k,
                        got.as_mut_ptr(),
                        1,
                        f.mr as isize,
                        a.as_ptr(),
                        f.mr as isize,
                        b.as_ptr(),
                        b_rs,
                        b_cs,
                        0.5,
                        2.0,
                        &aux,
                    )
                };
                let want = naive(f.mr, f.nr, m, n, k, &a, &b, b_rs, b_cs);
                for j in 0..n {
                    for i in 0..m {
                        let (x, y) = (got[j * f.mr + i], want[j * f.mr + i]);
                        assert!(
                            (x - y).abs() <= 1e-12 * y.abs().max(1.0),
                            "{} {m}x{n} strides {b_rs}/{b_cs} at ({i},{j}): {x} vs {y}",
                            f.id
                        );
                    }
                }
            }
        }
    }
}

/// The f32 table is a separate one per dtype, and it must be right for the
/// same reason: the shim and the lane count are per-dtype claims.
#[test]
fn f32_families_match_a_naive_product() {
    tprims_kernel_gemm::register();
    let families: Vec<_> = Registry::families::<f32>(CpuFeatures::detect(), false)
        .into_iter()
        .filter(|f| f.id.starts_with("gemm."))
        .collect();
    assert!(!families.is_empty());
    for f in families {
        f.validate().unwrap_or_else(|e| panic!("{e}"));
        let k = 13;
        let a: Vec<f32> = (0..f.mr * k)
            .map(|x| ((x * 31 % 17) as f32 - 8.0) / 6.0)
            .collect();
        let b: Vec<f32> = (0..k * f.nr)
            .map(|x| ((x * 41 % 13) as f32 - 6.0) / 4.0)
            .collect();
        let mut got = vec![1.0f32; f.mr * f.nr];
        let aux = UkrAux {
            a_next: a.as_ptr(),
            b_next: b.as_ptr(),
            inner: None,
            opaque: f.opaque,
        };
        let UkrFn::Direct(u) = f.ukr else { panic!() };
        // SAFETY: panels and tile cover the family's declared extents.
        unsafe {
            u(
                f.mr,
                f.nr,
                k,
                got.as_mut_ptr(),
                1,
                f.mr as isize,
                a.as_ptr(),
                f.mr as isize,
                b.as_ptr(),
                f.nr as isize,
                1,
                0.5,
                2.0,
                &aux,
            )
        };
        for j in 0..f.nr {
            for i in 0..f.mr {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[p * f.mr + i] * b[p * f.nr + j];
                }
                let want = 0.5 * 1.0 + 2.0 * s;
                let x = got[j * f.mr + i];
                assert!(
                    (x - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "{} at ({i},{j}): {x} vs {want}",
                    f.id
                );
            }
        }
    }
}

/// The head is the largest tile of the widest table, and no other entry may
/// take Auto from it.
#[test]
fn only_the_widest_tables_largest_tile_is_auto_eligible() {
    tprims_kernel_gemm::register();
    let all: Vec<_> = Registry::families::<f64>(CpuFeatures::NONE, true)
        .into_iter()
        .filter(|f| f.origin == Origin::Gemm)
        .collect();
    let head = all.iter().find(|f| f.allow_auto).expect("a head");
    let widest = all
        .iter()
        .map(|f| f.isa)
        .max_by_key(|isa| match isa {
            Isa::Avx512 => 3,
            Isa::Avx2 | Isa::Neon => 2,
            Isa::Portable => 1,
            _ => 0,
        })
        .unwrap();
    assert_eq!(head.isa, widest);
    assert!(all
        .iter()
        .filter(|f| f.isa == widest)
        .all(|f| f.mr * f.nr <= head.mr * head.nr));
}
