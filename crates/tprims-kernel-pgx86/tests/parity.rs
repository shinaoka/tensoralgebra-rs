//! The engine's arithmetic, against a naive product: it is called, not
//! reimplemented, so the only claims here are the strides, the conjugate flags
//! and the alpha/beta convention.
use num_complex::Complex64 as C;
use tprims_kernel_pgx86::{available, gemm};

#[test]
fn pgx86_matches_naive_for_real_dtypes() {
    if !available() {
        return;
    }
    // Column-major A (m x k), row-major B (k x n), column-major D (m x n).
    for &(m, n, k) in &[(1usize, 1usize, 1usize), (7, 5, 3), (64, 33, 129)] {
        let a: Vec<f64> = (0..m * k).map(|x| (x % 7) as f64 - 3.0).collect();
        let b: Vec<f64> = (0..k * n).map(|x| (x % 5) as f64 - 2.0).collect();
        let mut d = vec![0.0f64; m * n];
        // SAFETY: the buffers cover the advertised shapes with these strides,
        // and `d` aliases neither operand.
        unsafe {
            gemm(
                m,
                n,
                k,
                d.as_mut_ptr(),
                1,
                m as isize,
                true,
                a.as_ptr(),
                1,
                m as isize,
                false,
                b.as_ptr(),
                1,
                k as isize,
                false,
                1.0,
                1,
            )
        };
        for j in 0..n {
            for i in 0..m {
                let mut s = 0.0;
                for p in 0..k {
                    s += a[p * m + i] * b[j * k + p];
                }
                assert_eq!(d[j * m + i], s, "{m}x{n}x{k} at ({i},{j})");
            }
        }
    }
}

#[test]
fn pgx86_scales_by_alpha_and_accumulates_over_existing_d() {
    if !available() {
        return;
    }
    let (m, n, k) = (9usize, 6usize, 11usize);
    let a: Vec<f32> = (0..m * k).map(|x| (x % 3) as f32 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|x| (x % 4) as f32 - 2.0).collect();
    for beta_zero in [true, false] {
        let mut d = vec![1.5f32; m * n];
        // SAFETY: as above; `Accum::Add` is the `beta_zero = false` case, which
        // is what makes the pre-scaled `D` an input.
        unsafe {
            gemm(
                m,
                n,
                k,
                d.as_mut_ptr(),
                1,
                m as isize,
                beta_zero,
                a.as_ptr(),
                1,
                m as isize,
                false,
                b.as_ptr(),
                1,
                k as isize,
                false,
                2.0,
                1,
            )
        };
        for j in 0..n {
            for i in 0..m {
                let mut s = 0.0;
                for p in 0..k {
                    s += a[p * m + i] * b[j * k + p];
                }
                let want = if beta_zero { 2.0 * s } else { 2.0 * s + 1.5 };
                assert!((d[j * m + i] - want).abs() < 1e-4, "beta_zero={beta_zero}");
            }
        }
    }
}

#[test]
fn pgx86_handles_conjugating_operands() {
    if !available() {
        return;
    }
    let (m, n, k) = (5usize, 4usize, 3usize);
    let a: Vec<C> = (0..m * k)
        .map(|x| C::new(x as f64, 1.0 - x as f64))
        .collect();
    let b: Vec<C> = (0..k * n)
        .map(|x| C::new(2.0 - x as f64, x as f64))
        .collect();
    // Conjugate both operands, and combine with conjugation of the output by
    // conjugating a conjugating call.
    for (cj_a, cj_b) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut d = vec![C::new(0.0, 0.0); m * n];
        // SAFETY: complex column-major A and D, row-major B, no aliasing.
        unsafe {
            gemm(
                m,
                n,
                k,
                d.as_mut_ptr(),
                1,
                m as isize,
                true,
                a.as_ptr(),
                1,
                m as isize,
                cj_a,
                b.as_ptr(),
                1,
                k as isize,
                cj_b,
                C::new(1.0, 0.0),
                1,
            )
        };
        for j in 0..n {
            for i in 0..m {
                let mut s = C::new(0.0, 0.0);
                for p in 0..k {
                    let (av, bv) = (a[p * m + i], b[j * k + p]);
                    s += (if cj_a { av.conj() } else { av }) * (if cj_b { bv.conj() } else { bv });
                }
                assert!(
                    (d[j * m + i] - s).norm() < 1e-12,
                    "conj=({cj_a},{cj_b}) at ({i},{j}): {} vs {}",
                    d[j * m + i],
                    s
                );
            }
        }
    }
}
