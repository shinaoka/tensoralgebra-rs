//! Benchmark for 2D transpose-scale paths.
//!
//! This tracks cases where a raw pointer naive loop previously matched or beat
//! the strided path. Run benchmarks sequentially; do not run 1T and 4T jobs at
//! the same time.

use num_complex::{Complex32, Complex64};
use std::fmt::Debug;
use std::hint::black_box;
use std::ops::Mul;
use std::time::Instant;
use strided_kernel::{copy_transpose_scale_into, map_into};
use strided_view::{ElementOpApply, StridedArray};

trait BenchScalar:
    Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + ElementOpApply
    + Mul<Output = Self>
    + num_traits::Zero
    + num_traits::One
    + Send
    + Sync
    + 'static
{
    const NAME: &'static str;

    fn from_index(i: usize) -> Self;
    fn scale_three() -> Self;
}

impl BenchScalar for f32 {
    const NAME: &'static str = "f32";

    #[inline]
    fn from_index(i: usize) -> Self {
        i as f32 * 0.25 + 1.0
    }

    #[inline]
    fn scale_three() -> Self {
        3.0
    }
}

impl BenchScalar for f64 {
    const NAME: &'static str = "f64";

    #[inline]
    fn from_index(i: usize) -> Self {
        i as f64 * 0.25 + 1.0
    }

    #[inline]
    fn scale_three() -> Self {
        3.0
    }
}

impl BenchScalar for Complex32 {
    const NAME: &'static str = "c32";

    #[inline]
    fn from_index(i: usize) -> Self {
        Complex32::new(i as f32 * 0.25 + 1.0, i as f32 * -0.125)
    }

    #[inline]
    fn scale_three() -> Self {
        Complex32::new(3.0, -0.5)
    }
}

impl BenchScalar for Complex64 {
    const NAME: &'static str = "c64";

    #[inline]
    fn from_index(i: usize) -> Self {
        Complex64::new(i as f64 * 0.25 + 1.0, i as f64 * -0.125)
    }

    #[inline]
    fn scale_three() -> Self {
        Complex64::new(3.0, -0.5)
    }
}

impl BenchScalar for u64 {
    const NAME: &'static str = "u64";

    #[inline]
    fn from_index(i: usize) -> Self {
        i as u64 + 1
    }

    #[inline]
    fn scale_three() -> Self {
        3
    }
}

fn bench<F: FnMut()>(mut f: F, warmup: usize, nruns: usize) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(nruns);
    for _ in 0..nruns {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let iqr = times[3 * times.len() / 4] - times[times.len() / 4];
    (med, iqr)
}

#[inline(never)]
fn naive_transpose_scale<T: BenchScalar>(dst: *mut T, src: *const T, n: usize, scale: T) {
    unsafe {
        if scale == T::zero() {
            for j in 0..n {
                let col = j * n;
                for i in 0..n {
                    *dst.add(col + i) = T::zero();
                }
            }
            return;
        }

        for j in 0..n {
            let col = j * n;
            for i in 0..n {
                *dst.add(col + i) = scale * *src.add(i * n + j);
            }
        }
    }
}

fn make_col_major<T: BenchScalar>(n: usize) -> StridedArray<T> {
    StridedArray::<T>::from_fn_col_major(&[n, n], |idx| T::from_index(idx[0] + idx[1] * n))
}

fn run_case<T: BenchScalar>(n: usize, scale: T, scale_name: &str, warmup: usize, nruns: usize) {
    let a = make_col_major::<T>(n);
    let a_view = a.view();
    let a_t = a_view.transpose_2d().unwrap();
    let a_perm = a_view.permute(&[1, 0]).unwrap();
    let mut b = StridedArray::<T>::col_major(&[n, n]);
    let src_ptr = a.data().as_ptr();
    let dst_ptr = b.data_mut().as_mut_ptr();

    let (naive_med, naive_iqr) = bench(
        || {
            naive_transpose_scale(dst_ptr, src_ptr, n, scale);
            black_box(dst_ptr);
        },
        warmup,
        nruns,
    );

    let (strided_med, strided_iqr) = bench(
        || {
            copy_transpose_scale_into(&mut b.view_mut(), &a_view, scale).unwrap();
            black_box(dst_ptr);
        },
        warmup,
        nruns,
    );

    let (map_med, map_iqr) = bench(
        || {
            map_into(&mut b.view_mut(), &a_t, |x| scale * x).unwrap();
            black_box(dst_ptr);
        },
        warmup,
        nruns,
    );

    println!(
        "{:<4} n={:<4} scale={:<5} {:>11.3} ({:>6.3}) {:>11.3} ({:>6.3}) {:>11.3} ({:>6.3})",
        T::NAME,
        n,
        scale_name,
        naive_med,
        naive_iqr,
        strided_med,
        strided_iqr,
        map_med,
        map_iqr
    );

    if scale == T::one() {
        let (copy_med, copy_iqr) = bench(
            || {
                strided_perm::copy_into(&mut b.view_mut(), &a_perm).unwrap();
                black_box(dst_ptr);
            },
            warmup,
            nruns,
        );
        println!(
            "{:<4} n={:<4} scale={:<5} copy_into  {:>11.3} ({:>6.3})",
            T::NAME,
            n,
            scale_name,
            copy_med,
            copy_iqr
        );
    }
}

fn run_dtype<T: BenchScalar>(sizes: &[usize], warmup: usize, nruns: usize) {
    for &n in sizes {
        run_case::<T>(n, T::zero(), "0", warmup, nruns);
        run_case::<T>(n, T::one(), "1", warmup, nruns);
        run_case::<T>(n, T::scale_three(), "3", warmup, nruns);
    }
}

fn main() {
    let warmup = std::env::var("WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let nruns = std::env::var("NRUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11);
    let sizes: Vec<usize> = std::env::var("SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|v| v.trim().parse().expect("invalid SIZES entry"))
                .collect()
        })
        .unwrap_or_else(|| vec![1000, 1024, 2048]);

    println!("2D transpose-scale benchmark");
    println!("warmup={warmup} nruns={nruns} sizes={sizes:?}");
    println!("columns: dtype n scale naive_ms(iqr) strided_ms(iqr) map_ms(iqr)");
    println!("{}", "-".repeat(92));

    run_dtype::<f32>(&sizes, warmup, nruns);
    run_dtype::<f64>(&sizes, warmup, nruns);
    run_dtype::<Complex32>(&sizes, warmup, nruns);
    run_dtype::<Complex64>(&sizes, warmup, nruns);
    run_dtype::<u64>(&sizes, warmup, nruns);
}
