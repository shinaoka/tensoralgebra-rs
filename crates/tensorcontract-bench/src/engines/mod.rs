//! Shared plumbing for the harness: element traits, timing, and the
//! per-engine runners.

pub mod premise;
pub mod sweep;
pub mod verify;

use std::os::raw::c_int;
use std::time::Instant;

use num_complex::Complex;
use rand::Rng;
use tensorcontract::Element;

use crate::blas::GemmScalar;
use crate::tblis;

/// An element type the harness can drive through every engine.
#[allow(dead_code)] // some members are only used under optional features
pub trait BenchElem: Element + GemmScalar {
    const NAME: &'static str;
    const TBLIS_TYPE: c_int;
    /// The same shape in the corresponding real type, for ratio reporting.
    const REAL_NAME: &'static str;

    fn tblis_scalar(v: f64) -> tblis::tblis_scalar;
    fn sample(rng: &mut impl Rng) -> Self;
    fn from_f64(v: f64) -> Self;
}

impl BenchElem for f32 {
    const NAME: &'static str = "f32";
    const TBLIS_TYPE: c_int = tblis::TYPE_SINGLE;
    const REAL_NAME: &'static str = "f32";
    fn tblis_scalar(v: f64) -> tblis::tblis_scalar {
        tblis::tblis_scalar::f32(v as f32)
    }
    fn sample(rng: &mut impl Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl BenchElem for f64 {
    const NAME: &'static str = "f64";
    const TBLIS_TYPE: c_int = tblis::TYPE_DOUBLE;
    const REAL_NAME: &'static str = "f64";
    fn tblis_scalar(v: f64) -> tblis::tblis_scalar {
        tblis::tblis_scalar::f64(v)
    }
    fn sample(rng: &mut impl Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

impl BenchElem for Complex<f32> {
    const NAME: &'static str = "c32";
    const TBLIS_TYPE: c_int = tblis::TYPE_SCOMPLEX;
    const REAL_NAME: &'static str = "f32";
    fn tblis_scalar(v: f64) -> tblis::tblis_scalar {
        tblis::tblis_scalar::c32(v as f32, 0.0)
    }
    fn sample(rng: &mut impl Rng) -> Self {
        Complex::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn from_f64(v: f64) -> Self {
        Complex::new(v as f32, 0.0)
    }
}

impl BenchElem for Complex<f64> {
    const NAME: &'static str = "c64";
    const TBLIS_TYPE: c_int = tblis::TYPE_DCOMPLEX;
    const REAL_NAME: &'static str = "f64";
    fn tblis_scalar(v: f64) -> tblis::tblis_scalar {
        tblis::tblis_scalar::c64(v, 0.0)
    }
    fn sample(rng: &mut impl Rng) -> Self {
        Complex::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn from_f64(v: f64) -> Self {
        Complex::new(v, 0.0)
    }
}

/// Best-of-`reps` wall time in seconds, after one warm-up call.
///
/// Minimum rather than mean: these are deterministic compute kernels, so the
/// spread is machine noise (frequency, interrupts, other tenants) and the
/// minimum is the least contaminated estimator.
pub fn timed(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// GFLOP/s given a multiply-accumulate count and a time.
pub fn gflops<T: Element>(macs: u64, secs: f64) -> f64 {
    (macs as f64) * (T::FLOPS_PER_MAC as f64) / secs / 1e9
}

/// Relative Frobenius error `||got - want|| / ||want||`.
pub fn rel_error<T: Element>(got: &[T], want: &[T]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (&g, &w) in got.iter().zip(want) {
        num += g.sub(w).norm().powi(2);
        den += w.norm().powi(2);
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

/// Force the process onto one core's worth of BLAS/TBLIS threads.
pub fn pin_single_threaded() {
    #[cfg(feature = "tblis")]
    unsafe {
        tblis::tblis_set_num_threads(1)
    };
    #[cfg(feature = "blas")]
    unsafe {
        crate::blas::openblas_set_num_threads(1)
    };
}
