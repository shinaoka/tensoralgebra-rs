//! Calls `private-gemm-x86` (sarah-ek/gemm-x64-v2, MIT) directly as the tprims
//! `PrivateGemmX86` matrix engine.
//!
//! Calls only; no source is copied, and `docs/provenance.md` records the
//! upstream. Its single entry point takes element strides and a scalar `alpha`,
//! accumulates into `D`, and splits the work over the rayon pool it is called
//! inside, so this crate is deliberately a wrapper and nothing more: the shape,
//! stride and `beta` decisions belong to the caller.
//!
//! Off x86-64 the engine is not compiled at all: [`available`] answers false and
//! [`gemm`] must not be called.
use num_complex::Complex;

/// A scalar the engine accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PgDtype {
    /// `f32`.
    F32,
    /// `f64`.
    F64,
    /// `Complex<f32>`.
    C32,
    /// `Complex<f64>`.
    C64,
}

/// Element types the engine can compute with.
pub trait PgScalar: Copy {
    /// Which kind this scalar is.
    const DTYPE: PgDtype;
}

impl PgScalar for f32 {
    const DTYPE: PgDtype = PgDtype::F32;
}
impl PgScalar for f64 {
    const DTYPE: PgDtype = PgDtype::F64;
}
impl PgScalar for Complex<f32> {
    const DTYPE: PgDtype = PgDtype::C32;
}
impl PgScalar for Complex<f64> {
    const DTYPE: PgDtype = PgDtype::C64;
}

/// Whether this build and this CPU can run the engine: x86-64 with AVX2 and FMA.
///
/// # Examples
/// ```
/// // False on a machine without AVX2+FMA, and on every non-x86-64 target.
/// let _ = tprims_kernel_pgx86::available();
/// ```
pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `D += alpha * op(A) * op(B)` through the engine, with element strides.
///
/// The engine accumulates, so a caller that needs `beta * D + alpha * A * B`
/// scales `D` first (and passes `beta_zero` when it did not, so the engine knows
/// it may skip reading `D`). Threads come from the rayon pool this call runs
/// inside; `n_threads` is how many of them the engine may use.
///
/// # Safety
/// `A` must be readable over `m x k` with the given element strides, `B` over
/// `k x n`, and `D` writable over `m x n`; `D` must not alias `A` or `B`.
/// `available()` must be true.
///
/// # Examples
/// ```no_run
/// // Not run: it needs an AVX2+FMA x86-64 host.
/// # #[cfg(target_arch = "x86_64")] {
/// let a = [1.0_f64, 2.0];
/// let b = [3.0_f64, 4.0];
/// let mut d = [0.0_f64; 4];
/// // SAFETY: 2x1 and 1x2 column-major operands, a 2x2 output, no aliasing.
/// unsafe {
///     tprims_kernel_pgx86::gemm(
///         2, 2, 1, d.as_mut_ptr(), 1, 2, true, a.as_ptr(), 1, 2, false, b.as_ptr(), 1, 1,
///         false, 1.0, 1,
///     );
/// }
/// assert_eq!(d, [3.0, 6.0, 4.0, 8.0]);
/// # }
/// ```
#[allow(clippy::too_many_arguments)] // INVARIANT: the engine's own argument set.
pub unsafe fn gemm<T: PgScalar>(
    m: usize,
    n: usize,
    k: usize,
    d: *mut T,
    d_rs: isize,
    d_cs: isize,
    beta_zero: bool,
    a: *const T,
    a_rs: isize,
    a_cs: isize,
    conj_a: bool,
    b: *const T,
    b_rs: isize,
    b_cs: isize,
    conj_b: bool,
    alpha: T,
    n_threads: usize,
) {
    debug_assert!(available(), "the engine needs an AVX2+FMA x86-64 host");
    #[cfg(target_arch = "x86_64")]
    {
        use private_gemm_x86::{Accum, DType, DstKind, IType, InstrSet};
        let dtype = match T::DTYPE {
            PgDtype::F32 => DType::F32,
            PgDtype::F64 => DType::F64,
            PgDtype::C32 => DType::C32,
            PgDtype::C64 => DType::C64,
        };
        let instr = if std::is_x86_feature_detected!("avx512f") {
            InstrSet::Avx512
        } else {
            InstrSet::Avx256
        };
        // SAFETY: forwarded unchanged from this function's contract; the null
        // row/column index and diagonal pointers are the dense, non-diagonal
        // case the engine documents.
        unsafe {
            private_gemm_x86::gemm(
                dtype,
                IType::U64,
                instr,
                m,
                n,
                k,
                d as *mut (),
                d_rs,
                d_cs,
                core::ptr::null(),
                core::ptr::null(),
                DstKind::Full,
                if beta_zero {
                    Accum::Replace
                } else {
                    Accum::Add
                },
                a as *const (),
                a_rs,
                a_cs,
                conj_a,
                core::ptr::null(),
                0,
                b as *const (),
                b_rs,
                b_cs,
                conj_b,
                &alpha as *const T as *const (),
                n_threads,
            )
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (
            m, n, k, d, d_rs, d_cs, beta_zero, a, a_rs, a_cs, conj_a, b, b_rs, b_cs, conj_b, alpha,
            n_threads,
        );
        unreachable!("the engine is not compiled on this target")
    }
}
