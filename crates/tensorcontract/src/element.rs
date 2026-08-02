//! Element types.
//!
//! The engine is generic over a *real* scalar type ([`Real`]) and over a
//! *storage* element type ([`Element`]) which is either that real type or an
//! interleaved complex pair built from it.
//!
//! Complex support is deliberately expressed as "two planes of `Real`". All
//! packing, micro-kernel and write-back code below the [`Element`] boundary
//! works exclusively in terms of `Element::Real`, which is what makes the
//! planar (split-complex) strategy fall out of the design rather than being
//! bolted on.

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use num_complex::Complex;

/// A real scalar usable as the arithmetic base of a contraction.
///
/// Implemented here for `f32`/`f64`. Downstream types (extended precision,
/// dual numbers for forward-mode AD, `bf16`, ...) can implement this and get a
/// correct — if unvectorised — contraction engine for free by also
/// implementing [`crate::kernel::KernelSet`].
pub trait Real:
    Copy
    + Send
    + Sync
    + fmt::Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + 'static
{
    const ZERO: Self;
    const ONE: Self;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn abs(self) -> Self;

    #[inline(always)]
    fn is_zero(self) -> bool {
        self == Self::ZERO
    }
}

macro_rules! impl_real {
    ($t:ty) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;

            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
        }
    };
}

impl_real!(f32);
impl_real!(f64);

/// A tensor storage element: a real scalar, or an interleaved complex pair.
///
/// The memory layout of complex elements is *interleaved* (`re`, `im`
/// adjacent), matching `TAPP_C32`/`TAPP_C64`, C99 `_Complex`, `std::complex`
/// and `num_complex::Complex`. The engine converts to planar form during
/// packing and back to interleaved at write-back.
pub trait Element: Copy + Send + Sync + fmt::Debug + PartialEq + 'static {
    /// The underlying real type.
    type Real: Real;

    const IS_COMPLEX: bool;
    /// Number of real planes per element: 1 for real, 2 for complex.
    const PLANES: usize;
    /// Real flops per multiply-accumulate: 2 for real, 8 for complex.
    const FLOPS_PER_MAC: u64;

    fn zero() -> Self;
    fn one() -> Self;
    fn re(self) -> Self::Real;
    fn im(self) -> Self::Real;
    fn from_parts(re: Self::Real, im: Self::Real) -> Self;

    fn conj(self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;

    /// Magnitude, used by the test oracle for error metrics.
    fn norm(self) -> f64 {
        let (r, i) = (self.re().to_f64(), self.im().to_f64());
        (r * r + i * i).sqrt()
    }
}

macro_rules! impl_element_real {
    ($t:ty) => {
        impl Element for $t {
            type Real = $t;
            const IS_COMPLEX: bool = false;
            const PLANES: usize = 1;
            const FLOPS_PER_MAC: u64 = 2;

            #[inline(always)]
            fn zero() -> Self {
                0.0
            }
            #[inline(always)]
            fn one() -> Self {
                1.0
            }
            #[inline(always)]
            fn re(self) -> Self::Real {
                self
            }
            #[inline(always)]
            fn im(self) -> Self::Real {
                0.0
            }
            #[inline(always)]
            fn from_parts(re: Self::Real, _im: Self::Real) -> Self {
                re
            }
            #[inline(always)]
            fn conj(self) -> Self {
                self
            }
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                self + rhs
            }
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                self - rhs
            }
        }
    };
}

impl_element_real!(f32);
impl_element_real!(f64);

macro_rules! impl_element_complex {
    ($t:ty) => {
        impl Element for Complex<$t> {
            type Real = $t;
            const IS_COMPLEX: bool = true;
            const PLANES: usize = 2;
            const FLOPS_PER_MAC: u64 = 8;

            #[inline(always)]
            fn zero() -> Self {
                Complex::new(0.0, 0.0)
            }
            #[inline(always)]
            fn one() -> Self {
                Complex::new(1.0, 0.0)
            }
            #[inline(always)]
            fn re(self) -> Self::Real {
                self.re
            }
            #[inline(always)]
            fn im(self) -> Self::Real {
                self.im
            }
            #[inline(always)]
            fn from_parts(re: Self::Real, im: Self::Real) -> Self {
                Complex::new(re, im)
            }
            #[inline(always)]
            fn conj(self) -> Self {
                Complex::new(self.re, -self.im)
            }
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                self + rhs
            }
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                self - rhs
            }
        }
    };
}

impl_element_complex!(f32);
impl_element_complex!(f64);

/// Convenient aliases matching the TAPP datatype names.
pub type C32 = Complex<f32>;
/// Convenient aliases matching the TAPP datatype names.
pub type C64 = Complex<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_layout_is_interleaved() {
        // The FFI/TAPP boundary and the packing code both assume this.
        assert_eq!(core::mem::size_of::<C64>(), 16);
        assert_eq!(core::mem::align_of::<C64>(), core::mem::align_of::<f64>());
        let z = C64::new(1.0, 2.0);
        let parts: [f64; 2] = unsafe { core::mem::transmute(z) };
        assert_eq!(parts, [1.0, 2.0]);
    }

    #[test]
    fn planes_and_flops() {
        assert_eq!(<f64 as Element>::PLANES, 1);
        assert_eq!(<C64 as Element>::PLANES, 2);
        assert_eq!(<C64 as Element>::FLOPS_PER_MAC, 8);
    }
}
