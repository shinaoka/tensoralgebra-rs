use num_complex::Complex;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// Element types supported by tprims-blas: `f32`, `f64`, `Complex<f32>`,
/// `Complex<f64>` (identical to `faer::c32` / `faer::c64`).
///
/// Both faer's and tensorcontract's element traits are supertraits; their
/// methods share names, so call them as `<T as tensorcontract::Element>::zero()`.
///
/// # Examples
///
/// ```
/// fn is_complex<T: tprims_blas::Scalar>() -> bool { T::IS_COMPLEX_SCALAR }
/// assert!(!is_complex::<f64>());
/// assert!(is_complex::<num_complex::Complex<f32>>());
/// ```
pub trait Scalar:
    sealed::Sealed
    + faer::traits::ComplexField<Real = <Self as Scalar>::Re>
    + tensorcontract::Element<Real = <Self as Scalar>::Re>
    + Copy
    + Send
    + Sync
    + 'static
{
    /// The real type, with tensorcontract's kernels.
    type Re: tensorcontract::KernelSet + faer::traits::RealField + Copy;
    /// Whether the type is complex.
    const IS_COMPLEX_SCALAR: bool;
}

impl Scalar for f32 {
    type Re = f32;
    const IS_COMPLEX_SCALAR: bool = false;
}
impl Scalar for f64 {
    type Re = f64;
    const IS_COMPLEX_SCALAR: bool = false;
}
impl Scalar for Complex<f32> {
    type Re = f32;
    const IS_COMPLEX_SCALAR: bool = true;
}
impl Scalar for Complex<f64> {
    type Re = f64;
    const IS_COMPLEX_SCALAR: bool = true;
}

pub(crate) fn zero<T: Scalar>() -> T {
    <T as tensorcontract::Element>::zero()
}

pub(crate) fn one<T: Scalar>() -> T {
    <T as tensorcontract::Element>::one()
}

pub(crate) fn mul<T: Scalar>(a: T, b: T) -> T {
    <T as tensorcontract::Element>::mul(a, b)
}
