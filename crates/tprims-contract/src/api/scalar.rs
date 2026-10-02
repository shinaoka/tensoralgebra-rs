//! The sealed storage-scalar vocabulary.

use tprims_kernel::{Element, Families};

use crate::api::DType;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// The storage and accumulation type of a contraction: `f32`, `f64`,
/// `Complex<f32>` or `Complex<f64>` (identical to `faer::c32` / `faer::c64`).
///
/// Sealed. The supertraits (the kernel registry's [`Families`] and faer's
/// element trait) are implementation vocabulary of this crate; a caller or a
/// second backend only names `T: Scalar`, and the neutral backend trait does
/// not require a kernel family or faer bound of its own.
///
/// Both element traits share method names, so call them as
/// `<T as tprims_kernel::Element>::zero()`.
///
/// # Examples
///
/// ```
/// use tprims_contract::api::{DType, Scalar};
/// fn dtype_of<T: Scalar>() -> DType { T::STORAGE }
/// assert_eq!(dtype_of::<f64>(), DType::F64);
/// assert_eq!(dtype_of::<num_complex::Complex<f32>>(), DType::C32);
/// ```
pub trait Scalar:
    sealed::Sealed
    + faer::traits::ComplexField<Real = <Self as Scalar>::Re>
    + Element<Real = <Self as Scalar>::Re>
    + Families
    + Copy
    + PartialEq
    + Send
    + Sync
    + 'static
{
    /// The real type: faer's field and the kernels' arithmetic type.
    type Re: faer::traits::RealField + tprims_kernel::Real;
    /// The problem-level name of the type.
    const STORAGE: DType;
}

impl Scalar for f32 {
    type Re = f32;
    const STORAGE: DType = DType::F32;
}
impl Scalar for f64 {
    type Re = f64;
    const STORAGE: DType = DType::F64;
}
impl Scalar for num_complex::Complex<f32> {
    type Re = f32;
    const STORAGE: DType = DType::C32;
}
impl Scalar for num_complex::Complex<f64> {
    type Re = f64;
    const STORAGE: DType = DType::C64;
}
