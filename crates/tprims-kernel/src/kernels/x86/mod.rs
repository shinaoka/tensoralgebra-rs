//! The x86 and x86-64 kernels.

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2_complex;

mod avx2;
pub use avx2::*;
