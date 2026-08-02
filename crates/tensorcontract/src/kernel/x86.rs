//! x86-64 micro-kernels.
//!
//! Populated in Phase 3; until then the dispatcher falls back to the portable
//! scalar kernels. When these land they must honour the same
//! [`PackFormat`](super::PackFormat) / [`TileFormat`](super::TileFormat)
//! contract, so that the three complex methods stay interchangeable and the
//! kernel-vs-reference tests in `kernel::tests` keep working unchanged.

use super::{ComplexMethod, KernelConfig};

pub fn config_real_f64() -> Option<KernelConfig<f64>> {
    None
}
pub fn config_cplx_f64(_method: ComplexMethod) -> Option<KernelConfig<f64>> {
    None
}
pub fn config_real_f32() -> Option<KernelConfig<f32>> {
    None
}
pub fn config_cplx_f32(_method: ComplexMethod) -> Option<KernelConfig<f32>> {
    None
}
