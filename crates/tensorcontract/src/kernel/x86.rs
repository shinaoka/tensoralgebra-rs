//! x86-64 micro-kernels.
//!
//! Populated in Phase 3; until then the dispatcher falls back to the portable
//! scalar kernels.

use super::KernelConfig;

pub fn config_real_f64() -> Option<KernelConfig<f64>> {
    None
}
pub fn config_cplx_f64() -> Option<KernelConfig<f64>> {
    None
}
pub fn config_real_f32() -> Option<KernelConfig<f32>> {
    None
}
pub fn config_cplx_f32() -> Option<KernelConfig<f32>> {
    None
}
