//! C ABI core of the tprims stack.
//!
//! DLPack 1.x types ([`dlpack`]), borrowed operands ([`tensor::tprims_tensor`])
//! converted to strided views without copying, status codes and the
//! thread-local last-error message ([`status`]), and executor handles
//! ([`exec`]). Every `extern "C"` entry point catches panics. Other parts'
//! C ABI crates build on these; only `tprims-bundle` produces a library.
pub mod dlpack;
pub mod exec;
pub mod status;
pub mod tensor;

/// ABI version: major * 10000 + minor * 100 + patch.
pub const ABI_VERSION: u32 = 100; // 0.1.0

/// The ABI version of the loaded library (`TPRIMS_ABI_VERSION` in the header).
#[no_mangle]
pub extern "C" fn tprims_abi_version() -> u32 {
    ABI_VERSION
}
