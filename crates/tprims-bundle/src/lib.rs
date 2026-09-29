//! `libtprims`: one shared (`cdylib`) and static library from the selected
//! C ABI parts (Cargo features `blas`, `contract`). Every part links into this
//! one library, so handles created by one part (executors) are valid in all
//! others. Headers: `crates/tprims-core/include/tprims/*.h`.
use std::ffi::{c_char, CStr};

#[cfg(feature = "blas")]
pub use tprims_blas_capi;
#[cfg(feature = "contract")]
pub use tprims_contract_capi;
pub use tprims_core;

/// Whether the loaded library contains `part` (`"core"`, `"blas"`,
/// `"contract"`): 1 or 0.
///
/// # Safety
///
/// `part` is null or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn tprims_has_part(part: *const c_char) -> i32 {
    if part.is_null() {
        return 0;
    }
    // SAFETY: NUL-terminated per the contract.
    let name = unsafe { CStr::from_ptr(part) }.to_bytes();
    let parts: &[&[u8]] = &[
        b"core",
        #[cfg(feature = "blas")]
        tprims_blas_capi::PART.as_bytes(),
        #[cfg(feature = "contract")]
        tprims_contract_capi::PART.as_bytes(),
    ];
    i32::from(parts.contains(&name))
}
