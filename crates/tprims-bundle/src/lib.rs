//! `libtprims`: one shared (`cdylib`) and static library from the selected
//! C ABI parts (Cargo features `blas`, `tapp`). Every part links into this
//! one library, so the executors of `tprims-core` (`TAPP_executor`) are valid
//! in all others; handles of another TAPP provider cannot be mixed in. The
//! `tapp` part is `tensorprimitives-tapp` as an `rlib`, exporting the TAPP
//! contraction API. Headers: the pinned upstream TAPP headers
//! `crates/tprims-core/include/tapp.h` and `tapp/*.h`, and
//! `crates/tprims-core/include/tprims/*.h` for the extensions.
use std::ffi::{c_char, CStr};

#[cfg(feature = "tapp")]
pub use tensorprimitives_tapp;
#[cfg(feature = "blas")]
pub use tprims_blas_capi;
pub use tprims_core;

/// Whether the loaded library contains `part` (`"core"`, `"blas"`, `"tapp"`):
/// 1 or 0.
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
        #[cfg(feature = "tapp")]
        b"tapp",
    ];
    i32::from(parts.contains(&name))
}
