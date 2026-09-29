//! Status codes, the thread-local last-error message, and the panic-catching
//! entry wrapper every `extern "C"` function uses.
use std::cell::RefCell;
use std::ffi::{c_char, CString};

/// `tprims_status`.
#[allow(non_camel_case_types)]
pub type tprims_status = i32;

/// Success.
pub const TPRIMS_OK: tprims_status = 0;
/// Null pointer or otherwise invalid argument.
pub const TPRIMS_ERR_INVALID_ARGUMENT: tprims_status = 1;
/// Inconsistent or overflowing shapes/strides.
pub const TPRIMS_ERR_SHAPE: tprims_status = 2;
/// Unsupported or mismatched element type.
pub const TPRIMS_ERR_DTYPE: tprims_status = 3;
/// Unsupported device.
pub const TPRIMS_ERR_DEVICE: tprims_status = 4;
/// Output marked read-only.
pub const TPRIMS_ERR_READ_ONLY: tprims_status = 5;
/// Output has zero or overlapping strides.
pub const TPRIMS_ERR_ALIASED: tprims_status = 6;
/// The operation would copy an operand but `TPRIMS_NO_MATERIALIZE` was given.
pub const TPRIMS_ERR_WOULD_MATERIALIZE: tprims_status = 7;
/// Numerical failure (singular, not positive definite, no convergence).
pub const TPRIMS_ERR_NUMERICAL: tprims_status = 8;
/// The executor has calls in flight.
pub const TPRIMS_BUSY: tprims_status = 9;
/// Closing a pool from one of its own workers.
pub const TPRIMS_ERR_WOULD_DEADLOCK: tprims_status = 10;
/// The executor was closed.
pub const TPRIMS_ERR_CLOSED: tprims_status = 11;
/// A panic was caught at the ABI boundary.
pub const TPRIMS_ERR_PANIC: tprims_status = 12;
/// Internal error.
pub const TPRIMS_ERR_INTERNAL: tprims_status = 13;

thread_local! {
    static LAST: RefCell<CString> = RefCell::new(CString::default());
}

/// An error crossing the ABI: a status and a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfiError {
    /// Status code.
    pub status: tprims_status,
    /// Message for `tprims_last_error`.
    pub message: String,
}

impl FfiError {
    /// A new error.
    pub fn new(status: tprims_status, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

/// Record `message` for [`tprims_last_error`] on this thread.
pub fn set_last_error(message: &str) {
    let c = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST.with(|l| *l.borrow_mut() = c);
}

/// Clear the message without allocating when it is already empty (the
/// success path of every call).
fn clear_last_error() {
    LAST.with(|l| {
        if !l.borrow().as_bytes().is_empty() {
            *l.borrow_mut() = CString::default();
        }
    });
}

/// Run an ABI body: catch panics, record the error message, return a status.
pub fn ffi(body: impl FnOnce() -> Result<(), FfiError>) -> tprims_status {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(Ok(())) => {
            clear_last_error();
            TPRIMS_OK
        }
        Ok(Err(e)) => {
            set_last_error(&e.message);
            e.status
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".into());
            set_last_error(&format!("panic: {msg}"));
            TPRIMS_ERR_PANIC
        }
    }
}

/// The message of the last failed call on this thread (empty after a
/// success). Valid until the next tprims call on this thread.
///
/// # Safety
///
/// Always safe to call; the returned pointer must not be freed.
#[no_mangle]
pub extern "C" fn tprims_last_error() -> *const c_char {
    LAST.with(|l| l.borrow().as_ptr())
}
