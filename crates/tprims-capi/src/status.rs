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
/// Destroying a pool from one of its own workers.
pub const TPRIMS_ERR_WOULD_DEADLOCK: tprims_status = 10;
// 11 was `TPRIMS_ERR_CLOSED`; an executor has no closed state any more
// (destruction consumes the handle), so the code is retired, not reused.
/// A panic was caught at the ABI boundary.
pub const TPRIMS_ERR_PANIC: tprims_status = 12;
/// Internal error.
pub const TPRIMS_ERR_INTERNAL: tprims_status = 13;
/// Index labels that do not describe a contraction (TAPP).
pub const TPRIMS_ERR_LABELS: tprims_status = 14;
/// A well-formed request this implementation declines (an unsupported
/// precision, element operation, or index class).
pub const TPRIMS_ERR_UNSUPPORTED: tprims_status = 15;

/// A one-line description of `status` (the text behind `TAPP_explain_error`).
pub fn describe(status: tprims_status) -> &'static str {
    match status {
        TPRIMS_OK => "success",
        TPRIMS_ERR_INVALID_ARGUMENT => "null pointer, invalid handle or invalid argument",
        TPRIMS_ERR_SHAPE => "inconsistent or overflowing extents or strides",
        TPRIMS_ERR_DTYPE => "unsupported or mismatched data type",
        TPRIMS_ERR_DEVICE => "unsupported device",
        TPRIMS_ERR_READ_ONLY => "output is read-only",
        TPRIMS_ERR_ALIASED => "output aliases an operand or overlaps itself",
        TPRIMS_ERR_WOULD_MATERIALIZE => "the operation would copy an operand",
        TPRIMS_ERR_NUMERICAL => "numerical failure",
        TPRIMS_BUSY => "executor has calls in flight",
        TPRIMS_ERR_WOULD_DEADLOCK => "executor destroyed from one of its own workers",
        TPRIMS_ERR_PANIC => "a panic was caught at the ABI boundary",
        TPRIMS_ERR_INTERNAL => "internal error",
        TPRIMS_ERR_LABELS => "invalid index labels",
        TPRIMS_ERR_UNSUPPORTED => "operation not supported by this implementation",
        _ => "unknown error",
    }
}

/// `TAPP_check_success`: whether `error` is [`TPRIMS_OK`]. Zero is the only
/// value upstream TAPP fixes; every other code is provider-defined.
#[no_mangle]
pub extern "C" fn TAPP_check_success(error: i32) -> bool {
    error == TPRIMS_OK
}

/// `TAPP_explain_error`: copy a description of `error` into `message`
/// (NUL-terminated, at most `maxlen - 1` characters). Returns the untruncated
/// length without the NUL, as `snprintf` does; `maxlen == 0` writes nothing.
///
/// # Safety
///
/// `message` is valid for `maxlen` bytes, or `maxlen` is zero.
#[no_mangle]
pub unsafe extern "C" fn TAPP_explain_error(
    error: i32,
    maxlen: usize,
    message: *mut c_char,
) -> usize {
    let s = describe(error).as_bytes();
    if !message.is_null() && maxlen > 0 {
        let n = s.len().min(maxlen - 1);
        // SAFETY: `n < maxlen` bytes are writable per the contract.
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr(), message as *mut u8, n);
            *message.add(n) = 0;
        }
    }
    s.len()
}

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
    // try_with: during thread teardown the slot may be gone; drop the message.
    let _ = LAST.try_with(|l| *l.borrow_mut() = c);
}

/// Clear the message without allocating when it is already empty (the
/// success path of every call).
fn clear_last_error() {
    let _ = LAST.try_with(|l| {
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
    LAST.try_with(|l| l.borrow().as_ptr())
        .unwrap_or(c"".as_ptr())
}
