//! TAPP C-ABI front end.
//!
//! Implements the C interface of the Tensor Algebra Processing Primitives
//! standard (arXiv:2601.07827, <https://github.com/TAPPorg/reference-implementation>)
//! on top of [`tprims_contract`], so that this engine, TBLIS and cuTENSOR are
//! drop-in swappable behind one header.
//!
//! The symbols exported here are ABI-compatible with the upstream
//! `api/include/tapp/*.h` headers: all handles are `intptr_t`, all fallible
//! calls return a `TAPP_error` (`int`, zero on success).
//!
//! # One contraction, end to end
//!
//! Five objects are created and five destroyed, in the order below. Every
//! fallible call is checked with [`TAPP_check_success`] rather than against a
//! numeric code, which is the only portable way — see the note under
//! [Coverage](#coverage). The example is Rust because that is what compiles as a
//! test; a C caller writes the same sequence, and
//! [`examples/c-consumer`][c-consumer] in the repository does.
//!
//! ```
//! use std::ffi::c_void;
//! use tensorprimitives_tapp::*;
//!
//! // D[i,j] = sum_k A[i,k] * B[k,j] over 2x2 column-major f64 tensors.
//! let extents = [2i64, 2];
//! let strides = [1i64, 2];                    // column-major, in elements
//! let (i, j, k) = (b'i' as i64, b'j' as i64, b'k' as i64);
//!
//! unsafe {
//!     let mut handle = 0isize;
//!     assert!(TAPP_check_success(TAPP_create_handle(&mut handle)));
//!     let mut exec = 0isize;
//!     assert!(TAPP_check_success(TAPP_create_executor(&mut exec)));
//!
//!     // One info serves all four operands: they share a shape here.
//!     let mut info = 0isize;
//!     assert!(TAPP_check_success(TAPP_create_tensor_info(
//!         &mut info, TAPP_F64, 2, extents.as_ptr(), strides.as_ptr(),
//!     )));
//!
//!     // Labels are what make this a contraction rather than a shape: `k` is in
//!     // A and B only, so it is summed; `i` and `j` survive into D.
//!     let (ia, ib, id) = ([i, k], [k, j], [i, j]);
//!     let mut plan = 0isize;
//!     assert!(TAPP_check_success(TAPP_create_tensor_product(
//!         &mut plan,
//!         handle,
//!         TAPP_IDENTITY, info, ia.as_ptr(),
//!         TAPP_IDENTITY, info, ib.as_ptr(),
//!         TAPP_IDENTITY, info, id.as_ptr(),   // C is required; reuse D's info
//!         TAPP_IDENTITY, info, id.as_ptr(),
//!         TAPP_F64,
//!     )));
//!
//!     let a = [1.0f64, 2.0, 3.0, 4.0];
//!     let b = [1.0f64, 0.0, 0.0, 1.0];        // the 2x2 identity
//!     let mut d = [0.0f64; 4];
//!     let (alpha, beta) = (1.0f64, 0.0f64);
//!
//!     // `alpha` and `beta` are read as the plan's element type, so they are
//!     // passed by pointer. A null `c` is TAPP_IN_PLACE, legal here only because
//!     // `beta` is zero.
//!     let mut status = 0isize;
//!     assert!(TAPP_check_success(TAPP_execute_product(
//!         plan,
//!         exec,
//!         &mut status,
//!         &alpha as *const f64 as *const c_void,
//!         a.as_ptr() as *const c_void,
//!         b.as_ptr() as *const c_void,
//!         &beta as *const f64 as *const c_void,
//!         std::ptr::null(),
//!         d.as_mut_ptr() as *mut c_void,
//!     )));
//!     assert_eq!(d, a);                       // multiplying by the identity
//!
//!     // Destroy in any order: a product does not borrow the infos it was built
//!     // from, so `info` could have gone immediately after the call above.
//!     TAPP_destroy_status(status);
//!     TAPP_destroy_tensor_product(plan);
//!     TAPP_destroy_tensor_info(info);
//!     TAPP_destroy_executor(exec);
//!     TAPP_destroy_handle(handle);
//! }
//! ```
//!
//! [c-consumer]: https://github.com/lkdvos/tensorprimitives-rs/tree/main/examples/c-consumer
//!
//! # Coverage
//!
//! | TAPP feature | status |
//! |---|---|
//! | `TAPP_F32` / `TAPP_F64` / `TAPP_C32` / `TAPP_C64` | supported |
//! | `TAPP_F16` / `TAPP_BF16` | rejected (`TAPP_ERROR_DATATYPE`) |
//! | Case 1 simple contraction | supported |
//! | Case 2 Hadamard / batch indices | supported |
//! | Case 3 repeated indices (diagonals) | supported |
//! | Case 4 isolated input indices (reductions) | supported |
//! | Case 5 isolated *output* indices (broadcast) | rejected, as TAPP permits |
//! | `TAPP_CONJUGATE` on any operand | supported (free: folded into packing) |
//! | computational precision | `TAPP_DEFAULT_PREC` or the storage precision (`F32F32_ACCUM_F32` for `f32`/`c32`, `F64F64_ACCUM_F64` for `f64`/`c64`); anything else is `TAPP_ERROR_UNSUPPORTED` |
//! | element operations | `TAPP_IDENTITY` and `TAPP_CONJUGATE`; any other value is `TAPP_ERROR_UNSUPPORTED` |
//! | executors | `0` and `TAPP_create_executor` are serial; `tprims_tapp_executor_create_rayon` owns a Rayon pool, which `TAPP_destroy_executor` joins |
//! | `D` overlapping `A`/`B`, or overlapping itself | rejected (`TAPP_ERROR_ALIASED`); `C == D` only with equal element mapping |
//! | mixed *storage* types across operands | **rejected** (`TAPP_ERROR_DATATYPE`) — one element type per plan |
//! | batched product | supported |
//! | `TAPP_IN_PLACE` (a null `C`) | only with `beta == 0`; a non-zero `beta` is refused rather than reinterpreted |
//! | `TAPP_attr_set` / `_get` / `_clear` | exported, and refuse every key: upstream specifies none |
//!
//! Note that `TAPP_ERROR_*` beyond `TAPP_SUCCESS` are **this implementation's
//! own numbering**: the `tprims_status` codes of `tprims-core`, shared by every
//! part of `libtprims`. Upstream `error.h` is a bare `typedef int TAPP_error`
//! with no enumerators, so zero is the only value the standard fixes and a
//! portable caller must go through [`TAPP_check_success`] rather than compare
//! codes.
//!
//! # The handle discipline, once
//!
//! Every `extern "C"` entry point here inherits the same safety contract from
//! the TAPP specification, and each one's `# Safety` section says only what is
//! specific to it on top of these three:
//!
//! * A handle argument (`isize`) must be either `0` or a live value produced by
//!   the matching `TAPP_create_*` call and not yet destroyed. `0` is rejected
//!   with `TAPP_ERROR_NULL` (an executor `0` is the default serial executor,
//!   not an error); a stale or foreign non-zero value is undefined behaviour,
//!   because nothing distinguishes it from a live one.
//! * Each handle may be destroyed once. An executor refuses with
//!   `TPRIMS_BUSY` while calls are in flight and the handle stays live; for the
//!   others no call may be in flight at the time, and the caller synchronizes
//!   destruction with the start of new calls.
//! * Data pointers must be valid for every offset the extents and strides
//!   recorded in the tensor infos generate: the raw ABI carries no allocation
//!   lengths. Overflow of the addressed range and output aliasing are checked;
//!   see [`tprims_contract::Plan::execute_raw`], which is what they reach.
//!
//! Handles are `Box::into_raw` pointers cast to `isize`, so they are *not*
//! interchangeable between processes and must not be serialised.
#![warn(missing_docs)]

use std::ffi::c_void;
use std::os::raw::c_int;

use num_complex::Complex;
use tprims_contract::api::{self, CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem};
use tprims_contract::{Plan, PlanConfig};
use tprims_core::exec::with_executor;
use tprims_core::status::{
    ffi, FfiError, TPRIMS_BUSY, TPRIMS_ERR_ALIASED, TPRIMS_ERR_DTYPE, TPRIMS_ERR_INTERNAL,
    TPRIMS_ERR_INVALID_ARGUMENT, TPRIMS_ERR_LABELS, TPRIMS_ERR_PANIC, TPRIMS_ERR_SHAPE,
    TPRIMS_ERR_UNSUPPORTED, TPRIMS_ERR_WOULD_DEADLOCK, TPRIMS_ERR_WOULD_MATERIALIZE,
};

// The executor, the status codes and `TAPP_check_success` / `TAPP_explain_error`
// live in `tprims-core`, so every part of `libtprims` shares one definition.
pub use tprims_core::exec::{
    tprims_rayon_opts, tprims_tapp_executor_create_rayon, tprims_tapp_executor_get_threads,
    tprims_tapp_executor_set_budget, TAPP_create_executor, TAPP_destroy_executor,
};
pub use tprims_core::status::{TAPP_check_success, TAPP_explain_error};

// ---------------------------------------------------------------- datatypes

// The `datatype_t` enumerators, in the order the upstream header declares them.
// The numeric values are ABI and must not be reordered — TBLIS 1.3 and 2.0 differ
// by exactly such a reordering of their own `type_t`, which produces plausible
// wrong answers rather than an error.

/// `TAPP_F32`: single-precision real. Handled by [`tprims_contract`] as `f32`.
pub const TAPP_F32: c_int = 0;
/// `TAPP_F64`: double-precision real, i.e. `f64`.
pub const TAPP_F64: c_int = 1;
/// `TAPP_C32`: single-precision complex, interleaved — layout-compatible with
/// C99 `float _Complex` and `num_complex::Complex<f32>`.
pub const TAPP_C32: c_int = 2;
/// `TAPP_C64`: double-precision complex, interleaved.
pub const TAPP_C64: c_int = 3;
/// `TAPP_F16`: accepted by the enumeration, rejected by this implementation
/// with [`TAPP_ERROR_DATATYPE`]. There is no storage scalar for it.
pub const TAPP_F16: c_int = 4;
/// `TAPP_BF16`: as [`TAPP_F16`], rejected.
pub const TAPP_BF16: c_int = 5;

/// `TAPP_DEFAULT_PREC`: compute at the storage precision.
pub const TAPP_DEFAULT_PREC: c_int = -1;
/// `TAPP_F32F32_ACCUM_F32`: accepted for `f32` and `c32` storage.
pub const TAPP_F32F32_ACCUM_F32: c_int = 0;
/// `TAPP_F64F64_ACCUM_F64`: accepted for `f64` and `c64` storage.
pub const TAPP_F64F64_ACCUM_F64: c_int = 1;

/// `TAPP_IDENTITY`: read the operand as stored. The default for every operand.
pub const TAPP_IDENTITY: c_int = 0;
/// `TAPP_CONJUGATE`: complex-conjugate the operand. Free here — it is folded
/// into packing or into the write-back, never a separate pass.
pub const TAPP_CONJUGATE: c_int = 1;

// ------------------------------------------------------------------- errors

/// No error. The only value for which `TAPP_check_success` returns `true`.
pub const TAPP_SUCCESS: c_int = 0;
/// A required pointer was null, or a handle was `0` / not live. (The tprims
/// status `TPRIMS_ERR_INVALID_ARGUMENT`.)
pub const TAPP_ERROR_NULL: c_int = TPRIMS_ERR_INVALID_ARGUMENT;
/// An unsupported element type, or the four operands did not agree on one.
pub const TAPP_ERROR_DATATYPE: c_int = TPRIMS_ERR_DTYPE;
/// Extents or strides are inconsistent: a label with two different extents, a
/// negative extent, or a shape or stride that overflows.
pub const TAPP_ERROR_SHAPE: c_int = TPRIMS_ERR_SHAPE;
/// The index labels do not describe a contraction: wrong count for the rank, or
/// `C` and `D` labelled differently.
pub const TAPP_ERROR_LABELS: c_int = TPRIMS_ERR_LABELS;
/// A well-formed request this implementation declines: TAPP case 5 (an
/// output-only index, a broadcast), a computational precision other than the
/// storage precision, an element operation other than identity or conjugation,
/// or a null `C` with a nonzero `beta`.
pub const TAPP_ERROR_UNSUPPORTED: c_int = TPRIMS_ERR_UNSUPPORTED;
/// A bug or an allocation failure.
pub const TAPP_ERROR_INTERNAL: c_int = TPRIMS_ERR_INTERNAL;
/// `TAPP_destroy_executor` with calls in flight; the handle stays live.
pub const TAPP_ERROR_BUSY: c_int = TPRIMS_BUSY;
/// `TAPP_destroy_executor` from one of the executor's own workers; the handle
/// stays live.
pub const TAPP_ERROR_WOULD_DEADLOCK: c_int = TPRIMS_ERR_WOULD_DEADLOCK;
/// A panic was caught at the ABI boundary and reported instead of unwinding.
pub const TAPP_ERROR_PANIC: c_int = TPRIMS_ERR_PANIC;
/// Output memory that overlaps an input or itself, or `C` and `D` that overlap
/// without being the same mapping.
pub const TAPP_ERROR_ALIASED: c_int = TPRIMS_ERR_ALIASED;

fn fail(status: c_int, message: impl Into<String>) -> FfiError {
    FfiError::new(status, message)
}

fn null(what: &str) -> FfiError {
    fail(
        TPRIMS_ERR_INVALID_ARGUMENT,
        format!("{what} is null or zero"),
    )
}

/// The C status of a contraction error: the one table of the ABI.
///
/// A label count or a `C`/`D` label-set mismatch is `LABELS`; extents, checked
/// sizes and layout-range incompatibilities are `SHAPE`; an output (or `C`)
/// that overlaps itself or an input is `ALIASED`; an unsupported operation
/// (including an output-only label) or an unusable forced family is
/// `UNSUPPORTED`; a wrong storage scalar is `DTYPE`.
fn map_err(e: api::Error) -> FfiError {
    use api::{ConfigError, Error as E, ShapeError, Unsupported};
    let status = match &e {
        E::Shape(ShapeError::LabelCount { .. } | ShapeError::OutputLabelMismatch) => {
            TPRIMS_ERR_LABELS
        }
        E::Shape(_) | E::Layout(_) => TPRIMS_ERR_SHAPE,
        E::Alias(_) => TPRIMS_ERR_ALIASED,
        E::Unsupported(Unsupported::WouldMaterialize { .. }) => TPRIMS_ERR_WOULD_MATERIALIZE,
        E::Unsupported(_) | E::Select(_) | E::Exec(_) => TPRIMS_ERR_UNSUPPORTED,
        E::Config(ConfigError::DtypeMismatch { .. }) => TPRIMS_ERR_DTYPE,
        E::Config(ConfigError::CLabels) => TPRIMS_ERR_LABELS,
        E::Config(_) => TPRIMS_ERR_INVALID_ARGUMENT,
        _ => TPRIMS_ERR_INTERNAL,
    };
    fail(status, e.to_string())
}

// ------------------------------------------------------------------ handles

/// Library handle. Stateless so far, but a *non*-zero-sized allocation on
/// purpose: with an empty struct `Box::into_raw` returns `NonNull::dangling()`,
/// so every handle the library ever issued was the same value, two live handles
/// were indistinguishable, and a C program that created two and destroyed both
/// was performing a double free — harmless only for as long as the state stayed
/// empty. The field costs eight bytes once per library handle.
///
/// Note what this does *not* buy, since an earlier comment here claimed it:
/// handle validity still cannot be checked. Nothing can distinguish a pointer
/// this crate produced from an arbitrary non-zero `intptr_t`, so `0` remains the
/// only value the ABI can reject.
struct HandleState {
    /// Reserved for state a future version needs; also what keeps the type
    /// non-zero-sized. Not read.
    _reserved: u64,
}

/// Create a library handle.
///
/// # Safety
/// `handle` must be a valid, writable `*mut isize` (or null, which is rejected).
/// On success it receives a handle to be released with
/// [`TAPP_destroy_handle`].
#[no_mangle]
pub unsafe extern "C" fn TAPP_create_handle(handle: *mut isize) -> c_int {
    ffi(|| {
        if handle.is_null() {
            return Err(null("handle out-parameter"));
        }
        // SAFETY: non-null and writable per the contract.
        unsafe { *handle = Box::into_raw(Box::new(HandleState { _reserved: 0 })) as isize };
        Ok(())
    })
}

/// Release a handle from [`TAPP_create_handle`].
///
/// # Safety
/// `handle` must be live and not already destroyed. See the crate docs on the
/// handle discipline.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_handle(handle: isize) -> c_int {
    ffi(|| {
        if handle == 0 {
            return Err(null("handle"));
        }
        // SAFETY: a live handle from `TAPP_create_handle`.
        drop(unsafe { Box::from_raw(handle as *mut HandleState) });
        Ok(())
    })
}

// ------------------------------------------------------------- attributes
//
// `api/include/tapp/attributes.h` declares three functions over an opaque
// `TAPP_attr` (`intptr_t`) and a `TAPP_key` (`int`), and specifies no keys. They
// are implemented here as refusals rather than left out, because a symbol that
// is merely absent turns a C program that includes `<tapp.h>` and calls one into
// a **link** failure — the least diagnosable kind — whereas a refusal is a value
// the caller can check and explain. Found by the conformance suite, which
// re-declares the whole header and so notices what is missing from it.
//
// Prototypes verified against the upstream header (TAPPorg/reference-implementation,
// api/include/tapp/attributes.h), not reconstructed.

/// Set an attribute. Always fails with [`TAPP_ERROR_UNSUPPORTED`]: this
/// implementation defines no attribute keys, and upstream specifies none.
///
/// # Safety
/// Trivially safe — no argument is dereferenced — and `unsafe` only to match the
/// declared C signature.
#[no_mangle]
pub unsafe extern "C" fn TAPP_attr_set(_attr: isize, _key: c_int, _value: *mut c_void) -> c_int {
    ffi(|| {
        Err(fail(
            TPRIMS_ERR_UNSUPPORTED,
            "no attribute keys are defined",
        ))
    })
}

/// Get an attribute. Always fails with [`TAPP_ERROR_UNSUPPORTED`]; `value` is
/// set to null first if it is non-null, so a caller that ignores the return code
/// reads a defined value rather than whatever was on its stack.
///
/// # Safety
/// `value` must be null or valid for one pointer-sized write.
#[no_mangle]
pub unsafe extern "C" fn TAPP_attr_get(
    _attr: isize,
    _key: c_int,
    value: *mut *mut c_void,
) -> c_int {
    if !value.is_null() {
        // SAFETY: non-null and writable per the contract.
        unsafe { *value = std::ptr::null_mut() };
    }
    ffi(|| {
        Err(fail(
            TPRIMS_ERR_UNSUPPORTED,
            "no attribute keys are defined",
        ))
    })
}

/// Clear an attribute. Always fails with [`TAPP_ERROR_UNSUPPORTED`].
///
/// # Safety
/// Trivially safe; `unsafe` only to match the declared C signature.
#[no_mangle]
pub unsafe extern "C" fn TAPP_attr_clear(_attr: isize, _key: c_int) -> c_int {
    ffi(|| {
        Err(fail(
            TPRIMS_ERR_UNSUPPORTED,
            "no attribute keys are defined",
        ))
    })
}

/// Release a status object. A no-op: execution here is synchronous, so it never
/// produces a status to release. Provided because the header declares it and a
/// conforming caller will call it.
///
/// # Safety
/// Trivially safe; `unsafe` only to match the declared C signature.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_status(_status: isize) -> c_int {
    ffi(|| Ok(()))
}

// ------------------------------------------------------------- tensor infos

/// One tensor as the caller described it: extents and strides in elements,
/// kept as `i64` because the setters can change them after creation. Everything
/// is validated, once, when a product is created.
#[derive(Clone, Debug, Default)]
struct TapLayout {
    extents: Vec<i64>,
    strides: Vec<i64>,
}

impl TapLayout {
    fn ndim(&self) -> usize {
        self.extents.len()
    }

    /// Truncate or extend; new modes get extent 1 and stride 0.
    fn resize(&mut self, ndim: usize) {
        self.extents.resize(ndim, 1);
        self.strides.resize(ndim, 0);
    }
}

struct TensorInfo {
    dtype: c_int,
    layout: TapLayout,
}

/// Bytes per element of a supported datatype.
fn elem_size(dtype: c_int) -> Option<usize> {
    match dtype {
        TAPP_F32 => Some(4),
        TAPP_F64 | TAPP_C32 => Some(8),
        TAPP_C64 => Some(16),
        _ => None,
    }
}

/// Describe one tensor: element type, rank, extents and strides (in elements).
///
/// Strides may be negative or zero; extents may not be negative. A rank of 0 is
/// legal and describes a scalar, in which case `extents` and `strides` are not
/// read and may be null. Overflow of the addressed range is checked when a
/// product is created, since the setters can change the description.
///
/// # Safety
/// `info` must be a valid, writable `*mut isize`. When `nmode > 0`, `extents`
/// and `strides` must each be valid for `nmode` `i64` reads. On success `*info`
/// receives a handle to release with [`TAPP_destroy_tensor_info`].
#[no_mangle]
pub unsafe extern "C" fn TAPP_create_tensor_info(
    info: *mut isize,
    dtype: c_int,
    nmode: c_int,
    extents: *const i64,
    strides: *const i64,
) -> c_int {
    ffi(move || {
        if info.is_null() {
            return Err(null("info out-parameter"));
        }
        if nmode < 0 {
            return Err(fail(TPRIMS_ERR_SHAPE, "negative mode count"));
        }
        if elem_size(dtype).is_none() {
            return Err(fail(TPRIMS_ERR_DTYPE, "unsupported datatype"));
        }
        let n = nmode as usize;
        if n > 0 && (extents.is_null() || strides.is_null()) {
            return Err(null("extents or strides"));
        }
        // SAFETY: `n` reads are valid per the contract.
        let (e, s) = unsafe {
            if n == 0 {
                (Vec::new(), Vec::new())
            } else {
                (
                    std::slice::from_raw_parts(extents, n).to_vec(),
                    std::slice::from_raw_parts(strides, n).to_vec(),
                )
            }
        };
        let layout = TapLayout {
            extents: e,
            strides: s,
        };
        // SAFETY: non-null and writable per the contract.
        unsafe { *info = Box::into_raw(Box::new(TensorInfo { dtype, layout })) as isize };
        Ok(())
    })
}

/// Release a tensor info from [`TAPP_create_tensor_info`].
///
/// A [`TAPP_create_tensor_product`] built from it does **not** borrow it — the
/// plan copies everything it needs — so an info may be destroyed while products
/// derived from it are still in use.
///
/// # Safety
/// `info` must be live and not already destroyed.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_tensor_info(info: isize) -> c_int {
    ffi(|| {
        if info == 0 {
            return Err(null("info"));
        }
        // SAFETY: a live handle from `TAPP_create_tensor_info`.
        drop(unsafe { Box::from_raw(info as *mut TensorInfo) });
        Ok(())
    })
}

/// The rank recorded in `info`, or `-1` if the handle is `0`.
///
/// # Safety
/// `info` must be `0` or live.
#[no_mangle]
pub unsafe extern "C" fn TAPP_get_nmodes(info: isize) -> c_int {
    // SAFETY: zero or live per the contract.
    match unsafe { (info as *const TensorInfo).as_ref() } {
        Some(t) => t.layout.ndim() as c_int,
        None => -1,
    }
}

/// Change the rank in place, truncating or extending.
///
/// New modes get extent 1 and stride 0, which is the identity for this engine:
/// extent-1 axes are dropped during planning. So growing the rank and then
/// setting extents and strides is well defined, but growing it and *not* doing
/// so leaves the tensor describing the same elements it did before.
///
/// # Safety
/// `info` must be `0` or live.
#[no_mangle]
pub unsafe extern "C" fn TAPP_set_nmodes(info: isize, nmodes: c_int) -> c_int {
    ffi(|| {
        // SAFETY: zero or live per the contract.
        let t = unsafe { (info as *mut TensorInfo).as_mut() }.ok_or_else(|| null("info"))?;
        if nmodes < 0 {
            return Err(fail(TPRIMS_ERR_SHAPE, "negative mode count"));
        }
        t.layout.resize(nmodes as usize);
        Ok(())
    })
}

/// Copy `info`'s extents out. Returns nothing, as TAPP declares it, so query
/// the rank with [`TAPP_get_nmodes`] first.
///
/// # Safety
/// `info` must be `0` or live, and `extents` must be null or valid for
/// `TAPP_get_nmodes(info)` `i64` writes.
#[no_mangle]
pub unsafe extern "C" fn TAPP_get_extents(info: isize, extents: *mut i64) {
    // SAFETY: zero or live per the contract.
    if let Some(t) = unsafe { (info as *const TensorInfo).as_ref() } {
        if !extents.is_null() {
            // SAFETY: `ndim` writes are valid per the contract.
            unsafe {
                std::ptr::copy_nonoverlapping(t.layout.extents.as_ptr(), extents, t.layout.ndim())
            };
        }
    }
}

/// Replace `info`'s extents, keeping its rank.
///
/// # Safety
/// `info` must be `0` or live, and `extents` must be null or valid for
/// `TAPP_get_nmodes(info)` `i64` reads.
#[no_mangle]
pub unsafe extern "C" fn TAPP_set_extents(info: isize, extents: *const i64) -> c_int {
    ffi(|| {
        // SAFETY: zero or live per the contract.
        let t = unsafe { (info as *mut TensorInfo).as_mut() }.ok_or_else(|| null("info"))?;
        if extents.is_null() {
            return Err(null("extents"));
        }
        let n = t.layout.ndim();
        // SAFETY: `n` reads are valid per the contract.
        t.layout
            .extents
            .copy_from_slice(unsafe { std::slice::from_raw_parts(extents, n) });
        Ok(())
    })
}

/// Copy `info`'s strides out, in elements. As [`TAPP_get_extents`], this
/// returns nothing.
///
/// # Safety
/// `info` must be `0` or live, and `strides` must be null or valid for
/// `TAPP_get_nmodes(info)` `i64` writes.
#[no_mangle]
pub unsafe extern "C" fn TAPP_get_strides(info: isize, strides: *mut i64) {
    // SAFETY: zero or live per the contract.
    if let Some(t) = unsafe { (info as *const TensorInfo).as_ref() } {
        if !strides.is_null() {
            // SAFETY: `ndim` writes are valid per the contract.
            unsafe {
                std::ptr::copy_nonoverlapping(t.layout.strides.as_ptr(), strides, t.layout.ndim())
            };
        }
    }
}

/// Replace `info`'s strides, keeping its rank.
///
/// # Safety
/// `info` must be `0` or live, and `strides` must be null or valid for
/// `TAPP_get_nmodes(info)` `i64` reads.
#[no_mangle]
pub unsafe extern "C" fn TAPP_set_strides(info: isize, strides: *const i64) -> c_int {
    ffi(|| {
        // SAFETY: zero or live per the contract.
        let t = unsafe { (info as *mut TensorInfo).as_mut() }.ok_or_else(|| null("info"))?;
        if strides.is_null() {
            return Err(null("strides"));
        }
        let n = t.layout.ndim();
        // SAFETY: `n` reads are valid per the contract.
        t.layout
            .strides
            .copy_from_slice(unsafe { std::slice::from_raw_parts(strides, n) });
        Ok(())
    })
}

// ----------------------------------------------------------------- products

/// The prepared contraction of one storage type.
enum Prepared {
    F32(Plan<f32>),
    F64(Plan<f64>),
    C32(Plan<Complex<f32>>),
    C64(Plan<Complex<f64>>),
}

struct Product {
    plan: Prepared,
}

fn op_of(op: c_int, name: &str) -> Result<Op, FfiError> {
    match op {
        TAPP_IDENTITY => Ok(Op::Identity),
        TAPP_CONJUGATE => Ok(Op::Conjugate),
        _ => Err(fail(
            TPRIMS_ERR_UNSUPPORTED,
            format!("element operation {op} on {name} is not supported"),
        )),
    }
}

/// One operand as the problem describes it: validated extents and strides at
/// the pointer's own origin (TAPP has no logical offsets), and the element
/// operation.
fn spec(t: &TensorInfo, op: Op) -> Result<OperandSpec, FfiError> {
    let layout =
        LayoutSpec::from_signed(&t.layout.extents, &t.layout.strides, 0).map_err(map_err)?;
    Ok(OperandSpec::new(layout).with_op(op))
}

fn build<T: api::Scalar>(problem: &Problem) -> Result<Plan<T>, FfiError> {
    // Standard calls use the default configuration: no tuning API in the C ABI.
    Plan::<T>::new(problem, &PlanConfig::default()).map_err(map_err)
}

/// Plan `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))` from four tensor
/// infos and their index labels.
///
/// All the work that depends only on shapes, strides and labels happens here —
/// the problem is lowered and validated once, and the index classification,
/// folding and scatter vectors are built — so this is the call to hoist out of a
/// loop. The plan snapshots the metadata, owns no data pointer and no executor,
/// and can be executed against different buffers and serial or Rayon executors.
///
/// `handle` must be a live library handle (zero is rejected). `prec` must be
/// [`TAPP_DEFAULT_PREC`] or the precision of the storage type
/// ([`TAPP_F32F32_ACCUM_F32`] for `f32`/`c32`, [`TAPP_F64F64_ACCUM_F64`] for
/// `f64`/`c64`); anything else is `TAPP_ERROR_UNSUPPORTED`, not ignored. Element
/// operations other than identity and conjugation are rejected too, and so are
/// four infos that do not agree on one storage type (TAPP permits mixed types;
/// this engine does not). Shapes and strides whose addressed range overflows
/// are `TAPP_ERROR_SHAPE`, and a `D` whose elements overlap one another is
/// `TAPP_ERROR_ALIASED`.
///
/// `C` is required, because the C signature has no way to omit it — pass
/// `beta == 0` at execution time to have it ignored, or pass `D`'s info for it.
///
/// # Safety
/// `plan_out` must be a valid, writable `*mut isize`. `a`, `b`, `c`, `d` must be
/// live tensor infos. Each `idx_*` must be valid for the corresponding info's
/// rank in `i64` reads, or may be null when that rank is 0. On success
/// `*plan_out` receives a handle to release with
/// [`TAPP_destroy_tensor_product`].
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn TAPP_create_tensor_product(
    plan_out: *mut isize,
    handle: isize,
    op_a: c_int,
    a: isize,
    idx_a: *const i64,
    op_b: c_int,
    b: isize,
    idx_b: *const i64,
    op_c: c_int,
    c: isize,
    idx_c: *const i64,
    op_d: c_int,
    d: isize,
    idx_d: *const i64,
    prec: c_int,
) -> c_int {
    ffi(move || {
        if plan_out.is_null() {
            return Err(null("plan out-parameter"));
        }
        if handle == 0 {
            return Err(null("library handle"));
        }
        // SAFETY: zero or live infos per the contract.
        let (ta, tb, tc, td) = unsafe {
            (
                (a as *const TensorInfo).as_ref(),
                (b as *const TensorInfo).as_ref(),
                (c as *const TensorInfo).as_ref(),
                (d as *const TensorInfo).as_ref(),
            )
        };
        let (Some(ta), Some(tb), Some(tc), Some(td)) = (ta, tb, tc, td) else {
            return Err(null("tensor info"));
        };

        // TAPP allows mixed storage types; this engine computes at a single
        // element type, so require all four to agree.
        if ta.dtype != tb.dtype || ta.dtype != tc.dtype || ta.dtype != td.dtype {
            return Err(fail(
                TPRIMS_ERR_DTYPE,
                "operands have different storage types",
            ));
        }
        let storage_prec = match ta.dtype {
            TAPP_F32 | TAPP_C32 => TAPP_F32F32_ACCUM_F32,
            _ => TAPP_F64F64_ACCUM_F64,
        };
        if prec != TAPP_DEFAULT_PREC && prec != storage_prec {
            return Err(fail(
                TPRIMS_ERR_UNSUPPORTED,
                format!("computational precision {prec} differs from the storage precision"),
            ));
        }
        let (oa, ob, oc, od) = (
            op_of(op_a, "A")?,
            op_of(op_b, "B")?,
            op_of(op_c, "C")?,
            op_of(op_d, "D")?,
        );

        let labels = |p: *const i64, t: &TensorInfo, name: &str| -> Result<Vec<i64>, FfiError> {
            let n = t.layout.ndim();
            if n == 0 {
                Ok(Vec::new())
            } else if p.is_null() {
                Err(null(&format!("labels of {name}")))
            } else {
                // SAFETY: `n` reads are valid per the contract.
                Ok(unsafe { std::slice::from_raw_parts(p, n) }.to_vec())
            }
        };
        let (la, lb, lc, ld) = (
            labels(idx_a, ta, "A")?,
            labels(idx_b, tb, "B")?,
            labels(idx_c, tc, "C")?,
            labels(idx_d, td, "D")?,
        );

        // The one lowering: labels, diagonals, reductions, extents, address
        // ranges and output injectivity are all checked by the problem.
        let dtype = match ta.dtype {
            TAPP_F32 => DType::F32,
            TAPP_F64 => DType::F64,
            TAPP_C32 => DType::C32,
            TAPP_C64 => DType::C64,
            _ => return Err(fail(TPRIMS_ERR_DTYPE, "unsupported datatype")),
        };
        let problem = Problem::from_labels(
            dtype,
            spec(ta, oa)?,
            spec(tb, ob)?,
            CSpec::Separate(spec(tc, oc)?),
            spec(td, od)?,
            &Labels::new(&la, &lb, &ld).with_c(&lc),
        )
        .map_err(map_err)?;
        let plan = match dtype {
            DType::F32 => Prepared::F32(build(&problem)?),
            DType::F64 => Prepared::F64(build(&problem)?),
            DType::C32 => Prepared::C32(build(&problem)?),
            DType::C64 => Prepared::C64(build(&problem)?),
        };
        // SAFETY: non-null and writable per the contract.
        unsafe { *plan_out = Box::into_raw(Box::new(Product { plan })) as isize };
        Ok(())
    })
}

/// Release a product from [`TAPP_create_tensor_product`].
///
/// # Safety
/// `plan` must be live and not already destroyed.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_tensor_product(plan: isize) -> c_int {
    ffi(|| {
        if plan == 0 {
            return Err(null("plan"));
        }
        // SAFETY: a live handle from `TAPP_create_tensor_product`.
        drop(unsafe { Box::from_raw(plan as *mut Product) });
        Ok(())
    })
}

/// Data pointers of one execution, as the caller passed them.
#[derive(Clone, Copy)]
struct Item {
    a: *const c_void,
    b: *const c_void,
    c: *const c_void,
    d: *mut c_void,
}

/// Everything an execution checks before it writes anything, the same for a
/// single call and for each item of a batch: the FFI-only null handling here,
/// then the plan's own semantic preflight (output overlap and `C`/`D`
/// aliasing), which is the one definition the safe API shares.
fn validate<T: api::Scalar>(plan: &Plan<T>, it: Item, beta_is_zero: bool) -> Result<(), FfiError> {
    if it.a.is_null() || it.b.is_null() || it.d.is_null() {
        return Err(null("A, B or D data pointer"));
    }
    if it.c.is_null() && !beta_is_zero {
        return Err(fail(
            TPRIMS_ERR_UNSUPPORTED,
            "a null C (TAPP_IN_PLACE) with a nonzero beta is ambiguous in TAPP and not supported; pass D as C to accumulate in place",
        ));
    }
    plan.check_raw(
        it.a as *const T,
        it.b as *const T,
        it.c as *const T,
        it.d as *const T,
    )
    .map_err(map_err)
}

/// Run one validated item on the executor's threads. The plan chooses the actual
/// width from its work estimate and the executor's budget.
///
/// # Safety
///
/// The pointers address every offset of this plan's operands, and `validate`
/// accepted `it`.
unsafe fn run<T: api::Scalar>(
    plan: &Plan<T>,
    exec: &tprims_exec::Exec<'_>,
    alpha: T,
    beta: T,
    it: Item,
    zero: T,
) -> Result<(), FfiError> {
    let beta = if it.c.is_null() { zero } else { beta };
    // SAFETY: forwarded; the executor lends the threads, none are spawned.
    unsafe {
        plan.execute_raw(
            exec,
            alpha,
            it.a as *const T,
            it.b as *const T,
            beta,
            it.c as *const T,
            it.d as *mut T,
        )
    }
    .map_err(map_err)
}

/// Read a scalar of the plan's element type.
///
/// # Safety
/// `p` is null or points to a `T` (possibly unaligned).
unsafe fn scalar<T: Copy>(p: *const c_void, name: &str) -> Result<T, FfiError> {
    if p.is_null() {
        return Err(null(name));
    }
    // SAFETY: non-null and a `T` per the contract.
    Ok(unsafe { std::ptr::read_unaligned(p as *const T) })
}

/// Execute `items` (validated first, as a whole, before any is written).
///
/// # Safety
/// As [`TAPP_execute_product`], for each item.
unsafe fn execute_items<T>(
    plan: &Plan<T>,
    exec: isize,
    alpha: *const c_void,
    beta: *const c_void,
    item: &dyn Fn(usize) -> Item,
    count: usize,
) -> Result<(), FfiError>
where
    T: api::Scalar + Default,
{
    let zero = T::default();
    // SAFETY: `alpha` and `beta` are `T`s per the contract.
    let (al, be) = unsafe { (scalar::<T>(alpha, "alpha")?, scalar::<T>(beta, "beta")?) };
    // Validate every item before the first write; nothing is allocated, so a
    // single product costs no more than it did before batches existed.
    for i in 0..count {
        validate(plan, item(i), be == zero)?;
    }
    // SAFETY: `exec` is zero or a live executor per the contract.
    unsafe {
        with_executor(exec, |x| {
            for i in 0..count {
                // SAFETY: validated above; the caller's pointer contract.
                run(plan, x, al, be, item(i), zero)?;
            }
            Ok(())
        })
    }
}

/// Dispatch on the plan's storage type.
///
/// # Safety
/// As [`execute_items`].
unsafe fn dispatch(
    p: &Product,
    exec: isize,
    alpha: *const c_void,
    beta: *const c_void,
    item: &dyn Fn(usize) -> Item,
    count: usize,
) -> Result<(), FfiError> {
    // SAFETY: forwarded.
    unsafe {
        match &p.plan {
            Prepared::F32(plan) => execute_items(plan, exec, alpha, beta, item, count),
            Prepared::F64(plan) => execute_items(plan, exec, alpha, beta, item, count),
            Prepared::C32(plan) => execute_items(plan, exec, alpha, beta, item, count),
            Prepared::C64(plan) => execute_items(plan, exec, alpha, beta, item, count),
        }
    }
}

/// Execute a planned product against data:
/// `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))`.
///
/// Synchronous: it returns when the contraction is done. `exec` is zero (the
/// default serial executor), or an executor of this library; the call uses at
/// most its budget (read once, here), runs small work on the calling thread
/// without entering the pool, and runs wider work on the executor's pool through
/// `Exec::broadcast` — never through an environment variable or a pool of its own.
/// `alpha` and `beta` are read as the plan's element type, so they are pointers
/// to an `f32`, `f64`, `float _Complex` or `double _Complex` accordingly.
///
/// `status`, if non-null, is set to `0` before the work starts. Upstream's own
/// reference implementation leaves it untouched, and `status.h` declares no
/// `TAPP_create_status`, so a caller following the idiomatic
/// `TAPP_status s; execute(.., &s, ..); TAPP_destroy_status(s);` would otherwise
/// pass an uninitialised value to the destructor. Writing zero costs a branch
/// and makes that sequence defined.
///
/// `C` and `D` may be separate buffers, of different layouts. `C == D` is an
/// in-place update, accepted only when `C` maps every element exactly as `D`
/// does once the labels are matched (`TAPP_ERROR_ALIASED` otherwise, even with
/// equal base pointers); any other overlap between `C` and `D` is rejected as
/// partial, and so is any overlap of `D` with `A` or `B`. Overlap is judged on
/// the byte range each operand addresses, which is conservative for
/// interleaved layouts. With `beta == 0`, `C` is not read.
///
/// `c` may be null — upstream's `TAPP_IN_PLACE` — **only together with
/// `beta == 0`**, meaning `D` is overwritten. A null `c` with a non-zero `beta`
/// returns [`TAPP_ERROR_UNSUPPORTED`] rather than silently discarding `D`,
/// because `product.h` leaves the meaning of that combination an open question.
/// In-place accumulation is expressible: pass `D`'s own pointer as `c`.
///
/// Every check runs before `D` is written.
///
/// # Safety
/// `plan` must be live and `exec` zero or live, not destroyed during the call.
/// `alpha` and `beta` must each be valid for one read of the plan's element
/// type. The raw TAPP ABI carries no allocation lengths: `a` and `b` must be
/// readable, and `d` writable, at every offset the plan's extents and strides
/// generate (and `c` readable unless it is null), which this cannot check.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn TAPP_execute_product(
    plan: isize,
    exec: isize,
    status: *mut isize,
    alpha: *const c_void,
    a: *const c_void,
    b: *const c_void,
    beta: *const c_void,
    c: *const c_void,
    d: *mut c_void,
) -> c_int {
    ffi(move || {
        if !status.is_null() {
            // SAFETY: non-null and writable per the contract.
            unsafe { *status = 0 };
        }
        // SAFETY: zero or live per the contract.
        let p = unsafe { (plan as *const Product).as_ref() }.ok_or_else(|| null("plan"))?;
        let it = Item { a, b, c, d };
        // SAFETY: forwarded contract.
        unsafe { dispatch(p, exec, alpha, beta, &|_| it, 1) }
    })
}

/// Execute the same plan against `num_batches` sets of data pointers.
///
/// Distinct from a Hadamard (batch) index inside the plan, which the engine's
/// own loop nest handles and which shares packed panels; this shares only the
/// plan and the executor. The pointer arrays are validated first and every item
/// is checked with the same rules as [`TAPP_execute_product`] before any `D` is
/// written; an error found in an item's *data* during execution (none are
/// currently possible past validation) would leave earlier items written — there
/// is no whole-batch rollback. One `alpha` and one `beta` apply to every item;
/// `c` may be null, meaning every item has a null `C` (`beta` must be zero).
/// Items run one after another, each on the executor's threads.
///
/// # Safety
/// `plan` must be live and `num_batches` non-negative. `a`, `b` and `d` must
/// each be valid for `num_batches` pointer reads, and `c` likewise unless it is
/// null. Every pointer so obtained must satisfy [`TAPP_execute_product`]'s
/// obligations.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn TAPP_execute_batched_product(
    plan: isize,
    exec: isize,
    status: *mut isize,
    num_batches: c_int,
    alpha: *const c_void,
    a: *const *const c_void,
    b: *const *const c_void,
    beta: *const c_void,
    c: *const *const c_void,
    d: *mut *mut c_void,
) -> c_int {
    ffi(move || {
        if !status.is_null() {
            // SAFETY: non-null and writable per the contract.
            unsafe { *status = 0 };
        }
        if num_batches < 0 {
            return Err(fail(TPRIMS_ERR_SHAPE, "negative batch count"));
        }
        // SAFETY: zero or live per the contract.
        let p = unsafe { (plan as *const Product).as_ref() }.ok_or_else(|| null("plan"))?;
        if a.is_null() || b.is_null() || d.is_null() {
            return Err(null("A, B or D pointer array"));
        }
        let n = num_batches as usize;
        let item = |i: usize| {
            // SAFETY: `n` pointer reads per array per the contract.
            unsafe {
                Item {
                    a: *a.add(i),
                    b: *b.add(i),
                    c: if c.is_null() {
                        std::ptr::null()
                    } else {
                        *c.add(i)
                    },
                    d: *d.add(i),
                }
            }
        };
        // SAFETY: forwarded contract.
        unsafe { dispatch(p, exec, alpha, beta, &item, n) }
    })
}

// ----------------------------------------------------------- implementation
//                                                              identification

/// Non-standard extension: a human-readable name for this backend. Useful when
/// several TAPP implementations are linked into one benchmark driver.
#[no_mangle]
pub extern "C" fn TAPP_implementation_name() -> *const std::os::raw::c_char {
    c"tprims-rs: tprims-contract (packed BSMTC, faer, elementwise)".as_ptr()
}

/// The crate version of the *loaded library*, as a static NUL-terminated
/// `"major.minor.patch"` string. Never null; do not free the result.
///
/// Non-standard, and the counterpart to the `TAPP_VERSION_*` macros in
/// `<tapp.h>`: those describe the header a caller compiled against, this
/// describes the library it actually linked. When a distribution ships the two
/// separately — a JLL, a system package, an `LD_PRELOAD` — they can disagree,
/// and without this there is no way to find out. The [C consumer example] in the
/// repository compares them and fails if they differ.
///
/// [C consumer example]: https://github.com/lkdvos/tensorprimitives-rs/tree/main/examples/c-consumer
#[no_mangle]
pub extern "C" fn TAPP_implementation_version() -> *const std::os::raw::c_char {
    // `c"..."` cannot interpolate, so the NUL is appended by hand.
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const std::os::raw::c_char
}
