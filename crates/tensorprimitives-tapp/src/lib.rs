//! TAPP C-ABI front end.
//!
//! Implements the C interface of the Tensor Algebra Processing Primitives
//! standard (arXiv:2601.07827, <https://github.com/TAPPorg/reference-implementation>)
//! on top of [`tensorcontract`], so that this engine, TBLIS and cuTENSOR are
//! drop-in swappable behind one header.
//!
//! The symbols exported here are ABI-compatible with the upstream
//! `api/include/tapp/*.h` headers: all handles are `intptr_t`, all fallible
//! calls return a `TAPP_error` (`int`, zero on success).
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
//! | mixed precision (`prec` != storage) | accepted, computed at storage precision |
//! | mixed *storage* types across operands | **rejected** (`TAPP_ERROR_DATATYPE`) — one element type per plan |
//! | batched product | supported |
//! | `TAPP_IN_PLACE` (a null `C`) | only with `beta == 0`; a non-zero `beta` is refused rather than reinterpreted |
//! | `TAPP_attr_set` / `_get` / `_clear` | exported, and refuse every key: upstream specifies none |
//!
//! Note that `TAPP_ERROR_*` beyond `TAPP_SUCCESS` are **this crate's own
//! numbering**. Upstream `error.h` is a bare `typedef int TAPP_error` with no
//! enumerators, so zero is the only value the standard fixes and a portable
//! caller must go through [`TAPP_check_success`] rather than compare codes.
//!
//! # The handle discipline, once
//!
//! Every `extern "C"` entry point here inherits the same safety contract from
//! the TAPP specification, and each one's `# Safety` section says only what is
//! specific to it on top of these three:
//!
//! * A handle argument (`isize`) must be either `0` or a live value produced by
//!   the matching `TAPP_create_*` call and not yet destroyed. `0` is rejected
//!   with `TAPP_ERROR_NULL`; a stale or foreign non-zero value is undefined
//!   behaviour, because nothing distinguishes it from a live one.
//! * Each handle may be destroyed once, and no other call may be in flight
//!   against it at the time. Nothing here is internally synchronised.
//! * Data pointers must be valid for every offset the extents and strides
//!   recorded in the tensor infos generate — see
//!   [`tensorcontract::Plan::run_raw`], which is what they reach.
//!
//! Handles are `Box::into_raw` pointers cast to `isize`, so they are *not*
//! interchangeable between processes and must not be serialised.
#![warn(missing_docs)]

use std::os::raw::{c_char, c_int};

use num_complex::Complex;
use tensorcontract::{ElementOp, Layout, Operand, Plan};

// ---------------------------------------------------------------- datatypes

// The `datatype_t` enumerators, in the order the upstream header declares them.
// The numeric values are ABI and must not be reordered — TBLIS 1.3 and 2.0 differ
// by exactly such a reordering of their own `type_t`, which produces plausible
// wrong answers rather than an error.

/// `TAPP_F32`: single-precision real. Handled by [`tensorcontract`] as `f32`.
pub const TAPP_F32: c_int = 0;
/// `TAPP_F64`: double-precision real, i.e. `f64`.
pub const TAPP_F64: c_int = 1;
/// `TAPP_C32`: single-precision complex, interleaved — layout-compatible with
/// C99 `float _Complex` and `num_complex::Complex<f32>`.
pub const TAPP_C32: c_int = 2;
/// `TAPP_C64`: double-precision complex, interleaved.
pub const TAPP_C64: c_int = 3;
/// `TAPP_F16`: accepted by the enumeration, rejected by this implementation
/// with [`TAPP_ERROR_DATATYPE`]. There is no [`tensorcontract::Element`] for it.
pub const TAPP_F16: c_int = 4;
/// `TAPP_BF16`: as [`TAPP_F16`], rejected.
pub const TAPP_BF16: c_int = 5;

/// `TAPP_IDENTITY`: read the operand as stored. The default for every operand.
pub const TAPP_IDENTITY: c_int = 0;
/// `TAPP_CONJUGATE`: complex-conjugate the operand. Free here — it is folded
/// into packing or into the write-back, never a separate pass.
pub const TAPP_CONJUGATE: c_int = 1;

// ------------------------------------------------------------------- errors

/// No error. The only value for which `TAPP_check_success` returns `true`.
pub const TAPP_SUCCESS: c_int = 0;
/// A required pointer was null, or a handle was `0` / not live.
pub const TAPP_ERROR_NULL: c_int = 1;
/// An unsupported element type, or the four operands did not agree on one.
pub const TAPP_ERROR_DATATYPE: c_int = 2;
/// Extents or strides are inconsistent: a label with two different extents, or
/// a negative extent.
pub const TAPP_ERROR_SHAPE: c_int = 3;
/// The index labels do not describe a contraction: wrong count for the rank, or
/// `C` and `D` labelled differently.
pub const TAPP_ERROR_LABELS: c_int = 4;
/// A well-formed operation this implementation declines. Currently only TAPP
/// case 5, an output-only index (broadcast), which the standard does not
/// require.
pub const TAPP_ERROR_UNSUPPORTED: c_int = 5;
/// A failure with no more specific code. Raised when a Rust panic is caught at
/// the C boundary — the entry points that allocate or run the engine convert an
/// unwind into this rather than letting it abort the caller's process — and
/// reserved so that a future [`tensorcontract::Error`] variant has a mapping
/// rather than being silently misreported. No [`tensorcontract::Error`] maps to
/// it today.
pub const TAPP_ERROR_INTERNAL: c_int = 6;

fn explain(e: c_int) -> &'static str {
    match e {
        TAPP_SUCCESS => "success",
        TAPP_ERROR_NULL => "null pointer or invalid handle",
        TAPP_ERROR_DATATYPE => "unsupported or mismatched data type",
        TAPP_ERROR_SHAPE => "inconsistent extents or strides",
        TAPP_ERROR_LABELS => "invalid index labels",
        TAPP_ERROR_UNSUPPORTED => "operation not supported by this implementation",
        _ => "internal error",
    }
}

/// Stop a Rust panic at the C boundary and report it as [`TAPP_ERROR_INTERNAL`].
///
/// A panic that reaches an `extern "C"` frame aborts the process on any
/// supported toolchain. That is memory-safe, but for a library called from a
/// long-running host — a solver, an MPI rank, an interactive session — it turns
/// a recoverable condition into the loss of everything not yet written out. The
/// engine does panic on conditions a caller can plausibly hit: `Panel::new`
/// asserts rather than returning when a packing buffer cannot be allocated, and
/// a contraction large enough to exhaust memory is an ordinary user mistake, not
/// a bug.
///
/// So the entry points that allocate or run the engine convert a panic into an
/// error code. The panic hook still runs first, so the message and backtrace
/// reach stderr exactly as before and nothing becomes less diagnosable; what
/// changes is that the caller gets the chance to clean up.
///
/// This is *not* a claim that the engine is unwind-safe in the general sense.
/// It is `AssertUnwindSafe` because the only state a caller can still reach
/// after an error return is the handles it already owned: an entry point that
/// panics has either not yet published its output handle or not yet written the
/// output buffer, so no partially-built object escapes. `TAPP_execute_product`
/// is the one exception worth naming — a panic partway through leaves `D`
/// partially written, which is already true of the documented error paths in
/// [`TAPP_execute_batched_product`].
///
/// Allocation *failure* proper (Rust's allocation error handler) still aborts
/// and is not catchable here; this covers the assertion paths above it.
fn guard(f: impl FnOnce() -> c_int) -> c_int {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(_) => TAPP_ERROR_INTERNAL,
    }
}

fn map_err(e: tensorcontract::Error) -> c_int {
    use tensorcontract::Error::*;
    match e {
        RankMismatch { .. } | LabelCountMismatch { .. } => TAPP_ERROR_LABELS,
        ExtentMismatch { .. } | NegativeExtent { .. } => TAPP_ERROR_SHAPE,
        ExtentProductOverflow { .. } => TAPP_ERROR_SHAPE,
        BroadcastIndexUnsupported { .. } => TAPP_ERROR_UNSUPPORTED,
        OutputLabelMismatch => TAPP_ERROR_LABELS,
        UnsupportedDatatype => TAPP_ERROR_DATATYPE,
        NullPointer { .. } => TAPP_ERROR_NULL,
        _ => TAPP_ERROR_INTERNAL,
    }
}

/// Whether `error` is [`TAPP_SUCCESS`].
#[no_mangle]
pub extern "C" fn TAPP_check_success(error: c_int) -> bool {
    error == TAPP_SUCCESS
}

/// Copy a description of `error` into `message` (NUL-terminated, truncated to
/// `maxlen`). Returns the length that would have been written.
///
/// # Safety
/// `message` must be valid for `maxlen` bytes, or `maxlen` must be zero.
#[no_mangle]
pub unsafe extern "C" fn TAPP_explain_error(
    error: c_int,
    maxlen: usize,
    message: *mut c_char,
) -> usize {
    let s = explain(error).as_bytes();
    if !message.is_null() && maxlen > 0 {
        let n = s.len().min(maxlen - 1);
        std::ptr::copy_nonoverlapping(s.as_ptr(), message as *mut u8, n);
        *message.add(n) = 0;
    }
    s.len()
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

/// Execution resources. `nthreads == 0` means "engine default".
///
/// Public because it is what a `TAPP_executor` handle points at, so a caller
/// holding one can set the field directly. TAPP defines no portable way to say
/// "use this many threads", and the engine's default is
/// `TENSORCONTRACT_THREADS` — see [`tensorcontract::Plan::threads`].
pub struct ExecutorState {
    /// Threads to execute with, or 0 for the engine's own default.
    ///
    /// Not yet plumbed through: [`TAPP_execute_product`] ignores its executor
    /// argument. Set `TENSORCONTRACT_THREADS` to thread the engine today.
    pub nthreads: usize,
}

/// Create a library handle.
///
/// # Safety
/// `handle` must be a valid, writable `*mut isize` (or null, which is rejected).
/// On success it receives a handle to be released with
/// [`TAPP_destroy_handle`].
#[no_mangle]
pub unsafe extern "C" fn TAPP_create_handle(handle: *mut isize) -> c_int {
    if handle.is_null() {
        return TAPP_ERROR_NULL;
    }
    *handle = Box::into_raw(Box::new(HandleState { _reserved: 0 })) as isize;
    TAPP_SUCCESS
}

/// Release a handle from [`TAPP_create_handle`].
///
/// # Safety
/// `handle` must be live and not already destroyed. See the crate docs on the
/// handle discipline.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_handle(handle: isize) -> c_int {
    if handle == 0 {
        return TAPP_ERROR_NULL;
    }
    drop(Box::from_raw(handle as *mut HandleState));
    TAPP_SUCCESS
}

/// Create an executor, i.e. an [`ExecutorState`] with `nthreads == 0`.
///
/// # Safety
/// `exec` must be a valid, writable `*mut isize` (or null, which is rejected).
/// Release with [`TAPP_destroy_executor`].
#[no_mangle]
pub unsafe extern "C" fn TAPP_create_executor(exec: *mut isize) -> c_int {
    if exec.is_null() {
        return TAPP_ERROR_NULL;
    }
    *exec = Box::into_raw(Box::new(ExecutorState { nthreads: 0 })) as isize;
    TAPP_SUCCESS
}

/// Release an executor from [`TAPP_create_executor`].
///
/// # Safety
/// `exec` must be live and not already destroyed.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_executor(exec: isize) -> c_int {
    if exec == 0 {
        return TAPP_ERROR_NULL;
    }
    drop(Box::from_raw(exec as *mut ExecutorState));
    TAPP_SUCCESS
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
pub unsafe extern "C" fn TAPP_attr_set(
    _attr: isize,
    _key: c_int,
    _value: *mut std::ffi::c_void,
) -> c_int {
    TAPP_ERROR_UNSUPPORTED
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
    value: *mut *mut std::ffi::c_void,
) -> c_int {
    if !value.is_null() {
        *value = std::ptr::null_mut();
    }
    TAPP_ERROR_UNSUPPORTED
}

/// Clear an attribute. Always fails with [`TAPP_ERROR_UNSUPPORTED`].
///
/// # Safety
/// Trivially safe; `unsafe` only to match the declared C signature.
#[no_mangle]
pub unsafe extern "C" fn TAPP_attr_clear(_attr: isize, _key: c_int) -> c_int {
    TAPP_ERROR_UNSUPPORTED
}

/// Release a status object. A no-op: execution here is synchronous, so it never
/// produces a status to release. Provided because the header declares it and a
/// conforming caller will call it.
///
/// # Safety
/// Trivially safe; `unsafe` only to match the declared C signature.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_status(_status: isize) -> c_int {
    TAPP_SUCCESS
}

// ------------------------------------------------------------- tensor infos

struct TensorInfo {
    dtype: c_int,
    layout: Layout,
}

/// Describe one tensor: element type, rank, extents and strides (in elements).
///
/// Strides may be negative or zero; extents may not be negative. A rank of 0 is
/// legal and describes a scalar, in which case `extents` and `strides` are not
/// read and may be null.
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
    guard(move || {
        if info.is_null() || nmode < 0 {
            return TAPP_ERROR_NULL;
        }
        if !matches!(dtype, TAPP_F32 | TAPP_F64 | TAPP_C32 | TAPP_C64) {
            return TAPP_ERROR_DATATYPE;
        }
        let n = nmode as usize;
        if n > 0 && (extents.is_null() || strides.is_null()) {
            return TAPP_ERROR_NULL;
        }
        let e = if n == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(extents, n).to_vec()
        };
        let s = if n == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(strides, n).to_vec()
        };
        let layout = match Layout::new(e, s) {
            Ok(l) => l,
            Err(err) => return map_err(err),
        };
        *info = Box::into_raw(Box::new(TensorInfo { dtype, layout })) as isize;
        TAPP_SUCCESS
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
    if info == 0 {
        return TAPP_ERROR_NULL;
    }
    drop(Box::from_raw(info as *mut TensorInfo));
    TAPP_SUCCESS
}

/// The rank recorded in `info`, or `-1` if the handle is `0`.
///
/// # Safety
/// `info` must be `0` or live.
#[no_mangle]
pub unsafe extern "C" fn TAPP_get_nmodes(info: isize) -> c_int {
    match (info as *const TensorInfo).as_ref() {
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
    let Some(t) = (info as *mut TensorInfo).as_mut() else {
        return TAPP_ERROR_NULL;
    };
    if nmodes < 0 {
        return TAPP_ERROR_SHAPE;
    }
    t.layout.extents.resize(nmodes as usize, 1);
    t.layout.strides.resize(nmodes as usize, 0);
    TAPP_SUCCESS
}

/// Copy `info`'s extents out. Returns nothing, as TAPP declares it, so query
/// the rank with [`TAPP_get_nmodes`] first.
///
/// # Safety
/// `info` must be `0` or live, and `extents` must be null or valid for
/// `TAPP_get_nmodes(info)` `i64` writes.
#[no_mangle]
pub unsafe extern "C" fn TAPP_get_extents(info: isize, extents: *mut i64) {
    if let Some(t) = (info as *const TensorInfo).as_ref() {
        if !extents.is_null() {
            std::ptr::copy_nonoverlapping(t.layout.extents.as_ptr(), extents, t.layout.ndim());
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
    let Some(t) = (info as *mut TensorInfo).as_mut() else {
        return TAPP_ERROR_NULL;
    };
    if extents.is_null() {
        return TAPP_ERROR_NULL;
    }
    let n = t.layout.ndim();
    t.layout
        .extents
        .copy_from_slice(std::slice::from_raw_parts(extents, n));
    TAPP_SUCCESS
}

/// Copy `info`'s strides out, in elements. As [`TAPP_get_extents`], this
/// returns nothing.
///
/// # Safety
/// `info` must be `0` or live, and `strides` must be null or valid for
/// `TAPP_get_nmodes(info)` `i64` writes.
#[no_mangle]
pub unsafe extern "C" fn TAPP_get_strides(info: isize, strides: *mut i64) {
    if let Some(t) = (info as *const TensorInfo).as_ref() {
        if !strides.is_null() {
            std::ptr::copy_nonoverlapping(t.layout.strides.as_ptr(), strides, t.layout.ndim());
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
    let Some(t) = (info as *mut TensorInfo).as_mut() else {
        return TAPP_ERROR_NULL;
    };
    if strides.is_null() {
        return TAPP_ERROR_NULL;
    }
    let n = t.layout.ndim();
    t.layout
        .strides
        .copy_from_slice(std::slice::from_raw_parts(strides, n));
    TAPP_SUCCESS
}

// ----------------------------------------------------------------- products

struct Product {
    plan: Plan,
    dtype: c_int,
}

fn op_of(op: c_int) -> ElementOp {
    if op == TAPP_CONJUGATE {
        ElementOp::Conjugate
    } else {
        ElementOp::Identity
    }
}

/// Plan `D = alpha * op_A(A) * op_B(B) + beta * op_C(C)` from four tensor infos
/// and their index labels.
///
/// All the work that depends only on shapes, strides and labels happens here —
/// index classification, folding, and building the scatter vectors — so this is
/// the call to hoist out of a loop. `_prec` is accepted and ignored: TAPP allows
/// a compute precision distinct from storage, and this engine computes at the
/// storage precision, which the crate docs record as the deviation it is.
///
/// The four infos must agree on the element type; TAPP permits mixed storage
/// types and this engine does not. `C` is required, unlike
/// [`tensorcontract::Plan::new`], because the C signature has no way to omit
/// it — pass `beta == 0` at execution time to have it ignored, or pass `D`'s
/// info for it.
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
    _handle: isize,
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
    _prec: c_int,
) -> c_int {
    guard(move || {
        if plan_out.is_null() {
            return TAPP_ERROR_NULL;
        }
        let (Some(ta), Some(tb), Some(tc), Some(td)) = (
            (a as *const TensorInfo).as_ref(),
            (b as *const TensorInfo).as_ref(),
            (c as *const TensorInfo).as_ref(),
            (d as *const TensorInfo).as_ref(),
        ) else {
            return TAPP_ERROR_NULL;
        };

        // TAPP allows mixed storage types; this engine computes at a single
        // element type, so require all four to agree.
        if ta.dtype != tb.dtype || ta.dtype != tc.dtype || ta.dtype != td.dtype {
            return TAPP_ERROR_DATATYPE;
        }

        let labels = |p: *const i64, t: &TensorInfo| -> Option<Vec<i64>> {
            let n = t.layout.ndim();
            if n == 0 {
                Some(Vec::new())
            } else if p.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(p, n).to_vec())
            }
        };
        let (Some(la), Some(lb), Some(lc), Some(ld)) = (
            labels(idx_a, ta),
            labels(idx_b, tb),
            labels(idx_c, tc),
            labels(idx_d, td),
        ) else {
            return TAPP_ERROR_NULL;
        };

        let plan = Plan::new(
            Operand {
                layout: &ta.layout,
                idx: &la,
                op: op_of(op_a),
            },
            Operand {
                layout: &tb.layout,
                idx: &lb,
                op: op_of(op_b),
            },
            Some(Operand {
                layout: &tc.layout,
                idx: &lc,
                op: op_of(op_c),
            }),
            Operand {
                layout: &td.layout,
                idx: &ld,
                op: op_of(op_d),
            },
        );
        match plan {
            Ok(plan) => {
                *plan_out = Box::into_raw(Box::new(Product {
                    plan,
                    dtype: ta.dtype,
                })) as isize;
                TAPP_SUCCESS
            }
            Err(e) => map_err(e),
        }
    })
}

/// Release a product from [`TAPP_create_tensor_product`].
///
/// # Safety
/// `plan` must be live and not already destroyed.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_tensor_product(plan: isize) -> c_int {
    if plan == 0 {
        return TAPP_ERROR_NULL;
    }
    drop(Box::from_raw(plan as *mut Product));
    TAPP_SUCCESS
}

/// Execute a planned product against data.
///
/// Synchronous: it returns when the contraction is done, and `_exec` is ignored
/// (see [`ExecutorState::nthreads`]). `alpha` and `beta` are read as the plan's
/// element type, so they are pointers to an `f32`, `f64`, `float _Complex` or
/// `double _Complex` accordingly.
///
/// `status`, if non-null, is set to `0` before the work starts. Upstream's own
/// reference implementation leaves it untouched, and `status.h` declares no
/// `TAPP_create_status`, so a caller following the idiomatic
/// `TAPP_status s; execute(.., &s, ..); TAPP_destroy_status(s);` would otherwise
/// pass an uninitialised value to the destructor. Writing zero costs a branch
/// and makes that sequence defined.
///
/// `c` may be null — upstream's `TAPP_IN_PLACE` — **only together with
/// `beta == 0`**, meaning `D` is overwritten. A null `c` with a non-zero `beta`
/// returns [`TAPP_ERROR_UNSUPPORTED`] rather than silently discarding `D`,
/// because `product.h` leaves the meaning of that combination an open question
/// and the constant's name suggests the opposite of what this engine would do.
/// In-place accumulation is expressible today: pass `D`'s own pointer as `c`.
///
/// # Safety
/// `plan` must be live. `alpha` and `beta` must each be valid for one read of
/// the plan's element type. `a` and `b` must be readable, and `d` writable, at
/// every offset the plan generates — the same obligation as
/// [`tensorcontract::Plan::run_raw`], which this forwards to unchanged, and it
/// is *not* checked here. `d` must not alias `a` or `b`. `c` must satisfy the
/// same read obligation unless it is null or `beta` is zero.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn TAPP_execute_product(
    plan: isize,
    _exec: isize,
    status: *mut isize,
    alpha: *const std::ffi::c_void,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    c: *const std::ffi::c_void,
    d: *mut std::ffi::c_void,
) -> c_int {
    guard(move || {
        if !status.is_null() {
            *status = 0;
        }
        let Some(p) = (plan as *const Product).as_ref() else {
            return TAPP_ERROR_NULL;
        };
        if a.is_null() || b.is_null() || d.is_null() || alpha.is_null() || beta.is_null() {
            return TAPP_ERROR_NULL;
        }
        // A null `C` is upstream's `TAPP_IN_PLACE`, whose meaning `product.h` leaves
        // as an open `//TODO`. Reading it as "beta is zero" — which this did — turns
        // `beta = 1, C = TAPP_IN_PLACE`, i.e. the request a caller would write for
        // `D += alpha*A*B`, into a silent overwrite of `D`. Rather than guess which
        // way upstream will settle it, reject the ambiguous combination and keep the
        // unambiguous one: a null `C` with `beta == 0` overwrites `D`, and in-place
        // accumulation is expressible today by passing `D`'s own pointer for `C`.
        // Turning a wrong answer into a diagnosable error costs a caller nothing it
        // can already express.
        macro_rules! run {
            ($t:ty) => {{
                let al = *(alpha as *const $t);
                let be = *(beta as *const $t);
                if c.is_null() && be != <$t as tensorcontract::Element>::zero() {
                    return TAPP_ERROR_UNSUPPORTED;
                }
                let cp = if c.is_null() {
                    d as *const $t
                } else {
                    c as *const $t
                };
                let be = if c.is_null() {
                    <$t as tensorcontract::Element>::zero()
                } else {
                    be
                };
                p.plan
                    .run_raw::<$t>(al, a as *const $t, b as *const $t, be, cp, d as *mut $t);
                TAPP_SUCCESS
            }};
        }
        match p.dtype {
            TAPP_F32 => run!(f32),
            TAPP_F64 => run!(f64),
            TAPP_C32 => run!(Complex<f32>),
            TAPP_C64 => run!(Complex<f64>),
            _ => TAPP_ERROR_DATATYPE,
        }
    })
}

/// Execute the same plan against `num_batches` sets of data pointers.
///
/// A loop over [`TAPP_execute_product`], stopping at the first error and
/// returning it, so a partial batch may already have been written when it does.
/// One `alpha` and one `beta` apply to every batch. This is not the same thing
/// as a Hadamard (batch) index inside the plan, which is handled by the engine's
/// own loop nest and shares packed panels; this shares only the plan.
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
    alpha: *const std::ffi::c_void,
    a: *const *const std::ffi::c_void,
    b: *const *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    c: *const *const std::ffi::c_void,
    d: *mut *mut std::ffi::c_void,
) -> c_int {
    if num_batches < 0 {
        return TAPP_ERROR_SHAPE;
    }
    if a.is_null() || b.is_null() || d.is_null() {
        return TAPP_ERROR_NULL;
    }
    for i in 0..num_batches as usize {
        let ci = if c.is_null() {
            std::ptr::null()
        } else {
            *c.add(i)
        };
        let e = TAPP_execute_product(
            plan,
            exec,
            status,
            alpha,
            *a.add(i),
            *b.add(i),
            beta,
            ci,
            *d.add(i),
        );
        if e != TAPP_SUCCESS {
            return e;
        }
    }
    TAPP_SUCCESS
}

// ----------------------------------------------------------- implementation
//                                                              identification

/// Non-standard extension: a human-readable name for this backend. Useful when
/// several TAPP implementations are linked into one benchmark driver.
#[no_mangle]
pub extern "C" fn TAPP_implementation_name() -> *const c_char {
    c"tensorprimitives-rs: tensorcontract (planar-complex BSMTC)".as_ptr()
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
pub extern "C" fn TAPP_implementation_version() -> *const c_char {
    // `c"..."` cannot interpolate, so the NUL is appended by hand.
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A panic inside a guarded entry point becomes [`TAPP_ERROR_INTERNAL`]
    /// instead of aborting the caller's process.
    ///
    /// This tests the mechanism rather than an end-to-end engine panic, and
    /// deliberately so: the conditions that make the engine panic are
    /// allocation failures, which cannot be induced from a test without a fault
    /// injection point that would itself have to be maintained. What is checked
    /// here is the part that could silently regress — that `guard` is a real
    /// unwind boundary and returns the documented code — while the *placement*
    /// of the guards is checked by reading them at the three call sites.
    ///
    /// The panic hook is silenced for the duration so a passing run does not
    /// print a backtrace that looks like a failure.
    #[test]
    fn a_panic_is_caught_and_reported_rather_than_aborting() {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = guard(|| panic!("induced"));
        let passed_through = guard(|| TAPP_ERROR_SHAPE);
        std::panic::set_hook(prev);

        assert_eq!(caught, TAPP_ERROR_INTERNAL);
        assert_eq!(passed_through, TAPP_ERROR_SHAPE, "guard is transparent");
    }

    /// Drive a 2x3 * 3x2 complex contraction entirely through the C ABI.
    #[test]
    fn c_abi_roundtrip_complex() {
        unsafe {
            let mut handle = 0isize;
            assert_eq!(TAPP_create_handle(&mut handle), TAPP_SUCCESS);

            let ea = [2i64, 3];
            let sa = [1i64, 2];
            let eb = [3i64, 2];
            let sb = [1i64, 3];
            let ed = [2i64, 2];
            let sd = [1i64, 2];

            let mut ia = 0isize;
            let mut ib = 0isize;
            let mut id = 0isize;
            assert_eq!(
                TAPP_create_tensor_info(&mut ia, TAPP_C64, 2, ea.as_ptr(), sa.as_ptr()),
                TAPP_SUCCESS
            );
            assert_eq!(
                TAPP_create_tensor_info(&mut ib, TAPP_C64, 2, eb.as_ptr(), sb.as_ptr()),
                TAPP_SUCCESS
            );
            assert_eq!(
                TAPP_create_tensor_info(&mut id, TAPP_C64, 2, ed.as_ptr(), sd.as_ptr()),
                TAPP_SUCCESS
            );

            let la = [0i64, 2];
            let lb = [2i64, 1];
            let ld = [0i64, 1];
            let mut plan = 0isize;
            assert_eq!(
                TAPP_create_tensor_product(
                    &mut plan,
                    handle,
                    TAPP_IDENTITY,
                    ia,
                    la.as_ptr(),
                    TAPP_IDENTITY,
                    ib,
                    lb.as_ptr(),
                    TAPP_IDENTITY,
                    id,
                    ld.as_ptr(),
                    TAPP_IDENTITY,
                    id,
                    ld.as_ptr(),
                    -1,
                ),
                TAPP_SUCCESS
            );

            let av: Vec<Complex<f64>> = (0..6)
                .map(|i| Complex::new(i as f64, 1.0 + i as f64))
                .collect();
            let bv: Vec<Complex<f64>> = (0..6)
                .map(|i| Complex::new(2.0 * i as f64, -(i as f64)))
                .collect();
            let mut dv = vec![Complex::new(0.0, 0.0); 4];
            let alpha = Complex::new(1.0f64, 0.0);
            let beta = Complex::new(0.0f64, 0.0);

            assert_eq!(
                TAPP_execute_product(
                    plan,
                    0,
                    std::ptr::null_mut(),
                    &alpha as *const _ as *const _,
                    av.as_ptr() as *const _,
                    bv.as_ptr() as *const _,
                    &beta as *const _ as *const _,
                    std::ptr::null(),
                    dv.as_mut_ptr() as *mut _,
                ),
                TAPP_SUCCESS
            );

            // Reference: A is 2x3 col-major, B is 3x2 col-major.
            for i in 0..2usize {
                for j in 0..2usize {
                    let mut want = Complex::new(0.0f64, 0.0);
                    for k in 0..3usize {
                        want += av[i + 2 * k] * bv[k + 3 * j];
                    }
                    let got = dv[i + 2 * j];
                    assert!((got - want).norm() < 1e-12, "({i},{j}): {got} vs {want}");
                }
            }

            assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_tensor_info(ia), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_tensor_info(ib), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_tensor_info(id), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
        }
    }
}
