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
//! | batched product | supported |

// Every `extern "C"` entry point below inherits its safety contract from the
// TAPP specification: handles must be live values produced by the matching
// `TAPP_create_*` call, and data pointers must be valid for the extents and
// strides recorded in the tensor infos. Restating that on each of the ~20
// shims adds noise without adding information.
#![allow(clippy::missing_safety_doc)]

use std::os::raw::{c_char, c_int};

use num_complex::Complex;
use tensorcontract::{ElementOp, Layout, Operand, Plan};

// ---------------------------------------------------------------- datatypes

pub const TAPP_F32: c_int = 0;
pub const TAPP_F64: c_int = 1;
pub const TAPP_C32: c_int = 2;
pub const TAPP_C64: c_int = 3;
pub const TAPP_F16: c_int = 4;
pub const TAPP_BF16: c_int = 5;

pub const TAPP_IDENTITY: c_int = 0;
pub const TAPP_CONJUGATE: c_int = 1;

// ------------------------------------------------------------------- errors

pub const TAPP_SUCCESS: c_int = 0;
pub const TAPP_ERROR_NULL: c_int = 1;
pub const TAPP_ERROR_DATATYPE: c_int = 2;
pub const TAPP_ERROR_SHAPE: c_int = 3;
pub const TAPP_ERROR_LABELS: c_int = 4;
pub const TAPP_ERROR_UNSUPPORTED: c_int = 5;
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

fn map_err(e: tensorcontract::Error) -> c_int {
    use tensorcontract::Error::*;
    match e {
        RankMismatch { .. } | LabelCountMismatch { .. } => TAPP_ERROR_LABELS,
        ExtentMismatch { .. } | NegativeExtent { .. } => TAPP_ERROR_SHAPE,
        BroadcastIndexUnsupported { .. } => TAPP_ERROR_UNSUPPORTED,
        OutputLabelMismatch => TAPP_ERROR_LABELS,
        UnsupportedDatatype => TAPP_ERROR_DATATYPE,
        NullPointer { .. } => TAPP_ERROR_NULL,
        _ => TAPP_ERROR_INTERNAL,
    }
}

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

/// Library handle. Stateless here, but kept as a real allocation so that
/// handle validity can be checked and future state has somewhere to live.
struct HandleState {
    _private: (),
}

/// Execution resources. `nthreads == 0` means "engine default".
pub struct ExecutorState {
    pub nthreads: usize,
}

#[no_mangle]
pub unsafe extern "C" fn TAPP_create_handle(handle: *mut isize) -> c_int {
    if handle.is_null() {
        return TAPP_ERROR_NULL;
    }
    *handle = Box::into_raw(Box::new(HandleState { _private: () })) as isize;
    TAPP_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_handle(handle: isize) -> c_int {
    if handle == 0 {
        return TAPP_ERROR_NULL;
    }
    drop(Box::from_raw(handle as *mut HandleState));
    TAPP_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn TAPP_create_executor(exec: *mut isize) -> c_int {
    if exec.is_null() {
        return TAPP_ERROR_NULL;
    }
    *exec = Box::into_raw(Box::new(ExecutorState { nthreads: 0 })) as isize;
    TAPP_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_executor(exec: isize) -> c_int {
    if exec == 0 {
        return TAPP_ERROR_NULL;
    }
    drop(Box::from_raw(exec as *mut ExecutorState));
    TAPP_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_status(_status: isize) -> c_int {
    TAPP_SUCCESS
}

// ------------------------------------------------------------- tensor infos

struct TensorInfo {
    dtype: c_int,
    layout: Layout,
}

#[no_mangle]
pub unsafe extern "C" fn TAPP_create_tensor_info(
    info: *mut isize,
    dtype: c_int,
    nmode: c_int,
    extents: *const i64,
    strides: *const i64,
) -> c_int {
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
}

#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_tensor_info(info: isize) -> c_int {
    if info == 0 {
        return TAPP_ERROR_NULL;
    }
    drop(Box::from_raw(info as *mut TensorInfo));
    TAPP_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn TAPP_get_nmodes(info: isize) -> c_int {
    match (info as *const TensorInfo).as_ref() {
        Some(t) => t.layout.ndim() as c_int,
        None => -1,
    }
}

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

#[no_mangle]
pub unsafe extern "C" fn TAPP_get_extents(info: isize, extents: *mut i64) {
    if let Some(t) = (info as *const TensorInfo).as_ref() {
        if !extents.is_null() {
            std::ptr::copy_nonoverlapping(t.layout.extents.as_ptr(), extents, t.layout.ndim());
        }
    }
}

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

#[no_mangle]
pub unsafe extern "C" fn TAPP_get_strides(info: isize, strides: *mut i64) {
    if let Some(t) = (info as *const TensorInfo).as_ref() {
        if !strides.is_null() {
            std::ptr::copy_nonoverlapping(t.layout.strides.as_ptr(), strides, t.layout.ndim());
        }
    }
}

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
}

#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_tensor_product(plan: isize) -> c_int {
    if plan == 0 {
        return TAPP_ERROR_NULL;
    }
    drop(Box::from_raw(plan as *mut Product));
    TAPP_SUCCESS
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn TAPP_execute_product(
    plan: isize,
    _exec: isize,
    _status: *mut isize,
    alpha: *const std::ffi::c_void,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    c: *const std::ffi::c_void,
    d: *mut std::ffi::c_void,
) -> c_int {
    let Some(p) = (plan as *const Product).as_ref() else {
        return TAPP_ERROR_NULL;
    };
    if a.is_null() || b.is_null() || d.is_null() || alpha.is_null() || beta.is_null() {
        return TAPP_ERROR_NULL;
    }
    // `beta == 0` is the documented way to say "C is not read"; honour it even
    // if `C` is null.
    macro_rules! run {
        ($t:ty) => {{
            let al = *(alpha as *const $t);
            let be = *(beta as *const $t);
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
}

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
    c"tensorcontract-rs (planar-complex BSMTC)".as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;

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
