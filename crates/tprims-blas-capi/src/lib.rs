//! C ABI of `tprims-blas`: `tprims_blas_gemm`, `tprims_blas_gemm_batched`.
//! dtype dispatch happens here only; all operands share the dtype of C.
use std::ffi::c_void;

use num_complex::Complex;
use tprims_blas::{
    gemm, gemm_batched, BatchIn, BatchStrategy, Conj, Error, MatIn, Scalar, Selected,
};
use tprims_core::exec::{exec_ref, tprims_exec};
use tprims_core::status::*;
use tprims_core::tensor::{
    dtype_of, elem_size, layout, spans_overlap, tprims_tensor, view, view_mut, DType,
};

fn map(e: Error) -> FfiError {
    let st = match e {
        Error::Rank { .. } | Error::Shape(_) => TPRIMS_ERR_SHAPE,
        Error::AliasedOutput => TPRIMS_ERR_ALIASED,
        _ => TPRIMS_ERR_INTERNAL,
    };
    FfiError::new(st, e.to_string())
}

fn conj(c: i32) -> Conj {
    if c != 0 {
        Conj::Yes
    } else {
        Conj::No
    }
}

/// # Safety
/// `p` is null or points to a `T`.
unsafe fn scalar<T: Copy>(p: *const c_void, name: &str) -> Result<T, FfiError> {
    if p.is_null() {
        return Err(FfiError::new(
            TPRIMS_ERR_INVALID_ARGUMENT,
            format!("null {name}"),
        ));
    }
    // SAFETY: non-null, a `T` per the ABI contract (possibly unaligned).
    Ok(unsafe { std::ptr::read_unaligned(p as *const T) })
}

fn check_no_overlap(
    dt: DType,
    out: &tprims_core::tensor::Layout,
    ins: &[&tprims_core::tensor::Layout],
) -> Result<(), FfiError> {
    if ins.iter().any(|l| spans_overlap(out, l, elem_size(dt))) {
        return Err(FfiError::new(
            TPRIMS_ERR_ALIASED,
            "output overlaps an input",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set.
unsafe fn gemm_t<T: Scalar>(
    exec: *mut tprims_exec,
    alpha: *const c_void,
    a: tprims_tensor,
    ca: i32,
    b: tprims_tensor,
    cb: i32,
    beta: *const c_void,
    c: tprims_tensor,
    dt: DType,
) -> Result<(), FfiError> {
    let (la, lb, lc) = (layout(&a, dt)?, layout(&b, dt)?, layout(&c, dt)?);
    check_no_overlap(dt, &lc, &[&la, &lb])?;
    // SAFETY: the caller's DLPack contract for all three operands; C does not
    // overlap A or B (checked).
    let (av, bv, mut cv) = unsafe { (view::<T>(&la)?, view::<T>(&lb)?, view_mut::<T>(&c, &lc)?) };
    let (alpha, beta) = unsafe { (scalar::<T>(alpha, "alpha")?, scalar::<T>(beta, "beta")?) };
    let (mut ai, mut bi) = (MatIn::new(&av), MatIn::new(&bv));
    ai.conj = conj(ca);
    bi.conj = conj(cb);
    exec_ref(exec)?.with(|x| gemm(x, alpha, ai, bi, beta, &mut cv).map_err(map))
}

macro_rules! dispatch {
    ($dt:expr, $f:ident, $($arg:expr),*) => {
        match $dt {
            DType::F32 => $f::<f32>($($arg),*, $dt),
            DType::F64 => $f::<f64>($($arg),*, $dt),
            DType::C32 => $f::<Complex<f32>>($($arg),*, $dt),
            DType::C64 => $f::<Complex<f64>>($($arg),*, $dt),
        }
    };
}

/// `C = alpha * op(A) * op(B) + beta * C` for rank-2 DLPack operands of one
/// dtype (that of C); `alpha` and `beta` point to scalars of that dtype;
/// `conj_*` non-zero conjugates the operand.
///
/// # Safety
///
/// The DLPack contract for every operand; `alpha` and `beta` point to one
/// element of C's dtype; `exec` is a live handle.
#[no_mangle]
#[allow(clippy::too_many_arguments)] // INVARIANT: the C ABI of GEMM.
pub unsafe extern "C" fn tprims_blas_gemm(
    exec: *mut tprims_exec,
    alpha: *const c_void,
    a: tprims_tensor,
    conj_a: i32,
    b: tprims_tensor,
    conj_b: i32,
    beta: *const c_void,
    c: tprims_tensor,
) -> tprims_status {
    ffi(|| {
        let dt = dtype_of(&c)?;
        // SAFETY: forwarded ABI contract.
        unsafe { dispatch!(dt, gemm_t, exec, alpha, a, conj_a, b, conj_b, beta, c) }
    })
}

#[allow(clippy::too_many_arguments)] // INVARIANT: the batched GEMM argument set.
unsafe fn batched_t<T: Scalar>(
    exec: *mut tprims_exec,
    alpha: *const c_void,
    a: tprims_tensor,
    ca: i32,
    b: tprims_tensor,
    cb: i32,
    beta: *const c_void,
    c: tprims_tensor,
    strategy: i32,
    selected: *mut i32,
    dt: DType,
) -> Result<(), FfiError> {
    let (la, lb, lc) = (layout(&a, dt)?, layout(&b, dt)?, layout(&c, dt)?);
    check_no_overlap(dt, &lc, &[&la, &lb])?;
    // SAFETY: as in gemm_t.
    let (av, bv, mut cv) = unsafe { (view::<T>(&la)?, view::<T>(&lb)?, view_mut::<T>(&c, &lc)?) };
    let (alpha, beta) = unsafe { (scalar::<T>(alpha, "alpha")?, scalar::<T>(beta, "beta")?) };
    let (mut ai, mut bi) = (BatchIn::new(&av), BatchIn::new(&bv));
    ai.conj = conj(ca);
    bi.conj = conj(cb);
    let strat = match strategy {
        0 => BatchStrategy::Auto,
        1 => BatchStrategy::FaerLoop,
        2 => BatchStrategy::Tblis,
        s => {
            return Err(FfiError::new(
                TPRIMS_ERR_INVALID_ARGUMENT,
                format!("unknown strategy {s}"),
            ))
        }
    };
    let sel = exec_ref(exec)?
        .with(|x| gemm_batched(x, alpha, ai, bi, beta, &mut cv, strat).map_err(map))?;
    if !selected.is_null() {
        let code = match sel {
            Selected::FaerLoop {
                outer_parallel: false,
            } => 0,
            Selected::FaerLoop {
                outer_parallel: true,
            } => 1,
            Selected::Tblis {
                outer_parallel: false,
            } => 2,
            Selected::Tblis {
                outer_parallel: true,
            } => 3,
            _ => -1,
        };
        // SAFETY: non-null, an `int` per the ABI.
        unsafe { *selected = code };
    }
    Ok(())
}

/// Batched GEMM over rank-3 operands `[rows, cols, batch]`. `strategy`:
/// 0 auto, 1 faer loop, 2 TBLIS-style. When `selected` is non-null it
/// receives 0/1 (faer loop, items serial/spread over the pool) or 2/3
/// (TBLIS-style, likewise).
///
/// # Safety
///
/// As [`tprims_blas_gemm`]; `selected` is null or points to an `int`.
#[no_mangle]
#[allow(clippy::too_many_arguments)] // INVARIANT: the C ABI of batched GEMM.
pub unsafe extern "C" fn tprims_blas_gemm_batched(
    exec: *mut tprims_exec,
    alpha: *const c_void,
    a: tprims_tensor,
    conj_a: i32,
    b: tprims_tensor,
    conj_b: i32,
    beta: *const c_void,
    c: tprims_tensor,
    strategy: i32,
    selected: *mut i32,
) -> tprims_status {
    ffi(|| {
        let dt = dtype_of(&c)?;
        // SAFETY: forwarded ABI contract.
        unsafe {
            dispatch!(dt, batched_t, exec, alpha, a, conj_a, b, conj_b, beta, c, strategy, selected)
        }
    })
}

/// Marker so `tprims_has_part("blas")` can be answered by the bundle.
pub const PART: &str = "blas";
