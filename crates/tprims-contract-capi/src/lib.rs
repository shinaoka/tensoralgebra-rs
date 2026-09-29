//! C ABI of `tprims-contract`: contraction plans created once for fixed
//! layouts and executed many times.
#![allow(non_camel_case_types)]

use std::ffi::c_void;

use num_complex::Complex;
use tprims_blas::{Conj, Scalar};
use tprims_contract::{ContractPlan, DotGeneral, Error, Flags, Selected, Strategy};
use tprims_core::exec::{exec_ref, tprims_exec};
use tprims_core::status::*;
use tprims_core::tensor::{
    dtype_of, elem_size, layout, spans_overlap, tprims_tensor, view, view_mut, DType,
};

/// `TPRIMS_NO_MATERIALIZE`: refuse plans that would copy an operand.
pub const TPRIMS_NO_MATERIALIZE: u32 = 1;

/// Axis configuration (`dot_general`): `n_contract` pairs and `n_batch`
/// pairs; arrays may be null when their count is zero.
#[repr(C)]
#[derive(Debug)]
pub struct tprims_dot_general {
    /// Contracted axes of A.
    pub lhs_contract: *const usize,
    /// Contracted axes of B.
    pub rhs_contract: *const usize,
    /// Number of contracted pairs.
    pub n_contract: usize,
    /// Batch axes of A.
    pub lhs_batch: *const usize,
    /// Batch axes of B.
    pub rhs_batch: *const usize,
    /// Number of batch pairs.
    pub n_batch: usize,
}

enum Any {
    F32(ContractPlan<f32>),
    F64(ContractPlan<f64>),
    C32(ContractPlan<Complex<f32>>),
    C64(ContractPlan<Complex<f64>>),
}

/// A contraction plan (opaque in C).
pub struct tprims_contract_plan {
    plan: Any,
    dtype: DType,
}

fn map(e: Error) -> FfiError {
    let st = match e {
        Error::Config(_) => TPRIMS_ERR_INVALID_ARGUMENT,
        Error::Shape(_) | Error::LayoutMismatch(_) => TPRIMS_ERR_SHAPE,
        Error::AliasedOutput => TPRIMS_ERR_ALIASED,
        Error::WouldMaterialize { .. } => TPRIMS_ERR_WOULD_MATERIALIZE,
        _ => TPRIMS_ERR_INTERNAL,
    };
    FfiError::new(st, e.to_string())
}

/// # Safety
/// `p` is null (then `n` must be 0) or points to `n` values.
unsafe fn arr<'a>(p: *const usize, n: usize, name: &str) -> Result<&'a [usize], FfiError> {
    if n == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(FfiError::new(
            TPRIMS_ERR_INVALID_ARGUMENT,
            format!("null {name}"),
        ));
    }
    // SAFETY: non-null, `n` entries per the ABI.
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}

fn conj(c: i32) -> Conj {
    if c != 0 {
        Conj::Yes
    } else {
        Conj::No
    }
}

/// Create a plan for fixed layouts of A, B and C: shapes, strides, dtype and
/// device are used; data pointers are only validated (non-null for
/// non-empty tensors, aligned), not read. `strategy`: 0 auto,
/// 1 permute plus batched GEMM, 2 TBLIS-style. `flags`: `TPRIMS_NO_MATERIALIZE`.
///
/// # Safety
///
/// `cfg` points to a valid configuration; the tensors satisfy the DLPack
/// contract for their metadata; `out` points to writable storage.
#[no_mangle]
#[allow(clippy::too_many_arguments)] // INVARIANT: the C ABI of plan creation.
pub unsafe extern "C" fn tprims_contract_plan_create(
    cfg: *const tprims_dot_general,
    a: tprims_tensor,
    b: tprims_tensor,
    c: tprims_tensor,
    conj_a: i32,
    conj_b: i32,
    strategy: i32,
    flags: u32,
    out: *mut *mut tprims_contract_plan,
) -> tprims_status {
    ffi(|| {
        if cfg.is_null() || out.is_null() {
            return Err(FfiError::new(
                TPRIMS_ERR_INVALID_ARGUMENT,
                "null cfg or out",
            ));
        }
        // SAFETY: non-null per the contract.
        let g = unsafe { &*cfg };
        // SAFETY: arrays per the contract.
        let dg = unsafe {
            DotGeneral::new(
                arr(g.lhs_contract, g.n_contract, "lhs_contract")?,
                arr(g.rhs_contract, g.n_contract, "rhs_contract")?,
                arr(g.lhs_batch, g.n_batch, "lhs_batch")?,
                arr(g.rhs_batch, g.n_batch, "rhs_batch")?,
            )
        };
        let dt = dtype_of(&c)?;
        let (la, lb, lc) = (layout(&a, dt)?, layout(&b, dt)?, layout(&c, dt)?);
        let strat = match strategy {
            0 => Strategy::Auto,
            1 => Strategy::PermuteGemm,
            2 => Strategy::Tblis,
            s => {
                return Err(FfiError::new(
                    TPRIMS_ERR_INVALID_ARGUMENT,
                    format!("unknown strategy {s}"),
                ))
            }
        };
        let fl = Flags {
            no_materialize: flags & TPRIMS_NO_MATERIALIZE != 0,
        };
        let cj = (conj(conj_a), conj(conj_b));
        let l = |x: &tprims_core::tensor::Layout| (x.dims.clone(), x.strides.clone());
        let ((ad, as_), (bd, bs), (cd, cs)) = (l(&la), l(&lb), l(&lc));
        macro_rules! mk {
            ($t:ty) => {
                ContractPlan::<$t>::new(&dg, (&ad, &as_), (&bd, &bs), (&cd, &cs), cj, strat, fl)
                    .map_err(map)?
            };
        }
        let plan = match dt {
            DType::F32 => Any::F32(mk!(f32)),
            DType::F64 => Any::F64(mk!(f64)),
            DType::C32 => Any::C32(mk!(Complex<f32>)),
            DType::C64 => Any::C64(mk!(Complex<f64>)),
        };
        // SAFETY: `out` is writable per the contract.
        unsafe { *out = Box::into_raw(Box::new(tprims_contract_plan { plan, dtype: dt })) };
        Ok(())
    })
}

unsafe fn exec_t<T: Scalar>(
    p: &ContractPlan<T>,
    exec: *mut tprims_exec,
    alpha: *const c_void,
    a: tprims_tensor,
    b: tprims_tensor,
    beta: *const c_void,
    c: tprims_tensor,
    dt: DType,
) -> Result<(), FfiError> {
    let (la, lb, lc) = (layout(&a, dt)?, layout(&b, dt)?, layout(&c, dt)?);
    if spans_overlap(&lc, &la, elem_size(dt)) || spans_overlap(&lc, &lb, elem_size(dt)) {
        return Err(FfiError::new(
            TPRIMS_ERR_ALIASED,
            "output overlaps an input",
        ));
    }
    if alpha.is_null() || beta.is_null() {
        return Err(FfiError::new(
            TPRIMS_ERR_INVALID_ARGUMENT,
            "null alpha or beta",
        ));
    }
    // SAFETY: the DLPack contract; C does not overlap A or B; scalars point
    // to one element of the dtype.
    unsafe {
        let (av, bv, mut cv) = (view::<T>(&la)?, view::<T>(&lb)?, view_mut::<T>(&c, &lc)?);
        let (al, be) = (
            std::ptr::read_unaligned(alpha as *const T),
            std::ptr::read_unaligned(beta as *const T),
        );
        exec_ref(exec)?.with(|x| p.execute(x, al, &av, &bv, be, &mut cv).map_err(map))
    }
}

/// Execute a plan: `C = alpha * contract(op(A), op(B)) + beta * C`. The
/// tensors must have exactly the planned shapes, strides and dtype.
///
/// # Safety
///
/// `plan` and `exec` are live handles; DLPack contract for the tensors;
/// `alpha`/`beta` point to one element of the plan's dtype.
#[no_mangle]
pub unsafe extern "C" fn tprims_contract_plan_execute(
    plan: *const tprims_contract_plan,
    exec: *mut tprims_exec,
    alpha: *const c_void,
    a: tprims_tensor,
    b: tprims_tensor,
    beta: *const c_void,
    c: tprims_tensor,
) -> tprims_status {
    ffi(|| {
        if plan.is_null() {
            return Err(FfiError::new(TPRIMS_ERR_INVALID_ARGUMENT, "null plan"));
        }
        // SAFETY: live handle per the contract.
        let p = unsafe { &*plan };
        let dt = p.dtype;
        // SAFETY: forwarded.
        unsafe {
            match &p.plan {
                Any::F32(x) => exec_t(x, exec, alpha, a, b, beta, c, dt),
                Any::F64(x) => exec_t(x, exec, alpha, a, b, beta, c, dt),
                Any::C32(x) => exec_t(x, exec, alpha, a, b, beta, c, dt),
                Any::C64(x) => exec_t(x, exec, alpha, a, b, beta, c, dt),
            }
        }
    })
}

/// What the plan runs: 0 permute+GEMM without copies, 1 with copies,
/// 2 TBLIS-style, 3 elementwise; -1 for null.
///
/// # Safety
///
/// `plan` is null or a live plan.
#[no_mangle]
pub unsafe extern "C" fn tprims_contract_plan_selected(plan: *const tprims_contract_plan) -> i32 {
    if plan.is_null() {
        return -1;
    }
    // SAFETY: live handle per the contract.
    let p = unsafe { &*plan };
    let s = match &p.plan {
        Any::F32(x) => x.selected(),
        Any::F64(x) => x.selected(),
        Any::C32(x) => x.selected(),
        Any::C64(x) => x.selected(),
    };
    match s {
        Selected::PermuteGemm { materialized } if materialized.iter().all(|m| !m) => 0,
        Selected::PermuteGemm { .. } => 1,
        Selected::Tblis => 2,
        Selected::Elementwise => 3,
        _ => -1,
    }
}

/// Free a plan (null is a no-op).
///
/// # Safety
///
/// `plan` is null or a live plan not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn tprims_contract_plan_destroy(plan: *mut tprims_contract_plan) {
    if !plan.is_null() {
        // SAFETY: created by `tprims_contract_plan_create`.
        drop(unsafe { Box::from_raw(plan) });
    }
}

/// Marker for `tprims_has_part("contract")`.
pub const PART: &str = "contract";
