use std::ffi::c_void;

use tprims_contract_capi::*;
use tprims_core::dlpack::*;
use tprims_core::exec::*;
use tprims_core::status::*;
use tprims_core::tensor::tprims_tensor_borrow_raw;

struct T {
    data: Vec<f64>,
    shape: Vec<i64>,
    strides: Vec<i64>,
    dl: DLTensor,
}

fn make(data: Vec<f64>, shape: &[i64], strides: &[i64]) -> Box<T> {
    let mut t = Box::new(T {
        data,
        shape: shape.to_vec(),
        strides: strides.to_vec(),
        dl: DLTensor {
            data: std::ptr::null_mut(),
            device: DLDevice {
                device_type: kDLCPU,
                device_id: 0,
            },
            ndim: shape.len() as i32,
            dtype: DLDataType {
                code: kDLFloat,
                bits: 64,
                lanes: 1,
            },
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
    });
    t.dl.data = t.data.as_mut_ptr() as *mut c_void;
    t.dl.shape = t.shape.as_mut_ptr();
    t.dl.strides = t.strides.as_mut_ptr();
    t
}

#[test]
fn plan_create_execute_destroy() {
    // C[i, k] = sum_j A[i, j] B[j, k] with A 2x3 col-major, B 3x2 col-major.
    let mut a = make(vec![1., 2., 3., 4., 5., 6.], &[2, 3], &[1, 2]);
    let mut b = make(vec![1., 3., 5., 2., 4., 6.], &[3, 2], &[1, 3]);
    let mut c = make(vec![0.; 4], &[2, 2], &[1, 2]);
    let (lc, rc) = ([1usize], [0usize]);
    let cfg = tprims_dot_general {
        lhs_contract: lc.as_ptr(),
        rhs_contract: rc.as_ptr(),
        n_contract: 1,
        lhs_batch: std::ptr::null(),
        rhs_batch: std::ptr::null(),
        n_batch: 0,
    };
    for (strategy, sel) in [(1, 0), (2, 2)] {
        let mut plan: *mut tprims_contract_plan = std::ptr::null_mut();
        let st = unsafe {
            tprims_contract_plan_create(
                &cfg,
                tprims_tensor_borrow_raw(&mut a.dl, 0),
                tprims_tensor_borrow_raw(&mut b.dl, 0),
                tprims_tensor_borrow_raw(&mut c.dl, 0),
                0,
                0,
                strategy,
                TPRIMS_NO_MATERIALIZE,
                &mut plan,
            )
        };
        assert_eq!(st, TPRIMS_OK);
        assert_eq!(unsafe { tprims_contract_plan_selected(plan) }, sel);
        let ex = unsafe { tprims_exec_rayon_create(2, std::ptr::null()) };
        let (one, zero) = (1.0f64, 0.0f64);
        let st = unsafe {
            tprims_contract_plan_execute(
                plan,
                ex,
                &one as *const f64 as *const c_void,
                tprims_tensor_borrow_raw(&mut a.dl, 0),
                tprims_tensor_borrow_raw(&mut b.dl, 0),
                &zero as *const f64 as *const c_void,
                tprims_tensor_borrow_raw(&mut c.dl, 0),
            )
        };
        assert_eq!(st, TPRIMS_OK);
        assert_eq!(c.data, vec![35., 44., 44., 56.]);
        // Layout mismatch at execute.
        c.strides = vec![2, 1];
        c.dl.strides = c.strides.as_mut_ptr();
        let st = unsafe {
            tprims_contract_plan_execute(
                plan,
                ex,
                &one as *const f64 as *const c_void,
                tprims_tensor_borrow_raw(&mut a.dl, 0),
                tprims_tensor_borrow_raw(&mut b.dl, 0),
                &zero as *const f64 as *const c_void,
                tprims_tensor_borrow_raw(&mut c.dl, 0),
            )
        };
        assert_eq!(st, TPRIMS_ERR_SHAPE);
        c.strides = vec![1, 2];
        c.dl.strides = c.strides.as_mut_ptr();
        unsafe { tprims_contract_plan_destroy(plan) };
        assert_eq!(tprims_exec_close(ex), TPRIMS_OK);
        tprims_exec_release(ex);
    }
}

#[test]
fn create_errors_are_statuses() {
    let mut a = make(vec![0.; 6], &[2, 3], &[1, 2]);
    let mut c = make(vec![0.; 4], &[2, 2], &[1, 2]);
    let bad = [5usize];
    let cfg = tprims_dot_general {
        lhs_contract: bad.as_ptr(),
        rhs_contract: bad.as_ptr(),
        n_contract: 1,
        lhs_batch: std::ptr::null(),
        rhs_batch: std::ptr::null(),
        n_batch: 0,
    };
    let mut plan: *mut tprims_contract_plan = std::ptr::null_mut();
    let st = unsafe {
        tprims_contract_plan_create(
            &cfg,
            tprims_tensor_borrow_raw(&mut a.dl, 0),
            tprims_tensor_borrow_raw(&mut a.dl, 0),
            tprims_tensor_borrow_raw(&mut c.dl, 0),
            0,
            0,
            0,
            0,
            &mut plan,
        )
    };
    assert_eq!(st, TPRIMS_ERR_INVALID_ARGUMENT);
    assert!(plan.is_null());
    let null_cfg = tprims_dot_general {
        lhs_contract: std::ptr::null(),
        rhs_contract: std::ptr::null(),
        n_contract: 1,
        lhs_batch: std::ptr::null(),
        rhs_batch: std::ptr::null(),
        n_batch: 0,
    };
    let st = unsafe {
        tprims_contract_plan_create(
            &null_cfg,
            tprims_tensor_borrow_raw(&mut a.dl, 0),
            tprims_tensor_borrow_raw(&mut a.dl, 0),
            tprims_tensor_borrow_raw(&mut c.dl, 0),
            0,
            0,
            0,
            0,
            &mut plan,
        )
    };
    assert_eq!(st, TPRIMS_ERR_INVALID_ARGUMENT);
    assert_eq!(
        unsafe { tprims_contract_plan_selected(std::ptr::null()) },
        -1
    );
    unsafe { tprims_contract_plan_destroy(std::ptr::null_mut()) };
}
