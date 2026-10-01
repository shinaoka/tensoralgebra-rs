use std::ffi::c_void;

use tprims_blas_capi::{tprims_blas_gemm, tprims_blas_gemm_batched};
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
fn gemm_through_the_abi_writes_the_callers_memory() {
    // A: 2x3 column-major, B: 3x2 row-major (strides [2, 1]), C: 2x2 column-major.
    let mut a = make(vec![1., 2., 3., 4., 5., 6.], &[2, 3], &[1, 2]);
    let mut b = make(vec![1., 2., 3., 4., 5., 6.], &[3, 2], &[2, 1]);
    let mut c = make(vec![f64::NAN; 4], &[2, 2], &[1, 2]);
    let cptr = c.data.as_ptr();
    let (alpha, beta) = (1.0f64, 0.0f64);
    let mut e = 0;
    assert_eq!(
        unsafe { tprims_tapp_executor_create_rayon(&mut e, 2, std::ptr::null()) },
        TPRIMS_OK
    );
    let mut serial = 0;
    assert_eq!(unsafe { TAPP_create_executor(&mut serial) }, TPRIMS_OK);
    // Executor 0 is the default serial executor.
    for exec in [0, serial, e] {
        let st = unsafe {
            tprims_blas_gemm(
                exec,
                &alpha as *const f64 as *const c_void,
                tprims_tensor_borrow_raw(&mut a.dl, 0),
                0,
                tprims_tensor_borrow_raw(&mut b.dl, 0),
                0,
                &beta as *const f64 as *const c_void,
                tprims_tensor_borrow_raw(&mut c.dl, 0),
            )
        };
        assert_eq!(st, TPRIMS_OK);
        // [[1,3,5],[2,4,6]] x [[1,2],[3,4],[5,6]] = [[35,44],[44,56]] (column-major)
        assert_eq!(c.data, vec![35., 44., 44., 56.]);
        assert_eq!(c.data.as_ptr(), cptr);
    }
    assert_eq!(unsafe { TAPP_destroy_executor(e) }, TPRIMS_OK);
    assert_eq!(unsafe { TAPP_destroy_executor(serial) }, TPRIMS_OK);
}

#[test]
fn abi_errors_are_statuses() {
    let mut a = make(vec![0.; 4], &[2, 2], &[1, 2]);
    let mut c = make(vec![0.; 4], &[2, 2], &[1, 2]);
    let one = 1.0f64;
    let ex = 0;
    let call = |a: &mut DLTensor, c: &mut DLTensor, cflags: u64, alpha: *const c_void| unsafe {
        tprims_blas_gemm(
            ex,
            alpha,
            tprims_tensor_borrow_raw(a, 0),
            0,
            tprims_tensor_borrow_raw(a, 0),
            0,
            &one as *const f64 as *const c_void,
            tprims_tensor_borrow_raw(c, cflags),
        )
    };
    let al = &one as *const f64 as *const c_void;
    assert_eq!(
        call(&mut a.dl, &mut c.dl, DLPACK_FLAG_BITMASK_READ_ONLY, al),
        TPRIMS_ERR_READ_ONLY
    );
    assert_eq!(
        call(&mut a.dl, &mut c.dl, 0, std::ptr::null()),
        TPRIMS_ERR_INVALID_ARGUMENT
    );
    // Output aliasing an input.
    let a2: *mut DLTensor = &mut a.dl;
    assert_eq!(
        call(unsafe { &mut *a2 }, unsafe { &mut *a2 }, 0, al),
        TPRIMS_ERR_ALIASED
    );
    // dtype mismatch.
    c.dl.dtype.bits = 32;
    assert_eq!(call(&mut a.dl, &mut c.dl, 0, al), TPRIMS_ERR_DTYPE);
    c.dl.dtype.bits = 64;
}

#[test]
fn batched_gemm_reports_what_ran() {
    let (n, nb) = (2i64, 3i64);
    let data: Vec<f64> = (0..(n * n * nb)).map(|x| x as f64).collect();
    let mut a = make(data.clone(), &[n, n, nb], &[1, n, n * n]);
    let mut b = make(data, &[n, n, nb], &[1, n, n * n]);
    let mut c = make(vec![0.; (n * n * nb) as usize], &[n, n, nb], &[1, n, n * n]);
    let (one, zero) = (1.0f64, 0.0f64);
    let ex = 0;
    for (strategy, want) in [(1, 0), (2, 2)] {
        let mut sel = -9;
        let st = unsafe {
            tprims_blas_gemm_batched(
                ex,
                &one as *const f64 as *const c_void,
                tprims_tensor_borrow_raw(&mut a.dl, 0),
                0,
                tprims_tensor_borrow_raw(&mut b.dl, 0),
                0,
                &zero as *const f64 as *const c_void,
                tprims_tensor_borrow_raw(&mut c.dl, 0),
                strategy,
                &mut sel,
            )
        };
        assert_eq!((st, sel), (TPRIMS_OK, want));
        // item 2: [[8,10],[9,11]] squared (column-major [8,9,10,11])
        assert_eq!(&c.data[8..12], &[154., 171., 190., 211.]);
    }
}
