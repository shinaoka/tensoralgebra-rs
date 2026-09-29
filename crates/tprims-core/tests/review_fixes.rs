//! Regressions from the Phase 1f review.
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tprims_core::dlpack::*;
use tprims_core::exec::*;
use tprims_core::status::*;
use tprims_core::tensor::{
    layout, tprims_tensor_borrow_raw, tprims_tensor_borrow_versioned, DType,
};

#[test]
fn misaligned_data_is_rejected_before_any_view() {
    let mut buf = vec![0u8; 8 * 5 + 1];
    let mut shape = [4i64];
    let mut t = DLTensor {
        data: unsafe { buf.as_mut_ptr().add(1) } as *mut c_void, // not 8-aligned
        device: DLDevice {
            device_type: kDLCPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: DLDataType {
            code: kDLFloat,
            bits: 64,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    };
    let e = layout(&tprims_tensor_borrow_raw(&mut t, 0), DType::F64).unwrap_err();
    assert_eq!(e.status, TPRIMS_ERR_INVALID_ARGUMENT, "{}", e.message);
}

#[test]
fn unsupported_dlpack_major_version_is_not_borrowed() {
    let mut shape = [1i64];
    let mut x = [0.0f64];
    let mut m = DLManagedTensorVersioned {
        version: DLPackVersion { major: 2, minor: 0 },
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
        flags: 0,
        dl_tensor: DLTensor {
            data: x.as_mut_ptr() as *mut c_void,
            device: DLDevice {
                device_type: kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: DLDataType {
                code: kDLFloat,
                bits: 64,
                lanes: 1,
            },
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
    };
    assert!(unsafe { tprims_tensor_borrow_versioned(&mut m) }
        .view
        .is_null());
    m.version.major = 1;
    assert!(!unsafe { tprims_tensor_borrow_versioned(&mut m) }
        .view
        .is_null());
}

#[test]
fn concurrent_closes_both_return_after_workers_are_joined() {
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            std::thread::sleep(Duration::from_millis(100));
            DROPPED.fetch_add(1, Ordering::SeqCst);
        }
    }
    thread_local! { static G: std::cell::RefCell<Option<Guard>> = const { std::cell::RefCell::new(None) }; }
    let e = unsafe { tprims_exec_rayon_create(2, std::ptr::null()) };
    exec_ref(e)
        .unwrap()
        .with(|x| {
            x.broadcast(2, &|_| G.with(|g| *g.borrow_mut() = Some(Guard)))
                .unwrap();
            Ok(())
        })
        .unwrap();
    let p = e as usize;
    let closers: Vec<_> = (0..2)
        .map(|_| {
            std::thread::spawn(move || {
                // A close racing another close may see BUSY; it retries. No
                // close may report OK before the workers are joined.
                loop {
                    let st = tprims_exec_close(p as *mut tprims_exec);
                    if st != TPRIMS_BUSY {
                        return (st, DROPPED.load(Ordering::SeqCst));
                    }
                    std::thread::yield_now();
                }
            })
        })
        .collect();
    for c in closers {
        let (st, dropped) = c.join().unwrap();
        assert_eq!(st, TPRIMS_OK);
        assert_eq!(
            dropped, 2,
            "a close returned before the workers were joined"
        );
    }
    tprims_exec_release(e);
}
