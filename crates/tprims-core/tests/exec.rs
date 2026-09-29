use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tprims_core::exec::*;
use tprims_core::status::*;

#[test]
fn serial_handle_and_basic_queries() {
    let e = tprims_exec_serial();
    assert_eq!(tprims_exec_num_threads(e), 1);
    assert_eq!(tprims_exec_close(e), TPRIMS_OK);
    tprims_exec_release(e);
    assert_eq!(
        tprims_exec_close(std::ptr::null_mut()),
        TPRIMS_ERR_INVALID_ARGUMENT
    );
}

#[test]
fn owned_pool_runs_budgets_and_closes_synchronously() {
    let e = unsafe { tprims_exec_rayon_create(3, std::ptr::null()) };
    assert!(!e.is_null());
    assert_eq!(tprims_exec_num_threads(e), 3);
    assert_eq!(tprims_exec_set_budget(e, 2), TPRIMS_OK);
    assert_eq!(tprims_exec_num_threads(e), 2);
    assert_eq!(tprims_exec_set_budget(e, 0), TPRIMS_ERR_INVALID_ARGUMENT);
    let h = exec_ref(e).unwrap();
    let ran = h.with(|x| Ok(x.install(2, |p| p.threads()))).unwrap();
    assert_eq!(ran, 2);

    // TLS-destructor handshake: every worker drops a guard in its TLS when it
    // exits; close must not return before all of them ran.
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            std::thread::sleep(Duration::from_millis(20));
            DROPPED.fetch_add(1, Ordering::SeqCst);
        }
    }
    thread_local! { static G: std::cell::RefCell<Option<Guard>> = const { std::cell::RefCell::new(None) }; }
    h.with(|x| {
        x.broadcast(2, &|_| G.with(|g| *g.borrow_mut() = Some(Guard)))
            .unwrap();
        Ok(())
    })
    .unwrap();
    let exec_width = 3; // pool size; broadcast wakes all, but only width 2 set guards
    let _ = exec_width;
    assert_eq!(tprims_exec_close(e), TPRIMS_OK);
    assert_eq!(
        DROPPED.load(Ordering::SeqCst),
        2,
        "close returned before worker TLS teardown"
    );
    assert_eq!(tprims_exec_close(e), TPRIMS_OK, "second close is a no-op");
    assert_eq!(h.with(|_| Ok(())).unwrap_err().status, TPRIMS_ERR_CLOSED);
    assert_eq!(tprims_exec_num_threads(e), 0);
    tprims_exec_release(e);
}

#[test]
fn busy_and_self_worker_close() {
    let e = unsafe { tprims_exec_rayon_create(2, std::ptr::null()) };
    let h = exec_ref(e).unwrap();
    let inside = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let ptr = e as usize;
    let (i2, r2) = (inside.clone(), release.clone());
    let t = std::thread::spawn(move || {
        let h = exec_ref(ptr as *mut tprims_exec).unwrap();
        h.with(|_| {
            i2.store(true, Ordering::SeqCst);
            while !r2.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            Ok(())
        })
        .unwrap();
    });
    while !inside.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }
    assert_eq!(tprims_exec_close(e), TPRIMS_BUSY);
    release.store(true, Ordering::SeqCst);
    t.join().unwrap();
    // From a worker of the pool itself.
    let st = h
        .with(|x| Ok(x.install(2, move |_| tprims_exec_close(ptr as *mut tprims_exec))))
        .unwrap();
    assert_eq!(st, TPRIMS_ERR_WOULD_DEADLOCK);
    assert_eq!(tprims_exec_close(e), TPRIMS_OK);
    tprims_exec_retain(e);
    tprims_exec_release(e);
    tprims_exec_release(e);
}

#[test]
fn last_error_and_panics() {
    let st = ffi(|| Err(FfiError::new(TPRIMS_ERR_SHAPE, "bad shape")));
    assert_eq!(st, TPRIMS_ERR_SHAPE);
    let msg = unsafe { std::ffi::CStr::from_ptr(tprims_last_error()) }
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(msg, "bad shape");
    let st = ffi(|| panic!("boom"));
    assert_eq!(st, TPRIMS_ERR_PANIC);
    let msg = unsafe { std::ffi::CStr::from_ptr(tprims_last_error()) }
        .to_str()
        .unwrap()
        .to_string();
    assert!(msg.contains("boom"));
    assert_eq!(ffi(|| Ok(())), TPRIMS_OK);
}
