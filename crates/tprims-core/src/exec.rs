//! Executor handles: serial, or a Rayon pool created (and joined) by tprims
//! for C hosts without one.
#![allow(non_camel_case_types)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::thread::JoinHandle;

use ::tprims_exec::{Exec, Pool};

use crate::status::*;

/// Options for [`tprims_exec_rayon_create`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct tprims_rayon_opts {
    /// Worker stack size in bytes; 0 = 16 MiB (provider kernels recurse
    /// deeply, see tenferro-rs's CPU threading contract).
    pub stack_size: usize,
}

enum Kind {
    Serial,
    Pool {
        pool: RwLock<Option<Pool<'static>>>,
        joins: Mutex<Vec<JoinHandle<()>>>,
        budget: AtomicUsize,
    },
}

/// An executor handle (opaque in C).
pub struct tprims_exec {
    refs: AtomicUsize,
    kind: Kind,
}

impl tprims_exec {
    /// Run `f` with an [`Exec`] for this handle, holding it in flight.
    ///
    /// # Errors
    ///
    /// `TPRIMS_ERR_CLOSED` after close.
    pub fn with<R>(&self, f: impl FnOnce(&Exec<'_>) -> Result<R, FfiError>) -> Result<R, FfiError> {
        match &self.kind {
            Kind::Serial => f(&Exec::serial()),
            Kind::Pool { pool, budget, .. } => {
                let g = pool
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let p = g
                    .as_ref()
                    .ok_or_else(|| FfiError::new(TPRIMS_ERR_CLOSED, "executor is closed"))?;
                let exec = Exec::rayon(p)
                    .with_budget(budget.load(Ordering::Relaxed))
                    .map_err(|e| FfiError::new(TPRIMS_ERR_INVALID_ARGUMENT, e.to_string()))?;
                f(&exec)
            }
        }
    }

    fn close(&self) -> tprims_status {
        match &self.kind {
            Kind::Serial => TPRIMS_OK,
            Kind::Pool { pool, joins, .. } => {
                // A worker of this pool cannot join its own pool.
                if let Ok(g) = pool.try_read() {
                    if g.as_ref().is_some_and(|p| Exec::rayon(p).is_worker()) {
                        return TPRIMS_ERR_WOULD_DEADLOCK;
                    }
                }
                let mut g = match pool.try_write() {
                    Ok(g) => g,
                    Err(_) => return TPRIMS_BUSY,
                };
                let Some(p) = g.take() else {
                    return TPRIMS_OK; // already closed
                };
                drop(p.into_owned()); // starts worker shutdown
                drop(g);
                let handles: Vec<_> = std::mem::take(
                    &mut *joins
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                );
                for h in handles {
                    let _ = h.join();
                }
                TPRIMS_OK
            }
        }
    }
}

fn boxed(kind: Kind) -> *mut tprims_exec {
    Box::into_raw(Box::new(tprims_exec {
        refs: AtomicUsize::new(1),
        kind,
    }))
}

/// A serial executor (runs everything on the calling thread).
#[no_mangle]
pub extern "C" fn tprims_exec_serial() -> *mut tprims_exec {
    boxed(Kind::Serial)
}

/// Create a Rayon pool of `nthreads` workers owned by tprims. `opts` may be
/// null. Returns null on failure (see `tprims_last_error`).
///
/// # Safety
///
/// `opts` is null or valid.
#[no_mangle]
pub unsafe extern "C" fn tprims_exec_rayon_create(
    nthreads: usize,
    opts: *const tprims_rayon_opts,
) -> *mut tprims_exec {
    let mut out = std::ptr::null_mut();
    let st = ffi(|| {
        if nthreads == 0 {
            return Err(FfiError::new(
                TPRIMS_ERR_INVALID_ARGUMENT,
                "nthreads must be at least 1",
            ));
        }
        // SAFETY: null or valid per the contract.
        let o = if opts.is_null() {
            tprims_rayon_opts::default()
        } else {
            unsafe { *opts }
        };
        let stack = if o.stack_size == 0 {
            16 << 20
        } else {
            o.stack_size
        };
        let joins = std::sync::Arc::new(Mutex::new(Vec::new()));
        let j2 = joins.clone();
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .spawn_handler(move |t| {
                let h = std::thread::Builder::new()
                    .name(format!("tprims-worker-{}", t.index()))
                    .stack_size(stack)
                    .spawn(|| t.run())?;
                j2.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(h);
                Ok(())
            })
            .build()
            .map_err(|e| FfiError::new(TPRIMS_ERR_INTERNAL, e.to_string()))?;
        let handles = std::mem::take(
            &mut *joins
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        out = boxed(Kind::Pool {
            pool: RwLock::new(Some(Pool::owned(tp))),
            joins: Mutex::new(handles),
            budget: AtomicUsize::new(nthreads),
        });
        Ok(())
    });
    if st == TPRIMS_OK {
        out
    } else {
        std::ptr::null_mut()
    }
}

fn handle<'a>(e: *mut tprims_exec) -> Result<&'a tprims_exec, FfiError> {
    if e.is_null() {
        return Err(FfiError::new(TPRIMS_ERR_INVALID_ARGUMENT, "null executor"));
    }
    // SAFETY: non-null handles come from tprims and are alive (refcounted).
    Ok(unsafe { &*e })
}

/// Borrow the handle behind a pointer, for other parts' ABI functions.
///
/// # Errors
///
/// Null pointer.
pub fn exec_ref<'a>(e: *mut tprims_exec) -> Result<&'a tprims_exec, FfiError> {
    handle(e)
}

/// Number of worker threads (1 for serial); 0 for null or closed.
#[no_mangle]
pub extern "C" fn tprims_exec_num_threads(e: *mut tprims_exec) -> usize {
    handle(e)
        .ok()
        .map_or(0, |h| h.with(|x| Ok(x.budget().max(1))).unwrap_or(0))
}

/// Limit the threads an operation may occupy (clamped to the pool size).
#[no_mangle]
pub extern "C" fn tprims_exec_set_budget(e: *mut tprims_exec, max_threads: usize) -> tprims_status {
    ffi(|| {
        let h = handle(e)?;
        if max_threads == 0 {
            return Err(FfiError::new(
                TPRIMS_ERR_INVALID_ARGUMENT,
                "budget must be at least 1",
            ));
        }
        if let Kind::Pool { budget, .. } = &h.kind {
            budget.store(max_threads, Ordering::Relaxed);
        }
        Ok(())
    })
}

/// Close: an owned pool stops and its workers are joined before this
/// returns. `TPRIMS_BUSY` while calls are in flight, `TPRIMS_ERR_WOULD_DEADLOCK`
/// from one of its workers; closing twice is `TPRIMS_OK`. The handle stays
/// valid until its last `tprims_exec_release`.
#[no_mangle]
pub extern "C" fn tprims_exec_close(e: *mut tprims_exec) -> tprims_status {
    match handle(e) {
        Ok(h) => h.close(),
        Err(err) => {
            set_last_error(&err.message);
            err.status
        }
    }
}

/// Add a reference.
#[no_mangle]
pub extern "C" fn tprims_exec_retain(e: *mut tprims_exec) {
    if let Ok(h) = handle(e) {
        h.refs.fetch_add(1, Ordering::Relaxed);
    }
}

/// Drop a reference; the last one closes (joining an owned pool) and frees
/// the handle. On a worker of the pool itself the join is handed to a
/// detached thread.
#[no_mangle]
pub extern "C" fn tprims_exec_release(e: *mut tprims_exec) {
    let Ok(h) = handle(e) else { return };
    if h.refs.fetch_sub(1, Ordering::AcqRel) != 1 {
        return;
    }
    let p = SendPtr(e);
    let finish = move || {
        let p = p;
        // SAFETY: last reference; nobody else can observe the handle now.
        let b = unsafe { Box::from_raw(p.0) };
        while b.close() == TPRIMS_BUSY {
            std::thread::yield_now();
        }
    };
    if h.close() == TPRIMS_ERR_WOULD_DEADLOCK {
        std::thread::spawn(finish);
    } else {
        finish();
    }
}

struct SendPtr(*mut tprims_exec);
// SAFETY: the pointer is the last reference, moved to one thread.
unsafe impl Send for SendPtr {}
