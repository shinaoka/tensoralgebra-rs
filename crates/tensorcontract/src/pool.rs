//! A parked-thread pool with SPMD broadcast semantics.
//!
//! # Why this exists, and why it is not `rayon`
//!
//! Threading spawns `std::thread`s per [`execute`](crate::driver::execute) call.
//! That costs **~20–36 µs per thread** and is the entire story below about a
//! megabyte: a 0.22 ms `f32` contraction takes 2.2 ms on 64 threads, and per case
//! the damage reaches 45x (A43, D46). This removes that cost, where it
//! transfers -- see the caveat below.
//!
//! The shape the driver needs is **SPMD with barriers**: `pn` barriers of width
//! `pm`, every thread running the same loop nest and rendezvousing twice per
//! `(jc, pc)` iteration ([`crate::driver`]). That is what rules out the obvious
//! reuse:
//!
//! * **`rayon::scope`/`spawn` deadlocks by construction.** A task that blocks on
//!   a barrier occupies a worker thread; with fewer workers than barrier
//!   participants — which a caller's pool routinely has — the remaining
//!   participants are never scheduled and the barrier never opens. Work-stealing
//!   also fights the design directly: D21/D27 give every output element one owning
//!   thread precisely so that no reduction is parallelised.
//! * **`rayon::ThreadPool::broadcast` fits the shape** — one closure per worker is
//!   exactly SPMD — but it ties the parallel degree to the pool size, where
//!   `Plan::partition` chooses `(pm, pn)` per contraction, and a *global* pool
//!   inside a library fights the host runtime. That is concrete here rather than
//!   hypothetical: the Julia package and the C consumers both bring their own.
//!
//! So: one slot per worker, one `Condvar` each, and a shared completion counter.
//! About 150 lines, no dependency, and the parallel degree is per call.
//!
//! # What it does not do
//!
//! No work stealing, no nesting, no priorities. `broadcast` is the only operation
//! and it is synchronous: it returns when every worker has finished, which is what
//! makes borrowing caller stack data sound. Workers are never joined — they park
//! for the life of the process, like every thread pool. That is why this is
//! **opt-in** (`TENSORCONTRACT_POOL=on`): a library should not leave threads
//! parked in a process that only ever asked for one.
//!
//! # Why off by default
//!
//! Every threaded number committed before 2026-08-05 was measured with per-call
//! `std::thread::scope`, so turning this on silently would make those curves
//! irreproducible — the same reason `TENSORCONTRACT_PARTITION=legacy` stays
//! reachable. It is a run-time switch rather than a build-to-build diff for the
//! reason A15 records, so both arms interleave in one session.

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Condvar, Mutex, OnceLock};

/// One unit of broadcast work: a closure the workers borrow rather than own.
///
/// The lifetime is erased to `'static` so it can cross into a worker thread that
/// outlives the call. That is sound only because [`Pool::try_broadcast`] does not
/// return until every worker it woke has finished with the pointer — the same
/// argument [`std::thread::scope`] makes, and the reason `broadcast` cannot become
/// asynchronous without changing this type.
#[derive(Clone, Copy)]
struct Job(*const (dyn Fn(usize) + Sync));

// SAFETY: the pointee is `Sync`, so calling it from several threads at once is
// exactly what its bound permits. Lifetime validity is `broadcast`'s invariant,
// stated on `Job` and enforced by its blocking join.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

/// A worker's mailbox. One `Mutex`/`Condvar` per worker rather than one shared
/// pair, so the submitter wakes exactly the workers it needs: a `notify_all` on a
/// shared condvar would wake all 64 to have 62 of them look, see the job is not
/// theirs, and park again.
struct Slot {
    job: Mutex<Option<Job>>,
    wake: Condvar,
}

/// Completion accounting for the job in flight.
struct Done {
    /// Workers that have not yet finished the current job.
    left: usize,
    /// Set if any worker's closure unwound. The panic is re-raised on the
    /// submitting thread after every worker has finished, so a panic cannot leave
    /// the pool with a nonzero `left` and deadlock the next call.
    panicked: bool,
}

/// The process pool. Grows to the widest broadcast asked of it and never shrinks.
pub(crate) struct Pool {
    slots: Mutex<Vec<&'static Slot>>,
    /// Held for the whole of one broadcast, and taken with `try_lock`. One
    /// completion counter serves the pool, so two broadcasts must not overlap —
    /// and a caller that drives `contract` from several of its own threads is a
    /// real configuration, not a misuse. Rather than serialise those callers
    /// behind each other, a broadcast that cannot take this lock reports failure
    /// and the driver falls back to `std::thread::scope` for that call: slower by
    /// the spawn cost, never wrong.
    submit: Mutex<()>,
    done: Mutex<Done>,
    finished: Condvar,
}

impl Pool {
    fn new() -> Self {
        Pool {
            slots: Mutex::new(Vec::new()),
            submit: Mutex::new(()),
            done: Mutex::new(Done {
                left: 0,
                panicked: false,
            }),
            finished: Condvar::new(),
        }
    }

    /// Make sure at least `n` workers exist, spawning any that do not.
    ///
    /// Leaked deliberately: a worker parks for the life of the process, so its
    /// `Slot` must outlive every stack frame that could hand it a job. `Box::leak`
    /// is the honest spelling of that, and the leak is bounded by the widest
    /// broadcast the process ever performs.
    fn ensure(&'static self, n: usize) {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        while slots.len() < n {
            let slot: &'static Slot = Box::leak(Box::new(Slot {
                job: Mutex::new(None),
                wake: Condvar::new(),
            }));
            let index = slots.len();
            slots.push(slot);
            // If a worker cannot be spawned the pool stays narrower than asked and
            // `try_broadcast` declines, so the driver spawns the region itself. A
            // contraction must never fail because a thread could not be created.
            if std::thread::Builder::new()
                .name(format!("tensorcontract-{index}"))
                .spawn(move || self.worker(slot, index))
                .is_err()
            {
                slots.pop();
                break;
            }
        }
    }

    /// A worker: park, run one job, report, repeat. Never returns.
    ///
    /// `slot_index` is 0-based over the workers, and the **submitting thread owns
    /// broadcast index 0**, so this worker runs `f(slot_index + 1)`. Getting that
    /// off by one would run index 0 twice and never run the last index, which the
    /// driver would see as one strip computed twice and one never written.
    fn worker(&'static self, slot: &'static Slot, slot_index: usize) {
        loop {
            let job = {
                let mut guard = slot.job.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if let Some(job) = guard.take() {
                        break job;
                    }
                    guard = slot
                        .wake
                        .wait(guard)
                        .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
                }
            };
            // SAFETY: `broadcast` blocks until `left` reaches zero, and this
            // worker decrements `left` only after the call returns, so the closure
            // is alive for the whole of it. See `Job`.
            let outcome = catch_unwind(AssertUnwindSafe(|| unsafe { (*job.0)(slot_index + 1) }));
            let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
            if outcome.is_err() {
                done.panicked = true;
            }
            done.left -= 1;
            if done.left == 0 {
                self.finished.notify_all();
            }
        }
    }

    /// Run `f(0)` … `f(n - 1)`, one per thread, and return when all have finished.
    /// Reports `false` **having run nothing** if the pool cannot serve the request,
    /// so the caller can fall back.
    ///
    /// The submitting thread runs `f(0)` itself, so width `n` needs `n - 1` workers
    /// and width 1 needs none: a serial caller never causes a thread to exist.
    ///
    /// Two reasons it may decline, and both must decline **all or nothing**:
    ///
    /// * **Another broadcast is in flight.** One completion counter serves the
    ///   pool.
    /// * **Fewer than `n - 1` workers could be spawned.** Folding the surplus
    ///   indices onto the submitting thread would look like the obvious graceful
    ///   degradation and is a **deadlock**: the driver's barriers are `pm`-way per
    ///   column group, and indices `t` and `t + pn` share one, so running them
    ///   sequentially on one thread waits forever for a participant that has
    ///   already left. Declining costs the spawn saving for that call and nothing
    ///   else.
    fn try_broadcast(&'static self, n: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
        if n <= 1 {
            f(0);
            return true;
        }
        // Poisoning must not brick the pool. A worker's panic propagates through
        // this function while it holds `submit`, so the *next* broadcast would find
        // the mutex poisoned and decline forever after — the pool would silently
        // stop pooling for the life of the process the first time any contraction
        // panicked. `WouldBlock` is the only decline; poison is recovered from, as
        // it is everywhere else here, because none of this state is a value whose
        // invariants a panic can break.
        let _submitting = match self.submit.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return false,
        };
        self.ensure(n - 1);
        let workers: Vec<&'static Slot> = {
            let slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
            if slots.len() < n - 1 {
                return false;
            }
            slots.iter().take(n - 1).copied().collect()
        };

        // SAFETY: erases `f`'s lifetime so it can cross into a worker. Sound
        // because this function does not return until `left` reaches zero, and a
        // worker decrements `left` only after its call has returned. See `Job`.
        let job = Job(unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const (dyn Fn(usize) + Sync)>(
                f as *const _,
            )
        });

        {
            let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
            debug_assert_eq!(done.left, 0, "broadcast is not re-entrant");
            done.left = workers.len();
            done.panicked = false;
        }
        for slot in &workers {
            let mut guard = slot.job.lock().unwrap_or_else(|e| e.into_inner());
            *guard = Some(job);
            drop(guard);
            slot.wake.notify_one();
        }

        // The submitting thread takes index 0. Its own panic must not skip the
        // join below, or the next broadcast would see a stale `left` and hang.
        let mine = catch_unwind(AssertUnwindSafe(|| f(0)));

        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        while done.left != 0 {
            done = self
                .finished
                .wait(done)
                .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
        }
        let worker_panicked = done.panicked;
        done.panicked = false;
        drop(done);

        if let Err(payload) = mine {
            resume_unwind(payload);
        }
        assert!(!worker_panicked, "a tensorcontract pool worker panicked");
        true
    }
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(Pool::new)
}

/// Is the pool enabled? `TENSORCONTRACT_POOL=on`, default **off**.
///
/// Read once per process. It cannot affect correctness: the partition is
/// unchanged, so the result is bitwise identical either way, which is the
/// invariant the test suite asserts at every thread count.
pub(crate) fn enabled() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        std::env::var("TENSORCONTRACT_POOL")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "on" | "1" | "true"))
            .unwrap_or(false)
    })
}

/// Run `f(0)` … `f(n - 1)` concurrently on pooled threads and wait for all of
/// them. Returns `false` **having run nothing** if the pool declined, which the
/// caller must answer by spawning the region itself.
///
/// Only reached when [`enabled`] is true; the driver checks that and otherwise
/// uses [`std::thread::scope`], so both arms are measurable in one session.
pub(crate) fn try_broadcast(n: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
    pool().try_broadcast(n, f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    /// `try_broadcast` declines while another broadcast is in flight, which is the
    /// design (see `Pool::submit`) and which cargo's concurrent test threads
    /// otherwise trigger constantly. The tests below assert that a broadcast *ran*,
    /// so they take this first. It is a property of the test harness, not of the
    /// pool: the driver answers a decline by spawning the region itself.
    fn serially<R>(f: impl FnOnce() -> R) -> R {
        static LOCK: Mutex<()> = Mutex::new(());
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        f()
    }

    #[test]
    fn broadcast_runs_every_index_exactly_once() {
        serially(|| {
            for n in [1usize, 2, 3, 8, 17] {
                let seen: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                assert!(try_broadcast(n, &|i| {
                    seen[i].fetch_add(1, Ordering::Relaxed);
                }));
                for (i, c) in seen.iter().enumerate() {
                    assert_eq!(c.load(Ordering::Relaxed), 1, "index {i} of {n}");
                }
            }
        });
    }

    /// The property the driver actually depends on: the participants of a
    /// broadcast are *concurrent*, so a barrier between them opens. This is the
    /// test that would fail against `rayon::scope` in a pool too small for the
    /// barrier, which is why it is written as a barrier rather than as a counter.
    #[test]
    fn broadcast_participants_are_concurrent_enough_for_a_barrier() {
        serially(|| {
            let n = 8;
            let bar = Barrier::new(n);
            let after = AtomicUsize::new(0);
            assert!(try_broadcast(n, &|_| {
                bar.wait();
                after.fetch_add(1, Ordering::Relaxed);
            }));
            assert_eq!(after.load(Ordering::Relaxed), n);
        });
    }

    /// A worker's panic must reach the caller and must not leave the pool with a
    /// nonzero outstanding count, which would deadlock every later broadcast.
    #[test]
    fn a_worker_panic_propagates_and_leaves_the_pool_usable() {
        serially(|| {
            let hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let caught = catch_unwind(AssertUnwindSafe(|| {
                try_broadcast(4, &|i| {
                    if i == 3 {
                        panic!("worker {i}");
                    }
                });
            }));
            std::panic::set_hook(hook);
            assert!(caught.is_err(), "the panic must not be swallowed");

            let seen = AtomicUsize::new(0);
            assert!(try_broadcast(4, &|_| {
                seen.fetch_add(1, Ordering::Relaxed);
            }));
            assert_eq!(
                seen.load(Ordering::Relaxed),
                4,
                "the pool deadlocked or lost a worker"
            );
        });
    }

    /// Repeated broadcasts reuse the same threads — the whole point. Asserted
    /// through thread ids rather than through timing, so it is a property and not
    /// a benchmark.
    #[test]
    fn repeated_broadcasts_reuse_the_same_threads() {
        serially(|| {
            use std::collections::HashSet;
            use std::sync::Mutex as M;
            let ids: M<HashSet<std::thread::ThreadId>> = M::new(HashSet::new());
            for _ in 0..4 {
                assert!(try_broadcast(4, &|_| {
                    ids.lock().unwrap().insert(std::thread::current().id());
                }));
            }
            // Four indices, one of which is the caller: at most four distinct threads
            // over four rounds. Without reuse there would be up to thirteen.
            assert!(ids.lock().unwrap().len() <= 4);
        });
    }
}
