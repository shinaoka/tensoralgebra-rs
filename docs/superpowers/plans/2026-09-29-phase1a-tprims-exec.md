# Phase 1a: tprims-exec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An explicit execution context (`tprims-exec`) that borrows a host Rayon pool, enters it only for parallel work, and drives strided kernels, faer and tensorcontract's SPMD driver — with entry counters, tests for the width/nesting/concurrency rules, and 1T/4T benchmarks.

**Architecture:** `tprims-exec` owns `Pool<'p>` (a borrowed `&rayon::ThreadPool` plus an SPMD mutex and entry counters) and `Exec<'a>` (`Serial` or `Rayon { pool, budget }`). Three primitives: `install(k, op)` (faer-style), `for_each_partition(k, f)` (barrier-free), `broadcast(width, f)` (SPMD, full pool, serialized, refused when nested). strided-basic gains a feature-gated adapter `run_with_exec`; tensorcontract gains a public `Spmd` seam used instead of `thread::scope` when supplied.

**Tech Stack:** Rust 1.89+, rayon 1.10+, faer 0.24 (dev), strided-basic, tensorcontract.

**Spec:** `docs/superpowers/specs/2026-09-29-phase1-cpu-backend-design.md` (section 1a); rules in `PERFORMANCE_TIPS.md` (CPU Threading Contract, Tests And Benchmarks).

## Global Constraints

- Branch `phase1a-exec` from `phase0-consolidation` (stacked; PR base `phase0-consolidation`).
- Every cargo invocation uses `-j 16`.
- No ambient pool in tprims code: no `rayon::current_num_threads()`, no `ThreadPoolBuilder` outside tests/benches, no `std::thread::spawn`/`scope` in production paths of new code.
- Serial work (`k == 1`) never enters a pool; a caller already on a worker of the pool never re-enters (`current_thread_index().is_some()`).
- New crates: `version = "0.1.0"`, `edition.workspace`, `rust-version.workspace`, `license = "MIT OR Apache-2.0"`, `publish = false`, `[lints] workspace = true`.
- Imported crates: only additive, feature-gated or seam changes; upstream default behaviour unchanged (tensorcontract without a seam still uses `thread::scope`/its pool; strided without the feature is unchanged).
- Unit tests in `src/<module>/tests/*.rs` or crate `tests/`; no large inline test blocks (REPOSITORY_RULES Unit Test Organization).
- Commit trailer `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

- `Pool::borrow` on a pool whose size changes? (Rayon pools are fixed size — assert once, document.)
- `broadcast` called from a worker of the same pool (nested SPMD) must return `Err(Unavailable)` without running anything, and tensorcontract must then fall back to its serial path, not `thread::scope`.
- Two host threads calling `broadcast` concurrently on the same `Pool` must not deadlock (mutex) — test with barriers and a timeout.
- A panic inside `f` in `broadcast`/`for_each_partition` must propagate to the caller and leave the SPMD mutex usable (poison handled).
- `run_with_exec` below `MINTHREADLENGTH` must not enter the pool (entry counter stays 0).

---

### Task 1: `tprims-exec` crate — `Pool`, `Exec`, `install`, `for_each_partition`

**Files:**
- Create: `crates/tprims-exec/Cargo.toml`, `crates/tprims-exec/src/lib.rs`, `crates/tprims-exec/src/pool.rs`, `crates/tprims-exec/src/exec.rs`, `crates/tprims-exec/src/width.rs`, `crates/tprims-exec/src/error.rs`, `crates/tprims-exec/tests/entry.rs`
- Modify: root `Cargo.toml` (member `crates/tprims-exec`, workspace dep `tprims-exec = { version = "0.1.0", path = "crates/tprims-exec" }`)

**Interfaces (Produces):**
```rust
pub struct Pool<'p> { /* &'p rayon::ThreadPool, Mutex<()>, AtomicU64 counters */ }
impl<'p> Pool<'p> {
    pub fn borrow(pool: &'p rayon::ThreadPool) -> Self;
    pub fn size(&self) -> usize;
    pub fn stats(&self) -> PoolStats;          // entries, broadcasts, inline_runs
    pub fn reset_stats(&self);
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats { pub entries: u64, pub broadcasts: u64, pub inline_runs: u64 }
#[derive(Clone, Copy, Debug)] #[non_exhaustive]
pub enum Exec<'a> { Serial, Rayon { pool: &'a Pool<'a>, budget: NonZeroUsize } }
impl<'a> Exec<'a> {
    pub const fn serial() -> Exec<'static>;
    pub fn rayon(pool: &'a Pool<'a>) -> Self;                 // budget = pool.size()
    pub fn with_budget(self, max_threads: usize) -> Result<Self, ExecError>;
    pub fn budget(&self) -> usize;
    pub fn is_worker(&self) -> bool;                            // caller is a worker of this pool
    pub fn install<R: Send>(&self, k: usize, op: impl FnOnce(Par) -> R + Send) -> R;
    pub fn for_each_partition(&self, k: usize, f: &(dyn Fn(usize) + Sync));
    pub fn broadcast(&self, width: usize, f: &(dyn Fn(usize) + Sync)) -> Result<(), ExecError>;   // Task 2
    pub fn width_for(&self, serial_ns: f64, policy: &WidthPolicy) -> usize;                        // Task 2
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Par { Seq, Threads(NonZeroUsize) }
impl Par { pub fn threads(self) -> usize; }
#[derive(Debug, thiserror::Error, PartialEq, Eq)] #[non_exhaustive]
pub enum ExecError {
    #[error("thread budget must be at least 1")] ZeroBudget,
    #[error("SPMD width {width} exceeds pool size {pool}")] WidthExceedsPool { width: usize, pool: usize },
    #[error("co-scheduled execution unavailable in this context")] Unavailable,
}
```

- [ ] **Step 1: Manifest**

```toml
[package]
name = "tprims-exec"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license = "MIT OR Apache-2.0"
publish = false
description = "Explicit execution contexts for the tprims stack: borrowed Rayon pools, kernel-level entry"

[dependencies]
rayon = { workspace = true }
thiserror = { workspace = true }

[dev-dependencies]
faer = { workspace = true }

[lints]
workspace = true
```

- [ ] **Step 2: Write failing tests** (`crates/tprims-exec/tests/entry.rs`)

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use tprims_exec::{Exec, Par, Pool};

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap()
}

#[test]
fn serial_install_runs_inline_with_seq() {
    let caller = std::thread::current().id();
    let got = Exec::serial().install(4, |par| (par, std::thread::current().id()));
    assert_eq!(got, (Par::Seq, caller));
}

#[test]
fn width_one_never_enters_the_pool() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let caller = std::thread::current().id();
    let tid = exec.install(1, |par| { assert_eq!(par, Par::Seq); std::thread::current().id() });
    assert_eq!(tid, caller);
    exec.for_each_partition(1, &|i| assert_eq!(i, 0));
    assert_eq!(p.stats().entries, 0);
}

#[test]
fn parallel_install_enters_once_and_runs_on_a_worker() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let (par, idx) = exec.install(3, |par| (par, rayon::current_thread_index()));
    assert_eq!(par.threads(), 3);
    assert!(idx.is_some());
    assert_eq!(p.stats().entries, 1);
}

#[test]
fn nested_install_on_a_worker_does_not_reenter() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    exec.install(2, |_| {
        assert!(exec.is_worker());
        exec.install(2, |par| assert_eq!(par.threads(), 2));
    });
    let s = p.stats();
    assert_eq!((s.entries, s.inline_runs), (1, 1));
}

#[test]
fn budget_caps_width_and_zero_budget_is_an_error() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p).with_budget(2).unwrap();
    assert_eq!(exec.budget(), 2);
    assert_eq!(exec.install(8, |par| par.threads()), 2);
    assert!(Exec::rayon(&p).with_budget(0).is_err());
    assert_eq!(Exec::rayon(&p).with_budget(99).unwrap().budget(), 4);
}

#[test]
fn partition_runs_every_index_once_and_repartitions_above_budget() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p).with_budget(2).unwrap();
    let hits: Vec<AtomicUsize> = (0..7).map(|_| AtomicUsize::new(0)).collect();
    exec.for_each_partition(7, &|i| { hits[i].fetch_add(1, Ordering::Relaxed); });
    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
}

#[test]
fn faer_matmul_runs_on_the_borrowed_pool() {
    use faer::{linalg::matmul::matmul, Accum, Mat};
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let a = Mat::<f64>::from_fn(256, 256, |i, j| (i + 2 * j) as f64 * 1e-3);
    let b = Mat::<f64>::from_fn(256, 256, |i, j| (i * j % 7) as f64);
    let mut c = Mat::<f64>::zeros(256, 256);
    let mut r = Mat::<f64>::zeros(256, 256);
    matmul(r.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, faer::Par::Seq);
    exec.install(4, |par| {
        let fp = match par { Par::Seq => faer::Par::Seq, Par::Threads(n) => faer::Par::rayon(n.get()) };
        matmul(c.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, fp);
    });
    assert!((&c - &r).norm_max() <= 1e-9 * r.norm_max());
    assert_eq!(p.stats().entries, 1);
}
```

- [ ] **Step 3: Run to see them fail**

Run: `cargo test -j 16 -p tprims-exec`
Expected: compile errors (crate empty).

- [ ] **Step 4: Implement**

`src/lib.rs`:
```rust
//! Explicit execution contexts for the tprims stack.
//!
//! A host lends its Rayon pool through [`Pool::borrow`]; operations take an
//! [`Exec`]. Work of width one runs on the calling thread and never touches
//! the pool; wider work enters the pool once, and not at all when the caller
//! is already one of its workers. See `PERFORMANCE_TIPS.md`, CPU Threading
//! Contract.
//!
//! # Examples
//!
//! ```
//! use tprims_exec::{Exec, Pool};
//! let tp = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
//! let pool = Pool::borrow(&tp);
//! let exec = Exec::rayon(&pool);
//! let width = exec.install(2, |par| par.threads());
//! assert_eq!(width, 2);
//! assert_eq!(pool.stats().entries, 1);
//! ```
mod error;
mod exec;
mod pool;
mod width;

pub use error::ExecError;
pub use exec::{Exec, Par};
pub use pool::{Pool, PoolStats};
pub use width::WidthPolicy;
```

`src/pool.rs`:
```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// A Rayon pool lent by the host for the lifetime `'p`.
///
/// tprims never creates threads beyond this pool. The pool's size is fixed
/// at construction (Rayon pools cannot resize). The embedded mutex
/// serializes SPMD broadcasts so two co-scheduled kernels cannot interleave
/// on the same workers.
///
/// # Examples
///
/// ```
/// let tp = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
/// let pool = tprims_exec::Pool::borrow(&tp);
/// assert_eq!(pool.size(), 3);
/// ```
pub struct Pool<'p> {
    pub(crate) pool: &'p rayon::ThreadPool,
    pub(crate) spmd: Mutex<()>,
    entries: AtomicU64,
    broadcasts: AtomicU64,
    inline_runs: AtomicU64,
}

/// Counters of how often a [`Pool`] was entered, for tests and benchmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Entries from outside the pool (`install` or partitions).
    pub entries: u64,
    /// SPMD broadcasts started from outside the pool.
    pub broadcasts: u64,
    /// Parallel requests served in place because the caller was a worker.
    pub inline_runs: u64,
}

impl<'p> Pool<'p> {
    /// Borrow a host pool.
    pub fn borrow(pool: &'p rayon::ThreadPool) -> Self {
        Self { pool, spmd: Mutex::new(()), entries: AtomicU64::new(0), broadcasts: AtomicU64::new(0), inline_runs: AtomicU64::new(0) }
    }
    /// Number of workers.
    pub fn size(&self) -> usize { self.pool.current_num_threads() }
    /// Snapshot of the entry counters.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            entries: self.entries.load(Ordering::Relaxed),
            broadcasts: self.broadcasts.load(Ordering::Relaxed),
            inline_runs: self.inline_runs.load(Ordering::Relaxed),
        }
    }
    /// Zero the counters.
    pub fn reset_stats(&self) {
        self.entries.store(0, Ordering::Relaxed);
        self.broadcasts.store(0, Ordering::Relaxed);
        self.inline_runs.store(0, Ordering::Relaxed);
    }
    pub(crate) fn is_worker(&self) -> bool { self.pool.current_thread_index().is_some() }
    pub(crate) fn count_entry(&self) { self.entries.fetch_add(1, Ordering::Relaxed); }
    pub(crate) fn count_broadcast(&self) { self.broadcasts.fetch_add(1, Ordering::Relaxed); }
    pub(crate) fn count_inline(&self) { self.inline_runs.fetch_add(1, Ordering::Relaxed); }
}

impl std::fmt::Debug for Pool<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool").field("size", &self.size()).field("stats", &self.stats()).finish()
    }
}
```
(Move the Examples/test pattern for `Pool` into rustdoc as shown; `rayon::ThreadPoolBuilder` is allowed in doctests and tests only.)

`src/exec.rs`:
```rust
use std::num::NonZeroUsize;
use crate::{ExecError, Pool};

/// Parallelism granted to an operation by [`Exec::install`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Par {
    /// Run serially on the current thread.
    Seq,
    /// Up to this many workers of the pool the closure runs in.
    Threads(NonZeroUsize),
}

impl Par {
    /// Worker count (1 for `Seq`).
    pub fn threads(self) -> usize { match self { Par::Seq => 1, Par::Threads(n) => n.get() } }
}

/// Execution context passed to every operation that may run in parallel.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Exec<'a> {
    /// Everything on the calling thread.
    Serial,
    /// A borrowed pool with a thread budget `<= pool.size()`.
    Rayon { pool: &'a Pool<'a>, budget: NonZeroUsize },
}

impl<'a> Exec<'a> {
    /// The serial context.
    pub const fn serial() -> Exec<'static> { Exec::Serial }
    /// Use `pool` with its full size as budget.
    pub fn rayon(pool: &'a Pool<'a>) -> Self {
        let budget = NonZeroUsize::new(pool.size()).unwrap_or(NonZeroUsize::MIN);
        Exec::Rayon { pool, budget }
    }
    /// Limit the budget; values above the pool size are clamped.
    pub fn with_budget(self, max_threads: usize) -> Result<Self, ExecError> {
        let max = NonZeroUsize::new(max_threads).ok_or(ExecError::ZeroBudget)?;
        Ok(match self {
            Exec::Serial => Exec::Serial,
            Exec::Rayon { pool, .. } => {
                let cap = NonZeroUsize::new(pool.size()).unwrap_or(NonZeroUsize::MIN);
                Exec::Rayon { pool, budget: max.min(cap) }
            }
        })
    }
    /// Maximum threads an operation may occupy.
    pub fn budget(&self) -> usize { match self { Exec::Serial => 1, Exec::Rayon { budget, .. } => budget.get() } }
    /// Whether the calling thread is a worker of this context's pool.
    pub fn is_worker(&self) -> bool { match self { Exec::Serial => false, Exec::Rayon { pool, .. } => pool.is_worker() } }

    fn width(&self, k: usize) -> usize { k.clamp(1, self.budget()) }

    /// Run `op` with `min(k, budget)` threads of parallelism. Width one runs
    /// `op(Par::Seq)` inline; otherwise `op` runs inside the pool (entered
    /// once, or in place when the caller is already a worker).
    pub fn install<R: Send>(&self, k: usize, op: impl FnOnce(Par) -> R + Send) -> R {
        let k = self.width(k);
        match (self, NonZeroUsize::new(k)) {
            (Exec::Rayon { pool, .. }, Some(n)) if k > 1 => {
                if pool.is_worker() {
                    pool.count_inline();
                    op(Par::Threads(n))
                } else {
                    pool.count_entry();
                    pool.pool.install(|| op(Par::Threads(n)))
                }
            }
            _ => op(Par::Seq),
        }
    }

    /// Barrier-free partition: runs `f(i)` exactly once for each `i < k`,
    /// using at most `budget` workers. Tasks are not guaranteed to run
    /// concurrently, so `f` must not wait on another index.
    pub fn for_each_partition(&self, k: usize, f: &(dyn Fn(usize) + Sync)) {
        if k == 0 { return; }
        let lanes = self.width(k);
        if lanes == 1 {
            (0..k).for_each(f);
            return;
        }
        self.install(lanes, |_| {
            rayon::scope(|s| {
                for lane in 0..lanes {
                    s.spawn(move |_| {
                        let lo = lane * k / lanes;
                        let hi = (lane + 1) * k / lanes;
                        (lo..hi).for_each(f);
                    });
                }
            })
        });
    }
}
```
`src/error.rs`: the `ExecError` enum from Interfaces. `src/width.rs`: placeholder module for Task 2 containing `pub struct WidthPolicy;` is NOT allowed — create `width.rs` in Task 2 and add `mod width; pub use width::WidthPolicy;` then; in Task 1 omit both lines from `lib.rs`.

- [ ] **Step 5: Run tests**

Run: `cargo test -j 16 -p tprims-exec`
Expected: all 7 tests and doctests pass.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/tprims-exec
git commit -m "Add tprims-exec: borrowed pool, kernel-level entry, barrier-free partitions

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: SPMD `broadcast` and width policy

**Files:**
- Modify: `crates/tprims-exec/src/exec.rs`, `crates/tprims-exec/src/lib.rs`
- Create: `crates/tprims-exec/src/width.rs`, `crates/tprims-exec/tests/spmd.rs`

**Interfaces (Produces):**
```rust
impl Exec<'_> {
    /// Co-scheduled SPMD: `f(t)` for t < width all run concurrently (barriers allowed).
    /// Serial: width 1 runs inline, width > 1 → Err(Unavailable).
    /// Rayon: width > pool.size() → Err(WidthExceedsPool); caller is a worker → Err(Unavailable);
    /// else lock the pool's SPMD mutex and `ThreadPool::broadcast`, workers with index >= width return at once.
    pub fn broadcast(&self, width: usize, f: &(dyn Fn(usize) + Sync)) -> Result<(), ExecError>;
    pub fn width_for(&self, serial_ns: f64, policy: &WidthPolicy) -> usize;
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WidthPolicy { pub serial_below_ns: f64, pub entry_base_ns: f64, pub entry_per_thread_ns: f64 }
impl Default for WidthPolicy { /* 50_000.0, 5_000.0, 5_000.0 — provisional, from experiments/rayon-entry */ }
```

- [ ] **Step 1: Failing tests** (`tests/spmd.rs`)

```rust
use std::sync::{Barrier, atomic::{AtomicUsize, Ordering}};
use std::time::Duration;
use tprims_exec::{Exec, ExecError, Pool, WidthPolicy};

fn pool(n: usize) -> rayon::ThreadPool { rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap() }

fn with_timeout<F: FnOnce() + Send + 'static>(secs: u64, f: F) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || { f(); let _ = tx.send(()); });
    rx.recv_timeout(Duration::from_secs(secs)).expect("deadlock: timed out");
}

#[test]
fn broadcast_runs_width_participants_concurrently() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let bar = Barrier::new(3);
    let n = AtomicUsize::new(0);
    exec.broadcast(3, &|t| { assert!(t < 3); bar.wait(); n.fetch_add(1, Ordering::Relaxed); }).unwrap();
    assert_eq!(n.load(Ordering::Relaxed), 3);
    assert_eq!(p.stats().broadcasts, 1);
}

#[test]
fn broadcast_rejects_width_above_pool_and_serial_width_two() {
    let tp = pool(2);
    let p = Pool::borrow(&tp);
    assert_eq!(Exec::rayon(&p).broadcast(3, &|_| {}), Err(ExecError::WidthExceedsPool { width: 3, pool: 2 }));
    assert_eq!(Exec::serial().broadcast(2, &|_| {}), Err(ExecError::Unavailable));
    let ran = AtomicUsize::new(0);
    Exec::serial().broadcast(1, &|_| { ran.fetch_add(1, Ordering::Relaxed); }).unwrap();
    assert_eq!(ran.load(Ordering::Relaxed), 1);
}

#[test]
fn nested_broadcast_from_a_worker_is_unavailable_and_runs_nothing() {
    with_timeout(20, || {
        let tp = pool(4);
        let p = Pool::borrow(&tp);
        let exec = Exec::rayon(&p);
        let inner = AtomicUsize::new(0);
        exec.broadcast(4, &|_| {
            let r = exec.broadcast(2, &|_| { inner.fetch_add(1, Ordering::Relaxed); });
            assert_eq!(r, Err(ExecError::Unavailable));
        }).unwrap();
        assert_eq!(inner.load(Ordering::Relaxed), 0);
    });
}

#[test]
fn concurrent_broadcasts_from_two_host_threads_do_not_deadlock() {
    with_timeout(60, || {
        let tp = pool(4);
        let p = Pool::borrow(&tp);
        std::thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(|| {
                    let exec = Exec::rayon(&p);
                    for _ in 0..50 {
                        let bar = Barrier::new(4);
                        exec.broadcast(4, &|_| { bar.wait(); }).unwrap();
                    }
                });
            }
        });
        assert_eq!(p.stats().broadcasts, 100);
    });
}

#[test]
fn panic_in_broadcast_propagates_and_pool_stays_usable() {
    let tp = pool(2);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = exec.broadcast(2, &|t| if t == 1 { panic!("boom") });
    }));
    assert!(r.is_err());
    exec.broadcast(2, &|_| {}).unwrap();
}

#[test]
fn width_policy_is_serial_for_small_work_and_capped_by_budget() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let pol = WidthPolicy::default();
    assert_eq!(exec.width_for(1_000.0, &pol), 1);
    assert_eq!(exec.width_for(1e9, &pol), 4);
    assert_eq!(exec.with_budget(2).unwrap().width_for(1e9, &pol), 2);
    assert_eq!(Exec::serial().width_for(1e9, &pol), 1);
}

#[test]
fn os_thread_count_does_not_grow_after_partitions() {
    fn os_threads() -> usize { std::fs::read_dir("/proc/self/task").map(|d| d.count()).unwrap_or(0) }
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p).with_budget(2).unwrap();
    exec.for_each_partition(64, &|_| {});
    let before = os_threads();
    for _ in 0..20 { exec.for_each_partition(64, &|_| {}); exec.broadcast(4, &|_| {}).unwrap(); }
    if before > 0 { assert_eq!(os_threads(), before); }
}
```

- [ ] **Step 2: Run** — `cargo test -j 16 -p tprims-exec --test spmd` → FAIL (no `broadcast`/`WidthPolicy`).

- [ ] **Step 3: Implement** in `exec.rs`:

```rust
    pub fn broadcast(&self, width: usize, f: &(dyn Fn(usize) + Sync)) -> Result<(), ExecError> {
        if width == 0 { return Ok(()); }
        match self {
            Exec::Serial => {
                if width == 1 { f(0); Ok(()) } else { Err(ExecError::Unavailable) }
            }
            Exec::Rayon { pool, .. } => {
                let size = pool.size();
                if width > size { return Err(ExecError::WidthExceedsPool { width, pool: size }); }
                if pool.is_worker() { return Err(ExecError::Unavailable); }
                if width == 1 { f(0); return Ok(()); }
                // INVARIANT: the SPMD mutex serializes broadcasts on this pool, so
                // every worker runs at most one barrier-bearing job at a time.
                let _guard = pool.spmd.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.count_broadcast();
                pool.pool.broadcast(|ctx| { let t = ctx.index(); if t < width { f(t) } });
                Ok(())
            }
        }
    }

    pub fn width_for(&self, serial_ns: f64, policy: &crate::WidthPolicy) -> usize {
        let budget = self.budget();
        if budget == 1 || !(serial_ns >= policy.serial_below_ns) { return 1; }
        let cost = |k: usize| if k == 1 { serial_ns } else { policy.entry_base_ns + policy.entry_per_thread_ns * k as f64 + serial_ns / k as f64 };
        (1..=budget).min_by(|&a, &b| cost(a).total_cmp(&cost(b))).unwrap_or(1)
    }
```
Note: a panic in `f` makes Rayon's `broadcast` propagate the panic to the caller after all participants finish; the guard is dropped during unwinding and a poisoned mutex is recovered with `into_inner` (the mutex protects no data). `width.rs`:
```rust
/// Cost model for choosing a barrier-free partition width from the work.
///
/// `T_par(k) = entry_base_ns + entry_per_thread_ns * k + serial_ns / k` for
/// `k > 1`; work below `serial_below_ns` always runs serially. Defaults are
/// provisional values from `experiments/rayon-entry` (M5 Max, 2026-09-29);
/// kernels replace them with their own measured policy.
///
/// # Examples
///
/// ```
/// let p = tprims_exec::WidthPolicy::default();
/// assert_eq!(p.serial_below_ns, 50_000.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WidthPolicy { pub serial_below_ns: f64, pub entry_base_ns: f64, pub entry_per_thread_ns: f64 }
impl Default for WidthPolicy {
    fn default() -> Self { Self { serial_below_ns: 50_000.0, entry_base_ns: 5_000.0, entry_per_thread_ns: 5_000.0 } }
}
```
Add `mod width; pub use width::WidthPolicy;` to `lib.rs`.

- [ ] **Step 4: Run** — `cargo test -j 16 -p tprims-exec` → all pass (run the concurrency test 5 times: `for i in 1 2 3 4 5; do cargo test -j 16 -p tprims-exec --test spmd concurrent -q || break; done`).

- [ ] **Step 5: Commit** — "Add SPMD broadcast and width policy to tprims-exec".

### Task 3: strided adapter `run_with_exec`

**Files:**
- Modify: `strided/strided-basic/Cargo.toml` (optional dep `tprims-exec`, feature `tprims-exec = ["parallel", "dep:tprims-exec"]`), `strided/strided-basic/src/exec_context.rs`, `strided/strided-basic/src/lib.rs`
- Create: `strided/strided-basic/tests/tprims_exec_adapter.rs`
- Modify: `strided/strided-kernel/Cargo.toml` (feature `tprims-exec = ["strided-basic/tprims-exec"]`, re-export if strided-kernel re-exports `ExecContext`)

**Interfaces (Produces):**
```rust
// strided_basic (feature "tprims-exec")
pub fn run_with_exec<R: Send>(exec: &tprims_exec::Exec<'_>, len: usize, op: impl FnOnce(ExecContext) -> R + Send) -> R;
```
Semantics: `k = if len > MINTHREADLENGTH { exec.budget() } else { 1 }`; `exec.install(k, |par| { let ctx = match par { Par::Seq => ExecContext::serial(), Par::Threads(n) => ExecContext::max_threads(n.get()).expect("n >= 1") }; ctx.run(|| op(ctx)) })`. (Use `ExecContext::serial()` fallback instead of `expect` to satisfy the no-panic rule: `.unwrap_or(ExecContext::serial())`.)

- [ ] **Step 1: Failing test** (`tests/tprims_exec_adapter.rs`, `#![cfg(feature = "tprims-exec")]`)

```rust
#![cfg(feature = "tprims-exec")]
use strided_basic::{map_into, run_with_exec, StridedArray};
use tprims_exec::{Exec, Pool};

fn pool(n: usize) -> rayon::ThreadPool { rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap() }

#[test]
fn small_map_stays_on_caller_and_never_enters() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let src = StridedArray::<f64>::from_fn_col_major(&[64, 8], |i| (i[0] + i[1]) as f64);
    let mut dst = StridedArray::<f64>::col_major(&[64, 8]);
    run_with_exec(&exec, 64 * 8, |_| map_into(&mut dst.view_mut(), &src.view(), |x| 2.0 * x).unwrap());
    assert_eq!(p.stats().entries, 0);
    assert_eq!(dst.view().get(&[3, 2]), 10.0);
}

#[test]
fn large_map_enters_once_and_matches_serial() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let dims = [512, 256];
    let src = StridedArray::<f64>::from_fn_col_major(&dims, |i| (i[0] * 3 + i[1]) as f64);
    let src_t = src.view().permute(&[1, 0]).unwrap();
    let mut par = StridedArray::<f64>::col_major(&[256, 512]);
    let mut ser = StridedArray::<f64>::col_major(&[256, 512]);
    run_with_exec(&exec, 512 * 256, |_| map_into(&mut par.view_mut(), &src_t, |x| x + 1.0).unwrap());
    run_with_exec(&Exec::serial(), 512 * 256, |_| map_into(&mut ser.view_mut(), &src_t, |x| x + 1.0).unwrap());
    assert_eq!(par.into_data(), ser.into_data());
    assert_eq!(p.stats().entries, 1);
}
```
(Adjust `permute`/`get` calls to the actual `StridedView` API if they differ: `permute(&self, perm: &[usize])` returns `Result<StridedView>`; `get(&self, idx: &[usize]) -> T`.)

- [ ] **Step 2: Run** — `cargo test -j 16 -p strided-basic --features tprims-exec --test tprims_exec_adapter` → FAIL.

- [ ] **Step 3: Implement** in `exec_context.rs` (below `impl Default`):

```rust
/// Run a strided operation of `len` logical elements on a tprims execution
/// context: below the parallel threshold it runs serially on the caller and
/// never enters the pool; above it, the pool is entered once and strided's
/// fanout is bounded by the context's budget.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "tprims-exec")] {
/// use strided_basic::run_with_exec;
/// let n = run_with_exec(&tprims_exec::Exec::serial(), 10, |ctx| { assert!(ctx.is_serial()); 1 });
/// assert_eq!(n, 1);
/// # }
/// ```
#[cfg(feature = "tprims-exec")]
pub fn run_with_exec<R: Send>(
    exec: &tprims_exec::Exec<'_>,
    len: usize,
    op: impl FnOnce(ExecContext) -> R + Send,
) -> R {
    let k = if len > crate::threading::MINTHREADLENGTH { exec.budget() } else { 1 };
    exec.install(k, |par| {
        let ctx = match par {
            tprims_exec::Par::Seq => ExecContext::serial(),
            tprims_exec::Par::Threads(n) => ExecContext::max_threads(n.get()).unwrap_or(ExecContext::serial()),
        };
        ctx.run(|| op(ctx))
    })
}
```
`lib.rs`: `#[cfg(feature = "tprims-exec")] pub use exec_context::run_with_exec;`. Cargo: `tprims-exec = { workspace = true, optional = true }` and dev-dep `rayon.workspace = true` (already present if so).

- [ ] **Step 4: Run** — the adapter tests, then `cargo test -j 16 -p strided-basic --features tprims-exec` and the default `cargo test -j 16 -p strided-basic` (unchanged behaviour) → pass.

- [ ] **Step 5: Commit** — "strided-basic: run strided kernels on a tprims-exec context (feature tprims-exec)".

### Task 4: tensorcontract `Spmd` seam

**Files:**
- Modify: `tensorprimitives/crates/tensorcontract/src/driver.rs`, `src/lib.rs`, `src/batch.rs` (call sites of `execute_capped` pass `None`)
- Create: `tensorprimitives/crates/tensorcontract/src/spmd.rs`, `tensorprimitives/crates/tensorcontract/tests/spmd_seam.rs`
- Create: `crates/tprims-exec/tests/tensorcontract_seam.rs` (dev-dep `tensorcontract`)

**Interfaces (Produces):**
```rust
// tensorcontract::spmd
pub trait Spmd: Sync {
    /// Desired number of co-scheduled participants for this call (>= 1).
    fn width(&self) -> usize;
    /// Run `f(t)` for every t < p concurrently and return true, or run nothing and return false.
    fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool;
}
impl Plan {
    pub fn run_with<T>(&self, spmd: &dyn Spmd, alpha: T, a: TensorView<'_, T>, b: TensorView<'_, T>, beta: T, c: Option<TensorView<'_, T>>, d: TensorViewMut<'_, T>) -> Result<()>;
    pub unsafe fn run_raw_with<T>(&self, spmd: &dyn Spmd, alpha: T, a: *const T, b: *const T, beta: T, c: *const T, d: *mut T);
}
```
Driver change: `execute_capped(plan, alpha, a, b, beta, c, d, max_threads, spmd: Option<&dyn Spmd>)`. With `Some(s)`: `want = s.width().min(max_threads).max(1)` (ignores `plan.threads()`/`TENSORCONTRACT_THREADS`); after the partition `p = pm*pn`, if `p == 1` serial path; else `if s.broadcast(p, &cell) { return }` else recompute with `max_threads = 1` (nothing ran, so re-entering serially is safe) — never `thread::scope`. With `None`: existing behaviour unchanged.

- [ ] **Step 1: Failing test in tensorcontract** (`tests/spmd_seam.rs`): a counting `Spmd` implementation that spawns `p` scoped threads (test-only) — assert results equal `Plan::run` serial result bitwise for a 96x80x72 f64 GEMM `"ik,kj->ij"` with `width() = 4`, that `broadcast` was called with `p > 1`, and that a refusing `Spmd` (returns false) still produces the bitwise-identical result.

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use tensorcontract::{spmd::Spmd, Layout, Operand, Plan, TensorView, TensorViewMut};

struct Scoped { w: usize, calls: AtomicUsize, refuse: bool }
impl Spmd for Scoped {
    fn width(&self) -> usize { self.w }
    fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.refuse { return false; }
        std::thread::scope(|s| for t in 0..p { s.spawn(move || f(t)); });
        true
    }
}

fn gemm_case() -> (Layout, Layout, Layout, Vec<f64>, Vec<f64>) {
    let (m, k, n) = (96i64, 72i64, 80i64);
    let a = Layout::col_major(&[m, k]); let b = Layout::col_major(&[k, n]); let d = Layout::col_major(&[m, n]);
    let av = (0..(m * k)).map(|x| (x % 13) as f64 - 6.0).collect();
    let bv = (0..(k * n)).map(|x| (x % 7) as f64 * 0.5).collect();
    (a, b, d, av, bv)
}

#[test]
fn seam_matches_serial_bitwise_and_refusal_falls_back_serially() {
    let (la, lb, ld, av, bv) = gemm_case();
    let (ia, ib, id) = ([0i64, 1], [1i64, 2], [0i64, 2]);
    let plan = Plan::new(Operand::new(&la, &ia), Operand::new(&lb, &ib), None, Operand::new(&ld, &id)).unwrap().with_threads(1);
    let mut serial = vec![0.0; 96 * 80];
    plan.run(1.0, TensorView::new(&av, &la, &ia), TensorView::new(&bv, &lb, &ib), 0.0, None, TensorViewMut::new(&mut serial, &ld, &id)).unwrap();
    for refuse in [false, true] {
        let s = Scoped { w: 4, calls: AtomicUsize::new(0), refuse };
        let mut out = vec![0.0; 96 * 80];
        plan.run_with(&s, 1.0, TensorView::new(&av, &la, &ia), TensorView::new(&bv, &lb, &ib), 0.0, None, TensorViewMut::new(&mut out, &ld, &id)).unwrap();
        assert_eq!(out, serial);
        assert_eq!(s.calls.load(Ordering::Relaxed), 1);
    }
}
```
(Use the crate's real constructors: check `Operand::new`, `TensorView::new`, `TensorViewMut::new` signatures in `plan.rs`/`lib.rs` and adapt; the assertions stay.)

- [ ] **Step 2: Run** — `cargo test -j 16 -p tensorcontract --test spmd_seam` → FAIL.

- [ ] **Step 3: Implement** the trait (`src/spmd.rs`, `pub mod spmd;` in lib.rs with rustdoc stating the co-scheduling contract and that barriers inside `f` require all `p` participants to run concurrently), thread the `Option<&dyn Spmd>` through `execute_capped` (existing callers pass `None`; `execute` passes `None`), add `run_with`/`run_raw_with` mirroring `run`/`run_raw`. Mark the seam in the file header comment of `driver.rs`: "tprims: `Spmd` seam added (not upstream)".

- [ ] **Step 4: Tests** — `cargo test -j 16 -p tensorcontract` and `cargo test -j 16 -p tensorcontract --release` → all pass (upstream suite unchanged).

- [ ] **Step 5: tprims-exec end-to-end seam test** (`crates/tprims-exec/tests/tensorcontract_seam.rs`, dev-dep `tensorcontract = { workspace = true }`): a local adapter

```rust
struct ExecSpmd<'a> { exec: &'a tprims_exec::Exec<'a>, width: usize }
impl tensorcontract::spmd::Spmd for ExecSpmd<'_> {
    fn width(&self) -> usize { self.width }
    fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool { self.exec.broadcast(p, f).is_ok() }
}
```
Tests: (a) on a 4-worker `Pool`, the GEMM above with `width = 4` equals serial bitwise and `pool.stats().broadcasts == 1`; (b) called from inside `exec.install(2, ..)` (caller is a worker) the contraction still completes (broadcast refused → serial) within a 20 s timeout and `broadcasts == 0`.

- [ ] **Step 6: Commit** — "tensorcontract: add an Spmd seam so hosts can supply co-scheduled threads".

### Task 5: Benchmarks (1T/4T) and harness thread enforcement

**Files:**
- Create: `benchmarks/src/lib.rs` (module `threads`), `benchmarks/benchmarks/tprims/exec_entry/exec_entry.rs`, `benchmarks/benchmarks/tprims/exec_entry/README.md`, `benchmarks/benchmarks/tprims/README.md`
- Modify: `benchmarks/Cargo.toml` (deps `tprims-exec`, `strided-basic = { workspace = true, features = ["tprims-exec"] }`, `tensorcontract`, `rayon` non-optional for the new bin; `[[bin]] exec_entry`)

**Interfaces (Produces):**
```rust
// tprims_bench::threads
pub struct BenchThreads { pub requested: usize, pool: Option<rayon::ThreadPool> }
impl BenchThreads {
    /// Parse `--threads N` (default 1); reject conflicting RAYON_NUM_THREADS / OMP_NUM_THREADS /
    /// OPENBLAS_NUM_THREADS / TENSORCONTRACT_THREADS (set and != N → exit with error);
    /// build a bounded N-worker pool for N > 1.
    pub fn from_args() -> Self;
    pub fn with_exec<R>(&self, f: impl FnOnce(&tprims_exec::Exec<'_>, Option<&tprims_exec::Pool<'_>>) -> R) -> R;
    /// Print `# threads: requested=N pool=P budget=B` and assert P == B == N (N>1) or Serial (N==1).
    pub fn verify(&self);
}
```

- [ ] **Step 1: Harness module + unit test** (`benchmarks/src/threads/tests.rs`): conflicting-env detection is a pure function `fn conflicting_env(requested: usize, get: impl Fn(&str) -> Option<String>) -> Option<(String, String)>`; test: `RAYON_NUM_THREADS=2` with requested 4 → `Some(("RAYON_NUM_THREADS","2"))`; equal values or unset → `None`. Run `cargo test -j 16 -p tprims-bench --lib` → fail, implement, pass.

- [ ] **Step 2: `exec_entry` bench**, CSV `case,variant,threads,median_ns,samples`, `BENCH_RUNS` (default 200) and `BENCH_WARMUP` (20); `Instant`-based median like the existing `kernel_scaling` bin; `black_box` inputs/outputs. Cases:
  - `install_empty` at k = 1, 2, 4 (entry cost per width; k=1 must report ~0 and `entries == 0`);
  - `partition_empty` at k = 2, 4;
  - `broadcast_empty` at width 4 (full-pool SPMD entry);
  - `strided_map_f64` via `run_with_exec` at len 2^12 (below threshold) and 2^22 (above): 1T row with `Exec::serial()`, 4T row with the 4-worker pool;
  - `tensorcontract_gemm_f64` 512³ via `run_with` + `ExecSpmd` (copy the adapter into the bench) at width 1 and 4, correctness `CHECK` line comparing against the 1T result (max abs diff 0).
  After each case print the pool `entries`/`broadcasts` counters as a `# stats` line.

- [ ] **Step 3: Run 1T and 4T in the same session**

```bash
cargo build -j 16 --release -p tprims-bench --bin exec_entry
taskset -c 0 target/release/exec_entry --threads 1 | tee /tmp/claude-2000/-home-shinaoka-tensor4all/9e93717e-bbdb-4bfe-bcc4-cb6d49651a25/scratchpad/exec_entry_1t.csv
taskset -c 0-3 target/release/exec_entry --threads 4 | tee /tmp/claude-2000/-home-shinaoka-tensor4all/9e93717e-bbdb-4bfe-bcc4-cb6d49651a25/scratchpad/exec_entry_4t.csv
```
Expected: `# threads:` line verified; `strided_map_f64` at 2^22 and the 512³ GEMM faster at 4T than 1T (if not, that is a finding to record, not to hide); `install_empty k=1` entries 0.

- [ ] **Step 4: Record** results in `benchmarks/benchmarks/tprims/exec_entry/README.md`: tprims-rs commit, CPU (`lscpu | head -20`), taskset sets, profile (release, thin LTO), load average before/after, the two CSVs as a table. Label as a single-run observation, not a claim.

- [ ] **Step 5: Commit** — "tprims-bench: exec_entry benchmark at 1T and 4T with enforced thread counts".

### Task 6: Docs, gate, PR

- [ ] **Step 1:** Update `docs/decision-log.md` rows: "C ABI for execution" stays open; add "tprims-exec shape (Phase 1a)": `Pool` wrapper with SPMD mutex, `Exec::{Serial, Rayon}`, host callbacks deferred to Phase 2, strided integration through `run_with_exec` (no change to strided's `ExecContext` representation, which is `Copy` and lifetime-free), tensorcontract `Spmd` seam. Update `docs/architecture.md` execution-context section with the realized API (`Exec::Host` deferred).
- [ ] **Step 2:** Gate: fmt, clippy default + parallel features (add `strided-basic/tprims-exec` to the parallel lane in `.github/workflows/ci.yml` and AGENTS.md), `cargo test -j 16 --workspace`, parallel tests, `cargo test -j 16 -p strided-basic --features tprims-exec`, MSRV build, docs.
- [ ] **Step 3:** Push `phase1a-exec`, open PR with base `phase0-consolidation`, body listing the rulings and the benchmark table link; do not merge.
