use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// A Rayon pool lent by the host for the lifetime `'p`.
///
/// tprims never creates threads beyond this pool. Its size is fixed at
/// construction (Rayon pools cannot resize). The embedded mutex serializes
/// SPMD broadcasts so two co-scheduled kernels cannot interleave on the same
/// workers.
///
/// # Examples
///
/// ```
/// let tp = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
/// let pool = tprims_exec::Pool::borrow(&tp);
/// assert_eq!(pool.size(), 3);
/// assert_eq!(pool.stats().entries, 0);
/// ```
pub struct Pool<'p> {
    pub(crate) pool: &'p rayon::ThreadPool,
    pub(crate) spmd: Mutex<()>,
    entries: AtomicU64,
    broadcasts: AtomicU64,
    inline_runs: AtomicU64,
}

/// Counters of how often a [`Pool`] was entered, for tests and benchmarks.
///
/// # Examples
///
/// ```
/// assert_eq!(tprims_exec::PoolStats::default().entries, 0);
/// ```
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
        Self {
            pool,
            spmd: Mutex::new(()),
            entries: AtomicU64::new(0),
            broadcasts: AtomicU64::new(0),
            inline_runs: AtomicU64::new(0),
        }
    }

    /// Number of workers.
    pub fn size(&self) -> usize {
        self.pool.current_num_threads()
    }

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

    pub(crate) fn is_worker(&self) -> bool {
        self.pool.current_thread_index().is_some()
    }

    pub(crate) fn count_entry(&self) {
        self.entries.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn count_broadcast(&self) {
        self.broadcasts.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn count_inline(&self) {
        self.inline_runs.fetch_add(1, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for Pool<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("size", &self.size())
            .field("stats", &self.stats())
            .finish()
    }
}
