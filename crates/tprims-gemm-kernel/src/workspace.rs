//! Owning workspace: page-aligned, grow-only, reused per worker.
//!
//! A workspace belongs to exactly one owner — a `tprims_exec::Pool`, a serial
//! plan, or a test — and is *lent* to the driver for one call at a time. There
//! is deliberately no process-global arena: storage handed to a team is
//! returned to the provider that issued it, and a second owner can never see
//! it.
//!
//! Two payload shapes:
//!
//! * **worker buffers** — the packed A block, the accumulator tile and the
//!   scratch vector of one thread. They are allocated by the thread that will
//!   write them, which is what makes their first touch local to the node that
//!   keeps using them; a thread-local handle names its slot in the owner, and a
//!   handle outlives nothing: the storage is the owner's, the handle is only an
//!   index, and a slot that has been trimmed is simply re-created.
//! * **team sets** — the shared packed B panel, the team's scatter vectors and
//!   its barriers. They are taken exclusively for the duration of a call and
//!   returned on drop, so two concurrent executes can never share one.
//!
//! Re-entrancy is answered by allocating fresh call-local buffers rather than
//! blocking on the slot the same thread is already using: the inner call needs
//! somewhere to write, and waiting for itself would deadlock. Only the outer
//! call reuses storage, so the steady state still allocates nothing.
//!
//! `ArenaProvider::traced()` additionally records every growth's thread and
//! size, which is how the tests prove that a worker's buffers were allocated by
//! the worker. Tracing is off by default and costs a `bool` when off.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};

/// The 4096-byte alignment every buffer in the workspace has.
pub const PAGE: usize = 4096;

/// Bytes and counts one execute needs. Buffer sizes are bytes because the
/// provider is element-type erased: the driver knows the element type and
/// converts its real counts once.
///
/// A zero entry never allocates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceReq {
    /// This thread's packed A block.
    pub a_bytes: usize,
    /// This thread's accumulator tile, including any induced-method scratch.
    pub tile_bytes: usize,
    /// Elements reserved for this thread's scatter scratch.
    pub worker_scatter: usize,
    /// The team's shared packed B panel.
    pub b_bytes: usize,
    /// Elements reserved for the team's block-scatter vectors.
    pub team_scatter: usize,
    /// Barriers the team needs, or zero when nothing is shared.
    pub barriers: usize,
}

/// Page-aligned, grow-only, never zeroed storage.
///
/// The caller writes before reading, which is the contract the packing and
/// kernel code already has; zeroing would only double the traffic.
#[derive(Default, Debug)]
pub struct PageBuf {
    ptr: *mut u8,
    cap: usize,
    /// Owner key this buffer reports growths under, or zero when untraced.
    traced: u64,
}

// SAFETY: the buffer owns its allocation and hands out pointers only through
// `&mut self` or an explicit, contracted call.
unsafe impl Send for PageBuf {}

impl PageBuf {
    /// A buffer that reports its growths under `owner`, for tests that prove
    /// where storage was first touched. Zero means untraced.
    pub fn traced(owner: u64) -> Self {
        Self {
            traced: owner,
            ..Self::default()
        }
    }

    /// A pointer to at least `bytes` writable, page-aligned bytes, growing the
    /// allocation if needed and reusing it otherwise. `ensure(0)` returns a
    /// dangling but well-aligned pointer and allocates nothing.
    pub fn ensure(&mut self, bytes: usize) -> *mut u8 {
        if bytes == 0 {
            return std::ptr::NonNull::<u8>::dangling().as_ptr();
        }
        if bytes > self.cap {
            // Grow by doubling, so a sequence of slightly larger calls does not
            // reallocate every time.
            let want = bytes.next_power_of_two();
            let layout = std::alloc::Layout::from_size_align(want, PAGE).expect("workspace layout");
            // SAFETY: `want` is nonzero, and any old block was allocated with
            // this same alignment and a power-of-two size.
            let new = unsafe {
                match self.ptr.is_null() {
                    true => std::alloc::alloc(layout),
                    false => {
                        let old = std::alloc::Layout::from_size_align_unchecked(self.cap, PAGE);
                        std::alloc::realloc(self.ptr, old, want)
                    }
                }
            };
            assert!(
                !new.is_null(),
                "workspace allocation of {want} bytes failed"
            );
            self.ptr = new;
            self.cap = want;
            if self.traced != 0 {
                record(self.traced, want);
            }
        }
        self.ptr
    }

    /// Pointer to the current allocation, or a dangling pointer when empty.
    pub fn as_ptr(&self) -> *mut u8 {
        if self.ptr.is_null() {
            std::ptr::NonNull::<u8>::dangling().as_ptr()
        } else {
            self.ptr
        }
    }

    /// Bytes currently allocated.
    pub fn cap_bytes(&self) -> usize {
        self.cap
    }

    /// Release the allocation. Callers must hold the buffer exclusively.
    pub fn trim(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated with this alignment and a power-of-two size.
            unsafe {
                std::alloc::dealloc(
                    self.ptr,
                    std::alloc::Layout::from_size_align_unchecked(self.cap, PAGE),
                )
            };
            self.ptr = std::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for PageBuf {
    fn drop(&mut self) {
        self.trim();
    }
}

/// One thread's buffers, as the owner keeps them.
#[derive(Debug)]
struct WorkerSlot {
    a: PageBuf,
    tile: PageBuf,
    scratch: Vec<i64>,
    /// Set while a call is running on this thread, so a re-entrant call on the
    /// same thread takes the fresh-buffer path instead of aliasing.
    borrowed: bool,
}

impl WorkerSlot {
    /// `owner` is the tracing key, or zero for an untraced owner.
    fn new(owner: u64) -> Self {
        Self {
            a: PageBuf::traced(owner),
            tile: PageBuf::traced(owner),
            scratch: Vec::new(),
            borrowed: false,
        }
    }
}

/// A team's shared storage: one packed B panel, the scatter vectors and the
/// barriers that publish the panel between its threads.
#[derive(Debug, Default)]
pub struct TeamSet {
    /// The packed B panel.
    pub b: PageBuf,
    /// Block-scatter vectors for this call, capacity-reused.
    pub scatter: Vec<i64>,
    /// One barrier per column group when `pm > 1`, else empty.
    pub barriers: Vec<Barrier>,
    pm: usize,
    pn: usize,
}

impl TeamSet {
    fn prepare(&mut self, req: &WorkspaceReq, pm: usize, pn: usize) {
        self.scatter.clear();
        self.scatter.reserve(req.team_scatter);
        if (self.pm, self.pn) != (pm, pn) {
            // A barrier cannot be reset, so a differently shaped team needs new
            // ones; that is why the free list keeps sets and rebuilds here.
            self.barriers = (0..req.barriers).map(|_| Barrier::new(pm)).collect();
            self.pm = pm;
            self.pn = pn;
        }
    }
}

/// An exclusive loan of a [`TeamSet`], returned to its owner on drop.
#[derive(Debug)]
pub struct TeamLease<'a> {
    owner: &'a ArenaProvider,
    set: Option<TeamSet>,
}

impl std::ops::Deref for TeamLease<'_> {
    type Target = TeamSet;
    fn deref(&self) -> &TeamSet {
        self.set.as_ref().expect("team lease holds its set")
    }
}

impl std::ops::DerefMut for TeamLease<'_> {
    fn deref_mut(&mut self) -> &mut TeamSet {
        self.set.as_mut().expect("team lease holds its set")
    }
}

impl TeamLease<'_> {
    /// Size the shared panel and return its pointer, accounting the growth to
    /// the owner. The driver sizes it because only the driver knows the element
    /// type; the provider keeps the storage.
    pub fn panel(&mut self, bytes: usize) -> *mut u8 {
        let before = self.b.cap_bytes();
        let ptr = self.b.ensure(bytes);
        let after = self.b.cap_bytes();
        if after > before {
            self.owner
                .leased
                .fetch_add(after - before, Ordering::Relaxed);
        }
        ptr
    }
}

impl Drop for TeamLease<'_> {
    fn drop(&mut self) {
        if let Some(set) = self.set.take() {
            self.owner
                .leased
                .fetch_sub(set.b.cap_bytes(), Ordering::Relaxed);
            lock(&self.owner.teams).push(set);
        }
    }
}

/// Storage an execution may borrow. Implemented by the owner; the driver only
/// ever sees this trait, so nothing here depends on the driver.
pub trait WorkspaceProvider: Sync {
    /// Run `f` with this thread's A block, tile and scatter scratch. A
    /// re-entrant call on the same thread gets fresh buffers instead of the
    /// ones its outer call is using.
    fn with_worker(&self, req: &WorkspaceReq, f: &mut dyn FnMut(*mut u8, *mut u8, &mut Vec<i64>));

    /// Take the team set for `pm x pn`, exclusively, until the lease drops.
    fn take_team(&self, req: &WorkspaceReq, pm: usize, pn: usize) -> TeamLease<'_>;

    /// Release idle storage. Live leases and borrowed worker slots are kept.
    fn trim(&self);
}

/// A workspace owner: per-thread worker slots plus a free list of team sets.
///
/// One per owner — never process-global — so a lease always returns to the
/// provider that issued it and no second owner can borrow it.
#[derive(Debug)]
pub struct ArenaProvider {
    /// Process-unique and never recycled, so a thread-local handle can never be
    /// confused with another provider that happens to share an address.
    key: u64,
    workers: Mutex<HashMap<u64, Box<WorkerSlot>>>,
    teams: Mutex<Vec<TeamSet>>,
    next_worker: AtomicU64,
    /// Panel bytes currently leased out, which the free list no longer holds.
    leased: AtomicUsize,
    traced: u64,
}

impl Default for ArenaProvider {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// `(provider key, worker slot)` pairs this thread has used. Thread-local
    /// by construction, so a slot is never handed to a different thread after
    /// its owner exits: the handle dies with the thread.
    static HANDLES: std::cell::RefCell<Vec<(u64, u64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

static NEXT_PROVIDER: AtomicU64 = AtomicU64::new(1);
static TRACE: Mutex<Vec<(u64, std::thread::ThreadId, usize)>> = Mutex::new(Vec::new());

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn record(owner: u64, bytes: usize) {
    lock(&TRACE).push((owner, std::thread::current().id(), bytes));
}

impl ArenaProvider {
    /// A fresh owner with no storage.
    pub fn new() -> Self {
        Self {
            key: NEXT_PROVIDER.fetch_add(1, Ordering::Relaxed),
            workers: Mutex::new(HashMap::new()),
            teams: Mutex::new(Vec::new()),
            next_worker: AtomicU64::new(0),
            leased: AtomicUsize::new(0),
            traced: 0,
        }
    }

    /// A fresh owner that records where its storage is first touched, readable
    /// through [`ArenaProvider::trace_take`].
    pub fn traced() -> Self {
        let mut arena = Self::new();
        arena.traced = arena.key;
        arena
    }

    /// Take this owner's recorded buffer growths, leaving its log empty.
    ///
    /// Empty unless the owner was built with [`ArenaProvider::traced`].
    pub fn trace_take(&self) -> Vec<(std::thread::ThreadId, usize)> {
        let mut mine = Vec::new();
        lock(&TRACE).retain(|(owner, thread, bytes)| {
            if *owner == self.key {
                mine.push((*thread, *bytes));
                false
            } else {
                true
            }
        });
        mine
    }

    /// Bytes currently retained, for accounting in tests and diagnostics.
    pub fn retained_bytes(&self) -> usize {
        let workers: usize = lock(&self.workers)
            .values()
            .map(|s| s.a.cap_bytes() + s.tile.cap_bytes())
            .sum();
        let teams: usize = lock(&self.teams).iter().map(|t| t.b.cap_bytes()).sum();
        workers + teams + self.leased.load(Ordering::Relaxed)
    }

    fn slot(&self, index: u64) -> *mut WorkerSlot {
        let mut workers = lock(&self.workers);
        match workers.get_mut(&index) {
            // SAFETY: the `Box` keeps the slot's address stable across map
            // growth, and only this thread's handle can name this index.
            Some(slot) => &mut **slot as *mut WorkerSlot,
            None => std::ptr::null_mut(),
        }
    }

    fn handle_for_this_thread(&self) -> u64 {
        if let Some(index) = HANDLES.with(|h| {
            h.borrow()
                .iter()
                .find(|(k, _)| *k == self.key)
                .map(|(_, i)| *i)
        }) {
            return index;
        }
        let index = self.next_worker.fetch_add(1, Ordering::Relaxed);
        lock(&self.workers).insert(index, Box::new(WorkerSlot::new(self.traced)));
        HANDLES.with(|h| h.borrow_mut().push((self.key, index)));
        index
    }
}

impl WorkspaceProvider for ArenaProvider {
    fn with_worker(&self, req: &WorkspaceReq, f: &mut dyn FnMut(*mut u8, *mut u8, &mut Vec<i64>)) {
        let index = self.handle_for_this_thread();
        let slot = self.slot(index);
        if slot.is_null() {
            return fresh_worker(req, f);
        }
        // SAFETY: the slot belongs to this thread alone (the handle is
        // thread-local and the map entry never moves), and `borrowed` sends a
        // re-entrant call down the fresh path instead of aliasing it.
        let slot = unsafe { &mut *slot };
        if slot.borrowed {
            return fresh_worker(req, f);
        }
        slot.borrowed = true;
        slot.a.ensure(req.a_bytes);
        slot.tile.ensure(req.tile_bytes);
        slot.scratch.clear();
        slot.scratch.reserve(req.worker_scatter);
        let (a, tile) = (slot.a.as_ptr(), slot.tile.as_ptr());
        f(a, tile, &mut slot.scratch);
        slot.borrowed = false;
    }

    fn take_team(&self, req: &WorkspaceReq, pm: usize, pn: usize) -> TeamLease<'_> {
        let mut set = lock(&self.teams).pop().unwrap_or_else(|| TeamSet {
            b: PageBuf::traced(self.traced),
            ..TeamSet::default()
        });
        set.prepare(req, pm, pn);
        TeamLease {
            owner: self,
            set: Some(set),
        }
    }

    fn trim(&self) {
        lock(&self.teams).clear();
        let mut workers = lock(&self.workers);
        for slot in workers.values_mut() {
            if !slot.borrowed {
                slot.a.trim();
                slot.tile.trim();
                slot.scratch = Vec::new();
            }
        }
    }
}

/// Buffers for one call that either re-entered its owner or has none. They are
/// freed when the call returns, which is the price of re-entrancy.
fn fresh_worker(req: &WorkspaceReq, f: &mut dyn FnMut(*mut u8, *mut u8, &mut Vec<i64>)) {
    let mut a = PageBuf::default();
    let mut tile = PageBuf::default();
    let mut scratch = Vec::new();
    a.ensure(req.a_bytes);
    tile.ensure(req.tile_bytes);
    scratch.reserve(req.worker_scatter);
    f(a.as_ptr(), tile.as_ptr(), &mut scratch);
}
