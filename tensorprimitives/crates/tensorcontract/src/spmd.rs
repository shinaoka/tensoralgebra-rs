//! Host-supplied co-scheduled threads for the threaded driver (tprims
//! addition, not upstream).
//!
//! The threaded driver is SPMD: its `p` participants share packed panels and
//! synchronise on barriers, so all `p` must run *at the same time*. By default
//! the crate gets them from `std::thread::scope` (or its opt-in pool). A host
//! that owns its threads implements [`Spmd`] and calls [`Plan::run_with`]
//! instead; the driver then never spawns a thread of its own.
//!
//! [`Plan::run_with`]: crate::Plan::run_with

/// A source of co-scheduled threads.
///
/// # Examples
///
/// ```
/// use tensorcontract::spmd::Spmd;
///
/// /// Runs everything on the caller: only width one is ever offered.
/// struct Inline;
/// impl Spmd for Inline {
///     fn width(&self) -> usize { 1 }
///     fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
///         if p != 1 { return false; }
///         f(0);
///         true
///     }
/// }
/// assert_eq!(Inline.width(), 1);
/// ```
pub trait Spmd: Sync {
    /// The number of participants this call may use (at least 1). It replaces
    /// [`Plan::threads`](crate::Plan::threads) and `TENSORCONTRACT_THREADS`.
    fn width(&self) -> usize;

    /// Run `f(t)` for every `t < p` **concurrently** and return `true`, or run
    /// nothing and return `false`; the driver then executes the contraction
    /// serially on the calling thread. `p` never exceeds [`Spmd::width`].
    fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool;

    /// Storage this host offers the driver, or none.
    ///
    /// The provider is borrowed for the duration of the call rather than
    /// required to be `'static`, because it belongs to whoever owns the
    /// threads: a `tprims_exec::Pool` lends its arena to every operation that
    /// runs on it, and a serial plan lends its own. A host that shares pools
    /// with someone else answers `None` and gets the driver's per-call buffers.
    fn workspace(&self) -> Option<&dyn tprims_gemm_kernel::WorkspaceProvider> {
        None
    }
}
