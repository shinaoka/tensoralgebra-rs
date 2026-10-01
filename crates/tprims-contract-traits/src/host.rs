use std::any::TypeId;
use std::num::NonZeroUsize;
use std::ptr::NonNull;

/// Why a host cannot give what a backend asked for.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum HostError {
    /// Co-scheduled execution cannot be guaranteed here (serial host with
    /// width above one, or the caller is already inside the host's workers).
    /// Nothing ran.
    #[error("co-scheduled execution unavailable in this host")]
    Unavailable,
    /// The requested width exceeds what the host allows. Nothing ran.
    #[error("requested width {width} exceeds the host limit {limit}")]
    WidthExceeds {
        /// Requested width.
        width: usize,
        /// Largest width the host grants.
        limit: usize,
    },
    /// The host cannot serve this backend at the requested budget; the backend
    /// requires an executor binding the host does not offer.
    #[error("host lacks the capability this backend requires: {0}")]
    MissingCapability(&'static str),
}

/// Parallelism granted to an [`HostExecution::install`] region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Par {
    /// Run serially on the current thread.
    Seq,
    /// Up to this many workers.
    Threads(NonZeroUsize),
}

impl Par {
    /// Worker count (1 for `Seq`).
    pub fn threads(self) -> usize {
        match self {
            Par::Seq => 1,
            Par::Threads(n) => n.get(),
        }
    }
}

/// A borrowed host-execution capability: the minimal executor seam a
/// contraction backend needs. It carries no kernel family, packing buffer or
/// runtime type. Concrete runtimes adapt to it outside this crate.
///
/// All methods are called a bounded number of times per operation
/// execution, never inside a register-tile loop. `Self` is object safe.
pub trait HostExecution {
    /// Most threads one operation may occupy; at least 1. One budget governs
    /// outer batch and inner work.
    fn budget(&self) -> usize;

    /// Run `op` with `min(k, budget)` threads of parallelism (`k >= 1`).
    ///
    /// Width one runs `op(Par::Seq)` inline on the caller without entering any
    /// runtime. A wider region enters the host once, or runs in place when the
    /// caller is already one of its workers.
    fn install(&self, k: usize, op: &mut (dyn FnMut(Par) + Send));

    /// Barrier-free partition: runs `f(i)` exactly once for each `i < k` on at
    /// most `budget` workers. Tasks may not run concurrently, so `f` must never
    /// wait on another index.
    fn for_each_partition(&self, k: usize, f: &(dyn Fn(usize) + Sync));

    /// Co-scheduled SPMD execution: `f(t)` for every `t < width` runs
    /// concurrently on distinct workers (barriers counting `width`
    /// participants are allowed), or nothing runs and an error is returned.
    /// Ordinary task submission does not satisfy this. Width one runs `f(0)`
    /// inline.
    ///
    /// # Errors
    ///
    /// [`HostError::Unavailable`] when co-scheduling cannot be guaranteed
    /// (including nested entry from a worker), [`HostError::WidthExceeds`]
    /// when `width` exceeds the host.
    fn broadcast(&self, width: usize, f: &(dyn Fn(usize) + Sync)) -> Result<(), HostError>;

    /// Implementation-private downcast hook: `Some` only for an adapter that
    /// implements the `unsafe` [`NativeHost`] contract. A safe implementation
    /// cannot forge one, so the default (`None`) is the only thing it can say.
    fn native_host(&self) -> Option<&dyn NativeHost> {
        None
    }
}

/// A host that can hand its own native context back to the backend that
/// understands it.
///
/// # Safety
///
/// `native(kind)` must return `Some(p)` only when `kind` is the `TypeId` of a
/// type `K` chosen by the implementor, `p` is the address of a live value of
/// the type that `K` documents (for the adapter this is `Self`), and `p`
/// stays valid for the borrow of `&self`. It must return `None` for every
/// other `kind`. A backend dereferences `p` as exactly that documented type.
pub unsafe trait NativeHost {
    /// The address of the native context named by `kind`, or `None`.
    fn native(&self, kind: TypeId) -> Option<NonNull<()>>;
}

/// The calling thread only: budget one, no runtime.
///
/// # Examples
///
/// ```
/// use tprims_contract_traits::{HostExecution, SerialHost};
/// let n = std::sync::atomic::AtomicUsize::new(0);
/// SerialHost.for_each_partition(4, &|_| { n.fetch_add(1, std::sync::atomic::Ordering::Relaxed); });
/// assert_eq!(n.into_inner(), 4);
/// assert!(SerialHost.broadcast(2, &|_| {}).is_err());
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct SerialHost;

impl HostExecution for SerialHost {
    fn budget(&self) -> usize {
        1
    }

    fn install(&self, _k: usize, op: &mut (dyn FnMut(Par) + Send)) {
        op(Par::Seq)
    }

    fn for_each_partition(&self, k: usize, f: &(dyn Fn(usize) + Sync)) {
        (0..k).for_each(f)
    }

    fn broadcast(&self, width: usize, f: &(dyn Fn(usize) + Sync)) -> Result<(), HostError> {
        match width {
            0 => Ok(()),
            1 => {
                f(0);
                Ok(())
            }
            _ => Err(HostError::Unavailable),
        }
    }
}
