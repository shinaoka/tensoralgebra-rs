//! Adapter from [`tprims_exec::Exec`] to the neutral
//! [`HostExecution`](tprims_contract_traits::HostExecution) seam.

use std::any::TypeId;
use std::num::NonZeroUsize;
use std::ptr::NonNull;

use tprims_contract_traits::{Error, HostError, HostExecution, Par, Result};
use tprims_exec::{Exec, ExecError};

/// Marker whose `TypeId` names [`ExecHost`] in `HostExecution::native`.
struct ExecHostKind;

/// A [`tprims_exec::Exec`] lent as a neutral host.
///
/// Width-one work stays on the calling thread; wider work enters the borrowed
/// pool the way `Exec` does. A tprims plan executed through the neutral
/// interface recovers the `Exec` (and with it the pool's workspace) from this
/// adapter, so the trait path computes exactly what the concrete call computes.
///
/// # Examples
///
/// ```
/// use tprims_contract::ExecHost;
/// use tprims_contract_traits::HostExecution;
/// use tprims_exec::Exec;
/// let host = ExecHost::new(&Exec::serial());
/// assert_eq!(host.budget(), 1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ExecHost<'a> {
    exec: Exec<'a>,
}

impl<'a> ExecHost<'a> {
    /// Lend `exec` as a host.
    pub fn new(exec: &Exec<'a>) -> Self {
        Self { exec: *exec }
    }

    /// The lent execution context.
    pub fn exec(&self) -> &Exec<'a> {
        &self.exec
    }
}

fn map(e: ExecError) -> HostError {
    match e {
        ExecError::WidthExceedsPool { width, pool } => {
            HostError::WidthExceeds { width, limit: pool }
        }
        ExecError::WidthExceedsBudget { width, budget } => HostError::WidthExceeds {
            width,
            limit: budget,
        },
        _ => HostError::Unavailable,
    }
}

impl HostExecution for ExecHost<'_> {
    fn budget(&self) -> usize {
        self.exec.budget()
    }

    fn install(&self, k: usize, op: &mut (dyn FnMut(Par) + Send)) {
        self.exec.install(k, |par| {
            op(match par {
                tprims_exec::Par::Seq => Par::Seq,
                tprims_exec::Par::Threads(n) => Par::Threads(n),
            })
        })
    }

    fn for_each_partition(&self, k: usize, f: &(dyn Fn(usize) + Sync)) {
        self.exec.for_each_partition(k, f)
    }

    fn broadcast(
        &self,
        width: usize,
        f: &(dyn Fn(usize) + Sync),
    ) -> std::result::Result<(), HostError> {
        self.exec.broadcast(width, f).map_err(map)
    }

    fn native(&self, kind: TypeId) -> Option<NonNull<()>> {
        (kind == TypeId::of::<ExecHostKind>()).then(|| NonNull::from(self).cast())
    }
}

// `Par::Threads` carries a `NonZeroUsize` in both crates.
const _: fn(NonZeroUsize) -> tprims_exec::Par = tprims_exec::Par::Threads;

/// The `Exec` behind a neutral host.
///
/// An [`ExecHost`] hands back the context it lends. Any other host is served
/// only when its budget is one, on the calling thread; a wider foreign host
/// has no pool this implementation can enter, which fails explicitly.
///
/// # Errors
///
/// [`HostError::MissingCapability`] (as [`Error::Host`]) for a foreign host
/// with budget above one.
pub(crate) fn exec_of<'h>(host: &'h dyn HostExecution) -> Result<Exec<'h>> {
    if let Some(p) = host.native(TypeId::of::<ExecHostKind>()) {
        // SAFETY: `native` answers this kind only from `ExecHost::native`, which
        // returns its own address, valid for the borrow `'h` of `host`. The
        // lifetime parameter is erased by the cast; the stored `Exec` outlives
        // that borrow because `ExecHost<'a>` can only be named while `'a` is live.
        let h: &ExecHost<'h> = unsafe { p.cast::<ExecHost<'h>>().as_ref() };
        return Ok(h.exec);
    }
    if host.budget() <= 1 {
        return Ok(Exec::serial());
    }
    Err(Error::Host(HostError::MissingCapability(
        "tprims-contract executes on tprims_exec::Exec (wrap it in ExecHost) or a serial host",
    )))
}
