/// Errors from building or using an [`Exec`](crate::Exec).
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecError {
    /// A thread budget of zero was requested.
    #[error("thread budget must be at least 1")]
    ZeroBudget,
    /// An SPMD width larger than the pool was requested.
    #[error("SPMD width {width} exceeds pool size {pool}")]
    WidthExceedsPool {
        /// Requested width.
        width: usize,
        /// Workers in the pool.
        pool: usize,
    },
    /// Co-scheduled execution cannot be guaranteed here (serial context, or
    /// the caller is already a worker of the pool).
    #[error("co-scheduled execution unavailable in this context")]
    Unavailable,
}
