/// Errors from tprims-blas operations. Validation happens before any write.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// An operand has the wrong rank.
    #[error("{operand}: expected rank {expected}, got {got}")]
    Rank {
        /// Which operand.
        operand: &'static str,
        /// Required rank.
        expected: usize,
        /// Given rank.
        got: usize,
    },
    /// Operand shapes are inconsistent.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// The output has zero or overlapping strides, so distinct elements would
    /// share memory.
    #[error("output has zero or overlapping strides")]
    AliasedOutput,
    /// The TBLIS-style strategy rejected the problem.
    #[error(transparent)]
    Contract(#[from] tensorcontract::Error),
    /// The chosen engine or kernel family cannot run this problem.
    #[error(transparent)]
    Select(#[from] tprims_kernel::SelectError),
}

/// Result alias for tprims-blas.
pub type Result<T> = core::result::Result<T, Error>;
