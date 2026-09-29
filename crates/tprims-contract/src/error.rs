/// Errors from tprims-contract. Validation happens before any write.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The axis configuration is invalid (repeated, out of range, unequal
    /// list lengths).
    #[error("invalid dot_general configuration: {0}")]
    Config(String),
    /// Extents do not match.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// A view passed to `execute` differs from the layout the plan was built for.
    #[error("layout differs from the plan: {0}")]
    LayoutMismatch(String),
    /// The output has zero or overlapping strides.
    #[error("output has zero or overlapping strides")]
    AliasedOutput,
    /// The strategy would copy operands but `no_materialize` was requested.
    #[error("contraction needs materialized operands {operands:?} (A, B, C)")]
    WouldMaterialize {
        /// Which of A, B, C would be copied.
        operands: [bool; 3],
    },
    /// A lower layer failed.
    #[error("{0}")]
    Backend(String),
}

/// Result alias for tprims-contract.
pub type Result<T> = core::result::Result<T, Error>;
