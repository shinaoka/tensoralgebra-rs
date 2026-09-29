/// Errors from tprims-linalg. Validation happens before any write.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
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
    /// A square matrix was required.
    #[error("expected a square matrix, got {rows}x{cols}")]
    NotSquare {
        /// Rows.
        rows: usize,
        /// Columns.
        cols: usize,
    },
    /// Operand shapes are inconsistent.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// Cholesky found a non-positive pivot.
    #[error("matrix is not positive definite (pivot {index})")]
    NotPositiveDefinite {
        /// Failing pivot.
        index: usize,
    },
    /// A solve met a (numerically) zero pivot.
    #[error("matrix is singular (pivot {index})")]
    Singular {
        /// Failing pivot.
        index: usize,
    },
    /// An iterative decomposition did not converge.
    #[error("decomposition did not converge")]
    NoConvergence,
    /// The input contains NaN or infinity.
    #[error("input contains non-finite values")]
    NonFinite,
    /// An output view has zero or overlapping strides.
    #[error("output has zero or overlapping strides")]
    AliasedOutput,
}

/// Result alias for tprims-linalg.
pub type Result<T> = core::result::Result<T, Error>;
