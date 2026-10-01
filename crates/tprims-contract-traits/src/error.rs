/// Errors shared by every contraction backend.
///
/// Validation and unsupported-requirement failures happen before any output
/// element is written. [`Error::Backend`] keeps the implementation's own error
/// as a source; it is the only variant that may follow a partially executed
/// operation, and no rollback is promised then.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The axis configuration is invalid (repeated, out of range, unequal
    /// list lengths).
    #[error("invalid dot_general configuration: {0}")]
    Config(String),
    /// Extents do not match, or a size computation overflows.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// A view passed to `execute` differs from the layout the plan was built for.
    #[error("layout differs from the plan: {0}")]
    LayoutMismatch(String),
    /// The output has zero or overlapping strides.
    #[error("output has zero or overlapping strides")]
    AliasedOutput,
    /// The plan would copy operands but `no_materialize` was requested.
    #[error("contraction needs materialized operands {operands:?} (A, B, C)")]
    WouldMaterialize {
        /// Which of A, B, C would be copied.
        operands: [bool; 3],
    },
    /// The backend does not support the requested dtype, layout, operation or
    /// forced requirement. Nothing was written, so a consumer policy may try
    /// another provider.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The host cannot supply what the plan needs.
    #[error("host execution: {0}")]
    Host(#[from] crate::HostError),
    /// The implementation failed; its error is the [`source`](std::error::Error::source)
    /// payload. Output may be partially written when this follows the start of
    /// the computation.
    #[error("{0}")]
    Backend(Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl Error {
    /// Wrap an implementation error, keeping it downcastable.
    pub fn backend(e: impl Into<Box<dyn std::error::Error + Send + Sync + 'static>>) -> Self {
        Error::Backend(e.into())
    }

    /// Whether this is a no-mutation refusal after which another provider may
    /// be tried: [`Error::Unsupported`], [`Error::WouldMaterialize`] or a
    /// missing host capability.
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Error::Unsupported(_) | Error::WouldMaterialize { .. } | Error::Host(_)
        )
    }
}

/// Result alias for the contraction interface.
pub type Result<T> = core::result::Result<T, Error>;
