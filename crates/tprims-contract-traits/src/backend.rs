use std::fmt::Debug;

use strided_view::{StridedView, StridedViewMut};

use crate::{HostExecution, PlanningBudget, Problem, Requirements, Result};

/// The storage scalar of a contraction: plain data the views can hand across
/// threads. Implemented for every type that qualifies; a backend states which
/// concrete types it supports by implementing [`ContractionBackend`] for them.
pub trait Scalar: Copy + PartialEq + Send + Sync + 'static {}

impl<T: Copy + PartialEq + Send + Sync + 'static> Scalar for T {}

/// What a prepared plan reports about itself, independent of the backend.
///
/// # Examples
///
/// ```
/// let d = tprims_contract_traits::Diagnostics::new("naive", "loop-nest");
/// assert_eq!(d.materialized, [false; 3]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Diagnostics {
    /// Backend identity.
    pub backend: &'static str,
    /// Selected algorithm identifier (backend-defined, stable per backend).
    pub algorithm: &'static str,
    /// Which of A, B, C are copied into compact buffers on every execution.
    pub materialized: [bool; 3],
}

impl Diagnostics {
    /// Diagnostics of a plan that copies nothing.
    pub fn new(backend: &'static str, algorithm: &'static str) -> Self {
        Self {
            backend,
            algorithm,
            materialized: [false; 3],
        }
    }

    /// Record which operands are materialized.
    #[must_use]
    pub fn with_materialized(mut self, materialized: [bool; 3]) -> Self {
        self.materialized = materialized;
        self
    }
}

/// A prepared contraction for fixed layouts, reusable across executions.
///
/// The plan snapshots layout and configuration metadata; it holds no operand
/// payload or pointer, no worker thread, and nothing borrowed from the
/// backend or the executor used to prepare it. It never outlives resources it
/// needs: persistent provider state is owned by the plan.
pub trait PreparedContraction<T: Scalar> {
    /// `C = alpha * dot_general(op(A), op(B)) + beta * C` on views with exactly
    /// the prepared layouts.
    ///
    /// `beta == 0` reads no previous C value; `alpha == 0` or an empty
    /// contraction reads no A or B value. Views are valid initialized storage.
    ///
    /// # Errors
    ///
    /// [`Error::LayoutMismatch`](crate::Error::LayoutMismatch) when a view
    /// differs from the plan and [`Error::Host`](crate::Error::Host) when the host
    /// lacks a required capability: both before any write.
    /// [`Error::Backend`](crate::Error::Backend) for an implementation failure,
    /// after which C may be partially written.
    fn execute_into_accum(
        &self,
        host: &dyn HostExecution,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        c: &mut StridedViewMut<'_, T>,
    ) -> Result<()>;

    /// Immutable plan diagnostics.
    fn diagnostics(&self) -> &Diagnostics;
}

/// A prepared plan that can be stored in a runtime slot and shared across
/// threads. Concurrent executions on independent outputs are supported.
pub type BoxedPlan<T> = Box<dyn PreparedContraction<T> + Send + Sync>;

/// A complete contraction implementation for storage type `T`.
///
/// Object safe for each fixed `T`: a runtime slot holds
/// `Box<dyn ContractionBackend<T>>` and dispatches the dtype in the consumer.
pub trait ContractionBackend<T: Scalar>: Debug + Send + Sync {
    /// Backend identity.
    fn id(&self) -> &'static str;

    /// Validate `problem`, check `requirements` and build a reusable plan.
    ///
    /// `budget` is advisory: a backend may use it to size planning choices, but
    /// the plan runs correctly on any host width and no backend may reject a
    /// host later because it differs from the budget.
    ///
    /// # Errors
    ///
    /// Validation errors ([`Error::Config`](crate::Error::Config),
    /// [`Error::Shape`](crate::Error::Shape),
    /// [`Error::AliasedOutput`](crate::Error::AliasedOutput)),
    /// [`Error::WouldMaterialize`](crate::Error::WouldMaterialize) under
    /// `no_materialize`, [`Error::Unsupported`](crate::Error::Unsupported) for
    /// anything the backend cannot do. No output exists at this point.
    fn prepare(
        &self,
        problem: &Problem,
        requirements: &Requirements,
        budget: &PlanningBudget,
    ) -> Result<BoxedPlan<T>>;
}
