//! Implementation-independent interface of binary tensor contraction
//! (`dot_general` semantics) for the tprims stack.
//!
//! This crate holds only the reusable public contracts: the problem
//! description ([`Problem`], [`DotGeneral`], [`Layout`], [`Conj`]), the
//! canonical validation ([`validate_layouts`]), the typed [`Error`], the
//! object-safe [`ContractionBackend`] / [`PreparedContraction`] traits and the
//! minimal borrowed [`HostExecution`] seam. It depends on no implementation,
//! executor runtime or kernel layer; implementations (for example
//! `tprims-contract`) depend on it.
//!
//! The operation is `C = alpha * dot_general(op(A), op(B)) + beta * C`, output
//! axes `[lhs_free..., rhs_free..., batch...]`.
//!
//! ```
//! use tprims_contract_traits::*;
//!
//! let dot = DotGeneral::new(&[1], &[0], &[], &[]);
//! let lay = Layout::new(&[2, 2], &[1, 2]);
//! let problem = Problem::new(dot, lay.clone(), lay.clone(), lay, (Conj::No, Conj::No));
//! let v = problem.validate::<f64>().unwrap();
//! assert_eq!(v.shape.out_dims, vec![2, 2]);
//! ```
mod backend;
mod error;
mod host;
mod problem;

pub use backend::{BoxedPlan, ContractionBackend, Diagnostics, PreparedContraction, Scalar};
pub use error::{Error, Result};
pub use host::{HostError, HostExecution, NativeHost, Par, SerialHost};
pub use problem::{
    is_injective_layout, validate_layouts, Conj, DotGeneral, Layout, PlanningBudget, Problem,
    Requirements, Shape, Validated,
};
