//! The public description of a contraction: front ends, the validated
//! [`Problem`], the error type and (later) the backend trait.

mod dot_general;
mod error;
mod labels;
mod problem;
mod validate;

pub use dot_general::{DotGeneral, DotShape};
pub use error::{
    AliasError, ConfigError, Error, LayoutError, OperandId, Result, ShapeError, Unsupported,
};
pub use labels::Labels;
pub use problem::{CSpec, DType, LayoutSpec, Op, OperandSpec, Problem, RoleAxis, Roles, Span};
