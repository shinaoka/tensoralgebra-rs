//! BLAS-like operations for the tprims stack.
//!
//! Every operation takes an explicit [`tprims_exec::Exec`]: serial-size work
//! runs on the caller and never enters a pool, larger work enters the host's
//! borrowed pool once. Operands are `strided_view` views with signed strides;
//! conjugation is a per-operand flag ([`Conj`]).
//!
//! Batched GEMM has two strategies to compare: faer plus a loop over items,
//! and the TBLIS-style direct kernel of Lukas Devos's tensorprimitives-rs
//! (`tensorcontract`; D. A. Matthews, *High-Performance Tensor Contraction
//! without Transposition*, arXiv:1607.00291).
//!
//! # Examples
//!
//! ```
//! use strided_view::{StridedView, StridedViewMut};
//! use tprims_blas::{gemm, MatIn};
//! use tprims_exec::Exec;
//!
//! let a = [1.0, 2.0, 3.0, 4.0]; // [[1, 3], [2, 4]] column-major
//! let b = [1.0, 0.0, 0.0, 1.0];
//! let mut c = [0.0; 4];
//! let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
//! let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
//! let mut cv = StridedViewMut::new(&mut c, &[2, 2], &[1, 2], 0).unwrap();
//! gemm(&Exec::serial(), 1.0, MatIn::new(&av), MatIn::new(&bv), 0.0, &mut cv).unwrap();
//! assert_eq!(c, a);
//! ```
mod batched;
mod error;
mod gemm;
mod operand;
mod scalar;
mod tblis;
mod trsm;

pub use batched::{gemm_batched, BatchIn, BatchStrategy, Selected};
pub use error::{Error, Result};
pub use gemm::{gemm, GemmPolicy};
pub use operand::{Conj, MatIn};
pub use scalar::Scalar;
pub use trsm::{trsm, Diag, Op, Side, Uplo};
