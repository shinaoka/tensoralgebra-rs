//! Binary tensor contraction with batch indices (`dot_general` semantics)
//! for the tprims stack.
//!
//! [`DotGeneral`] names the contracting and batch axes of both operands;
//! free axes keep their order and the output is
//! `[lhs_free..., rhs_free..., batch...]`. A [`ContractPlan`] validates the
//! problem once and runs one of two strategies:
//!
//! - [`Strategy::PermuteGemm`]: fuse index groups into strided batched
//!   matrices and call `tprims_blas::gemm_batched`; operands whose strides do
//!   not fuse are copied once into compact buffers (reported, and refused
//!   under [`Flags::no_materialize`]). The approach follows tenferro-rs's CPU
//!   `dot_general` (reimplemented here, not copied).
//! - [`Strategy::Tblis`]: the TBLIS-style direct contraction of Lukas
//!   Devos's tensorprimitives-rs (`tensorcontract`; D. A. Matthews,
//!   *High-Performance Tensor Contraction without Transposition*,
//!   arXiv:1607.00291), which packs general strides straight into the
//!   micro-kernel panels.
//!
//! # Examples
//!
//! ```
//! use strided_view::{StridedView, StridedViewMut};
//! use tprims_blas::Conj;
//! use tprims_contract::{ContractPlan, DotGeneral, Flags, Strategy};
//! use tprims_exec::Exec;
//!
//! // C[i, k] = sum_j A[i, j] B[j, k]
//! let cfg = DotGeneral::new(&[1], &[0], &[], &[]);
//! let a = [1.0, 2.0, 3.0, 4.0];
//! let b = [1.0, 0.0, 0.0, 1.0];
//! let mut c = [0.0; 4];
//! let plan = ContractPlan::<f64>::new(&cfg, (&[2, 2], &[1, 2]), (&[2, 2], &[1, 2]), (&[2, 2], &[1, 2]),
//!     (Conj::No, Conj::No), Strategy::Auto, Flags::default()).unwrap();
//! let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
//! let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
//! let mut cv = StridedViewMut::new(&mut c, &[2, 2], &[1, 2], 0).unwrap();
//! plan.execute(&Exec::serial(), 1.0, &av, &bv, 0.0, &mut cv).unwrap();
//! assert_eq!(c, a);
//! ```
mod config;
mod error;
mod permute_gemm;
mod plan;
mod tblis;
mod util;

pub use config::{DotGeneral, Shape};
pub use error::{Error, Result};
pub use plan::{ContractPlan, Flags, Selected, Strategy};
