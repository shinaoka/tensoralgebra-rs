//! Dense tensor primitives, as one dependency.
//!
//! This crate is a **facade**: it contains no algorithms of its own and exists
//! so that a caller can depend on `tensorprimitives` and get whichever tensor
//! primitives they need, rather than tracking one crate per operation. The
//! implementations live in their own crates and can also be depended on
//! directly, which is the right choice for a library that needs exactly one of
//! them.
//!
//! | primitive | crate | feature | status |
//! |---|---|---|---|
//! | contraction | [`tensorcontract`] | `contract` (default) | implemented |
//! | transposition | `tensortranspose` | `transpose` | **not yet written** |
//!
//! The pattern is `futures` over `futures-core`, or `blas-src` over its
//! providers: a thin, stable name at the top, the substance underneath, and the
//! choice of what to compile left to the caller through features.
//!
//! # Which crate should you depend on?
//!
//! * **An application, or a library that wants the whole family** — depend on
//!   `tensorprimitives` and enable the features you use.
//! * **A library that needs contraction only** — depend on [`tensorcontract`]
//!   directly. You then compile exactly what you use, and this crate's feature
//!   set cannot affect you. Cargo features are additive across a dependency
//!   graph, so a library that enables features on a facade imposes them on every
//!   other consumer of that facade; the BLAS ecosystem learned this the hard way,
//!   which is why `blas-src`'s own documentation tells libraries not to select a
//!   backend.
//!
//! # Scope
//!
//! These are **primitives**, in TAPP's sense of the word: one operation per
//! call, executed as asked. There is no index-notation front end, no contraction
//! *ordering* for a network of tensors, and no autodiff. Those belong in a layer
//! above this one and are deliberately out of scope — see `DESIGN.md`.
//!
//! For the C interface to the same primitives, see the `tensorprimitives-tapp`
//! crate, which implements the TAPP standard's ABI over them.
//!
//! # Example
//!
//! Everything the contraction engine exports is re-exported here, so the
//! `tensorcontract` documentation's examples work unchanged through this crate:
//!
//! ```
//! use tensorprimitives::{contract, parse_einsum, Layout, TensorView, TensorViewMut};
//!
//! // D[i,j] = sum_k A[i,k] * B[k,j]
//! let (ia, ib, id) = parse_einsum("ik,kj->ij").unwrap();
//! let (la, lb, ld) = (
//!     Layout::col_major(&[2, 3]),
//!     Layout::col_major(&[3, 2]),
//!     Layout::col_major(&[2, 2]),
//! );
//! let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b = vec![1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0];
//! let mut d = vec![0.0f64; 4];
//!
//! contract(
//!     1.0,
//!     TensorView::new(&a, &la, &ia),
//!     TensorView::new(&b, &lb, &ib),
//!     0.0,
//!     None,
//!     TensorViewMut::new(&mut d, &ld, &id),
//! )?;
//! assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0]);
//! # Ok::<(), tensorprimitives::Error>(())
//! ```
#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

/// The contraction engine, re-exported whole.
///
/// Available under the `contract` feature (on by default). Re-exported as a
/// module as well as by the flattened items below, so that anything not in the
/// prelude — the introspection tier, `kernel`, `reference` — stays reachable
/// through this crate rather than forcing a second dependency.
#[cfg(feature = "contract")]
pub use tensorcontract as contract_engine;

#[cfg(feature = "contract")]
pub use tensorcontract::{
    contract, einsum_labels, parse_einsum, ComplexMethod, Element, Error, Layout, Operand, Plan,
    PlanStats, Real, Result, TensorView, TensorViewMut, C32, C64,
};
