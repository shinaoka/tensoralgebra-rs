//! Dense linear algebra for the tprims stack.
//!
//! Every operation takes an explicit [`tprims_exec::Exec`]. Inputs are
//! rank-2 `strided_view` views; each factorization copies its input once into
//! a library-owned column-major [`Matrix`] (factors are outputs the library
//! owns) and keeps faer's representation. Solves work in place on caller
//! views. The implementations are faer's (Sarah Quiñones, MIT).
//!
//! Conventions: [`svd`] returns `U`, non-increasing real `S` and `V` (not
//! `Vᴴ`); [`eigh`] returns non-decreasing real eigenvalues and reads the lower
//! triangle; [`cholesky`] reads the lower triangle; factorizations do not fail
//! on singular input, solves report [`Error::Singular`].
//!
//! # Examples
//!
//! ```
//! use tprims_exec::Exec;
//! use tprims_linalg::{cholesky, Matrix};
//!
//! let a = Matrix::<f64>::from_col_major(2, 2, vec![4.0, 2.0, 2.0, 3.0]).unwrap();
//! let f = cholesky(&Exec::serial(), &a.view()).unwrap();
//! let mut b = Matrix::<f64>::from_col_major(2, 1, vec![6.0, 5.0]).unwrap();
//! f.solve(&Exec::serial(), &mut b.view_mut()).unwrap();
//! assert!((b.get(0, 0) - 1.0).abs() < 1e-12 && (b.get(1, 0) - 1.0).abs() < 1e-12);
//! ```
pub mod batched;
mod cholesky;
mod error;
mod lu;
mod mat;
mod qr;
mod spectral;
mod util;

pub use batched::Status;
pub use cholesky::{cholesky, ldlt, Cholesky, Ldlt};
pub use error::{Error, Result};
pub use lu::{det, inv, logdet, lu, lu_full, solve, FullPivLu, Lu};
pub use mat::Matrix;
pub use qr::{lstsq, qr, qr_col_piv, ColPivQr, Lstsq, Qr};
pub use spectral::{eig, eigh, svd, Eig, EigScalar, Eigh, Svd, Vectors};
