//! Transpose-free dense tensor contraction with planar-complex packing.
//!
//! This crate computes
//!
//! ```text
//! D[idx_D] = alpha * op_A(A[idx_A]) * op_B(B[idx_B]) + beta * op_C(C[idx_C])
//! ```
//!
//! for dense, generally-strided tensors, without ever materialising a
//! transposed copy of an operand. It follows the block-scatter-matrix
//! algorithm of Matthews (arXiv:1607.00291) — a tensor contraction is a GEMM
//! over a scatter memory layout — inside BLIS's five-loop, two-level-packing
//! structure.
//!
//! # What is different here
//!
//! Complex operands are packed into **planar (split-complex)** panels: the
//! real and imaginary parts are separated into contiguous runs during the
//! scatter/gather pass that packing has to make anyway, and re-interleaved
//! only at write-back. That gives a real-SIMD micro-kernel with no in-register
//! shuffles, and a packed `A` panel half the size of BLIS's 1m "1e" format.
//! See [`kernel`] and [`pack`] for the details.
//!
//! # Example
//!
//! ```
//! use tensorcontract::{contract, parse_einsum, Layout, TensorView, TensorViewMut};
//!
//! // D[i,j] = sum_k A[i,k] * B[k,j]
//! let (ia, ib, id) = parse_einsum("ik,kj->ij").unwrap();
//! let la = Layout::col_major(&[2, 3]);
//! let lb = Layout::col_major(&[3, 2]);
//! let ld = Layout::col_major(&[2, 2]);
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
//! )
//! .unwrap();
//! assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0]);
//! ```

mod buffer;
mod driver;
mod pack;
mod writeback;

pub mod element;
pub mod error;
pub mod kernel;
pub mod layout;
pub mod plan;
pub mod reference;
pub mod scatter;

pub use element::{Element, Real, C32, C64};
pub use error::{Error, Result};
pub use kernel::KernelSet;
pub use layout::Layout;
pub use plan::{Class, ElementOp, Operand, Plan, PlanStats};

/// An immutable operand: data, layout and index labels.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a, T> {
    pub data: &'a [T],
    pub layout: &'a Layout,
    pub idx: &'a [i64],
    pub op: ElementOp,
}

impl<'a, T> TensorView<'a, T> {
    pub fn new(data: &'a [T], layout: &'a Layout, idx: &'a [i64]) -> Self {
        TensorView {
            data,
            layout,
            idx,
            op: ElementOp::Identity,
        }
    }
    /// Apply complex conjugation to this operand.
    pub fn conj(mut self) -> Self {
        self.op = ElementOp::Conjugate;
        self
    }
    fn operand(&self) -> Operand<'a> {
        Operand {
            layout: self.layout,
            idx: self.idx,
            op: self.op,
        }
    }
}

/// A mutable output operand.
#[derive(Debug)]
pub struct TensorViewMut<'a, T> {
    pub data: &'a mut [T],
    pub layout: &'a Layout,
    pub idx: &'a [i64],
    pub op: ElementOp,
}

impl<'a, T> TensorViewMut<'a, T> {
    pub fn new(data: &'a mut [T], layout: &'a Layout, idx: &'a [i64]) -> Self {
        TensorViewMut {
            data,
            layout,
            idx,
            op: ElementOp::Identity,
        }
    }
    /// Conjugate the result before storing it.
    pub fn conj(mut self) -> Self {
        self.op = ElementOp::Conjugate;
        self
    }
}

/// Plan and execute a contraction in one call.
///
/// For repeated contractions of the same shape, build a [`Plan`] once with
/// [`Plan::new`] and call [`Plan::run`] instead; planning cost is proportional
/// to `M + N + K` and is not negligible for small contractions.
pub fn contract<T>(
    alpha: T,
    a: TensorView<'_, T>,
    b: TensorView<'_, T>,
    beta: T,
    c: Option<TensorView<'_, T>>,
    d: TensorViewMut<'_, T>,
) -> Result<()>
where
    T: Element,
    T::Real: KernelSet,
{
    let plan = Plan::new(
        a.operand(),
        b.operand(),
        c.as_ref().map(|c| c.operand()),
        Operand {
            layout: d.layout,
            idx: d.idx,
            op: d.op,
        },
    )?;
    plan.run(alpha, a, b, beta, c, d)
}

impl Plan {
    /// Execute this plan against concrete slices.
    ///
    /// The slices are bounds-checked against the plan's scatter vectors before
    /// any unsafe access, so this entry point is safe.
    pub fn run<T>(
        &self,
        alpha: T,
        a: TensorView<'_, T>,
        b: TensorView<'_, T>,
        beta: T,
        c: Option<TensorView<'_, T>>,
        d: TensorViewMut<'_, T>,
    ) -> Result<()>
    where
        T: Element,
        T::Real: KernelSet,
    {
        self.check_bounds(
            a.data.len(),
            b.data.len(),
            c.as_ref().map(|c| c.data.len()),
            d.data.len(),
        )?;
        let cptr = match &c {
            Some(c) => c.data.as_ptr(),
            None => d.data.as_ptr(),
        };
        let beta = if c.is_none() { T::zero() } else { beta };
        // SAFETY: bounds validated above; `d` is an exclusive borrow so it
        // cannot alias `a`, `b` or `c`.
        unsafe {
            driver::execute(
                self,
                alpha,
                a.data.as_ptr(),
                b.data.as_ptr(),
                beta,
                cptr,
                d.data.as_mut_ptr(),
            );
        }
        Ok(())
    }

    /// Execute against raw pointers. Used by the TAPP C ABI.
    ///
    /// # Safety
    /// See [`driver::execute`]: every offset produced by the plan's scatter
    /// vectors must be in bounds for the corresponding pointer, and `d` must
    /// not alias `a` or `b`.
    pub unsafe fn run_raw<T>(
        &self,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *const T,
        d: *mut T,
    ) where
        T: Element,
        T::Real: KernelSet,
    {
        driver::execute(self, alpha, a, b, beta, c, d)
    }

    /// Every offset the plan can generate must land inside the slice.
    fn check_bounds(&self, na: usize, nb: usize, nc: Option<usize>, nd: usize) -> Result<()> {
        fn extremes(parts: [&[i64]; 3]) -> (i64, i64) {
            let mut lo = 0i64;
            let mut hi = 0i64;
            for p in parts {
                lo += p.iter().copied().min().unwrap_or(0);
                hi += p.iter().copied().max().unwrap_or(0);
            }
            (lo, hi)
        }
        let check = |parts: [&[i64]; 3], have: usize, tensor: &'static str| -> Result<()> {
            let (lo, hi) = extremes(parts);
            if lo < 0 || hi < 0 || (hi as usize) >= have.max(1) || have == 0 {
                return Err(Error::NullPointer { tensor });
            }
            Ok(())
        };
        check([&self.a_m, &self.a_k, &self.h_a], na, "A")?;
        check([&self.b_k, &self.b_n, &self.h_b], nb, "B")?;
        check([&self.d_m, &self.d_n, &self.h_d], nd, "D")?;
        if let Some(nc) = nc {
            check([&self.c_m, &self.c_n, &self.h_c], nc, "C")?;
        }
        Ok(())
    }
}

/// Turn einsum-style label strings into the `i64` label arrays the engine
/// takes, using each character's Unicode scalar value as the label.
///
/// ```
/// let l = tensorcontract::einsum_labels(&["ik", "kj", "ij"]);
/// assert_eq!(l[0], vec!['i' as i64, 'k' as i64]);
/// ```
pub fn einsum_labels(specs: &[&str]) -> Vec<Vec<i64>> {
    specs
        .iter()
        .map(|s| s.chars().map(|c| c as i64).collect())
        .collect()
}

/// Parse `"ab,bc->ac"` into `(idx_a, idx_b, idx_d)`.
pub fn parse_einsum(spec: &str) -> Option<(Vec<i64>, Vec<i64>, Vec<i64>)> {
    let (lhs, rhs) = spec.split_once("->")?;
    let (a, b) = lhs.split_once(',')?;
    let lab = |s: &str| s.trim().chars().map(|c| c as i64).collect();
    Some((lab(a), lab(b), lab(rhs)))
}
