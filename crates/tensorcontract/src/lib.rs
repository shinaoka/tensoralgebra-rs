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
//! Complex contraction can be induced from real arithmetic in more than one
//! way, and this crate implements **three interchangeable methods** so they can
//! be measured against each other on genuinely equal footing — same index
//! analysis, same scatter machinery, same five-loop driver, same write-back
//! scatter. Only packing, the micro-kernel and the tile recombination differ.
//!
//! | [`ComplexMethod`] | packing | kernel | note |
//! |---|---|---|---|
//! | `Planar` (default) | split real/imaginary planes, 2 reals per element | fused complex | smallest packed A, no in-register shuffles |
//! | `OneM` | BLIS "1e" for A (4 reals per element), "1r" for B | plain real | what BLIS and TBLIS 2.x use |
//! | `ThreeM` | real, imaginary and sum planes, 3 reals per element | Karatsuba | 25% fewer flops, weaker error bound |
//!
//! Select per plan with [`Plan::with_complex_method`], or globally with the
//! `TENSORCONTRACT_COMPLEX` environment variable (`planar` | `1m` | `3m`).
//! See [`kernel`] for the packed formats and the micro-kernel contract.
//!
//! # What is API, and what it promises
//!
//! The public surface is deliberately in three tiers, because they carry very
//! different promises:
//!
//! 1. **The contraction API** — [`contract`], [`Plan`], [`Layout`],
//!    [`TensorView`], [`TensorViewMut`], [`Element`], [`Error`], the per-plan
//!    choices [`Plan::with_complex_method`], [`Plan::with_threads`],
//!    [`Plan::with_blocking`], and the batched entry points
//!    [`batch::contract_batched`] / [`batch::BatchItem`]. Ordinary semver: a
//!    breaking change here needs a major version.
//! 2. **Introspection of the engine's own decisions** — [`PlanStats`],
//!    [`plan::Scatters`], [`Plan::transposes_gemm`], [`Plan::row_block`],
//!    [`Plan::partition`], [`Plan::partition_with`],
//!    [`Plan::amortised_threads`], [`Plan::work_fmas`],
//!    [`kernel::selected_config`], [`kernel::cache`] and
//!    friends. The *signatures* are semver-stable, and they exist so that a
//!    benchmark harness or an alternative execution strategy can describe
//!    exactly what this engine would do. The *values* are tuning outputs and
//!    will change whenever a heuristic is re-measured; that is not a breaking
//!    change, and no caller should encode one of these answers as a constant.
//! 3. **`#[doc(hidden)]` internals**, which are public only because sibling
//!    crates in this workspace need them. They are outside the semver
//!    guarantee entirely and may change or vanish without a major bump. Today
//!    that is `kernel::x86` — the SIMD kernels, whose register-block menus are
//!    re-measured per machine — and the driver's block-scatter matrix view.
//!    Neither has a page here, which is the point.
//!
//! [`kernel::scalar`] sits in tier 1 by intent: it is the documented route by
//! which a foreign scalar type gets a correct, unvectorised engine.
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

#![warn(missing_docs)]

#[cfg(feature = "std")]
pub mod batch;
mod buffer;
mod driver;
mod pack;
#[cfg(feature = "std")]
mod pool;
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
pub use kernel::{ComplexMethod, KernelSet};
pub use layout::Layout;
pub use plan::{Class, ElementOp, Operand, Plan, PlanStats};

/// An immutable operand: data, layout and index labels.
///
/// The three parts are independent by design. `layout` describes *where* the
/// elements are (extents and general strides), `idx` says *what each mode
/// means* by giving it a label shared with the other operands, and `data` is
/// merely the allocation they address. Nothing requires `data` to be exactly as
/// long as the layout needs — only long enough, which [`Plan::run`] checks.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a, T> {
    /// The backing allocation. Indexed at `sum_k i_k * layout.strides[k]`, so
    /// its length is checked against the largest offset the plan can generate
    /// rather than against `layout.len()`.
    pub data: &'a [T],
    /// Extents and strides, in elements. See [`Layout`].
    pub layout: &'a Layout,
    /// One index label per mode of `layout`, in the same order. Labels are
    /// arbitrary `i64`s and are matched by equality across the four operands;
    /// [`parse_einsum`] and [`einsum_labels`] produce them from strings.
    pub idx: &'a [i64],
    /// Element-wise operation applied while reading. Set by [`TensorView::conj`].
    pub op: ElementOp,
}

impl<'a, T> TensorView<'a, T> {
    /// An operand read as stored, with no element-wise operation.
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
///
/// The exclusive borrow is what makes [`Plan::run`] safe: it is the only reason
/// the engine may assume `D` does not alias `A`, `B` or `C`, which the
/// scatter write-back has no way to check.
#[derive(Debug)]
pub struct TensorViewMut<'a, T> {
    /// The allocation written into, exclusively borrowed.
    pub data: &'a mut [T],
    /// Extents and strides, in elements. See [`Layout`].
    pub layout: &'a Layout,
    /// One index label per mode of `layout`, in the same order. Every label
    /// here must also appear in `A` or `B`; an output-only label would be a
    /// broadcast, which is rejected ([`Error::BroadcastIndexUnsupported`]).
    pub idx: &'a [i64],
    /// Element-wise operation applied to the result before storing. Set by
    /// [`TensorViewMut::conj`].
    pub op: ElementOp,
}

impl<'a, T> TensorViewMut<'a, T> {
    /// An output stored as computed, with no element-wise operation.
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

    /// Execute against raw pointers, with no bounds checking.
    ///
    /// This is what the TAPP C ABI calls: at an FFI boundary the caller has
    /// already been handed raw pointers and slice lengths do not exist, so
    /// [`Plan::run`]'s check has nothing to check against.
    ///
    /// # Safety
    ///
    /// The plan's scatter vectors ([`Plan::scatters`]) enumerate every offset
    /// that will be touched, so the obligations are exactly:
    ///
    /// * for each of `a`, `b`, `d` (and `c` when `beta` is nonzero), every sum
    ///   of one offset from each of that operand's three scatter vectors —
    ///   row, column and Hadamard — must be a valid offset from the pointer,
    ///   readable for `a`/`b`/`c` and writable for `d`;
    /// * `d` must not alias `a`, `b` or `c`, except that `c == d` is allowed
    ///   and is how an in-place update is expressed;
    /// * `T` must be the element type the caller intends — nothing here
    ///   validates it, and a plan is element-type independent by construction.
    ///
    /// When `c` is not read (`beta` zero) it is never dereferenced and may be
    /// any pointer, including `d`.
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
