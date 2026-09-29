//! Dense tensor contraction, computed in place over the operands' own strides.
//!
//! This crate computes
//!
//! ```text
//! D[idx_D] = alpha * op_A(A[idx_A]) * op_B(B[idx_B]) + beta * op_C(C[idx_C])
//! ```
//!
//! for dense, generally-strided tensors of any rank, **without materialising a
//! transposed copy of an operand and without allocating a workspace.** Index
//! labels say what contracts with what; the engine works out the rest.
//!
//! # Quick start
//!
//! A contraction is three layouts, three label lists, and a call. Labels are
//! matched by equality across the operands, so a label appearing in `A`, `B`
//! and `D` alike is a *batch* index — something no single matrix multiply
//! expresses:
//!
//! ```
//! use tensorcontract::{contract, parse_einsum, Layout, TensorView, TensorViewMut};
//!
//! // D[h,i,j] = sum_k A[h,i,k] * B[h,k,j]   — `h` batches, `k` contracts.
//! let (ia, ib, id) = parse_einsum("hik,hkj->hij")?;
//! let (la, lb, ld) = (
//!     Layout::col_major(&[2, 2, 2]),
//!     Layout::col_major(&[2, 2, 2]),
//!     Layout::col_major(&[2, 2, 2]),
//! );
//!
//! let a: Vec<f64> = (1..=8).map(|x| x as f64).collect();
//! // B is the 2x2 identity in (k, j) for each h, so D comes back equal to A.
//! let b = vec![1.0f64, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
//! let mut d = vec![0.0f64; 8];
//!
//! contract(
//!     1.0,
//!     TensorView::new(&a, &la, &ia),
//!     TensorView::new(&b, &lb, &ib),
//!     0.0,
//!     None,
//!     TensorViewMut::new(&mut d, &ld, &id),
//! )?;
//! assert_eq!(d, a);
//! # Ok::<(), tensorcontract::Error>(())
//! ```
//!
//! # Expressing a contraction
//!
//! An operand is three independent things, which is why no reshaping or
//! copying is ever needed to hand one over:
//!
//! * a [`Layout`] — extents and strides, **in elements**, where a stride may be
//!   negative (walks backwards) or zero (reads one element repeatedly);
//! * an `idx` list of one `i64` label per mode, in layout order. Labels are
//!   arbitrary integers compared by equality; [`parse_einsum`] and
//!   [`einsum_labels`] build them from strings for convenience only;
//! * the data, as a slice ([`TensorView`], [`TensorViewMut`]) or a raw pointer
//!   ([`Plan::run_raw`]).
//!
//! Where a label appears decides what it does, and that is the whole of the
//! interface:
//!
//! | label appears in | role |
//! |---|---|
//! | `A` and `D` | free index of `A` — rows of the underlying GEMM |
//! | `B` and `D` | free index of `B` — columns |
//! | `A` and `B` | contracted (summed over) |
//! | `A`, `B` and `D` | batch / Hadamard index |
//! | `A` only, or `B` only | reduction — summed over, needing no workspace |
//! | twice within one operand | that operand's diagonal |
//! | `D` only | rejected: [`Error::BroadcastIndexUnsupported`] |
//!
//! [`plan`] documents how these classes are derived, folded and ordered.
//!
//! # What is supported
//!
//! | | |
//! |---|---|
//! | element types | `f32`, `f64`, [`C32`], [`C64`] — see [`element`] |
//! | operand ranks | any, including rank 0 |
//! | strides | general, including negative and zero |
//! | contraction, batch indices | yes |
//! | diagonals (a label twice in one operand) | yes |
//! | reductions (a label in one input only) | yes, with no temporary |
//! | conjugation on any operand | yes — free, folded into packing |
//! | mixed element types across operands | no: one element type per contraction |
//! | output-only labels (broadcast) | no, rejected rather than guessed at |
//!
//! The vocabulary is [TAPP][tapp]'s, whose "cases 1–5" this table covers in
//! order; the C ABI itself lives in the sibling `tensorprimitives-tapp` crate.
//! Anything rejected is rejected while *planning*, so a [`Plan`] that exists
//! will run — see [`error`].
//!
//! # Complex arithmetic
//!
//! Complex contraction has to be induced from real arithmetic, and there is
//! more than one way to do it. Three are implemented, sharing every other line
//! of the engine — the same index analysis, scatter machinery, five-loop driver
//! and write-back — and differing only in what packing emits, what the
//! micro-kernel computes, and how the accumulator is read back:
//!
//! | [`ComplexMethod`] | packed reals per element (A / B) | kernel | trade |
//! |---|---|---|---|
//! | `Planar` (default) | 2 / 2 | fused complex | smallest packed `A`, no in-register shuffles |
//! | `OneM` | 4 / 2 | one real GEMM | Van Zee's 1m ([1m][onem]); no complex kernel needed |
//! | `ThreeM` | 3 / 3 | Karatsuba | 25% fewer flops, weaker error bound, more traffic |
//!
//! **Planar is the default and is the one to use absent a reason.** The
//! deciding quantity between them is bytes moved per useful flop rather than
//! flop count, so `ThreeM`'s arithmetic saving does not reliably pay; which
//! method wins beyond that depends on the machine, and this crate quotes no
//! ranking it has not measured on the machine in question.
//!
//! ```
//! use tensorcontract::{kernel::ComplexMethod, parse_einsum, Layout, Operand};
//! use tensorcontract::{Plan, TensorView, TensorViewMut, C64};
//!
//! // D[i,j] = sum_k conj(A[i,k]) * B[k,j], with B the identity.
//! let (ia, ib, id) = parse_einsum("ik,kj->ij")?;
//! let l = Layout::col_major(&[2, 2]);
//!
//! // Conjugation belongs to the plan; the view handed to `run` must agree.
//! let plan = Plan::new(
//!     Operand::new(&l, &ia).conj(),
//!     Operand::new(&l, &ib),
//!     None,
//!     Operand::new(&l, &id),
//! )?
//! .with_complex_method(ComplexMethod::Planar);
//!
//! let a = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0),
//!              C64::new(5.0, 6.0), C64::new(7.0, 8.0)];
//! let b = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0),
//!              C64::new(0.0, 0.0), C64::new(1.0, 0.0)];
//! let mut d = vec![C64::new(0.0, 0.0); 4];
//!
//! plan.run(
//!     C64::new(1.0, 0.0),
//!     TensorView::new(&a, &l, &ia).conj(),   // conjugation costs nothing
//!     TensorView::new(&b, &l, &ib),
//!     C64::new(0.0, 0.0),
//!     None,
//!     TensorViewMut::new(&mut d, &l, &id),
//! )?;
//! assert_eq!(d, a.iter().map(|z| z.conj()).collect::<Vec<_>>());
//! # Ok::<(), tensorcontract::Error>(())
//! ```
//!
//! [`kernel`] has the packed panel formats, the micro-kernel contract, and the
//! environment override that selects a method process-wide.
//!
//! # Performance
//!
//! **Reuse a [`Plan`] across contractions of the same shape.** Planning is
//! `O(M + N + K)` — it builds the scatter vectors — which is not negligible for
//! small tensors. [`contract`] is the convenience form that plans and discards;
//! for anything in a loop, plan once:
//!
//! ```
//! use tensorcontract::{parse_einsum, Layout, Operand, Plan, TensorView, TensorViewMut};
//!
//! let (ia, ib, id) = parse_einsum("ik,kj->ij")?;
//! let l = Layout::col_major(&[2, 2]);
//! let plan = Plan::new(
//!     Operand::new(&l, &ia),
//!     Operand::new(&l, &ib),
//!     None,
//!     Operand::new(&l, &id),
//! )?;
//!
//! let identity = vec![1.0f64, 0.0, 0.0, 1.0];
//! for a in [vec![1.0f64, 2.0, 3.0, 4.0], vec![5.0f64, 6.0, 7.0, 8.0]] {
//!     let mut d = vec![0.0f64; 4];
//!     plan.run(
//!         1.0,
//!         TensorView::new(&a, &l, &ia),
//!         TensorView::new(&identity, &l, &ib),
//!         0.0,
//!         None,
//!         TensorViewMut::new(&mut d, &l, &id),
//!     )?;
//!     assert_eq!(d, a);       // same plan, different data
//! }
//! # Ok::<(), tensorcontract::Error>(())
//! ```
//!
//! **Threading is off by default** — one thread unless [`Plan::with_threads`]
//! asks otherwise. A library should not decide how many cores its caller has,
//! and thread spawn is tens of microseconds, which dominates a small
//! contraction outright. Any thread count gives bitwise the same result.
//!
#![cfg_attr(
    feature = "std",
    doc = "**Many small contractions are a batch, not a loop.** \
[`batch::contract_batched`] parallelises over independent items, paying one set \
of spawns for the whole batch instead of one per contraction — which is the \
regime where per-call threading loses."
)]
//!
//! Cache blocking, the register block and the row/column orientation are all
//! derived per machine and per element type; [`Plan::with_blocking`] overrides
//! them. What has actually been measured, on which machine, and what is
//! explicitly *not* claimed, is recorded in [`docs/results.md`][results] in the
//! [repository] — no performance number is restated here.
//!
//! # Algorithm
//!
//! A tensor contraction becomes a matrix multiplication once the operands are
//! addressed through **scatter vectors**: group the labels into classes,
//! linearise each class as a mixed-radix multi-index, and precompute
//! `rscat[i] = sum_l i_l * stride_l`. Matrix element `(i, j)` then lives at
//! `base + rscat[i] + cscat[j]`, and no transposition is needed because the
//! traversal, not the data, is what changes. The refinement that makes it fast
//! is the **block** scatter vector: for each aligned run of `MR` entries,
//! record whether they form an arithmetic progression, and if so pack that
//! block with ordinary strided loads instead of a gather.
//!
//! That matrix multiply then runs in BLIS's five-loop nest with two levels of
//! packing — `NC` on the columns, `KC` on the contraction, `MC` on the rows,
//! an L3-resident packed `B` and an L2-resident packed `A`, and a
//! register-blocked micro-kernel at the bottom. The only tensor-specific
//! change is that packing and write-back read through the scatter vectors.
//!
//! This is the block-scatter-matrix algorithm of [Matthews][bsmtc]. See
//! [`scatter`] for the vectors themselves, [`plan`] for the index analysis
//! that produces them, and [`kernel`] for the packed formats and blocking.
//!
//! # Where to read next
//!
//! | module | what it covers |
//! |---|---|
//! | (root) | [`contract`], [`Plan`], [`TensorView`] — the everyday API |
//! | [`plan`] | index analysis: label classes, folding, diagonals, reductions |
//! | [`layout`] | extents and general strides |
//! | [`element`] | `f32`/`f64`/[`C32`]/[`C64`], and the real-scalar boundary |
//! | [`kernel`] | complex methods, packed formats, micro-kernel contract, cache blocking |
//! | [`scatter`] | scatter and block-scatter vectors — the core data structure |
//! | [`error`] | what a rejected contraction reports, and why |
//! | [`reference`][mod@reference] | a naive oracle for tests; never the fast path |
#![cfg_attr(
    feature = "std",
    doc = "| [`batch`] | many independent contractions, batch as the parallel axis |"
)]
//!
//! # Stability
//!
//! The public surface is in three tiers, carrying different promises:
//!
//! 1. **The contraction API** — [`contract`], [`Plan`], [`Layout`],
//!    [`TensorView`], [`TensorViewMut`], [`Element`], [`Error`] and the
//!    per-plan choices. Ordinary semver. [`kernel::scalar`] is here by intent:
//!    it is the documented route by which a foreign scalar type gets a correct,
//!    unvectorised engine.
//! 2. **Introspection of the engine's own decisions** — [`PlanStats`],
//!    [`plan::Scatters`], [`Plan::transposes_gemm`], [`Plan::row_block`],
//!    [`Plan::partition`], [`kernel::selected_config`], [`kernel::cache`] and
//!    friends. The *signatures* are semver-stable. The *values* are tuning
//!    outputs and change whenever a heuristic is re-measured; that is not a
//!    breaking change, and **no caller should encode one of these answers as a
//!    constant.**
//! 3. **`#[doc(hidden)]` internals**, public only because sibling crates in this
//!    workspace need them, and outside the semver guarantee entirely:
//!    `kernel::x86` and `kernel::aarch64`, the SIMD kernels, whose
//!    register-block menus are re-measured per machine. They have no page
//!    here, which is the point.
//!
//! # References
//!
//! * D. A. Matthews, *High-Performance Tensor Contraction without
//!   Transposition*, SIAM J. Sci. Comput. 40(1), 2018 — [arXiv:1607.00291][bsmtc].
//!   The block-scatter-matrix algorithm this engine implements.
//! * *Tensor Algebra Processing Primitives* — [arXiv:2601.07827][tapp] and the
//!   [reference implementation][tappref]. The strided data model above mirrors
//!   its `TAPP_tensor_info`, and the case vocabulary is its own.
//! * F. G. Van Zee, *Implementing High-Performance Complex Matrix
//!   Multiplication via the 1m Method*, SIAM J. Sci. Comput. 42(5), 2020 —
//!   [1m][onem]; and Van Zee & Smith, *Implementing High-Performance Complex
//!   Matrix Multiplication via the 3m and 4m Methods*, ACM TOMS 44(1), 2017 —
//!   [3m/4m][threem]. `OneM` and `ThreeM` are theirs.
//! * F. G. Van Zee & R. A. van de Geijn, *BLIS: A Framework for Rapidly
//!   Instantiating BLAS Functionality*, ACM TOMS 41(3), 2015 — [BLIS][blis].
//!   The five-loop, two-level-packing structure.
//!
//! [bsmtc]: https://arxiv.org/abs/1607.00291
//! [tapp]: https://arxiv.org/abs/2601.07827
//! [tappref]: https://github.com/TAPPorg/reference-implementation
//! [onem]: https://doi.org/10.1137/19M1282040
//! [threem]: https://doi.org/10.1145/3086466
//! [blis]: https://doi.org/10.1145/2764454
//! [repository]: https://github.com/lkdvos/tensorprimitives-rs
//! [results]: https://github.com/lkdvos/tensorprimitives-rs/blob/main/docs/results.md

#![warn(missing_docs)]
// CI builds with `-D warnings`, so this is what keeps "every public type is
// `Debug`" true by construction instead of by review. Two types had already
// slipped through.
#![warn(missing_debug_implementations)]

/// Read one `TENSORCONTRACT_*` variable once per process, or fall back.
///
/// Nine switches were spelling this out by hand, and the copies had drifted:
/// `partition_override` returned the *legacy* rule without `std` where the
/// `std` default is the domain-aware one, so a `--no-default-features` build
/// silently partitioned differently. Naming the default once, outside the
/// `cfg`, makes that class of divergence unrepresentable — the two arms cannot
/// disagree because there is only one expression.
///
/// The variable name stays a literal at each call site on purpose, so
/// `grep TENSORCONTRACT_` still finds every switch in the crate.
///
/// `$ty` must be `Copy`; every switch is a small enum, `bool` or `usize`.
macro_rules! env_once {
    ($ty:ty, $var:literal, $default:expr, $parse:expr) => {{
        #[cfg(feature = "std")]
        {
            use std::sync::OnceLock;
            static ENV: OnceLock<$ty> = OnceLock::new();
            *ENV.get_or_init(|| match std::env::var($var) {
                Ok(v) => ($parse)(v.as_str()),
                Err(_) => $default,
            })
        }
        #[cfg(not(feature = "std"))]
        {
            $default
        }
    }};
}

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
pub use kernel::{Blocking, ComplexMethod, KernelSet};
pub use layout::Layout;
pub use plan::{ElementOp, Operand, Plan, PlanStats};

/// An immutable operand: data, layout and index labels.
///
/// The three parts are independent by design. `layout` describes *where* the
/// elements are (extents and general strides), `idx` says *what each mode
/// means* by giving it a label shared with the other operands, and `data` is
/// merely the allocation they address. Nothing requires `data` to be exactly as
/// long as the layout needs — only long enough, which [`Plan::run`] checks.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a, T> {
    /// The backing allocation. Indexed at `sum_k i_k * layout.strides()[k]`, so
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
    ///
    /// [`contract`] takes this as the request, building a plan to match. With an
    /// existing [`Plan`] the plan is authoritative and this must agree with it,
    /// or [`Plan::run`] returns [`Error::ElementOpMismatch`].
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
    ///
    /// Consumes the view and returns a new one, so dropping the result is a
    /// silent no-op — the conjugation simply does not happen. Hence `must_use`.
    #[must_use]
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
    ///
    /// Consumes the view and returns a new one, so dropping the result is a
    /// silent no-op — the conjugation simply does not happen. Hence `must_use`.
    #[must_use]
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
///
/// Unlike [`Plan::run`], this reads the conjugation off the views themselves,
/// because it builds the plan from them — so [`TensorView::conj`] takes effect
/// here with nothing else to keep in sync.
///
/// # Errors
///
/// Everything [`Plan::new`] rejects — an inconsistent description
/// ([`Error::RankMismatch`], [`Error::LabelCountMismatch`],
/// [`Error::ExtentMismatch`], [`Error::NegativeExtent`],
/// [`Error::OutputLabelMismatch`]), a contraction out of scope
/// ([`Error::BroadcastIndexUnsupported`]) or a shape too large to address
/// ([`Error::ExtentProductOverflow`]) — plus [`Error::NullPointer`] from
/// [`Plan::run`] if a slice is shorter than the offsets the plan generates.
/// [`Error::ElementOpMismatch`] cannot occur here: the plan is derived from
/// these very views.
///
/// ```
/// use tensorcontract::{contract, Layout, TensorView, TensorViewMut};
///
/// // D[i] = sum_j A[i,j] * v[j] — a matrix-vector product.
/// let (i, j) = (b'i' as i64, b'j' as i64);
/// let la = Layout::col_major(&[2, 3]);
/// let lv = Layout::col_major(&[3]);
/// let ld = Layout::col_major(&[2]);
///
/// let a: Vec<f64> = (1..=6).map(|x| x as f64).collect();   // A[i,j] = i + 2j + 1
/// let v = vec![1.0f64; 3];
/// let mut d = vec![0.0f64; 2];
///
/// contract(
///     1.0,
///     TensorView::new(&a, &la, &[i, j]),
///     TensorView::new(&v, &lv, &[j]),
///     0.0,
///     None,
///     TensorViewMut::new(&mut d, &ld, &[i]),
/// )?;
/// assert_eq!(d, vec![1.0 + 3.0 + 5.0, 2.0 + 4.0 + 6.0]);
/// # Ok::<(), tensorcontract::Error>(())
/// ```
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
    ///
    /// Each view's [`ElementOp`] must equal the one the plan was built with, or
    /// this returns [`Error::ElementOpMismatch`]. Conjugation is folded into the
    /// packing and write-back traversals, so it belongs to the plan (see
    /// [`Operand::conj`]) and cannot be varied per call; checking is what stops
    /// a `.conj()`ed view from being silently ignored. `C`'s op is checked only
    /// when a `C` is supplied, since otherwise it is never read.
    ///
    /// [`contract`] cannot hit this: it derives the plan from the same views it
    /// executes.
    ///
    /// # Errors
    ///
    /// Only the two things a plan cannot know until it is handed arguments:
    /// [`Error::ElementOpMismatch`] if a view's [`ElementOp`] differs from the
    /// plan's, and [`Error::NullPointer`] if a slice is empty or too short to
    /// hold every offset the plan's scatter vectors generate. Nothing about the
    /// contraction *itself* is rechecked — [`Plan::new`] settled all of it.
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
        self.check_ops(a.op, b.op, c.as_ref().map(|c| c.op), d.op)?;
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

    /// Each operand's element-wise op must be the one the plan was built with.
    ///
    /// `C` is exempt when absent: `beta` is forced to zero and it is never read,
    /// so a plan built with a conjugated `C` and run without one is consistent.
    fn check_ops(
        &self,
        a: ElementOp,
        b: ElementOp,
        c: Option<ElementOp>,
        d: ElementOp,
    ) -> Result<()> {
        let pairs = [
            (a.is_conj(), self.conj_a, "A"),
            (b.is_conj(), self.conj_b, "B"),
            (d.is_conj(), self.conj_d, "D"),
        ];
        for (given, planned, tensor) in pairs {
            if given != planned {
                return Err(Error::ElementOpMismatch { tensor });
            }
        }
        if let Some(c) = c {
            if c.is_conj() != self.conj_c {
                return Err(Error::ElementOpMismatch { tensor: "C" });
            }
        }
        Ok(())
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
/// Any iterable of string-likes will do, so a caller holding `String`s does not
/// have to build a `Vec<&str>` to be allowed to call this:
///
/// ```
/// use tensorcontract::einsum_labels;
///
/// let l = einsum_labels(&["ik", "kj", "ij"]);
/// assert_eq!(l[0], vec!['i' as i64, 'k' as i64]);
///
/// let owned: Vec<String> = vec!["ik".into(), "kj".into(), "ij".into()];
/// assert_eq!(einsum_labels(owned), l);
/// ```
pub fn einsum_labels(specs: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<Vec<i64>> {
    specs
        .into_iter()
        .map(|s| s.as_ref().chars().map(|c| c as i64).collect())
        .collect()
}

/// Parse `"ab,bc->ac"` into `(idx_a, idx_b, idx_d)`.
///
/// A convenience for tests and examples, not a general einsum front end: it
/// splits on `->` and one `,`, and every label is a single character. There is no
/// `C` operand in the notation, and no validation of what the labels *mean* — a
/// spec the engine will reject still parses.
///
/// # Errors
///
/// [`Error::EinsumSyntax`], and only that, for the two shapes of spec this
/// notation cannot read: one with no `->`, and one whose left-hand side has no
/// `,` separating `A` from `B`. Its `reason` says which. Everything else is
/// somebody else's error — an unusable set of labels is [`Plan::new`]'s to
/// reject, not this function's.
///
/// ```
/// use tensorcontract::parse_einsum;
///
/// let (a, b, d) = parse_einsum("ik,kj->ij")?;
/// assert_eq!(a, vec!['i' as i64, 'k' as i64]);
/// assert_eq!(b, vec!['k' as i64, 'j' as i64]);
/// assert_eq!(d, vec!['i' as i64, 'j' as i64]);
///
/// // An output-only label parses; `Plan::new` is what rejects it.
/// assert!(parse_einsum("i,j->ijk").is_ok());
/// // Missing the second operand or the arrow does not.
/// assert!(parse_einsum("ik->i").is_err());
/// # Ok::<(), tensorcontract::Error>(())
/// ```
pub fn parse_einsum(spec: &str) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
    let (lhs, rhs) = spec.split_once("->").ok_or(Error::EinsumSyntax {
        reason: "no `->`; the output labels are not optional",
    })?;
    let (a, b) = lhs.split_once(',').ok_or(Error::EinsumSyntax {
        reason: "no `,` before `->`; two input operands are required",
    })?;
    let lab = |s: &str| s.trim().chars().map(|c| c as i64).collect();
    Ok((lab(a), lab(b), lab(rhs)))
}
