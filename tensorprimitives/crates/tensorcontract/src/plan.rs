//! Index analysis and contraction planning.
//!
//! Given the four operand layouts and their index label lists, this module
//! works out how to view the contraction
//!
//! ```text
//! D[idx_D] = alpha * op_A(A[idx_A]) * op_B(B[idx_B]) + beta * op_C(C[idx_C])
//! ```
//!
//! as a (batched) matrix multiplication over scatter matrices.
//!
//! # Index classes
//!
//! Every distinct label is assigned to exactly one class:
//!
//! | class | in A | in B | in D | role |
//! |-------|------|------|------|------|
//! | `M`   | yes  | no   | yes  | free index of A, rows of the GEMM |
//! | `N`   | no   | yes  | yes  | free index of B, columns of the GEMM |
//! | `K`   | yes  | yes  | no   | contracted index |
//! | `H`   | yes  | yes  | yes  | Hadamard / batch index |
//!
//! Two further cases fold into `K` rather than needing special machinery:
//!
//! * a label present **only in A** (TAPP "isolated" index, i.e. a reduction
//!   `sum_i A[...,i,...]`) becomes a contraction index with `stride_B = 0`;
//! * likewise a label present only in B becomes one with `stride_A = 0`.
//!
//! Because the block-scatter machinery treats a zero stride as a perfectly
//! regular access pattern, this costs nothing and — unlike a pre-reduction
//! pass — needs no workspace. A label present only in the output (TAPP
//! "case 5", broadcasting) is rejected; TAPP does not require it.
//!
//! # Repeated labels
//!
//! A label repeated within a single tensor selects that tensor's diagonal.
//! Extents must agree and the strides are summed, after which the label is
//! treated as a single mode. This is applied per tensor before classification.
//!
//! # Ordering and folding
//!
//! Within a class the labels are ordered fastest-varying first, then adjacent
//! labels are merged whenever their extents and strides are compatible in
//! *every* operand (`s_next == s_prev * extent_prev`). Folding is what makes
//! an ordinary matrix multiply collapse to a single index per class — and
//! therefore to a plain GEMM with fully regular block scatter — and it
//! substantially raises the regular-block fraction for real contractions.
//!
//! The ordering heuristic is: class `M`, `N` and `H` are ordered by increasing
//! |stride| in `D`, class `K` by increasing |stride| in `A`. Rationale: the
//! output update is the one access that packing cannot hide, so `D` gets first
//! claim on contiguity; `K` only affects the packing of `A` and `B`. This is a
//! heuristic and a Phase 4 tuning knob.

use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::scatter::{build_scatter, run_structure, unbroken_fraction};

/// Index class.
///
/// Crate-internal: it appears in no public signature and no public field --
/// [`PlanStats`] exposes [`Axis`], not this -- and its only uses are inside
/// [`Plan::new`]'s classification pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    /// Free index of A (GEMM rows).
    M,
    /// Free index of B (GEMM columns).
    N,
    /// Contracted index (including isolated/reduction indices).
    K,
    /// Hadamard (batch) index.
    H,
}

/// One folded axis: an extent plus its stride in each operand.
///
/// Carrying all four strides together is what makes folding checkable: two
/// adjacent axes may be merged only when their strides are compatible in
/// *every* operand at once, so the test has to see them side by side. An axis
/// absent from an operand has stride 0 there, which is not a special case —
/// see the module docs on isolated indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Axis {
    /// Length of the axis after folding, i.e. the product of the extents that
    /// were merged into it.
    pub extent: i64,
    /// Stride in `A`, in elements; 0 if the axis does not appear in `A`.
    pub sa: i64,
    /// Stride in `B`.
    pub sb: i64,
    /// Stride in `C`.
    pub sc: i64,
    /// Stride in `D`.
    pub sd: i64,
}

#[derive(Clone, Copy, Debug)]
struct LabelInfo {
    id: i64,
    extent: i64,
    sa: i64,
    sb: i64,
    sc: i64,
    sd: i64,
    in_a: bool,
    in_b: bool,
    in_c: bool,
    in_d: bool,
}

/// Which element-wise operation to apply to an operand.
///
/// Conjugation is free here: on the input side it is folded into the pass that
/// packing has to make anyway, and on the output side into the write-back, so
/// there is never a separate traversal for it. That is why the engine offers no
/// way to *not* apply it lazily.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ElementOp {
    /// Use the values as stored. The default, and the only option that means
    /// anything for a real element type.
    #[default]
    Identity,
    /// Complex-conjugate while reading (inputs) or before storing (output).
    /// A no-op for real element types.
    Conjugate,
}

impl ElementOp {
    /// Whether this op conjugates.
    #[inline]
    pub fn is_conj(self) -> bool {
        matches!(self, ElementOp::Conjugate)
    }
}

/// An operand description: layout plus index labels plus element-wise op.
///
/// This is [`crate::TensorView`] with the data slice removed — everything
/// [`Plan::new`] needs, and nothing it does not. A plan is built from
/// `Operand`s and can then be executed against any data of any element type.
#[derive(Clone, Copy, Debug)]
pub struct Operand<'a> {
    /// Extents and strides, in elements.
    pub layout: &'a Layout,
    /// One index label per mode of `layout`, in the same order.
    pub idx: &'a [i64],
    /// Element-wise operation to apply. Recorded in the plan, so a conjugated
    /// plan and an unconjugated one are different plans.
    pub op: ElementOp,
}

impl<'a> Operand<'a> {
    /// An operand with [`ElementOp::Identity`].
    pub fn new(layout: &'a Layout, idx: &'a [i64]) -> Self {
        Operand {
            layout,
            idx,
            op: ElementOp::Identity,
        }
    }
    /// The same operand, complex-conjugated.
    #[must_use]
    pub fn conj(mut self) -> Self {
        self.op = ElementOp::Conjugate;
        self
    }
}

/// Diagnostics about a plan, useful for benchmarking write-ups and for
/// dispatch heuristics.
///
/// The four dimensions are the *matrix* shape the contraction was reduced to,
/// after diagonals are collapsed, extent-1 axes are dropped and compatible
/// axes are folded — so they are what the engine works on rather than what the
/// caller wrote. Reading them is the cheapest way to see whether folding did
/// what you expected.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanStats {
    /// Rows of the matrix view: the product of the `M` extents.
    pub m: usize,
    /// Columns: the product of the `N` extents.
    pub n: usize,
    /// Contraction depth: the product of the `K` extents, including any
    /// isolated (reduction) indices folded in.
    pub k: usize,
    /// Number of independent matrix products, i.e. the product of the Hadamard
    /// extents. 1 when there are none.
    pub batch: usize,
    /// The folded `M` axes, fastest-varying in `D` first.
    pub m_axes: Vec<Axis>,
    /// The folded `N` axes, fastest-varying in `D` first.
    pub n_axes: Vec<Axis>,
    /// The folded `K` axes, fastest-varying in `A` first.
    pub k_axes: Vec<Axis>,
    /// The folded Hadamard axes.
    pub h_axes: Vec<Axis>,
    /// True when `M`, `N` and `K` each folded to at most one axis **and there
    /// are no Hadamard axes** — i.e. the contraction is exactly *one* GEMM on
    /// strided matrices.
    ///
    /// A batch index therefore makes this false even when every other class
    /// folded perfectly: the work is then a sequence of GEMMs rather than one,
    /// which is the distinction the flag exists to draw. Check `batch == 1`
    /// alongside it if what you want is "no gather anywhere".
    pub is_pure_gemm: bool,
}

impl PlanStats {
    /// Multiply-accumulate count (batch * m * n * k).
    ///
    /// Multiply by [`crate::Element::FLOPS_PER_MAC`] for a flop count. This
    /// counts *useful* work — the padding the engine does on edge blocks is
    /// deliberately not included, so throughput computed from it is comparable
    /// against another library's.
    ///
    /// # Panics
    ///
    /// In a debug build, if the product exceeds `u64::MAX`; in release it wraps.
    /// Reaching that needs all four dimensions near `10^5`, which is only a few
    /// megabytes of scatter vectors and therefore a plan that builds fine even
    /// though nothing could execute it — so it is a real edge, not an
    /// unreachable one.
    pub fn macs(&self) -> u64 {
        (self.batch as u64) * (self.m as u64) * (self.n as u64) * (self.k as u64)
    }
}

/// Borrowed view of a plan's scatter vectors. Offsets are in elements.
///
/// Element `(i, j)` of batch `h` of operand `X` lives at
/// `x_row[i] + x_col[j] + h_x[h]` from that operand's base pointer. The three
/// vectors are independent, which is the whole point of the scatter
/// representation: no arithmetic relates them, so an arbitrary permutation of
/// modes costs a table lookup rather than a transposition.
#[derive(Clone, Copy, Debug)]
pub struct Scatters<'a> {
    /// Offsets of A's rows (free indices of A).
    pub a_m: &'a [i64],
    /// Offsets of A's columns (contracted indices).
    pub a_k: &'a [i64],
    /// Offsets of B's rows (contracted indices). Parallel to `a_k`: entry `p`
    /// of each names the same contraction index.
    pub b_k: &'a [i64],
    /// Offsets of B's columns (free indices of B).
    pub b_n: &'a [i64],
    /// Offsets of C's rows. Parallel to `d_m`, but with C's own strides —
    /// C and D must share labels, not layout.
    pub c_m: &'a [i64],
    /// Offsets of C's columns.
    pub c_n: &'a [i64],
    /// Offsets of D's rows.
    pub d_m: &'a [i64],
    /// Offsets of D's columns.
    pub d_n: &'a [i64],
    /// Per-batch base offsets for the Hadamard indices.
    pub h_a: &'a [i64],
    /// Per-batch base offsets in B. Parallel to `h_a`.
    pub h_b: &'a [i64],
    /// Per-batch base offsets in C.
    pub h_c: &'a [i64],
    /// Per-batch base offsets in D.
    pub h_d: &'a [i64],
}

/// A prepared contraction: everything that depends only on shapes, strides and
/// index labels, and not on the element type or the data pointers.
///
/// Scatter vectors are element-type independent, so one plan can be executed
/// for any element type. Blocking parameters and block-scatter vectors do
/// depend on the element type and are derived at execution time.
#[derive(Clone, Debug)]
pub struct Plan {
    pub(crate) a_m: Vec<i64>,
    pub(crate) a_k: Vec<i64>,
    pub(crate) b_k: Vec<i64>,
    pub(crate) b_n: Vec<i64>,
    pub(crate) c_m: Vec<i64>,
    pub(crate) c_n: Vec<i64>,
    pub(crate) d_m: Vec<i64>,
    pub(crate) d_n: Vec<i64>,
    pub(crate) h_a: Vec<i64>,
    pub(crate) h_b: Vec<i64>,
    pub(crate) h_c: Vec<i64>,
    pub(crate) h_d: Vec<i64>,
    /// Run structure of the output's `M` and `N` scatters, `(len, stride)`, or
    /// `(len, 0)` when there is none. Cached because
    /// [`Plan::transposes_gemm`] and [`Plan::row_block`] both turn on it and
    /// are called several times per execution, while the scatters themselves
    /// never change.
    pub(crate) d_m_run: (usize, i64),
    pub(crate) d_n_run: (usize, i64),
    pub(crate) conj_a: bool,
    pub(crate) conj_b: bool,
    pub(crate) conj_c: bool,
    pub(crate) conj_d: bool,
    /// Overrides the element-type-derived cache blocking when set.
    pub(crate) blocking: Option<crate::kernel::Blocking>,
    /// Overrides the default complex method when set.
    pub(crate) method: Option<crate::kernel::ComplexMethod>,
    /// Overrides the default thread count when set.
    pub(crate) threads: Option<usize>,
    pub(crate) kernel: Option<tprims_kernel::KernelChoice>,
    /// Set by [`Plan::with_selector`]: the family a caller's selector chose.
    pub(crate) forced: Option<crate::select::Forced>,
    /// Partition policy requested by [`Plan::with_partition`].
    pub(crate) partition: Option<(tprims_kernel::PartitionPolicy, tprims_kernel::PartitionOpts)>,
    /// The kernel layer's explicit tuning inputs (see [`Plan::with_tuning`]).
    pub(crate) tuning: tprims_kernel::Tuning,
    /// Orientation request (see [`Plan::with_orientation`]).
    pub(crate) orient: Orient,
    /// Row-block request (see [`Plan::with_row_block`]).
    pub(crate) row_block_mode: RowBlock,
    /// Partition rule or pin (see [`Plan::with_partition_mode`]).
    pub(crate) partition_mode: PartitionMode,
    /// Forced L3 domain count (see [`Plan::with_l3_domains`]).
    pub(crate) l3_domains: Option<usize>,
    pub(crate) resolved_cache: crate::resolve::Cache,
    /// The matrix shape and folded axes this plan reduced to. Public because
    /// it is the answer to "what did the index analysis actually decide", which
    /// nothing else reports.
    pub stats: PlanStats,
}

impl Plan {
    /// Explicit kernel-layer tuning: the ISA preference of the legacy menu, the
    /// blocking model and overrides, a coupled `kc` and the write-back mode. The
    /// default is the baseline with every knob unset.
    #[must_use]
    pub fn with_tuning(mut self, tuning: tprims_kernel::Tuning) -> Self {
        self.tuning = tuning;
        self.resolved_cache = Default::default();
        self
    }

    /// The plan's tuning inputs.
    pub fn tuning(&self) -> &tprims_kernel::Tuning {
        &self.tuning
    }

    /// Pin or rule the row/column orientation (default [`Orient::Rule`]).
    /// Affects role orientation only, never correctness.
    #[must_use]
    pub fn with_orientation(mut self, orient: Orient) -> Self {
        self.orient = orient;
        self.resolved_cache = Default::default();
        self
    }

    /// Request a micro-tile row block (default [`RowBlock::Auto`]).
    #[must_use]
    pub fn with_row_block(mut self, row_block: RowBlock) -> Self {
        self.row_block_mode = row_block;
        self.resolved_cache = Default::default();
        self
    }

    /// Choose the partition rule, or pin the layout outright (default
    /// [`PartitionMode::Domain`]). Distinct from [`Plan::with_partition`], which
    /// selects the packed driver's grid or dynamic tiles.
    #[must_use]
    pub fn with_partition_mode(mut self, mode: PartitionMode) -> Self {
        self.partition_mode = mode;
        self.resolved_cache = Default::default();
        self
    }

    /// Take the number of L3 domains the thread set spans as `n` instead of
    /// deriving it from the probed cache hierarchy (the derivation assumes
    /// compact placement).
    #[must_use]
    pub fn with_l3_domains(mut self, n: usize) -> Self {
        self.l3_domains = Some(n);
        self.resolved_cache = Default::default();
        self
    }

    /// Select a registered family, or restore automatic selection.
    /// Dtype agreement is checked by `resolved::<T>` before execution.
    ///
    /// # Errors
    /// Returns `KernelSelection` for an unknown/unbuilt id, unavailable CPU,
    /// ambiguous id or invalid descriptor. No forced id silently falls back.
    ///
    /// # Examples
    /// ```
    /// use tensorcontract::{KernelChoice, Layout, Operand, Plan};
    /// let l = Layout::col_major(&[2, 2]);
    /// let p = Plan::new(Operand::new(&l, &[0,2]), Operand::new(&l, &[2,1]),
    ///     None, Operand::new(&l, &[0,1]))?;
    /// let p = p.with_kernel(KernelChoice::Id("ref.f64.real.4x4".into()))?;
    /// assert_eq!(p.resolved::<f64>()?.family().id, "ref.f64.real.4x4");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_kernel(mut self, choice: tprims_kernel::KernelChoice) -> Result<Self> {
        if let Some(forced) = &self.forced {
            return Err(Error::KernelSelection(
                tprims_kernel::SelectError::Incompatible {
                    id: forced.id().into(),
                    reason: "a custom selector already chose this plan's kernel",
                },
            ));
        }
        if let tprims_kernel::KernelChoice::Id(id) = &choice {
            let cpu = tprims_kernel::CpuFeatures::detect();
            let mut failure = None;
            macro_rules! check {
                ($t:ty) => {
                    match tprims_kernel::Registry::select::<$t>(id, cpu) {
                        Ok(_) => {
                            failure = None;
                        }
                        Err(tprims_kernel::SelectError::DtypeMismatch { .. }) => {}
                        Err(e) => {
                            failure = Some(e);
                        }
                    }
                };
            }
            // INVARIANT: every id names one of the four sealed storage dtypes.
            // A dtype mismatch isn't a builder error before T is known.
            check!(f32);
            check!(f64);
            check!(crate::C32);
            check!(crate::C64);
            if let Some(e) = failure {
                return Err(Error::KernelSelection(e));
            }
        }
        self.kernel = Some(choice);
        self.resolved_cache = Default::default();
        Ok(self)
    }

    /// Resolve and cache the family/blocking for a built-in storage dtype.
    /// Clones and choice/blocking/method/width builders start fresh caches.
    ///
    /// # Errors
    /// Returns typed `SelectError` for invalid ids, dtype/CPU incompatibility,
    /// unsupported conjugation or overflowing blocking configuration.
    ///
    /// # Examples
    /// ```
    /// use tensorcontract::{Layout, Operand, Plan};
    /// let l = Layout::col_major(&[2, 2]);
    /// let p = Plan::new(Operand::new(&l, &[0,2]), Operand::new(&l, &[2,1]),
    ///     None, Operand::new(&l, &[0,1]))?;
    /// let first = p.resolved::<f64>()?;
    /// assert!(core::ptr::eq(first.family(), p.resolved::<f64>()?.family()));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resolved<T: tprims_kernel::Families>(
        &self,
    ) -> core::result::Result<tprims_kernel::ResolvedGemm<T::Real>, tprims_kernel::SelectError>
    where
        T::Real: crate::KernelSet,
    {
        self.resolved_cache.resolved::<T>(self)
    }

    /// Analyse a contraction. `c` may be `None`, in which case `beta` is
    /// ignored at execution time and `D` is overwritten.
    ///
    /// This is where every shape-dependent decision is made — classification,
    /// folding, and the scatter vectors — so it is the call to hoist out of a
    /// loop. The [`stats`][Plan::stats] field reports what it concluded.
    ///
    /// ```
    /// use tensorcontract::plan::Operand;
    /// use tensorcontract::{Layout, Plan};
    ///
    /// // D[h,i,j] = sum_k A[h,i,k] * B[h,k,j]: `h` batches, `k` contracts.
    /// let (h, i, j, k) = (b'h' as i64, b'i' as i64, b'j' as i64, b'k' as i64);
    /// let l = Layout::col_major(&[2, 3, 4]);
    /// let ld = Layout::col_major(&[2, 3, 3]);
    ///
    /// let plan = Plan::new(
    ///     Operand::new(&l, &[h, i, k]),
    ///     Operand::new(&Layout::col_major(&[2, 4, 3]), &[h, k, j]),
    ///     None,
    ///     Operand::new(&ld, &[h, i, j]),
    /// )
    /// .unwrap();
    ///
    /// // Every class folded to a single axis, and `h` became the batch.
    /// let s = &plan.stats;
    /// assert_eq!((s.m, s.n, s.k, s.batch), (3, 3, 4, 2));
    /// assert_eq!((s.m_axes.len(), s.n_axes.len(), s.k_axes.len()), (1, 1, 1));
    ///
    /// // Not `is_pure_gemm`, though: that means exactly *one* GEMM, and the
    /// // batch axis makes this a sequence of them.
    /// assert!(!s.is_pure_gemm);
    /// ```
    ///
    /// An unsupported contraction is rejected here rather than at execution:
    ///
    /// ```
    /// use tensorcontract::plan::Operand;
    /// use tensorcontract::{Error, Layout, Plan};
    ///
    /// let l = Layout::col_major(&[2, 2]);
    /// let (i, j, x) = (b'i' as i64, b'j' as i64, b'x' as i64);
    /// let ld = Layout::col_major(&[2, 2, 2]);
    ///
    /// // `x` appears only in the output: a broadcast, which TAPP calls case 5.
    /// let err = Plan::new(
    ///     Operand::new(&l, &[i, j]),
    ///     Operand::new(&l, &[j, i]),
    ///     None,
    ///     Operand::new(&ld, &[i, j, x]),
    /// )
    /// .unwrap_err();
    /// assert_eq!(err, Error::BroadcastIndexUnsupported { label: x });
    /// ```
    pub fn new(
        a: Operand<'_>,
        b: Operand<'_>,
        c: Option<Operand<'_>>,
        d: Operand<'_>,
    ) -> Result<Plan> {
        let ra = reduce_tensor("A", a.layout, a.idx)?;
        let rb = reduce_tensor("B", b.layout, b.idx)?;
        let rd = reduce_tensor("D", d.layout, d.idx)?;
        let rc = match c {
            Some(c) => reduce_tensor("C", c.layout, c.idx)?,
            // No C operand: mirror D so the C scatter vectors are well formed
            // and simply never read (beta is forced to zero).
            None => rd.clone(),
        };

        let mut labels: Vec<LabelInfo> = Vec::new();
        merge_into(&mut labels, &ra, |l, e, s| {
            l.sa = s;
            l.in_a = true;
            let _ = e;
        })?;
        merge_into(&mut labels, &rb, |l, e, s| {
            l.sb = s;
            l.in_b = true;
            let _ = e;
        })?;
        merge_into(&mut labels, &rc, |l, e, s| {
            l.sc = s;
            l.in_c = true;
            let _ = e;
        })?;
        merge_into(&mut labels, &rd, |l, e, s| {
            l.sd = s;
            l.in_d = true;
            let _ = e;
        })?;

        // C and D must describe the same index set.
        if labels.iter().any(|l| l.in_c != l.in_d) {
            return Err(Error::OutputLabelMismatch);
        }

        let mut m_ax = Vec::new();
        let mut n_ax = Vec::new();
        let mut k_ax = Vec::new();
        let mut h_ax = Vec::new();

        for l in &labels {
            let class = match (l.in_a, l.in_b, l.in_d) {
                (true, true, true) => Class::H,
                (true, false, true) => Class::M,
                (false, true, true) => Class::N,
                // contracted, or isolated in A / isolated in B (reductions)
                (true, _, false) | (false, true, false) => Class::K,
                (false, false, true) => {
                    return Err(Error::BroadcastIndexUnsupported { label: l.id })
                }
                // Two facts make this unreachable, and it takes both. Every
                // `LabelInfo` is created inside `merge_into`, whose only
                // constructing branch runs `set` immediately, and each of the
                // four `set` closures above turns on exactly one `in_*` flag —
                // so no label is in no operand at all. That leaves one way to
                // read `(false, false, false)`: a label in `C` alone. The
                // `in_c != in_d` check above has already rejected that as
                // `OutputLabelMismatch`, which is why `in_c` is absent from this
                // match rather than overlooked by it.
                (false, false, false) => unreachable!("label present in no operand"),
            };
            // Extent-1 axes contribute nothing; dropping them improves folding.
            if l.extent == 1 {
                continue;
            }
            let ax = Axis {
                extent: l.extent,
                sa: l.sa,
                sb: l.sb,
                sc: l.sc,
                sd: l.sd,
            };
            match class {
                Class::M => m_ax.push(ax),
                Class::N => n_ax.push(ax),
                Class::K => k_ax.push(ax),
                Class::H => h_ax.push(ax),
            }
        }

        // Order fastest-varying first, then fold.
        m_ax.sort_by_key(|x| (x.sd.unsigned_abs(), x.sa.unsigned_abs()));
        n_ax.sort_by_key(|x| (x.sd.unsigned_abs(), x.sb.unsigned_abs()));
        k_ax.sort_by_key(|x| (x.sa.unsigned_abs(), x.sb.unsigned_abs()));
        h_ax.sort_by_key(|x| (x.sd.unsigned_abs(), x.sa.unsigned_abs()));

        let m_ax = fold_axes(m_ax);
        let n_ax = fold_axes(n_ax);
        let k_ax = fold_axes(k_ax);
        let h_ax = fold_axes(h_ax);

        let a_m = build_scatter_for(&m_ax, |x| x.sa);
        let a_k = build_scatter_for(&k_ax, |x| x.sa);
        let b_k = build_scatter_for(&k_ax, |x| x.sb);
        let b_n = build_scatter_for(&n_ax, |x| x.sb);
        let c_m = build_scatter_for(&m_ax, |x| x.sc);
        let c_n = build_scatter_for(&n_ax, |x| x.sc);
        let d_m = build_scatter_for(&m_ax, |x| x.sd);
        let d_n = build_scatter_for(&n_ax, |x| x.sd);
        let h_a = build_scatter_for(&h_ax, |x| x.sa);
        let h_b = build_scatter_for(&h_ax, |x| x.sb);
        let h_c = build_scatter_for(&h_ax, |x| x.sc);
        let h_d = build_scatter_for(&h_ax, |x| x.sd);

        let stats = PlanStats {
            m: d_m.len(),
            n: d_n.len(),
            k: a_k.len(),
            batch: h_d.len(),
            is_pure_gemm: m_ax.len() <= 1 && n_ax.len() <= 1 && k_ax.len() <= 1 && h_ax.is_empty(),
            m_axes: m_ax,
            n_axes: n_ax,
            k_axes: k_ax,
            h_axes: h_ax,
        };

        let d_m_run = run_structure(&d_m).unwrap_or((d_m.len(), 0));
        let d_n_run = run_structure(&d_n).unwrap_or((d_n.len(), 0));

        Ok(Plan {
            a_m,
            a_k,
            b_k,
            b_n,
            c_m,
            c_n,
            d_m,
            d_n,
            h_a,
            h_b,
            h_c,
            h_d,
            d_m_run,
            d_n_run,
            conj_a: a.op.is_conj(),
            conj_b: b.op.is_conj(),
            conj_c: c.map(|c| c.op.is_conj()).unwrap_or(false),
            conj_d: d.op.is_conj(),
            blocking: None,
            method: None,
            threads: None,
            kernel: None,
            forced: None,
            partition: None,
            tuning: Default::default(),
            orient: Orient::default(),
            row_block_mode: RowBlock::default(),
            partition_mode: PartitionMode::default(),
            l3_domains: None,
            resolved_cache: Default::default(),
            stats,
        })
    }

    /// Choose how complex arithmetic is induced from real micro-kernels.
    ///
    /// Ignored for real element types. Without this, the plan uses
    /// [`ComplexMethod::Planar`].
    ///
    /// [`ComplexMethod::Planar`]: crate::kernel::ComplexMethod::Planar
    ///
    /// ```
    /// # use tensorcontract::{Layout, Plan, Operand, ComplexMethod};
    /// # let la = Layout::col_major(&[4, 5]);
    /// # let lb = Layout::col_major(&[5, 6]);
    /// # let ld = Layout::col_major(&[4, 6]);
    /// let plan = Plan::new(
    ///     Operand::new(&la, &[0, 2]),
    ///     Operand::new(&lb, &[2, 1]),
    ///     None,
    ///     Operand::new(&ld, &[0, 1]),
    /// )?
    /// .with_complex_method(ComplexMethod::ThreeM);
    /// # Ok::<(), tensorcontract::Error>(())
    /// ```
    #[must_use]
    pub fn with_complex_method(mut self, method: crate::kernel::ComplexMethod) -> Self {
        self.method = Some(method);
        self.resolved_cache = Default::default();
        self
    }

    /// The complex method this plan will execute with.
    pub fn complex_method(&self) -> crate::kernel::ComplexMethod {
        self.method.unwrap_or_default()
    }

    /// Override the cache blocking parameters this plan executes with.
    ///
    /// Mainly for parameter sweeps and for driving every level of the loop
    /// nest on small test problems; the defaults are derived from the element
    /// type and the target's cache sizes.
    #[must_use]
    pub fn with_blocking(mut self, blk: crate::kernel::Blocking) -> Self {
        self.blocking = Some(blk);
        self.resolved_cache = Default::default();
        self
    }

    /// Choose how the packed driver assigns output work to its team:
    /// [`PartitionPolicy::StaticGrid`] (the default, with its own cost model
    /// when `pm == pn == 0`) or the opt-in dynamic
    /// [`PartitionPolicy::DynamicTiles`].
    ///
    /// The policy is validated against the resolved family when the plan is
    /// resolved ([`Plan::resolved`], which every safe execution and the
    /// high-level planners call first), so an invalid job extent, a job count
    /// that overflows, or `DynamicTiles` with `align_c_lines` is a typed
    /// `KernelSelection` error before any compute, for empty problems too.
    /// Job extents are logical rows/columns of the *oriented* GEMM (the row
    /// role may be either user operand, see [`Plan::transposes_gemm`]).
    ///
    /// [`PartitionPolicy`]: tprims_kernel::PartitionPolicy
    /// [`PartitionPolicy::StaticGrid`]: tprims_kernel::PartitionPolicy::StaticGrid
    /// [`PartitionPolicy::DynamicTiles`]: tprims_kernel::PartitionPolicy::DynamicTiles
    #[must_use]
    pub fn with_partition(
        mut self,
        policy: tprims_kernel::PartitionPolicy,
        opts: tprims_kernel::PartitionOpts,
    ) -> Self {
        self.partition = Some((policy, opts));
        self.resolved_cache = Default::default();
        self
    }

    /// Plan for `n` threads: the width the family's blocking is resolved for
    /// and [`Plan::partition`] reports. Execution threads come only from the
    /// [`tprims_exec::Exec`] passed to `run_with` (width = its budget), and a
    /// plan run without one is serial whatever `n` is.
    ///
    /// Parallelism is a 2-D partition of the output, `ceil(M / MR)` row panels
    /// by `ceil(N / NR)` column blocks, so it is capped at their product — see
    /// [`Plan::partition`] for how the two axes are apportioned. A thread count
    /// above the cap is silently reduced.
    ///
    /// **`n == 0` is clamped to 1, not rejected.** One is the floor because
    /// there is no zero-thread execution to name: at `n == 1` the driver runs
    /// the whole contraction on the calling thread, so `0` and `1` would have to
    /// mean the same thing whatever the signature. Clamping keeps this a
    /// `Self`-returning builder rather than a `Result` for an argument no caller
    /// passes deliberately. [`Plan::threads`] reports what will actually be
    /// used, and is the way to read back a clamp.
    ///
    /// Each thread owns a disjoint set of output blocks and its own packed `A`,
    /// and no reduction is parallelised, so the summation order over `k` is the
    /// serial one at every thread count.
    ///
    /// Results are **bitwise identical** for every thread count, so this is
    /// never a numerical decision.
    #[must_use]
    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads = Some(n.max(1));
        self.resolved_cache = Default::default();
        self
    }

    /// The thread count this plan is resolved and partitioned for (see
    /// [`Plan::with_threads`]); it is not an execution width.
    ///
    /// Defaults to **1**.
    pub fn threads(&self) -> usize {
        self.threads.unwrap_or(1)
    }

    /// How execution will split the output across threads: `(pm, pn)`, the
    /// number of contiguous row strips of whole `MR` panels and the number of
    /// column groups of whole `NR` blocks. Their product is the number of
    /// threads that will actually run, and never exceeds [`Plan::threads`].
    ///
    /// One caveat: this is the partition at the **requested** thread count. The
    /// driver may run fewer -- `execute` takes a cap -- so
    /// [`Plan::partition_with`] is the form that answers "what will actually
    /// run" at a given width.
    ///
    /// Both counts are in the **oriented** directions, i.e. after the
    /// [`Plan::transposes_gemm`] swap: on a plan that computes `D^T = B^T A^T`
    /// the row strips run along `N`, so a `1 x 33` output parallelises 33 ways
    /// and not one way.
    ///
    /// This is a **tier-2** answer: the signature is stable, the value is a
    /// tuning output and will move when the rule is re-measured. Do not encode
    /// one of these answers as a constant.
    ///
    /// # The rule, in outline
    ///
    /// `pn == 1` whenever the `M` direction alone can fill the threads, which is
    /// the overwhelmingly common case. Only when `ceil(M / MR) < p` does the `N`
    /// direction get involved, and then `(pm, pn)` minimises
    ///
    /// ```text
    /// cost(pm, pn) = ceil(panels/pm) * (NR * ceil(blocks/pn) + PACK_WEIGHT)
    /// ```
    ///
    /// — one thread's share of the work, in micro-kernel lane-slots per unit of
    /// `k`, including the packed `A` block it builds for itself however few
    /// columns it owns. That second term is the only reason the objective is not
    /// simply "balance the tiles".
    ///
    /// That early return is correct only where the threads share one L3, and
    /// wrong by up to 4.3x where they span many, so it is **gated** on
    /// [`cache::l3_domains`](crate::kernel::cache::l3_domains) rather than
    /// removed: the row axis gives way to `1 x p` only when the thread set spans
    /// more than one domain, the column axis can fill the threads by itself, and
    /// the contraction is shallow. `columns_beat_rows` is that predicate, kept
    /// pure so its truth table can be pinned by a test on any machine.
    /// [`Plan::with_partition_mode`] selects `domain` (the default, D44) or
    /// `legacy`, or pins the layout outright.
    ///
    /// The derivation of `PACK_WEIGHT`, the three gate conditions and what each
    /// one is worth are in `docs/notebook/` — D29, D41, D42, D44 and A28, with the
    /// measurements in the threading chapter.
    pub fn partition(&self, mr: usize, nr: usize) -> (usize, usize) {
        self.partition_with(mr, nr, self.threads())
    }

    /// [`Plan::partition`] at a caller-supplied thread count.
    ///
    /// Exists because the driver may run at a width other than the requested
    /// one: `execute` takes a cap, and the batched entry point divides the
    /// available threads across items. `partition` is this at
    /// [`Plan::threads`].
    pub fn partition_with(&self, mr: usize, nr: usize, threads: usize) -> (usize, usize) {
        /// Cost of packing one `A` element relative to one micro-kernel lane-FMA.
        /// See [`Plan::partition`]; deliberately at the conservative end.
        const PACK_WEIGHT: usize = 8;

        let (rows, cols) = if self.transposes_gemm(mr) {
            (self.b_n.len(), self.a_m.len())
        } else {
            (self.a_m.len(), self.b_n.len())
        };
        let panels = rows.div_ceil(mr.max(1)).max(1);
        let blocks = cols.div_ceil(nr.max(1)).max(1);
        let p = threads.max(1);

        let domain_aware = match self.partition_mode {
            PartitionMode::Rule => false,
            PartitionMode::Domain => true,
            PartitionMode::Rows => return (p.min(panels), 1),
            PartitionMode::Cols => return (1, p.min(blocks)),
            PartitionMode::Pin(pm, pn) => {
                return (pm.clamp(1, panels), pn.clamp(1, blocks));
            }
        };
        // The row direction alone fills the threads: the 1-D partition, bit for
        // bit the pre-2-D behaviour, and the only case the corpus mostly needs.
        // Unless the threads do not share an L3, the column axis could fill them
        // just as well, and the case is shallow enough for the replicated `B`
        // panel to be what limits it — then, and only then, the axes swap.
        if panels >= p {
            let domains = if domain_aware {
                crate::kernel::cache::l3_domains(p, self.l3_domains)
            } else {
                1
            };
            if columns_beat_rows(blocks, self.a_k.len(), p, domains) {
                return (1, p.min(blocks));
            }
            return (p, 1);
        }
        let mut best = (1, 1);
        let mut best_cost = usize::MAX;
        for pm in 1..=panels {
            let pn = (p / pm).min(blocks);
            let cost = panels.div_ceil(pm) * (nr.max(1) * blocks.div_ceil(pn) + PACK_WEIGHT);
            // `<=`, so among equal-cost partitions the largest `pm` wins: the
            // `M` split needs no duplicated packing and is the measured one.
            if cost <= best_cost {
                best = (pm, pn);
                best_cost = cost;
            }
        }
        best
    }

    /// The plan's scatter vectors.
    ///
    /// Exposed so that alternative execution strategies (a TTGT baseline, a
    /// GPU offload path) can reuse the index analysis verbatim and be compared
    /// against the block-scatter engine on exactly equal footing.
    pub fn scatters(&self) -> Scatters<'_> {
        Scatters {
            a_m: &self.a_m,
            a_k: &self.a_k,
            b_k: &self.b_k,
            b_n: &self.b_n,
            c_m: &self.c_m,
            c_n: &self.c_n,
            d_m: &self.d_m,
            d_n: &self.d_n,
            h_a: &self.h_a,
            h_b: &self.h_b,
            h_c: &self.h_c,
            h_d: &self.h_d,
        }
    }

    /// The scatter vectors in the row/column orientation execution will use.
    ///
    /// Identical to [`Plan::scatters`] unless [`Plan::transposes_gemm`], in
    /// which case `(A, M)` and `(B, N)` are exchanged. Report block-scatter
    /// regularity against *this*, not [`Plan::scatters`]: after a swap the
    /// engine blocks `B`'s column scatter at `MR` and `A`'s row scatter at
    /// `NR`, so regularity measured on the unswapped vectors describes a
    /// traversal that never happens.
    pub fn oriented_scatters(&self, mr: usize) -> Scatters<'_> {
        let s = self.scatters();
        if !self.transposes_gemm(mr) {
            return s;
        }
        Scatters {
            a_m: s.b_n,
            a_k: s.b_k,
            b_k: s.a_k,
            b_n: s.a_m,
            c_m: s.c_n,
            c_n: s.c_m,
            d_m: s.d_n,
            d_n: s.d_m,
            h_a: s.h_b,
            h_b: s.h_a,
            h_c: s.h_c,
            h_d: s.h_d,
        }
    }

    /// Whether execution will compute `D^T = B^T A^T` rather than `D = A B`.
    ///
    /// The engine is symmetric under exchanging `(A, M)` with `(B, N)`: doing
    /// so transposes the matrix view of `C` and `D` and changes nothing else.
    /// It is worth doing when it makes the `MR` rows of a micro-tile a single
    /// contiguous run of `D`, because the write-back runs under the innermost
    /// loop and is the one access packing cannot hide. On the nine corpus cases
    /// Phase 3 profiled as a 2x defect this is worth up to 2.7x.
    ///
    /// The rule is a preference with a fallback, not a test with a veto:
    ///
    /// 1. **Prefer the arm whose micro-tile row block lands inside a single run
    ///    of `D`** — unit stride and a run of at least `MR`. If exactly one arm
    ///    manages that, take it. If both do, stay put; there is nothing to buy.
    /// 2. **Otherwise put the direction with the *shorter* run in the row
    ///    role**, whichever operand that is.
    ///
    /// Step 2 is what the Phase 4.1 rule was missing: it treated "the row block
    /// would be shattered either way" as a reason never to swap, and all nine of
    /// its known misses lived in that case (A14). The rule has to be
    /// antisymmetric under exchanging the two directions, because the `abcijk`
    /// families are exact mirror images of each other.
    ///
    /// Step 1 dominates step 2, and must: it is why `c64` (`MR = 16` against a
    /// run of 24) takes the opposite arm from `f32` (`MR = 48`) on the same
    /// shapes. That element-type dependence is why the choice is not a property
    /// of the plan alone — hence the `mr` argument.
    ///
    /// This is a **tier-2** answer: stable signature, tuning-output value.
    /// [`Plan::with_orientation`] can disable the swap or force it; neither
    /// affects correctness. The mirror-family table the rule was derived
    /// from, the scoring against both forced arms of all 392 corpus
    /// case-dtype-methods, and the 21 cases still on the slower arm are in
    /// `docs/notebook/` — the write-back chapter, Phase 4.1d.
    pub fn transposes_gemm(&self, mr: usize) -> bool {
        match self.orient {
            Orient::Rule => self.transposes_gemm_rule(mr),
            Orient::Force(v) => v,
            Orient::Legacy => self.transposes_gemm_legacy(mr),
        }
    }

    /// [`Plan::transposes_gemm`]'s rule with no environment override.
    ///
    /// Separate so the unit tests can assert what the *rule* decides even for a
    /// plan whose orientation is pinned.
    fn transposes_gemm_rule(&self, mr: usize) -> bool {
        // A row block lands inside one run when the rows are unit-stride and
        // the run is at least `MR` long.
        let fits = |(run, stride): (usize, i64)| stride == 1 && run >= mr;
        let (ab, ba) = (fits(self.d_m_run), fits(self.d_n_run));
        if ab != ba {
            return ba;
        }
        if ab {
            return false;
        }
        // A tie-break between two imperfect arms is not a tie when one of them
        // cannot fill a micro-tile: most of every `MR x NR` block would be
        // padding, which is arithmetic rather than a cache effect. The corpus
        // cannot test this — its shortest direction is 32 against a largest
        // `MR` of 48, and neither of the two cases where that bites reaches the
        // fallback — so this guard is inert on every measurement quoted here.
        let (m_rows, n_rows) = (self.d_m.len(), self.d_n.len());
        if (m_rows >= mr) != (n_rows >= mr) {
            return n_rows >= mr;
        }
        self.d_n_run.0 < self.d_m_run.0
    }

    /// The Phase 4.1 orientation rule, kept reachable as
    /// `Orient::Legacy` so that the current one can be measured
    /// against it as an A/B rather than a diff between two builds (A15).
    ///
    /// Swap only when `D`'s column direction is strictly more contiguous than
    /// its row direction *and* the swap leaves the row block unbroken. The
    /// second condition is a veto with no fallback, which is where all nine of
    /// its known misses live; see [`Plan::transposes_gemm`].
    fn transposes_gemm_legacy(&self, mr: usize) -> bool {
        let lead = |axes: &[Axis]| axes.first().map_or(u64::MAX, |a| a.sd.unsigned_abs());
        if lead(&self.stats.n_axes) >= lead(&self.stats.m_axes) {
            return false;
        }
        match self.stats.n_axes.first() {
            Some(ax) => ax.sd == 1 && ax.extent >= mr as i64,
            None => false,
        }
    }

    /// Choose the micro-tile row block `MR` from a menu of shapes the kernel
    /// set actually has, ordered fastest-in-isolation first.
    ///
    /// Returns `None` for "use the menu's default", which is also what
    /// `RowBlock::Base` gives.
    ///
    /// # Why `MR` is not just a kernel-tuning constant
    ///
    /// `MR` is the granularity at which the *output's* row scatter is blocked,
    /// so it decides which of the write-back's three paths each block
    /// takes. When `D`'s rows come in contiguous runs of `r` elements, an
    /// aligned `MR`-block lies inside one run — and so gets the unit-stride
    /// path — only when `MR` divides into the run pattern; otherwise it
    /// straddles a discontinuity, the block scatter reports [`IRREGULAR`], and
    /// the whole block falls back to a gather. That is a code-path change, not
    /// a tuning delta, and the shapes it wants are not the shapes peak
    /// throughput wants. The TCCG corpus rounds every stride-1 index up to a
    /// multiple of 24, while `MR` at `L = 16` lanes is 16, 32 or 48.
    ///
    /// [`IRREGULAR`]: crate::scatter::IRREGULAR
    ///
    /// # The rule, and the two guards it needs
    ///
    /// Take the first shape on the menu that makes *every* output row block a
    /// single run, but only when both of these hold. (A third guard existed and
    /// was removed in Phase 4.1d; see below.) Each guard is there
    /// because the grid in `bench-results/phase4c` measured what happens
    /// without it; none is a plausibility argument.
    ///
    /// 1. **The contraction is shallow** (`k <= 32`). The write-back costs a
    ///    constant per output element against `~4k` flops of kernel work, so
    ///    the path it takes only matters while `k` is small — and a shape off
    ///    the kernel's peak always costs something.
    /// 2. **The default is substantially broken** (`wb <= 0.75`). A shape change
    ///    is not free, so a marginal improvement cannot repay it.
    ///
    /// A third guard, against a shape change flipping the orientation as a side
    /// effect, was removed in Phase 4.1d once the orientation rule was fixed.
    /// The two rules are coupled and cannot be tuned separately.
    ///
    /// Unguarded, the rule is a **loss** — maximising the regular fraction alone
    /// scores 0.936 in `f32`. This is a **tier-2** answer: stable signature,
    /// tuning-output value. Both thresholds, what each guard is worth, the
    /// corpus firing count and the gain an oracle leaves on the table are in
    /// `docs/notebook/` (the write-back chapter, Phase 4.1c and 4.1d); the grid
    /// they were scored against is `bench-results/phase4c`.
    ///
    /// # What `menu` is, and what comes back
    ///
    /// `menu` is the kernel set's `(MR, NR)` shapes, default first, and the
    /// answer is a **position in it** — not an `MR`. The distinction is not
    /// cosmetic: two entries may share an `MR` and differ only in `NR`, which is
    /// a shape the measurement asked for and an `MR`-keyed menu could not hold
    /// (A35). The rule below reads only the `MR`, so entries of equal height
    /// score alike and the earlier one wins, which keeps the measured default in
    /// front.
    pub fn row_block(&self, menu: &[(usize, usize)]) -> Option<usize> {
        match self.row_block_mode {
            RowBlock::Base => None,
            RowBlock::Auto => self.preferred_row_block(menu),
            RowBlock::Pin(mr) => menu.iter().position(|&(m, _)| m == mr),
            RowBlock::Index(i) => (i < menu.len()).then_some(i),
        }
    }

    /// [`Plan::row_block`]'s rule with no environment override, so that it can
    /// be scored offline against measured ground truth. Returns a menu position.
    pub fn preferred_row_block(&self, menu: &[(usize, usize)]) -> Option<usize> {
        /// Above this contraction depth the write-back is amortised and the
        /// shape's own cost is all that is left. See [`Plan::row_block`], and
        /// note this is a different threshold from [`BANDWIDTH_BOUND_K`], which
        /// asks a different question.
        const SHALLOW_K: usize = 32;
        /// A default this regular already is not worth paying a shape change
        /// to improve.
        const BROKEN_ENOUGH: f64 = 0.75;

        let (&(default, _), rest) = menu.split_first()?;
        if self.stats.k > SHALLOW_K || self.row_block_score(default) > BROKEN_ENOUGH {
            return None;
        }
        rest.iter()
            .position(|&(mr, _)| self.row_block_score(mr) >= 1.0 - 1e-9)
            .map(|i| i + 1)
    }

    /// Fraction of the output's row blocks that would stay off
    /// the write-back's gather path at row block `mr`, evaluated in the
    /// orientation `mr` itself selects.
    ///
    /// `0.0` when the output's rows have no uniform run structure — then `MR`
    /// has no predictable effect and every shape scores alike, which leaves the
    /// tie-break to keep the default.
    pub fn row_block_score(&self, mr: usize) -> f64 {
        let (rows, run) = if self.transposes_gemm(mr) {
            (&self.d_n, self.d_n_run)
        } else {
            (&self.d_m, self.d_m_run)
        };
        // A zero stride marks "no uniform run structure", where `MR` has no
        // predictable effect.
        if run.1 == 0 {
            return 0.0;
        }
        unbroken_fraction(rows.len(), run.0, mr)
    }

    /// `true` when the contraction produces no output elements.
    pub fn is_empty(&self) -> bool {
        self.stats.m == 0 || self.stats.n == 0 || self.stats.batch == 0
    }

    /// `true` when the contraction dimension is empty, so `D = beta * C`.
    pub fn has_empty_contraction(&self) -> bool {
        self.stats.k == 0
    }
}

/// How the contraction is oriented into a matrix product: which operand plays
/// the GEMM row role. Orientation never changes the result, only which
/// register tile shape suits the output strides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Orient {
    /// [`Plan::transposes_gemm`]'s rule. The default.
    #[default]
    Rule,
    /// A pinned arm, for forced-arm measurement of both.
    Force(bool),
    /// The Phase 4.1 rule, for measuring the current one against it.
    Legacy,
}

impl Orient {
    /// Parse `rule | none | ab | swap | ba | legacy | phase41`
    /// (case-insensitively); `None` for any other spelling.
    pub fn parse(s: &str) -> Option<Orient> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rule" | "auto" => Some(Orient::Rule),
            "none" | "ab" => Some(Orient::Force(false)),
            "swap" | "ba" => Some(Orient::Force(true)),
            "legacy" | "phase41" => Some(Orient::Legacy),
            _ => None,
        }
    }
}

/// Which micro-tile row block a plan asks for.
///
/// The default is `Auto`; `Base` pins the Phase 3 shape, which is what the rule
/// was measured against. Both arms stay reachable so the comparison can be
/// repeated in one session rather than as a diff between two builds (A15), and
/// `Index` re-runs the whole grid the rule was derived from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowBlock {
    /// The kernel set's default shape — the Phase 3 choice.
    Base,
    /// [`Plan::row_block`]'s rule. The default.
    #[default]
    Auto,
    /// A specific logical `MR`, where the kernel set has one.
    Pin(usize),
    /// A position on the menu, which is comparable across element types and
    /// methods in a way a bare `MR` is not — `mr=16` names different shapes in
    /// `f32` and `f64`, `idx=1` names "the first alternate" in both.
    Index(usize),
}

impl RowBlock {
    /// Parse `base | default | auto | mr=<n> | idx=<i>` (case-insensitively);
    /// `None` for any other spelling.
    pub fn parse(s: &str) -> Option<RowBlock> {
        let v = s.trim().to_ascii_lowercase();
        match v.as_str() {
            "auto" => Some(RowBlock::Auto),
            "base" | "default" => Some(RowBlock::Base),
            _ => {
                let num = |p: &str| v.strip_prefix(p)?.parse::<usize>().ok();
                num("mr=")
                    .map(RowBlock::Pin)
                    .or_else(|| num("idx=").map(RowBlock::Index))
            }
        }
    }
}

/// The depth below which a contraction is bandwidth-bound enough for the
/// cross-domain `B` replication to be what limits it. The project's existing
/// memory-bound criterion is `min(n, k) <= 64` (see `CLAUDE.md`) and this is the
/// same 64; the corpus puts `k = 24` on one side of it and `k >= 204` on the
/// other, so nothing in it is near the boundary and the threshold is a
/// separation, not a tuned constant.
///
/// Named at length because `preferred_row_block` has its own `SHALLOW_K = 32`
/// meaning something else — there the question is whether the write-back is
/// amortised, here it is whether the case is bandwidth-bound. Two thresholds
/// that both mean "shallow" and are not the same number; keep them
/// distinguishable at the point of use.
const BANDWIDTH_BOUND_K: usize = 64;

/// The domain-aware gate: in the regime where the row axis alone fills the
/// threads (`panels >= p`, which the caller has already established), should the
/// column axis take them instead?
///
/// All three conditions were put here by measurement:
///
/// 1. **`domains > 1`** — the thread set spans more than one L3. This is the
///    mechanism, and the only quantity A36 separates from thread count: Ice Lake
///    runs 32 threads over one domain and reads 1.014, Zen2 runs 4 over one and
///    reads 1.031, while three multi-domain points rise monotonically to 2.38x.
///    Passing `1` is also how [`Plan::partition`] expresses the legacy default.
/// 2. **`blocks >= p`** — the column axis can fill the threads by itself, so the
///    swap costs no parallelism. Without it the corpus's narrow cases lose 2–5x
///    by running on a fraction of their cores.
/// 3. **`k <= BANDWIDTH_BOUND_K`** — the penalty being dodged is bandwidth, so it can
///    only dominate where the case is bandwidth-bound, and `k` is this corpus's
///    knob for that. The whole effect was measured on the `k = 24` family; the
///    wide compute-bound families (`ijkl-*`, `ij-ik-kj`, `k` 2704–5184) were
///    measured *losing* 25% in the complex methods from the same swap. The guard
///    confines the rule to the population the evidence covers.
///
/// A pure function of four numbers so that the whole truth table can be pinned
/// by a test on any machine, rather than only on a chiplet one.
fn columns_beat_rows(blocks: usize, k: usize, p: usize, domains: usize) -> bool {
    domains > 1 && blocks >= p && k <= BANDWIDTH_BOUND_K
}

/// The partition rule, or a pinned layout. Every partition gives bitwise
/// identical results.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PartitionMode {
    /// [`Plan::partition`]'s rule with the `panels >= p` early return
    /// unconditional (`legacy`). Every threaded number committed before
    /// 2026-08-04 was measured with it, so reproducing one means asking for it
    /// by name.
    Rule,
    /// The same rule with the early return **gated on the L3 domain count**, so
    /// a thread set that spans several L3s takes the column axis on shallow
    /// contractions wide enough to afford it. **The default** (D44), and a no-op
    /// on any machine where one L3 serves the thread set. See
    /// [`Plan::partition`] and [`columns_beat_rows`].
    #[default]
    Domain,
    /// One-dimensional over the oriented `M` direction — the partition Phase 4
    /// item 4 shipped, kept reachable so the 2-D rule can be measured against
    /// it as an A/B rather than a diff between two builds.
    Rows,
    /// One-dimensional over the oriented `N` direction, which is the other
    /// extreme and the one that duplicates the packed `A` block the most.
    Cols,
    /// A pinned `(pm, pn)`, clamped only by the panel and block counts — so a
    /// pin also sets the thread count, rather than being capped by it.
    Pin(usize, usize),
}

impl PartitionMode {
    /// Parse `legacy | rule | domain | domains | m | rows | 1d | n | cols |
    /// <pm>x<pn>` (case-insensitively); `None` for any other spelling.
    pub fn parse(s: &str) -> Option<PartitionMode> {
        let v = s.trim().to_ascii_lowercase();
        match v.as_str() {
            "m" | "rows" | "1d" => Some(PartitionMode::Rows),
            "n" | "cols" => Some(PartitionMode::Cols),
            "domain" | "domains" => Some(PartitionMode::Domain),
            "legacy" | "rule" => Some(PartitionMode::Rule),
            _ => {
                let (pm, pn) = v.split_once('x')?;
                Some(PartitionMode::Pin(
                    pm.parse::<usize>().ok()?.max(1),
                    pn.parse::<usize>().ok()?.max(1),
                ))
            }
        }
    }
}

/// Collapse repeated labels within one tensor onto its diagonal, validating
/// rank and extents along the way. Returns `(label, extent, stride)` triples.
fn reduce_tensor(name: &'static str, layout: &Layout, idx: &[i64]) -> Result<Vec<(i64, i64, i64)>> {
    if layout.ndim() != idx.len() {
        return Err(Error::LabelCountMismatch {
            tensor: name,
            nmode: layout.ndim(),
            nlabel: idx.len(),
        });
    }
    let mut out: Vec<(i64, i64, i64)> = Vec::with_capacity(idx.len());
    for (k, &l) in idx.iter().enumerate() {
        let e = layout.extents()[k];
        let s = layout.strides()[k];
        if e < 0 {
            return Err(Error::NegativeExtent {
                label: l,
                extent: e,
            });
        }
        match out.iter_mut().find(|x| x.0 == l) {
            Some(slot) => {
                if slot.1 != e {
                    return Err(Error::ExtentMismatch {
                        label: l,
                        expected: slot.1,
                        found: e,
                    });
                }
                slot.2 += s; // diagonal: strides add
            }
            None => out.push((l, e, s)),
        }
        // Every scatter vector the plan builds is as long as the product of the
        // extents of *some subset* of one tensor's axes, so checking each
        // tensor's whole product here bounds all of them at once. Unchecked, the
        // product wraps: `Vec::with_capacity` of a wrapped length then gives a
        // plan that computes nothing and reports success in release, and aborts
        // the process in debug. A C caller cannot be given either.
        if product_overflows(&out) {
            return Err(Error::ExtentProductOverflow { tensor: name });
        }
    }
    Ok(out)
}

fn merge_into(
    labels: &mut Vec<LabelInfo>,
    reduced: &[(i64, i64, i64)],
    set: impl Fn(&mut LabelInfo, i64, i64),
) -> Result<()> {
    for &(id, extent, stride) in reduced {
        match labels.iter_mut().find(|x| x.id == id) {
            Some(l) => {
                if l.extent != extent {
                    return Err(Error::ExtentMismatch {
                        label: id,
                        expected: l.extent,
                        found: extent,
                    });
                }
                set(l, extent, stride);
            }
            None => {
                let mut l = LabelInfo {
                    id,
                    extent,
                    sa: 0,
                    sb: 0,
                    sc: 0,
                    sd: 0,
                    in_a: false,
                    in_b: false,
                    in_c: false,
                    in_d: false,
                };
                set(&mut l, extent, stride);
                labels.push(l);
            }
        }
    }
    Ok(())
}

/// Merge adjacent axes whose strides are compatible in every operand.
fn fold_axes(axes: Vec<Axis>) -> Vec<Axis> {
    let mut out: Vec<Axis> = Vec::with_capacity(axes.len());
    for ax in axes {
        if let Some(p) = out.last_mut() {
            if ax.sa == p.sa * p.extent
                && ax.sb == p.sb * p.extent
                && ax.sc == p.sc * p.extent
                && ax.sd == p.sd * p.extent
            {
                p.extent *= ax.extent;
                continue;
            }
        }
        out.push(ax);
    }
    out
}

/// Whether the extents of these `(label, extent, stride)` triples overflow an
/// `i64` (or a `usize`) when multiplied.
///
/// `usize` matters as well as `i64` on 32-bit targets, where a product that fits
/// an `i64` can still exceed an allocation's addressable length.
fn product_overflows(axes: &[(i64, i64, i64)]) -> bool {
    let mut total: i64 = 1;
    for &(_, e, _) in axes {
        total = match total.checked_mul(e) {
            Some(t) => t,
            None => return true,
        };
    }
    usize::try_from(total).is_err()
}

fn build_scatter_for(axes: &[Axis], pick: impl Fn(&Axis) -> i64) -> Vec<i64> {
    let extents: Vec<i64> = axes.iter().map(|a| a.extent).collect();
    let strides: Vec<i64> = axes.iter().map(pick).collect();
    build_scatter(&extents, &strides)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lay(e: &[i64]) -> Layout {
        Layout::col_major(e)
    }

    /// The domain-aware gate's whole truth table, machine-independently.
    ///
    /// Every one of the three conditions is here because measurement put it
    /// there, so each gets a case that turns it off on its own — a gate that
    /// quietly stopped consulting one of them would still look right on the
    /// corpus's `abcijk` family, which satisfies all three.
    #[test]
    fn domain_gate_needs_all_three_conditions() {
        // The measured population: 64 threads over 16 L3 domains, 1024 column
        // blocks against 768 row panels, `k = 24`. Worth up to 4.3x (A36).
        assert!(columns_beat_rows(1024, 24, 64, 16));
        // One L3 domain: the early return is *correct* here, at any thread count.
        // Ice Lake reaches 32 threads on one domain and reads 1.014 — the point
        // that separates domain count from thread count — so it is pinned at both
        // ends of the thread range.
        assert!(!columns_beat_rows(1024, 24, 32, 1));
        assert!(!columns_beat_rows(1024, 24, 4, 1));
        // Too few column blocks to feed the threads: swapping would run the case
        // on a fraction of its cores, which the corpus's narrow half pays 2-5x for.
        assert!(!columns_beat_rows(63, 24, 64, 16));
        assert!(columns_beat_rows(64, 24, 64, 16));
        // Deep enough to be compute-bound: the cross-domain penalty is a
        // bandwidth cost and cannot dominate here. `ijkl-*` and `ij-ik-kj` sit on
        // this side and were measured *losing* 25% in the complex methods.
        assert!(!columns_beat_rows(1024, 2704, 64, 16));
        assert!(columns_beat_rows(1024, BANDWIDTH_BOUND_K, 64, 16));
        assert!(!columns_beat_rows(1024, BANDWIDTH_BOUND_K + 1, 64, 16));
        // Serial is never a partition question.
        assert!(!columns_beat_rows(1024, 24, 1, 1));
    }

    #[test]
    fn plain_matmul_folds_to_pure_gemm() {
        // C[i,j] = A[i,k] B[k,j], all column-major.
        let a = lay(&[4, 5]);
        let b = lay(&[5, 6]);
        let d = lay(&[4, 6]);
        let p = Plan::new(
            Operand::new(&a, &[0, 2]),
            Operand::new(&b, &[2, 1]),
            None,
            Operand::new(&d, &[0, 1]),
        )
        .unwrap();
        assert_eq!(
            (p.stats.m, p.stats.n, p.stats.k, p.stats.batch),
            (4, 6, 5, 1)
        );
        assert!(p.stats.is_pure_gemm);
    }

    #[test]
    fn adjacent_free_indices_fold() {
        // D[a,b,j] = A[a,b,k] B[k,j] with a,b adjacent in both A and D.
        let a = lay(&[4, 3, 5]);
        let b = lay(&[5, 6]);
        let d = lay(&[4, 3, 6]);
        let p = Plan::new(
            Operand::new(&a, &[0, 1, 3]),
            Operand::new(&b, &[3, 2]),
            None,
            Operand::new(&d, &[0, 1, 2]),
        )
        .unwrap();
        assert_eq!(p.stats.m_axes.len(), 1, "a and b should fold into one axis");
        assert_eq!(p.stats.m, 12);
        assert!(p.stats.is_pure_gemm);
    }

    #[test]
    fn hadamard_index_recognised() {
        // D[b,i,j] = A[b,i,k] B[b,k,j]
        let a = lay(&[2, 4, 5]);
        let b = lay(&[2, 5, 6]);
        let d = lay(&[2, 4, 6]);
        let p = Plan::new(
            Operand::new(&a, &[9, 0, 3]),
            Operand::new(&b, &[9, 3, 1]),
            None,
            Operand::new(&d, &[9, 0, 1]),
        )
        .unwrap();
        assert_eq!(p.stats.batch, 2);
        assert_eq!((p.stats.m, p.stats.n, p.stats.k), (4, 6, 5));
    }

    #[test]
    fn isolated_index_becomes_zero_stride_contraction() {
        // D[i,j] = sum_{k,r} A[i,k,r] B[k,j]: r is isolated in A.
        let a = lay(&[4, 5, 3]);
        let b = lay(&[5, 6]);
        let d = lay(&[4, 6]);
        let p = Plan::new(
            Operand::new(&a, &[0, 2, 7]),
            Operand::new(&b, &[2, 1]),
            None,
            Operand::new(&d, &[0, 1]),
        )
        .unwrap();
        assert_eq!(p.stats.k, 15, "contraction extent is 5*3");
        // B must re-read the same column for each r.
        assert_eq!(p.b_k.len(), 15);
        assert_eq!(&p.b_k[..6], &[0, 1, 2, 3, 4, 0]);
    }

    #[test]
    fn repeated_label_takes_diagonal() {
        // D[j] = sum_i A[i,i] B[i,j] -- A's trace-like diagonal.
        let a = lay(&[4, 4]);
        let b = lay(&[4, 6]);
        let d = Layout::col_major(&[6]);
        let p = Plan::new(
            Operand::new(&a, &[0, 0]),
            Operand::new(&b, &[0, 1]),
            None,
            Operand::new(&d, &[1]),
        )
        .unwrap();
        assert_eq!(p.stats.k, 4);
        // stride 1 + stride 4 = 5
        assert_eq!(p.a_k, vec![0, 5, 10, 15]);
    }

    #[test]
    fn broadcast_output_index_is_rejected() {
        let a = lay(&[4, 5]);
        let b = lay(&[5, 6]);
        let d = lay(&[4, 6, 2]);
        let e = Plan::new(
            Operand::new(&a, &[0, 2]),
            Operand::new(&b, &[2, 1]),
            None,
            Operand::new(&d, &[0, 1, 8]),
        )
        .unwrap_err();
        assert!(matches!(e, Error::BroadcastIndexUnsupported { label: 8 }));
    }

    /// A label in `C` alone is the one input that would reach the
    /// `unreachable!` in the classification match, which reads only
    /// `(in_a, in_b, in_d)`. The `in_c != in_d` check is the whole reason that
    /// arm is dead, so it is pinned here rather than left as an argument in a
    /// comment.
    #[test]
    fn label_only_in_c_is_rejected_before_classification() {
        let a = lay(&[4, 5]);
        let b = lay(&[5, 6]);
        let d = lay(&[4, 6]);
        let c = lay(&[4, 6, 2]);
        let e = Plan::new(
            Operand::new(&a, &[0, 2]),
            Operand::new(&b, &[2, 1]),
            Some(Operand::new(&c, &[0, 1, 8])),
            Operand::new(&d, &[0, 1]),
        )
        .unwrap_err();
        assert_eq!(e, Error::OutputLabelMismatch);
    }

    /// `D[i,j] = A[i,k] B[k,j]` with `D` stored in the given strides.
    fn gemm_plan(m: i64, n: i64, d_strides: [i64; 2]) -> Plan {
        let a = lay(&[m, 7]);
        let b = lay(&[7, n]);
        let d = Layout::new(vec![m, n], d_strides.to_vec()).unwrap();
        Plan::new(
            Operand::new(&a, &[0, 2]),
            Operand::new(&b, &[2, 1]),
            None,
            Operand::new(&d, &[0, 1]),
        )
        .unwrap()
    }

    #[test]
    fn column_major_output_is_not_transposed() {
        // Rows already have the unit stride: nothing to gain.
        assert!(!gemm_plan(64, 64, [1, 64]).transposes_gemm_rule(16));
    }

    #[test]
    fn row_major_output_is_transposed() {
        // Columns have the unit stride and the run is long enough to cover MR.
        assert!(gemm_plan(64, 64, [64, 1]).transposes_gemm_rule(16));
    }

    #[test]
    fn transpose_taken_when_the_row_block_fits_exactly() {
        let p = gemm_plan(64, 16, [16, 1]);
        assert!(p.transposes_gemm_rule(16), "run == MR fits");
    }

    /// `D` with `M` rows in runs of `m_run` at stride 24 and `N` columns in one
    /// contiguous run of 24 — the shape of the whole `abcijk` family, where the
    /// two arms are mirror images and neither can fit a 48-row block.
    fn mirrored_plan(m_run: i64, outer: i64) -> Plan {
        // `outer` and `outer2` break the folds, so both directions keep short
        // runs while staying long enough overall to fill a micro-tile.
        let d = Layout::new(vec![24, m_run, 4, 8], vec![1, 24, outer, outer * 7 + 3]).unwrap();
        let a = lay(&[m_run, 4, 7]);
        let b = lay(&[7, 24, 8]);
        Plan::new(
            Operand::new(&a, &[1, 2, 4]),
            Operand::new(&b, &[4, 0, 3]),
            None,
            Operand::new(&d, &[0, 1, 2, 3]),
        )
        .unwrap()
    }

    #[test]
    fn orientation_falls_back_to_the_shorter_run_when_neither_arm_fits() {
        // At MR = 48 no arm can hold a row block inside a run: the columns run
        // 24 and the rows run 16 or 256. The tie is then broken by putting the
        // *shorter*-run direction in the row role — which is what separates the
        // `-mb` family (where swapping measured -14% in `f32`) from `e*bc`
        // (where it measured +40%), the distinction the previous rule missed.
        let short = mirrored_plan(16, 100_000);
        assert_eq!(short.d_m_run.0, 16);
        assert_eq!(short.d_n_run, (24, 1));
        assert!(
            !short.transposes_gemm_rule(48),
            "rows already have the shorter run"
        );

        let long = mirrored_plan(256, 1_000_000);
        assert_eq!(long.d_m_run.0, 256);
        assert!(
            long.transposes_gemm_rule(48),
            "columns have the shorter run"
        );

        // And step 1 still dominates: give it an `MR` the column run can hold
        // and both plans swap for that reason instead.
        assert!(short.transposes_gemm_rule(16));
        assert!(long.transposes_gemm_rule(16));
    }

    #[test]
    fn transpose_declined_when_columns_are_not_contiguous() {
        // Both directions strided: the swap cannot make the rows contiguous,
        // so the smaller stride alone does not justify it.
        assert!(!gemm_plan(64, 64, [512, 2]).transposes_gemm_rule(16));
    }

    /// `D[a,c,j] = A[a,c,k] B[k,j]` with `a` contiguous in `D` (extent 24) and
    /// `c` far away, so `D`'s rows come in runs of 24 — the shape the whole
    /// TCCG corpus has, since it rounds stride-1 extents up to multiples of 24.
    fn run24_plan() -> Plan {
        let d = Layout::new(vec![24, 4, 8], vec![1, 200, 4000]).unwrap();
        let a = lay(&[24, 4, 7]);
        let b = lay(&[7, 8]);
        Plan::new(
            Operand::new(&a, &[0, 1, 3]),
            Operand::new(&b, &[3, 2]),
            None,
            Operand::new(&d, &[0, 1, 2]),
        )
        .unwrap()
    }

    /// The row-block rule over a menu written as bare `MR`s, answering with the
    /// `MR` it chose rather than the menu position.
    ///
    /// The menu is positional since A35, so the rule returns an index — but
    /// every expectation below is about *which shape* is picked, and an index
    /// would make them say that less clearly while also going stale whenever an
    /// entry is inserted. `NR` is a placeholder here because the rule does not
    /// read it; `row_block_may_reach_an_nr_only_alternate` is the test that does.
    fn pick(p: &Plan, mrs: &[usize]) -> Option<usize> {
        let menu: Vec<(usize, usize)> = mrs.iter().map(|&mr| (mr, 6)).collect();
        p.preferred_row_block(&menu).map(|i| menu[i].0)
    }

    #[test]
    fn row_block_scores_follow_the_output_runs() {
        let p = run24_plan();
        assert_eq!(p.stats.m, 96, "two unfolded M axes");
        assert_eq!(p.row_block_score(24), 1.0);
        assert_eq!(p.row_block_score(8), 1.0);
        assert!((p.row_block_score(16) - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(p.row_block_score(48), 0.0);
    }

    #[test]
    fn row_block_picks_a_shape_that_tiles_the_run() {
        let p = run24_plan();
        // `c64` planar's menu: the default straddles a third of its blocks,
        // the first alternate none, so the rule moves. This is the case worth
        // 1.09-1.26x on the corpus.
        assert_eq!(pick(&p, &[16, 24, 8]), Some(24));
        // The default is already perfect: never trade kernel peak for nothing.
        assert_eq!(pick(&p, &[24, 16, 8]), None);
        // Ties keep the default, which is the fastest kernel.
        assert_eq!(pick(&p, &[8, 24]), None);
        assert_eq!(pick(&p, &[]), None);
    }

    /// The point of keying the menu by position (A35): two entries of the same
    /// height differing only in `NR`.
    ///
    /// `planar` `f32`/`c32` ships `32x6` where Phase 3's own sweep names `32x5`
    /// as 7.8% faster, and under the old `MR`-keyed menu that shape could not be
    /// put on the menu at all — the second entry would have been unreachable, so
    /// there was no way to A/B it at run time and the finding stayed
    /// untestable. This pins the three things that has to mean.
    #[test]
    fn row_block_may_reach_an_nr_only_alternate() {
        let p = run24_plan();
        let menu = [(24, 6), (24, 5)];
        // 1. The rule cannot tell them apart — it reads `MR` — and ties go to
        //    the earlier entry, so the measured default stays in front.
        assert_eq!(p.preferred_row_block(&menu), None);
        // 2. `idx=` reaches the second one, which is what makes it measurable.
        assert_eq!(menu.get(1), Some(&(24, 5)));
        // 3. `mr=` cannot distinguish them and resolves to the first, which is
        //    the documented limitation rather than a silent surprise.
        assert_eq!(menu.iter().position(|&(m, _)| m == 24), Some(0));
    }

    #[test]
    fn row_block_refuses_a_partial_improvement() {
        // 48 straddles everything and 16 fixes two blocks in three, but a
        // shape that does not clear the gather path outright cannot repay its
        // own cost: this is the `f32` menu, and taking it measured 0.88-0.95.
        let p = run24_plan();
        assert_eq!(pick(&p, &[48, 16]), None);
    }

    #[test]
    fn row_block_may_change_the_orientation() {
        // Until Phase 4.1d a shape change was forbidden from flipping the
        // orientation, because the orientation rule of the day picked the wrong
        // arm on one family and the shape change would hand it the decision.
        // With that rule fixed the veto only blocked wins: the six `c32` 1m
        // cases it had been suppressing measured 1.19-1.24x once it was gone.
        // So a shape is judged on its own regularity, in whatever orientation
        // it implies.
        let p = run24_plan();
        assert_eq!(pick(&p, &[16, 24, 8]), Some(24));
    }

    #[test]
    fn row_block_leaves_a_deep_contraction_alone() {
        // Same output structure, but `k` large enough that the write-back is
        // amortised: the shape change would cost and buy nothing.
        let d = Layout::new(vec![24, 4, 8], vec![1, 200, 4000]).unwrap();
        let a = lay(&[24, 4, 512]);
        let b = lay(&[512, 8]);
        let p = Plan::new(
            Operand::new(&a, &[0, 1, 3]),
            Operand::new(&b, &[3, 2]),
            None,
            Operand::new(&d, &[0, 1, 2]),
        )
        .unwrap();
        assert_eq!(p.stats.k, 512);
        assert!(
            (p.row_block_score(16) - 2.0 / 3.0).abs() < 1e-12,
            "would fire"
        );
        assert_eq!(pick(&p, &[16, 24, 8]), None);
    }

    #[test]
    fn row_block_leaves_a_fully_regular_output_alone() {
        // Column-major `D`: one run, so no shape can straddle anything.
        let p = gemm_plan(64, 64, [1, 64]);
        assert_eq!(pick(&p, &[16, 24, 8]), None);
    }

    #[test]
    fn extent_mismatch_is_rejected() {
        let a = lay(&[4, 5]);
        let b = lay(&[7, 6]);
        let d = lay(&[4, 6]);
        assert!(Plan::new(
            Operand::new(&a, &[0, 2]),
            Operand::new(&b, &[2, 1]),
            None,
            Operand::new(&d, &[0, 1]),
        )
        .is_err());
    }
}
