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
use crate::scatter::build_scatter;

/// Index class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Axis {
    pub extent: i64,
    pub sa: i64,
    pub sb: i64,
    pub sc: i64,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ElementOp {
    #[default]
    Identity,
    Conjugate,
}

impl ElementOp {
    #[inline]
    pub fn is_conj(self) -> bool {
        matches!(self, ElementOp::Conjugate)
    }
}

/// An operand description: layout plus index labels plus element-wise op.
#[derive(Clone, Copy, Debug)]
pub struct Operand<'a> {
    pub layout: &'a Layout,
    pub idx: &'a [i64],
    pub op: ElementOp,
}

impl<'a> Operand<'a> {
    pub fn new(layout: &'a Layout, idx: &'a [i64]) -> Self {
        Operand {
            layout,
            idx,
            op: ElementOp::Identity,
        }
    }
    pub fn conj(mut self) -> Self {
        self.op = ElementOp::Conjugate;
        self
    }
}

/// Diagnostics about a plan, useful for benchmarking write-ups and for
/// dispatch heuristics.
#[derive(Clone, Debug, Default)]
pub struct PlanStats {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub m_axes: Vec<Axis>,
    pub n_axes: Vec<Axis>,
    pub k_axes: Vec<Axis>,
    pub h_axes: Vec<Axis>,
    /// True when every class folded down to at most one axis, i.e. the
    /// contraction is exactly a (batched) GEMM on strided matrices.
    pub is_pure_gemm: bool,
}

impl PlanStats {
    /// Multiply-accumulate count (batch * m * n * k).
    pub fn macs(&self) -> u64 {
        (self.batch as u64) * (self.m as u64) * (self.n as u64) * (self.k as u64)
    }
}

/// Borrowed view of a plan's scatter vectors. Offsets are in elements.
#[derive(Clone, Copy, Debug)]
pub struct Scatters<'a> {
    /// Offsets of A's rows (free indices of A).
    pub a_m: &'a [i64],
    /// Offsets of A's columns (contracted indices).
    pub a_k: &'a [i64],
    pub b_k: &'a [i64],
    pub b_n: &'a [i64],
    pub c_m: &'a [i64],
    pub c_n: &'a [i64],
    pub d_m: &'a [i64],
    pub d_n: &'a [i64],
    /// Per-batch base offsets for the Hadamard indices.
    pub h_a: &'a [i64],
    pub h_b: &'a [i64],
    pub h_c: &'a [i64],
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
    pub(crate) conj_a: bool,
    pub(crate) conj_b: bool,
    pub(crate) conj_c: bool,
    pub(crate) conj_d: bool,
    /// Overrides the element-type-derived cache blocking when set.
    pub(crate) blocking: Option<crate::kernel::Blocking>,
    /// Overrides the default complex method when set.
    pub(crate) method: Option<crate::kernel::ComplexMethod>,
    pub stats: PlanStats,
}

impl Plan {
    /// Analyse a contraction. `c` may be `None`, in which case `beta` is
    /// ignored at execution time and `D` is overwritten.
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
            conj_a: a.op.is_conj(),
            conj_b: b.op.is_conj(),
            conj_c: c.map(|c| c.op.is_conj()).unwrap_or(false),
            conj_d: d.op.is_conj(),
            blocking: None,
            method: None,
            stats,
        })
    }

    /// Choose how complex arithmetic is induced from real micro-kernels.
    ///
    /// Ignored for real element types. Without this, the plan uses
    /// [`ComplexMethod::from_env`], i.e. `TENSORCONTRACT_COMPLEX` if set and
    /// [`ComplexMethod::Planar`] otherwise.
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
    pub fn with_complex_method(mut self, method: crate::kernel::ComplexMethod) -> Self {
        self.method = Some(method);
        self
    }

    /// The complex method this plan will execute with.
    pub fn complex_method(&self) -> crate::kernel::ComplexMethod {
        self.method
            .unwrap_or_else(crate::kernel::ComplexMethod::from_env)
    }

    /// Override the cache blocking parameters this plan executes with.
    ///
    /// Mainly for parameter sweeps and for driving every level of the loop
    /// nest on small test problems; the defaults are derived from the element
    /// type and the target's cache sizes.
    pub fn with_blocking(mut self, blk: crate::kernel::Blocking) -> Self {
        self.blocking = Some(blk);
        self
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
    /// Both conditions are needed, and the second one is the interesting one:
    ///
    /// 1. `D`'s column direction must be *strictly* more contiguous than its
    ///    row direction — otherwise there is nothing to gain.
    /// 2. The swap must leave the row block **unbroken**: the new leading axis
    ///    must have unit stride and an extent of at least `MR`.
    ///
    /// Condition 2 is measured, not assumed. On the `abcijk-*m{b}-*` family the
    /// swap makes the leading axis unit-stride but only 16 long, and the result
    /// is monotone in `run / MR`: +18% at 16/16 (`c64`), +23% at 16/24 (`f64`),
    /// 0% at 16/32 (`c32`), **−17%** at 16/48 (`f32`). A shattered row block
    /// keeps the swap's costs and loses its benefit, so the rule declines it.
    /// Note this makes the choice element-type dependent through `MR`, which is
    /// why it is not a property of the plan alone.
    ///
    /// `TENSORCONTRACT_ORIENT=none` disables the swap and `=swap` forces it;
    /// both exist to A/B the decision, and neither affects correctness.
    pub fn transposes_gemm(&self, mr: usize) -> bool {
        if let Some(forced) = orient_override() {
            return forced;
        }
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
    /// Returns `None` for "use the menu's default", which is also what an
    /// unrecognised or absent `TENSORCONTRACT_ROWBLOCK` gives.
    ///
    /// # Why `MR` is not just a kernel-tuning constant
    ///
    /// `MR` is the granularity at which the *output's* row scatter is blocked,
    /// so it decides which of [`crate::writeback`]'s three paths each block
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
    /// # The rule, and the three guards it needs
    ///
    /// Take the first shape on the menu that makes *every* output row block a
    /// single run, but only when all three of these hold. Each guard is there
    /// because the grid in `bench-results/phase4c` measured what happens
    /// without it; none is a plausibility argument.
    ///
    /// 1. **The contraction is shallow** (`k <= 32`). The write-back costs a
    ///    constant per output element against `~4k` flops of kernel work, so
    ///    the path it takes only matters while `k` is small — and a shape off
    ///    the kernel's peak always costs something. Measured: at `k = 24` the
    ///    winning shape gains 1.09–1.26x; at `k >= 204` the same change is
    ///    1.01–1.04x, i.e. nothing, and it is still being paid for. The corpus
    ///    jumps from `k = 24` to `k = 52`, so it resolves this boundary only to
    ///    somewhere in `(24, 52]`.
    /// 2. **The default is substantially broken** (`wb <= 0.75`). A shape
    ///    change is not free, so it cannot be repaid by a marginal improvement.
    ///    Measured: taking `f32` `48x8 -> 32x8` where the default was already
    ///    0.88 regular lost 7–9%. The corpus only takes the values 0, 0.67,
    ///    0.88 and 1.0, so any threshold in `(0.67, 0.88]` fits it equally.
    /// 3. **The orientation does not change.** `MR` is an input to
    ///    [`Plan::transposes_gemm`], so a shape change can silently flip the
    ///    row/column orientation — which is worth far more than the write-back
    ///    path is (1.4x against 1.15x) and whose rule is known to be wrong on
    ///    nine corpus cases (A14). A shape change must not re-decide it as a
    ///    side effect. Measured: without this guard, `c32` 3m moves the three
    ///    `abcijk-e*bc-*` cases from the good arm to the bad one and loses 19%.
    ///
    /// So guarded, the rule fires on 20 of 392 corpus case-dtype-methods, gains
    /// 1.07–1.26x on 19 of them, and loses 7% on one small case — inside the
    /// per-case noise floor. Unguarded it is a **loss**: maximising the
    /// fraction alone scores 0.936 in `f32`.
    ///
    /// # What this deliberately leaves on the table
    ///
    /// An oracle picking the fastest shape per case with hindsight scores
    /// 1.03–1.07 across every dtype and method, so most of the available gain
    /// is *not* reachable from the output's stride pattern. The largest single
    /// piece of it is the orientation: on the six `abcijk-e*bc-*` cases in
    /// `f32`/`c32`, shrinking `MR` to 16 flips them to `BA` and gains
    /// 1.24–1.39x **despite** paying ~30% in kernel shape — which says the
    /// orientation there is worth about 2x and should be bought directly, at
    /// the default `MR`, rather than through a shape change. See the Phase 4.1c
    /// report.
    pub fn row_block(&self, menu: &[usize]) -> Option<usize> {
        match row_block_override() {
            RowBlock::Base => None,
            RowBlock::Auto => self.preferred_row_block(menu),
            RowBlock::Pin(mr) => menu.contains(&mr).then_some(mr),
            RowBlock::Index(i) => menu.get(i).copied(),
        }
    }

    /// [`Plan::row_block`]'s rule with no environment override, so that it can
    /// be scored offline against measured ground truth.
    pub fn preferred_row_block(&self, menu: &[usize]) -> Option<usize> {
        /// Above this contraction depth the write-back is amortised and the
        /// shape's own cost is all that is left. See [`Plan::row_block`].
        const SHALLOW_K: usize = 32;
        /// A default this regular already is not worth paying a shape change
        /// to improve.
        const BROKEN_ENOUGH: f64 = 0.75;

        let (&default, rest) = menu.split_first()?;
        if self.stats.k > SHALLOW_K || self.row_block_score(default) > BROKEN_ENOUGH {
            return None;
        }
        let orient = self.transposes_gemm(default);
        rest.iter().copied().find(|&mr| {
            self.transposes_gemm(mr) == orient && self.row_block_score(mr) >= 1.0 - 1e-9
        })
    }

    /// Fraction of the output's row blocks that would stay off
    /// [`crate::writeback`]'s gather path at row block `mr`, evaluated in the
    /// orientation `mr` itself selects.
    ///
    /// `0.0` when the output's rows have no uniform run structure — then `MR`
    /// has no predictable effect and every shape scores alike, which leaves the
    /// tie-break to keep the default.
    pub fn row_block_score(&self, mr: usize) -> f64 {
        let rows = if self.transposes_gemm(mr) {
            &self.d_n
        } else {
            &self.d_m
        };
        match run_structure(rows) {
            Some(run) => unbroken_fraction(rows.len(), run, mr),
            None => 0.0,
        }
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

/// `TENSORCONTRACT_ORIENT=none|swap` pins the row/column orientation instead of
/// deriving it from `D`'s strides. Read once per process; for measurement only.
fn orient_override() -> Option<bool> {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static ENV: OnceLock<Option<bool>> = OnceLock::new();
        *ENV.get_or_init(|| match std::env::var("TENSORCONTRACT_ORIENT").ok()?.as_str() {
            "none" | "ab" => Some(false),
            "swap" | "ba" => Some(true),
            _ => None,
        })
    }
    #[cfg(not(feature = "std"))]
    {
        None
    }
}

/// What `TENSORCONTRACT_ROWBLOCK` asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowBlock {
    /// The kernel set's default shape — the Phase 3 choice.
    Base,
    /// [`Plan::row_block`]'s rule.
    Auto,
    /// A specific logical `MR`, where the kernel set has one.
    Pin(usize),
    /// A position on the menu, which is comparable across element types and
    /// methods in a way a bare `MR` is not — `mr=16` names different shapes in
    /// `f32` and `f64`, `idx=1` names "the first alternate" in both.
    Index(usize),
}

/// `TENSORCONTRACT_ROWBLOCK=base|auto|mr=<n>|idx=<i>` selects the micro-tile
/// row block. Read once per process.
///
/// The default is `auto`; `base` pins the Phase 3 shape, which is what the rule
/// was measured against. Both arms stay reachable at run time so the comparison
/// can be repeated in one session rather than as a diff between two builds
/// (A15), and `idx=<i>` re-runs the whole grid the rule was derived from.
fn row_block_override() -> RowBlock {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static ENV: OnceLock<RowBlock> = OnceLock::new();
        *ENV.get_or_init(|| {
            let Ok(v) = std::env::var("TENSORCONTRACT_ROWBLOCK") else {
                return RowBlock::Auto;
            };
            let v = v.trim().to_ascii_lowercase();
            match v.as_str() {
                "auto" => RowBlock::Auto,
                "base" | "default" => RowBlock::Base,
                _ => {
                    let parse = |p: &str| v.strip_prefix(p)?.parse::<usize>().ok();
                    if let Some(n) = parse("mr=") {
                        RowBlock::Pin(n)
                    } else if let Some(i) = parse("idx=") {
                        RowBlock::Index(i)
                    } else {
                        RowBlock::Base
                    }
                }
            }
        })
    }
    #[cfg(not(feature = "std"))]
    {
        RowBlock::Auto
    }
}

/// The run structure of a scatter vector: `Some((len, stride))` when it is a
/// concatenation of equal-length maximal arithmetic runs, `None` otherwise.
///
/// This is the shape of every output scatter the corpus produces — a leading
/// axis of extent `len` and stride `stride`, restarted by the outer axes — and
/// it is what makes the effect of a candidate `MR` computable in `O(M / MR)`
/// rather than by rebuilding a block scatter per candidate.
fn run_structure(scat: &[i64]) -> Option<(usize, i64)> {
    if scat.len() < 2 {
        return None;
    }
    let stride = scat[1] - scat[0];
    // First discontinuity ends the first run; every later run must match it.
    let len = scat
        .windows(2)
        .position(|w| w[1] - w[0] != stride)
        .map_or(scat.len(), |p| p + 1);
    if len < 2 || !scat.len().is_multiple_of(len) {
        return None;
    }
    for (i, w) in scat.windows(2).enumerate() {
        let want = if (i + 1) % len == 0 { None } else { Some(stride) };
        if want.is_some_and(|s| w[1] - w[0] != s) {
            return None;
        }
        // A run boundary that happens to continue the progression would mean
        // the runs are longer than measured, contradicting maximality.
        if want.is_none() && w[1] - w[0] == stride {
            return None;
        }
    }
    Some((len, stride))
}

/// Fraction of aligned `mr`-blocks of a `total`-row scatter that fall inside a
/// single run, given runs of `(len, _)` rows.
///
/// Exactly the fraction that reaches [`crate::writeback`]'s non-gather paths.
fn unbroken_fraction(total: usize, (len, _): (usize, i64), mr: usize) -> f64 {
    if total == 0 || mr == 0 {
        return 1.0;
    }
    let nblk = total.div_ceil(mr);
    let whole = (0..nblk)
        .filter(|b| {
            let lo = b * mr;
            let hi = (lo + mr).min(total) - 1;
            lo / len == hi / len
        })
        .count();
    whole as f64 / nblk as f64
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
        let e = layout.extents[k];
        let s = layout.strides[k];
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
        assert!(!gemm_plan(64, 64, [1, 64]).transposes_gemm(16));
    }

    #[test]
    fn row_major_output_is_transposed() {
        // Columns have the unit stride and the run is long enough to cover MR.
        assert!(gemm_plan(64, 64, [64, 1]).transposes_gemm(16));
    }

    #[test]
    fn transpose_declined_when_it_would_shatter_the_row_block() {
        // Same layout, but the contiguous run is shorter than MR, so the
        // swapped micro-tile rows would still straddle a discontinuity. That
        // case measured *slower* than not swapping; see `transposes_gemm`.
        let p = gemm_plan(64, 16, [16, 1]);
        assert!(p.transposes_gemm(16), "run == MR is taken");
        assert!(!p.transposes_gemm(32), "run < MR is declined");
    }

    #[test]
    fn transpose_declined_when_columns_are_not_contiguous() {
        // Both directions strided: the swap cannot make the rows contiguous,
        // so the smaller stride alone does not justify it.
        assert!(!gemm_plan(64, 64, [512, 2]).transposes_gemm(16));
    }

    #[test]
    fn run_structure_recognises_equal_length_runs() {
        // Four contiguous runs of 24, restarted by an outer axis.
        let s = build_scatter(&[24, 4], &[1, 200]);
        assert_eq!(run_structure(&s), Some((24, 1)));
        // A single unbroken run is the whole vector.
        assert_eq!(run_structure(&build_scatter(&[24], &[1])), Some((24, 1)));
        // Constant non-unit stride is still a run: `MR` cannot break it.
        assert_eq!(run_structure(&build_scatter(&[12], &[4])), Some((12, 4)));
        // Unequal runs have no uniform structure.
        assert_eq!(run_structure(&[0, 1, 2, 100, 101, 200]), None);
        assert_eq!(run_structure(&[7]), None);
    }

    #[test]
    fn unbroken_fraction_counts_straddling_blocks() {
        // 96 rows in runs of 24. Only shapes that tile a run keep every block.
        let total = 96;
        let run = (24, 1);
        assert_eq!(unbroken_fraction(total, run, 24), 1.0);
        assert_eq!(unbroken_fraction(total, run, 8), 1.0);
        // 16 into 24 straddles every second block of a 48-row period.
        assert!((unbroken_fraction(total, run, 16) - 2.0 / 3.0).abs() < 1e-12);
        // 32 and 48 cross a boundary every time.
        assert_eq!(unbroken_fraction(total, run, 32), 0.0);
        assert_eq!(unbroken_fraction(total, run, 48), 0.0);
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
        assert_eq!(p.preferred_row_block(&[16, 24, 8]), Some(24));
        // The default is already perfect: never trade kernel peak for nothing.
        assert_eq!(p.preferred_row_block(&[24, 16, 8]), None);
        // Ties keep the default, which is the fastest kernel.
        assert_eq!(p.preferred_row_block(&[8, 24]), None);
        assert_eq!(p.preferred_row_block(&[]), None);
    }

    #[test]
    fn row_block_refuses_a_partial_improvement() {
        // 48 straddles everything and 16 fixes two blocks in three, but a
        // shape that does not clear the gather path outright cannot repay its
        // own cost: this is the `f32` menu, and taking it measured 0.88-0.95.
        let p = run24_plan();
        assert_eq!(p.preferred_row_block(&[48, 16]), None);
    }

    #[test]
    fn row_block_refuses_to_change_the_orientation() {
        // The alternate would be perfectly regular, but only because it flips
        // the GEMM orientation — a decision worth more than this one and made
        // by a rule known to be wrong on nine corpus cases. Declining it is
        // what keeps `c32` 3m off a 19% regression.
        let d = Layout::new(vec![24, 4, 32], vec![1, 200, 4000]).unwrap();
        let a = lay(&[24, 4, 7]);
        let b = lay(&[7, 32]);
        let p = Plan::new(
            Operand::new(&a, &[0, 1, 3]),
            Operand::new(&b, &[3, 2]),
            None,
            Operand::new(&d, &[0, 1, 2]),
        )
        .unwrap();
        for &mr in &[16usize, 24, 8] {
            assert!(!p.transposes_gemm(mr), "this fixture must not swap");
        }
        // Same fixture, but with `D` stored so that the column direction is the
        // contiguous one: now the shapes disagree about the orientation.
        let d = Layout::new(vec![4, 32, 24], vec![200, 4000, 1]).unwrap();
        let q = Plan::new(
            Operand::new(&a, &[0, 1, 3]),
            Operand::new(&b, &[3, 2]),
            None,
            Operand::new(&d, &[1, 2, 0]),
        )
        .unwrap();
        if q.transposes_gemm(8) != q.transposes_gemm(24) {
            assert_eq!(q.preferred_row_block(&[24, 8]), None);
        }
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
        assert!((p.row_block_score(16) - 2.0 / 3.0).abs() < 1e-12, "would fire");
        assert_eq!(p.preferred_row_block(&[16, 24, 8]), None);
    }

    #[test]
    fn row_block_leaves_a_fully_regular_output_alone() {
        // Column-major `D`: one run, so no shape can straddle anything.
        let p = gemm_plan(64, 64, [1, 64]);
        assert_eq!(p.preferred_row_block(&[16, 24, 8]), None);
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
