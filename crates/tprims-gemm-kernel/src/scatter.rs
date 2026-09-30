//! Scatter and block-scatter vectors.
//!
//! A tensor participating in a contraction is viewed as a matrix whose rows
//! and columns are mixed-radix multi-indices over a *class* of index labels
//! (see `tensorcontract::plan`). The row scatter vector `rscat[i]` gives the element
//! offset of matrix row `i` relative to the tensor base; likewise `cscat[j]`
//! for columns. Element `(i, j)` then lives at `base + rscat[i] + cscat[j]`.
//!
//! The *block* scatter vector records, for each aligned block of `blk`
//! consecutive scatter entries, whether those entries form an arithmetic
//! progression, and with what stride. When they do, the corresponding slice of
//! the matrix can be traversed with ordinary strided (often unit-stride)
//! accesses — the "regular" fast path. When they do not, the generic gather
//! path is used. This is the block-scatter-matrix layout of Matthews
//! (arXiv:1607.00291).
//!
//! Note that a *zero* block stride is meaningful and legal here: it arises for
//! contraction indices that are absent from one operand (TAPP "isolated"
//! indices, i.e. reductions), where the same element is read repeatedly.
//! Irregular blocks are therefore flagged with a dedicated sentinel rather
//! than with zero as in some other implementations.
//!
//! Everything public here is a pure function of `i64` slices with no coupling
//! to the rest of the engine, which is why it is API rather than an internal
//! detail: it is the only way for a caller — the project's own benchmark
//! harness included — to describe the traversal the engine is about to make,
//! and to report regularity at the same granularity the engine sees. Combine
//! with `tensorcontract::Plan::oriented_scatters` and
//! `tensorcontract::kernel::plan_config`, which give the vectors and the block sizes
//! actually used.
//!
//! # Forcing the gather path
//!
//! Both packing and the output write-back take the regular path when the block
//! scatter permits it. Setting `TENSORCONTRACT_WRITEBACK=gather` makes the
//! write-back use the general scatter loop unconditionally, disabling the
//! block-scatter row addressing and the `alpha = 1, beta = 0` copy along with
//! it. Results are unaffected — this exists so the fast path is an A/B switch at
//! run time rather than a rebuild, in the same spirit as
//! `TENSORCONTRACT_KERNEL=scalar` (see `tensorcontract::kernel`). Read once per
//! process.

/// Sentinel stored in a block-scatter vector for a block whose scatter entries
/// are *not* an arithmetic progression.
pub const IRREGULAR: i64 = i64::MIN;

/// Build the scatter vector for a mixed-radix index group.
///
/// `extents`/`strides` are ordered fastest-varying first. The returned vector
/// has length `extents.iter().product()` (1 for an empty group, i.e. a single
/// zero offset).
///
/// # Panics
///
/// Panics if `extents` and `strides` have different lengths — but only as a
/// `debug_assert_eq!`, because inside the crate both always come from one
/// `tensorcontract::Layout`. In a release build a *short* `strides` still panics, on
/// the index rather than the assertion, while a long one has its tail silently
/// ignored; a caller assembling the two slices separately should check the
/// lengths itself.
///
/// ```
/// use tprims_gemm_kernel::scatter::build_scatter;
///
/// // Two modes: extent 2 stride 1 (fastest), extent 3 stride 10.
/// // Entry n is the element offset of the n'th index tuple.
/// assert_eq!(build_scatter(&[2, 3], &[1, 10]), vec![0, 1, 10, 11, 20, 21]);
///
/// // An empty group is one element at offset zero, not zero elements.
/// assert_eq!(build_scatter(&[], &[]), vec![0]);
/// ```
pub fn build_scatter(extents: &[i64], strides: &[i64]) -> Vec<i64> {
    debug_assert_eq!(extents.len(), strides.len());
    let total: i64 = extents.iter().product();
    let total = total.max(0) as usize;
    let mut out = Vec::with_capacity(total);
    if total == 0 {
        return out;
    }
    if extents.is_empty() {
        out.push(0);
        return out;
    }

    // Odometer over the mixed-radix multi-index.
    let n = extents.len();
    let mut counter = vec![0i64; n];
    let mut offset = 0i64;
    for _ in 0..total {
        out.push(offset);
        for d in 0..n {
            counter[d] += 1;
            offset += strides[d];
            if counter[d] < extents[d] {
                break;
            }
            offset -= strides[d] * extents[d];
            counter[d] = 0;
        }
    }
    debug_assert_eq!(out.len(), total);
    out
}

/// Derive the block-scatter vector for `scat` at block size `blk`.
///
/// Entry `b` covers `scat[b*blk .. min((b+1)*blk, len)]` and is the common
/// difference of that run, or [`IRREGULAR`]. Runs of length 0 or 1 are
/// trivially regular and report a stride of 0.
///
/// This is where the register block earns or loses the fast path: the *same*
/// scatter vector is fully regular at one block size and fully irregular at
/// another, which is why `MR` is a performance decision and not just a tile
/// shape.
///
/// # Panics
///
/// Panics if `blk` is 0. There is no block structure to describe at a block
/// size of zero, and the block count below would divide by it.
///
/// ```
/// use tprims_gemm_kernel::scatter::{build_block_scatter, build_scatter, IRREGULAR};
///
/// let scat = build_scatter(&[2, 3], &[1, 10]);   // [0, 1, 10, 11, 20, 21]
///
/// // At blk = 2 every block sits inside one run of stride 1: all strided loads.
/// assert_eq!(build_block_scatter(&scat, 2), vec![1, 1, 1]);
///
/// // At blk = 3 every block straddles a run boundary: all gathers.
/// assert_eq!(build_block_scatter(&scat, 3), vec![IRREGULAR, IRREGULAR]);
/// ```
pub fn build_block_scatter(scat: &[i64], blk: usize) -> Vec<i64> {
    assert!(blk > 0);
    let nblk = scat.len().div_ceil(blk);
    let mut out = Vec::with_capacity(nblk);
    for b in 0..nblk {
        let lo = b * blk;
        let hi = (lo + blk).min(scat.len());
        out.push(run_stride(&scat[lo..hi]));
    }
    out
}

/// Common difference of a run, or [`IRREGULAR`].
#[inline]
fn run_stride(run: &[i64]) -> i64 {
    if run.len() <= 1 {
        return 0;
    }
    let s = run[1] - run[0];
    for w in run.windows(2) {
        if w[1] - w[0] != s {
            return IRREGULAR;
        }
    }
    s
}

/// The run structure of a scatter vector: `Some((len, stride))` when it is a
/// concatenation of equal-length *maximal* arithmetic runs, `None` otherwise.
///
/// This is the shape every output scatter in the corpus has — a leading axis of
/// extent `len` and stride `stride`, restarted by the outer axes — and it is
/// what makes the effect of a candidate block size computable in `O(len)`
/// rather than by rebuilding a block scatter per candidate. It is also the
/// quantity both the orientation rule and the row-block rule turn on, so it
/// lives here rather than in either of them.
///
/// ```
/// use tprims_gemm_kernel::scatter::{build_scatter, run_structure};
///
/// // Three maximal runs of length 2, each of stride 1.
/// let scat = build_scatter(&[2, 3], &[1, 10]);
/// assert_eq!(run_structure(&scat), Some((2, 1)));
///
/// // One contiguous axis is a single run spanning everything.
/// assert_eq!(run_structure(&build_scatter(&[6], &[1])), Some((6, 1)));
/// ```
pub fn run_structure(scat: &[i64]) -> Option<(usize, i64)> {
    if scat.len() < 2 {
        return None;
    }
    let stride = scat[1] - scat[0];
    // The first discontinuity ends the first run; every later run must match.
    let len = scat
        .windows(2)
        .position(|w| w[1] - w[0] != stride)
        .map_or(scat.len(), |p| p + 1);
    if len < 2 || !scat.len().is_multiple_of(len) {
        return None;
    }
    for (i, w) in scat.windows(2).enumerate() {
        let at_boundary = (i + 1) % len == 0;
        if !at_boundary && w[1] - w[0] != stride {
            return None;
        }
        // A boundary that happens to continue the progression would mean the
        // runs are longer than measured, contradicting maximality.
        if at_boundary && w[1] - w[0] == stride {
            return None;
        }
    }
    Some((len, stride))
}

/// Fraction of aligned `blk`-blocks of a `total`-entry scatter that fall inside
/// a single run, given runs of `len` entries.
///
/// Exactly the fraction that reaches a strided rather than a gather traversal.
///
/// ```
/// use tprims_gemm_kernel::scatter::unbroken_fraction;
///
/// // Six entries in runs of 2. Blocks of 2 align with the runs exactly.
/// assert_eq!(unbroken_fraction(6, 2, 2), 1.0);
/// // Blocks of 4 cannot: the first straddles a boundary, the second does not.
/// assert_eq!(unbroken_fraction(6, 2, 4), 0.5);
/// ```
pub fn unbroken_fraction(total: usize, len: usize, blk: usize) -> f64 {
    if total == 0 || blk == 0 || len == 0 {
        return 1.0;
    }
    let nblk = total.div_ceil(blk);
    let whole = (0..nblk)
        .filter(|b| {
            let lo = b * blk;
            let hi = (lo + blk).min(total) - 1;
            lo / len == hi / len
        })
        .count();
    whole as f64 / nblk as f64
}

/// Fraction of blocks in a block-scatter vector that are regular. Used for
/// diagnostics and for the planar-vs-TTGT dispatch heuristic.
///
/// ```
/// use tprims_gemm_kernel::scatter::{regular_fraction, IRREGULAR};
///
/// assert_eq!(regular_fraction(&[1, 1, IRREGULAR, 1]), 0.75);
/// // A zero block stride is regular: it is how a reduction's repeated read
/// // appears, and only IRREGULAR means the gather path.
/// assert_eq!(regular_fraction(&[0, 0]), 1.0);
/// ```
pub fn regular_fraction(bs: &[i64]) -> f64 {
    if bs.is_empty() {
        return 1.0;
    }
    let reg = bs.iter().filter(|&&s| s != IRREGULAR).count();
    reg as f64 / bs.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_column_major() {
        // extents (2,3) strides (1,2), fastest first -> 0,1,2,3,4,5
        let s = build_scatter(&[2, 3], &[1, 2]);
        assert_eq!(s, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn scatter_permuted() {
        // A 2x3 tensor stored row-major, viewed with the *second* index
        // fastest: extents (3,2) strides (1,3).
        let s = build_scatter(&[3, 2], &[1, 3]);
        assert_eq!(s, vec![0, 1, 2, 3, 4, 5]);
        // and with the first index fastest: extents (2,3) strides (3,1)
        let s = build_scatter(&[2, 3], &[3, 1]);
        assert_eq!(s, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn scatter_empty_group_is_single_zero() {
        assert_eq!(build_scatter(&[], &[]), vec![0]);
    }

    #[test]
    fn block_scatter_regularity() {
        let s = build_scatter(&[2, 3], &[3, 1]); // 0,3,1,4,2,5
        let bs = build_block_scatter(&s, 2);
        assert_eq!(bs, vec![3, 3, 3]);
        let bs = build_block_scatter(&s, 3);
        assert_eq!(bs, vec![IRREGULAR, IRREGULAR]);
        assert_eq!(regular_fraction(&bs), 0.0);
    }

    #[test]
    fn zero_stride_is_regular_not_irregular() {
        let s = build_scatter(&[4], &[0]);
        assert_eq!(s, vec![0, 0, 0, 0]);
        let bs = build_block_scatter(&s, 2);
        assert_eq!(bs, vec![0, 0]);
        assert_eq!(regular_fraction(&bs), 1.0);
    }

    #[test]
    fn run_structure_recognises_equal_length_runs() {
        // Four contiguous runs of 24, restarted by an outer axis.
        assert_eq!(
            run_structure(&build_scatter(&[24, 4], &[1, 200])),
            Some((24, 1))
        );
        // A single unbroken run is the whole vector.
        assert_eq!(run_structure(&build_scatter(&[24], &[1])), Some((24, 1)));
        // Constant non-unit stride is still one run: no block size breaks it.
        assert_eq!(run_structure(&build_scatter(&[12], &[4])), Some((12, 4)));
        // Unequal runs have no uniform structure.
        assert_eq!(run_structure(&[0, 1, 2, 100, 101, 200]), None);
        assert_eq!(run_structure(&[7]), None);
    }

    #[test]
    fn unbroken_fraction_counts_straddling_blocks() {
        // 96 entries in runs of 24. Only block sizes that tile a run survive.
        assert_eq!(unbroken_fraction(96, 24, 24), 1.0);
        assert_eq!(unbroken_fraction(96, 24, 8), 1.0);
        // 16 into 24 straddles every second block of a 48-entry period.
        assert!((unbroken_fraction(96, 24, 16) - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(unbroken_fraction(96, 24, 32), 0.0);
        assert_eq!(unbroken_fraction(96, 24, 48), 0.0);
    }

    #[test]
    fn ragged_tail_block() {
        let s: Vec<i64> = (0..5).collect();
        let bs = build_block_scatter(&s, 2);
        assert_eq!(bs, vec![1, 1, 0]); // last block has a single entry
    }
}
