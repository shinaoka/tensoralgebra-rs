//! Scatter and block-scatter vectors.
//!
//! A tensor participating in a contraction is viewed as a matrix whose rows
//! and columns are mixed-radix multi-indices over a *class* of index labels
//! (see [`crate::plan`]). The row scatter vector `rscat[i]` gives the element
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

/// Sentinel stored in a block-scatter vector for a block whose scatter entries
/// are *not* an arithmetic progression.
pub const IRREGULAR: i64 = i64::MIN;

/// Build the scatter vector for a mixed-radix index group.
///
/// `extents`/`strides` are ordered fastest-varying first. The returned vector
/// has length `extents.iter().product()` (1 for an empty group, i.e. a single
/// zero offset).
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

/// Fraction of blocks in a block-scatter vector that are regular. Used for
/// diagnostics and for the planar-vs-TTGT dispatch heuristic.
pub fn regular_fraction(bs: &[i64]) -> f64 {
    if bs.is_empty() {
        return 1.0;
    }
    let reg = bs.iter().filter(|&&s| s != IRREGULAR).count();
    reg as f64 / bs.len() as f64
}

/// A matrix view of a tensor through scatter vectors, with block-scatter
/// metadata for the two axes.
#[derive(Debug)]
pub struct BlockScatterMatrix<'a> {
    pub rscat: &'a [i64],
    pub cscat: &'a [i64],
    pub rbs: &'a [i64],
    pub cbs: &'a [i64],
    pub rblk: usize,
    pub cblk: usize,
}

impl BlockScatterMatrix<'_> {
    #[inline]
    pub fn nrows(&self) -> usize {
        self.rscat.len()
    }
    #[inline]
    pub fn ncols(&self) -> usize {
        self.cscat.len()
    }

    /// Row stride of the block containing row `i0`, where `i0` must be a
    /// multiple of `rblk`. `None` if the block is irregular.
    #[inline]
    pub fn row_stride_at(&self, i0: usize) -> Option<i64> {
        debug_assert_eq!(i0 % self.rblk, 0);
        match self.rbs[i0 / self.rblk] {
            IRREGULAR => None,
            s => Some(s),
        }
    }

    #[inline]
    pub fn col_stride_at(&self, j0: usize) -> Option<i64> {
        debug_assert_eq!(j0 % self.cblk, 0);
        match self.cbs[j0 / self.cblk] {
            IRREGULAR => None,
            s => Some(s),
        }
    }
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
    fn ragged_tail_block() {
        let s: Vec<i64> = (0..5).collect();
        let bs = build_block_scatter(&s, 2);
        assert_eq!(bs, vec![1, 1, 0]); // last block has a single entry
    }
}
