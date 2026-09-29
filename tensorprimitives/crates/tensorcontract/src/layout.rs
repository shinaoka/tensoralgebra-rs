//! Tensor layouts: extents plus general (possibly negative, possibly zero)
//! strides, measured in *elements*.
//!
//! This mirrors TAPP's `TAPP_tensor_info` and C++23 `layout_stride`: a tensor
//! is a base pointer plus `offset(i) = sum_k i_k * stride_k`.

use core::fmt;

use crate::error::{Error, Result};

/// Shape and strides of a dense strided tensor. Strides are in elements, not
/// bytes, and may be negative or zero.
///
/// Extents and strides live in **one** buffer of length `2 * ndim`, extents
/// first. That is not a micro-optimisation, it is what makes "as many strides as
/// extents" unrepresentable rather than merely checked: there is no pair of
/// lengths that could disagree. Two public `Vec` fields used to allow
/// `Layout { extents: vec![2, 3], strides: vec![1] }`, which sailed past
/// [`Layout::new`]'s only validation and reached an index panic during planning.
/// Reach the two halves with [`extents`][Self::extents] and
/// [`strides`][Self::strides].
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Layout {
    /// Extents in `[0..ndim)`, strides in `[ndim..2 * ndim)`. The length is
    /// always even, and every constructor below preserves that.
    dims: Vec<i64>,
}

impl Layout {
    /// A layout from explicit extents and strides.
    ///
    /// This is the one place where two independently-sized sequences meet, and
    /// so the one place that can fail. Nothing else is checkable without
    /// knowing the allocation; the remaining conditions are checked when a plan
    /// is built or run.
    ///
    /// # Errors
    ///
    /// [`Error::RankMismatch`] if `extents` and `strides` differ in length.
    ///
    /// ```
    /// use tensorcontract::Layout;
    ///
    /// // A reversed axis: stride -1 walks backwards from the base pointer.
    /// let reversed = Layout::new(vec![4], vec![-1])?;
    /// assert_eq!(reversed.ndim(), 1);
    ///
    /// // A broadcast axis: stride 0 reads one element repeatedly. Legal, and
    /// // how a reduction is expressed internally.
    /// let broadcast = Layout::new(vec![2, 3], vec![1, 0])?;
    /// assert_eq!(broadcast.len(), 6);            // six index tuples...
    /// assert_eq!(broadcast.storage_len(), 2);    // ...over two elements
    ///
    /// // Mismatched lengths are the one thing rejected here.
    /// assert!(Layout::new(vec![2, 3], vec![1]).is_err());
    /// # Ok::<(), tensorcontract::Error>(())
    /// ```
    pub fn new(extents: Vec<i64>, strides: Vec<i64>) -> Result<Self> {
        if extents.len() != strides.len() {
            return Err(Error::RankMismatch {
                extents: extents.len(),
                strides: strides.len(),
            });
        }
        // Reuse `extents`' allocation rather than building a third buffer.
        let mut dims = extents;
        dims.extend_from_slice(&strides);
        Ok(Layout { dims })
    }

    /// Column-major ("Fortran order", first index fastest) dense layout.
    ///
    /// ```
    /// use tensorcontract::Layout;
    ///
    /// assert_eq!(Layout::col_major(&[2, 3, 4]).strides(), &[1, 2, 6]);
    /// ```
    pub fn col_major(extents: &[i64]) -> Self {
        let mut dims = Vec::with_capacity(2 * extents.len());
        dims.extend_from_slice(extents);
        let mut acc = 1i64;
        for &e in extents {
            dims.push(acc);
            acc *= e;
        }
        Layout { dims }
    }

    /// Row-major ("C order", last index fastest) dense layout.
    ///
    /// Nothing in the engine prefers one order to the other: both are just
    /// strides, and mixing them across operands costs nothing — which is what
    /// "transpose-free" means in practice.
    ///
    /// ```
    /// use tensorcontract::Layout;
    ///
    /// assert_eq!(Layout::row_major(&[2, 3, 4]).strides(), &[12, 4, 1]);
    /// ```
    pub fn row_major(extents: &[i64]) -> Self {
        let n = extents.len();
        let mut dims = Vec::with_capacity(2 * n);
        dims.extend_from_slice(extents);
        dims.resize(2 * n, 0);
        let mut acc = 1i64;
        for k in (0..n).rev() {
            dims[n + k] = acc;
            acc *= extents[k];
        }
        Layout { dims }
    }

    /// Extent of each mode.
    ///
    /// Extents must be non-negative; an extent of 1 is dropped during planning,
    /// and an extent of 0 makes the whole contraction empty.
    #[inline]
    pub fn extents(&self) -> &[i64] {
        &self.dims[..self.ndim()]
    }

    /// Stride of each mode, in elements, in the same order as
    /// [`extents`][Self::extents].
    ///
    /// A zero stride means the mode reads the same element repeatedly, which is
    /// legal and is how reductions are expressed internally; a negative stride
    /// walks backwards.
    #[inline]
    pub fn strides(&self) -> &[i64] {
        &self.dims[self.ndim()..]
    }

    /// Number of modes.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len() / 2
    }

    /// Extents, mutably. The rank cannot change through this — a `&mut [i64]`
    /// has a fixed length — so the invariant survives any write.
    ///
    /// This exists for FFI boundaries, where the rank is fixed by one call
    /// ([`resize`][Self::resize]) and the values arrive in later ones; the TAPP
    /// front end's `TAPP_set_extents` is exactly that shape.
    #[inline]
    pub fn extents_mut(&mut self) -> &mut [i64] {
        let n = self.ndim();
        &mut self.dims[..n]
    }

    /// Strides, mutably. As [`extents_mut`][Self::extents_mut], the rank is
    /// fixed.
    #[inline]
    pub fn strides_mut(&mut self) -> &mut [i64] {
        let n = self.ndim();
        &mut self.dims[n..]
    }

    /// Change the rank, padding with identity axes or truncating.
    ///
    /// New modes get extent 1 and stride 0, which is the identity for this
    /// engine: extent-1 axes are dropped during planning, so growing the rank
    /// and *not* setting the new values leaves the layout describing the same
    /// elements it did before. Both halves are resized together, so the two can
    /// never come to disagree.
    ///
    /// ```
    /// use tensorcontract::Layout;
    ///
    /// let mut l = Layout::col_major(&[2, 3]);
    /// l.resize(4);
    /// assert_eq!(l.extents(), &[2, 3, 1, 1]);
    /// assert_eq!(l.strides(), &[1, 2, 0, 0]);
    ///
    /// l.resize(1);
    /// assert_eq!(l.extents(), &[2]);
    /// assert_eq!(l.strides(), &[1]);
    /// ```
    pub fn resize(&mut self, ndim: usize) {
        let n = self.ndim();
        if ndim == n {
            return;
        }
        if ndim > n {
            // Insert the new extents at the split, then pad the strides.
            self.dims.splice(n..n, core::iter::repeat_n(1, ndim - n));
            self.dims.resize(2 * ndim, 0);
        } else {
            // Drop the stride tail first, so the split is still at `n`.
            self.dims.truncate(n + ndim);
            self.dims.drain(ndim..n);
        }
    }

    /// Number of elements addressed (product of extents).
    ///
    /// This counts *index tuples*, not distinct memory locations: with a zero
    /// stride somewhere, or a repeated label selecting a diagonal, several
    /// tuples land on the same element.
    ///
    /// # Panics
    ///
    /// In debug builds, if the product of the extents overflows `i64`. A release
    /// build wraps; planning rejects the overflow separately with
    /// [`Error::ExtentProductOverflow`].
    pub fn len(&self) -> i64 {
        self.extents().iter().product()
    }

    /// Whether any extent is zero, so the layout addresses nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size of the smallest allocation that can back this layout, assuming a
    /// base pointer at offset zero and non-negative strides.
    pub fn storage_len(&self) -> i64 {
        let mut n = 1i64;
        for (&e, &s) in self.extents().iter().zip(self.strides()) {
            if e > 0 {
                n += (e - 1) * s.abs();
            }
        }
        n
    }
}

/// Printed as the two halves it represents, not as the buffer it is stored in.
///
/// The derive would show `Layout { dims: [2, 3, 1, 2] }`, which is the storage
/// talking rather than the tensor.
impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Layout")
            .field("extents", &self.extents())
            .field("strides", &self.strides())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_layouts() {
        let cm = Layout::col_major(&[2, 3, 4]);
        assert_eq!(cm.strides(), &[1, 2, 6]);
        assert_eq!(cm.extents(), &[2, 3, 4]);
        let rm = Layout::row_major(&[2, 3, 4]);
        assert_eq!(rm.strides(), &[12, 4, 1]);
        assert_eq!(rm.extents(), &[2, 3, 4]);
        assert_eq!(cm.len(), 24);
        assert_eq!(cm.storage_len(), 24);
    }

    /// The invariant that used to be breakable: two public `Vec` fields let a
    /// caller build a layout with more extents than strides, and planning then
    /// panicked indexing `strides`. There is now no way to express it.
    #[test]
    fn rank_mismatch_is_rejected_at_the_only_place_it_can_arise() {
        assert_eq!(
            Layout::new(vec![2, 3], vec![1]),
            Err(Error::RankMismatch {
                extents: 2,
                strides: 1
            })
        );
        // And every other constructor is structurally unable to disagree.
        for n in 0..4 {
            let e: Vec<i64> = (1..=n as i64).collect();
            for l in [Layout::col_major(&e), Layout::row_major(&e)] {
                assert_eq!(l.extents().len(), l.strides().len());
                assert_eq!(l.ndim(), n);
            }
        }
    }

    #[test]
    fn debug_shows_extents_and_strides_not_the_buffer() {
        let s = format!("{:?}", Layout::col_major(&[2, 3]));
        assert!(s.contains("extents"), "{s}");
        assert!(s.contains("strides"), "{s}");
        assert!(!s.contains("dims"), "{s}");
    }
}
