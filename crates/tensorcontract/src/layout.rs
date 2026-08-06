//! Tensor layouts: extents plus general (possibly negative, possibly zero)
//! strides, measured in *elements*.
//!
//! This mirrors TAPP's `TAPP_tensor_info` and C++23 `layout_stride`: a tensor
//! is a base pointer plus `offset(i) = sum_k i_k * stride_k`.

use crate::error::{Error, Result};

/// Shape and strides of a dense strided tensor. Strides are in elements, not
/// bytes, and may be negative or zero.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Layout {
    /// Extent of each mode. Must be non-negative; an extent of 1 is dropped
    /// during planning, and an extent of 0 makes the whole contraction empty.
    pub extents: Vec<i64>,
    /// Stride of each mode, in elements, same order as `extents`. A zero stride
    /// means the mode reads the same element repeatedly, which is legal and is
    /// how reductions are expressed internally; a negative stride walks
    /// backwards.
    pub strides: Vec<i64>,
}

impl Layout {
    /// A layout from explicit extents and strides.
    ///
    /// The only thing validated here is that the two have the same length —
    /// nothing else is checkable without knowing the allocation, and the
    /// remaining conditions are checked when a plan is built or run.
    ///
    /// ```
    /// use tensorcontract::Layout;
    ///
    /// // A reversed axis: stride -1 walks backwards from the base pointer.
    /// let reversed = Layout::new(vec![4], vec![-1]).unwrap();
    /// assert_eq!(reversed.ndim(), 1);
    ///
    /// // A broadcast axis: stride 0 reads one element repeatedly. Legal, and
    /// // how a reduction is expressed internally.
    /// let broadcast = Layout::new(vec![2, 3], vec![1, 0]).unwrap();
    /// assert_eq!(broadcast.len(), 6);            // six index tuples...
    /// assert_eq!(broadcast.storage_len(), 2);    // ...over two elements
    ///
    /// // Mismatched lengths are the one thing rejected here.
    /// assert!(Layout::new(vec![2, 3], vec![1]).is_err());
    /// ```
    pub fn new(extents: Vec<i64>, strides: Vec<i64>) -> Result<Self> {
        if extents.len() != strides.len() {
            return Err(Error::RankMismatch {
                extents: extents.len(),
                strides: strides.len(),
            });
        }
        Ok(Layout { extents, strides })
    }

    /// Column-major ("Fortran order", first index fastest) dense layout.
    ///
    /// ```
    /// use tensorcontract::Layout;
    ///
    /// assert_eq!(Layout::col_major(&[2, 3, 4]).strides, vec![1, 2, 6]);
    /// ```
    pub fn col_major(extents: &[i64]) -> Self {
        let mut strides = Vec::with_capacity(extents.len());
        let mut acc = 1i64;
        for &e in extents {
            strides.push(acc);
            acc *= e;
        }
        Layout {
            extents: extents.to_vec(),
            strides,
        }
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
    /// assert_eq!(Layout::row_major(&[2, 3, 4]).strides, vec![12, 4, 1]);
    /// ```
    pub fn row_major(extents: &[i64]) -> Self {
        let mut strides = vec![0i64; extents.len()];
        let mut acc = 1i64;
        for k in (0..extents.len()).rev() {
            strides[k] = acc;
            acc *= extents[k];
        }
        Layout {
            extents: extents.to_vec(),
            strides,
        }
    }

    /// Number of modes.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.extents.len()
    }

    /// Number of elements addressed (product of extents).
    ///
    /// This counts *index tuples*, not distinct memory locations: with a zero
    /// stride somewhere, or a repeated label selecting a diagonal, several
    /// tuples land on the same element.
    pub fn len(&self) -> i64 {
        self.extents.iter().product()
    }

    /// Whether any extent is zero, so the layout addresses nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size of the smallest allocation that can back this layout, assuming a
    /// base pointer at offset zero and non-negative strides.
    pub fn storage_len(&self) -> i64 {
        let mut n = 1i64;
        for (&e, &s) in self.extents.iter().zip(&self.strides) {
            if e > 0 {
                n += (e - 1) * s.abs();
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_layouts() {
        let cm = Layout::col_major(&[2, 3, 4]);
        assert_eq!(cm.strides, vec![1, 2, 6]);
        let rm = Layout::row_major(&[2, 3, 4]);
        assert_eq!(rm.strides, vec![12, 4, 1]);
        assert_eq!(cm.len(), 24);
        assert_eq!(cm.storage_len(), 24);
    }
}
