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
    pub extents: Vec<i64>,
    pub strides: Vec<i64>,
}

impl Layout {
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

    #[inline]
    pub fn ndim(&self) -> usize {
        self.extents.len()
    }

    /// Number of elements addressed (product of extents).
    pub fn len(&self) -> i64 {
        self.extents.iter().product()
    }

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
