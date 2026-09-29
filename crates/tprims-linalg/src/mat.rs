use strided_view::{StridedView, StridedViewMut};
use tprims_blas::Scalar;

use crate::{Error, Result};

/// A library-owned dense column-major matrix (factors, results).
///
/// # Examples
///
/// ```
/// let mut m = tprims_linalg::Matrix::<f64>::zeros(2, 3).unwrap();
/// m.set(1, 2, 5.0);
/// assert_eq!(m.get(1, 2), 5.0);
/// assert_eq!(m.data()[1 + 2 * 2], 5.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Scalar> Matrix<T> {
    /// A zero matrix.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] when `rows * cols` overflows.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self> {
        let len = rows
            .checked_mul(cols)
            .filter(|&l| {
                l.checked_mul(std::mem::size_of::<T>())
                    .is_some_and(|b| b <= isize::MAX as usize)
            })
            .ok_or_else(|| Error::Shape(format!("{rows}x{cols} matrix is too large")))?;
        Ok(Self {
            rows,
            cols,
            data: vec![<T as tensorcontract::Element>::zero(); len],
        })
    }

    /// A zero matrix whose size was already validated by [`Matrix::zeros`] or
    /// by an existing matrix of at least this many elements.
    pub(crate) fn zeros_validated(rows: usize, cols: usize) -> Self {
        // INVARIANT: callers pass extents whose product was checked (see doc).
        Self {
            rows,
            cols,
            data: vec![<T as tensorcontract::Element>::zero(); rows * cols],
        }
    }

    /// Wrap column-major data.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] when `data.len() != rows * cols`.
    pub fn from_col_major(rows: usize, cols: usize, data: Vec<T>) -> Result<Self> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(Error::Shape(format!(
                "{rows}x{cols} from {} elements",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Copy a rank-2 view (any strides) into a new column-major matrix.
    ///
    /// # Errors
    ///
    /// [`Error::Rank`] for a view that is not rank 2.
    pub fn from_view(v: &StridedView<'_, T>) -> Result<Self> {
        if v.ndim() != 2 {
            return Err(Error::Rank {
                operand: "matrix",
                expected: 2,
                got: v.ndim(),
            });
        }
        let (rows, cols) = (v.dims()[0], v.dims()[1]);
        let (rs, cs) = (v.strides()[0], v.strides()[1]);
        let p = v.ptr();
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::Shape(format!("{rows}x{cols} matrix is too large")))?;
        let mut data = Vec::with_capacity(len);
        for j in 0..cols {
            for i in 0..rows {
                // SAFETY: the view was bounds-checked at construction and
                // (i, j) is inside its extents.
                data.push(unsafe { *p.offset(i as isize * rs + j as isize * cs) });
            }
        }
        Ok(Self { rows, cols, data })
    }

    /// Rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Column-major data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Element `(i, j)`; panics when out of range (like slice indexing).
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(
            i < self.rows && j < self.cols,
            "index ({i}, {j}) out of range"
        );
        self.data[i + j * self.rows]
    }

    /// Set element `(i, j)`; panics when out of range.
    pub fn set(&mut self, i: usize, j: usize, v: T) {
        assert!(
            i < self.rows && j < self.cols,
            "index ({i}, {j}) out of range"
        );
        self.data[i + j * self.rows] = v;
    }

    /// Borrow as a strided view.
    pub fn view(&self) -> StridedView<'_, T> {
        // INVARIANT: compact column-major layout always fits `data`.
        StridedView::new(
            &self.data,
            &[self.rows, self.cols],
            &[1, self.rows.max(1) as isize],
            0,
        )
        .unwrap_or_else(|_| unreachable!("compact layout"))
    }

    /// Borrow as a mutable strided view.
    pub fn view_mut(&mut self) -> StridedViewMut<'_, T> {
        let (r, c) = (self.rows, self.cols);
        // INVARIANT: compact column-major layout always fits `data`.
        StridedViewMut::new(&mut self.data, &[r, c], &[1, r.max(1) as isize], 0)
            .unwrap_or_else(|_| unreachable!("compact layout"))
    }

    pub(crate) fn as_faer(&self) -> faer::MatRef<'_, T> {
        faer::MatRef::from_column_major_slice(&self.data, self.rows, self.cols)
    }

    pub(crate) fn as_faer_mut(&mut self) -> faer::MatMut<'_, T> {
        faer::MatMut::from_column_major_slice_mut(&mut self.data, self.rows, self.cols)
    }
}
