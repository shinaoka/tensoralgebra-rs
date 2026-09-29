use strided_view::StridedView;

use crate::scalar::{mul, zero};
use crate::{Error, Result, Scalar};

/// Per-operand complex conjugation (DLPack has no conjugation flag).
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_blas::Conj::No, tprims_blas::Conj::Yes);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conj {
    /// Use the elements as stored.
    No,
    /// Use their complex conjugates (no effect for real types).
    Yes,
}

impl Conj {
    pub(crate) fn to_faer(self) -> faer::Conj {
        match self {
            Conj::No => faer::Conj::No,
            Conj::Yes => faer::Conj::Yes,
        }
    }
}

/// A rank-2 input operand: a borrowed view plus a conjugation flag.
/// Transposition is expressed by the view's strides (`permute(&[1, 0])`).
///
/// # Examples
///
/// ```
/// let d = [1.0f64; 4];
/// let v = strided_view::StridedView::new(&d, &[2, 2], &[1, 2], 0).unwrap();
/// let a = tprims_blas::MatIn::new(&v).conj();
/// assert_eq!(a.conj, tprims_blas::Conj::Yes);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MatIn<'v, 'a, T> {
    /// The borrowed view.
    pub view: &'v StridedView<'a, T>,
    /// Conjugation applied on read.
    pub conj: Conj,
}

impl<'v, 'a, T> MatIn<'v, 'a, T> {
    /// Use `view` as stored.
    pub fn new(view: &'v StridedView<'a, T>) -> Self {
        Self {
            view,
            conj: Conj::No,
        }
    }

    /// Conjugate on read.
    pub fn conj(mut self) -> Self {
        self.conj = Conj::Yes;
        self
    }
}

/// Validated rank-2 layout. Strides of size-1 extents are normalized to 1 so
/// provider views never see a meaningless stride.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mat2 {
    pub rows: usize,
    pub cols: usize,
    pub rs: isize,
    pub cs: isize,
}

pub(crate) fn mat2(name: &'static str, dims: &[usize], strides: &[isize]) -> Result<Mat2> {
    if dims.len() != 2 {
        return Err(Error::Rank {
            operand: name,
            expected: 2,
            got: dims.len(),
        });
    }
    let norm = |e: usize, s: isize| if e <= 1 { 1 } else { s };
    Ok(Mat2 {
        rows: dims[0],
        cols: dims[1],
        rs: norm(dims[0], strides[0]),
        cs: norm(dims[1], strides[1]),
    })
}

/// Sufficient test that distinct `(i, j)` map to distinct addresses.
pub(crate) fn check_injective(extents_strides: &[(usize, isize)]) -> Result<()> {
    // An empty layout addresses nothing.
    if extents_strides.iter().any(|&(e, _)| e == 0) {
        return Ok(());
    }
    let mut axes: Vec<(usize, usize)> = extents_strides
        .iter()
        .filter(|(e, _)| *e > 1)
        .map(|&(e, s)| (e, s.unsigned_abs()))
        .collect();
    axes.sort_by_key(|&(_, s)| s);
    let mut span = 1usize; // smallest stride the next axis must reach
    for (e, s) in axes {
        if s < span {
            return Err(Error::AliasedOutput);
        }
        span = s.checked_mul(e).ok_or(Error::AliasedOutput)?;
    }
    Ok(())
}

/// `C = beta * C` over a validated layout; `beta == 0` writes zeros without
/// reading (C may hold NaN).
///
/// # Safety
///
/// `ptr` with `m`'s extents and strides must address writable elements of
/// one allocation, exclusively borrowed by the caller.
pub(crate) unsafe fn scale_in_place<T: Scalar>(ptr: *mut T, m: &Mat2, beta: T) {
    let is_zero = beta == zero();
    for j in 0..m.cols {
        for i in 0..m.rows {
            // INVARIANT: i < rows, j < cols; offsets validated by the view.
            let p = unsafe { ptr.offset(i as isize * m.rs + j as isize * m.cs) };
            unsafe { *p = if is_zero { zero() } else { mul(beta, *p) } };
        }
    }
}
