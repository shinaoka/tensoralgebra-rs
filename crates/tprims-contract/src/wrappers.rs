use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_blas::{is_injective_layout, Scalar};
use tprims_exec::strided::run_with_exec;
use tprims_exec::Exec;

use crate::{Error, Result};

fn check_out<T>(c: &StridedViewMut<'_, T>) -> Result<()> {
    let l: Vec<(usize, isize)> = c
        .dims()
        .iter()
        .copied()
        .zip(c.strides().iter().copied())
        .collect();
    if is_injective_layout(&l) {
        Ok(())
    } else {
        Err(Error::AliasedOutput)
    }
}

/// `C[perm[0], perm[1], ...] = A`: copy with an axis permutation, so that
/// axis `k` of C is axis `perm[k]` of A.
///
/// # Errors
///
/// [`Error::Config`] for an invalid permutation, [`Error::Shape`] for C's
/// extents, [`Error::AliasedOutput`].
///
/// # Examples
///
/// ```
/// use strided_view::{StridedView, StridedViewMut};
/// let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3 column-major
/// let mut c = [0.0; 6];
/// let av = StridedView::new(&a, &[2, 3], &[1, 2], 0).unwrap();
/// let mut cv = StridedViewMut::new(&mut c, &[3, 2], &[1, 3], 0).unwrap();
/// tprims_contract::permute(&tprims_exec::Exec::serial(), &av, &[1, 0], &mut cv).unwrap();
/// assert_eq!(c, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
/// ```
pub fn permute<T: Scalar>(
    exec: &Exec<'_>,
    a: &StridedView<'_, T>,
    perm: &[usize],
    c: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    let r = a.ndim();
    let mut seen = vec![false; r];
    if perm.len() != r
        || perm
            .iter()
            .any(|&p| p >= r || std::mem::replace(&mut seen[p], true))
    {
        return Err(Error::Config(format!(
            "{perm:?} is not a permutation of 0..{r}"
        )));
    }
    let want: Vec<usize> = perm.iter().map(|&p| a.dims()[p]).collect();
    if c.dims() != want.as_slice() {
        return Err(Error::Shape(format!(
            "C has extents {:?}, expected {want:?}",
            c.dims()
        )));
    }
    check_out(c)?;
    let src = a.permute(perm).map_err(|e| Error::Backend(e.to_string()))?;
    let len = c.len();
    run_with_exec(exec, len, |_| strided_basic::copy_into(c, &src))
        .map_err(|e| Error::Backend(e.to_string()))
}

/// `C = alpha * A + beta * C` elementwise over equal extents (any strides);
/// `beta == 0` never reads C.
///
/// # Errors
///
/// [`Error::Shape`], [`Error::AliasedOutput`].
///
/// # Examples
///
/// ```
/// use strided_view::{StridedView, StridedViewMut};
/// let a = [1.0, 2.0];
/// let mut c = [10.0, 20.0];
/// let av = StridedView::new(&a, &[2], &[1], 0).unwrap();
/// let mut cv = StridedViewMut::new(&mut c, &[2], &[1], 0).unwrap();
/// tprims_contract::add(&tprims_exec::Exec::serial(), 2.0, &av, 0.5, &mut cv).unwrap();
/// assert_eq!(c, [7.0, 14.0]);
/// ```
pub fn add<T: Scalar>(
    exec: &Exec<'_>,
    alpha: T,
    a: &StridedView<'_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    if a.dims() != c.dims() {
        return Err(Error::Shape(format!(
            "A {:?} vs C {:?}",
            a.dims(),
            c.dims()
        )));
    }
    check_out(c)?;
    let len = c.len();
    if beta == <T as Element>::zero() {
        return run_with_exec(exec, len, |_| {
            strided_basic::map_into(c, a, move |x| Element::mul(alpha, x))
        })
        .map_err(|e| Error::Backend(e.to_string()));
    }
    if len == 0 {
        return Ok(());
    }
    let (dims, cs, as_) = (
        c.dims().to_vec(),
        c.strides().to_vec(),
        a.strides().to_vec(),
    );
    // SAFETY: both views are non-empty and bounds-checked with these extents;
    // C is exclusive and injective (checked above); A is a distinct borrow.
    unsafe {
        crate::util::zip_update(
            exec,
            &dims,
            (c.as_mut_ptr(), &cs),
            [(a.ptr(), &as_)],
            true,
            &move |y, [x]| Element::add(Element::mul(alpha, x), Element::mul(beta, y)),
        )
    };
    Ok(())
}
