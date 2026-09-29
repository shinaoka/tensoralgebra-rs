use tensorcontract::Element;
use tprims_blas::Scalar;

/// `C = beta * C` over a rank-n strided layout; `beta == 0` writes zeros
/// without reading.
///
/// # Safety
///
/// `origin` is the element at logical index 0 of a non-empty, bounds-checked,
/// exclusively borrowed view with these extents and strides.
pub(crate) unsafe fn scale<T: Scalar>(origin: *mut T, dims: &[usize], strides: &[isize], beta: T) {
    if dims.contains(&0) {
        return;
    }
    let zero = beta == <T as Element>::zero();
    let mut idx = vec![0usize; dims.len()];
    let mut off = 0isize;
    loop {
        // SAFETY: `off` is the offset of `idx`, inside the view.
        unsafe {
            let p = origin.offset(off);
            *p = if zero {
                <T as Element>::zero()
            } else {
                Element::mul(beta, *p)
            };
        }
        let mut k = 0;
        loop {
            if k == dims.len() {
                return;
            }
            idx[k] += 1;
            off += strides[k];
            if idx[k] < dims[k] {
                break;
            }
            off -= strides[k] * dims[k] as isize;
            idx[k] = 0;
            k += 1;
        }
    }
}
