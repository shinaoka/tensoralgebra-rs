//! Accumulator write-back: `D = op_D(alpha * AB + beta * op_C(C))`.
//!
//! The micro-kernel leaves an `MR x NR` tile in a small stack buffer, planar
//! for complex. Re-interleaving happens here, once per output element per
//! `KC`-block, which is the only cost the planar representation adds relative
//! to a native interleaved kernel. That cost is
//! `~c / (4 * min(K, KC))` of the tile's compute, with `c` the interleave-ops
//! per element — negligible for `K >= KC` and only material for very small `K`
//! (see the dispatch note in `DESIGN.md`).
//!
//! Keeping write-back out of the kernel means one kernel serves the regular
//! fast path, the gather path, and every edge block.

use crate::element::{Element, Real};

/// Write one micro-tile back to `C`/`D`.
///
/// `c_r`/`c_c` and `d_r`/`d_c` are the scatter offsets of the tile's rows and
/// columns, each of length `mrem`/`nrem`.
///
/// To accumulate a later `KC` block, callers pass `beta = 1`, `c_base =
/// d_base`, `c_r = d_r`, `c_c = d_c` and `conj_c = conj_d`. Because
/// conjugation is additive and involutive, that reproduces
/// `op_D(alpha * sum_p AB_p + beta * op_C(C))` exactly.
///
/// # Safety
/// `ab` must hold `mr * nr` (real) or `2 * mr * nr` (complex) values; the
/// scatter offsets must be in bounds for their tensors.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn writeback<T: Element>(
    ab: *const T::Real,
    mr: usize,
    nr: usize,
    mrem: usize,
    nrem: usize,
    alpha: T,
    beta: T,
    c_base: *const T,
    c_r: &[i64],
    c_c: &[i64],
    conj_c: bool,
    d_base: *mut T,
    d_r: &[i64],
    d_c: &[i64],
    conj_d: bool,
) {
    let beta_is_zero = beta == T::zero();
    let im_plane = mr * nr;

    for j in 0..nrem {
        let col = j * mr;
        let dcj = *d_c.get_unchecked(j);
        let ccj = *c_c.get_unchecked(j);
        for i in 0..mrem {
            let re = *ab.add(col + i);
            let im = if T::IS_COMPLEX {
                *ab.add(im_plane + col + i)
            } else {
                T::Real::ZERO
            };
            let mut v = T::from_parts(re, im).mul(alpha);
            if !beta_is_zero {
                let cv = *c_base.offset((*c_r.get_unchecked(i) + ccj) as isize);
                let cv = if conj_c { cv.conj() } else { cv };
                v = v.add(cv.mul(beta));
            }
            if conj_d {
                v = v.conj();
            }
            *d_base.offset((*d_r.get_unchecked(i) + dcj) as isize) = v;
        }
    }
}

/// `D = op_D(beta * op_C(C))` over a whole (batched) matrix view.
///
/// Used when the contraction extent is zero, and as the `beta` pre-pass is not
/// needed elsewhere, this is the only place that touches `D` without a kernel.
///
/// # Safety
/// The scatter offsets must be in bounds for their tensors.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn scale_only<T: Element>(
    beta: T,
    c_base: *const T,
    c_r: &[i64],
    c_c: &[i64],
    conj_c: bool,
    d_base: *mut T,
    d_r: &[i64],
    d_c: &[i64],
    conj_d: bool,
) {
    let beta_is_zero = beta == T::zero();
    for (j, (&dcj, &ccj)) in d_c.iter().zip(c_c).enumerate() {
        let _ = j;
        for (&dri, &cri) in d_r.iter().zip(c_r) {
            let mut v = T::zero();
            if !beta_is_zero {
                let cv = *c_base.offset((cri + ccj) as isize);
                let cv = if conj_c { cv.conj() } else { cv };
                v = cv.mul(beta);
            }
            if conj_d {
                v = v.conj();
            }
            *d_base.offset((dri + dcj) as isize) = v;
        }
    }
}
