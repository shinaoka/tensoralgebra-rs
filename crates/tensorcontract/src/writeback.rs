//! Accumulator write-back: `D = op_D(alpha * AB + beta * op_C(C))`.
//!
//! The micro-kernel leaves a tile in a small stack buffer, in whichever
//! [`TileFormat`] its complex method produces. Recombining that into
//! interleaved complex happens here, once per output element per `KC`-block.
//!
//! That recombination is the only per-output cost the induced methods add over
//! a hypothetical native interleaved kernel, and it differs between them:
//!
//! | tile format | work per output element |
//! |---|---|
//! | [`TileFormat::Planar`] | two loads (the interleave itself) |
//! | [`TileFormat::OneM`] | two loads, already adjacent in memory |
//! | [`TileFormat::ThreeM`] | three loads and three adds |
//!
//! Relative to the tile's own compute this is `~c / (4 * min(K, KC))` with `c`
//! the ops per element — negligible for `K >= KC`, rising like `1/(8K)` for
//! small `K`, and material only at `K` in the low single digits.
//!
//! Keeping write-back out of the kernel means one kernel serves the regular
//! fast path, the gather path, and every edge block.

use crate::element::{Element, Real};
use crate::kernel::TileFormat;

/// Read complex element `(i, j)` out of an accumulator tile.
#[inline(always)]
unsafe fn tile_value<R: Real>(
    ab: *const R,
    fmt: TileFormat,
    mr: usize,
    nr: usize,
    i: usize,
    j: usize,
) -> (R, R) {
    match fmt {
        TileFormat::Real => (*ab.add(j * mr + i), R::ZERO),
        TileFormat::Planar => (*ab.add(j * mr + i), *ab.add(mr * nr + j * mr + i)),
        // One real `2*mr x nr` tile: row 2i is Re, row 2i+1 is Im.
        TileFormat::OneM => {
            let base = j * (2 * mr) + 2 * i;
            (*ab.add(base), *ab.add(base + 1))
        }
        // Karatsuba recombination: Cr = M1 - M2, Ci = M3 - M1 - M2.
        TileFormat::ThreeM => {
            let plane = mr * nr;
            let off = j * mr + i;
            let m1 = *ab.add(off);
            let m2 = *ab.add(plane + off);
            let m3 = *ab.add(2 * plane + off);
            (m1 - m2, m3 - m1 - m2)
        }
    }
}

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
/// `ab` must hold a tile of the size implied by `fmt`, `mr` and `nr`; the
/// scatter offsets must be in bounds for their tensors.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn writeback<T: Element>(
    ab: *const T::Real,
    fmt: TileFormat,
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

    for j in 0..nrem {
        let dcj = *d_c.get_unchecked(j);
        let ccj = *c_c.get_unchecked(j);
        for i in 0..mrem {
            let (re, im) = tile_value::<T::Real>(ab, fmt, mr, nr, i, j);
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
    for (&dcj, &ccj) in d_c.iter().zip(c_c) {
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
