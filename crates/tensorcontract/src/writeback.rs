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
//!
//! # Why this file has a fast path at all
//!
//! On compute-bound shapes the above is exact and write-back is noise. On the
//! low-arithmetic-intensity corpus cases it is not: with `k` in the tens, a
//! micro-tile's write-back costs a constant per output element against only
//! `~4k` flops of kernel work, and `perf` puts more than half of the run time
//! in this file. So the row offsets are taken from the *block* scatter when the
//! block is regular, exactly as [`crate::pack`] does, which removes the
//! scatter-table load from the innermost loop and lets the plane recombination
//! vectorise.

use crate::element::{Element, Real};
use crate::kernel::TileFormat;
use crate::scatter::IRREGULAR;

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
/// columns, each of length `mrem`/`nrem`. `c_rs`/`d_rs` are the corresponding
/// *block* scatter entries for the row runs — the common difference of `c_r` /
/// `d_r`, or [`IRREGULAR`]. They are redundant with `c_r`/`d_r` and only ever
/// select a faster loop; passing [`IRREGULAR`] for both is always correct.
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
    c_rs: i64,
    conj_c: bool,
    d_base: *mut T,
    d_r: &[i64],
    d_c: &[i64],
    d_rs: i64,
    conj_d: bool,
) {
    let beta_is_zero = beta == T::zero();
    // `C` is only read when beta is nonzero, so its regularity only matters
    // then. `beta = 0` with a scattered `C` is the common case (the harness and
    // most callers overwrite `D`), and it must not be pushed onto the slow path.
    let c_ok = beta_is_zero || c_rs != IRREGULAR;
    let (d0, c0) = (*d_r.get_unchecked(0), *c_r.get_unchecked(0));

    if d_rs == IRREGULAR || !c_ok {
        writeback_rows::<T, _, _>(
            ab,
            fmt,
            mr,
            nr,
            mrem,
            nrem,
            alpha,
            beta,
            beta_is_zero,
            c_base,
            |i| *c_r.get_unchecked(i),
            c_c,
            conj_c,
            d_base,
            |i| *d_r.get_unchecked(i),
            d_c,
            conj_d,
        )
    } else if d_rs == 1 && (beta_is_zero || c_rs == 1) {
        // Unit stride: a micro-tile column is a contiguous run of `D`. Worth
        // its own instantiation rather than folding into the strided one,
        // because only a *compile-time* unit stride lets LLVM turn the plane
        // recombination into vector loads and interleaved stores. This is the
        // case `Plan::transposes_gemm` exists to create.
        writeback_rows::<T, _, _>(
            ab,
            fmt,
            mr,
            nr,
            mrem,
            nrem,
            alpha,
            beta,
            beta_is_zero,
            c_base,
            |i| c0 + i as i64,
            c_c,
            conj_c,
            d_base,
            |i| d0 + i as i64,
            d_c,
            conj_d,
        )
    } else {
        writeback_rows::<T, _, _>(
            ab,
            fmt,
            mr,
            nr,
            mrem,
            nrem,
            alpha,
            beta,
            beta_is_zero,
            c_base,
            |i| c0 + c_rs * i as i64,
            c_c,
            conj_c,
            d_base,
            |i| d0 + d_rs * i as i64,
            d_c,
            conj_d,
        )
    }
}

/// The write-back loop, with the row offsets supplied by `c_row`/`d_row`.
///
/// Instantiated once per row-addressing mode. The `alpha == 1, beta == 0, no
/// conjugation` case gets its own inner loop because it is both the common one
/// and the only one that reduces to a straight tile-to-`D` copy.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn writeback_rows<T: Element, CR, DR>(
    ab: *const T::Real,
    fmt: TileFormat,
    mr: usize,
    nr: usize,
    mrem: usize,
    nrem: usize,
    alpha: T,
    beta: T,
    beta_is_zero: bool,
    c_base: *const T,
    c_row: CR,
    c_c: &[i64],
    conj_c: bool,
    d_base: *mut T,
    d_row: DR,
    d_c: &[i64],
    conj_d: bool,
) where
    CR: Fn(usize) -> i64,
    DR: Fn(usize) -> i64,
{
    let plain = beta_is_zero && !conj_d && alpha == T::one();

    for j in 0..nrem {
        let dcj = *d_c.get_unchecked(j);
        let ccj = *c_c.get_unchecked(j);
        if plain {
            for i in 0..mrem {
                let (re, im) = tile_value::<T::Real>(ab, fmt, mr, nr, i, j);
                *d_base.offset((d_row(i) + dcj) as isize) = T::from_parts(re, im);
            }
        } else {
            for i in 0..mrem {
                let (re, im) = tile_value::<T::Real>(ab, fmt, mr, nr, i, j);
                let mut v = T::from_parts(re, im).mul(alpha);
                if !beta_is_zero {
                    let cv = *c_base.offset((c_row(i) + ccj) as isize);
                    let cv = if conj_c { cv.conj() } else { cv };
                    v = v.add(cv.mul(beta));
                }
                if conj_d {
                    v = v.conj();
                }
                *d_base.offset((d_row(i) + dcj) as isize) = v;
            }
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
