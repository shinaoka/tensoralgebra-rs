//! Packing: gather a block of a scatter matrix into a register-blocked panel,
//! emitting whichever complex representation the chosen method needs.
//!
//! A tensor contraction has to make a scatter/gather pass over every element of
//! `A` and `B` regardless of how complex numbers are represented. That makes
//! packing the natural place to change representation: the loads are identical
//! whatever comes out, only the store pattern differs. All three complex
//! methods therefore cost the same *reads* here and differ only in stores:
//!
//! | format | reals stored per complex element |
//! |---|---|
//! | [`PackFormat::Planar`] | 2 |
//! | [`PackFormat::ThreeM`] | 3 (the sum plane is formed on the fly) |
//! | [`PackFormat::OneE`] | 4 |
//!
//! BLIS's 1m must emit A in "1e" format — the `2x2` block
//! `[[re, -im], [im, re]]` — so its packed A panel is twice the size of the
//! planar one. That is the concrete structural difference between the methods,
//! and `Blocking::derive` accounts for it by giving 1m a proportionally
//! smaller `MC` so the L2 budget stays equal.
//!
//! Conjugation of an operand is absorbed here for free by negating the
//! imaginary part as it is written, in every format.

use crate::element::Element;
use crate::kernel::PackFormat;
use crate::scatter::IRREGULAR;

/// Number of `T::Real` values a packed panel needs.
#[inline]
pub(crate) fn panel_len(vlen: usize, vr: usize, kc: usize, fmt: PackFormat) -> usize {
    vlen.div_ceil(vr) * vr * kc * fmt.reals_per_element()
}

/// Gather `vscat.len() x kscat.len()` elements into `ceil(vscat.len()/vr)`
/// slivers of `vr` "vector-axis" values per logical k-step.
///
/// The same routine packs both operands: for `A` the vector axis is the free
/// index `M` and for `B` it is the free index `N`; in both cases `kscat` walks
/// the contraction index. Element `(v, p)` of the block is at
/// `base + vscat[v] + kscat[p]`.
///
/// Sliver `s` starts at `s * vr * kc * fmt.reals_per_element()`; within it,
/// logical k-step `p` starts at `p * vr * fmt.reals_per_element()` and is laid
/// out per [`PackFormat`].
///
/// Rows past the end of a partial sliver are zero-filled so the micro-kernel
/// can always run at full register-block size.
///
/// # Safety
/// `base` must be valid for reads at every `vscat[v] + kscat[p]`, and `out`
/// valid for `panel_len(vscat.len(), vr, kscat.len(), fmt)` writes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn pack_panel<T: Element>(
    base: *const T,
    vscat: &[i64],
    vbs: &[i64],
    kscat: &[i64],
    vr: usize,
    conj: bool,
    fmt: PackFormat,
    out: *mut T::Real,
) {
    let per_k = vr * fmt.reals_per_element();
    let kc = kscat.len();
    let vlen_total = vscat.len();
    let nsliv = vlen_total.div_ceil(vr);
    let sliver_stride = per_k * kc;

    for (s, &bs) in vbs.iter().enumerate().take(nsliv) {
        let v0 = s * vr;
        let vlen = (vlen_total - v0).min(vr);
        let dst = out.add(s * sliver_stride);

        if vlen == vr && bs != IRREGULAR {
            // Regular block: the vr values for a fixed k are an arithmetic
            // progression, so no scatter table lookups are needed.
            let b0 = vscat[v0];
            for (p, &koff) in kscat.iter().enumerate() {
                let src = base.offset((b0 + koff) as isize);
                let o = dst.add(p * per_k);
                if bs == 1 {
                    for t in 0..vr {
                        emit::<T>(*src.add(t), o, vr, t, conj, fmt);
                    }
                } else {
                    for t in 0..vr {
                        emit::<T>(*src.offset((bs * t as i64) as isize), o, vr, t, conj, fmt);
                    }
                }
            }
        } else {
            // Irregular or partial block: full gather, zero-filling the tail.
            for (p, &koff) in kscat.iter().enumerate() {
                let o = dst.add(p * per_k);
                for t in 0..vr {
                    let z = if t < vlen {
                        *base.offset((*vscat.get_unchecked(v0 + t) + koff) as isize)
                    } else {
                        T::zero()
                    };
                    emit::<T>(z, o, vr, t, conj, fmt);
                }
            }
        }
    }
}

/// Write one gathered element into lane `t` of a logical k-step.
#[inline(always)]
unsafe fn emit<T: Element>(
    z: T,
    o: *mut T::Real,
    vr: usize,
    t: usize,
    conj: bool,
    fmt: PackFormat,
) {
    let re = z.re();
    match fmt {
        PackFormat::Real => *o.add(t) = re,
        PackFormat::Planar => {
            let im = if conj { -z.im() } else { z.im() };
            *o.add(t) = re;
            *o.add(vr + t) = im;
        }
        PackFormat::ThreeM => {
            let im = if conj { -z.im() } else { z.im() };
            *o.add(t) = re;
            *o.add(vr + t) = im;
            *o.add(2 * vr + t) = re + im;
        }
        PackFormat::OneE => {
            // Real 2x2 block [[re, -im], [im, re]], stored as two real k-steps
            // of 2*vr, with the two rows of each complex lane adjacent.
            let im = if conj { -z.im() } else { z.im() };
            *o.add(2 * t) = re;
            *o.add(2 * t + 1) = im;
            *o.add(2 * vr + 2 * t) = -im;
            *o.add(2 * vr + 2 * t + 1) = re;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::C64;
    use crate::scatter::build_block_scatter;

    fn pack_c(
        data: &[C64],
        vscat: &[i64],
        kscat: &[i64],
        vr: usize,
        conj: bool,
        fmt: PackFormat,
    ) -> Vec<f64> {
        let vbs = build_block_scatter(vscat, vr);
        let mut out = vec![f64::NAN; panel_len(vscat.len(), vr, kscat.len(), fmt)];
        unsafe {
            pack_panel::<C64>(
                data.as_ptr(),
                vscat,
                &vbs,
                kscat,
                vr,
                conj,
                fmt,
                out.as_mut_ptr(),
            )
        };
        out
    }

    #[test]
    fn packs_real_regular_block_with_zero_fill() {
        // 3x2 block of a 4x2 column-major matrix, vr = 2 -> 2 slivers,
        // the second one half empty.
        let data: Vec<f64> = (0..8).map(|x| x as f64).collect();
        let vscat = vec![0i64, 1, 2];
        let kscat = vec![0i64, 4];
        let vbs = build_block_scatter(&vscat, 2);
        let mut out = vec![-1.0f64; panel_len(3, 2, 2, PackFormat::Real)];
        unsafe {
            pack_panel::<f64>(
                data.as_ptr(),
                &vscat,
                &vbs,
                &kscat,
                2,
                false,
                PackFormat::Real,
                out.as_mut_ptr(),
            )
        };
        // sliver 0: k=0 -> [0,1], k=1 -> [4,5]
        // sliver 1: k=0 -> [2,0], k=1 -> [6,0]
        assert_eq!(out, vec![0.0, 1.0, 4.0, 5.0, 2.0, 0.0, 6.0, 0.0]);
    }

    #[test]
    fn packs_complex_into_planar_planes() {
        let data: Vec<C64> = (0..4)
            .map(|i| C64::new(i as f64, 10.0 + i as f64))
            .collect();
        let out = pack_c(&data, &[0, 1], &[0, 2], 2, false, PackFormat::Planar);
        // k=0: re[0,1] im[10,11]; k=1: re[2,3] im[12,13]
        assert_eq!(out, vec![0.0, 1.0, 10.0, 11.0, 2.0, 3.0, 12.0, 13.0]);
    }

    #[test]
    fn packs_complex_into_one_e_blocks() {
        let data: Vec<C64> = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
        let out = pack_c(&data, &[0, 1], &[0], 2, false, PackFormat::OneE);
        assert_eq!(
            out,
            vec![
                1.0, 2.0, 3.0, 4.0, // real step 0: [re0, im0, re1, im1]
                -2.0, 1.0, -4.0, 3.0, // real step 1: [-im0, re0, -im1, re1]
            ]
        );
        assert_eq!(out.len(), 4 * 2, "1m stores four reals per complex element");
    }

    #[test]
    fn packs_complex_into_three_m_planes() {
        let data: Vec<C64> = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
        let out = pack_c(&data, &[0, 1], &[0], 2, false, PackFormat::ThreeM);
        // [re | im | re+im]
        assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0, 3.0, 7.0]);
    }

    #[test]
    fn conjugation_negates_the_imaginary_part_in_every_format() {
        let data: Vec<C64> = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];

        let planar = pack_c(&data, &[0, 1], &[0], 2, true, PackFormat::Planar);
        assert_eq!(planar, vec![1.0, 3.0, -2.0, -4.0]);

        let threem = pack_c(&data, &[0, 1], &[0], 2, true, PackFormat::ThreeM);
        assert_eq!(threem, vec![1.0, 3.0, -2.0, -4.0, -1.0, -1.0]);

        let onee = pack_c(&data, &[0, 1], &[0], 2, true, PackFormat::OneE);
        assert_eq!(onee, vec![1.0, -2.0, 3.0, -4.0, 2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn packed_size_ratios_match_the_method_costs() {
        let data: Vec<C64> = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
        let n = |fmt| pack_c(&data, &[0, 1], &[0, 0], 2, false, fmt).len();
        assert_eq!(n(PackFormat::Planar), 8);
        assert_eq!(n(PackFormat::ThreeM), 12);
        assert_eq!(n(PackFormat::OneE), 16);
    }

    #[test]
    fn irregular_block_uses_gather_path() {
        let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let vscat = vec![0i64, 3, 1]; // not an arithmetic progression
        let kscat = vec![0i64];
        let vbs = build_block_scatter(&vscat, 3);
        assert_eq!(vbs, vec![IRREGULAR]);
        let mut out = vec![0.0f64; panel_len(3, 3, 1, PackFormat::Real)];
        unsafe {
            pack_panel::<f64>(
                data.as_ptr(),
                &vscat,
                &vbs,
                &kscat,
                3,
                false,
                PackFormat::Real,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(out, vec![0.0, 3.0, 1.0]);
    }
}
