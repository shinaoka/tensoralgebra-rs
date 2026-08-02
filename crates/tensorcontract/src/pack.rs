//! Packing: gather a block of a scatter matrix into a register-blocked panel,
//! splitting complex data into planar (real/imaginary) planes on the way.
//!
//! This is the heart of the project's thesis. A tensor contraction has to make
//! a scatter/gather pass over every element of `A` and `B` regardless of how
//! complex numbers are represented. Emitting planar output from that pass
//! costs nothing extra — the loads are the same, only the store addresses
//! differ — but it hands the micro-kernel real and imaginary parts already
//! separated into contiguous runs, so no in-register shuffles are needed.
//!
//! By contrast BLIS's 1m method must emit A in "1e" format, writing **four**
//! reals per complex element (the 2x2 block `[[re, -im], [im, re]]`). Its
//! packed A panel is therefore twice the size of ours, doubling both the store
//! traffic of the packing pass and the L2 footprint of the panel.
//!
//! Conjugation of an operand is absorbed here for free by negating the
//! imaginary plane as it is written.

use crate::element::Element;
use crate::scatter::IRREGULAR;

/// Number of `T::Real` values a packed panel needs.
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn panel_len<T: Element>(vlen: usize, vr: usize, kc: usize) -> usize {
    vlen.div_ceil(vr) * vr * kc * T::PLANES
}

/// Gather `vscat.len() x kscat.len()` elements into `ceil(vscat.len()/vr)`
/// slivers of `vr` "vector-axis" values per k-step.
///
/// The same routine packs both operands: for `A` the vector axis is the free
/// index `M` and for `B` it is the free index `N`; in both cases `kscat` walks
/// the contraction index. Element `(v, p)` of the block is at
/// `base + vscat[v] + kscat[p]`.
///
/// Output layout, with `P = T::PLANES`:
///
/// ```text
/// out[s * (vr*kc*P) + p * (vr*P) + plane * vr + t]
/// ```
///
/// Rows past the end of a partial sliver are zero-filled so the micro-kernel
/// can always run at full register-block size.
///
/// # Safety
/// `base` must be valid for reads at every `vscat[v] + kscat[p]`, and `out`
/// valid for `panel_len::<T>(vscat.len(), vr, kscat.len())` writes.
pub(crate) unsafe fn pack_panel<T: Element>(
    base: *const T,
    vscat: &[i64],
    vbs: &[i64],
    kscat: &[i64],
    vr: usize,
    conj: bool,
    out: *mut T::Real,
) {
    let planes = T::PLANES;
    let kc = kscat.len();
    let vlen_total = vscat.len();
    let nsliv = vlen_total.div_ceil(vr);
    let sliver_stride = vr * kc * planes;

    for (s, &bs) in vbs.iter().enumerate().take(nsliv) {
        let v0 = s * vr;
        let vlen = (vlen_total - v0).min(vr);
        let dst = out.add(s * sliver_stride);

        if vlen == vr && bs != IRREGULAR {
            // Regular block: the vr values for a fixed k are an arithmetic
            // progression, so no scatter table lookups are needed.
            let b0 = vscat[v0];
            if bs == 1 {
                for (p, &koff) in kscat.iter().enumerate() {
                    let src = base.offset((b0 + koff) as isize);
                    let o = dst.add(p * planes * vr);
                    pack_run_unit::<T>(src, o, vr, conj);
                }
            } else {
                for (p, &koff) in kscat.iter().enumerate() {
                    let src = base.offset((b0 + koff) as isize);
                    let o = dst.add(p * planes * vr);
                    pack_run_strided::<T>(src, bs, o, vr, conj);
                }
            }
        } else {
            // Irregular or partial block: full gather, zero-filling the tail.
            for (p, &koff) in kscat.iter().enumerate() {
                let o = dst.add(p * planes * vr);
                for t in 0..vr {
                    let z = if t < vlen {
                        *base.offset((*vscat.get_unchecked(v0 + t) + koff) as isize)
                    } else {
                        T::zero()
                    };
                    *o.add(t) = z.re();
                    if T::IS_COMPLEX {
                        *o.add(vr + t) = if conj { -z.im() } else { z.im() };
                    }
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn pack_run_unit<T: Element>(src: *const T, o: *mut T::Real, vr: usize, conj: bool) {
    if T::IS_COMPLEX {
        if conj {
            for t in 0..vr {
                let z = *src.add(t);
                *o.add(t) = z.re();
                *o.add(vr + t) = -z.im();
            }
        } else {
            for t in 0..vr {
                let z = *src.add(t);
                *o.add(t) = z.re();
                *o.add(vr + t) = z.im();
            }
        }
    } else {
        for t in 0..vr {
            *o.add(t) = (*src.add(t)).re();
        }
    }
}

#[inline(always)]
unsafe fn pack_run_strided<T: Element>(
    src: *const T,
    stride: i64,
    o: *mut T::Real,
    vr: usize,
    conj: bool,
) {
    if T::IS_COMPLEX {
        for t in 0..vr {
            let z = *src.offset((stride * t as i64) as isize);
            *o.add(t) = z.re();
            *o.add(vr + t) = if conj { -z.im() } else { z.im() };
        }
    } else {
        for t in 0..vr {
            *o.add(t) = (*src.offset((stride * t as i64) as isize)).re();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::C64;
    use crate::scatter::build_block_scatter;

    #[test]
    fn packs_real_regular_block_with_zero_fill() {
        // 3x2 block of a 4x2 column-major matrix, vr = 2 -> 2 slivers,
        // the second one half empty.
        let data: Vec<f64> = (0..8).map(|x| x as f64).collect();
        let vscat = vec![0i64, 1, 2];
        let kscat = vec![0i64, 4];
        let vbs = build_block_scatter(&vscat, 2);
        let mut out = vec![-1.0f64; panel_len::<f64>(3, 2, 2)];
        unsafe {
            pack_panel::<f64>(
                data.as_ptr(),
                &vscat,
                &vbs,
                &kscat,
                2,
                false,
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
        let vscat = vec![0i64, 1];
        let kscat = vec![0i64, 2];
        let vbs = build_block_scatter(&vscat, 2);
        let mut out = vec![0.0f64; panel_len::<C64>(2, 2, 2)];
        unsafe {
            pack_panel::<C64>(
                data.as_ptr(),
                &vscat,
                &vbs,
                &kscat,
                2,
                false,
                out.as_mut_ptr(),
            )
        };
        // k=0: re[0,1] im[10,11]; k=1: re[2,3] im[12,13]
        assert_eq!(out, vec![0.0, 1.0, 10.0, 11.0, 2.0, 3.0, 12.0, 13.0]);
    }

    #[test]
    fn conjugation_negates_the_imaginary_plane() {
        let data: Vec<C64> = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
        let vscat = vec![0i64, 1];
        let kscat = vec![0i64];
        let vbs = build_block_scatter(&vscat, 2);
        let mut out = vec![0.0f64; panel_len::<C64>(2, 2, 1)];
        unsafe {
            pack_panel::<C64>(
                data.as_ptr(),
                &vscat,
                &vbs,
                &kscat,
                2,
                true,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(out, vec![1.0, 3.0, -2.0, -4.0]);
    }

    #[test]
    fn irregular_block_uses_gather_path() {
        let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let vscat = vec![0i64, 3, 1]; // not an arithmetic progression
        let kscat = vec![0i64];
        let vbs = build_block_scatter(&vscat, 3);
        assert_eq!(vbs, vec![IRREGULAR]);
        let mut out = vec![0.0f64; panel_len::<f64>(3, 3, 1)];
        unsafe {
            pack_panel::<f64>(
                data.as_ptr(),
                &vscat,
                &vbs,
                &kscat,
                3,
                false,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(out, vec![0.0, 3.0, 1.0]);
    }
}
