//! Portable scalar micro-kernels.
//!
//! These are the reference semantics for every vectorised kernel, and the path
//! by which an arbitrary [`Real`] type (extended precision, dual numbers for
//! forward-mode AD, ...) gets a working contraction engine.
//!
//! They are written so that LLVM can auto-vectorise the `f32`/`f64`
//! instantiations reasonably well, but they are not the fast path.

use super::{Blocking, KernelConfig, Ukr};
use crate::element::Real;

/// `ab[j * MR + i] = sum_p a[p * MR + i] * b[p * NR + j]`
///
/// # Safety
/// `a` must be valid for `MR * kc` reads, `b` for `NR * kc`, `ab` for
/// `MR * NR` writes.
pub unsafe fn real_ukr<T: Real, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const T,
    b: *const T,
    ab: *mut T,
) {
    let mut acc = [[T::ZERO; MR]; NR];
    for p in 0..kc {
        let ap = a.add(p * MR);
        let bp = b.add(p * NR);
        for (j, accj) in acc.iter_mut().enumerate() {
            let bv = *bp.add(j);
            for (i, s) in accj.iter_mut().enumerate() {
                *s += *ap.add(i) * bv;
            }
        }
    }
    for (j, accj) in acc.iter().enumerate() {
        for (i, &v) in accj.iter().enumerate() {
            *ab.add(j * MR + i) = v;
        }
    }
}

/// Planar complex kernel.
///
/// `a` holds `[re_0..re_{MR-1}, im_0..im_{MR-1}]` per k-step, `b` likewise
/// with `NR`. Writes the real plane to `ab[0 .. MR*NR]` and the imaginary
/// plane to `ab[MR*NR .. 2*MR*NR]`, both column-major within the tile.
///
/// # Safety
/// `a` valid for `2 * MR * kc` reads, `b` for `2 * NR * kc`, `ab` for
/// `2 * MR * NR` writes.
pub unsafe fn cplx_ukr<T: Real, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const T,
    b: *const T,
    ab: *mut T,
) {
    let mut acc_re = [[T::ZERO; MR]; NR];
    let mut acc_im = [[T::ZERO; MR]; NR];
    for p in 0..kc {
        let are = a.add(p * 2 * MR);
        let aim = are.add(MR);
        let bre = b.add(p * 2 * NR);
        let bim = bre.add(NR);
        for j in 0..NR {
            let br = *bre.add(j);
            let bi = *bim.add(j);
            for i in 0..MR {
                let ar = *are.add(i);
                let ai = *aim.add(i);
                acc_re[j][i] += ar * br;
                acc_re[j][i] += -(ai * bi);
                acc_im[j][i] += ar * bi;
                acc_im[j][i] += ai * br;
            }
        }
    }
    for j in 0..NR {
        for i in 0..MR {
            *ab.add(j * MR + i) = acc_re[j][i];
            *ab.add(MR * NR + j * MR + i) = acc_im[j][i];
        }
    }
}

/// Blocking parameters derived from cache sizes rather than measured; these
/// are a starting point that Phase 4 retunes.
fn default_blocking<T>(planes: usize) -> Blocking {
    let esz = core::mem::size_of::<T>() * planes;
    // Target: packed A block ~= half of a 1 MiB L2, packed B block ~= 3 MiB.
    let kc = if esz <= 4 { 384 } else { 256 };
    let mc = (512 * 1024 / (kc * esz)).max(8);
    let nc = (3 * 1024 * 1024 / (kc * esz)).max(8);
    Blocking { mc, kc, nc }
}

pub fn config_real<T: Real, const MR: usize, const NR: usize>() -> KernelConfig<T> {
    KernelConfig {
        ukr: Ukr {
            mr: MR,
            nr: NR,
            func: real_ukr::<T, MR, NR>,
            name: "scalar-real",
        },
        blk: default_blocking::<T>(1),
    }
}

pub fn config_cplx<T: Real, const MR: usize, const NR: usize>() -> KernelConfig<T> {
    KernelConfig {
        ukr: Ukr {
            mr: MR,
            nr: NR,
            func: cplx_ukr::<T, MR, NR>,
            name: "scalar-cplx",
        },
        blk: default_blocking::<T>(2),
    }
}
