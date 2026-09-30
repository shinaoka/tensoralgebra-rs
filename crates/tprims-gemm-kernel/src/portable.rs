//! Project-owned portable reference kernels (MIT OR Apache-2.0), not a port.
//! Ordinary dot-product loops, without intrinsics. Family/context separation
//! follows BLIS (Van Zee & van de Geijn, TOMS 2015).

use crate::*;

/// Overwrite a column-major real MR×NR tile with packed A·B.
///
/// # Examples
/// ```
/// let a = [1.0_f64; 8]; let b = [1.0_f64; 8]; let mut tile = [f64::NAN; 16];
/// // SAFETY: two full 4-lane slivers and a distinct 4x4 output tile.
/// unsafe { tprims_gemm_kernel::portable::real_tile::<f64, 4, 4>(
///     2, a.as_ptr(), b.as_ptr(), tile.as_mut_ptr()) };
/// assert_eq!(tile, [2.0; 16]);
/// ```
///
/// # Safety
/// MR/NR are positive; A/B cover `kc*MR`/`kc*NR` readable reals and tile
/// covers `MR*NR` writable reals. Output does not alias either input.
pub unsafe fn real_tile<R: Real, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const R,
    b: *const R,
    tile: *mut R,
) {
    let mut acc = [[R::ZERO; MR]; NR];
    // INVARIANT: the caller supplies complete packed slivers and the full
    // writable tile, including padded lanes; accumulation needs zero starts.
    // SAFETY: the function's panel/tile extent and non-aliasing contract.
    unsafe {
        for p in 0..kc {
            let ap = a.add(p * MR);
            let bp = b.add(p * NR);
            for (j, col) in acc.iter_mut().enumerate() {
                let bj = *bp.add(j);
                for (i, sum) in col.iter_mut().enumerate() {
                    *sum += *ap.add(i) * bj;
                }
            }
        }
        for (j, col) in acc.iter().enumerate() {
            for (i, &value) in col.iter().enumerate() {
                *tile.add(j * MR + i) = value;
            }
        }
    }
}

/// Native complex product of interleaved panels into an interleaved tile.
///
/// # Examples
/// ```
/// let a = [1.0_f64; 8]; // four values 1+i
/// let b = [0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]; // four values i
/// let mut tile = [f64::NAN; 32];
/// // SAFETY: full 4-lane complex panels and a distinct 4x4 complex tile.
/// unsafe { tprims_gemm_kernel::portable::cplx_tile::<f64, 4, 4>(
///     1, a.as_ptr(), b.as_ptr(), tile.as_mut_ptr()) };
/// assert!(tile.chunks_exact(2).all(|z| z == [-1.0, 1.0]));
/// ```
///
/// # Safety
/// MR/NR are positive. A/B cover `2*kc*MR`/`2*kc*NR` readable reals and
/// tile covers `2*MR*NR` writable reals, without aliasing either input.
pub unsafe fn cplx_tile<R: Real, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const R,
    b: *const R,
    tile: *mut R,
) {
    let mut re = [[R::ZERO; MR]; NR];
    let mut im = [[R::ZERO; MR]; NR];
    // INVARIANT: full packed slivers and tile are covered by the caller;
    // only the accumulators are zero-initialized, not the output buffer.
    // SAFETY: the function's extent and non-aliasing contract.
    unsafe {
        for p in 0..kc {
            let ap = a.add(p * 2 * MR);
            let bp = b.add(p * 2 * NR);
            for (j, (rcol, icol)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
                let (br, bi) = (*bp.add(2 * j), *bp.add(2 * j + 1));
                for (i, (r, z)) in rcol.iter_mut().zip(icol.iter_mut()).enumerate() {
                    let (ar, ai) = (*ap.add(2 * i), *ap.add(2 * i + 1));
                    *r += ar * br - ai * bi;
                    *z += ar * bi + ai * br;
                }
            }
        }
        for (j, (rcol, icol)) in re.iter().zip(im.iter()).enumerate() {
            for (i, (&r, &z)) in rcol.iter().zip(icol.iter()).enumerate() {
                *tile.add(2 * (j * MR + i)) = r;
                *tile.add(2 * (j * MR + i) + 1) = z;
            }
        }
    }
}

macro_rules! families {
    ($r:ty, $real:ident, $cplx:ident, $list:ident, $real_id:literal, $cplx_id:literal) => {
        static $real: KernelFamily<$r> = KernelFamily {
            id: $real_id,
            origin: Origin::Portable,
            isa: Isa::Portable,
            required: CpuFeatures::NONE,
            imp: KernelImpl::Reference,
            priority: 0,
            complex: None,
            mr: 4,
            nr: 4,
            a_per_k: 4,
            b_per_k: 4,
            tile_bound: 16,
            c_pref: CPref::Any,
            ukr: UkrFn::Tile(real_tile::<$r, 4, 4>),
            b_access: BAccess::Packed,
            c_update: CUpdate::ScratchTile,
            blocks: Blocksizes {
                mc: (64, 64),
                kc: (256, 256),
                nc: (1024, 1024),
            },
            caps: Caps {
                scatter_pack: true,
                conj_a: true,
                conj_b: true,
            },
            allow_auto: true,
        };
        static $cplx: KernelFamily<$r> = KernelFamily {
            id: $cplx_id,
            complex: Some(ComplexScheme {
                method: Method::Native,
                a: Layout::Interleaved,
                b: Layout::Interleaved,
                tile: TileFormat::Interleaved,
            }),
            a_per_k: 8,
            b_per_k: 8,
            tile_bound: 32,
            ukr: UkrFn::Tile(cplx_tile::<$r, 4, 4>),
            ..$real
        };
        /// Static real and native-interleaved complex fallback descriptors.
        #[doc = concat!("\n# Examples\n```\nlet families = tprims_gemm_kernel::portable::",
                                    stringify!($list), "();\nassert_eq!(families.len(), 2);\n",
                                    "assert!(families.iter().all(|f| f.validate().is_ok()));\n```")]
        pub fn $list() -> &'static [&'static KernelFamily<$r>] {
            static LIST: [&KernelFamily<$r>; 2] = [&$real, &$cplx];
            &LIST
        }
    };
}
families!(
    f32,
    REAL32,
    CPLX32,
    families_f32,
    "portable.f32.4x4",
    "portable.c32.native.4x4"
);
families!(
    f64,
    REAL64,
    CPLX64,
    families_f64,
    "portable.f64.4x4",
    "portable.c64.native.4x4"
);
