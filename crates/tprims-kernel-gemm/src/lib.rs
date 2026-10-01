//! Adapter exposing the public microkernels of `gemm-f64`/`gemm-f32`
//! (sarah-ek/gemm, MIT) as tprims `CUpdate::Direct` kernel families.
//!
//! Calls only: no gemm source is copied, and `docs/provenance.md` records the
//! upstream. The microkernel module is undocumented upstream, so the versions
//! are pinned exactly and a bump has to be re-checked.
//!
//! The microkernel signature differs from [`DirectUkrFn`] only in argument
//! order and in the `alpha_status`/conjugation tail, so one context-free shim
//! per dtype adapts it: the real function pointer travels through
//! [`KernelFamily::opaque`], which the driver copies into [`UkrAux::opaque`]
//! for every call. A shim rather than a wrapper is what lets the family carry
//! the upstream pointer while staying a plain `fn` item.
use core::mem::size_of;

use gemm_common::microkernel::MicroKernelFn;
use tprims_gemm_kernel::{
    BAccess, Blocking, Blocksizes, CPref, CpuFeatures, Isa, KernelFamily, KernelImpl, Origin,
    UkrAux, UkrFn,
};

/// A `DirectUkrFn` that forwards to the real microkernel named by
/// `aux.opaque`, in gemm's argument order.
///
/// # Safety
/// `aux.opaque` must be the [`MicroKernelFn`] this family was built from, and
/// the panel/output extents must satisfy the family's declaration — the
/// driver's obligations, unchanged by the adapter.
macro_rules! shim {
    ($name:ident, $r:ty) => {
        /// # Safety
        /// As for the family's [`UkrFn::Direct`] arm.
        unsafe fn $name(
            m: usize,
            n: usize,
            k: usize,
            d: *mut $r,
            rs_d: isize,
            cs_d: isize,
            a: *const $r,
            a_cs: isize,
            b: *const $r,
            b_rs: isize,
            b_cs: isize,
            alpha_d: $r,
            beta_ab: $r,
            aux: &UkrAux<$r>,
        ) {
            // SAFETY: the family's `opaque` is the pointer its `ukr` was built
            // from — a process-constant microkernel of exactly this signature.
            let f: MicroKernelFn<$r> = unsafe { core::mem::transmute(aux.opaque) };
            // gemm's status: 0 skips reading dst, 1 says alpha is one, 2 is the
            // general case. It is what makes `beta == 0` free there too.
            let status: u8 = if alpha_d == 0.0 {
                0
            } else if alpha_d == 1.0 {
                1
            } else {
                2
            };
            // SAFETY: the caller's extent and stride obligations, mapped onto
            // gemm's argument order: (m, n, k, dst, lhs, rhs, dst_cs, dst_rs,
            // lhs_cs, rhs_rs, rhs_cs, alpha, beta, status, conj*, next_lhs).
            unsafe {
                f(
                    m, n, k, d, a, b, cs_d, rs_d, a_cs, b_rs, b_cs, alpha_d, beta_ab, status,
                    false, false, false, aux.a_next,
                )
            }
        }
    };
}

shim!(shim_f64, f64);
shim!(shim_f32, f32);

/// Build one ISA's families from its compiled microkernel table.
///
/// `n` is the module's SIMD lane count, which upstream keeps private; the
/// parity test compares every family against a naive product, so a wrong `n`
/// cannot pass unnoticed.
macro_rules! menu {
    ($fn_name:ident, $r:ty, $n:expr, $ukr:path, $mr_div_n:path, $nr:path,
     $label:literal, $isa:expr, $required:expr, $shim:path) => {
        fn $fn_name(out: &mut Vec<&'static KernelFamily<$r>>) {
            const N: usize = $n;
            let ukr = $ukr;
            let mr_div_n: usize = $mr_div_n;
            let nr: usize = $nr;
            for i in 0..mr_div_n {
                for j in 0..nr {
                    let f = ukr[i][j];
                    let mr = (i + 1) * N;
                    let nr = j + 1;
                    // The project default blocking, at the size the plan fixes.
                    let blk = Blocking::derive_at_depth(size_of::<$r>(), mr, nr, 256);
                    let mc = blk.mc.next_multiple_of(mr).max(mr);
                    let nc = blk.nc.next_multiple_of(nr).max(nr);
                    let id = Box::leak(
                        format!("gemm.{}.{}.{}x{}", $label, stringify!($r), mr, nr)
                            .into_boxed_str(),
                    );
                    let family = KernelFamily::<$r> {
                        id,
                        origin: Origin::Gemm,
                        isa: $isa,
                        required: $required,
                        imp: KernelImpl::Optimized,
                        priority: 0,
                        complex: None,
                        mr,
                        nr,
                        a_per_k: mr,
                        b_per_k: nr,
                        tile_bound: mr * nr,
                        c_pref: CPref::Any,
                        ukr: UkrFn::Direct($shim),
                        b_access: BAccess::Direct {
                            unit_stride: tprims_gemm_kernel::Axis::Col,
                        },
                        c_update: tprims_gemm_kernel::CUpdate::Direct,
                        blocks: Blocksizes {
                            mc: (mc, mc),
                            kc: (256, 256),
                            nc: (nc, nc),
                        },
                        caps: tprims_gemm_kernel::Caps {
                            scatter_pack: true,
                            conj_a: false,
                            conj_b: false,
                        },
                        opaque: f as *const (),
                        inner: None,
                        allow_auto: false,
                    };
                    out.push(Box::leak(Box::new(family)));
                }
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
menu!(
    menu_f64_fma,
    f64,
    4,
    ::gemm_f64::microkernel::fma::f64::UKR,
    ::gemm_f64::microkernel::fma::f64::MR_DIV_N,
    ::gemm_f64::microkernel::fma::f64::NR,
    "avx2",
    Isa::Avx2,
    CpuFeatures {
        avx2: true,
        fma: true,
        ..CpuFeatures::NONE
    },
    shim_f64
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
menu!(
    menu_f32_fma,
    f32,
    8,
    ::gemm_f32::microkernel::fma::f32::UKR,
    ::gemm_f32::microkernel::fma::f32::MR_DIV_N,
    ::gemm_f32::microkernel::fma::f32::NR,
    "avx2",
    Isa::Avx2,
    CpuFeatures {
        avx2: true,
        fma: true,
        ..CpuFeatures::NONE
    },
    shim_f32
);

#[cfg(all(feature = "x86-v4", any(target_arch = "x86", target_arch = "x86_64")))]
menu!(
    menu_f64_avx512,
    f64,
    8,
    ::gemm_f64::microkernel::avx512f::f64::UKR,
    ::gemm_f64::microkernel::avx512f::f64::MR_DIV_N,
    ::gemm_f64::microkernel::avx512f::f64::NR,
    "avx512",
    Isa::Avx512,
    CpuFeatures {
        avx512f: true,
        ..CpuFeatures::NONE
    },
    shim_f64
);
#[cfg(all(feature = "x86-v4", any(target_arch = "x86", target_arch = "x86_64")))]
menu!(
    menu_f32_avx512,
    f32,
    16,
    ::gemm_f32::microkernel::avx512f::f32::UKR,
    ::gemm_f32::microkernel::avx512f::f32::MR_DIV_N,
    ::gemm_f32::microkernel::avx512f::f32::NR,
    "avx512",
    Isa::Avx512,
    CpuFeatures {
        avx512f: true,
        ..CpuFeatures::NONE
    },
    shim_f32
);

#[cfg(target_arch = "aarch64")]
menu!(
    menu_f64_neon,
    f64,
    2,
    ::gemm_f64::microkernel::neon::f64::UKR,
    ::gemm_f64::microkernel::neon::f64::MR_DIV_N,
    ::gemm_f64::microkernel::neon::f64::NR,
    "neon",
    Isa::Neon,
    CpuFeatures {
        neon: true,
        ..CpuFeatures::NONE
    },
    shim_f64
);
#[cfg(target_arch = "aarch64")]
menu!(
    menu_f32_neon,
    f32,
    4,
    ::gemm_f32::microkernel::neon::f32::UKR,
    ::gemm_f32::microkernel::neon::f32::MR_DIV_N,
    ::gemm_f32::microkernel::neon::f32::NR,
    "neon",
    Isa::Neon,
    CpuFeatures {
        neon: true,
        ..CpuFeatures::NONE
    },
    shim_f32
);

menu!(
    menu_f64_scalar,
    f64,
    1,
    ::gemm_f64::microkernel::scalar::f64::UKR,
    ::gemm_f64::microkernel::scalar::f64::MR_DIV_N,
    ::gemm_f64::microkernel::scalar::f64::NR,
    "scalar",
    Isa::Portable,
    CpuFeatures::NONE,
    shim_f64
);
menu!(
    menu_f32_scalar,
    f32,
    1,
    ::gemm_f32::microkernel::scalar::f32::UKR,
    ::gemm_f32::microkernel::scalar::f32::MR_DIV_N,
    ::gemm_f32::microkernel::scalar::f32::NR,
    "scalar",
    Isa::Portable,
    CpuFeatures::NONE,
    shim_f32
);

macro_rules! families {
    ($r:ty, $list:ident, $menus:path) => {
        /// The compiled microkernel menu of every ISA this build has, in
        /// descending preference order.
        ///
        /// # Examples
        /// ```
        /// let families = tprims_kernel_gemm::families_f64();
        /// assert!(families.iter().all(|f| f.validate().is_ok()));
        /// assert!(families.iter().any(|f| f.id.starts_with("gemm.")));
        /// ```
        pub fn $list() -> &'static [&'static KernelFamily<$r>] {
            static LIST: std::sync::OnceLock<Vec<&'static KernelFamily<$r>>> =
                std::sync::OnceLock::new();
            LIST.get_or_init(|| {
                let mut out: Vec<&'static KernelFamily<$r>> = Vec::new();
                $menus(&mut out);
                // The widest available table's largest tile becomes the head:
                // it is the only Auto-eligible entry, and everything else is
                // there to be chosen by id.
                let head = out
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, f)| (f.isa != Isa::Portable, f.mr * f.nr))
                    .map(|(i, _)| i);
                let mut rebuilt: Vec<&'static KernelFamily<$r>> = Vec::with_capacity(out.len());
                for (index, f) in out.into_iter().enumerate() {
                    let mut f = *f;
                    f.priority = if Some(index) == head {
                        f.allow_auto = true;
                        250
                    } else {
                        240u16.saturating_sub(u16::try_from(index).unwrap_or(u16::MAX))
                    };
                    rebuilt.push(Box::leak(Box::new(f)));
                }
                rebuilt
            })
        }
    };
}

/// Every menu this build compiles, in descending ISA preference order.
fn menus_f32(out: &mut Vec<&'static KernelFamily<f32>>) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    menu_f32_fma(out);
    #[cfg(all(feature = "x86-v4", any(target_arch = "x86", target_arch = "x86_64")))]
    menu_f32_avx512(out);
    #[cfg(target_arch = "aarch64")]
    menu_f32_neon(out);
    menu_f32_scalar(out);
}

/// Every menu this build compiles, in descending ISA preference order.
fn menus_f64(out: &mut Vec<&'static KernelFamily<f64>>) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    menu_f64_fma(out);
    #[cfg(all(feature = "x86-v4", any(target_arch = "x86", target_arch = "x86_64")))]
    menu_f64_avx512(out);
    #[cfg(target_arch = "aarch64")]
    menu_f64_neon(out);
    menu_f64_scalar(out);
}

families!(f32, families_f32, menus_f32);
families!(f64, families_f64, menus_f64);

/// Register both real slots once. Does not initialize any executor or
/// workspace.
///
/// # Examples
/// ```
/// tprims_kernel_gemm::register();
/// assert!(tprims_gemm_kernel::list_kernels::<f64>()
///     .iter()
///     .any(|f| f.id.starts_with("gemm.")));
/// ```
pub fn register() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        // SAFETY: every descriptor is derived from one compiled upstream
        // microkernel table, so its tile extent, panel stride and required ISA
        // are the upstream declarations; the adapter passes the same values
        // through, and the parity test checks the arithmetic against a naive
        // product for every entry.
        unsafe {
            tprims_gemm_kernel::register::<f32>(families_f32);
            tprims_gemm_kernel::register::<f64>(families_f64);
        }
    });
}
