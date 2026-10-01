//! Descriptor adapters for Lukas Devos's tensorcontract kernels.
//! Calls the existing configs/bodies (lkdvos/tensorprimitives-rs, MIT OR
//! Apache-2.0); no kernel arithmetic is duplicated or ported here.

use std::sync::{Once, OnceLock};
use tprims_gemm_kernel::*;

type List<R> = Vec<&'static KernelFamily<R>>;

fn append_config<R: RealSlot>(
    out: &mut List<R>,
    label: &str,
    isa: Isa,
    required: CpuFeatures,
    priority: u16,
    cfg: KernelConfig<R>,
) {
    // Configs are process-constant compiled menus; effective-width blocking
    // is resolved separately when a plan selects a descriptor.
    // Descriptor defaults are unscaled. Resolution applies environment/model
    // overrides once; aligning the finite compiled menu needs no env lookup.
    let cfg = cfg.with_blocking(cfg.blk);
    let u = cfg.ukr;
    let complex = match u.tile_fmt {
        TileFormat::Real => None,
        TileFormat::Planar => Some(ComplexScheme {
            method: Method::Native,
            a: Layout::Planar,
            b: Layout::Planar,
            tile: u.tile_fmt,
        }),
        TileFormat::OneM => Some(ComplexScheme {
            method: Method::OneM,
            a: Layout::OneE,
            b: Layout::OneR,
            tile: u.tile_fmt,
        }),
        TileFormat::ThreeM => Some(ComplexScheme {
            method: Method::ThreeM,
            a: Layout::ThreeM,
            b: Layout::ThreeM,
            tile: u.tile_fmt,
        }),
        // These formats are not offered by the existing tc menus.
        TileFormat::Interleaved => Some(ComplexScheme {
            method: Method::Native,
            a: Layout::Interleaved,
            b: Layout::Interleaved,
            tile: u.tile_fmt,
        }),
        TileFormat::FourM => Some(ComplexScheme {
            method: Method::FourM,
            a: Layout::Planar,
            b: Layout::Planar,
            tile: u.tile_fmt,
        }),
    };
    // INVARIANT: RealSlot is sealed to f32/f64; ids name the storage dtype.
    let dtype = match (core::mem::size_of::<R>(), complex.is_some()) {
        (4, false) => "f32",
        (4, true) => "c32",
        (_, false) => "f64",
        (_, true) => "c64",
    };
    let method = match u.tile_fmt {
        TileFormat::Real => "",
        TileFormat::Planar => ".planar",
        TileFormat::OneM => ".1m",
        TileFormat::ThreeM => ".3m",
        TileFormat::Interleaved => ".native",
        TileFormat::FourM => ".4m",
    };
    let f = KernelFamily {
        id: Box::leak(format!("tc.{label}.{dtype}{method}.{}x{}", u.mr, u.nr).into_boxed_str()),
        origin: Origin::Tensorcontract,
        isa,
        required,
        imp: if isa == Isa::Portable {
            KernelImpl::Reference
        } else {
            KernelImpl::Optimized
        },
        priority: if u.tile_fmt == TileFormat::OneM {
            priority.saturating_sub(50)
        } else {
            priority
        },
        complex,
        mr: u.mr,
        nr: u.nr,
        a_per_k: u.a_per_k,
        b_per_k: u.b_per_k,
        tile_bound: u.tile,
        c_pref: CPref::Col,
        ukr: UkrFn::Tile(u.func),
        b_access: BAccess::Packed,
        c_update: CUpdate::ScratchTile,
        blocks: Blocksizes {
            mc: (cfg.blk.mc, cfg.blk.mc),
            kc: (cfg.blk.kc, cfg.blk.kc),
            nc: (cfg.blk.nc, cfg.blk.nc),
        },
        caps: Caps {
            scatter_pack: true,
            conj_a: true,
            conj_b: true,
        },
        opaque: core::ptr::null(),
        inner: None,
        allow_auto: u.tile_fmt != TileFormat::ThreeM,
    };
    // INVARIANT: this immutable descriptor set is finite (compiled menu
    // entries only), initialized once and retained for the process lifetime.
    out.push(Box::leak(Box::new(f)));
}

fn scalar_families<R: RealSlot>() -> List<R> {
    let mut out = Vec::new();
    append_config(
        &mut out,
        "scalar",
        Isa::Portable,
        CpuFeatures::NONE,
        10,
        crate::scalar::config_real::<R, 4, 4>(),
    );
    for method in ComplexMethod::ALL {
        append_config(
            &mut out,
            "scalar",
            Isa::Portable,
            CpuFeatures::NONE,
            10,
            crate::scalar::config_cplx::<R, 4, 4>(method),
        );
    }
    out
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
fn append_menu<R: RealSlot>(
    out: &mut List<R>,
    label: &str,
    isa: Isa,
    required: CpuFeatures,
    priority: u16,
    menu: crate::simd_isa::IsaConfigs<R>,
) {
    for (complex, method) in [
        (false, ComplexMethod::Planar),
        (true, ComplexMethod::Planar),
        (true, ComplexMethod::OneM),
        (true, ComplexMethod::ThreeM),
    ] {
        for i in 0..(menu.row_blocks)(complex, method).len() {
            // INVARIANT: indices come from this compiled menu; existing
            // row_block_menus_are_well_formed tests cover config_at coverage.
            let cfg = (menu.config_at)(complex, method, i).expect("compiled menu has a config");
            append_config(
                out,
                label,
                isa,
                required,
                priority.saturating_sub(u16::try_from(i).unwrap_or(u16::MAX)),
                cfg,
            );
        }
    }
}

macro_rules! family_list {
    ($r:ty, $name:ident, $sets:ident) => {
        /// All compiled real/complex families, including CPU-unavailable ISAs.
        /// Descriptors are a finite process-constant manifest, not a data cache.
        #[doc = concat!("\n# Examples\n```\nlet families = tprims_kernel_tensorcontract::",
            stringify!($name), "();\nassert!(families.iter().all(|f| f.validate().is_ok()));\n",
            "assert!(families.iter().any(|f| f.id.starts_with(\"tc.scalar.\")));\n```")]
        pub fn $name() -> &'static [&'static KernelFamily<$r>] {
            static LIST: OnceLock<List<$r>> = OnceLock::new();
            LIST.get_or_init(|| {
                #[allow(unused_mut)] // INVARIANT: SIMD targets append to the scalar list below.
                let mut out = scalar_families::<$r>();
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    append_menu(
                        &mut out,
                        "avx512",
                        Isa::Avx512,
                        CpuFeatures {
                            avx512f: true,
                            ..CpuFeatures::NONE
                        },
                        300,
                        crate::simd_isa::$sets(crate::simd_isa::Isa::Avx512),
                    );
                    append_menu(
                        &mut out,
                        "avx2",
                        Isa::Avx2,
                        CpuFeatures {
                            avx2: true,
                            fma: true,
                            ..CpuFeatures::NONE
                        },
                        200,
                        crate::simd_isa::$sets(crate::simd_isa::Isa::Avx2),
                    );
                }
                #[cfg(target_arch = "aarch64")]
                append_menu(
                    &mut out,
                    "neon",
                    Isa::Neon,
                    CpuFeatures {
                        neon: true,
                        ..CpuFeatures::NONE
                    },
                    200,
                    crate::simd_isa::$sets(crate::simd_isa::Isa::Neon),
                );
                out
            })
        }
    };
}
family_list!(f32, families_f32, isa_configs_f32);
family_list!(f64, families_f64, isa_configs_f64);

/// Register both real slots once. Does not initialize any executor or workspace.
///
/// # Examples
/// ```
/// tprims_kernel_tensorcontract::register();
/// assert!(tprims_gemm_kernel::list_kernels::<f64>().iter().any(|f| f.id.starts_with("tc.")));
/// ```
pub fn register() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: immutable compiled menus derive each pointer, tile/panel
        // footprint and ISA from the same scalar/SIMD config definitions.
        // Every available entry is tested against a packed-product oracle.
        unsafe {
            tprims_gemm_kernel::register::<f32>(families_f32);
            tprims_gemm_kernel::register::<f64>(families_f64);
        }
    });
}
