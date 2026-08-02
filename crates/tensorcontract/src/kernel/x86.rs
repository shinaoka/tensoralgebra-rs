//! x86-64 vectorised micro-kernels.
//!
//! These honour exactly the same [`PackFormat`](super::PackFormat) /
//! [`TileFormat`](super::TileFormat) contract as
//! [`scalar`](super::scalar), which is what keeps the three complex methods
//! interchangeable: the driver, the packing traversal and the write-back do not
//! know which kernel they are running.
//!
//! # Register blocking
//!
//! Write `MV` for the number of vector registers an `A` sliver occupies per
//! plane per k-step, `L` for the lane count (8 for AVX-512 `f64`, 16 for `f32`)
//! and `NR` for the column block. Then per logical k-step a kernel issues
//!
//! | | accumulator registers | load-port uops | FMA uops |
//! |---|---|---|---|
//! | real | `MV*NR` | `MV + NR` | `MV*NR` |
//! | planar | `2*MV*NR` | `2*MV + 2*NR` | `4*MV*NR` |
//! | 1m | `MV*NR` (real `2*MR x NR`) | `2*(MV + NR)` | `2*MV*NR` |
//! | 3m | `3*MV*NR` | `3*MV + 3*NR` | `3*MV*NR` |
//!
//! Skylake-SP retires two 512-bit FMAs and two loads per cycle, and LLVM emits
//! `vbroadcastsd` as its own uop rather than folding it into an FMA memory
//! operand (checked in the disassembly), so a kernel is issue-bound on FMAs
//! exactly when `FMA > loads`. Two things then decide the shape, and they pull
//! in opposite directions:
//!
//! 1. **The register file.** Accumulators plus one A plane plus the live
//!    broadcasts must fit in 32 zmm. Every candidate that does not spills and
//!    loses 30–50% — the sweep is full of these cliffs, and they are the reason
//!    3m cannot simply be given a big block.
//! 2. **Bytes per useful flop.** Once a shape is issue-bound the A sliver is a
//!    stream out of L2, not an L1 resident, and the packed footprint per flop
//!    becomes the binding constraint.
//!
//! The second point is the measured story of Phase 3, and it is not the one
//! the flop counts predict:
//!
//! * **planar** has the lowest bytes-per-flop of the three (two reals per
//!   complex element in *both* panels, and four FMAs from every (A-vector,
//!   B-scalar) pair with no in-register shuffling). It wins.
//! * **1m** issues the same FMAs but spread over two real k-steps and reads a
//!   packed `A` panel twice the size — 1.5x planar's bytes per flop.
//! * **3m** does 25% *fewer* FMAs and still loses, because it loads three
//!   planes of both operands to do it. Sweeping `kc` shows this directly: with
//!   both panels L1-resident (`kc = 16..64`) 3m is the fastest of the three, by
//!   roughly the margin its flop saving predicts; at the `kc` the engine
//!   actually uses it is no longer FMA-bound and the saving evaporates.
//!
//! Shapes were chosen by measuring, not by the reasoning above:
//! `cargo run --release -p tensorcontract --example kernel_shapes`, raw output
//! in `bench-results/phase3-kernel-shapes.txt`.

#![allow(clippy::missing_safety_doc)]

use super::{Blocking, ComplexMethod, KernelConfig, PackFormat, TileFormat, Ukr};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Generate the four AVX-512 kernels for one real type.
///
/// The bodies are written once, over `MV`/`NR` const generics, so that a shape
/// change is a one-line edit and every method is expressed in the same style —
/// a comparison between methods should not also be a comparison between two
/// people's hand-written assembly.
macro_rules! avx512_kernels {
    (
        $modname:ident, $t:ty, $v:ty, $lanes:expr,
        $zero:ident, $load:ident, $set1:ident, $fmadd:ident, $fnmadd:ident, $store:ident
    ) => {
        pub mod $modname {
            use super::*;

            /// Lanes per vector register.
            pub const L: usize = $lanes;

            /// `ab[j*MR + i] = sum_p a[p*MR + i] * b[p*NR + j]`, `MR = MV*L`.
            ///
            /// Also the 1m kernel: see [`onem`].
            #[target_feature(enable = "avx512f")]
            pub unsafe fn real<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                let mr = MV * L;
                let mut acc = [[$zero(); MV]; NR];
                for p in 0..kc {
                    let ap = a.add(p * mr);
                    let bp = b.add(p * NR);
                    let mut av = [$zero(); MV];
                    for u in 0..MV {
                        av[u] = $load(ap.add(u * L));
                    }
                    for j in 0..NR {
                        let bv = $set1(*bp.add(j));
                        for u in 0..MV {
                            acc[j][u] = $fmadd(av[u], bv, acc[j][u]);
                        }
                    }
                }
                for j in 0..NR {
                    for u in 0..MV {
                        $store(ab.add(j * mr + u * L), acc[j][u]);
                    }
                }
            }

            /// Planar / split-complex kernel over `MR = MV*L` complex rows.
            ///
            /// Both panels arrive as `[re plane | im plane]` per k-step, and the
            /// tile leaves as `[re plane | im plane]`, each `MR x NR`
            /// column-major. No shuffles anywhere: the four real products of a
            /// complex FMA are four `vfmadd`/`vfnmadd` on data that is already
            /// in the right lanes.
            #[target_feature(enable = "avx512f")]
            pub unsafe fn planar<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                let mr = MV * L;
                let mut cr = [[$zero(); MV]; NR];
                let mut ci = [[$zero(); MV]; NR];
                for p in 0..kc {
                    let are = a.add(p * 2 * mr);
                    let aim = are.add(mr);
                    let bre = b.add(p * 2 * NR);
                    let bim = bre.add(NR);
                    let mut ar = [$zero(); MV];
                    let mut ai = [$zero(); MV];
                    for u in 0..MV {
                        ar[u] = $load(are.add(u * L));
                        ai[u] = $load(aim.add(u * L));
                    }
                    for j in 0..NR {
                        let br = $set1(*bre.add(j));
                        let bi = $set1(*bim.add(j));
                        for u in 0..MV {
                            cr[j][u] = $fmadd(ar[u], br, cr[j][u]);
                            cr[j][u] = $fnmadd(ai[u], bi, cr[j][u]);
                            ci[j][u] = $fmadd(ar[u], bi, ci[j][u]);
                            ci[j][u] = $fmadd(ai[u], br, ci[j][u]);
                        }
                    }
                }
                let plane = mr * NR;
                for j in 0..NR {
                    for u in 0..MV {
                        $store(ab.add(j * mr + u * L), cr[j][u]);
                        $store(ab.add(plane + j * mr + u * L), ci[j][u]);
                    }
                }
            }

            /// Van Zee's 1m: one *real* kernel of shape `2*MR x NR` over `2*kc`
            /// real steps, fed by "1e" packed `A` and "1r" packed `B`.
            ///
            /// `MV` counts the vector registers of the *real* row block, so the
            /// complex micro-tile is `MV*L/2 x NR`.
            #[target_feature(enable = "avx512f")]
            pub unsafe fn onem<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                real::<MV, NR>(2 * kc, a, b, ab)
            }

            /// Karatsuba 3m over `MR = MV*L` complex rows.
            ///
            /// Accumulates `M1 = Ar*Br`, `M2 = Ai*Bi`, `M3 = (Ar+Ai)*(Br+Bi)`
            /// one plane at a time, so only `MV` A-registers are live at once
            /// and the three accumulator planes fit alongside them.
            #[target_feature(enable = "avx512f")]
            pub unsafe fn threem<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                let mr = MV * L;
                let mut m1 = [[$zero(); MV]; NR];
                let mut m2 = [[$zero(); MV]; NR];
                let mut m3 = [[$zero(); MV]; NR];
                for p in 0..kc {
                    let ao = a.add(p * 3 * mr);
                    let bo = b.add(p * 3 * NR);

                    let mut av = [$zero(); MV];
                    for u in 0..MV {
                        av[u] = $load(ao.add(u * L));
                    }
                    for j in 0..NR {
                        let bv = $set1(*bo.add(j));
                        for u in 0..MV {
                            m1[j][u] = $fmadd(av[u], bv, m1[j][u]);
                        }
                    }
                    for u in 0..MV {
                        av[u] = $load(ao.add(mr + u * L));
                    }
                    for j in 0..NR {
                        let bv = $set1(*bo.add(NR + j));
                        for u in 0..MV {
                            m2[j][u] = $fmadd(av[u], bv, m2[j][u]);
                        }
                    }
                    for u in 0..MV {
                        av[u] = $load(ao.add(2 * mr + u * L));
                    }
                    for j in 0..NR {
                        let bv = $set1(*bo.add(2 * NR + j));
                        for u in 0..MV {
                            m3[j][u] = $fmadd(av[u], bv, m3[j][u]);
                        }
                    }
                }
                let plane = mr * NR;
                for j in 0..NR {
                    for u in 0..MV {
                        $store(ab.add(j * mr + u * L), m1[j][u]);
                        $store(ab.add(plane + j * mr + u * L), m2[j][u]);
                        $store(ab.add(2 * plane + j * mr + u * L), m3[j][u]);
                    }
                }
            }

            // ---- plain-fn trampolines ---------------------------------------
            //
            // A `#[target_feature]` function cannot be coerced to a function
            // pointer, so each kernel gets a one-line wrapper. The wrapper is
            // not inlined into the caller (that is the point — the feature set
            // differs), which costs one `call` per micro-tile against
            // `kc * MR * NR` FMAs of work.

            pub unsafe fn tramp_real<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                real::<MV, NR>(kc, a, b, ab)
            }
            pub unsafe fn tramp_planar<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                planar::<MV, NR>(kc, a, b, ab)
            }
            pub unsafe fn tramp_onem<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                onem::<MV, NR>(kc, a, b, ab)
            }
            pub unsafe fn tramp_threem<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                threem::<MV, NR>(kc, a, b, ab)
            }
        }
    };
}

avx512_kernels!(
    avx512_f64,
    f64,
    __m512d,
    8,
    _mm512_setzero_pd,
    _mm512_loadu_pd,
    _mm512_set1_pd,
    _mm512_fmadd_pd,
    _mm512_fnmadd_pd,
    _mm512_storeu_pd
);

avx512_kernels!(
    avx512_f32,
    f32,
    __m512,
    16,
    _mm512_setzero_ps,
    _mm512_loadu_ps,
    _mm512_set1_ps,
    _mm512_fmadd_ps,
    _mm512_fnmadd_ps,
    _mm512_storeu_ps
);

// ---------------------------------------------------------------------------
// Configuration builders
// ---------------------------------------------------------------------------

/// Assemble the [`Ukr`] descriptors for one instruction set / type from a
/// *menu* of register blocks per method, the default first.
///
/// Every field is derived from `MV`/`NR` and the lane count, so a shape change
/// cannot desynchronise the descriptor from the kernel it describes — the
/// `panel_sizes_are_self_consistent` test in `kernel::tests` checks the
/// arithmetic, and `kernels_match_reference_*` checks the semantics.
///
/// # Why a menu rather than one shape
///
/// The first entry is the shape measured fastest in isolation and is what runs
/// unless something asks for otherwise. The alternates exist because peak
/// kernel throughput is not the only thing `MR` decides: it also sets the
/// granularity at which the output's row scatter is blocked, and therefore
/// whether the write-back takes its unit-stride path or its gather path. A
/// shape 10–20% off peak that moves whole block families onto the fast path
/// wins on any contraction that is nowhere near kernel-bound. See
/// [`super::row_block_for`].
macro_rules! configs {
    ($t:ty, $m:ident, $isa:literal,
     real   = [$(($rmv:literal, $rnr:literal)),+ $(,)?],
     planar = [$(($pmv:literal, $pnr:literal)),+ $(,)?],
     onem   = [$(($omv:literal, $onr:literal)),+ $(,)?],
     threem = [$(($tmv:literal, $tnr:literal)),+ $(,)?] $(,)?) => {
        /// Logical row blocks with a real kernel, default first.
        pub const REAL_ROW_BLOCKS: &[usize] = &[$($rmv * $m::L),+];
        /// Logical (complex) row blocks with a planar kernel, default first.
        pub const PLANAR_ROW_BLOCKS: &[usize] = &[$($pmv * $m::L),+];
        /// Ditto for 1m. The complex tile is half the real row block.
        pub const ONEM_ROW_BLOCKS: &[usize] = &[$($omv * $m::L / 2),+];
        /// Ditto for 3m.
        pub const THREEM_ROW_BLOCKS: &[usize] = &[$($tmv * $m::L),+];

        fn real_cfg<const MV: usize, const NR: usize>() -> KernelConfig<$t> {
            let mr = MV * $m::L;
            KernelConfig {
                ukr: Ukr {
                    mr,
                    nr: NR,
                    a_per_k: mr,
                    b_per_k: NR,
                    tile: mr * NR,
                    a_pack: PackFormat::Real,
                    b_pack: PackFormat::Real,
                    tile_fmt: TileFormat::Real,
                    func: $m::tramp_real::<MV, NR>,
                    name: concat!($isa, "-real"),
                },
                blk: Blocking::derive(core::mem::size_of::<$t>(), 1, 1),
            }
        }

        fn planar_cfg<const MV: usize, const NR: usize>() -> KernelConfig<$t> {
            let mr = MV * $m::L;
            KernelConfig {
                ukr: Ukr {
                    mr,
                    nr: NR,
                    a_per_k: 2 * mr,
                    b_per_k: 2 * NR,
                    tile: 2 * mr * NR,
                    a_pack: PackFormat::Planar,
                    b_pack: PackFormat::Planar,
                    tile_fmt: TileFormat::Planar,
                    func: $m::tramp_planar::<MV, NR>,
                    name: concat!($isa, "-planar"),
                },
                blk: Blocking::derive(core::mem::size_of::<$t>(), 2, 2),
            }
        }

        /// The real kernel is `MR2 x NR`; the complex tile is half as tall.
        /// `MR2` is even for any `MV >= 1` since `L` is a power of two.
        fn onem_cfg<const MV: usize, const NR: usize>() -> KernelConfig<$t> {
            let mr2 = MV * $m::L;
            KernelConfig {
                ukr: Ukr {
                    mr: mr2 / 2,
                    nr: NR,
                    a_per_k: 2 * mr2,
                    b_per_k: 2 * NR,
                    tile: mr2 * NR,
                    a_pack: PackFormat::OneE,
                    b_pack: PackFormat::Planar,
                    tile_fmt: TileFormat::OneM,
                    func: $m::tramp_onem::<MV, NR>,
                    name: concat!($isa, "-1m"),
                },
                // Four reals per complex element in packed A, so the same L2
                // budget buys half the rows planar gets.
                blk: Blocking::derive(core::mem::size_of::<$t>(), 4, 2),
            }
        }

        fn threem_cfg<const MV: usize, const NR: usize>() -> KernelConfig<$t> {
            let mr = MV * $m::L;
            KernelConfig {
                ukr: Ukr {
                    mr,
                    nr: NR,
                    a_per_k: 3 * mr,
                    b_per_k: 3 * NR,
                    tile: 3 * mr * NR,
                    a_pack: PackFormat::ThreeM,
                    b_pack: PackFormat::ThreeM,
                    tile_fmt: TileFormat::ThreeM,
                    func: $m::tramp_threem::<MV, NR>,
                    name: concat!($isa, "-3m"),
                },
                blk: Blocking::derive(core::mem::size_of::<$t>(), 3, 3),
            }
        }

        /// The real kernel at logical row block `mr`, or `None` if the menu
        /// has no shape of that height.
        pub fn real_config_at(mr: usize) -> Option<KernelConfig<$t>> {
            $( if mr == $rmv * $m::L { return Some(real_cfg::<$rmv, $rnr>()); } )+
            None
        }

        /// The complex kernel for `method` at logical row block `mr`.
        pub fn cplx_config_at(method: ComplexMethod, mr: usize) -> Option<KernelConfig<$t>> {
            match method {
                ComplexMethod::Planar => {
                    $( if mr == $pmv * $m::L { return Some(planar_cfg::<$pmv, $pnr>()); } )+
                }
                ComplexMethod::OneM => {
                    $( if mr == $omv * $m::L / 2 { return Some(onem_cfg::<$omv, $onr>()); } )+
                }
                ComplexMethod::ThreeM => {
                    $( if mr == $tmv * $m::L { return Some(threem_cfg::<$tmv, $tnr>()); } )+
                }
            }
            None
        }

        pub fn cplx_row_blocks(method: ComplexMethod) -> &'static [usize] {
            match method {
                ComplexMethod::Planar => PLANAR_ROW_BLOCKS,
                ComplexMethod::OneM => ONEM_ROW_BLOCKS,
                ComplexMethod::ThreeM => THREEM_ROW_BLOCKS,
            }
        }

        pub fn real_config() -> KernelConfig<$t> {
            real_config_at(REAL_ROW_BLOCKS[0]).expect("default shape is on the menu")
        }

        pub fn cplx_config(method: ComplexMethod) -> KernelConfig<$t> {
            cplx_config_at(method, cplx_row_blocks(method)[0])
                .expect("default shape is on the menu")
        }
    };
}

/// AVX-512 shapes for `f64` / `c64` (`L = 8`), measured at the `kc = 256` that
/// [`Blocking::derive`] picks for 8-byte reals.
///
/// | kernel | `MV x NR` | `MR x NR` (logical) | acc | load | FMA | bytes/flop | GF/s |
/// |---|---|---|---|---|---|---|---|
/// | real | `3 x 8` | `24 x 8` | 24 | 11 | 24 | 0.67 | 88.4 |
/// | planar | `2 x 6` | `16 x 6` | 24 | 16 | 48 | **0.46** | **102.8** |
/// | 1m | `3 x 8` | `12 x 8` | 24 | 22 | 48 | 0.67 | 91.1 |
/// | 3m | `1 x 10` | `8 x 10` | 30 | 33 | 30 | 0.68 | 87.8 |
///
/// Note the shapes are *not* interchangeable between methods. Every candidate
/// needing more than 32 live vector registers — accumulators plus one A plane
/// plus the broadcasts — falls off a cliff of 30–50%, which is what rules out
/// e.g. `3m 24 x 4` (36 accumulators). Within the survivors the ranking follows
/// bytes-per-useful-flop, not FMA count: see the module docs.
///
/// The alternates on each menu, and what they cost at the operating `kc`
/// (fraction of the default's throughput, from the Phase 4.1c re-run of
/// `examples/kernel_shapes`):
///
/// | method | menu, `MR` (`NR`) | cost of each alternate |
/// |---|---|---|
/// | real | 24 (8), 16 (8), 8 (8) | 0.82, 0.79 |
/// | planar | 16 (6), 24 (3), 8 (8) | 1.02, 0.93 |
/// | 1m | 12 (8), 16 (6), 8 (8) | 1.02, 0.86 |
/// | 3m | 8 (10), 16 (4), 24 (3) | 0.89, 0.87 |
///
/// Those within a couple of percent of 1.00 are ties at the sweep's own
/// repeatability, not free lunches; the menu is ordered by the Phase 3 choice,
/// which is not re-litigated here.
pub mod cfg_avx512_f64 {
    use super::*;
    configs!(
        f64, avx512_f64, "avx512",
        real   = [(3, 8), (2, 8), (1, 8)],
        planar = [(2, 6), (3, 3), (1, 8)],
        onem   = [(3, 8), (4, 6), (2, 8)],
        threem = [(1, 10), (2, 4), (3, 3)],
    );
}

/// AVX-512 shapes for `f32` / `c32` (`L = 16`), measured at `kc = 384`.
///
/// | kernel | `MV x NR` | `MR x NR` (logical) | bytes/flop | GF/s |
/// |---|---|---|---|---|
/// | real | `3 x 8` | `48 x 8` | 0.29 | 179.9 |
/// | planar | `2 x 6` | `32 x 6` | **0.20** | 195.7 |
/// | 1m | `4 x 6` | `32 x 6` | 0.37 | 175.4 |
/// | 3m | `1 x 10` | `16 x 10` | 0.24 | 194.9 |
///
/// 1m prefers a wider real row block here than it does in `f64` because the
/// doubled lane count already makes its "1e" panel large; the other three
/// methods land on the same `MV x NR` in both precisions.
///
/// The alternates and their cost at the operating `kc`, as for `f64`:
///
/// | method | menu, `MR` (`NR`) | cost of each alternate |
/// |---|---|---|
/// | real | 48 (8), 32 (8), 16 (10) | 0.87, 1.02 |
/// | planar | 32 (6), 48 (4), 16 (12) | 0.95, 0.90 |
/// | 1m | 32 (6), 24 (8), 16 (8), 8 (12) | 1.03, 0.86, 0.86 |
/// | 3m | 16 (10), 32 (4), 48 (3) | 0.91, 0.84 |
///
/// The 32-bit menus are the ones that matter for the write-back: the corpus
/// rounds every stride-1 index up to a multiple of **24**, and at `L = 16` no
/// full-width `MR` divides 24 except 1m's, so the others can only reduce the
/// straddling fraction rather than eliminate it. `real 16 (10)` costing nothing
/// measurable is the important entry — it is the shape the `f32` cases still on
/// the gather path would need.
pub mod cfg_avx512_f32 {
    use super::*;
    configs!(
        f32, avx512_f32, "avx512",
        real   = [(3, 8), (2, 8), (1, 10)],
        planar = [(2, 6), (3, 4), (1, 12)],
        onem   = [(4, 6), (3, 8), (2, 8), (1, 12)],
        threem = [(1, 10), (2, 4), (3, 3)],
    );
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

/// Whether AVX-512F is usable. Checked once per process.
fn have_avx512() -> bool {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static YES: OnceLock<bool> = OnceLock::new();
        *YES.get_or_init(|| is_x86_feature_detected!("avx512f"))
    }
    #[cfg(not(feature = "std"))]
    {
        // No runtime detection without `std`; take it only if the whole crate
        // was compiled for it.
        cfg!(target_feature = "avx512f")
    }
}

/// The four entry points the [`super::KernelSet`] impls call, per type: the
/// default shape, the menu of row blocks, and the config at a chosen one. Each
/// yields `None` when the CPU has no AVX-512, which sends the caller to the
/// portable scalar path.
macro_rules! dispatch {
    ($t:ty, $cfg:ident, $real:ident, $cplx:ident, $rows:ident, $real_at:ident, $cplx_at:ident) => {
        pub fn $real() -> Option<KernelConfig<$t>> {
            have_avx512().then($cfg::real_config)
        }

        pub fn $cplx(method: ComplexMethod) -> Option<KernelConfig<$t>> {
            have_avx512().then(|| $cfg::cplx_config(method))
        }

        /// Row blocks with a kernel, default first; empty when unavailable.
        pub fn $rows(complex: bool, method: ComplexMethod) -> &'static [usize] {
            if !have_avx512() {
                return &[];
            }
            if complex {
                $cfg::cplx_row_blocks(method)
            } else {
                $cfg::REAL_ROW_BLOCKS
            }
        }

        pub fn $real_at(mr: usize) -> Option<KernelConfig<$t>> {
            have_avx512().then(|| $cfg::real_config_at(mr)).flatten()
        }

        pub fn $cplx_at(method: ComplexMethod, mr: usize) -> Option<KernelConfig<$t>> {
            have_avx512()
                .then(|| $cfg::cplx_config_at(method, mr))
                .flatten()
        }
    };
}

dispatch!(
    f64,
    cfg_avx512_f64,
    config_real_f64,
    config_cplx_f64,
    row_blocks_f64,
    config_real_f64_at,
    config_cplx_f64_at
);
dispatch!(
    f32,
    cfg_avx512_f32,
    config_real_f32,
    config_cplx_f32,
    row_blocks_f32,
    config_real_f32_at,
    config_cplx_f32_at
);
