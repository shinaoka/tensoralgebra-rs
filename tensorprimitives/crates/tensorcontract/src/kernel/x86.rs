//! x86-64 vectorised micro-kernels.
//!
//! **Not part of the public API.** This module is `#[doc(hidden)]` and outside
//! the crate's semver guarantee; it is public only so that
//! `examples/kernel_shapes` can measure one kernel family at a time. The
//! register-block menus below are re-measured whenever the reference machine
//! changes, so nothing here is stable. Reach the kernels through
//! [`super::KernelSet`].
//!
//! These honour exactly the same [`PackFormat`] / [`TileFormat`] contract as
//! [`scalar`](super::scalar), which is what keeps the three complex methods
//! interchangeable: the driver, the packing traversal and the write-back do not
//! know which kernel they are running.
//!
//! # Register blocking
//!
//! Write `MV` for the number of vector registers an `A` sliver occupies per
//! plane per k-step, `L` for the lane count (8 for AVX-512 `f64`, 16 for
//! AVX-512 `f32`; 4 and 8 for AVX2) and `NR` for the column block. Then per
//! logical k-step a kernel issues
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
//!    broadcasts must fit in the architectural register file — 32 zmm under
//!    AVX-512, **16 ymm under AVX2**. Every candidate that does not spills and
//!    loses 30–50% — the sweep is full of these cliffs, and they are the reason
//!    3m cannot simply be given a big block.
//! 2. **Bytes per useful flop.** Once a shape is issue-bound the A sliver is a
//!    stream out of L2, not an L1 resident, and the packed footprint per flop
//!    becomes the binding constraint.
//!
//! The second point is the measured story, and it is not the one the flop
//! counts predict:
//!
//! * **planar** has the lowest bytes-per-flop of the three (two reals per
//!   complex element in *both* panels, and four FMAs from every (A-vector,
//!   B-scalar) pair with no in-register shuffling). It wins the corpus on both
//!   AVX-512 machines measured.
//! * **1m** issues the same FMAs but spread over two real k-steps and reads a
//!   packed `A` panel twice the size — 1.5x planar's bytes per flop.
//! * **3m** does 25% *fewer* FMAs and still loses, because it loads three
//!   planes of both operands to do it.
//!
//! The accounting above — bytes moved per useful flop, not flop count — is
//! arithmetic and holds everywhere. **Whether 3m's flop saving ever *pays* is a
//! per-microarchitecture question with no settled answer.** This module used to
//! claim that it pays with both panels L1-resident; that is a Cascade Lake
//! result and it does not transfer. At `kc = 16`, which is that regime, 3m leads
//! planar by 1.105 (`c64`) and 1.155 (`c32`) on Cascade Lake and trails at 0.567
//! and 0.662 on Ice Lake, where it wins at no depth, no shape and neither
//! precision while already running its own Ice Lake-optimal block. Ranking
//! anything below planar without naming the machine is a mistake this project
//! has made twice.
//!
//! The **AVX-512** shapes were chosen by measuring, not by the reasoning above:
//! `cargo run --release -p tensorcontract --example kernel_shapes`, raw output
//! in `bench-results/phase3-kernel-shapes.txt`.
//!
//! The **AVX2** shapes are *provisional and unmeasured* — see
//! [`cfg_avx2_f64`] — because the reference machine has no AVX2-only CPU to
//! measure them on and the same `examples/kernel_shapes` run is the calibration
//! path once one is available. Halving the register file is not a small change:
//! with 16 ymm, planar's two accumulator planes and 3m's three leave almost no
//! choice of aspect ratio, so AVX2 register blocks are 2–6x smaller than their
//! AVX-512 counterparts and are much closer to the register bound.

use super::{Blocking, ComplexMethod, KernelConfig, PackFormat, TileFormat, Ukr};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

simd_kernels!(
    avx512_f64,
    f64,
    __m512d,
    8,
    ["avx512f"],
    _mm512_setzero_pd,
    _mm512_loadu_pd,
    _mm512_set1_pd,
    _mm512_fmadd_pd,
    _mm512_fnmadd_pd,
    _mm512_storeu_pd
);

simd_kernels!(
    avx512_f32,
    f32,
    __m512,
    16,
    ["avx512f"],
    _mm512_setzero_ps,
    _mm512_loadu_ps,
    _mm512_set1_ps,
    _mm512_fmadd_ps,
    _mm512_fnmadd_ps,
    _mm512_storeu_ps
);

// AVX2 needs `fma` as well as `avx2`: the 256-bit loads, broadcasts and stores
// come from AVX/AVX2 but `_mm256_fmadd_*` / `_mm256_fnmadd_*` are FMA3. Every
// CPU that has AVX2 has FMA3 in practice, but they are separate CPUID bits, so
// both are detected and both are enabled here.
simd_kernels!(
    avx2_f64,
    f64,
    __m256d,
    4,
    ["avx2", "fma"],
    _mm256_setzero_pd,
    _mm256_loadu_pd,
    _mm256_set1_pd,
    _mm256_fmadd_pd,
    _mm256_fnmadd_pd,
    _mm256_storeu_pd
);

simd_kernels!(
    avx2_f32,
    f32,
    __m256,
    8,
    ["avx2", "fma"],
    _mm256_setzero_ps,
    _mm256_loadu_ps,
    _mm256_set1_ps,
    _mm256_fmadd_ps,
    _mm256_fnmadd_ps,
    _mm256_storeu_ps
);

// ---------------------------------------------------------------------------
// Configuration builders
// ---------------------------------------------------------------------------

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
        f64,
        avx512_f64,
        "avx512",
        real = [(3, 8), (2, 8), (1, 8)],
        planar = [(2, 6), (3, 3), (1, 8)],
        onem = [(3, 8), (4, 6), (2, 8)],
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
/// | planar | 32 (6), 48 (4), 16 (12), **32 (5)** | 0.95, 0.90, **1.078** |
/// | 1m | 32 (6), 24 (8), 16 (8), 8 (12) | 1.03, 0.86, 0.86 |
/// | 3m | 16 (10), 32 (4), 48 (3) | 0.91, 0.84 |
///
/// **`planar 32 (5)` is faster than the default it sits behind**, which no other
/// entry on any menu is: 210.9 GF/s against 195.7 in the same Phase 3 sweep, and
/// ahead at `kc = 64` too. The default appears to have been chosen on the
/// bolded bytes-per-flop above rather than on the throughput in the same output
/// file (A35) — the failure mode this project has now recorded four times. It is
/// **last on the menu, not first**, because a kernel margin is not a corpus
/// margin: `NR` also sets the `jr` loop count and the packed-`B` sliver
/// geometry. `TENSORCONTRACT_ROWBLOCK=idx=3` is the arm that settles it.
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
        f32,
        avx512_f32,
        "avx512",
        real = [(3, 8), (2, 8), (1, 10)],
        // `(2, 5)` is the shape A35 found and could not reach: Phase 3's own
        // sweep names `32x5` at **210.9 GF/s** against the shipped `32x6`'s
        // **195.7**, 7.8% faster at the operating `kc` and also ahead at
        // `kc = 64`. It is **appended, not inserted**, on purpose — the entries
        // before it keep the positions `bench-results/phase4c` swept, so that
        // grid's `idx=` numbering still means what it meant.
        //
        // It cannot be *chosen by the rule*, which reads `MR` and sees a tie
        // with the default. That is deliberate: a 7.8% kernel margin is not a
        // 7.8% corpus margin, since `NR` also moves the `jr` loop count and the
        // packed-`B` sliver geometry. Making it reachable at `idx=3` is what
        // turns A35 from untestable into an A/B, which is the whole change.
        planar = [(2, 6), (3, 4), (1, 12), (2, 5)],
        onem = [(4, 6), (3, 8), (2, 8), (1, 12)],
        threem = [(1, 10), (2, 4), (3, 3)],
    );
}

/// AVX2 shapes for `f64` / `c64` (`L = 4`). **Provisional and unmeasured** — see
/// the note below on which half of the choice is modelled.
///
/// 16 ymm instead of 32 zmm, and the same accumulator-plane counts, so the
/// register bound bites much harder than it does on AVX-512. With `live =
/// accumulators + one A plane + the live broadcasts`:
///
/// | kernel | `MV x NR` | `MR x NR` (logical) | acc | live | load | FMA | f/l | bytes/flop |
/// |---|---|---|---|---|---|---|---|---|
/// | real | `2 x 6` | `8 x 6` | 12 | 15 | 8 | 12 | 1.50 | **1.17** |
/// | planar | `1 x 5` | `4 x 5` | 10 | 14 | 12 | 20 | 1.67 | **0.90** |
/// | 1m | `2 x 6` | `4 x 6` | 12 | 15 | 16 | 24 | 1.50 | 1.17 |
/// | 3m | `1 x 4` | `4 x 4` | 12 | 14 | 15 | 12 | 0.80 | 1.50 |
///
/// # What is modelled and what is guessed
///
/// **Modelled, and trustworthy** (D19: the uop model gets the *cliffs* right):
///
/// * `live <= 16` for every shape on every menu. That is the only hard
///   statement here, and it is the one that matters most, because a spill costs
///   30–50% — far more than any ranking error among shapes that fit. It was
///   **checked in the disassembly**, which needs no benchmark: every shipped
///   default in both precisions (six distinct kernels — 1m reuses `real`)
///   allocates 15–16 distinct ymm with *zero* stack traffic in the loop, while
///   `planar 1 x 6` (`live = 16`) spills one register and `planar 2 x 3`
///   (`live = 18`) spills seven. The model's cliff is where the model says.
/// * `acc >= 10`, i.e. enough independent accumulator chains to cover an FMA
///   latency of ~5 cycles against two FMA ports. Below that the kernel is
///   latency-bound however good its byte traffic is. Both survive only just:
///   planar's default has exactly 10 and 3m's exactly 12.
/// * Which shape has the lowest bytes-per-useful-flop within each method, which
///   is the quantity Phase 3 measured to be decisive at the operating `kc`.
///
/// **Guessed** (D19: the same model gets the *ranking* wrong):
///
/// * That the byte-traffic argument still picks the winner when the register
///   file is half as large. It may not: with `MR` down to 4 complex rows the A
///   sliver is small enough that it could stay L1-resident, which is precisely
///   the regime where Phase 3 measured 3m to be *fastest*. So the ranking
///   between the three complex methods is genuinely open on AVX2, more open
///   than it was on AVX-512.
/// * `NR` for planar — though less than it was. `1 x 6` reaches `live = 16`
///   exactly and would give 12 accumulators and 0.83 bytes/flop, strictly
///   better on both counts *if* it allocated cleanly; the disassembly says it
///   does not quite (one spill/reload pair per k-step, against `1 x 5`'s none),
///   which settles the choice without a measurement. What stays guessed is the
///   *price*: one spill of a broadcast is nothing like the 30–50% a spilled
///   accumulator costs, so `1 x 6` might still win. **It could now go on the menu
///   as an alternate** — that was impossible when this was written, because the
///   menu was keyed by `MR` and both shapes have `MR = 4`, but D43 re-keyed it by
///   *position* precisely so an `NR`-only alternate is expressible (that is what
///   made A35's `32x5` reachable in `f32` planar). Appending `(1, 6)` here would
///   make it an `idx=` arm; it has not been done because nothing has asked to
///   measure it and an unmeasured menu entry is a liability.
/// * 3m's default. `1 x 4` is load-port-bound (`f/l = 0.80`) and `2 x 2`
///   (`MR = 8`, on the menu) is exactly balanced at `f/l = 1.00` but moves 25%
///   more bytes per flop. `1 x 4` follows the AVX-512 precedent, where the
///   measured 3m winner `1 x 10` was *also* load-bound and *also* the lowest
///   bytes/flop of its candidates — but that is an analogy, not a measurement.
/// * That one shape per method can be right for all AVX2 hardware at all. The
///   AVX2 microarchitectures spread wider than the AVX-512 ones: FMA latency is
///   5 on Haswell/Broadwell and 4 on later Intel and Zen 3, which moves the
///   `acc >= 10` floor, and Zen 1 splits every 256-bit op into two 128-bit
///   halves, which moves everything. These shapes are a starting point on
///   whatever machine gets measured first, not a tuning.
///
/// Calibrate with `cargo run --release -p tensorcontract --example
/// kernel_shapes` **on an AVX2 machine** — the example sweeps the AVX2 grid too
/// and prints `acc`/`load`/`FMA`/`bytes-per-flop` next to measured GF/s, so the
/// menus below can be replaced from its output the way the AVX-512 ones were.
///
/// # The menus
///
/// | method | menu, `MR` (`NR`) | why the alternates exist |
/// |---|---|---|
/// | real | 8 (6), 12 (4), 4 (8) | `12 (4)` is `live = 16` with a better `f/l`; `4 (8)` is load-bound and only there for the row-block rule |
/// | planar | 4 (5), 8 (2) | `8 (2)` is the only other aspect ratio that fits at all |
/// | 1m | 4 (6), 6 (4), 2 (8) | as real, halved: 1m's complex tile is half its real row block |
/// | 3m | 4 (4), 8 (2) | the load-bound / balanced pair described above |
///
/// Note what the small blocks do to the write-back: **every** `MR` on every
/// menu here (2, 4, 6, 8, 12) divides 24, and the TCCG corpus rounds every
/// stride-1 extent up to a multiple of 24, so the gather-path problem that
/// motivated the row-block menu on AVX-512 does not arise at all in `f64` on
/// AVX2, which is one reason these menus can stay short.
///
/// `tcbench shapes` says so quantitatively, and it costs no CPU to re-check
/// (`TENSORCONTRACT_KERNEL=avx2 tcbench shapes`). Over the 392 corpus
/// case-dtype-methods, `Plan::row_block` would change shape on **26 under
/// AVX-512 and 12 under AVX2**, and the 12 are all `f32` — it is inert in `f64`
/// and in all three complex methods in both precisions. The sharper number is
/// the other one: **81** case-dtype-methods have no shape on their AVX-512 menu
/// that clears the gather path at all, and **0** on their AVX2 menu. Smaller
/// register blocks are worse for the kernel and better for the write-back, and
/// on AVX2 the write-back side of that trade is simply won.
pub mod cfg_avx2_f64 {
    use super::*;
    configs!(
        f64,
        avx2_f64,
        "avx2",
        real = [(2, 6), (3, 4), (1, 8)],
        planar = [(1, 5), (2, 2)],
        onem = [(2, 6), (3, 4), (1, 8)],
        threem = [(1, 4), (2, 2)],
    );
}

/// AVX2 shapes for `f32` / `c32` (`L = 8`). **Provisional and unmeasured**; the
/// reasoning, and the modelled/guessed split, is in [`cfg_avx2_f64`].
///
/// The `MV x NR` grid is deliberately *identical* to the `f64` one, so that a
/// single-vs-double comparison on AVX2 is a comparison of the hardware and not
/// of two shape choices — the same reason `examples/kernel_shapes` sweeps one
/// candidate list for both types. (AVX-512 1m is the one place the project
/// departed from that, and it departed from it on measured evidence, which
/// there is none of here.) So the logical row blocks are twice the `f64` ones:
///
/// | kernel | `MV x NR` | `MR x NR` (logical) | acc | live | bytes/flop |
/// |---|---|---|---|---|---|
/// | real | `2 x 6` | `16 x 6` | 12 | 15 | 0.46 |
/// | planar | `1 x 5` | `8 x 5` | 10 | 14 | **0.33** |
/// | 1m | `2 x 6` | `8 x 6` | 12 | 15 | 0.46 |
/// | 3m | `1 x 4` | `8 x 4` | 12 | 14 | 0.56 |
///
/// Menus: real 16 (6), 24 (4), 8 (8); planar 8 (5), 16 (2); 1m 8 (6), 12 (4),
/// 4 (8); 3m 8 (4), 16 (2). Here `real 24 (4)`, `1m 12 (4)` and the `8`s divide
/// the corpus's multiple-of-24 extents while the `16`s do not, so the row-block
/// rule has a real choice to make in `f32` on AVX2 as it does on AVX-512 —
/// unlike `f64`, where every shape on every menu divides 24 already.
pub mod cfg_avx2_f32 {
    use super::*;
    configs!(
        f32,
        avx2_f32,
        "avx2",
        real = [(2, 6), (3, 4), (1, 8)],
        planar = [(1, 5), (2, 2)],
        onem = [(2, 6), (3, 4), (1, 8)],
        threem = [(1, 4), (2, 2)],
    );
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

/// An instruction set this file has kernels for, widest first.
///
/// Not a CPU-feature bitset: it names a *kernel family*, i.e. one instantiation
/// of `simd_kernels!` plus the register-block menus that go with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// AVX-512F. Measured; the reference machine.
    Avx512,
    /// AVX2 + FMA3. Provisional register blocks — see [`cfg_avx2_f64`].
    Avx2,
}

impl Isa {
    /// The name used in test failure messages and in `tcbench info` output.
    pub fn name(self) -> &'static str {
        match self {
            Isa::Avx512 => "avx512",
            Isa::Avx2 => "avx2",
        }
    }
}

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

/// Whether AVX2 *and* FMA3 are usable. Checked once per process.
///
/// Both bits are required and neither implies the other in CPUID, even though
/// no shipping CPU has one without the other: the kernels' loads and broadcasts
/// are AVX/AVX2 and their `vfmadd`/`vfnmadd` are FMA3.
fn have_avx2() -> bool {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static YES: OnceLock<bool> = OnceLock::new();
        *YES.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(all(target_feature = "avx2", target_feature = "fma"))
    }
}

/// Every ISA whose kernels this CPU can execute, widest first, **ignoring
/// `TENSORCONTRACT_KERNEL`**.
///
/// Dispatch uses [`selected_isa`]; this exists for the kernel-contract tests,
/// which must check every kernel the machine can run rather than only the one
/// it would choose. That is the only way the AVX2 kernels get exercised at all
/// on an AVX-512 reference machine under a plain `cargo test`.
pub fn available_isas() -> &'static [Isa] {
    match (have_avx512(), have_avx2()) {
        (true, true) => &[Isa::Avx512, Isa::Avx2],
        (true, false) => &[Isa::Avx512],
        (false, true) => &[Isa::Avx2],
        (false, false) => &[],
    }
}

/// The ISA the engine will actually dispatch to, or `None` for the portable
/// scalar path. Cached for the process, like the feature detection it wraps.
///
/// `TENSORCONTRACT_KERNEL` overrides the choice: `scalar` takes the portable
/// path, `avx2` and `avx512` pin a family. A pinned family the CPU cannot run
/// falls through to scalar rather than faulting — pinning is a testing and A/B
/// facility, not a promise that the hardware exists.
pub fn selected_isa() -> Option<Isa> {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static ISA: OnceLock<Option<Isa>> = OnceLock::new();
        *ISA.get_or_init(pick_isa)
    }
    #[cfg(not(feature = "std"))]
    {
        pick_isa()
    }
}

fn pick_isa() -> Option<Isa> {
    use super::KernelForce;
    match super::kernel_force() {
        KernelForce::Scalar => None,
        KernelForce::Avx2 => have_avx2().then_some(Isa::Avx2),
        KernelForce::Avx512 => have_avx512().then_some(Isa::Avx512),
        // Pinning NEON on x86 is a request this target cannot honour, and
        // falling through to scalar is what pinning an absent family already
        // does here. `kernel::aarch64` declines `avx2`/`avx512` the same way, so
        // one sweep script can pass the same arm list to every machine.
        KernelForce::Neon => None,
        // Widest first. AVX-512 is measured and AVX2 is not, so there is no
        // shape-dependent choice between them to make here.
        KernelForce::Auto => available_isas().first().copied(),
    }
}

/// One ISA's five kernel-set entry points for one element type.
///
/// A table of function pointers rather than five `match isa` statements per
/// type, because everything that wants this wants *all* of it at once: dispatch
/// wants the selected ISA's five, and the contract tests want each available
/// ISA's five in turn.
#[derive(Clone, Copy)]
pub struct IsaConfigs<T> {
    pub isa: Isa,
    pub real: fn() -> KernelConfig<T>,
    pub cplx: fn(ComplexMethod) -> KernelConfig<T>,
    pub row_blocks: fn(bool, ComplexMethod) -> &'static [(usize, usize)],
    pub config_at: fn(bool, ComplexMethod, usize) -> Option<KernelConfig<T>>,
}

/// Written by hand for two reasons: the derive would demand `T: Debug` for a
/// struct that never stores a `T`, and the other five fields are function
/// pointers whose addresses tell a reader nothing. The `isa` *is* the identity of
/// the table.
impl<T> core::fmt::Debug for IsaConfigs<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IsaConfigs")
            .field("isa", &self.isa)
            .finish()
    }
}

/// The five entry points the [`super::KernelSet`] impls call, per type: the
/// default shape, the menu of row blocks, and the config at a chosen one, each
/// resolved through [`selected_isa`]. All yield `None`/`&[]` when no vectorised
/// ISA is available or `TENSORCONTRACT_KERNEL=scalar` is set, which sends the
/// caller to the portable scalar path.
macro_rules! dispatch {
    ($t:ty, $cfg512:ident, $cfg2:ident, $sets:ident,
     $real:ident, $cplx:ident, $rows:ident, $real_at:ident, $cplx_at:ident) => {
        /// This type's kernel set for a named ISA, whether or not the CPU has it.
        pub fn $sets(isa: Isa) -> IsaConfigs<$t> {
            match isa {
                Isa::Avx512 => IsaConfigs {
                    isa,
                    real: $cfg512::real_config,
                    cplx: $cfg512::cplx_config,
                    row_blocks: $cfg512::row_blocks,
                    config_at: $cfg512::config_at,
                },
                Isa::Avx2 => IsaConfigs {
                    isa,
                    real: $cfg2::real_config,
                    cplx: $cfg2::cplx_config,
                    row_blocks: $cfg2::row_blocks,
                    config_at: $cfg2::config_at,
                },
            }
        }

        pub fn $real() -> Option<KernelConfig<$t>> {
            let s = $sets(selected_isa()?);
            Some((s.real)())
        }

        pub fn $cplx(method: ComplexMethod) -> Option<KernelConfig<$t>> {
            let s = $sets(selected_isa()?);
            Some((s.cplx)(method))
        }

        /// Row blocks with a kernel, default first; empty when unavailable.
        pub fn $rows(complex: bool, method: ComplexMethod) -> &'static [(usize, usize)] {
            match selected_isa() {
                Some(isa) => ($sets(isa).row_blocks)(complex, method),
                None => &[],
            }
        }

        pub fn $real_at(mr: usize) -> Option<KernelConfig<$t>> {
            let s = $sets(selected_isa()?);
            (s.config_at)(false, ComplexMethod::Planar, mr)
        }

        pub fn $cplx_at(method: ComplexMethod, mr: usize) -> Option<KernelConfig<$t>> {
            let s = $sets(selected_isa()?);
            (s.config_at)(true, method, mr)
        }
    };
}

dispatch!(
    f64,
    cfg_avx512_f64,
    cfg_avx2_f64,
    isa_configs_f64,
    config_real_f64,
    config_cplx_f64,
    row_blocks_f64,
    config_real_f64_at,
    config_cplx_f64_at
);
dispatch!(
    f32,
    cfg_avx512_f32,
    cfg_avx2_f32,
    isa_configs_f32,
    config_real_f32,
    config_cplx_f32,
    row_blocks_f32,
    config_real_f32_at,
    config_cplx_f32_at
);
