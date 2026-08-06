//! Micro-kernel shape sweep.
//!
//! Times each candidate register block in isolation, with both packed panels
//! hot, so that the register blocking in `kernel::x86` is chosen from measured
//! throughput rather than from a uop count. This is *not* an end-to-end
//! benchmark: it deliberately excludes packing, write-back and cache blocking,
//! because those are what the rest of the engine is for. It is the right tool
//! for one question only — given this method, which `MR x NR` should it use?
//!
//! Throughput is reported as **useful** GFLOP/s: `2*MR*NR*kc` for real and
//! `8*MR*NR*kc` for complex, whatever the method actually executes. 3m
//! therefore shows its 25% flop saving as extra throughput, which is the honest
//! way to compare it against planar and 1m.
//!
//! It sweeps whichever instruction sets this target has: AVX-512 and AVX2 on
//! x86, NEON on aarch64. The AVX-512 grid is the one Phase 3 chose the shipped
//! AVX-512 shapes from. **Two of the three families are still unmeasured, and
//! this is their calibration path** — the `best per method` lines at the bottom
//! are exactly what belongs in the corresponding `cfg_*` menu, replacing a model
//! with a measurement:
//!
//! * **AVX2** — picked from the register budget and the uop model on an
//!   AVX-512-only reference machine. Run this on a Haswell/Zen box.
//! * **NEON** — picked from the register budget alone. Run this on the Apple
//!   machine; it is the only thing that makes `cfg_neon_f64` / `cfg_neon_f32`
//!   more than a model, and A34 says register blocks are per-microarchitecture
//!   rather than per-ISA, so the budget agreeing with BLIS's chosen ARM shapes is
//!   corroboration and not evidence about this engine.
//!
//! NEON and AVX-512 share the candidate grid because both have 32 vector
//! registers and the budget counts registers, not lanes — see `cases!`. Their
//! rankings have no reason to agree.
//!
//! ```text
//! cargo run --release -p tensorcontract --example kernel_shapes
//! ```

// The sweep measures whichever vectorised kernel module this target has, so it
// is gated on there being one at all. It still has to *compile* everywhere,
// because `cargo test` builds examples and an example that fails to build fails
// the suite for every user of that target.
//
// It was x86-only until `kernel::aarch64` existed, and the *generic* half never
// was: `Kind`, `Case`, its cost model, `time` and `sweep` only ever needed a
// function pointer and a lane count. Widening this was mostly deleting a cfg.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
mod sweep {
    use std::time::Instant;

    #[cfg(target_arch = "aarch64")]
    use tensorcontract::kernel::aarch64::{neon_f32, neon_f64};
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use tensorcontract::kernel::x86::{avx2_f32, avx2_f64, avx512_f32, avx512_f64};

    const TRIALS: usize = 5;

    /// Which method a case belongs to; also fixes how its uop cost is counted.
    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Real,
        Planar,
        OneM,
        ThreeM,
    }

    impl Kind {
        fn label(self) -> &'static str {
            match self {
                Kind::Real => "real",
                Kind::Planar => "planar",
                Kind::OneM => "1m",
                Kind::ThreeM => "3m",
            }
        }
    }

    struct Case<T: 'static> {
        kind: Kind,
        /// Vector registers per A plane per k-step (the *real* row block for 1m).
        mv: usize,
        nr: usize,
        func: unsafe fn(usize, *const T, *const T, *mut T),
    }

    impl<T: 'static> Case<T> {
        /// Logical (complex, for complex methods) rows of the micro-tile.
        fn mr(&self, lanes: usize) -> usize {
            match self.kind {
                Kind::OneM => self.mv * lanes / 2,
                _ => self.mv * lanes,
            }
        }

        /// Reals per A sliver / per B sliver / in the tile, per logical k-step.
        fn sizes(&self, lanes: usize) -> (usize, usize, usize) {
            let mr = self.mr(lanes);
            match self.kind {
                Kind::Real => (mr, self.nr, mr * self.nr),
                Kind::Planar => (2 * mr, 2 * self.nr, 2 * mr * self.nr),
                Kind::OneM => (4 * mr, 2 * self.nr, 2 * mr * self.nr),
                Kind::ThreeM => (3 * mr, 3 * self.nr, 3 * mr * self.nr),
            }
        }

        /// Useful flops per logical k-step for one tile.
        fn flops_per_k(&self, lanes: usize) -> usize {
            let per = if self.kind == Kind::Real { 2 } else { 8 };
            per * self.mr(lanes) * self.nr
        }

        /// Accumulator registers, load-port uops and FMA uops per logical k-step.
        fn cost(&self) -> (usize, usize, usize) {
            let (mv, nr) = (self.mv, self.nr);
            match self.kind {
                Kind::Real => (mv * nr, mv + nr, mv * nr),
                Kind::Planar => (2 * mv * nr, 2 * mv + 2 * nr, 4 * mv * nr),
                // Two real k-steps make one complex one.
                Kind::OneM => (mv * nr, 2 * (mv + nr), 2 * mv * nr),
                Kind::ThreeM => (3 * mv * nr, 3 * mv + 3 * nr, 3 * mv * nr),
            }
        }

        /// Live vector registers: accumulators, plus the A plane(s) the body holds,
        /// plus the broadcasts. Above the architectural register count the
        /// allocator spills and the shape falls off a 30–50% cliff, so this is the
        /// one column of the model that reliably predicts the measurement (D19) —
        /// and the one that changes most between AVX-512 and AVX2.
        fn live(&self) -> usize {
            let (acc, _, _) = self.cost();
            match self.kind {
                // 3m loads one plane at a time, and 1m is the real kernel.
                Kind::Real | Kind::OneM | Kind::ThreeM => acc + self.mv + 1,
                // planar holds both A planes and both B broadcasts at once.
                Kind::Planar => acc + 2 * self.mv + 2,
            }
        }
    }

    fn time<T: Copy + Default + 'static>(
        c: &Case<T>,
        lanes: usize,
        kc: usize,
        fill: impl Fn(usize) -> T,
    ) -> f64 {
        let (a_per_k, b_per_k, tile) = c.sizes(lanes);
        let a: Vec<T> = (0..a_per_k * kc).map(&fill).collect();
        let b: Vec<T> = (0..b_per_k * kc).map(&fill).collect();
        let mut ab = vec![T::default(); tile];

        // Enough repetitions that even the widest tile runs for a few hundred ms,
        // and few enough that the narrow ones do not take all day.
        let reps = (400_000_000 / (kc * tile.max(1))).max(200);

        let mut best = f64::INFINITY;
        for t in 0..TRIALS + 1 {
            let t0 = Instant::now();
            for _ in 0..reps {
                unsafe { (c.func)(kc, a.as_ptr(), b.as_ptr(), ab.as_mut_ptr()) };
                std::hint::black_box(ab.as_ptr());
            }
            let s = t0.elapsed().as_secs_f64();
            if t > 0 {
                best = best.min(s);
            }
        }
        std::hint::black_box(&ab);
        (c.flops_per_k(lanes) * kc * reps) as f64 / best / 1e9
    }

    /// Build the **32-register** candidate list for one element type. The `$m`
    /// module supplies the kernels; the shapes are the same for both types so that
    /// the `f32`/`f64` comparison is a comparison of the hardware, not of two shape
    /// choices. Some candidates are deliberately over the 32-register budget, so
    /// that the cliff shows up in the output rather than having to be believed.
    ///
    /// **Used for NEON as well as AVX-512, and that is not a shortcut.** The
    /// budget is counted in *registers* — accumulators plus A planes plus
    /// broadcasts — and both files have 32, so the candidate `(MV, NR)` grid is
    /// the same problem. What differs is `$lanes`, which turns each `MV` into a
    /// different logical `MR`: `(3, 8)` is `24x8` on AVX-512 `f64` and `6x8` on
    /// NEON `f64`. The *rankings* have no reason to agree, which is the whole
    /// point of running it (A34).
    macro_rules! cases {
        ($m:ident, $t:ty) => {
            vec![
                (
                    Kind::Real,
                    2,
                    8,
                    $m::tramp_real::<2, 8> as unsafe fn(usize, *const $t, *const $t, *mut $t),
                ),
                (Kind::Real, 1, 8, $m::tramp_real::<1, 8>),
                (Kind::Real, 1, 10, $m::tramp_real::<1, 10>),
                (Kind::Real, 1, 12, $m::tramp_real::<1, 12>),
                (Kind::Real, 3, 8, $m::tramp_real::<3, 8>),
                (Kind::Real, 3, 9, $m::tramp_real::<3, 9>),
                (Kind::Real, 3, 10, $m::tramp_real::<3, 10>),
                (Kind::Real, 4, 6, $m::tramp_real::<4, 6>),
                (Kind::Real, 4, 7, $m::tramp_real::<4, 7>),
                (Kind::Real, 5, 5, $m::tramp_real::<5, 5>),
                (Kind::Real, 5, 6, $m::tramp_real::<5, 6>),
                (Kind::Real, 6, 4, $m::tramp_real::<6, 4>),
                (Kind::Real, 8, 3, $m::tramp_real::<8, 3>),
                (Kind::Planar, 1, 8, $m::tramp_planar::<1, 8>),
                (Kind::Planar, 1, 12, $m::tramp_planar::<1, 12>),
                (Kind::Planar, 2, 4, $m::tramp_planar::<2, 4>),
                (Kind::Planar, 2, 5, $m::tramp_planar::<2, 5>),
                (Kind::Planar, 2, 6, $m::tramp_planar::<2, 6>),
                (Kind::Planar, 2, 7, $m::tramp_planar::<2, 7>),
                (Kind::Planar, 3, 3, $m::tramp_planar::<3, 3>),
                (Kind::Planar, 3, 4, $m::tramp_planar::<3, 4>),
                (Kind::Planar, 3, 5, $m::tramp_planar::<3, 5>),
                (Kind::Planar, 4, 3, $m::tramp_planar::<4, 3>),
                (Kind::OneM, 1, 8, $m::tramp_onem::<1, 8>),
                (Kind::OneM, 1, 10, $m::tramp_onem::<1, 10>),
                (Kind::OneM, 1, 12, $m::tramp_onem::<1, 12>),
                (Kind::OneM, 2, 8, $m::tramp_onem::<2, 8>),
                (Kind::OneM, 3, 8, $m::tramp_onem::<3, 8>),
                (Kind::OneM, 3, 10, $m::tramp_onem::<3, 10>),
                (Kind::OneM, 4, 6, $m::tramp_onem::<4, 6>),
                (Kind::OneM, 4, 7, $m::tramp_onem::<4, 7>),
                (Kind::OneM, 5, 6, $m::tramp_onem::<5, 6>),
                (Kind::OneM, 6, 4, $m::tramp_onem::<6, 4>),
                (Kind::ThreeM, 1, 8, $m::tramp_threem::<1, 8>),
                (Kind::ThreeM, 1, 10, $m::tramp_threem::<1, 10>),
                (Kind::ThreeM, 2, 3, $m::tramp_threem::<2, 3>),
                (Kind::ThreeM, 2, 4, $m::tramp_threem::<2, 4>),
                (Kind::ThreeM, 2, 5, $m::tramp_threem::<2, 5>),
                (Kind::ThreeM, 3, 3, $m::tramp_threem::<3, 3>),
                (Kind::ThreeM, 3, 4, $m::tramp_threem::<3, 4>),
                (Kind::ThreeM, 4, 2, $m::tramp_threem::<4, 2>),
            ]
            .into_iter()
            .map(|(kind, mv, nr, func)| Case { kind, mv, nr, func })
            .collect::<Vec<_>>()
        };
    }

    /// The AVX2 candidate list. A separate grid because 16 ymm is a different
    /// problem from 32 zmm, not a scaled one: nearly every AVX-512 shape above
    /// spills here, and the surviving aspect ratios are so few for planar and 3m
    /// that the sweep is more about confirming the register bound than exploring.
    /// A handful of over-budget shapes are included on purpose, for the same reason
    /// as in `cases!`.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    macro_rules! cases_avx2 {
        ($m:ident, $t:ty) => {
            vec![
                (
                    Kind::Real,
                    2,
                    6,
                    $m::tramp_real::<2, 6> as unsafe fn(usize, *const $t, *const $t, *mut $t),
                ),
                (Kind::Real, 1, 6, $m::tramp_real::<1, 6>),
                (Kind::Real, 1, 8, $m::tramp_real::<1, 8>),
                (Kind::Real, 2, 4, $m::tramp_real::<2, 4>),
                (Kind::Real, 2, 5, $m::tramp_real::<2, 5>),
                (Kind::Real, 3, 3, $m::tramp_real::<3, 3>),
                (Kind::Real, 3, 4, $m::tramp_real::<3, 4>),
                (Kind::Real, 4, 2, $m::tramp_real::<4, 2>),
                (Kind::Real, 4, 3, $m::tramp_real::<4, 3>),
                (Kind::Real, 5, 2, $m::tramp_real::<5, 2>),
                (Kind::Planar, 1, 3, $m::tramp_planar::<1, 3>),
                (Kind::Planar, 1, 4, $m::tramp_planar::<1, 4>),
                (Kind::Planar, 1, 5, $m::tramp_planar::<1, 5>),
                (Kind::Planar, 1, 6, $m::tramp_planar::<1, 6>),
                (Kind::Planar, 1, 7, $m::tramp_planar::<1, 7>),
                (Kind::Planar, 2, 2, $m::tramp_planar::<2, 2>),
                (Kind::Planar, 2, 3, $m::tramp_planar::<2, 3>),
                (Kind::Planar, 3, 1, $m::tramp_planar::<3, 1>),
                (Kind::OneM, 1, 6, $m::tramp_onem::<1, 6>),
                (Kind::OneM, 1, 8, $m::tramp_onem::<1, 8>),
                (Kind::OneM, 2, 4, $m::tramp_onem::<2, 4>),
                (Kind::OneM, 2, 5, $m::tramp_onem::<2, 5>),
                (Kind::OneM, 2, 6, $m::tramp_onem::<2, 6>),
                (Kind::OneM, 3, 3, $m::tramp_onem::<3, 3>),
                (Kind::OneM, 3, 4, $m::tramp_onem::<3, 4>),
                (Kind::OneM, 4, 2, $m::tramp_onem::<4, 2>),
                (Kind::OneM, 4, 3, $m::tramp_onem::<4, 3>),
                (Kind::ThreeM, 1, 2, $m::tramp_threem::<1, 2>),
                (Kind::ThreeM, 1, 3, $m::tramp_threem::<1, 3>),
                (Kind::ThreeM, 1, 4, $m::tramp_threem::<1, 4>),
                (Kind::ThreeM, 1, 5, $m::tramp_threem::<1, 5>),
                (Kind::ThreeM, 2, 2, $m::tramp_threem::<2, 2>),
                (Kind::ThreeM, 2, 3, $m::tramp_threem::<2, 3>),
                (Kind::ThreeM, 3, 1, $m::tramp_threem::<3, 1>),
            ]
            .into_iter()
            .map(|(kind, mv, nr, func)| Case { kind, mv, nr, func })
            .collect::<Vec<_>>()
        };
    }

    fn sweep<T: Copy + Default + 'static>(
        title: &str,
        lanes: usize,
        regs: usize,
        kcs: &[usize],
        cases: Vec<Case<T>>,
        fill: impl Fn(usize) -> T + Copy,
    ) {
        println!("\n=== {title} (L = {lanes} lanes/register, {regs} registers) ===\n");
        println!(
            "`live` is accumulators + A plane(s) + broadcasts; `!` marks a shape over budget,\n\
             which is `live >= regs`, not `live > regs`: **measured on AVX2** (Phase 4 part 11,\n\
             A30) every `live == 16` shape collapses to 21-33 GF/s beside a 48-53 GF/s sibling\n\
             at `live <= 15`, so one register is not available to the shape. Verified on AVX2\n\
             only -- the AVX-512 sweep predates this column -- and flagged the conservative way.\n"
        );
        print!(
            "{:<8} {:>8} {:>4} {:>5} {:>5} {:>4} {:>6} {:>6}",
            "method", "MRxNR", "acc", "live", "load", "fma", "f/l", "B/flop"
        );
        for kc in kcs {
            print!("{:>13}", format!("GF/s kc={kc}"));
        }
        println!();

        let mut prev: Option<Kind> = None;
        // Best shape per method, at the *last* (largest, most realistic) kc.
        let mut best: Vec<(Kind, usize, usize, f64)> = Vec::new();
        for c in &cases {
            if prev.is_some_and(|p| p != c.kind) {
                println!();
            }
            prev = Some(c.kind);
            let (acc, loads, fma) = c.cost();
            let (a_per_k, b_per_k, _) = c.sizes(lanes);
            let bytes = ((a_per_k + b_per_k) * core::mem::size_of::<T>()) as f64;
            let live = c.live();
            print!(
                "{:<8} {:>4}x{:<3} {:>4} {:>4}{} {:>5} {:>4} {:>6.2} {:>6.3}",
                c.kind.label(),
                c.mr(lanes),
                c.nr,
                acc,
                live,
                // `>=`, not `>`: see the legend above. A30.
                if live >= regs { "!" } else { " " },
                loads,
                fma,
                fma as f64 / loads as f64,
                bytes / c.flops_per_k(lanes) as f64,
            );
            let mut last = 0.0;
            for &kc in kcs {
                last = time(c, lanes, kc, fill);
                print!("{last:>13.1}");
            }
            println!();
            match best.iter_mut().find(|b| b.0 == c.kind) {
                Some(b) if b.3 < last => *b = (c.kind, c.mv, c.nr, last),
                Some(_) => {}
                None => best.push((c.kind, c.mv, c.nr, last)),
            }
        }
        println!("\nbest per method at kc = {}:", kcs[kcs.len() - 1]);
        for (k, mv, nr, gf) in &best {
            println!("  {:<8} MV={mv} NR={nr}   {gf:>7.1} GF/s", k.label());
        }
    }

    /// The shared preamble, so the two `run`s cannot describe the metric
    /// differently.
    fn preamble() {
        println!("Micro-kernel register-block sweep, packed panels hot.");
        println!(
            "Useful flops: 2*MR*NR*kc real, 8*MR*NR*kc complex (so 3m's saving shows up as GF/s)."
        );
        println!(
            "kc is swept because it decides where the A sliver lives: at kc=16 both panels are\n\
         L1-resident and the kernel is purely issue-limited; at kc=256 the A sliver is an L2\n\
         stream, which is what the driver actually does. The gap between the two columns is\n\
         the kernel's exposure to L2 bandwidth, and it is what the `B/flop` column predicts."
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn run() {
        let have_avx512 = is_x86_feature_detected!("avx512f");
        let have_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        if !have_avx512 && !have_avx2 {
            eprintln!("no AVX-512 and no AVX2+FMA on this CPU; nothing to sweep");
            return;
        }
        preamble();

        // The last column is the `kc` `Blocking::derive` actually picks for that
        // element size, so the "best per method" line selects at the operating
        // point rather than at a flattering one.
        if have_avx512 {
            sweep(
                "AVX-512 f64 / c64",
                8,
                32,
                &[16, 64, 256],
                cases!(avx512_f64, f64),
                |i| ((i % 17) as f64 - 8.0) / 9.0,
            );
            sweep(
                "AVX-512 f32 / c32",
                16,
                32,
                &[16, 64, 384],
                cases!(avx512_f32, f32),
                |i| ((i % 17) as f32 - 8.0) / 9.0,
            );
        }
        // Worth running even on a CPU that has AVX-512: it will not reproduce an
        // AVX2-only machine's memory system or its FMA latency, but the shapes'
        // *relative* issue behaviour and every register cliff are visible here, and
        // that is more than the model alone gives.
        if have_avx2 {
            sweep(
                "AVX2 f64 / c64",
                4,
                16,
                &[16, 64, 256],
                cases_avx2!(avx2_f64, f64),
                |i| ((i % 17) as f64 - 8.0) / 9.0,
            );
            sweep(
                "AVX2 f32 / c32",
                8,
                16,
                &[16, 64, 384],
                cases_avx2!(avx2_f32, f32),
                |i| ((i % 17) as f32 - 8.0) / 9.0,
            );
        }
    }

    /// NEON needs no feature detection: it is mandatory in the AArch64 base
    /// architecture, so there is exactly one kernel family and it is always
    /// present. `regs` is 32, the same file AVX-512 has — which is why this
    /// reuses `cases!` rather than needing a grid of its own; see that macro.
    ///
    /// **This is the run that turns `cfg_neon_f64` / `cfg_neon_f32` from a
    /// register-budget model into a measurement**, which is what A34 requires and
    /// what the AVX2 shapes are still waiting for. The `best per method` lines are
    /// what belongs in those two menus.
    #[cfg(target_arch = "aarch64")]
    pub fn run() {
        preamble();
        sweep(
            "NEON f64 / c64",
            2,
            32,
            &[16, 64, 256],
            cases!(neon_f64, f64),
            |i| ((i % 17) as f64 - 8.0) / 9.0,
        );
        sweep(
            "NEON f32 / c32",
            4,
            32,
            &[16, 64, 384],
            cases!(neon_f32, f32),
            |i| ((i % 17) as f32 - 8.0) / 9.0,
        );
    }
}

fn main() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    sweep::run();
    // On a target with no vectorised kernel module the only kernels are the
    // portable scalar ones, whose shapes are const generics chosen by the caller
    // rather than by measurement. There is nothing to sweep, so say so rather
    // than failing to build.
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    println!("kernel_shapes measures the vectorised micro-kernels; this target has none.");
}
