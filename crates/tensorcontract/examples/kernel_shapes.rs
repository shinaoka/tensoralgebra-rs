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
//! ```text
//! cargo run --release -p tensorcontract --example kernel_shapes
//! ```

use std::time::Instant;

use tensorcontract::kernel::x86::{avx512_f32, avx512_f64};

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

/// Build the candidate list for one element type. The `$m` module supplies the
/// kernels; the shapes are the same for both types so that the `f32`/`f64`
/// comparison is a comparison of the hardware, not of two shape choices.
macro_rules! cases {
    ($m:ident, $t:ty) => {
        vec![
            (
                Kind::Real,
                2,
                8,
                $m::tramp_real::<2, 8> as unsafe fn(usize, *const $t, *const $t, *mut $t),
            ),
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

fn sweep<T: Copy + Default + 'static>(
    title: &str,
    lanes: usize,
    kcs: &[usize],
    cases: Vec<Case<T>>,
    fill: impl Fn(usize) -> T + Copy,
) {
    println!("\n=== {title} (L = {lanes} lanes/register) ===\n");
    print!(
        "{:<8} {:>8} {:>4} {:>5} {:>4} {:>6} {:>6}",
        "method", "MRxNR", "acc", "load", "fma", "f/l", "B/flop"
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
        print!(
            "{:<8} {:>4}x{:<3} {:>4} {:>5} {:>4} {:>6.2} {:>6.3}",
            c.kind.label(),
            c.mr(lanes),
            c.nr,
            acc,
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

fn main() {
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("no AVX-512 on this CPU; nothing to sweep");
        return;
    }
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

    // The last column is the `kc` `Blocking::derive` actually picks for that
    // element size, so the "best per method" line selects at the operating
    // point rather than at a flattering one.
    sweep(
        "AVX-512 f64 / c64",
        8,
        &[16, 64, 256],
        cases!(avx512_f64, f64),
        |i| ((i % 17) as f64 - 8.0) / 9.0,
    );
    sweep(
        "AVX-512 f32 / c32",
        16,
        &[16, 64, 384],
        cases!(avx512_f32, f32),
        |i| ((i % 17) as f32 - 8.0) / 9.0,
    );
}
