//! Phase 1 empirical premise check.
//!
//! The project's working thesis is that existing transpose-free engines
//! (TBLIS, via BLIS's 1m induced method) leave more on the table for *complex*
//! contractions than for real ones, and most of all in memory-bound or
//! awkward-stride cases. That is folklore, so it gets measured before any of
//! the design is committed to.
//!
//! Method. For a representative subset of the TCCG corpus, spanning
//! bandwidth-bound to compute-bound, measure:
//!
//! * `GEMM(m,n,k)` in the same dtype — the machine ceiling for that shape,
//!   from a tuned vendor library (OpenBLAS);
//! * the same contraction through TBLIS and through TTGT.
//!
//! Then report **efficiency** = contraction GF/s / GEMM GF/s, separately for
//! real and complex, and the ratio of the two. If complex efficiency tracks
//! real efficiency, the thesis is refuted and the project should pivot. If
//! complex efficiency is systematically lower, the headroom is real and the
//! size of the gap bounds what a better complex path can win.
//!
//! Note on normalisation: complex GF/s counts 8 flops per complex MAC, and a
//! complex MAC is exactly four real FMAs, so on a real-SIMD machine the
//! achievable peak is the *same* GF/s number for both domains. An efficiency
//! ratio of 1.0 therefore means "no complex penalty".

use std::process::ExitCode;

use num_complex::Complex;
#[cfg(feature = "blas")]
use rand::SeedableRng;
#[cfg(feature = "blas")]
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "blas")]
use super::{gflops, timed};
use super::{pin_single_threaded, BenchElem};
use crate::corpus;
use crate::report::{Results, Table};
use crate::Options;

/// How many cases to sample from the corpus, spread evenly across TCCG's
/// bandwidth-bound to compute-bound ordering.
const SUBSET_SIZE: usize = 12;

fn subset() -> Vec<corpus::Case> {
    let sorted = corpus::sorted_cases();
    if sorted.len() <= SUBSET_SIZE {
        return sorted;
    }
    // Evenly spaced picks so the sample spans the whole difficulty range
    // rather than clustering at one end.
    (0..SUBSET_SIZE)
        .map(|i| sorted[i * (sorted.len() - 1) / (SUBSET_SIZE - 1)].clone())
        .collect()
}

pub fn run(opts: &Options) -> ExitCode {
    pin_single_threaded();
    crate::report::print_environment();
    println!();

    #[cfg(not(feature = "tblis"))]
    eprintln!("warning: built without the `tblis` feature; TBLIS columns will be empty");
    #[cfg(not(feature = "blas"))]
    eprintln!("warning: built without the `blas` feature; no GEMM ceiling or TTGT");

    let cases: Vec<_> = subset()
        .into_iter()
        .filter(|c| {
            opts.case_filter
                .as_ref()
                .map(|f| c.name.contains(f.as_str()))
                .unwrap_or(true)
        })
        .collect();

    println!(
        "premise check: {} cases at {} MiB nominal tensor size, {} reps, stress={}\n",
        cases.len(),
        opts.size_mib,
        opts.reps,
        opts.stress.name()
    );

    let mut results = Results::default();
    let mut ceilings: Vec<(String, String, f64)> = Vec::new();

    for case in &cases {
        let s = corpus::size_case_stressed(case, opts.tensor_bytes(), opts.stress);
        let (m, n, k) = s.mnk();
        eprintln!(
            "  {} : m={m} n={n} k={k} macs={:.3e}",
            case.name,
            s.macs() as f64
        );

        if opts.wants("f64") {
            super::sweep::run_case::<f64>(&s, opts, &mut results);
            ceilings.push((
                case.name.into(),
                "f64".into(),
                gemm_ceiling::<f64>(m, n, k, opts),
            ));
        }
        if opts.wants("c64") {
            super::sweep::run_case::<Complex<f64>>(&s, opts, &mut results);
            ceilings.push((
                case.name.into(),
                "c64".into(),
                gemm_ceiling::<Complex<f64>>(m, n, k, opts),
            ));
        }
        if opts.wants("f32") {
            super::sweep::run_case::<f32>(&s, opts, &mut results);
            ceilings.push((
                case.name.into(),
                "f32".into(),
                gemm_ceiling::<f32>(m, n, k, opts),
            ));
        }
        if opts.wants("c32") {
            super::sweep::run_case::<Complex<f32>>(&s, opts, &mut results);
            ceilings.push((
                case.name.into(),
                "c32".into(),
                gemm_ceiling::<Complex<f32>>(m, n, k, opts),
            ));
        }
    }

    report(&results, &ceilings, opts);
    if let Some(path) = &opts.csv {
        let _ = results.write_csv(path);
        println!("\nwrote {path}");
    }
    ExitCode::SUCCESS
}

/// Peak achievable for this shape: a dense GEMM of the same `m`, `n`, `k` from
/// a tuned vendor library, so packing/transposition overheads are the only
/// difference from the contraction measurement.
#[allow(unused_variables)]
fn gemm_ceiling<T: BenchElem>(m: u64, n: u64, k: u64, opts: &Options) -> f64 {
    #[cfg(feature = "blas")]
    {
        let (m, n, k) = (m as usize, n as usize, k as usize);
        if m == 0 || n == 0 || k == 0 {
            return f64::NAN;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let a: Vec<T> = (0..m * k).map(|_| T::sample(&mut rng)).collect();
        let b: Vec<T> = (0..k * n).map(|_| T::sample(&mut rng)).collect();
        let mut c: Vec<T> = vec![T::zero(); m * n];
        let secs = timed(opts.reps.min(3), || unsafe {
            T::gemm(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr())
        });
        gflops::<T>((m * n * k) as u64, secs)
    }
    #[cfg(not(feature = "blas"))]
    {
        f64::NAN
    }
}

fn report(results: &Results, ceilings: &[(String, String, f64)], opts: &Options) {
    let ceiling = |case: &str, dtype: &str| -> Option<f64> {
        ceilings
            .iter()
            .find(|(c, d, _)| c == case && d == dtype)
            .map(|(_, _, g)| *g)
            .filter(|g| g.is_finite() && *g > 0.0)
    };

    for (real, cplx) in [("f64", "c64"), ("f32", "c32")] {
        if !(opts.wants(real) && opts.wants(cplx)) {
            continue;
        }
        for engine in ["tblis", "ttgt", "planar"] {
            if !results.rows.iter().any(|r| r.engine == engine) {
                continue;
            }
            let mut t = Table::new(&[
                "case",
                "regA",
                &format!("{real} GF/s"),
                &format!("gemm {real}"),
                "eff",
                &format!("{cplx} GF/s"),
                &format!("gemm {cplx}"),
                "eff",
                "eff ratio",
            ]);
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for case in subset() {
                let (Some(rr), Some(rc)) = (
                    results.get(case.name, real, engine),
                    results.get(case.name, cplx, engine),
                ) else {
                    continue;
                };
                let gr = ceiling(case.name, real);
                let gc = ceiling(case.name, cplx);
                let er = gr.map(|g| rr.gflops / g);
                let ec = gc.map(|g| rc.gflops / g);
                let ratio = match (er, ec) {
                    (Some(a), Some(b)) if a > 0.0 => {
                        sum += b / a;
                        cnt += 1;
                        format!("{:.3}", b / a)
                    }
                    _ => "-".into(),
                };
                let f = |x: Option<f64>| x.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into());
                let p = |x: Option<f64>| x.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into());
                t.row(vec![
                    case.name.to_string(),
                    format!("{:.2}", rr.reg_a),
                    format!("{:.1}", rr.gflops),
                    f(gr),
                    p(er),
                    format!("{:.1}", rc.gflops),
                    f(gc),
                    p(ec),
                    ratio,
                ]);
            }
            if cnt > 0 {
                t.row(vec![
                    "MEAN".into(),
                    "".into(),
                    "".into(),
                    "".into(),
                    "".into(),
                    "".into(),
                    "".into(),
                    "".into(),
                    format!("{:.3}", sum / cnt as f64),
                ]);
            }
            println!("\n=== {engine}: complex vs real efficiency against the GEMM ceiling ===");
            t.print();
        }
    }

    println!(
        "\ninterpretation: 'eff' is contraction GF/s divided by same-shape GEMM GF/s.\n\
         'eff ratio' is complex eff over real eff. Below 1.0 means the engine loses\n\
         more to overheads on complex data than on real data -- the headroom this\n\
         project targets. At or above 1.0 the complex-weakness thesis is refuted."
    );
}
