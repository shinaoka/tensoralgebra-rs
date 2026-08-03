//! `tcbench` — correctness and performance harness.
//!
//! Subcommands:
//!
//! * `verify` — engine vs TTGT and TBLIS over the whole corpus at benchmark
//!   sizes, in every dtype and under every stride-stress mode.
//! * `premise` — the Phase 1 empirical premise check: measure real-vs-complex
//!   efficiency for TBLIS and TTGT against a same-shape GEMM ceiling, to
//!   confirm or refute the hypothesis that complex contraction is where the
//!   headroom is.
//! * `sweep` — the full TCCG corpus across engines and dtypes, reporting
//!   GFLOP/s and the per-case complex efficiency ratio.
//!
//! Run single-threaded by default so that kernel and packing efficiency, not
//! thread scaling, is what is being compared.

mod blas;
mod corpus;
mod engines;
mod report;
mod tblis;
mod ttgt;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");

    let opts = Options::parse(&args[2.min(args.len())..]);

    match cmd {
        "verify" => engines::verify::run(&opts),
        "premise" => engines::premise::run(&opts),
        "sweep" => engines::sweep::run(&opts),
        "shapes" => engines::shapes::run(&opts),
        "orient" => engines::orient::run(&opts),
        "info" => {
            report::print_environment();
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "usage: tcbench <verify|premise|sweep|shapes|orient|info> [options]\n\
                 \n\
                 `shapes` and `orient` are analyses, not benchmarks: they touch\n\
                 no data and cost no CPU, so they are safe to run while a\n\
                 measurement is in flight, and their output is reproducible.\n\
                 \n\
                 options:\n\
                 \x20 --size <MiB>      tensor size target for TCCG sizing (default 200)\n\
                 \x20 --reps <n>        timed repetitions per measurement (default 5)\n\
                 \x20 --dtype <list>    comma separated: f32,f64,c32,c64 (default all)\n\
                 \x20 --case <substr>   only cases whose name contains this\n\
                 \x20 --engines <list>  comma separated, default all of:\n\
                 \x20                   planar,1m,3m  (this engine's three complex methods)\n\
                 \x20                   ttgt,tblis    (external baselines)\n\
                 \x20 --csv <path>      also write machine-readable results\n\
                 \x20 --stress <mode>   none|ragged|padded: perturb TCCG extents/layouts\n\
                 \x20                   so the block-scatter gather path is exercised\n"
            );
            ExitCode::FAILURE
        }
    }
}

/// Parsed command line.
pub struct Options {
    pub size_mib: f64,
    pub reps: usize,
    pub dtypes: Vec<String>,
    pub case_filter: Option<String>,
    pub engines: Vec<String>,
    pub csv: Option<String>,
    pub stress: corpus::Stress,
}

impl Options {
    fn parse(args: &[String]) -> Options {
        let mut o = Options {
            size_mib: 200.0,
            reps: 5,
            dtypes: vec!["f32".into(), "f64".into(), "c32".into(), "c64".into()],
            case_filter: None,
            engines: crate::engines::sweep::ENGINE_ORDER
                .iter()
                .map(|s| s.to_string())
                .collect(),
            csv: None,
            stress: corpus::Stress::None,
        };
        let mut i = 0;
        while i < args.len() {
            let next = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
            match args[i].as_str() {
                "--size" => {
                    o.size_mib = next(i).parse().unwrap_or(200.0);
                    i += 1;
                }
                "--reps" => {
                    o.reps = next(i).parse().unwrap_or(5);
                    i += 1;
                }
                "--dtype" => {
                    o.dtypes = next(i).split(',').map(|s| s.to_string()).collect();
                    i += 1;
                }
                "--case" => {
                    o.case_filter = Some(next(i));
                    i += 1;
                }
                "--engines" => {
                    o.engines = next(i).split(',').map(|s| s.to_string()).collect();
                    i += 1;
                }
                "--csv" => {
                    o.csv = Some(next(i));
                    i += 1;
                }
                "--stress" => {
                    o.stress = corpus::Stress::parse(&next(i)).unwrap_or_else(|| {
                        eprintln!("unknown --stress value; expected none|ragged|padded");
                        std::process::exit(2)
                    });
                    i += 1;
                }
                other => eprintln!("warning: ignoring unknown option {other}"),
            }
            i += 1;
        }
        o
    }

    pub fn wants(&self, dtype: &str) -> bool {
        self.dtypes.iter().any(|d| d == dtype)
    }
    pub fn engine(&self, name: &str) -> bool {
        self.engines.iter().any(|e| e == name)
    }
    pub fn tensor_bytes(&self) -> f64 {
        self.size_mib * 1024.0 * 1024.0
    }
}
