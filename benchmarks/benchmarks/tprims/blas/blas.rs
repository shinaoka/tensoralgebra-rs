//! tprims-blas at an enforced thread count: gemm, batched gemm (faer loop
//! and TBLIS-style, compared), trsm. f64 and c64.
//!
//! Usage: `blas --threads N [--corpus FILE] [--engine ENGINE] [--list]`. CSV
//! `case,variant,threads,median_ns,samples`; `CHECK` lines compare the two
//! batched strategies. `--engine faer|packed` selects the matrix engine
//! of the single-GEMM cases (default `faer`); `--partition dynamic:JM,JN`
//! (with `--engine packed`) selects the opt-in `DynamicTiles` scheduler and
//! tags the row labels `_dynJMxJN`; the resolved family is printed
//! to stderr once per case. With `--corpus` only the file's `gemm_batched`
//! entries run (any dtype, recorded strides; `tprims_bench::corpus`), and
//! `--list` prints their names. Environment: `BENCH_RUNS` (default 50, large
//! cases capped), `BENCH_WARMUP` (5), `BENCH_FILTER` (substring of the
//! built-in case groups), `BENCH_CASE` (exact corpus entry name).
use std::hint::black_box;

use num_complex::{Complex32, Complex64};
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_bench::corpus::{Corpus, Dtype, Entry, GemmBatchedEntry};
use tprims_bench::threads::BenchThreads;
use tprims_bench::timing::{env_usize, median_ns};
use tprims_blas::{
    gemm, gemm_batched, gemm_grouped, gemm_with, trsm, BatchIn, BatchStrategy, Conj, Diag, Engine,
    EngineChoice, GemmConfig, GroupedJob, MatIn, Op, Scalar, Side, Uplo,
};
use tprims_exec::Exec;

struct Cfg {
    threads: usize,
    warmup: usize,
    runs: usize,
    filter: Option<String>,
    /// Which engine the single-GEMM cases use. faer unless asked otherwise.
    engine: EngineChoice,
    /// `--partition dynamic:JM,JN`: the opt-in dynamic scheduler of the packed
    /// engine; `None` keeps the static grid and every row label unchanged.
    partition: Option<tprims_gemm_kernel::PartitionPolicy>,
}

impl Cfg {
    /// `--engine {faer|packed}`, default faer.
    fn engine_from_args() -> EngineChoice {
        let args: Vec<String> = std::env::args().collect();
        let Some(i) = args.iter().position(|a| a == "--engine") else {
            return EngineChoice::Faer;
        };
        let name = args
            .get(i + 1)
            .unwrap_or_else(|| panic!("--engine needs a value"));
        match name.as_str() {
            "faer" => EngineChoice::Faer,
            "packed" => EngineChoice::Packed,
            other => panic!("unknown engine {other}; use faer or packed"),
        }
    }
}

impl Cfg {
    fn want(&self, case: &str) -> bool {
        self.filter.as_deref().is_none_or(|f| case.contains(f))
    }
    fn runs_for(&self, flops: f64) -> usize {
        if flops > 5e9 {
            self.runs.min(5)
        } else if flops > 5e8 {
            self.runs.min(15)
        } else {
            self.runs
        }
    }
    fn row(&self, case: &str, variant: &str, ns: f64, runs: usize) {
        println!("{case},{variant},{},{ns:.0},{runs}", self.threads);
    }
}

fn fill<T: Scalar>(len: usize, seed: u64) -> Vec<T> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let re = (s % 1000) as f64 / 1000.0 - 0.5;
            let im = ((s >> 20) % 1000) as f64 / 1000.0 - 0.5;
            <T as Element>::from_parts(
                tensorcontract::Real::from_f64(re),
                tensorcontract::Real::from_f64(im),
            )
        })
        .collect()
}

fn name<T: Scalar>() -> &'static str {
    if T::IS_COMPLEX_SCALAR {
        "c64"
    } else {
        "f64"
    }
}

fn mul_cost<T: Scalar>() -> f64 {
    if T::IS_COMPLEX_SCALAR {
        8.0
    } else {
        2.0
    }
}

fn gemm_cases<T: Scalar>(cfg: &Cfg, exec: &Exec<'_>) {
    let case = format!("gemm_{}", name::<T>());
    if !cfg.want(&case) {
        return;
    }
    // (m, n, k, A transposed)
    let mut shapes = vec![
        (8, 8, 8, false),
        (32, 32, 32, false),
        (128, 128, 128, false),
        (512, 512, 512, false),
        (512, 512, 512, true),
        (1024, 1024, 1024, false),
        (256, 1024, 64, false),
        (1024, 64, 256, false),
    ];
    if !T::IS_COMPLEX_SCALAR {
        shapes.push((2048, 2048, 2048, false));
    }
    for (m, n, k, transposed) in shapes {
        let a: Vec<T> = fill(m * k, 1);
        let b: Vec<T> = fill(k * n, 2);
        let mut c: Vec<T> = fill(m * n, 3);
        let s = if transposed {
            [k as isize, 1]
        } else {
            [1, m as isize]
        };
        let av = StridedView::new(&a, &[m, k], &s, 0).expect("a");
        let bv = StridedView::new(&b, &[k, n], &[1, k as isize], 0).expect("b");
        let runs = cfg.runs_for(mul_cost::<T>() * (m * n * k) as f64);
        let one = <T as Element>::one();
        let zero = <T as Element>::zero();
        assert!(
            cfg.partition.is_none() || matches!(cfg.engine, EngineChoice::Packed),
            "--partition applies to --engine packed"
        );
        let config = tprims_bench::partition::apply(
            GemmConfig {
                engine: cfg.engine.clone(),
                ..Default::default()
            },
            cfg.partition,
        );
        let mut reported = false;
        let ns = median_ns(cfg.warmup.min(runs), runs, || {
            let mut cv = StridedViewMut::new(&mut c, &[m, n], &[1, m as isize], 0).expect("c");
            match &config.engine {
                // The default engine keeps the plain entry point, so its
                // numbers stay comparable with earlier runs.
                EngineChoice::Faer => {
                    gemm(exec, one, MatIn::new(&av), MatIn::new(&bv), zero, &mut cv).expect("gemm")
                }
                _ => {
                    let sel = gemm_with(
                        exec,
                        &config,
                        one,
                        MatIn::new(&av),
                        MatIn::new(&bv),
                        zero,
                        &mut cv,
                    )
                    .expect("gemm");
                    if !reported {
                        reported = true;
                        let engine = match sel.engine {
                            Engine::Faer => "faer",
                            Engine::Packed => "packed",
                            other => panic!("unexpected engine {other:?}"),
                        };
                        eprintln!(
                            "engine={} case={case}: family={:?} mr={} nr={} kc={}",
                            engine, sel.family_id, sel.mr, sel.nr, sel.kc
                        );
                    }
                }
            }
        });
        black_box(&c);
        let v = if m == n && n == k {
            format!("n{n}")
        } else {
            format!("m{m}_n{n}_k{k}")
        };
        cfg.row(
            &case,
            &format!(
                "{v}{}{}",
                if transposed { "_at" } else { "" },
                tprims_bench::partition::suffix(cfg.partition)
            ),
            ns,
            runs,
        );
    }
}

fn batched_cases<T: Scalar>(cfg: &Cfg, exec: &Exec<'_>) {
    let case = format!("gemm_batched_{}", name::<T>());
    if !cfg.want(&case) {
        return;
    }
    for (n, nb) in [
        (2, 1),
        (2, 1024),
        (2, 4096),
        (4, 1024),
        (8, 1024),
        (32, 256),
        (128, 16),
        (512, 1),
    ] {
        let a: Vec<T> = fill(n * n * nb, 4);
        let b: Vec<T> = fill(n * n * nb, 5);
        let st = [1, n as isize, (n * n) as isize];
        let av = StridedView::new(&a, &[n, n, nb], &st, 0).expect("a");
        let bv = StridedView::new(&b, &[n, n, nb], &st, 0).expect("b");
        let runs = cfg.runs_for(mul_cost::<T>() * (n * n * n * nb) as f64);
        let mut outs = Vec::new();
        for (tag, strategy) in [
            ("faer", BatchStrategy::FaerLoop),
            ("tblis", BatchStrategy::Tblis),
        ] {
            let mut c: Vec<T> = vec![<T as Element>::zero(); n * n * nb];
            let mut sel = None;
            let ns = median_ns(cfg.warmup.min(runs), runs, || {
                let mut cv = StridedViewMut::new(&mut c, &[n, n, nb], &st, 0).expect("c");
                sel = Some(
                    gemm_batched(
                        exec,
                        <T as Element>::one(),
                        BatchIn::new(&av),
                        BatchIn::new(&bv),
                        <T as Element>::zero(),
                        &mut cv,
                        strategy,
                    )
                    .expect("batched"),
                );
            });
            cfg.row(&case, &format!("{tag}_n{n}_b{nb}"), ns, runs);
            println!("# selected {tag}_n{n}_b{nb}: {sel:?}");
            outs.push(c);
        }
        let err = outs[0]
            .iter()
            .zip(&outs[1])
            .map(|(x, y)| {
                let d = Element::add(
                    *x,
                    Element::mul(
                        *y,
                        <T as Element>::from_parts(
                            tensorcontract::Real::from_f64(-1.0),
                            tensorcontract::Real::from_f64(0.0),
                        ),
                    ),
                );
                let (re, im) = (Element::re(d), Element::im(d));
                let (re, im): (f64, f64) = (
                    tensorcontract::Real::to_f64(re),
                    tensorcontract::Real::to_f64(im),
                );
                (re * re + im * im).sqrt()
            })
            .fold(0.0, f64::max);
        println!(
            "CHECK {case} n{n}_b{nb} threads={} faer_vs_tblis_max_abs={err:e} {}",
            cfg.threads,
            if err < 1e-9 * n as f64 {
                "ok"
            } else {
                "MISMATCH"
            }
        );
    }
}

/// `(rows, inner, cols)` per grouped job.
type Shapes = Vec<(usize, usize, usize)>;

/// Grouped GEMM over variable-size jobs (compact column-major blocks in shared
/// buffers) against a loop of single `gemm` calls on the same blocks.
fn grouped_cases<T: Scalar>(cfg: &Cfg, exec: &Exec<'_>) {
    let case = format!("gemm_grouped_{}", name::<T>());
    if !cfg.want(&case) {
        return;
    }
    // (label, jobs): many small, a moderate mix, a few large.
    let sets: [(&str, Shapes); 3] = [
        (
            "small_x512",
            (0..512)
                .map(|i| (4 + i % 13, 4 + i % 7, 4 + i % 11))
                .collect(),
        ),
        (
            "mixed_x64",
            (0..64)
                .map(|i| (16 + 7 * (i % 9), 16 + 5 * (i % 7), 16 + 3 * (i % 11)))
                .collect(),
        ),
        (
            "large_x4",
            vec![
                (512, 384, 256),
                (256, 512, 384),
                (384, 256, 512),
                (320, 320, 320),
            ],
        ),
    ];
    for (label, shapes) in sets {
        let (mut oa, mut ob, mut oc) = (0, 0, 0);
        let mut jobs = Vec::new();
        for &(m, k, n) in &shapes {
            jobs.push(GroupedJob {
                a_offset: oa,
                b_offset: ob,
                c_offset: oc,
                rows: m,
                inner: k,
                cols: n,
            });
            oa += m * k;
            ob += k * n;
            oc += m * n;
        }
        let (a, b): (Vec<T>, Vec<T>) = (fill(oa, 6), fill(ob, 7));
        let flops: f64 = shapes.iter().map(|&(m, k, n)| (m * n * k) as f64).sum();
        let runs = cfg.runs_for(mul_cost::<T>() * flops);
        let (one, zero) = (<T as Element>::one(), <T as Element>::zero());
        let mut outs = Vec::new();
        let mut c: Vec<T> = vec![zero; oc];
        let mut sel = None;
        let ns = median_ns(cfg.warmup.min(runs), runs, || {
            sel = Some(
                gemm_grouped(exec, one, &a, Conj::No, &b, Conj::No, zero, &mut c, &jobs)
                    .expect("grouped"),
            );
        });
        cfg.row(&case, &format!("grouped_{label}"), ns, runs);
        println!("# selected grouped_{label}: {sel:?}");
        outs.push(c);
        let mut c: Vec<T> = vec![zero; oc];
        let ns = median_ns(cfg.warmup.min(runs), runs, || {
            for j in &jobs {
                let (m, k, n) = (j.rows, j.inner, j.cols);
                let av = StridedView::new(&a, &[m, k], &[1, m as isize], j.a_offset as isize)
                    .expect("a");
                let bv = StridedView::new(&b, &[k, n], &[1, k as isize], j.b_offset as isize)
                    .expect("b");
                let mut cv =
                    StridedViewMut::new(&mut c, &[m, n], &[1, m as isize], j.c_offset as isize)
                        .expect("c");
                gemm(exec, one, MatIn::new(&av), MatIn::new(&bv), zero, &mut cv).expect("gemm");
            }
        });
        cfg.row(&case, &format!("gemm_loop_{label}"), ns, runs);
        outs.push(c);
        println!(
            "CHECK {case} {label} threads={} grouped_vs_loop_equal={}",
            cfg.threads,
            if outs[0] == outs[1] { "ok" } else { "MISMATCH" }
        );
    }
}

/// One recorded `gemm_batched` entry, both strategies.
fn corpus_batched<T: Scalar>(cfg: &Cfg, exec: &Exec<'_>, e: &GemmBatchedEntry) {
    let ((oa, la), (ob, lb), (oc, lc)) = (e.a.span(), e.b.span(), e.c.span());
    let a: Vec<T> = fill(la, 4);
    let b: Vec<T> = fill(lb, 5);
    let av = StridedView::new(&a, &e.a.dims, &e.a.strides, oa as isize).expect("a");
    let bv = StridedView::new(&b, &e.b.dims, &e.b.strides, ob as isize).expect("b");
    let runs = cfg.runs_for(mul_cost::<T>() * (e.m * e.n * e.k * e.batch) as f64);
    let mut outs = Vec::new();
    for (tag, strategy) in [
        ("faer", BatchStrategy::FaerLoop),
        ("tblis", BatchStrategy::Tblis),
    ] {
        let mut c: Vec<T> = vec![<T as Element>::zero(); lc];
        let mut sel = None;
        let ns = median_ns(cfg.warmup.min(runs), runs, || {
            let mut cv =
                StridedViewMut::new(&mut c, &e.c.dims, &e.c.strides, oc as isize).expect("c");
            sel = Some(
                gemm_batched(
                    exec,
                    <T as Element>::one(),
                    BatchIn::new(&av),
                    BatchIn::new(&bv),
                    <T as Element>::zero(),
                    &mut cv,
                    strategy,
                )
                .expect("batched"),
            );
        });
        cfg.row(&e.name, tag, ns, runs);
        println!("# selected {} {tag}: {sel:?}", e.name);
        outs.push(c);
    }
    let mag = |z: T| {
        let (re, im): (f64, f64) = (
            tensorcontract::Real::to_f64(Element::re(z)),
            tensorcontract::Real::to_f64(Element::im(z)),
        );
        re.hypot(im)
    };
    let scale = outs[0].iter().map(|&z| mag(z)).fold(1.0, f64::max);
    let err = outs[0]
        .iter()
        .zip(&outs[1])
        .map(|(&x, &y)| {
            mag(Element::add(
                x,
                Element::mul(
                    y,
                    <T as Element>::from_parts(
                        tensorcontract::Real::from_f64(-1.0),
                        tensorcontract::Real::from_f64(0.0),
                    ),
                ),
            ))
        })
        .fold(0.0, f64::max)
        / scale;
    let tol = if matches!(e.dtype, Dtype::F32 | Dtype::C32) {
        1e-4
    } else {
        1e-12
    };
    println!(
        "CHECK {} threads={} faer_vs_tblis_rel={err:e} {}",
        e.name,
        cfg.threads,
        if err < tol { "ok" } else { "MISMATCH" }
    );
}

/// The `gemm_batched` entries of `--corpus FILE`, if given.
fn corpus_entries() -> Option<Vec<GemmBatchedEntry>> {
    let args: Vec<String> = std::env::args().collect();
    let i = args.iter().position(|a| a == "--corpus")?;
    let path = args
        .get(i + 1)
        .unwrap_or_else(|| panic!("--corpus needs a file"));
    let corpus = Corpus::load(path).unwrap_or_else(|e| panic!("{e}"));
    Some(
        corpus
            .entries
            .into_iter()
            .filter_map(|e| match e {
                Entry::GemmBatched(g) => Some(g),
                _ => None,
            })
            .collect(),
    )
}

fn trsm_cases<T: Scalar>(cfg: &Cfg, exec: &Exec<'_>) {
    let case = format!("trsm_{}", name::<T>());
    if !cfg.want(&case) {
        return;
    }
    for n in [32usize, 256, 1024] {
        let mut a: Vec<T> = fill(n * n, 6);
        for i in 0..n {
            a[i + i * n] = Element::add(
                a[i + i * n],
                <T as Element>::from_parts(
                    tensorcontract::Real::from_f64(n as f64),
                    tensorcontract::Real::from_f64(0.0),
                ),
            );
        }
        let b0: Vec<T> = fill(n * n, 7);
        let av = StridedView::new(&a, &[n, n], &[1, n as isize], 0).expect("a");
        let runs = cfg.runs_for(mul_cost::<T>() * (n * n * n) as f64 / 2.0);
        let mut b = b0.clone();
        let ns = median_ns(cfg.warmup.min(runs), runs, || {
            b.copy_from_slice(&b0);
            let mut bv = StridedViewMut::new(&mut b, &[n, n], &[1, n as isize], 0).expect("b");
            trsm(
                exec,
                Side::Left,
                Uplo::Lower,
                Op::N,
                Diag::NonUnit,
                <T as Element>::one(),
                &av,
                &mut bv,
            )
            .expect("trsm");
        });
        black_box(&b);
        cfg.row(&case, &format!("left_lower_n{n}_nrhs{n}"), ns, runs);
    }
}

fn main() {
    let corpus = corpus_entries();
    if std::env::args().any(|a| a == "--list") {
        for e in corpus.iter().flatten() {
            println!("{}", e.name);
        }
        return;
    }
    let threads = BenchThreads::from_args();
    threads.verify();
    let cfg = Cfg {
        threads: threads.requested,
        warmup: env_usize("BENCH_WARMUP", 5),
        runs: env_usize("BENCH_RUNS", 50),
        filter: std::env::var("BENCH_FILTER").ok().filter(|f| !f.is_empty()),
        engine: Cfg::engine_from_args(),
        partition: tprims_bench::partition::from_args(),
    };
    println!("case,variant,threads,median_ns,samples");
    if let Some(entries) = corpus {
        let exact = std::env::var("BENCH_CASE").ok().filter(|f| !f.is_empty());
        threads.with_exec(|exec, _| {
            for e in entries
                .iter()
                .filter(|e| exact.as_deref().is_none_or(|x| x == e.name))
            {
                match e.dtype {
                    Dtype::F32 => corpus_batched::<f32>(&cfg, exec, e),
                    Dtype::F64 => corpus_batched::<f64>(&cfg, exec, e),
                    Dtype::C32 => corpus_batched::<Complex32>(&cfg, exec, e),
                    Dtype::C64 => corpus_batched::<Complex64>(&cfg, exec, e),
                }
            }
        });
        return;
    }
    threads.with_exec(|exec, _| {
        gemm_cases::<f64>(&cfg, exec);
        gemm_cases::<Complex64>(&cfg, exec);
        batched_cases::<f64>(&cfg, exec);
        batched_cases::<Complex64>(&cfg, exec);
        grouped_cases::<f64>(&cfg, exec);
        grouped_cases::<Complex64>(&cfg, exec);
        trsm_cases::<f64>(&cfg, exec);
        trsm_cases::<Complex64>(&cfg, exec);
    });
}
