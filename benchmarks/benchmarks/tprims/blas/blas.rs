//! tprims-blas at an enforced thread count: gemm, batched gemm (faer loop
//! and TBLIS-style, compared), trsm. f64 and c64.
//!
//! Usage: `blas --threads N [--corpus FILE] [--list]`. CSV
//! `case,variant,threads,median_ns,samples`; `CHECK` lines compare the two
//! batched strategies. With `--corpus` only the file's `gemm_batched`
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
    gemm, gemm_batched, trsm, BatchIn, BatchStrategy, Diag, MatIn, Op, Scalar, Side, Uplo,
};
use tprims_exec::Exec;

struct Cfg {
    threads: usize,
    warmup: usize,
    runs: usize,
    filter: Option<String>,
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
        let ns = median_ns(cfg.warmup.min(runs), runs, || {
            let mut cv = StridedViewMut::new(&mut c, &[m, n], &[1, m as isize], 0).expect("c");
            gemm(exec, one, MatIn::new(&av), MatIn::new(&bv), zero, &mut cv).expect("gemm");
        });
        black_box(&c);
        let v = if m == n && n == k {
            format!("n{n}")
        } else {
            format!("m{m}_n{n}_k{k}")
        };
        cfg.row(
            &case,
            &format!("{v}{}", if transposed { "_at" } else { "" }),
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
        trsm_cases::<f64>(&cfg, exec);
        trsm_cases::<Complex64>(&cfg, exec);
    });
}
