//! tprims-linalg at an enforced thread count: single-matrix factorizations
//! and solves, and the batched module (including the small-matrix,
//! large-batch regression case). f64 and c64.
//!
//! Usage: `linalg --threads N`. CSV `case,variant,threads,median_ns,samples`.
//! Environment: `BENCH_RUNS` (default 30, capped for large cases),
//! `BENCH_WARMUP` (3), `BENCH_FILTER` (substring), `BENCH_CASE` (exact case
//! name; `run.sh` uses it to measure every case in its own process, because
//! parallel cases leave the pool in a state that slows later serial rows).
use std::hint::black_box;

use num_complex::Complex64;
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_bench::threads::BenchThreads;
use tprims_bench::timing::{env_usize, median_ns};
use tprims_blas::Scalar;
use tprims_exec::Exec;
use tprims_linalg::{
    batched, cholesky, eig, eigh, lu, qr, solve, svd, EigScalar, Matrix, Status, Vectors,
};

struct Cfg {
    threads: usize,
    warmup: usize,
    runs: usize,
    filter: Option<String>,
    exact: Option<String>,
}

impl Cfg {
    fn want(&self, case: &str) -> bool {
        if let Some(exact) = &self.exact {
            return case == exact;
        }
        self.filter.as_deref().is_none_or(|f| case.contains(f))
    }
    fn runs_for(&self, n: usize) -> usize {
        if n >= 512 {
            self.runs.min(3)
        } else if n >= 128 {
            self.runs.min(10)
        } else {
            self.runs
        }
    }
    fn time(&self, case: &str, variant: &str, n: usize, mut f: impl FnMut()) {
        if !self.want(case) {
            return;
        }
        let runs = self.runs_for(n);
        let ns = median_ns(self.warmup.min(runs), runs, &mut f);
        println!("{case},{variant},{},{ns:.0},{runs}", self.threads);
    }
}

fn name<T: Scalar>() -> &'static str {
    if T::IS_COMPLEX_SCALAR {
        "c64"
    } else {
        "f64"
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

/// Hermitian positive definite: (A + Aᴴ)/2 + n I, as a column-major buffer.
fn hpd<T: Scalar>(n: usize, seed: u64) -> Vec<T> {
    let a: Vec<T> = fill(n * n, seed);
    let mut h = vec![<T as Element>::zero(); n * n];
    let half = <T as Element>::from_parts(
        tensorcontract::Real::from_f64(0.5),
        tensorcontract::Real::from_f64(0.0),
    );
    for j in 0..n {
        for i in 0..n {
            let v = Element::mul(
                half,
                Element::add(a[i + j * n], Element::conj(a[j + i * n])),
            );
            h[i + j * n] = v;
        }
        let d = <T as Element>::from_parts(
            tensorcontract::Real::from_f64(n as f64),
            tensorcontract::Real::from_f64(0.0),
        );
        h[j + j * n] = Element::add(RePart::re_part(h[j + j * n]), d);
    }
    h
}

trait RePart {
    fn re_part(self) -> Self;
}

impl<T: Scalar> RePart for T {
    fn re_part(self) -> Self {
        <T as Element>::from_parts(Element::re(self), tensorcontract::Real::from_f64(0.0))
    }
}

fn single<T: EigScalar>(cfg: &Cfg, exec: &Exec<'_>) {
    for n in [2usize, 4, 8, 32, 128, 512] {
        let a = Matrix::from_col_major(n, n, fill::<T>(n * n, 3)).expect("a");
        let h = Matrix::from_col_major(n, n, hpd::<T>(n, 4)).expect("h");
        let b0 = fill::<T>(n, 5);
        let v = format!("n{n}");
        let t = name::<T>();
        cfg.time(&format!("cholesky_{t}"), &v, n, || {
            black_box(cholesky(exec, &h.view()).expect("chol"));
        });
        cfg.time(&format!("lu_{t}"), &v, n, || {
            black_box(lu(exec, &a.view()).expect("lu"));
        });
        cfg.time(&format!("solve_{t}"), &v, n, || {
            let mut b = b0.clone();
            let mut bv = StridedViewMut::new(&mut b, &[n, 1], &[1, n as isize], 0).expect("b");
            let _ = solve(exec, &a.view(), &mut bv);
            black_box(&b);
        });
        cfg.time(&format!("qr_{t}"), &v, n, || {
            black_box(qr(exec, &a.view()).expect("qr"));
        });
        cfg.time(&format!("svd_{t}"), &v, n, || {
            black_box(svd(exec, &a.view(), Vectors::Thin).expect("svd"));
        });
        cfg.time(&format!("eigh_{t}"), &v, n, || {
            black_box(eigh(exec, &h.view(), true).expect("eigh"));
        });
        cfg.time(&format!("eig_{t}"), &v, n, || {
            black_box(eig(exec, &a.view(), true).expect("eig"));
        });
    }
}

fn batch<T: Scalar>(cfg: &Cfg, exec: &Exec<'_>) {
    let t = name::<T>();
    for (n, nb) in [(2usize, 1024usize), (4, 1024), (8, 1024), (32, 64)] {
        let v = format!("n{n}_b{nb}");
        let st = [1, n as isize, (n * n) as isize];
        let a = fill::<T>(n * n * nb, 6);
        let mut h = Vec::with_capacity(n * n * nb);
        for i in 0..nb {
            h.extend(hpd::<T>(n, 10 + i as u64));
        }
        let b0 = fill::<T>(n * nb, 7);
        let check = |s: &[Status]| assert!(s.iter().all(|x| *x == Status::Ok), "batched status");
        cfg.time(&format!("batched_solve_{t}"), &v, n, || {
            let mut b = b0.clone();
            let av = StridedView::new(&a, &[n, n, nb], &st, 0).expect("a");
            let mut bv = StridedViewMut::new(&mut b, &[n, 1, nb], &[1, n as isize, n as isize], 0)
                .expect("b");
            black_box(batched::solve(exec, &av, &mut bv).expect("solve"));
        });
        cfg.time(&format!("batched_cholesky_{t}"), &v, n, || {
            let mut w = h.clone();
            let mut wv = StridedViewMut::new(&mut w, &[n, n, nb], &st, 0).expect("h");
            check(&batched::cholesky(exec, &mut wv).expect("chol"));
        });
        cfg.time(&format!("batched_eigh_{t}"), &v, n, || {
            let hv = StridedView::new(&h, &[n, n, nb], &st, 0).expect("h");
            let mut w = vec![tensorcontract::Real::from_f64(0.0); n * nb];
            let mut vecs = vec![<T as Element>::zero(); n * n * nb];
            let mut wv = StridedViewMut::new(&mut w, &[n, nb], &[1, n as isize], 0).expect("w");
            let mut vv = StridedViewMut::new(&mut vecs, &[n, n, nb], &st, 0).expect("v");
            check(&batched::eigh::<T>(exec, &hv, &mut wv, Some(&mut vv)).expect("eigh"));
        });
        cfg.time(&format!("batched_svd_{t}"), &v, n, || {
            let av = StridedView::new(&a, &[n, n, nb], &st, 0).expect("a");
            let mut s = vec![tensorcontract::Real::from_f64(0.0); n * nb];
            let mut sv = StridedViewMut::new(&mut s, &[n, nb], &[1, n as isize], 0).expect("s");
            check(&batched::svd::<T>(exec, &av, &mut sv, None).expect("svd"));
        });
    }
}

fn main() {
    let threads = BenchThreads::from_args();
    threads.verify();
    let cfg = Cfg {
        threads: threads.requested,
        warmup: env_usize("BENCH_WARMUP", 3),
        runs: env_usize("BENCH_RUNS", 30),
        filter: std::env::var("BENCH_FILTER").ok().filter(|f| !f.is_empty()),
        exact: std::env::var("BENCH_CASE").ok().filter(|f| !f.is_empty()),
    };
    println!("case,variant,threads,median_ns,samples");
    threads.with_exec(|exec, _| {
        single::<f64>(&cfg, exec);
        single::<Complex64>(&cfg, exec);
        batch::<f64>(&cfg, exec);
        batch::<Complex64>(&cfg, exec);
    });
}
