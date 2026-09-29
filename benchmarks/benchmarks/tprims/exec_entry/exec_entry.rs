//! Entry cost of tprims-exec primitives, and strided / tensorcontract kernels
//! driven through `Exec`, at an enforced thread count.
//!
//! Usage: `exec_entry --threads N`. CSV `case,variant,threads,median_ns,samples`
//! plus `# stats` lines with pool entry counters. Environment: `BENCH_RUNS`
//! (default 200), `BENCH_WARMUP` (default 20), `BENCH_FILTER`.
use std::hint::black_box;

use strided_basic::{map_into, StridedArray};
use tensorcontract::spmd::Spmd;
use tensorcontract::{Layout, Operand, Plan, TensorView, TensorViewMut};
use tprims_bench::threads::BenchThreads;
use tprims_bench::timing::{env_usize, median_ns};
use tprims_exec::strided::run_with_exec;
use tprims_exec::{Exec, Pool};

struct ExecSpmd<'a> {
    exec: &'a Exec<'a>,
    width: usize,
}

impl Spmd for ExecSpmd<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn broadcast(&self, p: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
        self.exec.broadcast(p, f).is_ok()
    }
}

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
    fn row(&self, case: &str, variant: &str, ns: f64, runs: usize) {
        println!("{case},{variant},{},{ns:.0},{runs}", self.threads);
    }
}

fn stats(case: &str, pool: Option<&Pool<'_>>) {
    if let Some(p) = pool {
        let s = p.stats();
        println!(
            "# stats {case}: entries={} broadcasts={} inline={}",
            s.entries, s.broadcasts, s.inline_runs
        );
        p.reset_stats();
    }
}

fn entry_cases(cfg: &Cfg, exec: &Exec<'_>, pool: Option<&Pool<'_>>) {
    for k in [1usize, 2, 4] {
        if k > cfg.threads || !cfg.want("install_empty") {
            continue;
        }
        let ns = median_ns(cfg.warmup, cfg.runs, || {
            black_box(exec.install(k, |par| black_box(par.threads())));
        });
        cfg.row("install_empty", &format!("k{k}"), ns, cfg.runs);
        stats("install_empty", pool);
    }
    for k in [2usize, 4] {
        if k > cfg.threads || !cfg.want("partition_empty") {
            continue;
        }
        let ns = median_ns(cfg.warmup, cfg.runs, || {
            exec.for_each_partition(k, &|i| {
                black_box(i);
            })
        });
        cfg.row("partition_empty", &format!("k{k}"), ns, cfg.runs);
        stats("partition_empty", pool);
    }
    if cfg.threads > 1 && cfg.want("broadcast_empty") {
        let w = cfg.threads;
        let ns = median_ns(cfg.warmup, cfg.runs, || {
            exec.broadcast(w, &|t| {
                black_box(t);
            })
            .expect("broadcast");
        });
        cfg.row("broadcast_empty", &format!("w{w}"), ns, cfg.runs);
        stats("broadcast_empty", pool);
    }
}

fn strided_cases(cfg: &Cfg, exec: &Exec<'_>, pool: Option<&Pool<'_>>) {
    if !cfg.want("strided_map_f64") {
        return;
    }
    for log2 in [12u32, 22] {
        let n = 1usize << log2;
        let rows = 1usize << (log2 / 2);
        let dims = [rows, n / rows];
        let src = StridedArray::<f64>::from_fn_col_major(&dims, |i| (i[0] + 3 * i[1]) as f64);
        let src_view = src.view();
        let src_t = src_view.permute(&[1, 0]).expect("permute");
        let mut dst = StridedArray::<f64>::col_major(&[dims[1], dims[0]]);
        let runs = if log2 >= 20 {
            cfg.runs.min(30)
        } else {
            cfg.runs
        };
        let ns = median_ns(cfg.warmup.min(5), runs, || {
            run_with_exec(exec, n, |_| {
                map_into(&mut dst.view_mut(), &src_t, |x| x * 1.5 + 1.0).expect("map")
            });
        });
        black_box(&dst);
        cfg.row("strided_map_f64", &format!("transpose_2^{log2}"), ns, runs);
        stats("strided_map_f64", pool);
    }
}

fn gemm_case(cfg: &Cfg, exec: &Exec<'_>, pool: Option<&Pool<'_>>) {
    if !cfg.want("tensorcontract_gemm_f64") {
        return;
    }
    let n = 512usize;
    let a: Vec<f64> = (0..n * n).map(|x| (x % 13) as f64 - 6.0).collect();
    let b: Vec<f64> = (0..n * n).map(|x| (x % 7) as f64 * 0.5).collect();
    let l = Layout::col_major(&[n as i64, n as i64]);
    let (ia, ib, id) = ([0i64, 2], [2i64, 1], [0i64, 1]);
    let plan = Plan::new(
        Operand::new(&l, &ia),
        Operand::new(&l, &ib),
        None,
        Operand::new(&l, &id),
    )
    .expect("plan")
    .with_threads(1);
    let mut reference = vec![0.0; n * n];
    plan.run(
        1.0,
        TensorView::new(&a, &l, &ia),
        TensorView::new(&b, &l, &ib),
        0.0,
        None,
        TensorViewMut::new(&mut reference, &l, &id),
    )
    .expect("reference");
    let spmd = ExecSpmd {
        exec,
        width: exec.budget(),
    };
    let mut d = vec![0.0; n * n];
    let runs = cfg.runs.min(20);
    let ns = median_ns(cfg.warmup.min(3), runs, || {
        plan.run_with(
            &spmd,
            1.0,
            TensorView::new(&a, &l, &ia),
            TensorView::new(&b, &l, &ib),
            0.0,
            None,
            TensorViewMut::new(&mut d, &l, &id),
        )
        .expect("run");
    });
    let max_diff = d
        .iter()
        .zip(&reference)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    println!(
        "CHECK tensorcontract_gemm_f64 threads={} max_abs_diff={max_diff:e} {}",
        cfg.threads,
        if max_diff == 0.0 { "ok" } else { "MISMATCH" }
    );
    cfg.row("tensorcontract_gemm_f64", "512^3", ns, runs);
    stats("tensorcontract_gemm_f64", pool);
}

fn main() {
    let threads = BenchThreads::from_args();
    threads.verify();
    let cfg = Cfg {
        threads: threads.requested,
        warmup: env_usize("BENCH_WARMUP", 20),
        runs: env_usize("BENCH_RUNS", 200),
        filter: std::env::var("BENCH_FILTER").ok().filter(|f| !f.is_empty()),
    };
    println!("case,variant,threads,median_ns,samples");
    threads.with_exec(|exec, pool| {
        entry_cases(&cfg, exec, pool);
        strided_cases(&cfg, exec, pool);
        gemm_case(&cfg, exec, pool);
    });
}
