//! tprims-contract at an enforced thread count: a predeclared corpus of
//! binary contractions, both strategies (permute + batched GEMM, TBLIS-style
//! direct), f64 and c64. Planning and execution are timed separately.
//!
//! Usage: `contract --threads N`. CSV `case,variant,threads,median_ns,samples`
//! where `variant` is `<strategy>_<plan|exec>`; `# selected` lines report
//! what each plan runs (including materialized operands); `CHECK` lines
//! compare the strategies' results. Environment: `BENCH_RUNS` (default 20),
//! `BENCH_WARMUP` (3), `BENCH_CASE` (exact case name).
use std::hint::black_box;

use num_complex::Complex64;
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_bench::threads::BenchThreads;
use tprims_bench::timing::{env_usize, median_ns};
use tprims_blas::{Conj, Scalar};
use tprims_contract::{ContractPlan, DotGeneral, Flags, Strategy};
use tprims_exec::Exec;

/// One corpus entry. `a_order` / `b_order` give the storage order of the
/// logical axes (fastest first), so operands can be stored "transposed".
pub struct Case {
    pub name: &'static str,
    pub a: &'static [usize],
    pub b: &'static [usize],
    pub a_order: &'static [usize],
    pub b_order: &'static [usize],
    pub lc: &'static [usize],
    pub rc: &'static [usize],
    pub lb: &'static [usize],
    pub rb: &'static [usize],
}

/// The predeclared corpus (also listed in README.md).
pub const CORPUS: &[Case] = &[
    Case {
        name: "tiny_matmul",
        a: &[2, 2],
        b: &[2, 2],
        a_order: &[0, 1],
        b_order: &[0, 1],
        lc: &[1],
        rc: &[0],
        lb: &[],
        rb: &[],
    },
    Case {
        name: "matmul_256",
        a: &[256, 256],
        b: &[256, 256],
        a_order: &[0, 1],
        b_order: &[0, 1],
        lc: &[1],
        rc: &[0],
        lb: &[],
        rb: &[],
    },
    Case {
        name: "batched_64_b32",
        a: &[64, 64, 32],
        b: &[64, 64, 32],
        a_order: &[0, 1, 2],
        b_order: &[0, 1, 2],
        lc: &[1],
        rc: &[0],
        lb: &[2],
        rb: &[2],
    },
    Case {
        name: "permuted_fusable",
        a: &[64, 32, 32],
        b: &[32, 32, 64],
        a_order: &[2, 1, 0],
        b_order: &[2, 0, 1],
        lc: &[1, 2],
        rc: &[1, 0],
        lb: &[],
        rb: &[],
    },
    // A's contracted axes 1 and 2 are separated in storage by free axis 0: A must be copied.
    Case {
        name: "permuted_nonfusable",
        a: &[64, 32, 32],
        b: &[32, 32, 64],
        a_order: &[1, 0, 2],
        b_order: &[1, 0, 2],
        lc: &[1, 2],
        rc: &[1, 0],
        lb: &[],
        rb: &[],
    },
    Case {
        name: "network_ijkl_klmn",
        a: &[16, 16, 16, 16],
        b: &[16, 16, 16, 16],
        a_order: &[0, 1, 2, 3],
        b_order: &[0, 1, 2, 3],
        lc: &[2, 3],
        rc: &[0, 1],
        lb: &[],
        rb: &[],
    },
    Case {
        name: "large_ijk_jkl",
        a: &[256, 64, 64],
        b: &[64, 64, 256],
        a_order: &[0, 1, 2],
        b_order: &[0, 1, 2],
        lc: &[1, 2],
        rc: &[0, 1],
        lb: &[],
        rb: &[],
    },
];

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

/// Strides for storing logical `dims` in `order` (fastest first).
fn strides(dims: &[usize], order: &[usize]) -> Vec<isize> {
    let mut s = vec![0isize; dims.len()];
    let mut acc = 1isize;
    for &a in order {
        s[a] = acc;
        acc *= dims[a] as isize;
    }
    s
}

fn run<T: Scalar>(case: &Case, exec: &Exec<'_>, threads: usize, warmup: usize, runs: usize) {
    let tag = if T::IS_COMPLEX_SCALAR { "c64" } else { "f64" };
    let name = format!("{}_{tag}", case.name);
    let (sa, sb) = (strides(case.a, case.a_order), strides(case.b, case.b_order));
    let ad: Vec<T> = fill(case.a.iter().product(), 1);
    let bd: Vec<T> = fill(case.b.iter().product(), 2);
    let cfg = DotGeneral::new(case.lc, case.rc, case.lb, case.rb);
    let cdims = cfg.validate(case.a, case.b).expect("config").out_dims;
    let sc = strides(&cdims, &(0..cdims.len()).collect::<Vec<_>>());
    let clen: usize = cdims.iter().product();
    let av = StridedView::new(&ad, case.a, &sa, 0).expect("a");
    let bv = StridedView::new(&bd, case.b, &sb, 0).expect("b");
    let mut outs = Vec::new();
    for (tag, strategy) in [("pg", Strategy::PermuteGemm), ("tblis", Strategy::Tblis)] {
        let mk = || {
            ContractPlan::<T>::new(
                &cfg,
                (case.a, &sa),
                (case.b, &sb),
                (&cdims, &sc),
                (Conj::No, Conj::No),
                strategy,
                Flags::default(),
            )
            .expect("plan")
        };
        let ns = median_ns(warmup, runs, || {
            black_box(mk());
        });
        println!("{name},{tag}_plan,{threads},{ns:.0},{runs}");
        let plan = mk();
        println!("# selected {name} {tag}: {:?}", plan.selected());
        let mut c = vec![<T as Element>::zero(); clen];
        let ns = median_ns(warmup, runs, || {
            let mut cv = StridedViewMut::new(&mut c, &cdims, &sc, 0).expect("c");
            plan.execute(
                exec,
                <T as Element>::one(),
                &av,
                &bv,
                <T as Element>::zero(),
                &mut cv,
            )
            .expect("exec");
        });
        println!("{name},{tag}_exec,{threads},{ns:.0},{runs}");
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
    println!(
        "CHECK {name} threads={threads} pg_vs_tblis_rel={err:e} {}",
        if err < 1e-12 { "ok" } else { "MISMATCH" }
    );
}

fn main() {
    let threads = BenchThreads::from_args();
    threads.verify();
    let warmup = env_usize("BENCH_WARMUP", 3);
    let runs = env_usize("BENCH_RUNS", 20);
    let exact = std::env::var("BENCH_CASE").ok().filter(|f| !f.is_empty());
    println!("case,variant,threads,median_ns,samples");
    threads.with_exec(|exec, _| {
        for case in CORPUS {
            for complex in [false, true] {
                let full = format!("{}_{}", case.name, if complex { "c64" } else { "f64" });
                if exact.as_deref().is_some_and(|e| e != full) {
                    continue;
                }
                let r = if case.name.starts_with("large") || case.name.starts_with("matmul") {
                    runs.min(10)
                } else {
                    runs
                };
                if complex {
                    run::<Complex64>(case, exec, threads.requested, warmup, r);
                } else {
                    run::<f64>(case, exec, threads.requested, warmup, r);
                }
            }
        }
    });
}
