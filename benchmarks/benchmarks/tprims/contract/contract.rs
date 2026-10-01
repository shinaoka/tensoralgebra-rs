//! tprims-contract at an enforced thread count: a predeclared corpus of
//! binary contractions, or a recorded one (`--corpus FILE`, see
//! `tprims_bench::corpus`), both strategies (permute + batched GEMM,
//! TBLIS-style direct). Planning and execution are timed separately.
//!
//! Usage: `contract --threads N [--corpus FILE] [--list]`. CSV
//! `case,variant,threads,median_ns,samples` where `variant` is
//! `<strategy>_<plan|exec>`; `# selected` lines report what each plan runs
//! (including materialized operands); `CHECK` lines compare the strategies'
//! results. `--list` prints the case names and exits. Environment:
//! `BENCH_RUNS` (default 20), `BENCH_WARMUP` (3), `BENCH_CASE` (exact case
//! name). The built-in corpus runs in f64 and c64; a corpus file sets each
//! entry's dtype.
use std::hint::black_box;

use num_complex::{Complex32, Complex64};
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_bench::corpus::{Corpus, DotGeneralEntry, Dtype, Entry, Operand};
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

/// A built-in case as a corpus entry: operands stored in `a_order` /
/// `b_order`, output compact column-major.
fn entry(case: &Case, dtype: Dtype) -> DotGeneralEntry {
    let cfg = DotGeneral::new(case.lc, case.rc, case.lb, case.rb);
    let cdims = cfg.validate(case.a, case.b).expect("config").out_dims;
    let order: Vec<usize> = (0..cdims.len()).collect();
    DotGeneralEntry {
        name: format!("{}_{}", case.name, dtype.tag()),
        dtype,
        a: Operand {
            dims: case.a.to_vec(),
            strides: strides(case.a, case.a_order),
        },
        b: Operand {
            dims: case.b.to_vec(),
            strides: strides(case.b, case.b_order),
        },
        c: Operand {
            strides: strides(&cdims, &order),
            dims: cdims,
        },
        lc: case.lc.to_vec(),
        rc: case.rc.to_vec(),
        lb: case.lb.to_vec(),
        rb: case.rb.to_vec(),
        conj: [false, false],
        calls: None,
        time_share: None,
    }
}

fn conj(c: bool) -> Conj {
    if c {
        Conj::Yes
    } else {
        Conj::No
    }
}

fn run<T: Scalar>(
    e: &DotGeneralEntry,
    exec: &Exec<'_>,
    threads: usize,
    warmup: usize,
    runs: usize,
) {
    let name = &e.name;
    let ((oa, la), (ob, lb), (oc, lc)) = (e.a.span(), e.b.span(), e.c.span());
    let ad: Vec<T> = fill(la, 1);
    let bd: Vec<T> = fill(lb, 2);
    let cfg = DotGeneral::new(&e.lc, &e.rc, &e.lb, &e.rb);
    let av = StridedView::new(&ad, &e.a.dims, &e.a.strides, oa as isize).expect("a");
    let bv = StridedView::new(&bd, &e.b.dims, &e.b.strides, ob as isize).expect("b");
    let mut outs = Vec::new();
    // `--partition dynamic:JM,JN` adds a separately labelled TBLIS row that uses
    // the opt-in dynamic scheduler; the pg and static tblis rows are unchanged.
    let dynamic = tprims_bench::partition::from_args();
    let mut variants = vec![
        ("pg".to_string(), Strategy::PermuteGemm, None),
        ("tblis".to_string(), Strategy::Tblis, None),
    ];
    if dynamic.is_some() {
        variants.push((
            format!("tblis{}", tprims_bench::partition::suffix(dynamic)),
            Strategy::Tblis,
            dynamic,
        ));
    }
    for (tag, strategy, partition) in variants.iter().map(|(t, s, p)| (t.as_str(), *s, *p)) {
        let gemm = tprims_bench::partition::apply(
            tprims_blas::GemmConfig {
                engine: if partition.is_some() {
                    tprims_blas::EngineChoice::Packed
                } else {
                    Default::default()
                },
                ..Default::default()
            },
            partition,
        );
        let mk = || {
            ContractPlan::<T>::new_with(
                &gemm,
                &cfg,
                (&e.a.dims, &e.a.strides),
                (&e.b.dims, &e.b.strides),
                (&e.c.dims, &e.c.strides),
                (conj(e.conj[0]), conj(e.conj[1])),
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
        let mut c = vec![<T as Element>::zero(); lc];
        let ns = median_ns(warmup, runs, || {
            let mut cv =
                StridedViewMut::new(&mut c, &e.c.dims, &e.c.strides, oc as isize).expect("c");
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
    let tol = if matches!(e.dtype, Dtype::F32 | Dtype::C32) {
        1e-4
    } else {
        1e-12
    };
    println!(
        "CHECK {name} threads={threads} pg_vs_tblis_rel={err:e} {}",
        if err < tol { "ok" } else { "MISMATCH" }
    );
}

fn dispatch(e: &DotGeneralEntry, exec: &Exec<'_>, threads: usize, warmup: usize, runs: usize) {
    match e.dtype {
        Dtype::F32 => run::<f32>(e, exec, threads, warmup, runs),
        Dtype::F64 => run::<f64>(e, exec, threads, warmup, runs),
        Dtype::C32 => run::<Complex32>(e, exec, threads, warmup, runs),
        Dtype::C64 => run::<Complex64>(e, exec, threads, warmup, runs),
    }
}

/// Built-in cases (f64 and c64) or the `dot_general` entries of `--corpus`.
fn cases() -> Vec<DotGeneralEntry> {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == "--corpus") {
        Some(i) => {
            let path = args
                .get(i + 1)
                .unwrap_or_else(|| panic!("--corpus needs a file"));
            let corpus = Corpus::load(path).unwrap_or_else(|e| panic!("{e}"));
            corpus
                .entries
                .into_iter()
                .filter_map(|e| match e {
                    Entry::DotGeneral(d) => Some(d),
                    _ => None,
                })
                .collect()
        }
        None => CORPUS
            .iter()
            .flat_map(|c| [entry(c, Dtype::F64), entry(c, Dtype::C64)])
            .collect(),
    }
}

fn main() {
    let cases = cases();
    if std::env::args().any(|a| a == "--list") {
        for c in &cases {
            println!("{}", c.name);
        }
        return;
    }
    let threads = BenchThreads::from_args();
    threads.verify();
    let warmup = env_usize("BENCH_WARMUP", 3);
    let runs = env_usize("BENCH_RUNS", 20);
    let exact = std::env::var("BENCH_CASE").ok().filter(|f| !f.is_empty());
    println!("case,variant,threads,median_ns,samples");
    threads.with_exec(|exec, _| {
        for e in &cases {
            if exact.as_deref().is_some_and(|x| x != e.name) {
                continue;
            }
            let mults: usize = e.c.dims.iter().product::<usize>()
                * e.lc.iter().map(|&i| e.a.dims[i]).product::<usize>();
            let r = if mults >= 1 << 24 { runs.min(10) } else { runs };
            dispatch(e, exec, threads.requested, warmup, r);
        }
    });
}
