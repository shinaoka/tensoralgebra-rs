//! The TCCG corpus sweep: every case, every requested dtype, every engine.

use std::process::ExitCode;

use num_complex::Complex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensorcontract::kernel::{selected_config, selected_kernel_name, KernelSet};
use tensorcontract::plan::Operand;
use tensorcontract::scatter::{build_block_scatter, regular_fraction};
use tensorcontract::Plan;

use super::{gflops, pin_single_threaded, rel_error, timed, BenchElem};
use crate::corpus::{self, Sized};
use crate::report::{Results, Row, Table};
#[cfg(feature = "blas")]
use crate::ttgt::{ttgt, TtgtScratch};
use crate::Options;

pub fn run(opts: &Options) -> ExitCode {
    pin_single_threaded();
    crate::report::print_environment();
    println!();

    let mut results = Results::default();
    let cases: Vec<_> = corpus::corpus()
        .into_iter()
        .filter(|c| {
            opts.case_filter
                .as_ref()
                .map(|f| c.name.contains(f.as_str()))
                .unwrap_or(true)
        })
        .collect();

    println!(
        "sweeping {} cases at {} MiB nominal tensor size, {} reps, stress={}\n",
        cases.len(),
        opts.size_mib,
        opts.reps,
        opts.stress.name()
    );

    for case in &cases {
        let s = corpus::size_case_stressed(case, opts.tensor_bytes(), opts.stress);
        if opts.wants("f32") {
            run_case::<f32>(&s, opts, &mut results);
        }
        if opts.wants("f64") {
            run_case::<f64>(&s, opts, &mut results);
        }
        if opts.wants("c32") {
            run_case::<Complex<f32>>(&s, opts, &mut results);
        }
        if opts.wants("c64") {
            run_case::<Complex<f64>>(&s, opts, &mut results);
        }
    }

    print_tables(&results, opts);
    if let Some(path) = &opts.csv {
        match results.write_csv(path) {
            Ok(()) => println!("\nwrote {path}"),
            Err(e) => eprintln!("failed to write {path}: {e}"),
        }
    }
    ExitCode::SUCCESS
}

/// Measure one case for one element type across all requested engines.
pub fn run_case<T>(s: &Sized, opts: &Options, results: &mut Results)
where
    T: BenchElem,
    T::Real: KernelSet,
{
    let plan = match Plan::new(
        Operand::new(&s.la, &s.idx_a),
        Operand::new(&s.lb, &s.idx_b),
        None,
        Operand::new(&s.lc, &s.idx_c),
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{} [{}]: planning failed: {e}", s.case.name, T::NAME);
            return;
        }
    };

    let (m, n, k) = s.mnk();
    let macs = s.macs();

    let (mr, nr, _blk) = selected_config::<T>();
    let sc = plan.scatters();
    let reg_a = regular_fraction(&build_block_scatter(sc.a_m, mr));
    let reg_b = regular_fraction(&build_block_scatter(sc.b_n, nr));

    let mut rng = ChaCha8Rng::seed_from_u64(0x5EED);
    let a: Vec<T> = (0..s.elems_a()).map(|_| T::sample(&mut rng)).collect();
    let b: Vec<T> = (0..s.elems_b()).map(|_| T::sample(&mut rng)).collect();
    let mut d: Vec<T> = vec![T::zero(); s.elems_c()];

    let mut reference: Option<Vec<T>> = None;
    let check = |name: &str, d: &Vec<T>, reference: &mut Option<Vec<T>>| -> String {
        match reference {
            None => {
                *reference = Some(d.clone());
                String::new()
            }
            Some(r) => {
                let err = rel_error(d, r);
                let tol = if core::mem::size_of::<T::Real>() == 4 {
                    1e-3
                } else {
                    1e-10
                };
                if err > tol {
                    format!("MISMATCH({name}) rel_err={err:.2e}")
                } else {
                    String::new()
                }
            }
        }
    };

    let push = |engine: &str, secs: f64, notes: String, results: &mut Results| {
        results.push(Row {
            case: s.case.name.to_string(),
            group: s.case.group.to_string(),
            dtype: T::NAME.to_string(),
            engine: engine.to_string(),
            m,
            n,
            k,
            macs,
            secs,
            gflops: gflops::<T>(macs, secs),
            reg_a,
            reg_b,
            notes,
        });
    };

    // ---- our engine ------------------------------------------------------
    if opts.engine("planar") {
        let secs = timed(opts.reps, || unsafe {
            plan.run_raw::<T>(
                T::one(),
                a.as_ptr(),
                b.as_ptr(),
                T::zero(),
                d.as_ptr(),
                d.as_mut_ptr(),
            )
        });
        let notes = check("planar", &d, &mut reference);
        push(
            "planar",
            secs,
            format!("{} {notes}", selected_kernel_name::<T>())
                .trim()
                .to_string(),
            results,
        );
    }

    // ---- TTGT ------------------------------------------------------------
    #[cfg(feature = "blas")]
    if opts.engine("ttgt") {
        let mut scratch = TtgtScratch::<T>::new(&plan);
        let mut dt: Vec<T> = vec![T::zero(); s.elems_c()];
        let secs = timed(opts.reps, || {
            ttgt(
                &plan,
                T::one(),
                &a,
                &b,
                T::zero(),
                &dt.clone(),
                &mut dt,
                &mut scratch,
            )
        });
        let notes = check("ttgt", &dt, &mut reference);
        push("ttgt", secs, notes, results);
    }
    // ---- TBLIS -----------------------------------------------------------
    #[cfg(feature = "tblis")]
    if opts.engine("tblis") {
        use crate::tblis as tb;
        let mut oa = tb::Operand::new(&s.la.extents, &s.la.strides, s.case.a);
        let mut ob = tb::Operand::new(&s.lb.extents, &s.lb.strides, s.case.b);
        let mut oc = tb::Operand::new(&s.lc.extents, &s.lc.strides, s.case.c);
        let mut dt: Vec<T> = vec![T::zero(); s.elems_c()];
        let ta = oa.tensor(
            T::TBLIS_TYPE,
            T::tblis_scalar(1.0),
            a.as_ptr() as *mut std::ffi::c_void,
        );
        let tbv = ob.tensor(
            T::TBLIS_TYPE,
            T::tblis_scalar(1.0),
            b.as_ptr() as *mut std::ffi::c_void,
        );
        let secs = timed(opts.reps, || {
            let mut tc = oc.tensor(
                T::TBLIS_TYPE,
                T::tblis_scalar(0.0),
                dt.as_mut_ptr() as *mut std::ffi::c_void,
            );
            unsafe {
                tb::tblis_tensor_mult(
                    std::ptr::null(),
                    std::ptr::null(),
                    &ta,
                    oa.labels(),
                    &tbv,
                    ob.labels(),
                    &mut tc,
                    oc.labels(),
                )
            }
        });
        let notes = check("tblis", &dt, &mut reference);
        push("tblis", secs, notes, results);
    }
}

fn print_tables(results: &Results, opts: &Options) {
    let engines: Vec<&str> = ["planar", "ttgt", "tblis"]
        .into_iter()
        .filter(|e| results.rows.iter().any(|r| &r.engine == e))
        .collect();

    for dtype in ["f32", "f64", "c32", "c64"] {
        if !opts.wants(dtype) {
            continue;
        }
        let mut headers = vec![
            "case".to_string(),
            "m".into(),
            "n".into(),
            "k".into(),
            "regA".into(),
            "regB".into(),
        ];
        for e in &engines {
            headers.push(format!("{e} GF/s"));
        }
        let hrefs: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
        let mut t = Table::new(&hrefs);
        let mut any = false;
        for case in crate::corpus::corpus() {
            let Some(first) = results.get(case.name, dtype, engines.first().copied().unwrap_or(""))
            else {
                continue;
            };
            any = true;
            let mut cells = vec![
                case.name.to_string(),
                first.m.to_string(),
                first.n.to_string(),
                first.k.to_string(),
                format!("{:.2}", first.reg_a),
                format!("{:.2}", first.reg_b),
            ];
            for e in &engines {
                cells.push(
                    results
                        .get(case.name, dtype, e)
                        .map(|r| format!("{:.1}", r.gflops))
                        .unwrap_or_else(|| "-".into()),
                );
            }
            t.row(cells);
        }
        if any {
            println!("\n=== {dtype} ===");
            t.print();
        }
    }

    print_ratio_table(results, "f64", "c64");
    print_ratio_table(results, "f32", "c32");
}

/// The headline metric: complex GFLOP/s divided by real GFLOP/s for the same
/// contraction. A value of 1.0 means the engine extracts the same fraction of
/// the machine's FMA throughput from complex data as from real data; anything
/// below that is the complex penalty.
fn print_ratio_table(results: &Results, real: &str, cplx: &str) {
    let engines: Vec<&str> = ["planar", "ttgt", "tblis"]
        .into_iter()
        .filter(|e| results.rows.iter().any(|r| &r.engine == e))
        .collect();
    if engines.is_empty() {
        return;
    }
    let mut headers = vec!["case".to_string()];
    for e in &engines {
        headers.push(format!("{e} {cplx}/{real}"));
    }
    let hrefs: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
    let mut t = Table::new(&hrefs);
    let mut sums = vec![(0.0f64, 0usize); engines.len()];
    let mut any = false;
    for case in crate::corpus::corpus() {
        let mut cells = vec![case.name.to_string()];
        let mut has = false;
        for (i, e) in engines.iter().enumerate() {
            let r = results.get(case.name, real, e);
            let c = results.get(case.name, cplx, e);
            match (r, c) {
                (Some(r), Some(c)) if r.gflops > 0.0 => {
                    let ratio = c.gflops / r.gflops;
                    sums[i].0 += ratio;
                    sums[i].1 += 1;
                    cells.push(format!("{ratio:.3}"));
                    has = true;
                }
                _ => cells.push("-".into()),
            }
        }
        if has {
            any = true;
            t.row(cells);
        }
    }
    if !any {
        return;
    }
    let mut mean = vec!["MEAN".to_string()];
    for (s, n) in &sums {
        mean.push(if *n > 0 {
            format!("{:.3}", s / *n as f64)
        } else {
            "-".into()
        });
    }
    t.row(mean);
    println!("\n=== complex efficiency ratio ({cplx} GF/s over {real} GF/s; 1.0 = no penalty) ===");
    t.print();
}
