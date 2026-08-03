//! The TCCG corpus sweep: every case, every requested dtype, every engine.

use std::process::ExitCode;

use num_complex::Complex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensorcontract::kernel::{plan_config, selected_kernel_name, ComplexMethod, KernelSet};
use tensorcontract::plan::Operand;
use tensorcontract::scatter::{build_block_scatter, regular_fraction};
use tensorcontract::Plan;

use super::{gflops, pin_single_threaded, rel_error, timed, BenchElem};
use crate::corpus::{self, Sized};
use crate::report::{Results, Row, Table};
#[cfg(feature = "blas")]
use crate::ttgt::{ttgt, TtgtScratch};
use crate::Options;

/// Column order for the report tables: our three complex methods first, then
/// the external baselines.
pub const ENGINE_ORDER: &[&str] = &["planar", "1m", "3m", "ttgt", "tblis"];

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

    // Regularity is reported against the orientation the engine actually
    // executes in, which is not necessarily `A`-rows / `B`-columns.
    // `MR` is element-type and method dependent — and since Phase 4.1c, plan
    // dependent too — and so is the orientation. Each engine row therefore
    // reports against its own `MR`; the baselines' rows use the default
    // method's, so that column means one thing per row.
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

    let push =
        |engine: &str, secs: f64, reg_a: f64, reg_b: f64, notes: String, results: &mut Results| {
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

    // ---- our engine, once per complex method -----------------------------
    //
    // For a real element type all three methods reduce to the same real path,
    // so it is measured once and reported under each requested engine name.
    // That keeps the complex-over-real efficiency ratio well defined for every
    // method without pretending to have measured the real path three times.
    let mut real_measured: Option<(f64, f64, f64, String)> = None;
    for method in ComplexMethod::ALL {
        let name = method.name();
        if !opts.engine(name) {
            continue;
        }
        if let Some((secs, ra, rb, notes)) = &real_measured {
            push(name, *secs, *ra, *rb, notes.clone(), results);
            continue;
        }

        let p = plan.clone().with_complex_method(method);
        let (mr, nr, blk) = plan_config::<T>(&p);
        let psc = p.oriented_scatters(mr);
        let orient = if p.transposes_gemm(mr) { "BA" } else { "AB" };
        let reg_a = regular_fraction(&build_block_scatter(psc.a_m, mr));
        let reg_b = regular_fraction(&build_block_scatter(psc.b_n, nr));

        let secs = timed(opts.reps, || unsafe {
            p.run_raw::<T>(
                T::one(),
                a.as_ptr(),
                b.as_ptr(),
                T::zero(),
                d.as_ptr(),
                d.as_mut_ptr(),
            )
        });
        // `MR x NR` goes in the notes because it is no longer a constant per
        // dtype and method: a CSV without it cannot be re-read later. So does
        // the blocking, for the same reason — the item 2 grid varies it per arm
        // and a coupled `kc` gives each dtype and method a different `mc`.
        // The thread count appears only when it is not 1, so single-core CSVs
        // keep the exact format every committed measurement was written in.
        let threads = p.threads();
        let tag = if threads == 1 {
            String::new()
        } else {
            format!("t{threads}/{} ", p.strips(mr))
        };
        let notes = format!(
            "{} {mr}x{nr} {orient} {}x{}x{} {tag}{}",
            selected_kernel_name::<T>(method),
            blk.mc,
            blk.kc,
            blk.nc,
            check(name, &d, &mut reference)
        )
        .trim()
        .to_string();
        push(name, secs, reg_a, reg_b, notes.clone(), results);
        if !T::IS_COMPLEX {
            real_measured = Some((secs, reg_a, reg_b, notes));
        }
    }

    // Regularity for the baselines' rows: report at the default method's
    // register block so the column means one thing per row. (Only consumed
    // when a baseline feature is enabled.)
    #[cfg(any(feature = "blas", feature = "tblis"))]
    let (reg_a, reg_b) = {
        let (mr, nr, _) = plan_config::<T>(&plan);
        let sc = plan.oriented_scatters(mr);
        (
            regular_fraction(&build_block_scatter(sc.a_m, mr)),
            regular_fraction(&build_block_scatter(sc.b_n, nr)),
        )
    };

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
        push("ttgt", secs, reg_a, reg_b, notes, results);
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
        push("tblis", secs, reg_a, reg_b, notes, results);
    }
}

fn print_tables(results: &Results, opts: &Options) {
    let engines: Vec<&str> = ENGINE_ORDER
        .iter()
        .copied()
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
    let engines: Vec<&str> = ENGINE_ORDER
        .iter()
        .copied()
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
