//! Micro-tile shape analysis: what a row-block choice would do, without
//! running anything.
//!
//! `MR` is not only a kernel-tuning constant. It is the granularity at which
//! the output's row scatter is blocked, so it decides whether each block of `D`
//! reaches the write-back's unit-stride path or falls back to a gather — and,
//! through `Plan::transposes_gemm`, it is also an input to the row/column
//! orientation. This command evaluates every shape on every kernel menu against
//! every corpus case and reports the two quantities that follow from it:
//!
//! * `wb`, the fraction of output row blocks that stay off the gather path;
//! * `regA`, the block-scatter regularity of the *row operand*, which is what
//!   packing sees, and which a narrower `MR` can move in either direction.
//!
//! It costs no CPU time and touches no data, so it can be run while a benchmark
//! is in flight and its output is exactly reproducible. Use it to decide which
//! measurements are worth making; use `sweep` to make them.

use std::process::ExitCode;

use num_complex::Complex;
use tensorcontract::kernel::{plan_config, ComplexMethod, KernelSet};
use tensorcontract::plan::Operand;
use tensorcontract::scatter::{build_block_scatter, regular_fraction};
use tensorcontract::{Element, Plan};

use crate::corpus::{self, Sized};
use crate::Options;

pub fn run(opts: &Options) -> ExitCode {
    crate::report::print_environment();
    println!();

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
        "row-block menus over {} cases, stress={}\n\n\
         `wb` is the fraction of output row blocks that avoid the write-back's gather\n\
         path at that MR; `regA` is the row operand's block-scatter regularity. `>` marks\n\
         the shape TENSORCONTRACT_ROWBLOCK=auto would pick, `*` the kernel default.\n",
        cases.len(),
        opts.stress.name()
    );

    let mut rows = Vec::new();
    for case in &cases {
        let s = corpus::size_case_stressed(case, opts.tensor_bytes(), opts.stress);
        if opts.wants("f32") {
            collect::<f32>(&s, &mut rows);
        }
        if opts.wants("f64") {
            collect::<f64>(&s, &mut rows);
        }
        if opts.wants("c32") {
            collect::<Complex<f32>>(&s, &mut rows);
        }
        if opts.wants("c64") {
            collect::<Complex<f64>>(&s, &mut rows);
        }
    }

    print_table(&rows);
    if let Some(path) = &opts.csv {
        match write_csv(path, &rows) {
            Ok(()) => println!("\nwrote {path}"),
            Err(e) => eprintln!("failed to write {path}: {e}"),
        }
    }
    ExitCode::SUCCESS
}

/// One shape on one menu, scored for one case.
struct Shape {
    mr: usize,
    nr: usize,
    swap: bool,
    wb: f64,
    reg_a: f64,
    is_default: bool,
    chosen: bool,
}

struct Row {
    case: String,
    dtype: &'static str,
    method: &'static str,
    m: u64,
    n: u64,
    k: u64,
    shapes: Vec<Shape>,
}

impl Row {
    fn default_shape(&self) -> &Shape {
        self.shapes.iter().find(|s| s.is_default).unwrap()
    }
    fn chosen_shape(&self) -> &Shape {
        self.shapes.iter().find(|s| s.chosen).unwrap()
    }
    /// Does the rule move, and does moving change a code path?
    fn moves(&self) -> bool {
        self.chosen_shape().mr != self.default_shape().mr
    }
}

fn collect<T>(s: &Sized, out: &mut Vec<Row>)
where
    T: Element + crate::engines::BenchElem,
    T::Real: KernelSet,
{
    let Ok(plan) = Plan::new(
        Operand::new(&s.la, &s.idx_a),
        Operand::new(&s.lb, &s.idx_b),
        None,
        Operand::new(&s.lc, &s.idx_c),
    ) else {
        return;
    };
    let (m, n, k) = s.mnk();

    for method in ComplexMethod::ALL {
        // A real element type runs the same kernel whatever the method is.
        if !T::IS_COMPLEX && method != ComplexMethod::default() {
            continue;
        }
        let menu = <T::Real as KernelSet>::row_blocks(T::IS_COMPLEX, method);
        if menu.is_empty() {
            continue;
        }
        let pick = plan
            .preferred_row_block(menu)
            .unwrap_or_else(|| menu[0]);
        let shapes = menu
            .iter()
            .map(|&mr| {
                let nr = <T::Real as KernelSet>::config_at(T::IS_COMPLEX, method, mr)
                    .map_or(0, |c| c.ukr.nr);
                let swap = plan.transposes_gemm(mr);
                let sc = plan.oriented_scatters(mr);
                Shape {
                    mr,
                    nr,
                    swap,
                    wb: plan.row_block_score(mr),
                    reg_a: regular_fraction(&build_block_scatter(sc.a_m, mr)),
                    is_default: mr == menu[0],
                    chosen: mr == pick,
                }
            })
            .collect();
        out.push(Row {
            case: s.case.name.to_string(),
            dtype: T::NAME,
            method: if T::IS_COMPLEX { method.name() } else { "real" },
            m,
            n,
            k,
            shapes,
        });
    }
    // Keep the executed configuration honest: whatever the analysis says, this
    // is the shape the engine would use under the current environment.
    if let Some(r) = out.last_mut() {
        let (mr, _, _) = plan_config::<T>(&plan);
        debug_assert!(r.shapes.iter().any(|s| s.mr == mr));
    }
}

fn print_table(rows: &[Row]) {
    let moved: Vec<&Row> = rows.iter().filter(|r| r.moves()).collect();

    println!(
        "{:<22} {:>4} {:>6} {:>7} {:>7} {:>7}  shapes",
        "case", "type", "method", "m", "n", "k"
    );
    for r in rows {
        if !r.moves() {
            continue;
        }
        print!(
            "{:<22} {:>4} {:>6} {:>7} {:>7} {:>7} ",
            r.case, r.dtype, r.method, r.m, r.n, r.k
        );
        for s in &r.shapes {
            let mark = match (s.is_default, s.chosen) {
                (true, true) => "*>",
                (true, false) => " *",
                (false, true) => " >",
                (false, false) => "  ",
            };
            print!(
                "{mark}{}x{} wb={:.2} regA={:.2}   ",
                s.mr, s.nr, s.wb, s.reg_a
            );
        }
        println!();
    }

    println!(
        "\n{} of {} case-dtype-methods would change shape.",
        moved.len(),
        rows.len()
    );

    // The interesting sub-population: those that gain a *code path*, i.e. that
    // go from mostly-gather to mostly-regular, rather than merely improving.
    let onto_fast: Vec<&&Row> = moved
        .iter()
        .filter(|r| r.default_shape().wb < 0.5 && r.chosen_shape().wb >= 0.99)
        .collect();
    println!(
        "{} of those move a majority-gather output onto the regular path outright.",
        onto_fast.len()
    );

    // And the population the rule cannot help, which is where the remaining
    // headroom would have to come from.
    let stuck: Vec<&Row> = rows
        .iter()
        .filter(|r| r.shapes.iter().all(|s| s.wb < 0.99))
        .collect();
    println!(
        "{} have no shape on their menu that clears the gather path at all.",
        stuck.len()
    );

    let mut by_dtype: Vec<(&str, usize, usize)> = Vec::new();
    for r in rows {
        let e = match by_dtype.iter_mut().find(|e| e.0 == r.dtype) {
            Some(e) => e,
            None => {
                by_dtype.push((r.dtype, 0, 0));
                by_dtype.last_mut().unwrap()
            }
        };
        e.1 += 1;
        e.2 += r.moves() as usize;
    }
    println!("\nby dtype:");
    for (d, tot, mv) in by_dtype {
        println!("  {d:<4} {mv:>3} / {tot:<3} would change shape");
    }
}

fn write_csv(path: &str, rows: &[Row]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "case,dtype,method,m,n,k,mr,nr,orient,wb,reg_a,default,chosen")?;
    for r in rows {
        for s in &r.shapes {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{:.6},{:.6},{},{}",
                r.case,
                r.dtype,
                r.method,
                r.m,
                r.n,
                r.k,
                s.mr,
                s.nr,
                if s.swap { "BA" } else { "AB" },
                s.wb,
                s.reg_a,
                s.is_default,
                s.chosen,
            )?;
        }
    }
    Ok(())
}
