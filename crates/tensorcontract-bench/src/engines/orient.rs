//! Row/column orientation analysis: the structural features of both arms.
//!
//! The engine is symmetric under exchanging `(A, M)` with `(B, N)`, so every
//! contraction can be computed either as `D = A B` or as `D^T = B^T A^T`. Which
//! is faster is worth up to 2x and nobody has found the rule: the shipped one
//! (`Plan::transposes_gemm`) is right on 63 of 72 measured `abcijk` case-dtypes
//! and its nine misses are large and systematic (A14).
//!
//! This command emits, for every case and both arms, the structural quantities
//! a discriminant could plausibly be built from — so that candidates can be
//! scored against the forced-arm measurements in `bench-results/phase4d`
//! offline, without running anything. It is the same method that settled the
//! row-block rule in Phase 4.1c, and it exists because the alternative — guess a
//! rule, spend two hours measuring it, learn one bit — is how the last two
//! attempts at this went.
//!
//! Features per arm, all evaluated at the *default* register block, which is
//! what `TENSORCONTRACT_ROWBLOCK=base` pins the measurement to:
//!
//! | feature | why it is a candidate |
//! |---|---|
//! | `row_run` / `row_stride` | the write-back's innermost access; the current rule is `row_stride == 1 && row_run >= MR` |
//! | `col_run` / `col_stride` | the observation in Phase 4 part 4 that `e*bc`'s column direction folds to 576 where `-mb`'s is 24 |
//! | `wb` | fraction of output row blocks off the gather path |
//! | `reg_a` / `reg_b` | packing regularity of the row and column operands — part 4 noted the *faster* arm sometimes has the *worse* one |
//! | `m` / `n` / `k` | the shape itself; `MR`-vs-`m` edge waste differs between arms |

use std::process::ExitCode;

use num_complex::Complex;
use tensorcontract::kernel::{selected_config, ComplexMethod, KernelSet};
use tensorcontract::plan::{Operand, Scatters};
use tensorcontract::scatter::{build_block_scatter, regular_fraction, run_structure};
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

    println!(
        "orientation features for {} cases, {} arm-rows, stress={}\n",
        cases.len(),
        rows.len(),
        opts.stress.name()
    );
    println!(
        "{:<22} {:>4} {:>7} {:>3} {:>8} {:>8} {:>8} {:>8} {:>5} {:>5} {:>5} {:>5}",
        "case", "type", "method", "arm", "m", "n", "row_run", "col_run", "rstr", "wb", "regA",
        "regB"
    );
    for r in &rows {
        println!(
            "{:<22} {:>4} {:>7} {:>3} {:>8} {:>8} {:>8} {:>8} {:>5} {:>5.2} {:>5.2} {:>5.2}{}",
            r.case,
            r.dtype,
            r.method,
            r.arm,
            r.m,
            r.n,
            r.row_run,
            r.col_run,
            r.row_stride,
            r.wb,
            r.reg_a,
            r.reg_b,
            if r.rule_picks { "  <- rule" } else { "" },
        );
    }

    if let Some(path) = &opts.csv {
        match write_csv(path, &rows) {
            Ok(()) => println!("\nwrote {path}"),
            Err(e) => eprintln!("failed to write {path}: {e}"),
        }
    }
    ExitCode::SUCCESS
}

struct Row {
    case: String,
    dtype: &'static str,
    method: &'static str,
    arm: &'static str,
    mr: usize,
    nr: usize,
    m: usize,
    n: usize,
    k: usize,
    row_run: usize,
    row_stride: i64,
    col_run: usize,
    col_stride: i64,
    wb: f64,
    reg_a: f64,
    reg_b: f64,
    rule_picks: bool,
}

/// The two arms of one plan, as (row scatter, column scatter, row operand's row
/// scatter, column operand's column scatter).
fn arm<'a>(s: &Scatters<'a>, swap: bool) -> (&'a [i64], &'a [i64], &'a [i64], &'a [i64]) {
    if swap {
        (s.d_n, s.d_m, s.b_n, s.a_m)
    } else {
        (s.d_m, s.d_n, s.a_m, s.b_n)
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
    let sc = plan.scatters();

    for method in ComplexMethod::ALL {
        if !T::IS_COMPLEX && method != ComplexMethod::default() {
            continue;
        }
        // The default shape, matching `TENSORCONTRACT_ROWBLOCK=base`.
        let (mr, nr, _) = selected_config::<T>(method);
        let picked = plan.transposes_gemm(mr);
        for (swap, name) in [(false, "AB"), (true, "BA")] {
            let (rows, cols, op_rows, op_cols) = arm(&sc, swap);
            let (row_run, row_stride) = run_structure(rows).unwrap_or((rows.len(), 0));
            let (col_run, col_stride) = run_structure(cols).unwrap_or((cols.len(), 0));
            out.push(Row {
                case: s.case.name.to_string(),
                dtype: T::NAME,
                method: if T::IS_COMPLEX { method.name() } else { "real" },
                arm: name,
                mr,
                nr,
                m: rows.len(),
                n: cols.len(),
                k: sc.a_k.len(),
                row_run,
                row_stride,
                col_run,
                col_stride,
                wb: tensorcontract::scatter::unbroken_fraction(rows.len(), row_run, mr),
                reg_a: regular_fraction(&build_block_scatter(op_rows, mr)),
                reg_b: regular_fraction(&build_block_scatter(op_cols, nr)),
                rule_picks: picked == swap,
            });
        }
    }
}

fn write_csv(path: &str, rows: &[Row]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        f,
        "case,dtype,method,arm,mr,nr,m,n,k,row_run,row_stride,col_run,col_stride,wb,reg_a,reg_b,rule_picks"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{}",
            r.case,
            r.dtype,
            r.method,
            r.arm,
            r.mr,
            r.nr,
            r.m,
            r.n,
            r.k,
            r.row_run,
            r.row_stride,
            r.col_run,
            r.col_stride,
            r.wb,
            r.reg_a,
            r.reg_b,
            r.rule_picks,
        )?;
    }
    Ok(())
}
