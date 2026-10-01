//! Cross-implementation verification at realistic sizes.
//!
//! The exhaustive semantic testing (repeated indices, reductions, batch
//! indices, conjugation, negative strides, every loop-nest edge case) lives in
//! `crates/tensorcontract/tests/correctness.rs` and runs against a brute-force
//! oracle. This subcommand does the complementary job: run the *whole TCCG
//! corpus* at benchmark sizes and confirm that this engine, TTGT and TBLIS all
//! agree. That is what catches blocking and packing bugs that only appear once
//! a problem is bigger than one cache block.

use std::process::ExitCode;

use num_complex::Complex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensorcontract::kernel::KernelSet;
use tensorcontract::plan::Operand;
use tensorcontract::{Element, Plan};

#[cfg(any(feature = "blas", feature = "tblis"))]
use super::rel_error;
use super::{pin_single_threaded, BenchElem};
use crate::corpus::{self, Sized};
use crate::report::Table;
use crate::Options;

pub fn run(opts: &Options) -> ExitCode {
    pin_single_threaded();
    crate::report::print_environment();

    let cases: Vec<_> = corpus::corpus()
        .into_iter()
        .filter(|c| {
            opts.case_filter
                .as_ref()
                .map(|f| c.name.contains(f.as_str()))
                .unwrap_or(true)
        })
        .collect();

    // Small enough that every case is quick, large enough to span several
    // cache blocks in at least one dimension.
    println!(
        "\nverifying {} cases at {} MiB nominal size\n",
        cases.len(),
        opts.size_mib
    );

    let mut t = Table::new(&[
        "case", "dtype", "m", "n", "k", "vs ttgt", "vs tblis", "status",
    ]);
    let mut failures = 0usize;

    for case in &cases {
        let s = corpus::size_case_stressed(case, opts.tensor_bytes(), opts.stress);
        if opts.wants("f32") {
            failures += check::<f32>(&s, &mut t);
        }
        if opts.wants("f64") {
            failures += check::<f64>(&s, &mut t);
        }
        if opts.wants("c32") {
            failures += check::<Complex<f32>>(&s, &mut t);
        }
        if opts.wants("c64") {
            failures += check::<Complex<f64>>(&s, &mut t);
        }
    }

    t.print();
    if failures == 0 {
        println!("\nall comparisons within tolerance");
        ExitCode::SUCCESS
    } else {
        println!("\n{failures} comparison(s) FAILED");
        ExitCode::FAILURE
    }
}

fn tol<T: Element>() -> f64 {
    if core::mem::size_of::<T::Real>() == 4 {
        2e-3
    } else {
        1e-10
    }
}

fn check<T>(s: &Sized, t: &mut Table) -> usize
where
    T: BenchElem,
    T::Real: KernelSet,
{
    let plan = match Plan::new(
        Operand::new(&s.la, &s.idx_a),
        Operand::new(&s.lb, &s.idx_b),
        None,
        Operand::new(&s.lc, &s.idx_c),
    )
    .map(|p| crate::knobs::get().apply(p))
    {
        Ok(p) => p,
        Err(e) => {
            t.row(vec![
                s.case.name.into(),
                T::NAME.into(),
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
                format!("PLAN ERROR: {e}"),
            ]);
            return 1;
        }
    };
    let (m, n, k) = s.mnk();

    let mut rng = ChaCha8Rng::seed_from_u64(0xA11CE);
    let a: Vec<T> = (0..s.elems_a()).map(|_| T::sample(&mut rng)).collect();
    let b: Vec<T> = (0..s.elems_b()).map(|_| T::sample(&mut rng)).collect();
    let mut d: Vec<T> = vec![T::zero(); s.elems_c()];

    unsafe {
        plan.run_raw::<T>(
            T::one(),
            a.as_ptr(),
            b.as_ptr(),
            T::zero(),
            d.as_ptr(),
            d.as_mut_ptr(),
        )
    };

    let mut fails = 0usize;

    let ttgt_err: Option<f64> = {
        #[cfg(feature = "blas")]
        {
            let mut scratch = crate::ttgt::TtgtScratch::<T>::new(&plan);
            let mut dt: Vec<T> = vec![T::zero(); s.elems_c()];
            let empty: Vec<T> = Vec::new();
            crate::ttgt::ttgt(
                &plan,
                T::one(),
                &a,
                &b,
                T::zero(),
                &empty,
                &mut dt,
                &mut scratch,
            );
            Some(rel_error(&dt, &d))
        }
        #[cfg(not(feature = "blas"))]
        {
            None
        }
    };

    let tblis_err: Option<f64> = {
        #[cfg(feature = "tblis")]
        {
            use crate::tblis as tb;
            let mut oa = tb::Operand::new(s.la.extents(), s.la.strides(), s.case.a);
            let mut ob = tb::Operand::new(s.lb.extents(), s.lb.strides(), s.case.b);
            let mut oc = tb::Operand::new(s.lc.extents(), s.lc.strides(), s.case.c);
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
            };
            Some(rel_error(&dt, &d))
        }
        #[cfg(not(feature = "tblis"))]
        {
            None
        }
    };

    let fmt = |e: Option<f64>| e.map(|v| format!("{v:.1e}")).unwrap_or_else(|| "-".into());
    let limit = tol::<T>();
    let bad = [ttgt_err, tblis_err]
        .iter()
        .flatten()
        .any(|&e| e.is_nan() || e > limit);
    if bad {
        fails += 1;
    }
    t.row(vec![
        s.case.name.into(),
        T::NAME.into(),
        m.to_string(),
        n.to_string(),
        k.to_string(),
        fmt(ttgt_err),
        fmt(tblis_err),
        if bad { "FAIL".into() } else { "ok".into() },
    ]);
    fails
}
