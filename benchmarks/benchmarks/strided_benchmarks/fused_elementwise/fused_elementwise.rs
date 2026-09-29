//! Benchmark runtime-DAG fused elementwise kernels against per-op reused buffers.
//!
//! Run thread-count variants sequentially; do not run benchmark processes in
//! parallel.

use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_kernel::{
    fused_elementwise_into, map_into, zip_map2_into, FusedInst, FusedOp, FusedPlan, StridedArray,
};

#[derive(Clone, Copy)]
struct BenchStats {
    median_ms: f64,
    iqr_ms: f64,
}

struct CaseResult {
    name: &'static str,
    n: usize,
    elements: usize,
    per_op: BenchStats,
    fused: BenchStats,
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_sizes() -> Vec<usize> {
    std::env::var("SIZES")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|entry| entry.trim().parse().expect("invalid SIZES entry"))
                .collect()
        })
        .unwrap_or_else(|| vec![256, 512, 1024])
}

fn median_iqr(samples: &mut [Duration]) -> BenchStats {
    samples.sort();
    let median = samples[samples.len() / 2].as_secs_f64() * 1e3;
    let p25 = samples[samples.len() / 4].as_secs_f64() * 1e3;
    let p75 = samples[samples.len() * 3 / 4].as_secs_f64() * 1e3;
    BenchStats {
        median_ms: median,
        iqr_ms: p75 - p25,
    }
}

fn bench(mut f: impl FnMut(), warmup: usize, nruns: usize) -> BenchStats {
    for _ in 0..warmup {
        f();
    }

    let mut samples = Vec::with_capacity(nruns);
    for _ in 0..nruns {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
    }
    median_iqr(&mut samples)
}

fn dims(n: usize) -> [usize; 2] {
    [n, n]
}

fn make_input(n: usize, seed: f64) -> StridedArray<f64> {
    let dims = dims(n);
    StridedArray::<f64>::from_fn_col_major(&dims, |idx| {
        seed + 0.001 * idx[0] as f64 + 0.000_01 * idx[1] as f64
    })
}

fn make_constant(n: usize, value: f64) -> StridedArray<f64> {
    let dims = dims(n);
    StridedArray::<f64>::from_fn_col_major(&dims, |_| value)
}

fn check_close(case: &str, expected: &[f64], actual: &[f64]) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "{case}: length mismatch {} != {}",
            expected.len(),
            actual.len()
        ));
    }

    for (i, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
        let tolerance = 1e-10 * lhs.abs().max(1.0);
        if (lhs - rhs).abs() > tolerance {
            return Err(format!(
                "{case}: mismatch at {i}: expected {lhs:?}, actual {rhs:?}, tolerance {tolerance:?}"
            ));
        }
    }
    Ok(())
}

fn add_mul_plan() -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![3],
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![2, 0],
            },
        ],
    }
}

fn exp_mul_add_plan() -> FusedPlan {
    FusedPlan {
        input_count: 3,
        outputs: vec![5],
        ops: vec![
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![3, 2],
            },
            FusedInst {
                op: FusedOp::Exp,
                inputs: vec![4],
            },
        ],
    }
}

fn long_chain_plan() -> FusedPlan {
    FusedPlan {
        input_count: 4,
        outputs: vec![8],
        ops: vec![
            FusedInst {
                op: FusedOp::Divide,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Maximum,
                inputs: vec![4, 2],
            },
            FusedInst {
                op: FusedOp::Minimum,
                inputs: vec![5, 3],
            },
            FusedInst {
                op: FusedOp::Sqrt,
                inputs: vec![6],
            },
            FusedInst {
                op: FusedOp::Rsqrt,
                inputs: vec![7],
            },
        ],
    }
}

fn interpreter_fallback_plan() -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![4],
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Negate,
                inputs: vec![2],
            },
            FusedInst {
                op: FusedOp::Exp,
                inputs: vec![3],
            },
        ],
    }
}

fn validate_add_mul(n: usize) -> Result<(), String> {
    let shape = dims(n);
    let a = make_input(n, 1.0);
    let b = make_input(n, 2.0);
    let mut tmp = StridedArray::<f64>::col_major(&shape);
    let mut expected = StridedArray::<f64>::col_major(&shape);
    let mut actual = StridedArray::<f64>::col_major(&shape);

    zip_map2_into(&mut tmp.view_mut(), &a.view(), &b.view(), |x, y| x + y)
        .map_err(|err| err.to_string())?;
    zip_map2_into(&mut expected.view_mut(), &tmp.view(), &a.view(), |x, y| {
        x * y
    })
    .map_err(|err| err.to_string())?;
    fused_elementwise_into(
        &mut [actual.view_mut()],
        &[a.view(), b.view()],
        &add_mul_plan(),
    )
    .map_err(|err| err.to_string())?;

    check_close("add_mul", expected.data(), actual.data())
}

fn validate_exp_mul_add(n: usize) -> Result<(), String> {
    let shape = dims(n);
    let a = make_input(n, 0.25);
    let b = make_input(n, 0.5);
    let c_scalar = StridedArray::<f64>::from_parts(vec![0.125], &[1, 1], &[1, 1], 0)
        .map_err(|err| err.to_string())?;
    let c_view = c_scalar.view();
    let c_broadcast = c_view.broadcast(&shape).map_err(|err| err.to_string())?;
    let mut tmp_mul = StridedArray::<f64>::col_major(&shape);
    let mut tmp_add = StridedArray::<f64>::col_major(&shape);
    let mut expected = StridedArray::<f64>::col_major(&shape);
    let mut actual = StridedArray::<f64>::col_major(&shape);

    zip_map2_into(&mut tmp_mul.view_mut(), &a.view(), &b.view(), |x, y| x * y)
        .map_err(|err| err.to_string())?;
    zip_map2_into(
        &mut tmp_add.view_mut(),
        &tmp_mul.view(),
        &c_broadcast,
        |x, y| x + y,
    )
    .map_err(|err| err.to_string())?;
    map_into(&mut expected.view_mut(), &tmp_add.view(), |x| x.exp())
        .map_err(|err| err.to_string())?;
    fused_elementwise_into(
        &mut [actual.view_mut()],
        &[a.view(), b.view(), c_broadcast],
        &exp_mul_add_plan(),
    )
    .map_err(|err| err.to_string())?;

    check_close("broadcast_exp_mul_add", expected.data(), actual.data())
}

fn validate_long_chain(n: usize) -> Result<(), String> {
    let shape = dims(n);
    let a = make_input(n, 2.0);
    let b = make_input(n, 1.0);
    let lo = make_constant(n, 0.25);
    let hi = make_constant(n, 4.0);
    let mut tmp_div = StridedArray::<f64>::col_major(&shape);
    let mut tmp_max = StridedArray::<f64>::col_major(&shape);
    let mut tmp_min = StridedArray::<f64>::col_major(&shape);
    let mut tmp_sqrt = StridedArray::<f64>::col_major(&shape);
    let mut expected = StridedArray::<f64>::col_major(&shape);
    let mut actual = StridedArray::<f64>::col_major(&shape);

    zip_map2_into(&mut tmp_div.view_mut(), &a.view(), &b.view(), |x, y| x / y)
        .map_err(|err| err.to_string())?;
    zip_map2_into(
        &mut tmp_max.view_mut(),
        &tmp_div.view(),
        &lo.view(),
        |x, y| x.max(y),
    )
    .map_err(|err| err.to_string())?;
    zip_map2_into(
        &mut tmp_min.view_mut(),
        &tmp_max.view(),
        &hi.view(),
        |x, y| x.min(y),
    )
    .map_err(|err| err.to_string())?;
    map_into(&mut tmp_sqrt.view_mut(), &tmp_min.view(), |x| x.sqrt())
        .map_err(|err| err.to_string())?;
    map_into(&mut expected.view_mut(), &tmp_sqrt.view(), |x| {
        1.0 / x.sqrt()
    })
    .map_err(|err| err.to_string())?;
    fused_elementwise_into(
        &mut [actual.view_mut()],
        &[a.view(), b.view(), lo.view(), hi.view()],
        &long_chain_plan(),
    )
    .map_err(|err| err.to_string())?;

    check_close("long_chain", expected.data(), actual.data())
}

fn validate_interpreter_fallback(n: usize) -> Result<(), String> {
    let shape = dims(n);
    let a = make_input(n, 0.25);
    let b = make_input(n, 0.5);
    let mut tmp_add = StridedArray::<f64>::col_major(&shape);
    let mut tmp_neg = StridedArray::<f64>::col_major(&shape);
    let mut expected = StridedArray::<f64>::col_major(&shape);
    let mut actual = StridedArray::<f64>::col_major(&shape);

    zip_map2_into(&mut tmp_add.view_mut(), &a.view(), &b.view(), |x, y| x + y)
        .map_err(|err| err.to_string())?;
    map_into(&mut tmp_neg.view_mut(), &tmp_add.view(), |x| -x).map_err(|err| err.to_string())?;
    map_into(&mut expected.view_mut(), &tmp_neg.view(), |x| x.exp())
        .map_err(|err| err.to_string())?;
    fused_elementwise_into(
        &mut [actual.view_mut()],
        &[a.view(), b.view()],
        &interpreter_fallback_plan(),
    )
    .map_err(|err| err.to_string())?;

    check_close("interpreter_fallback", expected.data(), actual.data())
}

fn validate_all_cases(n: usize) -> Result<(), String> {
    validate_add_mul(n)?;
    validate_exp_mul_add(n)?;
    validate_long_chain(n)?;
    validate_interpreter_fallback(n)
}

fn bench_add_mul(n: usize, warmup: usize, nruns: usize) -> CaseResult {
    let shape = dims(n);
    let a = make_input(n, 1.0);
    let b = make_input(n, 2.0);
    let mut tmp = StridedArray::<f64>::col_major(&shape);
    let mut out = StridedArray::<f64>::col_major(&shape);
    let plan = add_mul_plan();

    let per_op = bench(
        || {
            zip_map2_into(&mut tmp.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();
            zip_map2_into(&mut out.view_mut(), &tmp.view(), &a.view(), |x, y| x * y).unwrap();
            black_box(out.data().as_ptr());
        },
        warmup,
        nruns,
    );
    let fused = bench(
        || {
            fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan).unwrap();
            black_box(out.data().as_ptr());
        },
        warmup,
        nruns,
    );

    CaseResult {
        name: "add_mul",
        n,
        elements: n * n,
        per_op,
        fused,
    }
}

fn bench_exp_mul_add(n: usize, warmup: usize, nruns: usize) -> CaseResult {
    let shape = dims(n);
    let a = make_input(n, 0.25);
    let b = make_input(n, 0.5);
    let c_scalar = StridedArray::<f64>::from_parts(vec![0.125], &[1, 1], &[1, 1], 0).unwrap();
    let c_view = c_scalar.view();
    let c_broadcast = c_view.broadcast(&shape).unwrap();
    let mut tmp_mul = StridedArray::<f64>::col_major(&shape);
    let mut tmp_add = StridedArray::<f64>::col_major(&shape);
    let mut out = StridedArray::<f64>::col_major(&shape);
    let plan = exp_mul_add_plan();

    let per_op = bench(
        || {
            zip_map2_into(&mut tmp_mul.view_mut(), &a.view(), &b.view(), |x, y| x * y).unwrap();
            zip_map2_into(
                &mut tmp_add.view_mut(),
                &tmp_mul.view(),
                &c_broadcast,
                |x, y| x + y,
            )
            .unwrap();
            map_into(&mut out.view_mut(), &tmp_add.view(), |x| x.exp()).unwrap();
            black_box(out.data().as_ptr());
        },
        warmup,
        nruns,
    );
    let fused = bench(
        || {
            fused_elementwise_into(
                &mut [out.view_mut()],
                &[a.view(), b.view(), c_broadcast.clone()],
                &plan,
            )
            .unwrap();
            black_box(out.data().as_ptr());
        },
        warmup,
        nruns,
    );

    CaseResult {
        name: "broadcast_exp_mul_add",
        n,
        elements: n * n,
        per_op,
        fused,
    }
}

fn bench_long_chain(n: usize, warmup: usize, nruns: usize) -> CaseResult {
    let shape = dims(n);
    let a = make_input(n, 2.0);
    let b = make_input(n, 1.0);
    let lo = make_constant(n, 0.25);
    let hi = make_constant(n, 4.0);
    let mut tmp_div = StridedArray::<f64>::col_major(&shape);
    let mut tmp_max = StridedArray::<f64>::col_major(&shape);
    let mut tmp_min = StridedArray::<f64>::col_major(&shape);
    let mut tmp_sqrt = StridedArray::<f64>::col_major(&shape);
    let mut out = StridedArray::<f64>::col_major(&shape);
    let plan = long_chain_plan();

    let per_op = bench(
        || {
            zip_map2_into(&mut tmp_div.view_mut(), &a.view(), &b.view(), |x, y| x / y).unwrap();
            zip_map2_into(
                &mut tmp_max.view_mut(),
                &tmp_div.view(),
                &lo.view(),
                |x, y| x.max(y),
            )
            .unwrap();
            zip_map2_into(
                &mut tmp_min.view_mut(),
                &tmp_max.view(),
                &hi.view(),
                |x, y| x.min(y),
            )
            .unwrap();
            map_into(&mut tmp_sqrt.view_mut(), &tmp_min.view(), |x| x.sqrt()).unwrap();
            map_into(&mut out.view_mut(), &tmp_sqrt.view(), |x| 1.0 / x.sqrt()).unwrap();
            black_box(out.data().as_ptr());
        },
        warmup,
        nruns,
    );
    let fused = bench(
        || {
            fused_elementwise_into(
                &mut [out.view_mut()],
                &[a.view(), b.view(), lo.view(), hi.view()],
                &plan,
            )
            .unwrap();
            black_box(out.data().as_ptr());
        },
        warmup,
        nruns,
    );

    CaseResult {
        name: "long_chain",
        n,
        elements: n * n,
        per_op,
        fused,
    }
}

fn bench_interpreter_fallback(n: usize, warmup: usize, nruns: usize) -> CaseResult {
    let shape = dims(n);
    let a = make_input(n, 0.25);
    let b = make_input(n, 0.5);
    let mut tmp_add = StridedArray::<f64>::col_major(&shape);
    let mut tmp_neg = StridedArray::<f64>::col_major(&shape);
    let mut out = StridedArray::<f64>::col_major(&shape);
    let plan = interpreter_fallback_plan();

    let per_op = bench(
        || {
            zip_map2_into(&mut tmp_add.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();
            map_into(&mut tmp_neg.view_mut(), &tmp_add.view(), |x| -x).unwrap();
            map_into(&mut out.view_mut(), &tmp_neg.view(), |x| x.exp()).unwrap();
            black_box(out.data().as_ptr());
        },
        warmup,
        nruns,
    );
    let fused = bench(
        || {
            fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan).unwrap();
            black_box(out.data().as_ptr());
        },
        warmup,
        nruns,
    );

    CaseResult {
        name: "interpreter_fallback",
        n,
        elements: n * n,
        per_op,
        fused,
    }
}

fn main() {
    let warmup = parse_env_usize("WARMUP", 3);
    let nruns = parse_env_usize("NRUNS", 11).max(1);
    let sizes = parse_sizes();
    let strided_rs_hash = std::env::var("STRIDED_RS_HASH").unwrap_or_else(|_| "unknown".into());

    println!("Fused elementwise benchmark");
    println!("strided-rs={strided_rs_hash}");
    println!("warmup={warmup} nruns={nruns} sizes={sizes:?}");
    println!("columns: case n elements per_op_ms(iqr) fused_ms(iqr) speedup");
    println!("{}", "-".repeat(96));

    for n in sizes {
        validate_all_cases(n).unwrap();
        for result in [
            bench_add_mul(n, warmup, nruns),
            bench_exp_mul_add(n, warmup, nruns),
            bench_long_chain(n, warmup, nruns),
            bench_interpreter_fallback(n, warmup, nruns),
        ] {
            println!(
                "{:<24} n={:<5} elements={:<9} {:>9.3} ({:>6.3}) {:>9.3} ({:>6.3}) {:>7.2}x",
                result.name,
                result.n,
                result.elements,
                result.per_op.median_ms,
                result.per_op.iqr_ms,
                result.fused.median_ms,
                result.fused.iqr_ms,
                result.per_op.median_ms / result.fused.median_ms,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fused_cases_match_reference() {
        super::validate_all_cases(8).unwrap();
    }
}
