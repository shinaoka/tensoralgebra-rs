//! Dense kernels consumed by tenferro; setup/restore/checks are outside samples.
use std::{env, hint::black_box, mem::MaybeUninit, time::Instant};
use strided_kernel::{
    axpby_accum, copy_into_uninit, embed_diagonal_into_uninit, map_into, mul_into_uninit,
    triangular_mask_into_uninit, with_execution_policy, ExecutionPolicy, StridedView,
    StridedViewMut,
};

// Stable Callgrind boundary. Explicit Sequential policy guarantees that no
// worker-thread kernel work is omitted by --toggle-collect=profile_dense.
#[no_mangle]
#[inline(never)]
pub fn profile_dense(operation: &mut dyn FnMut()) {
    operation();
}

fn run(case: &str, runs: usize, timing: bool) {
    let (shape, stride, kind): (Vec<usize>, usize, &str) = match case {
        "mul_2048" => (vec![2048, 2048], 1, "mul"),
        "mul_odd" => (vec![1025, 1025], 1, "mul"),
        "mul_strided" => (vec![1024, 1024], 2, "mul"),
        "axpby_1m" => (vec![1024, 1024], 1, "axpby"),
        "tril_1024" => (vec![1024, 1024], 1, "tril"),
        "triu_rect" => (vec![513, 1025, 2], 1, "triu"),
        "diag_rank2" => (vec![128, 64], 1, "diag"),
        "copy_contiguous" => (vec![1024, 1025], 1, "copy"),
        "copy_transpose" => (vec![1024, 1025], 1, "copy"),
        "copy_lm" => (vec![12, 12, 1100], 1, "copy"),
        "copy_small" => (vec![3, 5], 1, "copy"),
        "copy_negative" => (vec![33, 65], 1, "copy"),
        "copy_rank6" => (vec![2, 3, 4, 5, 6, 7], 1, "copy"),
        _ => panic!("unknown case {case}"),
    };
    let n: usize = shape.iter().product();
    let a: Vec<f64> = (0..n * stride)
        .map(|i| (i % 29) as f64 / 32.0 - 0.5)
        .collect();
    let b: Vec<f64> = (0..n).map(|i| (i % 17) as f64 / 16.0 + 0.25).collect();
    let mut y = b.clone();
    let out_len = if kind == "diag" { n * shape[0] } else { n };
    let mut output = vec![MaybeUninit::<f64>::uninit(); out_len];
    let mut source_strides: Vec<_> = strided_kernel::col_major_strides(&shape)
        .into_iter()
        .map(|s| s * stride as isize)
        .collect();
    let mut source_offset = 0;
    match case {
        "copy_transpose" | "copy_small" | "copy_rank6" => {
            let mut step = 1;
            for axis in (0..shape.len()).rev() {
                source_strides[axis] = step;
                step *= shape[axis] as isize;
            }
        }
        "copy_lm" => source_strides = vec![1100, 13200, 1],
        "copy_negative" => {
            source_strides[0] = -1;
            source_offset = 32;
        }
        _ => {}
    }
    let baseline_map = env::var("COPY_MAP_BASELINE").as_deref() == Ok("1");
    let strides = strided_kernel::col_major_strides(&shape);
    let av = StridedView::<f64>::new(&a, &shape, &source_strides, source_offset).unwrap();
    let bv = StridedView::<f64>::new(&b, &shape, &strides, 0).unwrap();
    let mut nanos = Vec::new();
    for sample in 0..=runs {
        // In-place state restoration and borrowed-view preparation are excluded.
        if kind != "copy" {
            y.copy_from_slice(&b);
        }
        let mut dest = StridedViewMut::new(&mut output, &shape, &strides, 0).unwrap();
        let mut operation = || {
            match kind {
                "copy" if baseline_map => map_into(&mut dest, &av, MaybeUninit::new).unwrap(),
                "copy" => copy_into_uninit(&mut dest, &av).unwrap(),
                "mul" => mul_into_uninit(&mut dest, &av, &bv).unwrap(),
                "axpby" => axpby_accum(&mut y, &a, 0.25, 0.5).unwrap(),
                "tril" | "triu" => triangular_mask_into_uninit(
                    dest.data_mut(),
                    &a,
                    &shape,
                    -1,
                    kind == "triu",
                    0.0,
                )
                .unwrap(),
                "diag" => {
                    embed_diagonal_into_uninit(dest.data_mut(), &a, &shape, 0, 1, 0.0).unwrap()
                }
                _ => unreachable!(),
            }
            black_box(&dest);
        };
        if sample == 0 {
            operation();
        } else {
            let start = timing.then(Instant::now);
            profile_dense(&mut operation);
            if let Some(start) = start {
                nanos.push(start.elapsed().as_nanos());
            }
        }
    }
    if kind == "axpby" {
        for ((&actual, &x), &old) in y.iter().zip(&a).zip(&b) {
            assert_eq!(actual, 0.25 * x + 0.5 * old);
        }
    } else {
        for (flat, value) in output.iter().enumerate() {
            let expected = match kind {
                "copy" => {
                    let mut index = flat;
                    let mut offset = source_offset;
                    for (&dim, &step) in shape.iter().zip(&source_strides) {
                        offset += (index % dim) as isize * step;
                        index /= dim;
                    }
                    a[offset as usize]
                }
                "mul" => a[flat * stride] * b[flat],
                "tril" | "triu" => {
                    let row = flat % shape[0];
                    let col = (flat / shape[0]) % shape[1];
                    let keep = if kind == "triu" {
                        row as i64 <= col as i64 + 1
                    } else {
                        row as i64 >= col as i64 + 1
                    };
                    if keep {
                        a[flat]
                    } else {
                        0.0
                    }
                }
                "diag" => {
                    let row = flat % shape[0];
                    let diagonal = (flat / shape[0]) % shape[0];
                    let column = flat / (shape[0] * shape[0]);
                    if row == diagonal {
                        a[row + shape[0] * column]
                    } else {
                        0.0
                    }
                }
                _ => unreachable!(),
            };
            // SAFETY: successful kernels above fully initialized output.
            assert_eq!(
                unsafe { value.assume_init() },
                expected,
                "{case} index {flat}"
            );
        }
    }
    if timing {
        nanos.sort_unstable();
        println!("{case},1,{runs},{}", nanos[nanos.len() / 2]);
    } else {
        println!("CHECK {case} passed; threads=1 policy=Sequential samples={runs} copy_map_baseline={baseline_map}");
    }
}

fn main() {
    let timing = env::args().any(|arg| arg == "--time");
    let runs = env::var("BENCH_RUNS")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .unwrap();
    assert!(runs > 0);
    let filter = env::var("BENCH_INSTANCE").unwrap_or_default();
    let cases = [
        "mul_2048",
        "mul_odd",
        "mul_strided",
        "axpby_1m",
        "tril_1024",
        "triu_rect",
        "diag_rank2",
        "copy_contiguous",
        "copy_transpose",
        "copy_lm",
        "copy_small",
        "copy_negative",
        "copy_rank6",
    ];
    assert!(filter.is_empty() || cases.contains(&filter.as_str()));
    // No ambient/all-core provider: every call uses the serial execution path.
    with_execution_policy(ExecutionPolicy::Sequential, || {
        for case in cases {
            if filter.is_empty() || filter == case {
                run(case, runs, timing);
            }
        }
    });
}
