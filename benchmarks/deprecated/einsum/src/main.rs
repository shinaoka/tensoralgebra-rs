use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;
use strided_opteinsum::{EinsumCode, EinsumError, EinsumNode, EinsumOperand};
use strided_view::StridedArray;

// ---------------------------------------------------------------------------
// JSON schema
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct BenchmarkInstance {
    name: String,
    format_string_colmajor: String,
    shapes_colmajor: Vec<Vec<usize>>,
    dtype: String,
    num_tensors: usize,
    paths: PathInfo,
}

#[derive(Deserialize)]
struct PathInfo {
    opt_size: PathMeta,
    opt_flops: PathMeta,
}

#[derive(Deserialize)]
struct PathMeta {
    path: Vec<[usize; 2]>,
    log2_size: f64,
    log10_flops: f64,
}

type PathStrategy = (&'static str, fn(&PathInfo) -> &PathMeta);

// ---------------------------------------------------------------------------
// Format string parsing
// ---------------------------------------------------------------------------

/// Parse a colmajor einsum format string into per-tensor index chars and output index chars.
///
/// Example: "ba,dca,feb->ki" -> (vec![vec!['b','a'], vec!['d','c','a'], vec!['f','e','b']], vec!['k','i'])
fn parse_format_string(s: &str) -> (Vec<Vec<char>>, Vec<char>) {
    let (inputs_str, output_str) = s.split_once("->").expect("format_string must contain '->'");
    let input_indices: Vec<Vec<char>> = inputs_str
        .split(',')
        .map(|operand| operand.chars().collect())
        .collect();
    let output_indices: Vec<char> = output_str.chars().collect();
    (input_indices, output_indices)
}

// ---------------------------------------------------------------------------
// Contraction path -> EinsumNode tree
// ---------------------------------------------------------------------------

/// Convert a flat contraction path (list of index pairs) into a nested EinsumNode tree.
///
/// Path convention (opt_einsum / cotengra):
/// - Each step [i, j] refers to the current list of tensors
/// - Remove higher index first, then lower; contract; append result to end
fn build_contraction_tree(input_indices: &[Vec<char>], path: &[[usize; 2]]) -> EinsumNode {
    let mut nodes: Vec<EinsumNode> = input_indices
        .iter()
        .enumerate()
        .map(|(i, ids)| EinsumNode::Leaf {
            ids: ids.clone(),
            tensor_index: i,
        })
        .collect();

    for &pair in path {
        let (i, j) = if pair[0] < pair[1] {
            (pair[0], pair[1])
        } else {
            (pair[1], pair[0])
        };
        let node_j = nodes.remove(j);
        let node_i = nodes.remove(i);
        nodes.push(EinsumNode::Contract {
            args: vec![node_i, node_j],
        });
    }

    assert_eq!(
        nodes.len(),
        1,
        "contraction path should reduce to a single node"
    );
    nodes.pop().unwrap()
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

fn create_operands(shapes: &[Vec<usize>], dtype: &str) -> Vec<EinsumOperand<'static>> {
    match dtype {
        "float64" => shapes
            .iter()
            .map(|shape| {
                let arr = StridedArray::<f64>::col_major(shape);
                EinsumOperand::from(arr)
            })
            .collect(),
        "complex128" => {
            use num_complex::Complex64;
            shapes
                .iter()
                .map(|shape| {
                    let arr = StridedArray::<Complex64>::col_major(shape);
                    EinsumOperand::from(arr)
                })
                .collect()
        }
        other => panic!("unsupported dtype: {other}"),
    }
}

struct BenchResult {
    median_ms: f64,
    q1_ms: f64,
    q3_ms: f64,
}

impl BenchResult {
    fn iqr_ms(&self) -> f64 {
        self.q3_ms - self.q1_ms
    }
}

const NUM_WARMUP: usize = 3;
const NUM_TIMED: usize = 15;

fn run_instance(
    instance: &BenchmarkInstance,
    path_meta: &PathMeta,
) -> Result<BenchResult, EinsumError> {
    let (input_indices, output_indices) = parse_format_string(&instance.format_string_colmajor);
    assert_eq!(
        input_indices.len(),
        instance.num_tensors,
        "parsed tensor count mismatch"
    );

    let root = build_contraction_tree(&input_indices, &path_meta.path);
    let code = EinsumCode {
        root,
        output_ids: output_indices,
    };

    // Warmup
    for _ in 0..NUM_WARMUP {
        let operands = create_operands(&instance.shapes_colmajor, &instance.dtype);
        code.evaluate(operands, None)?;
    }

    // Timed runs
    let mut durations = Vec::with_capacity(NUM_TIMED);
    for _ in 0..NUM_TIMED {
        let operands = create_operands(&instance.shapes_colmajor, &instance.dtype);
        let t0 = Instant::now();
        let result = code.evaluate(operands, None)?;
        let elapsed = t0.elapsed();
        black_box(&result);
        durations.push(elapsed);
    }

    durations.sort();
    let median = durations[durations.len() / 2];
    let q1 = durations[durations.len() / 4];
    let q3 = durations[3 * durations.len() / 4];
    Ok(BenchResult {
        median_ms: median.as_secs_f64() * 1e3,
        q1_ms: q1.as_secs_f64() * 1e3,
        q3_ms: q3.as_secs_f64() * 1e3,
    })
}

// ---------------------------------------------------------------------------
// Backend name (compile-time)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "faer", not(feature = "blas")))]
const BACKEND_NAME: &str = "strided-opteinsum(faer)";
#[cfg(all(feature = "blas", not(feature = "faer")))]
const BACKEND_NAME: &str = "strided-opteinsum(blas)";

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn load_instances() -> Vec<BenchmarkInstance> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/instances");
    let mut paths: Vec<_> = std::fs::read_dir(&data_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", data_dir.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();

    paths
        .iter()
        .filter_map(|path| {
            let json_str = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Warning: skip {} (read failed: {e})", path.display());
                    return None;
                }
            };
            match serde_json::from_str(&json_str) {
                Ok(instance) => Some(instance),
                Err(e) => {
                    eprintln!("Warning: skip {} (parse failed: {e})", path.display());
                    None
                }
            }
        })
        .collect()
}

fn main() {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/instances");
    let mut instances = load_instances();
    if let Ok(filter) = std::env::var("BENCH_INSTANCE") {
        instances.retain(|i| i.name == filter);
        if instances.is_empty() {
            eprintln!("BENCH_INSTANCE={filter:?}: no matching instance found");
            std::process::exit(1);
        }
    }

    let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into());
    let omp_threads = std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "unset".into());

    println!("{BACKEND_NAME} benchmark suite");
    println!("==================================");
    println!(
        "Loaded {} instances from {}",
        instances.len(),
        data_dir.display()
    );
    println!("Backend: {BACKEND_NAME}");
    println!("RAYON_NUM_THREADS={rayon_threads}, OMP_NUM_THREADS={omp_threads}");
    println!("Timing: median of {NUM_TIMED} runs ({NUM_WARMUP} warmup)");

    let strategies: &[PathStrategy] = &[
        ("opt_flops", |p| &p.opt_flops),
        ("opt_size", |p| &p.opt_size),
    ];

    for &(strategy_name, get_path) in strategies {
        println!();
        println!("Strategy: {strategy_name}");
        println!(
            "{:<50} {:>8} {:>10} {:>12} {:>12} {:>10}",
            "Instance", "Tensors", "log10FLOPS", "log2SIZE", "Median (ms)", "IQR (ms)"
        );
        println!("{}", "-".repeat(108));

        for instance in &instances {
            let path_meta = get_path(&instance.paths);
            match run_instance(instance, path_meta) {
                Ok(result) => {
                    println!(
                        "{:<50} {:>8} {:>10.2} {:>12.2} {:>12.3} {:>10.3}",
                        instance.name,
                        instance.num_tensors,
                        path_meta.log10_flops,
                        path_meta.log2_size,
                        result.median_ms,
                        result.iqr_ms(),
                    );
                }
                Err(e) => {
                    println!(
                        "{:<50} {:>8} {:>10.2} {:>12.2} {:>12} {:>10}",
                        instance.name,
                        instance.num_tensors,
                        path_meta.log10_flops,
                        path_meta.log2_size,
                        "SKIP",
                        "-",
                    );
                    eprintln!("  -> {} (backend error: {e})", instance.name);
                }
            }
        }
    }
}
