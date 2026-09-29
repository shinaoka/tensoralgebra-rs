use serde::Deserialize;
use std::fmt;
use std::hint::black_box;
use std::time::{Duration, Instant};
#[allow(unused_imports)]
#[cfg(feature = "parallel")]
use strided_perm::copy_into_col_major_par;
#[cfg(feature = "parallel")]
use strided_perm::copy_into_par;
use strided_perm::{copy_into, copy_into_col_major};
use strided_view::{col_major_strides, StridedArray};

const PATTERN_PATH: &str = "benchmarks/strided_benchmarks/permute/patterns.json";

// ---------------------------------------------------------------------------
// Pattern schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PatternSuite {
    version: u32,
    index_base: u32,
    semantics: String,
    data: String,
    patterns: Vec<PermutePattern>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PermutePattern {
    id: String,
    label: String,
    dtype: String,
    shape: Vec<usize>,
    perm: Vec<usize>,
    src_layout: LayoutPattern,
    dst_layout: LayoutPattern,
    participants: Vec<Participant>,
    #[allow(dead_code)]
    notes: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LayoutPattern {
    ColMajor,
    ExplicitStrides { strides: Vec<isize> },
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Participant {
    Naive,
    StridedPerm,
    StridedPermColMajor,
    Hptt,
    JuliaBase,
    StridedJl,
    Memcpy,
}

#[derive(Debug)]
struct PatternError(String);

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PatternError {}

fn load_pattern_suite_from_str(json: &str) -> Result<PatternSuite, Box<dyn std::error::Error>> {
    let suite: PatternSuite = serde_json::from_str(json)?;
    validate_pattern_suite(&suite)?;
    Ok(suite)
}

fn validate_pattern_suite(suite: &PatternSuite) -> Result<(), PatternError> {
    if suite.version != 1 {
        return Err(PatternError(format!(
            "unsupported pattern schema version {}",
            suite.version
        )));
    }
    if suite.index_base != 0 {
        return Err(PatternError(
            "permute patterns must use index_base = 0".into(),
        ));
    }
    if suite.semantics != "out[i0,...,ik] = src[i_perm0,...,i_permk]" {
        return Err(PatternError(format!(
            "unsupported semantics {:?}",
            suite.semantics
        )));
    }
    if suite.data != "deterministic_index_value" {
        return Err(PatternError(format!(
            "unsupported data mode {:?}",
            suite.data
        )));
    }

    for pattern in &suite.patterns {
        validate_pattern(pattern)?;
    }
    Ok(())
}

fn validate_pattern(pattern: &PermutePattern) -> Result<(), PatternError> {
    if pattern.dtype != "f64" {
        return Err(PatternError(format!(
            "{} uses unsupported dtype {:?}",
            pattern.id, pattern.dtype
        )));
    }
    if pattern.shape.is_empty() {
        return Err(PatternError(format!("{} has empty shape", pattern.id)));
    }
    if pattern.shape.len() != pattern.perm.len() {
        return Err(PatternError(format!(
            "{} shape rank {} != perm rank {}",
            pattern.id,
            pattern.shape.len(),
            pattern.perm.len()
        )));
    }

    let mut seen = vec![false; pattern.perm.len()];
    for &axis in &pattern.perm {
        if axis >= pattern.perm.len() {
            return Err(PatternError(format!(
                "{} perm axis {} is out of range for rank {}",
                pattern.id,
                axis,
                pattern.perm.len()
            )));
        }
        if seen[axis] {
            return Err(PatternError(format!(
                "{} perm axis {} appears more than once",
                pattern.id, axis
            )));
        }
        seen[axis] = true;
    }

    validate_layout(
        &pattern.id,
        "src_layout",
        &pattern.src_layout,
        pattern.shape.len(),
    )?;
    validate_layout(
        &pattern.id,
        "dst_layout",
        &pattern.dst_layout,
        pattern.shape.len(),
    )?;

    Ok(())
}

fn validate_layout(
    id: &str,
    name: &str,
    layout: &LayoutPattern,
    rank: usize,
) -> Result<(), PatternError> {
    if let LayoutPattern::ExplicitStrides { strides } = layout {
        if strides.len() != rank {
            return Err(PatternError(format!(
                "{id} {name} stride rank {} != shape rank {rank}",
                strides.len()
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2
    }
}

fn bench_n(label: &str, warmup: usize, iters: usize, bytes: usize, mut f: impl FnMut()) {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }
    let med = median(&mut samples);
    let ms = med.as_secs_f64() * 1e3;
    let gbps = (bytes as f64) / med.as_secs_f64() / 1e9;
    // p25 and p75
    let p25 = samples[samples.len() / 4].as_secs_f64() * 1e3;
    let p75 = samples[samples.len() * 3 / 4].as_secs_f64() * 1e3;
    println!("  {label:30} {ms:8.3} ms  ({p25:.3} / {p75:.3})  {gbps:6.2} GB/s");
}

fn make_deterministic_array(dims: &[usize], strides: &[isize]) -> StridedArray<f64> {
    let total: usize = dims.iter().product();
    let data: Vec<f64> = (0..total).map(|i| i as f64 + 1.0).collect();
    StridedArray::from_parts(data, dims, strides, 0).unwrap()
}

/// Generic strided copy using odometer iteration.
///
/// Iterates over `dims` in col-major order, reading from `src_ptr` using
/// `src_strides` and writing to `dst_ptr` using `dst_strides`.
unsafe fn naive_strided_copy(
    src_ptr: *const f64,
    dst_ptr: *mut f64,
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) {
    let rank = dims.len();
    let total: usize = dims.iter().product();

    let mut idx = vec![0usize; rank];
    let mut src_off = 0isize;
    let mut dst_off = 0isize;

    for _ in 0..total {
        *dst_ptr.offset(dst_off) = *src_ptr.offset(src_off);
        for d in 0..rank {
            idx[d] += 1;
            src_off += src_strides[d];
            dst_off += dst_strides[d];
            if idx[d] < dims[d] {
                break;
            }
            src_off -= (idx[d] as isize) * src_strides[d];
            dst_off -= (idx[d] as isize) * dst_strides[d];
            idx[d] = 0;
        }
    }
}

/// Convenience: naive permuted copy from col-major source.
///
/// B[i_0, ..., i_{N-1}] = A[i_{perm[0]}, ..., i_{perm[N-1]}]
// ---------------------------------------------------------------------------
// Unified pattern runner
// ---------------------------------------------------------------------------

struct PreparedPattern {
    src: StridedArray<f64>,
    out_shape: Vec<usize>,
    src_perm_strides: Vec<isize>,
    dst_strides: Vec<isize>,
    reference: Vec<f64>,
}

fn load_pattern_suite() -> Result<PatternSuite, Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(PATTERN_PATH)?;
    load_pattern_suite_from_str(&json)
}

fn layout_strides(pattern: &PermutePattern, layout: &LayoutPattern) -> Vec<isize> {
    match layout {
        LayoutPattern::ColMajor => col_major_strides(&pattern.shape),
        LayoutPattern::ExplicitStrides { strides } => strides.clone(),
    }
}

fn output_shape(pattern: &PermutePattern) -> Vec<usize> {
    pattern
        .perm
        .iter()
        .map(|&axis| pattern.shape[axis])
        .collect()
}

fn prepare_pattern(pattern: &PermutePattern) -> PreparedPattern {
    let src_strides = layout_strides(pattern, &pattern.src_layout);
    let src = make_deterministic_array(&pattern.shape, &src_strides);
    let src_perm = src.view().permute(&pattern.perm).unwrap();
    let out_shape = output_shape(pattern);
    let src_perm_strides = src_perm.strides().to_vec();
    let dst_strides = col_major_strides(&out_shape);
    let mut reference = vec![0.0f64; out_shape.iter().product()];
    unsafe {
        naive_strided_copy(
            src.data().as_ptr(),
            reference.as_mut_ptr(),
            &out_shape,
            &src_perm_strides,
            &dst_strides,
        );
    }

    PreparedPattern {
        src,
        out_shape,
        src_perm_strides,
        dst_strides,
        reference,
    }
}

fn verify_output(label: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {} != {}",
        actual.len(),
        expected.len()
    );
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "{label}: mismatch at element {i}");
    }
}

fn timing_counts(total: usize) -> (usize, usize) {
    if total >= 1 << 23 {
        (3, 15)
    } else {
        (5, 40)
    }
}

fn run_pattern(pattern: &PermutePattern) {
    let prepared = prepare_pattern(pattern);
    let total = prepared.reference.len();
    let bytes = total * std::mem::size_of::<f64>() * 2;
    let (warmup, iters) = timing_counts(total);
    let src_perm = prepared.src.view().permute(&pattern.perm).unwrap();

    println!(
        "=== {} ===\n  id={} elems={} bytes(r+w)={}",
        pattern.label, pattern.id, total, bytes
    );

    for participant in &pattern.participants {
        match participant {
            Participant::Naive => {
                let mut dst = vec![0.0f64; total];
                unsafe {
                    naive_strided_copy(
                        prepared.src.data().as_ptr(),
                        dst.as_mut_ptr(),
                        &prepared.out_shape,
                        &prepared.src_perm_strides,
                        &prepared.dst_strides,
                    );
                }
                verify_output("naive_odometer", &dst, &prepared.reference);
                bench_n("naive_odometer", warmup, iters, bytes, || {
                    unsafe {
                        naive_strided_copy(
                            prepared.src.data().as_ptr(),
                            dst.as_mut_ptr(),
                            &prepared.out_shape,
                            &prepared.src_perm_strides,
                            &prepared.dst_strides,
                        )
                    };
                    black_box(dst.as_ptr());
                });
            }
            Participant::Memcpy => {
                if pattern.perm.iter().copied().eq(0..pattern.perm.len())
                    && matches!(pattern.src_layout, LayoutPattern::ColMajor)
                    && matches!(pattern.dst_layout, LayoutPattern::ColMajor)
                {
                    let mut dst = vec![0.0f64; total];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            prepared.src.data().as_ptr(),
                            dst.as_mut_ptr(),
                            total,
                        )
                    };
                    verify_output("std::ptr::copy_nonoverlapping", &dst, &prepared.reference);
                    bench_n(
                        "std::ptr::copy_nonoverlapping",
                        warmup,
                        iters,
                        bytes,
                        || {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    prepared.src.data().as_ptr(),
                                    dst.as_mut_ptr(),
                                    total,
                                )
                            };
                            black_box(dst.as_ptr());
                        },
                    );
                } else {
                    println!("  std::ptr::copy_nonoverlapping skipped: requires identity col-major pattern");
                }
            }
            Participant::StridedPerm => {
                let mut dst = StridedArray::<f64>::col_major(&prepared.out_shape);
                copy_into(&mut dst.view_mut(), &src_perm).unwrap();
                verify_output("strided_perm::copy_into", dst.data(), &prepared.reference);
                bench_n("strided_perm::copy_into", warmup, iters, bytes, || {
                    copy_into(&mut dst.view_mut(), &src_perm).unwrap();
                    black_box(dst.data().as_ptr());
                });
                #[cfg(feature = "parallel")]
                {
                    copy_into_par(&mut dst.view_mut(), &src_perm).unwrap();
                    verify_output(
                        "strided_perm::copy_into_par",
                        dst.data(),
                        &prepared.reference,
                    );
                    bench_n("strided_perm::copy_into_par", warmup, iters, bytes, || {
                        copy_into_par(&mut dst.view_mut(), &src_perm).unwrap();
                        black_box(dst.data().as_ptr());
                    });
                }
            }
            Participant::StridedPermColMajor => {
                let mut dst = StridedArray::<f64>::col_major(&prepared.out_shape);
                copy_into_col_major(&mut dst.view_mut(), &src_perm).unwrap();
                verify_output(
                    "strided_perm::copy_into_col_major",
                    dst.data(),
                    &prepared.reference,
                );
                bench_n(
                    "strided_perm::copy_into_col_major",
                    warmup,
                    iters,
                    bytes,
                    || {
                        copy_into_col_major(&mut dst.view_mut(), &src_perm).unwrap();
                        black_box(dst.data().as_ptr());
                    },
                );
                #[cfg(feature = "parallel")]
                {
                    copy_into_col_major_par(&mut dst.view_mut(), &src_perm).unwrap();
                    verify_output(
                        "strided_perm::copy_into_col_major_par",
                        dst.data(),
                        &prepared.reference,
                    );
                    bench_n(
                        "strided_perm::copy_into_col_major_par",
                        warmup,
                        iters,
                        bytes,
                        || {
                            copy_into_col_major_par(&mut dst.view_mut(), &src_perm).unwrap();
                            black_box(dst.data().as_ptr());
                        },
                    );
                }
            }
            Participant::Hptt => run_hptt_participant(pattern, &prepared, warmup, iters, bytes),
            Participant::JuliaBase | Participant::StridedJl => {}
        }
    }

    println!();
}

#[cfg(feature = "hptt")]
fn run_hptt_participant(
    pattern: &PermutePattern,
    prepared: &PreparedPattern,
    warmup: usize,
    iters: usize,
    bytes: usize,
) {
    if !matches!(pattern.src_layout, LayoutPattern::ColMajor)
        || !matches!(pattern.dst_layout, LayoutPattern::ColMajor)
    {
        println!("  hptt skipped: requires contiguous source and destination");
        return;
    }

    let mut dst = vec![0.0f64; prepared.reference.len()];
    hptt::transpose_f64(
        &pattern.perm,
        1.0,
        prepared.src.data(),
        &pattern.shape,
        0.0,
        &mut dst,
        1,
        hptt::MemoryOrder::ColumnMajor,
    )
    .expect("hptt correctness run");
    verify_output("hptt", &dst, &prepared.reference);

    let threads = current_thread_count();
    bench_n(&format!("hptt ({threads}T)"), warmup, iters, bytes, || {
        hptt::transpose_f64(
            &pattern.perm,
            1.0,
            prepared.src.data(),
            &pattern.shape,
            0.0,
            &mut dst,
            threads,
            hptt::MemoryOrder::ColumnMajor,
        )
        .unwrap();
        black_box(dst.as_ptr());
    });
}

#[cfg(not(feature = "hptt"))]
fn run_hptt_participant(
    _pattern: &PermutePattern,
    _prepared: &PreparedPattern,
    _warmup: usize,
    _iters: usize,
    _bytes: usize,
) {
    println!("  hptt skipped: rebuild with --features hptt");
}

#[cfg(feature = "hptt")]
fn current_thread_count() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

fn selected_patterns<'a>(suite: &'a PatternSuite) -> Vec<&'a PermutePattern> {
    match std::env::var("PATTERN_ID") {
        Ok(id) => suite
            .patterns
            .iter()
            .filter(|pattern| pattern.id == id)
            .collect(),
        Err(_) => suite.patterns.iter().collect(),
    }
}

fn run_patterns(suite: &PatternSuite) {
    let patterns = selected_patterns(suite);
    assert!(
        !patterns.is_empty(),
        "PATTERN_ID did not match any pattern in {PATTERN_PATH}"
    );

    println!("--- Correctness verification and benchmarks ---");
    for pattern in patterns {
        run_pattern(pattern);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("strided-perm permutation benchmarks");
    println!("====================================");
    println!("Element size: {} bytes", std::mem::size_of::<f64>());
    #[cfg(feature = "parallel")]
    {
        let nthreads = rayon::current_num_threads();
        println!("Parallel feature: enabled ({nthreads} threads)");
    }
    #[cfg(not(feature = "parallel"))]
    println!("Parallel feature: disabled (single-threaded only)");
    println!("Format: label  median_ms  (p25 / p75)  bandwidth_GB/s");
    println!("Patterns: {PATTERN_PATH}");
    if let Ok(id) = std::env::var("PATTERN_ID") {
        println!("Pattern filter: {id}");
    }
    println!();

    let suite = load_pattern_suite()?;
    run_patterns(&suite);

    println!("Done.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_perm_pattern_with_explicit_strides_without_seed() {
        let json = r#"
        {
          "version": 1,
          "index_base": 0,
          "semantics": "out[i0,...,ik] = src[i_perm0,...,i_permk]",
          "data": "deterministic_index_value",
          "patterns": [
            {
              "id": "tn_light_415_24d_scattered_to_colmajor",
              "label": "24D scattered -> col-major",
              "dtype": "f64",
              "shape": [2, 2, 2, 2],
              "perm": [0, 2, 1, 3],
              "src_layout": {
                "kind": "explicit_strides",
                "strides": [1, 2, 8, 4]
              },
              "dst_layout": { "kind": "col_major" },
              "participants": ["naive", "strided_perm", "julia_base", "strided_jl"],
              "notes": "HPTT cannot represent arbitrary source strides."
            }
          ]
        }
        "#;

        let suite = load_pattern_suite_from_str(json).unwrap();

        assert_eq!(suite.patterns.len(), 1);
        assert_eq!(
            suite.patterns[0].id,
            "tn_light_415_24d_scattered_to_colmajor"
        );
        assert_eq!(suite.patterns[0].perm, vec![0, 2, 1, 3]);
        assert_eq!(
            suite.patterns[0].src_layout,
            LayoutPattern::ExplicitStrides {
                strides: vec![1, 2, 8, 4]
            }
        );
        assert!(!suite.patterns[0].participants.contains(&Participant::Hptt));
    }

    #[test]
    fn bundled_patterns_are_valid() {
        let suite = load_pattern_suite_from_str(include_str!("patterns.json")).unwrap();

        assert!(suite.patterns.iter().any(|p| p.id == "transpose_2d_1024"));
        assert!(suite
            .patterns
            .iter()
            .any(|p| p.id == "tn_light_415_24d_scattered_to_colmajor"));
    }

    #[test]
    fn rejects_seed_in_pattern_schema() {
        let json = r#"
        {
          "version": 1,
          "index_base": 0,
          "semantics": "out[i0,...,ik] = src[i_perm0,...,i_permk]",
          "data": "deterministic_index_value",
          "patterns": [
            {
              "id": "bad_seed",
              "label": "bad seed",
              "dtype": "f64",
              "shape": [2, 2],
              "perm": [1, 0],
              "src_layout": { "kind": "col_major" },
              "dst_layout": { "kind": "col_major" },
              "participants": ["naive"],
              "seed": 42
            }
          ]
        }
        "#;

        assert!(load_pattern_suite_from_str(json).is_err());
    }
}
