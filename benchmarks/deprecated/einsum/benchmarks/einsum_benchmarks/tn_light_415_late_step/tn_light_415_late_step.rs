//! Minimal benchmark reproducing TN light 415 late-step contraction of tensornetwork_permutation_light_415.
//!
//! Binary einsum with all dims = 2 (binary tensor network).
//! m=4, k=256, n=8192, batch=8.
//!
//! The key bottleneck: B has 24 binary dims with scattered strides after
//! canonical reorder. `prepare_input_owned` must copy 16M elements to
//! contiguous layout before BLAS GEMM.
//!
//! Build & run:
//!   cargo run --release --no-default-features --features blas --bin tn_light_415_late_step
//!   OMP_NUM_THREADS=4 cargo run --release --no-default-features --features blas --bin tn_light_415_late_step

use std::time::Instant;
use strided_view::StridedArray;

fn bench<F: FnMut()>(mut f: F, warmup: usize, nruns: usize) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(nruns);
    for _ in 0..nruns {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let q1 = times[times.len() / 4];
    let q3 = times[3 * times.len() / 4];
    (med, q3 - q1)
}

fn main() {
    let warmup = 3;
    let nruns = 15;

    // A: 13 dims of size 2, col-major (contiguous). 8192 elements.
    let a_dims: Vec<usize> = vec![2; 13];
    let a = StridedArray::<f64>::col_major(&a_dims);
    // B: 24 dims of size 2, col-major (contiguous). 16M elements.
    let b_dims: Vec<usize> = vec![2; 24];
    let b = StridedArray::<f64>::col_major(&b_dims);
    // C: 18 dims of size 2, col-major. 262144 elements.
    let c_dims: Vec<usize> = vec![2; 18];

    // Exact einsum labels reconstructed from TN light 415 late-step contraction's plan
    // These labels produce the same scattered-stride pattern as the real benchmark.
    let ia: Vec<char> = "caxydefghizjb".chars().collect();
    let ib: Vec<char> = "hklicxmnopdqfrstyjuzgvwe".chars().collect();
    let ic: Vec<char> = "abklwmnopqrstuvxyz".chars().collect();

    println!("TN light 415 late-step benchmark (all dims=2)");
    println!(
        "A: {:?} = {} elements",
        a.dims(),
        a.dims().iter().product::<usize>()
    );
    println!(
        "B: {:?} = {} elements",
        b.dims(),
        b.dims().iter().product::<usize>()
    );
    println!(
        "C: output {:?} = {} elements",
        &c_dims,
        c_dims.iter().product::<usize>()
    );
    println!(
        "Einsum: {},{}->{}",
        ia.iter().collect::<String>(),
        ib.iter().collect::<String>(),
        ic.iter().collect::<String>()
    );
    println!("{}", "=".repeat(70));

    // --- 1) Full einsum2_into_owned (scattered strides, what the real benchmark does) ---
    let (med, iqr) = bench(
        || {
            let mut c_arr = StridedArray::<f64>::col_major(&c_dims);
            strided_einsum2::einsum2_into_owned(
                c_arr.view_mut(),
                a.clone(),
                b.clone(),
                &ic,
                &ia,
                &ib,
                1.0,
                0.0,
                false,
                false,
            )
            .unwrap();
        },
        warmup,
        nruns,
    );
    println!(
        "einsum2 full (scattered B):    {:.3} ms (IQR {:.3} ms)",
        med, iqr
    );

    // --- 2) Isolate the copy cost: permute B to canonical order, then copy ---
    // Reconstruct the right_perm that einsum2 computes internally
    let right_perm: Vec<usize> = vec![
        4, 10, 23, 12, 20, 0, 3, 17, 1, 2, 6, 7, 8, 9, 11, 13, 14, 15, 18, 21, 22, 5, 16, 19,
    ];
    let b_perm = b.permuted(&right_perm).unwrap();
    println!(
        "\nB after canonical reorder: dims={:?} strides={:?}",
        b_perm.dims(),
        b_perm.strides()
    );

    let (med_copy_b, iqr_copy_b) = bench(
        || {
            let mut b_dest = StridedArray::<f64>::col_major(b_perm.dims());
            strided_perm::copy_into(&mut b_dest.view_mut(), &b_perm.view()).unwrap();
        },
        warmup,
        nruns,
    );
    println!(
        "copy_into B (16M, scattered):  {:.3} ms (IQR {:.3} ms)",
        med_copy_b, iqr_copy_b
    );

    // --- 3) Copy cost for A ---
    let left_perm: Vec<usize> = vec![1, 12, 0, 4, 5, 6, 7, 8, 9, 11, 2, 3, 10];
    let a_perm = a.permuted(&left_perm).unwrap();

    let (med_copy_a, iqr_copy_a) = bench(
        || {
            let mut a_dest = StridedArray::<f64>::col_major(a_perm.dims());
            strided_perm::copy_into(&mut a_dest.view_mut(), &a_perm.view()).unwrap();
        },
        warmup,
        nruns,
    );
    println!(
        "copy_into A (8K, scattered):   {:.3} ms (IQR {:.3} ms)",
        med_copy_a, iqr_copy_a
    );

    // --- 4) einsum2 with pre-contiguous data (isolates GEMM cost) ---
    // Make A and B contiguous in canonical order, then use labels matching canonical order
    let mut a_contig = StridedArray::<f64>::col_major(a_perm.dims());
    strided_perm::copy_into(&mut a_contig.view_mut(), &a_perm.view()).unwrap();
    let mut b_contig = StridedArray::<f64>::col_major(b_perm.dims());
    strided_perm::copy_into(&mut b_contig.view_mut(), &b_perm.view()).unwrap();

    // Labels in canonical order after permutation
    let ia_canon: Vec<char> = left_perm.iter().map(|&i| ia[i]).collect();
    let ib_canon: Vec<char> = right_perm.iter().map(|&i| ib[i]).collect();

    let (med_gemm, iqr_gemm) = bench(
        || {
            let mut c_arr = StridedArray::<f64>::col_major(&c_dims);
            strided_einsum2::einsum2_into_owned(
                c_arr.view_mut(),
                a_contig.clone(),
                b_contig.clone(),
                &ic,
                &ia_canon,
                &ib_canon,
                1.0,
                0.0,
                false,
                false,
            )
            .unwrap();
        },
        warmup,
        nruns,
    );
    println!(
        "einsum2 (contiguous, ~GEMM):   {:.3} ms (IQR {:.3} ms)",
        med_gemm, iqr_gemm
    );

    // --- Summary ---
    println!("\n--- Summary ---");
    println!("Full einsum2 (scattered):  {:.3} ms", med);
    println!(
        "  copy B (dominant):       {:.3} ms ({:.0}%)",
        med_copy_b,
        med_copy_b / med * 100.0
    );
    println!("  copy A:                  {:.3} ms", med_copy_a);
    println!(
        "  GEMM only (~):           {:.3} ms ({:.0}%)",
        med_gemm,
        med_gemm / med * 100.0
    );
}
