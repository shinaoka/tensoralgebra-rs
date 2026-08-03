//! Locate the optional C baselines.
//!
//! * `tblis` feature: needs `TBLIS_ROOT` (an install prefix with
//!   `lib/libtblis.{so,a}` and `include/tblis.h`).
//! * `blas`  feature: needs `OPENBLAS_ROOT` or `BLAS_ROOT`, or a system
//!   `libopenblas` on the default search path.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=TBLIS_ROOT");
    println!("cargo:rerun-if-env-changed=OPENBLAS_ROOT");
    println!("cargo:rerun-if-env-changed=BLAS_ROOT");

    if env::var("CARGO_FEATURE_TBLIS").is_ok() {
        let root = env::var("TBLIS_ROOT")
            .expect("feature `tblis` requires TBLIS_ROOT to point at a TBLIS install prefix");
        for sub in ["lib", "lib64"] {
            println!("cargo:rustc-link-search=native={root}/{sub}");
        }
        println!("cargo:rustc-link-lib=dylib=tblis");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // TBLIS pulls in hwloc and OpenMP through TCI.
        println!("cargo:rustc-link-lib=dylib=gomp");
    }

    if env::var("CARGO_FEATURE_BLAS").is_ok() {
        if let Ok(root) = env::var("OPENBLAS_ROOT").or_else(|_| env::var("BLAS_ROOT")) {
            for sub in ["lib", "lib64"] {
                println!("cargo:rustc-link-search=native={root}/{sub}");
            }
        }
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}
