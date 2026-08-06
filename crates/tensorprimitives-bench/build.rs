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
    println!("cargo:rerun-if-env-changed=TBLIS_GNU_RUNTIME");
    println!("cargo:rerun-if-env-changed=LIBOMP_ROOT");

    if env::var("CARGO_FEATURE_TBLIS").is_ok() {
        let root = env::var("TBLIS_ROOT")
            .expect("feature `tblis` requires TBLIS_ROOT to point at a TBLIS install prefix");
        for sub in ["lib", "lib64"] {
            println!("cargo:rustc-link-search=native={root}/{sub}");
        }
        println!("cargo:rustc-link-lib=dylib=tblis");
        // The C++ runtime and OpenMP are spelled differently on Darwin, and
        // neither GNU name exists there: `libstdc++` was removed from the SDK
        // years ago and Apple ships no `libgomp` at all. A build with the GNU
        // names fails at the link step with two undefined libraries, which reads
        // as a broken TBLIS install rather than as a platform mismatch.
        //
        // The pairing matters as much as the names: link `c++`/`omp` only if
        // TBLIS itself was built with Apple clang. A TBLIS built with Homebrew
        // `g++` wants `stdc++`/`gomp` from `/opt/homebrew/lib/gcc/current`, and
        // mixing the two C++ runtimes in one process is its own class of
        // failure. `TBLIS_GNU_RUNTIME=1` selects that combination.
        let apple = env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple");
        let gnu_runtime = env::var("TBLIS_GNU_RUNTIME").is_ok();
        if apple && !gnu_runtime {
            println!("cargo:rustc-link-lib=dylib=c++");
            // **No OpenMP by default here, unlike the GNU arm.** A TBLIS
            // configured with `BLIS_THREAD_MODEL=pthread` -- which is what the
            // Darwin build uses -- links libc++ and libSystem and nothing else,
            // so an unconditional `-lomp` would demand a keg-only Homebrew
            // library the baseline does not need. Set `LIBOMP_ROOT` to opt in,
            // which is what a TBLIS built with `=openmp` requires.
            if let Ok(prefix) = env::var("LIBOMP_ROOT") {
                println!("cargo:rustc-link-search=native={prefix}/lib");
                println!("cargo:rustc-link-lib=dylib=omp");
            }
        } else {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            // TBLIS pulls in hwloc and OpenMP through TCI.
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    }

    if env::var("CARGO_FEATURE_BLAS").is_ok() {
        // Accelerate is a framework, not a library on a search path, and it
        // supplies the classic CBLAS entry points this crate declares. The
        // `framework=` link kind is the only way to express `-framework`;
        // `dylib=openblas` cannot.
        if env::var("CARGO_FEATURE_ACCELERATE").is_ok() {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        } else {
            if let Ok(root) = env::var("OPENBLAS_ROOT").or_else(|_| env::var("BLAS_ROOT")) {
                for sub in ["lib", "lib64"] {
                    println!("cargo:rustc-link-search=native={root}/{sub}");
                }
            }
            println!("cargo:rustc-link-lib=dylib=openblas");
        }
    }
}
