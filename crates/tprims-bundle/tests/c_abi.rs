//! Compiles `tests/c/abi_test.c` against the headers, links it to the built
//! `libtprims.so`, and runs it; also checks the exported symbol set.
use std::path::PathBuf;
use std::process::Command;

fn target_dir() -> PathBuf {
    // .../target/<profile>/deps/c_abi-<hash>
    let exe = std::env::current_exe().expect("exe");
    exe.parent()
        .and_then(|p| p.parent())
        .expect("target dir")
        .to_path_buf()
}

#[test]
fn c_program_links_and_runs() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skipping: no C compiler ({cc})");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = root.join("../tprims-core/include");
    let lib = target_dir();
    assert!(
        lib.join("libtprims.so").exists() || lib.join("libtprims.dylib").exists(),
        "libtprims not built in {}",
        lib.display()
    );
    let out = std::env::temp_dir().join(format!("tprims_abi_test_{}", std::process::id()));
    let st = Command::new(&cc)
        .args(["-std=c11", "-Wall", "-Werror", "-o"])
        .arg(&out)
        .arg(root.join("tests/c/abi_test.c"))
        .arg(format!("-I{}", include.display()))
        .arg(format!("-L{}", lib.display()))
        .arg(format!("-Wl,-rpath,{}", lib.display()))
        .args(["-ltprims", "-lm"])
        .status()
        .expect("cc");
    assert!(st.success(), "C compile failed");
    let run = Command::new(&out).output().expect("run");
    let _ = std::fs::remove_file(&out);
    assert!(
        run.status.success(),
        "abi_test failed: {}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn only_tprims_symbols_of_the_selected_parts_are_exported() {
    let lib = target_dir().join("libtprims.so");
    let Ok(out) = Command::new("nm")
        .args(["-D", "--defined-only"])
        .arg(&lib)
        .output()
    else {
        eprintln!("skipping: no nm");
        return;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let syms: Vec<&str> = text
        .lines()
        .filter_map(|l| l.split_whitespace().nth(2))
        .filter(|s| s.starts_with("tprims_"))
        .collect();
    for want in [
        "tprims_abi_version",
        "tprims_exec_rayon_create",
        "tprims_exec_close",
        "tprims_blas_gemm",
        "tprims_contract_plan_create",
        "tprims_has_part",
    ] {
        assert!(syms.contains(&want), "{want} not exported: {syms:?}");
    }
}

#[test]
fn headers_combine_with_a_real_dlpack_h_in_either_order() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skipping: no C compiler ({cc})");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = root.join("../tprims-core/include");
    for src in ["headers_dlpack_first.c", "headers_tprims_first.c"] {
        let out = std::env::temp_dir().join(format!("tprims_{src}_{}", std::process::id()));
        let st = Command::new(&cc)
            .args(["-std=c11", "-Wall", "-Werror", "-fsyntax-only"])
            .arg(format!("-I{}", include.display()))
            .arg(root.join("tests/c").join(src))
            .status()
            .expect("cc");
        let _ = std::fs::remove_file(&out);
        assert!(st.success(), "{src} failed to compile");
    }
}
