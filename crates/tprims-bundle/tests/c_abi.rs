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

/// The bundle's file name and the `nm` arguments listing its exported,
/// defined symbols on this platform.
fn library_and_nm_args() -> (&'static str, &'static [&'static str]) {
    if cfg!(target_os = "macos") {
        ("libtprims.dylib", &["-gU"])
    } else {
        ("libtprims.so", &["-D", "--defined-only"])
    }
}

/// `tprims_*` names in `nm` output (`[address] type name` per line), with
/// the Mach-O leading underscore removed when `macho`.
fn tprims_symbols(nm_output: &str, macho: bool) -> Vec<String> {
    nm_output
        .lines()
        .filter_map(|l| l.split_whitespace().last())
        .map(|s| {
            if macho {
                s.strip_prefix('_').unwrap_or(s)
            } else {
                s
            }
        })
        .filter(|s| s.starts_with("tprims_"))
        .map(str::to_owned)
        .collect()
}

#[test]
fn nm_output_is_parsed_for_elf_and_macho() {
    let elf = "0000000000012340 T tprims_abi_version\n                 U malloc\n0000000000012350 T tprims_blas_gemm\n";
    assert_eq!(
        tprims_symbols(elf, false),
        ["tprims_abi_version", "tprims_blas_gemm"]
    );
    let macho = "0000000000003f10 T _tprims_abi_version\n0000000000003f20 T _tprims_has_part\n0000000000003f30 T _other\n";
    assert_eq!(
        tprims_symbols(macho, true),
        ["tprims_abi_version", "tprims_has_part"]
    );
}

#[test]
fn only_tprims_symbols_of_the_selected_parts_are_exported() {
    let (name, args) = library_and_nm_args();
    let lib = target_dir().join(name);
    assert!(
        lib.exists(),
        "{} not built: run `cargo build -p tprims-bundle` with the same profile first",
        lib.display()
    );
    let out = Command::new("nm")
        .args(args)
        .arg(&lib)
        .output()
        .expect("nm is required to inspect the exported symbols");
    assert!(
        out.status.success(),
        "nm {args:?} {} failed: {}",
        lib.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let syms = tprims_symbols(
        &String::from_utf8_lossy(&out.stdout),
        cfg!(target_os = "macos"),
    );
    for want in [
        "tprims_abi_version",
        "tprims_exec_rayon_create",
        "tprims_exec_close",
        "tprims_blas_gemm",
        "tprims_contract_plan_create",
        "tprims_has_part",
    ] {
        assert!(
            syms.iter().any(|s| s == want),
            "{want} not exported: {syms:?}"
        );
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
