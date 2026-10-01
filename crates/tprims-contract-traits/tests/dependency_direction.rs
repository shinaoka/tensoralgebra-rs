//! The interface crate sits below every implementation: its normal dependency
//! closure contains none of them, no executor runtime, no kernel layer, no
//! faer and no tenferro.
use std::process::Command;

fn tree(package: &str) -> String {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = Command::new(cargo)
        .args([
            "tree",
            "-p",
            package,
            "-e",
            "normal",
            "--prefix",
            "none",
            "--offline",
        ])
        .output()
        .expect("run cargo tree");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn crate_names(tree: &str) -> Vec<&str> {
    tree.lines()
        .filter_map(|l| l.split_whitespace().next())
        .collect()
}

#[test]
fn the_interface_crate_depends_on_no_implementation() {
    let deps = tree("tprims-contract-traits");
    let names = crate_names(&deps);
    for forbidden in [
        "tprims-contract",
        "tprims-blas",
        "tprims-exec",
        "tprims-core",
        "tprims-gemm-kernel",
        "tensorcontract",
        "tprims-linalg",
        "faer",
        "rayon",
        "tenferro-rs",
    ] {
        assert!(
            !names.contains(&forbidden),
            "{forbidden} is in the interface crate's closure:\n{deps}"
        );
    }
}

#[test]
fn the_test_backend_depends_on_the_interface_but_not_on_tprims_contract() {
    let deps = tree("tprims-contract-testkit");
    let names = crate_names(&deps);
    assert!(names.contains(&"tprims-contract-traits"), "{deps}");
    for forbidden in [
        "tprims-contract ",
        "tprims-blas",
        "tprims-exec",
        "tensorcontract",
        "faer",
    ] {
        assert!(
            !deps.lines().any(|l| l.starts_with(forbidden)),
            "{forbidden} in the test backend's closure:\n{deps}"
        );
    }
    // Implementations depend on the interface, never the reverse.
    let contract = tree("tprims-contract");
    assert!(crate_names(&contract).contains(&"tprims-contract-traits"));
}
