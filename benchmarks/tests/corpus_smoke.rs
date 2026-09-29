//! Replays the example corpus through the benchmark binaries once, at one
//! thread, and requires every strategy comparison to agree.
use std::process::Command;

fn replay(bin: &str) -> String {
    let corpus = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/benchmarks/tprims/corpus/example.json"
    );
    let out = Command::new(bin)
        .args(["--threads", "1", "--corpus", corpus])
        .env("BENCH_RUNS", "1")
        .env("BENCH_WARMUP", "0")
        .env_remove("RAYON_NUM_THREADS")
        .output()
        .expect("run");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{bin} failed: {text}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    text
}

fn checks(text: &str) -> Vec<&str> {
    text.lines().filter(|l| l.starts_with("CHECK")).collect()
}

#[test]
fn contract_replays_every_dot_general_entry() {
    let text = replay(env!("CARGO_BIN_EXE_contract"));
    let c = checks(&text);
    assert_eq!(c.len(), 4, "{text}");
    assert!(c.iter().all(|l| l.ends_with(" ok")), "{text}");
}

#[test]
fn blas_replays_every_gemm_batched_entry() {
    let text = replay(env!("CARGO_BIN_EXE_blas"));
    let c = checks(&text);
    assert_eq!(c.len(), 2, "{text}");
    assert!(c.iter().all(|l| l.ends_with(" ok")), "{text}");
}
