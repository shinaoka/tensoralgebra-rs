# Project guidance

Before acting, read the shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(fallback: `../tensor4all-agent-rules/rules/index.md`); load only the files
relevant to the task and do not vendor them here. Then read
`REPOSITORY_RULES.md`. Read `PERFORMANCE_TIPS.md` in full before implementing
or reviewing kernels, planning, caches, execution/threading code or
benchmarks, and before creating a PR that touches them. Work inside
`strided/`, `tensorprimitives/` or `benchmarks/` also follows that
directory's own `AGENTS.md`; where it conflicts with this file, this file and
`PERFORMANCE_TIPS.md` win for cross-cutting execution and benchmark rules.

This is a research repository, not a production tensor-algebra library. Read `README.md`, `docs/research-map.md`, `docs/experiments.md`, and `docs/provenance.md` before adding an experiment.

- Keep BLIS-style GEMM, TBLIS-style contraction, batched linalg, and executor/FFI choices provisional until comparable measurements support a decision.
- Before timing a numerical path, check known values and reconstruction or residuals. Record the provider version, build flags, CPU, layout, dtype, shape, batch size, thread count, and timed boundary with results.
- Cite the original paper and any implementation consulted in the source file when code is written. Clearly label a port or close translation, preserve upstream notices, and review imported tests file by file.
- Keep host-controlled threading explicit. Do not introduce an ambient global pool in a library experiment without identifying it as the behavior under test.
- Do not publish a package or present benchmark claims from unrecorded measurements.

## Layout

- `crates/`: new `tprims-*` crates.
- `strided/`: strided-rs, imported with history (`strided-traits`, `-view`,
  `-perm`, `-basic`, `-fused`, `-kernel`).
- `tensorprimitives/`: tensorprimitives-rs by Lukas Devos, imported with
  history (`tensorcontract`, `tensorprimitives-tapp`, `tensorprimitives-bench`).
- `benchmarks/`: strided-rs-benchmark-suite, imported with history, package
  `tprims-bench`. Every new operation adds rows here at 1T and 4T.
- `experiments/`: standalone measurement probes, excluded from the workspace.
- `deprecated/` directories are frozen: not built, not edited.

## Build

- Every cargo invocation uses `-j 16`.
- Local gate before a PR: `cargo fmt --all -- --check`,
  `cargo clippy -j 16 --workspace --all-targets -- -D warnings`, the same with
  `--features strided-basic/parallel,strided-basic/tprims-exec,strided-kernel/parallel,strided-perm/parallel,tprims-bench/parallel`,
  `cargo test -j 16 --workspace`, and
  `cargo test -j 16 -p strided-basic -p strided-kernel -p strided-perm --features parallel`,
  the strided crates alone without features (the workspace run unifies
  `parallel` on through `tprims-bench`), and
  `cargo test -j 16 -p tensorcontract --release`. Build `cargo build -j 16 -p tprims-bundle`
  before the workspace tests (the C ABI test links the built `libtprims.so`). CI uses stable clippy,
  which may be newer than a local toolchain.
