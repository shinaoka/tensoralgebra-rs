# Project guidance

Before acting, read the shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(fallback: `../tensor4all-agent-rules/rules/index.md`); load only the files
relevant to the task and do not vendor them here. Then read
`REPOSITORY_RULES.md`. Read `PERFORMANCE_TIPS.md` in full before implementing
or reviewing kernels, planning, caches, execution/threading code or
benchmarks, and before creating a PR that touches them. Work inside
`tensorprimitives/` or `benchmarks/` also follows that directory's own
`AGENTS.md`; where it conflicts with this file, this file and
`PERFORMANCE_TIPS.md` win for cross-cutting execution and benchmark rules.

This is a research repository, not a production tensor-algebra library. Read `README.md`, `docs/research-map.md`, `docs/experiments.md`, and `docs/provenance.md` before adding an experiment.

- Keep BLIS-style GEMM, TBLIS-style contraction and executor/FFI choices provisional until comparable measurements support a decision.
- Before timing a numerical path, check known values and reconstruction or residuals. Record the provider version, build flags, CPU, layout, dtype, shape, batch size, thread count, and timed boundary with results.
- Cite the original paper and any implementation consulted in the source file when code is written. Clearly label a port or close translation, preserve upstream notices, and review imported tests file by file.
- Keep host-controlled threading explicit. Do not introduce an ambient global pool in a library experiment without identifying it as the behavior under test.
- Do not publish a package or present benchmark claims from unrecorded measurements.

## Layout

- `crates/`: new `tprims-*` crates.
- `tensorprimitives/`: tensorprimitives-rs by Lukas Devos, imported with
  history (`tensorcontract`, `tensorprimitives-tapp`, `tensorprimitives-bench`).
- `benchmarks/`: package `tprims-bench` (started as an import of
  strided-rs-benchmark-suite). Every new operation adds rows here at 1T and 4T.
- `experiments/`: standalone measurement probes, excluded from the workspace.
- `.agents/skills/` (canonical, read by Codex and pi), mirrored byte for byte
  in `.claude/skills/`, with OpenCode commands in `.opencode/commands/`;
  `python3 scripts/check-agent-skills.py` checks the mirrors. Run benchmarks
  with the `tprims-benchmark` skill.
- strided-rs is an external git dependency pinned to the v0.4.4 commit,
  spelled exactly as in tenferro-rs (`Cargo.toml`); strided changes and
  strided benchmarks go to tensor4all/strided-rs and
  tensor4all/strided-rs-benchmark-suite.

## Build

- The build job count depends on the host: set `CARGO_BUILD_JOBS` (16 on the
  shared 64-core EPYC workstation); scripts and CI do not hardcode `-j`.
- Local gate before a PR: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`, `cargo test -p tprims-exec` (without
  the `strided` feature the workspace unifies on), and
  `cargo test -p tensorcontract --release`. Build `cargo build -p tprims-bundle`
  before the workspace tests (the C ABI test links the built `libtprims.so`). CI uses stable clippy,
  which may be newer than a local toolchain.
- Before a PR, also check README and `docs/` against the implementation
  (REPOSITORY_RULES.md, Public Surface Drift): diagrams and dependency tables
  against `cargo tree`, names, features, commands and status against the code.
