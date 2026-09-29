# Phase 0: repository consolidation

Date: 2026-09-29. Status: approved design, pending spec review.

## Goal

Before Phase 1 (the tenferro-rs CPU backend), bring the code that Phase 1
builds on into this repository with its history, make it one Cargo workspace,
and give the repository the rules that Phase 1 work must follow. Phase 1
steps 1a to 1f then start from a repository in which `strided-*`,
`tensorcontract` and the benchmark suite are local crates.

## Decisions taken with the maintainer

| Question | Decision |
| --- | --- |
| Phase 1 scope for now | 1a, 1b, 1d, 1c, 1f in that order; 1e (tenferro integration) later, in tenferro-rs |
| tensorprimitives-rs | Import now with `git subtree add`, no `--squash` |
| strided-rs | Import now with `git subtree add`, no `--squash` |
| strided-rs-benchmark-suite | Import now with `git subtree add`, no `--squash` |
| Placement of `tprims-exec` / `tprims-core` | This repository. With `strided-*` imported, the repository-level cycle in the decision log disappears |
| strided-rs crates outside the stack | Move `strided-rs` (facade), `strided-einsum2`, `strided-opteinsum`, `mdarray-opteinsum`, `ndarray-opteinsum` to `strided/deprecated/` now, out of the workspace |
| Einsum benchmarks | Keep only `benchmarks/strided_benchmarks/` alive; the einsum half is frozen under `benchmarks/deprecated/` |
| Upstream repositories and crates.io | Unchanged in Phase 0. Crate names stay `strided-*`. Archiving upstream and moving the publish source is follow-up work |
| Workflow | A branch and a PR per phase step |
| Benchmarks | Every Phase 1 implementation adds benchmarks, measured at 1 and 4 threads in the same run |
| Rules | Port the rules Phase 1 needs from tenferro-rs, especially the performance rules |

## Layout after Phase 0

```text
tprims-rs/
  Cargo.toml              root workspace (the only [workspace])
  AGENTS.md  REPOSITORY_RULES.md  PERFORMANCE_TIPS.md
  crates/                 new tprims-* crates (created from Phase 1a on)
  strided/                subtree of tensor4all/strided-rs
    strided-traits strided-view strided-perm strided-basic strided-fused strided-kernel
    deprecated/           strided-rs facade, einsum2, opteinsum, mdarray-/ndarray-opteinsum
  tensorprimitives/       subtree of lkdvos/tensorprimitives-rs
    crates/tensorcontract crates/tensorprimitives-tapp crates/tensorprimitives-bench
  benchmarks/             subtree of tensor4all/strided-rs-benchmark-suite
    benchmarks/strided_benchmarks/   kept alive
    deprecated/                      einsum benchmarks, Python/Julia runners
  experiments/            unchanged, excluded from the workspace
  docs/
```

## Commit sequence (one PR)

1. **Three subtree imports**, each a separate merge commit, from `origin/main`
   of the upstream repository, never from a local checkout:

   | Source | Prefix | Commit |
   | --- | --- | --- |
   | `https://github.com/tensor4all/strided-rs` | `strided/` | `71b7cb9` |
   | `https://github.com/lkdvos/tensorprimitives-rs` | `tensorprimitives/` | `8cda75e` |
   | `https://github.com/tensor4all/strided-rs-benchmark-suite` | `benchmarks/` | `0550611` |

   If `main` has moved when the import runs, use the new head and record it.
   The import commits contain no other change.

2. **Deprecations.** `git mv` the five out-of-stack strided crates to
   `strided/deprecated/<crate>/` and the einsum benchmark half (`src/main.rs`,
   `src/main.jl`, `benchmarks/einsum_benchmarks/`, the Python extraction
   pipeline, `Project.toml`/`Manifest.toml`, `pyproject.toml`/`uv.lock`, einsum
   data and results) to `benchmarks/deprecated/`. Each `deprecated/` directory
   gets a README stating why it is frozen and what replaces it
   (`tprims-contract` for binary contraction). Deprecated code is not built.

3. **One workspace.**
   - Root `Cargo.toml` declares every live crate as a member:
     `strided/strided-{traits,view,perm,basic,fused,kernel}`,
     `tensorprimitives/crates/{tensorcontract,tensorprimitives-tapp,tensorprimitives-bench}`,
     `benchmarks`. `experiments/*` and every `deprecated/` path are excluded.
   - Delete `strided/Cargo.toml` and `tensorprimitives/Cargo.toml` (they hold
     only `[workspace]` tables) after hoisting their content. `[workspace.package]` values that differ per origin
     (version, authors, repository) move into each crate's own manifest so
     that no published metadata changes: `strided-*` keep version `0.4.4`,
     their authors and `repository = tensor4all/strided-rs`; `tensorcontract`
     keeps its metadata.
   - `[workspace.dependencies]` is the union of both origins; a version
     conflict is resolved to one version and noted in the commit message.
   - `rust-version = "1.89"` for the workspace (tensorprimitives' floor).
   - Profiles: `[profile.dev] debug = 0` from strided-rs;
     `[profile.release]` `opt-level = 3, lto = "thin", codegen-units = 1,
     debug = 1` and `[profile.bench] inherits = "release"` from
     tensorprimitives-rs; `release-with-debug` from the benchmark suite. The
     benchmark suite's release numbers recorded before the import were built
     without thin LTO; results after Phase 0 record the new profile.
   - The benchmark package drops its einsum dependencies, rewrites
     `path = "../strided-rs/..."` to workspace dependencies, and is renamed
     `tprims-bench` (binary names unchanged). Phase 1 benchmarks are added
     here.
   - The nested `.github/` directories are removed (GitHub ignores them);
     their jobs are ported to the root CI below.
   - One root `Cargo.lock`; the nested lock files are removed.

4. **Rules** (next section).

5. **CI.** `.github/workflows/ci.yml` on Linux: `cargo fmt --check`,
   `cargo clippy -j 16 --workspace --all-targets -- -D warnings`,
   `cargo test -j 16 --workspace`, and doc build. Clippy findings that the
   upstream CIs did not enforce are fixed in a separate commit or given an
   `// INVARIANT:`-justified allow; the import commits stay untouched.
   Coverage gating, macOS and Windows are Phase 1f / Phase 2 work.

6. **Documentation.** Update `README.md`, `docs/architecture.md`,
   `docs/decision-log.md` (repository placement: **Decided: monorepo**;
   subtree imports done; Phase 0 row), and `docs/provenance.md` (the three
   imports, source commits, retained licenses: strided-rs MIT OR Apache-2.0
   with its `NOTICE` and `THIRD-PARTY-LICENSES` for HPTT-derived code,
   tensorprimitives-rs MIT OR Apache-2.0, benchmark suite license file as
   found). Remove the statements "no source migration has been performed" and
   "no third-party code is included". The repository license stays a pending
   maintainer choice; imported code keeps its own licenses.

## Rules ported from tenferro-rs

`AGENTS.md` keeps its research-repository guidance and adds, as tenferro-rs
does: read the shared `tensor4all-agent-rules` first (never vendored), then
`REPOSITORY_RULES.md`, then `PERFORMANCE_TIPS.md` in full before
performance-sensitive work. The subtrees' own `AGENTS.md` files remain and
apply inside their directories.

`PERFORMANCE_TIPS.md`, adapted to tprims:

| tenferro section | Treatment |
| --- | --- |
| Performance-Sensitive Safety Contracts | Port (`// INVARIANT:` for raw scratch, unchecked indexing after validation; no zero-fill in shared hot-path acquisition) |
| Complexity Budget | Port, restated for plans (planning, not graphs) |
| Materialization And Copies | Port, tied to DLPack zero copy and `TPRIMS_NO_MATERIALIZE` |
| Dense Layout And Linear Algebra | Rewrite: arbitrary strides at the boundary (DLPack), no hidden layout normalization; keep the batched-linalg rule (scratch and workspace query hoisted out of the batch loop, no per-item allocation, small-matrix large-batch regression case) |
| Range Checks And Slicing | Port (validate once, carry metadata inward, negative strides allowed when range-proven) |
| Faer Integration | Port; parallelism comes from the `Exec` context, never chosen in a helper |
| Performance Anti-Patterns | Port (dispatch once per call, no per-element runtime match, no heap allocation in hot loops, no planning inside execution) |
| Performance-Sensitive Tests And Benchmarks | Port with tprims specifics: release-mode Criterion; 1T and 4T rows of every tensor-sized case in the same run, a 4T time not faster than 1T is a finding; effective thread count built from the `Exec` context and asserted at startup, conflicting thread environment variables fail the run; every public operation has benchmark rows; build jobs `-j 16` separate from measured threads; pinned idle core; A/A noise floor |
| Performance-Gated Experiment Protocol | Port unchanged in substance |
| Cache Ownership | Port for plan and scratch caches: explicit owner, bound, clear, stats |
| CPU Threading Contract | Rewrite around `tprims-exec`: no ambient pool; serial work never enters a pool; a parallel kernel enters the borrowed pool itself, only if the caller is not already a worker; width from work; budget, active width and dispatch width distinct; no threads created beyond the pool; provider (BLAS/OpenMP) thread variables stay with the provider |
| Audit Procedure | Port, without the routing script |

`REPOSITORY_RULES.md`, adapted: Public Surface Drift; Public Surface
Discipline (small public API, `Debug` on public types); Public Boundary
Safety Audits (checked shape/stride/offset arithmetic, no panics on user
input, typed errors); Invariant Markers; No Ad Hoc Fixes; Unsafe Code
Boundary; File Organization and Unit Test Organization; Work Logs And Design
Records; CI Cost Discipline; Publication Order And Publish-Safety (no new
crates.io package without explicit approval); Generic Over Scalar Type; PR
Content Hygiene. Not ported: AD, oracle, GPU, device transfer, session-entry
audit, extension boundary, tenferro tensor data model.

Where a ported rule names a tenferro type (`CpuContext`, `TypedTensorView`,
`TENFERRO_*` variables), it is restated in tprims terms or dropped; no rule
refers to a mechanism this repository does not have.

## Done when

- `git log --follow` shows upstream authorship for a sample file from each
  import (for example `strided/strided-perm/src/lib.rs`,
  `tensorprimitives/crates/tensorcontract/src/lib.rs`).
- `cargo fmt --check`, `cargo clippy -j 16 --workspace --all-targets -- -D warnings`
  and `cargo test -j 16 --workspace` pass at the root.
- The strided crates' tests pass with the same count as upstream at the
  imported commit (minus the deprecated crates), and tensorcontract's tests
  pass.
- `cargo build -j 16 --release -p tprims-bench --bins` succeeds and one
  strided benchmark runs at 1T and 4T.
- Root CI is green on the PR.

## After Phase 0

Each Phase 1 step gets its own spec, plan and PR, in the order 1a
(`tprims-exec`, including explicit execution for `strided-perm` /
`strided-kernel`), 1b (`tprims-blas`), 1d (`tprims-linalg`), 1c
(`tprims-contract`, with `tensorcontract` as the TBLIS-style strategy), 1f
(C ABI slice). Every step's completion criteria include `tprims-bench`
benchmarks at 1T and 4T in the same run, with the effective thread count
verified, recorded under the experiment protocol.

Follow-up outside Phase 0: archive or redirect `tensor4all/strided-rs` and
`tensor4all/strided-rs-benchmark-suite`, move `strided-*` publication to this
repository, and coordinate the tenferro-rs dependency.
