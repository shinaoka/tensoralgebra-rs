# Phase 0 Repository Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import strided-rs, tensorprimitives-rs and strided-rs-benchmark-suite into tprims-rs with full history, make one Cargo workspace, port the tenferro-rs rules Phase 1 needs, and add Linux CI.

**Architecture:** Three `git subtree add` merge commits (no squash) under `strided/`, `tensorprimitives/`, `benchmarks/`; then separate commits that deprecate out-of-stack code, hoist the nested workspaces into one root workspace, port rules, add CI, and update docs. The import commits never contain hand edits.

**Tech Stack:** git subtree, Cargo workspaces, Rust 1.89 floor (local toolchain 1.97.1), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-29-phase0-repository-consolidation-design.md`

## Global Constraints

- Work in `/home/shinaoka/tensor4all/tprims-rs` on branch `phase0-consolidation` (already created from `origin/main`, contains the spec commit).
- Every cargo invocation passes `-j 16`.
- Import from upstream `main` URLs, never from local checkouts. Pinned heads (verified 2026-09-29): strided-rs `71b7cb9e4b027f37c16fa79fc411a069042d9f6d`, tensorprimitives-rs `8cda75e11ed26f46c0c22f9629004c84dbabc8e5`, strided-rs-benchmark-suite `05506112eb40f7cff7ac102689614c180a7d6120`. If `git ls-remote` shows a different head, import that head and record it.
- `git subtree add` without `--squash`.
- Published crate metadata must not change: `strided-*` keep version `0.4.4`, `license = "MIT OR Apache-2.0"`, `authors = ["Satoshi Terasaki", "Hiroshi Shinaoka"]`, `repository = "https://github.com/tensor4all/strided-rs"`; `tensorcontract`/`tensorprimitives-*` keep version `0.1.0`, `authors = ["Lukas Devos <ldevos@flatironinstitute.org>"]`, `repository`/`homepage = "https://github.com/lkdvos/tensorprimitives-rs"`.
- Workspace `rust-version = "1.89"`.
- Do not publish anything. Do not modify upstream repositories.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

## Review Focus

- A crate that silently changes its published metadata (version/authors/repository) when `*.workspace = true` is rewritten: check `cargo metadata` output for every strided and tensorprimitives package against upstream.
- Feature unification in one workspace: strided `parallel` features or `num-complex` features unified differently than upstream could change behavior; test both default and `--features parallel` runs as upstream CI did.
- `tensorcontract`'s `readme = "../../README.md"` and `include`-style relative paths must still resolve (`cargo package --list -p tensorcontract --allow-dirty` must succeed).
- The benchmark crate must not compile the deprecated einsum crates after the move (`cargo tree -p tprims-bench | grep -c einsum` = 0).
- `git log --follow` must still reach upstream authorship through the deprecation `git mv` commit.

---

### Task 1: Subtree imports

**Files:**
- Create (via subtree): `strided/**`, `tensorprimitives/**`, `benchmarks/**`

- [ ] **Step 1: Confirm upstream heads**

```bash
cd /home/shinaoka/tensor4all/tprims-rs
git status --short   # must be empty
git ls-remote https://github.com/tensor4all/strided-rs main
git ls-remote https://github.com/lkdvos/tensorprimitives-rs main
git ls-remote https://github.com/tensor4all/strided-rs-benchmark-suite main
```
Expected: the three pinned hashes in Global Constraints (or record new ones).

- [ ] **Step 2: Import strided-rs**

```bash
git subtree add --prefix=strided https://github.com/tensor4all/strided-rs main \
  -m "Import strided-rs at 71b7cb9 with history (git subtree, no squash)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Import tensorprimitives-rs**

```bash
git subtree add --prefix=tensorprimitives https://github.com/lkdvos/tensorprimitives-rs main \
  -m "Import tensorprimitives-rs at 8cda75e with history (git subtree, no squash)

Authored by Lukas Devos; MIT OR Apache-2.0.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Import strided-rs-benchmark-suite**

```bash
git subtree add --prefix=benchmarks https://github.com/tensor4all/strided-rs-benchmark-suite main \
  -m "Import strided-rs-benchmark-suite at 0550611 with history (git subtree, no squash)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Verify history is preserved**

```bash
git log --follow --format='%an %h' -- strided/strided-perm/src/lib.rs | tail -3
git log --follow --format='%an %h' -- tensorprimitives/crates/tensorcontract/src/lib.rs | tail -3
git log --format='%an' -- benchmarks | sort | uniq -c
```
Expected: upstream authors (Hiroshi Shinaoka / Satoshi Terasaki; lkdvos / Lukas Devos). No push yet.

### Task 2: Deprecate out-of-stack code

**Files:**
- Move: `strided/{strided-rs,strided-einsum2,strided-opteinsum,mdarray-opteinsum,ndarray-opteinsum}` → `strided/deprecated/<same name>`
- Move: `benchmarks/src/main.rs`, `benchmarks/src/main.jl`, `benchmarks/benchmarks/einsum_benchmarks`, `benchmarks/scripts/{convert_tensornetwork.py,create_lightweight_instance.py,generate_dataset.py,run_all.sh,run_all_julia.sh,run_all_rust.sh,format_results.py}`, `benchmarks/{Project.toml,Manifest.toml,pyproject.toml,uv.lock}`, `benchmarks/data`, `benchmarks/result` → `benchmarks/deprecated/einsum/<same relative path>`
- Create: `strided/deprecated/README.md`, `benchmarks/deprecated/README.md`

- [ ] **Step 1: Move strided crates**

```bash
cd /home/shinaoka/tensor4all/tprims-rs
for c in strided-rs strided-einsum2 strided-opteinsum mdarray-opteinsum ndarray-opteinsum; do
  git mv strided/$c strided/deprecated/$c
done
```
(`strided/deprecated/` already exists upstream with `benches/`.)

- [ ] **Step 2: Move einsum benchmark half**

```bash
mkdir -p benchmarks/deprecated/einsum/src benchmarks/deprecated/einsum/scripts benchmarks/deprecated/einsum/benchmarks
git mv benchmarks/src/main.rs benchmarks/src/main.jl benchmarks/deprecated/einsum/src/
git mv benchmarks/benchmarks/einsum_benchmarks benchmarks/deprecated/einsum/benchmarks/
for f in convert_tensornetwork.py create_lightweight_instance.py generate_dataset.py run_all.sh run_all_julia.sh run_all_rust.sh format_results.py; do
  git mv benchmarks/scripts/$f benchmarks/deprecated/einsum/scripts/
done
git mv benchmarks/Project.toml benchmarks/Manifest.toml benchmarks/pyproject.toml benchmarks/uv.lock benchmarks/deprecated/einsum/
git mv benchmarks/data benchmarks/result benchmarks/deprecated/einsum/
```
If `benchmarks/result/amd-cpu/dense-kernels` or `uninit-copy-kernels` is referenced from `benchmarks/benchmarks/strided_benchmarks/**/README.md`, keep `benchmarks/result/` in place instead (check first with `grep -rn 'result/' benchmarks/benchmarks/strided_benchmarks`), and move only einsum-specific result folders.

- [ ] **Step 3: Write the two READMEs**

`strided/deprecated/README.md`:
```markdown
# Frozen strided-rs code

These crates came with the strided-rs import (upstream `71b7cb9`) and are
outside the tprims stack: the `strided-rs` facade (the stack has no facade),
`strided-einsum2` (binary contraction moves to `tprims-contract`),
`strided-opteinsum` and the `mdarray-`/`ndarray-opteinsum` adapters (N-ary
einsum stays above the stack). They are not workspace members and are not
built or tested. Do not edit them; the published 0.4.x releases remain on
crates.io and in the upstream repository.
```

`benchmarks/deprecated/README.md`:
```markdown
# Frozen einsum benchmarks

The einsum half of strided-rs-benchmark-suite (upstream `0550611`): the
strided-opteinsum runner, the OMEinsum.jl runner, the Python metadata
pipeline, fixtures and results. It depends on crates frozen under
`strided/deprecated/` and is not built. Live benchmarks are in
`benchmarks/benchmarks/strided_benchmarks/` and the `tprims-bench` package.
```

- [ ] **Step 4: Commit**

```bash
git add -A strided/deprecated benchmarks
git commit -m "Freeze out-of-stack strided crates and einsum benchmarks under deprecated/

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git log --follow --format='%an' -- strided/deprecated/strided-einsum2/src/lib.rs | sort | uniq -c
```
Expected: upstream authors still listed.

### Task 3: One root workspace

**Files:**
- Create: `Cargo.toml`, `Cargo.lock` (generated)
- Delete: `strided/Cargo.toml`, `tensorprimitives/Cargo.toml`, `strided/Cargo.lock` (if present), `tensorprimitives/Cargo.lock`, `benchmarks/Cargo.lock`, `strided/.github/`, `tensorprimitives/.github/`
- Modify: every `Cargo.toml` under `strided/strided-*/`, `tensorprimitives/crates/*/`, and `benchmarks/Cargo.toml`

- [ ] **Step 1: Write the root manifest**

Copy every `[workspace.dependencies]` entry from `strided/Cargo.toml` (paths prefixed with `strided/`, deprecated crates omitted) and from `tensorprimitives/Cargo.toml`, and copy every `[profile.*]` table from both files (strided's `[profile.dev] debug = 0` with its comment and any `[profile.ci]`; tensorprimitives' `[profile.release]` and `[profile.bench]`; the benchmark suite's `[profile.release-with-debug]`). The result must look like:

```toml
[workspace]
resolver = "2"
members = [
    "strided/strided-traits",
    "strided/strided-view",
    "strided/strided-perm",
    "strided/strided-basic",
    "strided/strided-fused",
    "strided/strided-kernel",
    "tensorprimitives/crates/tensorcontract",
    "tensorprimitives/crates/tensorprimitives-tapp",
    "tensorprimitives/crates/tensorprimitives-bench",
    "benchmarks",
]
exclude = ["experiments", "strided/deprecated", "benchmarks/deprecated"]

[workspace.package]
edition = "2021"
rust-version = "1.89"

[workspace.dependencies]
strided-basic = { version = "0.4.4", path = "strided/strided-basic", default-features = false }
strided-fused = { version = "0.4.4", path = "strided/strided-fused", default-features = false }
strided-traits = { version = "0.4.4", path = "strided/strided-traits" }
strided-view = { version = "0.4.4", path = "strided/strided-view" }
strided-perm = { version = "0.4.4", path = "strided/strided-perm" }
strided-kernel = { version = "0.4.4", path = "strided/strided-kernel" }
tensorcontract = { version = "0.1.0", path = "tensorprimitives/crates/tensorcontract" }

approx = "0.5"
criterion = "0.5"
faer = { version = "0.24", default-features = false, features = ["std", "rayon"] }
faer-traits = "0.24"
num-complex = "0.4"
num-traits = "0.2"
pulp = "0.22"
rand = "0.8"
rand_distr = "0.4"
rayon = "1.10"
smallvec = "1"
thiserror = "1.0"
# ...any remaining entries from strided/Cargo.toml still used by a live crate

[profile.dev]
debug = 0

[profile.test]
debug = 0

# Hosted CI starts from cold caches, where incremental state has no value.
[profile.ci]
inherits = "test"
incremental = false
strip = "symbols"

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
debug = 1

[profile.bench]
inherits = "release"

[profile.release-with-debug]
inherits = "release"
debug = true
```
Drop dependency entries used only by deprecated crates (`cblas-inject`, `cblas-sys`, `mdarray`, `ndarray`, `omeco`, deprecated strided crates) — verify with `grep -rn '<name>' strided/strided-*/Cargo.toml tensorprimitives/crates/*/Cargo.toml benchmarks/Cargo.toml`. `num-complex`: strided uses default features, tensorcontract used `default-features = false, features = ["std"]`; with default features on, `std` is enabled — identical in effect.

- [ ] **Step 2: Inline per-origin package metadata**

In each `strided/strided-{traits,view,perm,basic,fused,kernel}/Cargo.toml` replace
```toml
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
repository.workspace = true
```
with
```toml
version = "0.4.4"
edition.workspace = true
rust-version.workspace = true
license = "MIT OR Apache-2.0"
authors = ["Satoshi Terasaki", "Hiroshi Shinaoka"]
repository = "https://github.com/tensor4all/strided-rs"
```
In each `tensorprimitives/crates/*/Cargo.toml` replace `version/license/repository/homepage/authors .workspace = true` with the literal values from the old `tensorprimitives/Cargo.toml` `[workspace.package]` (`version = "0.1.0"`, `license = "MIT OR Apache-2.0"`, `repository = "https://github.com/lkdvos/tensorprimitives-rs"`, `homepage = "https://github.com/lkdvos/tensorprimitives-rs"`, `authors = ["Lukas Devos <ldevos@flatironinstitute.org>"]`), keeping `edition.workspace`/`rust-version.workspace`. Leave the explanatory comments in those manifests intact.

- [ ] **Step 3: Rewrite the benchmark manifest**

`benchmarks/Cargo.toml`: set `name = "tprims-bench"`, add `publish = false`, `edition.workspace = true`, `rust-version.workspace = true`; delete the `strided-opteinsum`/`strided-einsum2` dependencies, the `faer`/`blas` features and their `default`, the `tn_light_415_late_step` `[[bin]]`, and any `[profile.*]` table (now at root); replace `path = "../strided-rs/<crate>"` dependencies with `<crate> = { workspace = true, ... }` keeping their `default-features`/`features`. New features block:
```toml
[features]
default = []
parallel = ["strided-kernel/parallel", "strided-perm/parallel", "dep:rayon"]
hptt = ["dep:hptt"]
```
Check `benchmarks/benchmarks/strided_benchmarks/kernel_scaling/run.sh` and `erased_replay/run.sh`: they generate standalone manifests pointing at `STRIDED_RS_DIR` / run inside strided-rs. Change their default `STRIDED_RS_DIR` to the repository's `strided/` directory (`$(git rev-parse --show-toplevel)/strided`) and run the `erased_replay` bench as `cargo bench -j 16 -p strided-kernel --bench erased_policy_thresholds --features parallel` from the repository root.

- [ ] **Step 4: Delete nested workspace files and CI**

```bash
git rm strided/Cargo.toml tensorprimitives/Cargo.toml
git rm -r strided/.github tensorprimitives/.github
git rm --ignore-unmatch strided/Cargo.lock tensorprimitives/Cargo.lock benchmarks/Cargo.lock
```
Keep upstream `.gitignore` files. Add `/target` to the root `.gitignore` if absent.

- [ ] **Step 5: Build and test**

```bash
cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys;[print(p["name"],p["version"],p["authors"],p["repository"]) for p in json.load(sys.stdin)["packages"]]'
cargo build -j 16 --workspace --all-targets
cargo test -j 16 --workspace --no-fail-fast 2>&1 | tee /tmp/claude-2000/-home-shinaoka-tensor4all/9e93717e-bbdb-4bfe-bcc4-cb6d49651a25/scratchpad/phase0-test.log | grep -E '^test result|FAILED|panicked' | tail -40
cargo test -j 16 -p strided-basic -p strided-kernel -p strided-perm --features parallel --no-fail-fast 2>&1 | grep -E '^test result|FAILED' | tail -20
cargo tree -j 16 -p tprims-bench | grep -c einsum   # expect 0
cargo package -j 16 --list -p tensorcontract --allow-dirty >/dev/null && echo package-ok
```
Expected: metadata matches Global Constraints; all tests pass. Compare pass counts with upstream (run `cargo test -j 16 --workspace` in the scratchpad clones `.../scratchpad/strided-rs` and `.../scratchpad/tensorprimitives-rs` if a count looks off). A failure caused by thin LTO or unified features is fixed in this task and noted in the commit message.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Make tprims-rs one Cargo workspace

Hoist strided-rs and tensorprimitives-rs [workspace] tables into the root,
inline per-origin package metadata so published metadata is unchanged,
rename the benchmark package tprims-bench and drop its einsum dependencies,
remove nested CI and lock files.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 4: fmt and clippy clean

**Files:** whatever clippy flags in live crates (never `deprecated/`).

- [ ] **Step 1: Run the gates**

```bash
cargo fmt --all -- --check
cargo clippy -j 16 --workspace --all-targets -- -D warnings 2>&1 | grep -E '^(error|warning)' | sort | uniq -c | sort -rn
cargo clippy -j 16 --workspace --all-targets --features strided-basic/parallel,strided-kernel/parallel,strided-perm/parallel -- -D warnings 2>&1 | tail -5
```

Known before the import (clippy on the upstream clone, toolchain 1.97.1): `strided-traits` lib test fails with `suspicious use of + in Mul impl` (`clippy::suspicious_arithmetic_impl`); fix with an `#[allow]` plus `// INVARIANT:` if the `+` is intentional (e.g. a tropical or test semiring), since behaviour must not change. More findings may appear once that crate builds.

- [ ] **Step 2: Fix findings**

Fix each lint in the source (mechanical fixes such as `needless_range_loop`, `manual_memcpy` must keep the same loop order and bounds; for hot kernels prefer an `#[allow(clippy::...)]` with an adjacent `// INVARIANT: <why>` line over rewriting). If a lint is new with toolchain 1.97 and fires across upstream code in a way that would take more than ~30 edits, add it to a root `[workspace.lints.clippy]` table as `allow` with a comment naming the reason, and add `[lints] workspace = true` to each live crate manifest. `cargo fmt --all` for format drift.

- [ ] **Step 3: Re-run tests and commit**

```bash
cargo test -j 16 --workspace --no-fail-fast 2>&1 | grep -E 'FAILED|test result: FAILED' ; echo "exit=$?"
git commit -am "Make the consolidated workspace clippy and fmt clean

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
Expected: grep finds nothing.

### Task 5: Port rules from tenferro-rs

**Files:**
- Modify: `AGENTS.md`
- Create: `REPOSITORY_RULES.md`, `PERFORMANCE_TIPS.md`, `CLAUDE.md` (single line `@AGENTS.md`)

Source: `git -C /home/shinaoka/tensor4all/tenferro-rs show 5a4e7fd844c9e06a2276f60dae18edfe502b98b0:PERFORMANCE_TIPS.md` and `...:REPOSITORY_RULES.md` (read-only; never write in that checkout). Also read `strided/REPOSITORY_RULES.md` (its CPU Threading Contract and Performance And Benchmark Discipline sections are already strided-specific and should be reconciled, not duplicated).

- [ ] **Step 1: AGENTS.md**

Keep the existing research-repository bullets. Prepend:
```markdown
Before acting, read the shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(fallback: `../tensor4all-agent-rules/rules/index.md`); load only the files
relevant to the task and do not vendor them here. Then read
`REPOSITORY_RULES.md`. Read `PERFORMANCE_TIPS.md` in full before implementing
or reviewing kernels, planning, caches, execution/threading code or
benchmarks, and before creating a PR that touches them. Work inside
`strided/`, `tensorprimitives/` or `benchmarks/` also follows that
directory's own `AGENTS.md`; where it conflicts with this file, this file
wins for cross-cutting execution and benchmark rules.
```
Add a "Layout" section listing `crates/`, `strided/`, `tensorprimitives/`, `benchmarks/`, `experiments/`, and the `deprecated/` freeze rule. Add "Build: every cargo invocation uses `-j 16`."

- [ ] **Step 2: PERFORMANCE_TIPS.md**

Header paragraph: "tprims performance, layout, scratch, threading and benchmark contracts, applied on top of the shared `rules/rust/performance.md` and `rules/rust/numerical.md`." Then these sections, each copied from tenferro's file and edited with the substitution rules below:

| Section | Action |
|---|---|
| Audit Procedure | copy; scope = `crates/`, `strided/strided-*`, `tensorprimitives/crates/`, `benchmarks/`; drop routing-script sentence |
| Performance-Sensitive Safety Contracts | copy first two bullets; drop the `Arc`/`StorageMut` bullet |
| Complexity Budget | copy; replace "graph construction, metadata propagation ... compilation" with "plan construction, index classification, scatter/fold analysis, and batch scheduling" |
| Materialization And Copies | copy bullets 1, 2, 4, 5 (drop the GPU transfer bullet); add: "Across the C ABI, DLPack operands are never copied to cross the boundary; a forced internal copy is reported as the selected strategy and becomes an error under `TPRIMS_NO_MATERIALIZE`." |
| Dense Layout And Linear Algebra | rewrite (text below) |
| Range Checks And Slicing | copy unchanged except "column-major shape/stride/offset semantics" → "shape/stride/offset semantics of the input view" |
| Faer Integration | copy; replace "`CpuContext`" with "the `tprims-exec` context passed to the operation" |
| Performance Anti-Patterns | copy bullets 1-7; drop the trace/tracked-eager bullet |
| Performance-Sensitive Tests And Benchmarks | copy bullets 1-5 and 7-12; drop AD bullets; apply tprims edits (text below) |
| Performance-Gated Experiment Protocol | copy; replace the `TENFERRO_PROFILE_CPU_SESSION` sentence with "count pool entries, provider calls and allocations with explicit counters rather than inferring them from timing" |
| Cache Ownership | copy bullets 1-3, 5 and 7, replacing `Engine` with "the plan or context object that owns it"; drop the backend-construction bullet |
| CPU Threading Contract | rewrite (text below) |

Substitutions everywhere: `tenferro` → `tprims`; `CpuBackend`/`CpuContext`/`CpuExecutionContext` → `Exec`; `tenferro-benchmark` → `tprims-bench`; drop sentences that only make sense with graphs, AD, eager/traced modes, GPU or CubeCL.

Rewritten **Dense Layout And Linear Algebra**:
```markdown
## Dense Layout And Linear Algebra

- Operands are strided views with signed element strides and an offset
  (`strided-view`, DLPack at the C ABI). Honour the given layout; never
  normalize to a canonical order behind the caller's back.
- Compact buffers the library allocates (packed operands, factors, scratch)
  are column-major, with compute dimensions on the left and batch dimensions
  on the right, so each batch item is contiguous.
- A kernel that needs a specific layout either consumes the strides directly
  (faer `MatRef`, TBLIS packing) or materializes through `strided-perm`
  explicitly, reporting it (see Materialization And Copies).
- Batched linear algebra runs as one tight loop per call: compute scratch
  requirements before execution, query provider workspace once per call,
  allocate scratch once and reuse it across the batch, and write each result
  directly into the batch output. No per-item `Vec` for pivots or
  permutations, no per-item allocation, no per-item workspace query. Small
  matrices at large batch (2x2, batch 1024) are the regression case.

Audit hints:

- Detect: layout normalization copies before a kernel that accepts strides;
  batch dimensions left of compute dimensions in library-allocated buffers;
  workspace queries, `Vec` allocation or per-item scratch inside a batch loop.
- Fix: consume strides, hoist scratch and workspace queries, write in place.
```

Rewritten **CPU Threading Contract**:
```markdown
## CPU Threading Contract

- There is no ambient pool. Every operation that may run in parallel takes a
  `tprims_exec::Exec` argument: `Serial`, a Rayon pool borrowed from the host,
  or a host broadcast executor. The global Rayon pool is never used
  implicitly, and no code calls `rayon::current_num_threads()` or builds a
  `ThreadPoolBuilder` to decide policy.
- Serial work runs on the calling thread and never enters a pool. A kernel
  picks its width from its own work; only when that width is greater than one
  does it enter the pool, and it does not enter if the calling thread is
  already a worker of that pool.
- Keep three widths distinct: the budget set by the host, the active width
  chosen from the work, and the dispatch width actually woken. SPMD kernels
  with barriers use a full-pool broadcast only; medium widths use
  barrier-free partitions; a partition larger than the budget is
  repartitioned. No code creates threads beyond the pool (no scoped-thread
  fallback, no crate-private pool) in a production path.
- faer parallelism is derived from `Exec`: `Par::Seq` for width one,
  `Par::rayon(k)` inside the borrowed pool's `install` otherwise. Never pick
  `Par` inside a helper.
- One budget governs batch-level and inner parallelism: a batch that fans out
  over items runs each item serially.
- Thresholds are measured per kernel and machine and live in one place per
  kernel family; they are policy values, not scattered constants.
- BLAS/OpenMP provider threading stays controlled by provider variables
  (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, ...); tprims makes no placement
  promise for provider threads.

Audit hints:

- Detect: `rayon::current_num_threads`, `par_iter`/`rayon::join`/`scope`
  outside an `Exec`-entered region, `ThreadPoolBuilder` in library code,
  `std::thread::scope`/`spawn` in a production path, `Par::rayon` chosen in a
  helper, a pool entered for work below the kernel's threshold.
- Fix: thread `Exec` through, enter only in the parallel branch, derive
  widths from the work and the budget.
```

Additions to **Performance-Sensitive Tests And Benchmarks** (append as bullets):
```markdown
- Benchmarks live in `tprims-bench` (`benchmarks/`). Every public operation
  and every alternative implementation (for example faer-loop versus
  TBLIS-style batched GEMM) has rows.
- Every tensor-sized case is measured at one and four threads in the same
  run. The harness builds the `Exec` from the requested count (a bounded
  pool for 4T, `Exec::Serial` for 1T), asserts the effective width at
  startup, and fails when `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` or `TENSORCONTRACT_THREADS` conflict with it. A 4T
  row that is not faster than its 1T row is a finding.
- Record the tprims-rs commit, CPU, `taskset` core set, profile, thread count
  and timed boundary beside every published table; raw output stays out of
  git except under a result page.
```

- [ ] **Step 3: REPOSITORY_RULES.md**

Header: "tprims-specific rules on top of the shared tensor4all rules. Performance, layout, threading and benchmark contracts live in `PERFORMANCE_TIPS.md`." Copy these tenferro sections, applying the Step 2 substitutions: Public Surface Drift; Public Surface Discipline (drop the tensor-operation-naming, `IntoRankShape` and `_view` suffix bullets); Public Boundary Safety Audits (bullets 1-6 and the typed-error/`source()` bullet; drop dtype-conversion, integer CUDA parity, AD seed, traced/eager bullets); Invariant Markers (drop the tutorial bullet); Work Logs And Design Records (worklogs under `docs/worklogs/`); CI Cost Discipline; Publication Order And Publish-Safety (add: "No crate from this repository is published without the maintainer's explicit approval for that package; the imported `strided-*` and `tensorcontract` crates keep being published from their upstream repositories until a maintainer moves publication here."); No Ad Hoc Fixes; Unsafe Code Boundary; File Organization; Unit Test Organization; Generic Over Scalar Type; PR Content Hygiene. Add a new section:
```markdown
## Imported Code

- `strided/`, `tensorprimitives/` and `benchmarks/` were imported with
  `git subtree` and keep upstream history; see `docs/provenance.md`.
- Files there keep their upstream copyright and license notices. A change to
  imported code is an ordinary commit here; say in the message when it
  diverges from upstream behaviour.
- `deprecated/` directories are frozen: not built, not edited.
```
If a copied section references a script, file or mechanism that does not exist here (`repository-rules-review.py`, `check-pr-fast.sh`, `docs/spec/`), delete that sentence.

- [ ] **Step 4: Commit**

```bash
printf '@AGENTS.md\n' > CLAUDE.md
git add AGENTS.md CLAUDE.md REPOSITORY_RULES.md PERFORMANCE_TIPS.md
git commit -m "Port repository and performance rules from tenferro-rs

Adapted from tenferro-rs 5a4e7fd REPOSITORY_RULES.md and PERFORMANCE_TIPS.md;
threading and dense-layout contracts rewritten for explicit Exec contexts and
DLPack strides.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
grep -n -i -E 'tenferro|CpuContext|CubeCL|TracedTensor' REPOSITORY_RULES.md PERFORMANCE_TIPS.md
```
Expected: the grep hits only the provenance sentence in the file header, if any.

### Task 6: Root CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: -D warnings

jobs:
  fmt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - run: cargo fmt --all -- --check

  clippy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - run: cargo clippy -j 16 --workspace --all-targets -- -D warnings
      - run: cargo clippy -j 16 --workspace --all-targets --features strided-basic/parallel,strided-kernel/parallel,strided-perm/parallel -- -D warnings

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test -j 16 --workspace --no-fail-fast
      - run: cargo test -j 16 -p strided-basic -p strided-kernel -p strided-perm --features parallel --no-fail-fast
      - run: cargo test -j 16 -p tensorcontract --release
      - run: cargo run -j 16 --release -p tensorprimitives-bench -- info

  msrv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: "1.89.0"
      - run: cargo build -j 16 --workspace --all-targets

  doc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo doc -j 16 --workspace --no-deps
        env:
          RUSTDOCFLAGS: -D warnings
```

- [ ] **Step 2: Check MSRV locally if the toolchain is installable**

```bash
rustup toolchain list | grep -q 1.89 || rustup toolchain install 1.89.0 --profile minimal
cargo +1.89.0 build -j 16 --workspace --all-targets
RUSTDOCFLAGS='-D warnings' cargo doc -j 16 --workspace --no-deps
```
If the failure is a dependency requiring a newer rustc (not our source), first regenerate the lock MSRV-aware: `CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo generate-lockfile` (or add `resolver.incompatible-rust-versions = "fallback"` under `[resolver]` in `.cargo/config.toml`), commit the lock, and rerun both the stable and 1.89 builds. If strided crates fail on 1.89 (they had no MSRV job), fix the construct (not the floor) when the fix is small; otherwise raise the workspace `rust-version` to the lowest version that builds, record the measured failure in the commit message and in `docs/decision-log.md`. If `rustup` is unavailable, skip and rely on CI.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "Add root CI: fmt, clippy, tests, MSRV 1.89, docs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 7: Documentation and provenance

**Files:**
- Modify: `README.md`, `docs/architecture.md`, `docs/decision-log.md`, `docs/provenance.md`

- [ ] **Step 1: provenance.md**

Replace "No source code, tests, or benchmark harnesses from the referenced projects have been copied here." with an "Imported code" table:

| Directory | Upstream | Commit | License | Notices kept |
| --- | --- | --- | --- | --- |
| `strided/` | tensor4all/strided-rs | `71b7cb9` | MIT OR Apache-2.0 | `strided/LICENSE-*`, `strided/NOTICE`, `strided/THIRD-PARTY-LICENSES` (HPTT-derived code in `strided-perm`) |
| `tensorprimitives/` | lkdvos/tensorprimitives-rs (Lukas Devos) | `8cda75e` | MIT OR Apache-2.0 | `tensorprimitives/LICENSE-*`, per-crate `LICENSE-*` |
| `benchmarks/` | tensor4all/strided-rs-benchmark-suite | `0550611` | MIT (`benchmarks/LICENSE`) | `benchmarks/LICENSE` |

Import method: `git subtree add` without `--squash`, history and authorship preserved. Update the tensorprimitives-rs row in the license table to "Imported (Phase 0)".

- [ ] **Step 2: decision-log.md**

- Row "Repository placement of `tprims-exec` / `tprims-core`?" → "**Decided: monorepo.** strided-rs, tensorprimitives-rs and the benchmark suite are imported with history (Phase 0); `tprims-exec`/`tprims-core` live in `crates/`. No repository-level cycle remains." Evidence: "Maintainer decision, 2026-09-29".
- Row "How is tensorprimitives-rs code brought in?" → "**Done:** imported in Phase 0 with `git subtree add` (no squash) at `8cda75e`."
- Add rows: "strided-rs source location" → "**Decided:** imported under `strided/`; facade and einsum crates frozen under `strided/deprecated/`; crates.io publication stays upstream until moved explicitly." and "Benchmark suite" → "**Decided:** imported under `benchmarks/` as `tprims-bench`; einsum half frozen; every Phase 1 implementation adds 1T and 4T rows in the same run."

- [ ] **Step 3: README.md and architecture.md**

README: change the Status line to say the repository now contains imported strided-rs, tensorprimitives-rs and benchmark-suite code plus design notes; add a short "Repository layout" list (same as the spec's layout); replace "No source migration or crate publication has been performed." with "strided-rs, tensorprimitives-rs and strided-rs-benchmark-suite were imported with history in Phase 0 ([provenance](docs/provenance.md)); no crate has been published from this repository." Add a Phase 0 row above the Phase 1 table. architecture.md: in "Relationship to `strided-rs`", say the crates now live under `strided/`; mark `strided-einsum2`, `strided-opteinsum`, `mdarray-/ndarray-opteinsum` and the facade as frozen under `strided/deprecated/`; in "Status" and "Sources and provenance", remove "no third-party code is included" / "No upstream source or tests have been copied"; add Phase 0 to the implementation-order table.

- [ ] **Step 4: Commit**

```bash
git add README.md docs
git commit -m "Document the Phase 0 imports, layout and decisions

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 8: Final verification, benchmark smoke run, PR

- [ ] **Step 1: Full gate**

```bash
cargo fmt --all -- --check
cargo clippy -j 16 --workspace --all-targets -- -D warnings
cargo test -j 16 --workspace --no-fail-fast 2>&1 | grep -E 'test result' | awk '{s+=$4; f+=$6} END {print "passed",s,"failed",f}'
cargo build -j 16 --release -p tprims-bench --features parallel --bins
```
Expected: failed 0.

- [ ] **Step 2: 1T and 4T smoke run of one strided benchmark**

```bash
cd benchmarks
BENCH_RUNS=3 BENCH_FILTER=copy taskset -c 0 ../target/release/kernel_scaling --threads 1 --time | head -5
BENCH_RUNS=3 BENCH_FILTER=copy taskset -c 0-3 ../target/release/kernel_scaling --threads 4 --time | head -5
cd ..
```
Expected: CSV rows with `threads` 1 and 4; the binary's thread verification passes. (Smoke only; no numbers are published.)

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin phase0-consolidation
gh pr create --repo tensor4all/tprims-rs --base main --head phase0-consolidation \
  --title "Phase 0: import strided-rs, tensorprimitives-rs and the benchmark suite; one workspace; rules; CI" \
  --body "$(cat <<'EOF'
Implements docs/superpowers/specs/2026-09-29-phase0-repository-consolidation-design.md.

- git subtree imports (no squash, history preserved): strided-rs 71b7cb9 → strided/, tensorprimitives-rs 8cda75e → tensorprimitives/, strided-rs-benchmark-suite 0550611 → benchmarks/
- Out-of-stack strided crates and the einsum benchmarks frozen under deprecated/
- One root workspace; published metadata of imported crates unchanged; benchmark package renamed tprims-bench
- REPOSITORY_RULES.md / PERFORMANCE_TIPS.md ported from tenferro-rs 5a4e7fd, threading and layout contracts rewritten for explicit Exec and DLPack strides
- Root CI (fmt, clippy, tests, MSRV 1.89, docs)

Merge with a merge commit (not squash) so the subtree history is kept.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Do **not** enable squash auto-merge: squashing would destroy the imported history. Leave the PR for the maintainer to merge with a merge commit (`gh pr merge --merge`) unless the maintainer has said otherwise.

- [ ] **Step 4: Watch CI**

```bash
gh pr checks --repo tensor4all/tprims-rs --watch
```
Fix failures locally, reproduce, then push one batched fix commit.
