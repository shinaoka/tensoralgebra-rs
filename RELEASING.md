# Releasing

Four artifacts come out of one version number, in a fixed order, and three of the
four steps cannot be undone. This is the order and the reason for it.

```
        git tag  ──►  crates.io  ──►  Yggdrasil PR  ──►  General registry
       (revocable)   (permanent)      (a JLL)          (a Julia package)
```

Publishing is a **human step and is deliberately not automated**. The `release`
workflow checks everything and publishes nothing; see
`.github/workflows/release.yml`.

---

## 0. Before anything

```bash
cargo test --workspace --release
cargo clippy --workspace --all-targets
cargo fmt --all --check
TENSORCONTRACT_KERNEL=scalar cargo test --workspace --release
TENSORCONTRACT_THREADS=4 cargo test --workspace --release
```

CI covers all of this plus the eleven cross-compilation targets, the three C link
modes and the vendored offline build. Do not release off a red CI.

**Note what "CI" means here.** The workflow triggers on `push` to `main` and on
`pull_request` only, so commits on a topic branch get *no* CI at all. Running the
block above locally is not a substitute: it does not cover the cross-compilation
matrix, the C link modes, the musl `cdylib` check or the offline build. Get the
work onto `main`, or open a PR, and let CI go green *before* step 2.

**The Julia suite is deliberately not a gate here.** It used to be, and it could
not be satisfied: `julia/TensorPrimitives` depends on `tensorprimitives_tapp_jll`,
the JLL is built in step 4 from the tag created in step 2, and step 2 is what this
section is supposed to gate. Testing the Julia side therefore belongs to step 4,
where the recipe dry-run builds a local JLL from the tagged commit, and to step 5,
which cannot run before the JLL is registered. Do not reinstate it above.

## 1. Bump the version — in five places, which CI then checks

The `consistency` job in `release.yml` fails if any of these disagree, so this list
is enforced rather than remembered:

| file | what |
|---|---|
| `Cargo.toml` | `[workspace.package] version` — the source of truth |
| `crates/tensorprimitives-tapp/include/tapp.h` | `TAPP_VERSION_MAJOR/MINOR/PATCH` and `TAPP_VERSION_STRING` |
| `CHANGELOG.md` | a `## X.Y.Z` section, with `— unreleased` removed |
| `julia/TensorPrimitives/Project.toml` | `version` |

`tapp.h` is checked from the Rust side too, by
`the_header_version_macros_match_the_crate_version` in `abi_layout.rs`, which reads
the header as text so it holds on a machine with no C compiler.

Semver policy is D30's three tiers, stated in `crates/tensorcontract/src/lib.rs`.
The clause that does the work: for tier-2 introspection, **signatures are
semver-stable and values are not**. Without that, every tuning change would be a
breaking change.

## 2. Tag and cut the GitHub release — in one command

```bash
git commit -am "Release vX.Y.Z"
git push origin main            # let CI go green before the next command

# Write the notes to a file first; see "the notes are not the changelog" below.
gh release create vX.Y.Z --target "$(git rev-parse HEAD)" \
  --title "vX.Y.Z" --notes-file path/to/notes.md
```

**Do not push the tag separately.** `release.yml`'s `artifacts` job triggers on the
tag and immediately runs `gh release upload`; if the tag exists before the release
does, that job fails with `release not found` within about thirty seconds, and no
human can create the release faster. This is not a hazard to be careful about, it
is a race that is always lost — it happened on v0.1.0. `gh release create --target`
creates the tag and the release together, so the upload has somewhere to go.

Two consequences of doing it this way, both accepted deliberately:

* **The tag is lightweight, not annotated.** No tagger name, date or message, and
  `git describe` needs `--tags` to see it. That is the price of atomicity. The
  Yggdrasil checksum in step 4 is unaffected: it hashes the source tarball, not
  the tag object.
* **`git push --tags` and `--follow-tags` remain forbidden**, now for their own
  reason rather than as part of this step. `backup/pre-split-2026-08-03` is a local
  safety net pointing at c1229e6, which is NOT an ancestor of main — pushing it
  would publish two orphaned commits, and deleting it locally would strand them.

**The notes are not the changelog.** Extracting the version's CHANGELOG section
verbatim gives a 300-line release body written for a maintainer, and on v0.1.0 it
also advertised the Julia package, which cannot be installed until steps 4 and 5
have run. Write the body for someone deciding whether to install *today*: what it
is, how to depend on it, what is measured and what is not, what is not yet
available, and links into the repository pinned at the tag rather than relative
paths, which do not resolve in a release body.

This is the one revocable step, and it is what the Yggdrasil recipe pins by
checksum, so do it before step 4.

## 3. crates.io — **irreversible**

A published version can be yanked but never removed, and the name is claimed
permanently. Dry-run first; CI already does the engine's.

**The order is fixed by cargo, not by preference.** `tensorprimitives-tapp`
depends on `tensorcontract` by path *and* version, and packaging rewrites that into
a registry dependency which then has to resolve — so `cargo package -p
tensorprimitives-tapp` fails at "failed to prepare local package for uploading"
until the engine is in the index. `cargo package --workspace` *appears* to work and
is not a reliable gate: it was observed verifying the dependent crate against a
stale extraction of the engine.

```bash
cargo publish --locked -p tensorcontract
# wait for the index to update -- a minute or so
cargo publish --locked -p tensorprimitives-tapp
```

`tensorprimitives-bench` is `publish = false` and stays that way: it needs
`TBLIS_ROOT` at build time.

## 4. The JLL, via Yggdrasil

**The recipe is not in this repository.** It lives where Yggdrasil requires it,
at `T/tensorprimitives_tapp/build_tarballs.jl` in
[JuliaPackaging/Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil) — that is
the only copy that builds anything, and a second one here could only drift from it.
A working copy was kept in-tree until 0.1.0; the last revision containing it is
`git show 79d4b9e:packaging/yggdrasil/build_tarballs.jl`, which is the place to
start from if you need to recreate it rather than edit the Yggdrasil one.

Update `version = v"X.Y.Z"` and the `GitSource` commit — **the commit the tag
points at, not the tag**, because a tag can be moved under a pin and this one was,
twice. The source is a `GitSource` and not an archive over
`.../archive/refs/tags/vX.Y.Z.tar.gz` **for a reason no checksum can address**:
BinaryBuilder rejects auto-generated GitHub archives outright, since GitHub does
not guarantee they are byte-stable. Do not reintroduce the URL and do not try to
fix the resulting error by recomputing the digest.

```bash
git rev-parse vX.Y.Z^{commit}
```

Then dry-run before submitting — the last time this was done it found four defects
that reading the file did not, including a licence-directory name the auditor
rejects:

```bash
# see julia/TensorPrimitives/README.md for the BBROOT setup
cd <fork>/T/tensorprimitives_tapp
julia --project=$BBROOT/env build_tarballs.jl x86_64-linux-gnu --verbose
```

This builds the recipe exactly as submitted, from the pinned commit, so the tag
from step 2 has to exist first. Add `--deploy=local` to get a develop-able JLL in
`$BBROOT/depot/dev` and test `julia/TensorPrimitives` against it — that is where
the Julia suite is gated, per step 0.

**If you regenerate the recipe with `BinaryBuilder.run_wizard()`** — which is worth
doing, its output is the house style a reviewer expects — keep
`CARGO_TARGET_DIR=${WORKSPACE}/target` in the script. The wizard's interactive build
mounts an overlay over `srcdir`, and on this kernel a copy-up inside it fails with
`Cross-device link (os error 18)` the moment cargo creates `target/`. `build_tarballs.jl`
creates no such overlay, so a dry-run will never show you this.
`docs/notebook/packaging.md` has the four things the wizard's printer cannot emit
and must be added back by hand.

Read the audit log for the two things a Yggdrasil reviewer greps for: a missing
licence file, and "could not be resolved and could not be auto-mapped". Exactly two
unresolved-library warnings are **expected** — `libgcc_s.so.1`, which Julia ships,
and `bcryptprimitives.dll` on Windows — and the recipe explains both in a comment.
A third would be a real finding. Then open the PR from a branch that is not
`master`.

PR title: `[tensorprimitives_tapp] Build vX.Y.Z`.

**One platform is excluded: check whether it still holds.** `i686-w64-mingw32`,
because BinaryBuilder's Rust toolchain does not work there — a claim that still
stands verbatim in its own `docs/src/build_tips.md`. `riscv64-linux-gnu` and
`aarch64-unknown-freebsd` *used* to be excluded too, and stopped needing to be on
2026-07-31.

**Run that check in an environment resolved from Yggdrasil's own manifest, not from
a `Pkg.add("BinaryBuilder")`.** This is the trap that kept the stale exclusion
alive: Yggdrasil's `Project.toml` pins BinaryBuilder *and* BinaryBuilderBase to
`master` via `[sources]`, and the released BinaryBuilderBase lags it by whole Rust
toolchains — 1.44.0 stops at Rust 1.94.0, where those two platforms have no shard
and `choose_shards` errors, while the 1.47.0 that CI resolves has 1.97.0, where
they build. A local dry-run against the released stack cannot see a platform that
CI can build, and says so in a way that looks permanent.

```bash
D=$BBROOT/env-ygg && mkdir -p $D
for f in Project.toml Manifest.toml; do
  gh api repos/JuliaPackaging/Yggdrasil/contents/$f?ref=master --jq .content \
    | base64 -d > $D/$f
done
julia --project=$D -e 'using Pkg; Pkg.instantiate()'
```

```julia
# in that environment
using BinaryBuilder, BinaryBuilderBase
for p in supported_platforms()
    try
        BinaryBuilderBase.choose_shards(p; compilers = [:c, :rust])
    catch
        @info "no shard" triplet(p)
    end
end
```

Omit `preferred_rust_version` there, deliberately: it defaults to the newest shard,
and the two platforms enabled in 2026 exist **only** at 1.97.0 — pinning 1.94.0
puts them back out of reach. Do not reintroduce a Rust pin below that.

The Apple targets need you to accept the Xcode SDK licence
(`BINARYBUILDER_AUTOMATIC_APPLE=true`). That is a legal agreement; read it first.

## 5. The Julia package, via the General registry

Only after the JLL is registered — `julia/TensorPrimitives/Project.toml` depends on
it, so the package cannot resolve before then. Then register with `@JuliaRegistrator`
on a commit, noting the `subdir`:

```
@JuliaRegistrator register subdir=julia/TensorPrimitives
```

## After

Open a `## X.Y.Z+1 — unreleased` section in `CHANGELOG.md` and record the release
in `docs/notebook/`.

Keep the CHANGELOG's "Confidence" table honest. It is the most useful thing in
this repository for anyone deciding whether to depend on it, and it is the first
thing that rots: it states, per component, what is measured and what is merely
correct. If a release changes what is known rather than what the code does, that
table is the release.
