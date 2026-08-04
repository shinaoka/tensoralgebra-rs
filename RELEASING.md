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

and the Julia side, against a locally built JLL (see
`julia/TensorPrimitives/README.md` for how to make one):

```bash
julia --project=julia/TensorPrimitives -e 'using Pkg; Pkg.test()'
```

CI covers all of this plus the eleven cross-compilation targets, the three C link
modes and the vendored offline build. Do not release off a red CI.

## 1. Bump the version — in five places, which CI then checks

The `consistency` job in `release.yml` fails if any of these disagree, so this list
is enforced rather than remembered:

| file | what |
|---|---|
| `Cargo.toml` | `[workspace.package] version` — the source of truth |
| `crates/tensorprimitives-tapp/include/tapp.h` | `TAPP_VERSION_MAJOR/MINOR/PATCH` and `TAPP_VERSION_STRING` |
| `CHANGELOG.md` | a `## X.Y.Z` section, with `— unreleased` removed |
| `julia/TensorPrimitives/Project.toml` | `version` |
| `packaging/yggdrasil/build_tarballs.jl` | `version = v"X.Y.Z"` |

`tapp.h` is checked from the Rust side too, by
`the_header_version_macros_match_the_crate_version` in `abi_layout.rs`, which reads
the header as text so it holds on a machine with no C compiler.

Semver policy is D30's three tiers, stated in `crates/tensorcontract/src/lib.rs`.
The clause that does the work: for tier-2 introspection, **signatures are
semver-stable and values are not**. Without that, every tuning change would be a
breaking change.

## 2. Tag, and cut a GitHub release

```bash
git commit -am "Release vX.Y.Z"
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin main --follow-tags
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file <(...)   # from CHANGELOG.md
```

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

Fill in the source checksum, which needs the tag from step 2 to exist:

```bash
curl -sL https://github.com/lkdvos/tensorprimitives-rs/archive/refs/tags/vX.Y.Z.tar.gz \
  | sha256sum
```

Put it in `packaging/yggdrasil/build_tarballs.jl`. Then dry-run the recipe locally
before submitting — the last time this was done it found four defects that reading
the file did not, including a licence-directory name the auditor rejects:

```bash
# see julia/TensorPrimitives/README.md for the BBROOT setup
cd packaging/yggdrasil
TAPP_LOCAL_SRC=$BBROOT/src julia --project=$BBROOT/env build_tarballs.jl \
    x86_64-linux-gnu --verbose
```

Read the audit log for the two things a Yggdrasil reviewer greps for: a missing
licence file, and "could not be resolved and could not be auto-mapped". One
unresolved-library warning is **expected** — `libgcc_s.so.1`, which Julia ships —
and the recipe explains why in a comment. Then:

```bash
# fork JuliaPackaging/Yggdrasil, on a branch that is not `master`
cp packaging/yggdrasil/build_tarballs.jl \
   <fork>/T/tensorprimitives_tapp/build_tarballs.jl
# remove nothing: the TAPP_LOCAL_SRC switch is inert without the variable
```

PR title: `[tensorprimitives_tapp] Build vX.Y.Z`.

**Note the two platform exclusions and check whether they still hold.**
`i686-w64-mingw32` is excluded because BinaryBuilder's Rust toolchain does not work
there; `riscv64-linux-gnu` and `aarch64-unknown-freebsd` are excluded because no
Rust toolchain shard exists for them at any available version. Re-check the second
one against the BinaryBuilderBase in play — it is a property of the shard list, not
of this code:

```julia
for p in supported_platforms()
    try
        BinaryBuilderBase.choose_shards(p; preferred_rust_version = v"1.94.0",
                                        compilers = [:c, :rust])
    catch
        @info "no shard" triplet(p)
    end
end
```

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
in `DECISIONS.md`.

Keep the CHANGELOG's "Confidence" table honest. It is the most useful thing in
this repository for anyone deciding whether to depend on it, and it is the first
thing that rots: it states, per component, what is measured and what is merely
correct. If a release changes what is known rather than what the code does, that
table is the release.
