## Packaging and distribution

Phase 5: the quality gates that had never run, the API tiers, the TAPP conformance suite, the shipped C header and its consumer, cross-compilation, the JLL and the Julia package.

### Phase 5 part 1: what preparing v0.1 found

Packaging was expected to be tidying. It was mostly **discovering that the
project's own quality gates had never run**, which is a more useful result and
worth recording in full so the lesson survives.

#### CI was not gating anything it claimed to gate

Three of five jobs could not have been passing, each for an independent reason,
and none had ever been noticed because nobody had run the commands locally with
the flags the workflow uses:

| job | why it could not pass |
|---|---|
| `msrv` | pinned toolchain **1.75** against a declared `rust-version` of 1.89, so cargo refuses before compiling anything |
| `lint` | `cargo fmt --all -- --check` against a tree with drift in five files, mostly macro-adjacent code in `kernel/x86.rs` and `plan.rs` |
| `docs` | `cargo doc` under `-D warnings` against **17 rustdoc errors**, nine of them public documentation linking into private modules |

The general lesson, which is the same one A15 and A20 taught in the measurement
domain: **a gate nobody has watched fail is not a gate.** Every command in the
workflow was re-verified locally before being trusted, and the new job set is
smaller and states what each job proves.

Two of the new jobs cover code paths that had **never been exercised in CI at
all**: the threaded driver (threading is off by default, so no test ran it) and
the analytical blocking model. The x86 runners have AVX2 but not AVX-512, so
CI's default path is now the AVX2 kernels — the ones whose register blocks are
provisional (D26) — which makes the untuned path the *automatically* tested one.

#### Defects that would have shipped

* **`examples/kernel_shapes` did not compile off x86**, and `cargo test` builds
  examples, so the suite failed for every aarch64 user. Now a cfg-gated module.
* **The declared MSRV was wrong** in the other direction too — see D31.
* **The `trace` feature was declared, described, and implemented nowhere.**
  Removed rather than advertised.
* **The TAPP crate cannot be packaged before the engine is published.** It
  depends on the engine by path *and* version, and packaging rewrites that
  into a registry dependency which must then resolve — so
  `cargo package -p tensorprimitives-tapp` fails at "failed to prepare local
  package for uploading" until `tensorcontract 0.1.0` is in the index.
  `cargo package --workspace` appears to work, and does set up a temporary
  registry to satisfy the dependency, but it is **not a reliable gate**: it was
  observed verifying the dependent crate against a *stale* extraction of the
  engine as soon as the dependent used an engine API added since the previous
  packaging run — which is precisely the case such a job exists to catch. It
  passed earlier in this session and then failed on exactly that change, which is
  how the behaviour was found. CI therefore gates `cargo package -p
  tensorcontract` only, and the ordering stands as the documented publish
  procedure: `tensorcontract`, wait for the index, then `tensorprimitives-tapp`.
* **The `std` feature promised something it does not deliver.** Disabling it
  compiles, but the crate has no `#![no_std]` and uses `Vec`, so it is not a
  no-std build. The feature is now documented as the seam a future port would
  widen rather than as a claim.
* **The README described the Phase 2 engine** — scalar kernels, "performance not
  yet meaningful" — two phases and two instruction sets out of date.

#### What the API decision cost and bought

D30's three tiers. The part worth carrying forward is that this project *needs*
a tier whose values are explicitly unstable: it ships measured heuristics, and it
re-measures them every phase. Publishing without saying so would have forced a
choice between freezing the tuning and breaking semver at each phase boundary.

Two side effects of the documentation pass are worth keeping. `kernel::scalar`
had been *documented* as the route by which a foreign scalar type gets a correct
engine for free and never demonstrated; it now carries a worked example that
compiles and runs as a doctest for 0.11 s. And both blanket
`allow(clippy::missing_safety_doc)` attributes are gone — the x86 one had been
hiding that the four generated kernels have *different* panel and tile bounds
(1x, 2x, 3x the real kernel's, by packing format) and that the plain-`fn`
trampolines drop the `#[target_feature]` attribute but not the obligation.

#### The TAPP conformance suite, and the four gaps it found

`crates/tensorprimitives-tapp` had **one test** — a happy-path `c64` contraction —
behind a coverage table claiming four datatypes, TAPP cases 1–4, conjugation on
any operand, two documented rejections and mixed precision. For the crate whose
entire purpose is that a C caller can swap this engine for TBLIS behind one
header, that was the thinnest-tested part of the workspace, and every claim in
the table was unverified. It is now **84 tests**, all through the `extern "C"`
entry points rather than the Rust `Plan` behind them, because the bugs this layer
can have are exactly the ones invisible from there.

Two design points worth copying. Numerics go against the brute-force oracle
*plus* two hand-computed anchors, so a shared engine/oracle bug cannot pass. And
the suite was **falsified before it was trusted**: perturbing one hand-computed
constant failed exactly the hand-checked test, while perturbing the oracle call's
`alpha` failed 33 of 34, leaving only the anchor — which is the correct pattern
and confirms the two mechanisms are independent. A conformance suite nobody has
watched fail is not evidence, which is A15's lesson in a third domain.
`abi_layout.rs` additionally re-declares the whole upstream header in an
`extern "C"` block and drives a contraction through it, so a renamed `#[no_mangle]`
is a link error in our own tests rather than a downstream C build's discovery.

**Four gaps, all fixed rather than filed** (none was a wrong number):

1. **An extent product could abort the caller's process.** Nothing between
   `TAPP_create_tensor_info` and `build_scatter` checked that a tensor's extents
   multiply to something representable. The product wrapped: in release the plan
   built, reported success and computed *nothing*; in debug the multiply panicked
   inside an `extern "C"` function, which the compiler turns into a process abort.
   `reduce_tensor` now folds with `checked_mul` and reports
   `Error::ExtentProductOverflow` → `TAPP_ERROR_SHAPE`. Checking per tensor bounds
   every scatter vector, since each is as long as the product of some *subset* of
   one tensor's axes.
2. **A null `C` silently discarded `D`.** Upstream defines `TAPP_IN_PLACE` as
   `NULL` and leaves its meaning an open `//TODO`; this crate read it as
   `beta = 0`, so `beta = 1, C = TAPP_IN_PLACE` — precisely what a caller writes
   for `D += alpha*A*B` — overwrote `D` and reported success. The ambiguous
   combination is now refused; in-place accumulation is expressible by passing
   `D`'s own pointer as `C`.
3. **Every library handle was the value `1`.** `HandleState` was zero-sized, so
   `Box::into_raw` returned `NonNull::dangling()`: two live handles were
   indistinguishable and a C program creating two and destroying both was
   double-freeing — harmlessly, for exactly as long as the state stayed empty.
   It has a reserved field now, and the comment claiming this made handle
   validity *checkable* is gone: it never did, and nothing can.
4. **Three declared symbols did not exist.** `TAPP_attr_set`/`_get`/`_clear` were
   absent, so a C program including `<tapp.h>` and calling one failed to **link**
   — the least diagnosable failure available. Now exported as refusals, which is
   conformant (upstream specifies no keys) and diagnosable. Prototypes fetched
   from the upstream header, not reconstructed.

Also corrected: `execute` now writes `0` through a non-null `status`, so the
idiomatic create/execute/destroy sequence stops handing an uninitialised value to
the destructor; and the coverage table now admits that mixed *storage* types are
rejected (§1.6 of [`../design.md`](../design.md) lists them as in TAPP's scope, and they are — just
not here), and that `TAPP_ERROR_*` beyond zero are this crate's own numbering,
since upstream `error.h` fixes only `TAPP_SUCCESS`.

| # | Assumption | Status |
|---|---|---|
| A26 | The TAPP layer is thin enough that the engine's own correctness tests cover it. | **Refuted.** Everything the layer can get wrong — datatype-tag dispatch, `intptr_t` handle casts, label arrays read at a rank the info supplies, `beta` on a null `C`, the `status` and `prec` arguments — is invisible from the Rust API and had no test. Four gaps on first contact, one of which aborted the caller's process. **Test an FFI layer as its caller, not as its callee.** |

#### Four gaps between "the ABI is correct" and "a distribution can ship it"

1. **Nothing reported a version.** `TAPP_implementation_name()` returns a
   free-form string with no number in it, and the crate version was visible only
   to cargo. A distribution that ships `include/` and `lib/` as separate packages —
   which is what a JLL, a system package, or a stale `-I` all produce — could have
   them from different versions with nothing anywhere to notice. There are now
   `TAPP_VERSION_*` macros describing the header and
   `TAPP_implementation_version()` describing the linked library, and **both halves
   are tested**: `abi_layout.rs` reads the header as *text* and compares to
   `CARGO_PKG_VERSION`, so the check holds on a machine with no C toolchain, and
   `examples/c-consumer` compares the macro its own compiler saw against the string
   its own link returned.
2. **The `cdylib` had no SONAME, and on macOS something worse than none.** rustc
   emits `-soname` only for the `dylib` crate type, never for `cdylib`, so every
   consumer recorded a bare filename. On Mach-O, ld64 defaults `LC_ID_DYLIB` to the
   `-o` path — an absolute build-tree path — so a plain `cargo build` produces a
   library that is not relocatable at all. Both are set at link time via `RUSTFLAGS`
   (D40). **The claim that stood here — that a macOS build "would have shipped
   unrelocatable with a clean audit" — was wrong**, and is corrected in
   "The audit was already doing two of these" below: inside BinaryBuilder the audit
   sets both, and the flags were only ever load-bearing everywhere else.
3. **Eight of the 23 prototypes in `tapp.h` had never been seen by a C compiler
   or a linker.** `main.c` called the ones that do work; the setters,
   `TAPP_get_strides`, the batched product, `TAPP_destroy_status` and two attribute
   functions were checked only by `abi_layout.rs`'s Rust transcription, which is a
   second hand-maintained copy rather than an oracle. The header's `TAPP_ERROR_NULL`
   prose bug — fixed in this pass — was exactly that failure class one level
   further out: prose, which nothing tests at all.
4. **There were no install rules.** The library was consumable only by knowing
   cargo's directory layout. `install.sh` plus a pkg-config template fixes that, and
   the example gained a third link mode (`-DTAPP_PREFIX=`) that consumes an
   installed prefix — the only mode in which the header comes from outside the
   source tree, and therefore the only one that would catch an install that forgot
   to copy it. The recipe calls the same script, so there is one definition of the
   layout rather than two that drift.

#### Cross-compilation: one prediction inverted, one silent failure (D39)

The repository had never been cross-compiled. Eleven targets now are, in CI, and
the two findings are both worth more than the eleven passes.

The prediction was that **`i686` would fail**, because `kernel::x86` is gated on
`any(target_arch = "x86", target_arch = "x86_64")` — so the AVX-512 intrinsics are
instantiated on 32-bit x86 too — and `core::arch::x86` genuinely lacks the
intrinsics taking 64-bit integer operands. It compiles: the kernels are f32/f64
FMA-shaped and use none of them. The cfg was left alone rather than narrowed to
`x86_64` on suspicion. Note what this does *not* say: in 32-bit mode only
`zmm0`–`zmm7` are encodable, so the register blocks — chosen against a 32-register
file — will spill on `i686`. That is a performance property, it is unmeasured, and
it is recorded here rather than acted on.

**musl fails silently.** With the target default (`crt-static` on) cargo prints
`dropping unsupported crate type cdylib` and **exits 0**. A musl JLL would have
been a tarball containing a header, a pkg-config file, two licences and no library
— green, and inexplicable downstream. `-C target-feature=-crt-static` fixes it, and
because the symptom is a warning rather than an error the `cross-musl` job asserts
the warning *still appears* without the flag, so the day it stops being needed is
visible instead of assumed.

#### What running the recipe found that reading it did not

The BinaryBuilder recipe was dry-run locally, on this workstation, before being
considered done. Four corrections, none of which a careful reading produced:

* **The available Rust shards stop at 1.94.0, not 1.97.0.** Master's
  `Artifacts.toml` advertises 1.97.0; the BinaryBuilderBase that the *released*
  BinaryBuilder resolves to offers 1.57.0 through 1.94.0. `choose_shards` **errors**
  on a version with no shard, so an optimistic pin is a hard build failure rather
  than a graceful fallback. This is [[prefers-verified-releases]] again, in a new
  place: check what is released, not what the development head says.
* **`riscv64-linux-gnu` and `aarch64-unknown-freebsd` have no Rust toolchain at
  any available version.** Enumerated rather than guessed, by calling
  `choose_shards` on all 18 supported platforms. Both would take the scalar path
  anyway. Filtered, with the query that establishes it recorded in `RELEASING.md`
  so it can be re-run rather than re-derived.
* **The licence directory name.** The auditor looks under
  `share/licenses/tensorprimitives_tapp` — the *package* name, underscore — while
  `install.sh` uses the crate name with a hyphen. The recipe's first draft argued
  itself out of calling `install_license` on the grounds that install.sh already
  did the job, and the audit answered "Unable to find valid license file", which is
  one of the two things a Yggdrasil reviewer greps the log for. install.sh gained
  `--no-licenses`.
* **`CompilerSupportLibraries_jll` makes the `libgcc_s.so.1` warning worse.** It
  adds a missing-artifact-mapping warning of its own — CSL's artifacts are keyed by
  libgfortran version — and does not silence the original. Julia ships libgcc_s in
  its own libdir, so the library loads and runs; that was settled by the Julia test
  suite loading it, not by argument. Reverted to an empty dependency list with the
  reasoning attached in the recipe, since a reviewer will ask.

**Thirteen of the fifteen platforms were then built and audited locally**, and all
thirteen produced a tarball with a real shared library in it and an identical
layout: `lib/` (or `bin/` on Windows), `include/tapp.h`, `lib/pkgconfig/`,
`share/licenses/`. `x86_64-w64-mingw32` behaves as predicted — the artifact is
`tensorprimitives_tapp.dll` with no `lib` prefix, it lands in `bin/` because that is
what `${libdir}` is there, the import library goes to `lib/`, and the two-name
`LibraryProduct` finds it without renaming anything. `i686` linked, which `cargo
check` could not establish. The two not built are the Apple targets, and only
because accepting the Xcode SDK licence is not a decision to take on someone's
behalf.

Exactly **three** distinct audit warnings across all thirteen, all understood:
the `cpuid` one below, `libgcc_s.so.1` on ELF, and `bcryptprimitives.dll` on
Windows — the last a Windows 10+ system DLL that Rust's standard library imports
for randomness and that the auditor's system-library list does not know. A fourth
would be a real finding, and the recipe says so.

And one **prediction confirmed verbatim**, which is why D40 exists. The audit log
reads: *"contains a `cpuid` instruction; refusing to analyze for minimum instruction
set, as it may dynamically select the proper instruction set internally. Would have
chosen avx512, instead choosing x86_64."* `check_isa` fails a build whose minimum
instruction set exceeds the platform's, and this is a generic x86-64 binary full of
AVX-512 kernel bodies. It ships only because the auditor abandons the analysis on
finding a `cpuid`, which `kernel::cache` and `std_detect` happen to supply. Nobody
promised that, so CI asserts it.

#### The Julia side, and why the backend shape is the right one

`julia/TensorPrimitives` is two layers: `LibTAPP`, a complete `ccall` wrapper with
handles as distinct Julia types and finalizers; and `TAPPBackend`, a
`TensorOperations.jl` backend. (It has since moved to its own repository and the
backend to an extension in `TensorOperations.jl` itself — see "The wrapper leaves
the tree" below. Everything in this section about the *mapping* still holds; one
sentence about `tensoradd!` did not.) The second is the one that matters, for a reason
that is about this engine specifically rather than about convenience.

`TensorOperations.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ...)`
hands over exactly what TAPP takes: arbitrary extents, element strides, and
per-operand conjugation. So `pA`/`pB`/`pAB` become a **relabelling** —
`_labels` is twenty lines of index bookkeeping — and nothing permutes, copies, or
allocates a temporary. A backend over a GEMM would have to. This makes the
transpose-free claim something a Julia user can observe rather than read, and it
puts this engine on the same footing as `TensorOperationsTBLIS.jl` for comparison,
which is the shape the eventual three-way benchmark wants.

The backend is opt-in and deliberately **not** registered with `select_backend`:
loading the package changes nothing for code that does not ask. Only
`tensorcontract!` is implemented. TAPP can express `tensoradd!` and `tensortrace!`
too — a trace is a repeated label, an add is a contraction against a rank-0 operand —
but each is a correctness surface, and acquiring one for free is how a wrong answer
gets shipped.

> **Correction, 2026-08-18.** This paragraph used to end "`tensoradd!` and
> `tensortrace!` fall through to TensorOperations' own backends." **They do not.**
> They throw. See "The wrapper leaves the tree" below, and A64.

64 tests, every one checked against TensorOperations' own backend on the same
inputs rather than against a rewritten expectation. The two that were worth
writing: non-contiguous strided views, where a strides-in-elements or base-pointer
error would show and nothing else would; and `β == 0` against an all-`NaN` output,
because "overwrite" and "multiply by zero" differ exactly there, and TAPP spells the
former as a null `C` operand.

#### Two things a reader should not take from this section

* **No performance was measured *here*.** Nothing in this section is a throughput
  claim and no code in it can change one: the engine was not touched. The two Phase
  4 measurements this branch was written alongside have since landed on `main`
  (parts 10 and 11), and the CHANGELOG's confidence table has been updated from *those*
  — five rows, none of them because of anything in this section. If a reader arrives
  at a new Julia backend and infers that something got faster, the answer is that a
  binary distribution was built, not a kernel.
* **Threading is still not reachable from Julia**, because
  `TAPP_execute_product` ignores its executor argument and `TENSORCONTRACT_THREADS`
  is read once per process. The Julia wrapper documents that rather than papering
  over it, and deliberately wires nothing to `TAPP_create_executor` — plumbing a
  knob through an inert object would be worse than the absence.

### Interlude: making the C surface consumable

Prompted by a concrete external ask — a colleague evaluating this for the
**NDA** C++ array library (TRIQS, Flatiron) — the question was whether a C++
project can use this without shipping a Rust compiler. Answering it honestly
turned up four gaps between "the ABI is correct" and "a C++ project can link
it", all of which were on the Phase 5 packaging list and none of which touched
the engine. See D36–D38.

#### The gap that mattered

The TAPP layer had 23 verified `extern "C"` symbols, a conformance suite driving
those symbols, and `abi_layout.rs` re-declaring the whole upstream header so a
dropped export is a link error. What it did not have was **a header**, and
nothing anywhere invoked a C compiler. The correctness of the ABI was thoroughly
established *from Rust*; whether a C caller could actually use it was inference.

That inference held — the C consumer passed on its first real run, including the
complex path and `TAPP_CONJUGATE` — but it held by luck as much as design, and
it would not have survived the first wrong prototype.

#### What the toolchain objection was actually worth

Little, once examined, and the examination is the useful part. NDA already
requires CMake ≥ 3.22, a concepts-capable C++ compiler, and HDF5 + MPI + OpenMP
on by default. Against that, `rustc` is marginal — and **a C++20 compiler is the
harder constraint on cluster environments**: the stock compiler on this very
workstation is gcc 8.5, which cannot compile nda at all, while `rustup` installs
a pinned toolchain into `~/.cargo` with no root and no system integration. The
MSRV, 1.89, was released 2025-08-04, one year ago to the day.

So the real blockers are maturity and measurement coverage, not the build.
Recorded so the toolchain argument is not re-litigated.

#### A claim corrected in the making

The vendoring story was initially stated as "three crates", from the runtime
dependency tree (`num-complex` → `num-traits`, plus `autocfg` at build time).
Running it showed **~17**: `cargo vendor` is workspace-wide and includes
dev-dependencies, so `rand` and its tree (`getrandom`, `libc`, `zerocopy`,
`syn`, …) come along. Both numbers are true of different things — a library-only
vendor from the published crate is three; vendoring this repo so the *tests* run
offline is seventeen — and the CI job now states both and exercises the second,
since a test step is the only thing that keeps the dev-dependency half honest.

| # | Assumption | Status |
|---|---|---|
| A47 | The shipped header agrees with the library it describes. | **Now tested rather than assumed.** `examples/c-consumer` compiles the header with a C compiler, links the built library and checks numerical results in `f64` and `c64`; CI runs it in both the corrosion and prebuilt modes. Previously no C compiler saw the header at any point. |
| A48 | A Rust panic reaching the C boundary is acceptable because it is memory-safe. | **Rejected as a policy.** Memory-safe but process-fatal, and the engine panics on allocation conditions a caller can hit. D38 converts it to an error code on the three entry points that can raise it. |

### Releasing v0.1.0, and what running the procedure found

Published 2026-08-07: `tensorcontract` 0.1.0 and `tensorprimitives-tapp` 0.1.0 on
crates.io, tag `v0.1.0` at `ca916fa`, GitHub release with the
`x86_64-linux-gnu` tarball attached. Verified after the fact rather than assumed:
a fresh crate depending on `tensorcontract = "0.1"` resolved from the registry
(checksum `98ffea44…`) computes a batched contraction correctly, and docs.rs
rendered all eight modules with the crate reporting 100% documented.

#### The release procedure had two defects, in series, and the first hid the second

Neither was findable by reading. `RELEASING.md` step 2 said to push the tag and
then create the release; `release.yml`'s `artifacts` job triggers on the tag and
runs `gh release upload` about thirty seconds later. That is not a hazard to be
careful about, it is a race no human can win, and it failed as
`release not found`.

Fixing it exposed the second: `HTTP 403: Resource not accessible by integration`.
The repository's default workflow token is `contents: read` and `release.yml`
declared no `permissions:` block, so the job could locate the release and then not
write to it. **The permission had been missing since the workflow was written and
had produced no symptom**, because the upload never got far enough to be refused.

Both are fixed and both are now *demonstrated* — the "Attach to the release" step
has run to completion exactly once, on the third tag push. The lesson is narrow
and worth keeping: a release procedure that has never been executed end to end is
untested code, and its failures stack rather than queue.

The tag moved twice during this. That was free only because nothing consumed it
yet — no crates.io publish, no Yggdrasil submission pinning the source-tarball
checksum, no asset downloads. That window closes at the first `cargo publish`,
which is the argument for doing the cheap fixes before the irreversible step
rather than after.

#### A gate that could never have been satisfied

Step 0 required `Pkg.test()` on `julia/TensorPrimitives` against a locally built
JLL. The package depends on `tensorprimitives_tapp_jll`, the JLL is built in step
4 from the tag created in step 2, and step 0 exists to gate step 2. The check now
lives in step 4, where it already happens via `TAPP_LOCAL_SRC`.

#### The same overclaim three times in one session

The pattern is worth naming because it recurred in three unrelated places and was
caught three different ways: **advertising a capability that the configuration
being advertised does not actually have.**

* `error.rs` gated `impl std::error::Error` on `feature = "std"`. The crate is not
  `no_std` and never was, so the gate bought nothing and cost the error type its
  interoperability in exactly the configuration the feature exists for. Found by
  the API-guidelines pass; the counterfactual was verified by reinstating the gate
  and watching `Box<dyn Error + Send + Sync>` refuse an `Error`.
* The CHANGELOG's `no-std`-adjacent bullet advertised that the crate compiles
  without the feature, omitting that the error type was degraded when it did.
* The v0.1.0 release notes, drafted from the CHANGELOG, described the Julia
  package as present when it cannot be installed until steps 4 and 5 have run.
  Caught by the maintainer asking whether a `tensorcontract` release should be
  mentioning Julia at all.

Two of the three were written *while removing* the first, which is the part to
remember: the fix and the overclaim came from the same hand in the same hour.

#### What the API-guidelines pass changed, and what it did not

All 54 items of the Rust API Guidelines checklist, before first publication
because every breaking change was free then: 12 fixed, 21 already passing, 21 not
applicable. The deliberate deviations are D59–D63 rather than left to look like
oversights — unsealed traits, no serde, bare `i64` labels, selective
`#[non_exhaustive]`, terse BLIS-domain names.

One real defect, D58: `Layout`'s two public `Vec` fields let a caller build a rank
mismatch that `Layout::new` existed to reject, reaching an index panic in safe
code at `plan.rs:1165`. Confirmed by running it first. One buffer of length
`2 * ndim` makes it unrepresentable rather than detected.

**This release changed what the code does, not what is known.** The Confidence
table is unchanged and was checked rather than assumed: every row concerns
measurement, and no measurement was taken here.

#### The JLL source pin: an archive checksum, then the source it could not be

The recipe ships a `GitSource` pinned to commit `ca916fa`, which is what tag
`v0.1.0` points at. It reached that by way of a dead end worth recording, because
the dead end looked exactly like success.

Step 4 says to pin `.../archive/refs/tags/v$(version).tar.gz` by sha256, so that
was done: `7fb56aebc6baca50b7aacdef6eb12abbba47c080e5f3583510d334dace2f7a84`,
computed over the right artifact after a near miss (`gh api /repos/.../tarball/<tag>`
is a *different* archive, prefixed `lkdvos-tensorprimitives-rs-<sha>/` rather than
`tensorprimitives-rs-0.1.0/`, so its digest differs and the recipe's `cd` glob
would not have matched it either). All of which was wasted: **BinaryBuilder rejects
that URL outright.**

```
ArgumentError: The archive automatically generated by GitHub ... may not have a
stable checksum in the future, thus cannot be used as a reliable source
```

GitHub does not guarantee its auto-generated archives are byte-stable
(github.blog, 2023-02-21), and BinaryBuilder has turned that into a hard error
rather than a warning. **The fix for a rejected source is a different source, not
a recomputed digest** — the objection is to the URL, and no amount of care over
the checksum addresses it. The recipe now carries that sentence, because the
natural reaction to a checksum error is to recompute the checksum.

A commit rather than the tag, for a reason this release supplied: `v0.1.0` was
moved twice while the release plumbing was being fixed, so a pin a tag can
relocate under is not a pin.

Changing the source also moved the checkout directory, which would have failed
next: `GitSource` checks out into `srcdir/tensorprimitives-rs` with **no** version
suffix, where the script's `cd .../tensorprimitives-rs-*/` required one. The glob
is now `tensorprimitives-rs*/`, matching both that and the
`tensorprimitives-rs-0.1.0` prefix the `TAPP_LOCAL_SRC` dry-run path produces.
Predicted by reading, confirmed by the dry-run.

Submitted as
[JuliaPackaging/Yggdrasil#14374](https://github.com/JuliaPackaging/Yggdrasil/pull/14374),
rebased onto a current upstream master. Worth knowing for next time: `gh repo sync`
on the fork silently did nothing — the fork sat 1,995 commits behind through a
rebase that was therefore a no-op — and the base only became current after
rebasing onto the `upstream` remote directly. Check the merge-base, not the command's
exit status.

#### Two things checked rather than carried over

The **platform exclusions were re-verified**, which `RELEASING.md` step 4 asks for
because they are a property of the shard list and not of this code. Against
BinaryBuilderBase 1.44.0, `choose_shards` at `preferred_rust_version = v"1.94.0"`
fails for exactly `riscv64-linux-gnu` and `aarch64-unknown-freebsd` out of 18
supported platforms — the same two the recipe filters, unchanged from when the
comment was written against BinaryBuilder 0.6.6's older base. Caveat: that resolved
BinaryBuilderBase standalone, not the version BinaryBuilder itself pins. The risk
direction is benign either way — a filtered platform that gained a shard is a
missed platform, whereas an *unfiltered* platform without one is a hard
`choose_shards` error, and there are none of those.

**The recipe's script logic was pre-flighted on the host**, which needs no
BinaryBuilder: from an extraction of the tagged tree the `--locked` build succeeds
(so `Cargo.lock` is committed and current, and `tensorprimitives-bench` never
builds), `install.sh` writes `lib/`, `include/` and `lib/pkgconfig/`,
`--no-licenses` correctly leaves `share/licenses` to `install_license`, and the
SONAME is set from `RUSTFLAGS`.

#### A concern that dissolved on measurement

The source tarball is 92% benchmark data: 42.5 MB of 46.4 MB uncompressed is
`bench-results/` across 2,558 files, against 816 KB of crates. That was raised as
an argument for pinning a slim purpose-built source asset. It does not survive the
numbers — **the whole repository is 8.6 MB of git**, because committed CSVs
compress well and the history is short, so a `GitSource` clone costs about what the
6.6 MB archive did. Nothing to fix, and the tidier-looking option would have added
a hand-built artifact to every release for no measured gain.

#### Review feedback: the recipe was mostly comments

The first Yggdrasil review of #14374 asked for one thing, and it was not technical:

> Please don't leave a whole bunch of LLM-generated comments that describe the
> code. Only comment the important things (e.g., why a platform is disabled, what
> patches are for).

The submitted file was **230 lines, 158 of them comment-only** — more prose than
recipe, and most of it re-explaining what the next line said. It is now 91 lines
with 27 comment lines, and the surviving comments are exactly the ones a reader
cannot recover from the code: the two platform filters, the `crt-static` and
SONAME / `LC_ID_DYLIB` link flags, the two-name `LibraryProduct`, the licence
directory that `install.sh` and the auditor disagree about, the two expected
unresolved-library warnings, and why the Rust pin cannot be raised optimistically.
Everything cut is either in this chapter or in `RELEASING.md` step 4, which is
where it belonged in the first place.

The `TAPP_LOCAL_SRC` switch went with it, deliberately: an `ENV`-conditional
`sources` is a second non-idiomatic thing to defend in review, and it earns nothing
now — its purpose was dry-running before a tag existed, and the release procedure
tags in step 2, before the recipe work in step 4. A dry-run from the pinned
`GitSource` also tests the submitted form exactly, which the `DirectorySource` path
never did.

Two live-document defects fell out of the same pass, both in `RELEASING.md` step 4,
and both would have misled the *next* release rather than a reader: it still told
you to compute a sha256 over `.../archive/refs/tags/vX.Y.Z.tar.gz`, which
BinaryBuilder rejects outright, and it still described the Julia-suite gate as
running from an untagged export via a variable that no longer exists.

#### Regenerating the recipe with the wizard, and a machine-specific trap

The trimmed recipe was thrown away and regenerated with `BinaryBuilder.run_wizard()`
rather than hand-edited, on the grounds that the review comment was about house
style and the wizard *is* the house style: its output carries six template
comments and nothing else. 52 lines, 17 comment-only, 11 of those the template's.

**The wizard cannot express four things this recipe needs**, all verified against
BinaryBuilder 0.6.6 rather than guessed:

* `preferred_rust_version`. `print_build_tarballs`' `kwargs_vec` handles
  `julia_compat`, `compilers`, `preferred_gcc_version` and `preferred_llvm_version`
  and nothing else, and step 2 prompts only for GCC and LLVM. Dropped rather than
  restored: `choose_shards` defaults it to `maximum(getversion.(Rust_builds))`,
  so the old pin was a no-op at 1.94.0 anyway.
* A `LibraryProduct` with two names — the printer emits one, `normalize_name` of
  the file it found.
* `FileProduct("include/tapp.h", …)`. Step 4 offers only what `readmeta` can open,
  so a header is never a candidate.
* `supported_platforms()` **minus** a filter. The printer prints either
  `supported_platforms()`, when the selection equals all of it, or a literal list.
  The filter form is worth restoring on its own merits: a literal list has nowhere
  to say *why* a platform is excluded, which is exactly what the reviewer asked to
  keep, and it silently stops tracking a 19th platform if one is ever added.

**The wizard does not run on this machine without one change to the recipe.** Its
interactive build mounts an overlay over `srcdir` to turn source modifications
into patches — `lowerdir=/workspace/srcdir, upperdir=/meta/upper` — and on RHEL 8's
4.18 kernel, inside a user namespace, the copy-up needed to create a directory
there fails:

```
error: failed to create directory `/workspace/srcdir/tensorprimitives-rs/target`
Caused by: Cross-device link (os error 18)
```

`cargo` is the first thing in the session that writes inside `srcdir`. The fix is
`export CARGO_TARGET_DIR=${WORKSPACE}/target`, which is outside the overlay —
`/workspace` is a plain bind mount — and it stays in the submitted recipe, coupled
to `install.sh --artifacts` through the same variable so the two cannot drift.
This is why the 2026-08-07 dry-run never saw it: `build_tarballs.jl` creates no
such overlay, only `Wizard.interactive_build` does, and `srcdir_overlay` is not
reachable from `run_wizard()`.

**12 of the 15 platforms were validated in the session**, read out of the saved
wizard state rather than off the screen: all 11 Linux targets plus
`x86_64-unknown-freebsd`. The other three:

* `x86_64-w64-mingw32` failed, **spuriously**. `match_files` compares
  `normalize_name` of the prefix contents against the products recorded on Linux:
  `libtensorprimitives_tapp` against mingw's `tensorprimitives_tapp.dll`. The file
  that would match, the import library `libtensorprimitives_tapp.dll.a`, is
  filtered out first because ObjectFile reads ELF, MachO and COFF and has no `ar`
  reader. The platform built and audited fine on 2026-08-07 with the two-name
  product, which is the thing the wizard's check cannot express. Disabled at step
  6 to reach deployment, then restored by hand.
* Both Apple targets were never attempted — the Xcode SDK licence was declined at
  step 1, and macOS is dropped silently at that point rather than at build time.

Two smaller wizard behaviours worth knowing next time: step 5a is dead code now —
its candidate filter is `plat isa typeof(p)` and every platform is a `Platform`
since BinaryPlatforms replaced the per-OS types, so it always finds zero
candidates and is skipped. And editing the script mid-session goes through
`change_script!`, which does `empty!(state.validated_platforms)`, so one late edit
re-validates every platform that had already passed. Type the whole script,
conditionals included, in step 3.

#### Two exclusions re-checked on review, and one of them was already stale

A second reviewer asked, on the platform filter, whether the missing toolchains
weren't already available as of Yggdrasil#14189. They were. That PR — "[RootFS]
Update Rust to 1.97.0 and enable more platforms", merged 2026-07-31 — added both
`aarch64-unknown-freebsd` and `riscv64-linux-gnu` as Rust targets, the latter under
the renamed target triple `riscv64gc-unknown-linux-gnu`.

**The reason the exclusion looked permanent is worth keeping, because it will
recur.** The check `RELEASING.md` prescribed was run against a standalone
`Pkg.add("BinaryBuilder")`, which resolves the released BinaryBuilderBase. That is
not what builds the merged recipe. Yggdrasil's `Project.toml` pins both
BinaryBuilder and BinaryBuilderBase to `master` through `[sources]`, and the gap
between released and master is measured in whole Rust toolchains:

| stack | Rust shards | `choose_shards` with `[:c, :rust]` |
|---|---|---|
| BinaryBuilderBase 1.44.0, released | 1.57.0 … 1.94.0 | fails for exactly those two |
| BinaryBuilderBase 1.47.0, what Yggdrasil master pins | 1.57.0 … **1.97.0** | fails for none of the 18 |

So a local dry-run against the released stack cannot see a platform CI can build,
and reports it in a form that reads as a property of the shard list rather than of
the version being resolved. The procedure now instantiates an environment from
Yggdrasil's own `Project.toml` and `Manifest.toml`, and that environment is what
future dry-runs should use.

**Both platforms were then built rather than assumed**, since shard availability is
not a build and both were new ground for this crate — riscv64 takes the scalar path,
and `aarch64-unknown-freebsd` is the first non-Linux entry into `kernel::aarch64`.
Each produced a tarball with one library and the header, one expected
`libgcc_s.so.1` warning, both licences, and `already has SONAME
"libtensorprimitives_tapp.so"` — so the link-time SONAME holds on FreeBSD and on
riscv64. (That log line is `ensure_soname`'s **early return**, and was read at the
time as evidence that the flag was doing something necessary. It is not: see the
next section.) D40's `patchelf --page-size 65536` hazard never arises on either
64 KiB-page target because nothing rewrites the ELF. Build times were 10.5 s and
8.7 s against setup times of 1m 45s and 2m 17s, which is the usual shape here.

**The `i686-w64-mingw32` exclusion still stands** and was checked the same way, not
carried over: BinaryBuilder's `docs/src/build_tips.md` on master still says in as
many words that its Rust toolchain does not work on that platform.

One consequence for the recipe: the two new platforms exist **only** at Rust 1.97.0,
so `preferred_rust_version = v"1.94.0"` — the pin this recipe carried until the
wizard rewrite dropped it, because `print_build_tarballs` cannot emit it — would now
silently cost two platforms. Dropping it was right for a reason that had nothing to
do with why it was dropped. The platform count goes 15 → 17.

#### The audit was already doing two of these

A third review comment, on the two link-flag branches of the build script:
*"Isn't this done automatically by the audit?"* It is — both of them, by **two
different passes**, which is why reading the one function whose name matches read
like a complete answer and was not. Verified against BinaryBuilder `master` on
2026-08-15 rather than from the version the dry-runs used.

| what the recipe set | what sets it anyway | where |
|---|---|---|
| ELF `-Wl,-soname,libtensorprimitives_tapp.so` | `ensure_soname` finds no `DT_SONAME`, runs `patchelf --set-soname $(basename(path))`, reads it back and verifies | `src/auditor/soname_matching.jl:43`, from `Auditor.jl:231` |
| Mach-O `-Wl,-install_name,@rpath/libtensorprimitives_tapp.dylib` | `fix_identity_mismatch` runs `install_name_tool -id @rpath/$(basename(old_id))` | `src/auditor/dynamic_linkage.jl:404`, from `Auditor.jl:515` |

Both are gated on `autofix`, which `build_tarballs` defaults to `true`
(`AutoBuild.jl:764`) and Yggdrasil does not turn off. `basename` of ld64's default
ID is the library's own filename, so the value the audit writes is the value the
flag was writing — identical, not merely equivalent.

**What made the error durable is worth more than the four lines it cost.** D40 was
written from `ensure_soname` alone, and that function *is* a dead end on Mach-O:
ld64 always emits an `LC_ID_DYLIB`, so `get_soname` returns non-`nothing` and it
returns `true` without looking at the value. The conclusion drawn — "the audit says
nothing" — required a second premise nobody stated, that no other pass touches the
ID. Nothing was measured wrong and nothing shipped wrong; the flags were redundant
rather than harmful, since both routes produce the same bytes.

And the confirming evidence was in the audit logs from the first dry-run, reading
as its own opposite: `already has SONAME "libtensorprimitives_tapp.so"` is the
early return, printed **because** the link flag had already done the job. Had the
flag been absent the log would have said `Set SONAME of … to …` and the tarball
would have been identical. **A passing audit does not distinguish a check that
fixed something from a check that skipped it**, so a log line is not evidence about
which of the two happened (A63).

What survives, and is unchanged: every build site the audit never sees still needs
the flags — `install.sh`'s warning, the `c-consumer` job in `ci.yml`, the
`artifacts` job in `release.yml`, and the hand-build recipe in
`examples/c-consumer/README.md`. rustc's omission is real; it is repaired for free
in exactly one place.

The `crt-static` branch stays, and the same question has the opposite answer there:
with the musl default, cargo drops the `cdylib` and exits 0, so there is no library
in the prefix for any audit pass to inspect or repair. That failure surfaces as a
missing `LibraryProduct`, which is D39's silent-failure mode one step later.

**Re-checked rather than assumed**, since the flag has been in the recipe since
before the toolchain moved: on rustc 1.97.1 — one patch above the 1.97.0 that
`choose_shards` now picks — both `x86_64-unknown-linux-musl` and
`aarch64-unknown-linux-musl` still warn that they are dropping the unsupported
`cdylib` crate type without the flag, and are silent with it. Upstream's position is
unchanged: the musl specs set `crt_static_default = true` and
`crt_static_allows_dylibs = false`, and rust-lang/rust#110509 and
rust-lang/cargo#8607 are both still open. The `cross-musl` job's negative control
is what will notice the day this changes.

#### The wrapper leaves the tree, and a claim about it that was never true

`julia/TensorPrimitives` is now
[TensorPrimitives.jl](https://github.com/lkdvos/TensorPrimitives.jl), with the five
commits that ever touched it replayed at the root so the `file:line` citations above
still resolve to something. Two facts about the ecosystem decided the shape, and both
postdate everything written above:

* **TBLIS.jl v0.3.0** (`QuantumKitHub/TBLIS.jl`, 2026-08-03) is a *pure* `tblis_jll`
  wrapper — one hard dependency, Clang.jl-generated bindings committed per target
  triple under `src/lib/`, a `gen/` harness that pins the generator, and **no `ext/`
  at all.** It knows nothing about TensorOperations.
* **The bridge moved upstream.** `ext/TensorOperationsTBLISExt.jl` now lives inside
  TensorOperations (PR #290, shipped in v5.8.0), with `TBLISBackend <: AbstractBackend`
  declared in TensorOperations' own `src/backends.jl`. The dependency points
  TensorOperations → TBLIS.jl, and the standalone `TensorOperationsTBLIS.jl` is
  superseded.

So "a wrapper in TBLIS.jl's style with a TensorOperations extension" is **three
repositories**, not one package: the wrapper, this engine's JLL, and an extension in
TensorOperations. It is also a strictly ordered release chain — the JLL must be in
General before the wrapper can resolve, and the wrapper must be registered before
TensorOperations can name it as a weak dependency. Worth knowing before promising
anyone a date.

The one design consequence for the wrapper: the extension lives in another package,
so `_handle()` and `_executor()` cannot stay private-by-underscore. An extension
cannot reach into another package's internals, so the process-wide context becomes
documented API — which is also the name the plan-once-execute-many path never had.

**The claim that was never true.** This chapter said `tensoradd!` and `tensortrace!`
"fall through to TensorOperations' own backends". They throw. `interface.jl:40-59`
routes any backend that is not `DefaultBackend` or `NoBackend` to a catch-all whose
`else` branch is `throw(ArgumentError("Unknown backend $backend …"))`, so
`@tensor backend = TAPPBackend() C[i, j] := A[j, i]` — a permutation, which is a
`tensoradd!` — has never worked, and neither has any expression containing a trace or
a partial trace. Nothing in the suite noticed because **every one of the nine backend
testsets drives `tensorcontract!`**, which is the one method that exists. A backend is
all-or-nothing unless it delegates explicitly; the fix is three-line `tensoradd!` and
`tensortrace!` methods that forward to `select_backend`'s choice, which makes the
sentence true rather than deleting it.

That is A64, and it is the same shape as A15 and A20 one domain further out: the
capability was described, the description was plausible, and the test that would have
falsified it was the one test nobody wrote. Reading TensorOperations' own dispatch
chain is what settled it — not reading our backend, which is correct as far as it goes.

**Two smaller things this pass tidied.** The version lockstep is gone: the wrapper had
been pinned to the Rust workspace version by a `consistency` step that read
`julia/TensorPrimitives/Project.toml`, which would now fail on a path that does not
exist, so the release checks four version statements rather than five. And the record
disagreed with itself about the size of the Julia suite — 64 tests in this chapter,
76 in the CHANGELOG. Neither number survives the split, and both are now moot: the
suite is **52 tests in the wrapper and 117 in the extension**, from a run against a
locally built library rather than from a count of `@test` lines. The extension's are
all against `StridedNative()` on the same inputs, and they include the two delegated
operations and a network mixing all three, which is what the old suite could not have
covered because those paths threw.

**And the JLL is not actually available.** Yggdrasil#14374 merged 2026-08-15, but three
days later `JuliaBinaryWrappers/tensorprimitives_tapp_jll.jl` does not exist, General
has no `jll/T/tensorprimitives_tapp_jll`, and the merge commit's only check-run reads
"This check was skipped" against a commit status still `pending`. A merged recipe is
not a published JLL, which is worth writing down because every downstream step in this
section waits on it.
