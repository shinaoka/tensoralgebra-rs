# Changelog

All notable changes to `tensorcontract` and `tensorprimitives-tapp`. The reasoning
behind every entry, and all of the data, lives in
[`DECISIONS.md`](DECISIONS.md) — this file records *what* changed and how well
it is known, not why.

The two crates share a version. `tensorprimitives-bench` is not published.

## 0.1.0 — unreleased

First public prerelease. The engine is correct and framework-complete, the
x86 micro-kernels are real, and performance work is partly done and
[documented case by case](DECISIONS.md). Read "Confidence" below before quoting
a number.

### Added — crate layout

One crate per primitive, named for its operation:

* `tensorcontract` — the contraction engine, described below.
* `tensortranspose` — planned, for dedicated transpose kernels. Not yet written
  and not published as a placeholder.
* `tensorprimitives-tapp` — the TAPP C ABI over whichever primitives exist.

`tensorprimitives` is the project name, not a crate. A facade re-exporting a
single primitive would be indirection rather than abstraction, and one can be
added later without breaking existing dependents.

### Added — engine

* **Transpose-free dense contraction** of generally-strided real and complex
  tensors, computing
  `D[idx_D] = alpha * op_A(A) * op_B(B) + beta * op_C(C)`.
  Block-scatter-matrix tensor contraction (Matthews, arXiv:1607.00291) inside
  BLIS's five-loop, two-level-packing structure: no transposed copies, no
  temporary workspace, no FFI in the hot path.
* **Three interchangeable complex methods** — `Planar` (default), `OneM`,
  `ThreeM` — sharing index analysis, scatter machinery, driver and write-back,
  and differing only in pack format, micro-kernel and tile recombination. Chosen
  per plan with `Plan::with_complex_method` or globally with
  `TENSORCONTRACT_COMPLEX`.
* **Index analysis** with mode folding, per-tensor diagonals (repeated labels),
  stride-0 reductions, and `einsum`-style label parsing. Broadcast *output*
  indices are rejected rather than silently mishandled.
* **Scatter and block-scatter layouts**, with an irregular-block sentinel
  distinct from a legal zero block stride, so a reduction stays on the fast path.
* **Micro-kernels** for `f32`, `f64` and all three complex methods, generated
  from one body per method over `(lanes, MR, NR)`:
  * **AVX-512** — the measured configuration.
  * **AVX2 + FMA** — same source text, different instantiation.
  * **portable scalar** — always present, always correct.
  Dispatch is `avx512f` → `avx2 + fma` → scalar at run time; a pinned instruction
  set the CPU lacks falls back to scalar rather than faulting.
* **Cache blocking** with hardcoded constants by default, plus an analytical
  model derived from probed cache descriptors (sysfs, then x86 `CPUID`, then
  conservative built-ins) behind `TENSORCONTRACT_BLOCKMODEL=model`.
* **Multi-threading** behind `Plan::with_threads` / `TENSORCONTRACT_THREADS`,
  default **1**. A 2-D `pm x pn` static partition of the *output*: every element
  has one owning thread accumulating over the full `K` in the original order, so
  results are **bitwise identical to serial at every thread count and every
  partition**. That is what the test suite asserts.
* **Brute-force reference implementation** (`reference`) for oracle testing.
* **Runtime switches** for every fast path, so any change can be A/B-tested in
  one session instead of as a diff between two builds: `TENSORCONTRACT_COMPLEX`,
  `_KERNEL`, `_BLOCKMODEL`, `_MC`/`_KC`/`_NC`, `_MC_PCT`/`_NC_PCT`,
  `_KC_COUPLE`, `_DEEPEN`, `_PARTITION`, `_THREADS`, `_ORIENT`, `_WRITEBACK`,
  `_ROWBLOCK`.
  Documented in `CLAUDE.md`; they exist for measurement and none of them affects
  correctness.
* **`no-std`-adjacent feature gating**: the `std` feature (on by default) gates
  every environment-variable read, and the crate compiles without it. It is not
  a `#![no_std]` crate.

### Added — `tensorprimitives-tapp`

* TAPP (Tensor Algebra Processing Primitives, arXiv:2601.07827) C ABI over the
  engine, as `lib`, `cdylib` and `staticlib`, verified against the upstream
  headers from `TAPPorg/reference-implementation` and exercised end to end
  through the C entry points. `num-complex` layout compatibility with
  `TAPP_C32`/`TAPP_C64` is pinned by a test.

### Performance

Single core, full 49-case TCCG corpus, geometric mean over the corpus unless
stated. **Every entry names its machine, because none of them may be compared
with another**: the optimisation results below are Xeon Gold 6244 (Cascade Lake),
the baseline comparison is Ice Lake-SP, and the AVX2 results are Zen2.

Noise floors are derived in session, never imported. The Cascade Lake sessions
measured **±1.3% on a corpus geomean and ±6% per case**; the exclusive Ice Lake
node measured **0.998–1.001 with 0 of 980 case points outside ±6%**, roughly 5x
tighter. Drift is a function of how far apart two arms are and of the machine —
0.02% at a minute to 4.4% across a cold start on a boost-happy node (A32), and
undetectable over four hours on an exclusive SMT-off one. Take the floor from the
session, and be aware that problem size is a separate confound from time (A46).

* Write-back **orientation** fix (compute `D^T = B^T A^T` when `D`'s column
  direction is the contiguous one): **+12–17%** in all four dtypes, closing a
  profiled 2x defect on the cases it affects, with no case left slower.
* Regular-block write-back on top: a further **2–15%**, most of it in single
  precision.
* A **menu of register blocks per method** with a guarded selection rule:
  **1.026** in `c64` planar (1.124 on the 12 cases it fires on); inside the
  noise floor elsewhere.
* A corrected, antisymmetric **orientation rule**: **1.028** in `f32`,
  **1.022** in `c32` 1m, best case **1.48x**, 64-bit flat. Within 0.4% of a
  hindsight oracle in `f32`.
* Of the three complex methods, **planar is fastest** over the corpus on both
  AVX-512 machines measured. The ordering below it, and the inversion on
  memory-bound shapes where 3m's flop saving pays, are **Cascade Lake results that
  do not transfer**: on Ice Lake 3m is last in every column and wins 0 of 49 cases
  (0.694 `c64`, 0.744 `c32` against planar, where Cascade Lake had 0.956 and
  0.921), and the inversion is absent (A44). Partly confounded with the register
  blocks being wrong there (A34). The ranking also depends on the instruction set:
  at the kernel level on **AVX2**, 3m comes first in `f32`/`c32`, the opposite of
  AVX-512 (A24). Do not treat the ranking as a property of the engine.
* **Against the baselines on Ice Lake** — median over the 12-case premise set at
  200 MiB, single core, everything single-threaded — the engine reaches
  **1.51x `f64`, 1.31x `c64`, 1.78x `f32`, 1.40x `c32`** over **TBLIS 2.0-dev**,
  and 1.95–2.20x over TTGT across the full corpus. Measured twice in separate
  allocations, agreeing to 0.997–1.003 on all twenty columns. Not comparable with
  any Cascade Lake number, and not evidence of what the Phase 4 work bought.

### Confidence — what is measured and what is only correct

The distinction matters more than the numbers, so it is stated per item.

| Component | Status |
|---|---|
| Correctness of everything below, in every dtype, method, ISA and thread count | **Tested** against a brute-force oracle, and in the harness against TTGT and TBLIS. Randomised extents and axis orders, diagonals, reductions, negative strides, empty and scalar cases, all 16 conjugation masks, shapes that cross every cache-blocking level, and the irregular block-scatter path |
| AVX-512 register blocks | **Measured** (`examples/kernel_shapes`), per method and element type, at the `kc` the engine uses — **on one microarchitecture.** Two later findings qualify this and neither is fixed in this release: register blocks turn out to be a property of the *microarchitecture*, not of the instruction set, so Cascade Lake's winners are not Ice Lake's (A34); and one of the eight shipped configurations disagrees with the sweep that chose it — `planar` `f32`/`c32` ships `32x6` where the sweep's own output names `32x5`, **7.8% faster** at the operating `kc` (A35). The default is unchanged, but the shape is no longer unreachable: the row-block menu is keyed by position rather than by `MR` (D43), so `32x5` is on the menu and `TENSORCONTRACT_ROWBLOCK=idx=3` runs it. A kernel margin is not a corpus margin, which is why this is an arm and not a retune |
| Write-back, orientation and row-block rules | **Measured** — whole-grid sweeps over all 392 corpus case-dtype-methods, scored offline |
| **AVX2 register blocks** | **Measured, and all eight shipped shapes are the winners** — on Zen2 (`worker5040`), where the `avx2`-without-`avx512` dispatch branch also executed on real hardware for the first time. They stopped being a guess with no code change. Read with A34 above: this is one AVX2 microarchitecture, not AVX2 in general |
| **Default cache blocking (`MC`/`KC`/`NC`)** | **Swept on two machines, and closed with a negative result: there is no few-percent win here.** `KC` is first-order — it decides whether the `A` sliver is an L1 resident or an L2 stream — but on the reference machine `kc = 384` is the optimum and the shipped 256 is close to it, while `MC` is a wide plateau a sixteenfold range moves by at most 3%. Pinned `kc = 512` is a *machine-specific* result and must not ship: 1.050 on Zen2, **0.961** on Cascade Lake. Coupled deepening looked like +3% on both machines and **failed its end-to-end A/B** — both grids' `base` arm was 1–2% slow, inflating every `arm/base` ratio identically, so the two-machine agreement was a shared artefact rather than a replication; control-corrected the treatment is ≈0.977. `TENSORCONTRACT_DEEPEN=on` survives as an off-by-default record of that experiment, not as a pending improvement. Shipped defaults remain `kc = 384/256` by real size, fitted to one Cascade Lake workstation. Three candidate changes and a mechanism all died on measurement; do not reopen without a new machine or a new mechanism |
| **The analytical blocking model** | **Measured, and it loses. Off by default and staying there.** On the first unseen machine it is worse in **11 of 12 columns**, by up to 7.2% in the complex methods, and the whole loss is attributable to its `kc` (A33). It was built to solve portability and does not; the hypothesis is refuted rather than pending |
| **Threading** | **Scaling measured on six machines, off by default and now off for a measured reason.** Below ~1 MiB at 64 threads it is 1.2–10x *slower* than serial — per-call thread spawn is ~20–36 µs per thread — and the optimal thread count walks 4 → 64 across the size range, so a fixed default is wrong at every size but one (D46). Bitwise-identical results are asserted at every thread count *and every partition*. Scaling is strongly topology-dependent: Zen2 saturates by 16–32 and declines at 64, Ice Lake reaches 48% of linear at 32 in `c64` 3m. A 4.3x partition win on Zen2 is **absent on Ice Lake** (A36), so the first machine alone would have produced the wrong rule. Threads are still spawned per call rather than pooled |
| **The domain-aware partition** (`TENSORCONTRACT_PARTITION=domain`) | **Measured on two topologies, it met its pre-registered prediction, and it is now the default (D44).** It supplies the input `Plan::partition` was missing (how many L3 domains the thread set spans) and *gates* the `panels >= p` early return on it rather than removing it, which would be wrong where one L3 serves the socket. Predicted from another node's grid before the run: 1.423 on the 144 case-dtype-methods it moves, 1.138 corpus. Measured: **1.433 and 1.133** drift-corrected, on exactly 144 of 392, and **0 of 392 moved on a one-L3-per-socket machine**. The effect is monotone in domain count and flat in thread count, including a fixed-16-thread arm that varies only the packing (1.19x packed over 4 domains, **2.74x spread over 16**). Corpus scaling at 64 Zen2 cores goes 5.81 → 7.54 (`f64`) and 5.51 → 7.88 (`f32`). `TENSORCONTRACT_PARTITION=legacy` restores the ungated rule, which is what every threaded number committed before 2026-08-04 was measured with. Reproduce the prediction with `scripts/partition-score-rule.py`; raw arms in `bench-results/worker5479-zen2` and `worker6150-icelake` |
| Absolute throughput off the reference machine | **Measured on Ice Lake, twice, and it is the best-known set here — but still not comparable with the reference machine.** Two full engine-vs-baseline runs in separate allocations agree to 0.997–1.003 on all twenty dtype × engine columns, each internally bracketed by a repeat arm 2.5 h away reading 0.998–1.001 with 0 of 980 case points outside ±6%. The Zen2 sessions remain within-session only. **No number from any of them may be differenced against a `ccqlin038` number** — which is why the improvement the Phase 4 work bought is still unmeasured end to end |
| **The complex-method ranking, off Cascade Lake** | **Measured and it does not transfer (A44).** Planar still wins the corpus, but 3m falls from 0.956/0.921 against planar to 0.694/0.744 on Ice Lake, is last in every column, and wins 0 of 49 cases — and the memory-bound inversion that Phase 4 item 3 was to exploit is absent. Confounded with A34's wrong register blocks and not separable without an Ice Lake shape sweep. Item 3 must not ship as an unconditional rule |
| **The TBLIS 2.0 baseline's build** | **Measured, and it does not matter at these sizes — but it decides whether the baseline runs at all.** The install every number here used was configured `BLIS_CONFIG_FAMILY=auto`, hence skx-only, and **SIGILLs on any machine without AVX-512**. Rebuilt multi-config it is identical at 64 MiB and 200 MiB (0.997–1.000 against a 0.998–1.003 floor), so the comparison numbers stand; an earlier claim that the skx build understated TBLIS 2.0 by up to 1.68x was measured at 8 MiB and is withdrawn (A45, A46) |

`K`-parallelism is deliberately absent, on evidence rather than by omission —
see `DECISIONS.md` A21.

### Added — distribution surface

* **A version that can be asked at run time.** `TAPP_VERSION_MAJOR/MINOR/PATCH`,
  `TAPP_VERSION_STRING` and `TAPP_VERSION_AT_LEAST()` in `<tapp.h>` describe the
  header; `TAPP_implementation_version()` describes the *linked library*. Nothing
  reported a version before, so a header and a shared library from different
  packages could disagree silently. Both halves are now tested: `abi_layout.rs`
  checks the macros against `CARGO_PKG_VERSION` by reading the header as text,
  and `examples/c-consumer` compares the macro a C compiler saw against the string
  the library it linked returns.
* **A prefix install.** `crates/tensorprimitives-tapp/install.sh --prefix=DIR`
  lays out `lib/`, `include/`, `lib/pkgconfig/` and `share/licenses/`, generating
  a `tensorprimitives-tapp.pc`. The BinaryBuilder recipe calls the same script, so
  a JLL and a manual install present the same layout by construction.
* **A third link mode in `examples/c-consumer`**, `-DTAPP_PREFIX=<dir>`, which
  consumes an installed prefix through `pkg_check_modules`. It is the only mode in
  which the header is found outside the source tree. CI runs all three.
* **Every declaration in `<tapp.h>` is now referenced from C.** Eight prototypes
  — the setters, `TAPP_get_strides`, the batched product, `TAPP_destroy_status`,
  `TAPP_attr_set`/`_clear` — were declared and never compiled or linked by any C
  translation unit, so their only check was a second hand-maintained Rust
  transcription.
* **`SONAME` / install-name are set at link time.** rustc emits neither for a
  `cdylib`: on ELF there is no `DT_SONAME` at all, and on Mach-O `LC_ID_DYLIB` is
  the absolute build-tree path. `RUSTFLAGS='-C link-arg=-Wl,-soname,…'` (and the
  `-install_name,@rpath/…` counterpart) fix both without rewriting binaries, which
  matters because `patchelf` needs `--page-size 65536` on every 64 KiB-page
  architecture. Documented for hand-building sites, used by CI and by the recipe,
  and `install.sh` warns when they are missing.

### Added — Julia and a binary distribution

* **A BinaryBuilder recipe**, `packaging/yggdrasil/build_tarballs.jl`, producing
  `tensorprimitives_tapp_jll` over 15 platforms. It calls the same `install.sh` a
  site install does, so the layout a JLL presents is the layout CI tests. It carries
  a `TAPP_LOCAL_SRC` switch so the file that gets submitted is the file that was
  dry-run. **13 of the 15 have been built and audited**, each producing an identical
  layout containing a real shared library; the two Apple targets are unbuilt because
  they require accepting the Xcode SDK licence. `riscv64-linux-gnu` and
  `aarch64-unknown-freebsd` are excluded — no Rust toolchain shard exists for them —
  as is `i686-w64-mingw32`, per BinaryBuilder's own documentation.
* **`julia/TensorPrimitives`**, a Julia package in two layers: `LibTAPP`, a
  complete `ccall` wrapper over `<tapp.h>` with handles as distinct Julia types and
  finalizers; and `TAPPBackend`, a `TensorOperations.jl` backend. The contraction
  mapping needs no permutation and no temporary — arbitrary extents, element strides
  and per-operand conjugation go straight to the ABI — which is the engine's central
  claim, now exercised from Julia. 76 tests, each checked against
  `TensorOperations`' own backend on the same inputs.
* **`RELEASING.md`** and a `release` CI workflow that checks a release and
  publishes nothing: the five places a version is written must agree, the recipe
  must not still hold a placeholder checksum, and `cargo publish --dry-run` must
  pass.

### Added — portability

* **Eleven cross-compilation targets checked in CI**, which had never been done:
  `i686`, `aarch64`, `armv7`, `ppc64le` and `riscv64` Linux, both `musl`
  architectures, `x86_64-pc-windows-gnu`, both Apple targets, and FreeBSD. All
  compile unmodified — including 32-bit x86, where the AVX-512 intrinsics *are*
  instantiated and were expected not to exist.
* On `musl` the `cdylib` requires `-C target-feature=-crt-static`. Without it
  cargo prints `dropping unsupported crate type cdylib` and **exits 0**, so the
  build succeeds and ships no shared library. CI asserts both arms.

### Not included

* No AArch64/NEON, RISC-V or GPU kernels. Off x86 the engine runs the portable
  scalar path, which is correct but not competitive.
* No dispatch of the complex method by shape, though the inversion that would
  justify it is measured.
* No thread pool, no `pc`-loop fusion, no pack-free fast path for already
  unit-stride block scatter, no software prefetch.
* `tensorprimitives-bench` (the TCCG harness, the TBLIS and OpenBLAS-TTGT
  baselines, the GEMM roofline and the stride-stress modes) is in the repository
  but `publish = false`.

### Fixed before first release

Found by the new TAPP conformance suite, which was written because the C ABI had
one test. All four were behaviour a C caller could observe:

* Extents whose product overflows are now a `TAPP_ERROR_SHAPE` error. Previously
  the product wrapped: a release build reported success and computed nothing, and
  a debug build aborted the calling process from inside `extern "C"`.
* A null `C` (`TAPP_IN_PLACE`) with a non-zero `beta` is now
  `TAPP_ERROR_UNSUPPORTED` instead of silently overwriting `D`. With `beta == 0`
  it still overwrites, as before; in-place accumulation is expressible by passing
  `D`'s own pointer as `C`.
* Library handles are now distinct allocations. They were all the value `1`
  (`HandleState` was zero-sized), so two handles were indistinguishable and
  destroying both was a double free — harmless only while the state stayed empty.
* `TAPP_attr_set` / `TAPP_attr_get` / `TAPP_attr_clear` are exported and return
  `TAPP_ERROR_UNSUPPORTED`. They were declared by `<tapp.h>` and missing from the
  library, so calling one was a link failure.
* `TAPP_execute_product` writes `0` through a non-null `status`, so the idiomatic
  create/execute/`TAPP_destroy_status` sequence no longer passes an uninitialised
  value.
* `<tapp.h>`'s prose named the wrong error code for a null `C` with a non-zero
  `beta` — `TAPP_ERROR_NULL`, where the library returns and the conformance suite
  asserts `TAPP_ERROR_UNSUPPORTED`. Prose is the one part of the header nothing
  tested; see the header/library agreement checks added for this release.

### Notes

* **MSRV is 1.89**, where the AVX-512 intrinsics and `is_x86_feature_detected!`
  stabilised. CI pins exactly that version, because the claim had drifted twice:
  the pin once sat *below* the declared floor, and the `CPUID` cache probe once
  called `__cpuid_count` outside an `unsafe` block, which is only a safe function
  from 1.94 and silently raised the real floor by five releases.
* **Publish order is fixed by cargo**: **`tensorcontract` first**, then
  `tensorprimitives-tapp` once the engine is in the index. The TAPP crate depends
  on the engine by path *and* version, so it cannot even be packaged beforehand — `cargo package` fails at "failed to prepare local
  package for uploading" until `tensorcontract 0.1.0` is in the registry.
* Licensed MIT OR Apache-2.0.
* Any statement about TBLIS in `DECISIONS.md` names a version. v1.3.0 and
  2.0-dev differ by ~5x on complex data and swap two ABI enumerators; treating
  them as one library is the single easiest way to get a wrong answer here.
