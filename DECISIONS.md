# Decisions, assumptions and phase reports

Newest phase report last. Every material decision is recorded here so the run
is auditable after the fact.

---

## Resume here

**State as of 2026-08-03.** Phases 1, 2 and 3 are complete and their gates are
met. **Phase 4 is in progress: items 1 (the write-back), 1c (the micro-tile
row block) and 1d (the orientation rule) are done; item 2 is next.**

What exists:

* `crates/tensorcontract` — the engine. Correct and framework-complete: index
  analysis with folding, scatter/block-scatter, packing in four formats, three
  complex methods, five-loop driver, scattered write-back, brute-force oracle.
  Micro-kernels are **AVX-512 for `f32`/`f64` and all three complex methods**,
  runtime-dispatched, with the portable scalar path retained behind
  `TENSORCONTRACT_KERNEL=scalar`. No AVX2 path yet.
* `crates/tensorcontract-tapp` — TAPP C ABI, verified against the upstream
  headers, exercised end to end through the C entry points.
* `crates/tensorcontract-bench` — `tcbench` with `verify` / `premise` /
  `sweep` / `info`, plus two *analyses* that touch no data and cost no CPU, so
  they are safe to run while a benchmark is in flight: `shapes` (what each
  register block would do to the write-back) and `orient` (the structural
  features of both orientation arms). TBLIS (1.3 and 2.x) and OpenBLAS-TTGT
  baselines; the TCCG corpus; GEMM roofline; stride-stress modes; CSV output.
* `bench-results/` — raw CSVs for every measurement quoted in this file, all
  committed. `phase3-*` is the Phase 3 set; `phase4/rm-*` is the Phase 4
  re-measurement on an exclusive machine and is the one to compare against.
  **`phase4c/rb-*` and `phase4d/or-*` are the two grids**: every register block,
  and both orientation arms, for all 392 corpus case-dtype-methods. They are the
  most valuable asset here — a candidate rule can be scored against them offline
  for free, and that is how both Phase 4.1c and 4.1d were settled.
* `scripts/` — `env.sh` (toolchain + single-threading), `phase3-bench.sh` (the
  Phase 3 measurement set), `phase4-remeasure.sh` (the A/B pattern to copy),
  `phase4c-rowblock.sh` and `phase4d-orient.sh` (the two grids),
  `compare-sweeps.py` (turns two sweep CSVs into the ratio tables in this file),
  and `rowblock-decompose.py` / `rowblock-score-rules.py` /
  `orient-score-rules.py` (score candidate rules offline against a grid — the
  pattern to copy for any further discrete tuning decision).

Sanity check on a fresh checkout, in this order:

```bash
cargo test --workspace --release              # 6 green suites
TENSORCONTRACT_KERNEL=scalar cargo test --workspace --release
cargo clippy --workspace --all-targets        # silent
cargo build --release -p tensorcontract-bench
cd bench-results/phase4 && python3 ../../scripts/compare-sweeps.py \
    rm-A-f64c64.csv,rm-A-f32c32.csv rm-A2-f64c64.csv,rm-A2-f32c32.csv
```

The last one re-derives the published noise floor from committed data and
should print geomeans of 0.99–1.01 without touching the CPU. If it does not,
the analysis tooling has drifted from the numbers in this file.

Settled; do not re-open without new data:

1. The complex-weakness thesis is refuted against TBLIS 2.0 and confirmed
   against TBLIS 1.3.0, for a mundane reason (1.x has no complex micro-kernel
   outside Sandy Bridge). See the Phase 1 report.
2. The real headroom is low arithmetic intensity in either domain.
3. The TCCG corpus cannot test awkward-stride claims without `--stress`.
4. The corpus is 49 cases, not 48.
5. **Write-back regularity is not a performance predictor.** The fraction of
   output row blocks that avoid the gather path explains the write-back path
   and nothing else: maximising it scores 0.936 in `f32` as a row-block rule
   (A16) and 0.936 as an orientation rule (part 6). It has now pointed the
   wrong way three times. It is a *gate* on a change, never an objective.
6. **The three-way comparison is done and planar wins**, by 3–8% geometric
   mean over the corpus in both precisions — but the ranking inverts on
   memory-bound shapes, where 3m wins. The mechanism is bytes moved per useful
   flop, not flop count and not shuffles. See the Phase 3 report. *Caveat added
   in Phase 4:* the margin has narrowed, because 3m gained most from the
   write-back work (`c32` 3m is now the best cumulative improvement at 1.167,
   against planar's 1.116). The mechanism is unchanged; the margin is not.
   Re-measure before quoting a ranking.

**Phase 4 item 1 is done** — see the Phase 4 report, parts 1–4. The 2x
write-back defect is closed in all four dtypes (2.75x / 2.02x / 2.17x / 2.23x
on the affected cases) for **+12–17% corpus geometric mean**, with no case left
slower beyond noise.

**Measure with `scripts/phase4-remeasure.sh`, not ad hoc.** Everything in
Phase 4 was first measured on a shared workstation, and two conclusions had to
be corrected: the write-back was reported as "a wash in double" (it is +2–3%)
and a 4–5% `f32` regression was reported that does not exist. The script runs
`A, B, A'` so a repeat brackets the treatment, and records sibling-CPU
occupancy. **Noise floor on an exclusive machine: ±1.3% on a 49-case geomean,
±6% per case.** Quote that, not a control-group inference.

Two things came out of item 1 that are worth carrying forward:

* The cause was **not** the mechanism Phase 3 named. It was the row/column
  *orientation* of the matrix view, not the write-back's own L2 traffic. See
  A10.
* The orientation choice depends on `MR`, so it is not a property of the plan
  alone (A11), and it must only be taken when the swap leaves the micro-tile
  row block unbroken — the guard is empirical and the case it guards against
  is **not fully explained**. Do not remove it without re-measuring
  `abcijk-*mb-*` in `f32`.

**Phase 4 item 1c is done** — see the Phase 4 report, part 5. The row-block
menu and its guarded rule ship on by default; `c64` planar gains **1.026**
corpus geomean (1.124 on the 12 cases it fires on), `c32` 1m 1.006 (1.108 on
7), everything else is inside the noise floor, and one case out of 392 is left
7% slower. It also **prices the orientation at ~2x** on the nine known misses
and shows `MR` is the wrong instrument for buying it.

**Phase 4 item 1d is done too** — see the Phase 4 report, part 6. The
orientation rule's missing discriminant was found: the rule must be
*antisymmetric* under exchanging the two directions, which neither previous
version was. Condition 2 became a preference with a fallback ("row block fits,
else the shorter run goes in the row role"), which is 12 better / 0 worse over
both arms of all 392 case-dtype-methods and closes `f32` to within 0.4% of a
hindsight oracle. Removing the row-block rule's now-obsolete guard 3 on top is
worth another 1.218 on six `c32` 1m cases. End to end against Phase 4.1:
**`f32` 1.028, `c32` 1m 1.022, `c32` planar 1.013, `c32` 3m 1.010**, 64-bit
flat, best case 1.48x, one genuine per-case regression at 7%.

**Item 2 is in progress: the grid is measuring.** Launched 2026-08-03 08:11 on
`ccqlin038` cpu4, 19 arms x 2 dtype pairs, ~7 h, writing
`bench-results/phase4e/bl-*`. Read Phase 4 report **part 7** for the design and
for what the grid deliberately does not cover. On resume: if
`bench-results/phase4e/run.log` ends in `grid complete`, run
`scripts/blocking-score-rules.py bench-results/phase4e/features.csv
bench-results/phase4e` and write the results into part 7; if it does not, the
run was interrupted and the arms that exist are still valid (each arm is one
independent CSV) — re-run the missing ones with the same script, but **not**
while anything else uses the machine.

**Item 2, in priority order:**

1. **Sweep `MC`/`KC`/`NC`.** Still the untouched Phase 2 heuristic. Phase 3
   showed the whole method ranking turns on whether the `A` sliver is an L1
   resident or an L2 stream, which makes `KC` a first-order parameter. Phase 4
   part 3 adds a design constraint on the sweep itself: `MC` is bounded from
   *both* sides (A13), so the sweep must separate the two rather than look for
   one best value. Read part 3 before starting — the obvious depth-adaptive
   shortcut is already measured and rejected. Part 5 adds a second input: the
   *register block* is depth-conditional too (A18), so the sweep should vary
   the shape alongside `kc` rather than hold the Phase 3 table fixed.
2. **Dispatch the complex method by shape.** The inversion is measured and
   large enough to exploit: 3m on memory-bound shapes, planar otherwise. Note
   Phase 4.1 moved the ranking: `c32` 3m gained the most from the write-back
   work (1.167 cumulative against planar's 1.116), so re-measure before
   dispatching — the Phase 3 margins are no longer current.
3. **Close the remaining orientation gap.** 21 case-dtype-methods still take
   the slower arm, worth up to 1.36x, and they are a different population from
   the `abcijk` family the current rule was derived on. Both arms of all of
   them are in `bench-results/phase4d/or-*`, so a candidate costs nothing to
   score with `scripts/orient-score-rules.py`.
4. Then the rest of the Phase 4 list: threading, small-`k` handling, fusing the
   `pc` loop so `C` is touched once, a pack-free fast path for unit-stride
   block scatter, prefetch.

Do not re-derive the register blocks; they are measured and recorded in
`kernel::x86`, with the sweep in `examples/kernel_shapes` if the machine
changes. Since Phase 4.1c each method carries a *menu* of them and the default
is still the Phase 3 choice — the menu adds alternates, it does not re-litigate
the default.

---

## Environment

All measurements in this file were taken on:

| | |
|---|---|
| host | `ccqlin038.flatironinstitute.org` (Flatiron / CCQ workstation) |
| CPU | Intel Xeon Gold 6244, Cascade Lake-SP, 2 sockets x 8 cores, 3.6 GHz base |
| ISA | AVX-512F/DQ/BW/VL/VNNI, 2 FMA units per core |
| cache | 32 KiB L1d, 1 MiB L2 per core, 25.3 MiB shared L3 per socket |
| memory | 251 GiB, 2 NUMA nodes |
| toolchain | rustc 1.97.1, gcc 13.3.0 (module), cmake 3.31.6; workspace MSRV 1.89 (see D20) |
| TBLIS (A) | v1.3.0 (tag `c4f81e0`, 2 Jul 2025) — the latest **stable** release, re-checked 2026-08-02: still the newest tag |
| TBLIS (B) | `develop` @ `555320c` (4 Dec 2025), version string 2.0, BLIS auto-configured for `skx` — an **unreleased** development snapshot, now 8 months past the newest tag (`v2.0-beta2`) |
| BLAS | OpenBLAS 0.3.29 (`openblas/single-0.3.29` module) |

Both TBLIS baselines were rebuilt from source for Phase 3, at the same commits,
under `../baselines/tblis-{1.3.0,2.0}-install` relative to the repo. They are
not in the repo and not in the build; `scripts/phase3-bench.sh` takes their
prefixes from `TBLIS_ROOT_13` and `TBLIS_ROOT_2X`. The rebuild reproduces the
Phase 1 headline to three digits, which is the check that it is the same
baseline — see the Phase 3 report.

Both TBLIS versions are measured. They behave completely differently on
complex data, and the difference is the single most important result in this
document — see the Phase 1 report.

Note a silent ABI break between them: `type_t` swaps `TYPE_DOUBLE` and
`TYPE_SCOMPLEX` (1.3: `DOUBLE=1, SCOMPLEX=2`; 2.0: `SCOMPLEX=1, DOUBLE=2`).
Mixing them produces plausible-looking wrong numbers with no error. The harness
has a `tblis13` cargo feature and a startup self-check
(`tblis::verify_type_tags`) that multiplies a known matrix in each dtype and
aborts on mismatch.

All runs are **single-threaded** (`TBLIS_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
engine is not yet threaded).

Reference for single-core ceiling: OpenBLAS `dgemm` reaches ~96 GF/s and
`zgemm` ~96 GF/s on large square shapes here, so the two domains have
effectively the same ceiling, as expected.

---

## Standing assumptions

| # | Assumption | Status |
|---|---|---|
| A1 | TAPP can express everything needed, including complex and conjugation. | **Confirmed** from the actual headers: `TAPP_C32`/`TAPP_C64`, `TAPP_CONJUGATE` per operand, `int64_t` label arrays, `intptr_t` handles. No kill condition. |
| A2 | TBLIS `develop` supports TAPP in-tree (per arXiv:2601.07827). | **Refuted.** No TAPP source in `master` or `develop` of `devinamatthews/tblis` v2.0. TBLIS is benchmarked through its native `tblis_tensor_mult` C API instead. |
| A2b | "TBLIS" is one thing. | **Refuted.** v1.3.0 (latest release) and 2.0-dev differ by ~5x on complex, because 1.x has no complex micro-kernel outside Sandy Bridge. Any statement about TBLIS's complex performance must name a version. Both are now measured. |
| A3 | Achievable GF/s peak is the same for real and complex on real-SIMD hardware. | Confirmed: `dgemm` 96 GF/s vs `zgemm` 96 GF/s. This validates the efficiency-ratio metric. |
| A4 | The TCCG corpus exercises the irregular/gather path. | **Refuted.** TCCG rounds stride-1 extents to multiples of 24, which divides every register block in use, so `regA = 1.00` everywhere. Added `--stress ragged` / `--stress padded` to probe it. |
| A5 | Shapes must be held fixed across dtypes for the real-vs-complex ratio to mean anything. | Adopted. Deviates from TCCG's per-precision sizing; documented in `corpus.rs`. |
| A6 | This host is the reference machine; single-core is the headline. | Adopted. Threading deferred to Phase 4. |
| A7 | The three complex methods differ mainly in flop count. | **Refuted in Phase 3.** They differ mainly in bytes moved per useful flop. |
| A8 | One register block per method is enough. | **Refuted in Phase 3.** The best shape depends on method, element type and `kc`. |
| A9 | Absolute performance figures are meaningful. | Adopted from Phase 3 onward; explicitly *not* true before it. |
| A10 | The Phase 3 write-back defect is the write-back's own L2 traffic. | **Refuted in Phase 4.** It was the row/column orientation of the matrix view. |
| A11 | The row/column orientation is a property of the plan. | **Refuted in Phase 4.** It depends on `MR`, hence on element type and method. |
| A12 | Write-back overhead matters equally in both precisions. | **Refuted in Phase 4.** ~2x more of the budget in `f32`/`c32` than in `f64`/`c64`. |

---

## Build-vs-reuse decisions

| Layer | Decision | Why |
|---|---|---|
| N-d array crate | **Build** (`Layout`, 20 lines) | The engine needs pointer + extents + strides. A dependency would leak into the public API and the TAPP C ABI for no gain. |
| `num-complex` | **Reuse** | `#[repr(C)]`, layout-identical to `TAPP_C32/C64` and C99 `_Complex`; ecosystem standard. Layout pinned by a test. |
| SIMD abstraction (`pulp`, `macerator`) | **Build** on `core::arch` | `portable_simd` is unstable on 1.97. Register-blocked kernels want explicit register control. The `Ukr` function pointer keeps a `pulp` backend addable later without touching the driver. |
| GEMM kernels (`gemm`, `matrixmultiply`, `microgemm`, `faer`) | **Build** | All expose *matrix* multiply, not a panel-panel kernel over externally packed buffers, and none has a planar-complex path. `matrixmultiply` (now with AVX-512 `cgemm`/`zgemm`) is a benchmark target, not a foundation. |
| `rayon` | **Build** on `std::thread::scope` (deferred) | BLIS parallelism wants static partitioning with a shared packed-B panel; work-stealing fights that. |
| `tblis`/`tblis-ffi` crates | **Build** ~100 lines of FFI | The benchmark must control which TBLIS is measured — crucially which BLIS config its kernels were built for. Those crates vendor their own build. This turned out to be load-bearing: the headline result *is* a version difference, which a crate that vendors one fixed build could not have surfaced. Struct layout pinned by a `sizeof`/`offsetof` test, and the 1.3-vs-2.0 `type_t` swap by a runtime self-check. |
| `criterion` | **Build** a small harness | Criterion targets many fast iterations of a cache-resident routine. These are 0.1–5 s measurements on 64–200 MiB working sets; best-of-N after warm-up plus a CSV is the right tool. |
| `opt-einsum-path` | **Out of scope** | This engine executes one binary contraction; ordering is a caller concern. |

---

## Design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Reductions (TAPP case 4) become contraction indices with stride 0 in the operand that lacks them. | Needs no workspace, unlike a pre-reduction pass. TAPP itself warns case 4 may need unbounded workspace. Works because block-scatter treats stride 0 as regular. |
| D2 | Irregular block-scatter entries are flagged with `i64::MIN`, not `0`. | Follows from D1: a zero block stride is a *legal, regular* pattern here, so it cannot double as the sentinel. |
| D3 | Diagonals (repeated labels) are handled by summing strides, per tensor, before classification. | Keeps the rest of the engine free of repeated labels. |
| D4 | Broadcast output indices (TAPP case 5) are rejected. | TAPP does not require support; supporting it needs either workspace or redundant compute. |
| D5 | Class ordering: `M`/`N`/`H` by `|stride_D|`, `K` by `|stride_A|`. | The output update is the one access packing cannot hide, so `D` gets first claim on contiguity. Heuristic; a Phase 4 knob. |
| D6 | Scatter vectors are materialised in full at plan time; block-scatter vectors at execute time. | Scatter is element-type independent (so one plan serves all dtypes, which the TAPP layer needs); block-scatter depends on `MR`/`NR` and is `O(M/MR)`, i.e. free. |
| D7 | Micro-kernel writes to a stack tile; `alpha`/`beta`/scatter write-back/re-interleave happen outside. | One kernel serves the regular path, the gather path and every edge block. Cost is one extra store/load of an L1-resident tile. |
| D8 | Conjugation is folded into packing (negate the imaginary plane). | Free, and it is what makes `TAPP_CONJUGATE` cost nothing. |
| D9 | On accumulate passes, `op_C` is set equal to `op_D`. | Conjugation is additive and involutive, so `conj(alpha*AB_p + conj(stored))` accumulates correctly across `K` blocks. Verified by an all-16-masks test forced through multiple `K` blocks. |
| D10 | Blocking is overridable per plan and via `TENSORCONTRACT_MC/KC/NC`. | Lets the test suite drive every level of the five-loop nest on oracle-sized tensors, and lets Phase 4 sweep parameters. |
| D11 | Corpus is TCCG's **full 49-case** set, not 48. | See `DESIGN.md` §5.2: the brief's "48" does not correspond to any list in upstream `benchmark.py`. 49 is the full set and a superset of the 25-case reduced set; `_sortedTCs` is a re-labelling, not a sixth group. |
| D12 | Both TBLIS v1.3.0 and 2.0-dev are benchmarked, behind a `tblis13` cargo feature, with a runtime ABI self-check. | The two releases swap `TYPE_DOUBLE`/`TYPE_SCOMPLEX`, so a mismatch is silent rather than fatal. Given the result hinges on the version difference, guessing was not acceptable. |
| D17 | Micro-kernels are macro-generated over `(MV, NR)` const generics from one body per method, not hand-written per shape. | A comparison between three methods must not also be a comparison between three hand-tunings. One body per method, one shape parameterisation, and the shape is then chosen by measurement. It also made the shape sweep possible at all. |
| D18 | `#[target_feature]` kernels are reached through one-line plain-`fn` trampolines. | A `#[target_feature]` function cannot be coerced to a function pointer, which the `Ukr` contract requires. Cost is one `call` per micro-tile against `kc*MR*NR` FMAs — unmeasurable. |
| D19 | Register blocks were chosen by measured throughput at the `kc` the engine actually uses, per method and per element type. | The uop model gets the *cliffs* right (spills above 32 live vector registers) but the *ranking* wrong: it predicts 3m fastest, and 3m is fastest only when the panels are L1-resident. See the Phase 3 report. |
| D20 | Workspace MSRV raised `1.75` → `1.89`. | AVX-512 intrinsics and `is_x86_feature_detected!("avx512f")` were stabilised in Rust 1.89. The alternative — feature-gating the AVX-512 path so 1.75 still builds — would make the project's headline measurement an opt-in extra. 1.89 is a year old. |

---

## Phase 1 report

**Gate:** self-approved design doc; premise resolved with data; green scaffolded
repo; working harness with baselines wired in.

**Status: gate met. Kill/pivot condition triggered.**

### Delivered

* `DESIGN.md` — literature review (cited), ecosystem survey with per-layer
  build-vs-reuse calls, full engine design, benchmark/test framework,
  self-scrutiny.
* Cargo workspace: `tensorcontract` (core), `tensorcontract-tapp` (C ABI),
  `tensorcontract-bench` (harness). CI (build/test/clippy/fmt/docs/MSRV +
  a scalar-fallback job), dual MIT/Apache-2.0, MSRV 1.75.
* Working engine, correct end-to-end (this is the Phase 2 gate, met early — see
  the Phase 2 report).
* Harness `tcbench` with `verify` / `premise` / `sweep` / `info`, TBLIS and
  OpenBLAS-TTGT baselines wired in, CSV output, GEMM roofline annotation, and
  stride-stress modes.
* Raw results in `bench-results/`.

### The premise check

Hypothesis under test, from the brief:

> TBLIS underperforms on complex contractions, worst in memory-bound / awkward-stride
> cases, because interleaved-complex storage and the scatter/block-scatter packing
> compound and force more work onto the slow full-scatter (gather) path.

Method: 12 cases sampled evenly across TCCG's bandwidth-bound-to-compute-bound
ordering, at 64 MiB nominal tensor size, best of 3, single-threaded. For each,
measure the contraction and a same-shape vendor GEMM, in both a real and the
matching complex dtype. Report `eff = contraction / GEMM` and
`eff ratio = complex eff / real eff`.

Because a complex MAC is four real FMAs counted as 8 flops, the achievable GF/s
peak is the same number in both domains (confirmed: `dgemm` 96, `zgemm` 96).
So `eff ratio < 1` means a complex-specific penalty; `>= 1` refutes the thesis.

**Results — mean `eff ratio` over 12 cases:**

| engine | dtypes | stress | mean eff ratio |
|---|---|---|---|
| **TBLIS v1.3.0** (latest release) | f64 / c64 | none | **0.215** |
| TBLIS 2.0-dev | f64 / c64 | none | **1.060** |
| TBLIS 2.0-dev | f64 / c64 | ragged (`regA` 0.80–1.00) | **1.027** |
| TBLIS 2.0-dev | f64 / c64 | padded strided views | **1.059** |
| TBLIS 2.0-dev | f32 / c32 | none | **1.150** |
| TTGT | f64 / c64 | none | **1.170** |

**Ceiling-free cross-check — raw complex/real GF/s ratio for the same shape:**

| run | n | min | median | max | mean | below 1.0 |
|---|---|---|---|---|---|---|
| **TBLIS v1.3.0 f64→c64** | 12 | **0.19** | **0.33** | 1.15 | 0.40 | **11** |
| TBLIS 2.0-dev f64→c64 | 12 | 0.98 | 1.91 | 2.67 | 1.76 | 1 |
| TBLIS 2.0-dev f64→c64 ragged | 12 | 1.01 | 1.82 | 2.28 | 1.68 | 0 |
| TBLIS 2.0-dev f64→c64 padded | 12 | 1.03 | 1.90 | 2.53 | 1.76 | 0 |
| TBLIS 2.0-dev f32→c32 | 12 | 1.07 | 1.95 | 2.07 | 1.70 | 0 |
| TTGT f64→c64 | 12 | 1.10 | 2.12 | 2.62 | 1.94 | 0 |

### Verdict: the observation is real, the explanation is not, and it is already fixed

The answer depends entirely on which TBLIS you measure, and the two differ by
almost a factor of five.

**Against v1.3.0, the latest stable release, the complex-weakness claim is
emphatically true.** Complex efficiency against the GEMM ceiling is 0.09–0.20
across every case, versus 0.37–1.08 for real: a mean `eff ratio` of **0.215**.

But the *mechanism* is not the one the brief proposes, and the diagnostic is
unmistakable. TBLIS 1.3.0's complex throughput is essentially **flat across
shapes** — 4.1 to 9.1 GF/s, a 2.2x spread — while its real throughput spans
6.2 to 48.4 GF/s, a 7.8x spread. A memory- or scatter-bound effect would track
shape. A flat ceiling means one fixed-throughput kernel is the bottleneck
regardless of what it is fed.

Reading `src/configs/*/config.hpp` in v1.3.0 confirms it directly. The
`TBLIS_CONFIG_GEMM_UKR` macro takes four slots, `(float, double, scomplex,
dcomplex)`:

```
skx1:        TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_6x16, bli_dgemm_asm_6x8,  _, _)
skx2:        TBLIS_CONFIG_GEMM_UKR(_,                  bli_dgemm_opt_6x32_l1, _, _)
haswell:     TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_24x4, bli_dgemm_asm_12x4, _, _)
zen:         TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_6x16, bli_dgemm_asm_6x8,  _, _)
knl:         TBLIS_CONFIG_GEMM_UKR(bli_sgemm_opt_30x16_knc, bli_dgemm_opt_30x8_knc, _, _)
sandybridge: TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_8x8,  bli_dgemm_asm_8x4,
                                   bli_cgemm_asm_8x4,  bli_zgemm_asm_4x4)
```

**Sandy Bridge is the only configuration with complex micro-kernels.** On every
post-2012 x86 target — Haswell, Zen, Skylake-X, KNL — TBLIS 1.x runs complex
tensor contraction on the generic templated fallback while real gets hand-tuned
BLIS assembly. That is the entire effect. It has nothing to do with
interleaved storage, nothing to do with scatter/gather, and nothing to do with
the block-scatter fast path: `regA = 1.00` on every case measured.

**Against 2.0-dev the claim is refuted.** Rebasing onto BLIS-as-framework
brings BLIS's 1m induced method, and complex immediately regains full shape
sensitivity (11.2–82.9 GF/s, a 7.4x spread matching real) and lands at or above
parity — mean `eff ratio` 1.06 in f64, 1.15 in f32, and still 1.03 when
irregular block scatter is forced. The upstream release notes for `v2.0-beta2`
say as much: "a major update … which incorporates BLIS as the core framework"
with improvements including complex number support.

**Why complex is not intrinsically disadvantaged.** The folklore reasoning ran:
complex data is 2x the bytes, scatter/gather is the bottleneck, therefore
complex suffers more. The missing term is arithmetic intensity. A complex MAC
does 4x the flops of a real MAC on 2x the bytes, so complex contraction has
**2x the arithmetic intensity** of the same-shape real contraction. Packing,
indexing and write-back costs are amortised over twice as much arithmetic. Once
a real complex kernel exists (2.0), the memory-bound shapes where complex was
predicted to be worst are where it looks *best*: `abcijk-ikmb-mjac` runs at
8.8 GF/s in f64 and 23.5 GF/s in c64; `abjcd-dkbac-jk` at 5.5 vs 11.2.

BLIS's 1m does inflate the packed A panel 2x (four reals per complex element in
"1e" format versus two in planar). That cost is real but is not on the critical
path at these shapes, and the intensity advantage swamps it.

### What this means for the project

The gap the project set out to exploit **exists in the wild today** — anyone
using the packaged, released TBLIS for complex tensor contraction on modern x86
is getting roughly a fifth of the achievable throughput. But:

* it is a missing-kernel bug, not an algorithmic opening, so beating it proves
  nothing about planar packing;
* it is already closed upstream, and will disappear from the wild the moment
  2.0 ships;
* the correct opponent for any new complex method is 2.0/BLIS 1m, and against
  that opponent there is no complex-specific headroom to take.

So the planar-complex thesis is refuted as a *research* proposition, while the
practical observation that motivated it is validated as a *packaging* problem.
Both halves are worth reporting.

### What the data says the real headroom is

Not complex — **low arithmetic intensity**, in either domain:

* TBLIS `eff` against the GEMM ceiling ranges from **0.34 to 1.05**. It is
  0.85–0.87 on the big compute-bound `ijkl` cases and collapses to 0.34–0.53 on
  small-`k` / skinny shapes (`abjcd-dkbac-jk`, `ajbc-ckba-jk`,
  `abcijk-*`, all with `k = 24`).
* The gap is worse in **f32** (mean `eff` ≈ 0.6) than f64, because the same
  overhead is amortised over half the bytes of arithmetic.
* TTGT is 2–4x behind TBLIS on those same low-intensity shapes (`eff` 0.15–0.35),
  confirming that materialising a transposed copy is what hurts — the original
  BSMTC insight, still valid.

So the defensible target is **small-`k` and skinny tensor contractions**, where
the best available transpose-free engine leaves 50–65% of the machine on the
table, in *both* domains. That is a larger and better-evidenced gap than the
one the project set out to close.

### Kill/pivot condition

`DESIGN.md` §6 named this as the single most likely failure mode, and the
Phase 1 gate exists precisely to catch it before implementation is committed
to. Per the operating rules, this is escalated rather than worked around.
Options, with the evidence for each:

1. **Re-aim at low arithmetic intensity** (recommended). Keep everything built:
   the data model, index analysis, block-scatter machinery, TAPP surface,
   corpus and harness are all domain-agnostic and all still needed. Change the
   target from "complex vs real" to "small-`k` / skinny shapes", where TBLIS
   measurably gives up 50–65%. Plausible mechanisms, in order of expected
   value: fusing the `pc` loop so `C` is touched once instead of `K/KC` times;
   skipping packing of `A` entirely when the block-scatter is already regular
   and unit-stride (a "pack-free" fast path); dispatching to a
   small-`k`-specialised kernel; and the write-back fast path for regular
   blocks. Planar complex stays in the design because it is *free* and it is
   what makes `TAPP_CONJUGATE`, mixed real x complex operands and 3m natural —
   it is simply no longer the headline claim.
2. **Pursue 3m instead.** Untouched by this result: 3m's advantage is a 25%
   *flop* reduction, not a bandwidth one, and planar packing makes it cheap to
   build. Smaller, more speculative, and carries a numerical-stability caveat.
3. **Wrap TBLIS.** Honest answer if the goal is a usable Rust tensor
   contraction today, but no research contribution, and it keeps the C++
   dependency the brief wanted to remove.
4. **Stop.** The negative result is itself publishable, and the brief says so:
   there is no public systematic complex tensor-contraction benchmark, this
   repository now is one, and "complex contraction is not the weak spot; low
   arithmetic intensity is, and here is why" is a useful correction to
   circulating folklore.

**Recommendation: option 1**, with the Phase 1 negative result written up as a
standalone finding.

---

## Phase 2 report

**Gate:** numerically correct across the full matrix (shapes, permutations,
dtypes, traces, degenerate cases) vs oracle, TTGT, TBLIS. Performance measured
as a baseline, not a goal.

**Status: gate met.** Phase 2 was completed alongside Phase 1 because the
premise check needed a working engine to sit alongside the baselines.

Implemented: tensor data model; index analysis with folding; scatter and
block-scatter construction; planar-complex packing with conjugation folded in;
reference scalar micro-kernel; five-loop driver; scattered write-back with
`alpha`/`beta`/`op_C`/`op_D`; TAPP C-ABI export.

Correctness evidence:

* 1000 randomised contractions vs the brute-force oracle across
  `f32`/`f64`/`c32`/`c64`, each run under both tiny `(1,2,1)` blocking and the
  real blocking, covering free/contracted/Hadamard/isolated indices, repeated
  labels, random stride permutations, random conjugation masks, and
  `alpha`/`beta` including zero — all within `1e-11` (f64) / `2e-4` (f32).
* Targeted degenerate cases: empty contraction extent, zero-sized output,
  scalar output (full double contraction), negative strides via a reversed
  axis, all 16 conjugation flag combinations forced through multiple `K` blocks.
* Large pure-GEMM cases crossing the real `MC`/`KC`/`NC` boundaries with
  awkward remainders, in all four dtypes.
* Cross-implementation: all 49 corpus cases x 4 dtypes agree with **both** TBLIS
  and TTGT to `~2e-16` (f64/c64) and `~1.5e-7` (f32/c32), under `none`,
  `ragged` and `padded` stride stress.
* TAPP C ABI exercised end-to-end through the C entry points on a complex case.

Performance baseline: the micro-kernels are the portable scalar fallback
(Phase 3 was not reached), so the `planar` engine's absolute numbers are not
meaningful yet and are not reported as a result.

---

## Phase 2b report: three interchangeable complex methods

**Direction decision.** After the Phase 1 result, the chosen direction is to
keep all three induced-complex methods available and switchable, so that the
comparison can be made properly rather than argued from first principles. This
supersedes the four options listed at the end of the Phase 1 report.

### What was built

`ComplexMethod::{Planar, OneM, ThreeM}`, selected per plan with
`Plan::with_complex_method` or globally with `TENSORCONTRACT_COMPLEX`.

The three share the *entire* engine except three things, each named in the
`Ukr` the method selects:

| | `a_pack` / `b_pack` | kernel | `tile_fmt` |
|---|---|---|---|
| planar | `Planar` / `Planar` | fused complex, 4 FMAs per k per output | `Planar` |
| 1m | `OneE` / `Planar` | plain real, `2*MR x NR` over `2*KC` | `OneM` |
| 3m | `ThreeM` / `ThreeM` | Karatsuba, 3 FMAs per k per output | `ThreeM` |

`crates/tensorcontract/src/driver.rs` contains no branch on the method at all —
it reads sliver widths, tile size and formats off the `Ukr`. That is what makes
the comparison fair: same index analysis, same scatter traversal, same loop
arithmetic, same write-back scatter.

| # | Decision | Rationale |
|---|---|---|
| D13 | `Blocking::derive` takes reals-per-element for A and B rather than `size_of::<Element>()`. | 1m's packed A carries four reals per complex element instead of two, so an element-size-based rule would hand it double the L2 footprint and quietly rig the comparison. Deriving from the actual packed footprint gives every method the same L2 budget and gives 1m a proportionally smaller `MC`. |
| D14 | B's "1r" packing under 1m is bit-identical to planar packing, and shares the code path. | Not a shortcut: `[re_0..re_{NR-1}, im_0..im_{NR-1}]` per logical k-step *is* both formats. Worth stating because it means 1m's cost over planar is entirely on the A side. |
| D15 | 3m gets a 100x looser test tolerance than the other two. | Its error bound is relative to `\|Ar\|\|Br\| + \|Ai\|\|Bi\|` rather than the complex magnitudes, so it loses relative accuracy under cancellation. That is the documented price of the 25% flop saving; the tolerance records it rather than hiding it. |
| D16 | Kernels take the *logical* `kc` and know their own panel layout. | 1m internally runs `2*kc` real steps. Exposing that to the driver would leak the method into the loop nest. |

### Correctness

* The randomised oracle sweep now runs **every complex problem under all three
  methods** — 300 problems x 3 methods x 2 blockings for `c64`, likewise
  `c32` — plus the all-16-conjugation-masks test and the large blocking-boundary
  cases, each across all three.
* `kernel::tests` checks each method's kernel directly against the definition,
  including a packing helper that builds panels in each `PackFormat` and a
  reader for each `TileFormat`, so a mis-specified format is caught at the
  kernel boundary rather than end to end.
* All 49 corpus cases agree with **both** TBLIS and TTGT to ~2e-16 under each
  of the three methods (`TENSORCONTRACT_COMPLEX=planar|1m|3m tcbench verify`).

### Measurement, and why it does not yet mean much

`tcbench premise --size 16 --engines planar,1m,3m --dtype f64,c64`, 12 cases,
single core. Geometric-mean `c64` throughput relative to planar:

| method | relative c64 GF/s | mean complex/real efficiency ratio |
|---|---|---|
| planar | 1.000 | 0.726 |
| 1m | 1.086 | 0.783 |
| 3m | 1.129 | 0.817 |

**This ranking is an artifact of the scalar kernels and must not be quoted as
a result.** All three currently run portable scalar loops, and the numbers
mostly reflect how well LLVM auto-vectorises three different loop shapes: 1m's
inner loop is a plain real GEMM kernel, which LLVM handles best, while planar's
entire argument is *fewer shuffles in a hand-written SIMD kernel* — which does
not exist yet. 3m's edge is more likely real, since a 25% flop reduction
survives any kernel quality, but even that needs confirming.

The honest three-way comparison is the Phase 3 gate.

Raw data: `bench-results/methods-f64c64.csv`.

---

## Phase 3 report: vectorised micro-kernels

**Gate:** correctness unchanged; single-core throughput against the baselines
and a GEMM roofline; **an honest three-way planar/1m/3m comparison with real
kernels.** All three met. This is the measurement the project exists to
produce.

### What was built

`crates/tensorcontract/src/kernel/x86.rs`, previously four `None`s, now holds
AVX-512 kernels for all four shapes — `real`, `planar`, `onem`, `threem` —
macro-generated over `(MV, NR)` const generics for both `f32` and `f64`,
selected by runtime `avx512f` detection with the scalar path untouched behind
`TENSORCONTRACT_KERNEL=scalar`. `MV` is the number of vector registers an `A`
sliver occupies per plane per k-step.

**Nothing outside that file changed.** The `Ukr` contract carried the new
kernels unmodified, which is the design claim from Phase 2 discharged.

Also added: `examples/kernel_shapes`, a register-block sweep used to choose the
shapes (D19), and `scripts/phase3-bench.sh`, which reproduces this entire
report from a clean checkout given the two TBLIS prefixes.

### Correctness

Unchanged, and checked at three levels:

* `kernel::tests` validates each selected kernel directly against the
  mathematical definition through its own `PackFormat`/`TileFormat`. Every
  kernel here passed on first execution.
* `cargo test --workspace --release` green, and green again under
  `TENSORCONTRACT_KERNEL=scalar`.
* `tcbench verify` — all 49 corpus cases x `f32`/`f64`/`c32`/`c64` x all three
  complex methods, against **both** TBLIS 2.0-dev and TTGT. `f32`/`f64` agree
  exactly (`0.0e0`); `c32` to ~1.3e-7, `c64` to ~2.5e-16.

### The three-way comparison

Full 49-case TCCG corpus, 64 MiB nominal tensors, single core, geometric mean
GF/s. `tblis` is 2.0-dev at `555320c`. Unperturbed corpus, so `regA = 1.00`
throughout and the gather path is not involved.

| engine | c64 | vs planar | c32 | vs planar |
|---|---|---|---|---|
| **planar** | **43.6** | 1.000 | **79.9** | 1.000 |
| 1m | 42.2 | 0.967 | 78.2 | 0.979 |
| 3m | 41.7 | 0.956 | 73.6 | 0.921 |
| tblis 2.0-dev | 44.1 | 1.011 | 71.2 | 0.891 |
| ttgt | 23.2 | 0.532 | 45.3 | 0.567 |

**Planar wins, in both precisions, but by 3–8% rather than by a lot.** The
scalar-kernel ranking of Phase 2b (planar 1.00, 1m 1.09, 3m 1.13) is now
reversed, exactly as that report predicted it would be once the ranking stopped
measuring LLVM's auto-vectoriser.

**But the aggregate hides the actual finding, which is that the ranking is
shape-dependent and inverts.** Splitting the same corpus by arithmetic
intensity:

| c64 subset | planar | 1m | 3m | tblis |
|---|---|---|---|---|
| `min(n,k) > 64` — compute-bound, 25 cases | **69.0** | 65.4 | 62.2 | 71.3 |
| `min(n,k) <= 64` — memory-bound, 24 cases | 27.0 | 26.7 | **27.4** | 26.7 |

and the micro-kernel sweep says why. Timing each kernel in isolation while
sweeping `kc`, which decides whether the `A` sliver is an L1 resident or an L2
stream (`bench-results/phase3-kernel-shapes.txt`, `f64`):

| method | best shape | GF/s at `kc=64` | GF/s at `kc=256` | bytes / useful flop |
|---|---|---|---|---|
| planar | `16x6` | 107.7 | **102.8** | **0.46** |
| 1m | `12x8` | 93.2 | 91.1 | 0.67 |
| 3m | `8x10` | **117.1** | 87.8 | 0.68 |

So:

1. **3m's 25% flop saving is real and it is not free.** With both panels
   L1-resident 3m is the fastest of the three, by roughly the margin the flop
   count predicts. At the `kc = 256` the engine actually uses, it is the
   slowest. 3m loads *three* planes of both operands to save one of four
   products, so per useful flop it moves 1.5x planar's bytes; once the kernel
   stops being FMA-issue-bound that is what decides it.
2. **Planar wins on bytes, not on shuffles.** Its advantage over 1m is that
   "1e" packing carries four reals per complex element of `A` against planar's
   two. The original argument for planar — fewer in-register shuffles — is not
   what the data rewards, because at these shapes none of the three methods is
   shuffle-limited: LLVM emits `vbroadcastsd` as its own uop and every winning
   shape is FMA-issue-bound (verified in the disassembly).
3. **When the contraction is memory-bound the kernel's byte traffic stops
   mattering and its flop count starts to.** That is the inversion above, and
   it is a direct argument for the shape-dispatch item already listed in
   Phase 4.

Every candidate needing more than 32 live vector registers loses 30–50%. That
cliff, not the flop count, is what bounds 3m's usable shapes: it needs three
accumulator planes, so it cannot be given a block wide enough to amortise its
loads.

### Against the baselines

Full corpus, 64 MiB, geometric mean GF/s, `planar` for complex:

| dtype | this engine | tblis 2.0-dev | ttgt | best-of-49 count (ours / tblis / ttgt) |
|---|---|---|---|---|
| f64 | **30.5** | 27.0 | 13.1 | 22 / 16 / 11 |
| f32 | **54.6** | 42.7 | 24.4 | 23 / 13 / 13 |
| c64 | 43.6 | **44.1** | 23.2 | 13 / 12 / 12 |
| c32 | **79.9** | 71.2 | 45.3 | 10 / 11 / 16 |

Against **TBLIS v1.3.0** (`c4f81e0`, still the latest stable release — no new
tag has appeared since Phase 1), 12-case premise set at 200 MiB: `c64` planar
**43.8** vs **7.3** GF/s, a 6.0x gap, and `f64` 29.2 vs 21.5. The Phase 1
finding that 1.x has no complex micro-kernel outside Sandy Bridge reproduces
unchanged.

The rebuilt TBLIS 2.0-dev reproduces Phase 1's headline to three digits — mean
complex-over-real efficiency ratio **1.058** here against 1.06 in Phase 1 —
which is the check that the rebuilt baseline is the same baseline.

Complex-over-real efficiency ratio against a same-shape GEMM ceiling, 12-case
premise set at 200 MiB: planar 0.976 / 1m 0.931 / 3m 0.979 (`c64`), and 1.156 /
1.144 / 1.134 (`c32`). Against the roofline directly, the engine reaches
**0.81–0.84** of a same-shape `zgemm` on the large compute-bound cases and
**0.69–0.70** of `dgemm` — complex is the *easier* domain for us too, for the
arithmetic-intensity reason established in Phase 1.

### Irregular strides

`--stress ragged`, `c64`, 64 MiB. 40 of 49 cases become genuinely irregular
(mean observed `regA = 0.656`; the remaining 9 stay at 1.00). On those 40:

| engine | GF/s |
|---|---|
| planar | **43.7** |
| 3m | 42.5 |
| 1m | 42.3 |
| tblis 2.0-dev | 39.4 |

An 11% lead for the transpose-free path where strides are awkward — the one
regime where block-scatter is doing work a TTGT-style engine cannot avoid
paying for. Note these numbers are *not* comparable to the unstressed table
above: `--stress ragged` changes the extents, so it is a different set of
shapes, not the same shapes made harder.

### A concrete 2x defect, localised

Nine corpus cases of the form `abcijk-{ij,ik,jk}m{a,b,c}-*` have identical
`m`, `n`, `k` and `regA = regB = 1.00`, and differ only in which output axis
leads the `M` group. Throughput is monotone in that axis's stride in `D`,
9 cases out of 9:

| `M` leading axis | its stride in `D` | planar `c64` GF/s |
|---|---|---|
| `a` | 1 | 40.5, 40.5, 40.5 |
| `b` | `n_a` | 36.2, 36.4, 36.6 |
| `c` | `n_a * n_b` | 17.7, 17.8, 17.8 |

Packing is identical across the nine (same operand layouts, fully regular block
scatter), so the cost is in the **scattered write-back**. `perf stat` on three
of them, same binary, same reps, confirms it and identifies the mechanism:

| `M` leading axis | cycles | instructions | IPC | dTLB store misses | **L2 demand misses** |
|---|---|---|---|---|---|
| `a` (stride 1) | 5.19e9 | 6.50e9 | 1.25 | 4.11e6 | 27.5e6 |
| `b` (stride `n_a`) | 5.48e9 | 6.46e9 | 1.18 | 4.21e6 | 33.0e6 |
| `c` (stride `n_a*n_b`) | **8.32e9** | 6.49e9 | **0.78** | 4.07e6 | **62.9e6** |

Instruction count and retired stores are flat to within 1%, so this is purely
stalling, not extra work. **The first-guess mechanism was wrong**: DTLB misses
are flat, so it is not TLB pressure. It is L2 miss traffic — 2.3x more of it —
from poor cache-line utilisation and reuse distance on the `C`/`D` update when
consecutive tile rows are far apart in the output.

A controlled follow-up: rebuilding *planar alone* with 3m's wider `8x10` tile
instead of its own `16x6`, changing nothing else, moves these cases the way the
mechanism predicts and moves the others back:

| case | `16x6` | `8x10` | |
|---|---|---|---|
| `abcijk-ijmc-mkab` — `D` contiguous along `N` | 17.8 | **19.8** | +11% |
| `abcijk-ijmb-mkac` — intermediate | 36.2 | 33.8 | −7% |
| `abcijk-ijma-mkbc` — `D` contiguous along `M` | 40.5 | 35.9 | −11% |
| `ijkl-imjn-lnkm` — compute-bound control | 79.4 | 79.4 | 0% |

So micro-tile **aspect ratio should follow the output's stride pattern** — a
real lever, worth about ±11%, and free to apply since the kernels are already
parameterised over `(MV, NR)`. It also explains why 3m beats planar by 1.36x on
exactly these three cases while losing everywhere else.

But ±11% does not explain a 2x. Aspect ratio is a tuning knob; the 2x is the
write-back's L2 traffic itself, and closing it needs the "vectorised write-back
for regular blocks" item on the Phase 4 list — or blocking the `jr`/`ir` loops
against the output's layout rather than only against the packed panels.

### Assumptions added

| # | Assumption | Status |
|---|---|---|
| A7 | The three complex methods differ mainly in flop count. | **Refuted.** They differ mainly in bytes moved per useful flop, and that is what decides the ranking at realistic `kc`. Flop count decides it only when the panels are L1-resident or the contraction is memory-bound. |
| A8 | One register block per method is enough. | **Refuted, and it matters.** The best shape depends on the method, the element type and `kc`, and neighbouring shapes differ by 30–50% across the 32-register cliff. |
| A9 | Absolute performance is meaningful now that kernels are vectorised. | Adopted. It was explicitly not meaningful before this phase. |

### What is *not* done

* **No AVX2 path.** Dispatch is AVX-512 or the scalar fallback. The macro takes
  it without restructuring; deferred to Phase 5's multi-arch work, since the
  reference machine is AVX-512 and the gate is a comparison on it.
* **Blocking is still the Phase 2 heuristic.** `MC`/`KC`/`NC` come from a fixed
  cache-budget rule, never swept. Given that the whole Phase 3 result turns on
  where the `A` sliver lives, `KC` in particular is now known to be a
  first-order parameter rather than a detail. Phase 4.
* **Still single-threaded.**

Raw data: `bench-results/phase3-*.csv`, transcript in
`bench-results/phase3-log.txt`, kernel sweep in
`bench-results/phase3-kernel-shapes.txt`. Reproduce with
`scripts/phase3-bench.sh`.

## Phase 4 report, part 1: the write-back

Phase 3 left a profiled 2x defect on nine corpus cases and named two candidate
fixes. Both were built. The first one is not the one Phase 3 predicted.

### What was built

1. **Row/column orientation** (`Plan::transposes_gemm`, applied in `driver`).
   The engine is symmetric under exchanging `(A, M)` with `(B, N)`; doing so
   computes `D^T = B^T A^T`, which is the same contraction seen through the
   transposed matrix view of `C` and `D`. The driver binds `am`/`ak`/`ptr_a` to
   whichever operand plays the row role and nothing downstream branches again.
   For 1m this also swaps the two *different* pack formats ("1e" for rows, "1r"
   for columns), which is right: the kernel's contract is about the row and
   column panels, not about which user tensor they came from.
2. **A regular-block write-back** (`writeback`). The row offsets now come from
   the output's *block* scatter, exactly as packing takes the operands' — three
   instantiations of one inlined body: gather, strided, and unit-stride. Only
   the last lets LLVM vectorise the plane recombination, and it is the case the
   orientation rule exists to create. The `alpha == 1, beta == 0, no
   conjugation` combination gets its own inner loop, hoisted out of the `i`
   loop, because it reduces to a straight tile-to-`D` copy.

`TENSORCONTRACT_ORIENT=none|swap` pins the orientation for A/B measurement.

### The orientation rule, and the condition that is not obvious

Swap when, and only when:

1. `D`'s column direction is **strictly** more contiguous than its row
   direction (compared on the leading axis of each class, i.e. the smallest
   |stride| in `D`, since that is the axis the register block is cut along); and
2. the swap leaves the micro-tile's row block **unbroken** — the new leading
   axis has unit stride *and* extent at least `MR`.

Condition 2 was not anticipated and is the reason the first version of this
rule regressed. On the `abcijk-*mb-*` family the swap makes the leading axis
unit-stride but only 24 long, and the outcome is monotone in `run / MR`:

| dtype | kernel `MR` | run / `MR` | swap vs no swap |
|---|---|---|---|
| `c64` | 16 | 1.50 | **+18%** |
| `f64` | 24 | 1.00 | **+23%** |
| `c32` | 32 | 0.75 | 0% |
| `f32` | 48 | 0.50 | **−17%** |

A shattered row block keeps the swap's costs and loses its benefit. Note this
makes the choice element-type dependent through `MR`, so it is not a property
of the plan alone — `transposes_gemm` and `oriented_scatters` both take `MR`.

Those figures were taken on a shared workstation and were re-measured on an
exclusive one at 24 reps (see part 4). They hold: `f32` `-mb` reproduces at
0.862 / 0.836 / 0.848 across the three cases, against a per-case noise floor of
±6%, and the monotone ordering is intact.

**The −17% is not fully explained and should not be presented as if it were.**
`perf stat` on the `f32` case shows instructions flat to 1%, dTLB misses flat,
and L1 load misses *down* 27% under the swap, yet 4% more cycles at IPC 1.05 →
1.00. Counting cache lines per micro-tile also favours the swap (24 fully
written lines against 48 half-written ones), as does the packing pattern. So
the rule's condition 2 is an empirical guard rail, not a derived one.

### Result: the 2x defect is closed, with no case left slower

Full 49-case corpus, `--size 64 --reps 3`, single core, against the *recorded*
Phase 3 sweep (`bench-results/phase3-sweep-*.csv`) — same basis, same machine.
Cases the rule does not swap are a control group: the change cannot affect
them, so their spread is a direct read of run-to-run noise.

| | `f32` | `f64` | `c32` planar | `c64` planar |
|---|---|---|---|---|
| geomean, all 49 | **1.047** | **1.105** | **1.046** | **1.102** |
| geomean, swapped cases | 1.620 | 1.700 | 1.485 | 1.801 |
| geomean, control group | 0.985 | 1.040 | 0.996 | 1.029 |
| worst swapped case | 1.493 | 1.315 | 1.094 | 1.153 |

The nine cases Phase 3 profiled, `c64` planar, all three `M`-leading strides:

| `M` leading axis | its stride in `D` | Phase 3 | now |
|---|---|---|---|
| `a` | 1 | 40.5 | 40.9 |
| `b` | `n_a` | 36.2 | 44.1 |
| `c` | `n_a * n_b` | 17.7 | **47.2** |

Throughput is no longer monotone in that stride — it is flat to ±8%, and the
case that was 2.3x slower than its sibling is now the fastest of the three.
The `-mc` family gains **2.6–2.7x** in `c64` planar.

A first version of the rule without condition 2 swapped 12 of 49 cases and
scored 1.07–1.19 geomean *with* a −15% worst case; the guarded rule swaps 6–12
depending on element type, scores slightly lower on paper, and has no
regression beyond noise. Raw data for both: `bench-results/phase4/orient-*.csv`.

### Part 2: the regular-block write-back, on top of the orientation fix

First measured as a diff between two builds half an hour apart, which gave
`c64` 3m = 0.973 and the conclusion "a wash in double". Both were wrong. With
`TENSORCONTRACT_WRITEBACK=gather` the comparison became a runtime A/B inside one
session (`rm-B-*.csv` → `rm-A-*.csv`), and it is positive everywhere:

| | planar | 1m | 3m |
|---|---|---|---|
| `f32` / `f64` (one real path) | 1.071 / 1.019 | — | — |
| `c32` | 1.104 | 1.121 | **1.153** |
| `c64` | 1.032 | 1.017 | 1.020 |

Every entry clears the ±1.3% geomean noise floor. It pays about 5x more in
single precision than double, which is the ratio to expect: the per-output
cost it removes is fixed, while the kernel work it hides behind scales with the
element size, so halving the element doubles the relative overhead.

The lesson is the methodological one. Nothing about the code changed between
the two measurements — only that the arms were interleaved instead of
sequential. A build-to-build diff cannot distinguish a 3% effect from drift;
put the switch behind an environment variable and the same question answers
itself in one run.

### Cumulative Phase 4.1 result

Full corpus, against the recorded Phase 3 sweep, exclusive machine
(`rm-A-*.csv`), against a ±1.3% geomean noise floor:

| dtype | planar | 1m | 3m |
|---|---|---|---|
| `f32` | **1.142** | — | — |
| `f64` | **1.127** | — | — |
| `c32` | 1.116 | 1.122 | **1.167** |
| `c64` | **1.118** | 1.058 | 1.048 |

And the nine cases Phase 3 profiled as a 2x defect, planar, GF/s, averaged over
the three cases sharing each `M`-leading axis:

| `M` leading axis | `c64` | `f64` | `c32` | `f32` |
|---|---|---|---|---|
| `a` (stride 1) | 40.5 → 43.5 | 32.4 → 35.5 | 82.8 → 85.8 | 52.8 → 52.3 |
| `b` (stride `n_a`) | 36.4 → 44.9 | 25.0 → 31.9 | 63.3 → 67.3 | 40.6 → 40.0 |
| `c` (stride `n_a n_b`) | 17.8 → **49.0** | 13.8 → **27.9** | 40.1 → **87.0** | 27.0 → **60.1** |

The 2x defect is gone in every precision — 2.75x, 2.02x, 2.17x, 2.23x on the
affected cases — and the ordering has inverted: what was the slowest of the
three is now the fastest. The 4–5% loss on the `f32` `a`/`b` rows reported
before the machine was quiet does **not** survive re-measurement; those rows
are flat to within noise, and the paragraph explaining them as a write-back
side effect was explaining contention.

### Part 3: a negative result on depth-adaptive `MC` — do not redo this

The obvious first move on item 2 looked free and is not. `Blocking::derive`
sizes `MC` and `NC` so a `KC`-deep packed block fits its cache budget, but a
third of the corpus contracts over `k = 24` against a `KC` of 256. Those cases
pack an `A` block a tenth the size of its L2 budget, and the only consequence
is that `B` is re-streamed `M/MC` times for nothing. Re-deriving against
`min(k, KC)` — widening `MC` about tenfold — should be pure profit.

It is not. Unconditionally, `abcijk-jkm*` planar:

| case | `f32` | `f64` | `c32` | `c64` |
|---|---|---|---|---|
| `-ma` | 50.6 → **32.5** | 35.5 → 30.1 | 83.1 → **61.2** | 42.9 → 43.2 |
| `-mb` | 39.4 → **24.3** | 32.0 → 29.8 | 65.0 → **53.2** | 45.0 → 43.4 |
| `-mc` | 56.3 → 56.3 | 27.6 → **31.7** | 93.5 → **104.1** | 48.6 → **53.4** |

The mechanism for the losses is visible: `MC` also bounds the strip of `D` that
one `jr` pass touches and the *next* pass revisits. When the output's rows are
strided, that strip is `MC` distinct cache lines, and a tenfold `MC` takes it
from 24 KiB (L1-resident) to 350 KiB. The packed-`A` budget does not model this
at all.

Gating the widening on "every `D` row block is unit-stride" — where the strip
is `NR` sequential runs, streamed and written once — removes the `f32`/`c32`
losses, but a clean A/B on `f64` (all three cases unit-stride, pinned blocking
vs adaptive, 15 reps) shows the rule still does not hold:

| case | pinned `mc=256 kc=256 nc=1536` | depth-adaptive |
|---|---|---|
| `-ma` | 37.1 | 30.5 (**−18%**) |
| `-mb` | 32.9 | 29.9 (−9%) |
| `-mc` | 28.4 | 32.0 (**+13%**) |

Three cases, one sign each way, no rule. **Backed out.** `Blocking::derive_at_depth`
is kept because the sweep needs to vary `kc` and get budget-consistent `mc`/`nc`
with it, but the driver does not call it. The lesson for item 2 is that `MC` is
a two-sided constraint — packed-`A` residency below, output-strip residency
above — and the sweep has to be designed to separate them rather than to find a
single best `MC`.

### Part 4: what an exclusive machine changed, and the rule's 9 known misses

Everything above was first measured while other work shared the workstation.
The harness pins to one logical CPU, but nothing stopped a co-tenant landing on
its **hyperthread sibling**, which shares the 32 KiB L1d and 1 MiB L2 — the
exact resources all of this is about. `scripts/phase4-remeasure.sh` redid it
with exclusive access, running `A, B, A'` so the identical repeat brackets the
treatment, and recording `/proc/stat` occupancy for the pinned core and its
sibling next to every result (`*.cpu`: cpu4 at 100%, cpu20 at 0.5–1.0%
throughout).

**Measured noise floor**, A vs A′ at the reps the corpus sweeps use:

| | geomean over 49 cases | per case |
|---|---|---|
| spread | 0.992 – 1.013 (**±1.3%**) | 0.92 – 1.06 (**±6%**) |

That is tighter than the ±3% / ±13% inferred earlier from the contended control
group, so the corpus geomeans stand and the per-case claims get a real error
bar. **Quote these, not a control-group inference.**

Then, since the orientation A/B forced both arms on all 18 `abcijk` cases in
all four dtypes at 24 reps, it also grades the rule itself. **The rule picks
the better arm, or ties within noise, on 63 of 72 case-dtypes.** The 9 misses
are large, systematic, and all 32-bit:

| case family | dtype | rule | better | left on the table |
|---|---|---|---|---|
| `abcijk-e{i,j,k}bc-*` | `f32` | AB | BA | **1.43–1.47x** |
| `abcijk-e{i,j,k}bc-*` | `c32` | AB | BA | **1.35–1.36x** |
| `abcijk-e{i,j,k}ac-*` | `f32` | AB | BA | 1.18x |

These are unrealised gains, not regressions — the rule picks the pre-Phase-4
behaviour there — but 1.4x on six corpus cases is larger than anything else
left on the Phase 4 list.

**Condition 2 is a proxy, and these show it is the wrong one.** The `e*bc`
cases have the *same* post-swap row structure as the `-mb` family that
condition 2 correctly rejects — leading axis unit-stride, extent 24, against an
`MR` of 32 or 48 — and yet swapping gains 1.4x where `-mb` loses 15%. So
`run / MR` cannot be the discriminant; it merely correlates on the cases it was
derived from. The `e*ac` row is worse still: there the rule declines at
condition *1*, and in `f32` swapping to the **less** contiguous row direction
wins by 1.18x while in `c64` it loses 18% — opposite signs for the same shape
in different precisions.

Two structural differences are visible but neither has been tested: after the
swap, `e*bc`'s column direction folds to a run of 576 in `D` where `-mb`'s is
24; and `e*bc`'s row operand packs *less* regularly than `-mb`'s, i.e. the
faster arm is the one with worse block-scatter regularity. Do not adopt either
as a rule without measuring it.

The useful asset from this is the data: `rm-orient-{none,swap}.csv` hold both
arms for all 72 case-dtypes, so a candidate discriminant can be scored offline
against ground truth without running a single new benchmark. Failing that, the
orientation is one cheap binary choice per plan and plans are reusable, which
makes empirical selection — run both once, keep the faster — the honest
fallback.

### Where the next lever is, and why it is now sharper

The `f32` `a`/`b` rows above gained least — and they are the ones still on the
write-back's **gather** path, for a structural reason rather than a measured
regression: their leading axis in `D` has unit stride but extent 24, while
the `f32` real kernel's `MR` is 48. A 48-row block therefore straddles a
discontinuity and the block scatter reports it irregular — the same quantity
that condition 2 of the orientation rule turns on. Choosing `MR` to divide the
output's leading contiguous run would convert those blocks to the unit-stride
path outright. That reframes the Phase 3 "micro-tile aspect ratio should follow
the output's stride pattern" item from a ±11% tuning knob into a way of
reaching a qualitatively faster code path, and it is cheap because the kernels
are already parameterised over `(MV, NR)`.

Note this is the *same quantity* — `run` against `MR` — that part 4 shows is
not the true discriminant for the orientation. The two items are therefore
coupled: `MR` is simultaneously a free parameter of the aspect-ratio choice and
an input to the orientation rule, and every one of the nine orientation misses
is a 32-bit case where `MR` (32 or 48) exceeds the run (24). Changing `MR` to
16 for those shapes would satisfy condition 2 and make the rule pick BA without
any new discriminant. **Do 1c before trying to fix the orientation rule** — it
may dissolve the problem rather than require solving it.

> **Superseded by parts 5 and 6.** It did not dissolve it: shrinking `MR` does
> flip those cases to `BA`, but they gain 1.17–1.39x *while paying ~30% in
> kernel shape*, which prices the orientation at ~2x and makes `MR` the wrong
> instrument. The discriminant was found in part 6 and is unrelated to
> `run / MR`: the rule has to be **antisymmetric** under exchanging the two
> directions. Read part 6 before acting on anything in this section.

### Part 5: item 1c, the micro-tile row block — and what it prices

Part 4 predicted that choosing `MR` to divide the output's contiguous run would
"dissolve" the orientation problem. It does not. It **prices** it, which is more
useful, and the write-back gain it was aimed at is real but small and needs
three guards to be positive at all.

#### What was built

`kernel::x86` now builds each method from a **menu** of register blocks rather
than one, default first, using the same const-generic kernels Phase 3 wrote.
The alternates were priced by re-running `examples/kernel_shapes`, extended with
the `MV = 1` real and 1m cases it never covered
(`bench-results/phase4c/kernel-shapes.txt`). `Plan::row_block` chooses from the
menu, `Plan::row_block_score` reports the fraction of output row blocks that
would stay off the gather path at a given `MR` — evaluated in the orientation
*that* `MR` selects, since the two are coupled — and
`TENSORCONTRACT_ROWBLOCK=base|auto|mr=<n>|idx=<i>` makes every arm reachable at
run time. `tcbench shapes` scores every shape against every corpus case without
running anything, which is how the measurements below were chosen.

#### The grid, and why it was worth 2 h

`scripts/phase4c-rowblock.sh` pins *every* shape on *every* menu across the
whole corpus (`bench-results/phase4c/rb-*`, sibling CPU 0.6–1.3% throughout).
That is worth far more than an A/B of the rule, because it splits each
(dtype, method, shape) into cases where the shape changes nothing the
write-back can see — their ratio is the shape's own cost — and cases where it
does. Any candidate rule can then be scored offline against ground truth, which
is what `scripts/rowblock-score-rules.py` does:

| rule | `f32` | `c32` planar | `c32` 1m | `c32` 3m | `c64` planar |
|---|---|---|---|---|---|
| maximise the regular fraction | **0.936** | 1.006 | 1.026 | 0.987 | 1.040 |
| + only where `k <= 32` | 1.006 | 1.018 | 1.027 | 0.987 | 1.034 |
| + only reaching *full* regularity | 0.997 | 1.004 | 1.027 | 0.987 | 1.034 |
| + only if the orientation is unchanged | 0.997 | 1.004 | 1.015 | **1.000** | 1.034 |
| oracle, best shape per case with hindsight | 1.042 | 1.051 | 1.068 | 1.033 | 1.054 |

The obvious rule — the one part 4 proposed — is a **loss**. Each guard is
measured, not argued:

1. **`k <= 32`.** The write-back costs a constant per output element against
   `~4k` flops of kernel work, so which path it takes only matters while `k` is
   small, and a shape off the kernel's peak always costs something. At `k = 24`
   the winning shape gains 1.09–1.26x; at `k >= 204` the identical change is
   1.01–1.04x and still being paid for. The corpus jumps from `k = 24` to
   `k = 52`, so it resolves this boundary only to somewhere in `(24, 52]`.
2. **The default must be substantially broken** (regular fraction `<= 0.75`).
   Taking `f32` `48x8 -> 32x8`, where the default was already 0.88 regular, lost
   7–9%. The corpus only produces the values 0, 0.67, 0.88 and 1.0, so any
   threshold in `(0.67, 0.88]` fits it equally.
3. **The orientation must not change.** `MR` is an input to
   `transposes_gemm`, so a shape change can silently flip it. Without this
   guard `c32` 3m moves the three `abcijk-e*bc-*` cases from the good arm to
   the bad one and loses **19%**.

#### Result

Validation A/B (`bench-results/phase4c/v-*`, `base, auto, base'`, exclusive
machine, sibling CPU 0.7–0.9%). Session noise floor from the bracketing repeat:
geomean 0.992–1.008.

| | corpus geomean | fired cases | n | untouched cases |
|---|---|---|---|---|
| `c64` planar | **1.026** | **1.124** | 12 | 0.996 |
| `c32` 1m | 1.006 | **1.108** | 7 | 0.990 |
| `c32` planar | 1.005 | 0.977 | 1 | 1.005 |
| everything else | 0.992 – 1.000 | — | 0 | 0.992 – 1.000 |

The rule fires on **20 of 392** case-dtype-methods. Per case it runs 1.065 to
1.202 on 19 of them — `abcijk-ijma-mkbc` in `c64` planar goes 43.2 → 52.0 GF/s
— and 0.931 on the twentieth, `ajbdc-ckbad-jk` in `c32` 1m, a case that runs at
8.5 GF/s. That is the one case left slower, and it is only just outside the
per-case noise floor. The untouched cases are a control group of 372 and sit
inside the noise floor in every dtype and method.

`f32` and `f64` are untouched by construction: at `L = 16` lanes no full-width
`MR` divides the corpus's runs of 24 except 1m's, and `f64`'s default `MR = 24`
already tiles them. The `f32` gather-path cases part 4 pointed at are therefore
**not** reachable this way — see below.

#### What this prices: the orientation is worth ~2x, and `MR` is a bad way to buy it

Part 4's hope was that shrinking `MR` to 16 would satisfy the orientation rule's
condition 2 on the nine misses and make it pick BA "without any new
discriminant". The flip does happen — the grid confirms `abcijk-e*bc-*` in
`f32`/`c32` switches to BA at `MR = 16`. But:

| | `f32` real, `48x8 -> 16x10` | `c32` planar, `32x6 -> 16x12` |
|---|---|---|
| cases where nothing else changes | **0.70** | 0.96 – 0.99 |
| the six `e*bc` / `e*ac` cases | **1.17 – 1.39** | 1.15 – 1.29 |

Those cases gain 1.17–1.39x *while paying about 30% in kernel shape*, so the
orientation there is worth roughly **2x** on its own. Buying it through a shape
change is a bad trade, and the rule above correctly declines to (guard 3, plus
guard 2 for `f32`). **The conclusion is the opposite of part 4's prediction:
1c does not dissolve the orientation problem, it shows the orientation is the
larger lever and must be attacked directly, at the default `MR`.**

The oracle row above is the other half of the picture: picking the best shape
per case with hindsight scores only 1.03–1.07 anywhere. The shape lever is close
to exhausted. The orientation lever is not.

#### Two effects seen, deliberately not shipped

The grid also shows, at `k <= 24` and with no write-back change at all:

* `c64` 3m gains **1.088** from a *wider* `MR` (16 or 24 against its default 8);
* `c32` 1m gains **1.099** from `24x8` against its Phase 3 default `32x6`.

Both are register blocks that are simply better at small `k` than the ones
chosen at `kc = 256/384`, which is a *shape-by-depth* effect and belongs with
item 2's `MC`/`KC`/`NC` sweep. Neither has a mechanism yet and neither has been
validated on anything but this grid, so neither ships. Note the second one means
the Phase 3 register-block table is depth-conditional, not wrong.

## Phase 4 report, part 6 (item 1d): the orientation rule, and its discriminant

Part 5 named this the biggest measured lever and priced it at ~2x on the nine
known misses. A14 recorded the discriminant as unknown. **It is now found**, and
the answer is that the old rule was not wrong so much as *incomplete*: it had a
veto with no fallback, and every one of its misses lived in the case the veto
sent to the default.

### The grid, again

`scripts/phase4d-orient.sh` forces both arms — `TENSORCONTRACT_ORIENT=none|swap`
— over the whole corpus in every dtype and method, with
`TENSORCONTRACT_ROWBLOCK=base` pinning the shape so the two rules cannot
confound each other (`bench-results/phase4d/or-*`, sibling CPU 0.6–1.1%). That
is both arms of all 392 case-dtype-methods, against the 72 single-family points
that were the entire basis before. `tcbench orient` then dumps the structural
features of each arm, and `scripts/orient-score-rules.py` scores candidates
against ground truth, free and unlimited.

### The discriminant: the rule has to be antisymmetric, and the old one was not

The `abcijk` families are **exact mirror images** of one another — the same
output structure with the roles of the two directions exchanged. Any correct
rule must therefore be antisymmetric under that exchange. The old rule was
phrased entirely in terms of the *column* direction's properties, so it could
not be, and that is the whole defect. Written symmetrically, `f32` says:

| family | faster arm | its row run | its column run |
|---|---|---|---|
| `-mb` | `AB` | 16 | 24 |
| `e*ac` | `BA` | 16 | 24 |
| `e*bc` | `BA` | 24 | 4096 |
| `-ma` | `AB` | 24 | 256 |

In every one, the faster arm is the one whose **row direction has the shorter
run**. So condition 2 is promoted from a veto to a preference, with a defined
fallback:

1. Prefer an arm whose micro-tile row block lands inside a single run — unit
   stride, run at least `MR`. One arm: take it. Both: stay put.
2. Otherwise put the shorter-run direction in the row role.

Step 1 still dominates, and must: it is why `c64` (`MR = 16` against a run of
24) takes the *opposite* arm from `f32` (`MR = 48`) on identical shapes.

Scored over both arms of all 392: **12 cases better beyond noise, none worse.**
In `f32` it scores 1.128 against never swapping where the hindsight oracle
scores 1.132 — that dtype's orientation question is essentially closed.

| rule | `f32` | `c32` planar | `c32` 1m | `c64` planar |
|---|---|---|---|---|
| Phase 4.1 (`legacy`) | 1.092 | 1.063 | 1.049 | 1.102 |
| **fits, else shorter run** | **1.128** | **1.082** | 1.067 | 1.102 |
| oracle (hindsight) | 1.132 | 1.093 | 1.088 | 1.118 |

### Two part-4 hypotheses, tested and refuted

Part 4 offered two structural differences as possible discriminants and warned
against adopting either without measuring. Both are now measured, and both are
wrong:

* **"the column direction folds to a longer run"** — as a rule on its own it
  scores 0.976–1.008, i.e. nothing; as a guard on top of the working rule it
  *lowers* every column (1.078 against 1.128 in `f32`).
* **"maximise write-back regularity"** — 0.936 in `f32`. This is the third time
  that quantity has pointed the wrong way (see A16); it explains the write-back
  path and nothing else.

### The two rules are coupled, and tuning them separately is wrong

The end-to-end A/B of the new rule in the shipped configuration exposed
something the grid could not, because the grid pinned the shape: three `c32` 1m
cases went **0.80**. The 2x2 explains it exactly:

| `abcijk-e*ac-*`, `c32` 1m | `32x6` (default) | `24x8` |
|---|---|---|
| `AB` | 77–79 GF/s | **93 GF/s** |
| `BA` | 75–76 GF/s | — |

At the default shape the new rule prefers `BA` — a 3% error, inside the
per-case noise floor, which is why "0 worse" did not flag it. But the row-block
rule's guard 3 forbade a shape change that flips the orientation, so choosing
`BA` also **blocked** the `24x8` shape that is worth 1.20x. A 3% mistake was
amplified twentyfold by the interaction.

The fix is to remove guard 3, which the orientation fix has made obsolete: the
shapes it existed to veto (`c32` 3m's `e*bc`) are now rejected by guard 2
anyway, because under the corrected orientation their alternates no longer
reach full regularity. Measured directly — 6 newly-firing cases against 12
control cases in the same run:

| | geomean | range |
|---|---|---|
| newly firing (`c32` 1m) | **1.218** | 1.185 – 1.240 |
| control (unchanged) | 1.002 | 0.969 – 1.039 |

**The lesson is methodological and worth more than the numbers.** Both grids
pinned the other lever to isolate their own, which is correct experimental
design and is exactly why neither could see this. A rule validated at a pinned
operating point has only been validated *there*. The end-to-end A/B in the
shipped configuration is not a formality.

### Result, end to end, in the configuration that ships

`bench-results/phase4d/vf-*`: `legacy, new, legacy'` on one build with
`TENSORCONTRACT_ORIENT=legacy|rule`, everything else at its shipped setting, so
this is the whole of item 1d (orientation rule + the guard-3 removal) against
Phase 4.1. Exclusive machine, sibling CPU 0.8–1.0%.

| | shipped / Phase 4.1 | noise floor | best case |
|---|---|---|---|
| `f32` | **1.028** | 0.996 | 1.410 |
| `f64` | 0.998 | 1.003 | 1.057 |
| `c32` planar | **1.013** | 0.998 | 1.311 |
| `c32` 1m | **1.022** | 1.000 | 1.480 |
| `c32` 3m | 1.010 | 0.998 | 1.222 |
| `c64` planar | 1.004 | 1.005 | 1.042 |
| `c64` 1m | 0.999 | 1.012 | 1.046 |
| `c64` 3m | 1.004 | 0.993 | 1.050 |

Every 32-bit column clears the ±1.3% geomean floor; the 64-bit ones are flat,
which is expected — their `MR` already fitted the corpus's runs, so the old
rule was not in its failing case there. The `abcijk-e*bc-*` family, the largest
of the nine misses, goes **1.41–1.48x**.

Four cases are slower beyond the per-case floor. Three are the same `f32` case
under its three method labels (`abcijk-jkma-mibc`, 0.931, with the bracketing
repeat at 1.013, so it is real); the fourth is `ijkl-mink-jnlm` in `c64` 1m at
0.937 against a repeat of 0.960, i.e. mostly drift. **One genuine per-case
regression**, at 7%, against six cases gained at 1.3–1.5x.

### What is left

21 of 392 case-dtype-methods still take the slower arm by more than the noise
floor, worth up to 1.36x. They are a different population from the `abcijk`
family this rule was derived on — `abjcd-dkbac-jk`, `ajbdc-ckbad-jk`,
`abjc-cbka-kj`, `aqrs-pa-pqrs`, mostly in 1m — and all are failures to swap. The
oracle gap outside `f32` is 1.088 against 1.067 (`c32` 1m) and 1.118 against
1.102 (`c64` planar), so there is roughly 2% of corpus geomean still on the
table per dtype. `bench-results/phase4d/or-*` holds both arms for all of them,
so the next candidate costs nothing to score.

### Assumptions added

| # | Assumption | Status |
|---|---|---|
| A19 | The orientation discriminant is a property of `D`'s column direction (the form both previous rules took). | **Refuted.** The corpus families are mirror images, so a correct rule must be antisymmetric under exchanging the two directions; a rule phrased about one of them cannot be. Written symmetrically — row block fits, else the shorter run goes in the row role — it is 12 better / 0 worse over both arms of all 392, and closes the `f32` case to within 0.4% of an oracle. |
| A20 | A rule validated with the other levers pinned is validated. | **Refuted.** The orientation rule was 12/0 with the shape pinned and still cost 20% on three cases in the shipped configuration, because a 3% orientation error blocked a 20% shape change. The levers must be validated jointly, end to end, even when each was isolated correctly for derivation. |

| # | Assumption | Status |
|---|---|---|
| A16 | Choosing `MR` to divide the output's contiguous run converts whole block families to the write-back's fast path, so maximising that fraction is the rule. | **Refuted as a rule, confirmed as a mechanism.** Unguarded it scores 0.936 in `f32`. It needs three guards — shallow `k`, a default that is substantially broken, and no change of orientation — after which it fires on 20 of 392 case-dtype-methods for 1.11–1.12x on those and 1.026 corpus geomean in `c64` planar. |
| A17 | Shrinking `MR` will fix the orientation rule's nine misses by satisfying its condition 2. | **Refuted as a fix, and it prices the problem.** The flip does happen, and those cases gain 1.17–1.39x *while paying ~30% in kernel shape* — so the orientation alone is worth ~2x there and must be bought at the default `MR`. `MR` is the wrong instrument. |
| A18 | The register blocks measured at the operating `kc` are the right ones at every depth. | **Refuted.** At `k <= 24`, `c64` 3m prefers `MR` 16–24 over its default 8 (1.088) and `c32` 1m prefers `24x8` over `32x6` (1.099), with no write-back change involved. The Phase 3 table is depth-conditional. Belongs with item 2. |
| A13 | `MC` is bounded only by keeping the packed `A` block in L2. | **Refuted.** It also bounds the `D` strip a `jr` pass revisits, which is the binding constraint whenever the output's rows are strided. |
| A10 | The Phase 3 write-back defect is the write-back's own L2 traffic, needing a vectorised inner loop. | **Refuted as the primary cause.** It was the *orientation*: the row direction of the matrix view was the strided one, so the innermost loop jumped a row stride per micro-tile row. Choosing the orientation costs nothing and recovers the whole 2x. The vectorised inner loop is real but second-order, and only in single precision. |
| A11 | The row/column orientation is a property of the plan. | **Refuted.** The right choice depends on `MR`, hence on element type and complex method. |
| A12 | Write-back overhead matters equally in both precisions. | **Refuted.** It is per output element and the kernel work it hides behind scales with element size, so removing it is worth ~5x more in `f32`/`c32` (7–15%) than in `f64`/`c64` (2–3%). |
| A14 | The orientation rule's `run >= MR` condition is the right discriminant. | **Refuted, and unresolved.** It is right 63/72 but misses 9 cases by 1.18–1.47x, all 32-bit, including shapes with the same post-swap row structure as the ones it correctly rejects. The true discriminant is unknown; `rm-orient-{none,swap}.csv` hold both arms for all 72 case-dtypes to score candidates against. |
| A15 | A sequential build-to-build A/B is good enough for a few-percent effect. | **Refuted.** It reported `c64` 3m at 0.973 where a paired runtime A/B gives 1.020. Put the switch behind an environment variable and interleave the arms. |

## Phase 4 report, part 7 (item 2): the `MC`/`KC`/`NC` grid

**Status: the grid is measuring. Design and rationale below; results follow in
this section when it lands.** Started 2026-08-03 08:11 on `ccqlin038` cpu4,
19 arms x 2 dtype pairs, ~7 h. If this section still says "measuring" and
`bench-results/phase4e/run.log` ends in `grid complete`, the run finished and
the analysis is what is missing — `scripts/blocking-score-rules.py
bench-results/phase4e/features.csv bench-results/phase4e` reproduces it.

### Why this shape of experiment

The blocking is the last untouched Phase 2 heuristic: `kc` is 384 for 4-byte
reals and 256 otherwise, and `mc`/`nc` follow from a 512 KiB L2 budget for the
packed `A` block and 3 MiB of L3 for `B`, divided by the packed footprint each
method actually produces. Three earlier results dictate how it has to be
measured rather than leaving it a free choice:

1. **`KC` is first-order, not a tuning knob.** Phase 3 traced the entire
   complex-method ranking to whether the `A` sliver is an L1 resident or an L2
   stream at the operating `kc` — 3m is the *fastest* of the three when panels
   are L1-resident and third when they are not. `kc` is the parameter that
   decides which regime the engine is in.
2. **`MC` is bounded from both sides (A13).** Below by packed-`A` residency in
   L2, above by the strip of `D` that one `jr` pass touches and the next
   revisits. A sweep that varies the two together sees only their sum, and
   part 3's three-case experiment — which found +13% on one case and −18% on
   another and was backed out — is what that confound looks like. So the `kc`
   axis is measured **twice**: `_KC` moves the panel depth with `mc`/`nc`
   pinned, `_KC_COUPLE` re-derives them against the same budgets at the new
   depth. The pair separates the bounds; either arm alone does not.
3. **An absolute `MC` rigs the method comparison.** 1m derives half of planar's
   `mc` because its packed `A` carries four reals per complex element instead of
   two, and that proportionality is exactly what keeps the three-way comparison
   honest. So the `mc`/`nc` axes are swept as **percentages** of the derived
   value (`_MC_PCT`, `_NC_PCT`), not as absolute numbers.

The 19 arms: `base` (three times — first, middle, last, so drift over seven
hours is measured and every treatment is bracketed), `kc` ∈ {64, 128, 256, 384,
512} pinned, the same five coupled, `mc` ∈ {25, 50, 200, 400}% and `nc` ∈ {25,
400}%. Every arm is a runtime switch, so no arm is a rebuild (A15), and the
shipped row-block and orientation rules stay **on** — neither reads the
blocking, so there is no confound to pin away, and leaving them on means the
grid is measured in the configuration that ships (A20).

This is the *whole-grid* pattern from items 1c and 1d, for the third time and
for the same reason: a candidate rule scored against a grid that already exists
costs nothing, so the eventual A/B is spent on a rule that survived all 392
case-dtype-methods rather than on the first plausible one. `_MC_PCT` and
`_KC_COUPLE` exist so that a rule about `mc` can be *expressed* against the
grid at all.

Two properties of the design worth knowing when reading the output:

* A third of the corpus contracts over `k <= 24`, so for those cases every
  pinned-`kc` arm is bit-for-bit the same computation. Their spread across arms
  is a **per-case noise floor measured inside this very run**, alongside the
  three `base` repeats — not one imported from another session.
* The sweep CSV's `notes` column now records the `mc x kc x nc` each row
  actually ran with, so an arm is self-describing and a mislabelled one is
  detectable after the fact. Same reason `MR x NR` went in there in 4.1c.

### What the grid cannot answer

The truly depth-adaptive rule — `kc = min(k, KC)`, re-derived — is not an arm,
because `Blocking::derive` is per element type and does not know the
contraction's depth; expressing it needs the *driver* to re-derive, which is a
code change and not a switch. `ck64` is the closest the grid gets (for a `k =
24` case it widens `mc` fourfold where true adaptation would widen it tenfold),
so the coupled family brackets that rule's *direction* without measuring it.
That is deliberately the cheap half: if moderate coupled widening is broadly
bad, the adaptive rule is dead and agrees with part 3; if it is broadly good,
the driver-level switch is worth building and measuring.

Also not in the grid: the depth-conditional *register block* (A18). `c64` 3m
prefers a wider `MR` at `k <= 24` than at the operating `kc`, so the shape and
`kc` interact, and the honest version of this experiment varies both. The grid
holds the shape at whatever the shipped row-block rule picks. Sweeping the
product of the two grids is 19 x 5 arms, which is a week; the intended order is
to settle `kc` first and then re-run the row-block grid at the chosen `kc`,
because that is the direction the coupling runs — `kc` decides the regime, and
the shape is chosen inside it.

## Phases 4 (rest) – 5

In progress. See "Resume here" at the top of this file.
