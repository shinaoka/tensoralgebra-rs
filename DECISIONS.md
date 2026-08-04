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
* `crates/tensorprimitives-tapp` — TAPP C ABI, verified against the upstream
  headers, exercised end to end through the C entry points.
* `crates/tensorprimitives-bench` — `tcbench` with `verify` / `premise` /
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
  pattern to copy for any further discrete tuning decision). For running any of
  it on a cluster node: `node-session.sh` (stages a whole session),
  `topology.py` (what machine is this, and where may a thread go),
  `run-arms.py` (independent arms one per L3 domain, with per-domain occupancy),
  `validate-placement.sh` + `placement-spread.py` (is concurrent placement safe
  here). See part 10.

Sanity check on a fresh checkout, in this order:

```bash
cargo test --workspace --release              # 6 green suites
TENSORCONTRACT_KERNEL=scalar cargo test --workspace --release
cargo clippy --workspace --all-targets        # silent
cargo build --release -p tensorprimitives-bench
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

**Since the last measurement, four things were built and none was measured.**
All are correct, all are off or inert by default, and the whole set is green
across ten combinations of the switches (default, `scalar`, `avx2`, 8 threads
crossed with each, partition pinned to `m`/`n`/`2x2`, `blockmodel=model`, and a
debug build):

* **AVX2 kernels and ISA dispatch** (`avx512f` → `avx2+fma` → scalar), one macro
  body per method across both ISAs. The AVX-512 instruction stream is
  byte-identical to before — verified by disassembling both trees — so every
  number in this file stands. Register blocks are *provisional and unmeasured*.
  See the AVX2 interlude and D24–D26.
* **The analytical cache-blocking model** (part 9, D23), behind
  `TENSORCONTRACT_BLOCKMODEL=model`, default `legacy`. Predicts a `kc` 2.4–8x
  smaller than the constants, i.e. an L1-resident `A` sliver.
* **2-D threading** (part 8b, D27–D29), still off by default.
* **The corpus's parallel width and the case against K-parallelism** (A21).

**v0.1 packaging is in progress** — see the Phase 5 report, part 1, which is
mostly a list of quality gates that had never run. CI is fixed and expanded,
docs and the public API surface are settled (D30), the README is current, the
CHANGELOG exists, `cargo package` succeeds for all three crates, and MSRV is
verified at 1.89 (D31). Publishing is a human step and is not automated.

**Both pending measurements are done.** They ran on Rusty `rome` nodes on
2026-08-03 — jobs 6745376 (`worker5040`, 179 min: register blocks, thread scaling,
placement validation, the blocking grid) and 6745978 (`worker5175`, 204 min: the
traffic-sensitive blocking arms, sequentially). Read parts 7, 8, 8b, 9, 10 and 11.
**Nothing they measured is comparable to a single-core number elsewhere in this
file**: different machine, different cache hierarchy, and AVX2 rather than AVX-512.

Headlines, and four decisions they hand back:

* **`KC` is first-order and the default is too shallow.** `kc = 512` is worth 5.0%
  in `f64` and 3.3% in `c64` against a 0.5% floor; `kc64` costs 15–24%. `f32`/`c32`
  are already optimal. **Recommended, not taken** — a one-line change to
  `Blocking::derive`, and it is a Zen2/AVX2 result.
* **`MC` is a wide plateau** — 25% to 400% of the derived value moves the geomean
  ≤3%. Not worth tuning. Item retired.
* **The analytical blocking model loses in 11 of 12 columns** on the first foreign
  machine, by up to 7.2% in the complex methods, and part 7 attributes the whole
  loss to its `kc`. **Leave `TENSORCONTRACT_BLOCKMODEL=legacy`** (A33). D23's
  probing and descriptors stay; the default does not change.
* **Threading scales to 8 and saturates by 16–32**, declining at 64 with cores idle
  two-thirds of the time. The 2-D partition is vindicated at 1.6–3.4x, but
  `Plan::partition`'s `panels >= p` early return costs up to **4.3x** on 18 cases.
  **Recommended: default stays off** until the spawn cost and that early return are
  addressed — both fixable, neither structural.
* **D26's eight AVX2 register blocks are all confirmed** as the measured winners, so
  they stop being a guess with no code change.

Concurrent placement (24 arms, one per L3 domain) was validated at +0.3% on the
corpus and **rejected at −3.2% on the memory-bound half**, which is why the
traffic-sensitive arms got their own sequential job. It returned 20.4x on the grid.

What the two runs left open, in priority order:

1. **Fix `Plan::partition`'s early return.** The `panels >= p` fast path costs up
   to 4.3x on the 18 memory-bound `abcijk` cases at 64 threads (part 8b). The
   mechanism is *not* identified — write-back false sharing across sixteen L3
   domains and barrier span are both plausible — and `TENSORCONTRACT_PARTITION`
   makes either cheap to test. **Do not change the rule before measuring one of
   them**; three plausible mechanisms have already been wrong in this phase.
2. **Fuse the `pc` loop so `C` is touched once rather than `K/KC` times.** Promoted
   from the tail of the list: the whole measured `kc` effect is this quantity in
   disguise (+8.2% where deeper panels halve the pass count, +2.4% where they
   cannot), so the fusion should capture more of it than a bigger constant and make
   `kc` stop being first-order. **Superseded:** "re-derive the model's `kc`" was item
   2 here and is withdrawn — eq. (4)-(6) is arithmetically correct and its objective
   is unreachable, since the measured optimum puts the `A` micro-panel at twice the
   whole L1 (part 9).
3. **Pool the threads.** Spawning per `execute` call is the leading suspect for
   occupancy falling to 29–36% at 64 threads (part 8), and it is first-order for the
   small repeated contractions Phase 1 identified as the real headroom.
4. **Re-run the row-block grid at the chosen `kc`**, which was always the intended
   order (part 7): `kc` decides the regime and the shape is chosen inside it. Now
   that `kc = 512` is indicated for 8-byte reals, the Phase 3 register-block table
   was chosen at a depth the engine may no longer use.
5. **Close the remaining orientation gap.** 21 case-dtype-methods still take the
   slower arm, worth up to 1.36x, and they are a different population from the
   `abcijk` family the rule was derived on. Both arms are in
   `bench-results/phase4d/or-*`, so a candidate costs nothing to score with
   `scripts/orient-score-rules.py`. Untouched by the `rome` runs.
6. **Dispatch the complex method by shape** (item 3), still unstarted, and now with
   a new input: on AVX2 the kernel-level ranking puts 3m *first* in `f32`/`c32`
   (part 11), the opposite of AVX-512.
7. Then the rest of the Phase 4 list: small-`k` handling, fusing the `pc` loop so
   `C` is touched once, a pack-free fast path for unit-stride block scatter,
   prefetch.

An `icelake` session remains the open portability question: `rome` answered AVX2 and
a second cache hierarchy, but AVX-512 on a *different Intel* hierarchy — which is
where the legacy constants are wrong in a different direction — was deliberately not
covered. `scripts/rusty-phase4.sbatch` runs there unchanged with `-C icelake`.

Do not re-derive the register blocks. The AVX-512 ones are measured (Phase 3)
and recorded in `kernel::x86`; the **AVX2 ones are now measured too** and all eight
were confirmed as the winners they were guessed to be (part 11), so neither set is
open. `examples/kernel_shapes` re-derives them if the machine changes. Since Phase
4.1c each method carries a *menu* of them and the default is still the measured
choice — the menu adds alternates, it does not re-litigate the default. The one
caveat is item 4 above: the shapes were chosen at the old `kc`.

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
| D23 | The cache blocking is computed from **probed cache descriptors** via BLIS's analytical model, not from constants — and ships **off by default** behind `TENSORCONTRACT_BLOCKMODEL=legacy\|model`. | `Blocking::derive` held the only three machine-specific numbers in the engine: `kc = 384/256` by real size, a 512 KiB packed-`A` budget and a 3 MiB packed-`B` budget — half of `ccqlin038`'s L2 and a slice of its L3. Nothing about them transfers. `kernel::cache` probes sysfs (no `unsafe`, no dependency, and the only source that reports cache *sharing*), then x86 `CPUID` leaf 4 / `0x8000001D`, then conservative built-ins; a failed probe degrades to the next source and can never fail a contraction. The model is driven by the **reals per element the kernel packs**, not `size_of::<Element>()`, so 1m still gets half the rows planar does out of one L2 — the invariant the three-way comparison rests on. Off by default because the pending grid defines its arms *relative to the derived defaults*, so changing the derivation would silently change what that measurement means, and because A15 requires the old behaviour stay reachable as a run-time switch rather than a build-to-build diff. Build-vs-reuse: no crate, because `raw-cpuid`/`num_cpus`-style crates do not report per-level sharing and sysfs is ~100 lines of parsing. |
| D24 | The instruction set is a **parameter of the kernel macro**, not a second set of bodies. `simd_kernels!` takes the lane count, a target-feature list and six intrinsics; AVX-512 and AVX2 are two instantiations of the same four bodies. | D17's argument — a three-method comparison must not also be a comparison between three hand-tunings — applies unchanged to two instruction sets, and would be violated the moment somebody hand-tuned AVX2 planar and not AVX2 3m. It also made the change auditable: because the AVX-512 arm expands from unchanged source text, its instruction stream is *verifiably* untouched (7816 zmm instructions, byte-identical, checked independently at merge). |
| D25 | Dispatch is `avx512f` → `avx2 + fma` → scalar, and `TENSORCONTRACT_KERNEL` grows to `scalar\|avx2\|avx512\|auto`. A pinned ISA the CPU lacks falls back to scalar rather than faulting. | Without the pin, AVX2 kernels could be compiled on the reference machine and never executed on it, leaving the only coverage on hardware nobody here has. Same rule as `_ORIENT`/`_WRITEBACK`/`_ROWBLOCK`: every new fast path gets a run-time switch, because a build-to-build diff has already produced one wrong sign here (A15). Both `avx2` and `fma` are detected — separate CPUID bits, and the FMAs need the second. |
| D26 | AVX2 register blocks are **provisional and explicitly unmeasured**, chosen from the register budget and the uop model, with a menu of alternates and `examples/kernel_shapes` extended to an AVX2 grid as the calibration path. | D19 splits this cleanly and the split was honoured: `live <= 16` and `acc >= 10` are load-bearing and were *verified in the disassembly at no CPU cost* (all shipped defaults allocate 15–16 distinct ymm with zero stack traffic; `planar 1x6` at `live=16` spills one register, `planar 2x3` at `live=18` spills seven — the cliff is exactly where the count puts it), while "which of the shapes that fit is fastest" is recorded as a guess. Publishing a modelled shape as measured would corrupt the one thing the register-block table is good for. |
| D33 | The project is `tensorprimitives`, a facade over per-operation crates (`tensorcontract` now, `tensortranspose` planned), rather than one crate named after the operation it happens to implement first. | Three names were rejected on specific grounds. `tensoroperations` would claim the name of a *frontend* — index notation, contraction ordering — that `DESIGN.md` puts explicitly out of scope. `scattergemm` and `bsmtc` name the algorithm, which stops being true the moment a dedicated transpose kernel lands: a transpose is neither a GEMM nor block-scatter-matrix contraction. And a single crate named `tensorcontract` would have to grow non-contraction operations under a contraction name. "Primitives" is TAPP's own word for this layer — one operation per call, executed as asked — so the name states the scope and aligns with the standard the project implements. The shape is `futures`/`futures-core`: applications depend on the facade, libraries depend on the operation crate directly, since cargo features are additive across a dependency graph and a library enabling a facade feature imposes it on every other consumer — the point `blas-src`'s own documentation leads with. Precedent checked rather than assumed: Rust's BLAS ecosystem always names the implementation for itself (`openblas-src`, `netlib-src`, `blis-src`) and never after the interface, and TAPP has no provider-naming convention at all — its TBLIS and cuTENSOR interfaces live inside the reference repository rather than as published providers. |
| D34 | **No published facade crate** until there are at least two implemented primitives. `tensorprimitives` is the project and repository name; the published crates are one per operation plus the family-level `-tapp` ABI. | A facade re-exporting a single primitive is indirection, not abstraction: it would be `tensorcontract` plus a version to keep in lockstep, a re-export surface that has to track the engine's API, and a crate republished on every release with nothing in it changed. The argument that justified the *feature* structure also undercuts the facade — libraries are told to depend on the operation crate directly, because cargo features are additive across a dependency graph, which leaves the facade serving only applications, for whom the alternative is one extra line in a manifest. And the asymmetry decides it: adding a facade later is purely additive and breaks no existing dependent, whereas publishing one now is permanent. `futures` and `tracing` earned theirs by having several substantial members on a shared cadence. Accepted cost: crates.io will show `tensorprimitives-tapp` with no `tensorprimitives` stem, and the umbrella name stays unclaimed — cheaper than a maintained no-op, and publishing a shell to hold a name is name-reservation with extra steps. For C callers the family-level surface already exists: the TAPP crate is the umbrella that will expose transposition beside contraction. |
| D35 | **No Rust TAPP consumer-bindings crate** (a `tapp-abi`) until something links against it. This project ships a TAPP *provider* — it defines the symbols — and does not publish declarations for calling *into* other implementations. | Decisive fact: the upstream reference implementation has **no releases and no tags**, so an ABI crate would have nothing to version against and drift would be silent — the exact failure mode that TBLIS's `type_t` enum swap already cost this project once. The standard is also visibly unsettled: `attributes.h` specifies no keys, `product.h` leaves `TAPP_IN_PLACE` an open `//TODO`, `error.h` fixes only `TAPP_SUCCESS`, and the `status` out-parameter has no defined semantics — all four found by our own conformance suite. One argument previously offered for the split is **withdrawn**: it would *not* "leave something for other Rust providers", because a provider defines these symbols rather than declaring them, and two Rust providers could not coexist in one link. The real consumer is Rust code calling a C or CUDA TAPP implementation, of which there are currently none. Adding it later is purely additive, so it waits for a trigger — most plausibly the benchmark harness driving another TAPP implementation, which supplies a second consumer *and* the only real test of the declarations, since a declarations crate nothing links against is untested by construction. When that happens the name is `tapp-abi`, not `tapp-sys` (a `-sys` crate corresponds 1:1 to a native library; TAPP is a standard with several), and the seed is `abi_layout.rs`, which already maintains an independent transcription of all 23 symbols. |
| D32 | The TAPP conformance suite drives the **C symbols**, checks against the oracle *plus* hand-computed anchors, and is **falsified by perturbation** before being believed. | Testing the layer through its Rust types would exercise exactly the code that cannot be wrong. Two anchors sit alongside the oracle so a shared engine/oracle bug cannot pass, and both mechanisms were confirmed live by breaking them and checking *which* tests fail: one constant broke one test, a perturbed oracle broke 33 of 34. `abi_layout.rs` re-declares the upstream header so symbol export is a link error here rather than downstream. Where the ABI *cannot* detect an error — any non-zero bogus handle, a data buffer shorter than its extents — that is a comment rather than a test, because a test that invokes undefined behaviour is not evidence of anything. |
| D30 | The published surface is **three tiers**, stated in the crate docs, not one flat `pub`. Tier 1 is the contraction API (`contract`, `Plan`, `Layout`, `TensorView`/`TensorViewMut`, `Element`/`Real`, `Error`, the `with_*` builders, `kernel::scalar` as the documented extension point) under ordinary semver. Tier 2 is introspection of the engine's own decisions — `PlanStats`, `plan::Scatters`, `Plan::{transposes_gemm, row_block, partition, oriented_scatters, ..}`, `kernel::{selected_config, plan_config, selected_kernel_name}`, `kernel::cache`, the `scatter` builders — where **signatures are semver-stable and values are not**. Tier 3 is `#[doc(hidden)]`. | 158 undocumented public items across seven `pub` modules was the symptom; the disease was that nothing had decided which of them a user may depend on, and publishing settles that whether or not we answer it. Tier 2 is the load-bearing part and exists because of how this project works: every one of those functions returns the output of a measured heuristic, and Phase 4 has already moved three of them (A14 moved `transposes_gemm`, A16–A18 moved `row_block`, item 2 will move `Blocking`). Without saying outright that a tier-2 *value* is not part of the contract, the project must either freeze its own tuning or break semver every phase. The corollary is why they stay public at all: a harness or an alternative execution strategy can ask the engine what it would do instead of guessing, which is exactly what `tcbench orient` and `tcbench shapes` rely on. Only `kernel::x86` (register-block menus *are* per-machine measurements) and `scatter::BlockScatterMatrix` (not correctly constructible from outside) went to tier 3. |
| D36 | **Ship a header** — `crates/tensorprimitives-tapp/include/tapp.h` — rather than telling C callers to fetch the standard's own. Upstream's headers remain supported and equivalent; this one versions with the implementation. | D35 established that upstream has **no releases and no tags**, and then left C consumers depending on exactly that untagged `main`. A consumer pinning this crate was pinning its declarations to a moving target, with drift as silent as the TBLIS `type_t` swap. The header is not a second source of truth: every prototype was checked against upstream at `main`, `abi_layout.rs` independently pins the same 23 symbols from the Rust side, and `examples/c-consumer` now compiles *this* header and checks numerical results — so the two transcriptions are cross-checked by construction rather than by discipline. Two departures are marked in the file itself, both where upstream is underspecified: `TAPP_IN_PLACE` (an open `//TODO` upstream, so the definition is guarded and yields to any upstream one) and the non-zero `TAPP_error` codes (upstream fixes only `TAPP_SUCCESS`). |
| D37 | The C consumer is a **CMake project driven by corrosion**, run by CI in *both* the corrosion and prebuilt-library modes. | `abi_layout.rs` verified linkage from Rust, so no C compiler had ever seen the header and a wrong prototype would have reached a downstream user first. The two modes fail differently and both are real: prebuilt is what a site building the Rust half separately does and needs no network; corrosion is what a CMake project actually writes, and a break in cargo-from-CMake is invisible to the prebuilt path. Corrosion earns its place by resolving the platform link line — it derived `gcc_s;util;rt;pthread;m;dl;c` for a static link here, which is the usual first hand-written failure. The example checks *numbers*, not return codes, so a stride convention or complex-layout error fails rather than passing quietly. Linux only for now: macOS would add `.dylib` naming and install-name handling and is the obvious next job, left out rather than added blind. |
| D38 | A **panic boundary** (`guard`) on the three entry points that allocate or run the engine, mapping a caught unwind to `TAPP_ERROR_INTERNAL`. | A panic reaching an `extern "C"` frame aborts. That is memory-safe and, for a library called from a solver or an MPI rank, still the wrong outcome: `Panel::new` asserts rather than returning when a packing buffer cannot be allocated, and a contraction too large for memory is an ordinary user mistake. The panic hook still runs first, so nothing becomes less diagnosable. Scope is deliberate rather than uniform — the getters return `void` and could only swallow silently, and `TAPP_execute_batched_product` inherits the guard through the inner call — and `AssertUnwindSafe` is justified in the doc comment rather than assumed: an entry point that panics has not yet published its handle, the one exception being a partially written `D`, which the batched path already documents. Allocation *failure* proper still aborts and is not catchable here. Tested at the mechanism (`guard` itself), not end to end, because inducing an engine panic needs a fault-injection point that would have to be maintained — recorded so the limit is not mistaken for coverage. |
| D31 | MSRV stays **1.89** (D20), and CI pins exactly it. | The floor briefly became 1.94 by accident: the `CPUID` cache probe called `__cpuid_count` outside `unsafe` on the strength of a comment claiming it had been safe since 1.87. It had not — 1.89 through 1.93 fail with `E0133` and 1.94 is the first that compiles. Since that regression arrived with the probe rather than with any requirement, the fix is the `unsafe` block (plus `allow(unused_unsafe)` for toolchains where it is redundant), not five releases of downstream compatibility. The pin had already drifted the other way — it read 1.75, *below* the declared floor, so that job could never have passed. |
| D27 | Threading partitions the output in **two dimensions**: `pm` row strips of whole `MR` panels by `pn` column groups of whole `NR` blocks, with `pn > 1` only when `ceil(M/MR) < p`. The `N` cut is made *inside* loop 5, per `NC` block, not over the whole range. | D21's 1-D cap costs real throughput on the 16 case-dtype-methods whose row axis cannot fill 8 threads (part 8). Both axes partition the *output*, so D21's invariant survives intact — one owning thread per element, accumulating over the full `K` in the original order, hence bitwise identical to serial at every thread count *and* every `(pm, pn)`. Cutting `N` inside loop 5 is what keeps the packed `B` panel single and L3-sized (a top-level split would want `pn` panels and `pn` times the L3 budget) and keeps loops 5 and 4 identical across threads, which is what makes the barrier counts agree structurally rather than by bookkeeping. Barriers become per column group and `pm`-way; a pure `N` split synchronises nowhere at all. |
| D28 | The packed `A` block stays per thread, duplicated `pn` times, rather than packed once per row strip behind a barrier. | The duplication costs one packed element per `NR * ceil(blocks/pn)` lane-FMAs and is only ever paid when `M` is narrow; `Plan::partition` prices it and refuses to split `N` when a thread would have too few `NR` blocks to amortise it. The alternative needs a barrier *inside* loop 3 — `M/MC` times more often than the ones above it — and leaves the block in one thread's L2 for the others to pull across L3, when `MC` exists precisely to make it an L2 resident. |
| D29 | In the narrow-`M` regime `(pm, pn)` minimises `ceil(panels/pm) * (NR*ceil(blocks/pn) + PACK_WEIGHT)` with `PACK_WEIGHT = 8`, and `TENSORCONTRACT_PARTITION=m\|n\|<pm>x<pn>` pins the partition. | Maximising thread count alone either oversubscribes (3 panels, 8 threads → `3x3` = 9) or wastes threads; balancing tiles alone ignores the `A` duplication and takes an 8th thread that costs more than it returns. The weight is the rule's only modelled quantity, so it is one named constant and its insensitivity was checked offline against the committed feature table: **every weight in `[4, 64]` gives an identical partition on all 392 case-dtype-methods at 2/4/8/16/32 threads**. The env switch makes the axis choice a run-time A/B rather than a build diff (A15). |
| D21 | Threading parallelises the `M` direction only, into contiguous strips of whole `MR` panels, with a per-thread packed `A` and a **shared** packed `B`. | The `pc` loop accumulates into `D` in place, so parallelising it would need a per-thread temporary or atomics; `M` instead gives every output element one owning thread. That makes the result **bitwise identical to serial at every thread count** — a stronger invariant than agreeing with the oracle, and one a test can assert directly. Strips of whole panels keep each thread's row blocks aligned with the block scatter, so the write-back fast path and the orientation rule are unaffected. `B` is shared because `NC` is sized for L3, which is a per-socket resource. |
| D22 | The default thread count stays **1** until scaling is measured on the reference machine. | Every performance number in this file is a single-core measurement, and the item 2 blocking grid is designed against the serial engine. A default that changed with the machine's core count would make committed numbers irreproducible from a bare checkout. `TENSORCONTRACT_THREADS` and `Plan::with_threads` opt in; flipping the default is one line in `Plan::threads`. |
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
* Cargo workspace: `tensorcontract` (core), `tensorprimitives-tapp` (C ABI),
  `tensorprimitives-bench` (harness). CI (build/test/clippy/fmt/docs/MSRV +
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

**Status: measured on `worker5040` (Zen2, AVX2), 40 arms in 34.6 min wall against
707 min of arm-time — a 20.4x return on the concurrent placement.** Results
first, then the design that produced them. The `ccqlin038` run this section was
written for never happened; it was stopped after two arms (see "Resume here") and
the grid moved to a cluster node, which changed the machine and the instruction
set. **Nothing here is comparable to a single-core number elsewhere in this file.**

### The floor, and why this grid supports global conclusions only

Three independent estimates from inside the run:

| estimate | `f64`/`c64` | `f32`/`c32` |
|---|---|---|
| `basem` vs `base` (mid-run repeat) | 1.000–1.004 | 0.997–1.004 |
| `base2` vs `base` (end-run repeat) | 0.991–0.996 | 0.983–0.999 |
| arms that are **bit-for-bit identical** to `base` (`kc256` in 8-byte, `kc384` in 4-byte) | 0.995–0.999 | 0.979–0.992 |
| per-case spread over pinned-`kc` arms at `k <= 64`, where they are the same computation | **6.2%** | 1.9% |

The third row is the sharpest instrument in the grid and it was free: for the
4-byte dtypes `TENSORCONTRACT_KC=384` *is* the default, so that arm must read
1.000 and reads 0.979–0.992. So the geomean floor is **~0.5% in 64-bit and ~2% in
32-bit**, and the per-case floor is **6.2% in `f64`**.

That last number governs how the rest of this section may be read. The grid has 19
arms, so choosing the best per case harvests noise: the "oracle" row below claims
1.077 for `f64` against the best *global* arm's 1.050, and with a 6.2% per-case
floor that 2.7% gap is not distinguishable from picking maxima out of noise. The
demonstration is in the run itself — the top of the per-case wins list includes
`kc64` beating `base` by 1.130 on `ajbdc-ckbad-jk`, a case with `k = 24` where
**every pinned-`kc` arm is bit-for-bit the same computation.** A third of the
corpus is in that position. So: `193 of 588 case-dtype-methods gain more than the
per-case floor` is substantially noise, no per-case blocking rule is supportable
from this run, and every conclusion below is a *global* one.

### `KC` is first-order, and the direction is deeper — the opposite of what part 9 predicts

Geomean against `base`, per dtype and method (identical across methods in the real
dtypes because the real path is method-independent):

| arm | `f32` | `f64` | `c32` planar / 1m / 3m | `c64` planar / 1m / 3m |
|---|---|---|---|---|
| `kc64` | 0.760 | 0.848 | 0.798 / 0.794 / 0.783 | 0.883 / 0.889 / 0.891 |
| `kc128` | 0.885 | 0.948 | 0.915 / 0.916 / 0.903 | 0.956 / 0.969 / 0.960 |
| `kc256` | 0.960 | *0.995* | 0.982 / 0.989 / 0.981 | *0.997 / 0.999 / 0.999* |
| `kc384` | *0.979* | 1.038 | *0.992 / 0.991 / 0.990* | 1.026 / 1.021 / 1.029 |
| `kc512` | 0.985 | **1.050** | 1.000 / 0.999 / 0.999 | **1.033 / 1.034 / 1.033** |

*Italic* entries are the arms that are bit-for-bit `base`, i.e. the floor.

1. **Shallow `kc` is catastrophic**: `kc64` costs 15–24%. The Phase 2 heuristic is
   nowhere near that bad, but it is on the wrong side of the optimum.
2. **`f64` gains 5.0% at `kc512`** — ten times the 64-bit floor — and `c64` gains
   3.3% in all three methods. `f32` and `c32` are already at their optimum
   (`base` is `kc = 384` there) and go nowhere.
3. **This contradicts part 9's central prediction.** The analytical model wants
   `kc` *smaller* — on this machine 256→128 for `c64` 1m and 384→160 for `c32` 1m —
   to make the `A` sliver an L1 resident, which Phase 3 found the method ranking to
   turn on. Measured, the complex methods prefer `kc` **deeper or unchanged**, and
   nothing prefers it shallower. Whatever the L1-residency argument buys, on Zen2
   it is smaller than what deeper panels buy.
4. `kc512`-`f64c64` is one of the 13 arms that crossed a socket boundary from
   `base`, and crossing reads ~1.1% *low*, so the 1.050 is if anything an
   underestimate.

### Coupling adds nothing, and `MC` is a plateau

| arm | `f32` | `f64` | `c64` planar |
|---|---|---|---|
| `ck512` (depth + re-derived `mc`/`nc`) | 0.988 | 1.031 | 1.006 |
| `kc512` (depth alone) | 0.985 | **1.050** | **1.033** |
| `mc25` | 0.981 | 1.019 | 0.998 |
| `mc50` | 0.971 | 1.029 | 1.005 |
| `mc200` | 0.986 | 1.015 | 1.013 |
| `mc400` | 0.987 | 1.020 | 1.017 |

Two answers to questions part 7 was built to separate:

* **The coupled arms are no better than the pinned ones and in `f64` are 1.9%
  worse.** Re-deriving `mc`/`nc` at the new depth is not where the effect is —
  panel depth alone is. The pair of axes did its job: the two bounds are
  separable and only one of them matters.
* **`MC` is a wide plateau.** Scaling the derived `mc` from 25% to 400% — a
  sixteenfold range — moves the geomean by at most 3%, all of it within about two
  floors. A13's two-sided bound is presumably real, but **the interval between the
  bounds is wide enough that `MC` is not worth tuning.** That is a clean negative
  result and it retires an item.
* The per-case rules the scorer tries (`couple kc at k <= 32/64`) reach at most
  1.018–1.025 in `f64` — *worse* than simply setting `kc = 512` globally. Combined
  with the 6.2% per-case floor, there is no case for a per-case blocking rule here.

### The `nc` and `model` arms, re-measured sequentially

`nc25`, `nc400` and `model` change how much memory traffic an arm generates, and
the concurrent placement is **not** neutral for such arms — it was rejected at
−3.2% on the memory-bound half (part 10). A uniform penalty cancels in `arm /
base`; a traffic-dependent one does not. So they were held back and re-measured
sequentially on one core (job 6745978, `worker5175`, warm-up arm discarded),
where `base2` reads **1.000–1.004** — a 0.4% floor, against the placed grid's
0.983–0.996.

| arm | dtype/method | **clean** | placed | delta |
|---|---|---|---|---|
| `nc25` | `f64` | 0.976 | 0.991 | −0.014 |
| | `c64` 3m | 0.940 | 0.948 | −0.008 |
| | `c32` 3m | 0.943 | 0.942 | +0.000 |
| `nc400` | `f64` | 0.995 | 0.994 | +0.002 |
| | `c64` 3m | 1.001 | 0.998 | +0.003 |
| `model` | `f64` | **1.011** | 1.024 | −0.013 |
| | `f32` | **0.975** | 0.984 | −0.009 |
| | `c64` planar / 1m / 3m | **0.972 / 0.967 / 0.956** | 0.977 / 0.975 / 0.959 | ≤0.009 |
| | `c32` planar / 1m / 3m | **0.984 / 0.929 / 0.928** | 0.991 / 0.947 / 0.938 | ≤0.018 |

**The confound was real and small: `|delta| <= 1.8` percentage points, mostly under
1.** No conclusion moves. So holding these arms back was the right procedure — a
−3.2% rejection can swamp a 2% effect and there was no way to know in advance that
it would not — and the answer is that the placed grid was quotable after all. Worth
recording in that order, because the next person will face the same choice with the
same absence of information.

`nc` is confirmed to have nothing in it: shrinking it costs 2–6% (worst in 3m),
enlarging it does nothing, and the `nc`-only oracle is 1.008–1.016.

### The analytical model loses on the first machine it was supposed to help, and its `kc` is why

`model` is **below `base` in 11 of 12 columns**, and its one gain — `f64` at 1.011 —
is barely twice the 0.4% floor. In the complex methods it costs **1.6% to 7.2%**.
See part 9 for what that does to D23.

The cause is attributable, and attributing it is exactly what part 7's
single-parameter arms were for. The model predicts a shallower `kc` for every
complex method; the pinned-`kc` arms price that depth directly, and the two agree:

| method | model's `kc` (legacy) | `model` arm | pinned-`kc` arm at that depth |
|---|---|---|---|
| `c64` 3m | 128 (256) | 0.959 | `kc128` = **0.960** |
| `c64` 1m | 128 (256) | 0.975 | `kc128` = 0.969 |
| `c64` planar | 192 (256) | 0.977 | between `kc128` 0.956 and `kc256` 0.997 |
| `c32` 1m | 160 (384) | 0.947 | between `kc128` 0.916 and `kc256` 0.989 |
| `c32` 3m | 170 (384) | 0.938 | between `kc128` 0.903 and `kc256` 0.981 |
| `c32` planar | 256 (384) | 0.991 | `kc256` = 0.982 |

**The model arm's damage is its `kc` and nothing else.** Its `mc` rises 4–6x and its
`nc` about 20x, and neither shows up — which is consistent rather than surprising,
since `mc` is a plateau and `nc` has nothing in it. So the model's `mc`/`nc`
reconstruction is harmless and its `kc` equation is the whole problem.

### Why deeper `kc` wins, from the grid and no extra machine time

The hypothesis first recorded here — that eq. (4)–(6) fails to carry the
per-element real counts through — **was wrong, and is retracted.** `model_kc` takes
its ways ratio from `a_step`/`b_step`, which are `mr * a_reals * real_bytes` and
`nr * b_reals * real_bytes`, so the panel formats *are* accounted for. Worked by
hand for `c64` 1m on Zen2: `a_step` = 128 B/k, `b_step` = 96, `c_ar` =
⌊7·128/224⌋ = 4 ways, `kc` = 4·4096/128 = **128**, which is exactly what the model
reports. The arithmetic is right.

What is wrong is the **objective**. At the measured optimum, `kc >= 512`, that same
`A` micro-panel is 65 KB against a 32 KiB L1 — *twice the whole cache*. So the
engine's best depth is one where the paper's central premise, that a micro-panel
occupies whole L1 ways and the next one evicts the last, does not hold at all. No
repair of eq. (4)–(6) reaches `kc = 512`, because the equation is answering a
different question.

The grid says what the engine is actually buying, using the `kc512` arm's *own*
identical-computation subset as the control. For `k <= 256` there is one `pc` pass
at either depth, so those cases are the same computation and their ratio is the
arm's bias; for `k > 256` the pass count `K/KC` halves:

| `pc` passes at `base` (`kc = 256`) | n | `kc512` / `base` |
|---|---|---|
| 1 — same computation, i.e. the control | 204 | 1.024 |
| 8+ — pass count halves | 90 | **1.082** |

So the effect is **+5.7% net of the arm's own bias, and it lives entirely where
deeper panels reduce the number of times `C` is re-touched.** That quantity is
already on the Phase 4 list as its own item — "fusing the `pc` loop so `C` is
touched once rather than `K/KC` times" — and deeper `kc` is a partial, free version
of that fusion. The analytical model has no term for `C` traffic at all, which is
why it points the wrong way: it optimises a residency this engine does not benefit
from, and ignores the one that dominates.

Two consequences worth acting on rather than admiring:

* **The `kc` recommendation is really a `C`-traffic recommendation.** Raising the
  constant captures part of the win; fusing the `pc` loop should capture more of it
  and make `kc`'s depth much less important. Prefer the fusion.
* **A prediction for the reference-machine grid** (`scripts/ccq-blocking-night.sh`,
  which adds `kc768`/`kc1024`): the `k > 256` population should keep gaining as
  depth rises until the pass count reaches 1, and the `k <= 256` population should
  show only each arm's bias. If instead deeper arms help the `k <= 256` cases too,
  this account is wrong and something about buffer size, not pass count, is doing
  the work.

### What this means for the default `KC`

The recommendation is **`kc = 512` for 8-byte reals**, worth 5.0% in `f64` and
3.3% in `c64` against floors of 0.5%, with the 4-byte default already correct at
384. It is a one-line change to `Blocking::derive` and it is deliberately **not**
made in the same commit as the measurement. Note it is also a change measured on
*Zen2 with AVX2 register blocks*: on `ccqlin038` the same question was never
answered, so this is a recommendation for this machine class and an argument for
the model arm rather than for a new hardcoded constant everywhere.

### Why this shape of experiment

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

## Phase 4 report, part 8 (item 4): threading, implemented and unmeasured

**Status: built, correct, and not yet measured.** It ships **off** (D22): the
default thread count is 1, so nothing in this file changes and the item 2 grid
still measures the engine it was designed against.

### Why this got done before item 2 finished

The workstation was needed for other work, which made a seven-hour
noise-sensitive measurement the wrong thing to be holding the machine for and a
structural implementation the right one — threading's *design and correctness*
cost almost no CPU, while its measurement wants more of the machine than the
blocking grid does (a whole socket, not one core).

There is also an ordering argument, and it is worth recording because it cuts
one way and not the other. Under this parallelisation the packed `A` block stays
per-thread in L2, so item 2's `kc` and `mc` conclusions will carry over
unchanged. The packed `B` panel is **shared**, and `B_BUDGET` currently charges
3 MiB of a 25 MiB L3 as though one core owned the cache — so `NC` becomes a
per-socket question the moment threading is on. **Tonight's `nc` arms are
therefore single-core results and must be labelled as such**; the `kc` and `mc`
arms are not affected.

### The scheme

`M` is cut once into `p` contiguous strips of whole `MR` panels. Each thread
runs loops 3, 2 and 1 over its own strip with its own packed `A` block; the
packed `B` panel is shared, packed cooperatively (each thread takes a slice of
its `NR` slivers), and bracketed by two barriers per `(jc, pc)` iteration — one
so nobody is still reading the previous panel, one so the new one is complete.
Loops 5 and 4 are identical across threads, which is what makes the barrier
counts agree without tracking them.

Consequences, all deliberate (D21):

* **No reduction anywhere.** Every output element has one owning thread which
  accumulates over the full `K` in the original order.
* **Bitwise identical to serial at any thread count**, therefore. This is the
  strongest available correctness invariant and it is asserted directly, not
  approximated by a tolerance: a strip that dropped a row, one that
  double-counted, or a thread reading `B` across a barrier would each break it,
  and the last of those is exactly the kind of bug a tolerance check waves
  through.
* **Strips are whole `MR` panels**, so each thread's row blocks stay aligned
  with the block scatter. The write-back fast path, the row-block rule and the
  orientation rule are all untouched — threading does not re-open any of
  Phase 4.1.
* **The serial path is unchanged**: at `p == 1` the difference from the
  pre-threading driver is two `Option` checks per `(jc, pc)`, nowhere near the
  hot loops.

### Correctness

`cargo test --workspace --release` green, and green again under
`TENSORCONTRACT_KERNEL=scalar`, at `TENSORCONTRACT_THREADS` of 1, 2, 4 and 8 —
i.e. the *entire* existing suite, oracle comparisons included, also runs through
the threaded driver, which was free coverage worth taking. Five new tests cover
bitwise agreement with serial across all four dtypes and all three methods, a
blocking that forces hundreds of barrier round-trips, a batch axis (whose
failure mode is a deadlock rather than a wrong answer, so it is worth isolating),
and thread counts far exceeding the panel count.

`Plan::strips` reports how many strips a plan will really use, and the tests
assert it — following the precedent of the orientation tests, which assert the
heuristic actually fires so that a test cannot quietly stop testing anything.
That assertion earned itself immediately: it caught that the strips run along
the **oriented** row direction, so a `1 x 33` output parallelises into two
strips along 33. Skinny-`M` is therefore not automatically serial; skinny in
*both* directions is.

### What is not known yet

Everything quantitative. `scripts/phase4f-threads.sh` is written and
smoke-tested and measures 1/2/4/8 threads on physical cores of one socket
(it refuses a cpuset containing hyperthread siblings, since that measures a
different question). The only number in hand is from that smoke test — one case,
8 MiB, one rep, on a shared machine, so an indication and nothing more:
`ijkl-imjn-lnkm` at **3.63–3.73x on 4 threads** in `f64`/`c64` and 2.69x in
`c32` planar. Treat as evidence that the parallelism is real, not as a result.

Three limits are structural and known in advance, listed in the order they will
bite:

1. **Parallelism is capped at `ceil(M / MR)` strips.** A contraction whose
   oriented row direction is short cannot use the cores however much work it
   contains. **Now sized, from committed data and at no CPU cost**
   (`scripts/thread-width.py`): **20 of 392 case-dtype-methods cannot fill 8
   threads from `M` alone** — four distinct cases, `aqrs-pa-pqrs`,
   `ij-ikl-ljk`, `ij-kil-lkj` and `ijk-il-jlk`. They are not the slow ones:
   `ijk-il-jlk` in `c32` runs at 0.86 of the fastest throughput the engine
   reaches in that dtype, so this is real work, not a corner. **All 20 reach
   width 8 from `M x N` together**, which needs no accumulators and no
   reduction, so the fix is a 2-D partition and not a `K` split.
2. **Threads are spawned per `execute` call** via `std::thread::scope`, not
   reused from a pool. Irrelevant at corpus sizes, first-order for small
   repeated contractions — which is exactly the low-arithmetic-intensity
   population Phase 1 identified as the real headroom.
3. **`NC`'s L3 budget is still per-core**, as above.

### `K`-parallelism is not needed, and the reason is structural

Worth recording so it is not re-opened: parallelising the `pc` loop — the one
axis that would require per-thread accumulators and a reduction — is **not
needed by anything in this corpus**, and the argument generalises beyond it.

Needing it means the output has fewer than `p` micro-tiles in total, i.e. an
output of order 1–3 thousand elements while `K` is large. But arithmetic
intensity at the matrix level is bounded by roughly `2MN / ((M + N) * bytes)`,
so small `M` *and* small `N` cap it from above. Compute-bound requires a large
output, and a large output has plenty of tiles: the two conditions are in
tension, which is why the intersection is empty here rather than merely
unpopulated.

Measured confirmation is in `scripts/thread-width.py` above: every case short of
width 8 in `M` has `ceil(N / NR) >= 8`. And if a user ever does bring such a
shape, the feature is *cheap* precisely where it is needed — the output being
tiny is what makes per-thread accumulator tiles L1-resident and the final
reduction negligible. So this is a demand-driven feature, not a Phase 4 item.

### Measured, on `worker5040` (Zen2, 64 cores of one socket, AVX2)

**Everything in this subsection is within-session.** Nothing in it is comparable
to the single-core numbers elsewhere in this file. Each arm ran on exactly as many
cores as it had threads, taken in order so that `t4` is one 4-core L3 domain and
`t8` is two; occupancy over all 128 cores was recorded per arm and the 64–127
cores outside the cpuset were idle at **0.0% mean, 1.1% max** throughout, so the
node was genuinely exclusive rather than nominally so.

#### The noise floor, and a defect in the A/B/A' pattern itself

| pair | `t1` vs `t1b` | when both ran |
|---|---|---|
| `f64`/`c64` | **0.953–0.966** | `t1` first thing, `t1b` after ~50 min of 64-core load |
| `f32`/`c32` | **0.981–0.989** | both after the node was already hot |

The first pair's repeat is 3.4–4.7% slower than its opening arm, uniformly across
every case, dtype and method — a direction, not a spread, so it is drift and not
noise. The second pair, measured entirely on a hot node, drifts only 1.1–1.9%.
That difference is the diagnosis: **the opening arm of the session was measured at
single-core boost on a cold package and nothing else was.**

This is a real limitation of the `A, B, A'` pattern this project standardised on
(A15, and the header of `phase4-remeasure.sh`): bracketing detects drift, but when
the drift is a *cold-start* transient the bracket reports it as a floor of ±4.7%
and silently inflates every ratio measured against `A`. On `ccqlin038` the effect
was invisible because a shared workstation is never cold. **Future sessions on a
boost-happy machine need a warm-up arm that is run and discarded**, and until one
exists the honest floor for the `f64`/`c64` column below is asymmetric: its
scaling ratios are understated by roughly 4%.

A second floor, and the one that matters more, comes free from the partition arms:
on the 270 case-dtype-methods where `TENSORCONTRACT_PARTITION=m` and the rule
chose *bit-for-bit the same partition*, two adjacent hot arms differ by

| | geomean | median case | p95 case |
|---|---|---|---|
| `f64`/`c64` | 0.980 | 6% | **31%** |
| `f32`/`c32` | 0.994 | 6% | **26%** |

So at 64 threads the per-case median floor is ±6% — the same as `ccqlin038`'s
single-core figure — but the **tail explodes to ±26–31%**. Per-case claims at high
thread counts are close to worthless here; geomeans over the corpus are good to
about ±2%.

#### Scaling saturates at 16–32 threads and then declines

Per-case geometric mean against the same session's `t1`:

| threads | 2 | 4 | 8 | 16 | 32 | 64 | cores busy at 64 |
|---|---|---|---|---|---|---|---|
| `f64` | 1.90 | 3.41 | 4.72 | **5.76** | 5.68 | 5.37 | 36% |
| `c64` planar | 1.96 | 3.73 | 6.22 | 7.80 | **8.39** | 7.65 | 36% |
| `f32` | 1.90 | 3.51 | 5.53 | 6.36 | **6.56** | 6.01 | 29% |
| `c32` planar | 1.95 | 3.75 | 6.14 | 7.72 | **8.37** | 7.55 | 29% |

Four things follow, and the occupancy column is doing most of the work:

1. **Scaling is clean to 8 threads** (4.7–6.2x) and saturates by 16–32. Beyond the
   peak it *declines*: 64 cores are slower than 32 in every dtype.
2. **The cores are idle, not saturated.** Busy fraction falls 98% → 95% → 88% →
   73% → 49% → 36% as threads double. At `t64` roughly two thirds of the wall
   clock is not compute, which rules out memory bandwidth as the primary limit and
   points at the two structural limits part 8 listed in advance: threads spawned
   per `execute` call rather than pooled, and two barriers per `(jc, pc)`
   iteration now crossing 16 separate L3 domains. **Limit #2 was named before this
   run and is now the leading suspect, priced.**
3. **Complex scales better than real** at every thread count (7.65 against 5.37 at
   64 in 64-bit). That is the twice-arithmetic-intensity argument from Phase 1
   appearing somewhere it was never claimed — complex is less exposed to the
   memory system, so it holds up longer as cores are added. A *new* instance of a
   settled result, not a re-derivation of it.
4. **Per case at 64 threads: median 5.89x, zero cases above 32x, 193 of 294 below
   8x.** The best case reaches 17.8x. Nothing in this corpus scales well at 64
   threads.

#### The prediction, scored

Part 10 predicted, before the run, that the genuinely partition-limited family at
64 threads would be `ij-ikl-ljk` and `ij-kil-lkj`, and that if anything else
flattened it would be contention or a bug rather than the partition. Scored:

* **Right about those two.** They are the two worst cases in the whole corpus at
  `t64` — 2.13x and 2.29x — and the CSV's notes column confirms they ran `27x2`,
  the clamped partition the prediction named.
* **Wrong about the rest, and the miss is the more valuable half.** The next worst
  are the `abcijk` family at 2.43–2.47x running `64x1` — full parallel width, no
  partition limit at all. They are the memory-bound `k = 24` cases, and the reason
  they flatten turns out not to be contention either. See part 8b.

#### The default stays off, with numbers

Threading is off by default (D22) purely because it was unmeasured. It is now
measured, and the recommendation is **still off** — but for a specific and fixable
reason rather than a general one. It scales cleanly to 8 threads in every dtype;
what fails is 16 and beyond, where the loss is concentrated in per-call thread
spawning, barrier cost across 16 L3 domains, and one wrong branch in
`Plan::partition` (part 8b). None of those is structural. Turning it on is a
decision for the user, per D22, and is deliberately not taken in the same commit
as the measurement.

| # | Assumption | Status |
|---|---|---|
| A21 | Some compute-bound contractions will need `K`-parallelism, hence per-thread accumulators and a reduction. | **Refuted for this corpus, and argued structurally.** 20 of 392 case-dtype-methods cannot fill 8 threads from `M`, all 20 can from `M x N`, and needing `K` requires fewer than `p` micro-tiles in the whole output — which bounds arithmetic intensity at `~2MN/((M+N)*bytes)` and so bounds the case away from compute-bound. Build the 2-D partition; leave `K` unbuilt until a real shape demands it. *Confirmed again in part 10 at every thread count to 128 in both instruction sets, i.e. 16x the width it was argued at.* |
| A31 | `A, B, A'` bracketing is enough to establish a session's noise floor. | **Refuted on a machine with boost headroom.** The session's opening arm ran at single-core boost on a cold package and nothing else did, so its repeat came back 3.4–4.7% slower *uniformly* — reported by the bracket as a ±4.7% floor, while every ratio measured against that opening arm was inflated by the same amount. The second dtype pair, measured entirely hot, drifts 1.1–1.9%. Invisible on `ccqlin038` because a shared workstation is never cold. **Run and discard a warm-up arm.** |

## Phase 4 report, part 8b (item 4): the partition becomes 2-D

**Status: built, correct, unmeasured.** Still off by default (D22).

Part 8 named the 1-D partition's first limit and sized it before the fix was
built: 16 of 392 case-dtype-methods cannot fill 8 threads from `M` alone, over
four cases (`aqrs-pa-pqrs` at 2–6 row panels; `ij-ikl-ljk`, `ij-kil-lkj` and
`ijk-il-jlk` at 5–7), and every one of them is 26–23435 `NR` blocks wide. (That
count is at the *default* register block, which is what `tcbench orient` emits;
the shipped row-block rule changes `MR` on some cases, so a case at the boundary
can move either way.) The output is now cut into a `pm x pn` grid (D27–D29).
Three things are worth carrying forward.

**The two axes are cut in different places, and that is the design.** Row strips
are cut once, outside everything; column groups are cut inside loop 5, per `NC`
block. So every thread walks the same `(h, jc, pc)` sequence and a thread's cell
changes only how much work happens inside an iteration, never how many
iterations there are. That preserves both properties the 1-D scheme rested on —
one L3-sized shared `B` panel, and barrier counts that agree because loops 5 and
4 are identical across threads — while making the `N` axis available. A thread
whose column group is empty in a tail `jc` block still takes both of that
block's barriers and *then* skips loop 3.

**A negative result on indexing, and the most valuable thing this produced.**
Indexing the shared `B` panel by *absolute sliver*, which is what the 1-D scheme
did, is wrong the moment `N` is split: a group's sliver range **moves** between
`jc` blocks (a tail block has fewer slivers to divide) and its stride changes
with `pc_len`, so one group's next block lands on another group's current one —
and by design nothing orders them. Each group now owns a fixed slice, strided by
the worst-case sliver size so groups on different `pc` blocks cannot overlap
either. It surfaced only under test-level oversubscription and never in a
standalone repro; the barrier-stress case, which has one sliver per `jc` block
and therefore ranges that never move, passed throughout. **Worth remembering as
the shape of bug this parallelisation produces: not a missing barrier, but a
buffer whose ownership map is not constant.**

**The one modelled decision, priced without the machine.** `PACK_WEIGHT = 8` was
replayed offline over all 392 case-dtype-methods at five thread counts, and every
weight in `[4, 64]` gives an identical partition everywhere; the only two cases
that move at all choose between candidates 1.4% apart in modelled cost. At 8
threads the rule spills most of the narrow cases onto the column axis and
deliberately leaves `ij-ikl-ljk` / `ij-kil-lkj` 1-D **on 7 of 8 threads by
choice** — 37 column blocks split eight ways is five per thread against a whole
packed `A` block each. A hand estimate puts 1-D ahead by ~2.6%, which is an
estimate, which is why `TENSORCONTRACT_PARTITION` exists and why
`scripts/phase4f-threads.sh` now runs `=m` and `=n` arms at 8 threads beside the
scaling curve. Verified behaviourally at merge: `aqrs-pa-pqrs` runs `2x4` in
`f64` and `1x8` in `c64`, a wide case stays `8x1`, and the sweep CSV records
`t<threads>/<pm>x<pn>` so the partition is recoverable from the data.

### Measured, on `worker5040` at 64 threads: the 2-D split is right and the rule's first line is wrong

Two arms, both against the rule at the same thread count, on the cases where they
actually change the partition (everything else is a control group and is what the
±2% geomean / ±6% median-case floor in part 8 was derived from).

**`TENSORCONTRACT_PARTITION=m` — forcing the old 1-D `M` split — loses everywhere
the 2-D rule fires:**

| dtype | cases changed | geomean |
|---|---|---|
| `f64` | 12 | **0.607** |
| `f32` | 12 | **0.607** |
| `c64` | 6 | **0.297** |
| `c32` | 12 | **0.521** |

So the 2-D extension is worth **1.6x to 3.4x** on exactly the population it was
built for, in all four dtypes, far outside any floor. D27–D29 are vindicated, and
part 8b's hand estimate that leaving two cases 1-D on 7 of 8 threads was worth
~2.6% turns out to have been the wrong thing to worry about — at 64 threads the
axis choice is worth factors, not percent.

**`TENSORCONTRACT_PARTITION=n` — forcing 1-D `N` — beats the rule in the real
dtypes and loses badly in the complex ones:**

| dtype | cases changed | geomean | max |
|---|---|---|---|
| `f64` | 141 | **1.055** | 4.32 |
| `f32` | 144 | **1.100** | 3.14 |
| `c64` | 147 | 0.678 | 1.82 |
| `c32` | 141 | 0.748 | 2.34 |

And the real-dtype gain is not spread thinly — it is concentrated on the
memory-bound `abcijk` family (18 cases, `k = 24`), which gains a **geomean 2.25–2.26x
and up to 4.32x**:

```
4.32x  abcijk-jkmc-miab  f64   rule t64/64x1 -> t64/1x64
4.23x  abcijk-ijmc-mkab  f64   rule t64/64x1 -> t64/1x64
```

**The diagnosis is sharper than A28 guessed, and it is not `PACK_WEIGHT`.** These
cases have 768 row panels against 64 threads, so `Plan::partition` never reaches
the cost model at all — it returns from the first line:

```rust
if panels >= p { return (p, 1); }
```

That early return encodes "if `M` alone can fill the threads, `M` is the right
axis". At 8 threads on a socket-wide L3 that was true and cheap. At 64 threads on
Zen2's sixteen 4-core L3 domains it costs **up to 4.3x on 18 of 49 corpus cases**,
and it is the same population — the `k = 24` memory-bound family — that part 8's
scaling curve showed flattening at 2.43x for no visible reason. The two findings
are one finding.

**`PACK_WEIGHT`'s direction is right, which is why the fix is not "always consult
the cost model".** The complex methods *lose* 25–32% from the same `1x64` switch,
and that is exactly what the packing term predicts: a complex packed `A` carries
two to four reals per element, so duplicating it across 64 column groups costs more
than the real path pays. So the rule is right for complex and wrong for real on
this family, and a corrected rule needs the early return removed *and* a term that
captures whatever makes the real path prefer `N` here — which this run has not
identified. Two candidates, both cheap to test with the switch that already exists:
write-back locality (under `64x1` sixty-four threads write interleaved row strips
of `D`, and false sharing across separate L3s is expensive on this topology, where
under `1x64` each thread owns a contiguous column group), and the cost of the two
barriers per `(jc, pc)` iteration now spanning 16 L3 domains. **Do not change the
rule until one of those is measured** — this is the third time in Phase 4 that a
plausible mechanism for a real effect turned out to be the wrong one.

| # | Assumption | Status |
|---|---|---|
| A28 | The 2-D partition rule and `PACK_WEIGHT` behave at node scale as they do at 8 threads. | **Refuted, but not where predicted.** `PACK_WEIGHT` is fine — it correctly keeps the complex methods off the `N` axis, which costs them 25–32% when forced. What fails is the `panels >= p` early return that precedes it: worth up to 4.3x on the 18 memory-bound `abcijk` cases in the real dtypes at 64 threads. The 2-D machinery itself is vindicated at 1.6–3.4x on the cases it fires on. |

Correctness is unchanged in kind and stronger in coverage: bitwise identity with
serial at every thread count *and* every partition, plus a `Split` expectation on
every threaded case pinning whether it is meant to be 1-D, 2-D, a genuine grid or
clamped to serial — written in units of `MR`/`NR`, since a shape two row panels
deep in `f32` is fourteen with the portable kernels. Green in release and debug,
and across all ten combinations of the three features merged today.

## Phase 4 report, part 9: blocking that transfers

Item 2's grid measures *this* machine, and nothing in it makes the result
transfer, because three constants in `Blocking::derive` are `ccqlin038`'s cache
sizes written down by hand. This removes them (D23): descriptors are probed at
run time — sysfs, then `CPUID`, then built-ins, with the source reported by
`tcbench info` so a number is traceable to a probe rather than to a guess — and
fed to BLIS's analytical model, whose thesis is precisely that this layer needs
no empirical search.

Source: Low, Igual, Smith & Quintana-Ortí, "Analytical Modeling Is Enough for
High-Performance BLIS", ACM TOMS 43(2):12, 2016 (DOI 10.1145/2925987), read as
the actual PDF plus FLAME Working Note #74, not from memory.

**What is the paper's and what is not** — worth stating precisely, because the
two halves have different standing:

* **As written:** `kc` from eq. (4)–(6) — whole L1 ways for the `A` micro-panel
  so the next one evicts the last, one way reserved for the unpacked `C`
  micro-tile, and the 2-way fallback. Equations (1)–(3), which choose `mr`/`nr`,
  are deliberately *not* used: this project measures register blocks (D19).
* **Reconstructed:** §4.3.1 says only that `mc` and `nc` follow "in a similar
  manner" and never writes the inequalities. The reconstruction — reserve the
  ways the streaming operand needs plus one for `C`, give the rest to the
  resident block — is not a guess: it reproduces the paper's own Table III `mc`
  **exactly** for SandyBridge (96), Kaveri (1792) and the TI C6678 (128), two of
  which are asserted as unit tests. The SandyBridge row is BLIS's real shipped
  configuration (`mr=8, nr=4, kc=256, mc=96`) recovered from cache geometry
  alone, which is better evidence than matching a table would be.
* **Does not reproduce:** the Intel Dunnington row (model 1280 against the
  paper's 384). Its `kc` does not follow eq. (4) either — the paper uses 2 ways
  of `A_r` where the formula asks for 3 — so that row appears to carry a
  constraint the paper never states. Recorded rather than fudged.
* **Unvalidated inference:** `nc`, by the same symmetry one level out. The paper
  declines to validate `nc` because three of its four machines have no L3.

### What it predicts here, which is a computation and not a measurement

`cargo run --release --example blocking_model` prints it; safe to run while a
benchmark is in flight. Post-rounding, i.e. what execution would use:

| dtype | method | legacy `mc/kc/nc` | model `mc/kc/nc` |
|---|---|---|---|
| `f64` | – | 264 / 256 / 1536 | 1104 / **106** / 25040 |
| `f32` | – | 384 / 384 / 2048 | 1824 / **128** / 41472 |
| `c64` | planar | 128 / 256 / 768 | 720 / **80** / 16590 |
| `c64` | 1m | 72 / 256 / 768 | 540 / **53** / 25040 |
| `c32` | 1m | 96 / 384 / 1026 | 1216 / **48** / 55296 |

One prediction dominates: **`kc` falls everywhere**, 2.4x in `f64` and 8x in
`c32` 1m, whose fat "1e" panel is what forces it. That converts the `A` sliver
from an L2 stream into an **L1 resident** — exactly the quantity Phase 3 found
the entire method ranking to turn on, and in the direction that favours 3m. `mc`
rises 4–6x (the model gives `A_c` 14 of 16 L2 ways) and `nc` about 20x, which
for most corpus cases means one `jc` block. The packed-`A` footprint stays
equalised across methods (894–914 KiB against legacy's 512–576), so the
1m-versus-planar fairness invariant holds and 1m's `mc` is still the smaller.

### Status — superseded by measurement; read the next subsection

The claim as written in this section was: the **portability** problem is solved —
uniformly decent with no hand tuning on any machine whose cache hierarchy it can
see — while the **optimality** question on `ccqlin038` was untouched and
deliberately so. `A13`'s upper bound on `mc` was noted as absent from the model and
`mc` rising 4–6x flagged as the arm that bound would punish. The model was left as
a *second arm of the pending grid* rather than a change, to be judged end to end in
the shipping configuration (A20).

That judgement has now happened, and it went against the model.

### Measured on the first foreign machine, and it loses

**`worker5040`/`worker5175` (Zen2, AVX2) is exactly the test this model was built
for** — a machine whose cache hierarchy is nothing like the one the legacy
constants were hand-fitted to (512 KiB private L2 against those constants' 512 KiB
`A`-block budget, i.e. they ask for the entire L2; 16 MiB L3 per four cores against
25 MiB per eight). If the model were going to win anywhere it should have won here.

It does not. Measured in the clean single-core regime (part 7), `model` against
`base`:

| | `f32` | `f64` | `c32` planar / 1m / 3m | `c64` planar / 1m / 3m |
|---|---|---|---|---|
| `model` | 0.975 | **1.011** | 0.984 / 0.929 / 0.928 | 0.972 / 0.967 / 0.956 |

**Below `base` in 11 of 12 columns**, by 1.6–7.2% in the complex methods, against a
0.4% floor. Its single gain is `f64` at +1.1%.

**Two things this does and does not mean.** It does *not* refute the paper: BLIS's
own configuration is recovered exactly for SandyBridge, Kaveri and the TI C6678 by
the reconstruction, and those unit tests still pass. What it refutes is the
inference this project drew from it — that an analytically derived blocking is
*therefore* a safe default for **this** engine. The failure is localised: part 7
attributes the whole loss to the model's `kc`, which it predicts shallower for every
complex method (`c64` 1m 256→128, `c32` 1m 384→160), while the grid's pinned-`kc`
arms independently show shallower is worse and deeper is better. The model's `mc`
(up 4–6x) and `nc` (up ~20x) cost nothing measurable, so A13's missing upper bound —
the thing this section flagged as the risk — **was not the problem**. The problem
was the half of the derivation taken straight from the paper.

**The reason is now known, and it is not a bug in the equation** — see part 7,
"Why deeper `kc` wins". The hypothesis first recorded here, that eq. (4)–(6) fails
to carry the per-element real and plane counts through, is **retracted**:
`model_kc` derives its ways ratio from `a_step`/`b_step`, which already include
`a_reals`/`b_reals`, and reproduces its own published prediction exactly when worked
by hand. The equation is correct.

Its *objective* is what does not fit this engine. At the measured optimum,
`kc >= 512`, the `A` micro-panel is 65 KB against a 32 KiB L1 — twice the whole
cache — so the premise that a micro-panel occupies whole L1 ways is simply not where
this engine wants to operate, and **no repair of eq. (4)–(6) can reach that depth.**
What deeper panels actually buy is fewer `pc` passes, hence fewer times `C` is
re-touched: scored against the `kc512` arm's own identical-computation control, the
gain is +8.2% where the pass count halves against +2.4% where it cannot change. The
model has no term for `C` traffic at all.

So the useful repair is not to the model but to the driver: **fuse the `pc` loop**,
which is already a Phase 4 item, and `kc` stops being first-order. That reorders the
"re-derive the model's `kc`" item in "Resume here" — it is no longer the promising
one.

**Recommendation: leave `TENSORCONTRACT_BLOCKMODEL` defaulting to `legacy`.** D23
is unchanged as a decision — the probing, the descriptors and `tcbench info` are
all worth having and are not in question — but the model must not become the
default on this evidence. Flipping it is the user's call and is deliberately not
done in the same commit as this measurement.

| # | Assumption | Status |
|---|---|---|
| A33 | An analytically derived blocking is a safe default on an unseen machine, so it solves portability. | **Refuted on the first unseen machine.** The model loses in 11 of 12 columns on Zen2, by up to 7.2% in the complex methods, against hand-fitted constants belonging to a completely different hierarchy. Localised to its `kc`; its `mc`/`nc` are harmless. "Analytical" bought traceability and cost throughput, and the two were assumed to come together. |

| # | Assumption | Status |
|---|---|---|
| A22 | The model may assume one thread per physical core. | **Assumed, and the project's own rules justify it.** `cores_sharing` divides a level's logical-CPU sharing by the logical CPUs per core, so here a `shared_by = 2` L2 is one core's and a `shared_by = 16` L3 is eight cores'. Oversubscribing hyperthread siblings really would halve a thread's L1/L2, and is not modelled — the measurement rules already treat that configuration as invalid. |
| A23 | Under threading, every cache budget must be divided by the thread count. | **Refuted; it is asymmetric.** The packed `B` panel is *shared*, so `nc`'s L3 budget is a per-socket resource used cooperatively and must **not** be divided. What shrinks `nc` is the per-thread packed `A` blocks, all of which sit in the same L3: the model charges `min(t, cores sharing that L3)` of them. This corrects the limit part 8 recorded as "`NC`'s L3 budget is still charged per core" — in the model arm only. |

## Phase 4/5 interlude: AVX2 kernels, and where the model can be trusted

Dispatch was a single `is_x86_feature_detected!("avx512f")`, so Zen2/Zen3, most
laptops and the *largest partition of the cluster* ran this engine at **scalar**
speed. That is the widest gap between what this document measures and what a
user would experience, and it blocks any prerelease. It is now closed for
`f32`/`f64` and all three complex methods.

`avx512_kernels!` became `simd_kernels!` with the instruction set as a further
macro parameter (D24); the four bodies are not duplicated, and 1m is still
literally `real::<MV,NR>(2*kc, ..)` on both ISAs. Dispatch resolves through a
`OnceLock`-cached `selected_isa()`, and `TENSORCONTRACT_KERNEL` now takes
`scalar|avx2|avx512|auto` (D25).

**The AVX-512 path is unchanged, and that is checked rather than argued:** its
7816 zmm instructions in `examples/kernel_shapes` are byte-identical to the
pre-merge tree, verified independently at merge time by disassembling both. So
every AVX-512 number in this file stands, and tonight's grid still measures the
engine it was designed against.

### The register blocks, and the line between model and guess

16 ymm instead of 32 zmm with the same accumulator-plane counts, so the register
bound binds much harder. Defaults, as `MV x NR` and logical `MR x NR`:

| method | `MV x NR` | f64/c64 | f32/c32 | acc | live |
|---|---|---|---|---|---|
| real | `2 x 6` | `8 x 6` | `16 x 6` | 12 | 15 |
| planar | `1 x 5` | `4 x 5` | `8 x 5` | 10 | 14 |
| 1m | `2 x 6` | `4 x 6` | `8 x 6` | 12 | 15 |
| 3m | `1 x 4` | `4 x 4` | `8 x 4` | 12 | 14 |

`real`/`1m` at `2 x 6` is BLIS's `haswell` `dgemm 6x8` / `sgemm 6x16` with the
operand roles swapped. **These are not measured and must not be quoted as if they
were** (D26). The modelled half — register budget and accumulator count — was
confirmed in the disassembly at zero CPU cost, which is the fourth time the uop
model has been right about *cliffs*. The guessed half is which of the fitting
shapes is fastest; notably `planar 1x6` is better on both accumulator count and
bytes/flop and is rejected only on a single spill, and 3m's `1x4` is chosen by
analogy with the measured AVX-512 winner `1x10` (also load-bound, also lowest
bytes/flop). Alternates are on the menu, so the whole-grid pattern applies
unchanged once AVX2 hardware is available.

### One real result that needed no machine time

`tcbench shapes` under both ISAs, which is exactly what the zero-cost analyses
were built for:

| | AVX-512 | AVX2 |
|---|---|---|
| case-dtype-methods where `Plan::row_block` changes shape | 26 / 392 | 12 / 392 (all `f32`) |
| case-dtype-methods with **no** menu shape that clears the gather path | 81 | **0** |

Every `f64` AVX2 row block on every menu (2, 4, 6, 8, 12) divides 24, and the
corpus rounds every stride-1 extent to a multiple of 24. So Phase 4.1c's
row-block rule is **inert** in `f64` and in all three complex methods on AVX2,
and still has work to do only in `f32`.

### Correctness, and what is not done

Every kernel family the CPU can execute — default shape plus every menu entry,
real plus three complex methods, both precisions — is checked against the
mathematical definition under a plain `cargo test`, so the AVX2 kernels are
*executed* on this AVX-512 machine rather than merely compiled. A second test
pins what must not drift between ISAs: shapes may differ, pack formats, tile
formats and sliver arithmetic may not, because the driver, packing traversal and
write-back are shared and know nothing about the ISA. The `Ukr` contract carried
a second instruction set unmodified, which discharges the Phase 2 design claim
again.

Not done: **no AVX2 performance number of any kind.** Register blocks, the method
ranking, and the cache budgets on an AVX2 machine's hierarchy are all unmeasured
— run `examples/kernel_shapes` then `scripts/phase3-bench.sh` on a Haswell or Zen
box. The `avx2`-without-`avx512` auto-selection branch has also never run on real
hardware, only its forced equivalent.

| # | Assumption | Status |
|---|---|---|
| A24 | The AVX-512 method ranking (planar > 1m > 3m at the operating `kc`) carries over to AVX2. | **Open, and probably not.** 16 ymm forces `MR` down to 4 complex rows in `f64`, which is the L1-resident regime where Phase 3 measured 3m *fastest*. AVX2's ranking is a separate experiment, not a re-run — and note the analytical model (part 9) pushes in the same direction on AVX-512. |
| A25 | Smaller register blocks are purely a cost. | **Refuted, at zero CPU cost.** They are worse for the kernel and better for the write-back, and on AVX2 the write-back side of the trade is simply won: 81 of 392 case-dtype-methods have no AVX-512 menu shape that clears the gather path, against none on AVX2. |

## Phase 5 report, part 1: what preparing v0.1 found

Packaging was expected to be tidying. It was mostly **discovering that the
project's own quality gates had never run**, which is a more useful result and
worth recording in full so the lesson survives.

### CI was not gating anything it claimed to gate

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

### Defects that would have shipped

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

### What the API decision cost and bought

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

### The TAPP conformance suite, and the four gaps it found

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
rejected (§1.6 of `DESIGN.md` lists them as in TAPP's scope, and they are — just
not here), and that `TAPP_ERROR_*` beyond zero are this crate's own numbering,
since upstream `error.h` fixes only `TAPP_SUCCESS`.

| # | Assumption | Status |
|---|---|---|
| A26 | The TAPP layer is thin enough that the engine's own correctness tests cover it. | **Refuted.** Everything the layer can get wrong — datatype-tag dispatch, `intptr_t` handle casts, label arrays read at a rank the info supplies, `beta` on a null `C`, the `status` and `prec` arguments — is invisible from the Rust API and had no test. Four gaps on first contact, one of which aborted the caller's process. **Test an FFI layer as its caller, not as its callee.** |

### Still open before publishing

* **Defaults.** Threading and the blocking model are both off, which is honest
  but means a 64-core machine gets one thread and a foreign cache hierarchy gets
  constants fitted to `ccqlin038`. Flipping either needs the two pending
  measurements, not a decision.
* Publication itself, which is a human step and deliberately not automated.

## Phase 4 report, part 10: taking the two pending measurements to a cluster node

**Status: designed, tooled and pre-registered. Nothing is measured yet.** This
section is written *before* the run so that the node choice, the placement
hypothesis and its accept/reject rule are on the record and cannot be adjusted to
fit whatever comes back. Results go into parts 7, 8 and 8b, and a new part for
the AVX2 calibration.

### The node choice *is* the experiment design

The two pending measurements want an exclusive machine, and `ccqlin038` is a
shared workstation. Rusty has one, but **no Cascade Lake node**, so the choice of
partition picks which question gets answered:

| feature | S:C:T | cores | ISA consequence |
|---|---|---|---|
| `rome` (Zen2) | 2:64:1 | 128 | **no AVX-512** — runs the AVX2 kernels |
| `icelake` | 2:32:1 | 64 | AVX-512, Intel, closest to the reference machine |
| `genoa` (Zen4) | 2:48:1 | 96 | AVX-512 (double-pumped), a third cache topology |

`T:1` in all three: **SMT is off on Rusty's CPU nodes**, so the hyperthread
sibling that forced two Phase 4 retractions is absent rather than merely handled.
That is checked on the node by `scripts/topology.py` rather than believed from
`sinfo`, and `phase4f-threads.sh` keeps its sibling guard regardless — a guard
that trivially passes costs nothing and stops being load-bearing only when
someone proves it is.

**Chosen: `rome`.** Three reasons, in order of weight:

1. **It converts a documented guess into a measurement.** The AVX2 register
   blocks are provisional and explicitly unmeasured (D26): chosen from a register
   budget and a uop model on a machine that cannot execute them competitively.
   `rome` is the largest partition on the cluster and AVX2 is what most users
   get, so this is the widest remaining gap between what this file measures and
   what a user experiences.
2. **Its topology is the one where concurrent placement can work.** Zen2 shares
   one L3 between four cores, so one arm per L3 domain gets a *private* L3 —
   better isolation than `ccqlin038` ever had, where eight cores share 25 MiB. On
   `icelake` the L3 is a whole 32-core socket, which leaves about two usable
   domains and no way to run the 40-job grid concurrently at all.
3. **It tests the analytical model harder than `icelake` would.** The point of
   D23 was that the legacy constants are `ccqlin038`'s cache sizes written down
   by hand. Zen2's are further away than Ice Lake's in the direction that
   matters: a 512 KiB L2 against a legacy `A`-block budget of 512 KiB, i.e. the
   constants ask for the entire L2 where the model would reserve ways for the
   streaming operand. If the model is going to win anywhere, it should win here,
   and if it does not, that is a finding about the model rather than about the
   node.

What this deliberately gives up, stated so it is not discovered later as a
surprise: `rome` cannot say anything about AVX-512 blocking on a second Intel
hierarchy, which is the `icelake` question and remains open. And the blocking
grid measured on `rome` sits on top of register blocks that are themselves
unmeasured until the `shapes` stage runs — so the calibration must be read
before the grid, and the grid's conclusions are conditional on the shapes the
engine actually shipped that day, not on the calibrated ones. Recalibrating and
re-running the grid is a second session, not this one.

### Nothing measured there is comparable to anything above this line

Different machine, different cache hierarchy, different instruction set. Every
ratio must be computed *within* the session, and **the ±1.3% geomean / ±6%
per-case noise floor is `ccqlin038`'s and does not transfer.** The floor is
re-derived on the node from bracketing repeats the scripts already run:
`phase4f-threads.sh` now compares `t1` against its bracketing `t1b` and prints
that first, `validate-placement.sh` runs the same arm solo twice for the same
purpose, and the grid keeps its three `base` repeats. No effect gets quoted
before the floor it is measured against.

### The placement problem, and the hypothesis

An exclusive 128-core node makes both obvious answers wrong. One arm at a time
leaves 127 cores idle, and 40 grid jobs at ~22 min each (on the *reference*
machine; AVX2 will be slower per core) is a day's work. Forty arms at once
corrupts the quantity being measured, because arms sharing an L3 or a memory
controller perturb each other and `NC` is *sized for L3*.

So the placement is treated as a hypothesis: **one measurement thread per L3
domain, every other core in that domain idle, SMT siblings idle, and concurrency
held below the domain count** — 24 of 32 by default on `rome` — because memory
bandwidth and the interconnect stay shared however threads are placed.

The accept/reject rule, fixed now:

* **Accept** if placed-vs-solo lies inside the solo-vs-solo floor on the
  geometric mean, and no case moves by more than the per-case floor.
* **Reject** otherwise, run the grid sequentially, and record the rejection.

Both arms are measured on the full corpus *and* on the `abcijk` family, which is
exactly the 18 cases that contract over `k = 24` — the memory-bound half, i.e.
where bandwidth contention should show up first, bandwidth being the one resource
no placement can privatise. A rejected placement is a result worth writing down:
it tells the next person on a different machine what to expect, and the mechanism
(a private L3 per CCX against a socket-wide L3 on Intel) predicts that the answer
differs by machine.

### What was built for it, all of it zero-CPU

* `scripts/topology.py` — L3 domains, SMT siblings, NUMA and cache descriptors
  from the *allocation's* affinity mask, so it describes the job and not the
  node, plus the placement plan. Emitted as JSON so the runner and this file read
  the same facts.
* `scripts/run-arms.py` — runs independent arms one per L3 domain, and records
  `/proc/stat` occupancy for **every core in each arm's own L3 domain** across
  exactly that arm's window, plus the observed overlap with other arms. The
  reference scripts sample the pinned core and its sibling; this widens that to
  the domain, because on this placement the domain is the unit that has to be
  clean. That recording is what made the earlier retractions detectable.
* `scripts/validate-placement.sh` + `scripts/placement-spread.py` — the
  hypothesis test above, including the spread *between* slots, which prices
  position within the node. A uniform slowdown cancels in the ratios the grid is
  scored on; a position-dependent one does not, and would turn slot assignment
  into a per-arm bias.
* `phase4e-blocking.sh` now runs both regimes through one code path
  (`<cpu>` sequential, `auto` placed) from a single arm list, so the two cannot
  drift apart. `phase4f-threads.sh` derives its cpuset from the allocation
  instead of a hardcoded `0-7`, scales its thread counts to the core count, gives
  each arm exactly as many cores as it has threads (so L3 sharing is a known
  function of `nt` rather than the scheduler's choice), records occupancy over the
  whole allocation as an exclusivity check, and labels the cross-socket arm as the
  separate question it is.
* `scripts/node-session.sh` — stages the session so the ordering constraints are
  enforced rather than remembered: one stage at a time, only `prep` compiles, and
  a guard refuses to start a measurement while any `cargo`/`rustc`/`tcbench` of
  the user's is alive.
* `scripts/placement-verdict.py` — the accept/reject rule **as code**, so an
  unattended run can act on it and, more to the point, so the threshold cannot
  drift once the numbers are visible. Strict reading of the pre-registered rule:
  the per-case bound is the solo pair's *worst* case, not a percentile, because a
  percentile is a knob and a knob chosen after seeing the data is how a
  pre-registered rule stops being one. Every placed replicate must pass, not just
  the one on the reference core, since the grid assigns arms to slots arbitrarily.
* `scripts/rusty-phase4.sbatch` — the whole session unattended. Stage order is
  `prep, shapes, threads, validate, grid`: `shapes` moves ahead of the
  higher-priority `threads` because on an AVX2 node it is the highest-value single
  deliverable and costs twenty minutes, so putting it behind a four-hour stage
  risks the cheap irreplaceable thing for nothing. Stages are independent and a
  failure does not abort the rest.

**The rejection path returns data rather than nothing.** A rejected placement
means the full grid cannot run — sequential is ~23 h on this node and does not fit
a 12 h allocation — so the batch script runs a *scoped* grid sequentially inside
the wall time that is left, in the order `base model base2` then the `kc` family:
the model arm is a whole different derivation rather than a point in the grid and
needs its brackets to mean anything, and `kc` is the first-order parameter. Arms
that do not fit are named in `skipped-arms.txt` and in the log, because a bounded
run that does not say what it dropped reads as complete coverage. A scoped grid is
reported as scoped — offline rule scoring against a partial grid is not the asset
that scoring against a whole one is.

### The prediction, made first

`scripts/thread-width.py` no longer hardcodes `P = 8` — it takes a list, and it
now replays `Plan::partition` (`PACK_WEIGHT` included) instead of only reporting
raw widths, so it predicts what the *rule* will ask for rather than what the shape
allows. Replayed over all 392 case-dtype-methods:

| P | no fill from `M` | rule goes 2-D | leaves threads idle | of which int-div waste | of which a genuine limit | needs `K` |
|---|---|---|---|---|---|---|
| **AVX-512 blocks (the reference machine's)** ||||||
| 8 | 16 | 12 | 4 | 0 | 4 | 0 |
| 16 | 26 | 22 | 12 | 4 | 8 | 0 |
| 32 | 32 | 30 | 16 | 4 | 12 | 0 |
| 64 | 33 | 32 | 18 | 12 | 6 | 0 |
| 128 | 72 | 68 | 49 | 46 | 3 | 0 |
| **AVX2 blocks (what `rome` will run)** ||||||
| 8 | 1 | 1 | 0 | 0 | 0 | 0 |
| 16 | 8 | 6 | 2 | 0 | 2 | 0 |
| 32 | 23 | 15 | 10 | 0 | 10 | 0 |
| 64 | 32 | 26 | 16 | 0 | 16 | 0 |
| 128 | 32 | 32 | 20 | 6 | 14 | 0 |

Four things fall out of it, none of which needed the machine:

1. **`K`-parallelism is never needed, at any thread count up to a whole 128-core
   node, in either ISA.** A21 was argued at 8 threads and the structural argument
   was claimed to generalise; this is that claim checked to 16x the width. A21 is
   now confirmed far outside the regime it was made in.
2. **The 2-D rule will be exercised 5x harder than it was designed against.** It
   fires on 12 of 392 at 8 threads and 32–68 at 128. `PACK_WEIGHT` was priced by
   replay at 2–32 threads and every weight in `[4, 64]` gave the same partition;
   that replay does not cover 64 or 128, so the `=m` / `=n` arms at the top thread
   count are the load-bearing part of the threading run, not the scaling curve.
3. **Most of the apparent shortfall at large `P` is arithmetic, not judgement.**
   `pn = min(p / pm, blocks)` is integer division, so a rule that wants `9 x 14`
   out of 128 threads gets 126 and idles two. Separating that from a genuine
   refusal matters, because only the second kind can show up as a visibly flat
   scaling curve: at `P = 128` on AVX-512 blocks, 46 of the 49 are integer-division
   waste of ≤2%, and just **three** are real.
4. **AVX2's smaller register blocks buy parallel width**, and by a lot: at 8
   threads 16 of 392 case-dtype-methods cannot fill the threads from `M` on
   AVX-512 blocks against **1** on AVX2, because `MR` is 3–6x smaller and the row
   panel count rises accordingly. This is a fresh instance of A25 — smaller
   register blocks are not purely a cost — in a dimension A25 did not consider,
   and it means the `rome` scaling curve should look *better* than the reference
   machine's would, for a reason that has nothing to do with the cores.

The genuinely partition-limited population on AVX2 is one family: `ij-ikl-ljk`
and `ij-kil-lkj`, which cap at `27x1` of 32 threads and `54x2` of 128 (84%).
Those are the curves that must flatten. If any *other* case flattens, that is
contention or a bug, not the partition — which is what makes this prediction
worth having.

Caveat on the table: the `frac of best` column in the full output ranks cases by
throughput measured on `ccqlin038`, used only as a compute-bound proxy for
sorting. It is not a cross-machine performance claim and is regenerated from the
node's own `t1` arm once that exists.

### Result: the placement is clean on the corpus and *not* clean on the memory-bound half

Measured on `worker5040`, 24 concurrent arms one per L3 domain against a solo arm.
The honest comparison is against the **hot** solo arm (`t1b`, same core), because
the reused solo pair straddles the cold-start transient A31 describes:

| | geomean vs hot solo | range over slots |
|---|---|---|
| full corpus, socket 0 slots (12) | **1.0029** | 1.0017–1.0050 |
| full corpus, socket 1 slots (12) | 1.0215 | 1.0181–1.0255 |
| **memory-bound `abcijk` half, all 24** | **0.9684** | 0.9616–0.9769 |

Read in order:

1. **On the corpus as a whole the placement costs 0.3% — nothing.** Twenty-four
   arms running at once, each with a private 16 MiB L3 and three idle cores in its
   domain, measure what a solo arm measures. A27 is confirmed, and it bought a
   **23.7x** speedup: 533 minutes of arm-time in 22.5 minutes of wall clock, which
   is the difference between the grid being affordable and not.
2. **The 2.2% socket split is thermal, not spatial.** The `threads` stage had run
   for two hours entirely on socket 0, so socket 1 was cold when the placed round
   started; its slots read 2.2% *faster*. `placement-spread.py` was built expecting
   memory-system position to be the variable, and on this node it is temperature
   history instead. Confirmed by the memory-bound round twenty minutes later, when
   both sockets had been loaded and the split had shrunk to 0.6%.
3. **On the memory-bound half the placement is rejected, decisively.** 3.2%
   geomean and up to 10% per case, against a solo-pair floor of **0.02%** — the
   cleanest floor in this whole project, from two arms 69 seconds apart. Every one
   of the 24 slots fails. Bandwidth is the resource no placement can privatise, the
   `abcijk` family at `k = 24` is where that bites first, and it does.

**A gap between the rule as written and the rule as coded, which is mine.** Part 10
pre-registered that both the full corpus *and* the `abcijk` family would be
measured, but `rusty-phase4.sbatch` called `placement-verdict.py` with its default
prefix, so only the full-corpus arms were scored and the grid launched placed on an
ACCEPT that had not consulted the memory-bound arms. Scored after the fact with
`--prefix mbp --solo mbsolo`, they REJECT. The verdict script now takes `--solo` so
the rule can be applied to the arm set it was written for; the sequencing error had
already happened.

**What that does to the grid, and what it does not.** The grid is scored on
*ratios between arms*, and every arm ran under the identical 24-way placement, so a
uniform 3.2% penalty on memory-bound cases cancels in `arm / base`. The exposure is
not uniformity but **interaction**: an arm that changes memory traffic suffers
different contention, and that is exactly what the `nc` arms and the `model` arm do
— `model` raises `nc` about 4.7x here. So:

* `kc` and `mc` arms: the penalty is close to uniform and the ratios stand.
* `nc25`, `nc400`, `model`: **confounded at the same order as the effect**, and
  `model` is the single most interesting arm in the grid.

The fix is cheap and does not need the whole grid re-run: measure `base`, `nc25`,
`nc400`, `model`, `base2` sequentially on one core — 10 jobs, ~3.7 h — and compare
those against each other in the clean regime, using `base`/`base2` to tie them back
to the placed run. `scripts/rusty-phase4-seq.sbatch` does exactly that and nothing
else. **Until it has run, do not quote the `nc` or `model` arms from the placed
grid.**

| # | Assumption | Status |
|---|---|---|
| A27 | Concurrent arms placed one per L3 domain measure the same thing as a solo arm. | **Confirmed on the corpus (+0.3%), refuted on the memory-bound half (−3.2%, up to −10%).** Private per-CCX L3 is enough for compute-bound work and irrelevant to bandwidth: `abcijk` at `k = 24` is bandwidth-bound and 24 arms contend. Use the placement for arms that do not change memory traffic; measure the `nc` and `model` arms sequentially. |
| A32 | A session's drift is a single number. | **Refuted; it is a function of how far apart the two arms are.** Same node, same core, same corpus: 0.02% at 69 s apart, 1.1–1.9% at ~1 h, 4.4% across the cold-start transient. Quoting one floor for a whole session is what let a 4.4% cold-start artifact be mistaken for the precision of a repeat. |

### What would invalidate the run

Checked before anything is believed, and each one recorded rather than assumed:
the allocation is genuinely exclusive (`scontrol show job`, once, plus the
per-core occupancy every arm now records for the whole allocation); no compile
overlapped a measurement (`node-session.sh` refuses); the bracketing repeats
agree within the floor derived in the same session; and no number is compared
across machines.

| # | Assumption | Status |
|---|---|---|
| A27 | Concurrent arms placed one per L3 domain measure the same thing as a solo arm. | **Hypothesis, with the test and the accept/reject rule pre-registered above.** Zen2's private-per-CCX L3 is what makes it plausible; shared memory bandwidth is what makes it doubtful, which is why the memory-bound `abcijk` half is measured separately. |
| A28 | The 2-D partition rule and `PACK_WEIGHT` behave at node scale as they do at 8 threads. | **Open, and the reason the threading run is worth more than a scaling curve.** The rule fires on 12 of 392 at 8 threads and 32–68 at 128; `PACK_WEIGHT`'s indifference was only ever replayed to 32. |
| A29 | Parallel width is a property of the contraction. | **Refuted at no CPU cost — it is a property of the contraction *and the ISA*.** Row panels scale as `M / MR`, so AVX2's smaller register blocks give the same corpus 3–6x more of them: 1 of 392 case-dtype-methods short of 8 threads against 16 on AVX-512. |

## Phase 4 report, part 11: the AVX2 register blocks, measured

**First result of the `rome` session (job 6745376, `worker5040`).** The AVX2
register blocks were shipped as an explicit guess (D26) — the register budget and
the accumulator count were modelled, but *which of the fitting shapes is fastest*
was not, because the reference machine cannot execute AVX2 competitively. It can
now be answered.

The machine, from the engine's own probes rather than from `sinfo`:

| | |
|---|---|
| host | `worker5040` (Rusty, `gen` partition, `rome`) |
| CPU | AMD EPYC Zen2, 2 x 64 cores, **SMT off** (`sinfo` `2:64:1`, confirmed on the node) |
| ISA | `avx2 fma`, **no AVX-512** |
| cache | 32 KiB L1d 8-way **private**; 512 KiB L2 8-way **private**; 16 MiB L3 16-way **per 4 cores** |
| domains | 32 L3 domains of 4 cores each |

Two consequences before any number: the **`avx2`-without-`avx512` auto-selection
branch has now executed on real hardware** for the first time (it had only ever
been forced on an AVX-512 machine), and the whole 128-core node has **no
hyperthread sibling**, so the contention that forced two Phase 4 retractions is
structurally absent here rather than merely avoided.

### All four shipped shapes are the measured winners

`examples/kernel_shapes`, packed panels hot, useful GF/s. At the `kc` the driver
actually uses:

| method | shipped `MV x NR` | logical `MR x NR` | GF/s | runner-up | GF/s |
|---|---|---|---|---|---|
| **f64 / c64, `kc = 256`** ||||||
| real | `2 x 6` | 8 x 6 | **53.2** | 8 x 5 | 52.8 |
| planar | `1 x 5` | 4 x 5 | **50.0** | 4 x 4 | 41.9 |
| 1m | `2 x 6` | 4 x 6 | **53.5** | 4 x 5 | 52.9 |
| 3m | `1 x 4` | 4 x 4 | **48.9** | 8 x 2 | 48.1 |
| **f32 / c32, `kc = 384`** ||||||
| real | `2 x 6` | 16 x 6 | **106.6** | 16 x 5 | 105.4 |
| planar | `1 x 5` | 8 x 5 | **104.6** | 8 x 4 | 84.2 |
| 1m | `2 x 6` | 8 x 6 | **107.0** | 8 x 5 | 106.5 |
| 3m | `1 x 4` | 8 x 4 | **112.6** | 16 x 2 | 97.8 |

**Eight for eight.** No change to `cfg_avx2_f64` / `cfg_avx2_f32` is indicated, so
D26's guessed half turns out to have been right — and the *reason* is worth more
than the confirmation: three of the four margins over the runner-up are 0.8–1.6%,
i.e. inside anything this sweep can resolve, while the margin over the *rejected*
shapes is 20–35%. The choice was never between close alternatives; it was between
shapes that fit the register file and shapes that do not.

**The spill has a price and it is now measured.** The interlude recorded that
`planar 1x6` is better on both accumulator count and bytes per flop and was
rejected "only on a single spill". That single spill costs **35% in `f64`** (32.5
against 50.0) and **38% in `f32`** (64.5 against 104.6). Rejecting it was correct
and the margin is not subtle.

### A correction: the register budget is 15 ymm, not 16

The sweep's `!` marker flags `live > 16` — and the data says that threshold is one
register optimistic. Every shape with `live == 16` collapses just as the flagged
ones do:

| shape | `live` | flagged? | GF/s at operating `kc` | fast sibling |
|---|---|---|---|---|
| `real 12x4` f64 | 16 | no | 21.3 | `real 8x6` (15) 53.2 |
| `1m 6x4` f64 | 16 | no | 20.9 | `1m 4x6` (15) 53.5 |
| `planar 4x6` f64 | 16 | no | 32.5 | `planar 4x5` (14) 50.0 |
| `real 16x3` f64 | 17 | yes | 21.5 | — |
| `3m 8x5` f32 | 17 | yes | 45.9 | `3m 8x4` (14) 112.6 |

So the boundary between "fast" and "collapsed" sits at `live <= 15`, not
`live <= 16`: one ymm is not available to the shape, and every shipped default
happens to sit at 14 or 15. **The uop model was right about the existence and
location of a cliff for the fifth time, and wrong about its threshold by exactly
one register** — which is the kind of error a whole-grid sweep is for, and which
would have been invisible from a sweep of the menu alone. The `!` predicate in
`examples/kernel_shapes` should flag `live > 15`; it is an analysis annotation and
not a code path, so it is deliberately **not** changed while this job holds the
node and shares a `target/` directory with it.

### A24: the AVX-512 ranking does not carry over, and 3m leads in single precision

A24 guessed "open, and probably not". At the kernel level it is now measurably
*not*:

| | AVX-512 (Phase 3, end to end) | AVX2 kernel, operating `kc` | AVX2 kernel, `kc = 16` |
|---|---|---|---|
| `f64`/`c64` | planar > 1m > 3m | 1m 53.5 ≈ real 53.2 > planar 50.0 > 3m 48.9 | **3m 54.8** > 1m 53.6 > real 51.6 > planar 44.3 |
| `f32`/`c32` | planar > 1m > 3m | **3m 112.6** > 1m 107.0 ≈ real 106.6 > planar 104.6 | 3m 110.7 > 1m 107.2 > real 105.6 > planar 88.5 |

3m is **first in single precision at the depth the driver uses**, where on
AVX-512 it was last by 8% over the corpus. The mechanism is the one Phase 3
identified and needs no revision: 16 ymm forces `MR` down to 4 complex rows in
`f64`, which is the L1-resident regime where 3m's 25% flop saving is not consumed
by extra plane traffic. Note also that 3m is fastest of all four at `kc = 16` in
*both* precisions, which is Phase 3's finding reproduced on a different ISA.

**What this does not settle.** These are *kernel* numbers: packed panels hot, no
packing, no write-back, no cache blocking, which is exactly what the rest of the
engine is. The corpus-level AVX2 ranking is a separate measurement and is not in
this session. Do not quote the table above as a method ranking — it is the
kernel's contribution to one, and Phase 3's whole lesson was that the ranking is
decided by bytes moved per useful flop across the *driver*, not inside the kernel.

| # | Assumption | Status |
|---|---|---|
| A24 | The AVX-512 method ranking (planar > 1m > 3m at the operating `kc`) carries over to AVX2. | **Refuted at the kernel level, in the direction predicted.** 3m is first in `f32`/`c32` at the operating `kc` and first in both precisions at `kc = 16`; planar is last or next-to-last in every AVX2 column. End-to-end confirmation is not in this session. |
| A30 | The register-block sweep's `live <= 16` budget is the real one. | **Refuted, by one register.** Every `live == 16` shape collapses to 21–33 GF/s beside a 48–53 GF/s sibling at `live <= 15`. All eight shipped defaults sit at 14 or 15, so nothing shipped is affected — but the annotation is wrong and would mislead the next person choosing a shape. |

### The register block is per-microarchitecture, not per-ISA

An Ice Lake session (job 6746817, `worker6016`, `STAGES="shapes threads"`) ran the
same sweep on a *second AVX-512 Intel core*, which had never been done — every
AVX-512 shape in `kernel::x86` was measured on Cascade Lake. Three of the four
`f64`/`c64` defaults are still the winners there. `real` is not, and the reversal is
symmetric:

| shape (`MV x NR`) | acc | live | Cascade Lake @ `kc = 256` | Ice Lake @ `kc = 256` |
|---|---|---|---|---|
| `real 3 x 8` — **shipped** | 24 | 28 | **88.4** | 68.3 |
| `real 3 x 9` | 27 | 31 | 81.3 | **74.6** |

Each machine prefers the other's loser by about 9%: `3x8` wins by 8.7% on Cascade
Lake and loses by 9.2% on Ice Lake. Both machines were swept over the same candidate
set — `3x9` was on Cascade Lake's list and was correctly rejected there — so this is
not a coverage gap, it is a genuine disagreement between two microarchitectures with
the *same ISA and the same 32 registers*.

The mechanism is visible in the depth columns: at `kc = 64` both machines prefer
`3x9` (109.7 against 107.6 on Cascade Lake, 97.7 against 90.9 on Ice Lake), and the
flip happens only at the operating depth. `3x9` carries 27 accumulator registers to
`3x8`'s 24 and `live = 31` against 28 — the A30 budget exactly — so it puts more
pressure on the L1 that also holds the `A` micro-panel. **Ice Lake's L1d is 48 KiB
12-way against Cascade Lake's 32 KiB 8-way**, which is precisely the room `3x9`
needs and does not get on the older core.

**Dispatch selects shapes by ISA only**, so on every Ice Lake machine this engine
currently runs a `real` kernel 9.2% off its own optimum — and Ice Lake is the second
largest CPU partition on this cluster.

Worth noting what makes the fix cheap: the engine *already* probes cache descriptors
for D23, and L1 geometry alone separates these two cores (48 KiB/12-way against
32 KiB/8-way) with no CPUID model table and no new machinery. That is a better
discriminator than a vendor/family list because it names the thing that actually
causes the difference.

Not done, and it should be: `c64`/`c32` were only checked against the shipped
default here, `f32`/`c32` on Ice Lake are unanalysed, and whether a single compromise
shape exists that is within noise of both optima is unknown. None of that needs a new
allocation — `bench-results/worker6016-*/kernel-shapes.txt` and
`bench-results/phase3-kernel-shapes.txt` are both committed and the comparison is
arithmetic.

| # | Assumption | Status |
|---|---|---|
| A34 | Register blocks are a property of the instruction set, so one measurement per ISA is enough (the premise of D19 and of `cfg_avx512_*` / `cfg_avx2_*`). | **Refuted.** Cascade Lake and Ice Lake, same ISA and same 32 registers, disagree by ~9% in opposite directions on the `real` `f64` shape, because Ice Lake's 48 KiB 12-way L1 accommodates an accumulator footprint Cascade Lake's 32 KiB 8-way does not. Shapes are per-microarchitecture; probed L1 geometry is enough to tell these two apart. |

## Phase 5 interlude: making the C surface consumable

Prompted by a concrete external ask — a colleague evaluating this for the
**NDA** C++ array library (TRIQS, Flatiron) — the question was whether a C++
project can use this without shipping a Rust compiler. Answering it honestly
turned up four gaps between "the ABI is correct" and "a C++ project can link
it", all of which were on the Phase 5 packaging list and none of which touched
the engine. See D36–D38.

### The gap that mattered

The TAPP layer had 23 verified `extern "C"` symbols, a conformance suite driving
those symbols, and `abi_layout.rs` re-declaring the whole upstream header so a
dropped export is a link error. What it did not have was **a header**, and
nothing anywhere invoked a C compiler. The correctness of the ABI was thoroughly
established *from Rust*; whether a C caller could actually use it was inference.

That inference held — the C consumer passed on its first real run, including the
complex path and `TAPP_CONJUGATE` — but it held by luck as much as design, and
it would not have survived the first wrong prototype.

### What the toolchain objection was actually worth

Little, once examined, and the examination is the useful part. NDA already
requires CMake ≥ 3.22, a concepts-capable C++ compiler, and HDF5 + MPI + OpenMP
on by default. Against that, `rustc` is marginal — and **a C++20 compiler is the
harder constraint on cluster environments**: the stock compiler on this very
workstation is gcc 8.5, which cannot compile nda at all, while `rustup` installs
a pinned toolchain into `~/.cargo` with no root and no system integration. The
MSRV, 1.89, was released 2025-08-04, one year ago to the day.

So the real blockers are maturity and measurement coverage, not the build.
Recorded so the toolchain argument is not re-litigated.

### A claim corrected in the making

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
| A31 | The shipped header agrees with the library it describes. | **Now tested rather than assumed.** `examples/c-consumer` compiles the header with a C compiler, links the built library and checks numerical results in `f64` and `c64`; CI runs it in both the corrosion and prebuilt modes. Previously no C compiler saw the header at any point. |
| A32 | A Rust panic reaching the C boundary is acceptable because it is memory-safe. | **Rejected as a policy.** Memory-safe but process-fatal, and the engine panics on allocation conditions a caller can hit. D38 converts it to an error code on the three entry points that can raise it. |

## Phases 4 (rest) – 5

In progress. See "Resume here" at the top of this file.
