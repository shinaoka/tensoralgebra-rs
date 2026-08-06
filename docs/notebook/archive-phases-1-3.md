## Archive: phases 1–3

Closed and unlikely to be reopened. Kept in full because the Phase 3 tables are still the Cascade Lake reference data and the Phase 1 premise check is the project's founding result.

### Phase 1 report: the premise check

**Gate:** self-approved design doc; premise resolved with data; green scaffolded
repo; working harness with baselines wired in.

**Status: gate met. Kill/pivot condition triggered.**

#### Delivered

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

#### The premise check

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

#### Verdict: the observation is real, the explanation is not, and it is already fixed

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

#### What this means for the project

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

#### What the data says the real headroom is

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

#### Kill/pivot condition

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

### Phase 2 report: a correct, framework-complete engine

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

### Phase 2b report: three interchangeable complex methods

**Direction decision.** After the Phase 1 result, the chosen direction is to
keep all three induced-complex methods available and switchable, so that the
comparison can be made properly rather than argued from first principles. This
supersedes the four options listed at the end of the Phase 1 report.

#### What was built

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

*Decisions introduced here: D13, D14, D15, D16 — stated in [Design decisions](#design-decisions).*

#### Correctness

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

#### Measurement, and why it does not yet mean much

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

### Phase 3 report: vectorised micro-kernels

**Gate:** correctness unchanged; single-core throughput against the baselines
and a GEMM roofline; **an honest three-way planar/1m/3m comparison with real
kernels.** All three met. This is the measurement the project exists to
produce.

#### What was built

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

#### Correctness

Unchanged, and checked at three levels:

* `kernel::tests` validates each selected kernel directly against the
  mathematical definition through its own `PackFormat`/`TileFormat`. Every
  kernel here passed on first execution.
* `cargo test --workspace --release` green, and green again under
  `TENSORCONTRACT_KERNEL=scalar`.
* `tcbench verify` — all 49 corpus cases x `f32`/`f64`/`c32`/`c64` x all three
  complex methods, against **both** TBLIS 2.0-dev and TTGT. `f32`/`f64` agree
  exactly (`0.0e0`); `c32` to ~1.3e-7, `c64` to ~2.5e-16.

#### The three-way comparison

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
   L1-resident 3m is the fastest of the three *on this machine*, by roughly the
   margin the flop count predicts. At the `kc = 256` the engine actually uses, it
   is the slowest. (**Overtaken:** A44 shows the L1-resident half of this is a
   Cascade Lake property, absent on Ice Lake. The bytes-per-flop account in the
   next sentence is the part that generalises.) 3m loads *three* planes of both
   operands to save one of four
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

#### Against the baselines

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

#### Irregular strides

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

#### A concrete 2x defect, localised

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

#### Assumptions added

| # | Assumption | Status |
|---|---|---|
| A7 | The three complex methods differ mainly in flop count. | **Refuted.** They differ mainly in bytes moved per useful flop, and that is what decides the ranking at realistic `kc`. Phase 3 added "flop count decides it only when the panels are L1-resident or the contraction is memory-bound"; **A44 has since refuted that qualifier too** — both halves of it are Cascade Lake results that do not transfer. The bytes-per-flop account is the part that survives. |
| A8 | One register block per method is enough. | **Refuted, and it matters.** The best shape depends on the method, the element type and `kc`, and neighbouring shapes differ by 30–50% across the 32-register cliff. |
| A9 | Absolute performance is meaningful now that kernels are vectorised. | Adopted. It was explicitly not meaningful before this phase. |

#### What is *not* done

* **No AVX2 path.** Dispatch is AVX-512 or the scalar fallback. The macro takes
  it without restructuring; deferred to Phase 5's multi-arch work, since the
  reference machine is AVX-512 and the gate is a comparison on it.
* **Blocking is still the Phase 2 heuristic.** `MC`/`KC`/`NC` come from a fixed
  cache-budget rule, never swept. Given that the whole Phase 3 result turns on
  where the `A` sliver lives, `KC` in particular is now known to be a
  first-order parameter rather than a detail. Phase 4.
* **Still single-threaded.**

> **All three have since been addressed** and this list is historical: AVX2
> kernels landed in the Phase 4/5 interlude and their register blocks are
> measured (part 11); `MC`/`KC`/`NC` were swept on two machines and item 2 is
> closed with a negative result (part 7); threading is built and measured on seven
> nodes (parts 8, 8b, 12, 16), though the thread count is still 1 by default.

Raw data: `bench-results/phase3-*.csv`, transcript in
`bench-results/phase3-log.txt`, kernel sweep in
`bench-results/phase3-kernel-shapes.txt`. Reproduce with
`scripts/phase3-bench.sh`.
