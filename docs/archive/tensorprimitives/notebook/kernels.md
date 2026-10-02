## Micro-kernels across instruction sets

One macro body per method across two instruction sets, and what measuring the second one taught about the first.

### Interlude: AVX2 kernels, and where the model can be trusted

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

#### The register blocks, and the line between model and guess

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

#### One real result that needed no machine time

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

#### Correctness, and what is not done

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
| A24 | The AVX-512 method ranking (planar > 1m > 3m at the operating `kc`) carries over to AVX2. | **Refuted at the kernel level** (see the master table). When this was written the reasoning was that 16 ymm forces `MR` down to 4 complex rows in `f64`, which is the regime where Phase 3 measured 3m fastest — but A44 has since shown that measurement to be a Cascade Lake result that does not transfer, so the premise no longer supports the conclusion. AVX2's ranking is a separate experiment, not a re-run. |
| A25 | Smaller register blocks are purely a cost. | **Refuted, at zero CPU cost.** They are worse for the kernel and better for the write-back, and on AVX2 the write-back side of the trade is simply won: 81 of 392 case-dtype-methods have no AVX-512 menu shape that clears the gather path, against none on AVX2. |

### Part 11: the AVX2 register blocks, measured

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

#### All four shipped shapes are the measured winners

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

#### A correction: the register budget is 15 ymm, not 16

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

#### A24: the AVX-512 ranking does not carry over, and 3m leads in single precision

A24 guessed "open, and probably not". At the kernel level it is now measurably
*not*:

| | AVX-512 (Phase 3, end to end) | AVX2 kernel, operating `kc` | AVX2 kernel, `kc = 16` |
|---|---|---|---|
| `f64`/`c64` | planar > 1m > 3m | 1m 53.5 ≈ real 53.2 > planar 50.0 > 3m 48.9 | **3m 54.8** > 1m 53.6 > real 51.6 > planar 44.3 |
| `f32`/`c32` | planar > 1m > 3m | **3m 112.6** > 1m 107.0 ≈ real 106.6 > planar 104.6 | 3m 110.7 > 1m 107.2 > real 105.6 > planar 88.5 |

3m is **first in single precision at the depth the driver uses**, where on
AVX-512 it was last by 8% over the corpus. The mechanism proposed at the time was
Phase 3's: 16 ymm forces `MR` down to 4 complex rows in `f64`, which is the
L1-resident regime where 3m's 25% flop saving was thought not to be consumed by
extra plane traffic; 3m being fastest of all four at `kc = 16` in *both*
precisions read as Phase 3's finding reproduced on a different ISA.

> **Overtaken, 2026-08-05.** A44 refutes the premise: the L1-resident advantage is
> a Cascade Lake result and 3m does not win at `kc = 16` on Ice Lake in either
> precision. The AVX2 numbers below stand as measurements on that Zen2 node; the
> *explanation* borrowed from Phase 3 does not, and the ranking is per
> microarchitecture rather than per ISA (A34, A56).

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

#### The register block is per-microarchitecture, not per-ISA

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

The full comparison, shipped shape against each machine's own best at the operating
`kc`, from the two committed sweeps:

| section | method | shipped | CL @ shipped | CL best | IL @ shipped | IL best | IL / best |
|---|---|---|---|---|---|---|---|
| f64/c64 | real | `24x8` | **88.4** | `24x8` 88.4 | 68.3 | `24x9` 74.6 | **0.916** |
| f64/c64 | planar | `16x6` | 102.8 | `16x6` 102.8 | 108.4 | `16x6` 108.4 | 1.000 |
| f64/c64 | 1m | `12x8` | 91.1 | `12x8` 91.1 | 69.7 | `12x8` 69.7 | 1.000 |
| f64/c64 | 3m | `8x10` | 87.8 | `8x10` 87.8 | 63.8 | `8x10` 63.8 | 1.000 |
| f32/c32 | real | `48x8` | 179.9 | `48x8` 179.9 | 150.3 | `48x9` 172.6 | **0.871** |
| f32/c32 | planar | `32x6` | 195.7 | **`32x5` 210.9** | 221.8 | `32x6` 221.8 | 1.000 |
| f32/c32 | 1m | `32x6` | 175.4 | `32x6` 175.4 | 136.4 | `24x8` 151.3 | **0.902** |
| f32/c32 | 3m | `16x10` | 194.9 | `16x10` 194.9 | 134.5 | `16x10` 134.5 | 1.000 |

Three of eight shipped shapes are **9–13% off on Ice Lake**, and `real` wants `NR+1`
in *both* precisions — a coherent signature, not scatter.

#### A Phase 3 defect this turned up, on the reference machine

The `CL best` column has an entry that is not the shipped shape: **`planar` `f32`/`c32`
should be `32x5` (210.9) and ships as `32x6` (195.7) — 7.8% off, on `ccqlin038`,
in the default complex method.** This is not an Ice Lake finding; it has been true
since Phase 3.

Phase 3's own sweep output says so in as many words —
`bench-results/phase3-kernel-shapes.txt` contains

```
best per method at kc = 384:
  planar   MV=2 NR=5     210.9 GF/s
```

— while `cfg_avx512_f32` ships `planar = [(2, 6), (3, 4), (1, 12)]`, in which `(2, 5)`
does not appear at all, not even as an alternate. `32x5` also wins at `kc = 64`
(210.3 against 205.4) and loses only at `kc = 16`, so it is not a single-depth fluke.

**Why it was probably chosen wrong is the interesting part.** The doc comment on
`cfg_avx512_f32` bolds `32x6`'s bytes-per-flop (**0.20**, against `32x5`'s 0.231) and
records 195.7 beside it. So the shape appears to have been selected on the
bytes-moved-per-useful-flop model — the project's own central mechanism from Phase 3
— *over* the measured throughput sitting in the same output file. That is the
failure mode this document has now recorded four times in other guises (A16 and part
6: write-back regularity "has now pointed the wrong way three times... it is a *gate*
on a change, never an objective"). D19 says this project measures register blocks;
here it modelled one and the sweep disagreed.

**Not a fix, a finding.** `NR` changes the `jr` loop count and the packed-`B` sliver
geometry, so a 7.8% kernel margin is not a 7.8% corpus margin, and `32x5` is absent
from the menu so no runtime switch can A/B it.

**And it cannot be added to the menu — a structural finding, not an oversight.** The
row-block menu is keyed by `MR`: `cplx_config_at` dispatches on `mr` alone, and
`row_block_menus_are_well_formed` asserts no `MR` appears twice, precisely because a
duplicate would make the later entry unreachable. `(2, 5)` shares `MV = 2` with the
shipped `(2, 6)`, hence the same `MR`, so **the menu has no way to express an
`NR`-only alternate** and `TENSORCONTRACT_ROWBLOCK=idx=` can never reach it.

Two routes out, and choosing is a design decision rather than a fix:

* **Swap the default and measure build-to-build.** Two lines, but it gives up A15's
  preference for runtime switches, and a build-to-build diff already produced one
  wrong sign in Phase 4.
* **Key the menu by position rather than by `MR`**, so entries carry `(MV, NR)` and
  `idx=` selects an index — which is what `idx=` already implies. Touches `configs!`,
  `cplx_config_at`, `row_blocks`, `Plan::row_block` and the well-formedness test. Not
  large, but it changes hot-path config selection and should not ride along with a
  measurement.

So A35 is **documented and untestable end-to-end**, which is a worse state than
merely unfixed and is recorded as such. Worth noting how it was found: entirely inside
committed Phase 3 raw output, months later, at no machine cost. That is the return on
keeping the sweeps.

| # | Assumption | Status |
|---|---|---|
| A34 | Register blocks are a property of the instruction set, so one measurement per ISA is enough (the premise of D19 and of `cfg_avx512_*` / `cfg_avx2_*`). | **Refuted.** Cascade Lake and Ice Lake, same ISA and same 32 registers, disagree by up to 13% on three of eight shapes, and `real` wants `NR+1` in both precisions — Ice Lake's 48 KiB 12-way L1 accommodates an accumulator footprint Cascade Lake's 32 KiB 8-way does not. Shapes are per-microarchitecture; probed L1 geometry separates these two with no CPUID table. |
| A35 | The shipped register blocks are the ones the Phase 3 sweep selected. | **Refuted for one of eight, and now testable.** `planar` `f32`/`c32` ships `32x6` where the sweep's own output names `32x5`, 7.8% faster at the operating `kc`. The bolded bytes-per-flop in the config's doc comment suggests it was chosen by model over measurement, which is the same error A16 records for write-back regularity. It was unreachable until the menu was re-keyed by position (D43); `32x5` is now the last planar `f32` entry and `TENSORCONTRACT_ROWBLOCK=idx=3` selects it end to end. Still unfixed on purpose: verify before changing, because kernel margin ≠ corpus margin. |
