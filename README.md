# tensorprimitives-rs

A native-Rust, transpose-free dense tensor contraction engine with **three
interchangeable complex methods** (planar, 1m, 3m) behind one switch, plus a
systematic benchmark of complex-vs-real tensor contraction across TBLIS, TTGT
and a vendor GEMM ceiling.

```
D[idx_D] = alpha * op_A(A[idx_A]) * op_B(B[idx_B]) + beta * op_C(C[idx_C])
```

No transposed copies, no temporary workspace, no FFI in the hot path. The
algorithm is Matthews' block-scatter-matrix tensor contraction
(arXiv:1607.00291) inside BLIS's five-loop, two-level-packing structure.

> **Status: prerelease.** The engine is correct, vectorised, and benchmarked
> against TBLIS on three machines. Tuning is a different matter, and the section
> on [what is and is not tuned](#what-is-tuned-and-what-is-not) says exactly
> which claims are measured, on which microarchitecture, and which are not
> claimed at all. The API may still change.
>
> What you get today, by hardware:
>
> | | micro-kernels | tuned? |
> |---|---|---|
> | x86-64 with AVX-512F | AVX-512, per method | register blocks **measured on Cascade Lake**, and they do not transfer — an Ice Lake machine runs three of the eight shipped shapes 9–13% off its own optimum. Cache blocking is fitted to the same Cascade Lake workstation and has now been swept: it is near optimal there |
> | x86-64 with AVX2+FMA | AVX2, per method | register blocks **measured on Zen2**, and all eight shipped shapes were confirmed as the winners |
> | anything else | portable scalar | correct; slow by design |
>
> Threading and an analytical cache-blocking model are implemented and **off by
> default**, and both have now been measured — see [Switches](#switches).
> Threading reaches 10.5x (`f64`) and 14.2x (`c64` 3m) on 32 Ice Lake cores and
> 7.5–7.9x on 64 Zen2 cores; it is off because it has only been measured on
> *large* contractions, and threads are spawned per call rather than pooled. The
> blocking model is off because it was measured and **lost**, on the first
> machine it was meant to help. Turning either on is one call or one environment
> variable, and results are bitwise identical either way.
>
> The project's original thesis — that complex contraction is where existing
> engines leave the most on the table, because interleaved storage compounds
> with scatter/gather — did not survive the Phase 1 premise check. The observed
> weakness is real against the *released* TBLIS but has a much more mundane
> cause, and is already fixed in TBLIS 2.0. See the headline finding below and
> [`DECISIONS.md`](DECISIONS.md) for the data.

## Headline finding

**Which TBLIS you measure changes the answer by a factor of five.**

Efficiency here is contraction GF/s divided by same-shape vendor GEMM GF/s.
Because a complex MAC is exactly four real FMAs, the achievable GF/s ceiling is
the same in both domains (measured: `dgemm` 96 GF/s, `zgemm` 96 GF/s), so the
number below — **each engine's own complex efficiency divided by its own real
efficiency** — is 1.0 when that engine treats complex data as well as it treats
real data. Single-core, Xeon Gold 6244, 12 TCCG contractions:

| | complex ÷ real efficiency, **within the same engine** |
|---|---|
| **TBLIS v1.3.0** — latest *stable* release | **0.215** |
| TBLIS 2.0-dev | **1.06** |
| TBLIS 2.0-dev, irregular strides forced | **1.03** |
| TBLIS 2.0-dev, f32→c32 | **1.15** |
| TTGT (OpenBLAS) | **1.17** |

**This is not a speed comparison between engines, and it must not be read as
one.** An engine can score well by being uniformly bad. TTGT has the *highest*
ratio in the table and is by a wide margin the *slowest* of the three: on those
same 12 cases its median `c64` throughput is 17.4 GF/s against TBLIS 2.0's 42.4.
Its ratio is high because its **real** path is worse still — 6.8 GF/s median in
`f64`, where the transpose-transpose-GEMM-transpose overhead dominates a small
GEMM — and complex, with twice the arithmetic intensity, amortises that overhead
better. That is the same mechanism as everywhere else in this project, and here
it produces a flattering ratio for the slowest engine. The ratio answers exactly
one question: *is complex penalised relative to real, in this engine?*

Against the released TBLIS, complex tensor contraction really is roughly **5x
less efficient** than real. But the cause is not the subtle one usually
proposed (interleaved storage compounding with scatter/gather). The diagnostic:
v1.3.0's complex throughput is nearly **flat across shapes** — 4.1 to 9.1 GF/s,
a 2.2x spread — while its real throughput spans 6.2 to 48.4 GF/s. A
memory-bound effect would track shape; a flat ceiling means one fixed kernel is
the bottleneck.

`src/configs/*/config.hpp` in v1.3.0 confirms it. `TBLIS_CONFIG_GEMM_UKR` takes
`(float, double, scomplex, dcomplex)`, and **Sandy Bridge is the only
configuration that fills the complex slots**. On Haswell, Zen, Skylake-X and
KNL, TBLIS 1.x runs complex on the generic templated fallback while real gets
hand-tuned BLIS assembly. Block scatter is fully regular (`regA = 1.00`) in
every case measured, so the gather path is not involved at all.

TBLIS 2.0 fixes this by adopting BLIS as its core framework, which brings the
1m induced method. Complex immediately regains full shape sensitivity
(11.2–82.9 GF/s, matching real) and reaches parity or better. So the
opportunity is real in the wild today, already closed upstream, and not
evidence for any algorithmic claim.

Complex is not intrinsically disadvantaged: it does 4x the flops on 2x the
bytes, i.e. **twice the arithmetic intensity**, so packing and indexing
overheads amortise better. Once a real complex kernel exists, the memory-bound
shapes where complex was predicted to be worst are where it looks best.

The remaining headroom is **low arithmetic intensity in either domain**: TBLIS
2.0's efficiency against the GEMM ceiling ranges from 0.85 on large
compute-bound contractions down to **0.34** on small-`k` skinny ones.

For orientation, since the table above deliberately says nothing about it: on
those same 12 cases, measured in the same run, median `c64` throughput was
**42.5 GF/s for this engine, 42.4 for TBLIS 2.0-dev and 17.4 for TTGT**; in
`f64`, 30.7 / 21.8 / 6.8. Twelve cases are not the corpus and these predate the
Phase 4 work, so the 49-case numbers below are the ones to quote — but parity
with TBLIS 2.0 on complex is roughly where this engine sits.

A methodological note worth carrying forward: the standard TCCG corpus rounds
every stride-1 extent up to a multiple of 24. That is regular only at a register
block which *divides* 24 — true of the blocks TBLIS uses in the comparison above,
and **not** true of this engine's `f32`/`c32` blocks (`MR` 16, 32, 48), where
`reg_a` falls below 1.0 on 43% of case-dtype-methods. So the corpus does exercise
the gather path here, periodically; what it cannot produce is *aperiodic*
irregularity, which is what the `--stress ragged` / `--stress padded` modes added
here are for. Any claim about awkward strides needs one of them named.

## Performance, and what is actually measured

Single core, **Intel Ice Lake-SP, AVX-512** (2 x 32 cores, SMT off; 48 KiB
12-way L1d, 1280 KiB 20-way L2 per core, 48 MiB L3 per socket — the geometry the
engine itself probed), full 49-case
TCCG corpus at 64 MiB nominal tensor size, planar method, GF/s counting 2 flops
per real MAC and 8 per complex one. **Measured 2026-08-04/05** on an exclusive
cluster node, against **TBLIS 2.0-dev** (`develop` @ `555320c`) and **TBLIS
v1.3.0** in the same runs. Raw CSVs and provenance in
[`bench-results/worker6156-icelake/`](bench-results/README.md):

| dtype | min | median | geomean | max |
|---|---|---|---|---|
| `f32` | 11.0 | 74.8 | 77.2 | **168.8** |
| `f64` | 7.1 | 44.7 | 43.7 | **71.6** |
| `c32` | 25.5 | 121.0 | 113.9 | **179.3** |
| `c64` | 15.7 | 62.0 | 63.3 | **91.1** |

The corpus deliberately spans compute-bound and badly memory-bound contractions,
which is why min and max differ by an order of magnitude; the worst cases are
memory-bound for every engine measured, TBLIS included.

**How well this is known.** The whole set was run **twice**, in separate
allocations three hours apart, and the two runs agree across all twenty
dtype × engine columns to **0.997–1.003**. Within each run, a repeat arm 2.5 h
from its original reproduces it to 0.998–1.001 with **0 of 980** case points
outside ±6%. A discarded warm-up arm precedes everything, and every arm records
occupancy for its whole L3 domain — all of them came back with no co-tenants.

**Against the baselines**, median over the 12-case premise set at 200 MiB, same
run, same core, everything single-threaded:

| dtype | this engine | TBLIS 2.0-dev | TTGT (OpenBLAS) | engine ÷ TBLIS 2.0 |
|---|---|---|---|---|
| `f64` | 46.5 | 30.8 | 14.0 | **1.51x** |
| `c64` | 64.0 | 48.8 | 25.9 | **1.31x** |
| `f32` | 81.1 | 45.6 | 25.9 | **1.78x** |
| `c32` | 123.2 | 88.3 | 49.2 | **1.40x** |

Over the full 49-case corpus the engine is 1.95–2.20x TTGT in every dtype.

> **What these numbers are not.** They are Ice Lake, and the previous set was
> Cascade Lake, so **the difference between them is not the improvement the
> optimisation work bought.** Two different things changed at once. The Phase 4
> gains are real and separately measured — the write-back orientation fix is
> +12–17% of corpus geometric mean in all four dtypes on the reference machine,
> the orientation rule up to 1.48x per case — but attributing any part of the
> table above to them would be wrong. Nothing in this repo permits comparing a
> number from one machine with a number from another.
>
> The engine is also **handicapped** in this table: its register blocks were
> chosen on Cascade Lake and three of the eight are 9–13% off on Ice Lake, and its
> complex-method ranking does not transfer here either. It wins these columns
> while running shapes picked for a different microarchitecture.

**The counterintuitive part, and the project's main technical result:** complex
throughput is *higher* than real on the same shapes — 62.0 against 44.7 GF/s in
double, 121.0 against 74.8 in single, at the median. A complex MAC is four real
FMAs on twice the bytes, so complex contraction has **twice the arithmetic
intensity** and amortises packing and indexing overhead better. Complex is not
the weak spot; low arithmetic intensity is, in either domain. This is a
within-run ratio, so it holds where the absolute numbers are not comparable: it
came out 1.449 (`c64`/`f64`) and 1.474 (`c32`/`f32`) here, against 1.416 and
1.432 on Cascade Lake — the same result on two microarchitectures.

**On the three methods, and this changed.** Planar wins the corpus geometric mean
in both precisions on both AVX-512 machines measured. But the *ordering below it*
and the much-quoted inversion on memory-bound shapes are **Cascade Lake results
that do not transfer.** On Cascade Lake 3m was the fastest of the three on
memory-bound shapes, where its 25% flop saving pays; on Ice Lake 3m is last in
every column, wins 0 of 49 cases, and sits at 0.694 (`c64`) and 0.744 (`c32`)
against planar where Cascade Lake had it at 0.956 and 0.921. That is partly
confounded with the wrong register blocks above and cannot be fully separated
without an Ice Lake shape sweep. The deciding quantity is still bytes moved per
useful flop — the mechanism is intact — but **do not treat the ranking, or the
inversion, as a property of the engine.** On AVX2 the kernel-level ordering
differs again, putting 3m first in `f32`/`c32`; that is a kernel measurement with
panels packed and hot, not a corpus ranking, and the corpus-level AVX2 comparison
has not been made.

## What is tuned, and what is not

The most useful thing measurement established here is where the tuning stops.

**Register blocks are per-microarchitecture, not per-ISA** — which was assumed
the other way round, and dispatch still selects them by instruction set alone.
Cascade Lake and Ice Lake, same ISA and same 32 registers, disagree by up to 13%
on three of the eight shipped shapes, each machine preferring the other's loser
by about 9%. Ice Lake's 48 KiB 12-way L1d accommodates an accumulator footprint
Cascade Lake's 32 KiB 8-way does not. So on any Ice Lake machine this engine
currently runs a `real` kernel 9.2% off its own optimum. The AVX-512 blocks are
measured on Cascade Lake, the AVX2 blocks on Zen2 — where all eight were
confirmed as the winners — and neither set is a claim about anything else.

**Cache blocking is fitted to the reference machine, and swept.** `KC` is
first-order: it decides whether the `A` sliver is an L1 resident or an L2 stream,
which is what the whole complex-method ranking turns on. But there is no
few-percent win left in `MC`/`KC`/`NC` there — `kc = 384` is the measured optimum
and the shipped 256 is close to it, `MC` is a plateau a sixteenfold range moves
by at most 3%, and a deeper coupled setting that looked like +3% on two machines
failed its end-to-end A/B. That item is closed, on evidence. It says nothing
about a machine with a different hierarchy.

**The analytical blocking model is off by default because it lost**, not because
it is unmeasured. It was built to solve exactly the portability problem above,
and on the first unseen machine it was worse in 11 of 12 columns by up to 7.2%.
On the reference machine it costs 14–34%. Its `kc` sinks it on one machine and
its `mc` on the other, so both halves of the derivation are wrong — in different
places. The probing and the cache descriptors it introduced are worth having and
are still used; the default is not changing.

**Threading is off by default, and not for lack of data.** Scaling is measured
on two topologies and is strongly topology-dependent: Zen2 saturates by 16–32
threads and declines at 64, while Ice Lake reaches 10.6x (`f64`) and 15.3x
(`c64` 3m) on 32 cores and is still climbing. One thing is still missing and one has been fixed: threads are spawned per call rather than pooled, which is the whole story below ~1 MiB where 64 threads run 1.2–10x *slower* than serial; but `Plan::partition` now knows how many L3 domains the thread set spans, and gating its early return on that — rather than removing it — is the default (`TENSORCONTRACT_PARTITION=domain`). That is worth 1.13 corpus geometric mean at 64 Zen2 threads and provably changes nothing on a machine with one L3 per socket. Results are bitwise identical at every thread count and every partition, so none of this is ever a correctness or accuracy decision.

**One known defect, stated because it ships.** In `planar` `f32`/`c32` the
register block is `32x6`, where the Phase 3 sweep's own output names `32x5` as
**7.8% faster** at the operating `kc`. The shipped shape appears to have been
picked by a bytes-per-flop model over the measurement sitting next to it. It is
not fixed here — but it *is* now testable, which it was not when it was found. The row-block menu was keyed by `MR`, and `32x5` shares its `MR` with the shipped `32x6`, so the menu could not express an `NR`-only alternate and no runtime switch could reach it. Re-keying the menu by position fixed that: `TENSORCONTRACT_ROWBLOCK=idx=3` selects `32x5` end to end. It stays unfixed on purpose, because a kernel margin is not a corpus margin — `NR` changes the loop count and the packed sliver geometry — so the corpus effect has to be measured before the default moves. It was found inside committed raw output months after the fact, at no machine cost, which is the argument for committing raw output.

**Not measured, and so not claimed:** absolute throughput on any machine other
than the Cascade Lake reference; the corpus-level method ranking on AVX2;
anything on non-x86 hardware.

## Switches

Every performance-relevant choice is reachable at run time, so an A/B is a
process restart rather than a rebuild. None of them changes results.

| variable | effect |
|---|---|
| `TENSORCONTRACT_COMPLEX` | `planar` \| `1m` \| `3m` — the complex method |
| `TENSORCONTRACT_THREADS` | thread count, default **1**. Results are bitwise identical at any count, so this is never a correctness or accuracy decision. Off by default for the two reasons under [what is tuned](#what-is-tuned-and-what-is-not), and because it keeps every committed single-core number reproducible |
| `TENSORCONTRACT_KERNEL` | `auto` (default) \| `scalar` \| `avx2` \| `avx512` — pin the instruction set. A pinned ISA the CPU lacks falls back to scalar, so `avx2` is how the AVX2 kernels get exercised on an AVX-512 machine |
| `TENSORCONTRACT_BLOCKMODEL` | `legacy` (default) \| `model` — derive cache blocking from probed cache descriptors instead of hardcoded constants. Measured on a foreign machine and **worse in 11 of 12 columns**, so `legacy` is the default on evidence |
| `TENSORCONTRACT_PARTITION` | `domain` (default) \| `legacy` — which rule apportions the threads over the output. `domain` gates the "row axis fills the threads, so use it" shortcut on how many L3 domains the thread set spans. Measured: it changes **nothing at all** on a machine with one L3 per socket (the identical partition on 392 of 392 corpus cases) and is worth **1.13 geometric mean over the corpus at 64 threads** on a chiplet machine, up to 3.0x on individual memory-bound cases. Single-threaded callers never reach it. `legacy` is the ungated rule; `m`, `n` and `<pm>x<pn>` pin the partition outright, for A/B measurement |
| `TENSORCONTRACT_DEEPEN` | `on` — coupled deepening: `kc = 512` for `f64` real geometry with `mc`/`nc` re-derived at that depth. **Off by default because it failed its end-to-end A/B**, and kept only as a record of the experiment. It looked like +3% on two machines; both grids' `base` arm was 1–2% slow, so every `arm/base` ratio was inflated identically and the agreement was a shared artefact rather than a replication. Corrected, the treatment is ≈0.977 |

Plus the levers that exist so a fast path can be A/B-tested at run time rather
than as a diff between two builds — `TENSORCONTRACT_ORIENT` (`none` | `swap` |
`legacy`, the row/column orientation), `_ROWBLOCK` (`base` | `auto` | `mr=<n>` |
`idx=<i>`, the micro-tile row block — `idx=3` reaches the `32x5` shape the known
defect below is about), `_WRITEBACK` (`gather` forces the general scatter path),
and `_MC`/`_KC`/`_NC` with their `_MC_PCT`/`_NC_PCT`/`_KC_COUPLE` relatives for
cache blocking. None of them affects correctness; they exist for measurement, and
a build-to-build diff already produced one wrong sign here.

`Plan::with_complex_method`, `Plan::with_threads` and `Plan::with_blocking` are
the programmatic equivalents, and take precedence.

Threading partitions the *output* — row strips of micro-panels by column groups,
never the contraction index — so every output element has exactly one owning
thread accumulating over the full contraction in the original order. That is why
the result is bitwise identical to serial at every thread count, and it is
asserted in the test suite rather than assumed.

## Three complex methods, one engine

Complex contraction can be induced from real arithmetic in several ways, and
which one wins depends on shape, element type and machine. Rather than pick
one, the engine implements three and lets you switch:

| [`ComplexMethod`] | A / B reals per complex elt | FMAs per k per tile | accumulator planes |
|---|---|---|---|
| `Planar` (default) | 2 / 2 | `4*MR*NR` | 2 |
| `OneM` — BLIS's 1m, what TBLIS 2.x uses | **4** / 2 | `4*MR*NR` | 2 (as `2*MR x NR` real) |
| `ThreeM` — Karatsuba | 3 / 3 | **`3*MR*NR`** | 3 |

```rust
let plan = Plan::new(a, b, None, d)?.with_complex_method(ComplexMethod::ThreeM);
```
```bash
TENSORCONTRACT_COMPLEX=1m ./target/release/tcbench sweep --dtype c64
./target/release/tcbench premise --engines planar,1m,3m --dtype f64,c64
```

The three share the *entire* engine — index analysis, scatter machinery,
five-loop driver, write-back scatter. `driver.rs` contains no branch on the
method; everything a method changes is declared on the `Ukr` it selects
(`a_pack`, `b_pack`, `tile_fmt` and the sliver widths). Cache blocking is
derived from the reals a method actually packs, not from `size_of::<Element>()`,
so 1m automatically gets a smaller `MC` and every method sees the same L2
budget — otherwise the comparison would be quietly rigged.

`ThreeM` trades accuracy for its 25% flop saving: its error bound is relative to
`|Ar||Br| + |Ai||Bi|` rather than the complex magnitudes, so it can lose
relative accuracy under cancellation. It is opt-in for that reason, and the
test suite gives it a correspondingly looser tolerance rather than hiding it.

## Layout

| crate | what |
|---|---|
| `crates/tensorcontract` | the contraction engine: data model, index analysis, scatter/block-scatter, packing, micro-kernels, five-loop driver, brute-force oracle |
| `crates/tensortranspose` | dedicated transpose kernels — **planned, not yet written** |
| `crates/tensorprimitives-tapp` | TAPP C-ABI front end (`lib` / `cdylib` / `staticlib`) |
| `crates/tensorprimitives-bench` | `tcbench`: correctness and performance harness, TCCG corpus, TBLIS and TTGT baselines |
| `julia/TensorPrimitives` | Julia wrapper over the TAPP surface, including a `TensorOperations.jl` backend |
| `packaging/yggdrasil` | the BinaryBuilder recipe that produces `tensorprimitives_tapp_jll` |

`tensorprimitives` is the project and the repository, not a crate: with one
primitive implemented, a facade re-exporting it would be indirection rather than
abstraction, and it can be added later without breaking anyone. Depend on the
operation crate you need.

## Usage

```rust
use tensorcontract::{contract, parse_einsum, Layout, TensorView, TensorViewMut};

let (ia, ib, id) = parse_einsum("ik,kj->ij").unwrap();
let la = Layout::col_major(&[2, 3]);
let lb = Layout::col_major(&[3, 2]);
let ld = Layout::col_major(&[2, 2]);
let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
let b = vec![1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0];
let mut d = vec![0.0f64; 4];

contract(
    1.0,
    TensorView::new(&a, &la, &ia),
    TensorView::new(&b, &lb, &ib),
    0.0,
    None,
    TensorViewMut::new(&mut d, &ld, &id),
)?;
```

Build a `Plan` once and reuse it for repeated contractions of the same shape.
Operands may be conjugated (`.conj()`), strides may be negative or zero, labels
may repeat within a tensor (diagonals), and indices present in only one input
are summed over.

Supported element types: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Any
other real scalar type — extended precision, dual numbers for forward-mode AD —
works by implementing `kernel::KernelSet` with the provided generic scalar
kernels.

## TAPP

`crates/tensorprimitives-tapp` implements the C interface of
[TAPP](https://github.com/TAPPorg/reference-implementation) (arXiv:2601.07827),
so this engine is swappable with TBLIS and cuTENSOR behind one header. Covered:
datatypes `F32`/`F64`/`C32`/`C64`, conjugation on any operand, and TAPP cases
1–4 (contraction, Hadamard/batch indices, repeated indices, isolated input
indices). Case 5 (output broadcasting) is rejected, as TAPP permits.

Note: despite the TAPP paper's claim, TBLIS `develop` (v2.0) has no in-tree
TAPP support, so the benchmark drives it through `tblis_tensor_mult` directly.

Also note a silent ABI break between TBLIS releases: `type_t` swaps
`TYPE_DOUBLE` and `TYPE_SCOMPLEX` between 1.3 and 2.0, so mixing headers and
libraries yields plausible wrong numbers with no error. Select the ABI with the
`tblis13` cargo feature; the harness self-checks at startup and aborts on
mismatch.

### From C or C++

The header is shipped rather than fetched — `crates/tensorprimitives-tapp/include/tapp.h`
versions with the implementation, because upstream TAPP has no releases and no
tags. `examples/c-consumer` is a CMake project that consumes it three ways
(corrosion, a prebuilt library, an installed prefix), and CI compiles and runs all
three. `examples/c-consumer/README.md` has the details, including the link flags
cargo will not set for you.

```bash
cargo build --release -p tensorprimitives-tapp
crates/tensorprimitives-tapp/install.sh --prefix=/opt/tapp
cc myprog.c $(PKG_CONFIG_PATH=/opt/tapp/lib/pkgconfig \
              pkg-config --cflags --libs tensorprimitives-tapp) -lm
```

### From Julia

`julia/TensorPrimitives` wraps the same ABI, including a `TensorOperations.jl`
backend, so an existing `@tensor` expression routes through this engine by adding
one keyword:

```julia
@tensor backend = TAPPBackend() C[i, j] := conj(A[i, k, l]) * B[l, k, j]
```

Neither it nor its JLL is registered yet. `packaging/yggdrasil/build_tarballs.jl`
builds the JLL and `julia/TensorPrimitives/README.md` says how to run it locally.

## Running the benchmarks

```bash
# One-time: build the TBLIS baseline (see scripts/env.sh for the exact recipe)
export TBLIS_ROOT=/path/to/tblis-install
source scripts/env.sh

cargo build --release -p tensorprimitives-bench --features tblis,blas

# correctness: whole corpus vs TBLIS and TTGT, all dtypes
./target/release/tcbench verify --size 4

# the Phase 1 premise check
./target/release/tcbench premise --size 64 --reps 3 --dtype f64,c64 \
    --engines tblis,ttgt --csv /tmp/premise-f64c64.csv

# the same against the last stable TBLIS release (note the feature and ABI)
cargo build --release -p tensorprimitives-bench --features tblis13,blas
TBLIS_ROOT=/path/to/tblis-1.3.0-install \
    ./target/release/tcbench premise --size 64 --dtype f64,c64 --engines tblis

# same, with the gather path actually exercised
./target/release/tcbench premise --size 64 --stress ragged --dtype f64,c64

# full corpus sweep
./target/release/tcbench sweep --size 32 --csv /tmp/sweep.csv
```

Or run the whole comparison set the way this project runs it — a discarded
warm-up arm, both TBLIS ABIs from prebuilt binaries so nothing compiles mid-run,
and two repeat arms that derive the session's own noise floor instead of
importing one:

```bash
TBLIS_ROOT_2X=../baselines/tblis-2.0-install \
TBLIS_ROOT_13=../baselines/tblis-1.3.0-install \
  scripts/compare-bench.sh prep                              # the only compile
  scripts/compare-bench.sh bench-results/$(hostname -s)-$(scripts/arch-label.sh)
```

**It wants an exclusive machine for about three and a half hours**, and that is
not pedantry: pinning is not enough, because the pinned core's SMT sibling shares
L1d and L2, which is what every cache-blocking measurement here turns on. Two
Phase 4 conclusions had to be corrected after re-measuring on a quiet machine.
[`scripts/README.md`](scripts/README.md) indexes the rest, including the Slurm
wrappers and the offline analyses that cost no CPU at all.

Raw results from every run quoted anywhere in this repo are committed under
[`bench-results/`](bench-results/README.md), one `PROVENANCE.txt` per directory
saying which machine and date produced it.

## Testing

```bash
cargo test --workspace --release                       # includes 1000 randomised
TENSORCONTRACT_KERNEL=scalar cargo test --workspace --release   # portable path
TENSORCONTRACT_KERNEL=avx2 cargo test --workspace --release      # AVX2 kernels
```

`TENSORCONTRACT_KERNEL` takes `scalar`, `avx2`, `avx512` or `auto` (the
default, meaning the widest the CPU has). Pinning an instruction set narrower
than the CPU's is how the AVX2 path is exercised on an AVX-512 machine; the
kernel-contract tests in `kernel/mod.rs` additionally run *every* kernel family
the CPU supports on every `cargo test`, whatever is selected.

The engine is checked against a brute-force oracle that shares no code with it,
under both realistic and deliberately tiny cache blocking so that every level
of the five-loop nest and every partial block is exercised on tensors small
enough to verify exhaustively.

## License

MIT OR Apache-2.0.
