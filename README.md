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

> **Status: prerelease.** The engine is correct, vectorised and benchmarked
> against TBLIS on a single machine. It is *not* yet tuned for machines other
> than that one, and the table under [Performance](#performance-and-what-is-actually-measured) says exactly which claims are measured and which are not.
> The API may still change.
>
> What you get today, by hardware:
>
> | | micro-kernels | tuned? |
> |---|---|---|
> | x86-64 with AVX-512F | AVX-512, per method | register blocks **measured** on a Cascade Lake Xeon; cache blocking hand-set for that machine and not yet swept |
> | x86-64 with AVX2+FMA | AVX2, per method | correct; register blocks **modelled, not measured** |
> | anything else | portable scalar | correct; slow by design |
>
> Threading and an analytical cache-blocking model are implemented and **off by
> default**, because neither has been measured yet — see
> [Switches](#switches). Turning them on is one call or one environment
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
every stride-1 extent up to a multiple of 24, which divides every plausible
register block, so its block-scatter vectors are **fully regular** and the
irregular gather path never runs. Any claim about awkward strides needs the
`--stress ragged` / `--stress padded` modes added here.

## Performance, and what is actually measured

Single core, Xeon Gold 6244 (Cascade Lake, AVX-512), full 49-case TCCG corpus at
64 MiB nominal tensor size, planar method, GF/s counting 2 flops per real MAC and
8 per complex one. Raw CSVs in [`bench-results/`](bench-results/):

| dtype | min | median | geomean | max |
|---|---|---|---|---|
| `f32` | 4.8 | 61.4 | 62.3 | **161.2** |
| `f64` | 1.5 | 35.4 | 34.4 | **70.7** |
| `c32` | 8.1 | 93.7 | 89.2 | **168.4** |
| `c64` | 4.0 | 49.5 | 48.7 | **83.5** |

For scale, OpenBLAS on the same core reaches ~96 GF/s for both `dgemm` and
`zgemm` on large square shapes. The corpus deliberately spans compute-bound and
badly memory-bound contractions, which is why the min and max differ by 20x or
more; the single worst case (`adbjc-cbdka-kj`) is memory-bound for every engine
measured, TBLIS included.

**The counterintuitive part, and the project's main technical result:** complex
throughput is *higher* than real on the same shapes — 49.5 against 35.4 GF/s in
double, 93.7 against 61.4 in single, at the median. A complex MAC is four real
FMAs on twice the bytes, so complex contraction has **twice the arithmetic
intensity** and amortises packing and indexing overhead better. Complex is not
the weak spot; low arithmetic intensity is, in either domain.

**On the three methods:** planar wins on the corpus geometric mean in both
precisions, by single-digit percent, and the ranking **inverts on memory-bound
shapes** where 3m's 25% flop saving wins. The deciding quantity is bytes moved
per useful flop, not flop count and not in-register shuffles. Exact margins moved
during optimisation work and are being re-measured, so this README does not quote
a number for them — see the Phase 3 and Phase 4 reports in
[`DECISIONS.md`](DECISIONS.md), which record the mechanism and every measurement.

**Not measured, and so not claimed:** anything on AVX2 hardware; anything
threaded; anything with the analytical cache-blocking model enabled; any machine
whose cache hierarchy differs from the one above while the default (hardcoded)
blocking is in force.

## Switches

Every performance-relevant choice is reachable at run time, so an A/B is a
process restart rather than a rebuild. None of them changes results.

| variable | effect |
|---|---|
| `TENSORCONTRACT_COMPLEX` | `planar` \| `1m` \| `3m` — the complex method |
| `TENSORCONTRACT_THREADS` | thread count, default **1**. Results are bitwise identical at any count |
| `TENSORCONTRACT_KERNEL` | `auto` (default) \| `scalar` \| `avx2` \| `avx512` — pin the instruction set |
| `TENSORCONTRACT_BLOCKMODEL` | `legacy` (default) \| `model` — derive cache blocking from probed cache descriptors instead of hardcoded constants |

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
    --engines tblis,ttgt --csv bench-results/premise-f64c64.csv

# the same against the last stable TBLIS release (note the feature and ABI)
cargo build --release -p tensorprimitives-bench --features tblis13,blas
TBLIS_ROOT=/path/to/tblis-1.3.0-install \
    ./target/release/tcbench premise --size 64 --dtype f64,c64 --engines tblis

# same, with the gather path actually exercised
./target/release/tcbench premise --size 64 --stress ragged --dtype f64,c64

# full corpus sweep
./target/release/tcbench sweep --size 32 --csv bench-results/sweep.csv
```

Raw results from the runs quoted above are committed under `bench-results/`.

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
