# tensorprimitives-rs

A native-Rust, transpose-free dense tensor contraction engine for real and
complex tensors of any rank, with **three interchangeable complex methods**
(planar, 1m, 3m) behind one switch.

```
D[idx_D] = alpha * op_A(A[idx_A]) * op_B(B[idx_B]) + beta * op_C(C[idx_C])
```

No transposed copies, no temporary workspace, no FFI in the hot path. The
algorithm is Matthews' block-scatter-matrix tensor contraction
(arXiv:1607.00291) inside BLIS's five-loop, two-level-packing structure.

| crate | what |
|---|---|
| [`tensorcontract`](crates/tensorcontract) | the contraction engine |
| [`tensorprimitives-tapp`](crates/tensorprimitives-tapp) | TAPP C-ABI front end (`lib` / `cdylib` / `staticlib`) |
| [`tensorprimitives-bench`](crates/tensorprimitives-bench) | `tcbench`: the correctness and benchmark harness (not published) |
| [`julia/TensorPrimitives`](julia/TensorPrimitives) | Julia wrapper, with a `TensorOperations.jl` backend |

`tensorprimitives` is the project and the repository, not a crate. Depend on the
operation crate you need.

> **Status: prerelease.** The engine is correct, vectorised on x86-64 and
> AArch64, and benchmarked against TBLIS on four machines. The API may still
> change.
>
> **Tuning is per-microarchitecture**, and the honest account of what is measured
> — on which machine, and what is not claimed at all — is
> [`docs/results.md`](docs/results.md). Register blocks are measured on Cascade
> Lake (AVX-512), Zen2 (AVX2) and an M3 Max (NEON); on other hardware they are
> reasonable, not optimal. The NEON menu is the only one measured with an error
> bar, and six of its eight winners were ties — which is also what the AVX-512 and
> AVX2 menus' single arms cannot tell you. Cache blocking is fitted to Cascade
> Lake everywhere, including on AArch64. Threading and an analytical
> cache-blocking model are implemented and **off by default**, both for measured
> reasons.
>
> Anything without AVX-512, AVX2 or NEON runs a portable scalar path: correct,
> and slow by design — though "slow" turned out to be a good deal less slow than
> this project assumed before measuring it, because LLVM vectorises that path
> unasked. [`docs/results.md`](docs/results.md) has the figure.

## Usage

```rust
use tensorcontract::{contract, parse_einsum, Layout, TensorView, TensorViewMut};

let (ia, ib, id) = parse_einsum("ik,kj->ij")?;
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
may repeat within a tensor (diagonals), and indices present in only one input are
summed over.

Supported element types: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Any other
real scalar type — extended precision, dual numbers for forward-mode AD — works
by implementing `kernel::KernelSet`; `kernel::scalar` is a worked example.

### Many small contractions

If you have a batch of independent contractions rather than one big one, the
batch is the right parallel axis: `batch::contract_batched` pays **one** spawn
set for the whole batch instead of one per contraction, and each item runs
serially inside. Every item's bounds are checked before any item runs, so a batch
containing one bad item writes to no output at all — which a loop over
`Plan::run` cannot give you, and a reason to prefer it even at one thread.

```rust
use tensorcontract::batch::{contract_batched, BatchItem};

let mut items: Vec<BatchItem<'_, f64>> = /* one per contraction */ vec![];
contract_batched(&mut items)?;
```

## Three complex methods, one engine

Complex contraction can be induced from real arithmetic in several ways, and
which one wins depends on shape, element type and machine. Rather than pick one,
the engine implements three and lets you switch:

| `ComplexMethod` | A / B reals per complex elt | FMAs per k per tile | accumulator planes |
|---|---|---|---|
| `Planar` (default) | 2 / 2 | `4*MR*NR` | 2 |
| `OneM` — Van Zee's 1m | **4** / 2 | `4*MR*NR` | 2 (as `2*MR x NR` real) |
| `ThreeM` — Karatsuba | 3 / 3 | **`3*MR*NR`** | 3 |

```rust
let plan = Plan::new(a, b, None, d)?.with_complex_method(ComplexMethod::ThreeM);
```

The three share the *entire* engine — index analysis, scatter machinery,
five-loop driver, write-back scatter. `driver.rs` contains no branch on the
method; everything a method changes is declared on the `Ukr` it selects. Cache
blocking is derived from the reals a method actually packs, not from
`size_of::<Element>()`, so 1m automatically gets a smaller `MC` and every method
sees the same L2 budget — otherwise the comparison would be quietly rigged.

`ThreeM` trades accuracy for its 25% flop saving: its error bound is relative to
`|Ar||Br| + |Ai||Bi|` rather than the complex magnitudes, so it can lose relative
accuracy under cancellation. It is opt-in for that reason, and the test suite
gives it a correspondingly looser tolerance rather than hiding it.

**Which is fastest is not a property of the engine.** Planar wins the corpus on
both AVX-512 machines measured; everything below that changes with the
microarchitecture. See [`docs/results.md`](docs/results.md) before quoting a
ranking.

## Performance

**This README quotes no throughput numbers, on purpose.** Every number this
project has is machine-specific, and the two comparable sets were taken on
different microarchitectures — so what the optimisation work bought end to end is
still unmeasured. Publishing a headline figure without that context is the exact
mistake the measurement rules here exist to prevent.

[`docs/results.md`](docs/results.md) has the numbers, each with its machine, its
date, its noise floor and its raw data — and says plainly what is *not* claimed.
[`docs/refuted.md`](docs/refuted.md) has the ideas that were measured and lost,
each with what would reopen it; it is the fastest way to find out whether an
obvious-looking optimisation here has already failed.

## Switches

Every performance-relevant choice is reachable at run time, so an A/B is a
process restart rather than a rebuild. **None of them affects correctness.**
Output is bitwise identical across every switch that only divides work — threads,
partition, pool, orientation, row block and the blocking overrides — because each
output element has one owning thread accumulating over the full contracted extent
in the original order. The two that change the arithmetic are not bitwise
identical and cannot be: `TENSORCONTRACT_KERNEL`, because the scalar path's
`acc += a * b` is two roundings where a vector kernel's FMA is one, and
`TENSORCONTRACT_COMPLEX=3m`, which computes three products where planar computes
four.

| variable | effect |
|---|---|
| `TENSORCONTRACT_COMPLEX` | `planar` (default) \| `1m` \| `3m` — the complex method. `3m` is **not bitwise identical** to the other two, see above |
| `TENSORCONTRACT_THREADS` | thread count, default **1**. Bitwise identical at any count, so never a correctness or accuracy decision |
| `TENSORCONTRACT_KERNEL` | `auto` (default) \| `scalar` \| `avx2` \| `avx512` \| `neon` — pin the instruction set. A pinned ISA the CPU lacks falls back to scalar. **Not bitwise identical across arms**, see above |
| `TENSORCONTRACT_PARTITION` | `domain` (default) \| `legacy` — which rule apportions threads over the output; or `m` \| `n` \| `<pm>x<pn>` to pin it |
| `TENSORCONTRACT_POOL` | `on` — reuse parked threads instead of spawning per call. Off by default: measured on two machine classes, and it does not transfer |
| `TENSORCONTRACT_BLOCKMODEL` | `legacy` (default) \| `model` — cache blocking from probed cache descriptors instead of hardcoded constants. `legacy` is the default on evidence |
| `TENSORCONTRACT_ORIENT` | `none` \| `swap` \| `legacy` — pin the row/column orientation |
| `TENSORCONTRACT_ROWBLOCK` | `base` \| `auto` \| `mr=<n>` \| `idx=<i>` — pin the micro-tile row block |
| `TENSORCONTRACT_WRITEBACK` | `gather` — force the general scatter write-back |
| `TENSORCONTRACT_MC` / `_KC` / `_NC` | override cache blocking, with `_MC_PCT` / `_NC_PCT` / `_KC_COUPLE` relatives |

Why each default is what it is, and what it cost to find out, is in
[`docs/results.md`](docs/results.md). `Plan::with_complex_method`,
`Plan::with_threads` and `Plan::with_blocking` are the programmatic equivalents
and take precedence.

## TAPP

`crates/tensorprimitives-tapp` implements the C interface of
[TAPP](https://github.com/TAPPorg/reference-implementation) (arXiv:2601.07827),
so this engine is swappable with TBLIS and cuTENSOR behind one header. Covered:
datatypes `F32`/`F64`/`C32`/`C64`, conjugation on any operand, and TAPP cases 1–4
(contraction, Hadamard/batch indices, repeated indices, isolated input indices).
Case 5 (output broadcasting) is rejected, as TAPP permits.

Note that despite the TAPP paper's claim, TBLIS `develop` (v2.0) has **no
in-tree TAPP support**, so the benchmark drives it through `tblis_tensor_mult`
directly.

Note also a silent ABI break between TBLIS releases: `type_t` swaps
`TYPE_DOUBLE` and `TYPE_SCOMPLEX` between 1.3 and 2.0, so mixing headers and
libraries yields plausible wrong numbers with no error. Select the ABI with the
`tblis13` cargo feature; the harness self-checks at startup and aborts on
mismatch.

### From C or C++

The header is shipped rather than fetched —
`crates/tensorprimitives-tapp/include/tapp.h` versions with the implementation,
because upstream TAPP has no releases and no tags. `examples/c-consumer` is a
CMake project that consumes it three ways (corrosion, a prebuilt library, an
installed prefix), and CI compiles and runs all three.

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

Neither it nor its JLL is registered yet.

## Documentation

**To use the library**, the API documentation is the place to start — it carries
the index notation, the supported cases, the complex methods and the stability
tiers, and nothing below is needed to call the engine:

| | |
|---|---|
| [`tensorcontract`](https://docs.rs/tensorcontract) | the engine's API documentation |
| [`tensorprimitives-tapp`](https://docs.rs/tensorprimitives-tapp) | the TAPP C ABI, and its coverage of the specification |
| [`examples/contract.rs`](crates/tensorcontract/examples/contract.rs) | runnable: a batch index, a reduction, a diagonal, complex with plan reuse |

**To audit the measurements**, or before proposing a performance idea:

| | |
|---|---|
| [`docs/results.md`](docs/results.md) | what is measured, on which machine, and what is not claimed |
| [`docs/refuted.md`](docs/refuted.md) | ideas measured and lost, each with what would reopen it |
| [`docs/open-questions.md`](docs/open-questions.md) | what is still unknown |
| [`docs/design.md`](docs/design.md) | the architecture |
| [`docs/decisions.md`](docs/decisions.md) | every decision and standing assumption |
| [`docs/notebook/`](docs/notebook/README.md) | the full measurement narrative |
| [`bench-results/`](bench-results/README.md) | raw CSVs for every number, one `PROVENANCE.txt` per directory |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | building, testing, and running the benchmarks |

## License

MIT OR Apache-2.0.
