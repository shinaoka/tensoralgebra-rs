# tensorcontract-rs

A native-Rust, transpose-free dense tensor contraction engine, plus a
systematic benchmark of complex-vs-real tensor contraction across TBLIS, TTGT
and a vendor GEMM ceiling.

```
D[idx_D] = alpha * op_A(A[idx_A]) * op_B(B[idx_B]) + beta * op_C(C[idx_C])
```

No transposed copies, no temporary workspace, no FFI in the hot path. The
algorithm is Matthews' block-scatter-matrix tensor contraction
(arXiv:1607.00291) inside BLIS's five-loop, two-level-packing structure.

> **Status.** The engine is correct and framework-complete (Phase 2 gate met);
> the micro-kernels are still the portable scalar fallback, so absolute
> performance is not yet meaningful. The project's original thesis — that
> complex contraction is where existing engines leave the most on the table —
> was **refuted** by the Phase 1 premise check. See
> [`DECISIONS.md`](DECISIONS.md) for the data and the recommended change of
> direction.

## Headline finding

Contrary to circulating folklore, TBLIS is not weak on complex contractions.
Measured single-core on a Xeon Gold 6244 over 12 TCCG contractions:

| | mean complex efficiency / real efficiency |
|---|---|
| TBLIS f64→c64 | **1.06** |
| TBLIS f64→c64, irregular strides forced | **1.03** |
| TBLIS f32→c32 | **1.15** |
| TTGT f64→c64 | **1.17** |

Efficiency is contraction GF/s divided by same-shape vendor GEMM GF/s. Since a
complex MAC is exactly four real FMAs, the achievable GF/s ceiling is the same
in both domains (measured: `dgemm` 96 GF/s, `zgemm` 96 GF/s), so 1.0 means "no
complex penalty". On raw GF/s for the same shape, complex runs at a **median
1.9x** the rate of real.

The reason is arithmetic intensity: complex does 4x the flops on 2x the bytes,
so packing, indexing and write-back overheads are amortised over twice as much
arithmetic. In exactly the memory-bound shapes where complex was predicted to
be worst, it is best.

The real headroom is **low arithmetic intensity in either domain**: TBLIS's
efficiency against the GEMM ceiling ranges from 0.85 on large compute-bound
contractions down to **0.34** on small-`k` skinny ones.

A methodological note worth carrying forward: the standard TCCG corpus rounds
every stride-1 extent up to a multiple of 24, which divides every plausible
register block, so its block-scatter vectors are **fully regular** and the
irregular gather path never runs. Any claim about awkward strides needs the
`--stress ragged` / `--stress padded` modes added here.

## Layout

| crate | what |
|---|---|
| `crates/tensorcontract` | the engine: data model, index analysis, scatter/block-scatter, packing, micro-kernels, five-loop driver, brute-force oracle |
| `crates/tensorcontract-tapp` | TAPP C-ABI front end (`lib` / `cdylib` / `staticlib`) |
| `crates/tensorcontract-bench` | `tcbench`: correctness and performance harness, TCCG corpus, TBLIS and TTGT baselines |

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

`crates/tensorcontract-tapp` implements the C interface of
[TAPP](https://github.com/TAPPorg/reference-implementation) (arXiv:2601.07827),
so this engine is swappable with TBLIS and cuTENSOR behind one header. Covered:
datatypes `F32`/`F64`/`C32`/`C64`, conjugation on any operand, and TAPP cases
1–4 (contraction, Hadamard/batch indices, repeated indices, isolated input
indices). Case 5 (output broadcasting) is rejected, as TAPP permits.

Note: despite the TAPP paper's claim, TBLIS `develop` (v2.0) has no in-tree
TAPP support, so the benchmark drives it through `tblis_tensor_mult` directly.

## Running the benchmarks

```bash
# One-time: build the TBLIS baseline (see scripts/env.sh for the exact recipe)
export TBLIS_ROOT=/path/to/tblis-install
source scripts/env.sh

cargo build --release -p tensorcontract-bench --features tblis,blas

# correctness: whole corpus vs TBLIS and TTGT, all dtypes
./target/release/tcbench verify --size 4

# the Phase 1 premise check
./target/release/tcbench premise --size 64 --reps 3 --dtype f64,c64 \
    --engines tblis,ttgt --csv bench-results/premise-f64c64.csv

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
```

The engine is checked against a brute-force oracle that shares no code with it,
under both realistic and deliberately tiny cache blocking so that every level
of the five-loop nest and every partial block is exercised on tensors small
enough to verify exhaustively.

## License

MIT OR Apache-2.0.
