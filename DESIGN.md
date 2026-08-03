# Design: a transpose-free tensor contraction engine in Rust

Phase 1 design document, updated after the premise check and the
three-methods decision.

Status: **complete.** The working thesis it was written to serve — that planar
complex packing is a research win over BLIS's 1m — did not survive the Phase 1
premise check (see [`DECISIONS.md`](DECISIONS.md) §Phase 1 report). The project
has been re-aimed: rather than advocating for planar, the engine now implements
**all three induced-complex methods behind one switch** so they can be measured
against each other and against TBLIS on genuinely equal footing
(§3.4, §3.5, and `DECISIONS.md` §Phase 2b).

---

## 1. Literature review

### 1.1 Block-scatter-matrix tensor contraction (BSMTC)

Matthews, *High-Performance Tensor Contraction without Transposition*
(arXiv:1607.00291, SIAM J. Sci. Comput. 2018) observes that a tensor
contraction

```
D[idx_D] = A[idx_A] * B[idx_B]
```

is a matrix multiplication once you agree to address the operands through
**scatter vectors**. Group the index labels into classes (free-in-A, free-in-B,
contracted), linearise each class as a mixed-radix multi-index, and precompute
`rscat[i] = sum_l i_l * stride_l`. Then matrix element `(i, j)` of an operand
lives at `base + rscat[i] + cscat[j]` and the whole BLIS GEMM machinery applies
unchanged, with the scatter lookup absorbed into the packing routines.

The refinement that makes it fast is the **block** scatter vector: for each
aligned run of `MR` consecutive scatter entries, record whether those entries
form an arithmetic progression and with what stride. When they do — which is
most of the time, because tensors are dense and index folding makes them more
so — the packing loop uses ordinary strided (usually unit-stride) loads instead
of a gather. Only genuinely irregular blocks pay for the scatter table.

Reported results: TBLIS stays within 5–10% of the equivalent GEMM, versus up to
a 50% penalty for transpose-based approaches, with up to 17x speedups over
TTDT on awkward shapes.

### 1.2 BLIS: five loops and two-level packing

The GEMM skeleton is BLIS's: `NC` blocking on the columns of B, `KC` on the
contraction, `MC` on the rows of A, with B packed into L3-resident panels and A
into L2-resident panels, and a register-blocked `MR x NR` micro-kernel at the
bottom. Everything in this project reuses that structure verbatim; the only
tensor-specific change is that the two packing routines read through scatter
vectors, and the write-back does too.

### 1.3 Induced complex methods: 4m, 3m, 1m

Van Zee & Smith (*3m and 4m methods*, ACM TOMS 2017) and Van Zee (*1m method*,
SIAM J. Sci. Comput. 2020, and `blis6_toms_rev0`) address a real problem: a
BLAS-like framework does not want to write and maintain a separate assembly
micro-kernel for every complex datatype. Their answer is to *induce* complex
GEMM from a real micro-kernel by changing the packed format.

* **4m** performs four real GEMMs (`ar*br`, `ai*bi`, `ar*bi`, `ai*br`). It
  underperforms, mainly because it makes four passes over the packed data and
  four updates to `C`.
* **3m** uses a Karatsuba-style identity to get away with three real GEMMs
  (a 25% flop saving) at a small cost in the error bound.
* **1m** performs a *single* real GEMM. A complex `m x k` matrix `A` is packed
  in **"1e" format**: each complex element becomes the real `2 x 2` block
  `[[re, -im], [im, re]]`, so the packed panel is `2m x 2k` reals. `B` is packed
  in **"1r" format**: each element becomes the `2 x 1` column `[re; im]`. One
  real GEMM of those two produces the complex product in column-interleaved
  form. Total real flops are `2 * (2m)(2k)(n) = 8mnk`, exactly the complex flop
  count — no waste.

The 1m paper reports performance generally competitive with hand-written
complex kernels. The structural cost that motivated this project's thesis is
that **1m's packed A panel holds four reals per complex element instead of
two** — twice the store traffic in the packing pass and twice the L2 footprint.

### 1.4 GETT and cuTENSOR

Springer & Bientinesi, *Design of a High-Performance GEMM-like Tensor-Tensor
Multiplication* (arXiv:1607.00145) take the other route: keep an explicit,
fused packing/permutation step (GETT), generate code per contraction, and
select among candidate loop orders. cuTENSOR is the GPU descendant. The `tccg`
repository's `benchmark/benchmark.py` defines the benchmark corpus this project
uses (see §5).

### 1.5 Scatter-copy as a bottleneck

*Strassen's Algorithm for Tensor Contraction* (arXiv:1704.03092) explicitly
flags the packing/scatter-copy step as a limiter in TBLIS-style engines: once
the micro-kernel is at peak, moving data through the scatter is what is left.
This is the observation the thesis was built on — correctly, as far as it goes;
what it does not imply is that *complex* data suffers disproportionately.

### 1.6 TAPP

*Tensor Algebra Processing Primitives* (arXiv:2601.07827), reference
implementation at <https://github.com/TAPPorg/reference-implementation>,
BSD-3. A small C interface: opaque `intptr_t` handles for library, executor,
tensor info, plan and status; `TAPP_create_tensor_product(...)` builds a plan
from four tensor infos plus four `int64_t*` label arrays plus four element ops;
`TAPP_execute_product(...)` runs it. Datatypes are `TAPP_F32/F64/C32/C64`
(complex interleaved) plus `F16/BF16`; element ops are `TAPP_IDENTITY` and
`TAPP_CONJUGATE`.

Scope, taken from the paper: contractions with free, contracted and Hadamard
indices (case 1–2), repeated indices within a tensor (case 3), isolated indices
in the *inputs*, i.e. reductions (case 4). Isolated indices in the *output*
(broadcasting, case 5) are explicitly not required. Mixed storage precisions
are permitted; a `TAPP_prectype` requests a computational precision.

**Verified against the headers, not the paper.** The paper does not print the C
prototypes; the ones above were read from `api/include/tapp/*.h` in the
reference implementation. One claim in the paper did *not* check out: TBLIS is
described as "the first library to officially support the TAPP interface
directly within its own repository", but neither `master` nor `develop` of
`devinamatthews/tblis` (v2.0) contains any TAPP source. This project therefore
benchmarks TBLIS through its native `tblis_tensor_mult` C API and implements
TAPP itself.

---

## 2. Rust ecosystem survey and build-vs-reuse

| Layer | Options considered | Decision |
|---|---|---|
| N-d arrays | `ndarray`, `RSTSR` | **Neither in core.** The engine needs base pointer + extents + strides, nothing more. A dependency here would leak into the public API and the TAPP ABI for no benefit. `Layout` is 20 lines. |
| Complex scalar | `num-complex` | **Reuse.** `#[repr(C)]`, matches `TAPP_C32/C64`, C99 `_Complex` and `std::complex` layout exactly, and is the ecosystem standard. Verified interleaved by a layout test. |
| SIMD abstraction | `pulp`, `macerator`, `std::simd`, raw `core::arch` | **Raw `core::arch` with runtime dispatch.** `std::simd` (`portable_simd`) is still unstable on the pinned toolchain (1.97). `pulp` is mature and powers `faer`, but a register-blocked micro-kernel needs explicit control over which values live in which registers, and an abstraction layer mostly gets in the way. The `Ukr` function-pointer indirection means a `pulp` or `macerator` backend can be added later without touching the driver. |
| GEMM micro-kernels | `gemm`, `matrixmultiply`, `microgemm`, `faer` | **Own kernels.** All of these expose *matrix* multiplication, not a panel-panel kernel operating on externally packed buffers, and none has a planar-complex path. `matrixmultiply` does now ship AVX-512 `cgemm`/`zgemm` kernels; it is the right thing to benchmark against, not to build on. |
| Threading | `rayon`, `std::thread::scope` | **`std::thread::scope`** when Phase 4 arrives. BLIS-style parallelism wants static partitioning with a *shared* packed-B panel; work-stealing fights that. No dependency needed. |
| Path optimisation | `opt-einsum-path` | Out of scope. This engine executes one binary contraction; contraction *ordering* is a caller concern. |
| Baselines | `tblis`/`tblis-ffi` crates, hand-rolled FFI | **Hand-rolled, ~100 lines.** The benchmark must control which TBLIS is measured — version, branch, and above all which BLIS configuration its kernels were built for (`skx` here). The published crates vendor their own build. Struct layout is pinned by a test against a `sizeof`/`offsetof` probe. |
| Benchmark stats | `criterion`, custom | **Custom.** Criterion's model is many fast iterations of a cache-resident routine. These measurements are 0.1–5 s each on 64–200 MiB working sets; the useful estimator is best-of-N after a warm-up, and the useful output is a CSV of GF/s. |

---

## 3. Engine design

### 3.1 Data model

An operand is a base pointer plus `Layout { extents, strides }`, strides in
elements, permitted to be negative or zero — the same model as TAPP's tensor
info and C++23 `layout_stride`. No ownership, no reshaping, no views: the
caller owns memory.

### 3.2 Index analysis (`plan.rs`)

Every distinct label is classified:

| class | in A | in B | in D | role |
|-------|------|------|------|------|
| `M` | yes | no | yes | GEMM rows |
| `N` | no | yes | yes | GEMM columns |
| `K` | yes | yes | no | contracted |
| `H` | yes | yes | yes | Hadamard / batch |

Two design choices here are worth calling out.

**Reductions need no workspace.** A label present only in `A` (TAPP case 4,
`sum_i A[...,i,...]`) is folded into class `K` with `stride_B = 0`. The B
scatter vector then simply repeats, and the contraction
`sum_{p,i} A[m,p,i] B[p,n]` computes `sum_p (sum_i A[m,p,i]) B[p,n]` — which is
the right answer. The natural alternative, a pre-reduction pass into a
temporary, would allocate; TAPP itself warns that case 4 "may require unbounded
amounts of workspace". This costs nothing because the block-scatter code treats
a zero stride as a perfectly regular access pattern. That in turn is why
irregular blocks are flagged with a dedicated `i64::MIN` sentinel rather than
with `0` as in some other implementations — `0` is a meaningful stride here.

**Diagonals are a stride sum.** A label repeated within one tensor selects its
diagonal; extents must agree and the strides add, after which it is one mode.
Applied per tensor before classification, so the rest of the engine never sees
a repeated label.

Output-only labels (case 5, broadcast) are rejected, as TAPP permits.

**Folding.** Within a class, labels are ordered fastest-varying first and
adjacent ones merged when `s_next == s_prev * extent_prev` holds in *every*
operand. This is what collapses an ordinary matrix multiply to one axis per
class, and what pushes the regular-block fraction toward 1. Ordering heuristic:
`M`, `N`, `H` by increasing `|stride|` in `D`, `K` by increasing `|stride|` in
`A`. Rationale: the output update is the one access packing cannot hide, so `D`
gets first claim on contiguity. This is a heuristic, and a tuning knob.

### 3.3 Scatter and block scatter (`scatter.rs`)

Full scatter vectors are materialised at plan time: `O(M + N + K)` `i64`s,
tens of kilobytes for the entire TCCG corpus. Block-scatter vectors are derived
at execution time because they depend on `MR`/`NR`, which depend on the element
type; they are `O(M/MR + N/NR)` and cost nothing.

Scatter vectors are element-type *independent*, so one `Plan` executes for any
element type — which is what lets the TAPP layer store a single dtype-erased
plan.

### 3.4 Planar complex packing (`pack.rs`) — the intended contribution

One routine packs both operands. For `A` the "vector axis" is `M` and for `B`
it is `N`; both stream over `K`. Output layout, `P` planes:

```
out[s * (vr*kc*P) + p * (vr*P) + plane * vr + t]
```

so for a complex operand each k-step emits `[re_0..re_{VR-1}, im_0..im_{VR-1}]`.
In BLIS terms this is **"1r" format for both operands**, driven by a genuinely
complex micro-kernel — as against 1m's "1e" for A and "1r" for B driven by a
real one.

The argument was: a tensor contraction has to make a scatter/gather pass over
every element of `A` and `B` no matter how complex numbers are stored. Emitting
planar output from that pass costs nothing extra — same loads, different store
addresses — and buys a micro-kernel that needs no in-register shuffles plus a
packed `A` panel half the size of 1m's. Unlike matrix GEMM, where a separate
planarisation pass would be pure overhead (which is precisely why 1m exists),
here it rides along for free.

Conjugation is absorbed here at zero cost by negating the imaginary plane as it
is written, which is why `TAPP_CONJUGATE` on any operand is free.

Partial slivers are zero-filled so the micro-kernel always runs at full
register-block size.

### 3.5 Micro-kernel interface (`kernel/`)

```rust
pub struct Ukr<T> {
    pub mr: usize,          // logical (complex) rows of the micro-tile
    pub nr: usize,
    pub a_per_k: usize,     // reals in an A sliver per logical k-step
    pub b_per_k: usize,
    pub tile: usize,        // reals in the accumulator tile
    pub a_pack: PackFormat,
    pub b_pack: PackFormat,
    pub tile_fmt: TileFormat,
    pub func: unsafe fn(kc: usize, a: *const T, b: *const T, ab: *mut T),
    pub name: &'static str,
}
```

This struct is the whole of the method abstraction. `driver.rs` contains **no
branch on the complex method**: it reads sliver widths, panel sizes, tile size
and formats off the selected `Ukr`. Swapping methods swaps a `Ukr`, nothing
else — which is precisely what makes the three comparable.

Kernels take the *logical* `kc` (in complex elements) and know their own panel
layout, so 1m's internal doubling to `2*kc` real steps never leaks out.

The kernel *overwrites* the accumulator tile; `alpha`, `beta`, the scattered
write-back and the recombination of planes happen afterwards in `writeback.rs`.
Keeping them separate is what lets one kernel serve the regular fast path, the
gather path and every edge block. The cost is one extra store/load of a tile
that is L1-resident by construction.

Tile formats, read by `writeback::tile_value`:

| `TileFormat` | layout | complex value |
|---|---|---|
| `Real` | `ab[j*MR+i]` | `(x, 0)` |
| `Planar` | two `MR x NR` planes | `(re, im)` |
| `OneM` | one real `2*MR x NR` | row `2i` is Re, row `2i+1` is Im |
| `ThreeM` | three `MR x NR` planes | `(M1-M2, M3-M1-M2)` |

Complex needs more accumulator register state than real — two planes for
planar, a doubled real row count for 1m, three planes for 3m — so the complex
micro-tile is smaller. That is inherent to complex arithmetic, not to any one
method.

Dispatch is by `trait KernelSet`, implemented for `f32`/`f64` with runtime CPU
feature detection — AVX-512F, else AVX2+FMA, else the portable scalar path —
and `TENSORCONTRACT_KERNEL` pins a narrower one so each is testable on hardware
that has something wider. All the vectorised kernels are generated from one
macro body per method over `(MV, NR)` const generics, with the instruction set
as a further macro parameter, so neither the three complex methods nor the two
instruction sets differ by anything but their shapes. Any other `Real` type
(extended precision, dual numbers for forward-mode AD, `bf16`) gets a correct
engine by implementing `KernelSet` with the generic scalar kernels — the
genericity is real, not aspirational, and is exercised by the
`TENSORCONTRACT_KERNEL=scalar` CI job.

### 3.6 Blocking, and the write-back cost model

Defaults are derived from cache sizes (packed A ≈ half of L2, packed B ≈ 3 MiB
of L3) and overridable per plan or via `TENSORCONTRACT_MC/KC/NC`.

Those numbers are one machine's, which is the one thing in the blocking that
does not transfer. There is therefore a second derivation,
`TENSORCONTRACT_BLOCKMODEL=model`: the BLIS analytical model (Low, Igual, Smith
& Quintana-Ortí, ACM TOMS 2016) over cache descriptors probed at run time from
Linux sysfs, x86 `CPUID`, or conservative built-in defaults, in that order —
`KC` from the L1 subject to associativity, `MC` from the L2, `NC` from the L3,
with no machine-specific constant anywhere in it. It is **off by default**: the
committed measurements are all against the hardcoded constants and the pending
`MC`/`KC`/`NC` grid defines its arms relative to them, so the model is an A/B
switch rather than a new default until it has been measured. See
`kernel::cache` for the equations, what had to be inferred, and how the
threading interacts (`NC`'s L3 budget is shared, not divided, because the
packed `B` panel is shared — but each thread's packed `A` is charged against
the same L3).

Critically, `Blocking::derive` takes the **reals per element the method
actually packs**, not `size_of::<Element>()`. 1m stores four reals per complex
element where planar stores two, so an element-size-based rule would silently
give 1m twice the L2 footprint and rig the comparison. Deriving from the packed
footprint gives every method the same L2 budget and 1m a proportionally
smaller `MC`.

The per-output recombination cost at write-back differs by method — two loads
for planar and 1m, three loads and three adds for 3m — and is roughly
`c / (4 * min(K, KC))` of a tile's compute with `c` the ops per element.
Negligible for `K >= KC`, rising like `1/(8K)` for small `K`, material only at
`K` in the low single digits. That is where a shape-driven method dispatch (or
a fall-back to TTGT) belongs, in Phase 4.

### 3.7 TAPP export

`crates/tensorcontract-tapp` exports the C ABI as `lib`, `cdylib` and
`staticlib`. Handles are `Box::into_raw` pointers cast to `isize`. A single
dtype-erased `Plan` is stored and `TAPP_execute_product` dispatches on the
recorded datatype. Coverage table is in the crate docs; cases 1–4 and
conjugation on all four operands are supported, case 5 is rejected,
`F16`/`BF16` are rejected.

---

## 4. Threading (deferred to Phase 4)

BLIS-style: parallelise loop 3 (`ic`) and loop 2 (`jr`), with threads in the
same `jc`/`pc` iteration sharing one packed B panel. `std::thread::scope` with
static partitioning; NUMA-aware first-touch for the packed buffers. Not
implemented — all measurements in this repository are single-core.

---

## 5. Benchmark and testing framework

### 5.1 Correctness

Three independent layers:

1. **Brute-force oracle** (`reference.rs`), sharing no code with the engine,
   walking raw layouts and labels. It independently defines the semantics of
   diagonals, reductions, Hadamard indices, conjugation and general strides.
2. **Randomised property tests** (`tests/correctness.rs`): 1000 random
   contractions across `f32`/`f64`/`c32`/`c64` with random ranks, random label
   assignments across all index classes, random stride permutations, random
   conjugation masks, and `alpha`/`beta` including zero — each run twice, once
   with cache blocking set to `(1, 2, 1)` so that every level of the five-loop
   nest and every partial block fires on a tensor small enough for the oracle,
   and once with the real blocking. Plus targeted cases: empty contraction
   dimension, zero-sized output, scalar output, negative strides, and all 16
   conjugation flag combinations forced through multiple `K` blocks.
3. **Cross-implementation** (`tcbench verify`): the whole 49-case corpus at
   benchmark sizes, in all four dtypes and all three stride-stress modes,
   against both TTGT/OpenBLAS and TBLIS.

### 5.2 The corpus, and a correction to the brief

The brief specifies "the TCCG set of 48 contractions". Reading the upstream
`benchmark/benchmark.py`:

* `_fullbenchmark = 1` gives 19 CCSD + 3 AO2MO + 8 InTensLi + 18 CCSD(T) + 1
  transpose-C = **49 distinct** contractions;
* `_fullbenchmark = 0` gives a 25-case reduced set;
* `_sortedTCs` is **not** a sixth group. Its 24 entries are those same
  reduced-set cases re-written in TCCG's normalised labels and ordered from
  bandwidth-bound to compute-bound. Concatenating every list and
  de-duplicating yields 25, not 49 — as a unit test in `corpus.rs` records.

This repository uses the full 49-case set, a strict superset of both published
variants, and keeps `_sortedTCs` as the presentation ordering.

Extent selection reproduces TCCG's rule exactly, with one deliberate deviation:
the element size used for sizing is fixed at 8 bytes for *every* dtype. TCCG
re-sizes per precision, which would hand complex runs smaller tensors and make
the real-vs-complex ratio meaningless.

### 5.3 The corpus does not exercise the gather path

TCCG rounds every stride-1 extent **up to a multiple of 24**. Since 24 is
divisible by every plausible register block (4, 8, 12, 24; and the others in
use divide the extents too), a run of `MR` consecutive rows never straddles an
index boundary, and the block-scatter vectors come out **fully regular** — the
harness measures `regA = 1.00` on every case.

This matters a great deal for the thesis, which is specifically about awkward
strides forcing work onto the slow gather path: **the standard corpus cannot
test that claim at all.** Two perturbation modes were added:

* `--stress ragged` — subtract one from every extent, so nothing divides the
  register block and every index-boundary crossing produces an irregular block
  (observed `regA` 0.80–1.00);
* `--stress padded` — each tensor becomes a strided view into a larger buffer
  with its leading dimension padded, as a slice of a bigger array would be.

### 5.4 Metrics

The headline metric is the **complex efficiency ratio**. Because a complex MAC
is exactly four real FMAs and is counted as 8 flops, the achievable GF/s peak
is the *same number* in both domains on real-SIMD hardware. So:

* `eff = contraction GF/s / same-shape GEMM GF/s` (the roofline annotation,
  using a tuned vendor GEMM as the ceiling), computed separately for real and
  complex;
* `eff ratio = complex eff / real eff`. Below 1.0 means the engine loses more
  to overhead on complex data than on real data. At or above 1.0 the
  complex-weakness thesis is refuted.

A ceiling-free cross-check is also reported: the raw `c64/f64` GF/s ratio.

---

## 6. Self-scrutiny

Written before the premise check was run, and left standing because the
"single most likely failure mode" is exactly what happened.

**Assumptions.**

1. That the achievable complex peak equals the achievable real peak in GF/s.
   This underpins the whole metric. It is right for real-SIMD FMA hardware.
2. That the TCCG corpus is representative of where complex tensor contraction
   actually hurts. §5.3 shows it is not even representative of where *scatter*
   hurts.
3. That halving the packed-A footprint relative to 1m translates into
   measurable time. Only true if packing store bandwidth is on the critical
   path.
4. That TBLIS-as-built is a fair stand-in for "the state of the art in complex
   tensor contraction".

**Alternatives to planar.** All three are now implemented rather than argued
about, which is the resolution of this section.

* **1m** — the incumbent. Its A-panel inflation is real but may be irrelevant.
* **3m** — 25% fewer flops, weaker error bound. Planar-style packing makes it
  natural (pack `re`, `im`, and `re+im` in one pass). This is the one complex
  idea with a *flop-count* advantage rather than a bandwidth one, and the
  premise check does not bear on it.
* **TTGT with a good permute** (HPTT-class) — the measurements show TTGT is
  2–4x off TBLIS on low-intensity shapes, so the transpose really does cost.
* **Wrap TBLIS** — no research contribution, but the honest answer if the goal
  is a working Rust tensor contraction today.
* **Contribute upstream** — a 3m or planar path inside BLIS/TBLIS would be a
  much smaller change than a new engine, and this repository is now a
  ready-made A/B harness for arguing that case.

**Single most likely failure mode, and how Phase 1 detects it early.**

> The complex-weakness premise is folklore and simply is not true, because
> complex arithmetic has 4x the arithmetic intensity per element pair and
> therefore *hides* memory and indexing overheads better than real arithmetic
> does. The Phase 1 premise check detects this directly: it measures
> `complex eff / real eff` for TBLIS on real shapes, and if that number is at
> or above 1.0 the thesis is dead.

That is what the measurement found. See `DECISIONS.md`.
