# Strided Permutation Benchmark

This benchmark compares materializing strided/permuted `f64` tensors into a
col-major destination. The benchmark patterns live in
[`patterns.json`](patterns.json), and the Rust and Julia runners consume that
same file.

The unified pattern set incorporates the third-party transpose comparison from
[ultimatile/297160e33005b42595c3e2a416e0549b](https://gist.github.com/ultimatile/297160e33005b42595c3e2a416e0549b):
naive, HPTT, and `strided-perm` are compared on the same col-major `f64`
transpose semantics, with correctness checked before timing.

## Pattern Schema

Each pattern records the operation, not the result:

- `shape`: source tensor shape.
- `perm`: 0-indexed permutation with semantics
  `out[i0,...,ik] = src[i_perm0,...,i_permk]`.
- `src_layout` and `dst_layout`: currently `col_major` or explicit source
  strides.
- `participants`: implementations that should be reported for the pattern.
- `data`: fixed to `deterministic_index_value`; no random seed is recorded.

HPTT rows are emitted only for patterns with contiguous source and destination
layouts. Explicit source-stride cases such as `tn_light_415_24d_scattered_to_colmajor`
are intentionally excluded from HPTT because the public HPTT API does not
represent arbitrary source strides.

## Run

Run thread-count variants sequentially. Do not run benchmark processes in
parallel.

Rust, serial:

```bash
RUSTFLAGS="-C target-cpu=native" \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --bin permute
```

Rust, parallel `strided-perm`:

```bash
RUSTFLAGS="-C target-cpu=native" \
  RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 \
  cargo run --release --features parallel --bin permute
```

Rust with HPTT rows:

```bash
RUSTFLAGS="-C target-cpu=native" \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --features hptt --bin permute
```

Rust with parallel `strided-perm` and HPTT using the same Rayon thread count:

```bash
RUSTFLAGS="-C target-cpu=native" \
  RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 \
  cargo run --release --features parallel,hptt --bin permute
```

Julia Base and Strided.jl:

```bash
JULIA_NUM_THREADS=1 julia --project=. \
  benchmarks/strided_benchmarks/permute/permute.jl

JULIA_NUM_THREADS=4 julia --project=. \
  benchmarks/strided_benchmarks/permute/permute.jl
```

To run one pattern while investigating:

```bash
PATTERN_ID=transpose_2d_1024 cargo run --release --features hptt --bin permute

PATTERN_ID=transpose_2d_1024 julia --project=. \
  benchmarks/strided_benchmarks/permute/permute.jl
```

## Implementations

| Runner | Implementations |
|---|---|
| Rust | naive odometer, `std::ptr::copy_nonoverlapping`, `strided_perm::copy_into`, `strided_perm::copy_into_col_major`, parallel variants, optional `hptt` crate |
| Julia | Base `permutedims!` or generic `copyto!`, Strided.jl `@strided` materialization |

## Results

### Apple M5 Max MacBook Pro

Measured on macOS 26.5.1, 2026-07-02. Median milliseconds. macOS CPU pinning was unavailable.

- `strided-rs`: `7cdc813`
- Rust compiler flags: `RUSTFLAGS="-C target-cpu=native"`
- Rust `strided-rs` columns use the faster of `copy_into` and `copy_into_col_major` for the row.
- HPTT is shown only for contiguous-source/destination cases supported by the public HPTT API.

#### 1 Thread

`RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`.

| Pattern | naive (Rust) | strided-rs | HPTT | Julia Base | Strided.jl |
|---|---:|---:|---:|---:|---:|
| 2D 256^2 transpose | 0.058 | **0.016** | 0.019 | 0.038 | 0.027 |
| 2D 1024^2 transpose | 1.859 | 0.474 | **0.424** | 1.226 | 0.850 |
| 2D 2048^2 transpose | 12.103 | 4.965 | **3.846** | 11.437 | 5.909 |
| 3D 256^3 [2,0,1] | 131.466 | **16.326** | 20.034 | 112.162 | 100.505 |
| 3D 256^3 [1,0,2] | 30.006 | 11.190 | **9.697** | 15.332 | 11.280 |
| 6D rotation | 432.744 | 193.423 | **186.347** | 188.184 | 188.356 |
| 20D reverse | 1.900 | **0.987** | 2.049 | 2.563 | 1.126 |
| 13D reverse | 0.010 | **0.004** | - | 0.008 | 0.007 |
| 13D cyclic | 0.010 | **0.003** | - | 0.008 | 0.003 |
| 24D scattered -> col-major | 22.058 | **5.636** | - | 113.098 | 7.825 |
| 24D contiguous TN permutation | 20.000 | **3.260** | 4.256 | 16.769 | 5.996 |

Copy baseline (`24D 2^24` contiguous): `std::ptr::copy_nonoverlapping` 2.014 ms, `strided_perm::copy_into` 1.943 ms.

#### 4 Threads

`RAYON_NUM_THREADS=4`, `OMP_NUM_THREADS=1`, `JULIA_NUM_THREADS=4`.

| Pattern | naive (Rust) | strided-rs serial | strided-rs parallel | HPTT | Julia Base | Strided.jl |
|---|---:|---:|---:|---:|---:|---:|
| 2D 256^2 transpose | 0.102 | 0.015 | **0.014** | 0.026 | 0.038 | 0.028 |
| 2D 1024^2 transpose | 1.847 | 0.513 | **0.154** | 0.166 | 1.215 | 0.264 |
| 2D 2048^2 transpose | 11.829 | 4.857 | **1.307** | 1.369 | 11.497 | 1.777 |
| 3D 256^3 [2,0,1] | 79.311 | 15.588 | 7.695 | **6.138** | 116.826 | 34.740 |
| 3D 256^3 [1,0,2] | 29.687 | 11.602 | 3.725 | **3.351** | 15.272 | 3.560 |
| 6D rotation | 416.697 | 187.492 | **60.897** | 63.384 | 197.772 | 70.050 |
| 20D reverse | 2.472 | 1.117 | 0.709 | 1.474 | 2.496 | **0.546** |
| 13D reverse | 0.009 | **0.004** | 0.004 | - | 0.008 | 0.007 |
| 13D cyclic | 0.009 | **0.002** | 0.002 | - | 0.008 | 0.003 |
| 24D scattered -> col-major | 20.718 | 5.544 | **2.058** | - | 112.834 | 3.261 |
| 24D contiguous TN permutation | 18.713 | 3.281 | **1.158** | 1.418 | 16.757 | 1.733 |

Copy baseline (`24D 2^24` contiguous): `std::ptr::copy_nonoverlapping` 2.034 ms, `strided_perm::copy_into` 1.918 ms, `strided_perm::copy_into_par` 1.944 ms.

**Notes:**

- The original problematic 2D 1024^2 transpose is now faster than naive in both serial and parallel `strided-rs` paths.
- HPTT remains strongest on some contiguous transpose shapes, especially 3D permutations, but `strided-rs` is faster on scattered 24D cases that HPTT cannot express through its public API.
- Small 13D cases do not benefit from threading; the parallel path falls back to essentially the same cost.
