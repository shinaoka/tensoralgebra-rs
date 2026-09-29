# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmark suite for [strided-rs](https://github.com/tensor4all/strided-rs) comparing **strided-opteinsum** (Rust) against **OMEinsum.jl** (Julia) using the [einsum benchmark](https://benchmark.einsum.org/) — 168 standardized einsum problems across 7 categories.

Only metadata (shapes, dtypes, contraction paths) is stored as JSON. Tensors are generated at benchmark time (zero-filled).

## Build & Run Commands

### Python (dataset generation)
```bash
uv sync                                    # Install dependencies
uv run python scripts/generate_dataset.py  # Export benchmark metadata to data/instances/
```

### Rust (strided-opteinsum benchmark)
```bash
cargo build --release    # Build
cargo run --release      # Run benchmarks
```
**Note:** `Cargo.toml` references `strided-opteinsum` and `strided-view` via local paths (`../strided-rs/`). The strided-rs repo must be cloned as a sibling directory.

### Julia (OMEinsum.jl benchmark)
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'  # Install dependencies
julia --project=. src/main.jl                          # Run benchmarks
```

## Architecture

### Three-language stack
- **Python** (`scripts/generate_dataset.py`): Filters instances from `einsum_benchmark` package by FLOPS/SIZE/dtype/tensor-count thresholds, exports JSON metadata with both row-major and column-major formats.
- **Rust** (`src/main.rs`): Loads JSON instances, builds `EinsumNode` contraction trees from pre-computed paths, benchmarks `strided-opteinsum` with warmup (2 runs) + timed (5 runs).
- **Julia** (`src/main.jl`): Same benchmark protocol but with two execution modes: (1) pre-computed path using `DynamicEinCode` pairwise contractions, (2) OMEinsum's `optimize_code` with `TreeSA()` optimizer.

### Data flow
`einsum_benchmark` (Python) → `data/instances/*.json` → Rust/Julia benchmark runners

### Column-major convention
NumPy is row-major; strided-rs and Julia are column-major. JSON files contain both `format_string`/`shapes` (row-major) and `format_string_colmajor`/`shapes_colmajor` (column-major). The conversion reverses each tensor's shape and each operand's index labels. Rust and Julia runners use the `_colmajor` fields.

### Contraction path format
Paths follow the opt_einsum/cotengra convention: each step `[i, j]` contracts tensors at those indices in the current list. Higher index is removed first, then lower; result is appended to end. Two strategies are benchmarked: `opt_flops` and `opt_size`.

## OpenBLAS Version Requirement

**Do NOT use the system OpenBLAS (apt install libopenblas-dev) for benchmarking.** It is too old (0.3.8 on Ubuntu 20.04) and ~2x slower than Julia's bundled version on AMD EPYC. Use OpenBLAS >= 0.3.29.

See [`docs/environment-setup.md`](docs/environment-setup.md) for installation and configuration details.

## Benchmark Execution Rules

**NEVER run multiple benchmarks concurrently.** Benchmark runs must be executed sequentially (one at a time). Running 1T and 4T benchmarks in parallel causes CPU resource contention and produces unreliable results.

## Benchmark Results Guidelines

When recording benchmark results in README.md, always include the OpenBLAS version for **both** strided-rs and OMEinsum.jl:

- **strided-rs OpenBLAS**: Check with `ldd target/release/strided-rs-benchmark-suite | grep openblas` and verify the version in the library filename.
- **OMEinsum.jl OpenBLAS**: Check with `julia -e 'using LinearAlgebra; println(BLAS.get_config())'`. The bundled version can be found at `~/.julia/juliaup/julia-<version>/lib/julia/libopenblas64_*.so`.
