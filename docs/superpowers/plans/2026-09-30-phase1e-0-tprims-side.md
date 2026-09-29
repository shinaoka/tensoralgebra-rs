# Phase 1e-0: tprims-side preparation — plan

**Spec:** `docs/superpowers/specs/2026-09-30-phase1e-tenferro-integration-design.md`
(deliverable 1e-0, and the P2 tooling). Executed inline.

## Global constraints

- Build jobs from `CARGO_BUILD_JOBS`; measurements follow the
  `tprims-benchmark` skill (idle cores of one L3 domain, `pinned.sh`).
- Thread counts for P2: 1T, 4T, 8T gate; 16T reported.
- dtypes f32, f64, c32, c64.

## Tasks

1. **Corpus schema and loader** (`benchmarks/src/corpus.rs`, serde): a JSON
   file `{ "source": {...}, "entries": [...] }`; an entry is either
   `dot_general` (operands `a`, `b`, `c` as `{dims, strides}` in elements,
   `lc`, `rc`, `lb`, `rb`, `conj: [bool; 2]`) or `gemm_batched` (`m`, `n`,
   `k`, `batch`, per-operand strides of the `[rows, cols, batch]` views), with
   `dtype`, `name`, and optional `calls` / `time_share` from the P1 profile.
   Validation: rank and length agreement, injective output, known dtype.
   Unit tests on valid and invalid entries.
2. **`contract --corpus FILE`** and `--list`: runs every `dot_general`
   entry with both strategies, any of the four dtypes, operands laid out with
   the recorded strides (offset so the lowest address is element 0), same CSV
   and `CHECK` lines as the built-in corpus (tolerance 1e-12 for f64/c64,
   1e-4 for f32/c32).
3. **`blas --corpus FILE`** and `--list`: `gemm_batched` entries, faer loop
   vs TBLIS-style, same CSV and `CHECK` lines.
4. **Runners for any thread list**: `contract/run.sh` and a new
   `blas/run.sh` take `OUT CPUS THREADS...` (CPUS sized for the largest
   count, prefixes for smaller ones) and an optional `CORPUS` file; one
   process per case, thread counts paired per case, through `pinned.sh`.
5. **`tprims_blas::gemm_grouped`**: variable-size jobs matching tenferro's
   grouped GEMM request (shape taken from the tenferro interface digest),
   scheduled like `gemm_batched` (items over the pool when there are at least
   as many items as the budget, otherwise each item with inner parallelism),
   correctness tests against per-job `gemm`, benchmark rows at 1T/4T/8T.
6. Docs: bench READMEs (corpus mode, runners), skill step for corpus runs,
   status line in README.
