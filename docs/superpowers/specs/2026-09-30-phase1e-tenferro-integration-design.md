# Phase 1e: tenferro-rs integration — design

Date: 2026-09-30. Status: design for maintainer review. Inputs: the
maintainer decisions of 2026-09-30 (below), a read-only survey of tenferro-rs
`origin/main` @ `c22ee10e9` and tenferro-benchmark `origin/main` @ `5d6f457`,
and the Phase 1 results in `benchmarks/benchmarks/tprims/`.

## Goal

tprims becomes an optional CPU implementation that tenferro-rs can run and
that we can accept or reject op by op from measurements. tenferro-rs itself
gains only generic injection points; nothing in tenferro's default build
depends on tprims.

## Maintainer decisions (2026-09-30)

1. tenferro-linalg gets an injection point, like the existing GEMM/contraction
   provider bundle, so the tenferro changes can merge to `origin/main`. tprims
   is an optional implementation tested through those points.
2. tenferro-benchmark can switch the implementation to tprims by a cargo
   feature; acceptance is decided there.
3. 1e is split: **1e-a** GEMM, grouped/batched GEMM and `dot_general`;
   **1e-b** linear algebra. One spec, separate PRs.
4. Contraction and batched GEMM: after sufficient EPYC benchmarks, switch to
   the TBLIS-style implementation if it is faster, **globally** (not by shape
   range).
5. The TBLIS decision uses a `dot_general`/contraction size sweep, faer
   (permute+GEMM) vs TBLIS, at 1T and 4T (8T added, see P2), with typical sizes taken from
   tenferro-benchmark runtime profiles.
6. strided-rs is external to both repositories and pinned to the same commit
   (done in tprims-rs #18), so tenferro and tprims share one `StridedView`.

## Constraints from the survey

- GEMM and `dot_general` already have runtime injection:
  `CpuGemmProvider` (`gemm`, `strided_batched_gemm`, `grouped_gemm`),
  `CpuGeneralContractionProvider::dot_general`, installed through the
  provider bundle (`CpuBackend::with_provider_bundle`). A provider returns
  `CpuProviderOutcome::Unsupported(reason)` to fall through to the built-in
  path. Precedent: `ext/tenferro-cpu-tblis`, a separate workspace tested by
  the `extensions` CI profile.
- Linear algebra has no injection: `CpuLinalgProvider {Faer, Blas}` is a
  compile-time `match` in ~25 `*_entered` functions
  (`tenferro-linalg/src/cpu/backend.rs`). Composites (`det`, `slogdet`,
  `inv`, `lstsq`, `pinv`, `norm`) are built from the primitives and follow
  whatever the primitives do.
- Kernels see `CpuExecutionContext` (`ParallelMode`, `thread_budget()`), but
  no pool handle; the session callback already runs on a worker of the
  context's `Arc<rayon::ThreadPool>`. tprims-exec needs a borrowed pool.
- tenferro's git-pin content check applies to `[workspace.dependencies]` of
  the root workspace; an `ext/` crate with its own workspace is outside it.
- tenferro CI is expensive (GPU lanes cascade on PRs to `main`): pushes are
  batched and reproduced locally first.

## Architecture

```
tenferro-rs (origin/main)                      tprims-rs
  tenferro-cpu:   provider bundle  <─────┐       tprims-exec / -blas /
                  + pool accessor         │       -contract / -linalg
  tenferro-linalg: CpuLinalgKernels trait │            ▲
                  (new injection point)   │            │ git (pinned rev)
  ext/tenferro-cpu-tprims (own workspace) ─┴────────────┘
        ▲ path
tenferro-benchmark: feature `tprims` installs the ext provider
```

### tenferro-rs changes (merge to `origin/main`, no tprims dependency)

**T1. Pool access for providers.** `CpuExecutionContext` gains
`rayon_pool(&self) -> Option<&rayon::ThreadPool>`: the pool of the running
session when the context allows inner parallelism, `None` otherwise. Nothing
else about sessions changes (they stay closure-scoped). Tested with a mock
provider that records the pool identity and budget.

**T2. Linalg injection point.** A new object-safe trait in tenferro-linalg,
provisionally `CpuLinalgKernels`, one method per primitive the `*_entered`
functions dispatch today (cholesky, triangular_solve, lu, lu_factor,
full_piv_lu, solve and its batched forms, qr, rank_revealing_qr, the
Householder family, svd / svd_full / svd_values, eigh / eigh_values, eig /
eig_values). Each takes typed strided operands plus `&CpuExecutionContext`
and returns `CpuProviderOutcome`; `Unsupported` falls through to the existing
faer/LAPACK arm. It is installed through the same provider bundle as the GEMM
providers. The trait is generic; its first non-test implementation is tprims.
The name `CpuLinalgProvider` is taken by the existing enum, hence the
different name. Tested in tenferro with a mock implementation: routing, every
`Unsupported` fallback, and error propagation. Existing linalg and AD suites
run unchanged with no kernels installed.

**T3. `ext/tenferro-cpu-tprims`** (separate workspace, like
`ext/tenferro-cpu-tblis`): path dependencies on tenferro crates, git
dependency on tprims-rs at a pinned commit. Implements `CpuGemmProvider`,
`CpuGeneralContractionProvider` and `CpuLinalgKernels` with tprims, returning
`Unsupported` for everything outside the op table. Added to the `extensions`
CI profile. It is merged to `main` once green (maintainer decision,
2026-09-30).

### Execution mapping

| tenferro context | tprims `Exec` |
| --- | --- |
| `ParallelMode::Sequential`, or budget 1 | `Exec::Serial` |
| `ParallelMode::Outer` (tenferro spreads lanes) | `Exec::Serial` per lane |
| `ParallelMode::Inner`, budget b > 1 | `Exec::rayon(Pool::borrow(ctx.rayon_pool()))` with budget b |

The provider runs on a worker of that pool, so tprims-exec enters nothing
and runs in place (Phase 1a contract). One rayon build in the binary is
required and holds (rayon 1.x unifies).

### Op table

Each op is routed to tprims only when it passes correctness and the
acceptance gate; otherwise its provider method returns `Unsupported`.

| tenferro op | tprims | Notes |
| --- | --- | --- |
| `gemm` | `tprims_blas::gemm` | conj flags map to `Conj`; `alpha`/`beta` |
| `strided_batched_gemm` | `tprims_blas::gemm_batched` | batch axis last in both |
| `grouped_gemm` | `tprims_blas::gemm_grouped` (new in 1e-a) | variable-size jobs; scheduled like `gemm_batched` |
| `dot_general` | `tprims_contract::ContractPlan` (`Strategy::Auto`) | plan cached per layout key, as `SessionCachedDot` does |
| uninit GEMM variants | `Unsupported` | later |
| cholesky, triangular_solve | `cholesky`, `trsm` | |
| lu, lu_factor, full_piv_lu, solve (+ batched) | `lu`, `lu_full`, `solve`, `batched::solve` | pivot format converted in the adapter |
| qr, rank_revealing_qr | `qr`, `qr_col_piv` | |
| Householder family | `Unsupported` | no tprims representation yet |
| svd, svd_full, svd_values (+ batched) | `svd`, `batched::svd` | singular vectors compared up to gauge |
| eigh, eigh_values (+ batched) | `eigh`, `batched::eigh` | |
| eig, eig_values | `eig` | eigenvalue order and vector normalization converted |
| dtype other than f32/f64/c32/c64 | `Unsupported` | |

## Correctness

- **tenferro side (T1, T2):** mock-provider tests for routing and fallback;
  the full existing suite with no provider installed must pass unchanged
  (this is what merges to `main`).
- **ext side (T3):** A/B tests in one binary, default backend vs the tprims
  provider, for every routed op over f32/f64/c32/c64, contiguous, transposed
  and offset strided operands, batched and unbatched, empty and 1x1 shapes.
  Gauge-dependent results (SVD/eig vectors, QR signs) are compared through
  reconstruction and invariants, not elementwise. tenferro's linalg AD rule
  tests run with the provider installed.

## Performance and acceptance

All runs on the EPYC 7713P (64 cores, 8 CCDs of 8 cores sharing 32 MiB L3,
one NUMA node) follow the `tprims-benchmark` skill
(`.agents/skills/tprims-benchmark/SKILL.md`): idle cores of one L3 domain
chosen by `benchmarks/scripts/idle_cpus.py`, every measurement through
`benchmarks/scripts/pinned.sh` (idle before and after), runs sequential,
build jobs from `CARGO_BUILD_JOBS`. CPU affinity is Linux-only; the
decisions below use Linux runs.

**P1. Shape profile.** The ext provider logs, behind an environment variable,
each call's op, dtype, shape, strides and contraction config to a JSONL file.
Running the tenferro-benchmark CPU suites with it yields call counts and time
shares; the corpus is the shapes covering most of the time plus the most
frequent small ones, committed as JSON with the benchmark commit that
produced it.

**P2. TBLIS decision (in tprims-bench).** `contract` and `blas` gain a
`--corpus <json>` mode sweeping the P1 corpus plus the existing synthetic
cases, faer/permute+GEMM vs TBLIS-style, at 1T, 4T and 8T (one full L3
domain; 16T reported only), in at least three
sessions with an A/A noise measurement per session. Rule, applied separately
to contraction (`Strategy::Auto`) and batched GEMM (default
`BatchStrategy`): switch the default to TBLIS if the geometric mean of
`time_pg / time_tblis` over the corpus exceeds 1 by more than the A/A noise at
each of 1T, 4T and 8T in every session; otherwise keep the current default.
8T gates because TBLIS reuses packed panels in L3, which a full CCD
contends for most, and a global switch must not regress there. Cases
where the chosen default loses are listed in the result page. The decision is
recorded in `docs/decision-log.md`.

**P3. Acceptance (in tenferro-benchmark).** A cargo feature `tprims` makes
the benchmark binaries install the ext provider. Baseline = default build,
candidate = `--features tprims`, compared with the existing ABBA paired
timing (`scripts/run_paired_timing.sh`) at 1T, 4T and 8T (16T reported, not
gated). An op family is routed to tprims when its
geometric-mean ratio is ≤ 1 within the A/A noise and no case regresses beyond
twice the noise (thresholds approved by the maintainer, 2026-09-30);
otherwise it stays `Unsupported`. Results go to
`result/amd-cpu/cpu/` with both commits.

## Deliverables and order

| Step | Repository | Content |
| --- | --- | --- |
| 1e-0 | tprims-rs | `gemm_grouped`; `--corpus` mode in `contract`/`blas` benches |
| 1e-a1 | tenferro-rs | T1 (pool accessor), mock-provider tests → PR to `main` |
| 1e-a2 | tenferro-rs | T3 ext crate with GEMM/grouped/dot_general providers, A/B tests, shape log |
| 1e-a3 | tenferro-benchmark | feature `tprims`; P1 profile, P3 acceptance for 1e-a ops |
| 1e-a4 | tprims-rs | P2 sweep and TBLIS decision |
| 1e-b1 | tenferro-rs | T2 (`CpuLinalgKernels`), mock tests → PR to `main` |
| 1e-b2 | tenferro-rs | linalg provider in the ext crate, A/B + AD tests |
| 1e-b3 | tenferro-benchmark | P3 acceptance for linalg ops |

Each tenferro-rs step reproduces the relevant CI profiles locally before its
single push; PRs to tenferro `main` follow tenferro's own rules and review.

## Out of scope

Making tprims tenferro's default backend; publishing tprims to crates.io;
GPU; the C ABI; replacing faer inside tprims; the Householder family and
uninit GEMM in tprims.

## Decided in review (2026-09-30)

- `ext/tenferro-cpu-tprims` merges to tenferro `main` once green.
- P3 thresholds as written.
- 8T gates P2 and P3 alongside 1T and 4T; 16T is reported only. (8T was
  requested by the maintainer; gating it was the recommendation adopted when
  implementation was allowed to proceed. Cost if wrong: a TBLIS switch that
  would pass at 1T/4T is held back by 8T.)
- Measurement procedure lives in the `tprims-benchmark` skill, not in this
  spec.
