# contract

`tprims-contract` at an enforced thread count on a predeclared corpus, both
strategies: permute plus batched GEMM (`pg`) and TBLIS-style direct through
Lukas Devos's `tensorcontract` (`tblis`). Planning (`ContractPlan::new`) and
execution (`execute`, alpha = 1, beta = 0) are timed separately.

| Case | A (storage order) | B (storage order) | Contraction |
| --- | --- | --- | --- |
| `tiny_matmul` | 2 x 2 | 2 x 2 | `ij,jk->ik` |
| `matmul_256` | 256 x 256 | 256 x 256 | `ij,jk->ik` |
| `batched_64_b32` | 64 x 64 x 32 | 64 x 64 x 32 | `ijb,jkb->ikb` |
| `permuted_fusable` | 64 x 32 x 32, stored (c, b, a) | 32 x 32 x 64, stored (d, b, c) | `abc,cbd->ad` |
| `permuted_nonfusable` | 64 x 32 x 32, stored (b, a, c) | 32 x 32 x 64, stored (b, a, d) | `abc,cbd->ad` (A must be copied) |
| `network_ijkl_klmn` | 16^4 | 16^4 | `ijkl,klmn->ijmn` |
| `large_ijk_jkl` | 256 x 64 x 64 | 64 x 64 x 256 | `ijk,jkl->il` |

```bash
cargo build --release -p tprims-bench --bin contract
cpus=$(python3 benchmarks/scripts/idle_cpus.py pick 8)            # one idle L3 domain
benchmarks/benchmarks/tprims/contract/run.sh /tmp/contract "$cpus" 1 4 8
CORPUS=path/to/corpus.json benchmarks/benchmarks/tprims/contract/run.sh /tmp/c2 "$cpus" 1 4 8
```

`run.sh` (`benchmarks/scripts/paired.sh`) runs every case in its own process,
all thread counts of a case back to back, each through `pinned.sh`, and keeps
the `# selected` (materialized operands) and `CHECK` (pg vs tblis) lines in a
log and the commit, CPU, core set and corpus hash in `manifest.txt`.
`contract --corpus FILE` replays the `dot_general` entries of a corpus
(`tprims_bench::corpus`; example: `../corpus/example.json`) in any of
f32/f64/c32/c64 with the recorded strides. The observation below predates
corpus mode and used the older `run.sh CPUS1 CPUS4 OUT` (1T/4T only).

## Observation, 2026-09-30 (single run, not a claim)

- tprims-rs `0e68641` (clean tree), release profile (thin LTO,
  codegen-units 1), rustc 1.97.1, faer 0.24.4.
- AMD EPYC 7713P, shared host (load average about 4); cores 57-60 (one L3
  domain) busy fraction <= 0.01 over 3 s before and after. Sub-microsecond
  rows are within the ~20% A/A noise floor measured for `linalg`.
- Raw output: [results/](results/) (CSVs, and `*.selected.log` with the `# selected` and `CHECK` lines). All `CHECK` lines ok (relative
  difference <= 1.5e-15); every case ran copy-free under `pg` except
  `permuted_nonfusable`, where A was materialized.

| Case | pg plan | pg 1T | pg 4T | pg 1T/4T | tblis plan | tblis 1T | tblis 4T | tblis 1T/4T | tblis/pg 1T | tblis/pg 4T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_matmul_f64` | 1.1 µs | 360 ns | 460 ns | 0.78 | 1.6 µs | 730 ns | 730 ns | 1.00 | 2.03 | 1.59 |
| `tiny_matmul_c64` | 1.3 µs | 500 ns | 380 ns | 1.32 | 1.8 µs | 870 ns | 720 ns | 1.21 | 1.74 | 1.89 |
| `matmul_256_f64` | 1.2 µs | 634.0 µs | 206.1 µs | 3.08 | 5.3 µs | 1.14 ms | 322.1 µs | 3.54 | 1.80 | 1.56 |
| `matmul_256_c64` | 1.2 µs | 2.48 ms | 854.0 µs | 2.90 | 5.3 µs | 2.86 ms | 999.8 µs | 2.86 | 1.15 | 1.17 |
| `batched_64_b32_f64` | 1.3 µs | 429.3 µs | 127.5 µs | 3.37 | 3.2 µs | 684.8 µs | 518.5 µs | 1.32 | 1.59 | 4.07 |
| `batched_64_b32_c64` | 1.2 µs | 1.54 ms | 675.2 µs | 2.28 | 3.2 µs | 2.23 ms | 964.3 µs | 2.31 | 1.45 | 1.43 |
| `permuted_fusable_f64` | 1.2 µs | 286.9 µs | 144.5 µs | 1.99 | 5.5 µs | 304.7 µs | 161.1 µs | 1.89 | 1.06 | 1.11 |
| `permuted_fusable_c64` | 1.2 µs | 837.9 µs | 260.4 µs | 3.22 | 4.6 µs | 1.01 ms | 317.2 µs | 3.17 | 1.20 | 1.22 |
| `permuted_nonfusable_f64` | 2.1 µs | 248.4 µs | 175.5 µs | 1.41 | 5.6 µs | 303.0 µs | 161.7 µs | 1.87 | 1.22 | 0.92 |
| `permuted_nonfusable_c64` | 1.9 µs | 719.3 µs | 399.1 µs | 1.80 | 4.7 µs | 825.7 µs | 350.2 µs | 2.36 | 1.15 | 0.88 |
| `network_ijkl_klmn_f64` | 1.5 µs | 737.6 µs | 239.2 µs | 3.08 | 6.8 µs | 1.36 ms | 277.2 µs | 4.90 | 1.84 | 1.16 |
| `network_ijkl_klmn_c64` | 1.4 µs | 2.47 ms | 812.3 µs | 3.04 | 5.8 µs | 2.85 ms | 995.7 µs | 2.86 | 1.15 | 1.23 |
| `large_ijk_jkl_f64` | 1.3 µs | 10.68 ms | 3.16 ms | 3.38 | 14.2 µs | 12.46 ms | 4.03 ms | 3.09 | 1.17 | 1.28 |
| `large_ijk_jkl_c64` | 1.2 µs | 44.03 ms | 12.77 ms | 3.45 | 14.2 µs | 47.82 ms | 14.97 ms | 3.19 | 1.09 | 1.17 |

### Findings

1. **Permute plus batched GEMM wins wherever the operands fuse**: TBLIS-style
   is 1.1-2.0x slower at 1T and 1.1-1.9x at 4T (4.1x for `batched_64_b32` at
   4T, where the pg path spreads the batch over the pool and tensorcontract's
   batch axis is serial inside each SPMD team).
2. **When an operand must be copied, TBLIS-style is competitive and wins at
   4T** (`permuted_nonfusable`: tblis/pg 0.92 f64, 0.88 c64 at 4T), because
   the pg path pays for a copy of A that the direct kernel avoids
   (hypothesis: the copy, although parallel above strided's threshold, is
   bandwidth-bound; pg itself only scales 1.4x there). This is the first case where the direct strategy is preferable;
   an `Auto` rule "TBLIS when pg would materialize a large operand" is a
   candidate, to be decided on a wider corpus (decided in P2 below).
3. Planning costs 1-2 us for pg and 1.6-14 us for tblis (scatter vectors of
   length M + N + K); both are outside `execute`.

## Phase 1e P2: tenferro shape corpus (2026-09-30)

- Corpus: [`../corpus/tenferro-p1.json`](../corpus/tenferro-p1.json), 57
  `dot_general` shape groups (3150 calls) logged from tenferro-benchmark's
  einsum suite (25 instances) and `cpu/public_api` quick run through the
  tprims provider (`TPRIMS_SHAPE_LOG`, 1T), weighted by call count.
- Runs: tprims-rs `d8e565a`, release profile, AMD EPYC 7713P, CPUs 0-7 (one
  CCD, idle-checked before and after every run by `pinned.sh`), 1T/4T/8T
  paired per case, `BENCH_RUNS=7`, three complete sessions
  ([`results/2026-09-30-p2/`](results/2026-09-30-p2/), manifests inside). A
  first session 3 stopped after 11 cases (idle retries exhausted) and was
  discarded and rerun in full.
- Decision ([`decision.txt`](results/2026-09-30-p2/decision.txt)): workload
  time pg / tblis = 1.42-1.47 (1T), 1.57-1.66 (4T), 1.70-1.87 (8T), noise
  1.7-4.7%; unweighted geometric mean 0.93-1.09. All `CHECK` lines ok.

Workload time by what permute+GEMM must copy (session 3):

| pg copies | cases | 1T pg | 1T tblis | ratio | 8T pg | 8T tblis | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A and B | 23 | 12.32 s | 8.05 s | 1.53 | 3.48 s | 1.59 s | 2.19 |
| B | 9 | 3.33 s | 2.39 s | 1.39 | 0.77 s | 0.44 s | 1.76 |
| A | 6 | 1.17 s | 0.66 s | 1.78 | 0.36 s | 0.16 s | 2.21 |
| nothing | 19 | 0.83 s | 0.88 s | 0.94 | 0.16 s | 0.41 s | 0.40 |

Findings:

1. The tensor-network contractions of tensor4all are high rank with small
   extents, and their contracted axes are rarely adjacent in storage, so
   permute+GEMM copies at least one operand in 38 of 57 groups, which carry
   over 95% of the workload time. TBLIS-style packs straight from the
   strides and skips that copy; the gap widens with threads (hypothesis: the
   copy is bandwidth-bound and scales worse than the arithmetic).
2. Without a copy the comparison is GEMM kernel against GEMM kernel, and
   faer wins (also seen in the batched-GEMM corpus); the difference is
   presumably already in the GEMM kernel, blocking or threading, to be
   checked separately.
3. Hence `Strategy::Auto` now picks TBLIS-style exactly when permute+GEMM
   would copy (decision log), which is also the best of both columns above.

## Hadamard products: elementwise path vs TBLIS-style (2026-09-30)

Whether the dedicated all-batch path could be dropped in favour of the
TBLIS-style kernel. Corpus [`../corpus/hadamard.json`](../corpus/hadamard.json):
24 cases, f64 and c64, vectors of 2^10-2^24 elements, 256^2 and 2048^2
matrices, rank-4 and rank-12 tensors, some with B stored transposed (`_bT`).
The `pg` rows run tprims-contract's elementwise path (`PermuteGemm` and
`Auto` both route all-batch problems there), `tblis` rows the TBLIS-style
kernel. Same setup as P2: tprims-rs `d8e565a`, EPYC 7713P CPUs 0-7, 1T/4T/8T
paired per case, three sessions
([`results/2026-09-30-hadamard/`](results/2026-09-30-hadamard/)). All `CHECK`
lines ok.

TBLIS-style time / elementwise time, median over sessions:

| threads | range over the 24 cases | f64, 2^22 vector | c64, 2^22 vector | f64, 256^2 |
| --- | --- | --- | --- | --- |
| 1T | 7.0-24.4x | 24.4x | 11.2x | 21.5x |
| 4T | 11.2-92.1x | 92.1x | 46.0x | 21.7x |
| 8T | 10.1-131.6x | 131.6x | 61.7x | 18.0x |

The TBLIS-style kernel runs a Hadamard product as a batch of 1x1x1 GEMMs,
paying micro-kernel and packing overhead per element, and its batch axis is
serial inside the SPMD team, so it does not scale with threads while the
elementwise path does (hypothesis consistent with the growth from 1T to 8T).
The dedicated elementwise path stays.
