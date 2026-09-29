# Erased Replay Rank/Layout Scaling

This page publishes the before/after evidence for the strided-rs #213
incremental-offset remediation. It covers generic gather and additive scatter,
dynamic slice/update, erased axis reduction, generic pad, and integer
divide/remainder zero preflight.

## Reproduce

The measurement source is
`strided-kernel/benches/erased_policy_thresholds.rs` at strided-rs
`b40cd2f6d83c35ca23b24a8fb371ca061495729c`. Run all groups sequentially:

```bash
CPUS=0-3 STRIDED_RS_REF=b40cd2f6d83c35ca23b24a8fb371ca061495729c \
  ./benchmarks/strided_benchmarks/erased_replay/run.sh
```

The script refuses a different strided-rs checkout and uses a hash-specific
Cargo target directory. Setup and plan compilation remain outside Criterion
timed regions. It does not automate the load gate: inspect the selected cores
and the rest of their L3 domain immediately before starting it.

## Environment

- measured: 2026-08-24
- CPU: AMD EPYC 7713P, one 8-core L3 domain, four pinned cores
- OS: Linux 6.8.0-101-generic x86_64
- Rust: 1.97.1
- Criterion: 10 samples, 300 ms warmup, 1 s measurement
- sizes: 4,096; 32,768; 262,144; 1,048,576 elements
- contexts: serial and `ExecContext::max_threads(4)`; groups ran sequentially
- load gate: selected cores and the rest of their L3 domain checked before each run

Most baseline/candidate pairs used the same L3 domain. The generic-gather pair
used different domains; its worklog records the approximately 17% control shift,
which is small relative to the reported generic speedups.

Implementation checkpoints:

| Family | Candidate commit | PR |
|---|---|---|
| generic gather | `acdeea3f` | strided-rs #242 |
| dynamic slice/update | `75fb0f70` | strided-rs #244 |
| erased axis reduction | `37ce20b7` | strided-rs #245 |
| generic pad | `f875cc89` | strided-rs #246 |
| integer zero preflight | `53ecd771` | strided-rs #248 |
| generic additive scatter | `b40cd2f6` | strided-rs #252 |

## Medium Results

`N = 262,144`. Times are Criterion point estimates in milliseconds.

### Representative compact rank-8 timings

| Family | Serial baseline → candidate | Speedup | 4-thread baseline → candidate | Speedup |
|---|---:|---:|---:|---:|
| gather | 27.882 → 2.456 | 11.35x | 2.633 → 0.697 | 3.78x |
| dynamic slice | 17.956 → 0.618 | 29.07x | 1.669 → 0.219 | 7.63x |
| dynamic update | 19.572 → 0.632 | 30.97x | 1.700 → 0.254 | 6.69x |
| single-axis reduction | 4.950 → 0.846 | 5.85x | 1.231 → 0.229 | 5.39x |
| pad | 12.965 → 1.694 | 7.65x | 1.722 → 0.503 | 3.42x |
| integer divide preflight | 4.103 → 1.267 | 3.24x | 4.349 → 1.147 | 3.79x |
| additive scatter | 21.269 → 1.392 | 15.28x | 20.870 → 1.383 | 15.09x |

### Rank scaling

| Family | Serial rank 2 | Serial rank 4 | Serial rank 8 | 4T rank 2 | 4T rank 4 | 4T rank 8 |
|---|---:|---:|---:|---:|---:|---:|
| gather | 5.76x | 7.56x | 11.35x | 2.35x | 2.65x | 3.78x |
| dynamic slice | 8.74x | 17.26x | 29.07x | 2.79x | 4.56x | 7.63x |
| dynamic update | 8.05x | 16.23x | 30.97x | 2.50x | 3.81x | 6.69x |
| single-axis reduction | 2.57x | 3.33x | 5.85x | 2.32x | 3.05x | 5.39x |
| pad | 2.46x | 4.42x | 7.65x | 1.34x | 2.09x | 3.42x |
| integer divide preflight | 1.33x | 1.82x | 3.24x | 1.84x | 2.27x | 3.79x |
| additive scatter | 1.56x | 6.24x | 15.28x | 1.60x | 6.18x | 15.09x |

### Generic layout controls

| Family | Serial non-unit | Serial negative/crop | 4T non-unit | 4T negative/crop |
|---|---:|---:|---:|---:|
| gather | 5.71x | 5.72x | 2.34x | 2.34x |
| dynamic slice | 8.97x | 7.18x | 2.79x | 2.34x |
| dynamic update | 8.01x | 7.38x | 2.46x | 2.22x |
| single-axis reduction | 3.37x | 3.35x | 3.02x | 2.85x |
| pad | 1.99x | 2.29x | 1.41x | 1.34x |
| integer divide preflight | 1.57x | 1.32x | 1.71x | 1.59x |
| additive scatter | 1.58x | 1.57x | 1.60x | 1.58x |

Existing specialized controls were retained in every upstream matrix. No
accepted implementation had a material specialized-control regression.

## Interpretation

The baseline rank slope came from rebuilding coordinates and checked offsets
inside tensor-sized loops. Preparing or maintaining incremental offsets removes
that rank-multiplied work. Four-thread gains are smaller where memory traffic
or existing partition overhead dominates, but remain positive for the generic
layouts.

The scatter series uses the corrected typed-storage black-box pair documented
in `docs/worklogs/2026-08-25-issue-251-generic-scatter.md`; its earlier
descriptor-only black-box baseline is explicitly INCONCLUSIVE. The pad series
uses the corrected same-domain run at strided-rs worklog commit
`03ed964e`; its earlier 7.5%-load candidate is explicitly reclassified
INCONCLUSIVE. The integer-preflight series includes a documented discarded run
where a shared Cargo target reused the baseline binary. Accepted
baseline/candidate runs used separate target directories; unchanged Add
controls were independently rerun. See strided-rs #213 and the dated worklogs
for confidence intervals, correctness gates, and measurement incident details.

## Review provenance

- Initial publication commit `4dc92e95` received an exact read-only
  `reviewer-flash` high-thinking verdict of **Correct-to-merge**; follow-up
  `ff0998de` incorporated its reproducibility findings before PR #34 merged.
- Corrected pad evidence commits `03ed964e` (strided-rs worklog) and `68d719e0`
  (this page) received a joint exact read-only `reviewer-flash` high-thinking
  verdict of **Correct-to-merge** before PRs #249 and #35 merged.
- PR bodies preserve the exact reviewed commits and verdicts; no benchmark
  implementation or public library API was added in this repository.
