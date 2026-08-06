# The measurement record

This directory is the evidence behind every performance claim the project makes.
It exists because the claims are only worth what their provenance is worth, and
because roughly half of the obvious ideas here have already been measured and
lost.

If you are here to *use* the library, you want the [top-level
README](../README.md) instead. Nothing below is needed to call the engine.

## Where to look

| when | read |
|---|---|
| **what is measured, and how well** | [`results.md`](results.md) — the single source for every number quoted anywhere else in the repository |
| **before proposing a performance idea** | [`refuted.md`](refuted.md). Each entry says what killed it *and what would reopen it* |
| what is still unknown | [`open-questions.md`](open-questions.md) |
| before designing a measurement | [`measurement-rules.md`](measurement-rules.md) — machines, baselines, noise floors, and nine rules that each cost a measurement |
| the full account behind any `A<n>` or `D<n>` | [`decisions.md`](decisions.md), then the [`notebook/`](notebook/README.md) chapter its last column names |
| the architecture | [`design.md`](design.md) |
| the raw data | [`../bench-results/`](../bench-results/README.md), one `PROVENANCE.txt` per directory |

## If you read nothing else

Eight results, each of which cost a measurement, and each of which this project
got wrong at least once first.

1. **The complex-weakness thesis is refuted.** The founding hypothesis — that
   interleaved complex storage compounds with scatter/gather — is false. There
   *is* a ~5x gap against TBLIS **v1.3.0**, for the mundane reason that 1.x has
   no complex micro-kernel outside Sandy Bridge, and **none** against 2.0-dev.
2. **The real headroom is low arithmetic intensity in either domain**, not
   complex. TBLIS 2.0 runs 0.85 of a same-shape GEMM ceiling on large
   compute-bound contractions and **0.34** on small-`k` skinny ones.
3. **The corpus is 49 cases, not 48** (D11, [`design.md`](design.md) §5.2).
4. **The corpus is not "fully regular."** That shorthand is wrong and steered
   conclusions for three phases (A4). TCCG rounds stride-1 extents to multiples
   of 24, which is regular only at a register block that *divides* 24. Quote the
   `--stress` mode and the observed `reg_a` with any awkward-stride claim.
5. **Write-back regularity is not a performance predictor.** It is a gate on a
   change, never an objective; it has pointed the wrong way three times.
6. **The complex-method ranking and the memory-bound inversion are Cascade Lake
   results** and do not transfer (A44). Planar wins the corpus on both AVX-512
   machines measured; nothing below that may be quoted without naming the
   machine.
7. **`K`-parallelism is ruled out on evidence** (A21), and D45 does not reopen it.
8. **Register blocks are measured**, and are per-*microarchitecture*, not per-ISA
   (A34). Do not re-derive them; `examples/kernel_shapes` re-derives them if the
   machine changes.

And one rule that generalises all of them: **anything touching threads or caches
is a per-microarchitecture claim until shown otherwise** (A56). Four have now
failed to transfer — register blocks, the thread partition, the complex-method
ranking, and the thread pool. Two machine classes is the minimum for a
recommendation, and this project has been wrong on one class twice.

## Before believing any number here

* **No number measured on one machine may be compared with a number measured on
  another.** Every ratio is within-session, against a floor derived in that
  session, and a floor is meaningless without its thread count as well as its
  session.
* Every number has committed raw data. If you cannot find the
  `PROVENANCE.txt`, treat the number as unsupported.
* The fresh-checkout sanity check is in
  [`measurement-rules.md`](measurement-rules.md); its last step re-derives a
  published noise floor from committed data without touching the CPU.
