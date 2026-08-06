# The notebook

The measurement narrative, one file per topic. This is the *long* form: what was
tried, what happened, what it cost, and what was got wrong on the way.

You probably want [`../results.md`](../results.md) (what is measured),
[`../refuted.md`](../refuted.md) (what was measured and lost, with what would
reopen it) or [`../decisions.md`](../decisions.md) (the A- and D-numbers) first.
Come here when one of those points you at a chapter.

**A report names the numbers it introduced and does not restate them.** A
decision goes in [`../decisions.md`](../decisions.md) and nowhere else.

## Chapters

Live chapters first, closed phases last.

| chapter | reports | subject |
|---|---|---|
| [`sessions.md`](sessions.md) | part 10 | node choice as experiment design, the placement pre-registration, the zero-CPU tooling |
| [`shape-rules.md`](shape-rules.md) | parts 1–5, 6, 13 | the 2x write-back defect and its real cause; the row-block menu and rule; the orientation rule and its discriminant; the menu keyed by position |
| [`cache-blocking.md`](cache-blocking.md) | parts 7, 9 | the `MC`/`KC`/`NC` grid and the A/B that closes item 2; the analytical model, and it loses |
| [`kernels.md`](kernels.md) | AVX2 interlude, part 11 | AVX2 from the same macro bodies; the AVX2 register blocks measured; A24, A34, A35 |
| [`threading.md`](threading.md) | parts 8, 8b, 12, 14–19 | the scheme and its scaling on seven nodes; the 2-D partition; the domain-aware gate; load imbalance; what TBLIS does; the two experiments that decide the default; the answers to the spawn cost and the batch axis |
| [`packaging.md`](packaging.md) | Phase 5 parts 1, 2, C-surface interlude | quality gates, the API tiers, the TAPP conformance suite, the C header and consumer, cross-compilation, the JLL and the Julia package |
| [`baselines.md`](baselines.md) | Phase 5 part 3 | the engine against TBLIS and TTGT, re-measured; A44–A46 |
| [`archive-phases-1-3.md`](archive-phases-1-3.md) | Phases 1, 2, 2b, 3 | closed and unlikely to reopen: the premise check, the correct engine, the three methods, the AVX-512 kernels |

## Part number → chapter

For citations written before the reports were grouped by topic.

| part | chapter |
|---|---|
| 1, 2, 3, 4, 5 (one section, "part 1") | [`shape-rules.md`](shape-rules.md) |
| 6 | [`shape-rules.md`](shape-rules.md) |
| 7 | [`cache-blocking.md`](cache-blocking.md) |
| 8, 8b | [`threading.md`](threading.md) |
| 9 | [`cache-blocking.md`](cache-blocking.md) |
| 10 | [`sessions.md`](sessions.md) |
| 11 | [`kernels.md`](kernels.md) |
| 12 | [`threading.md`](threading.md) |
| 13 | [`shape-rules.md`](shape-rules.md) |
| 14, 15, 16, 17, 18, 19 | [`threading.md`](threading.md) |
| Phase 5 parts 1, 2, and the C-surface interlude | [`packaging.md`](packaging.md) |
| Phase 5 part 3 | [`baselines.md`](baselines.md) |

These files were one 5277-line `DECISIONS.md` until 0.1.0. The split was pure
motion — `git log --follow` on any chapter reaches the whole history — and the
trim that followed it is a separate commit, so the two are reviewable apart.
