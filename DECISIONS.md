# Decisions, assumptions and phase reports

The audit trail: every material decision, every assumption and what became of it,
and every phase report with the data behind it.

**How this file is organised.** Reference material first — [state](#resume-here),
[index](#index), [machines](#environment), [measurement rules](#measurement-rules),
[assumptions](#standing-assumptions), [decisions](#design-decisions) — then the
reports, grouped by **topic**. Each report keeps the `part N` name it was written
under, so every existing citation still resolves; the [index](#index) maps part
numbers to chapters.

**Refuted results live in [`REFUTED.md`](REFUTED.md)**, one entry per thing tried,
with a confidence level, evidence, and — the field that matters — what would
reopen it. The full accounts stay here and `REFUTED.md` points at them. It is not
part of the per-session reading list; consult it before proposing any performance
idea.

**How to add to this file.** A report goes into the topic chapter it belongs to,
not at the end. A decision goes in the [decisions table](#design-decisions) and
nowhere else — reports name the numbers they introduced and stop. An assumption is
*restated* wherever its status changes; that duplication is deliberate and is how
a refutation is tracked. "Resume here" is **rewritten**, never appended to. A
refutation gets an entry in `REFUTED.md` in the same commit as the measurement.

---

## Resume here

**State, 2026-08-06.** Tests green (9 suites), `clippy --workspace --all-targets`
silent, `fmt` clean. Workspace MSRV 1.89. Two lines of work are open and they are
independent; the Apple one is the live one.

### START HERE if you are picking up the Apple Silicon work

Read **parts 20, 21 and 22** and the two Environment subsections, then this.

**Stages A and B are both complete.** The NEON micro-kernel is built, green, its
register blocks are measured, and its end-to-end value is measured against a
floor. Nothing here is under construction.

**What it bought** (part 22, `bench-results/CKF6QCDVPD-m3max/neon-ab/`, 12
premise shapes, 64 MiB, 1 thread, floor ±1%):

| | scalar | NEON | |
|---|---|---|---|
| `f64` | 19.90 | **36.60** GF/s | **1.840** |
| `c64` planar / 1m / 3m | 24.34 / 23.31 / 29.85 | 44.46 / 41.74 / **50.45** | 1.826 / 1.791 / 1.690 |
| against TBLIS 2.0-dev | `f64` 0.54 | **1.00** | `c64` 3m 0.66 → **1.12** |
| against OpenBLAS-TTGT | `f64` 0.63 | **1.16** | `c64` 3m 0.85 → **1.42** |

Controls the switch cannot reach moved 0.996–1.006. Best-shape efficiency is
**81.4% of the 64.8 GF/s NEON FMA peak**, against the scalar path's 80.5% of a
ceiling halved by the missing `fmla` — same efficiency, twice the ceiling, which
is A58 confirmed by removal.

**Three things this changes that are easy to quote wrongly.**

1. **The 0.46x-of-TBLIS figure is superseded.** At 64 MiB the scalar path is
   **0.54**; the NEON path is 1.00. Quote 0.54, and only for the scalar arm.
2. **3m leads on this machine with a tuned kernel.** 3m/planar goes 1.226 →
   **1.135** — it narrowed and did not invert, and A59 guessed otherwise. So
   "3m is a fair-comparison method, not a candidate default" is now a claim
   about **x86**; here it is the fastest complex method and the arm that passes
   TBLIS.
3. **`f64` real ships `16x3`**, whose `MR = 16` does not divide TCCG's 24. The
   guarded row-block rule demotes the 12 affected cases; do not "fix" this by
   changing the menu.

| Stage B step | state |
|---|---|
| extract `simd_kernels!` / `configs!` into `kernel::simd` | **done.** Bodies unchanged; `#[macro_use] mod simd;` must stay declared before the ISA modules |
| `kernel::aarch64`, `Isa::Neon`, `TENSORCONTRACT_KERNEL=neon` | **done.** All four kernels, `f64` and `f32` |
| generalise `examples/kernel_shapes` off x86 | **done** — NEON reuses the AVX-512 candidate grid, because both have 32 registers and the budget counts registers not lanes |
| run `kernel_shapes` and replace the budget-derived menus | **done, three arms, part 21.** Six of the eight winners are **ties** and say so; the budget got three of eight wrong (A60) |
| the scalar-vs-neon A/B at 64 MiB with its own floor | **done, part 22.** `scripts/macos-neon-ab.sh` |
| re-measure the complex-method ranking | **done, part 22.** It narrowed, it did not invert |
| the blocking constants (`kernel/mod.rs:320-333`) | **not started, and this is now the only open Apple engine item.** A NEON kernel is no longer instruction-bound, so the cache effects those constants exist for are finally visible. `legacy_blocking_is_unchanged` pins the current values. It would also re-open `f64` real `16x3`, whose win is measured at `kc = 256` and which is **bimodal at `kc = 16`** |

**Cross-compile after any change here.** The x86 tests cannot run on this machine,
and that is not theoretical: `cargo check --target x86_64-unknown-linux-gnu` is
what caught a missing `KernelForce::Neon` arm in x86's `pick_isa`. Three targets
are cheap and all pass — `x86_64`, `i686`, `aarch64` linux.

#### What is still owed on this machine

**One measurement, and it is the interrupted Stage A corpus session.** Part 22
gave the machine a floor and a 64 MiB TBLIS number, which closes one of the
three gaps part 20 left. Two remain, and both need the corpus rather than the 12
premise shapes: **no `ragged` arm** — and `ragged` is the only source of
irregularity here — and **no per-case corpus spread**, so nothing per-case over
the 49 cases is quotable.

```bash
scripts/macos-session.sh prep  bench-results/CKF6QCDVPD-m3max   # REQUIRED, see below
scripts/macos-session.sh bench bench-results/CKF6QCDVPD-m3max 64 3   # ~5 h
```

It is now a **more valuable** run than when it was planned, because its default
arm exercises the NEON kernel over all 49 cases rather than the portable path
over 49. Consider `SIZE=32` or dropping `1m`/`3m` from `ENGINES` if 5 h is too
long; either is fine as long as the write-up says which.

**What that Stage A session did and did not write.** `info`, `orient`, all three
`verify` arms and both `premise` arms completed and are committed. **The five
corpus `sweep` arms, `ragged` and the TBLIS sweep were never written** — one arm
costs ~50 min here — and neither were its floor files. Part 22 has since supplied
a floor (±1% on geomeans, ~±10% per case) and the 64 MiB TBLIS number, so part
20's ratios are no longer floorless; what is still missing is corpus-wide, which
is what the run above is for.

**`prep` must be re-run before it, and this changed on 2026-08-06.** The three
binaries in that directory's `bin/` were built at 09:29, before the NEON kernel
existed — `strings` finds no `neon-real` in any of them — so running `bench`
against them would measure the portable path and label it as the current engine.
That is D51's hazard arriving from the other side: not a build racing a running
job, but a stale build outliving its source. After `prep`, **do not compile while
it runs** — the arms invoke those binaries.

**Do not** try to port the driver scripts, thread anything on this machine, or make
the analytical block model the default here. The reasons are in part 20 and in the
plan's out-of-scope section; A33 and A57 cover the last one.

### Phase state

| phase | state |
|---|---|
| 1 — exploration and design | **complete.** Premise resolved with data; the founding thesis is refuted and the displaced finding is better (see `REFUTED.md`) |
| 2 / 2b — correct engine, three complex methods | **complete** |
| 3 — micro-kernels | **complete.** AVX-512 for `f32`/`f64` and all three complex methods; AVX2 added later; **NEON added 2026-08-06 and its register blocks measured** (part 21). AVX-512 and NEON blocks are measured, AVX2's are still budget-derived |
| 4 — profiling and improvement | **in progress.** Items 1, 1c, 1d closed with wins; item 2 closed with a negative result; **item 3 dropped** (below); item 4 built and measured on seven nodes, and its three answers to the per-call spawn cost — the amortisation guard, the thread pool, the batched API — are built and measured (parts 17–19): **the pool's recommendation is WITHDRAWN and both switches stay off (D53), the guard does not ship (D52), the batched API is unmeasured.** Threads still default to 1 |
| 5 — packaging | **in progress.** C surface, distribution surface, Julia consumer all exist. **Nothing published, nothing tagged**, on purpose |
| — Apple Silicon | **Stage A complete** (part 20) and **Stage B complete** (parts 21, 22): the NEON kernel is built, green, its register blocks are measured, and the end-to-end A/B puts it at **1.840** with a 1% floor — the engine reaches **1.00x TBLIS 2.0-dev** in `f64` and **1.12x** in `c64` with 3m. What remains is the cut-short Stage A corpus session (no `ragged` arm, no per-case corpus spread) and the blocking constants |

### The one open engine question

**What Phase 4 bought, end to end, is still unmeasured.** The current
engine-vs-baseline numbers are Ice Lake (jobs 6753260 and 6754877, `worker6156`)
and the Phase 3 table they would be differenced against is Cascade Lake. Two
things changed at once, so the difference is not attributable. Only a
`ccqlin038` run supplies it:

```bash
TBLIS_ROOT_2X=../baselines/tblis-2.0-install \
TBLIS_ROOT_13=../baselines/tblis-1.3.0-install \
  scripts/compare-bench.sh prep                       # the only compile
  scripts/compare-bench.sh bench-results/$(hostname -s)-$(scripts/arch-label.sh)
```

~3.5 h, and it wants the machine to itself.

### Item 3 is dropped, and what replaces it

**Phase 4 item 3 — "dispatch 3m on memory-bound shapes, planar otherwise" — is
dropped as a rule** (user's decision, 2026-08-05). Its premise was the ranking
inversion Phase 3 measured on Cascade Lake, and A44 shows the inversion is absent
on Ice Lake, where 3m is last in every column and wins 0 of 49 cases. An
unconditional rule is a pessimisation there; a per-microarchitecture rule needs
per-microarchitecture shape tables that do not exist.

**Nothing is owed to it.** The experiment that would have deconfounded A44 from
A34 had already been run and committed — `bench-results/worker6016-icelake/kernel-shapes.txt`
covers 3m — and it says the confound does not exist: 3m already runs its Ice Lake
optimal shape. The collapse is uniform across 3m's whole shape space, and at the
L1-resident `kc = 16` 3m leads by 10–16% on Cascade Lake and trails by 34–43% on
Ice Lake. So **3m's L1-resident advantage is a Cascade Lake result too**, which
this project had carried as settled mechanism. See part 3, "the A34 confound does
not exist", and `REFUTED.md`.

**Consequence for quoting: the ranking and the inversion are Cascade Lake
results.** Never state either as a property of the engine.

### What to do next, in order

**The previous list's items 1–4 are done** (part 17: the per-job build tree, the
guard, the pool, the batched API) **and two of the three answers are now measured**
(part 18, job 6760092). The pool wins, the guard is refuted in the form built, and
the batched API remains unmeasured.

1. **Nothing is waiting on you, and one thing needs undoing in your head if you read
   the earlier recommendation.** `TENSORCONTRACT_POOL=on` was recommended as a default
   on Zen2 evidence (D53) and **the recommendation is withdrawn**: on Ice Lake the same
   switch costs up to 2.5x, and at the pre-registered decision cell it reads 0.80
   against a 0.97 threshold (part 19). Both switches stay off. `TENSORCONTRACT_AMORTISE`
   does not ship either (D52).

   The pool is still worth up to 11.6x on Zen2, so it is a *conditional* switch with an
   implementation defect to find, not a dead end. The suspected cause is 32 long-lived
   per-thread buffers at repeating addresses contending in one shared L3 (A55), and the
   cheap test is to offset each worker's buffer by a per-worker amount and re-run both
   nodes' arms:

   ```bash
   ARMS="base;pool:TENSORCONTRACT_POOL=on" STAGES=small SIZES="1 16 64" \
     sbatch --constraint=icelake scripts/rusty-phase4.sbatch   # ~1.8 h
   ```

2. **Find out why the pool's win scales with size — do not build the fix that was
   proposed for it.** Part 18 offered the per-thread `Panel` allocation as the
   candidate and the same part refutes it: `ap_len` is identical at all four sizes, so
   hoisting those allocations has no measured basis (A53). What is solid is a fixed
   ~37 µs/thread; the scaling component is unexplained, and the two untested candidates
   are scheduler placement of freshly created threads and barrier skew growing with
   `N/NC` x `K/KC`. **This is a question, not a task**, and the cheapest form of it is a
   microbenchmark that separates spawn from placement from barrier count — no corpus
   and no exclusive node needed.
3. **Block-sparse**, the unbuilt half of the batched API. The batch axis exists and
   is static; block-sparse is where items differ in *size* rather than in
   regularity, which is the one place D47's null result does not reach, and where
   a dynamic claim over the batch belongs.
4. **Close the remaining orientation gap.** 21 case-dtype-methods still take the
   slower arm, worth up to 1.36x, and they are a different population from the
   `abcijk` family the rule was derived on. Both arms are already in
   `bench-results/phase4d/or-*`, so a candidate costs no machine time:
   `scripts/orient-score-rules.py`.
5. **Re-run the row-block grid at the chosen `kc`.** `kc` decides the regime and
   the shape is chosen inside it; the Phase 3 shapes were chosen at a depth the
   engine may no longer use.
6. **A35's A/B**, reachable since D43 and never run:
   `scripts/ab.sh bench-results/ab-rowblock-idx3 "TENSORCONTRACT_ROWBLOCK=idx=3"`.
7. Then the rest of Phase 4: small-`k` handling, `pc`-loop fusion **for small `K`
   only** (it is not the general enabler earlier drafts implied — see
   `REFUTED.md`), a pack-free fast path for already-unit-stride block scatter,
   prefetch.
8. Route the batch axis through the thread pool too, removing its one remaining
   spawn set.

Items 2, 3, 4 and 8 need no cluster time; items 1, 5 and 6 want an exclusive machine.

### Settled — do not re-derive

Each of these cost a measurement. The full entry, with what would reopen it, is
in [`REFUTED.md`](REFUTED.md).

1. **The complex-weakness thesis** is refuted against TBLIS 2.0-dev (1.06) and
   confirmed against TBLIS v1.3.0 (0.215), for a mundane reason: 1.x has no
   complex micro-kernel outside Sandy Bridge.
2. **The real headroom is low arithmetic intensity in either domain**, not
   complex. TBLIS 2.0 runs 0.85 of a same-shape GEMM ceiling on large
   compute-bound contractions and 0.34 on small-`k` skinny ones.
3. **The corpus is 49 cases, not 48** (D11, `DESIGN.md` §5.2).
4. **The corpus is not "fully regular"** — that shorthand is wrong and steered
   conclusions for three phases (A4). Quote the `--stress` mode and the observed
   `reg_a` with any awkward-stride claim.
5. **Write-back regularity is not a performance predictor.** It is a gate on a
   change, never an objective; it has pointed the wrong way three times.
6. **The complex-method ranking and the memory-bound inversion are Cascade Lake
   results** and do not transfer (A44). Planar wins the corpus on both AVX-512
   machines measured; nothing below that may be quoted without naming the machine.
7. **`K`-parallelism is ruled out on evidence** (A21), and D45 does not reopen it.
8. **Register blocks are measured** — AVX-512 on Cascade Lake, AVX2 on Zen2 — and
   are per-*microarchitecture*, not per-ISA (A34). Do not re-derive them;
   `examples/kernel_shapes` re-derives them if the machine changes.

### Before believing anything in this file

* **No number measured on one machine may be compared with a number measured on
  another.** Every ratio here is within-session against a floor derived in that
  session. See [measurement rules](#measurement-rules); floors are per session
  *and per thread count*.
* Raw CSVs for every number are committed under
  [`bench-results/`](bench-results/README.md), one `PROVENANCE.txt` per directory.
* Sanity check on a fresh checkout, in this order:

```bash
cargo test --workspace --release              # 9 green suites
TENSORCONTRACT_KERNEL=scalar cargo test --workspace --release
cargo clippy --workspace --all-targets        # silent
cargo build --release -p tensorprimitives-bench
cd bench-results/phase4 && python3 ../../scripts/compare-sweeps.py \
    rm-A-f64c64.csv,rm-A-f32c32.csv rm-A2-f64c64.csv,rm-A2-f32c32.csv
```

The last one re-derives a published noise floor from committed data and should
print geomeans of 0.99–1.01 without touching the CPU. If it does not, the
analysis tooling has drifted from the numbers in this file.

---

## Index

Reference sections, kept current:

| | |
|---|---|
| [Resume here](#resume-here) | state, the open question, what to do next |
| [Environment](#environment) | the reference machine, the cluster nodes, both TBLIS baselines, the ABI break |
| [Measurement rules](#measurement-rules) | the noise floors, each with its session and thread count, and the rules that produced them |
| [Standing assumptions](#standing-assumptions) | A1–A56, one line each, with where the full account is |
| [Build-vs-reuse decisions](#build-vs-reuse-decisions) | every dependency taken or declined |
| [Design decisions](#design-decisions) | D1–D47, complete, in one place |

Reports, by topic. Live chapters first, closed phases last:

| chapter | reports | subject |
|---|---|---|
| [Running a measurement session](#running-a-measurement-session) | part 10 | node choice as experiment design, the placement pre-registration, the zero-CPU tooling |
| [The write-back and the two shape rules](#the-write-back-and-the-two-shape-rules) | parts 1–5, 6, 13 | the 2x defect and its real cause; the row-block menu and rule; the orientation rule and its discriminant; the menu keyed by position |
| [Cache blocking](#cache-blocking) | parts 7, 9 | the `MC`/`KC`/`NC` grid and the A/B that closes item 2; the analytical model, and it loses |
| [Micro-kernels across instruction sets](#micro-kernels-across-instruction-sets) | AVX2 interlude, part 11 | AVX2 from the same macro bodies; the AVX2 register blocks measured; A24, A34, A35 |
| [Threading](#threading) | parts 8, 8b, 12, 14, 15, 16, 17, 18, 19 | the scheme and its scaling on seven nodes; the 2-D partition; the domain-aware gate; load imbalance; what TBLIS does; the two experiments that decide the default; three answers to the spawn cost and the batch axis |
| [Packaging and distribution](#packaging-and-distribution) | Phase 5 parts 1, 2, C-surface interlude | quality gates, the API tiers, the TAPP conformance suite, the C header and consumer, cross-compilation, the JLL and the Julia package |
| [The baseline comparison](#the-baseline-comparison) | Phase 5 part 3 | the engine against TBLIS and TTGT, re-measured; A44–A46 |
| [Apple Silicon](#part-20--apple-silicon-and-what-the-portable-path-actually-costs) | parts 20, 21, 22 | the portable path's real cost and the one instruction behind it; three honest-reporting fixes; the NEON register blocks measured, and the budget that filtered well and chose badly; the kernel worth **1.84x**, the engine reaching TBLIS, and 3m leading a fourth ordering |
| [Archive: phases 1–3](#archive-phases-13) | Phase 1, 2, 2b, 3 | closed and unlikely to be reopened: the premise check, the correct engine, the three methods, the AVX-512 kernels and the three-way comparison |

**Part number → chapter**, for citations written before this file was
reorganised:

| part | chapter |
|---|---|
| 1, 2, 3, 4, 5 (one section, "part 1") | the write-back and the two shape rules |
| 6 | the write-back and the two shape rules |
| 7 | cache blocking |
| 8, 8b | threading |
| 9 | cache blocking |
| 10 | running a measurement session |
| 11 | micro-kernels across instruction sets |
| 12 | threading |
| 13 | the write-back and the two shape rules |
| 14, 15, 16, 17, 18, 19 | threading |
| 20, 21, 22 | Apple Silicon |
| Phase 5 parts 1, 2, and the C-surface interlude | packaging and distribution |
| Phase 5 part 3 | the baseline comparison |

Two things a citation may trip over:

* **`A31` and `A32` were each used for two unrelated claims.** The methodology
  ones keep their numbers, because every citation in the repository means those:
  `A31` is the cold-start artefact, `A32` is drift-as-a-function-of-separation.
  The Phase 5 C-surface pair was **renumbered to `A47` (the shipped header agrees
  with the library) and `A48` (a panic at the C boundary is acceptable)**. Any
  citation of A31/A32 predating 2026-08-05 means the methodology ones.
* **`A27` appears twice on purpose** — once as the pre-registered hypothesis,
  once with its verdict — as do A7, A8, A9, A10, A11, A12, A24, A28, A36 and A41.
  That is how a refutation is tracked, not a collision. The threading branch's
  `A37`–`A43` kept their numbers when the docs branch merged; the docs branch's
  three became `A44`–`A46`.
* **`A49` does not exist.** Part 17's three assumptions were drafted as A47–A49
  and renumbered to `A50`–`A52` when the C-surface pair took A47/A48. A gap is
  cheaper than a third meaning for a number.
* Two references to "parts 11–13" in the Phase 5 reports were written when those
  parts did not exist and meant **parts 10 and 11**; they have been corrected in
  place.

---

## Environment

### The reference machine

The only machine the shipped blocking constants and the AVX-512 register blocks
were fitted to, and the one every `phase3-*` / `phase4*` number was measured on:

| | |
|---|---|
| host | `ccqlin038.flatironinstitute.org` (Flatiron / CCQ workstation) |
| CPU | Intel Xeon Gold 6244, Cascade Lake-SP, 2 sockets x 8 cores, 3.6 GHz base |
| ISA | AVX-512F/DQ/BW/VL/VNNI, 2 FMA units per core |
| cache | 32 KiB L1d, 1 MiB L2 per core, 25.3 MiB shared L3 per socket |
| memory | 251 GiB, 2 NUMA nodes |
| toolchain | rustc 1.97.1, gcc 13.3.0 (module), cmake 3.31.6; workspace MSRV 1.89 (D20) |
| BLAS | OpenBLAS 0.3.29 (`openblas/single-0.3.29` module) |

It is **shared during working hours.** Long benchmarks go off-hours; the
zero-CPU analyses are what to run during the day.

Reference for the single-core ceiling: OpenBLAS `dgemm` reaches ~96 GF/s and
`zgemm` ~96 GF/s on large square shapes here, so the two domains have effectively
the same ceiling (A3), which is what makes the efficiency-ratio metric mean
something.

### The cluster nodes

Rusty and Popeye are Slurm systems and **have no Cascade Lake**, so every cluster
session answers a portability question rather than "is it fast". Submitting is a
human step and is never automated.

| node | uarch | ISA | topology | date | job | what it measured |
|---|---|---|---|---|---|---|
| `worker5040` | Zen2 (`rome`) | AVX2, **no AVX-512** | 64 cores/socket, 4 per 16 MiB L3 → 16 domains | 2026-08-03 | 6745376 | AVX2 register blocks; first thread scaling; the blocking grid; placement validation |
| `worker5175` | Zen2 | AVX2 | as above | 2026-08-03 | 6745978 | the three traffic-changing blocking arms, sequentially (A27) |
| `worker6016` | Ice Lake-SP | AVX-512 | 2x32 cores, **one 48 MiB L3 per socket** | 2026-08-03 | 6746817 | the second Intel hierarchy: A34 (blocks are per-uarch) and A36 (the partition win is absent) |
| `worker5137` | Zen2 | AVX2 | as `worker5040` | 2026-08-04 | 6751550 | that the partition effect follows **L3 domains spanned**, not thread count |
| `worker5479` | Zen2 | AVX2 | as `worker5040` | 2026-08-04 | 6753208 | the domain-gate confirmation run, Zen2 arm |
| `worker6150` | Ice Lake-SP | AVX-512 | as `worker6016` | 2026-08-04 | 6753209 | the same, Ice Lake arm: **0 of 392 cases moved** |
| `worker5139` | Zen2 | AVX2 | as `worker5040` | 2026-08-04 | 6754849 | small contractions: the spawn cost, and why a fixed thread default is wrong |
| `worker5178` | Zen2 | AVX2 | as `worker5040` | 2026-08-04 | 6755009 | `--stress ragged`: load imbalance does not cost |
| `worker6156` | Ice Lake-SP | AVX-512, SMT off | as `worker6016` | 2026-08-04/05 | 6753260, 6754877 | **the engine against its baselines**, twice; the TBLIS-build A/B |

### The Apple Silicon machine

The first non-x86 machine, and the first that is **not a cluster node and cannot be
pinned**. It runs `kernel::scalar`, because there is no NEON path.

| | |
|---|---|
| host | `CKF6QCDVPD`, Apple M3 Max (`T6031`), macOS 25.5, `aarch64-apple-darwin` |
| cores | 12 P in 2 clusters of 6, plus 4 E. **No SMT** |
| ISA | NEON, 4 x 128-bit FP/SIMD pipes per P-core; FP16, BF16, DotProd. **No SVE, no SME** |
| cache | P: 128 KiB L1d per core, **16 MiB L2 shared per 6-core cluster**. E: 64 KiB / 4 MiB. **No L3 Darwin will name**; a ~48 MiB SLC exists and is not exported |
| line / page | **128-byte line**, 16 KiB pages |
| memory | 64 GiB unified, one NUMA domain |
| clock | 3.6 GHz base, **4.05 GHz P-core turbo — on AC power only**, which is why the driver records the power state |
| toolchain | rustc 1.91.1, Apple clang, cmake 4.4.2 (Homebrew), OpenBLAS 0.3.34 (Homebrew) |
| driver | `scripts/macos-session.sh` — **the only one that runs here**; 17 others cannot |

**Peak, for reading every number on this machine against something.** Four
128-bit FP pipes at 4.05 GHz is 16 f64 flops/cycle with FMA, i.e. **64.8 GF/s per
P-core** (129.6 in `f32`). Roughly half a 2-FMA-unit AVX-512 core. But the
portable path cannot reach it — see part 20 — because `acc += a * b` is two
instructions where `fmla` is one, which halves the ceiling to **32.4 GF/s**.

### The three baselines here, and which one is a roofline

| | |
|---|---|
| OpenBLAS | 0.3.34, Homebrew, `DYNAMIC_ARCH`. **Selects its `neoversen1` kernel on Apple P-cores**, not `vortex`. NEON. **This is the roofline** |
| Accelerate | no install. **Not a roofline — it is off-ISA**, see below |
| TBLIS | 2.0-dev @ `555320c`, the same commit the Linux runs used, built `BLIS_CONFIG_FAMILY=arm64` / `BLIS_THREAD_MODEL=pthread`. BLIS selects **`firestorm`**, Apple's own P-core sub-configuration (`BLIS_ARCH_DEBUG=1` prints it), with real ARM assembly kernels — `bli_dgemm_armv8a_asm_6x8` and `8x6r`, `bli_sgemm_armv8a_asm_12x8r` and `8x12`, confirmed with `nm` and not `strings` (A45) |
| TBLIS 1.3.0 | **deliberately not built.** No complex micro-kernel outside Sandy Bridge and no ARM kernel, so here it would be a reference path measured against a reference path |

**Accelerate is off-ISA and the measurement proves it, so A3's roofline metric does
not mean here what it means on x86.** Its TTGT reaches a **geomean 79.2 GF/s in
`f64` and 99.4 in `c64`, peaking at 334.8** on the premise shapes — **5.2x the
64.8 GF/s NEON FMA peak of a P-core.** No NEON kernel can produce that number, so
it is not NEON: Accelerate's GEMM goes through Apple's undocumented AMX matrix
coprocessor. OpenBLAS on the same shapes reads 30.7 and 35.0. **Quote the engine
against OpenBLAS; quote Accelerate separately and say what it is.** A ratio against
Accelerate is not an efficiency-vs-achievable-ceiling number, and merging the two
CSVs would silently make it look like one.

The recipe for all three is in `scripts/macos-session.sh`'s header. Both TBLIS and
OpenBLAS carry absolute install names, so **no `DYLD_LIBRARY_PATH`** — which
matters, because SIP strips `DYLD_*` across `/bin/sh` wrappers and every Linux
driver here exports `LD_LIBRARY_PATH`.

### The two TBLIS baselines, and a trap in each

| | |
|---|---|
| TBLIS (A) | **v1.3.0** (tag `c4f81e0`, 2 Jul 2025) — the latest *stable* release, re-checked 2026-08-02: still the newest tag |
| TBLIS (B) | `develop` @ `555320c` (4 Dec 2025), version string **2.0** — an *unreleased* development snapshot, 8 months past the newest tag (`v2.0-beta2`) |

They behave completely differently on complex data and the difference is this
project's single most important measured result. **Any statement about TBLIS
performance must name the version** (A2b).

**Trap 1 — a silent ABI break.** `type_t` swaps `TYPE_DOUBLE` and
`TYPE_SCOMPLEX` (1.3: `DOUBLE=1, SCOMPLEX=2`; 2.0: `SCOMPLEX=1, DOUBLE=2`).
Mixing them produces plausible wrong numbers with no error. The harness has a
`tblis13` cargo feature and a startup self-check (`tblis::verify_type_tags`) that
multiplies a known matrix in each dtype and aborts on mismatch (D12).

**Trap 2 — TBLIS 2.0's install is ISA-specific.** It was configured with
`BLIS_CONFIG_FAMILY=auto`, which detects the *build* machine, so its BLIS is
**skx-only** and SIGILLs on any machine without AVX-512 — it killed a `rome` job
two seconds in. `../baselines/tblis-2.0-x86_64-install` is the same source with
`BLIS_CONFIG_FAMILY=x86_64`, multi-configuration with runtime dispatch, and is
what a non-AVX-512 node needs. **Keep both**: the `auto` build is what every
committed number used, so prefer it wherever it runs.
`scripts/rusty-compare.sbatch` chooses by grepping `avx512f` out of
`/proc/cpuinfo` and records the choice in each run's `PROVENANCE.txt`. TBLIS
1.3.0 needs none of this care — it is genuinely multi-config. Verify a claim like
this with `nm`, not `strings`: the config *name* table is compiled in whether or
not the kernels are (A45).

Both baselines live outside the repo at `../baselines/tblis-{1.3.0,2.0}-install`
(built 2026-08-02, same commits as Phase 1). If they are gone, rebuild them — it
is ~30 min unattended, and `scripts/env.sh` documents the recipe. TBLIS 1.3.0
uses autotools; build the harness against it with `--features tblis13,blas`.

Single-core measurements pin everything single-threaded
(`TBLIS_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `TENSORCONTRACT_THREADS=1`);
`scripts/env.sh` does it.

---

## Measurement rules

**There is no project-wide noise floor.** A floor is a property of a session, a
machine *and a thread count*, and quoting one outside its own is how three
published numbers here turned out to be artefacts. Every floor below is derived
in the session it labels.

| session | machine | width | floor |
|---|---|---|---|
| Phase 4 A/Bs, exclusive | `ccqlin038`, Cascade Lake | 1 thread | **±1.3% on a 49-case geomean, ±6% per case.** Anything smaller is not a result |
| `compare-bench.sh`, exclusive SMT-off | `worker6156`, Ice Lake-SP | 1 thread | **0.995–1.003** at minutes' separation, **0.998–1.001** at 2.5 h, **0 of 980** case points outside ±6%. The tightest in this file, ~5x `ccqlin038` |
| `rusty-phase4.sbatch`, exclusive | Zen2 `rome` | 1 thread | **0.02% at 69 s, 1.1–1.9% at ~1 h, 4.4% across the opening cold start** (A31, A32) |
| partition arms, exclusive | `worker5479`, Zen2 | **64 threads** | per-case **p10 0.885 / p90 1.107, tails 0.80–1.55**, on repeats of an *identical* partition. Only per-family geomeans (1–2%) are quotable (A39) |
| partition arms, exclusive | `worker6150`, Ice Lake-SP | **32 threads** | per-case **p10 0.915 / p90 1.108** over 392 identical-partition repeats. So A39 is a property of threaded measurement here, not of one machine |

Nine rules, each of which cost something to learn. The full accounts are in
[`REFUTED.md`](REFUTED.md) under *Measurement methodology*.

1. **Name the thread count and the session with every floor.** A per-case ratio
   at 64 threads is not readable at all (A39).
2. **Derive the floor in-session, from repeats adjacent to the arms being
   compared** (A32). Identical-configuration arms already in a grid are free
   repeats — find them before booking machine time.
3. **Run and discard a warm-up arm** (A31) — and do not treat it as sufficient
   (A40). Every current script does; `phase3-bench.sh` predates it.
4. **Include columns the change cannot touch, and check they do not move.** This
   drift-corrected the domain gate's headline from 1.449 to 1.433 and caught a
   contaminated A/B before that.
5. **Measure at the size you publish at** (A46). Problem size is a confound
   separate from time, and none of the timing rules catches it.
6. **Prefer a runtime switch to a rebuild** (A15). A build-to-build A/B produced
   a wrong *sign* on a real effect. Every fast path gets an environment variable.
7. **Sweep the grid once and score rules offline** — as many candidates as you
   like, for free. `partition-score-rule.py` predicted the domain gate to 1%
   before a node was booked; the row-block and orientation rules were both settled
   this way and their grids are still committed.
8. **Finish with an end-to-end A/B in the configuration that ships** (A20). A rule
   validated with the other levers pinned is validated only there.
9. **Do not place traffic-changing arms concurrently** (A27). One arm per L3
   domain is free on the corpus (+0.3%) and **wrong on memory-bound cases**
   (−3.2%, up to −10%).

Two operational rules with the same standing:

* **An exclusive machine means exclusive.** Pinning is not enough: the pinned
  core's SMT sibling shares L1d and L2, which is what every cache-blocking
  measurement here turns on. Two Phase 4 conclusions had to be corrected after
  re-measuring on a quiet machine.
* **The submit directory is frozen for the duration of a cluster job.** A job runs
  in the submit directory and uses its `target/`, so `cargo build` — or
  `cargo test`, which relinks the same artefacts — replaces the very binary a
  running arm invokes, and the job will not notice.

---

## Standing assumptions

One line each, with the chapter holding the full account. **The report row is
authoritative**; this is an index. Refuted entries with a reopening condition are
in [`REFUTED.md`](REFUTED.md).

| # | Assumption | Verdict | Full account |
|---|---|---|---|
| A1 | TAPP can express everything needed, including complex and conjugation. | **Confirmed** from the actual headers: `TAPP_C32`/`TAPP_C64`, `TAPP_CONJUGATE` per operand, `int64_t` labels, `intptr_t` handles. No kill condition. | Phase 1 |
| A2 | TBLIS `develop` supports TAPP in-tree (per arXiv:2601.07827). | **Refuted.** No TAPP source in `master` or `develop`. TBLIS is driven through native `tblis_tensor_mult`. | Phase 1 |
| A2b | "TBLIS" is one thing. | **Refuted.** v1.3.0 and 2.0-dev differ by ~5x on complex and swap two ABI enumerators. Name the version, always. | Phase 1 |
| A3 | The achievable GF/s ceiling is the same for real and complex on real-SIMD hardware. | **Confirmed**: `dgemm` 96 GF/s vs `zgemm` 96. This is what validates the efficiency-ratio metric. | Phase 1 |
| A4 | The TCCG corpus exercises the irregular/gather path. | **Refuted in Phase 1, and the refutation expired in Phase 3.** It does exercise it here — `reg_a < 1.0` on 42.9% of 392 at `--size 64`, 45 of them *entirely* — because the shipped `f32`/`c32` blocks do not divide 24. The fraction depends on the size and the ISA, so quote both. What it cannot produce is *aperiodic* irregularity. | threading (part 14); `REFUTED.md` |
| A5 | Shapes must be held fixed across dtypes for the real-vs-complex ratio to mean anything. | **Adopted.** Deviates from TCCG's per-precision sizing; documented in `corpus.rs`. | Phase 1 |
| A6 | This host is the reference machine and single-core is the headline. | **Adopted.** | Phase 1 |
| A7 | The three complex methods differ mainly in flop count. | **Refuted in Phase 3.** They differ mainly in **bytes moved per useful flop**. | Phase 3 |
| A8 | One register block per method is enough. | **Refuted in Phase 3.** The best shape depends on method, element type and `kc` — and later on microarchitecture (A34). | Phase 3 |
| A9 | Absolute performance figures are meaningful. | **Adopted from Phase 3 onward**, explicitly not before it. | Phase 3 |
| A10 | The Phase 3 write-back defect is the write-back's own L2 traffic. | **Refuted in Phase 4.** It was the row/column **orientation** of the matrix view. The vectorised inner loop is real but second-order, and only in single precision. | shape rules (part 1) |
| A11 | The row/column orientation is a property of the plan. | **Refuted.** It depends on `MR`, hence on element type and complex method. | shape rules (part 1) |
| A12 | Write-back overhead matters equally in both precisions. | **Refuted.** ~5x more of the budget in `f32`/`c32` (7–15%) than in `f64`/`c64` (2–3%). | shape rules (part 1) |
| A13 | `MC` is bounded only by keeping the packed `A` block in L2. | **Refuted.** It also bounds the `D` strip a `jr` pass revisits, which binds whenever the output's rows are strided. Useful as a bound, worthless as a rule. | shape rules; cache blocking; `REFUTED.md` |
| A14 | The orientation rule's `run >= MR` condition is the right discriminant. | **Refuted, and was unresolved** — right 63/72, missing 9 cases by 1.18–1.47x. Resolved by A19. | shape rules (parts 1, 6) |
| A15 | A sequential build-to-build A/B is good enough for a few-percent effect. | **Refuted.** Reported `c64` 3m at 0.973 where a paired runtime A/B gives 1.020 — a wrong sign. | `REFUTED.md` |
| A16 | Maximising the fraction of output row blocks off the gather path is the row-block rule. | **Refuted as a rule, confirmed as a mechanism.** Unguarded it scores 0.936 in `f32`. It is a gate, never an objective — and it has pointed the wrong way three times. | shape rules (part 5); `REFUTED.md` |
| A17 | Shrinking `MR` will fix the orientation rule's nine misses. | **Refuted as a fix, and it prices the problem.** The flip happens and those cases gain 1.17–1.39x *while paying ~30% in kernel shape*: the orientation is worth ~2x and `MR` is the wrong instrument. | shape rules (part 5) |
| A18 | The register blocks measured at the operating `kc` are right at every depth. | **Refuted.** At `k <= 24`, `c64` 3m prefers `MR` 16–24 over its default 8 (1.088) and `c32` 1m prefers `24x8` over `32x6` (1.099). The table is depth-conditional. | shape rules (part 5) |
| A19 | The orientation discriminant is a property of `D`'s column direction. | **Refuted.** The corpus families are mirror images, so a correct rule must be **antisymmetric** under exchanging the two directions. Written symmetrically it is 12 better / 0 worse over all 392. | shape rules (part 6) |
| A20 | A rule validated with the other levers pinned is validated. | **Refuted.** 12/0 with the shape pinned, and still 20% worse on three cases in the shipped configuration. | `REFUTED.md` |
| A21 | Some compute-bound contractions will need `K`-parallelism. | **Refuted for this corpus, and argued structurally** — needing `K` bounds arithmetic intensity away from compute-bound. Confirmed to 128 threads in both ISAs, and corroborated by TBLIS never parallelising `PC`. | threading (parts 8, 10, 15); `REFUTED.md` |
| A22 | The blocking model may assume one thread per physical core. | **Assumed, and the project's own rules justify it.** Oversubscribing SMT siblings would halve a thread's L1/L2 and the measurement rules already treat that as invalid. | cache blocking (part 9) |
| A23 | Under threading, every cache budget must be divided by the thread count. | **Refuted; it is asymmetric.** The packed `B` panel is *shared*, so `nc`'s L3 budget must not be divided; what shrinks `nc` is the per-thread packed `A` blocks. | cache blocking (part 9) |
| A24 | The AVX-512 method ranking carries over to AVX2. | **Refuted at the kernel level, in the direction predicted.** 3m is first in `f32`/`c32` on AVX2 and planar last or next-to-last in every column. No end-to-end confirmation. | kernels (part 11) |
| A25 | Smaller register blocks are purely a cost. | **Refuted, at zero CPU cost.** Worse for the kernel, better for the write-back: 81 of 392 case-dtype-methods have no AVX-512 menu shape that clears the gather path, against none on AVX2. | kernels (AVX2 interlude) |
| A26 | The TAPP layer is thin enough that the engine's own tests cover it. | **Refuted.** Four gaps on first contact, one of which aborted the caller's process. **Test an FFI layer as its caller, not as its callee.** | packaging (part 1) |
| A27 | Concurrent arms, one per L3 domain, measure what a solo arm measures. | **Confirmed on the corpus (+0.3%), refuted on the memory-bound half (−3.2%, up to −10%).** Pre-registered with its accept/reject rule. | session (part 10); `REFUTED.md` |
| A28 | The 2-D partition rule and `PACK_WEIGHT` behave at node scale as at 8 threads. | **Refuted, but not where predicted.** `PACK_WEIGHT` is fine; the `panels >= p` early return preceding it is what fails, by up to 4.3x. | threading (part 8b) |
| A29 | Parallel width is a property of the contraction. | **Refuted at no CPU cost — of the contraction *and the ISA*.** AVX2's smaller blocks give the same corpus 3–6x more row panels. | session (part 10) |
| A30 | The register-block sweep's `live <= 16` budget is the real one. | **Refuted, by one register.** Every `live == 16` shape collapses to 21–33 GF/s beside a 48–53 GF/s sibling at `live <= 15`. All eight shipped defaults sit at 14–15, so nothing shipped is affected. | kernels (part 11) |
| A31 | `A, B, A'` bracketing is enough to establish a session's noise floor. | **Refuted on a machine with boost headroom.** Reported ±4.7% where the true repeat precision was 0.02%. **Run and discard a warm-up arm.** | `REFUTED.md` |
| A32 | A session's drift is a single number. | **Refuted; it is a function of how far apart the arms are** — 0.02% at a minute, 1–2% at an hour, 4.4% across a cold start. Refined: on an exclusive SMT-off node it does not grow with separation at all. | `REFUTED.md` |
| A33 | An analytically derived blocking is a safe default on an unseen machine, so it solves portability. | **Refuted on the first unseen machine** — worse in 11 of 12 columns, by up to 7.2%. Its `kc` sinks it there and its `mc` on the reference machine. | cache blocking (part 9); `REFUTED.md` |
| A34 | Register blocks are a property of the instruction set. | **Refuted.** Cascade Lake and Ice Lake disagree by up to 13% on three of eight shapes. Shapes are **per-microarchitecture**; probed L1 geometry separates these two. | kernels (part 11); `REFUTED.md` |
| A35 | The shipped register blocks are the ones the Phase 3 sweep selected. | **Refuted for one of eight.** `planar` `f32`/`c32` ships `32x6` where the sweep names `32x5`, 7.8% faster. Reachable at `TENSORCONTRACT_ROWBLOCK=idx=3` since D43; unfixed on purpose. | kernels (part 11); shape rules (part 13); `REFUTED.md` |
| A36 | The 4.3x a 1-D `N` partition wins on the memory-bound family is a property of the partition rule. | **Refuted; it is a property of the topology** — how many L3 domains the shared packed-`B` panel spans. So gate the early return, do not remove it. | threading (parts 8, 12); `REFUTED.md` |
| A37 | `l3_domains(p)` may assume **compact placement**. | **Assumed, and true of every measurement here.** The error is in the safe direction — it under-counts, so the rule falls back to legacy behaviour. `TENSORCONTRACT_L3_DOMAINS` overrides it. | threading (part 12) |
| A38 | A cross-domain traffic term in the partition cost model can be calibrated. | **Refuted, four ways.** The separating weights are 1% apart and want opposite answers. Hence a binary gate over the two arms that were measured. | threading (part 12); `REFUTED.md` |
| A39 | Per-case ratios at 64 threads are readable at ±6%. | **Refuted.** p10 0.885 / p90 1.107 with tails to 1.55 on identical-partition repeats. Only per-family geomeans are quotable. Confirmed on a second machine and width. | threading (part 12); `REFUTED.md` |
| A40 | A discarded warm-up arm removes the opening-arm artefact. | **Partly.** Three of four brackets inside 0.3%; the exception tracks **position in the session**, so a hotter warm-up would not fix it. Keep it; do not treat it as a floor. | threading (part 12); `REFUTED.md` |
| A41 | Blocks of a block-scatter contraction are equal-cost. | **False in the premise, true in the consequence.** Blocks differ in cost — 123 of 392 case-dtype-methods are internally mixed — but making it 7.5x more prevalent and aperiodic moves parallel efficiency by less than the floor. | threading (parts 14, 16); `REFUTED.md` |
| A42 | Block-scatter load imbalance is solved in mature implementations, so a cost-aware partition would be reinvention. | **False.** TBLIS computes the same regularity sentinel and feeds it to no scheduling decision, in either version. | threading (part 15) |
| A43 | Per-call thread spawn is a second-order cost. | **Refuted at small sizes.** ~20–36 µs per thread; a 0.22 ms contraction takes 2.2 ms on 64 threads. First-order for exactly the workload Phase 1 named as the headroom. | threading (part 16); `REFUTED.md` |
| A44 | The complex-method ranking, the memory-bound inversion, and 3m's L1-resident advantage are properties of the engine and the shape. | **Refuted, and not confounded after all.** 3m already runs its Ice Lake-optimal shape, so A34 cannot explain it; the collapse is uniform across 3m's shape space, and even at L1-resident depth 3m leads on Cascade Lake and trails badly on Ice Lake. **Item 3 is dead.** | comparison (part 3); `REFUTED.md` |
| A45 | A baseline built from the right source at the right commit is the right baseline. | **Refuted, on portability.** `BLIS_CONFIG_FAMILY=auto` fits BLIS to the build host: skx-only, SIGILLs without AVX-512. Record a baseline's **configuration**. | comparison (part 3); `REFUTED.md` |
| A46 | This file's measurement rules (A27, A31, A32) cover the ways a comparison can mislead. | **Refuted: they are all about time, and problem size is a separate axis.** A 1.68x claim measured at 8 MiB vanishes at the sizes actually published, and is withdrawn. | comparison (part 3); `REFUTED.md` |
| A47 | The shipped header agrees with the library it describes. *(was the second `A31`)* | **Now tested rather than assumed.** `examples/c-consumer` compiles the header with a C compiler, links the built library and checks numbers; CI runs three link modes. | packaging (C-surface interlude) |
| A48 | A Rust panic reaching the C boundary is acceptable because it is memory-safe. *(was the second `A32`)* | **Rejected as a policy.** Memory-safe but process-fatal, and the engine panics on allocation conditions a caller can hit. D38 converts it to an error code. | packaging (C-surface interlude) |
| A50 | A guard fitted to the sub-megabyte regime will misjudge the saturation above it. | **Confirmed, and the guard is scoped accordingly** — larger constants score better at 16 MiB precisely because capping threads helps against a bandwidth ceiling, which is a second mechanism. The constant is chosen on small-size evidence alone. | threading (part 17) |
| A51 | A pool can degrade gracefully when it has fewer workers than the requested width. | **False for this driver, and it would deadlock.** Indices `t` and `t + pn` share a `pm`-way barrier. `try_broadcast` declines all-or-nothing and the caller spawns instead. | threading (part 17) |
| A52 | Mutex poisoning is a detail in a pool whose state has no invariants a panic can break. | **False in effect, and a test found it.** A worker's panic poisons the submission mutex, after which the pool declines *permanently*. Recover from poison; decline only on `WouldBlock`. | threading (part 17) |
| A53 | The cost a thread pool removes is thread creation. | **Refuted.** Size-dependent — 37 µs/thread at 0.25 MiB, 66–94 µs at 16 MiB — where thread creation is not. Named candidate: the per-thread packed-`A` allocation and its page faults, which suggests a cheaper unbuilt fix. | threading (part 18) |
| A54 | A rule that consumes a caller's parameter can be validated at one value of it. | **Refuted; A20 in a third form.** D48's guard scores 0 points slower at `requested = 64` and loses 39% at `requested = 4`. Score across the caller's parameter. | threading (part 18) |
| A55 | A thread pool changes thread lifetime, not memory locality. | **Refuted.** Reusing a thread reuses its allocator arena, so every worker gets the same packed-`A` buffer address back each call. A pool is an allocation-locality change, which is why its benefit turns on cache topology. | threading (part 19) |
| A56 | The pool's benefit is topology-independent, so one machine class suffices to recommend it. | **Refuted — 11.6x on Zen2, 0.40–0.80 on Ice Lake.** The fourth threading or kernel choice here that fails to transfer, after A34, A36 and A44. | threading (part 19) |
| A57 | The L2 is private to a core, so the analytical model may budget all of it to one thread. | **False on Apple Silicon, and it had been invisible because every x86 machine measured makes it true.** `model_mc` reads `l2.ways` and `l2.bytes_per_way()` and never divides by `cores_sharing(l2)`; on Cascade Lake, Zen2 and Ice Lake the L2 is per-core (`shared_by` ≤ 2, SMT siblings) so the omission cannot be seen. Six M3 Max P-cores share one 16 MiB L2, so the model hands each the whole thing — a 6x over-allocation, and it derives a **12 MB packed `A` block** for `f64`. A **third, structural** reason A33 stands, not a reopening. | Apple Silicon (part 20) |
| A58 | "The portable scalar path is not competitive" is a statement about vectorisation. | **False, and it understated the path by an order of magnitude.** LLVM vectorises `real_ukr` to NEON unasked — eight `float64x2_t` accumulators, `ld1r.2d` broadcasts, even the lane-indexed `fmul.2d v, v, v[0]`. What it will not do is contract `acc += a * b` into `fmla`, because that is two roundings and IEEE forbids it. So the path runs at two instructions per MAC where the machine offers one, **halving the ceiling available to it to 32.4 GF/s**, and on its best shape it reaches **80.5% of that** — against OpenBLAS's 87.1% of the full 64.8. The gap is **one instruction form, not a missing kernel**: **0.65x of OpenBLAS-TTGT** in `f64` over 12 shapes at 64 MiB, not 0.05x. A NEON kernel is worth ~2x, not 10x. **Confirmed by removing it** (part 22): the NEON kernel buys **1.840** end to end and reaches **81.4% of the full 64.8 GF/s ceiling**, against the scalar path's 80.5% of the halved one — same efficiency, twice the ceiling. | Apple Silicon (parts 20, 22) |
| A59 | The complex-method ranking is decided by bytes moved per useful flop, everywhere. | **Incomplete, not refuted — it depends on which resource binds.** On the portable path 3m leads planar by **1.229** (`c64`, premise, 64 MiB), a third ordering after Cascade Lake's and Ice Lake's. Mechanism: with no FMA the kernel is *instruction-throughput*-bound, so 3m's 25% saving in products is a 25% saving in the binding resource, and the bytes-per-flop accounting that decides it on x86 is not what binds. **Re-measured in Stage B and the guess was wrong in direction of magnitude, not sign** (part 22): with a real FMA kernel 3m's lead **narrows from 1.226 to 1.135 and does not invert**. It gains least from NEON (1.690 against planar's 1.826), which is the mechanism behaving as stated, and still finishes 13.5% ahead — 13x the session floor. So A59 stands as *incomplete*: which resource binds does decide the ranking, and on this microarchitecture 3m wins under both bindings. A **fourth ordering**, and the only one taken with a tuned kernel on its own machine. | Apple Silicon (parts 20, 22) |
| A60 | The register budget that correctly rejects bad shapes can also choose the good one. | **False, and the two halves are separate skills.** The budget filters well — the `!` over-budget flag tracks collapse closely — and then names the wrong winner in **three of eight** columns on an M3 Max. The sharp case: it proposed `f64` real `6x8`, which is BLIS's own `armv8a_asm_6x8`, and that agreement was recorded as corroboration; the machine prefers `16x3` by 4.9% (2.6x the floor), a shape the budget calls **over** its limit at `live = 33`. A34 on a fourth ISA. A library's chosen shape agreeing with a model is not evidence about this engine's kernels. | Apple Silicon (part 21) |
| A61 | A rule fitted on one microarchitecture's register blocks will not hold for another ISA's. | **False here, and worth naming because so little transfers.** The guarded row-block rule was derived on Cascade Lake, on AVX-512, for `MR` in {16, 32, 48}. On NEON `f64` the measured shape `16x3` costs write-back regularity on 12 of 49 cases (`wb` 1.00 → 0.67, the whole `abcijk` family, since 16 does not divide TCCG's 24) and **the unmodified rule demotes exactly those 12** — all have `k = 24` against its `k <= 32` guard. 0 of 392 left with no regular shape. Against A56's four non-transferring choices, this one transfers. | Apple Silicon (part 21) |
| A62 | A `best per method` line from a register-block sweep is a result. | **False without a floor: six of eight were ties.** One arm prints eight confident winners; three arms on the same quiet machine put six of them inside the session's own p90 spread (1.89% `f64`, 3.64% `f32`), and expose three shapes that are **unstable rather than noisy** — 1m `3x10` reads 37.8 / 54.9 / 54.8, a 45% swing, which is a shape doing two different things and not a bad measurement of one. A register block may not be changed on one arm; `scripts/kernel-shapes-compare.py` is the gate. | Apple Silicon (part 21) |

---

## Build-vs-reuse decisions

| Layer | Decision | Why |
|---|---|---|
| N-d array crate | **Build** (`Layout`, 20 lines) | The engine needs pointer + extents + strides. A dependency would leak into the public API and the TAPP C ABI for no gain. |
| `num-complex` | **Reuse** | `#[repr(C)]`, layout-identical to `TAPP_C32/C64` and C99 `_Complex`; ecosystem standard. Layout pinned by a test. |
| SIMD abstraction (`pulp`, `macerator`) | **Build** on `core::arch` | `portable_simd` is unstable on 1.97. Register-blocked kernels want explicit register control. The `Ukr` function pointer keeps a `pulp` backend addable later without touching the driver. |
| GEMM kernels (`gemm`, `matrixmultiply`, `microgemm`, `faer`) | **Build** | All expose *matrix* multiply, not a panel-panel kernel over externally packed buffers, and none has a planar-complex path. `matrixmultiply` (now with AVX-512 `cgemm`/`zgemm`) is a benchmark target, not a foundation. |
| `rayon`, for the intra-contraction path | **Build** on `std::thread::scope` | The parallelism is SPMD-with-barriers — `pn` barriers of width `pm`, two rendezvous per `(jc, pc)` — and a rayon task that blocks on a barrier inside a bounded pool deadlocks. Work-stealing also fights static partitioning with a shared packed-`B` panel. `ThreadPool::broadcast` fits the shape but ties the parallel degree to the pool size, and a global pool inside a library fights the host runtime (Julia, C consumers). See `REFUTED.md`. **rayon does fit the batched API**, where the axis is the batch and there are no barriers. |
| `tblis`/`tblis-ffi` crates | **Build** ~100 lines of FFI | The benchmark must control which TBLIS is measured — crucially which BLIS config its kernels were built for. Those crates vendor their own build. Load-bearing: the headline result *is* a version difference, which a crate vendoring one fixed build could not have surfaced. Struct layout pinned by a `sizeof`/`offsetof` test, and the 1.3-vs-2.0 `type_t` swap by a runtime self-check. |
| `criterion` | **Build** a small harness | Criterion targets many fast iterations of a cache-resident routine. These are 0.1–5 s measurements on 64–200 MiB working sets; best-of-N after warm-up plus a CSV is the right tool. |
| `opt-einsum-path` | **Out of scope** | This engine executes one binary contraction; ordering is a caller concern. |

---

## Design decisions

D1–D56, complete and in one place. A report names the numbers it introduced and
does not restate them.

| # | Decision | Rationale |
|---|---|---|
| D1 | Reductions (TAPP case 4) become contraction indices with stride 0 in the operand that lacks them. | Needs no workspace, unlike a pre-reduction pass. TAPP itself warns case 4 may need unbounded workspace. Works because block-scatter treats stride 0 as regular. |
| D2 | Irregular block-scatter entries are flagged with `i64::MIN`, not `0`. | Follows from D1: a zero block stride is a *legal, regular* pattern here, so it cannot double as the sentinel. |
| D3 | Diagonals (repeated labels) are handled by summing strides, per tensor, before classification. | Keeps the rest of the engine free of repeated labels. |
| D4 | Broadcast output indices (TAPP case 5) are rejected. | TAPP does not require support; supporting it needs either workspace or redundant compute. |
| D5 | Class ordering: `M`/`N`/`H` by `|stride_D|`, `K` by `|stride_A|`. | The output update is the one access packing cannot hide, so `D` gets first claim on contiguity. Heuristic; a Phase 4 knob. |
| D6 | Scatter vectors are materialised in full at plan time; block-scatter vectors at execute time. | Scatter is element-type independent (so one plan serves all dtypes, which the TAPP layer needs); block-scatter depends on `MR`/`NR` and is `O(M/MR)`, i.e. free. |
| D7 | Micro-kernel writes to a stack tile; `alpha`/`beta`/scatter write-back/re-interleave happen outside. | One kernel serves the regular path, the gather path and every edge block. Cost is one extra store/load of an L1-resident tile. |
| D8 | Conjugation is folded into packing (negate the imaginary plane). | Free, and it is what makes `TAPP_CONJUGATE` cost nothing. |
| D9 | On accumulate passes, `op_C` is set equal to `op_D`. | Conjugation is additive and involutive, so `conj(alpha*AB_p + conj(stored))` accumulates correctly across `K` blocks. Verified by an all-16-masks test forced through multiple `K` blocks. |
| D10 | Blocking is overridable per plan and via `TENSORCONTRACT_MC/KC/NC`. | Lets the test suite drive every level of the five-loop nest on oracle-sized tensors, and lets Phase 4 sweep parameters. |
| D11 | Corpus is TCCG's **full 49-case** set, not 48. | See `DESIGN.md` §5.2: the brief's "48" does not correspond to any list in upstream `benchmark.py`. 49 is the full set and a superset of the 25-case reduced set; `_sortedTCs` is a re-labelling, not a sixth group. |
| D12 | Both TBLIS v1.3.0 and 2.0-dev are benchmarked, behind a `tblis13` cargo feature, with a runtime ABI self-check. | The two releases swap `TYPE_DOUBLE`/`TYPE_SCOMPLEX`, so a mismatch is silent rather than fatal. Given the result hinges on the version difference, guessing was not acceptable. |
| D13 | `Blocking::derive` takes reals-per-element for A and B rather than `size_of::<Element>()`. | 1m's packed A carries four reals per complex element instead of two, so an element-size-based rule would hand it double the L2 footprint and quietly rig the comparison. Deriving from the actual packed footprint gives every method the same L2 budget and gives 1m a proportionally smaller `MC`. |
| D14 | B's "1r" packing under 1m is bit-identical to planar packing, and shares the code path. | Not a shortcut: `[re_0..re_{NR-1}, im_0..im_{NR-1}]` per logical k-step *is* both formats. Worth stating because it means 1m's cost over planar is entirely on the A side. |
| D15 | 3m gets a 100x looser test tolerance than the other two. | Its error bound is relative to `\|Ar\|\|Br\| + \|Ai\|\|Bi\|` rather than the complex magnitudes, so it loses relative accuracy under cancellation. That is the documented price of the 25% flop saving; the tolerance records it rather than hiding it. |
| D16 | Kernels take the *logical* `kc` and know their own panel layout. | 1m internally runs `2*kc` real steps. Exposing that to the driver would leak the method into the loop nest. |
| D17 | Micro-kernels are macro-generated over `(MV, NR)` const generics from one body per method, not hand-written per shape. | A comparison between three methods must not also be a comparison between three hand-tunings. One body per method, one shape parameterisation, and the shape is then chosen by measurement. It also made the shape sweep possible at all. |
| D18 | `#[target_feature]` kernels are reached through one-line plain-`fn` trampolines. | A `#[target_feature]` function cannot be coerced to a function pointer, which the `Ukr` contract requires. Cost is one `call` per micro-tile against `kc*MR*NR` FMAs — unmeasurable. |
| D19 | Register blocks were chosen by measured throughput at the `kc` the engine actually uses, per method and per element type. | The uop model gets the *cliffs* right (spills above 32 live vector registers) but the *ranking* wrong: it predicts 3m fastest, and 3m is fastest only when the panels are L1-resident. See the Phase 3 report. |
| D20 | Workspace MSRV raised `1.75` → `1.89`. | AVX-512 intrinsics and `is_x86_feature_detected!("avx512f")` were stabilised in Rust 1.89. The alternative — feature-gating the AVX-512 path so 1.75 still builds — would make the project's headline measurement an opt-in extra. 1.89 is a year old. |
| D21 | Threading parallelises the `M` direction only, into contiguous strips of whole `MR` panels, with a per-thread packed `A` and a **shared** packed `B`. | The `pc` loop accumulates into `D` in place, so parallelising it would need a per-thread temporary or atomics; `M` instead gives every output element one owning thread. That makes the result **bitwise identical to serial at every thread count** — a stronger invariant than agreeing with the oracle, and one a test can assert directly. Strips of whole panels keep each thread's row blocks aligned with the block scatter, so the write-back fast path and the orientation rule are unaffected. `B` is shared because `NC` is sized for L3, which is a per-socket resource. |
| D22 | The default thread count stays **1** until scaling is measured on the reference machine. | Every performance number in this file is a single-core measurement, and the item 2 blocking grid is designed against the serial engine. A default that changed with the machine's core count would make committed numbers irreproducible from a bare checkout. `TENSORCONTRACT_THREADS` and `Plan::with_threads` opt in; flipping the default is one line in `Plan::threads`. |
| D23 | The cache blocking is computed from **probed cache descriptors** via BLIS's analytical model, not from constants — and ships **off by default** behind `TENSORCONTRACT_BLOCKMODEL=legacy\|model`. | `Blocking::derive` held the only three machine-specific numbers in the engine: `kc = 384/256` by real size, a 512 KiB packed-`A` budget and a 3 MiB packed-`B` budget — half of `ccqlin038`'s L2 and a slice of its L3. Nothing about them transfers. `kernel::cache` probes sysfs (no `unsafe`, no dependency, and the only source that reports cache *sharing*), then x86 `CPUID` leaf 4 / `0x8000001D`, then conservative built-ins; a failed probe degrades to the next source and can never fail a contraction. The model is driven by the **reals per element the kernel packs**, not `size_of::<Element>()`, so 1m still gets half the rows planar does out of one L2 — the invariant the three-way comparison rests on. Off by default because the pending grid defines its arms *relative to the derived defaults*, so changing the derivation would silently change what that measurement means, and because A15 requires the old behaviour stay reachable as a run-time switch rather than a build-to-build diff. Build-vs-reuse: no crate, because `raw-cpuid`/`num_cpus`-style crates do not report per-level sharing and sysfs is ~100 lines of parsing. |
| D24 | The instruction set is a **parameter of the kernel macro**, not a second set of bodies. `simd_kernels!` takes the lane count, a target-feature list and six intrinsics; AVX-512 and AVX2 are two instantiations of the same four bodies. | D17's argument — a three-method comparison must not also be a comparison between three hand-tunings — applies unchanged to two instruction sets, and would be violated the moment somebody hand-tuned AVX2 planar and not AVX2 3m. It also made the change auditable: because the AVX-512 arm expands from unchanged source text, its instruction stream is *verifiably* untouched (7816 zmm instructions, byte-identical, checked independently at merge). |
| D25 | Dispatch is `avx512f` → `avx2 + fma` → scalar, and `TENSORCONTRACT_KERNEL` grows to `scalar\|avx2\|avx512\|auto`. A pinned ISA the CPU lacks falls back to scalar rather than faulting. | Without the pin, AVX2 kernels could be compiled on the reference machine and never executed on it, leaving the only coverage on hardware nobody here has. Same rule as `_ORIENT`/`_WRITEBACK`/`_ROWBLOCK`: every new fast path gets a run-time switch, because a build-to-build diff has already produced one wrong sign here (A15). Both `avx2` and `fma` are detected — separate CPUID bits, and the FMAs need the second. |
| D26 | AVX2 register blocks are **provisional and explicitly unmeasured**, chosen from the register budget and the uop model, with a menu of alternates and `examples/kernel_shapes` extended to an AVX2 grid as the calibration path. | D19 splits this cleanly and the split was honoured: `live <= 16` and `acc >= 10` are load-bearing and were *verified in the disassembly at no CPU cost* (all shipped defaults allocate 15–16 distinct ymm with zero stack traffic; `planar 1x6` at `live=16` spills one register, `planar 2x3` at `live=18` spills seven — the cliff is exactly where the count puts it), while "which of the shapes that fit is fastest" is recorded as a guess. Publishing a modelled shape as measured would corrupt the one thing the register-block table is good for. |
| D27 | Threading partitions the output in **two dimensions**: `pm` row strips of whole `MR` panels by `pn` column groups of whole `NR` blocks, with `pn > 1` only when `ceil(M/MR) < p`. The `N` cut is made *inside* loop 5, per `NC` block, not over the whole range. | D21's 1-D cap costs real throughput on the 16 case-dtype-methods whose row axis cannot fill 8 threads (part 8). Both axes partition the *output*, so D21's invariant survives intact — one owning thread per element, accumulating over the full `K` in the original order, hence bitwise identical to serial at every thread count *and* every `(pm, pn)`. Cutting `N` inside loop 5 is what keeps the packed `B` panel single and L3-sized (a top-level split would want `pn` panels and `pn` times the L3 budget) and keeps loops 5 and 4 identical across threads, which is what makes the barrier counts agree structurally rather than by bookkeeping. Barriers become per column group and `pm`-way; a pure `N` split synchronises nowhere at all. |
| D28 | The packed `A` block stays per thread, duplicated `pn` times, rather than packed once per row strip behind a barrier. | The duplication costs one packed element per `NR * ceil(blocks/pn)` lane-FMAs and is only ever paid when `M` is narrow; `Plan::partition` prices it and refuses to split `N` when a thread would have too few `NR` blocks to amortise it. The alternative needs a barrier *inside* loop 3 — `M/MC` times more often than the ones above it — and leaves the block in one thread's L2 for the others to pull across L3, when `MC` exists precisely to make it an L2 resident. |
| D29 | In the narrow-`M` regime `(pm, pn)` minimises `ceil(panels/pm) * (NR*ceil(blocks/pn) + PACK_WEIGHT)` with `PACK_WEIGHT = 8`, and `TENSORCONTRACT_PARTITION=m\|n\|<pm>x<pn>` pins the partition. | Maximising thread count alone either oversubscribes (3 panels, 8 threads → `3x3` = 9) or wastes threads; balancing tiles alone ignores the `A` duplication and takes an 8th thread that costs more than it returns. The weight is the rule's only modelled quantity, so it is one named constant and its insensitivity was checked offline against the committed feature table: **every weight in `[4, 64]` gives an identical partition on all 392 case-dtype-methods at 2/4/8/16/32 threads**. The env switch makes the axis choice a run-time A/B rather than a build diff (A15). |
| D30 | The published surface is **three tiers**, stated in the crate docs, not one flat `pub`. Tier 1 is the contraction API (`contract`, `Plan`, `Layout`, `TensorView`/`TensorViewMut`, `Element`/`Real`, `Error`, the `with_*` builders, `kernel::scalar` as the documented extension point) under ordinary semver. Tier 2 is introspection of the engine's own decisions — `PlanStats`, `plan::Scatters`, `Plan::{transposes_gemm, row_block, partition, oriented_scatters, ..}`, `kernel::{selected_config, plan_config, selected_kernel_name}`, `kernel::cache`, the `scatter` builders — where **signatures are semver-stable and values are not**. Tier 3 is `#[doc(hidden)]`. | 158 undocumented public items across seven `pub` modules was the symptom; the disease was that nothing had decided which of them a user may depend on, and publishing settles that whether or not we answer it. Tier 2 is the load-bearing part and exists because of how this project works: every one of those functions returns the output of a measured heuristic, and Phase 4 has already moved three of them (A14 moved `transposes_gemm`, A16–A18 moved `row_block`, item 2 will move `Blocking`). Without saying outright that a tier-2 *value* is not part of the contract, the project must either freeze its own tuning or break semver every phase. The corollary is why they stay public at all: a harness or an alternative execution strategy can ask the engine what it would do instead of guessing, which is exactly what `tcbench orient` and `tcbench shapes` rely on. Only `kernel::x86` (register-block menus *are* per-machine measurements) and `scatter::BlockScatterMatrix` (not correctly constructible from outside) went to tier 3. |
| D31 | MSRV stays **1.89** (D20), and CI pins exactly it. | The floor briefly became 1.94 by accident: the `CPUID` cache probe called `__cpuid_count` outside `unsafe` on the strength of a comment claiming it had been safe since 1.87. It had not — 1.89 through 1.93 fail with `E0133` and 1.94 is the first that compiles. Since that regression arrived with the probe rather than with any requirement, the fix is the `unsafe` block (plus `allow(unused_unsafe)` for toolchains where it is redundant), not five releases of downstream compatibility. The pin had already drifted the other way — it read 1.75, *below* the declared floor, so that job could never have passed. |
| D32 | The TAPP conformance suite drives the **C symbols**, checks against the oracle *plus* hand-computed anchors, and is **falsified by perturbation** before being believed. | Testing the layer through its Rust types would exercise exactly the code that cannot be wrong. Two anchors sit alongside the oracle so a shared engine/oracle bug cannot pass, and both mechanisms were confirmed live by breaking them and checking *which* tests fail: one constant broke one test, a perturbed oracle broke 33 of 34. `abi_layout.rs` re-declares the upstream header so symbol export is a link error here rather than downstream. Where the ABI *cannot* detect an error — any non-zero bogus handle, a data buffer shorter than its extents — that is a comment rather than a test, because a test that invokes undefined behaviour is not evidence of anything. |
| D33 | The project is `tensorprimitives`, a facade over per-operation crates (`tensorcontract` now, `tensortranspose` planned), rather than one crate named after the operation it happens to implement first. | Three names were rejected on specific grounds. `tensoroperations` would claim the name of a *frontend* — index notation, contraction ordering — that `DESIGN.md` puts explicitly out of scope. `scattergemm` and `bsmtc` name the algorithm, which stops being true the moment a dedicated transpose kernel lands: a transpose is neither a GEMM nor block-scatter-matrix contraction. And a single crate named `tensorcontract` would have to grow non-contraction operations under a contraction name. "Primitives" is TAPP's own word for this layer — one operation per call, executed as asked — so the name states the scope and aligns with the standard the project implements. The shape is `futures`/`futures-core`: applications depend on the facade, libraries depend on the operation crate directly, since cargo features are additive across a dependency graph and a library enabling a facade feature imposes it on every other consumer — the point `blas-src`'s own documentation leads with. Precedent checked rather than assumed: Rust's BLAS ecosystem always names the implementation for itself (`openblas-src`, `netlib-src`, `blis-src`) and never after the interface, and TAPP has no provider-naming convention at all — its TBLIS and cuTENSOR interfaces live inside the reference repository rather than as published providers. |
| D34 | **No published facade crate** until there are at least two implemented primitives. `tensorprimitives` is the project and repository name; the published crates are one per operation plus the family-level `-tapp` ABI. | A facade re-exporting a single primitive is indirection, not abstraction: it would be `tensorcontract` plus a version to keep in lockstep, a re-export surface that has to track the engine's API, and a crate republished on every release with nothing in it changed. The argument that justified the *feature* structure also undercuts the facade — libraries are told to depend on the operation crate directly, because cargo features are additive across a dependency graph, which leaves the facade serving only applications, for whom the alternative is one extra line in a manifest. And the asymmetry decides it: adding a facade later is purely additive and breaks no existing dependent, whereas publishing one now is permanent. `futures` and `tracing` earned theirs by having several substantial members on a shared cadence. Accepted cost: crates.io will show `tensorprimitives-tapp` with no `tensorprimitives` stem, and the umbrella name stays unclaimed — cheaper than a maintained no-op, and publishing a shell to hold a name is name-reservation with extra steps. For C callers the family-level surface already exists: the TAPP crate is the umbrella that will expose transposition beside contraction. |
| D35 | **No Rust TAPP consumer-bindings crate** (a `tapp-abi`) until something links against it. This project ships a TAPP *provider* — it defines the symbols — and does not publish declarations for calling *into* other implementations. | Decisive fact: the upstream reference implementation has **no releases and no tags**, so an ABI crate would have nothing to version against and drift would be silent — the exact failure mode that TBLIS's `type_t` enum swap already cost this project once. The standard is also visibly unsettled: `attributes.h` specifies no keys, `product.h` leaves `TAPP_IN_PLACE` an open `//TODO`, `error.h` fixes only `TAPP_SUCCESS`, and the `status` out-parameter has no defined semantics — all four found by our own conformance suite. One argument previously offered for the split is **withdrawn**: it would *not* "leave something for other Rust providers", because a provider defines these symbols rather than declaring them, and two Rust providers could not coexist in one link. The real consumer is Rust code calling a C or CUDA TAPP implementation, of which there are currently none. Adding it later is purely additive, so it waits for a trigger — most plausibly the benchmark harness driving another TAPP implementation, which supplies a second consumer *and* the only real test of the declarations, since a declarations crate nothing links against is untested by construction. When that happens the name is `tapp-abi`, not `tapp-sys` (a `-sys` crate corresponds 1:1 to a native library; TAPP is a standard with several), and the seed is `abi_layout.rs`, which already maintains an independent transcription of all 23 symbols. |
| D36 | **Ship a header** — `crates/tensorprimitives-tapp/include/tapp.h` — rather than telling C callers to fetch the standard's own. Upstream's headers remain supported and equivalent; this one versions with the implementation. | D35 established that upstream has **no releases and no tags**, and then left C consumers depending on exactly that untagged `main`. A consumer pinning this crate was pinning its declarations to a moving target, with drift as silent as the TBLIS `type_t` swap. The header is not a second source of truth: every prototype was checked against upstream at `main`, `abi_layout.rs` independently pins the same 23 symbols from the Rust side, and `examples/c-consumer` now compiles *this* header and checks numerical results — so the two transcriptions are cross-checked by construction rather than by discipline. Two departures are marked in the file itself, both where upstream is underspecified: `TAPP_IN_PLACE` (an open `//TODO` upstream, so the definition is guarded and yields to any upstream one) and the non-zero `TAPP_error` codes (upstream fixes only `TAPP_SUCCESS`). |
| D37 | The C consumer is a **CMake project driven by corrosion**, run by CI in *both* the corrosion and prebuilt-library modes. | `abi_layout.rs` verified linkage from Rust, so no C compiler had ever seen the header and a wrong prototype would have reached a downstream user first. The two modes fail differently and both are real: prebuilt is what a site building the Rust half separately does and needs no network; corrosion is what a CMake project actually writes, and a break in cargo-from-CMake is invisible to the prebuilt path. Corrosion earns its place by resolving the platform link line — it derived `gcc_s;util;rt;pthread;m;dl;c` for a static link here, which is the usual first hand-written failure. The example checks *numbers*, not return codes, so a stride convention or complex-layout error fails rather than passing quietly. Linux only for now: macOS would add `.dylib` naming and install-name handling and is the obvious next job, left out rather than added blind. |
| D38 | A **panic boundary** (`guard`) on the three entry points that allocate or run the engine, mapping a caught unwind to `TAPP_ERROR_INTERNAL`. | A panic reaching an `extern "C"` frame aborts. That is memory-safe and, for a library called from a solver or an MPI rank, still the wrong outcome: `Panel::new` asserts rather than returning when a packing buffer cannot be allocated, and a contraction too large for memory is an ordinary user mistake. The panic hook still runs first, so nothing becomes less diagnosable. Scope is deliberate rather than uniform — the getters return `void` and could only swallow silently, and `TAPP_execute_batched_product` inherits the guard through the inner call — and `AssertUnwindSafe` is justified in the doc comment rather than assumed: an entry point that panics has not yet published its handle, the one exception being a partially written `D`, which the batched path already documents. Allocation *failure* proper still aborts and is not catchable here. Tested at the mechanism (`guard` itself), not end to end, because inducing an engine panic needs a fault-injection point that would have to be maintained — recorded so the limit is not mistaken for coverage. |
| D39 | **Eleven cross targets are checked in CI, and `musl` gets `-C target-feature=-crt-static`.** `kernel::x86`'s cfg stays `any(target_arch = "x86", target_arch = "x86_64")` — 32-bit x86 is *not* narrowed out. | The repository had never been cross-compiled: CI was x86-64 Linux plus aarch64 macOS, and a binary distribution attempts about a dozen platforms. Checking them turned one expectation over and found one silent failure. The expectation was that `i686` would break, because `kernel::x86` instantiates the AVX-512 intrinsics on 32-bit x86 too and the `core::arch::x86` module genuinely lacks the intrinsics taking 64-bit integer operands; the kernels are f32/f64 FMA-shaped and use none of them, so `i686` compiles as-is and the cfg needs no change. The silent failure is `musl`: with the target's default `crt-static`, cargo prints `dropping unsupported crate type cdylib` and **exits 0**, so a musl build succeeds while producing no shared library — a JLL tarball with a header and no library in it, and nothing downstream to say why. `-C target-feature=-crt-static` fixes it, and because the symptom is a warning rather than an error the `cross-musl` job asserts the warning still appears without the flag, so the day it stops being needed is visible. `cargo check`, not `cargo build`: a linker per target is a cross toolchain CI has no business installing, and the link step is covered by the BinaryBuilder recipe, which brings its own. The nine non-musl targets — `i686`, `aarch64`, `armv7`, `ppc64le`, `riscv64`, `x86_64-windows-gnu`, both Apple targets and FreeBSD — all pass unmodified. |
| D40 | **The `cdylib`'s SONAME and Mach-O install name are set at link time via `RUSTFLAGS`, not by rewriting the binary afterwards** — and CI asserts three properties of the built library that no Rust test can see: the SONAME, the exported-symbol count, and the presence of a `cpuid` instruction. | rustc emits `-soname` only for the `dylib` crate type and **never for `cdylib`**, so the shared library this project ships has no `DT_SONAME` at all: every consumer records a bare filename and the library cannot be versioned. On Mach-O it is worse than absent — ld64 defaults `LC_ID_DYLIB` to the `-o` path, i.e. an absolute build-tree path, and BinaryBuilder's `ensure_soname` autofix *returns early when an ID is already present without inspecting its value*, so a macOS build ships unrelocatable and the audit says nothing. Link-time flags fix both with one mechanism, and avoid ELF rewriting entirely — `patchelf` needs `--page-size 65536` on every 64 KiB-page architecture (`aarch64`, `ppc64le`, `riscv64`) or it corrupts section alignment. `-C rpath` was rejected as the macOS route: it does produce `-install_name @rpath/…`, but it also bakes build-tree `LC_RPATH` entries in. The third assertion is the load-bearing one and was not obvious: BinaryBuilder's `check_isa` audit **fails** a build whose detected minimum instruction set exceeds the platform's, and this is a generic-x86-64 binary containing AVX-512 kernel bodies. It passes only because `analyze_instruction_set` abandons the analysis when the binary contains a `cpuid`, on the grounds that it may be dispatching at run time — which is precisely what `kernel::x86` does. That exemption is undocumented as a contract and depends on `kernel::cache`'s `__cpuid_count` and `std_detect`'s `__cpuid` staying linked in; if a future sysfs-only cache probe removed both, the consequence would be an unrelated-looking Yggdrasil failure months later reading "Minimum instruction set detected … is avx512". Asserted in the `c-consumer` job so the dependency is visible where it can be acted on. |
| D41 | `Plan::partition`'s `panels >= p` early return is **gated** on `l3_domains(p) > 1`, `blocks >= p` and `k <= 64`, and then swaps to `1 x min(p, blocks)`. Behind `TENSORCONTRACT_PARTITION=domain`, off by default. | Removing the early return is wrong on a one-L3-per-socket machine, where it is worth nothing and the code is already right; A36 identifies the domain count as the discriminator and the engine already probes it. The other two conditions are the ones that stop the fix from costing 2–5x on the narrow half and 25% on the complex compute-bound half. Binary rather than modelled because a traffic term cannot be calibrated (this part) and because the two arms it chooses between are the two that were measured. |
| D42 | The choice is made in a pure function, `columns_beat_rows(blocks, k, p, domains)`, with the legacy default expressed as `domains = 1`. | The gate's whole truth table is then pinnable by a unit test on any machine rather than only on a chiplet one, and each of the three conditions gets a case that turns it off alone — the `abcijk` family satisfies all three, so a gate that quietly stopped consulting one of them would still look right on the corpus. |
| D43 | The row-block menu is addressed by **position**, not by `MR`: `row_blocks` yields `(MR, NR)` and `config_at` takes an index. A repeated `MR` is legal; a repeated shape is not. | An `MR`-keyed menu cannot express an `NR`-only alternate, and the Phase 3 sweep produced one that beats the shipped default (A35). Keying by position makes it reachable at run time (A15's preference for a switch over a rebuild), makes `idx=` honest, and costs the rule nothing because the rule never read `NR`. Appending rather than inserting keeps the committed grid's index numbering valid. |
| D44 | **The domain-aware rule is the default.** `TENSORCONTRACT_PARTITION` unset means `domain`; `legacy` asks for the ungated rule by name. | Taken by the user on 2026-08-04, after the measurement above and separately from D22. It is a strictly smaller decision than turning threads on: a single-threaded caller cannot reach it (`l3_domains(1) == 1`), it made the identical decision on 392 of 392 cases where one L3 serves the thread set, and it is worth 1.133 corpus geomean where sixteen do. There is no machine on which it is known to cost anything, and every partition gives bitwise identical results, so it is never a numerical decision. `legacy` stays reachable because every threaded number committed before this date was measured with it. |
| D45 | **Bitwise identity with serial is no longer a design constraint.** | The user's call, 2026-08-04. It remains *true* today and the tests still assert it, so nothing is being given up yet — but future work may trade it. Two things this does not license, recorded so they are not assumed: concurrent read-modify-write on an output block is a data race regardless of what one thinks of float ordering, so exclusive access per block is still required; and A21 (no `K`-parallelism) was argued from *shape* — needing it implies fewer than `p` micro-tiles in the whole output, which bounds arithmetic intensity — so it survives independently. |
| D46 | Do **not** flip `TENSORCONTRACT_THREADS` to a fixed non-1 default. | Measured: at 64 threads, contractions below ~1 MiB run 1.2–10x *slower* than serial, and the optimal thread count walks 4 → 64 across the size range. A fixed default is wrong at every size but one, and wrong by an order of magnitude at the small end — which is the regime Phase 1 named as this project's real headroom. |
| D47 | Do not build cost-weighted or dynamic partitioning for the dense path on current evidence. | `--stress ragged` raises heterogeneous cases from 11.7% to 87.2% at aperiodic fractions and parallel efficiency does not move (0.976–1.124). TBLIS reached the same conclusion by inaction (part 15). Revisit for block-sparse, where the imbalance is block *size*, not block *regularity*. |
| D48 | The thread-count **amortisation guard** — each thread must be given at least `MIN_FMAS_PER_THREAD = 3e6` real FMAs — behind `TENSORCONTRACT_AMORTISE=on`, **off by default**. | D46 said no *fixed* thread count can be the default because the optimum walks 4 → 64 across 0.25 → 64 MiB. A guard makes a default safe rather than optimal: it converts an order-of-magnitude regression into "no worse than serial", which is the property a default needs. Stated as FMAs per thread it holds the spawn *fraction* constant (~0.2 at 30 µs and 20 G FMAs/s) rather than the count, so it is one constant and not a table. Calibrated offline against `bench-results/worker5139-zen2/phase4g/`, where it leaves **0 of 588 points slower than serial at all four sizes** against 565/229/57/3 unguarded, and where every value in `[3e6, 8e6]` does the same — a plateau, per D29's precedent of checking a modelled constant's insensitivity. Weighted by real FMAs per MAC (1 real, 4 planar/1m, 3 for 3m) because an unweighted `m*n*k` misjudges the two domains in opposite directions. Off by default because every threaded number committed before 2026-08-05 was measured uncapped, `phase4f-threads.sh` asks for an exact width and means it, and this has only been scored offline — the state `TENSORCONTRACT_DEEPEN` was in before it failed its A/B (A20). |
| D49 | A **parked-thread pool** with one operation, `try_broadcast(n, f)`, behind `TENSORCONTRACT_POOL=on`, **off by default**. Build, not `rayon`. | Per-call spawn is ~20–36 µs per thread and first-order below a megabyte (A43); a pool removes it rather than steering around it. `rayon` is refused for this axis on a structural ground, not a preference: the driver is SPMD-with-barriers, and a task that blocks on a barrier inside a bounded pool deadlocks when the pool has fewer workers than participants. `ThreadPool::broadcast` fits the shape but ties the parallel degree to the pool size where `Plan::partition` chooses it per contraction, and a global pool inside a library fights the host runtime — concretely here, since the Julia package and the C consumers bring their own. What is built is ~200 lines of `std`: a mailbox per worker so the submitter wakes only the workers it needs, a shared completion counter, and the submitting thread taking index 0 so a serial caller never causes a thread to exist. `try_broadcast` declines **all-or-nothing** (A48) and recovers from mutex poisoning (A49). Off by default for D48's reasons. |
| D50 | A **batched API** — `batch::BatchItem`, `contract_batched` — with the batch as the *only* parallel axis and each item running serially, via `driver::execute_capped`. Fork-join on `std::thread::scope`; `rayon` fits and is still not used. | It removes the spawn *count* rather than the cost: one spawn set per batch instead of one per contraction, which is the right axis for the many-small-contractions workload Phase 1 named as the real headroom. Nesting the axes would spend the saving again inside every item, hence the cap of 1 per item. Soundness is the borrow checker's: items live in a `&mut [BatchItem]`, so `chunks_mut` *proves* the outputs disjoint and there is no unsafe block in the fan-out. Validation is all-or-nothing before any item runs, which a loop over `Plan::run` cannot give and which is the reason to prefer this entry point even at one thread. `rayon` genuinely fits this axis — no barriers, plain fork-join — and is declined only because ten lines of `std` do it and the crate already owns a pool that can be extended here; recorded so the objection is not mistaken for the inner path's, which is a soundness one. The split is static and contiguous, and **block-sparse is not built**: that is where items differ in *size* rather than in regularity, which is the one place D47's null result does not reach. |
| D51 | Every measurement script resolves its binary from `TC_TARGET` (default `target/`), and the `.sbatch` wrappers set it to `target-job-$SLURM_JOB_ID`. | A cluster job runs in the submit directory and used its `target/`, so a `cargo build` on the workstation — or a `cargo test`, which relinks the same artefacts — replaced the executable each arm invokes, mid-session, and the job did not notice. It happened on 2026-08-04 (part 13) and was inert only by luck; "treat the submit directory as frozen" is a rule that depends on remembering, and this removes the class instead. `node-session.sh` exports it as `CARGO_TARGET_DIR` so the build and the arms agree by construction. The two jobs that never compile *snapshot* the prebuilt binaries into it, which also closes the failure that killed job 6753197: a concurrent `prep` left a 0-byte executable that every arm ran for 0.0 s with `rc=0`. |
| D52 | **The amortisation guard does not ship.** `TENSORCONTRACT_AMORTISE` stays reachable as an off-by-default record of the experiment, like `TENSORCONTRACT_DEEPEN`. | Measured on `worker5086` (part 18): it is a *trade*, not a win — it rescues an over-threaded caller by up to 10.8x and costs a correctly-threaded one 10–39% at 2–8 threads on sub-megabyte work, and which side a caller lands on is something a library cannot know. Per case it leaves **12–14% of points more than 10% slower** at those widths, far outside A39's floor. On top of the pool it is pure loss, reaching **0.32**, because it rations a spawn cost the pool has removed. And the form is wrong, not just the constant: a fixed spawn-*fraction* threshold caps at 2 threads where measurement says 4 is optimal, because the real trade is marginal rather than fractional — and fitting a marginal model is where A33 died. Its calibration was sound and validated at one value of the caller's thread count, which is A20 in a third form (A54). |
| D53 | **`TENSORCONTRACT_POOL=on` was recommended as the default on 2026-08-05 and the recommendation is WITHDRAWN the same day.** The pool stays a switch, off by default (D49); it is **topology-conditional at best**. | Recommended on Zen2 evidence — no per-family cell below 0.99 at any size or width, up to 11.6x, 2.0–2.8x at 16 MiB (part 18) — with the flip deliberately left to the user pending a second machine class. That second machine class **reversed it**: on Ice Lake, 32 cores over one 48 MiB L3, the pool costs up to **2.5x** (46 of 60 per-family cells below 0.97, worst 0.403, 36–48% of individual points below 0.90 at 64 MiB), and at the pre-registered decision cell — 64 MiB, `t32` — it reads **0.80** against the `< 0.97` threshold written down as "a real surprise" (part 19). The Zen2 measurement stands and was always labelled with its machine; what is withdrawn is the default. The a-priori argument that a pool could not be topology-dependent is itself refuted (A55): reusing a thread reuses its allocator arena, so a pool is an allocation-locality change and not merely a thread-lifetime one. **Do not flip this default.** What would revive it: fixing the suspected cause — 32 long-lived buffers at repeating addresses contending in one shared L3 — by offsetting each worker's buffer, then re-running both nodes' arms. |
| D54 | **A third cache source, Darwin `sysctl`, tried *after* `CPUID`; it reports the **performance** cores and reports **no L3**.** `CacheSource::Sysctl`, `hw.perflevel0.{l1dcachesize,l2cachesize,cpusperl2}` and `hw.cachelinesize`. Associativity is **assumed** (`ASSUMED_WAYS = 8`) because Darwin exports none, and the sets are derived from it so `bytes_per_way * ways == size` still holds. | Off x86 the probe fell to `BUILTIN` — 32 KiB L1d, 256 KiB L2, an 8 MiB L3 shared by 4 — on a machine with 128 KiB, a 16 MiB cluster L2 and no L3 at all. Every descriptor was wrong and `tcbench info` reported them as if probed, which is worse than a blank: `l3_domains` was being derived from a fabricated `shared_by = 4`. Three specifics decided the shape. **After `CPUID`, not before**, because an Intel Mac has both and only `CPUID` reports associativity and sets. **`perflevel0`, not the unprefixed keys**, because on a heterogeneous part `hw.l1dcachesize`/`hw.l2cachesize` answer for the *last* perflevel — the efficiency cores — so the naive read returns 64 KiB and 4 MiB for an M3 Max whose P-cores have 128 KiB and 16 MiB, a silent 2x/4x under-block that looks like a successful probe. **`l3: None`**, because a P-cluster's L2 is the last level Darwin names and the ~48 MiB SLC behind it is not exported; reporting the cluster L2 as both L2 and L3 would double-count it in every budget. Assuming the associativity rather than declining the probe is the right trade because `size`, `line` and `shared_by` are all real and are the only fields the **default** `legacy` path consults — `ways`/`sets` reach only `analytical`, which is not the default and is refuted (A33). **The consequence to know:** `l3_domains` treats a no-L3 part as one domain per thread, so on Apple Silicon it *over-counts* — six cores really share 16 MiB, so 12 threads span two hardware domains and this reports twelve. Nothing single-threaded can see it (`l3_domains(1) == 1`, D44's gate never fires) and `TENSORCONTRACT_L3_DOMAINS=2` expresses the physical count. Which of the two is the right input to the partition rule on a cluster-L2 topology is **unmeasured**, and that is what the override is for. |
| D55 | **The NEON register-block menus are the measured ones**, three arms of `examples/kernel_shapes` on an M3 Max, ordered by measurement, with the budget-derived incumbent kept wherever the methods tied. `f64` real `16x3`, planar `4x6`, 1m `2x8`, 3m `2x8`; `f32` real `8x8`, planar `8x6`, 1m `8x6`, 3m `4x8`. | A34 requires it: register blocks are per-microarchitecture, and a menu picked from a register budget is a model. The budget got three of eight wrong (A60). Six of the eight winners are **ties inside the session floor** and are labelled as such rather than promoted, which is the whole reason three arms were run (A62). Three shapes are excluded outright as *unstable* — 1m `3x10` swings 45% between arms — which is a stronger disqualification than measuring slower. **`f64` real `16x3` is the one that needed a second decision:** it is a real 4.9% kernel win at 2.6x the floor, and `MR = 16` costs write-back regularity on 12 of 49 `f64` cases because 16 does not divide TCCG's 24. It ships because the guarded row-block rule already demotes exactly those 12 to `4x8` on its `k <= 32` guard, leaving 0 of 392 without a regular shape (A61) — not because 12 cases were judged tolerable. |
| D56 | **`scripts/kernel-shapes-compare.py`, and the rule that a register block may not be changed on one sweep arm.** Two or more arms, scored on the per-cell median, floor as a p90, `>5%` arm-to-arm spread reclassified as *unstable* rather than noisy. | The sweep's `best per method` footer is the thing that gets copied into a `cfg_*` menu, and it is a single number with no error bar — exactly the shape of the mistake A46 records. Building the floor into the tool rather than the procedure is what makes it survive a handoff. The `>5%` reclassification earns its place separately: a bimodal cell is not a noisy measurement to be averaged, it is a shape that must not ship at any mean, and averaging would have hidden all three found here. |

---

## Running a measurement session

How a cluster session is designed and driven, and the pre-registration parts 7–9 and 11 consume.

### Part 10: taking the two pending measurements to a cluster node

**Status: designed, tooled and pre-registered. Nothing is measured yet.** This
section is written *before* the run so that the node choice, the placement
hypothesis and its accept/reject rule are on the record and cannot be adjusted to
fit whatever comes back. Results go into parts 7, 8 and 8b, and a new part for
the AVX2 calibration.

#### The node choice *is* the experiment design

The two pending measurements want an exclusive machine, and `ccqlin038` is a
shared workstation. Rusty has one, but **no Cascade Lake node**, so the choice of
partition picks which question gets answered:

| feature | S:C:T | cores | ISA consequence |
|---|---|---|---|
| `rome` (Zen2) | 2:64:1 | 128 | **no AVX-512** — runs the AVX2 kernels |
| `icelake` | 2:32:1 | 64 | AVX-512, Intel, closest to the reference machine |
| `genoa` (Zen4) | 2:48:1 | 96 | AVX-512 (double-pumped), a third cache topology |

`T:1` in all three: **SMT is off on Rusty's CPU nodes**, so the hyperthread
sibling that forced two Phase 4 retractions is absent rather than merely handled.
That is checked on the node by `scripts/topology.py` rather than believed from
`sinfo`, and `phase4f-threads.sh` keeps its sibling guard regardless — a guard
that trivially passes costs nothing and stops being load-bearing only when
someone proves it is.

**Chosen: `rome`.** Three reasons, in order of weight:

1. **It converts a documented guess into a measurement.** The AVX2 register
   blocks are provisional and explicitly unmeasured (D26): chosen from a register
   budget and a uop model on a machine that cannot execute them competitively.
   `rome` is the largest partition on the cluster and AVX2 is what most users
   get, so this is the widest remaining gap between what this file measures and
   what a user experiences.
2. **Its topology is the one where concurrent placement can work.** Zen2 shares
   one L3 between four cores, so one arm per L3 domain gets a *private* L3 —
   better isolation than `ccqlin038` ever had, where eight cores share 25 MiB. On
   `icelake` the L3 is a whole 32-core socket, which leaves about two usable
   domains and no way to run the 40-job grid concurrently at all.
3. **It tests the analytical model harder than `icelake` would.** The point of
   D23 was that the legacy constants are `ccqlin038`'s cache sizes written down
   by hand. Zen2's are further away than Ice Lake's in the direction that
   matters: a 512 KiB L2 against a legacy `A`-block budget of 512 KiB, i.e. the
   constants ask for the entire L2 where the model would reserve ways for the
   streaming operand. If the model is going to win anywhere, it should win here,
   and if it does not, that is a finding about the model rather than about the
   node.

What this deliberately gives up, stated so it is not discovered later as a
surprise: `rome` cannot say anything about AVX-512 blocking on a second Intel
hierarchy, which is the `icelake` question and remains open. And the blocking
grid measured on `rome` sits on top of register blocks that are themselves
unmeasured until the `shapes` stage runs — so the calibration must be read
before the grid, and the grid's conclusions are conditional on the shapes the
engine actually shipped that day, not on the calibrated ones. Recalibrating and
re-running the grid is a second session, not this one.

#### Nothing measured there is comparable to anything above this line

Different machine, different cache hierarchy, different instruction set. Every
ratio must be computed *within* the session, and **the ±1.3% geomean / ±6%
per-case noise floor is `ccqlin038`'s and does not transfer.** The floor is
re-derived on the node from bracketing repeats the scripts already run:
`phase4f-threads.sh` now compares `t1` against its bracketing `t1b` and prints
that first, `validate-placement.sh` runs the same arm solo twice for the same
purpose, and the grid keeps its three `base` repeats. No effect gets quoted
before the floor it is measured against.

#### The placement problem, and the hypothesis

An exclusive 128-core node makes both obvious answers wrong. One arm at a time
leaves 127 cores idle, and 40 grid jobs at ~22 min each (on the *reference*
machine; AVX2 will be slower per core) is a day's work. Forty arms at once
corrupts the quantity being measured, because arms sharing an L3 or a memory
controller perturb each other and `NC` is *sized for L3*.

So the placement is treated as a hypothesis: **one measurement thread per L3
domain, every other core in that domain idle, SMT siblings idle, and concurrency
held below the domain count** — 24 of 32 by default on `rome` — because memory
bandwidth and the interconnect stay shared however threads are placed.

The accept/reject rule, fixed now:

* **Accept** if placed-vs-solo lies inside the solo-vs-solo floor on the
  geometric mean, and no case moves by more than the per-case floor.
* **Reject** otherwise, run the grid sequentially, and record the rejection.

Both arms are measured on the full corpus *and* on the `abcijk` family, which is
exactly the 18 cases that contract over `k = 24` — the memory-bound half, i.e.
where bandwidth contention should show up first, bandwidth being the one resource
no placement can privatise. A rejected placement is a result worth writing down:
it tells the next person on a different machine what to expect, and the mechanism
(a private L3 per CCX against a socket-wide L3 on Intel) predicts that the answer
differs by machine.

#### What was built for it, all of it zero-CPU

* `scripts/topology.py` — L3 domains, SMT siblings, NUMA and cache descriptors
  from the *allocation's* affinity mask, so it describes the job and not the
  node, plus the placement plan. Emitted as JSON so the runner and this file read
  the same facts.
* `scripts/run-arms.py` — runs independent arms one per L3 domain, and records
  `/proc/stat` occupancy for **every core in each arm's own L3 domain** across
  exactly that arm's window, plus the observed overlap with other arms. The
  reference scripts sample the pinned core and its sibling; this widens that to
  the domain, because on this placement the domain is the unit that has to be
  clean. That recording is what made the earlier retractions detectable.
* `scripts/validate-placement.sh` + `scripts/placement-spread.py` — the
  hypothesis test above, including the spread *between* slots, which prices
  position within the node. A uniform slowdown cancels in the ratios the grid is
  scored on; a position-dependent one does not, and would turn slot assignment
  into a per-arm bias.
* `phase4e-blocking.sh` now runs both regimes through one code path
  (`<cpu>` sequential, `auto` placed) from a single arm list, so the two cannot
  drift apart. `phase4f-threads.sh` derives its cpuset from the allocation
  instead of a hardcoded `0-7`, scales its thread counts to the core count, gives
  each arm exactly as many cores as it has threads (so L3 sharing is a known
  function of `nt` rather than the scheduler's choice), records occupancy over the
  whole allocation as an exclusivity check, and labels the cross-socket arm as the
  separate question it is.
* `scripts/node-session.sh` — stages the session so the ordering constraints are
  enforced rather than remembered: one stage at a time, only `prep` compiles, and
  a guard refuses to start a measurement while any `cargo`/`rustc`/`tcbench` of
  the user's is alive.
* `scripts/placement-verdict.py` — the accept/reject rule **as code**, so an
  unattended run can act on it and, more to the point, so the threshold cannot
  drift once the numbers are visible. Strict reading of the pre-registered rule:
  the per-case bound is the solo pair's *worst* case, not a percentile, because a
  percentile is a knob and a knob chosen after seeing the data is how a
  pre-registered rule stops being one. Every placed replicate must pass, not just
  the one on the reference core, since the grid assigns arms to slots arbitrarily.
* `scripts/rusty-phase4.sbatch` — the whole session unattended. Stage order is
  `prep, shapes, threads, validate, grid`: `shapes` moves ahead of the
  higher-priority `threads` because on an AVX2 node it is the highest-value single
  deliverable and costs twenty minutes, so putting it behind a four-hour stage
  risks the cheap irreplaceable thing for nothing. Stages are independent and a
  failure does not abort the rest.

**The rejection path returns data rather than nothing.** A rejected placement
means the full grid cannot run — sequential is ~23 h on this node and does not fit
a 12 h allocation — so the batch script runs a *scoped* grid sequentially inside
the wall time that is left, in the order `base model base2` then the `kc` family:
the model arm is a whole different derivation rather than a point in the grid and
needs its brackets to mean anything, and `kc` is the first-order parameter. Arms
that do not fit are named in `skipped-arms.txt` and in the log, because a bounded
run that does not say what it dropped reads as complete coverage. A scoped grid is
reported as scoped — offline rule scoring against a partial grid is not the asset
that scoring against a whole one is.

#### The prediction, made first

`scripts/thread-width.py` no longer hardcodes `P = 8` — it takes a list, and it
now replays `Plan::partition` (`PACK_WEIGHT` included) instead of only reporting
raw widths, so it predicts what the *rule* will ask for rather than what the shape
allows. Replayed over all 392 case-dtype-methods:

| P | no fill from `M` | rule goes 2-D | leaves threads idle | of which int-div waste | of which a genuine limit | needs `K` |
|---|---|---|---|---|---|---|
| **AVX-512 blocks (the reference machine's)** ||||||
| 8 | 16 | 12 | 4 | 0 | 4 | 0 |
| 16 | 26 | 22 | 12 | 4 | 8 | 0 |
| 32 | 32 | 30 | 16 | 4 | 12 | 0 |
| 64 | 33 | 32 | 18 | 12 | 6 | 0 |
| 128 | 72 | 68 | 49 | 46 | 3 | 0 |
| **AVX2 blocks (what `rome` will run)** ||||||
| 8 | 1 | 1 | 0 | 0 | 0 | 0 |
| 16 | 8 | 6 | 2 | 0 | 2 | 0 |
| 32 | 23 | 15 | 10 | 0 | 10 | 0 |
| 64 | 32 | 26 | 16 | 0 | 16 | 0 |
| 128 | 32 | 32 | 20 | 6 | 14 | 0 |

Four things fall out of it, none of which needed the machine:

1. **`K`-parallelism is never needed, at any thread count up to a whole 128-core
   node, in either ISA.** A21 was argued at 8 threads and the structural argument
   was claimed to generalise; this is that claim checked to 16x the width. A21 is
   now confirmed far outside the regime it was made in.
2. **The 2-D rule will be exercised 5x harder than it was designed against.** It
   fires on 12 of 392 at 8 threads and 32–68 at 128. `PACK_WEIGHT` was priced by
   replay at 2–32 threads and every weight in `[4, 64]` gave the same partition;
   that replay does not cover 64 or 128, so the `=m` / `=n` arms at the top thread
   count are the load-bearing part of the threading run, not the scaling curve.
3. **Most of the apparent shortfall at large `P` is arithmetic, not judgement.**
   `pn = min(p / pm, blocks)` is integer division, so a rule that wants `9 x 14`
   out of 128 threads gets 126 and idles two. Separating that from a genuine
   refusal matters, because only the second kind can show up as a visibly flat
   scaling curve: at `P = 128` on AVX-512 blocks, 46 of the 49 are integer-division
   waste of ≤2%, and just **three** are real.
4. **AVX2's smaller register blocks buy parallel width**, and by a lot: at 8
   threads 16 of 392 case-dtype-methods cannot fill the threads from `M` on
   AVX-512 blocks against **1** on AVX2, because `MR` is 3–6x smaller and the row
   panel count rises accordingly. This is a fresh instance of A25 — smaller
   register blocks are not purely a cost — in a dimension A25 did not consider,
   and it means the `rome` scaling curve should look *better* than the reference
   machine's would, for a reason that has nothing to do with the cores.

The genuinely partition-limited population on AVX2 is one family: `ij-ikl-ljk`
and `ij-kil-lkj`, which cap at `27x1` of 32 threads and `54x2` of 128 (84%).
Those are the curves that must flatten. If any *other* case flattens, that is
contention or a bug, not the partition — which is what makes this prediction
worth having.

Caveat on the table: the `frac of best` column in the full output ranks cases by
throughput measured on `ccqlin038`, used only as a compute-bound proxy for
sorting. It is not a cross-machine performance claim and is regenerated from the
node's own `t1` arm once that exists.

#### Result: the placement is clean on the corpus and *not* clean on the memory-bound half

Measured on `worker5040`, 24 concurrent arms one per L3 domain against a solo arm.
The honest comparison is against the **hot** solo arm (`t1b`, same core), because
the reused solo pair straddles the cold-start transient A31 describes:

| | geomean vs hot solo | range over slots |
|---|---|---|
| full corpus, socket 0 slots (12) | **1.0029** | 1.0017–1.0050 |
| full corpus, socket 1 slots (12) | 1.0215 | 1.0181–1.0255 |
| **memory-bound `abcijk` half, all 24** | **0.9684** | 0.9616–0.9769 |

Read in order:

1. **On the corpus as a whole the placement costs 0.3% — nothing.** Twenty-four
   arms running at once, each with a private 16 MiB L3 and three idle cores in its
   domain, measure what a solo arm measures. A27 is confirmed, and it bought a
   **23.7x** speedup: 533 minutes of arm-time in 22.5 minutes of wall clock, which
   is the difference between the grid being affordable and not.
2. **The 2.2% socket split is thermal, not spatial.** The `threads` stage had run
   for two hours entirely on socket 0, so socket 1 was cold when the placed round
   started; its slots read 2.2% *faster*. `placement-spread.py` was built expecting
   memory-system position to be the variable, and on this node it is temperature
   history instead. Confirmed by the memory-bound round twenty minutes later, when
   both sockets had been loaded and the split had shrunk to 0.6%.
3. **On the memory-bound half the placement is rejected, decisively.** 3.2%
   geomean and up to 10% per case, against a solo-pair floor of **0.02%** — the
   cleanest floor in this whole project, from two arms 69 seconds apart. Every one
   of the 24 slots fails. Bandwidth is the resource no placement can privatise, the
   `abcijk` family at `k = 24` is where that bites first, and it does.

**A gap between the rule as written and the rule as coded, which is mine.** Part 10
pre-registered that both the full corpus *and* the `abcijk` family would be
measured, but `rusty-phase4.sbatch` called `placement-verdict.py` with its default
prefix, so only the full-corpus arms were scored and the grid launched placed on an
ACCEPT that had not consulted the memory-bound arms. Scored after the fact with
`--prefix mbp --solo mbsolo`, they REJECT. The verdict script now takes `--solo` so
the rule can be applied to the arm set it was written for; the sequencing error had
already happened.

**What that does to the grid, and what it does not.** The grid is scored on
*ratios between arms*, and every arm ran under the identical 24-way placement, so a
uniform 3.2% penalty on memory-bound cases cancels in `arm / base`. The exposure is
not uniformity but **interaction**: an arm that changes memory traffic suffers
different contention, and that is exactly what the `nc` arms and the `model` arm do
— `model` raises `nc` about 4.7x here. So:

* `kc` and `mc` arms: the penalty is close to uniform and the ratios stand.
* `nc25`, `nc400`, `model`: **confounded at the same order as the effect**, and
  `model` is the single most interesting arm in the grid.

The fix is cheap and does not need the whole grid re-run: measure `base`, `nc25`,
`nc400`, `model`, `base2` sequentially on one core — 10 jobs, ~3.7 h — and compare
those against each other in the clean regime, using `base`/`base2` to tie them back
to the placed run. `scripts/rusty-phase4-seq.sbatch` does exactly that and nothing
else. **Until it has run, do not quote the `nc` or `model` arms from the placed
grid.**

| # | Assumption | Status |
|---|---|---|
| A27 | Concurrent arms placed one per L3 domain measure the same thing as a solo arm. | **Confirmed on the corpus (+0.3%), refuted on the memory-bound half (−3.2%, up to −10%).** Private per-CCX L3 is enough for compute-bound work and irrelevant to bandwidth: `abcijk` at `k = 24` is bandwidth-bound and 24 arms contend. Use the placement for arms that do not change memory traffic; measure the `nc` and `model` arms sequentially. |
| A32 | A session's drift is a single number. | **Refuted; it is a function of how far apart the two arms are.** Same node, same core, same corpus: 0.02% at 69 s apart, 1.1–1.9% at ~1 h, 4.4% across the cold-start transient. Quoting one floor for a whole session is what let a 4.4% cold-start artifact be mistaken for the precision of a repeat. |

#### What would invalidate the run

Checked before anything is believed, and each one recorded rather than assumed:
the allocation is genuinely exclusive (`scontrol show job`, once, plus the
per-core occupancy every arm now records for the whole allocation); no compile
overlapped a measurement (`node-session.sh` refuses); the bracketing repeats
agree within the floor derived in the same session; and no number is compared
across machines.

| # | Assumption | Status |
|---|---|---|
| A27 | Concurrent arms placed one per L3 domain measure the same thing as a solo arm. | **Hypothesis, with the test and the accept/reject rule pre-registered above.** Zen2's private-per-CCX L3 is what makes it plausible; shared memory bandwidth is what makes it doubtful, which is why the memory-bound `abcijk` half is measured separately. |
| A28 | The 2-D partition rule and `PACK_WEIGHT` behave at node scale as they do at 8 threads. | **Open, and the reason the threading run is worth more than a scaling curve.** The rule fires on 12 of 392 at 8 threads and 32–68 at 128; `PACK_WEIGHT`'s indifference was only ever replayed to 32. |
| A29 | Parallel width is a property of the contraction. | **Refuted at no CPU cost — it is a property of the contraction *and the ISA*.** Row panels scale as `M / MR`, so AVX2's smaller register blocks give the same corpus 3–6x more of them: 1 of 392 case-dtype-methods short of 8 threads against 16 on AVX-512. |

## The write-back and the two shape rules

Phase 4 item 1 and its three follow-ons. The 2x defect, its real cause, and the two rules that buy it back — plus the menu re-keying that made a known-better shape reachable.

### Parts 1–5: the write-back, the orientation rule, and the micro-tile row block

Phase 3 left a profiled 2x defect on nine corpus cases and named two candidate
fixes. Both were built. The first one is not the one Phase 3 predicted.

#### What was built

1. **Row/column orientation** (`Plan::transposes_gemm`, applied in `driver`).
   The engine is symmetric under exchanging `(A, M)` with `(B, N)`; doing so
   computes `D^T = B^T A^T`, which is the same contraction seen through the
   transposed matrix view of `C` and `D`. The driver binds `am`/`ak`/`ptr_a` to
   whichever operand plays the row role and nothing downstream branches again.
   For 1m this also swaps the two *different* pack formats ("1e" for rows, "1r"
   for columns), which is right: the kernel's contract is about the row and
   column panels, not about which user tensor they came from.
2. **A regular-block write-back** (`writeback`). The row offsets now come from
   the output's *block* scatter, exactly as packing takes the operands' — three
   instantiations of one inlined body: gather, strided, and unit-stride. Only
   the last lets LLVM vectorise the plane recombination, and it is the case the
   orientation rule exists to create. The `alpha == 1, beta == 0, no
   conjugation` combination gets its own inner loop, hoisted out of the `i`
   loop, because it reduces to a straight tile-to-`D` copy.

`TENSORCONTRACT_ORIENT=none|swap` pins the orientation for A/B measurement.

#### The orientation rule, and the condition that is not obvious

Swap when, and only when:

1. `D`'s column direction is **strictly** more contiguous than its row
   direction (compared on the leading axis of each class, i.e. the smallest
   |stride| in `D`, since that is the axis the register block is cut along); and
2. the swap leaves the micro-tile's row block **unbroken** — the new leading
   axis has unit stride *and* extent at least `MR`.

Condition 2 was not anticipated and is the reason the first version of this
rule regressed. On the `abcijk-*mb-*` family the swap makes the leading axis
unit-stride but only 24 long, and the outcome is monotone in `run / MR`:

| dtype | kernel `MR` | run / `MR` | swap vs no swap |
|---|---|---|---|
| `c64` | 16 | 1.50 | **+18%** |
| `f64` | 24 | 1.00 | **+23%** |
| `c32` | 32 | 0.75 | 0% |
| `f32` | 48 | 0.50 | **−17%** |

A shattered row block keeps the swap's costs and loses its benefit. Note this
makes the choice element-type dependent through `MR`, so it is not a property
of the plan alone — `transposes_gemm` and `oriented_scatters` both take `MR`.

Those figures were taken on a shared workstation and were re-measured on an
exclusive one at 24 reps (see part 4). They hold: `f32` `-mb` reproduces at
0.862 / 0.836 / 0.848 across the three cases, against a per-case noise floor of
±6%, and the monotone ordering is intact.

**The −17% is not fully explained and should not be presented as if it were.**
`perf stat` on the `f32` case shows instructions flat to 1%, dTLB misses flat,
and L1 load misses *down* 27% under the swap, yet 4% more cycles at IPC 1.05 →
1.00. Counting cache lines per micro-tile also favours the swap (24 fully
written lines against 48 half-written ones), as does the packing pattern. So
the rule's condition 2 is an empirical guard rail, not a derived one.

#### Result: the 2x defect is closed, with no case left slower

Full 49-case corpus, `--size 64 --reps 3`, single core, against the *recorded*
Phase 3 sweep (`bench-results/phase3-sweep-*.csv`) — same basis, same machine.
Cases the rule does not swap are a control group: the change cannot affect
them, so their spread is a direct read of run-to-run noise.

| | `f32` | `f64` | `c32` planar | `c64` planar |
|---|---|---|---|---|
| geomean, all 49 | **1.047** | **1.105** | **1.046** | **1.102** |
| geomean, swapped cases | 1.620 | 1.700 | 1.485 | 1.801 |
| geomean, control group | 0.985 | 1.040 | 0.996 | 1.029 |
| worst swapped case | 1.493 | 1.315 | 1.094 | 1.153 |

The nine cases Phase 3 profiled, `c64` planar, all three `M`-leading strides:

| `M` leading axis | its stride in `D` | Phase 3 | now |
|---|---|---|---|
| `a` | 1 | 40.5 | 40.9 |
| `b` | `n_a` | 36.2 | 44.1 |
| `c` | `n_a * n_b` | 17.7 | **47.2** |

Throughput is no longer monotone in that stride — it is flat to ±8%, and the
case that was 2.3x slower than its sibling is now the fastest of the three.
The `-mc` family gains **2.6–2.7x** in `c64` planar.

A first version of the rule without condition 2 swapped 12 of 49 cases and
scored 1.07–1.19 geomean *with* a −15% worst case; the guarded rule swaps 6–12
depending on element type, scores slightly lower on paper, and has no
regression beyond noise. Raw data for both: `bench-results/phase4/orient-*.csv`.

#### Part 2: the regular-block write-back, on top of the orientation fix

First measured as a diff between two builds half an hour apart, which gave
`c64` 3m = 0.973 and the conclusion "a wash in double". Both were wrong. With
`TENSORCONTRACT_WRITEBACK=gather` the comparison became a runtime A/B inside one
session (`rm-B-*.csv` → `rm-A-*.csv`), and it is positive everywhere:

| | planar | 1m | 3m |
|---|---|---|---|
| `f32` / `f64` (one real path) | 1.071 / 1.019 | — | — |
| `c32` | 1.104 | 1.121 | **1.153** |
| `c64` | 1.032 | 1.017 | 1.020 |

Every entry clears the ±1.3% geomean noise floor. It pays about 5x more in
single precision than double, which is the ratio to expect: the per-output
cost it removes is fixed, while the kernel work it hides behind scales with the
element size, so halving the element doubles the relative overhead.

The lesson is the methodological one. Nothing about the code changed between
the two measurements — only that the arms were interleaved instead of
sequential. A build-to-build diff cannot distinguish a 3% effect from drift;
put the switch behind an environment variable and the same question answers
itself in one run.

#### Cumulative Phase 4.1 result

Full corpus, against the recorded Phase 3 sweep, exclusive machine
(`rm-A-*.csv`), against a ±1.3% geomean noise floor:

| dtype | planar | 1m | 3m |
|---|---|---|---|
| `f32` | **1.142** | — | — |
| `f64` | **1.127** | — | — |
| `c32` | 1.116 | 1.122 | **1.167** |
| `c64` | **1.118** | 1.058 | 1.048 |

And the nine cases Phase 3 profiled as a 2x defect, planar, GF/s, averaged over
the three cases sharing each `M`-leading axis:

| `M` leading axis | `c64` | `f64` | `c32` | `f32` |
|---|---|---|---|---|
| `a` (stride 1) | 40.5 → 43.5 | 32.4 → 35.5 | 82.8 → 85.8 | 52.8 → 52.3 |
| `b` (stride `n_a`) | 36.4 → 44.9 | 25.0 → 31.9 | 63.3 → 67.3 | 40.6 → 40.0 |
| `c` (stride `n_a n_b`) | 17.8 → **49.0** | 13.8 → **27.9** | 40.1 → **87.0** | 27.0 → **60.1** |

The 2x defect is gone in every precision — 2.75x, 2.02x, 2.17x, 2.23x on the
affected cases — and the ordering has inverted: what was the slowest of the
three is now the fastest. The 4–5% loss on the `f32` `a`/`b` rows reported
before the machine was quiet does **not** survive re-measurement; those rows
are flat to within noise, and the paragraph explaining them as a write-back
side effect was explaining contention.

#### Part 3: a negative result on depth-adaptive `MC` — do not redo this

The obvious first move on item 2 looked free and is not. `Blocking::derive`
sizes `MC` and `NC` so a `KC`-deep packed block fits its cache budget, but a
third of the corpus contracts over `k = 24` against a `KC` of 256. Those cases
pack an `A` block a tenth the size of its L2 budget, and the only consequence
is that `B` is re-streamed `M/MC` times for nothing. Re-deriving against
`min(k, KC)` — widening `MC` about tenfold — should be pure profit.

It is not. Unconditionally, `abcijk-jkm*` planar:

| case | `f32` | `f64` | `c32` | `c64` |
|---|---|---|---|---|
| `-ma` | 50.6 → **32.5** | 35.5 → 30.1 | 83.1 → **61.2** | 42.9 → 43.2 |
| `-mb` | 39.4 → **24.3** | 32.0 → 29.8 | 65.0 → **53.2** | 45.0 → 43.4 |
| `-mc` | 56.3 → 56.3 | 27.6 → **31.7** | 93.5 → **104.1** | 48.6 → **53.4** |

The mechanism for the losses is visible: `MC` also bounds the strip of `D` that
one `jr` pass touches and the *next* pass revisits. When the output's rows are
strided, that strip is `MC` distinct cache lines, and a tenfold `MC` takes it
from 24 KiB (L1-resident) to 350 KiB. The packed-`A` budget does not model this
at all.

Gating the widening on "every `D` row block is unit-stride" — where the strip
is `NR` sequential runs, streamed and written once — removes the `f32`/`c32`
losses, but a clean A/B on `f64` (all three cases unit-stride, pinned blocking
vs adaptive, 15 reps) shows the rule still does not hold:

| case | pinned `mc=256 kc=256 nc=1536` | depth-adaptive |
|---|---|---|
| `-ma` | 37.1 | 30.5 (**−18%**) |
| `-mb` | 32.9 | 29.9 (−9%) |
| `-mc` | 28.4 | 32.0 (**+13%**) |

Three cases, one sign each way, no rule. **Backed out.** `Blocking::derive_at_depth`
is kept because the sweep needs to vary `kc` and get budget-consistent `mc`/`nc`
with it, but the driver does not call it. The lesson for item 2 is that `MC` is
a two-sided constraint — packed-`A` residency below, output-strip residency
above — and the sweep has to be designed to separate them rather than to find a
single best `MC`.

#### Part 4: what an exclusive machine changed, and the rule's 9 known misses

Everything above was first measured while other work shared the workstation.
The harness pins to one logical CPU, but nothing stopped a co-tenant landing on
its **hyperthread sibling**, which shares the 32 KiB L1d and 1 MiB L2 — the
exact resources all of this is about. `scripts/phase4-remeasure.sh` redid it
with exclusive access, running `A, B, A'` so the identical repeat brackets the
treatment, and recording `/proc/stat` occupancy for the pinned core and its
sibling next to every result (`*.cpu`: cpu4 at 100%, cpu20 at 0.5–1.0%
throughout).

**Measured noise floor**, A vs A′ at the reps the corpus sweeps use:

| | geomean over 49 cases | per case |
|---|---|---|
| spread | 0.992 – 1.013 (**±1.3%**) | 0.92 – 1.06 (**±6%**) |

That is tighter than the ±3% / ±13% inferred earlier from the contended control
group, so the corpus geomeans stand and the per-case claims get a real error
bar. **Quote these, not a control-group inference.**

Then, since the orientation A/B forced both arms on all 18 `abcijk` cases in
all four dtypes at 24 reps, it also grades the rule itself. **The rule picks
the better arm, or ties within noise, on 63 of 72 case-dtypes.** The 9 misses
are large, systematic, and all 32-bit:

| case family | dtype | rule | better | left on the table |
|---|---|---|---|---|
| `abcijk-e{i,j,k}bc-*` | `f32` | AB | BA | **1.43–1.47x** |
| `abcijk-e{i,j,k}bc-*` | `c32` | AB | BA | **1.35–1.36x** |
| `abcijk-e{i,j,k}ac-*` | `f32` | AB | BA | 1.18x |

These are unrealised gains, not regressions — the rule picks the pre-Phase-4
behaviour there — but 1.4x on six corpus cases is larger than anything else
left on the Phase 4 list.

**Condition 2 is a proxy, and these show it is the wrong one.** The `e*bc`
cases have the *same* post-swap row structure as the `-mb` family that
condition 2 correctly rejects — leading axis unit-stride, extent 24, against an
`MR` of 32 or 48 — and yet swapping gains 1.4x where `-mb` loses 15%. So
`run / MR` cannot be the discriminant; it merely correlates on the cases it was
derived from. The `e*ac` row is worse still: there the rule declines at
condition *1*, and in `f32` swapping to the **less** contiguous row direction
wins by 1.18x while in `c64` it loses 18% — opposite signs for the same shape
in different precisions.

Two structural differences are visible but neither has been tested: after the
swap, `e*bc`'s column direction folds to a run of 576 in `D` where `-mb`'s is
24; and `e*bc`'s row operand packs *less* regularly than `-mb`'s, i.e. the
faster arm is the one with worse block-scatter regularity. Do not adopt either
as a rule without measuring it.

The useful asset from this is the data: `rm-orient-{none,swap}.csv` hold both
arms for all 72 case-dtypes, so a candidate discriminant can be scored offline
against ground truth without running a single new benchmark. Failing that, the
orientation is one cheap binary choice per plan and plans are reusable, which
makes empirical selection — run both once, keep the faster — the honest
fallback.

#### Where the next lever is, and why it is now sharper

The `f32` `a`/`b` rows above gained least — and they are the ones still on the
write-back's **gather** path, for a structural reason rather than a measured
regression: their leading axis in `D` has unit stride but extent 24, while
the `f32` real kernel's `MR` is 48. A 48-row block therefore straddles a
discontinuity and the block scatter reports it irregular — the same quantity
that condition 2 of the orientation rule turns on. Choosing `MR` to divide the
output's leading contiguous run would convert those blocks to the unit-stride
path outright. That reframes the Phase 3 "micro-tile aspect ratio should follow
the output's stride pattern" item from a ±11% tuning knob into a way of
reaching a qualitatively faster code path, and it is cheap because the kernels
are already parameterised over `(MV, NR)`.

Note this is the *same quantity* — `run` against `MR` — that part 4 shows is
not the true discriminant for the orientation. The two items are therefore
coupled: `MR` is simultaneously a free parameter of the aspect-ratio choice and
an input to the orientation rule, and every one of the nine orientation misses
is a 32-bit case where `MR` (32 or 48) exceeds the run (24). Changing `MR` to
16 for those shapes would satisfy condition 2 and make the rule pick BA without
any new discriminant. **Do 1c before trying to fix the orientation rule** — it
may dissolve the problem rather than require solving it.

> **Superseded by parts 5 and 6.** It did not dissolve it: shrinking `MR` does
> flip those cases to `BA`, but they gain 1.17–1.39x *while paying ~30% in
> kernel shape*, which prices the orientation at ~2x and makes `MR` the wrong
> instrument. The discriminant was found in part 6 and is unrelated to
> `run / MR`: the rule has to be **antisymmetric** under exchanging the two
> directions. Read part 6 before acting on anything in this section.

#### Part 5: item 1c, the micro-tile row block — and what it prices

Part 4 predicted that choosing `MR` to divide the output's contiguous run would
"dissolve" the orientation problem. It does not. It **prices** it, which is more
useful, and the write-back gain it was aimed at is real but small and needs
three guards to be positive at all.

##### What was built

`kernel::x86` now builds each method from a **menu** of register blocks rather
than one, default first, using the same const-generic kernels Phase 3 wrote.
The alternates were priced by re-running `examples/kernel_shapes`, extended with
the `MV = 1` real and 1m cases it never covered
(`bench-results/phase4c/kernel-shapes.txt`). `Plan::row_block` chooses from the
menu, `Plan::row_block_score` reports the fraction of output row blocks that
would stay off the gather path at a given `MR` — evaluated in the orientation
*that* `MR` selects, since the two are coupled — and
`TENSORCONTRACT_ROWBLOCK=base|auto|mr=<n>|idx=<i>` makes every arm reachable at
run time. `tcbench shapes` scores every shape against every corpus case without
running anything, which is how the measurements below were chosen.

##### The grid, and why it was worth 2 h

`scripts/phase4c-rowblock.sh` pins *every* shape on *every* menu across the
whole corpus (`bench-results/phase4c/rb-*`, sibling CPU 0.6–1.3% throughout).
That is worth far more than an A/B of the rule, because it splits each
(dtype, method, shape) into cases where the shape changes nothing the
write-back can see — their ratio is the shape's own cost — and cases where it
does. Any candidate rule can then be scored offline against ground truth, which
is what `scripts/rowblock-score-rules.py` does:

| rule | `f32` | `c32` planar | `c32` 1m | `c32` 3m | `c64` planar |
|---|---|---|---|---|---|
| maximise the regular fraction | **0.936** | 1.006 | 1.026 | 0.987 | 1.040 |
| + only where `k <= 32` | 1.006 | 1.018 | 1.027 | 0.987 | 1.034 |
| + only reaching *full* regularity | 0.997 | 1.004 | 1.027 | 0.987 | 1.034 |
| + only if the orientation is unchanged | 0.997 | 1.004 | 1.015 | **1.000** | 1.034 |
| oracle, best shape per case with hindsight | 1.042 | 1.051 | 1.068 | 1.033 | 1.054 |

The obvious rule — the one part 4 proposed — is a **loss**. Each guard is
measured, not argued:

1. **`k <= 32`.** The write-back costs a constant per output element against
   `~4k` flops of kernel work, so which path it takes only matters while `k` is
   small, and a shape off the kernel's peak always costs something. At `k = 24`
   the winning shape gains 1.09–1.26x; at `k >= 204` the identical change is
   1.01–1.04x and still being paid for. The corpus jumps from `k = 24` to
   `k = 52`, so it resolves this boundary only to somewhere in `(24, 52]`.
2. **The default must be substantially broken** (regular fraction `<= 0.75`).
   Taking `f32` `48x8 -> 32x8`, where the default was already 0.88 regular, lost
   7–9%. The corpus only produces the values 0, 0.67, 0.88 and 1.0, so any
   threshold in `(0.67, 0.88]` fits it equally.
3. **The orientation must not change.** `MR` is an input to
   `transposes_gemm`, so a shape change can silently flip it. Without this
   guard `c32` 3m moves the three `abcijk-e*bc-*` cases from the good arm to
   the bad one and loses **19%**.

##### Result

Validation A/B (`bench-results/phase4c/v-*`, `base, auto, base'`, exclusive
machine, sibling CPU 0.7–0.9%). Session noise floor from the bracketing repeat:
geomean 0.992–1.008.

| | corpus geomean | fired cases | n | untouched cases |
|---|---|---|---|---|
| `c64` planar | **1.026** | **1.124** | 12 | 0.996 |
| `c32` 1m | 1.006 | **1.108** | 7 | 0.990 |
| `c32` planar | 1.005 | 0.977 | 1 | 1.005 |
| everything else | 0.992 – 1.000 | — | 0 | 0.992 – 1.000 |

The rule fires on **20 of 392** case-dtype-methods. Per case it runs 1.065 to
1.202 on 19 of them — `abcijk-ijma-mkbc` in `c64` planar goes 43.2 → 52.0 GF/s
— and 0.931 on the twentieth, `ajbdc-ckbad-jk` in `c32` 1m, a case that runs at
8.5 GF/s. That is the one case left slower, and it is only just outside the
per-case noise floor. The untouched cases are a control group of 372 and sit
inside the noise floor in every dtype and method.

`f32` and `f64` are untouched by construction: at `L = 16` lanes no full-width
`MR` divides the corpus's runs of 24 except 1m's, and `f64`'s default `MR = 24`
already tiles them. The `f32` gather-path cases part 4 pointed at are therefore
**not** reachable this way — see below.

##### What this prices: the orientation is worth ~2x, and `MR` is a bad way to buy it

Part 4's hope was that shrinking `MR` to 16 would satisfy the orientation rule's
condition 2 on the nine misses and make it pick BA "without any new
discriminant". The flip does happen — the grid confirms `abcijk-e*bc-*` in
`f32`/`c32` switches to BA at `MR = 16`. But:

| | `f32` real, `48x8 -> 16x10` | `c32` planar, `32x6 -> 16x12` |
|---|---|---|
| cases where nothing else changes | **0.70** | 0.96 – 0.99 |
| the six `e*bc` / `e*ac` cases | **1.17 – 1.39** | 1.15 – 1.29 |

Those cases gain 1.17–1.39x *while paying about 30% in kernel shape*, so the
orientation there is worth roughly **2x** on its own. Buying it through a shape
change is a bad trade, and the rule above correctly declines to (guard 3, plus
guard 2 for `f32`). **The conclusion is the opposite of part 4's prediction:
1c does not dissolve the orientation problem, it shows the orientation is the
larger lever and must be attacked directly, at the default `MR`.**

The oracle row above is the other half of the picture: picking the best shape
per case with hindsight scores only 1.03–1.07 anywhere. The shape lever is close
to exhausted. The orientation lever is not.

##### Two effects seen, deliberately not shipped

The grid also shows, at `k <= 24` and with no write-back change at all:

* `c64` 3m gains **1.088** from a *wider* `MR` (16 or 24 against its default 8);
* `c32` 1m gains **1.099** from `24x8` against its Phase 3 default `32x6`.

Both are register blocks that are simply better at small `k` than the ones
chosen at `kc = 256/384`, which is a *shape-by-depth* effect and belongs with
item 2's `MC`/`KC`/`NC` sweep. Neither has a mechanism yet and neither has been
validated on anything but this grid, so neither ships. Note the second one means
the Phase 3 register-block table is depth-conditional, not wrong.

### Part 6 (item 1d): the orientation rule, and its discriminant

Part 5 named this the biggest measured lever and priced it at ~2x on the nine
known misses. A14 recorded the discriminant as unknown. **It is now found**, and
the answer is that the old rule was not wrong so much as *incomplete*: it had a
veto with no fallback, and every one of its misses lived in the case the veto
sent to the default.

#### The grid, again

`scripts/phase4d-orient.sh` forces both arms — `TENSORCONTRACT_ORIENT=none|swap`
— over the whole corpus in every dtype and method, with
`TENSORCONTRACT_ROWBLOCK=base` pinning the shape so the two rules cannot
confound each other (`bench-results/phase4d/or-*`, sibling CPU 0.6–1.1%). That
is both arms of all 392 case-dtype-methods, against the 72 single-family points
that were the entire basis before. `tcbench orient` then dumps the structural
features of each arm, and `scripts/orient-score-rules.py` scores candidates
against ground truth, free and unlimited.

#### The discriminant: the rule has to be antisymmetric, and the old one was not

The `abcijk` families are **exact mirror images** of one another — the same
output structure with the roles of the two directions exchanged. Any correct
rule must therefore be antisymmetric under that exchange. The old rule was
phrased entirely in terms of the *column* direction's properties, so it could
not be, and that is the whole defect. Written symmetrically, `f32` says:

| family | faster arm | its row run | its column run |
|---|---|---|---|
| `-mb` | `AB` | 16 | 24 |
| `e*ac` | `BA` | 16 | 24 |
| `e*bc` | `BA` | 24 | 4096 |
| `-ma` | `AB` | 24 | 256 |

In every one, the faster arm is the one whose **row direction has the shorter
run**. So condition 2 is promoted from a veto to a preference, with a defined
fallback:

1. Prefer an arm whose micro-tile row block lands inside a single run — unit
   stride, run at least `MR`. One arm: take it. Both: stay put.
2. Otherwise put the shorter-run direction in the row role.

Step 1 still dominates, and must: it is why `c64` (`MR = 16` against a run of
24) takes the *opposite* arm from `f32` (`MR = 48`) on identical shapes.

Scored over both arms of all 392: **12 cases better beyond noise, none worse.**
In `f32` it scores 1.128 against never swapping where the hindsight oracle
scores 1.132 — that dtype's orientation question is essentially closed.

| rule | `f32` | `c32` planar | `c32` 1m | `c64` planar |
|---|---|---|---|---|
| Phase 4.1 (`legacy`) | 1.092 | 1.063 | 1.049 | 1.102 |
| **fits, else shorter run** | **1.128** | **1.082** | 1.067 | 1.102 |
| oracle (hindsight) | 1.132 | 1.093 | 1.088 | 1.118 |

#### Two part-4 hypotheses, tested and refuted

Part 4 offered two structural differences as possible discriminants and warned
against adopting either without measuring. Both are now measured, and both are
wrong:

* **"the column direction folds to a longer run"** — as a rule on its own it
  scores 0.976–1.008, i.e. nothing; as a guard on top of the working rule it
  *lowers* every column (1.078 against 1.128 in `f32`).
* **"maximise write-back regularity"** — 0.936 in `f32`. This is the third time
  that quantity has pointed the wrong way (see A16); it explains the write-back
  path and nothing else.

#### The two rules are coupled, and tuning them separately is wrong

The end-to-end A/B of the new rule in the shipped configuration exposed
something the grid could not, because the grid pinned the shape: three `c32` 1m
cases went **0.80**. The 2x2 explains it exactly:

| `abcijk-e*ac-*`, `c32` 1m | `32x6` (default) | `24x8` |
|---|---|---|
| `AB` | 77–79 GF/s | **93 GF/s** |
| `BA` | 75–76 GF/s | — |

At the default shape the new rule prefers `BA` — a 3% error, inside the
per-case noise floor, which is why "0 worse" did not flag it. But the row-block
rule's guard 3 forbade a shape change that flips the orientation, so choosing
`BA` also **blocked** the `24x8` shape that is worth 1.20x. A 3% mistake was
amplified twentyfold by the interaction.

The fix is to remove guard 3, which the orientation fix has made obsolete: the
shapes it existed to veto (`c32` 3m's `e*bc`) are now rejected by guard 2
anyway, because under the corrected orientation their alternates no longer
reach full regularity. Measured directly — 6 newly-firing cases against 12
control cases in the same run:

| | geomean | range |
|---|---|---|
| newly firing (`c32` 1m) | **1.218** | 1.185 – 1.240 |
| control (unchanged) | 1.002 | 0.969 – 1.039 |

**The lesson is methodological and worth more than the numbers.** Both grids
pinned the other lever to isolate their own, which is correct experimental
design and is exactly why neither could see this. A rule validated at a pinned
operating point has only been validated *there*. The end-to-end A/B in the
shipped configuration is not a formality.

#### Result, end to end, in the configuration that ships

`bench-results/phase4d/vf-*`: `legacy, new, legacy'` on one build with
`TENSORCONTRACT_ORIENT=legacy|rule`, everything else at its shipped setting, so
this is the whole of item 1d (orientation rule + the guard-3 removal) against
Phase 4.1. Exclusive machine, sibling CPU 0.8–1.0%.

| | shipped / Phase 4.1 | noise floor | best case |
|---|---|---|---|
| `f32` | **1.028** | 0.996 | 1.410 |
| `f64` | 0.998 | 1.003 | 1.057 |
| `c32` planar | **1.013** | 0.998 | 1.311 |
| `c32` 1m | **1.022** | 1.000 | 1.480 |
| `c32` 3m | 1.010 | 0.998 | 1.222 |
| `c64` planar | 1.004 | 1.005 | 1.042 |
| `c64` 1m | 0.999 | 1.012 | 1.046 |
| `c64` 3m | 1.004 | 0.993 | 1.050 |

Every 32-bit column clears the ±1.3% geomean floor; the 64-bit ones are flat,
which is expected — their `MR` already fitted the corpus's runs, so the old
rule was not in its failing case there. The `abcijk-e*bc-*` family, the largest
of the nine misses, goes **1.41–1.48x**.

Four cases are slower beyond the per-case floor. Three are the same `f32` case
under its three method labels (`abcijk-jkma-mibc`, 0.931, with the bracketing
repeat at 1.013, so it is real); the fourth is `ijkl-mink-jnlm` in `c64` 1m at
0.937 against a repeat of 0.960, i.e. mostly drift. **One genuine per-case
regression**, at 7%, against six cases gained at 1.3–1.5x.

#### What is left

21 of 392 case-dtype-methods still take the slower arm by more than the noise
floor, worth up to 1.36x. They are a different population from the `abcijk`
family this rule was derived on — `abjcd-dkbac-jk`, `ajbdc-ckbad-jk`,
`abjc-cbka-kj`, `aqrs-pa-pqrs`, mostly in 1m — and all are failures to swap. The
oracle gap outside `f32` is 1.088 against 1.067 (`c32` 1m) and 1.118 against
1.102 (`c64` planar), so there is roughly 2% of corpus geomean still on the
table per dtype. `bench-results/phase4d/or-*` holds both arms for all of them,
so the next candidate costs nothing to score.

#### Assumptions added

| # | Assumption | Status |
|---|---|---|
| A19 | The orientation discriminant is a property of `D`'s column direction (the form both previous rules took). | **Refuted.** The corpus families are mirror images, so a correct rule must be antisymmetric under exchanging the two directions; a rule phrased about one of them cannot be. Written symmetrically — row block fits, else the shorter run goes in the row role — it is 12 better / 0 worse over both arms of all 392, and closes the `f32` case to within 0.4% of an oracle. |
| A20 | A rule validated with the other levers pinned is validated. | **Refuted.** The orientation rule was 12/0 with the shape pinned and still cost 20% on three cases in the shipped configuration, because a 3% orientation error blocked a 20% shape change. The levers must be validated jointly, end to end, even when each was isolated correctly for derivation. |

| # | Assumption | Status |
|---|---|---|
| A16 | Choosing `MR` to divide the output's contiguous run converts whole block families to the write-back's fast path, so maximising that fraction is the rule. | **Refuted as a rule, confirmed as a mechanism.** Unguarded it scores 0.936 in `f32`. It needs three guards — shallow `k`, a default that is substantially broken, and no change of orientation — after which it fires on 20 of 392 case-dtype-methods for 1.11–1.12x on those and 1.026 corpus geomean in `c64` planar. |
| A17 | Shrinking `MR` will fix the orientation rule's nine misses by satisfying its condition 2. | **Refuted as a fix, and it prices the problem.** The flip does happen, and those cases gain 1.17–1.39x *while paying ~30% in kernel shape* — so the orientation alone is worth ~2x there and must be bought at the default `MR`. `MR` is the wrong instrument. |
| A18 | The register blocks measured at the operating `kc` are the right ones at every depth. | **Refuted.** At `k <= 24`, `c64` 3m prefers `MR` 16–24 over its default 8 (1.088) and `c32` 1m prefers `24x8` over `32x6` (1.099), with no write-back change involved. The Phase 3 table is depth-conditional. Belongs with item 2. |
| A13 | `MC` is bounded only by keeping the packed `A` block in L2. | **Refuted.** It also bounds the `D` strip a `jr` pass revisits, which is the binding constraint whenever the output's rows are strided. |
| A10 | The Phase 3 write-back defect is the write-back's own L2 traffic, needing a vectorised inner loop. | **Refuted as the primary cause.** It was the *orientation*: the row direction of the matrix view was the strided one, so the innermost loop jumped a row stride per micro-tile row. Choosing the orientation costs nothing and recovers the whole 2x. The vectorised inner loop is real but second-order, and only in single precision. |
| A11 | The row/column orientation is a property of the plan. | **Refuted.** The right choice depends on `MR`, hence on element type and complex method. |
| A12 | Write-back overhead matters equally in both precisions. | **Refuted.** It is per output element and the kernel work it hides behind scales with element size, so removing it is worth ~5x more in `f32`/`c32` (7–15%) than in `f64`/`c64` (2–3%). |
| A14 | The orientation rule's `run >= MR` condition is the right discriminant. | **Refuted, and unresolved.** It is right 63/72 but misses 9 cases by 1.18–1.47x, all 32-bit, including shapes with the same post-swap row structure as the ones it correctly rejects. The true discriminant is unknown; `rm-orient-{none,swap}.csv` hold both arms for all 72 case-dtypes to score candidates against. |
| A15 | A sequential build-to-build A/B is good enough for a few-percent effect. | **Refuted.** It reported `c64` 3m at 0.973 where a paired runtime A/B gives 1.020. Put the switch behind an environment variable and interleave the arms. |

### Part 13: the row-block menu is keyed by position (A35)

Part 11 left A35 in the worst available state: a **measured** 7.8% shape that the
engine could not run, could not A/B, and could not put on a menu, because
`config_at` dispatched on `MR` alone and `32x5` shares its height with the
shipped `32x6`. "Documented and untestable" is worse than "unfixed", and this
closes that half without touching the default.

**The change is a re-keying, not a retune.** `row_blocks` now returns
`(MR, NR)` pairs and `config_at` takes a **menu position**; `Plan::row_block` and
`preferred_row_block` return an index. The rule itself is untouched — it reads
only `MR`, so entries of equal height tie and the earlier one wins, which keeps
the measured default in front by construction. `TENSORCONTRACT_ROWBLOCK=idx=<i>`
now means what its name always implied, and `mr=<n>` resolves to the first entry
of that height, which is the documented limitation rather than a silent surprise.

`(2, 5)` is **appended** to the `f32`/`c32` planar menu, not inserted after the
default: entries 0–2 keep the positions `bench-results/phase4c` swept, so that
grid's `idx=` numbering still means what it meant. Verified end to end —
`idx=0` runs `avx512-planar 32x6`, `idx=3` runs `avx512-planar 32x5`.

The well-formedness test changed with it, and the change is the interesting part:
a repeated `MR` is now **allowed** and a repeated *shape* is not. The old
invariant existed because a duplicate height made the later entry unreachable;
under positional keying it is reachable, and what is unreachable instead is a
duplicate `(MR, NR)`. The test says so, and says why, so the next person does not
restore the stronger version and delete the entry this part added.

**Not fixed, and deliberately.** The default is still `32x6`. A 7.8% *kernel*
margin is not a corpus margin — `NR` moves the `jr` loop count and the packed-`B`
sliver geometry as well as the register block — and this project has been wrong
about exactly that kind of extrapolation before. What changed is that settling it
now costs one sweep arm instead of a rebuild.

#### A provenance defect in the confirmation run, since it is mine

The two confirmation jobs (6753208 `worker5479` rome, 6753209 `worker6150` Ice
Lake) were submitted at commit `d62b9e2` and **run in place** —
`rusty-phase4.sbatch` does `cd "$SLURM_SUBMIT_DIR"` and every arm invokes
`./target/release/tcbench` from the shared checkout. I then wrote the D43 change
in the same working tree and rebuilt that binary at 14:44:49, **while 6753208's
threads stage was running**. Timeline:

| time | event |
|---|---|
| 14:36:49 | 6753208 prep builds at `d62b9e2`; threads stage starts |
| 14:38:58 | its discarded warm-up arm completes |
| **14:44:49** | **`target/release/tcbench` relinked from the D43 tree** |
| 14:45:26 | 6753209 prep finds the binary current, does *not* relink, starts its threads stage on the D43 binary |

So on `worker5479` the arms launched before 14:44:49 ran one binary and those
after ran another, and `worker6150` ran the second one throughout while its
`PROVENANCE` says `d62b9e2`. The project's rule is "do not compile while a
benchmark is in flight"; it is written about CPU contention on a shared
workstation, and it turns out to protect something else as well on a cluster —
the *binary*, through a shared `target/`.

**Checked rather than assumed, and the delta is inert.** D43 is a re-keying plus
one appended menu entry in `cfg_avx512_f32`, which an AVX2 node never consults at
all. On the machine where the menu did grow, the selected shape is unchanged on
**392 of 392 case-dtype-methods**: `worker6016`'s `features.csv` (Ice Lake, built
before any of this) and `worker6150`'s (Ice Lake, built from the D43 tree) agree
on `(mr, nr, arm)` everywhere, and on all 49 `f32`/`c32` planar entries in
particular. The runs therefore stand, and the partition arms — which is what they
were booked for — touch none of this code.

**What to do differently**, and it is cheaper than the rule it replaces: a
cluster job should build into its own `CARGO_TARGET_DIR` under
`bench-results/<node>-<arch>/`, so a submit-directory edit cannot reach a running
arm and the binary is archived beside the numbers it produced. Until that exists,
treat the submit directory as frozen for the duration of a job — including
`cargo test`, which relinks the same artefacts.

*Decisions introduced here: D43 — stated in [Design decisions](#design-decisions).*

## Cache blocking

Phase 4 item 2, closed with a negative result, and the analytical model that was supposed to make it portable.

### Part 7 (item 2): the `MC`/`KC`/`NC` grid

**Status: measured on `worker5040` (Zen2, AVX2), 40 arms in 34.6 min wall against
707 min of arm-time — a 20.4x return on the concurrent placement.** Results
first, then the design that produced them. The `ccqlin038` run this section was
written for never happened; it was stopped after two arms (see "Resume here") and
the grid moved to a cluster node, which changed the machine and the instruction
set. **Nothing here is comparable to a single-core number elsewhere in this file.**

#### The floor, and why this grid supports global conclusions only

Three independent estimates from inside the run:

| estimate | `f64`/`c64` | `f32`/`c32` |
|---|---|---|
| `basem` vs `base` (mid-run repeat) | 1.000–1.004 | 0.997–1.004 |
| `base2` vs `base` (end-run repeat) | 0.991–0.996 | 0.983–0.999 |
| arms that are **bit-for-bit identical** to `base` (`kc256` in 8-byte, `kc384` in 4-byte) | 0.995–0.999 | 0.979–0.992 |
| per-case spread over pinned-`kc` arms at `k <= 64`, where they are the same computation | **6.2%** | 1.9% |

The third row is the sharpest instrument in the grid and it was free: for the
4-byte dtypes `TENSORCONTRACT_KC=384` *is* the default, so that arm must read
1.000 and reads 0.979–0.992. So the geomean floor is **~0.5% in 64-bit and ~2% in
32-bit**, and the per-case floor is **6.2% in `f64`**.

That last number governs how the rest of this section may be read. The grid has 19
arms, so choosing the best per case harvests noise: the "oracle" row below claims
1.077 for `f64` against the best *global* arm's 1.050, and with a 6.2% per-case
floor that 2.7% gap is not distinguishable from picking maxima out of noise. The
demonstration is in the run itself — the top of the per-case wins list includes
`kc64` beating `base` by 1.130 on `ajbdc-ckbad-jk`, a case with `k = 24` where
**every pinned-`kc` arm is bit-for-bit the same computation.** A third of the
corpus is in that position. So: `193 of 588 case-dtype-methods gain more than the
per-case floor` is substantially noise, no per-case blocking rule is supportable
from this run, and every conclusion below is a *global* one.

#### `KC` is first-order, and the direction is deeper — the opposite of what part 9 predicts

Geomean against `base`, per dtype and method (identical across methods in the real
dtypes because the real path is method-independent):

| arm | `f32` | `f64` | `c32` planar / 1m / 3m | `c64` planar / 1m / 3m |
|---|---|---|---|---|
| `kc64` | 0.760 | 0.848 | 0.798 / 0.794 / 0.783 | 0.883 / 0.889 / 0.891 |
| `kc128` | 0.885 | 0.948 | 0.915 / 0.916 / 0.903 | 0.956 / 0.969 / 0.960 |
| `kc256` | 0.960 | *0.995* | 0.982 / 0.989 / 0.981 | *0.997 / 0.999 / 0.999* |
| `kc384` | *0.979* | 1.038 | *0.992 / 0.991 / 0.990* | 1.026 / 1.021 / 1.029 |
| `kc512` | 0.985 | **1.050** | 1.000 / 0.999 / 0.999 | **1.033 / 1.034 / 1.033** |

*Italic* entries are the arms that are bit-for-bit `base`, i.e. the floor.

1. **Shallow `kc` is catastrophic**: `kc64` costs 15–24%. The Phase 2 heuristic is
   nowhere near that bad, but it is on the wrong side of the optimum.
2. **`f64` gains 5.0% at `kc512`** — ten times the 64-bit floor — and `c64` gains
   3.3% in all three methods. `f32` and `c32` are already at their optimum
   (`base` is `kc = 384` there) and go nowhere.
3. **This contradicts part 9's central prediction.** The analytical model wants
   `kc` *smaller* — on this machine 256→128 for `c64` 1m and 384→160 for `c32` 1m —
   to make the `A` sliver an L1 resident, which Phase 3 found the method ranking to
   turn on. Measured, the complex methods prefer `kc` **deeper or unchanged**, and
   nothing prefers it shallower. Whatever the L1-residency argument buys, on Zen2
   it is smaller than what deeper panels buy.
4. `kc512`-`f64c64` is one of the 13 arms that crossed a socket boundary from
   `base`, and crossing reads ~1.1% *low*, so the 1.050 is if anything an
   underestimate.

#### Coupling adds nothing, and `MC` is a plateau

| arm | `f32` | `f64` | `c64` planar |
|---|---|---|---|
| `ck512` (depth + re-derived `mc`/`nc`) | 0.988 | 1.031 | 1.006 |
| `kc512` (depth alone) | 0.985 | **1.050** | **1.033** |
| `mc25` | 0.981 | 1.019 | 0.998 |
| `mc50` | 0.971 | 1.029 | 1.005 |
| `mc200` | 0.986 | 1.015 | 1.013 |
| `mc400` | 0.987 | 1.020 | 1.017 |

Two answers to questions part 7 was built to separate:

* **The coupled arms are no better than the pinned ones and in `f64` are 1.9%
  worse.** Re-deriving `mc`/`nc` at the new depth is not where the effect is —
  panel depth alone is. The pair of axes did its job: the two bounds are
  separable and only one of them matters.
* **`MC` is a wide plateau.** Scaling the derived `mc` from 25% to 400% — a
  sixteenfold range — moves the geomean by at most 3%, all of it within about two
  floors. A13's two-sided bound is presumably real, but **the interval between the
  bounds is wide enough that `MC` is not worth tuning.** That is a clean negative
  result and it retires an item.
* The per-case rules the scorer tries (`couple kc at k <= 32/64`) reach at most
  1.018–1.025 in `f64` — *worse* than simply setting `kc = 512` globally. Combined
  with the 6.2% per-case floor, there is no case for a per-case blocking rule here.

#### The `nc` and `model` arms, re-measured sequentially

`nc25`, `nc400` and `model` change how much memory traffic an arm generates, and
the concurrent placement is **not** neutral for such arms — it was rejected at
−3.2% on the memory-bound half (part 10). A uniform penalty cancels in `arm /
base`; a traffic-dependent one does not. So they were held back and re-measured
sequentially on one core (job 6745978, `worker5175`, warm-up arm discarded),
where `base2` reads **1.000–1.004** — a 0.4% floor, against the placed grid's
0.983–0.996.

| arm | dtype/method | **clean** | placed | delta |
|---|---|---|---|---|
| `nc25` | `f64` | 0.976 | 0.991 | −0.014 |
| | `c64` 3m | 0.940 | 0.948 | −0.008 |
| | `c32` 3m | 0.943 | 0.942 | +0.000 |
| `nc400` | `f64` | 0.995 | 0.994 | +0.002 |
| | `c64` 3m | 1.001 | 0.998 | +0.003 |
| `model` | `f64` | **1.011** | 1.024 | −0.013 |
| | `f32` | **0.975** | 0.984 | −0.009 |
| | `c64` planar / 1m / 3m | **0.972 / 0.967 / 0.956** | 0.977 / 0.975 / 0.959 | ≤0.009 |
| | `c32` planar / 1m / 3m | **0.984 / 0.929 / 0.928** | 0.991 / 0.947 / 0.938 | ≤0.018 |

**The confound was real and small: `|delta| <= 1.8` percentage points, mostly under
1.** No conclusion moves. So holding these arms back was the right procedure — a
−3.2% rejection can swamp a 2% effect and there was no way to know in advance that
it would not — and the answer is that the placed grid was quotable after all. Worth
recording in that order, because the next person will face the same choice with the
same absence of information.

`nc` is confirmed to have nothing in it: shrinking it costs 2–6% (worst in 3m),
enlarging it does nothing, and the `nc`-only oracle is 1.008–1.016.

#### The analytical model loses on the first machine it was supposed to help, and its `kc` is why

`model` is **below `base` in 11 of 12 columns**, and its one gain — `f64` at 1.011 —
is barely twice the 0.4% floor. In the complex methods it costs **1.6% to 7.2%**.
See part 9 for what that does to D23.

The cause is attributable, and attributing it is exactly what part 7's
single-parameter arms were for. The model predicts a shallower `kc` for every
complex method; the pinned-`kc` arms price that depth directly, and the two agree:

| method | model's `kc` (legacy) | `model` arm | pinned-`kc` arm at that depth |
|---|---|---|---|
| `c64` 3m | 128 (256) | 0.959 | `kc128` = **0.960** |
| `c64` 1m | 128 (256) | 0.975 | `kc128` = 0.969 |
| `c64` planar | 192 (256) | 0.977 | between `kc128` 0.956 and `kc256` 0.997 |
| `c32` 1m | 160 (384) | 0.947 | between `kc128` 0.916 and `kc256` 0.989 |
| `c32` 3m | 170 (384) | 0.938 | between `kc128` 0.903 and `kc256` 0.981 |
| `c32` planar | 256 (384) | 0.991 | `kc256` = 0.982 |

**The model arm's damage is its `kc` and nothing else.** Its `mc` rises 4–6x and its
`nc` about 20x, and neither shows up — which is consistent rather than surprising,
since `mc` is a plateau and `nc` has nothing in it. So the model's `mc`/`nc`
reconstruction is harmless and its `kc` equation is the whole problem.

#### Why deeper `kc` wins, from the grid and no extra machine time

The hypothesis first recorded here — that eq. (4)–(6) fails to carry the
per-element real counts through — **was wrong, and is retracted.** `model_kc` takes
its ways ratio from `a_step`/`b_step`, which are `mr * a_reals * real_bytes` and
`nr * b_reals * real_bytes`, so the panel formats *are* accounted for. Worked by
hand for `c64` 1m on Zen2: `a_step` = 128 B/k, `b_step` = 96, `c_ar` =
⌊7·128/224⌋ = 4 ways, `kc` = 4·4096/128 = **128**, which is exactly what the model
reports. The arithmetic is right.

What is wrong is the **objective**. At the measured optimum, `kc >= 512`, that same
`A` micro-panel is 65 KB against a 32 KiB L1 — *twice the whole cache*. So the
engine's best depth is one where the paper's central premise, that a micro-panel
occupies whole L1 ways and the next one evicts the last, does not hold at all. No
repair of eq. (4)–(6) reaches `kc = 512`, because the equation is answering a
different question.

The grid says what the engine is actually buying, using the `kc512` arm's *own*
identical-computation subset as the control. For `k <= 256` there is one `pc` pass
at either depth, so those cases are the same computation and their ratio is the
arm's bias; for `k > 256` the pass count `K/KC` halves:

| `pc` passes at `base` (`kc = 256`) | n | `kc512` / `base` |
|---|---|---|
| 1 — same computation, i.e. the control | 204 | 1.024 |
| 8+ — pass count halves | 90 | **1.082** |

So the effect is **+5.7% net of the arm's own bias, and it lives entirely where
deeper panels reduce the number of times `C` is re-touched.** That quantity is
already on the Phase 4 list as its own item — "fusing the `pc` loop so `C` is
touched once rather than `K/KC` times" — and deeper `kc` is a partial, free version
of that fusion. The analytical model has no term for `C` traffic at all, which is
why it points the wrong way: it optimises a residency this engine does not benefit
from, and ignores the one that dominates.

Two consequences worth acting on rather than admiring:

* **The `kc` recommendation is really a `C`-traffic recommendation.** Raising the
  constant captures part of the win; fusing the `pc` loop should capture more of it
  and make `kc`'s depth much less important. Prefer the fusion.
* **A prediction for the reference-machine grid** (`scripts/ccq-blocking-night.sh`,
  which adds `kc768`/`kc1024`): the `k > 256` population should keep gaining as
  depth rises until the pass count reaches 1, and the `k <= 256` population should
  show only each arm's bias. If instead deeper arms help the `k <= 256` cases too,
  this account is wrong and something about buffer size, not pass count, is doing
  the work.

#### The reference machine reverses it: `kc = 512` was a Zen2 result

**Run on `ccqlin038` overnight** (19 arms, `kc768`/`kc1024` added because Zen2 never
bracketed its own optimum, discarded warm-up arm, `basem`/`base2` both reading
1.010–1.020 — see the contamination note below). Geomean against `base`:

| arm | `f32` | `f64` | `c32` planar / 1m / 3m | `c64` planar / 1m / 3m |
|---|---|---|---|---|
| `kc64` | 0.753 | 0.808 | 0.728 / 0.748 / 0.742 | 0.827 / 0.835 / 0.853 |
| `kc256` | 0.983 | *0.994* | 0.979 / 0.999 / 0.985 | *1.013 / 0.996 / 1.001* |
| `kc384` | *1.001* | 1.003 | *1.002 / 1.009 / 0.997* | 1.014 / 0.974 / 0.998 |
| `kc512` | 1.013 | **0.961** | 1.011 / 1.007 / 1.004 | 0.998 / 0.918 / 0.977 |
| `kc768` | 0.944 | 0.902 | 0.987 / 0.891 / 0.974 | 0.961 / 0.869 / 0.963 |
| `kc1024` | 0.910 | 0.900 | 0.966 / 0.839 / 0.964 | 0.959 / 0.873 / 0.962 |
| `mc200` | 0.913 | 0.929 | 0.974 / 0.888 / 0.984 | 0.977 / 0.919 / 0.978 |
| `mc400` | **0.747** | 0.846 | 0.905 / 0.778 / 0.938 | 0.928 / 0.843 / 0.954 |
| `ck512` | 1.014 | **1.029** | 0.999 / 1.034 / 1.009 | 1.000 / 0.945 / 0.974 |
| `model` | **0.746** | 0.860 | 0.746 / 0.662 / 0.823 | 0.861 / 0.806 / 0.877 |

**Three conclusions from the Zen2 grid are hereby retracted.**

1. **`kc = 512` does not transfer and must not ship.** On Cascade Lake `f64` it is
   **0.961** — 3.9% *worse* than the current default — where on Zen2 it was 1.050.
   The optimum here is `kc = 384`, and deeper is monotonically worse.
2. **`MC` is not a plateau.** `mc400` costs **15–25%** here against Zen2's harmless
   1.020. A13's upper bound — the strip of `D` a `jr` pass revisits, the bound part 9
   flagged as absent from the model — **binds on this machine.**
3. **"Coupling adds nothing" was Zen2-only.** `ck512` is the *best* `f64` arm here at
   1.029.

#### The mechanism, and why the sign flips between machines

My C-traffic account (deeper panels cut how many times `C` is re-touched) predicted
that deeper `kc` would keep helping the `k > 256` population. **It is falsified**, and
by the very test that was pre-registered for it:

| arm | `k <= 256` (same work — the control) | `k > 256` (pass count drops) |
|---|---|---|
| `kc384` | 1.013 | 0.968 |
| `kc512` | 1.011 | **0.862** |
| `kc768` | 1.000 | **0.751** |
| `kc1024` | 1.007 | **0.735** |

Deeper `kc` *hurts* precisely where it reduces `C` traffic, monotonically, up to −26%.
So `C` traffic is real but is not the dominant term, and the prediction recorded in
part 9 was wrong.

What does account for both machines is **whether the packed `A` block still fits L2
at the default depth**. The `kc` arms pin `mc`, so raising `kc` inflates the `A`
footprint (`mc * kc * a_reals * bytes`) in proportion:

* **Cascade Lake**, 1 MiB L2: at `kc = 256`, `mc = 264` gives ~540 KB — resident. At
  `kc = 1024` it is ~2.2 MB, twice the whole L2, so residency is destroyed and the
  loss grows with depth. Consistently, `mc400` at fixed `kc` costs 15–25% for the
  same reason.
* **Zen2**, 512 KiB L2: at `kc = 256`, `mc = 256` already gives ~520 KB — the entire
  L2 with nothing left for the streaming `B` or `C`. There was **no residency to
  lose**, so deepening cost nothing there and the `C`-traffic saving showed up net
  positive. Consistently, `mc400` was harmless on Zen2.

One account, opposite signs, both machines — and it explains the `mc` arms too.

#### What should actually change: the coupled arm, which is stable across machines

Coupling re-derives `mc` at the new depth, holding the `A` footprint constant. That is
exactly the degree of freedom the pinned arms confound, and it is the only arm that
agrees on both machines:

| arm | Cascade Lake `f64` | Zen2 `f64` |
|---|---|---|
| `kc512` (pinned `mc`) | **0.961** | **1.050** |
| `ck512` (coupled) | **1.029** | **1.031** |

The pinned arm swings 9 points between machines; the coupled arm gives ~+3% on both.
**So the recommendation is coupled deepening — raise `kc` and re-derive `mc` against
the same L2 budget — not a bigger `kc` constant.** It is also the change with a
mechanism behind it rather than a fitted number, which is what makes it plausible on
a third machine.

**Scoped offline against both grids, which is where it stops being uniform.** `ck512`
per dtype and method, on each machine:

| dtype / method | Cascade Lake | Zen2 | worse of the two |
|---|---|---|---|
| **`f64` real** | **1.029** | **1.031** | **1.029** |
| `f32` real | 1.014 | 0.988 | 0.988 |
| `c32` planar / 1m / 3m | 0.999 / 1.034 / 1.009 | 1.006 / 1.007 / 1.004 | 0.999 |
| `c64` planar | 1.000 | 1.006 | 1.000 |
| `c64` 1m | **0.945** | 1.006 | **0.945** |
| `c64` 3m | 0.974 | 0.987 | 0.974 |

So the change is **`f64`-only**: +2.9% and +3.1%, the one column where both machines
agree on a gain well outside their floors. `f32` and `c32` are neutral, `c64` planar
is neutral, and **`c64` 1m loses 5.5%** on the reference machine.

The pattern has a mechanism, and it is the same one: coupling *shrinks* `mc` as it
deepens `kc`, and the methods that lose are exactly those whose derived `mc` is
already smallest — 1m derives half of planar's by design (see "The three complex
methods"), so at `kc = 512` its `mc` falls to a handful of `MR` panels and the `ic`
loop stops amortising anything. That predicts the loss ordering observed (`c64` 1m
worst, then `c64` 3m) and suggests the general form of the rule: couple only while
`mc` stays above some small multiple of `MR`. That form is scorable against these two
grids offline, for free, and has not been done.

#### The A/B, and it fails: there is no ~3% blocking win on the reference machine

`scripts/ab.sh bench-results/ab-deepen "TENSORCONTRACT_DEEPEN=on"` on `ccqlin038`,
warm-up arm discarded, `A, B, A'`. **Pre-registered rule: `f64` must beat the floor's
geomean with no dtype regressing beyond it. It does not. Do not ship it.**

| column | B vs A | can the switch touch it? |
|---|---|---|
| `f64` | 0.941 | **yes** |
| `c64` (same arm) | 0.963 | no — control |
| `f32` | 0.941 | no — control |
| `c32` | 0.963 | no — control |
| floor (`A'` vs `A`) | 0.990–0.999 | — |

**The controls are what make this readable.** `TENSORCONTRACT_DEEPEN` is scoped to
`real_bytes == 8 && a_reals == 1 && b_reals == 1`, so three of those four columns
*cannot* move, and they moved 3.7–5.9%. The per-arm occupancy says why: the B arm ran
under roughly double the L3-domain co-tenant load of A and `A'` (198.7% against 85.1%
and 94.3%, summed over the domain). So the raw 0.941 is mostly environment.

Correcting `f64` by its in-arm control gives **≈0.977**, and two controls *within* one
arm differ by 2.2%, so this run resolves about ±2%. Either way the treatment is
neutral-to-negative and nowhere near +2.9%.

**Why the grid said +2.9% and the A/B says ≈0.977.** Both grids' `base` arm was the
contaminated one — `basem`/`base2` read 1.010–1.020 on `ccqlin038`, i.e. `base` was
1–2% slow — which inflates *every* `arm / base` ratio in the grid, `ck512` included.
Corrected, the grid's 1.029 is ≈1.01. The two measurements now agree: coupled
deepening is worth approximately nothing on this machine, and the +3% "agreement
across two machines" was an artefact common to both.

**And the split explains it, which is the useful part.** Coupling changes
`264x256x1536` → `144x512x768` for *all* `f64` cases, so a case with `k <= 256` pays
the `mc` halving and gets no depth in return — `kc` is already clamped to `k`.
Control-corrected:

| population | n | B vs A, corrected |
|---|---|---|
| `k > 256` — deeper panel acts | 45 | **1.049** |
| `k <= 256` — pays `mc`, gains nothing | 102 | **0.947** |

Two thirds of the corpus is in the second row, so the net is negative. A rule
conditioned on `k > kc` would keep the first row — but that is precisely the
depth-adaptive `kc` that part 3 measured and rejected, and the reason is now visible:
the L2 budget makes depth and `mc` a strict trade, so buying depth for a deep-`k` case
costs the `D`-strip width that A13 bounds. On this machine `kc = 384` is the optimum
and the shipped 256 is close to it.

**Conclusion for item 2: the blocking on the reference machine is already near
optimal, and there is no few-percent win available from `MC`/`KC`/`NC`.** That is a
negative result, it closes the item for this machine class, and it is worth more than
the wrong default it prevented. `TENSORCONTRACT_DEEPEN` stays as an off-by-default
switch documenting the experiment rather than a pending change.

**Caveat, stated because it is the honest limit of this run.** The machine was not
quiet: every arm shows 6–14 co-tenants in its L3 domain and the load varied between
arms. The controls permit a correction and the corrected verdict is unambiguous, but a
repeat on a genuinely idle machine would tighten it. Given the corrected estimate would
have to be wrong by 5 points to reverse the decision, that repeat is confirmatory
rather than necessary.

#### The model is worse here, and fails through its other half

`model` costs **14% (`f64`) to 34% (`c32` 1m)** on this machine, against 2–7% on
Zen2. The reason is the mirror image of Zen2's: the model raises `mc` 4–6x, and
`mc400` alone costs 15–25% here. So the model's `kc` sinks it on Zen2 and its `mc`
sinks it on Cascade Lake — **both halves of the derivation are wrong, on different
machines.** A33 stands and is strengthened; `legacy` remains the recommendation.

#### A contamination note, since it is mine

`basem` and `base2` read **1.010–1.020** against `base`, i.e. the first `base` arm is
1–2% slow. That is my own doing: for roughly the first minute of the grid's opening
arm I was still running unpinned analysis and `git` on this machine, which the
per-arm occupancy record caught (`cpu0` at 51% during the warm-up). Everything after
was pinned to the other socket. A persistent 24.6% co-tenant (`herdr`, unrelated to
this work) shared the L3 throughout, which is constant across arms and cancels in
ratios. The `k <= 256` control column above is the cleanest read of the residual:
1.000–1.013. Ratios in this section carry that ~1% bias against `base`; the
conclusions all turn on effects of 3–26%, so none of them moves.

Per-case floor on this machine, from the `k <= 64` identical-arm spread: **~4%**
(against Zen2's 6.2%).

#### Why this shape of experiment

The blocking is the last untouched Phase 2 heuristic: `kc` is 384 for 4-byte
reals and 256 otherwise, and `mc`/`nc` follow from a 512 KiB L2 budget for the
packed `A` block and 3 MiB of L3 for `B`, divided by the packed footprint each
method actually produces. Three earlier results dictate how it has to be
measured rather than leaving it a free choice:

1. **`KC` is first-order, not a tuning knob.** Phase 3 traced the entire
   complex-method ranking to whether the `A` sliver is an L1 resident or an L2
   stream at the operating `kc` — 3m is the *fastest* of the three when panels
   are L1-resident and third when they are not. `kc` is the parameter that
   decides which regime the engine is in.
2. **`MC` is bounded from both sides (A13).** Below by packed-`A` residency in
   L2, above by the strip of `D` that one `jr` pass touches and the next
   revisits. A sweep that varies the two together sees only their sum, and
   part 3's three-case experiment — which found +13% on one case and −18% on
   another and was backed out — is what that confound looks like. So the `kc`
   axis is measured **twice**: `_KC` moves the panel depth with `mc`/`nc`
   pinned, `_KC_COUPLE` re-derives them against the same budgets at the new
   depth. The pair separates the bounds; either arm alone does not.
3. **An absolute `MC` rigs the method comparison.** 1m derives half of planar's
   `mc` because its packed `A` carries four reals per complex element instead of
   two, and that proportionality is exactly what keeps the three-way comparison
   honest. So the `mc`/`nc` axes are swept as **percentages** of the derived
   value (`_MC_PCT`, `_NC_PCT`), not as absolute numbers.

The 19 arms: `base` (three times — first, middle, last, so drift over seven
hours is measured and every treatment is bracketed), `kc` ∈ {64, 128, 256, 384,
512} pinned, the same five coupled, `mc` ∈ {25, 50, 200, 400}% and `nc` ∈ {25,
400}%. Every arm is a runtime switch, so no arm is a rebuild (A15), and the
shipped row-block and orientation rules stay **on** — neither reads the
blocking, so there is no confound to pin away, and leaving them on means the
grid is measured in the configuration that ships (A20).

This is the *whole-grid* pattern from items 1c and 1d, for the third time and
for the same reason: a candidate rule scored against a grid that already exists
costs nothing, so the eventual A/B is spent on a rule that survived all 392
case-dtype-methods rather than on the first plausible one. `_MC_PCT` and
`_KC_COUPLE` exist so that a rule about `mc` can be *expressed* against the
grid at all.

Two properties of the design worth knowing when reading the output:

* A third of the corpus contracts over `k <= 24`, so for those cases every
  pinned-`kc` arm is bit-for-bit the same computation. Their spread across arms
  is a **per-case noise floor measured inside this very run**, alongside the
  three `base` repeats — not one imported from another session.
* The sweep CSV's `notes` column now records the `mc x kc x nc` each row
  actually ran with, so an arm is self-describing and a mislabelled one is
  detectable after the fact. Same reason `MR x NR` went in there in 4.1c.

#### What the grid cannot answer

The truly depth-adaptive rule — `kc = min(k, KC)`, re-derived — is not an arm,
because `Blocking::derive` is per element type and does not know the
contraction's depth; expressing it needs the *driver* to re-derive, which is a
code change and not a switch. `ck64` is the closest the grid gets (for a `k =
24` case it widens `mc` fourfold where true adaptation would widen it tenfold),
so the coupled family brackets that rule's *direction* without measuring it.
That is deliberately the cheap half: if moderate coupled widening is broadly
bad, the adaptive rule is dead and agrees with part 3; if it is broadly good,
the driver-level switch is worth building and measuring.

Also not in the grid: the depth-conditional *register block* (A18). `c64` 3m
prefers a wider `MR` at `k <= 24` than at the operating `kc`, so the shape and
`kc` interact, and the honest version of this experiment varies both. The grid
holds the shape at whatever the shipped row-block rule picks. Sweeping the
product of the two grids is 19 x 5 arms, which is a week; the intended order is
to settle `kc` first and then re-run the row-block grid at the chosen `kc`,
because that is the direction the coupling runs — `kc` decides the regime, and
the shape is chosen inside it.

### Part 9: blocking that transfers

Item 2's grid measures *this* machine, and nothing in it makes the result
transfer, because three constants in `Blocking::derive` are `ccqlin038`'s cache
sizes written down by hand. This removes them (D23): descriptors are probed at
run time — sysfs, then `CPUID`, then built-ins, with the source reported by
`tcbench info` so a number is traceable to a probe rather than to a guess — and
fed to BLIS's analytical model, whose thesis is precisely that this layer needs
no empirical search.

Source: Low, Igual, Smith & Quintana-Ortí, "Analytical Modeling Is Enough for
High-Performance BLIS", ACM TOMS 43(2):12, 2016 (DOI 10.1145/2925987), read as
the actual PDF plus FLAME Working Note #74, not from memory.

**What is the paper's and what is not** — worth stating precisely, because the
two halves have different standing:

* **As written:** `kc` from eq. (4)–(6) — whole L1 ways for the `A` micro-panel
  so the next one evicts the last, one way reserved for the unpacked `C`
  micro-tile, and the 2-way fallback. Equations (1)–(3), which choose `mr`/`nr`,
  are deliberately *not* used: this project measures register blocks (D19).
* **Reconstructed:** §4.3.1 says only that `mc` and `nc` follow "in a similar
  manner" and never writes the inequalities. The reconstruction — reserve the
  ways the streaming operand needs plus one for `C`, give the rest to the
  resident block — is not a guess: it reproduces the paper's own Table III `mc`
  **exactly** for SandyBridge (96), Kaveri (1792) and the TI C6678 (128), two of
  which are asserted as unit tests. The SandyBridge row is BLIS's real shipped
  configuration (`mr=8, nr=4, kc=256, mc=96`) recovered from cache geometry
  alone, which is better evidence than matching a table would be.
* **Does not reproduce:** the Intel Dunnington row (model 1280 against the
  paper's 384). Its `kc` does not follow eq. (4) either — the paper uses 2 ways
  of `A_r` where the formula asks for 3 — so that row appears to carry a
  constraint the paper never states. Recorded rather than fudged.
* **Unvalidated inference:** `nc`, by the same symmetry one level out. The paper
  declines to validate `nc` because three of its four machines have no L3.

#### What it predicts here, which is a computation and not a measurement

`cargo run --release --example blocking_model` prints it; safe to run while a
benchmark is in flight. Post-rounding, i.e. what execution would use:

| dtype | method | legacy `mc/kc/nc` | model `mc/kc/nc` |
|---|---|---|---|
| `f64` | – | 264 / 256 / 1536 | 1104 / **106** / 25040 |
| `f32` | – | 384 / 384 / 2048 | 1824 / **128** / 41472 |
| `c64` | planar | 128 / 256 / 768 | 720 / **80** / 16590 |
| `c64` | 1m | 72 / 256 / 768 | 540 / **53** / 25040 |
| `c32` | 1m | 96 / 384 / 1026 | 1216 / **48** / 55296 |

One prediction dominates: **`kc` falls everywhere**, 2.4x in `f64` and 8x in
`c32` 1m, whose fat "1e" panel is what forces it. That converts the `A` sliver
from an L2 stream into an **L1 resident** — exactly the quantity Phase 3 found
the entire method ranking to turn on, and in the direction that favours 3m. `mc`
rises 4–6x (the model gives `A_c` 14 of 16 L2 ways) and `nc` about 20x, which
for most corpus cases means one `jc` block. The packed-`A` footprint stays
equalised across methods (894–914 KiB against legacy's 512–576), so the
1m-versus-planar fairness invariant holds and 1m's `mc` is still the smaller.

#### Status — superseded by measurement; read the next subsection

The claim as written in this section was: the **portability** problem is solved —
uniformly decent with no hand tuning on any machine whose cache hierarchy it can
see — while the **optimality** question on `ccqlin038` was untouched and
deliberately so. `A13`'s upper bound on `mc` was noted as absent from the model and
`mc` rising 4–6x flagged as the arm that bound would punish. The model was left as
a *second arm of the pending grid* rather than a change, to be judged end to end in
the shipping configuration (A20).

That judgement has now happened, and it went against the model.

#### Measured on the first foreign machine, and it loses

**`worker5040`/`worker5175` (Zen2, AVX2) is exactly the test this model was built
for** — a machine whose cache hierarchy is nothing like the one the legacy
constants were hand-fitted to (512 KiB private L2 against those constants' 512 KiB
`A`-block budget, i.e. they ask for the entire L2; 16 MiB L3 per four cores against
25 MiB per eight). If the model were going to win anywhere it should have won here.

It does not. Measured in the clean single-core regime (part 7), `model` against
`base`:

| | `f32` | `f64` | `c32` planar / 1m / 3m | `c64` planar / 1m / 3m |
|---|---|---|---|---|
| `model` | 0.975 | **1.011** | 0.984 / 0.929 / 0.928 | 0.972 / 0.967 / 0.956 |

**Below `base` in 11 of 12 columns**, by 1.6–7.2% in the complex methods, against a
0.4% floor. Its single gain is `f64` at +1.1%.

**Two things this does and does not mean.** It does *not* refute the paper: BLIS's
own configuration is recovered exactly for SandyBridge, Kaveri and the TI C6678 by
the reconstruction, and those unit tests still pass. What it refutes is the
inference this project drew from it — that an analytically derived blocking is
*therefore* a safe default for **this** engine. The failure is localised: part 7
attributes the whole loss to the model's `kc`, which it predicts shallower for every
complex method (`c64` 1m 256→128, `c32` 1m 384→160), while the grid's pinned-`kc`
arms independently show shallower is worse and deeper is better. The model's `mc`
(up 4–6x) and `nc` (up ~20x) cost nothing measurable, so A13's missing upper bound —
the thing this section flagged as the risk — **was not the problem**. The problem
was the half of the derivation taken straight from the paper.

**The reason is now known, and it is not a bug in the equation** — see part 7,
"Why deeper `kc` wins". The hypothesis first recorded here, that eq. (4)–(6) fails
to carry the per-element real and plane counts through, is **retracted**:
`model_kc` derives its ways ratio from `a_step`/`b_step`, which already include
`a_reals`/`b_reals`, and reproduces its own published prediction exactly when worked
by hand. The equation is correct.

Its *objective* is what does not fit this engine. At the measured optimum,
`kc >= 512`, the `A` micro-panel is 65 KB against a 32 KiB L1 — twice the whole
cache — so the premise that a micro-panel occupies whole L1 ways is simply not where
this engine wants to operate, and **no repair of eq. (4)–(6) can reach that depth.**
What deeper panels actually buy is fewer `pc` passes, hence fewer times `C` is
re-touched: scored against the `kc512` arm's own identical-computation control, the
gain is +8.2% where the pass count halves against +2.4% where it cannot change. The
model has no term for `C` traffic at all.

So the useful repair is not to the model but to the driver: **fuse the `pc` loop**,
which is already a Phase 4 item, and `kc` stops being first-order. That reorders the
"re-derive the model's `kc`" item in "Resume here" — it is no longer the promising
one.

**Recommendation: leave `TENSORCONTRACT_BLOCKMODEL` defaulting to `legacy`.** D23
is unchanged as a decision — the probing, the descriptors and `tcbench info` are
all worth having and are not in question — but the model must not become the
default on this evidence. Flipping it is the user's call and is deliberately not
done in the same commit as this measurement.

| # | Assumption | Status |
|---|---|---|
| A33 | An analytically derived blocking is a safe default on an unseen machine, so it solves portability. | **Refuted on the first unseen machine.** The model loses in 11 of 12 columns on Zen2, by up to 7.2% in the complex methods, against hand-fitted constants belonging to a completely different hierarchy. Localised to its `kc`; its `mc`/`nc` are harmless. "Analytical" bought traceability and cost throughput, and the two were assumed to come together. |

| # | Assumption | Status |
|---|---|---|
| A22 | The model may assume one thread per physical core. | **Assumed, and the project's own rules justify it.** `cores_sharing` divides a level's logical-CPU sharing by the logical CPUs per core, so here a `shared_by = 2` L2 is one core's and a `shared_by = 16` L3 is eight cores'. Oversubscribing hyperthread siblings really would halve a thread's L1/L2, and is not modelled — the measurement rules already treat that configuration as invalid. |
| A23 | Under threading, every cache budget must be divided by the thread count. | **Refuted; it is asymmetric.** The packed `B` panel is *shared*, so `nc`'s L3 budget is a per-socket resource used cooperatively and must **not** be divided. What shrinks `nc` is the per-thread packed `A` blocks, all of which sit in the same L3: the model charges `min(t, cores sharing that L3)` of them. This corrects the limit part 8 recorded as "`NC`'s L3 budget is still charged per core" — in the model arm only. |

## Micro-kernels across instruction sets

One macro body per method across two instruction sets, and what measuring the second one taught about the first.

### Interlude: AVX2 kernels, and where the model can be trusted

Dispatch was a single `is_x86_feature_detected!("avx512f")`, so Zen2/Zen3, most
laptops and the *largest partition of the cluster* ran this engine at **scalar**
speed. That is the widest gap between what this document measures and what a
user would experience, and it blocks any prerelease. It is now closed for
`f32`/`f64` and all three complex methods.

`avx512_kernels!` became `simd_kernels!` with the instruction set as a further
macro parameter (D24); the four bodies are not duplicated, and 1m is still
literally `real::<MV,NR>(2*kc, ..)` on both ISAs. Dispatch resolves through a
`OnceLock`-cached `selected_isa()`, and `TENSORCONTRACT_KERNEL` now takes
`scalar|avx2|avx512|auto` (D25).

**The AVX-512 path is unchanged, and that is checked rather than argued:** its
7816 zmm instructions in `examples/kernel_shapes` are byte-identical to the
pre-merge tree, verified independently at merge time by disassembling both. So
every AVX-512 number in this file stands, and tonight's grid still measures the
engine it was designed against.

#### The register blocks, and the line between model and guess

16 ymm instead of 32 zmm with the same accumulator-plane counts, so the register
bound binds much harder. Defaults, as `MV x NR` and logical `MR x NR`:

| method | `MV x NR` | f64/c64 | f32/c32 | acc | live |
|---|---|---|---|---|---|
| real | `2 x 6` | `8 x 6` | `16 x 6` | 12 | 15 |
| planar | `1 x 5` | `4 x 5` | `8 x 5` | 10 | 14 |
| 1m | `2 x 6` | `4 x 6` | `8 x 6` | 12 | 15 |
| 3m | `1 x 4` | `4 x 4` | `8 x 4` | 12 | 14 |

`real`/`1m` at `2 x 6` is BLIS's `haswell` `dgemm 6x8` / `sgemm 6x16` with the
operand roles swapped. **These are not measured and must not be quoted as if they
were** (D26). The modelled half — register budget and accumulator count — was
confirmed in the disassembly at zero CPU cost, which is the fourth time the uop
model has been right about *cliffs*. The guessed half is which of the fitting
shapes is fastest; notably `planar 1x6` is better on both accumulator count and
bytes/flop and is rejected only on a single spill, and 3m's `1x4` is chosen by
analogy with the measured AVX-512 winner `1x10` (also load-bound, also lowest
bytes/flop). Alternates are on the menu, so the whole-grid pattern applies
unchanged once AVX2 hardware is available.

#### One real result that needed no machine time

`tcbench shapes` under both ISAs, which is exactly what the zero-cost analyses
were built for:

| | AVX-512 | AVX2 |
|---|---|---|
| case-dtype-methods where `Plan::row_block` changes shape | 26 / 392 | 12 / 392 (all `f32`) |
| case-dtype-methods with **no** menu shape that clears the gather path | 81 | **0** |

Every `f64` AVX2 row block on every menu (2, 4, 6, 8, 12) divides 24, and the
corpus rounds every stride-1 extent to a multiple of 24. So Phase 4.1c's
row-block rule is **inert** in `f64` and in all three complex methods on AVX2,
and still has work to do only in `f32`.

#### Correctness, and what is not done

Every kernel family the CPU can execute — default shape plus every menu entry,
real plus three complex methods, both precisions — is checked against the
mathematical definition under a plain `cargo test`, so the AVX2 kernels are
*executed* on this AVX-512 machine rather than merely compiled. A second test
pins what must not drift between ISAs: shapes may differ, pack formats, tile
formats and sliver arithmetic may not, because the driver, packing traversal and
write-back are shared and know nothing about the ISA. The `Ukr` contract carried
a second instruction set unmodified, which discharges the Phase 2 design claim
again.

Not done: **no AVX2 performance number of any kind.** Register blocks, the method
ranking, and the cache budgets on an AVX2 machine's hierarchy are all unmeasured
— run `examples/kernel_shapes` then `scripts/phase3-bench.sh` on a Haswell or Zen
box. The `avx2`-without-`avx512` auto-selection branch has also never run on real
hardware, only its forced equivalent.

| # | Assumption | Status |
|---|---|---|
| A24 | The AVX-512 method ranking (planar > 1m > 3m at the operating `kc`) carries over to AVX2. | **Open, and probably not.** 16 ymm forces `MR` down to 4 complex rows in `f64`, which is the L1-resident regime where Phase 3 measured 3m *fastest*. AVX2's ranking is a separate experiment, not a re-run — and note the analytical model (part 9) pushes in the same direction on AVX-512. |
| A25 | Smaller register blocks are purely a cost. | **Refuted, at zero CPU cost.** They are worse for the kernel and better for the write-back, and on AVX2 the write-back side of the trade is simply won: 81 of 392 case-dtype-methods have no AVX-512 menu shape that clears the gather path, against none on AVX2. |

### Part 11: the AVX2 register blocks, measured

**First result of the `rome` session (job 6745376, `worker5040`).** The AVX2
register blocks were shipped as an explicit guess (D26) — the register budget and
the accumulator count were modelled, but *which of the fitting shapes is fastest*
was not, because the reference machine cannot execute AVX2 competitively. It can
now be answered.

The machine, from the engine's own probes rather than from `sinfo`:

| | |
|---|---|
| host | `worker5040` (Rusty, `gen` partition, `rome`) |
| CPU | AMD EPYC Zen2, 2 x 64 cores, **SMT off** (`sinfo` `2:64:1`, confirmed on the node) |
| ISA | `avx2 fma`, **no AVX-512** |
| cache | 32 KiB L1d 8-way **private**; 512 KiB L2 8-way **private**; 16 MiB L3 16-way **per 4 cores** |
| domains | 32 L3 domains of 4 cores each |

Two consequences before any number: the **`avx2`-without-`avx512` auto-selection
branch has now executed on real hardware** for the first time (it had only ever
been forced on an AVX-512 machine), and the whole 128-core node has **no
hyperthread sibling**, so the contention that forced two Phase 4 retractions is
structurally absent here rather than merely avoided.

#### All four shipped shapes are the measured winners

`examples/kernel_shapes`, packed panels hot, useful GF/s. At the `kc` the driver
actually uses:

| method | shipped `MV x NR` | logical `MR x NR` | GF/s | runner-up | GF/s |
|---|---|---|---|---|---|
| **f64 / c64, `kc = 256`** ||||||
| real | `2 x 6` | 8 x 6 | **53.2** | 8 x 5 | 52.8 |
| planar | `1 x 5` | 4 x 5 | **50.0** | 4 x 4 | 41.9 |
| 1m | `2 x 6` | 4 x 6 | **53.5** | 4 x 5 | 52.9 |
| 3m | `1 x 4` | 4 x 4 | **48.9** | 8 x 2 | 48.1 |
| **f32 / c32, `kc = 384`** ||||||
| real | `2 x 6` | 16 x 6 | **106.6** | 16 x 5 | 105.4 |
| planar | `1 x 5` | 8 x 5 | **104.6** | 8 x 4 | 84.2 |
| 1m | `2 x 6` | 8 x 6 | **107.0** | 8 x 5 | 106.5 |
| 3m | `1 x 4` | 8 x 4 | **112.6** | 16 x 2 | 97.8 |

**Eight for eight.** No change to `cfg_avx2_f64` / `cfg_avx2_f32` is indicated, so
D26's guessed half turns out to have been right — and the *reason* is worth more
than the confirmation: three of the four margins over the runner-up are 0.8–1.6%,
i.e. inside anything this sweep can resolve, while the margin over the *rejected*
shapes is 20–35%. The choice was never between close alternatives; it was between
shapes that fit the register file and shapes that do not.

**The spill has a price and it is now measured.** The interlude recorded that
`planar 1x6` is better on both accumulator count and bytes per flop and was
rejected "only on a single spill". That single spill costs **35% in `f64`** (32.5
against 50.0) and **38% in `f32`** (64.5 against 104.6). Rejecting it was correct
and the margin is not subtle.

#### A correction: the register budget is 15 ymm, not 16

The sweep's `!` marker flags `live > 16` — and the data says that threshold is one
register optimistic. Every shape with `live == 16` collapses just as the flagged
ones do:

| shape | `live` | flagged? | GF/s at operating `kc` | fast sibling |
|---|---|---|---|---|
| `real 12x4` f64 | 16 | no | 21.3 | `real 8x6` (15) 53.2 |
| `1m 6x4` f64 | 16 | no | 20.9 | `1m 4x6` (15) 53.5 |
| `planar 4x6` f64 | 16 | no | 32.5 | `planar 4x5` (14) 50.0 |
| `real 16x3` f64 | 17 | yes | 21.5 | — |
| `3m 8x5` f32 | 17 | yes | 45.9 | `3m 8x4` (14) 112.6 |

So the boundary between "fast" and "collapsed" sits at `live <= 15`, not
`live <= 16`: one ymm is not available to the shape, and every shipped default
happens to sit at 14 or 15. **The uop model was right about the existence and
location of a cliff for the fifth time, and wrong about its threshold by exactly
one register** — which is the kind of error a whole-grid sweep is for, and which
would have been invisible from a sweep of the menu alone. The `!` predicate in
`examples/kernel_shapes` should flag `live > 15`; it is an analysis annotation and
not a code path, so it is deliberately **not** changed while this job holds the
node and shares a `target/` directory with it.

#### A24: the AVX-512 ranking does not carry over, and 3m leads in single precision

A24 guessed "open, and probably not". At the kernel level it is now measurably
*not*:

| | AVX-512 (Phase 3, end to end) | AVX2 kernel, operating `kc` | AVX2 kernel, `kc = 16` |
|---|---|---|---|
| `f64`/`c64` | planar > 1m > 3m | 1m 53.5 ≈ real 53.2 > planar 50.0 > 3m 48.9 | **3m 54.8** > 1m 53.6 > real 51.6 > planar 44.3 |
| `f32`/`c32` | planar > 1m > 3m | **3m 112.6** > 1m 107.0 ≈ real 106.6 > planar 104.6 | 3m 110.7 > 1m 107.2 > real 105.6 > planar 88.5 |

3m is **first in single precision at the depth the driver uses**, where on
AVX-512 it was last by 8% over the corpus. The mechanism is the one Phase 3
identified and needs no revision: 16 ymm forces `MR` down to 4 complex rows in
`f64`, which is the L1-resident regime where 3m's 25% flop saving is not consumed
by extra plane traffic. Note also that 3m is fastest of all four at `kc = 16` in
*both* precisions, which is Phase 3's finding reproduced on a different ISA.

**What this does not settle.** These are *kernel* numbers: packed panels hot, no
packing, no write-back, no cache blocking, which is exactly what the rest of the
engine is. The corpus-level AVX2 ranking is a separate measurement and is not in
this session. Do not quote the table above as a method ranking — it is the
kernel's contribution to one, and Phase 3's whole lesson was that the ranking is
decided by bytes moved per useful flop across the *driver*, not inside the kernel.

| # | Assumption | Status |
|---|---|---|
| A24 | The AVX-512 method ranking (planar > 1m > 3m at the operating `kc`) carries over to AVX2. | **Refuted at the kernel level, in the direction predicted.** 3m is first in `f32`/`c32` at the operating `kc` and first in both precisions at `kc = 16`; planar is last or next-to-last in every AVX2 column. End-to-end confirmation is not in this session. |
| A30 | The register-block sweep's `live <= 16` budget is the real one. | **Refuted, by one register.** Every `live == 16` shape collapses to 21–33 GF/s beside a 48–53 GF/s sibling at `live <= 15`. All eight shipped defaults sit at 14 or 15, so nothing shipped is affected — but the annotation is wrong and would mislead the next person choosing a shape. |

#### The register block is per-microarchitecture, not per-ISA

An Ice Lake session (job 6746817, `worker6016`, `STAGES="shapes threads"`) ran the
same sweep on a *second AVX-512 Intel core*, which had never been done — every
AVX-512 shape in `kernel::x86` was measured on Cascade Lake. Three of the four
`f64`/`c64` defaults are still the winners there. `real` is not, and the reversal is
symmetric:

| shape (`MV x NR`) | acc | live | Cascade Lake @ `kc = 256` | Ice Lake @ `kc = 256` |
|---|---|---|---|---|
| `real 3 x 8` — **shipped** | 24 | 28 | **88.4** | 68.3 |
| `real 3 x 9` | 27 | 31 | 81.3 | **74.6** |

Each machine prefers the other's loser by about 9%: `3x8` wins by 8.7% on Cascade
Lake and loses by 9.2% on Ice Lake. Both machines were swept over the same candidate
set — `3x9` was on Cascade Lake's list and was correctly rejected there — so this is
not a coverage gap, it is a genuine disagreement between two microarchitectures with
the *same ISA and the same 32 registers*.

The mechanism is visible in the depth columns: at `kc = 64` both machines prefer
`3x9` (109.7 against 107.6 on Cascade Lake, 97.7 against 90.9 on Ice Lake), and the
flip happens only at the operating depth. `3x9` carries 27 accumulator registers to
`3x8`'s 24 and `live = 31` against 28 — the A30 budget exactly — so it puts more
pressure on the L1 that also holds the `A` micro-panel. **Ice Lake's L1d is 48 KiB
12-way against Cascade Lake's 32 KiB 8-way**, which is precisely the room `3x9`
needs and does not get on the older core.

**Dispatch selects shapes by ISA only**, so on every Ice Lake machine this engine
currently runs a `real` kernel 9.2% off its own optimum — and Ice Lake is the second
largest CPU partition on this cluster.

Worth noting what makes the fix cheap: the engine *already* probes cache descriptors
for D23, and L1 geometry alone separates these two cores (48 KiB/12-way against
32 KiB/8-way) with no CPUID model table and no new machinery. That is a better
discriminator than a vendor/family list because it names the thing that actually
causes the difference.

Not done, and it should be: `c64`/`c32` were only checked against the shipped
default here, `f32`/`c32` on Ice Lake are unanalysed, and whether a single compromise
shape exists that is within noise of both optima is unknown. None of that needs a new
allocation — `bench-results/worker6016-*/kernel-shapes.txt` and
`bench-results/phase3-kernel-shapes.txt` are both committed and the comparison is
arithmetic.

The full comparison, shipped shape against each machine's own best at the operating
`kc`, from the two committed sweeps:

| section | method | shipped | CL @ shipped | CL best | IL @ shipped | IL best | IL / best |
|---|---|---|---|---|---|---|---|
| f64/c64 | real | `24x8` | **88.4** | `24x8` 88.4 | 68.3 | `24x9` 74.6 | **0.916** |
| f64/c64 | planar | `16x6` | 102.8 | `16x6` 102.8 | 108.4 | `16x6` 108.4 | 1.000 |
| f64/c64 | 1m | `12x8` | 91.1 | `12x8` 91.1 | 69.7 | `12x8` 69.7 | 1.000 |
| f64/c64 | 3m | `8x10` | 87.8 | `8x10` 87.8 | 63.8 | `8x10` 63.8 | 1.000 |
| f32/c32 | real | `48x8` | 179.9 | `48x8` 179.9 | 150.3 | `48x9` 172.6 | **0.871** |
| f32/c32 | planar | `32x6` | 195.7 | **`32x5` 210.9** | 221.8 | `32x6` 221.8 | 1.000 |
| f32/c32 | 1m | `32x6` | 175.4 | `32x6` 175.4 | 136.4 | `24x8` 151.3 | **0.902** |
| f32/c32 | 3m | `16x10` | 194.9 | `16x10` 194.9 | 134.5 | `16x10` 134.5 | 1.000 |

Three of eight shipped shapes are **9–13% off on Ice Lake**, and `real` wants `NR+1`
in *both* precisions — a coherent signature, not scatter.

#### A Phase 3 defect this turned up, on the reference machine

The `CL best` column has an entry that is not the shipped shape: **`planar` `f32`/`c32`
should be `32x5` (210.9) and ships as `32x6` (195.7) — 7.8% off, on `ccqlin038`,
in the default complex method.** This is not an Ice Lake finding; it has been true
since Phase 3.

Phase 3's own sweep output says so in as many words —
`bench-results/phase3-kernel-shapes.txt` contains

```
best per method at kc = 384:
  planar   MV=2 NR=5     210.9 GF/s
```

— while `cfg_avx512_f32` ships `planar = [(2, 6), (3, 4), (1, 12)]`, in which `(2, 5)`
does not appear at all, not even as an alternate. `32x5` also wins at `kc = 64`
(210.3 against 205.4) and loses only at `kc = 16`, so it is not a single-depth fluke.

**Why it was probably chosen wrong is the interesting part.** The doc comment on
`cfg_avx512_f32` bolds `32x6`'s bytes-per-flop (**0.20**, against `32x5`'s 0.231) and
records 195.7 beside it. So the shape appears to have been selected on the
bytes-moved-per-useful-flop model — the project's own central mechanism from Phase 3
— *over* the measured throughput sitting in the same output file. That is the
failure mode this document has now recorded four times in other guises (A16 and part
6: write-back regularity "has now pointed the wrong way three times... it is a *gate*
on a change, never an objective"). D19 says this project measures register blocks;
here it modelled one and the sweep disagreed.

**Not a fix, a finding.** `NR` changes the `jr` loop count and the packed-`B` sliver
geometry, so a 7.8% kernel margin is not a 7.8% corpus margin, and `32x5` is absent
from the menu so no runtime switch can A/B it.

**And it cannot be added to the menu — a structural finding, not an oversight.** The
row-block menu is keyed by `MR`: `cplx_config_at` dispatches on `mr` alone, and
`row_block_menus_are_well_formed` asserts no `MR` appears twice, precisely because a
duplicate would make the later entry unreachable. `(2, 5)` shares `MV = 2` with the
shipped `(2, 6)`, hence the same `MR`, so **the menu has no way to express an
`NR`-only alternate** and `TENSORCONTRACT_ROWBLOCK=idx=` can never reach it.

Two routes out, and choosing is a design decision rather than a fix:

* **Swap the default and measure build-to-build.** Two lines, but it gives up A15's
  preference for runtime switches, and a build-to-build diff already produced one
  wrong sign in Phase 4.
* **Key the menu by position rather than by `MR`**, so entries carry `(MV, NR)` and
  `idx=` selects an index — which is what `idx=` already implies. Touches `configs!`,
  `cplx_config_at`, `row_blocks`, `Plan::row_block` and the well-formedness test. Not
  large, but it changes hot-path config selection and should not ride along with a
  measurement.

So A35 is **documented and untestable end-to-end**, which is a worse state than
merely unfixed and is recorded as such. Worth noting how it was found: entirely inside
committed Phase 3 raw output, months later, at no machine cost. That is the return on
keeping the sweeps.

| # | Assumption | Status |
|---|---|---|
| A34 | Register blocks are a property of the instruction set, so one measurement per ISA is enough (the premise of D19 and of `cfg_avx512_*` / `cfg_avx2_*`). | **Refuted.** Cascade Lake and Ice Lake, same ISA and same 32 registers, disagree by up to 13% on three of eight shapes, and `real` wants `NR+1` in both precisions — Ice Lake's 48 KiB 12-way L1 accommodates an accumulator footprint Cascade Lake's 32 KiB 8-way does not. Shapes are per-microarchitecture; probed L1 geometry separates these two with no CPUID table. |
| A35 | The shipped register blocks are the ones the Phase 3 sweep selected. | **Refuted for one of eight, and now testable.** `planar` `f32`/`c32` ships `32x6` where the sweep's own output names `32x5`, 7.8% faster at the operating `kc`. The bolded bytes-per-flop in the config's doc comment suggests it was chosen by model over measurement, which is the same error A16 records for write-back regularity. It was unreachable until the menu was re-keyed by position (D43); `32x5` is now the last planar `f32` entry and `TENSORCONTRACT_ROWBLOCK=idx=3` selects it end to end. Still unfixed on purpose: verify before changing, because kernel margin ≠ corpus margin. |

## Threading

Phase 4 item 4, end to end: the scheme, the 2-D partition, the domain-aware gate that is now the default, load imbalance, what TBLIS does, and the two experiments that decide the default.

### Part 8 (item 4): threading, implemented and measured

**Status when written: built, correct, and not yet measured.** It shipped **off**
(D22) so nothing else in this file changed and the item 2 grid still measured the
engine it was designed against.

> **Overtaken, 2026-08-04.** It is now measured on seven nodes (five Zen2, two Ice Lake). The measured
> subsections below are the ones to read; the forward-looking parts of this
> report were written before any of them. Threads are **still off by default**,
> now for a measured reason (D46) rather than for want of data, and the partition
> gate *is* on by default (D44).

#### Why this got done before item 2 finished

The workstation was needed for other work, which made a seven-hour
noise-sensitive measurement the wrong thing to be holding the machine for and a
structural implementation the right one — threading's *design and correctness*
cost almost no CPU, while its measurement wants more of the machine than the
blocking grid does (a whole socket, not one core).

There is also an ordering argument, and it is worth recording because it cuts
one way and not the other. Under this parallelisation the packed `A` block stays
per-thread in L2, so item 2's `kc` and `mc` conclusions will carry over
unchanged. The packed `B` panel is **shared**, and `B_BUDGET` currently charges
3 MiB of a 25 MiB L3 as though one core owned the cache — so `NC` becomes a
per-socket question the moment threading is on. **Tonight's `nc` arms are
therefore single-core results and must be labelled as such**; the `kc` and `mc`
arms are not affected.

#### The scheme

`M` is cut once into `p` contiguous strips of whole `MR` panels. Each thread
runs loops 3, 2 and 1 over its own strip with its own packed `A` block; the
packed `B` panel is shared, packed cooperatively (each thread takes a slice of
its `NR` slivers), and bracketed by two barriers per `(jc, pc)` iteration — one
so nobody is still reading the previous panel, one so the new one is complete.
Loops 5 and 4 are identical across threads, which is what makes the barrier
counts agree without tracking them.

Consequences, all deliberate (D21):

* **No reduction anywhere.** Every output element has one owning thread which
  accumulates over the full `K` in the original order.
* **Bitwise identical to serial at any thread count**, therefore. This is the
  strongest available correctness invariant and it is asserted directly, not
  approximated by a tolerance: a strip that dropped a row, one that
  double-counted, or a thread reading `B` across a barrier would each break it,
  and the last of those is exactly the kind of bug a tolerance check waves
  through.
* **Strips are whole `MR` panels**, so each thread's row blocks stay aligned
  with the block scatter. The write-back fast path, the row-block rule and the
  orientation rule are all untouched — threading does not re-open any of
  Phase 4.1.
* **The serial path is unchanged**: at `p == 1` the difference from the
  pre-threading driver is two `Option` checks per `(jc, pc)`, nowhere near the
  hot loops.

#### Correctness

`cargo test --workspace --release` green, and green again under
`TENSORCONTRACT_KERNEL=scalar`, at `TENSORCONTRACT_THREADS` of 1, 2, 4 and 8 —
i.e. the *entire* existing suite, oracle comparisons included, also runs through
the threaded driver, which was free coverage worth taking. Five new tests cover
bitwise agreement with serial across all four dtypes and all three methods, a
blocking that forces hundreds of barrier round-trips, a batch axis (whose
failure mode is a deadlock rather than a wrong answer, so it is worth isolating),
and thread counts far exceeding the panel count.

`Plan::strips` reports how many strips a plan will really use, and the tests
assert it — following the precedent of the orientation tests, which assert the
heuristic actually fires so that a test cannot quietly stop testing anything.
That assertion earned itself immediately: it caught that the strips run along
the **oriented** row direction, so a `1 x 33` output parallelises into two
strips along 33. Skinny-`M` is therefore not automatically serial; skinny in
*both* directions is.

#### What was not known yet, and the three structural limits

*Everything quantitative, at the time.* `scripts/phase4f-threads.sh` measures
1/2/4/8 threads on physical cores of one socket and refuses a cpuset containing
hyperthread siblings, since that measures a different question. **The smoke-test
figure this section originally quoted is removed rather than kept** — one case,
one rep, on a shared machine, superseded by the measured subsections below, and
the kind of number that gets quoted later as though it were a result.

Three limits are structural and were known in advance, listed in the order they
will bite:

1. **Parallelism is capped at `ceil(M / MR)` strips.** A contraction whose
   oriented row direction is short cannot use the cores however much work it
   contains. **Now sized, from committed data and at no CPU cost**
   (`scripts/thread-width.py`): **20 of 392 case-dtype-methods cannot fill 8
   threads from `M` alone** — four distinct cases, `aqrs-pa-pqrs`,
   `ij-ikl-ljk`, `ij-kil-lkj` and `ijk-il-jlk`. They are not the slow ones:
   `ijk-il-jlk` in `c32` runs at 0.86 of the fastest throughput the engine
   reaches in that dtype, so this is real work, not a corner. **All 20 reach
   width 8 from `M x N` together**, which needs no accumulators and no
   reduction, so the fix is a 2-D partition and not a `K` split.
2. **Threads are spawned per `execute` call** via `std::thread::scope`, not
   reused from a pool. Irrelevant at corpus sizes, first-order for small
   repeated contractions — which is exactly the low-arithmetic-intensity
   population Phase 1 identified as the real headroom.
3. **`NC`'s L3 budget is still per-core**, as above.

#### `K`-parallelism is not needed, and the reason is structural

Worth recording so it is not re-opened: parallelising the `pc` loop — the one
axis that would require per-thread accumulators and a reduction — is **not
needed by anything in this corpus**, and the argument generalises beyond it.

Needing it means the output has fewer than `p` micro-tiles in total, i.e. an
output of order 1–3 thousand elements while `K` is large. But arithmetic
intensity at the matrix level is bounded by roughly `2MN / ((M + N) * bytes)`,
so small `M` *and* small `N` cap it from above. Compute-bound requires a large
output, and a large output has plenty of tiles: the two conditions are in
tension, which is why the intersection is empty here rather than merely
unpopulated.

Measured confirmation is in `scripts/thread-width.py` above: every case short of
width 8 in `M` has `ceil(N / NR) >= 8`. And if a user ever does bring such a
shape, the feature is *cheap* precisely where it is needed — the output being
tiny is what makes per-thread accumulator tiles L1-resident and the final
reduction negligible. So this is a demand-driven feature, not a Phase 4 item.

#### Measured, on `worker5040` (Zen2, 64 cores of one socket, AVX2)

**Everything in this subsection is within-session.** Nothing in it is comparable
to the single-core numbers elsewhere in this file. Each arm ran on exactly as many
cores as it had threads, taken in order so that `t4` is one 4-core L3 domain and
`t8` is two; occupancy over all 128 cores was recorded per arm and the 64–127
cores outside the cpuset were idle at **0.0% mean, 1.1% max** throughout, so the
node was genuinely exclusive rather than nominally so.

##### The noise floor, and a defect in the A/B/A' pattern itself

| pair | `t1` vs `t1b` | when both ran |
|---|---|---|
| `f64`/`c64` | **0.953–0.966** | `t1` first thing, `t1b` after ~50 min of 64-core load |
| `f32`/`c32` | **0.981–0.989** | both after the node was already hot |

The first pair's repeat is 3.4–4.7% slower than its opening arm, uniformly across
every case, dtype and method — a direction, not a spread, so it is drift and not
noise. The second pair, measured entirely on a hot node, drifts only 1.1–1.9%.
That difference is the diagnosis: **the opening arm of the session was measured at
single-core boost on a cold package and nothing else was.**

This is a real limitation of the `A, B, A'` pattern this project standardised on
(A15, and the header of `phase4-remeasure.sh`): bracketing detects drift, but when
the drift is a *cold-start* transient the bracket reports it as a floor of ±4.7%
and silently inflates every ratio measured against `A`. On `ccqlin038` the effect
was invisible because a shared workstation is never cold. **Future sessions on a
boost-happy machine need a warm-up arm that is run and discarded**, and until one
exists the honest floor for the `f64`/`c64` column below is asymmetric: its
scaling ratios are understated by roughly 4%.

A second floor, and the one that matters more, comes free from the partition arms:
on the 270 case-dtype-methods where `TENSORCONTRACT_PARTITION=m` and the rule
chose *bit-for-bit the same partition*, two adjacent hot arms differ by

| | geomean | median case | p95 case |
|---|---|---|---|
| `f64`/`c64` | 0.980 | 6% | **31%** |
| `f32`/`c32` | 0.994 | 6% | **26%** |

So at 64 threads the per-case median floor is ±6% — the same as `ccqlin038`'s
single-core figure — but the **tail explodes to ±26–31%**. Per-case claims at high
thread counts are close to worthless here; geomeans over the corpus are good to
about ±2%.

##### Scaling saturates at 16–32 threads and then declines

Per-case geometric mean against the same session's `t1`:

| threads | 2 | 4 | 8 | 16 | 32 | 64 | cores busy at 64 |
|---|---|---|---|---|---|---|---|
| `f64` | 1.90 | 3.41 | 4.72 | **5.76** | 5.68 | 5.37 | 36% |
| `c64` planar | 1.96 | 3.73 | 6.22 | 7.80 | **8.39** | 7.65 | 36% |
| `f32` | 1.90 | 3.51 | 5.53 | 6.36 | **6.56** | 6.01 | 29% |
| `c32` planar | 1.95 | 3.75 | 6.14 | 7.72 | **8.37** | 7.55 | 29% |

Four things follow, and the occupancy column is doing most of the work:

1. **Scaling is clean to 8 threads** (4.7–6.2x) and saturates by 16–32. Beyond the
   peak it *declines*: 64 cores are slower than 32 in every dtype.
2. **The cores are idle, not saturated.** Busy fraction falls 98% → 95% → 88% →
   73% → 49% → 36% as threads double. At `t64` roughly two thirds of the wall
   clock is not compute, which rules out memory bandwidth as the primary limit and
   points at the two structural limits part 8 listed in advance: threads spawned
   per `execute` call rather than pooled, and two barriers per `(jc, pc)`
   iteration now crossing 16 separate L3 domains. **Limit #2 was named before this
   run and is now the leading suspect, priced.**
3. **Complex scales better than real** at every thread count (7.65 against 5.37 at
   64 in 64-bit). That is the twice-arithmetic-intensity argument from Phase 1
   appearing somewhere it was never claimed — complex is less exposed to the
   memory system, so it holds up longer as cores are added. A *new* instance of a
   settled result, not a re-derivation of it.
4. **Per case at 64 threads: median 5.89x, zero cases above 32x, 193 of 294 below
   8x.** The best case reaches 17.8x. Nothing in this corpus scales well at 64
   threads.

##### The prediction, scored

Part 10 predicted, before the run, that the genuinely partition-limited family at
64 threads would be `ij-ikl-ljk` and `ij-kil-lkj`, and that if anything else
flattened it would be contention or a bug rather than the partition. Scored:

* **Right about those two.** They are the two worst cases in the whole corpus at
  `t64` — 2.13x and 2.29x — and the CSV's notes column confirms they ran `27x2`,
  the clamped partition the prediction named.
* **Wrong about the rest, and the miss is the more valuable half.** The next worst
  are the `abcijk` family at 2.43–2.47x running `64x1` — full parallel width, no
  partition limit at all. They are the memory-bound `k = 24` cases, and the reason
  they flatten turns out not to be contention either. See part 8b.

##### Ice Lake: the same code scales twice as well, and it identifies the mechanism

A second session (job 6746817, `worker6016`, Ice Lake-SP, `STAGES="shapes threads"`,
one 48 MiB L3 per 32-core socket) ran the same arms. Its `t1`/`t1b` floor is
**0.4–0.6%**. Against the same session's `t1`:

| threads | Zen2 `f64` | **Ice Lake `f64`** | Zen2 `c64` 3m | **Ice Lake `c64` 3m** |
|---|---|---|---|---|
| 8 | 4.72 | 6.17 | 6.23 | 6.96 |
| 16 | 5.76 | **9.67** | 7.65 | **12.26** |
| 32 | 5.68 | **10.59** | 8.06 | **15.33** |

At 32 threads Ice Lake reaches 48% of linear in `c64` 3m and 33% in `f64`, against
Zen2's 25% and 18% — **roughly twice the parallel efficiency from identical code**,
and still rising at 32 where Zen2 had already turned over.

And the partition sweep, on the `abcijk` family, forced partition against the rule:

| threads | `pn/rule` `f64` | `pn/rule` `c64` | `pm/rule` (control) |
|---|---|---|---|
| 4 | 0.988 | 0.944 | 1.003–1.005 |
| 8 | 0.962 | 0.843 | 1.001–1.003 |
| 16 | 0.983 | 0.806 | 1.002 |
| 32 | **1.014** | 0.934 | 0.987–0.988 |

**The 4.3x that a 1-D `N` split won on Zen2 is simply absent here.** So the effect
is not a property of the partition — it is a property of the *topology*, and both
findings now have one mechanism:

`NC` sizes the shared packed-`B` panel for **an L3**. On Ice Lake there is one L3 per
socket, so "shared" means what the design assumed and the panel is genuinely shared.
On Zen2, 64 threads span **sixteen** separate 16 MiB L3s, so a panel every thread
must read whole is effectively replicated across sixteen caches and re-streamed from
memory. Under `64x1` every thread needs all of `B`, which maximises that traffic;
under `1x64` each thread owns a narrow column group and touches only its own slice,
which fits its local L3. That explains the 4.3x *and* the halved scaling efficiency
with a single cause, which neither a barrier-count nor a false-sharing story does.

**This retargets the fix.** Removing `Plan::partition`'s `panels >= p` early return
would be wrong: on a one-L3-per-socket machine the early return is *correct* and `pn`
buys nothing. What the rule is missing is **how many L3 domains the thread set
spans** — a quantity the engine already probes for D23 (`cores_sharing`). Keep
`pm = p` when the threads share one L3; split `N` so each thread's `B` slice fits its
local L3 when they do not.

**Confirmed** (job 6751550, `worker5137`). The sweep prices the axis choice against
thread count on the `abcijk` family, and on Zen2 four cores share an L3, so thread
count fixes the domain count:

| machine | threads | L3 domains spanned | `pn/rule` `f64` | `pn/rule` `c64` |
|---|---|---|---|---|
| Zen2 | 4 | **1** | 1.031 | 0.981 |
| **Ice Lake** | **32** | **1** | **1.014** | 0.934 |
| Zen2 | 16 | 4 | 1.173 | 1.236 |
| Zen2 | 64 | 16 | **2.004** | 1.242 |

`pm/rule` is 0.959–1.003 across all of it, which is the control (the rule already
chooses `pm = p` for these cases).

**Replicated in all four dtypes on that node**, single-domain point null every time:

| threads | domains | `f64` | `c64` | `f32` | `c32` |
|---|---|---|---|---|---|
| 4 | **1** | 1.031 | 0.981 | 0.988 | 0.966 |
| 16 | 4 | 1.173 | 1.236 | 1.252 | 1.270 |
| 64 | 16 | 2.004 | 1.242 | **2.380** | 1.269 |

`f32` reaches **2.38x**. `worker5137`'s scaling curve also reproduces `worker5040`'s
(`t64`: `f64` 5.54 against 5.37, `c64` 7.69 against 7.65), and its `t1`/`t1b` floor
came back 0.954–0.973 — **A31's cold-start artefact reproducing independently on a
third node**, since `t1` was again the session's opening arm.

**Ice Lake's 32-thread row is what makes this an answer rather than a correlation.**
Within Zen2 alone, threads and domains move together and cannot be separated. But Ice
Lake runs *32* threads across *one* domain and shows nothing (1.014), while Zen2 runs
*4* threads across one domain and also shows nothing (1.031). Two single-domain points
an order of magnitude apart in thread count, both null; three multi-domain points with
a monotone effect. **The driver is the number of L3 domains the thread set spans, not
the thread count.**

So the fix is now specified rather than guessed: `Plan::partition` should split `N`
when the thread set spans more than one L3 domain, and keep `pm = p` when it does not.
The input is `cores_sharing` on the L3, which `CacheHierarchy` already probes for D23.

**Residual gap, nameable and cheap.** The perfect within-machine separation would place
a *fixed* thread count either packed into few domains or spread one-per-domain across
many — 16 threads on 4 whole CCXs against 16 threads on 16 CCXs. `phase4f-threads.sh`
packs by construction (`cpuset_for` takes the first `nt` cores in domain order), so
that arm does not exist yet; it wants a `--spread` cpuset mode, which is a few lines.
The cross-machine control above makes it confirmatory rather than load-bearing.

##### The default: revised by the second machine

Threading is off by default (D22) purely because it was unmeasured. It is now
measured on two machines, and **the first machine alone would have given the wrong
recommendation.**

On Zen2 the honest reading was "off": scaling turned over at 16–32 threads and
reached 5.7–8.1x on 64 cores. On Ice Lake the same code reaches **10.6x (`f64`) and
15.3x (`c64` 3m) on 32 cores and is still climbing**, with a 0.4–0.6% floor. Those
are numbers worth having on by default.

So the recommendation is **conditional, and the condition is the topology**: enable
it where the threads share an L3 (one L3 per socket, i.e. Intel here), and treat
chiplet machines as the limited case until the partition is domain-aware. That is
not a satisfying default to ship — `TENSORCONTRACT_THREADS` still defaults to 1 —
but it is the shape of the answer, and the domain-aware partition is a small change
to `Plan::partition` over a quantity the engine already probes.

Two limits remain and are independent of topology: threads are spawned per
`execute` call rather than pooled (first-order for the small repeated contractions
Phase 1 named as the real headroom), and Zen2's occupancy of 29–36% at 64 threads
says most of the wall clock there is not compute. Turning the default on is the
user's decision per D22 and is deliberately not taken in the same commit as the
measurement.

| # | Assumption | Status |
|---|---|---|
| A21 | Some compute-bound contractions will need `K`-parallelism, hence per-thread accumulators and a reduction. | **Refuted for this corpus, and argued structurally.** 20 of 392 case-dtype-methods cannot fill 8 threads from `M`, all 20 can from `M x N`, and needing `K` requires fewer than `p` micro-tiles in the whole output — which bounds arithmetic intensity at `~2MN/((M+N)*bytes)` and so bounds the case away from compute-bound. Build the 2-D partition; leave `K` unbuilt until a real shape demands it. *Confirmed again in part 10 at every thread count to 128 in both instruction sets, i.e. 16x the width it was argued at.* |
| A31 | `A, B, A'` bracketing is enough to establish a session's noise floor. | **Refuted on a machine with boost headroom.** The session's opening arm ran at single-core boost on a cold package and nothing else did, so its repeat came back 3.4–4.7% slower *uniformly* — reported by the bracket as a ±4.7% floor, while every ratio measured against that opening arm was inflated by the same amount. The second dtype pair, measured entirely hot, drifts 1.1–1.9%. Invisible on `ccqlin038` because a shared workstation is never cold. **Run and discard a warm-up arm.** |

### Part 8b (item 4): the partition becomes 2-D

**Status when written: built, correct, unmeasured.** Measured in part 12; the
partition rule it introduced is now the default (D44), while the *thread count*
still is not (D46).

Part 8 named the 1-D partition's first limit and sized it before the fix was
built: 16 of 392 case-dtype-methods cannot fill 8 threads from `M` alone, over
four cases (`aqrs-pa-pqrs` at 2–6 row panels; `ij-ikl-ljk`, `ij-kil-lkj` and
`ijk-il-jlk` at 5–7), and every one of them is 26–23435 `NR` blocks wide. (That
count is at the *default* register block, which is what `tcbench orient` emits;
the shipped row-block rule changes `MR` on some cases, so a case at the boundary
can move either way.) The output is now cut into a `pm x pn` grid (D27–D29).
Three things are worth carrying forward.

**The two axes are cut in different places, and that is the design.** Row strips
are cut once, outside everything; column groups are cut inside loop 5, per `NC`
block. So every thread walks the same `(h, jc, pc)` sequence and a thread's cell
changes only how much work happens inside an iteration, never how many
iterations there are. That preserves both properties the 1-D scheme rested on —
one L3-sized shared `B` panel, and barrier counts that agree because loops 5 and
4 are identical across threads — while making the `N` axis available. A thread
whose column group is empty in a tail `jc` block still takes both of that
block's barriers and *then* skips loop 3.

**A negative result on indexing, and the most valuable thing this produced.**
Indexing the shared `B` panel by *absolute sliver*, which is what the 1-D scheme
did, is wrong the moment `N` is split: a group's sliver range **moves** between
`jc` blocks (a tail block has fewer slivers to divide) and its stride changes
with `pc_len`, so one group's next block lands on another group's current one —
and by design nothing orders them. Each group now owns a fixed slice, strided by
the worst-case sliver size so groups on different `pc` blocks cannot overlap
either. It surfaced only under test-level oversubscription and never in a
standalone repro; the barrier-stress case, which has one sliver per `jc` block
and therefore ranges that never move, passed throughout. **Worth remembering as
the shape of bug this parallelisation produces: not a missing barrier, but a
buffer whose ownership map is not constant.**

**The one modelled decision, priced without the machine.** `PACK_WEIGHT = 8` was
replayed offline over all 392 case-dtype-methods at five thread counts, and every
weight in `[4, 64]` gives an identical partition everywhere; the only two cases
that move at all choose between candidates 1.4% apart in modelled cost. At 8
threads the rule spills most of the narrow cases onto the column axis and
deliberately leaves `ij-ikl-ljk` / `ij-kil-lkj` 1-D **on 7 of 8 threads by
choice** — 37 column blocks split eight ways is five per thread against a whole
packed `A` block each. A hand estimate puts 1-D ahead by ~2.6%, which is an
estimate, which is why `TENSORCONTRACT_PARTITION` exists and why
`scripts/phase4f-threads.sh` now runs `=m` and `=n` arms at 8 threads beside the
scaling curve. Verified behaviourally at merge: `aqrs-pa-pqrs` runs `2x4` in
`f64` and `1x8` in `c64`, a wide case stays `8x1`, and the sweep CSV records
`t<threads>/<pm>x<pn>` so the partition is recoverable from the data.

#### Measured, on `worker5040` at 64 threads: the 2-D split is right and the rule's first line is wrong

Two arms, both against the rule at the same thread count, on the cases where they
actually change the partition (everything else is a control group and is what the
±2% geomean / ±6% median-case floor in part 8 was derived from).

**`TENSORCONTRACT_PARTITION=m` — forcing the old 1-D `M` split — loses everywhere
the 2-D rule fires:**

| dtype | cases changed | geomean |
|---|---|---|
| `f64` | 12 | **0.607** |
| `f32` | 12 | **0.607** |
| `c64` | 6 | **0.297** |
| `c32` | 12 | **0.521** |

So the 2-D extension is worth **1.6x to 3.4x** on exactly the population it was
built for, in all four dtypes, far outside any floor. D27–D29 are vindicated, and
part 8b's hand estimate that leaving two cases 1-D on 7 of 8 threads was worth
~2.6% turns out to have been the wrong thing to worry about — at 64 threads the
axis choice is worth factors, not percent.

**`TENSORCONTRACT_PARTITION=n` — forcing 1-D `N` — beats the rule in the real
dtypes and loses badly in the complex ones:**

| dtype | cases changed | geomean | max |
|---|---|---|---|
| `f64` | 141 | **1.055** | 4.32 |
| `f32` | 144 | **1.100** | 3.14 |
| `c64` | 147 | 0.678 | 1.82 |
| `c32` | 141 | 0.748 | 2.34 |

And the real-dtype gain is not spread thinly — it is concentrated on the
memory-bound `abcijk` family (18 cases, `k = 24`), which gains a **geomean 2.25–2.26x
and up to 4.32x**:

```
4.32x  abcijk-jkmc-miab  f64   rule t64/64x1 -> t64/1x64
4.23x  abcijk-ijmc-mkab  f64   rule t64/64x1 -> t64/1x64
```

**The diagnosis is sharper than A28 guessed, and it is not `PACK_WEIGHT`.** These
cases have 768 row panels against 64 threads, so `Plan::partition` never reaches
the cost model at all — it returns from the first line:

```rust
if panels >= p { return (p, 1); }
```

That early return encodes "if `M` alone can fill the threads, `M` is the right
axis". At 8 threads on a socket-wide L3 that was true and cheap. At 64 threads on
Zen2's sixteen 4-core L3 domains it costs **up to 4.3x on 18 of 49 corpus cases**,
and it is the same population — the `k = 24` memory-bound family — that part 8's
scaling curve showed flattening at 2.43x for no visible reason. The two findings
are one finding.

**`PACK_WEIGHT`'s direction is right, which is why the fix is not "always consult
the cost model".** The complex methods *lose* 25–32% from the same `1x64` switch,
and that is exactly what the packing term predicts: a complex packed `A` carries
two to four reals per element, so duplicating it across 64 column groups costs more
than the real path pays. So the rule is right for complex and wrong for real on
this family, and a corrected rule needs the early return removed *and* a term that
captures whatever makes the real path prefer `N` here — which this run has not
identified. Two candidates, both cheap to test with the switch that already exists:
write-back locality (under `64x1` sixty-four threads write interleaved row strips
of `D`, and false sharing across separate L3s is expensive on this topology, where
under `1x64` each thread owns a contiguous column group), and the cost of the two
barriers per `(jc, pc)` iteration now spanning 16 L3 domains. **Do not change the
rule until one of those is measured** — this is the third time in Phase 4 that a
plausible mechanism for a real effect turned out to be the wrong one.

| # | Assumption | Status |
|---|---|---|
| A36 | The 4.3x that a 1-D `N` partition wins on the memory-bound family is a property of the partition rule. | **Refuted by a second topology.** Absent on Ice Lake (`pn/rule` 1.014 at 32 threads against Zen2's 2.26x geomean), where one L3 serves the whole socket. It is a property of how many L3 domains the shared packed-`B` panel is spread across, which also explains Ice Lake scaling twice as well on identical code. So the fix is a domain-aware partition, **not** removing the `panels >= p` early return — on a one-L3-per-socket machine that early return is correct. |
| A28 | The 2-D partition rule and `PACK_WEIGHT` behave at node scale as they do at 8 threads. | **Refuted, but not where predicted.** `PACK_WEIGHT` is fine — it correctly keeps the complex methods off the `N` axis, which costs them 25–32% when forced. What fails is the `panels >= p` early return that precedes it: worth up to 4.3x on the 18 memory-bound `abcijk` cases in the real dtypes at 64 threads. The 2-D machinery itself is vindicated at 1.6–3.4x on the cases it fires on. |

Correctness is unchanged in kind and stronger in coverage: bitwise identity with
serial at every thread count *and* every partition, plus a `Split` expectation on
every threaded case pinning whether it is meant to be 1-D, 2-D, a genuine grid or
clamped to serial — written in units of `MR`/`NR`, since a shape two row panels
deep in `f32` is fourteen with the portable kernels. Green in release and debug,
and across all ten combinations of the three features merged today.

### Part 12: the partition becomes L3-domain-aware

**Status: built, measured, and it met its pre-registered prediction. Still off by
default — that is D22 and it is the user's call.** The prediction was made from
another node's committed grid before this one was booked; the result is at the end
of this section.

Part 8b diagnosed `Plan::partition`'s first line as costing up to 4.3x at 64
threads, and part 8 then refuted the obvious fix: removing the early return is
wrong, because on a machine with one L3 per socket the early return is *correct*
(Ice Lake, 32 threads on one domain, `pn/rule` 1.014). A36 is settled on three
nodes and in all four dtypes — the driver is the number of L3 domains the thread
set spans, not the thread count. This part turns that into code.

#### The change

`CacheHierarchy::l3_domains(threads)` is new and is three lines: `threads`
divided by the cores that share an L3, which the probe already reports for D23.
`tcbench info` prints it as a function of thread count, because a session that
says "16 domains" without saying at what width has said nothing.

`Plan::partition`'s early return is now **gated, not removed** (D41). In the
regime where the row axis alone fills the threads, the column axis takes them
instead when all three of these hold:

| condition | why it is there |
|---|---|
| `domains > 1` | The mechanism. `NC` sizes the shared packed-`B` panel for *an* L3; under `p x 1` every thread reads the whole panel, so it is replicated across every domain the thread set covers. Two single-domain points an order of magnitude apart in thread count (Zen2 t4, Ice Lake t32) are both null. |
| `blocks >= p` | The column axis must be able to fill the threads by itself, or the swap buys locality by giving up cores. The corpus's narrow half loses 2–5x on a forced `1 x p` for exactly this reason, and it is the largest population in the `n` arm. |
| `k <= 64` (`BANDWIDTH_BOUND_K`) | The penalty being dodged is **bandwidth**, so it cannot dominate a compute-bound case. `k` is this corpus's knob for that, and the guard confines the rule to the population the evidence covers — see the negative result below. |

It is a **binary** choice between `p x 1` and `1 x min(p, blocks)`, which are the
only two arms any session has measured, and it is reached only from the early
return: the cost model below it is untouched, so the narrow-`M` regime the 2-D
partition was built for behaves exactly as it did.

`TENSORCONTRACT_PARTITION` gains `domain` (and `legacy`, the default, spelled
out). It is a *rule*, not a pin, so unlike `m`/`n`/`<pm>x<pn>` the test suite
keeps asserting the rule's invariants under it — which is how "bitwise identical
to serial at every thread count **and every partition**" now covers the new path
as well. The correctness suite exercises the swapped branch on 60-odd shapes and
is green in every combination of `{default, domain} x {avx512, scalar} x
{1, 4, forced-16 domains}`.

#### The prediction, made before the node is booked

`scripts/partition-score-rule.py` replays the gate over a committed grid and
scores it against the arms that grid already measured. Both arms exist for every
case: the gate only ever selects the rule's own partition or the forced `1 x p`
one, and part 8b measured both on the whole corpus. So this is not a model of the
answer, it *is* the answer, up to the session-to-session drift of running it
again.

On `worker5137` (Zen2, 4 cores per 16 MiB L3), whole corpus at 64 threads — 16
domains — the gate moves **144 of 392 case-dtype-methods**, all of them the
`abcijk` family, and nothing else:

| dtype | n | predicted arm/rule |
|---|---|---|
| `f64` | 18 | **1.947** |
| `f32` | 18 | **2.647** |
| `c64` | 54 | 1.172 |
| `c32` | 54 | 1.266 |
| all changed | 144 | **1.423** |

which is **1.138 over the whole 392** with the unchanged cases counted at 1.000,
or per dtype 1.277 / 1.430 / 1.060 / 1.091. At 16 threads (4 domains) it moves
the same population for 1.242. At **4 threads — one domain — it changes nothing**,
and on **Ice Lake at any thread count up to the socket it changes nothing at
all**: the gate is bit-identical to the shipped rule there, which makes that
machine a free null control rather than a second experiment.

Put beside part 8's scaling curve, that is the difference between 5.97x and
~7.6x on 64 Zen2 cores in `f64`, and it closes most of the gap between Zen2's
parallel efficiency and Ice Lake's on identical code — which is what the
mechanism predicted it should do.

#### The negative result: a traffic term cannot be calibrated, and here is why not

The first design was the principled one — add a cross-domain `B`-replication term
to the cost model and let it decide — and it does not work. Four attempts, each
scored offline against the grid:

* A term in `min(pm, domains)` large enough to move the `k = 24` family moves
  **263 of 392 case-dtype-methods onto intermediate grids** like `4 x 16` and
  `8 x 8` that no session has ever run. The term saturates at `pm >= domains`, so
  its optimum is an intermediate, not `1 x p`.
* Restricted to the two extremes, the switch-over weight is **33.6** for `f64`
  `abcijk`, **56.7** for `f64` `ijkl`, **50.6** for `c64` `abcijk` and **56.2**
  for `c64` `ijkl`. The measurement wants the first three to switch and the
  fourth not to. There is no weight that does that: the two `ijkl` thresholds are
  1% apart and want opposite answers.
* Making the term count *reals* rather than elements — so 1m's four-reals-per-`A`
  and 3m's three-plane traffic enter, and the `D == 1` behaviour is provably
  unchanged for every method — moves the pair to 66.0 and 67.6 and leaves them
  still inverted.
* Adding the `A`-side traffic term that makes the model symmetric (`min(pn, d)`
  domains read each row strip) ranks `ijkl` as *more* inclined to the column axis
  than `abcijk`, which is backwards by a factor of two in the measurement.

So the mechanism that explains the domain count does **not** explain which wide
families want the swap, and four plausible accounts of it are refuted. The
threshold that does separate them is `k`, by a factor of 40 — and a threshold on
a bandwidth-boundness proxy is at least the right *kind* of quantity for a
bandwidth effect. It is stated as a confinement of scope, not as a mechanism.

**What it costs to be that conservative, stated rather than hidden.** The same
session says a forced column split is worth 1.168–1.801 on the wide
*compute-bound* families (`ijkl-*`, `ij-ik-kj`) in the real dtypes, and 0.715–0.876
in the complex ones. Flipping those too is a **0.916 geomean over the 96
case-dtype-methods involved** — a loss, because the complex populations are three
times the real ones — so leaving them alone is the right call on this evidence
even before the one-machine, one-session caveat. The `n` arm already prices that
variant in every future run; nothing new is needed to revisit it.

#### A methodological result: per-case ratios at 64 threads are not readable

The `m` arm at 64 threads produces the **same partition as the rule** on 366 of
392 case-dtype-methods, so those ratios are pure repeats and measure this
session's per-case precision directly. They come back geomean 0.997 with **p10
0.885 and p90 1.107, and tails to 0.80 and 1.55** — on the `abcijk` family in
`f64`, 0.801 to 1.545.

Per-family geomeans are good to 1–2% and everything quoted above is one. **No
per-case number at 64 threads on this corpus means anything**, which is why the
gate was not fitted to one, and why `partition-score-rule.py` prints that control
first and labels the per-case column as spread rather than as a result. This is
the same lesson as A31 and A32 in a third form: the floor is a property of the
measurement's shape, and it has to be re-derived in-session every time.

#### The threading default: recommended on, and not flipped here

D22 is the user's call and this commit does not take it. **The recommendation is
no longer conditional on topology, because the gate removed the condition** — the
numbers below are post-measurement:

* **Turn the gate on by default first.** It is a *strictly smaller* decision than
  turning threads on: measured a no-op on 392 of 392 cases on a one-L3-per-socket
  machine, and 1.133 corpus geomean (1.433 on the 144 it moves) at 64 threads on
  a chiplet one. Nothing that runs single-threaded can observe it at all, since
  `l3_domains(1) == 1`. There is no machine on which it is known to cost anything.
* **Then turn threads on.** With the gate, Zen2 goes from 5.5–5.8x to
  **7.5–7.9x** in the real dtypes at 64 cores and stops declining past 16
  threads; Ice Lake was already 10.5x (`f64`) and 14.2x (`c64` 3m) at 32. The
  topology-conditional recommendation in part 8 existed because the chiplet case
  was bad and unfixed. It is fixed.

The older reasoning, kept because the shape of the argument still holds:

* Where one L3 serves the thread set, threading was always worth having: Ice Lake
  reaches 10.5x (`f64`) and 14.2x (`c64` 3m) on 32 cores and is still climbing.
* On chiplet machines the shortfall was the partition, and the gate closes most of
  it: 5.81 -> 7.54 (`f64`) and 5.51 -> 7.88 (`f32`) at 64 Zen2 cores. What remains
  is occupancy (29–36%), i.e. the per-call thread spawn and the small-contraction
  work Phase 1 named — not the partition, which is now measured rather than
  suspected.
* **Two limits are unchanged and independent of topology**: threads are spawned
  per `execute` call rather than pooled, and `NC`'s L3 budget is charged per core
  in the legacy blocking.

Shipping the gate on by default is a separate decision from shipping *threads* on
by default, and it should be taken first, because it is a no-op on every
single-domain machine and a large win on the others.

#### The confirmation run, and what would falsify it

```bash
PARTITION_SWEEP=1 STAGES=threads sbatch -C rome scripts/rusty-phase4.sbatch
```

~1.5–2 h. `phase4f-threads.sh` now runs a `domain` arm beside `m` and `n`, over
the whole corpus at the top thread count and across the thread-count sweep on the
memory-bound family, and prints the prediction from this section next to the
result. Pre-registered, so it cannot be reinterpreted afterwards:

1. **`dom/rule` on the `abcijk` family must reproduce `n/rule`** to within the
   session's own floor, at 64 threads and at 16, in all four dtypes. It is the
   same partition, so anything else is contention.
2. **`dom/rule` on every other case must be 1.000** — the same code path runs.
   This is the column the change cannot touch, and it is the check that caught a
   contaminated A/B in part 7. If it moves, nothing else in the run counts.
3. **At 4 threads (one domain) `dom` must equal `rule` exactly.**
4. `t1` against `t1b` should now come back near 1.000 rather than 0.95–0.97,
   because a discarded warm-up arm runs first (A31). If it does not, the warm-up
   is at the wrong thread count and the floor is still a cold-start artefact.

A run on `-C icelake` is a stronger null than a repeat here: the gate must be
bit-identical to the rule at every thread count up to the socket.

#### A36's residual gap, now closable in the same run

Every separation of domain count from thread count so far has been
*cross-machine*: Zen2 at 4 threads on one domain and Ice Lake at 32 threads on
one domain are both null, an order of magnitude apart in width. That is a strong
control, but the within-machine version was missing because `phase4f-threads.sh`
packs its cpusets by construction, which ties the domain count to the thread
count and makes the two inseparable inside one node.

`SPREAD=1` supplies it, in about eight family-restricted arms per dtype pair. It
holds the thread count fixed and varies only the packing: on this node 16 threads
on cpus `0-15` (**4 domains**) against 16 threads on `0,4,8,…,60` (**16
domains**), verified against the committed `topology.json`. Both placements run
`rule`, `m`, `n` and `domain`, so the comparison is two `n/rule` ratios at one
thread count with the control beside them.

The engine cannot infer the spread placement — `l3_domains` assumes compact (A37)
and would under-count — so those arms declare the true span through
`TENSORCONTRACT_L3_DOMAINS`, which is what that override was added for. The
prediction, if A36's mechanism is right: `n/rule` near 1.17 packed (4 domains,
the measured `t16` figure) and near 2.0 spread (16 domains, the measured `t64`
figure), at the *same* 16 threads. If instead both read ~1.17, the driver is
thread count after all and D41 is wrong in a way three machines have not shown.

```bash
SPREAD=1 PARTITION_SWEEP=1 STAGES=threads sbatch -C rome scripts/rusty-phase4.sbatch
```

#### Result: all four checks pass, and the prediction was right to 1%

Run 2026-08-04, jobs 6753208 (`worker5479`, rome, 64 cores of one socket, 4 per
16 MiB L3) and 6753209 (`worker6150`, Ice Lake-SP, 2x32 cores, **one 48 MiB L3
per socket**, so one domain over the 32-core cpuset). 108 arms, **none flagged
non-exclusive**, 141 min and 70 min.

**Check 2 first, because it decides whether the rest counts.** The 248
case-dtype-methods the gate cannot touch read **1.0116** — a uniform drift
between the rule arm and the `dom` arm an hour later, which is A32's
drift-with-separation and not contamination. Every number below is quoted raw and
deflated by it.

**Zen2, 64 threads, 16 domains**, against the prediction made from `worker5137`'s
grid before this node existed:

| | predicted | measured | drift-corrected |
|---|---|---|---|
| case-dtype-methods moved | 144 of 392 | **144 of 392** | — |
| geomean on those | 1.423 | 1.449 | **1.433** |
| whole corpus | 1.138 | 1.146 | **1.133** |

Per dtype, measured: `f64` 2.207, `f32` 2.379, `c64` 1.217, `c32` 1.272. Check 1
(`dom` against the `n` arm, which is the same partition) is 1.019 raw and **1.007
corrected** — inside per-case noise, as it must be.

**Check 3, Ice Lake: the gate moved 0 of 392.** Not approximately a no-op — the
identical partition on every case at every thread count up to the socket. A
one-L3-per-socket machine is untouched by this change, which was the whole reason
for gating the early return rather than deleting it.

**The domain sweep is monotone in domains and flat in threads**, `abcijk` family,
gate against rule:

| threads | domains | `f64` | `c64` | `f32` | `c32` |
|---|---|---|---|---|---|
| 2 | **1** | 0.999 | 1.001 | 1.000 | 1.001 |
| 4 | **1** | 0.999 | 1.003 | 1.001 | 1.001 |
| 8 | 2 | 1.145 | 1.025 | 1.129 | 1.043 |
| 16 | 4 | 1.181 | 1.179 | 1.221 | 1.272 |
| 32 | 8 | 1.645 | 1.267 | 1.806 | 1.226 |
| 64 | 16 | 2.100 | 1.237 | 2.300 | 1.272 |

**And the spread arms close A36's residual gap within one machine.** Sixteen
threads throughout, same node, same cases, only the packing different:

| placement | domains | `f64` | `c64` | `f32` | `c32` |
|---|---|---|---|---|---|
| packed | 4 | 1.189 | 1.218 | 1.219 | 1.293 |
| **spread** | **16** | **2.736** | **1.487** | **3.045** | **1.744** |

Predicted in this section: "near 1.17 packed and near 2.0 spread". Packed landed
at 1.19. Spread landed *higher* than the `t64` figure, and the mechanism says it
should — 16 threads on 16 domains is one thread per 16 MiB L3, so the replicated
`B` panel under `16 x 1` is at its worst against the private cache each thread
could have had instead. Every prior separation of domain count from thread count
was cross-machine; this one holds the thread count fixed on one node.

**What it does to scaling**, corpus geomean over each session's own `t1`:

| machine | dtype | t8 | t16 | t32 | t64 | **t_top + gate** |
|---|---|---|---|---|---|---|
| Zen2 | `f64` | 4.62 | 5.77 | 5.76 | 5.81 | **7.54** |
| Zen2 | `f32` | 5.21 | 6.14 | 5.98 | 5.51 | **7.88** |
| Zen2 | `c64` | 6.11 | 7.83 | 8.15 | 7.65 | **8.24** |
| Zen2 | `c32` | 6.01 | 7.63 | 8.04 | 7.28 | **8.06** |
| Ice Lake | `f64` | 6.13 | 9.61 | **10.54** | — | 10.40 |
| Ice Lake | `c64` | 6.82 | 11.58 | **14.23** | — | 14.31 |

The gate turns Zen2's *decline* past 16 threads into a rise, and the Ice Lake
column moves only by session drift, since the partitions there were identical.

**Check 4, the warm-up arm: it works, but not completely, and the residual names
its own mechanism.** `t1` against `t1b`:

| node | `f64c64` | `f32c32` |
|---|---|---|
| Ice Lake | 0.997–0.999 | 0.998 |
| Zen2 | **0.972–0.977** | 1.001–1.003 |

Three of four brackets are now inside 0.3%, against 0.954–0.973 on three nodes
without a warm-up. The exception is the *first* dtype pair of the *longer*
session, and its second pair is clean — so the residual tracks **position in the
session**, not package temperature, which makes it A32 rather than A31 and means
a hotter warm-up would not fix it. Derive the floor from a bracket adjacent to the
arms being compared, which is exactly what check 2 does above.

*Decisions introduced here: D41, D42, D44 — stated in [Design decisions](#design-decisions).*

| # | Assumption | Status |
|---|---|---|
| A36 | The 4.3x that a 1-D `N` partition wins on the memory-bound family is a property of the partition rule. | **Refuted; now implemented as a property of the topology.** See part 8. D41 is the code. |
| A37 | `l3_domains(p)` may assume **compact placement**: `p` threads occupy `p` consecutive physical cores, filling one L3 domain before starting the next. | **Assumed, and true of every measurement in this file.** `phase4f-threads.sh` builds its cpusets that way by construction (`cpuset_for` takes the first `nt` cores in domain order) and a whole-node run leaves nothing to spread over. A *scattered* placement spans more domains than this counts, and the error is in the safe direction: it under-counts, so the rule falls back to the behaviour every committed number was measured with. `TENSORCONTRACT_L3_DOMAINS` overrides it, and `SPREAD=1` in `phase4f-threads.sh` is the arm that uses it. |
| A38 | A cross-domain traffic term in the partition cost model can be calibrated to select the column axis where measurement wants it. | **Refuted, four ways** — see the negative result above. The weights that separate the two `abcijk`/`ijkl` pairs are 1% apart and want opposite answers, in element units and in real units alike, and the symmetric two-sided model ranks the two families backwards. Do not re-derive it without a new mechanism. |
| A39 | Per-case ratios at 64 threads are readable at the ±6% the reference machine reports. | **Refuted on this corpus and this width.** Repeats of an identical partition run p10 0.885 / p90 1.107 with tails to 0.80–1.55. Only per-family geomeans (1–2%) are quotable at this thread count, and the control that shows it comes out of the same data at no cost. *Confirmed on two further nodes: Ice Lake at 32 threads reads p10 0.915 / p90 1.108 over 392 identical-partition repeats, so this is a property of threaded measurement here and not of one machine.* |
| A40 | A discarded warm-up arm at the top thread count removes the opening-arm artefact (the fix A31 asked for). | **Partly.** Three of four `t1`/`t1b` brackets came back inside 0.3%, against 0.954–0.973 on three nodes without one. The exception is the first dtype pair of the longer session, at 0.972–0.977, whose *second* pair is clean — so the residual tracks position in the session rather than package temperature, and a hotter or longer warm-up would not remove it. Keep the warm-up; do not treat it as a floor. Derive the floor from a bracket adjacent to the arms being compared, which is what the "columns the change cannot touch" control does for free. |

### Part 14: load imbalance, and a claim of ours that expired

Not a measurement — a correction, an argument, and the experiment that settles
it. Prompted by the observation that a block-scatter contraction has a load
imbalance a dense GEMM does not: some blocks sit on the regular fast path and
some on the gather path, so **equal block counts are not equal work**.

#### The correction: "the corpus is fully regular" is false, and it steered things

A4 recorded, in Phase 1, that TCCG rounds stride-1 extents to multiples of 24
"which divides every register block in use", so `regA = 1.00` everywhere. That
was true *then* and is still the right reading of the TBLIS comparison, which was
measured at TBLIS's register blocks. **It stopped being true of this engine in
Phase 3**, when `f32`/`c32` shipped `MR` of 16, 32 and 48 — none of which divides
24. Nobody re-checked the sentence, and it has been repeated in `CLAUDE.md`,
`README.md` and a memory ever since.

Measured, on the arm the orientation rule actually picks, over all 392 corpus
case-dtype-methods (`bench-results/worker6150-icelake/features.csv`).

**These fractions are `--size 64` figures and the size is part of the claim** — the
extents scale with it, so which runs straddle an `MR` boundary does too. At
`tcbench orient`'s default size the same AVX-512 engine reads 40.6 / 26.0 / 34.9,
and `bench-results/phase4d/features.csv` (generated at that default) reads
36.7 / 26.0 / 33.2. Reproduce the row below with
`./target/release/tcbench orient --size 64 --csv -` on any AVX-512 machine; it
costs no CPU and touches no data.

| quantity | fraction below 1.0 | values it takes |
|---|---|---|
| `reg_a` | **42.9%** | 0.0, 0.667, 0.889, 0.963 |
| `reg_b` | **38.3%** | down to 0.501 |
| `wb` (output row blocks off the gather path) | **35.7%** | 0.0, 0.667, 0.889, 0.963 |

The quantised values *are* the imbalance: `reg_a = 0.667` means one row block in
three is on the gather path and two are not, **within a single contraction**. 123
of 392 are genuinely mixed rather than uniformly good or uniformly bad.

The mechanism is worth stating because it is not what the old sentence implies:
irregularity here is produced by the interaction of the layout with `MR`, not by
the tensors. A 24-run straddles a 16-wide block. That is the same effect the
row-block and orientation rules were built for; we simply never connected it to
*threading*.

#### What the corpus still cannot show, and the experiment for it

Its straddling is **periodic** — every third block — so a static strip of many
blocks self-averages, and at 12 panels per strip (768 panels over 64 threads) the
averaging is imperfect but real. The case where a static partition is genuinely
bound by its unluckiest strip is *aperiodic* irregularity: ragged extents, real
user tensors, block-sparse blocks. The corpus cannot produce that, which is what
`--stress ragged` exists for (it subtracts 1 from every extent, destroying the
multiple-of-24 property).

So the question is decided by a comparison the harness can already make, and
`RAGGED=1` in `phase4f-threads.sh` now makes it: the scaling curve under ragged
against the scaling curve unperturbed, **each against its own `t1`**, because
ragged changes the shapes and only within-mode ratios mean anything. If ragged
scales visibly worse, static partitioning is losing to imbalance.

#### The design ladder, if it is

1. **Cost-weighted static cut.** Strips are currently cut by panel *count*. The
   per-block cost is already known before any thread starts — `a_m_bs` and
   `d_m_bs` carry the `IRREGULAR` sentinel — so cut for equal estimated cost
   instead. No scheduler, no new memory, no semantics changed.
2. **Dynamic `ic` claiming within a `(jc, pc)` iteration**, and this is the one
   worth knowing about: **it preserves bitwise identity for free.** The barriers
   already serialise `pc` iterations across a column group — a thread entering
   `pc+1` waits at "nobody is still reading the previous panel" — so `D`
   accumulates in `pc` order no matter *which* thread does which row block, and
   each block is touched once per `pc`. Full load balancing at no cost to the
   invariant. What it does cost is locality: today a thread owns the same rows for
   the whole run, so its `A` and `D` regions stay put. Mitigation is the standard
   one, claim your home range first and steal only when starved.
3. **A dependency-counted task DAG.** Replace the collective rendezvous with nodes
   — `pack-B(jc,pc)` → many `compute(ic,jc,pc)` → `pack-B(jc,pc+1)` gated on a
   completion counter. Same memory, same packing work, no rendezvous, automatic
   balancing. Node counts are coarse enough for the overhead to vanish: `abcijk`
   at 64 threads is roughly `24 x 4 x 1` nodes.

**`pc` fusion is *not* the general enabler this file previously implied.** Fusing
makes the packed `A` block `mc x K`; at `K = 3744` and `mc = 256` that is 7.7 MB
in `f64`, far past L2. It is viable only for small `K` — the memory-bound family,
where it is most wanted — but the road to a general task graph is 3, not fusion.

#### What is *not* a reason to do any of this

Barriers. They cost skew per iteration; imbalance costs *total*, because a
systematically unlucky strip does more work every iteration and the barrier
merely exposes it. Removing barriers does not fix imbalance, and the engine
already has a barrier-free configuration — `pm == 1`, which the domain-aware gate
now selects on chiplet machines for exactly the memory-bound family. Earlier
drafts of this argument had that emphasis backwards.

*Decisions introduced here: D45 — stated in [Design decisions](#design-decisions).*

| # | Assumption | Status |
|---|---|---|
| A41 | Blocks of a block-scatter contraction are equal-cost, so partitioning by block count balances the load. | **False, and measured false on the corpus we call regular** — 123 of 392 case-dtype-methods have mixed regularity within one contraction, at ratios of 1/3, 1/9 or 1/27 of blocks on the gather path, which the orientation work priced at roughly 2x. Whether it *costs* at thread scale is what `RAGGED=1` answers; on the unperturbed corpus the straddling is periodic and a static cut partly self-averages. |

### Part 15: what TBLIS actually does about threading

Prior art for part 14, read from source rather than assumed. Both trees are on
disk — `baselines/tblis-1.3.0` (git `c4f81e0`) and `baselines/tblis-2.0` (git
`555320c`, which vendors BLIS `358e689c`). Nothing was built. The four
load-bearing claims below were spot-checked directly against the source after the
survey, and all four hold.

#### The three findings that matter to us

**1. `PC` is never parallelised, in either version, and for our reason.** 1.3.0
gives the `kc` level exactly one gang — `communicator comm_kc = comm_nc.gang(
TCI_EVENLY, 1);` (`src/nodes/gemm.hpp:127`) — and 2.0 inherits BLIS's
`bli_rntm_set_pc_ways_only(1, rntm); // Disable pc_nt values.`
(`frame/base/bli_rntm.c:292`). Matthews states the reason in the BSMTC paper
itself (arXiv:1607.00291 §7.2): the `p` loop *"is not parallelized since this
would require additional synchronization and/or temporary buffers with
reduction."* **This is independent confirmation of A21** from a mature engine
that had every opportunity to do otherwise.

**2. TBLIS does nothing whatsoever about block-scatter load imbalance.** It
computes the same irregularity sentinel we do — `fill_block_stride` sets a
block's stride to `0` when it is not uniform (1.3.0
`src/matrix/block_scatter_matrix.hpp:232-247`; 2.0
`tblis/frame/base/block_scatter.cxx:219-236`) — and forks on it at pack and
write-back time, so its per-block cost varies as much as ours. **No partitioning
site consults it.** Every split is by count or length: `nodes/partm.hpp:40` (the
single decision point for all five levels in 1.3.0),
`gemm_ker_bsmtc.cxx:158-159` and `packm_blk_bsmtc.cxx:75,78,115` in 2.0. A
case-insensitive grep for `cost|imbalance|load.?balan` over `frame/3t` and
`frame/3m` returns **zero hits**. The decisive detail is ordering: in
`gemm_ker_bsmtc.cxx` the block-stride arrays are *built* by a count-split (180-183),
barriered (213), and then read by threads whose ranges were already fixed —
**the regularity information is produced after the scheduling decision and never
fed back into it.**

BLIS's one cost-weighted partitioner, `bli_thread_range_weighted_sub`, is gated
on triangular structure (`bli_thread_range.c:790-824`) and cannot fire on the
dense objects TBLIS builds. Its TLB partitioner disclaims this exact class in its
own comment: *"It makes no effort, however, to account for differences in
threads' workload that is attributable to differences in the number of edge-case
microtiles"* (`bli_thread_range_tlb.c:650-657`).

And the author measured the consequence and shipped anyway (§9.2): BSMTC's weak
scalability is *"only slightly less"* than BLIS's ~90%, **"possibly due to load
imbalance stemming from edge cases which must use the full scatter vector."**
Diagnosed, quantified as small on TCCG, never addressed.

**So a cost-aware partition is open ground rather than catching up.** Two
qualifiers, and the first corrects the survey itself: it is *not* true that our
corpus is fully regular and would measure nothing (see part 14 — 42.9% of
case-dtype-methods have `reg_a < 1.0` at our register blocks), so the unperturbed
corpus can show the periodic case and `--stress ragged` is for the aperiodic one.
Second, TBLIS's real escape hatch is upstream of scheduling: `sort_by_stride`
(`frame/3t/dense/mult.cxx:380-382`) reorders tensor *dimensions* to manufacture
regular blocks so the imbalance rarely bites. **We already have that** —
`plan.rs:373-376` orders `M`/`N`/`H` by increasing `|stride|` in `D` and `K` by
`|stride|` in `A`, then folds. So we are not missing their mitigation; we are
both left with the residue it does not remove.

**3. TBLIS has a dynamic, atomic-claim task scheduler — and uses it only for the
sparse formats.** `comm.do_tasks_deferred`, backed by a CAS on a slot
(`external/tci/src/tci/task_set.c:35-45`), appears in `3t/indexed/mult.cxx`,
`3t/indexed_dpd/mult.cxx` and the `1t/indexed*` family, and **never** in
`3t/dense`, `3m` or `1m` — verified by listing every call site. That is precisely
the split part 14 proposes and the user's instinct predicted: **static
partitioning for one dense contraction, dynamic fork-join for block-sparse.** The
most experienced implementation of this algorithm made the same division.

#### The structural divergence, which is a real trade

| | this engine | TBLIS |
|---|---|---|
| decomposition | flat `pm x pn` grid, one cell per thread | five nested gangs, `jc → kc(x1) → ic → jr → ir` |
| packed `A` | per thread, `pm * pn` copies | **shared** within an `ic`/`ir` gang, packed cooperatively; footprint scales with `jc_nt` |
| packed `B` | one panel per column group | one panel |
| barriers | 2 per `(jc, pc)` | ~3 per `PC` iteration + ~4 per `IC` iteration (2.0) |
| per-loop thread counts | `pm`/`pn` from the cost model | 1.3.0 honours `BLIS_{JC,IC,JR,IR}_NT`; **2.0 wipes them** — `bli_rntm_set_num_threads` clears the ways so only the total is honoured (`bli_rntm.c:257-266`) |

They buy memory and cache sharing with barrier depth; we buy barrier shallowness
with duplicated packing. The one thing they can do that our grid structurally
cannot is put several threads on a *single* `MC x NC` block (the `jr`/`ir` ways),
which keeps threads cache-coherent by construction — where our answer to the same
pressure is the topology-aware gate of D41. Worth knowing when the `jr`/`ir` axis
is next considered.

Two smaller notes: TBLIS also spawns its parallel region per call with no pool of
its own (`external/tci/src/tci/parallel.c:24` — `#pragma omp parallel
num_threads(nthread)`), so it has no advantage over our `std::thread::scope`
there; and `tblis_tensor_mult` takes a **communicator** as its first argument
rather than a thread count, which is a more composable API than
`Plan::with_threads` if this engine is ever nested inside a caller's parallel
region. Both baselines here are configured `TCI_USE_OPENMP_THREADS 1` with
`TCI_USE_SPIN_BARRIER 1`.

| # | Assumption | Status |
|---|---|---|
| A42 | Block-scatter load imbalance is a solved problem in mature implementations, so a cost-aware partition would be reinventing something. | **False.** TBLIS computes the same per-block regularity sentinel and feeds it to no scheduling decision in either version, and its author published the diagnosis without a fix. Its mitigation is upstream — reorder dimensions to manufacture regular blocks — and we already do the equivalent. |

### Part 16: the two experiments that decide the threading default

Jobs 6754849 (`worker5139`) and 6755009 (`worker5178`), both rome, both rc=0,
2026-08-04. One settles the default; the other kills a design direction.

#### Small contractions: the default cannot be a fixed thread count

Speedup against `t1` **at the same size**, so the fixed cost is inside the
measurement — `timed()` takes the best of `reps` with one `execute` per rep, and
threads are spawned per call.

| MiB | dtype | t2 | t4 | t8 | t16 | t32 | t64 | serial ms |
|---|---|---|---|---|---|---|---|---|
| 0.25 | `f64` | 1.56 | **2.12** | 1.50 | 0.85 | 0.41 | **0.14** | 0.32 |
| 0.25 | `f32` | 1.40 | **1.73** | 1.10 | 0.56 | 0.26 | **0.10** | 0.22 |
| 1 | `f64` | 1.81 | 3.24 | **4.16** | 3.60 | 2.25 | 1.10 | 4.55 |
| 1 | `f32` | 1.76 | 2.91 | **3.14** | 2.43 | 1.42 | 0.66 | 2.37 |
| 4 | `f64` | 1.86 | 3.68 | 5.38 | **5.70** | 4.18 | 2.61 | 6.31 |
| 16 | `f64` | 1.85 | 3.20 | 4.44 | 4.99 | **6.19** | 5.03 | 26.07 |
| 64 | `f64` | 1.90 | 3.41 | 4.91 | 6.10 | 6.28 | **7.50** | — |

**At 64 threads, threading costs up to 10x below a megabyte.** `f32` at 0.25 MiB
runs at **0.10** — a 0.22 ms contraction takes 2.2 ms. And the best thread count
walks monotonically with size: 4, 8, 16, 32, 64 at 0.25, 1, 4, 16, 64 MiB. **A
fixed default is wrong at every size but one.**

The fixed cost reads straight off the smallest rung, where parallel work is
negligible: `t64` on a 0.32 ms job takes 2.29 ms, so the overhead is ~2.3 ms, or
**~36 µs per thread**; at `t8` the same arithmetic gives ~21 µs. That is
per-call `std::thread` spawn, and it is what makes the sub-MiB region
catastrophic rather than merely inefficient.

**Two regimes, and only one is dangerous.** Below ~1 MiB the spawn cost dominates
and threading is a *loss*. Above it, the curve saturates early (best at `t8`–`t32`
rather than `t64`) — that is a bandwidth ceiling, not spawn, and it is benign: you
get 4x instead of 6x. A guard only has to fix the first.

> **Corrected by part 18, and this paragraph is the thing it corrects.** The
> saturation above ~1 MiB is *substantially fixed cost, not bandwidth*: pooling the
> threads makes `t64` beat unpooled best-at-any-width in all eight size x dtype rows,
> and at 16 MiB `f64` the peak goes 6.20 at `t32` to 13.19. What survives is that the
> optimum still sits below 64 threads, so a fixed default is still wrong at most
> sizes — D46 stands as a decision, but not for this reason. And "a guard only has to
> fix the first" turned out to be the sentence that mattered: the guard was scored
> only at `requested = 64` and loses 39% at `requested = 4` (A54, D52).

**So `TENSORCONTRACT_THREADS` must not simply be flipped on.** Two ways forward,
and they compose:

* **Amortisation guard** — cap the thread count so the spawn cost stays a bounded
  fraction of the estimated serial work. Fitting the 0.25 MiB rung, "spawn ≤ 50%
  of serial" gives `p <= 5`, and `t4` is indeed the measured optimum there. Small,
  needs only a flop estimate, and converts a 10x regression into "no worse than
  serial". It does *not* predict the saturation above 1 MiB, which is a different
  mechanism — do not fit it to that.
* **Pool the threads**, which removes the per-call cost outright and shrinks the
  dangerous region rather than steering around it. *Measured in part 18: it removes
  more than spawn, and the "dangerous region" is wider than this part thought.*

And it sharpens the batched-API argument: for many small contractions the right
parallel axis is *the batch*, giving one spawn per batch instead of one per
contraction, which is exactly the cost measured here.

#### Load imbalance from block-scatter irregularity: refuted, and cleanly

`--stress ragged` subtracts 1 from every extent, destroying TCCG's multiple-of-24
property. It did what it was meant to, and the perturbation is characterised
rather than assumed:

| | case-dtype-methods with `reg_a < 1.0` | values |
|---|---|---|
| unperturbed | **11.7%** | 0.667, 0.889, 0.963 — the periodic quantisation |
| ragged | **87.2%** | 0.348, 0.349, 0.696, 0.697, 0.789, 0.793, 0.87, 0.901 — aperiodic |

(The unperturbed figure is lower here than the 42.9% in part 14 because this is an
AVX2 node: `MR = 8` for `f64` *does* divide 24. The contrast is therefore cleaner,
not weaker.)

Scaling, each mode against **its own** `t1`:

| dtype | width | unperturbed | ragged | ragged/unperturbed |
|---|---|---|---|---|
| `f64` | 8 | 4.93 | 5.23 | 1.062 |
| `c64` | 8 | 5.96 | 6.13 | 1.028 |
| `f64` | 64 | 7.64 | 7.92 | 1.037 |
| `c64` | 64 | 8.21 | 8.60 | 1.048 |
| `f32` | 64 | 8.04 | 8.36 | 1.040 |
| `c32` | 64 | 7.91 | 8.89 | 1.124 |

**No degradation — 0.976 to 1.124, and if anything ragged scales slightly
better.** Seven and a half times as many heterogeneous cases, at aperiodic
fractions down to 35% of blocks irregular, and parallel efficiency does not move.

Why, most likely: at 64 threads occupancy is 32–39%, so threads are waiting on
memory rather than on each other, and compute-side imbalance hides inside that.
Static strips are also 12+ panels wide at 64 threads, which averages a good deal
even when the pattern is aperiodic.

**So the design ladder in part 14 is not worth building** — cost-weighted cuts and
dynamic `ic` claiming are solutions to a problem that does not measurably exist
here. Two limits on that conclusion, both real: it says nothing about **block-sparse
with wildly varying block sizes**, which is a far larger imbalance of a different
kind, and it was measured where the machine is bandwidth-bound. If a future kernel
or a smaller working set makes the engine compute-bound at scale, re-run it.

*Decisions introduced here: D46, D47 — stated in [Design decisions](#design-decisions).*

| # | Assumption | Status |
|---|---|---|
| A41 | Blocks of a block-scatter contraction are equal-cost, so partitioning by block count balances the load. | **False in the premise, true in the consequence.** Blocks genuinely differ in cost — part 14 measured the heterogeneity and it is large. But making it 7.5x more prevalent and aperiodic changes parallel efficiency by less than the noise floor, so the imbalance does not *cost* at these thread counts on this machine class. Stated this way because the premise may matter again where the consequence does not follow — a compute-bound machine, or block-sparse. |
| A43 | Per-call thread spawn is a second-order cost, worth fixing after the partition. | **Refuted at small sizes.** ~20–36 µs per thread, which is the entire story below 1 MiB: a 0.22 ms `f32` contraction takes 2.2 ms on 64 threads. It is first-order for exactly the workload Phase 1 identified as the headroom. *Part 18 removed it and found it worth up to 11.6x — and also that it is not the whole cost, nor confined to below 1 MiB (A53).* |

### Part 17: three answers to the spawn cost, and the batch axis

**Status: built, correct, and measured only offline.** All three are **off or
inert by default** and each is a run-time switch, so the next session can A/B them
against the shipped behaviour in one process-restart rather than as a diff between
two builds (A15). Written after part 16 established that per-call spawn is
first-order below a megabyte (A43) and that no *fixed* thread count can be the
default (D46).

Part 16 left one problem with three distinct answers, and they compose rather than
compete:

| answer | what it does to the ~20–36 µs/thread | switch |
|---|---|---|
| amortisation guard | **steers around it** — caps the thread count so it stays a bounded fraction of the work | `TENSORCONTRACT_AMORTISE=on` |
| thread pool | **removes it** — parked workers instead of `std::thread::scope`. Measured in part 18, and it removes *more* than spawn: the cost is size-dependent, so the wording "removes the ~20–36 µs/thread" throughout this part is an undercount (A53) | `TENSORCONTRACT_POOL=on` |
| batched API | **removes the count** — one spawn set per batch instead of one per contraction | `tensorcontract::batch` |

#### The amortisation guard, calibrated against a committed grid

The rule is one line: each thread must be given at least `MIN_FMAS_PER_THREAD`
real FMAs, so `p <= work_fmas / C` clamped to `[1, requested]`. Written that way
it holds the spawn **fraction** constant rather than the thread count — with spawn
`S` and machine rate `R` FMAs/s, spawn over serial time is
`(W/C) * S / (W/R) = S * R / C`, independent of the problem. At `S = 30 µs`,
`R ≈ 20 G` FMAs/s and `C = 3e6` that fraction is **0.2**.

`work_fmas` is `m * n * k` weighted by real FMAs per logical MAC — 1 real, 4
planar/1m, **3 for 3m**, which is exactly the flop saving that method exists for.
Unweighted `m * n * k` misjudges the two domains in opposite directions, because at
a fixed byte size a complex contraction has half the elements and four times the
arithmetic per element.

`C` was **not guessed**: `bench-results/worker5139-zen2/phase4g/` already measured
all 588 case-dtype-method points at 1, 2, 4, 8, 16, 32 and 64 threads at four
sizes, so candidate constants score exactly, offline, for free — the same method
that settled the row-block rule, the orientation rule and the partition gate.
**Reproduce every row below with `scripts/amortise-score-rule.py
bench-results/worker5139-zen2/phase4g`**, which also cross-checks the constant it
scores against `plan.rs` so the script and the engine cannot drift apart in silence.
(The table was first produced by an ad-hoc script and was therefore un-rederivable
from the repository for a day, which is the defect the rest of this file exists to
prevent.)
Corpus geometric mean against serial with 64 threads requested throughout, and the
count of points left slower than serial:

| nominal size | unguarded | `C = 2e6` | **`C = 3e6`** | `C = 4e6` | `C = 8e6` | per-case oracle |
|---|---|---|---|---|---|---|
| 0.25 MiB | **0.20**, 565 slower, worst **0.022** | 1.33, 0 | **1.66, 0** | 1.18, 0 | 1.04, 0 | 2.40 |
| 1 MiB | 1.32, 229 slower, worst 0.077 | 3.98, 0 | **3.50, 0** | 3.31, 0 | 2.50, 0 | 4.94 |
| 4 MiB | 2.91, 57 slower, worst 0.363 | 5.38, 0 | **4.60, 0** | 5.11, 0 | 3.81, 0 | 6.37 |
| 16 MiB | 5.48, 3 slower, worst 0.800 | 5.50, **3** | **5.67, 0** | 5.76, 0 | 6.65, 0 | 7.42 |

Three things this table says that the corpus-geomean table in part 16 could not:

* **The damage is far worse per case than per corpus.** Part 16 reported the 0.25
  MiB geomean at 0.14–0.10; per case the worst point runs at **0.022**, i.e. 45x
  slower on 64 threads than on one.
* **1 MiB is not safe either.** Part 16's corpus geomean at 1 MiB is 1.10 for
  `t64`, which reads as "no longer a loss". Per case, **229 of 588 points are still
  slower than serial there**, worst 0.077. "Below ~1 MiB" understates where the
  guard is needed.
* **The guard beats no guard at every size, not only at the small end.** At 16 MiB
  it also removes the three points that were slower unguarded.

`C = 3e6` is chosen and the choice is a plateau, not a peak: every value in
`[3e6, 8e6]` leaves 0 points slower at all four sizes, and the spread between them
is inside the ±11% per-case floor a 64-thread measurement has (A39). The choice
between *guard* and *no guard* is not inside any floor. Weighting by method is
worth having: unweighted, the same scoring gives 1.23 at 0.25 MiB against 1.66.

**What it deliberately does not do** is predict the saturation above ~1 MiB, where
the best count is `t8`–`t32` rather than `t64`. That is a bandwidth ceiling — a
different mechanism, and benign — and fitting one constant to two mechanisms is how
the analytical blocking model lost (A33). At 16 MiB this rule caps only 60 of 588
points and never below 36 threads, which is the intended near-inertness.

`C` bundles the machine's rate, so it is a fitted constant with `kc`'s caveat and
the opposite sign: **a faster machine wants a larger one.** A single unrepeated
observation on the shared 16-core reference machine — 8 threads on the 0.25 MiB
`ij-ik-kj` case — has the guard capping to 2 and *costing* throughput there, which
is consistent with a lower thread count making the total spawn cost smaller. That
is one rep on a shared machine and is not a measurement; it is recorded because it
is the direction the caveat predicts, and because it is why the guard is off by
default rather than on.

#### The thread pool, and the deadlock that rules out the obvious reuse

`crates/tensorcontract/src/pool.rs`, ~200 lines including its tests, no dependency.
One `Mutex`/`Condvar` mailbox per worker, a shared completion counter, and one
operation: `try_broadcast(n, f)` runs `f(0)` … `f(n-1)` and returns when all have
finished. The submitting thread takes index 0, so width `n` needs `n-1` workers and
**a serial caller never causes a thread to exist**.

Three design points that are not free choices:

1. **`rayon::scope`/`spawn` deadlocks by construction here.** Our parallelism is
   SPMD-with-barriers — `pn` barriers of width `pm`, two rendezvous per `(jc, pc)`
   iteration. A task that blocks on a barrier occupies a worker thread, so a pool
   with fewer workers than barrier participants never schedules the rest and the
   barrier never opens. `ThreadPool::broadcast` *does* fit the shape but ties the
   parallel degree to the pool size, where `Plan::partition` chooses `(pm, pn)` per
   contraction; and a global pool inside a library fights the host runtime, which
   for this project means the Julia package and the C consumers concretely rather
   than hypothetically. See `REFUTED.md`.
2. **`try_broadcast` declines all-or-nothing.** Two cases: another broadcast is in
   flight (one completion counter serves the pool), or fewer than `n-1` workers
   could be spawned. The tempting graceful degradation — fold the surplus indices
   onto the submitting thread — is a **deadlock**, because indices `t` and `t + pn`
   share a barrier and running them sequentially waits forever for a participant
   that has already left. Declining costs the spawn saving for that call and
   nothing else: the driver answers a decline with `std::thread::scope`, which is
   the shipped path anyway. (A51.)
3. **Poisoning must not brick the pool.** This was found by the test, not by
   reading: a worker's panic propagates through `try_broadcast` while it holds the
   submission mutex, so the *next* broadcast finds it poisoned. Handled with
   `TryLockError::Poisoned(e) => e.into_inner()`, so `WouldBlock` is the only
   decline. Without it, the pool would silently stop pooling for the life of any
   process in which one contraction ever panicked — and `Panel::new` asserts on an
   allocation it cannot satisfy, so that is reachable (D38, A52).

The driver now writes one thread's whole job as a closure of its grid index and
reaches it two ways, pooled or spawned, so the arms cannot drift apart. The
partition, the strips and therefore the arithmetic are identical, which is why the
result stays bitwise identical to serial either way — asserted by running the whole
suite at 2, 4 and 8 threads with the pool on.

**Directional only, and stated as such**: on the shared reference machine at 8
threads, 0.25 MiB `ij-ik-kj` reads 24.4 GF/s unpooled against 43.0 and 80.1 GF/s
pooled on two runs, and 16 MiB reads 328.3 against 329.7 — large at the small end,
a no-op at the large end, which is what the mechanism predicts. The 2x spread
between the two pooled runs is why this is not a number: it is a shared machine and
five reps. The measurement is `scripts/ab.sh` on an exclusive node.

#### The batched API

`crates/tensorcontract/src/batch.rs`: `BatchItem` plus `contract_batched` /
`contract_batched_with_threads`. Many independent contractions, **the batch as the
only parallel axis**, each item running serially — nesting the two axes would spend
the spawn saving again inside every item, which is the whole point of the batch.
`driver::execute_capped` is how "serially" is expressed, and it is additive:
`execute` is that with no cap.

Three things worth recording:

* **Soundness is the borrow checker's, not a comment's.** Items are held in a
  `&mut [BatchItem]` and each item's `d` is a `&mut` borrow, so the outputs are
  *proved* disjoint; `chunks_mut` hands each thread an exclusive slice. There is no
  unsafe block on the caller's side and none in the fan-out.
* **All or nothing on validation.** Every item's bounds are checked before any item
  runs, so a batch containing one bad item writes to no output at all. A loop over
  `Plan::run` cannot give that, and a partially executed batch leaves the caller
  unable to say which outputs are valid. This is the reason to prefer the batched
  entry point even at one thread.
* **`rayon` fits here and is still not used**, for a much weaker reason than the
  inner path's: fork-join over disjoint `&mut` chunks is ten lines of
  `std::thread::scope`, and the crate already owns a pool that can be extended to
  this axis. Recorded so the next person does not re-derive it — the objection here
  is *unnecessary*, not *unsound*.

The split is static and contiguous by item count. Balancing it by estimated work is
the obvious refinement and is deliberately not guessed at: D47 measured that
block-scatter load imbalance does not cost on the dense path, and the case where it
plausibly does is **block-sparse, where items differ in size rather than in
regularity**, which is not built. That is the same division TBLIS makes — a dynamic
atomic-claim scheduler for its sparse formats, static partitioning for dense (part
15) — reached independently.

**Block-sparse is not started**, and that is the honest state of the second half of
this item.

#### Two hygiene fixes that belong here

* **A per-job `CARGO_TARGET_DIR`.** A cluster job runs in the submit directory and
  used its `target/`, so a `cargo build` on the workstation — or a `cargo test`,
  which relinks the same artefacts — replaced the very executable each arm invokes,
  mid-session, with nothing reporting it. It happened on 2026-08-04 (part 13) and
  was inert only by luck. Every script now resolves its binary from `TC_TARGET`
  (default `target/`, so hand runs are unchanged) and the `.sbatch` wrappers set it
  to `target-job-$SLURM_JOB_ID`. The two prebuilt-binary jobs *snapshot* into it
  instead, because they never compile — which also closes the failure that killed
  job 6753197, where a concurrent `prep` left a 0-byte executable that every arm
  then ran for 0.0 s with `rc=0`.
* **The harness reports the thread count it *used*.** `sweep`'s `notes` column read
  `t{requested}`; under the guard that would claim `t64` for a contraction that ran
  on two threads. It now reports `t{used}/of{requested}` when they differ, because a
  CSV that misstates its own configuration is the provenance error this project has
  had to retract twice.

#### What is not measured

Everything, on a machine. All three answers are scored offline or checked for
correctness; none has an end-to-end A/B on an exclusive node, which is exactly the
state `TENSORCONTRACT_DEEPEN` was in when it looked like +3% and then failed (A20).
The order matters and is cheap: the pool first, since it should be a strict
improvement wherever threading is used at all, then the guard on top, then a
threading default. Each is `scripts/ab.sh` with one switch.

*Decisions introduced here: D48, D49, D50, D51 — stated in [Design decisions](#design-decisions).*

| # | Assumption | Status |
|---|---|---|
| A43 | Per-call thread spawn is a second-order cost, worth fixing after the partition. | **Refuted, and now worse than part 16 reported.** Per *case* rather than per corpus, the 0.25 MiB worst point runs at **0.022** — 45x slower on 64 threads than on one — and at 1 MiB, where the corpus geomean reads 1.10, **229 of 588 points are still slower than serial**. "Below ~1 MiB" understates it. |
| A50 | A guard fitted to the sub-megabyte regime will misjudge the saturation above it. | **Confirmed, and the guard is scoped accordingly.** `C = 3e6` caps only 60 of 588 points at 16 MiB and never below 36 threads. Larger constants score *better* at 16 MiB (6.65 at `C = 8e6`) precisely because capping threads helps against a bandwidth ceiling — which is a second mechanism, and fitting one constant to two is how A33's model lost. The constant is chosen on the small-size evidence alone. |
| A51 | A pool can degrade gracefully when it has fewer workers than the requested width. | **False for this driver, and it would deadlock.** Indices `t` and `t + pn` share a `pm`-way barrier, so folding surplus indices onto one thread waits forever for a participant that has already left. `try_broadcast` therefore declines all-or-nothing and the caller spawns instead. |
| A52 | Mutex poisoning is a detail in a pool whose state has no invariants a panic can break. | **False in effect, and the test found it.** A worker's panic propagates while the submitter holds the submission mutex, so the next broadcast finds it poisoned and declines — permanently. The pool would silently stop pooling for the life of any process in which one contraction panicked, and `Panel::new` asserts on an unsatisfiable allocation, so that is reachable. Recover from poison; decline only on `WouldBlock`. |

### Part 18: the pool ships, the guard does not, and the joint arm is why

Job **6760092**, `worker5086` (Zen2 `rome`, 64 cores of one socket, 16 L3 domains),
2026-08-05, 107 min, `--exclusive`, `OverSubscribe=NO`, engine at commit `a2f425f`.
Raw data in `bench-results/worker5086-zen2/phase4g/` — 225 CSVs, four arms over
4 sizes x 7 widths x 2 dtype pairs.

Part 17 built three answers to the per-call spawn cost and measured none of them.
This measures them, and the result is not what part 17 predicted.

**The design is a 2x2, and that is the whole reason this run is conclusive.** Arms:
`base`, `pool` (`TENSORCONTRACT_POOL=on`), `guard` (`TENSORCONTRACT_AMORTISE=on`),
`both`. One discarded warm-up serves all four (A31). Each arm's `t1` column is a free
control — no switch here can touch single-threaded work — so every row is deflated by
its own `t1` before being read, which is the "columns the change cannot touch" rule
applied per row rather than per table.

#### The result

Drift-corrected treatment/base, per-family geomeans (the only granularity a
64-thread measurement supports, A39):

| | `pool` t4 | `pool` t64 | `guard` t4 | `guard` t64 | `both` t4 | `both` t64 |
|---|---|---|---|---|---|---|
| 0.25 MiB `f64` | **1.53** | **10.30** | **0.61** | 7.86 | 0.63 | 8.48 |
| 0.25 MiB `f32` | **1.77** | **11.64** | **0.73** | 10.84 | 0.79 | 12.50 |
| 0.25 MiB `c64` | 1.20 | 7.65 | 0.86 | 4.51 | 0.96 | 6.22 |
| 0.25 MiB `c32` | 1.36 | 10.44 | 0.90 | 6.66 | 1.10 | 10.48 |
| 1 MiB `f64` | 1.13 | 5.18 | 0.89 | 2.91 | 0.98 | 4.88 |
| 4 MiB `f64` | 1.12 | 4.08 | 1.01 | 1.80 | 1.10 | 4.03 |
| 16 MiB `f64` | 0.99 | 2.38 | 0.99 | 1.08 | 1.01 | 2.41 |
| 16 MiB `c32` | 1.02 | 2.27 | 1.00 | 1.02 | 1.01 | 2.40 |

**The pool is a uniform win**: no per-family cell below 0.99, up to 11.6x, monotone
down with size and up with thread count — which is what a fixed cost divided by
growing work has to look like.

**The guard is a trade, not a win.** It rescues the over-threaded case (1.0–10.8x at
`t64`) and *penalises the correctly-threaded case* by 10–39% at `t2`–`t8` on the
smallest size, where part 16 measured `t4` to be the optimum. Which side a caller
lands on depends on whether their thread count was already well chosen — something a
library cannot know.

**And on top of the pool the guard is pure loss.** `both / pool`, drift-corrected:
0.32 at 0.25 MiB `f64` `t16`, 0.41 `f32`, 0.46 `c64`, 0.51 `c32`; 0.70–0.94 at 1 MiB;
inert (0.99–1.07) at 4 and 16 MiB. Never better than 1.07. The mechanism is plain:
the cap was fitted against a ~37 µs/thread spawn cost that pooling has removed, so it
rations a resource that is now cheap.

**Per case is where the two really separate**, and it is the reading that matters
because the geomeans understate the guard's damage by 3x. Control-corrected, over all
2352 points at each width:

| width | `pool` p10 | `pool` worst | `pool` < 0.90 | `both` p10 | `both` worst | `both` < 0.90 |
|---|---|---|---|---|---|---|
| 2 | 0.987 | 0.711 | 16 | 0.849 | 0.548 | **274** |
| 4 | 0.995 | 0.742 | 23 | 0.681 | 0.327 | **323** |
| 8 | 1.011 | 0.578 | 33 | 0.861 | 0.302 | **286** |
| 16 | 1.134 | 0.688 | 16 | 1.079 | 0.466 | 59 |
| 32 | 1.712 | 0.959 | 0 | 1.595 | 0.711 | 8 |
| 64 | 1.782 | 0.928 | 0 | 1.785 | 0.850 | 1 |

The pool's sub-0.90 tail is **0.7–1.4% of points**, which is what A39's floor already
produces on *identical* configurations — so it is not evidence of a systematic loss,
and it is not proof of none. The guard's is **12–14% of points**, far outside that
floor, which is.

**So: `TENSORCONTRACT_POOL` is recommended on (D53, a user decision, not taken here);
`TENSORCONTRACT_AMORTISE` does not ship (D52).**

#### Two controls that were not the point, and both replicate

* **The base arm reproduces part 16 on a fourth Zen2 node.** 16 MiB `f64` reads best
  6.20 at `t32` and 5.00 at `t64` here, against `worker5139`'s 6.19 and 5.03. Two
  within-session ratios agreeing to 0.6%, which is what makes the treatment arm beside
  it worth reading — and it is the same form of replication part 8 used.
* **D48's offline calibration reproduces**, simulated from this node's own base grid:
  no guard 0.20 with 564 of 588 points slower, `C = 3e6` 1.65 with 0 slower, oracle
  2.37 — against `worker5139`'s 0.20/565, 1.66/0, 2.40
  (`scripts/amortise-score-rule.py bench-results/worker5086-zen2/phase4g`). The
  *simulation* is sound. What it could not see is the next subsection.

Predicted against actual for the guard, `f64c64`, requesting 64 threads: simulated
1.72 / 3.84 / 5.04 / 5.99 with 0 points slower at each size, measured 1.56 / 3.43 /
4.60 / 6.06 with 3 / 0 / 0 / 0. The three are at 0.967, i.e. at the 0.97 threshold and
inside A39's floor. The gap between predicted and actual tracks the arm's own `t1`
drift (0.934–0.994; the guard arm ran ~50 min after base).

#### The calibration was validated at one setting of another lever — A20 again

D48's table reports "0 points slower than serial at all four sizes". That is true
**at `requested = 64`**, which is the only case the offline scoring simulated, and it
is the winning half of a trade. The losing half — 0.61 at `t4` — was invisible to it
by construction, and D48 was committed quoting the table without that qualifier.

This is the third instance of A20 in this project, and the first where the author of
the rule and the author of the check were the same. The general form: **a rule that
takes a caller's parameter must be scored across that parameter, not at one value of
it.** `partition-score-rule.py` gets this right — it sweeps `-p` and `-d` — and the
amortisation scoring did not.

**The deeper problem is the functional form, not the constant.** A fixed
spawn-*fraction* threshold says "4 threads on a 0.32 ms contraction is 46% overhead,
cap it", while measurement says 4 threads is the optimum there. The real trade is
marginal — does the *next* thread return more than it costs — and answering that needs
a scaling model. That is exactly where the analytical blocking model died (A33), so
the honest move is to record the form as refuted rather than to fit a second constant.

#### Part 16's "benign bandwidth ceiling" is substantially wrong

Part 16 split the size range into two regimes and called the one above ~1 MiB benign:
"the curve saturates early — that is a bandwidth ceiling, not spawn". Within this one
session, pooled `t64` beats unpooled *best-at-any-width* in **all eight** size x dtype
rows:

| 16 MiB | unpooled best | pooled `t64` | pooled best |
|---|---|---|---|
| `f64` | 6.20 @ t32 | 11.89 | 13.19 @ t32 |
| `c64` | 7.34 @ t16 | 13.29 | 13.29 @ t64 |
| `f32` | 6.73 @ t16 | 12.83 | 15.59 @ t32 |
| `c32` | 7.29 @ t32 | 13.46 | 15.94 @ t32 |

At 4 MiB the same pattern: `f64` 5.07 @ t16 unpooled against 11.93 @ t32 pooled. So a
large part of what part 16 attributed to bandwidth was fixed cost, and D46's second
regime is narrower than it claimed. What survives is that the *optimal thread count*
still saturates below 64 — pooled, the walk compresses toward `t32` rather than
reaching `t64` — so a fixed default remains wrong at most sizes even with a pool. D46
stands as a decision; its explanation does not.

#### And the cost the pool removes is not thread spawn

Median per-call time removed at `t64`, divided by 64:

| nominal size | µs/thread, `f64c64` | µs/thread, `f32c32` |
|---|---|---|
| 0.25 MiB | 37.2 | 37.2 |
| 1 MiB | 43.4 | 41.5 |
| 4 MiB | 54.0 | 49.8 |
| 16 MiB | **93.6** | **66.2** |

A thread-creation cost is size-independent. At 0.25 MiB this lands at 37.2 µs in
*both* dtype pairs — the top of A43's 20–36 µs, and a nice independent confirmation of
it. By 16 MiB it is 2.5x that. **So "the pool removes the ~20–36 µs/thread spawn",
which is how part 17 words it, is an incomplete account** (A53).

**A hypothesis was offered here and is now refuted, from committed data and at no
machine cost.** The candidate was the per-thread packed-`A` buffer: `Panel::new(ap_len)`
is called *inside* each thread's closure on every call, so 64 fresh threads mean 64
fresh allocator arenas faulting in fresh pages, and if `ap_len` grew with the problem
that would scale with size where spawn does not.

**`ap_len` does not grow.** It is `panel_len(min(mc, ceil(m/mr)*mr), mr, kc, ..)`, and
across all four sizes the footprint is *identical* — 65536 reals in `f64`, 22528 in
`c64`, 135168 in `f32`, 46080 in `c32` — because even at 0.25 MiB the median oriented
`m` is 384, comfortably above every `mc` in play, so the cap never binds. The buffer
at 0.25 MiB is the same buffer as at 16 MiB. Its allocation cannot explain a cost that
grows 2.5x between them.

**A trap found on the way, worth more than the hypothesis was.** The first attempt at
this check read the `mc` recorded in each CSV's `notes` column — and `notes` comes from
`plan_config`, which returns the *derived* blocking, while the driver caps it at
`driver.rs:404` and allocates from the capped value at `:445`. The two agree here only
because `m` is large; on a narrow case they would not, and any analysis of buffer
footprints from these CSVs has to apply the cap itself.

**So the mechanism of the size-scaling component is unidentified**, and the honest
reading is that the "µs/thread" framing above is partly an artefact of my own
arithmetic: it assumes the whole pooled-vs-unpooled difference is a fixed per-call cost
and divides by the thread count, so any proportional component appears as growth. The
difference is neither purely fixed (it grows 2.4 → 6.0 ms) nor purely proportional (the
ratio falls 9.6 → 2.1), and fitting two parameters to four points on one node would be
over-fitting.

What is solid: **a fixed component of ~37 µs/thread**, visible at 0.25 MiB where
parallel work is negligible, identical in both dtype pairs, and an independent
confirmation of A43's 20–36 µs. What is not: everything about the rest. Two candidates
remain, neither testable from this data — scheduler placement of freshly created
threads, whose penalty accrues over the thread's life and so scales with runtime; and
barrier skew, whose count grows with `N/NC` x `K/KC` and hence with size.

**Consequently the "third, smaller fix" this section originally proposed — hoisting the
`Panel` allocations out of the per-call closure — is not justified.** Its premise was
that the buffer grows, and the buffer does not. It might still help by removing a
constant allocation, but that is a different and much weaker claim, and nothing here
supports building it.

#### What is not measured

* **One node, one session.** The direction is trustworthy on this machine class; the
  magnitudes are not portable, and nothing here may be differenced against part 16's
  absolute speedups (different node).
* **Nothing at 64 MiB.** The corpus size the rest of this file uses is outside the
  sweep. The pool's win decays with size (2.0–2.8x at 16 MiB) and may be small there,
  but "may be" is the honest word. `THREADS="1 16 64" STAGES=threads` is ~55 min.
* **Nothing on an Intel hierarchy.** Every partition and threading finding so far has
  been topology-dependent (A36), and the pool has no reason to be — it touches where
  threads come from, not what they read — but that is an argument, not a measurement.
* **The batched API is unmeasured entirely.** It was built in part 17 and this run did
  not exercise it.

#### A methodological note on the monitoring, since it nearly cost the run's ending

The completion predicate watching this job's log grepped for the `finished :` banner.
`STAGES=small` takes the sbatch's early-exit path — `echo "outdir $OUT"; cat
"$SUMMARY"; exit 0` — which never prints that banner, so the predicate could not fire
on a successful run. What caught it was a quiet-timer: "no new log lines for 15
minutes" fired 16 minutes after the last write. **A watcher's success path needs the
same coverage discipline as its failure paths**, and a timeout on silence is what
makes an incomplete predicate survivable, because silence and success look identical
from outside.

*Decisions introduced here: D52, D53 — stated in [Design decisions](#design-decisions).*

| # | Assumption | Status |
|---|---|---|
| A43 | Per-call thread spawn is a second-order cost. | **Refuted, and now measured directly rather than inferred.** Removing it is worth up to 11.6x at 0.25 MiB and 64 threads, and 2.0–2.8x even at 16 MiB. The 0.25 MiB figure of 37.2 µs/thread, identical in both dtype pairs, independently confirms the 20–36 µs part 16 inferred from a different node. |
| A50 | A guard fitted to the sub-megabyte regime will misjudge the saturation above it. | **Confirmed, and the misjudgement is the other way round from the worry.** The guard is inert above 4 MiB (0.98–1.09), so it does no damage there. What was misjudged is the *saturation itself*: it is largely fixed cost, not bandwidth. |
| A53 | The cost a thread pool removes is thread creation, and the part that scales with size is the per-thread packed-`A` allocation. | **First half refuted, second half refuted too.** The pooled-vs-unpooled difference is not a pure fixed cost: it grows 2.4 → 6.0 ms with size while the ratio falls 9.6 → 2.1, so it has a proportional component and the "µs/thread" figure manufactures growth by dividing a mixed quantity by the thread count. The named candidate is *also* wrong: `ap_len` is byte-identical at all four sizes (65536 reals in `f64`, 22528 in `c64`), because `mc` never binds against `m` at these shapes. **A fixed ~37 µs/thread is solid** and confirms A43 independently; the scaling component's mechanism is **unknown**, with scheduler placement and barrier skew as untested candidates. Nothing supports hoisting the `Panel` allocations. |
| A54 | A rule that consumes a caller's parameter can be validated at one value of it. | **Refuted, and it is A20 in a third form.** D48's guard scores 0 points slower than serial at `requested = 64` and loses 39% at `requested = 4` on the same size. The offline scoring simulated only 64, so the losing half of the trade was invisible by construction. Score across the caller's parameter, as `partition-score-rule.py` does with `-p` and `-d`. |

### Part 19: the pool loses on Ice Lake, and D53 is withdrawn

Job **6762432**, `worker6194` (Ice Lake-SP, 32 cores of one socket, **one 48 MiB L3**),
2026-08-05, 107 min, `--exclusive`, engine at `ea10301`. Two arms, `base` and `pool`,
at 1 / 16 / 64 MiB. Raw data in `bench-results/worker6194-icelake/phase4g/` — 84
arm-runs, 21168 rows, 0 missing, 0 zero-second.

This was the strengthening evidence part 18 asked for before flipping D53: the same
arms on an Intel hierarchy, plus a 64 MiB point at the corpus size. **The reading rule
was fixed before the data existed** — at 64 MiB and `t32`, `>= 1.10` meant "matters
everywhere", `1.00–1.10` "small contractions only", `< 0.97` "the pool costs at corpus
size, which would be a real surprise".

It came in at **0.80**.

#### The result

`pool / base`, drift-corrected by each row's own `t1` (0.996–1.007 throughout, so the
control is clean and this is not session drift):

| MiB | dtype | t2 | t4 | t8 | t16 | t32 |
|---|---|---|---|---|---|---|
| 1 | `f64` | 1.054 | 1.006 | 0.591 | **0.457** | 0.527 |
| 1 | `c32` | 1.028 | 0.928 | 0.546 | **0.403** | 0.478 |
| 16 | `f64` | 0.994 | 0.875 | 0.740 | 0.505 | 0.587 |
| 16 | `c32` | 0.993 | 0.895 | 0.773 | 0.592 | 0.650 |
| 64 | `f64` | 0.993 | 0.932 | 0.872 | 0.791 | **0.802** |
| 64 | `c32` | 0.997 | 0.961 | 0.893 | 0.779 | **0.781** |

**46 of 60 cells below 0.97, worst 0.403** — the pool is up to **2.5x slower** than
per-call spawn on this machine. All four dtypes agree. Per case at 64 MiB, **210–284 of
588 points fall below 0.90** (36–48%), worst 0.094, which is systematic under any
reading of A39.

The shape: neutral at `t2` (1.01–1.08, where the pool has a single worker), degrading
through `t4`–`t8`, **worst at `t16`**, partial recovery at `t32`. Worst at `t16` rather
than at the top width is not explained and does not fit a simple monotone story.

So on Zen2 the pool is worth up to 11.6x and on Ice Lake it costs up to 2.5x. **D53's
recommendation cannot stand**, and the switch is topology-conditional at best.

#### The a-priori argument was wrong, and that is the transferable part

Recorded here before the run, as the reason a null result was expected rather than a
reversal: *"the pool has no reason to be topology-dependent — it touches where threads
come from, not what they read."*

**That is false, and it is false for a reason worth carrying.** Reusing a thread also
reuses its **allocator arena**, so each worker gets the *same* packed-`A` buffer address
back on every call, where a freshly spawned thread gets a fresh, well-spread one. **A
thread pool is an allocation-locality change as much as a thread-lifetime change** — it
changes what the threads read, not just where they come from (A55).

That reframing also makes the topology dependence unsurprising in hindsight, which is
the uncomfortable part: 32 long-lived buffers at repeating addresses all contend in
**one** 48 MiB 12-way L3 here, where Zen2 spread the same buffers across **sixteen**
private 16 MiB L3s — and Zen2 is exactly where the pool won.

#### The mechanism is a candidate, not a finding

The extra time the pool costs is neither a fixed per-call cost nor proportional to
runtime:

| MiB | width | base ms | pool ms | delta ms | delta/base |
|---|---|---|---|---|---|
| 1 | 32 | 0.609 | 1.046 | 0.431 | 0.71 |
| 16 | 32 | 2.304 | 3.944 | 1.671 | 0.73 |
| 64 | 32 | 5.355 | 7.484 | 1.781 | 0.33 |
| 64 | 8 | 8.293 | 9.284 | 0.939 | 0.11 |
| 64 | 16 | 6.061 | 7.857 | 1.459 | 0.24 |

It grows with width and with size, then plateaus. The implementation's *fixed* costs —
31 serial `notify_one` wakes and 31 threads contending on one completion mutex — are of
order 100–200 µs, an order of magnitude too small to be this.

Two candidates, and this data cannot separate them:

1. **Shared-L3 conflict between reused buffers**, per A55 above. Scales with width (more
   aliasing buffers) and with traffic (size). A cheap test: pad each worker's buffer by
   a per-worker offset so the addresses cannot alias, and re-run this arm.
2. **Barrier cost at 32 participants**, or the wake pattern interacting with it. `t16`
   being worse than `t32` is unexplained under (1) and might belong here.

Both locate the problem in **our implementation rather than in pooling as a concept**,
which is the reason D49 keeps the pool as a switch rather than deleting it.

#### What did not change

* **The Zen2 result stands**, and was labelled with its machine throughout. The pool
  really is worth up to 11.6x there. What is withdrawn is the *default*, not the
  measurement.
* **D52 stands.** The guard lost on Zen2, where the pool won; nothing here touches it.
* **A control replicated, and it is now the strongest baseline in this file.** The base
  arm at 64 MiB reproduces both committed Ice Lake curves to **0.1–1.5%**: `f64`
  `t8`/`t16`/`t32` = 6.14 / 9.62 / 10.55 against `worker6016`'s 6.17 / 9.67 / 10.59 and
  `worker6150`'s 6.13 / 9.61 / 10.54, and `c64` 3m `t8` = 6.96 against 6.96. Three
  nodes, three sessions, two scripts. It also cross-checks `phase4g`'s 64 MiB path
  against `phase4f`'s `threads` stage for the first time — they measure the same thing.

#### What this run says about the process, twice over

* **The pre-registered reading rule earned its keep.** Writing `< 0.97 → a real
  surprise` down *before* the data meant the reversal could not be narrated as
  "roughly neutral" afterwards. It cost two minutes.
* **Two of my own a-priori arguments were wrong in one day** — that the pool could not
  be topology-dependent (this part) and that the buffer grows with problem size (part
  18's A53). Both were stated as reasoning rather than measurement, and both were
  checked because they were written down as claims. The lesson is not "argue less"; it
  is that an argument recorded as an argument gets tested, and one folded into prose
  does not.

*Decisions introduced here: none. D53 is revised in place — see
[Design decisions](#design-decisions).*

| # | Assumption | Status |
|---|---|---|
| A55 | A thread pool changes thread lifetime, not memory locality — it touches where threads come from, not what they read. | **Refuted, and it was the reason a reversal was thought impossible.** Reusing a thread reuses its allocator arena, so every worker gets the same packed-`A` buffer address back each call where a fresh thread gets a well-spread one. On a one-L3-per-socket machine 32 such buffers contend in one 48 MiB cache; on Zen2 they spread over sixteen private L3s, which is where the pool won. A pool is an allocation-locality change. |
| A56 | The pool's benefit is topology-independent, so one machine class is enough to recommend it. | **Refuted — 11.6x on Zen2, 0.40–0.80 on Ice Lake.** This is the **fourth** threading or kernel choice here that fails to transfer between microarchitectures, after A34 (register blocks), A36 (the partition) and A44 (the method ranking). Treat "measured on one machine class" as a statement about that class until shown otherwise, for anything touching threads or caches. |

## Packaging and distribution

Phase 5: the quality gates that had never run, the API tiers, the TAPP conformance suite, the shipped C header and its consumer, cross-compilation, the JLL and the Julia package.

### Phase 5 part 1: what preparing v0.1 found

Packaging was expected to be tidying. It was mostly **discovering that the
project's own quality gates had never run**, which is a more useful result and
worth recording in full so the lesson survives.

#### CI was not gating anything it claimed to gate

Three of five jobs could not have been passing, each for an independent reason,
and none had ever been noticed because nobody had run the commands locally with
the flags the workflow uses:

| job | why it could not pass |
|---|---|
| `msrv` | pinned toolchain **1.75** against a declared `rust-version` of 1.89, so cargo refuses before compiling anything |
| `lint` | `cargo fmt --all -- --check` against a tree with drift in five files, mostly macro-adjacent code in `kernel/x86.rs` and `plan.rs` |
| `docs` | `cargo doc` under `-D warnings` against **17 rustdoc errors**, nine of them public documentation linking into private modules |

The general lesson, which is the same one A15 and A20 taught in the measurement
domain: **a gate nobody has watched fail is not a gate.** Every command in the
workflow was re-verified locally before being trusted, and the new job set is
smaller and states what each job proves.

Two of the new jobs cover code paths that had **never been exercised in CI at
all**: the threaded driver (threading is off by default, so no test ran it) and
the analytical blocking model. The x86 runners have AVX2 but not AVX-512, so
CI's default path is now the AVX2 kernels — the ones whose register blocks are
provisional (D26) — which makes the untuned path the *automatically* tested one.

#### Defects that would have shipped

* **`examples/kernel_shapes` did not compile off x86**, and `cargo test` builds
  examples, so the suite failed for every aarch64 user. Now a cfg-gated module.
* **The declared MSRV was wrong** in the other direction too — see D31.
* **The `trace` feature was declared, described, and implemented nowhere.**
  Removed rather than advertised.
* **The TAPP crate cannot be packaged before the engine is published.** It
  depends on the engine by path *and* version, and packaging rewrites that
  into a registry dependency which must then resolve — so
  `cargo package -p tensorprimitives-tapp` fails at "failed to prepare local
  package for uploading" until `tensorcontract 0.1.0` is in the index.
  `cargo package --workspace` appears to work, and does set up a temporary
  registry to satisfy the dependency, but it is **not a reliable gate**: it was
  observed verifying the dependent crate against a *stale* extraction of the
  engine as soon as the dependent used an engine API added since the previous
  packaging run — which is precisely the case such a job exists to catch. It
  passed earlier in this session and then failed on exactly that change, which is
  how the behaviour was found. CI therefore gates `cargo package -p
  tensorcontract` only, and the ordering stands as the documented publish
  procedure: `tensorcontract`, wait for the index, then `tensorprimitives-tapp`.
* **The `std` feature promised something it does not deliver.** Disabling it
  compiles, but the crate has no `#![no_std]` and uses `Vec`, so it is not a
  no-std build. The feature is now documented as the seam a future port would
  widen rather than as a claim.
* **The README described the Phase 2 engine** — scalar kernels, "performance not
  yet meaningful" — two phases and two instruction sets out of date.

#### What the API decision cost and bought

D30's three tiers. The part worth carrying forward is that this project *needs*
a tier whose values are explicitly unstable: it ships measured heuristics, and it
re-measures them every phase. Publishing without saying so would have forced a
choice between freezing the tuning and breaking semver at each phase boundary.

Two side effects of the documentation pass are worth keeping. `kernel::scalar`
had been *documented* as the route by which a foreign scalar type gets a correct
engine for free and never demonstrated; it now carries a worked example that
compiles and runs as a doctest for 0.11 s. And both blanket
`allow(clippy::missing_safety_doc)` attributes are gone — the x86 one had been
hiding that the four generated kernels have *different* panel and tile bounds
(1x, 2x, 3x the real kernel's, by packing format) and that the plain-`fn`
trampolines drop the `#[target_feature]` attribute but not the obligation.

#### The TAPP conformance suite, and the four gaps it found

`crates/tensorprimitives-tapp` had **one test** — a happy-path `c64` contraction —
behind a coverage table claiming four datatypes, TAPP cases 1–4, conjugation on
any operand, two documented rejections and mixed precision. For the crate whose
entire purpose is that a C caller can swap this engine for TBLIS behind one
header, that was the thinnest-tested part of the workspace, and every claim in
the table was unverified. It is now **84 tests**, all through the `extern "C"`
entry points rather than the Rust `Plan` behind them, because the bugs this layer
can have are exactly the ones invisible from there.

Two design points worth copying. Numerics go against the brute-force oracle
*plus* two hand-computed anchors, so a shared engine/oracle bug cannot pass. And
the suite was **falsified before it was trusted**: perturbing one hand-computed
constant failed exactly the hand-checked test, while perturbing the oracle call's
`alpha` failed 33 of 34, leaving only the anchor — which is the correct pattern
and confirms the two mechanisms are independent. A conformance suite nobody has
watched fail is not evidence, which is A15's lesson in a third domain.
`abi_layout.rs` additionally re-declares the whole upstream header in an
`extern "C"` block and drives a contraction through it, so a renamed `#[no_mangle]`
is a link error in our own tests rather than a downstream C build's discovery.

**Four gaps, all fixed rather than filed** (none was a wrong number):

1. **An extent product could abort the caller's process.** Nothing between
   `TAPP_create_tensor_info` and `build_scatter` checked that a tensor's extents
   multiply to something representable. The product wrapped: in release the plan
   built, reported success and computed *nothing*; in debug the multiply panicked
   inside an `extern "C"` function, which the compiler turns into a process abort.
   `reduce_tensor` now folds with `checked_mul` and reports
   `Error::ExtentProductOverflow` → `TAPP_ERROR_SHAPE`. Checking per tensor bounds
   every scatter vector, since each is as long as the product of some *subset* of
   one tensor's axes.
2. **A null `C` silently discarded `D`.** Upstream defines `TAPP_IN_PLACE` as
   `NULL` and leaves its meaning an open `//TODO`; this crate read it as
   `beta = 0`, so `beta = 1, C = TAPP_IN_PLACE` — precisely what a caller writes
   for `D += alpha*A*B` — overwrote `D` and reported success. The ambiguous
   combination is now refused; in-place accumulation is expressible by passing
   `D`'s own pointer as `C`.
3. **Every library handle was the value `1`.** `HandleState` was zero-sized, so
   `Box::into_raw` returned `NonNull::dangling()`: two live handles were
   indistinguishable and a C program creating two and destroying both was
   double-freeing — harmlessly, for exactly as long as the state stayed empty.
   It has a reserved field now, and the comment claiming this made handle
   validity *checkable* is gone: it never did, and nothing can.
4. **Three declared symbols did not exist.** `TAPP_attr_set`/`_get`/`_clear` were
   absent, so a C program including `<tapp.h>` and calling one failed to **link**
   — the least diagnosable failure available. Now exported as refusals, which is
   conformant (upstream specifies no keys) and diagnosable. Prototypes fetched
   from the upstream header, not reconstructed.

Also corrected: `execute` now writes `0` through a non-null `status`, so the
idiomatic create/execute/destroy sequence stops handing an uninitialised value to
the destructor; and the coverage table now admits that mixed *storage* types are
rejected (§1.6 of `DESIGN.md` lists them as in TAPP's scope, and they are — just
not here), and that `TAPP_ERROR_*` beyond zero are this crate's own numbering,
since upstream `error.h` fixes only `TAPP_SUCCESS`.

| # | Assumption | Status |
|---|---|---|
| A26 | The TAPP layer is thin enough that the engine's own correctness tests cover it. | **Refuted.** Everything the layer can get wrong — datatype-tag dispatch, `intptr_t` handle casts, label arrays read at a rank the info supplies, `beta` on a null `C`, the `status` and `prec` arguments — is invisible from the Rust API and had no test. Four gaps on first contact, one of which aborted the caller's process. **Test an FFI layer as its caller, not as its callee.** |

#### Still open before publishing

* **Defaults.** Threading and the blocking model are both off, which is honest
  but means a 64-core machine gets one thread and a foreign cache hierarchy gets
  constants fitted to `ccqlin038`. Flipping either needs the two pending
  measurements, not a decision.

  **Superseded, and not in the direction this expected.** Both measurements have
  since been taken (parts 10 and 11). The blocking model **loses** on the first unseen
  machine and stays off on evidence rather than for want of it (A33); threading's
  scaling turned out to be topology-dependent in a way `Plan::partition` does not
  model, and the one machine measured first would have produced the wrong rule
  (A36). So "flipping either needs a measurement" was right about the process and
  wrong about the outcome: the measurements argued for leaving both alone.
* Publication itself, which is a human step and deliberately not automated.

### Phase 5 part 2: the distribution surface, and a Julia consumer

**Status: complete except for the irreversible steps, which are deliberately not
taken.** No tag, nothing published to crates.io, no Yggdrasil PR. Prompted by the
question "could I try this from Julia", which turns out to be the same question as
"can this be distributed as a binary at all" — and the answer was no, for four
reasons that had nothing to do with the engine.

Part 1 ended with a list of two open items. This is the list that was actually
open, and every item on it was found by *doing* the thing rather than by reading
the code. That is the theme, and it is the same theme as part 1's "a gate nobody
has watched fail is not a gate".

#### Four gaps between "the ABI is correct" and "a distribution can ship it"

1. **Nothing reported a version.** `TAPP_implementation_name()` returns a
   free-form string with no number in it, and the crate version was visible only
   to cargo. A distribution that ships `include/` and `lib/` as separate packages —
   which is what a JLL, a system package, or a stale `-I` all produce — could have
   them from different versions with nothing anywhere to notice. There are now
   `TAPP_VERSION_*` macros describing the header and
   `TAPP_implementation_version()` describing the linked library, and **both halves
   are tested**: `abi_layout.rs` reads the header as *text* and compares to
   `CARGO_PKG_VERSION`, so the check holds on a machine with no C toolchain, and
   `examples/c-consumer` compares the macro its own compiler saw against the string
   its own link returned.
2. **The `cdylib` had no SONAME, and on macOS something worse than none.** rustc
   emits `-soname` only for the `dylib` crate type, never for `cdylib`, so every
   consumer recorded a bare filename. On Mach-O, ld64 defaults `LC_ID_DYLIB` to the
   `-o` path — an absolute build-tree path — and BinaryBuilder's `ensure_soname`
   autofix *returns early whenever an ID is present without inspecting its value*.
   So a macOS build would have shipped unrelocatable with a clean audit. See D40.
3. **Eight of the 23 prototypes in `tapp.h` had never been seen by a C compiler
   or a linker.** `main.c` called the ones that do work; the setters,
   `TAPP_get_strides`, the batched product, `TAPP_destroy_status` and two attribute
   functions were checked only by `abi_layout.rs`'s Rust transcription, which is a
   second hand-maintained copy rather than an oracle. The header's `TAPP_ERROR_NULL`
   prose bug — fixed in this pass — was exactly that failure class one level
   further out: prose, which nothing tests at all.
4. **There were no install rules.** The library was consumable only by knowing
   cargo's directory layout. `install.sh` plus a pkg-config template fixes that, and
   the example gained a third link mode (`-DTAPP_PREFIX=`) that consumes an
   installed prefix — the only mode in which the header comes from outside the
   source tree, and therefore the only one that would catch an install that forgot
   to copy it. The recipe calls the same script, so there is one definition of the
   layout rather than two that drift.

#### Cross-compilation: one prediction inverted, one silent failure (D39)

The repository had never been cross-compiled. Eleven targets now are, in CI, and
the two findings are both worth more than the eleven passes.

The prediction was that **`i686` would fail**, because `kernel::x86` is gated on
`any(target_arch = "x86", target_arch = "x86_64")` — so the AVX-512 intrinsics are
instantiated on 32-bit x86 too — and `core::arch::x86` genuinely lacks the
intrinsics taking 64-bit integer operands. It compiles: the kernels are f32/f64
FMA-shaped and use none of them. The cfg was left alone rather than narrowed to
`x86_64` on suspicion. Note what this does *not* say: in 32-bit mode only
`zmm0`–`zmm7` are encodable, so the register blocks — chosen against a 32-register
file — will spill on `i686`. That is a performance property, it is unmeasured, and
it is recorded here rather than acted on.

**musl fails silently.** With the target default (`crt-static` on) cargo prints
`dropping unsupported crate type cdylib` and **exits 0**. A musl JLL would have
been a tarball containing a header, a pkg-config file, two licences and no library
— green, and inexplicable downstream. `-C target-feature=-crt-static` fixes it, and
because the symptom is a warning rather than an error the `cross-musl` job asserts
the warning *still appears* without the flag, so the day it stops being needed is
visible instead of assumed.

#### What running the recipe found that reading it did not

The BinaryBuilder recipe was dry-run locally, on this workstation, before being
considered done. Four corrections, none of which a careful reading produced:

* **The available Rust shards stop at 1.94.0, not 1.97.0.** Master's
  `Artifacts.toml` advertises 1.97.0; the BinaryBuilderBase that the *released*
  BinaryBuilder resolves to offers 1.57.0 through 1.94.0. `choose_shards` **errors**
  on a version with no shard, so an optimistic pin is a hard build failure rather
  than a graceful fallback. This is [[prefers-verified-releases]] again, in a new
  place: check what is released, not what the development head says.
* **`riscv64-linux-gnu` and `aarch64-unknown-freebsd` have no Rust toolchain at
  any available version.** Enumerated rather than guessed, by calling
  `choose_shards` on all 18 supported platforms. Both would take the scalar path
  anyway. Filtered, with the query that establishes it recorded in `RELEASING.md`
  so it can be re-run rather than re-derived.
* **The licence directory name.** The auditor looks under
  `share/licenses/tensorprimitives_tapp` — the *package* name, underscore — while
  `install.sh` uses the crate name with a hyphen. The recipe's first draft argued
  itself out of calling `install_license` on the grounds that install.sh already
  did the job, and the audit answered "Unable to find valid license file", which is
  one of the two things a Yggdrasil reviewer greps the log for. install.sh gained
  `--no-licenses`.
* **`CompilerSupportLibraries_jll` makes the `libgcc_s.so.1` warning worse.** It
  adds a missing-artifact-mapping warning of its own — CSL's artifacts are keyed by
  libgfortran version — and does not silence the original. Julia ships libgcc_s in
  its own libdir, so the library loads and runs; that was settled by the Julia test
  suite loading it, not by argument. Reverted to an empty dependency list with the
  reasoning attached in the recipe, since a reviewer will ask.

**Thirteen of the fifteen platforms were then built and audited locally**, and all
thirteen produced a tarball with a real shared library in it and an identical
layout: `lib/` (or `bin/` on Windows), `include/tapp.h`, `lib/pkgconfig/`,
`share/licenses/`. `x86_64-w64-mingw32` behaves as predicted — the artifact is
`tensorprimitives_tapp.dll` with no `lib` prefix, it lands in `bin/` because that is
what `${libdir}` is there, the import library goes to `lib/`, and the two-name
`LibraryProduct` finds it without renaming anything. `i686` linked, which `cargo
check` could not establish. The two not built are the Apple targets, and only
because accepting the Xcode SDK licence is not a decision to take on someone's
behalf.

Exactly **three** distinct audit warnings across all thirteen, all understood:
the `cpuid` one below, `libgcc_s.so.1` on ELF, and `bcryptprimitives.dll` on
Windows — the last a Windows 10+ system DLL that Rust's standard library imports
for randomness and that the auditor's system-library list does not know. A fourth
would be a real finding, and the recipe says so.

And one **prediction confirmed verbatim**, which is why D40 exists. The audit log
reads: *"contains a `cpuid` instruction; refusing to analyze for minimum instruction
set, as it may dynamically select the proper instruction set internally. Would have
chosen avx512, instead choosing x86_64."* `check_isa` fails a build whose minimum
instruction set exceeds the platform's, and this is a generic x86-64 binary full of
AVX-512 kernel bodies. It ships only because the auditor abandons the analysis on
finding a `cpuid`, which `kernel::cache` and `std_detect` happen to supply. Nobody
promised that, so CI asserts it.

#### The Julia side, and why the backend shape is the right one

`julia/TensorPrimitives` is two layers: `LibTAPP`, a complete `ccall` wrapper with
handles as distinct Julia types and finalizers; and `TAPPBackend`, a
`TensorOperations.jl` backend. The second is the one that matters, for a reason
that is about this engine specifically rather than about convenience.

`TensorOperations.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ...)`
hands over exactly what TAPP takes: arbitrary extents, element strides, and
per-operand conjugation. So `pA`/`pB`/`pAB` become a **relabelling** —
`_labels` is twenty lines of index bookkeeping — and nothing permutes, copies, or
allocates a temporary. A backend over a GEMM would have to. This makes the
transpose-free claim something a Julia user can observe rather than read, and it
puts this engine on the same footing as `TensorOperationsTBLIS.jl` for comparison,
which is the shape the eventual three-way benchmark wants.

The backend is opt-in and deliberately **not** registered with `select_backend`:
loading the package changes nothing for code that does not ask. `tensoradd!` and
`tensortrace!` fall through to TensorOperations' own backends. TAPP can express
both — a trace is a repeated label, an add is a contraction against a rank-0
operand — but each is a correctness surface, and acquiring one for free is how a
wrong answer gets shipped.

64 tests, every one checked against TensorOperations' own backend on the same
inputs rather than against a rewritten expectation. The two that were worth
writing: non-contiguous strided views, where a strides-in-elements or base-pointer
error would show and nothing else would; and `β == 0` against an all-`NaN` output,
because "overwrite" and "multiply by zero" differ exactly there, and TAPP spells the
former as a null `C` operand.

#### Two things a reader should not take from this section

* **No performance was measured *here*.** Nothing in this section is a throughput
  claim and no code in it can change one: the engine was not touched. The two Phase
  4 measurements this branch was written alongside have since landed on `main`
  (parts 10 and 11), and the CHANGELOG's confidence table has been updated from *those*
  — five rows, none of them because of anything in this section. If a reader arrives
  at a new Julia backend and infers that something got faster, the answer is that a
  binary distribution was built, not a kernel.
* **Threading is still not reachable from Julia**, because
  `TAPP_execute_product` ignores its executor argument and `TENSORCONTRACT_THREADS`
  is read once per process. The Julia wrapper documents that rather than papering
  over it, and deliberately wires nothing to `TAPP_create_executor` — plumbing a
  knob through an inert object would be worse than the absence.

#### Still open before publishing

Unchanged from part 1 on defaults — and see the note added there: both measurements
have landed and both argued for leaving the defaults alone, so the honest v0.1
position is "off on evidence" rather than "off pending evidence". Plus:

* The **tag, the crates.io publish and the Yggdrasil PR**, in that order, for the
  reasons in `RELEASING.md`. All three are human steps.
* **One known unfixed defect is now release-facing.** A35 found, in committed data
  and at no machine cost, that `planar` `f32`/`c32` ships the register block `32x6`
  where the Phase 3 sweep's own output names `32x5`, 7.8% faster at the operating
  `kc`. It is recorded in the CHANGELOG's confidence table rather than quietly
  carried, because shipping a known register-block defect under a table that says
  "measured" is exactly the kind of claim this project has spent two phases learning
  not to make. Fixing it needs a corpus A/B — kernel margin is not corpus margin —
  so it is a decision for whoever cuts the tag, not something to slip in.
* **The repository is private.** Yggdrasil builds only from publicly downloadable
  sources, so the recipe's `ArchiveSource` cannot resolve until it is public and
  tagged. Making it public also publishes this file and 9 MB of benchmark CSVs,
  which is a decision rather than a side effect.
* **The Apple targets are unverified.** They need the Xcode SDK licence accepted
  (`BINARYBUILDER_AUTOMATIC_APPLE=true`), which is a legal agreement and therefore
  not something to accept on someone's behalf. They are also the two targets where
  the install-name work in D40 actually matters, so they should be the first thing
  built after that acceptance.

### Interlude: making the C surface consumable

Prompted by a concrete external ask — a colleague evaluating this for the
**NDA** C++ array library (TRIQS, Flatiron) — the question was whether a C++
project can use this without shipping a Rust compiler. Answering it honestly
turned up four gaps between "the ABI is correct" and "a C++ project can link
it", all of which were on the Phase 5 packaging list and none of which touched
the engine. See D36–D38.

#### The gap that mattered

The TAPP layer had 23 verified `extern "C"` symbols, a conformance suite driving
those symbols, and `abi_layout.rs` re-declaring the whole upstream header so a
dropped export is a link error. What it did not have was **a header**, and
nothing anywhere invoked a C compiler. The correctness of the ABI was thoroughly
established *from Rust*; whether a C caller could actually use it was inference.

That inference held — the C consumer passed on its first real run, including the
complex path and `TAPP_CONJUGATE` — but it held by luck as much as design, and
it would not have survived the first wrong prototype.

#### What the toolchain objection was actually worth

Little, once examined, and the examination is the useful part. NDA already
requires CMake ≥ 3.22, a concepts-capable C++ compiler, and HDF5 + MPI + OpenMP
on by default. Against that, `rustc` is marginal — and **a C++20 compiler is the
harder constraint on cluster environments**: the stock compiler on this very
workstation is gcc 8.5, which cannot compile nda at all, while `rustup` installs
a pinned toolchain into `~/.cargo` with no root and no system integration. The
MSRV, 1.89, was released 2025-08-04, one year ago to the day.

So the real blockers are maturity and measurement coverage, not the build.
Recorded so the toolchain argument is not re-litigated.

#### A claim corrected in the making

The vendoring story was initially stated as "three crates", from the runtime
dependency tree (`num-complex` → `num-traits`, plus `autocfg` at build time).
Running it showed **~17**: `cargo vendor` is workspace-wide and includes
dev-dependencies, so `rand` and its tree (`getrandom`, `libc`, `zerocopy`,
`syn`, …) come along. Both numbers are true of different things — a library-only
vendor from the published crate is three; vendoring this repo so the *tests* run
offline is seventeen — and the CI job now states both and exercises the second,
since a test step is the only thing that keeps the dev-dependency half honest.

| # | Assumption | Status |
|---|---|---|
| A47 | The shipped header agrees with the library it describes. | **Now tested rather than assumed.** `examples/c-consumer` compiles the header with a C compiler, links the built library and checks numerical results in `f64` and `c64`; CI runs it in both the corrosion and prebuilt modes. Previously no C compiler saw the header at any point. |
| A48 | A Rust panic reaching the C boundary is acceptable because it is memory-safe. | **Rejected as a policy.** Memory-safe but process-fatal, and the engine panics on allocation conditions a caller can hit. D38 converts it to an error code on the three entry points that can raise it. |

## The baseline comparison

The engine against TBLIS 2.0-dev, TBLIS v1.3.0 and TTGT, re-measured on the tightest floor in this file — and the finding that the method ranking does not travel.

### Phase 5 part 3: the comparison re-measured, and the method ranking does not travel

Job **6753260**, `worker6156` (Ice Lake-SP, AVX-512, 2 x 32 cores, SMT off),
2026-08-04, 231 min, `--constraint=icelake --exclusive`,
`scripts/rusty-compare.sbatch` → `scripts/compare-bench.sh`. Raw data in
`bench-results/worker6156-icelake/compare-tblis-skx/`.

**Why this run happened at all.** The newest engine-vs-baseline data in the repo
was `bench-results/phase3-sweep-*` from 2026-08-02, and every Phase 4 gain landed
after it, so the README's comparison understated the engine. This is the
re-measurement. It is **not** on the reference machine — that was a deliberate
choice, taken because `ccqlin038` is shared and the last A/B run there had 6–14
co-tenants per arm — and the cost of that choice is that **nothing here can be
differenced against the Phase 3 table.** Different microarchitecture, different
cache hierarchy. The improvement attributable to Phase 4 remains unmeasured; only
a `ccqlin038` run can supply it.

#### The measurement is the tightest in this file

| floor | arms | geomean spread | outside ±6% |
|---|---|---|---|
| near (`A2` vs `A`) | minutes apart | 0.995–1.003 | 1 of 980 |
| **session-span (`A3` vs `A`)** | **2.5 h apart** | **0.998–1.001** | **0 of 980** |

Identical arms repeated to within 0.4 s on 1424 s; every arm reports `self 100%,
co-tenants: none`. That is roughly **5x tighter than `ccqlin038`'s published
±1.3% geomean / ±6% per case**, which is the argument for A32 — derive the floor
in session — stated as a number.

**A32 is refined by this run, and in the direction of "it is not universal".**
The session-span floor is *no worse* than the near floor, so drift does not grow
with the distance between arms here. On Zen2 it did (0.02% at a minute, 1–2% at
an hour, 4.4% across a cold start). The 35-minute discarded warm-up arm bought
confirmation rather than a correction. A31 and A32 look like properties of boost
headroom and co-tenancy, not of measurement in general — keep the warm-up arm
(it is cheap and it is how you learn which case you are in), but do not expect it
to move anything on an exclusive SMT-off node.

#### The engine, this build, this machine

49-case corpus at 64 MiB, single core, planar, GF/s. **These columns are
independent of the TBLIS question below**, which cannot touch them.

| dtype | min | median | geomean | max |
|---|---|---|---|---|
| `f32` | 11.0 | 74.8 | 77.2 | 168.8 |
| `f64` | 7.1 | 44.7 | 43.7 | 71.6 |
| `c32` | 25.5 | 121.0 | 113.9 | 179.3 |
| `c64` | 15.7 | 62.0 | 63.3 | 91.1 |

Correctness verified on the node in all three methods before any timing.
Against TTGT (OpenBLAS, unaffected by the TBLIS question): **2.15x `f64`, 2.12x
`c64`, 2.20x `f32`, 1.95x `c32`** on corpus geometric mean.

**Complex beats real again, on a second microarchitecture.** `c64/f64` = 1.449,
`c32/f32` = 1.474, against Cascade Lake's 1.416 and 1.432. This is a within-run
ratio, so it is valid where the absolute numbers are not comparable, and it is
the project's central technical result replicating on new hardware to within
about two points. Twice the arithmetic intensity amortises overhead better; the
weak spot is low arithmetic intensity in either domain, not complex.

#### The method ranking does not transfer, and item 3's premise fails here

This is the result worth carrying forward, and it contradicts something the
Resume-here block lists as settled.

| relative to planar | Cascade Lake (Phase 3) | **Ice Lake (here)** |
|---|---|---|
| `c64` 1m | 0.967 | 0.868 |
| `c64` 3m | 0.956 | **0.694** |
| `c32` 1m | 0.979 | 1.004 |
| `c32` 3m | 0.921 | **0.744** |

And the inversion that Phase 4 item 3 was going to exploit is **absent**:

| subset | `c64` | `c32` |
|---|---|---|
| memory-bound (`min(n,k) <= 64`, n=24) | planar 49.3 > 1m 44.7 > 3m 35.9 | 1m 87.0 > planar 81.4 > 3m 67.3 |
| compute-bound (n=25) | planar 80.5 > 1m 67.0 > 3m 53.4 | planar 157.2 > 1m 148.6 > 3m 105.6 |

**3m is last in every column and wins 0 of 49 cases**, its per-case ratio against
planar running 0.63–0.84 — uniform, not a few catastrophic shapes. On Cascade
Lake 3m was the *fastest* of the three on the memory-bound subset. So:

* **Item 3 ("dispatch 3m on memory-bound shapes") is a pessimisation on Ice
  Lake** and must not be built as an unconditional rule. If it ships at all it has
  to be conditioned on the microarchitecture, which is the same conclusion A34
  reached for register blocks and A36 for the thread partition. That is three
  independent findings pointing one way.
* **This looked confounded with A34, and it is not.** See the next subsection: the
  deconfounding data was already committed and the confound does not exist.
* Practical consequence for quoting: **the ranking, and the inversion, are Cascade
  Lake results.** The README must say so rather than stating them as properties of
  the engine.

#### The A34 confound does not exist, and the data was already on disk

*Added 2026-08-05, from committed data, at no machine cost.* The subsection above
recorded A44 as confounded with A34 and named "extend the Ice Lake register-block
sweep to 3m" as the experiment that would separate them. **That experiment had
already run.** `bench-results/worker6016-icelake/kernel-shapes.txt` (job 6746817)
covers all four methods — 35 lines of 3m — at three depths in both AVX-512 dtype
pairs. This is a second instance of A35's lesson: the answer sat in committed raw
output for two days while the file above described it as unmeasured.

**The confound requires 3m to be running a shape that is wrong for Ice Lake. It is
not.** 3m ships `(1, 10)` in both precisions — `8x10` in `c64`, `16x10` in `c32` —
and the Ice Lake sweep names `MV=1 NR=10` as 3m's *best* shape in both. Shipped and
optimal coincide, on both machines, in both precisions. A34's shape error is real
and it lands on `real` (Ice Lake wants `MV=3 NR=9`) and on planar `f32` (A35's
`32x5`); it does not touch 3m at all, because there is no better shape to give it.

So the comparison can be made at each method's shipped shape, which is what the
corpus actually runs, at the operating `kc` for each dtype:

| | Cascade Lake | **Ice Lake** |
|---|---|---|
| `c64`, `kc = 256`: planar `16x6` / 3m `8x10` | 102.8 / 87.8 → **0.854** | 108.4 / 63.8 → **0.589** |
| `c32`, `kc = 384`: planar `32x6` / 3m `16x10` | 195.7 / 194.9 → **0.996** | 221.8 / 134.5 → **0.606** |

Those kernel-level ratios bracket the corpus-level 0.694 / 0.744 that A44 measured,
which is the consistency check. And the collapse is **uniform across 3m's whole
shape space** rather than a shape choice: every 3m shape is slower on Ice Lake at
every depth (`8x10` 87.8 → 63.8, `16x4` 82.5 → 59.5, `24x3` 76.1 → 56.2) while
planar `16x6` gets *faster* (102.8 → 108.4). Nor is it the register budget: 3m's
`16x10` is flagged `live = 32!` on Ice Lake, but the best *unflagged* 3m shape there
is slower still (`32x4` at 125.8 against `16x10`'s 134.5), so avoiding the spill
does not rescue it either.

**The bigger casualty is the mechanism, not the rule.** This project has carried
"3m's 25% flop saving is real, and with L1-resident panels 3m is the *fastest* of
the three" as settled mechanism rather than as a machine-specific result. At
`kc = 16`, which is that regime, at shipped shapes:

| | Cascade Lake | **Ice Lake** |
|---|---|---|
| `c64` 3m / planar | **1.105** | **0.567** |
| `c32` 3m / planar | **1.155** | **0.662** |

3m leads by 10–16% on Cascade Lake at L1-resident depth and trails by 34–43% on Ice
Lake at the same depth. **On Ice Lake 3m does not win at any depth, at any shape, in
either precision.** So the L1-resident advantage is a Cascade Lake property too, and
the sentence above must be qualified wherever it appears — it was carried into
`CLAUDE.md` unchecked during the 2026-08-05 clean-up, which is the same failure mode
as "the corpus is fully regular".

What survives is the *accounting*: 3m really does 3 products where planar does 4,
and it really moves 3 planes of both operands where planar moves 2. What does not
survive is the claim that the saving pays in a nameable regime. The one candidate
explanation visible in the committed columns is load-port pressure — 3m runs at
`f/l` 0.89–1.71 against planar's 1.78–3.75, i.e. it is load-bound where planar is
FMA-bound — and Ice Lake evidently punishes that. That is a **hypothesis from one
column of one sweep**, not a measurement.

**Consequence: item 3 is dead rather than blocked**, and no node time is owed to it.
The Ice Lake shape sweep that was on the to-do list is deleted from it.

#### The TBLIS 2.0 columns are under review, and why

This run used `../baselines/tblis-2.0-install` — the build every previously
committed TBLIS 2.0 number used. Its BLIS was configured with
`BLIS_CONFIG_FAMILY=auto`, which resolves against the *build* machine, so it is
**skx-only**: `nm` shows one `bli_cntx_init_skx`, 1364 `bli_` symbols, and no
skinny-GEMM (`sup`) kernel set. Rebuilding the same source at the same commit
with `BLIS_CONFIG_FAMILY=x86_64` gives thirteen contexts and 3809 symbols, and a
one-shot look on `ccqlin038` put it **up to 1.68x ahead** on the premise shapes,
clustered on the low-arithmetic-intensity cases.

Two things follow, and the first is uncomfortable:

1. **A mis-configured baseline flatters this engine on exactly the shapes the
   project identifies as the real headroom.** The README's "roughly parity with
   TBLIS 2.0 on complex" rests on it.
2. The Phase 1 headline is probably safe, because it is a ratio *within* TBLIS 2.0
   (complex efficiency ÷ real efficiency) and both halves moved together in every
   pair inspected. It should still be re-derived.

That 1.68x is `--size 8 --reps 2`, one shot, on a shared machine, with no warm-up
and no repeat arm — a signal, not a result, by this file's own standards.
`scripts/ab-tblis.sh` prices it properly (job 6754877): warm-up, A (skx),
B (multi-config), A′ (skx), the treatment being `LD_LIBRARY_PATH` on one binary,
with **`planar` in every arm as a control no TBLIS library can move**. It covers
the sweep corpus and then the premise shapes, the latter because a dry run showed
the two libraries indistinguishable over the sweep corpus at trivial size and the
original signal came from `premise`.

**Resolved by the next subsection: the TBLIS 2.0 columns from this run are
sound after all.** The A/B found the two builds identical at both measured sizes,
so the skx build understated nothing here, and the 1.68x was an artifact of the
8 MiB size it was taken at. Read this paragraph together with that retraction.

#### The TBLIS A/B lands, and it retracts the alarm — the 1.68x was a size artifact

Job **6754877**, `worker6156` (the same node, which is the best case for
comparability), 2026-08-04, stage 1 = 90 min. Data in
`bench-results/worker6156-icelake/ab-tblis/`.

**The two TBLIS 2.0 builds are indistinguishable at the sizes this project
publishes at.** Treatment is multi-config ÷ skx; `planar` is the in-arm control,
which no TBLIS library can move.

| | `f64` | `c64` | `f32` | `c32` |
|---|---|---|---|---|
| sweep corpus, 64 MiB — floor (`A2` vs `A`) | 1.001 | 0.998 | 1.001 | 0.999 |
| sweep corpus, 64 MiB — **treatment** | 1.000 | 0.997 | 0.998 | 0.997 |
| premise, 200 MiB — floor | 1.000 | 1.000 | — | — |
| premise, 200 MiB — **treatment** | 0.998 | 0.998 | — | — |

Every treatment ratio is inside its floor. Per case on the premise shapes the
spread is 0.979–1.002 against a floor spread of 0.992–1.003, and **the specific
shapes that produced the original 1.5–1.7x read 0.993–1.001**: `ajbc-ckba-jk`
0.993/1.000, `abrs-qb-aqrs` 0.997/0.999, `ijkl-imkn-jnlm` 1.000/1.000,
`ij-ikl-ljk` 0.999/0.999, `abjcd-dkbac-jk` 0.999/1.001.

**So the alarm is withdrawn.** The claim that the skx-only build understated
TBLIS 2.0 by up to 1.68x on low-arithmetic-intensity shapes, and that the
README's near-parity-with-TBLIS-2.0 statement therefore rested on a
mis-configured baseline, **was wrong.** Nothing in the README or the CHANGELOG
needed to change on that account, and the Phase 1 headline is untouched.

**What actually happened, because the process failure is the useful part.** The
1.68x came from one `premise --size 8 --reps 2` run: a problem size **8 to 25x
smaller** than anything this project publishes (sweeps at 64 MiB, premise at
200 MiB), on a shared machine, no warm-up, no repeat, no control. At 8 MiB the
operands are small enough that BLIS's skinny-GEMM (`sup`) kernels — present in the
multi-config build and absent from the skx-only one — decide the result. At 64 and
200 MiB they are irrelevant, so the two libraries converge exactly. The symbol
count that looked like a smoking gun (1364 against 3809 `bli_` symbols) was real
and had no consequence at the sizes that matter.

This is a **new failure mode for this file's methodology rules**, which are all
about time: discard a warm-up arm (A31), derive the floor in session (A32), price
co-tenancy per arm (A27). None of them would have caught this, because the
confound was **problem size**, not timing or contention. Hence A46.

It also vindicates a decision that was not mine: rewriting the docs on the
strength of the signal was considered and rejected in favour of measuring first.
Had the signal been acted on, a false retraction would have gone into the
release-facing documents — the exact failure mode the CHANGELOG's confidence
table exists to prevent, arriving through the person maintaining it.

**What survives.** The portability half, entirely: the skx-only build SIGILLs on
any machine without AVX-512, which killed job 6753261, and
`../baselines/tblis-2.0-x86_64-install` is now the build a non-AVX-512 node needs.
And a genuine positive result, cheaply held: the TBLIS 2.0 baseline is **measured
to be build-insensitive at publication sizes**, where before it was an
assumption nobody had tested.

#### Replicated, and the baselines placed

Job 6754877 stage 2 ran the **whole comparison set a second time** on the same
node, against the multi-config TBLIS 2.0 (231 min, rc=0). Data in
`compare-tblis-x86_64/`. So there are now two independent full runs, in separate
allocations three hours apart, and they agree across **all twenty dtype × engine
columns to 0.997–1.003**. The second run's own near and session floors are both
1.000–1.002. The `tblis` column agrees to 0.998–1.001, which confirms the A/B
verdict a second way — a full run does not see the build difference either.

That is the strongest evidence base this repo has for any comparison number: two
runs, each internally bracketed, agreeing at the level of their own floors.

**The Phase 1 headline replicates, and one number is identical to three digits.**
Efficiency against a same-shape GEMM ceiling, mean over the 12 premise cases at
200 MiB, complex efficiency ÷ real efficiency *within each engine*:

| engine | `f64`→`c64` | `f32`→`c32` | Cascade Lake, Phase 1 |
|---|---|---|---|
| **TBLIS v1.3.0** (latest stable) | **0.215** | — | **0.215** |
| TBLIS 2.0-dev | 1.413 | 1.306 | 1.06 / 1.15 |
| TTGT (OpenBLAS) | 1.307 | 1.348 | 1.17 |
| this engine, planar | 1.316 | 1.232 | — |
| this engine, 3m | 0.934 | 0.950 | — |

**TBLIS v1.3.0's 0.215 on Ice Lake is the same 0.215 measured on Cascade Lake**,
and that is not a coincidence to be marvelled at — it is the mechanism confirming
itself. 1.x has no complex micro-kernel for any post-Sandy-Bridge x86
configuration, so complex runs the generic template while real gets tuned
assembly, on *both* machines. The ratio is a property of the software, not of the
hardware, which is exactly what Phase 1 concluded from the flat-across-shapes
diagnostic. Nothing in this project has replicated so cleanly.

The refutation against 2.0-dev is *stronger* here than on the reference machine
(1.413 against 1.06). 3m is the only engine below 1.0, consistent with A44.

**Where the engine sits, median GF/s over those 12 cases** — same run, same core,
single-threaded throughout:

| dtype | this engine | TBLIS 2.0-dev | TTGT | engine ÷ TBLIS 2.0 |
|---|---|---|---|---|
| `f64` | 46.5 | 30.8 | 14.0 | **1.51x** |
| `c64` | 64.0 | 48.8 | 25.9 | **1.31x** |
| `f32` | 81.1 | 45.6 | 25.9 | **1.78x** |
| `c32` | 123.2 | 88.3 | 49.2 | **1.40x** |

This is a materially better position than the README's current claim of "roughly
parity with TBLIS 2.0 on complex", which was a Cascade Lake Phase 1 number
(42.5 against 42.4 in `c64`) taken before every Phase 4 gain. **Two caveats keep
it honest.** It is a different machine, so it is not evidence of how much Phase 4
bought — that remains unmeasured and needs a `ccqlin038` run. And the engine is
*handicapped* here: A34 says three of the eight shipped register blocks are 9–13%
off on Ice Lake, and A44 says the method ranking does not transfer. It wins these
columns while running shapes chosen for a different microarchitecture.

#### Assumptions added

| # | Assumption | Status |
|---|---|---|
| A44 | The complex-method ranking, the memory-bound inversion, **and 3m's L1-resident advantage** are properties of the engine and the shape. | **Refuted, and no longer confounded.** On Ice Lake 3m is last in every column and wins 0 of 49 cases (0.694 `c64`, 0.744 `c32` against planar, against Cascade Lake's 0.956 / 0.921), and the inversion is absent. The A34 confound was *assumed* and does not exist: 3m ships `(1, 10)` in both precisions and the Ice Lake sweep names that as 3m's own best shape, so there is no better shape to give it. At shipped shapes the kernel-level ratio goes 0.854 → 0.589 (`c64`) and 0.996 → 0.606 (`c32`), the collapse is uniform across 3m's whole shape space, and **at the L1-resident `kc = 16` 3m leads by 10–16% on Cascade Lake and trails by 34–43% on Ice Lake** — so the mechanism this project called settled is machine-specific too. Item 3 is **dead**, not blocked. |
| A45 | A baseline install built from the right source at the right commit is the right baseline. | **Refuted, but on portability only** — the performance half was measured and withdrawn, see the A/B subsection. `BLIS_CONFIG_FAMILY=auto` silently fits BLIS to the *build host*, so this TBLIS 2.0 is skx-only and **SIGILLs on any machine without AVX-512** (job 6753261). It also lacks the skinny-GEMM kernels, which looked like a large performance defect and turns out to cost **nothing at 64 or 200 MiB** (0.997–1.000 against a 0.998–1.003 floor). So: record a baseline's *configuration*, not just its version and commit; verify a multi-ISA claim with `nm` rather than `strings`, because BLIS compiles its config name table in whether or not the kernels are there; and do not infer a performance consequence from a symbol count. |
| A46 | This file's measurement rules (A27, A31, A32) cover the ways a comparison can mislead. | **Refuted: they are all about time, and problem size is a separate axis.** A one-shot `premise --size 8` run reported TBLIS 2.0's two builds differing by up to 1.68x. At the sizes actually published — 64 MiB sweeps, 200 MiB premise, i.e. 8–25x larger — they are identical to within the floor, because the kernels that differ only matter while the operands are small. A warm-up arm, an in-session floor and per-arm occupancy would each have passed the bad measurement through unchanged. **Measure at the size you publish at, and treat a result taken at a smaller size as being about that size.** |

---

## Part 20 — Apple Silicon, and what the portable path actually costs

**2026-08-06, `CKF6QCDVPD` (Apple M3 Max), `bench-results/CKF6QCDVPD-m3max/`.**
The first non-x86 measurement in the project. Stage A of the plan: make an
aarch64 run honest, then measure it. Stage B — a NEON micro-kernel — is **not
started**; see "Resume here".

### The result that changes a published claim

**The portable path costs about 1.5x, not the order of magnitude `CHANGELOG.md`
and the Julia README implied.** The properly sized measurement is `premise`, 12
shapes, 64 MiB, reps 3, one thread — geomean GF/s:

| | `f64` | `c64` |
|---|---|---|
| engine, `kernel::scalar` | 19.84 | planar 24.20, 1m 23.32, **3m 29.74** |
| OpenBLAS TTGT (NEON) | 30.68 | 35.02 |
| engine / OpenBLAS | **0.65** | **0.69** planar, 0.85 3m |

**Quote those, and quote them with their size.** A separate probe — one case
(`ij-ik-kj`), `--size 8`, reps 1 — read engine 25.9, OpenBLAS-TTGT 54.5, TBLIS
2.0-dev 55.8 in `f64`, i.e. **0.46x of TBLIS**. That number is real but it is one
case at one size, and **the corpus sweep that would confirm it at 64 MiB was not
run** (see "Resume here"). A46 exists because a one-shot small-size run produced a
1.68x that vanished at the published size; do not let 0.46 become the headline
until a 64 MiB sweep says so.

**The mechanism is one instruction, not a missing kernel.** The disassembly of
`kernel::scalar::real_ukr` on this target shows LLVM *does* vectorise it: eight
`float64x2_t` accumulators for the 4x4 tile, `ld1r.2d` broadcasts, and
`fmul.2d v20, v16, v18[0]` — the **lane-indexed** form, which it found without
being asked. What it does not emit is `fmla`. `real_ukr` writes
`*s += *ap.add(i) * bv`, two roundings, and LLVM may not contract that into an FMA
under IEEE rules; every `kernel::x86` body calls a true `_mm*_fmadd_*` instead.

So the portable path runs at two instructions per multiply-accumulate where the
machine offers one, which **halves the ceiling available to it from 64.8 to 32.4
GF/s**. On its best premise shape it reaches **80.5%** of that halved ceiling
(26.07 of 32.4); OpenBLAS on its best reaches **87.1%** of the full one (56.44 of
64.8). Two independently measured efficiencies, agreeing under one frequency
assumption — that is the evidence the model is right, and it is stronger than
either number alone. **The scalar kernel is not badly written; it is denied one
instruction form**, and that is the whole gap.

Do not apply the 32.4 ceiling to the complex columns: `c64` 3m reads 36.94 at
best, i.e. 114% of it, because *useful* flops for 3m exceed the products it
actually issues. That is 3m working as designed, not an error.

**Consequence for Stage B, and it is a smaller prize than expected.** A NEON
kernel's headline win is FMA contraction, worth up to 2x, plus better register
blocking — the 4x4 tile uses only 8 of 32 vector registers. Do not expect 10x, and
do not justify the work with the old claim.

### 3m wins here — and this is a statement about the *portable path*, not the machine

Premise shapes, 64 MiB, geomean over 12 cases, `c64`: **3m 29.74 GF/s, planar
24.20, 1m 23.32.** 3m leads planar by **1.229**.

That is a third ordering, after Cascade Lake (3m leads only at L1-resident `kc`)
and Ice Lake (3m last everywhere, 0 of 49). **It does not reopen A44 and it is not
evidence about Apple Silicon.** The mechanism is specific to the crippled kernel:
3m computes three products where planar computes four, and on a path that is
*instruction-throughput*-bound — two instructions per MAC, no FMA — a 25% saving in
products is a 25% saving in the binding resource. The bytes-per-useful-flop
accounting that decides it on x86 is not what binds here. **Once a NEON FMA kernel
exists the binding constraint changes and this ranking may well invert**, so
re-measure it in Stage B rather than carrying it forward. Recorded as A58.

The `f64` columns are a free control: planar, 1m and 3m all read 19.84 GF/s
exactly, because for real `f64` all three select the same real kernel. They agree
to the digit, which is what says the harness is not confusing methods.

### The floor, and what this machine cannot give

A second free control fell out of running `premise` twice, once per BLAS: the
engine columns are untouchable by a BLAS swap and moved **≤0.5%** (19.84 → 19.74
`f64`, 24.20 → 24.01 `c64`). That is the only floor figure available at the time
of writing; the three identical corpus arms that produce the real one were still
in flight (see "Resume here").

**There is no CPU pinning on Darwin** — no `sched_setaffinity`, and
`thread_policy_set(THREAD_AFFINITY_POLICY)` is a no-op on Apple Silicon. Every
other number in this file is a pinned single-core measurement and **none of these
is**. Add a laptop's DVFS and thermal envelope and 12 P-cores beside 4 E-cores the
scheduler may migrate between. The compensations — AC power (the 4.05 GHz turbo
requires it), a quiesced machine, thermal level recorded before and after,
everything pinned single-threaded, three identical arms rather than one — are all
in `scripts/macos-session.sh` and none of them is a substitute. The one thing that
is *better* here: **no SMT**, so the hyperthread sibling sharing L1d and L2, which
invalidated two Phase 4 conclusions, does not exist.

### Three things that were reporting fiction, and one that was hiding

* **The cache probe fell to `BUILTIN`** — 32 KiB L1d, 256 KiB L2, an 8 MiB L3
  shared by 4 — on a machine with 128 KiB, a 16 MiB cluster L2 and no L3, and
  `tcbench info` printed them as though probed, so `l3_domains` was derived from a
  fabricated `shared_by`. Fixed with a `sysctl` source (D54). **The trap inside
  the fix:** the unprefixed `hw.l1dcachesize` / `hw.l2cachesize` answer for the
  *last* perflevel — the **efficiency** cores — so the naive read returns 64 KiB
  and 4 MiB for a machine whose P-cores have 128 KiB and 16 MiB. A silent 2x/4x
  under-block that looks like a successful probe.
* **The host printed as `unknown`**, which is the one field every
  `bench-results/` directory name keys on, and `arch-label.sh` could not run at
  all. Both fixed.
* **A57, found by feeding the model true descriptors.** `model_mc` reads
  `l2.ways` and `l2.bytes_per_way()` and **never divides by `cores_sharing`**.
  Invisible on all three x86 machines here because their L2s are per-core; six M3
  Max P-cores share one 16 MiB L2, so the model gives each the whole thing and
  derives a **12 MB packed `A` block** for `f64`. A third and *structural* reason
  A33 stands — not a reopening, and not fixed, because the model is not the
  default and tuning a refuted path on one unmeasured machine is how A33 happened.
* **"Results are bitwise identical across all switches" is false on two rows.**
  `TENSORCONTRACT_KERNEL` (scalar's two roundings against x86's fused one, now
  confirmed in disassembly) and `COMPLEX=3m` (three products, not four — which
  `tests/conformance.rs:416` already said). Nothing pins cross-ISA identity, so
  nothing caught it. Corrected in CLAUDE.md and scoped to the switches for which
  it does hold.

### The corpus really is regular here, and the 42.9% figure is an x86 artefact

`reg_a = 1.000` on **784 of 784** case-dtype-method-arms at `--size 64`, and
`reg_b` likewise. CLAUDE.md's 42.9%-below-1.0 needs `MR` in {16, 32, 48}, none of
which divides the 24 that TCCG rounds every stride-1 extent up to; the scalar
path's `MR` is **4**, which does. So **`--stress ragged` is the only source of
irregularity on this machine**, and any write-back or gather-path claim taken here
without it is a claim about the fully regular case.

### What was portable, for the record

`cargo test --workspace --release --lib --bins --tests` and `--doc` pass green
unchanged, and `clippy --all-targets` is silent under `-D warnings` — CI's
`portable` job already covered this. All 49 cases verify in all four dtypes
against OpenBLAS-TTGT, Accelerate-TTGT and TBLIS, and in all three complex
methods. Every `scripts/*.py` is pure stdlib and runs here unmodified, so the
whole offline grid-scoring workflow works. What does not: **17 shell/Python
drivers**, on `taskset`, `/proc/stat`, sysfs or `os.sched_getaffinity`, plus
`examples/kernel_shapes`, which is x86-only by construction, plus `tcbench shapes`,
which emits a header and no rows because `row_blocks` returns `&[]` off x86.

**And a trap that will bite anyone who ports a driver:** BSD `pgrep` accepts `-a`
and does **not** print command lines, so the exclusivity guard in five drivers
cannot filter its own subshell and refuses to start — silently, and looking exactly
like a real co-tenant. `pgrep -fl`. `date -Is` also fails where `-Iseconds` works.

---

## Part 21 — The NEON register blocks, measured

**2026-08-06, `CKF6QCDVPD` (Apple M3 Max),
`bench-results/CKF6QCDVPD-m3max/kernel-shapes*.txt`.** Stage B's calibration
step. The NEON menus shipped a few hours earlier were derived from the
32-register budget alone and said so; this replaces them with three arms of
`examples/kernel_shapes` and a floor.

### Three arms, because one has no floor

~8 min each, no baselines, machine quiet, AC power. Scored with the new
`scripts/kernel-shapes-compare.py`, which is what converts three arms into a
floor, a list of unstable shapes and a per-method margin. **Scored at the
deepest `kc` column only** — 256 for `f64`, 384 for `f32` — and that is not a
preference: it is what `Blocking::derive` gives those element sizes, so these
are the shapes the driver runs. The `kc = 16` column is a regime the engine
never enters.

| | floor, median | floor, p90 | stable shapes |
|---|---|---|---|
| `f64` / `c64` | 0.73% | **1.89%** | 38 of 41 |
| `f32` / `c32` | 1.24% | **3.64%** | 41 of 41 |

Leads are scored against the p90. That the `f32` floor is double the `f64` one
is itself a small finding: the same shapes run at twice the throughput at
`L = 4` and are correspondingly more exposed to a laptop's clock.

**Three shapes are unstable rather than noisy** — an arm-to-arm disagreement
this size is a shape doing two different things, not a bad measurement of one
number. They are excluded from the floor and are not shippable at any mean:

| | arms | spread | |
|---|---|---|---|
| 1m `3x10` | 37.8 / 54.9 / 54.8 | **45.2%** | over the register budget |
| 3m `8x2` | 63.1 / 73.3 / 72.7 | 16.2% | |
| 3m `6x3` | 70.7 / 62.6 / 70.9 | 13.3% | |

### What shipped, and six of eight are ties

| | shape | GF/s | margin | |
|---|---|---|---|---|
| `f64` real | **`16x3`** | 58.3 | +4.9% over `4x8` | 2.6x floor — **measured** |
| `f64` planar | `4x6` | 56.2 | +0.2% | a tie with `4x5`, `2x12` |
| `f64` 1m | `2x8` | 55.9 | +1.1% | a tie with `4x6`, `3x8` |
| `f64` 3m | **`2x8`** | 79.3 | +7.6% over `4x3` | 4.0x floor — **measured** |
| `f32` real | `8x8` | 111.5 | +0.5% | a tie, six ways |
| `f32` planar | `8x6` | 112.1 | +0.1% | a tie with `8x5`, `4x12` |
| `f32` 1m | `8x6` | 111.2 | +1.4% | a tie with `6x8`, `12x4` |
| `f32` 3m | **`4x8`** | 159.1 | +7.1% over `8x3` | 2.0x floor — borderline |

**Saying "tie" is the deliverable, not a hedge.** A single arm would have
printed eight confident winners; six of them are inside this session's own
spread. Where a method tied, the budget-derived incumbent was kept and the menu
below the first entry is ordered by measurement without any claim that the order
is resolved.

### The budget was a good filter and a bad chooser

It filtered well in half the columns and then named the wrong winner in **three
of eight**. The filtering is worth stating precisely, because "the budget
works, it just picks badly" is too kind to it:

| | over-budget shapes | where they land |
|---|---|---|
| planar, both dtypes | 4 and 4 | **the bottom four of the column, every time** |
| 3m, both dtypes | 3 and 3 | **the bottom three, every time** |
| real, both dtypes | 4 and 4 | mixed — `f64` `16x3` is the column's **winner**, `f32` `12x10` is third |
| 1m, both dtypes | 3 and 3 | mixed — `f32` `6x10` is fifth of ten, and `f64` `3x10` is the unstable one |

So the flag is a clean separator for the two methods with the largest
accumulator footprint and an unreliable one for the two with the smallest. That
is not a refinement of the budget; it says the budget is modelling spill
pressure and the other two methods are limited by something else. `f64` real is the sharp case. The budget proposed
`6x8`, which is `bli_dgemm_armv8a_asm_6x8`, BLIS's own AArch64 shape, and that
agreement was recorded as corroboration. The machine prefers `16x3` by 4.9%,
2.6x the floor — and `16x3` is a shape the budget calls **over** its 32-register
limit at `live = 33`.

This is A34 restated on a fourth ISA rather than a new finding, and it is
recorded as A60 because the specific form matters: *the model that correctly
rejects is not thereby a model that correctly selects*, and a well-known
library's chosen shape agreeing with the model is not evidence about this
engine's kernels.

`16x3` is also the one shape whose two regimes disagree: 0.0% spread across
three arms at `kc = 64` and `256`, and bimodal at `kc = 16` (51.5 / 51.5 /
40.9). The engine never runs it at 16, so it ships — but a future change to
`Blocking::derive` that shallows `kc` for `f64` would need to re-check this,
and that is exactly the change item 6 of the Stage B list contemplates.

### The interaction the sweep cannot see, and why the default is safe anyway

`kernel_shapes` measures one kernel on hot packed panels. It knows nothing about
write-back, and `MR` is not only a kernel constant — it is the granularity at
which the *output's* row scatter is blocked.

`tcbench shapes`, which works here now that the NEON menus give `row_blocks`
something to return, says what that costs: **`MR = 16` drops write-back
regularity on 12 of 49 `f64` cases**, the whole `abcijk` family, `wb` 1.00 →
0.67 and `reg_a` → 0.67 on six of them. 16 does not divide the 24 that TCCG
rounds every stride-1 extent up to; 4, 6, 8 and 12 do. This is the first time
the corpus has been anything but perfectly regular on this machine — part 20
recorded `reg_a = 1.000` on 784 of 784 rows, at the scalar path's `MR = 4`.

**It ships as the default anyway, and the reason is a measurement, not a
tolerance:** the guarded row-block rule already demotes exactly those 12 to
`4x8`. Every one of them has `k = 24`, and the rule's first guard is `k <= 32`.
So `tcbench shapes` reports 12 of 392 case-dtype-methods changing shape, **0**
left with no regular shape on their menu, and **0** moved onto a majority-gather
path.

That rule was derived on Cascade Lake, on AVX-512, for `MR` in {16, 32, 48}, and
it transfers to NEON `f64` unmodified. Given A56 — four threading or kernel
choices that fail to transfer — a rule that does transfer is worth naming.
Recorded as A61.

---

## Part 22 — What the NEON kernel is worth, and it closes the Apple gap

**2026-08-06, `CKF6QCDVPD` (Apple M3 Max), `bench-results/CKF6QCDVPD-m3max/neon-ab/`.**
Four arms, 47 minutes, `scripts/macos-neon-ab.sh`. Stage B's end-to-end
measurement, run *after* the shapes were calibrated so that it measures what
ships (A20).

### The design, and it is better than the equivalent x86 A/B

One binary, four arms — `warm` (discarded), `A-scalar`, `B-neon`, `A2-scalar` —
with `TENSORCONTRACT_KERNEL` as the treatment. Both code paths compile in and
the switch chooses at run time, so this satisfies A15 without a build-to-build
diff. **The Phase 4 kernel work could not do this**: an AVX-512 machine cannot
un-have its own kernels, so "what is the vectorised kernel worth" has never been
answerable there. Here it is one environment variable.

`ttgt` and `tblis` ride along in every arm as columns the switch cannot reach.
They moved **0.996–1.006**. That is a control from outside the engine, and it is
the reason the engine numbers below can be read at all.

### The floor

| | geomean range | |
|---|---|---|
| engine columns, A vs A2 | **1.000–1.008** | quote ratios against this |
| baseline columns | 0.982–1.004 | |
| per case | 9 of 120 outside ±6% | worst *engine* case **1.105** |

So a geomean is readable to about ±1% and a per-case ratio to about ±10%. **This
session finally has a floor**, which part 20 did not — and it is still not a
pinned measurement, because Darwin has no CPU affinity API.

### The treatment

12 premise shapes, 64 MiB, reps 3, one thread, geomean GF/s:

| | scalar | NEON | ratio |
|---|---|---|---|
| `f64`, all three methods | 19.90 | **36.60** | **1.840** |
| `c64` planar | 24.34 | 44.46 | 1.826 |
| `c64` 1m | 23.31 | 41.74 | 1.791 |
| `c64` 3m | 29.85 | 50.45 | 1.690 |
| `f64` OpenBLAS TTGT *(control)* | 31.62 | 31.51 | 0.996 |
| `f64` TBLIS 2.0-dev *(control)* | 36.73 | 36.76 | 1.001 |
| `c64` OpenBLAS TTGT *(control)* | 35.20 | 35.42 | 1.006 |
| `c64` TBLIS 2.0-dev *(control)* | 44.97 | 45.19 | 1.005 |

**A58 predicted ~2x from FMA contraction alone and this is 1.84**, per-case
1.586–2.028 in `f64`. The earlier one-case `--size 8` probe read 1.91: right in
direction, slightly high in magnitude, which is the outcome A46 asks you to
check for rather than assume.

### The headline: the engine reaches its baselines on this machine

| | scalar | NEON |
|---|---|---|
| `f64` engine / TTGT | 0.63 | **1.16** |
| `f64` engine / TBLIS 2.0-dev | 0.54 | **1.00** |
| `c64` planar / TBLIS | 0.54 | 0.98 |
| `c64` 3m / TBLIS | 0.66 | **1.12** |
| `c64` 3m / TTGT | 0.85 | **1.42** |

Best-shape efficiency against the 64.8 GF/s P-core NEON FMA peak: **engine 52.76
= 81.4%**, OpenBLAS 56.88 = 87.8%, TBLIS 58.18 = 89.8%.

That last line is the cleanest confirmation A58 could have got. Part 20 measured
the scalar path at **80.5% of a ceiling halved to 32.4** by the missing `fmla`.
The NEON kernel reaches **81.4% of the full ceiling**. Same efficiency, twice the
ceiling — the gap really was one instruction form, and removing it recovers
exactly what the accounting said it would.

**The 0.46x figure is superseded and should stop being quoted.** Part 20 carried
it from a one-case `--size 8` probe, correctly labelled provisional. At 64 MiB
over 12 shapes the scalar path is **0.54** of TBLIS 2.0-dev. Direction right,
magnitude 17% off. That closes one of the three gaps the interrupted Stage A
session left; the other two — no `ragged` arm, no per-case corpus spread —
remain.

### 3m still leads, and A59 guessed wrong about which way

| | 3m / planar |
|---|---|
| scalar path | 1.226 |
| NEON path | **1.135** |

A59 expected the ordering to move or invert once a real FMA kernel removed the
instruction-throughput bottleneck that explained 3m's lead. **It narrowed and did
not invert.** 3m gains least from NEON (1.690 against planar's 1.826), which is
exactly what the mechanism predicts — the resource it saves is no longer the
binding one — and it still finishes ahead by 13.5%, an order of magnitude outside
the 1% floor.

So this is a **fourth ordering**, and the only one taken with a tuned kernel on
its own machine:

| machine | kernel | 3m |
|---|---|---|
| Cascade Lake, AVX-512 | measured | leads only at L1-resident `kc` |
| Ice Lake, AVX-512 | measured | last in every column, wins 0 of 49 |
| M3 Max, portable | none | leads planar by 1.226 |
| M3 Max, NEON | measured | **leads planar by 1.135** |

A44 is not reopened — the ranking stays per-microarchitecture, which is the
whole point of a fourth ordering. What changes is the scope of a sentence in
`CLAUDE.md`: "treat 3m as the method that makes the comparison honest, not as a
candidate default" is now a claim about **x86**. On this machine 3m is the
fastest complex method by a wide margin, on a properly tuned kernel, and it is
the arm that passes TBLIS.

Also, and for the third time: a one-case `--size 8` probe had suggested 3m's
lead *widens* under NEON. At 64 MiB it narrows. A46.

In `f64` all three methods read 36.60 exactly, because they select the same real
kernel — a control on the harness, not a coincidence.

### What this does not show

No `ragged` arm, so nothing here is about the irregular path — and `ragged` is
the only source of irregularity on this machine. No 49-case corpus sweep; these
are the 12 premise shapes. Nothing per-case finer than ~10%. Nothing about any
other aarch64 part, because register blocks are per-microarchitecture (A34) and
these were measured on this one. Nothing about threads.

---

## Archive: phases 1–3

Closed and unlikely to be reopened. Kept in full because the Phase 3 tables are still the Cascade Lake reference data and the Phase 1 premise check is the project's founding result.

### Phase 1 report: the premise check

**Gate:** self-approved design doc; premise resolved with data; green scaffolded
repo; working harness with baselines wired in.

**Status: gate met. Kill/pivot condition triggered.**

#### Delivered

* `DESIGN.md` — literature review (cited), ecosystem survey with per-layer
  build-vs-reuse calls, full engine design, benchmark/test framework,
  self-scrutiny.
* Cargo workspace: `tensorcontract` (core), `tensorprimitives-tapp` (C ABI),
  `tensorprimitives-bench` (harness). CI (build/test/clippy/fmt/docs/MSRV +
  a scalar-fallback job), dual MIT/Apache-2.0, MSRV 1.75.
* Working engine, correct end-to-end (this is the Phase 2 gate, met early — see
  the Phase 2 report).
* Harness `tcbench` with `verify` / `premise` / `sweep` / `info`, TBLIS and
  OpenBLAS-TTGT baselines wired in, CSV output, GEMM roofline annotation, and
  stride-stress modes.
* Raw results in `bench-results/`.

#### The premise check

Hypothesis under test, from the brief:

> TBLIS underperforms on complex contractions, worst in memory-bound / awkward-stride
> cases, because interleaved-complex storage and the scatter/block-scatter packing
> compound and force more work onto the slow full-scatter (gather) path.

Method: 12 cases sampled evenly across TCCG's bandwidth-bound-to-compute-bound
ordering, at 64 MiB nominal tensor size, best of 3, single-threaded. For each,
measure the contraction and a same-shape vendor GEMM, in both a real and the
matching complex dtype. Report `eff = contraction / GEMM` and
`eff ratio = complex eff / real eff`.

Because a complex MAC is four real FMAs counted as 8 flops, the achievable GF/s
peak is the same number in both domains (confirmed: `dgemm` 96, `zgemm` 96).
So `eff ratio < 1` means a complex-specific penalty; `>= 1` refutes the thesis.

**Results — mean `eff ratio` over 12 cases:**

| engine | dtypes | stress | mean eff ratio |
|---|---|---|---|
| **TBLIS v1.3.0** (latest release) | f64 / c64 | none | **0.215** |
| TBLIS 2.0-dev | f64 / c64 | none | **1.060** |
| TBLIS 2.0-dev | f64 / c64 | ragged (`regA` 0.80–1.00) | **1.027** |
| TBLIS 2.0-dev | f64 / c64 | padded strided views | **1.059** |
| TBLIS 2.0-dev | f32 / c32 | none | **1.150** |
| TTGT | f64 / c64 | none | **1.170** |

**Ceiling-free cross-check — raw complex/real GF/s ratio for the same shape:**

| run | n | min | median | max | mean | below 1.0 |
|---|---|---|---|---|---|---|
| **TBLIS v1.3.0 f64→c64** | 12 | **0.19** | **0.33** | 1.15 | 0.40 | **11** |
| TBLIS 2.0-dev f64→c64 | 12 | 0.98 | 1.91 | 2.67 | 1.76 | 1 |
| TBLIS 2.0-dev f64→c64 ragged | 12 | 1.01 | 1.82 | 2.28 | 1.68 | 0 |
| TBLIS 2.0-dev f64→c64 padded | 12 | 1.03 | 1.90 | 2.53 | 1.76 | 0 |
| TBLIS 2.0-dev f32→c32 | 12 | 1.07 | 1.95 | 2.07 | 1.70 | 0 |
| TTGT f64→c64 | 12 | 1.10 | 2.12 | 2.62 | 1.94 | 0 |

#### Verdict: the observation is real, the explanation is not, and it is already fixed

The answer depends entirely on which TBLIS you measure, and the two differ by
almost a factor of five.

**Against v1.3.0, the latest stable release, the complex-weakness claim is
emphatically true.** Complex efficiency against the GEMM ceiling is 0.09–0.20
across every case, versus 0.37–1.08 for real: a mean `eff ratio` of **0.215**.

But the *mechanism* is not the one the brief proposes, and the diagnostic is
unmistakable. TBLIS 1.3.0's complex throughput is essentially **flat across
shapes** — 4.1 to 9.1 GF/s, a 2.2x spread — while its real throughput spans
6.2 to 48.4 GF/s, a 7.8x spread. A memory- or scatter-bound effect would track
shape. A flat ceiling means one fixed-throughput kernel is the bottleneck
regardless of what it is fed.

Reading `src/configs/*/config.hpp` in v1.3.0 confirms it directly. The
`TBLIS_CONFIG_GEMM_UKR` macro takes four slots, `(float, double, scomplex,
dcomplex)`:

```
skx1:        TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_6x16, bli_dgemm_asm_6x8,  _, _)
skx2:        TBLIS_CONFIG_GEMM_UKR(_,                  bli_dgemm_opt_6x32_l1, _, _)
haswell:     TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_24x4, bli_dgemm_asm_12x4, _, _)
zen:         TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_6x16, bli_dgemm_asm_6x8,  _, _)
knl:         TBLIS_CONFIG_GEMM_UKR(bli_sgemm_opt_30x16_knc, bli_dgemm_opt_30x8_knc, _, _)
sandybridge: TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_8x8,  bli_dgemm_asm_8x4,
                                   bli_cgemm_asm_8x4,  bli_zgemm_asm_4x4)
```

**Sandy Bridge is the only configuration with complex micro-kernels.** On every
post-2012 x86 target — Haswell, Zen, Skylake-X, KNL — TBLIS 1.x runs complex
tensor contraction on the generic templated fallback while real gets hand-tuned
BLIS assembly. That is the entire effect. It has nothing to do with
interleaved storage, nothing to do with scatter/gather, and nothing to do with
the block-scatter fast path: `regA = 1.00` on every case measured.

**Against 2.0-dev the claim is refuted.** Rebasing onto BLIS-as-framework
brings BLIS's 1m induced method, and complex immediately regains full shape
sensitivity (11.2–82.9 GF/s, a 7.4x spread matching real) and lands at or above
parity — mean `eff ratio` 1.06 in f64, 1.15 in f32, and still 1.03 when
irregular block scatter is forced. The upstream release notes for `v2.0-beta2`
say as much: "a major update … which incorporates BLIS as the core framework"
with improvements including complex number support.

**Why complex is not intrinsically disadvantaged.** The folklore reasoning ran:
complex data is 2x the bytes, scatter/gather is the bottleneck, therefore
complex suffers more. The missing term is arithmetic intensity. A complex MAC
does 4x the flops of a real MAC on 2x the bytes, so complex contraction has
**2x the arithmetic intensity** of the same-shape real contraction. Packing,
indexing and write-back costs are amortised over twice as much arithmetic. Once
a real complex kernel exists (2.0), the memory-bound shapes where complex was
predicted to be worst are where it looks *best*: `abcijk-ikmb-mjac` runs at
8.8 GF/s in f64 and 23.5 GF/s in c64; `abjcd-dkbac-jk` at 5.5 vs 11.2.

BLIS's 1m does inflate the packed A panel 2x (four reals per complex element in
"1e" format versus two in planar). That cost is real but is not on the critical
path at these shapes, and the intensity advantage swamps it.

#### What this means for the project

The gap the project set out to exploit **exists in the wild today** — anyone
using the packaged, released TBLIS for complex tensor contraction on modern x86
is getting roughly a fifth of the achievable throughput. But:

* it is a missing-kernel bug, not an algorithmic opening, so beating it proves
  nothing about planar packing;
* it is already closed upstream, and will disappear from the wild the moment
  2.0 ships;
* the correct opponent for any new complex method is 2.0/BLIS 1m, and against
  that opponent there is no complex-specific headroom to take.

So the planar-complex thesis is refuted as a *research* proposition, while the
practical observation that motivated it is validated as a *packaging* problem.
Both halves are worth reporting.

#### What the data says the real headroom is

Not complex — **low arithmetic intensity**, in either domain:

* TBLIS `eff` against the GEMM ceiling ranges from **0.34 to 1.05**. It is
  0.85–0.87 on the big compute-bound `ijkl` cases and collapses to 0.34–0.53 on
  small-`k` / skinny shapes (`abjcd-dkbac-jk`, `ajbc-ckba-jk`,
  `abcijk-*`, all with `k = 24`).
* The gap is worse in **f32** (mean `eff` ≈ 0.6) than f64, because the same
  overhead is amortised over half the bytes of arithmetic.
* TTGT is 2–4x behind TBLIS on those same low-intensity shapes (`eff` 0.15–0.35),
  confirming that materialising a transposed copy is what hurts — the original
  BSMTC insight, still valid.

So the defensible target is **small-`k` and skinny tensor contractions**, where
the best available transpose-free engine leaves 50–65% of the machine on the
table, in *both* domains. That is a larger and better-evidenced gap than the
one the project set out to close.

#### Kill/pivot condition

`DESIGN.md` §6 named this as the single most likely failure mode, and the
Phase 1 gate exists precisely to catch it before implementation is committed
to. Per the operating rules, this is escalated rather than worked around.
Options, with the evidence for each:

1. **Re-aim at low arithmetic intensity** (recommended). Keep everything built:
   the data model, index analysis, block-scatter machinery, TAPP surface,
   corpus and harness are all domain-agnostic and all still needed. Change the
   target from "complex vs real" to "small-`k` / skinny shapes", where TBLIS
   measurably gives up 50–65%. Plausible mechanisms, in order of expected
   value: fusing the `pc` loop so `C` is touched once instead of `K/KC` times;
   skipping packing of `A` entirely when the block-scatter is already regular
   and unit-stride (a "pack-free" fast path); dispatching to a
   small-`k`-specialised kernel; and the write-back fast path for regular
   blocks. Planar complex stays in the design because it is *free* and it is
   what makes `TAPP_CONJUGATE`, mixed real x complex operands and 3m natural —
   it is simply no longer the headline claim.
2. **Pursue 3m instead.** Untouched by this result: 3m's advantage is a 25%
   *flop* reduction, not a bandwidth one, and planar packing makes it cheap to
   build. Smaller, more speculative, and carries a numerical-stability caveat.
3. **Wrap TBLIS.** Honest answer if the goal is a usable Rust tensor
   contraction today, but no research contribution, and it keeps the C++
   dependency the brief wanted to remove.
4. **Stop.** The negative result is itself publishable, and the brief says so:
   there is no public systematic complex tensor-contraction benchmark, this
   repository now is one, and "complex contraction is not the weak spot; low
   arithmetic intensity is, and here is why" is a useful correction to
   circulating folklore.

**Recommendation: option 1**, with the Phase 1 negative result written up as a
standalone finding.

---

### Phase 2 report: a correct, framework-complete engine

**Gate:** numerically correct across the full matrix (shapes, permutations,
dtypes, traces, degenerate cases) vs oracle, TTGT, TBLIS. Performance measured
as a baseline, not a goal.

**Status: gate met.** Phase 2 was completed alongside Phase 1 because the
premise check needed a working engine to sit alongside the baselines.

Implemented: tensor data model; index analysis with folding; scatter and
block-scatter construction; planar-complex packing with conjugation folded in;
reference scalar micro-kernel; five-loop driver; scattered write-back with
`alpha`/`beta`/`op_C`/`op_D`; TAPP C-ABI export.

Correctness evidence:

* 1000 randomised contractions vs the brute-force oracle across
  `f32`/`f64`/`c32`/`c64`, each run under both tiny `(1,2,1)` blocking and the
  real blocking, covering free/contracted/Hadamard/isolated indices, repeated
  labels, random stride permutations, random conjugation masks, and
  `alpha`/`beta` including zero — all within `1e-11` (f64) / `2e-4` (f32).
* Targeted degenerate cases: empty contraction extent, zero-sized output,
  scalar output (full double contraction), negative strides via a reversed
  axis, all 16 conjugation flag combinations forced through multiple `K` blocks.
* Large pure-GEMM cases crossing the real `MC`/`KC`/`NC` boundaries with
  awkward remainders, in all four dtypes.
* Cross-implementation: all 49 corpus cases x 4 dtypes agree with **both** TBLIS
  and TTGT to `~2e-16` (f64/c64) and `~1.5e-7` (f32/c32), under `none`,
  `ragged` and `padded` stride stress.
* TAPP C ABI exercised end-to-end through the C entry points on a complex case.

Performance baseline: the micro-kernels are the portable scalar fallback
(Phase 3 was not reached), so the `planar` engine's absolute numbers are not
meaningful yet and are not reported as a result.

---

### Phase 2b report: three interchangeable complex methods

**Direction decision.** After the Phase 1 result, the chosen direction is to
keep all three induced-complex methods available and switchable, so that the
comparison can be made properly rather than argued from first principles. This
supersedes the four options listed at the end of the Phase 1 report.

#### What was built

`ComplexMethod::{Planar, OneM, ThreeM}`, selected per plan with
`Plan::with_complex_method` or globally with `TENSORCONTRACT_COMPLEX`.

The three share the *entire* engine except three things, each named in the
`Ukr` the method selects:

| | `a_pack` / `b_pack` | kernel | `tile_fmt` |
|---|---|---|---|
| planar | `Planar` / `Planar` | fused complex, 4 FMAs per k per output | `Planar` |
| 1m | `OneE` / `Planar` | plain real, `2*MR x NR` over `2*KC` | `OneM` |
| 3m | `ThreeM` / `ThreeM` | Karatsuba, 3 FMAs per k per output | `ThreeM` |

`crates/tensorcontract/src/driver.rs` contains no branch on the method at all —
it reads sliver widths, tile size and formats off the `Ukr`. That is what makes
the comparison fair: same index analysis, same scatter traversal, same loop
arithmetic, same write-back scatter.

*Decisions introduced here: D13, D14, D15, D16 — stated in [Design decisions](#design-decisions).*

#### Correctness

* The randomised oracle sweep now runs **every complex problem under all three
  methods** — 300 problems x 3 methods x 2 blockings for `c64`, likewise
  `c32` — plus the all-16-conjugation-masks test and the large blocking-boundary
  cases, each across all three.
* `kernel::tests` checks each method's kernel directly against the definition,
  including a packing helper that builds panels in each `PackFormat` and a
  reader for each `TileFormat`, so a mis-specified format is caught at the
  kernel boundary rather than end to end.
* All 49 corpus cases agree with **both** TBLIS and TTGT to ~2e-16 under each
  of the three methods (`TENSORCONTRACT_COMPLEX=planar|1m|3m tcbench verify`).

#### Measurement, and why it does not yet mean much

`tcbench premise --size 16 --engines planar,1m,3m --dtype f64,c64`, 12 cases,
single core. Geometric-mean `c64` throughput relative to planar:

| method | relative c64 GF/s | mean complex/real efficiency ratio |
|---|---|---|
| planar | 1.000 | 0.726 |
| 1m | 1.086 | 0.783 |
| 3m | 1.129 | 0.817 |

**This ranking is an artifact of the scalar kernels and must not be quoted as
a result.** All three currently run portable scalar loops, and the numbers
mostly reflect how well LLVM auto-vectorises three different loop shapes: 1m's
inner loop is a plain real GEMM kernel, which LLVM handles best, while planar's
entire argument is *fewer shuffles in a hand-written SIMD kernel* — which does
not exist yet. 3m's edge is more likely real, since a 25% flop reduction
survives any kernel quality, but even that needs confirming.

The honest three-way comparison is the Phase 3 gate.

Raw data: `bench-results/methods-f64c64.csv`.

---

### Phase 3 report: vectorised micro-kernels

**Gate:** correctness unchanged; single-core throughput against the baselines
and a GEMM roofline; **an honest three-way planar/1m/3m comparison with real
kernels.** All three met. This is the measurement the project exists to
produce.

#### What was built

`crates/tensorcontract/src/kernel/x86.rs`, previously four `None`s, now holds
AVX-512 kernels for all four shapes — `real`, `planar`, `onem`, `threem` —
macro-generated over `(MV, NR)` const generics for both `f32` and `f64`,
selected by runtime `avx512f` detection with the scalar path untouched behind
`TENSORCONTRACT_KERNEL=scalar`. `MV` is the number of vector registers an `A`
sliver occupies per plane per k-step.

**Nothing outside that file changed.** The `Ukr` contract carried the new
kernels unmodified, which is the design claim from Phase 2 discharged.

Also added: `examples/kernel_shapes`, a register-block sweep used to choose the
shapes (D19), and `scripts/phase3-bench.sh`, which reproduces this entire
report from a clean checkout given the two TBLIS prefixes.

#### Correctness

Unchanged, and checked at three levels:

* `kernel::tests` validates each selected kernel directly against the
  mathematical definition through its own `PackFormat`/`TileFormat`. Every
  kernel here passed on first execution.
* `cargo test --workspace --release` green, and green again under
  `TENSORCONTRACT_KERNEL=scalar`.
* `tcbench verify` — all 49 corpus cases x `f32`/`f64`/`c32`/`c64` x all three
  complex methods, against **both** TBLIS 2.0-dev and TTGT. `f32`/`f64` agree
  exactly (`0.0e0`); `c32` to ~1.3e-7, `c64` to ~2.5e-16.

#### The three-way comparison

Full 49-case TCCG corpus, 64 MiB nominal tensors, single core, geometric mean
GF/s. `tblis` is 2.0-dev at `555320c`. Unperturbed corpus, so `regA = 1.00`
throughout and the gather path is not involved.

| engine | c64 | vs planar | c32 | vs planar |
|---|---|---|---|---|
| **planar** | **43.6** | 1.000 | **79.9** | 1.000 |
| 1m | 42.2 | 0.967 | 78.2 | 0.979 |
| 3m | 41.7 | 0.956 | 73.6 | 0.921 |
| tblis 2.0-dev | 44.1 | 1.011 | 71.2 | 0.891 |
| ttgt | 23.2 | 0.532 | 45.3 | 0.567 |

**Planar wins, in both precisions, but by 3–8% rather than by a lot.** The
scalar-kernel ranking of Phase 2b (planar 1.00, 1m 1.09, 3m 1.13) is now
reversed, exactly as that report predicted it would be once the ranking stopped
measuring LLVM's auto-vectoriser.

**But the aggregate hides the actual finding, which is that the ranking is
shape-dependent and inverts.** Splitting the same corpus by arithmetic
intensity:

| c64 subset | planar | 1m | 3m | tblis |
|---|---|---|---|---|
| `min(n,k) > 64` — compute-bound, 25 cases | **69.0** | 65.4 | 62.2 | 71.3 |
| `min(n,k) <= 64` — memory-bound, 24 cases | 27.0 | 26.7 | **27.4** | 26.7 |

and the micro-kernel sweep says why. Timing each kernel in isolation while
sweeping `kc`, which decides whether the `A` sliver is an L1 resident or an L2
stream (`bench-results/phase3-kernel-shapes.txt`, `f64`):

| method | best shape | GF/s at `kc=64` | GF/s at `kc=256` | bytes / useful flop |
|---|---|---|---|---|
| planar | `16x6` | 107.7 | **102.8** | **0.46** |
| 1m | `12x8` | 93.2 | 91.1 | 0.67 |
| 3m | `8x10` | **117.1** | 87.8 | 0.68 |

So:

1. **3m's 25% flop saving is real and it is not free.** With both panels
   L1-resident 3m is the fastest of the three, by roughly the margin the flop
   count predicts. At the `kc = 256` the engine actually uses, it is the
   slowest. 3m loads *three* planes of both operands to save one of four
   products, so per useful flop it moves 1.5x planar's bytes; once the kernel
   stops being FMA-issue-bound that is what decides it.
2. **Planar wins on bytes, not on shuffles.** Its advantage over 1m is that
   "1e" packing carries four reals per complex element of `A` against planar's
   two. The original argument for planar — fewer in-register shuffles — is not
   what the data rewards, because at these shapes none of the three methods is
   shuffle-limited: LLVM emits `vbroadcastsd` as its own uop and every winning
   shape is FMA-issue-bound (verified in the disassembly).
3. **When the contraction is memory-bound the kernel's byte traffic stops
   mattering and its flop count starts to.** That is the inversion above, and
   it is a direct argument for the shape-dispatch item already listed in
   Phase 4.

Every candidate needing more than 32 live vector registers loses 30–50%. That
cliff, not the flop count, is what bounds 3m's usable shapes: it needs three
accumulator planes, so it cannot be given a block wide enough to amortise its
loads.

#### Against the baselines

Full corpus, 64 MiB, geometric mean GF/s, `planar` for complex:

| dtype | this engine | tblis 2.0-dev | ttgt | best-of-49 count (ours / tblis / ttgt) |
|---|---|---|---|---|
| f64 | **30.5** | 27.0 | 13.1 | 22 / 16 / 11 |
| f32 | **54.6** | 42.7 | 24.4 | 23 / 13 / 13 |
| c64 | 43.6 | **44.1** | 23.2 | 13 / 12 / 12 |
| c32 | **79.9** | 71.2 | 45.3 | 10 / 11 / 16 |

Against **TBLIS v1.3.0** (`c4f81e0`, still the latest stable release — no new
tag has appeared since Phase 1), 12-case premise set at 200 MiB: `c64` planar
**43.8** vs **7.3** GF/s, a 6.0x gap, and `f64` 29.2 vs 21.5. The Phase 1
finding that 1.x has no complex micro-kernel outside Sandy Bridge reproduces
unchanged.

The rebuilt TBLIS 2.0-dev reproduces Phase 1's headline to three digits — mean
complex-over-real efficiency ratio **1.058** here against 1.06 in Phase 1 —
which is the check that the rebuilt baseline is the same baseline.

Complex-over-real efficiency ratio against a same-shape GEMM ceiling, 12-case
premise set at 200 MiB: planar 0.976 / 1m 0.931 / 3m 0.979 (`c64`), and 1.156 /
1.144 / 1.134 (`c32`). Against the roofline directly, the engine reaches
**0.81–0.84** of a same-shape `zgemm` on the large compute-bound cases and
**0.69–0.70** of `dgemm` — complex is the *easier* domain for us too, for the
arithmetic-intensity reason established in Phase 1.

#### Irregular strides

`--stress ragged`, `c64`, 64 MiB. 40 of 49 cases become genuinely irregular
(mean observed `regA = 0.656`; the remaining 9 stay at 1.00). On those 40:

| engine | GF/s |
|---|---|
| planar | **43.7** |
| 3m | 42.5 |
| 1m | 42.3 |
| tblis 2.0-dev | 39.4 |

An 11% lead for the transpose-free path where strides are awkward — the one
regime where block-scatter is doing work a TTGT-style engine cannot avoid
paying for. Note these numbers are *not* comparable to the unstressed table
above: `--stress ragged` changes the extents, so it is a different set of
shapes, not the same shapes made harder.

#### A concrete 2x defect, localised

Nine corpus cases of the form `abcijk-{ij,ik,jk}m{a,b,c}-*` have identical
`m`, `n`, `k` and `regA = regB = 1.00`, and differ only in which output axis
leads the `M` group. Throughput is monotone in that axis's stride in `D`,
9 cases out of 9:

| `M` leading axis | its stride in `D` | planar `c64` GF/s |
|---|---|---|
| `a` | 1 | 40.5, 40.5, 40.5 |
| `b` | `n_a` | 36.2, 36.4, 36.6 |
| `c` | `n_a * n_b` | 17.7, 17.8, 17.8 |

Packing is identical across the nine (same operand layouts, fully regular block
scatter), so the cost is in the **scattered write-back**. `perf stat` on three
of them, same binary, same reps, confirms it and identifies the mechanism:

| `M` leading axis | cycles | instructions | IPC | dTLB store misses | **L2 demand misses** |
|---|---|---|---|---|---|
| `a` (stride 1) | 5.19e9 | 6.50e9 | 1.25 | 4.11e6 | 27.5e6 |
| `b` (stride `n_a`) | 5.48e9 | 6.46e9 | 1.18 | 4.21e6 | 33.0e6 |
| `c` (stride `n_a*n_b`) | **8.32e9** | 6.49e9 | **0.78** | 4.07e6 | **62.9e6** |

Instruction count and retired stores are flat to within 1%, so this is purely
stalling, not extra work. **The first-guess mechanism was wrong**: DTLB misses
are flat, so it is not TLB pressure. It is L2 miss traffic — 2.3x more of it —
from poor cache-line utilisation and reuse distance on the `C`/`D` update when
consecutive tile rows are far apart in the output.

A controlled follow-up: rebuilding *planar alone* with 3m's wider `8x10` tile
instead of its own `16x6`, changing nothing else, moves these cases the way the
mechanism predicts and moves the others back:

| case | `16x6` | `8x10` | |
|---|---|---|---|
| `abcijk-ijmc-mkab` — `D` contiguous along `N` | 17.8 | **19.8** | +11% |
| `abcijk-ijmb-mkac` — intermediate | 36.2 | 33.8 | −7% |
| `abcijk-ijma-mkbc` — `D` contiguous along `M` | 40.5 | 35.9 | −11% |
| `ijkl-imjn-lnkm` — compute-bound control | 79.4 | 79.4 | 0% |

So micro-tile **aspect ratio should follow the output's stride pattern** — a
real lever, worth about ±11%, and free to apply since the kernels are already
parameterised over `(MV, NR)`. It also explains why 3m beats planar by 1.36x on
exactly these three cases while losing everywhere else.

But ±11% does not explain a 2x. Aspect ratio is a tuning knob; the 2x is the
write-back's L2 traffic itself, and closing it needs the "vectorised write-back
for regular blocks" item on the Phase 4 list — or blocking the `jr`/`ir` loops
against the output's layout rather than only against the packed panels.

#### Assumptions added

| # | Assumption | Status |
|---|---|---|
| A7 | The three complex methods differ mainly in flop count. | **Refuted.** They differ mainly in bytes moved per useful flop, and that is what decides the ranking at realistic `kc`. Flop count decides it only when the panels are L1-resident or the contraction is memory-bound. |
| A8 | One register block per method is enough. | **Refuted, and it matters.** The best shape depends on the method, the element type and `kc`, and neighbouring shapes differ by 30–50% across the 32-register cliff. |
| A9 | Absolute performance is meaningful now that kernels are vectorised. | Adopted. It was explicitly not meaningful before this phase. |

#### What is *not* done

* **No AVX2 path.** Dispatch is AVX-512 or the scalar fallback. The macro takes
  it without restructuring; deferred to Phase 5's multi-arch work, since the
  reference machine is AVX-512 and the gate is a comparison on it.
* **Blocking is still the Phase 2 heuristic.** `MC`/`KC`/`NC` come from a fixed
  cache-budget rule, never swept. Given that the whole Phase 3 result turns on
  where the `A` sliver lives, `KC` in particular is now known to be a
  first-order parameter rather than a detail. Phase 4.
* **Still single-threaded.**

> **All three have since been addressed** and this list is historical: AVX2
> kernels landed in the Phase 4/5 interlude and their register blocks are
> measured (part 11); `MC`/`KC`/`NC` were swept on two machines and item 2 is
> closed with a negative result (part 7); threading is built and measured on seven
> nodes (parts 8, 8b, 12, 16), though the thread count is still 1 by default.

Raw data: `bench-results/phase3-*.csv`, transcript in
`bench-results/phase3-log.txt`, kernel sweep in
`bench-results/phase3-kernel-shapes.txt`. Reproduce with
`scripts/phase3-bench.sh`.
