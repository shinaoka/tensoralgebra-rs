# Results — what is measured, and how well

The single source for every performance number this project quotes. The README
and the CHANGELOG's confidence table both cite this file rather than restating
it, so the three cannot drift.

Read [`measurement-rules.md`](measurement-rules.md) before designing a
measurement, and [`refuted.md`](refuted.md) before proposing an idea.

**The one rule that governs all of it: no number measured on one machine may be
compared with a number measured on another.** Every ratio here is within-session,
against a noise floor derived in that same session.

---

## 1. The headline finding: which TBLIS you measure changes the answer by 5x

Efficiency here is contraction GF/s divided by same-shape vendor GEMM GF/s.
Because a complex MAC is exactly four real FMAs, the achievable ceiling is the
same in both domains (measured: `dgemm` 96 GF/s, `zgemm` 96 GF/s), so the number
below — **each engine's own complex efficiency divided by its own real
efficiency** — is 1.0 when that engine treats complex data as well as real.
Single core, Xeon Gold 6244, 12 TCCG contractions.

| | complex ÷ real efficiency, **within the same engine** |
|---|---|
| **TBLIS v1.3.0** — latest *stable* release | **0.215** |
| TBLIS 2.0-dev | **1.06** |
| TBLIS 2.0-dev, irregular strides forced | **1.03** |
| TBLIS 2.0-dev, `f32`→`c32` | **1.15** |
| TTGT (OpenBLAS) | **1.17** |

**This is not a speed comparison between engines and must not be read as one.**
An engine can score well by being uniformly bad. TTGT has the *highest* ratio and
is by a wide margin the slowest of the three: on those same 12 cases its median
`c64` throughput is 17.4 GF/s against TBLIS 2.0's 42.4. Its ratio is high because
its **real** path is worse still — 6.8 GF/s median in `f64`, where the
transpose-GEMM-transpose overhead dominates a small GEMM — and complex, with twice
the arithmetic intensity, amortises that overhead better. The ratio answers
exactly one question: *is complex penalised relative to real, in this engine?*

Against the released TBLIS, complex contraction really is ~5x less efficient than
real — but not for the reason usually proposed. The diagnostic: v1.3.0's complex
throughput is nearly **flat across shapes** (4.1–9.1 GF/s, a 2.2x spread) while
its real throughput spans 6.2–48.4. A memory-bound effect would track shape; a
flat ceiling means one fixed kernel is the bottleneck.

`src/configs/*/config.hpp` in v1.3.0 confirms it. `TBLIS_CONFIG_GEMM_UKR` takes
`(float, double, scomplex, dcomplex)`, and **Sandy Bridge is the only
configuration that fills the complex slots.** On Haswell, Zen, Skylake-X and KNL,
TBLIS 1.x runs complex on the generic templated fallback while real gets
hand-tuned BLIS assembly.

TBLIS 2.0 fixes this by adopting BLIS as its core framework, which brings the 1m
induced method. Complex immediately regains full shape sensitivity (11.2–82.9
GF/s, matching real) and reaches parity. **So the opportunity is real in the wild
today, already closed upstream, and not evidence for any algorithmic claim.**

**Confidence:** `settled`, and replicated on a second microarchitecture — v1.3.0's
0.215 came back identically on Ice Lake, because the ratio is a property of the
software. Evidence: `bench-results/phase3-premise-*.csv`,
`bench-results/worker6156-icelake/compare-tblis-skx/` (job 6753260).

## 2. The displaced finding, which is the better one

**The real headroom is low arithmetic intensity in either domain.** TBLIS 2.0's
efficiency against the GEMM ceiling ranges from 0.85 on large compute-bound
contractions down to **0.34** on small-`k` skinny ones.

And complex is not intrinsically disadvantaged: 4x the flops on 2x the bytes is
**twice the arithmetic intensity**, so packing and indexing overheads amortise
*better*. Measured on this engine, complex throughput is *higher* than real on
the same shapes — 62.0 against 44.7 GF/s in double, 121.0 against 74.8 in single,
at the median. That is a within-run ratio, so it survives where absolute numbers
do not: 1.449 (`c64`/`f64`) and 1.474 (`c32`/`f32`) on Ice Lake against 1.416 and
1.432 on Cascade Lake — the same result on two microarchitectures.

## 3. Absolute throughput, Ice Lake

Single core, **Intel Ice Lake-SP, AVX-512** (2 x 32 cores, SMT off; 48 KiB 12-way
L1d, 1280 KiB 20-way L2 per core, 48 MiB L3 per socket — the geometry the engine
itself probed), full 49-case TCCG corpus at 64 MiB nominal size, planar method,
GF/s counting 2 flops per real MAC and 8 per complex one. **Measured 2026-08-04/05**
on an exclusive cluster node against **TBLIS 2.0-dev** (`develop` @ `555320c`) and
**TBLIS v1.3.0** in the same runs. Raw data:
[`bench-results/worker6156-icelake/`](https://github.com/tensor4all/tprims-rs/blob/0fc06f4578e20017e510807ccaaa72ab4bab08f4/tensorprimitives/bench-results/README.md).

| dtype | min | median | geomean | max |
|---|---|---|---|---|
| `f32` | 11.0 | 74.8 | 77.2 | **168.8** |
| `f64` | 7.1 | 44.7 | 43.7 | **71.6** |
| `c32` | 25.5 | 121.0 | 113.9 | **179.3** |
| `c64` | 15.7 | 62.0 | 63.3 | **91.1** |

The corpus deliberately spans compute-bound and badly memory-bound contractions,
which is why min and max differ by an order of magnitude; the worst cases are
memory-bound for every engine measured, TBLIS included.

Against the baselines, median over the 12-case premise set at 200 MiB, same run,
same core, everything single-threaded:

| dtype | this engine | TBLIS 2.0-dev | TTGT (OpenBLAS) | engine ÷ TBLIS 2.0 |
|---|---|---|---|---|
| `f64` | 46.5 | 30.8 | 14.0 | **1.51x** |
| `c64` | 64.0 | 48.8 | 25.9 | **1.31x** |
| `f32` | 81.1 | 45.6 | 25.9 | **1.78x** |
| `c32` | 123.2 | 88.3 | 49.2 | **1.40x** |

Over the full corpus the engine is 1.95–2.20x TTGT in every dtype.

**How well this is known.** The whole set was run **twice**, in separate
allocations three hours apart, agreeing across all twenty dtype × engine columns
to **0.997–1.003**. Within each run a repeat arm 2.5 h from its original
reproduces it to 0.998–1.001, with **0 of 980** case points outside ±6%. A
discarded warm-up arm precedes everything, and every arm recorded occupancy for
its whole L3 domain — all came back with no co-tenants.

### What these numbers are not

* **They are not the improvement the optimisation work bought.** They are Ice
  Lake; the comparable previous set is Cascade Lake. Two things changed at once.
  The Phase 4 gains are real and separately measured — the write-back orientation
  fix is +12–17% of corpus geomean in all four dtypes on the reference machine —
  but attributing any part of the table to them would be wrong. See
  [`open-questions.md`](open-questions.md) §1.
* **The engine is handicapped here.** Its register blocks were chosen on Cascade
  Lake and three of the eight are 9–13% off on Ice Lake; its complex-method
  ranking does not transfer either. It wins these columns while running shapes
  picked for a different microarchitecture.

## 4. Absolute throughput, Apple M3 Max

Single core, **Apple M3 Max, NEON** (12 P-cores in 2 clusters of 6, no SMT; 128 KiB
L1d per core, 16 MiB L2 per cluster, no L3 Darwin will name), 12-case premise set at
64 MiB nominal size, GF/s on the same counting convention as §3. **Measured
2026-08-06** against **TBLIS 2.0-dev** (`develop` @ `555320c`, `BLIS_CONFIG_FAMILY=arm64`,
BLIS selecting its `firestorm` sub-configuration) and OpenBLAS 0.3.34. Raw data:
[`bench-results/CKF6QCDVPD-m3max/`](https://github.com/tensor4all/tprims-rs/blob/0fc06f4578e20017e510807ccaaa72ab4bab08f4/tensorprimitives/bench-results/README.md).

**Read every number in this section with two caveats.** It is **not a pinned
measurement** — Darwin has no CPU affinity API, so nothing here is the pinned
single-core measurement §3 is. And the roofline here is **OpenBLAS, not Accelerate**:
Accelerate's TTGT reaches a geomean 79.2 GF/s in `f64` and peaks at 334.8, which is
**5.2x** a P-core's 64.8 GF/s NEON FMA peak, because its GEMM reaches Apple's
undocumented AMX coprocessor. That is not a NEON ceiling and a ratio against it is
not an efficiency figure.

| | engine ÷ TTGT | engine ÷ TBLIS 2.0-dev |
|---|---|---|
| `f64` | **1.16x** | **1.00x** |
| `c64`, planar | — | 0.98x |
| `c64`, 3m | **1.42x** | **1.12x** |

Best-shape efficiency against the 64.8 GF/s NEON FMA peak: engine **81.4%**,
OpenBLAS 87.8%, TBLIS 89.8%.

**What the NEON kernel is worth, measured end to end**, and this is the one A/B the
x86 kernels cannot have — an AVX-512 machine cannot un-have its own vector unit,
whereas here `TENSORCONTRACT_KERNEL` switches a real kernel against the portable one
inside a single binary:

| | scalar | NEON | ratio |
|---|---|---|---|
| `f64`, all three methods | 19.90 | 36.60 | **1.840** |
| `c64` planar | 24.34 | 44.46 | 1.826 |
| `c64` 1m | 23.31 | 41.74 | 1.791 |
| `c64` 3m | 29.85 | 50.45 | 1.690 |

Four baseline columns held at 0.996–1.006 across the same arms as controls, against
an engine-column floor of **1.000–1.008** — so the 1.84 is ~80x the floor.

**The portable path cost 1.5x, not an order of magnitude**, and that corrects a claim
this project published unmeasured. Before the kernel the engine ran at 0.63x of
OpenBLAS-TTGT and 0.54x of TBLIS 2.0-dev in `f64`. The reason was one instruction
form rather than a missing kernel: LLVM vectorises `kernel::scalar` to NEON unasked,
but will not contract `acc += a * b` into `fmla`, because that is two roundings and
IEEE forbids it — so the path ran two instructions per multiply-accumulate where the
machine offers one, halving its available ceiling to 32.4 GF/s. It then reached
**80.5% of that halved ceiling**, against the NEON kernel's 81.4% of the full one.
Same efficiency, twice the ceiling.

**On this machine 3m is the fastest complex method** — 1.135x planar, and it is the
arm that passes TBLIS. That is a **fourth** per-microarchitecture ordering (§5), and
the only one taken with a tuned kernel on its own machine. It also scopes "3m is not
a candidate default" to x86.

### What these numbers are not

* **Not pinned, and not comparable with §3.** Different machine, different ISA, and
  no affinity control.
* **Not a corpus result.** 12 premise shapes, not the 49-case corpus, and there is
  **no `ragged` arm** — so nothing here is about the irregular path, which on this
  machine is very nearly the only source of irregularity there is.
* **Nothing per-case finer than ~10%**, and nothing about threads.
* **Cache blocking here is still the Cascade Lake constants.** Only the register
  blocks were tuned to this machine.

## 5. The three complex methods

**Planar wins the corpus geometric mean in both precisions on both AVX-512
machines measured.** That much travels.

**The ordering below it, and the inversion on memory-bound shapes, do not.** On
Cascade Lake 3m was fastest on memory-bound shapes, where its 25% flop saving
pays. On Ice Lake 3m is last in every column, wins **0 of 49** cases, and sits at
0.694 (`c64`) and 0.744 (`c32`) against planar where Cascade Lake had 0.956 and
0.921.

That was first recorded as confounded with the wrong register blocks. **It is
not.** 3m ships the shape the Ice Lake kernel sweep names as 3m's *own* best, in
both precisions, so there is no better shape to give it; the collapse is uniform
across 3m's entire shape space. Even at the L1-resident depth where the flop
saving is supposed to pay, 3m leads planar by 10–16% on Cascade Lake and trails
by 34–43% on Ice Lake.

So the *accounting* holds everywhere — 3 products against 4, 3 planes of both
operands against 2, and bytes moved per useful flop is the deciding quantity —
while the claim that the saving **pays** in a nameable regime is a Cascade Lake
result. On AVX2 the *kernel-level* ordering differs again, putting 3m first in
`f32`/`c32`; that is a kernel measurement with panels packed and hot, not a corpus
ranking, and the corpus-level AVX2 comparison has not been made.

**And on the M3 Max 3m wins outright** — 1.135x planar with the NEON kernel, and it
is the arm that reaches 1.12x TBLIS 2.0-dev where planar reaches 0.98x (§4). That is
a **fourth** ordering from four machines, and the only one taken with a kernel tuned
to the machine it ran on. It also localises the advice: "treat 3m as the method that
makes the comparison honest rather than a candidate default" is an **x86** statement,
not a general one. What generalises is that *which resource binds* decides the
ranking, and that the answer is per-microarchitecture every time it has been checked.

**No ranking table is reproduced here on purpose.** Re-measure before quoting one.

## 6. What is tuned, and what is not

The most useful thing measurement established is where the tuning stops.

**Register blocks are per-microarchitecture, not per-ISA** — assumed the other way
round, and dispatch still selects by instruction set alone. Cascade Lake and Ice
Lake, same ISA and same 32 registers, disagree by up to 13% on three of eight
shipped shapes, each preferring the other's loser by ~9%: Ice Lake's 48 KiB 12-way
L1d accommodates an accumulator footprint Cascade Lake's 32 KiB 8-way does not. So
on any Ice Lake machine this engine runs a `real` kernel 9.2% off its own optimum.
AVX-512 blocks are measured on Cascade Lake, AVX2 on Zen2 (where all eight were
confirmed winners), NEON on an M3 Max, and no set claims anything about anywhere
else.

**Only the NEON menu was measured with an error bar, and that changed its
conclusions.** Three arms of the sweep instead of the single arm AVX-512 and AVX2
were each chosen from. Six of the eight winners then turned out to be **ties inside
the session's own spread** (1.89% `f64`, 3.64% `f32` p90) and are recorded as ties
rather than decisions; three candidate shapes were **unstable rather than slow** —
one reading 37.8 / 54.9 / 54.8 GF/s across arms, which is a shape doing two
different things and must not ship at any mean. The register budget that seeded the
menu named the wrong winner in **three of eight** columns, including one where it
agreed with BLIS's own hand-tuned `armv8a_asm_6x8` and the machine preferred a shape
the budget calls over-limit. **The consequence for the other two menus is that they
carry the same unknown error bar**, because a single arm cannot report one.

**Cache blocking is fitted to the reference machine, and swept.** `KC` is
first-order — it decides whether the `A` sliver is an L1 resident or an L2 stream.
But there is no few-percent win left in `MC`/`KC`/`NC` there: `kc = 384` is the
measured optimum and the shipped 256 is close, `MC` is a plateau a sixteenfold
range moves by at most 3%, and a deeper coupled setting that looked like +3% on
two machines failed its end-to-end A/B. Closed on evidence. Says nothing about a
different hierarchy.

**The analytical blocking model is off because it lost**, not because it is
unmeasured. Built to solve exactly the portability problem above, it was worse in
11 of 12 columns by up to 7.2% on the first unseen machine, and costs 14–34% on
the reference one. Its `kc` sinks it on one machine and its `mc` on the other —
both halves wrong, in different places. The cache probing it introduced is worth
having and is still used.

**Threading is off by default, and not for lack of data.** Scaling is measured on
seven nodes and is strongly topology-dependent: Zen2 saturates by 16–32 and
declines at 64; Ice Lake reaches 10.6x (`f64`) and 15.3x (`c64` 3m) on 32 cores
and is still climbing. Threads are spawned per call at ~20–36 µs each, which is
the whole story below ~1 MiB — there 64 threads run **1.2–10x slower than
serial** — and the optimal count walks 4 → 8 → 16 → 32 → 64 across 0.25 → 64 MiB,
so a fixed non-1 default is wrong at every size but one.

**The domain-aware partition is on by default** (D44). `Plan::partition` now knows
how many L3 domains the thread set spans, and *gating* its early return on that —
rather than removing it — is worth **1.13 corpus geomean at 64 Zen2 threads** and
provably changes nothing on a one-L3-per-socket machine (0 of 392 cases moved).

**The thread pool helps enormously on one machine class and hurts on another.**
`TENSORCONTRACT_POOL=on` is worth up to **11.6x** on 64 Zen2 cores at 0.25 MiB
(2.0–2.8x at 16 MiB) and up to **2.5x slower** on 32 Ice Lake cores sharing one
48 MiB L3. It stays off, and D53's recommendation to default it on is withdrawn.
Suspected cause in [`open-questions.md`](open-questions.md) §2. A thread-count
*amortisation guard* was built and is **not shipped** — it rescues an
over-threaded caller while costing a correctly-threaded one 10–39%, and on top of
the pool it is pure loss. `batch::contract_batched` pays one spawn set per batch
and is **not yet measured**.

Results are bitwise identical at every thread count and every partition, so none
of this is ever a correctness or accuracy decision.

### One known defect, stated because it ships

In `planar` `f32`/`c32` the register block is `32x6`, where the Phase 3 sweep's
own output names `32x5` as **7.8% faster** at the operating `kc`. The shipped
shape appears to have been picked by a bytes-per-flop model over the measurement
sitting next to it.

It is not fixed, but it *is* testable, which it was not when found: the menu was
keyed by `MR`, and `32x5` shares its `MR` with `32x6`, so no runtime switch could
reach it. Re-keying by position fixed that — `TENSORCONTRACT_ROWBLOCK=idx=3`
selects `32x5` end to end. It stays unfixed on purpose, because a kernel margin is
not a corpus margin (`NR` also changes the `jr` loop count and the packed sliver
geometry), so the corpus effect must be measured before the default moves.

It was found inside committed raw output months after the fact, at no machine
cost. That is the argument for committing raw output.

### Not measured, and so not claimed

* Absolute throughput on the Cascade Lake reference machine *since the Phase 4
  work landed*, so **what that work bought end to end is unmeasured**.
* The corpus-level method ranking on AVX2.
* **Anything on aarch64 beyond the 12 premise shapes** — no 49-case corpus sweep, no
  `ragged` arm, no per-case spread, and no threading (§4). And nothing about any
  aarch64 part other than the M3 Max, because register blocks are
  per-microarchitecture.
* Anything on non-x86, non-aarch64 hardware.
* Cache blocking anywhere but Cascade Lake; the aarch64 path runs its constants.
* The batched API, on any machine.

## 7. A note on the corpus

The TCCG corpus rounds every stride-1 extent up to a multiple of 24. That is
regular only at a register block which *divides* 24 — true of the blocks TBLIS
uses in §1, and **not** true of this engine's `f32`/`c32` blocks (`MR` 16, 32,
48), where `reg_a` falls below 1.0 on **42.9%** of case-dtype-methods at the
64 MiB size used here, a ninth of them *entirely*.

So the corpus does exercise the gather path here, periodically. Two qualifiers
travel with that number: the fraction moves with the tensor size, because the
extents do, and it is 11.7% on an AVX2 machine where `MR = 8` for `f64` does
divide 24. **Any awkward-stride claim must name its `--stress` mode and quote the
observed `reg_a`.** What the corpus cannot produce is *aperiodic* irregularity;
that is what `--stress ragged` is for.

"The corpus is fully regular" is false, and repeating it steered conclusions for
three phases.
