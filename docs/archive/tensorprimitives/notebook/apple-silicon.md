## Apple Silicon: a new ISA, and what the portable path cost

Stage A measured the platform before writing a line of it; Stage B built the NEON
kernel and measured what it bought. The chapter is kept whole rather than split
across `baselines.md` and `kernels.md` because the parts depend on each other: the
baseline choice is what makes the kernel result readable, and the cache probe is
what makes the blocking legible.

**Nothing in this chapter is a pinned measurement.** Darwin has no CPU affinity
API, so no number here is the pinned single-core measurement the rest of the
notebook assumes. Each part states its own floor.

### Part 20: Apple Silicon, and what the portable path actually costs

**2026-08-06, `CKF6QCDVPD` (Apple M3 Max), `bench-results/CKF6QCDVPD-m3max/`.**
The first non-x86 measurement in the project. Stage A of the plan: make an
aarch64 run honest, then measure it. Stage B — a NEON micro-kernel — is **not
started**; see "Resume here".

#### The result that changes a published claim

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
case at one size.

> **Superseded on the same day — do not quote 0.46.** Part 22 supplies the 64
> MiB TBLIS number this paragraph said was missing: over 12 shapes the scalar
> path is **0.54** of TBLIS 2.0-dev, and the NEON path is **1.00**. Direction
> right, magnitude 17% off, which is exactly the check A46 asks for and exactly
> why this figure was labelled provisional rather than promoted.

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

#### 3m wins here — and this is a statement about the *portable path*, not the machine

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

#### The floor, and what this machine cannot give

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

#### Three things that were reporting fiction, and one that was hiding

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

#### The corpus really is regular here, and the 42.9% figure is an x86 artefact

`reg_a = 1.000` on **784 of 784** case-dtype-method-arms at `--size 64`, and
`reg_b` likewise. CLAUDE.md's 42.9%-below-1.0 needs `MR` in {16, 32, 48}, none of
which divides the 24 that TCCG rounds every stride-1 extent up to; the scalar
path's `MR` is **4**, which does. So **`--stress ragged` is the only source of
irregularity on this machine**, and any write-back or gather-path claim taken here
without it is a claim about the fully regular case.

#### What was portable, for the record

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

### Part 21: The NEON register blocks, measured

**2026-08-06, `CKF6QCDVPD` (Apple M3 Max),
`bench-results/CKF6QCDVPD-m3max/kernel-shapes*.txt`.** Stage B's calibration
step. The NEON menus shipped a few hours earlier were derived from the
32-register budget alone and said so; this replaces them with three arms of
`examples/kernel_shapes` and a floor.

#### Three arms, because one has no floor

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

#### What shipped, and six of eight are ties

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

#### The budget was a good filter and a bad chooser

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

#### The interaction the sweep cannot see, and why the default is safe anyway

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

### Part 22: What the NEON kernel is worth, and it closes the Apple gap

**2026-08-06, `CKF6QCDVPD` (Apple M3 Max), `bench-results/CKF6QCDVPD-m3max/neon-ab/`.**
Four arms, 47 minutes, `scripts/macos-neon-ab.sh`. Stage B's end-to-end
measurement, run *after* the shapes were calibrated so that it measures what
ships (A20).

#### The design, and it is better than the equivalent x86 A/B

One binary, four arms — `warm` (discarded), `A-scalar`, `B-neon`, `A2-scalar` —
with `TENSORCONTRACT_KERNEL` as the treatment. Both code paths compile in and
the switch chooses at run time, so this satisfies A15 without a build-to-build
diff. **The Phase 4 kernel work could not do this**: an AVX-512 machine cannot
un-have its own kernels, so "what is the vectorised kernel worth" has never been
answerable there. Here it is one environment variable.

`ttgt` and `tblis` ride along in every arm as columns the switch cannot reach.
They moved **0.996–1.006**. That is a control from outside the engine, and it is
the reason the engine numbers below can be read at all.

#### The floor

| | geomean range | |
|---|---|---|
| engine columns, A vs A2 | **1.000–1.008** | quote ratios against this |
| baseline columns | 0.982–1.004 | |
| per case | 9 of 120 outside ±6% | worst *engine* case **1.105** |

So a geomean is readable to about ±1% and a per-case ratio to about ±10%. **This
session finally has a floor**, which part 20 did not — and it is still not a
pinned measurement, because Darwin has no CPU affinity API.

#### The treatment

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

#### The headline: the engine reaches its baselines on this machine

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

#### 3m still leads, and A59 guessed wrong about which way

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

#### What this does not show

No `ragged` arm, so nothing here is about the irregular path — and `ragged` is
the only source of irregularity on this machine. No 49-case corpus sweep; these
are the 12 premise shapes. Nothing per-case finer than ~10%. Nothing about any
other aarch64 part, because register blocks are per-microarchitecture (A34) and
these were measured on this one. Nothing about threads.

#### Assumptions added

The full rows, with their citations, are in [`../decisions.md`](../decisions.md).

| # | Assumption | Status |
|---|---|---|
| A57 | The L2 is private to a core, so the analytical model may budget all of it to one thread. | **False on Apple Silicon, and it had been invisible because every x86 machine measured makes it true.** `model_mc` reads `l2.ways` and `l2.bytes_per_way()` and never divides by `cores_sharing(l2)`; on Cascade Lake, Zen2 and Ice Lake the L2 is per-core (`shared_by` ≤ 2, SMT siblings) so the omission cannot be seen. Six M3 Max P-cores share one 16 MiB L2, so the model hands each the whole thing — a 6x over-allocation, and it derives a **12 MB packed `A` block** for `f64`. A **third, structural** reason A33 stands, not a reopening. |
| A58 | "The portable scalar path is not competitive" is a statement about vectorisation. | **False, and it understated the path by an order of magnitude.** LLVM vectorises `real_ukr` to NEON unasked — eight `float64x2_t` accumulators, `ld1r.2d` broadcasts, even the lane-indexed `fmul.2d v, v, v[0]`. What it will not do is contract `acc += a * b` into `fmla`, because that is two roundings and IEEE forbids it. So the path runs at two instructions per MAC where the machine offers one, **halving the ceiling available to it to 32.4 GF/s**, and on its best shape it reaches **80.5% of that** — against OpenBLAS's 87.1% of the full 64.8. The gap is **one instruction form, not a missing kernel**: **0.65x of OpenBLAS-TTGT** in `f64` over 12 shapes at 64 MiB, not 0.05x. A NEON kernel is worth ~2x, not 10x. **Confirmed by removing it** (part 22): the NEON kernel buys **1.840** end to end and reaches **81.4% of the full 64.8 GF/s ceiling**, against the scalar path's 80.5% of the halved one — same efficiency, twice the ceiling. |
| A59 | The complex-method ranking is decided by bytes moved per useful flop, everywhere. | **Incomplete, not refuted — it depends on which resource binds.** On the portable path 3m leads planar by **1.229** (`c64`, premise, 64 MiB), a third ordering after Cascade Lake's and Ice Lake's. Mechanism: with no FMA the kernel is *instruction-throughput*-bound, so 3m's 25% saving in products is a 25% saving in the binding resource, and the bytes-per-flop accounting that decides it on x86 is not what binds. **Re-measured in Stage B and the guess was wrong in direction of magnitude, not sign** (part 22): with a real FMA kernel 3m's lead **narrows from 1.226 to 1.135 and does not invert**. It gains least from NEON (1.690 against planar's 1.826), which is the mechanism behaving as stated, and still finishes 13.5% ahead — 13x the session floor. So A59 stands as *incomplete*: which resource binds does decide the ranking, and on this microarchitecture 3m wins under both bindings. A **fourth ordering**, and the only one taken with a tuned kernel on its own machine. |
| A60 | The register budget that correctly rejects bad shapes can also choose the good one. | **False, and the two halves are separate skills.** The budget filters well — the `!` over-budget flag tracks collapse closely — and then names the wrong winner in **three of eight** columns on an M3 Max. The sharp case: it proposed `f64` real `6x8`, which is BLIS's own `armv8a_asm_6x8`, and that agreement was recorded as corroboration; the machine prefers `16x3` by 4.9% (2.6x the floor), a shape the budget calls **over** its limit at `live = 33`. A34 on a fourth ISA. A library's chosen shape agreeing with a model is not evidence about this engine's kernels. |
| A61 | A rule fitted on one microarchitecture's register blocks will not hold for another ISA's. | **False here, and worth naming because so little transfers.** The guarded row-block rule was derived on Cascade Lake, on AVX-512, for `MR` in {16, 32, 48}. On NEON `f64` the measured shape `16x3` costs write-back regularity on 12 of 49 cases (`wb` 1.00 → 0.67, the whole `abcijk` family, since 16 does not divide TCCG's 24) and **the unmodified rule demotes exactly those 12** — all have `k = 24` against its `k <= 32` guard. 0 of 392 left with no regular shape. Against A56's four non-transferring choices, this one transfers. |
| A62 | A `best per method` line from a register-block sweep is a result. | **False without a floor: six of eight were ties.** One arm prints eight confident winners; three arms on the same quiet machine put six of them inside the session's own p90 spread (1.89% `f64`, 3.64% `f32`), and expose three shapes that are **unstable rather than noisy** — 1m `3x10` reads 37.8 / 54.9 / 54.8, a 45% swing, which is a shape doing two different things and not a bad measurement of one. A register block may not be changed on one arm; `scripts/kernel-shapes-compare.py` is the gate. |
