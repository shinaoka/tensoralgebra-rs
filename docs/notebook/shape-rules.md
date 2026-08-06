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
