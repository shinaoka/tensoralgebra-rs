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

Coupling re-derives `mc` at the new depth, holding the `A` footprint constant, and
it was the only arm that agreed on both machines — +2.9% on Cascade Lake and +3.1%
on Zen2, where the *pinned* `kc = 512` arm reads 0.961 and 1.050. That agreement is
why it was proposed as the change to make.

**It did not survive its own A/B** — see the next subsection, and `../refuted.md`.
Both grids' `base` arm was 1–2% slow, so the two-machine agreement was a shared
artefact rather than a replication. The recommendation, the per-method table behind
it and the general form it suggested (couple only while `mc` stays above a small
multiple of `MR`) are all deleted rather than preserved: they were argued from a
number that is not real. The mechanism paragraph above them stands, because it is
about *why* depth trades against `mc`, which the artefact does not touch.

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
the wrong default it prevented. `TENSORCONTRACT_DEEPEN` stayed for a while as an
off-by-default switch documenting the experiment; it was **removed before 0.1.0**,
and `REFUTED.md` plus `bench-results/ab-deepen/` is the record.

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
   stream at the operating `kc`, reporting 3m as the *fastest* of the three when
   panels are L1-resident and third when they are not. (**That reading is since
   refuted — A44:** it holds on Cascade Lake and reverses on Ice Lake. `kc` is
   still the parameter that decides which regime the engine is in, which is all
   this section needs from it.)
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

This section argued that the model solved the **portability** problem while leaving
**optimality** on `ccqlin038` untouched, and left it as a second arm of a pending
grid rather than a change, to be judged end to end (A20).

That judgement happened and went against the model.

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
