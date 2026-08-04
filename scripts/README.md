# `scripts/` — what each one is for

Twenty-seven files, and the overlaps are real but mostly not accidental. This
index exists so that "which script do I use" is answered here rather than by
reading five headers, and so that a script kept only because it produced
committed data is visibly that.

**Nothing here is deleted, and superseded is not the same as dead.**
`phase4-remeasure.sh` and `phase3-bench.sh` produced data that `DECISIONS.md`
cites; they stay, and they are marked.

Shared conventions: the shell scripts refuse to be sourced, `cd` to the repo
root, and take positional `[cpu] [outdir] [size_mib] [reps] [filter]` with
defaults `4 / <their own dir> / 64 / 3 / none`. Every measurement here is
single-core and pinned unless it says otherwise.

---

## Start here

| I want to… | use |
|---|---|
| compare the engine against TBLIS and TTGT | **`compare-bench.sh`** |
| A/B one runtime switch, end to end | **`ab.sh`** |
| decide between N discrete options | sweep the **whole grid** once (`phase4c`/`phase4d`/`phase4e-blocking.sh`), then score rules offline — see below |
| run a whole session on a cluster node | **`rusty-compare.sbatch`** or **`rusty-phase4.sbatch`**, or `node-session.sh` by hand |
| know if this machine can host concurrent arms | `validate-placement.sh`, then `placement-verdict.py` |
| analyse committed data, costing no CPU | `compare-sweeps.py`, the four `*-score-rules.py`, `thread-width.py` |

**The grid pattern is the most valuable thing in this directory.** When the
choice is discrete, measure every option once and score candidate rules against
the result offline, as often as you like, for free. Both the row-block rule and
the orientation rule were settled that way, and the grids are still in
[`../bench-results/`](../bench-results/README.md) for the next candidate.

---

## Measurement drivers

| script | what it does | status |
|---|---|---|
| `compare-bench.sh` | The engine against its baselines: `verify`, corpus sweeps, `premise` at 200 MiB, ragged stress, and the TBLIS **v1.3.0** comparison. Takes an output directory. `prep` builds the two binaries (one per TBLIS ABI) and is the only thing that compiles. ~3.5 h | current |
| `ab.sh` | Generic end-to-end A/B on one runtime switch: `warm-up, A, B, A'`. ~2 h | current |
| `phase3-bench.sh` | The Phase 3 measurement set. Produced `bench-results/phase3-*` | **superseded by `compare-bench.sh`** — it writes flat into `bench-results/` and would overwrite that committed data, it has no warm-up arm (it predates A31), and it compiles *between* measurement groups |
| `phase4-remeasure.sh` | The Phase 4 A/B with its B arm hardcoded to the write-back experiment. Produced `bench-results/phase4` | **superseded by `ab.sh`**, which adds the discarded warm-up arm and an explicit A-vs-A′ floor. Kept because its output is cited. Note the citation weight is inverted — `DECISIONS.md` mentions the old one far more often, because it is older, not because it is better |
| `phase4c-rowblock.sh` | **Grid.** Every register block, all 392 corpus case-dtype-methods | current |
| `phase4d-orient.sh` | **Grid.** Both orientation arms, forced, all 392 | current |
| `phase4e-blocking.sh` | **Grid.** The `MC`/`KC`/`NC` arms. Honours `ARMS_ONLY` | current, but the *item* is closed: see A33 and part 7 before spending a node on it |
| `phase4f-threads.sh` | Thread scaling, physical cores of one socket, `t1` brackets, a `--cross-socket` arm. Wants a whole socket. ~1 h | current |
| `ccq-blocking-night.sh` | The blocking grid on the **reference machine**, overnight: warm-up, grid, audit, score. ~7 h | current. Not superseded by `phase4e-blocking.sh` — it *calls* it twice. The real duplication is with `node-session.sh`, which implements the same warm-up/topology/grid orchestration independently |
| `validate-placement.sh` | The pre-registered test: can arms run one-per-L3-domain without changing what is measured? `solo1`/`solo2`/`placed` plus memory-bound variants | current |

## Orchestration

| script | what it does | status |
|---|---|---|
| `node-session.sh` | Stages a whole cluster session: `prep`, `threads`, `shapes`, `validate`, `grid`. **`prep` is the only stage that compiles** | current. Does *not* supersede the `phase4*` scripts — it invokes three of them as stages, and never touches `phase4c`/`phase4d`/`phase4-remeasure` |
| `rusty-compare.sbatch` | `compare-bench.sh` on one exclusive Rusty node. Constraint deliberately left to the submit line, because the node choice is the experiment | current |
| `rusty-phase4.sbatch` | The Phase 4 session on one exclusive Rusty node, 12 h, with a scoped-grid fallback if placement is rejected | current |
| `rusty-phase4-seq.sbatch` | The three traffic-changing blocking arms, sequentially, 6 h | current. **Complements** `rusty-phase4.sbatch` rather than replacing it — submit it *after*, and it exists because those arms cannot use concurrent placement (A27) |
| `run-arms.py` | Runs arms under an explicit placement and records **where each one ran**: one thread per L3 domain, SMT siblings idle, `/proc/stat` sampled over every core in the arm's own domain across exactly that arm's window. Drives `sweep` or `premise` | current. The shared runner — prefer it over calling `tcbench` by hand, because the occupancy record is what makes a result auditable |
| `env.sh` | Sourced: toolchain modules, `TBLIS_ROOT`, and pins everything single-threaded | current |
| `arch-label.sh` | Prints this machine's microarchitecture label, for a `bench-results/<host>-<uarch>/` directory name | current, and the authoritative copy. Two older in-line copies exist, in `rusty-phase4.sbatch` (equivalent) and `node-session.sh` (weaker — it labels every EPYC `zen`); both are left alone because they have produced committed data |

## Offline analysis — no CPU, safe while a benchmark is in flight

| script | what it does |
|---|---|
| `compare-sweeps.py` | Two sweep CSVs into the ratio tables used throughout `DECISIONS.md`, with the AB/BA split as a control |
| `rowblock-score-rules.py` | Score candidate row-block rules against the 1c grid |
| `rowblock-decompose.py` | Split a row-block grid into what the shape *costs* and what it *buys* |
| `orient-score-rules.py` | Score candidate orientation rules against the 1d grid |
| `blocking-score-rules.py` | Score `MC`/`KC`/`NC` rules against the item-2 grid. The most evolved of the four — it adds `arms` and `noise` sections the earlier ones lack |
| `thread-width.py` | How much parallel width the corpus has, and where, for any thread count — from committed data |
| `topology.py` | What machine is this, and where may a measurement thread go: L3 domains, SMT siblings, NUMA. JSON out. Imported by `run-arms.py` and called by most of the drivers |

### The three placement scripts answer three different questions

None supersedes another, which is not obvious from their names:

* **`placement-spread.py`** — *descriptive.* Does position in the node matter?
  Each concurrent replicate against a solo baseline, by slot, L3 and NUMA.
  Called by `validate-placement.sh`.
* **`placement-verdict.py`** — *the decision.* Implements the accept/reject rule
  that was pre-registered in `DECISIONS.md` part 10 before any data existed.
  Called by `rusty-phase4.sbatch`.
* **`grid-placement-audit.py`** — *post-hoc.* Did where each arm ran contaminate
  an already-run grid's `arm/base` ratios? It reports and **corrects nothing**.
  Reachable only through `ccq-blocking-night.sh` and cited in no document, which
  is the gap this line closes.

---

## The rules these scripts encode

Each is a lesson that cost a retraction, and they are why the newer scripts look
the way they do:

* **Run and discard a warm-up arm** (A31). `A, B, A'` cannot tell a cold-start
  transient from a noise floor: an exclusive machine's opening arm runs at
  single-core boost on a cold package and nothing else does, which came back once
  as a uniform 4.4% "floor" while inflating every ratio measured against A.
* **Derive the floor in session, and know which floor** (A32). Drift is a
  function of how far apart two arms are — 0.02% at a minute, 1–2% at an hour,
  4.4% across a cold start. Do not import ±1.3%/±6% from `DECISIONS.md`; those
  belong to one session on one machine.
* **Prefer a runtime switch to a rebuild** (A15), so both arms interleave in one
  session. `TENSORCONTRACT_ORIENT`, `_ROWBLOCK`, `_WRITEBACK`, `_DEEPEN`,
  `_PARTITION` exist for exactly this. A build-to-build diff already produced one
  wrong sign.
* **Do not place traffic-changing arms concurrently** (A27). One arm per L3
  domain is free on the corpus (+0.3%) and *not* free on the memory-bound half
  (−3.2%, up to −10%).
* **Record occupancy per arm.** A constant co-tenant cancels in an A-vs-B ratio;
  an intermittent one does not. Two Phase 4 conclusions had to be corrected
  because this was not recorded.
* **Validate jointly, not just pinned** (A20). A rule proven with the other
  levers pinned is proven only there — finish with an end-to-end A/B in the
  configuration that actually ships.

## Slurm

The `.sbatch` files are submitted by a human, never by a tool:

```bash
sbatch --constraint=icelake scripts/rusty-compare.sbatch
sbatch --constraint=rome    scripts/rusty-compare.sbatch
squeue -u $USER
```
