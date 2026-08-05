# `bench-results/` — every measurement this project quotes

Raw CSVs for every number in [`DECISIONS.md`](../DECISIONS.md), committed. This
is not an archive: two of these directories are *grids* that a candidate tuning
rule can still be scored against offline, for free, and that is how three Phase 4
decisions were settled without buying more machine time.

**Read the `PROVENANCE.txt` in a directory before reading its numbers.** Every
directory has one. It names the machine, the date, the Slurm job where there is
one, the script that produced it, and — the part that is easy to lose — what the
data is good for and what it cannot support.

## The one rule that governs all of it

**No number measured on one machine may be compared with a number measured on
another.** Different cache hierarchy, sometimes a different instruction set.
Every ratio here is within-session, against a noise floor derived in that same
session. `ccqlin038` is the reference machine and the only one the shipped
blocking constants were fitted to; the `worker*` directories are Rusty nodes and
answer portability questions, not "is it fast".

## Naming

Two schemes coexist, for a reason that is historical but not arbitrary:

* **`phase<N><letter>/` — by experiment.** The reference-machine work, before
  this project ever ran on a cluster. `ccqlin038` is implied and is in every
  file header.
* **`<host>-<uarch>/<experiment>/` — by machine.** Everything from the Slurm era,
  where *which machine* is the first thing you need to know. `ab-deepen` and
  `ccqlin038-blocking` are the transitional pair: named for the experiment and
  the host respectively, both on the reference machine.

**New directories use `<host>-<uarch>/<experiment>/`.** The label comes from the
machine via [`scripts/arch-label.sh`](../scripts/arch-label.sh), never from the
partition a script was first written for — hardcoding `-zen2` filed an Ice Lake
run under a Zen2 name once already, which is why `worker6016-icelake` carries a
note about its own rename.

The existing names are **not** being retrofitted. They are cited by path
throughout `DECISIONS.md`, which is append-only by design, and a rename would
turn every one of those citations into a dead reference to buy tidiness.

## The directories

| directory | machine | date | job | produced by | what it is |
|---|---|---|---|---|---|
| `phase4/` | ccqlin038 (Cascade Lake, AVX-512) | 2026-08-02 | — | `phase4-remeasure.sh` | Item 1, the write-back, re-measured on an exclusive machine. `rm-A`/`rm-B`/`rm-A2`. The run that corrected two earlier conclusions |
| `phase4c/` | ccqlin038 | 2026-08-02 | — | `phase4c-rowblock.sh` | **Grid.** Every register block in every method's menu, all 392 corpus case-dtype-methods |
| `phase4d/` | ccqlin038 | 2026-08-02 | — | `phase4d-orient.sh` | **Grid.** Both orientation arms, forced, all 392. 21 case-dtype-methods still take the slower arm — a candidate rule costs no machine time |
| `ccqlin038-blocking/` | ccqlin038 | 2026-08-03/04 | — | `ccq-blocking-night.sh` | The `MC`/`KC`/`NC` grid on the reference machine. **Its `base` arm is 1–2% slow**, so every `arm/base` ratio in it is inflated — read it with `ab-deepen/` |
| `ab-deepen/` | ccqlin038 | 2026-08-04 | — | `ab.sh` | The end-to-end A/B for `TENSORCONTRACT_DEEPEN`, and it fails. Closes Phase 4 item 2 |
| `worker5040-zen2/` | worker5040 (Zen2, AVX2, no AVX-512) | 2026-08-03 | 6745376 | `rusty-phase4.sbatch` | The first cluster session: AVX2 register blocks confirmed, thread scaling, the blocking grid, placement validation |
| `worker5175-zen2/` | worker5175 (Zen2) | 2026-08-03 | 6745978 | `rusty-phase4-seq.sbatch` | The three traffic-changing blocking arms, sequentially, because A27 rejected concurrent placement for them. **Directory name is wrong** — see its `PROVENANCE.txt` |
| `worker5137-zen2/` | worker5137 (Zen2) | 2026-08-04 | 6751550 | `rusty-phase4.sbatch`, threads only | Third node. Confirms the partition effect follows **L3 domains spanned**, not thread count |
| `worker6156-icelake/` | worker6156 (Ice Lake-SP, AVX-512) | 2026-08-04 | 6753260 | `rusty-compare.sbatch` | **The engine against its baselines**, first such run since Phase 3. Tightest floor in the repo (0.998-1.001 session-span). Source of **A44** (the method ranking does not travel). Holds three experiments: `compare-tblis-skx/` and `compare-tblis-x86_64/` (two independent full runs agreeing to 0.997-1.003) and `ab-tblis/` (the TBLIS-build A/B, A45/A46). **The README's numbers come from here** |
| `worker6016-icelake/` | worker6016 (Ice Lake-SP, AVX-512) | 2026-08-03 | 6746817 | `rusty-phase4.sbatch`, `STAGES="shapes threads"` | The second Intel hierarchy. Source of **A34** (register blocks are per-microarchitecture) and **A36** (the partition win is absent here). `kernel-shapes.txt` covers **all four methods including 3m** at three depths — which is what deconfounds **A44** from A34 and kills Phase 4 item 3, and it sat here unread for two days while `DECISIONS.md` called that measurement pending. Read a directory before booking a machine |
| `worker5479-zen2/` | worker5479 (Zen2) | 2026-08-04 | 6753208 | `PARTITION_SWEEP=1 STAGES=threads` | The domain-gate confirmation run, **Zen2 arm**: 144 of 392 moved at 1.433 corrected, corpus 1.133, against a prediction of 1.423 / 1.138 made before the node existed. Also the spread arms (A36's residual gap), A38 and A39. Contains `failed-6753261-sigill/`, a separate job with its own provenance |
| `worker6150-icelake/` | worker6150 (Ice Lake-SP, AVX-512) | 2026-08-04 | 6753209 | the same submission | The domain-gate confirmation run, **Ice Lake arm — the control**: the gate moved **0 of 392** at every width up to the socket. Confirms A39 at 32 threads on a second machine |
| `worker5139-zen2/` | worker5139 (Zen2) | 2026-08-04 | 6754849 | `STAGES="threads small"` | **Small contractions** (`phase4g/`): threading is 1.2–10x *slower* than serial below ~1 MiB and the optimal thread count walks 4 → 64 across the size range. Source of **D46** and **A43** |
| `worker5178-zen2/` | worker5178 (Zen2) | 2026-08-04 | 6755009 | `RAGGED=1 STAGES=threads` | **`--stress ragged`**: heterogeneous cases from 11.7% to 87.2% at aperiodic fractions, and parallel efficiency moves 0.976–1.124, i.e. not at all. Source of **D47** and A41's final form |
| `worker5086-zen2/` | worker5086 (Zen2) | 2026-08-05 | 6760092 | `ARMS="base;pool;guard;both" STAGES=small` | **The A/B for the three answers to per-call spawn.** Four arms, 225 CSVs. **The pool wins** (up to 11.6x at 0.25 MiB, 2.0–2.8x at 16 MiB, no per-family cell below 0.99) and **the guard does not ship** — it is a trade, and pure loss on top of the pool, reaching 0.32. The 2x2 design is what showed the two switches conflict; either alone looked good. Also: the base arm replicates part 16 on a fourth node, and the cost the pool removes turns out **not** to be thread creation (A53). Source of **D52**, **D53**, A53, A54 |

### Loose files at the top level

The Phase 3 measurement set, all on `ccqlin038`, all 2026-08-02, all from
`scripts/phase3-bench.sh` — plus one Phase 2b file:

| file | what |
|---|---|
| `phase3-sweep-{f64c64,f32c32}.csv` | The 49-case corpus against TBLIS 2.0-dev and TTGT. **These are the newest engine-vs-baseline numbers in the repo and they predate every Phase 4 gain**, so they understate the engine |
| `phase3-sweep-ragged-c64.csv` | The same under `--stress ragged` — the only committed data in which the block-scatter gather path actually runs (`regA` down to 0.0031). Note the unperturbed corpus is **not** fully regular either, contrary to a shorthand this project repeated for a while: at the shipped `f32`/`c32` register blocks `reg_a < 1.0` on 42.9% of the 392 case-dtype-methods at `--size 64`, 45 of them entirely (Phase 4 part 14 — and note the fraction moves with the size and the ISA; this very file's `phase4d/features.csv` reads 36.7% because it was generated at a different size). `--stress` is how you get *severe* irregularity, not the only way to leave 1.00 |
| `phase3-premise-{f64c64,f32c32}.csv` | Efficiency against a same-shape GEMM ceiling, 200 MiB, 12 cases — the metric the Phase 1 headline is stated in |
| `phase3-premise-tblis130-f64c64.csv` | The same against **TBLIS v1.3.0**, the latest stable release. This is the half of the headline that shows the 5x |
| `phase3-kernel-shapes.txt` | The AVX-512 register-block sweep. Cited from `kernel/x86.rs`, and the file **A35** was found in — months later, at no machine cost, which is the return on keeping raw output |
| `phase3-log.txt` | Full transcript of that session |
| `methods-f64c64.csv` | Phase 2b, 2026-08-01: the three complex methods before the AVX-512 kernels existed |

Five early-era `premise-*.csv` and an orphan `phase4e/features.csv` (byte-identical
to `ccqlin038-blocking/features.csv`) were deleted in the cleanup that added this
file; they are in git history if ever wanted.

## Working with it, without spending any CPU

```bash
# Turn two sweep CSVs into the ratio tables used throughout DECISIONS.md.
scripts/compare-sweeps.py BASE_CSVS NEW_CSVS

# Score candidate rules against a grid — as many candidates as you like, free.
scripts/rowblock-score-rules.py phase4c/shapes.csv   phase4c
scripts/orient-score-rules.py   phase4d/features.csv phase4d
scripts/blocking-score-rules.py ccqlin038-blocking/features.csv ccqlin038-blocking

# Re-derive the published noise floor from committed data. Should print
# geomeans of 0.99-1.01; if it does not, the tooling has drifted from the file.
cd phase4 && python3 ../../scripts/compare-sweeps.py \
    rm-A-f64c64.csv,rm-A-f32c32.csv rm-A2-f64c64.csv,rm-A2-f32c32.csv
```

Every CSV shares one header:

```
case,group,dtype,engine,m,n,k,macs,seconds,gflops,regular_a,regular_b,notes
```

`regular_a` / `regular_b` are the block-scatter regularity fractions — quote them
in any claim about awkward strides, and say which `--stress` mode produced them.
`notes` carries the kernel, the register block, the orientation arm and the
blocking actually used, which is what makes a row reproducible.

Sessions run through `scripts/run-arms.py` also leave, per arm, a `.txt`
transcript, a `.cpu` file recording occupancy for **every core in that arm's L3
domain** across exactly the arm's window, and a cumulative `arms.jsonl`. Read the
per-arm files, not just the means: a *constant* co-tenant cancels in an A-vs-B
ratio and an intermittent one does not, and that distinction is what made two
retractions detectable.

## What to read next

[`../scripts/README.md`](../scripts/README.md) for what produced any of this, and
[`../DECISIONS.md`](../DECISIONS.md) for what it means — its "Measurement rules"
section holds the noise floors, **each with its own session and thread count**,
and the nine rules these directories were accumulated under. The short form: run
and discard a warm-up arm (A31) and do not treat it as sufficient (A40); derive
the floor in session, from repeats adjacent to the arms compared (A32); do not
place traffic-changing arms concurrently (A27); include columns the change cannot
touch and check they do not move; measure at the size you publish at (A46); and
**a per-case ratio at 64 threads is not readable at all** (A39). The full account
of each, with what would reopen it, is in [`../REFUTED.md`](../REFUTED.md).
