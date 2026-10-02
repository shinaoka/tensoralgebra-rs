# Measurement rules, machines and baselines

Read this before designing a measurement. Every rule here cost one.

The fresh-checkout sanity check, the machines, both TBLIS baselines and the
noise floors are all here; a floor is meaningless without its session **and**
its thread count.

---

### Before believing anything in this file

* **No number measured on one machine may be compared with a number measured on
  another.** Every ratio here is within-session against a floor derived in that
  session. See [measurement rules](#measurement-rules); floors are per session
  *and per thread count*.
* Raw CSVs for every number are committed under
  [`bench-results/`](https://github.com/tensor4all/tprims-rs/blob/0fc06f4578e20017e510807ccaaa72ab4bab08f4/tensorprimitives/bench-results/README.md), one `PROVENANCE.txt` per directory.
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
pinned**. It runs `kernel::aarch64` as of Stage B (part 22); every number taken here
before that ran `kernel::scalar`, and the two are an arm apart under
`TENSORCONTRACT_KERNEL`.

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
[`refuted.md`](refuted.md) under *Measurement methodology*.

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

