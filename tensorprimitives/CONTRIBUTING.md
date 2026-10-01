# Contributing

## Ground rules

* **The test suite stays green.** Correctness is never traded for speed. Every
  performance change must leave `cargo test --workspace --release` passing,
  on every instruction set the CPU supports (the kernel-contract tests run each one).
* **Record decisions.** Anything non-obvious — a build-vs-reuse call, a
  heuristic, a deviation from a published method — goes in `docs/notebook/` with
  its rationale.
* **Negative results count.** A well-characterised "X does not beat Y in regime
  Z" is a valid outcome and should be written up rather than buried.

## Before opening a PR

```bash
cargo fmt --all
cargo clippy --workspace --all-targets    # must be warning-free
cargo test --workspace --release
```

If you touched anything in the hot path, also run the cross-implementation
check against TBLIS and TTGT:

```bash
source scripts/env.sh
cargo build --release -p tensorprimitives-bench --features tblis,blas
./target/release/tcbench verify --size 4
./target/release/tcbench verify --size 4 --stress ragged
./target/release/tcbench verify --size 4 --stress padded
```

## Running the benchmarks

```bash
# One-time: build the TBLIS baseline (see scripts/env.sh for the exact recipe)
export TBLIS_ROOT=/path/to/tblis-install
source scripts/env.sh

cargo build --release -p tensorprimitives-bench --features tblis,blas

# correctness: whole corpus vs TBLIS and TTGT, all dtypes
./target/release/tcbench verify --size 4

# the premise check
./target/release/tcbench premise --size 64 --reps 3 --dtype f64,c64 \
    --engines tblis,ttgt --csv /tmp/premise-f64c64.csv

# the same against the last stable TBLIS release (note the feature and ABI)
cargo build --release -p tensorprimitives-bench --features tblis13,blas
TBLIS_ROOT=/path/to/tblis-1.3.0-install \
    ./target/release/tcbench premise --size 64 --dtype f64,c64 --engines tblis

# same, with the gather path actually exercised
./target/release/tcbench premise --size 64 --stress ragged --dtype f64,c64

# full corpus sweep
./target/release/tcbench sweep --size 32 --csv /tmp/sweep.csv
```

Or run the whole comparison set the way this project runs it — a discarded
warm-up arm, both TBLIS ABIs from prebuilt binaries so nothing compiles mid-run,
and two repeat arms that derive the session's own noise floor instead of
importing one:

```bash
TBLIS_ROOT_2X=../baselines/tblis-2.0-install \
TBLIS_ROOT_13=../baselines/tblis-1.3.0-install \
  scripts/compare-bench.sh prep                              # the only compile
  scripts/compare-bench.sh bench-results/$(hostname -s)-$(scripts/arch-label.sh)
```

**It wants an exclusive machine for about three and a half hours**, and that is
not pedantry: pinning is not enough, because the pinned core's SMT sibling shares
L1d and L2, which is what every cache-blocking measurement here turns on. Two
Phase 4 conclusions had to be corrected after re-measuring on a quiet machine.
[`scripts/README.md`](scripts/README.md) indexes the rest, including the Slurm
wrappers and the offline analyses that cost no CPU at all.

Raw results from every run quoted anywhere in this repository are committed under
[`bench-results/`](bench-results/README.md), one `PROVENANCE.txt` per directory
saying which machine and date produced it. Read
[`docs/measurement-rules.md`](docs/measurement-rules.md) before designing a new
measurement — every rule in it cost one.

## Testing

```bash
cargo test --workspace --release                                 # includes 1000 randomised
```

The library reads no environment variables. The kernel-contract tests run
*every* kernel family the CPU supports on every `cargo test`, which is how the
AVX2 path is exercised on an AVX-512 machine; pinning an instruction set for a
whole plan is `Tuning::kernel_force` (`tcbench` parses `TENSORCONTRACT_KERNEL`
= `scalar`, `avx2`, `avx512` or `auto` into it).

The engine is checked against a brute-force oracle that shares no code with it,
under both realistic and deliberately tiny cache blocking, so that every level of
the five-loop nest and every partial block is exercised on tensors small enough
to verify exhaustively.

CI runs the switch matrix — `scalar`, `avx2`, `threads=4`, a pinned `2x2`
partition, the analytical blocking model, the thread pool, and all three
orientation arms — because each is a path nothing else enters.

## Adding a micro-kernel

Kernels live in `crates/tensorcontract/src/kernel/`. A kernel must:

1. Match the panel format documented in `kernel/mod.rs` — for complex, planar
   "1r" for both operands.
2. **Overwrite**, not accumulate into, the `MR x NR` accumulator tile. `alpha`,
   `beta`, scatter write-back and complex re-interleaving belong to
   `writeback.rs`.
3. Be registered through `KernelSet::config_real` / `config_cplx` with runtime
   feature detection, so the portable scalar path stays reachable.
4. Pass `kernel::tests::kernels_match_reference_*`, which checks every selected
   kernel against the scalar semantics element by element, and
   `every_available_x86_isa_matches_reference`, which does the same for every
   kernel family the CPU can run rather than only the selected one.

A new *instruction set* rather than a new method should be a new set of
arguments to `simd_kernels!` in `kernel/x86.rs` and a new `configs!` block, not
a new set of kernel bodies. One body per method across all ISAs is deliberate
(D17): a comparison between the three complex methods must not also be a
comparison between hand-tunings, and that argument does not stop at the ISA
boundary. Add the ISA to `x86::Isa`, give it a `KernelForce` variant so
it can be pinned on hardware that has something wider, and extend the AVX2/512
grids in `examples/kernel_shapes` so its register blocks can be calibrated.

## Benchmarking etiquette

* Single-threaded unless the change is specifically about threading; set
  `TBLIS_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` (`scripts/env.sh` does).
* Report GF/s against a same-shape vendor GEMM ceiling, not in isolation.
  Absolute GF/s says as much about the machine as about the code.
* When comparing real and complex, hold the *shape* fixed. Re-sizing per
  precision, as upstream TCCG does, makes the ratio meaningless.
* If a claim depends on irregular block scatter, say which `--stress` mode
  produced it and quote the observed `regA`. Do **not** repeat the old shorthand
  that the unperturbed TCCG corpus is "fully regular" — it is not, at this
  engine's register blocks. TCCG rounds stride-1 extents to multiples of 24,
  which is regular only for a block that divides 24, and the shipped `f32`/`c32`
  blocks are `MR` 16, 32 and 48. `reg_a < 1.0` on 42.9% of the corpus as the
  orientation rule actually picks it (Phase 4 part 14). `--stress` buys *severe*
  irregularity, not the only departure from 1.00.
