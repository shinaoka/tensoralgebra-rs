# Contributing

## Ground rules

* **The test suite stays green.** Correctness is never traded for speed. Every
  performance change must leave `cargo test --workspace --release` passing,
  including under `TENSORCONTRACT_KERNEL=scalar`.
* **Record decisions.** Anything non-obvious — a build-vs-reuse call, a
  heuristic, a deviation from a published method — goes in `DECISIONS.md` with
  its rationale.
* **Negative results count.** A well-characterised "X does not beat Y in regime
  Z" is a valid outcome and should be written up rather than buried.

## Before opening a PR

```bash
cargo fmt --all
cargo clippy --workspace --all-targets    # must be warning-free
cargo test --workspace --release
TENSORCONTRACT_KERNEL=scalar cargo test --workspace --release
TENSORCONTRACT_KERNEL=avx2 cargo test --workspace --release
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
boundary. Add the ISA to `x86::Isa`, give it a `TENSORCONTRACT_KERNEL` name so
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
