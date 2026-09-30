# Switchable GEMM engine implementation

## Workspace ownership correction

The maintainer rejected Task 8's process-global arena on 2026-09-30.
Retain the approved spec's pool ownership: worker-local A/tile buffers,
with an exclusive team B lease returned to its originating pool. Serial
plans own their workspace. TLS lookup does not own retained payloads;
dropping/trimming an owner must release idle storage even when a borrowed
host pool's threads remain alive. Active leases remain valid.

`ExecSpmd` lives in blas/contract, but that is not a reason for a global
arena. `tprims-exec::Pool` can own a provider from the thread-free kernel
contract without depending on the driver. No process-global workspace or
cross-pool team-buffer reuse is permitted. This changes ownership and
lifetime, not OS affinity or NUMA policy, and makes no performance claim.

TBLIS 1.x's global memory pools are a valid different choice: allocation
reuse is global while active buffers belong to a gang. That precedent does
not override this project's explicit context ownership. Reference:
`2026-09-30-research-blis-tblis-structure.md`, Part V.

## Validation and progress

Spec §6.2 and Task 8 now agree on ownership, provider borrowing, serial
ownership, re-entry, trim/drop and cross-pool isolation checks. The existing
process-default selection registry remains process-wide immutable metadata,
not a workspace cache. Design/plan/HANDOFF diff passes `git diff --check`.

Task 1 preserves all 367 passing workspace tests (including doctests) and
all 124 tensorcontract/kernel release tests, now distributed across three
crates. Workspace/all-targets build, std-disabled tensorcontract check,
formatting and warning-denying rustdoc pass. Clippy 1.98.0 passes with
`-D warnings`; the default 1.97.1 exits successfully but emits the existing
unknown-1.98-lint warning. No lint was disabled to obtain the clean result.
Evidence: `/home/shinaoka/tensor4all/.artifacts/gemm-engine-spec/` baseline
and `split-*` logs. Runtime kernel dependencies point only downward; a
path-only dev-dependency preserves the existing custom-scalar doctest.
The private-env blocking test follows the contract crate rather than
exposing its env helper solely for a test. No arithmetic bodies changed.

Task 2 adds CPU feature detection/masking, typed family descriptors,
validation/legacy bridging and per-real-type registration plus forced-id
errors and manifests. Conflicting descriptors sharing an id yield a typed
selection error, not the plan's input-derived debug assertion; enumeration
still de-duplicates ids. A typo under a built provider remains `UnknownId`,
not a misleading `NotBuilt`. Providers will supply portable and tc lists in
Task 3; execution is still the existing driver. The seven planned failing
tests first produced missing-type compile errors (`task2-red.log`). The
focused suite now has 11 passing tests, covering duplicate callback
registration, CPU masking, cross-dtype ids, missing features, unknown ids,
packing/tile bounds, checked geometry overflow and the legacy bridge.
Contract crate: 34 unit tests, 11 integration tests and 18 doctests pass.
`--no-default-features` check and all-targets clippy 1.98 with `-D warnings`
pass (`task2-*` evidence logs).

Source inspection corrected two plan transcription errors: tc 1m's logical
MR/NR need not be even, and its OneR B role uses `PackFormat::Planar`, not
`Real`. See `scalar.rs::config_cplx` and `onem_ukr`; a logical 3×5 regression
checks valid packing lengths and bridging. Generation eligibility in Task 6
still applies to the inner real family, not its already-halved descriptor.
The spec, plan and HANDOFF record this clarification; no arithmetic changed.

Implementation is not complete. Tasks 3–12 and final performance/CI gates
remain.
