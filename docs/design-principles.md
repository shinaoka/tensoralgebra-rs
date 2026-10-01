# Design principles

[Back to the visual overview](../README.md). This page states what the stack is trying to be and the rules that follow from it. The [architecture notes](architecture.md) apply these principles to concrete crates; the [decision log](decision-log.md) records which choices are settled.

## What the stack is

tprims is a set of CPU building blocks for dense tensor and matrix computation: an execution context, strided kernels, a GEMM microkernel, BLAS-like operations, dense linear algebra and binary tensor contraction. It is meant to sit **underneath** other libraries and applications, in Rust or through a C ABI from C, C++, Julia or Python. It is not an end-user array library and not tied to any one consumer.

## 1. Parts, not a facade

Every crate is a part that can be depended on alone. There is no umbrella crate that re-exports the stack.

- **Why:** a consumer that needs only GEMM, or only a contraction, should not compile, audit or version-lock the rest. tenferro-rs follows the same rule and it has worked: its crates are reused selectively by downstream projects.
- **Consequence:** each crate has its own documentation, tests and semver. Its public API must make sense without the others; a crate may not assume that its caller also uses a sibling.
- **Consequence:** convenience layers (array types, `ndarray`/`mdarray` adapters, einsum strings) belong to consumers or to thin adapter crates above the stack, never inside a part.

## 2. One direction of dependency

The dependency graph is acyclic and layered: execution context at the bottom, then strided kernels and the GEMM microkernel, then BLAS-like operations, then dense linear algebra, then contraction.

- A lower part never learns about a higher one.
- A part depends on the narrowest lower contract it needs. `tprims-contract` depends on the packed-kernel contract of `tprims-gemm-kernel`, not only on public GEMM, because that is what direct contraction actually requires.

## 3. Split only at a real boundary

A new crate is created when a separate consumer and a stable interface justify it, not to mirror a taxonomy.

- `tprims-linalg` keeps factorizations and solves together because solves depend on factor internals.
- Batched execution is a module of the crate that owns the per-item algorithm, not a crate of its own.
- Out-of-scope work (for example Krylov solvers) is added later as a new part, without reshaping existing ones.

## 4. Short names under one prefix

Crates are named `tprims-<part>`, where the part name says what it does: `tprims-exec`, `tprims-blas`, `tprims-linalg`, `tprims-contract`.

- The prefix identifies the family and avoids claiming generic crates.io names such as `blas3` or `tensordot`.
- The prefix is neutral: no consumer or project name, because the stack is meant for users beyond its original project.
- Crate, header and C symbol correspond one to one (`tprims-blas`, `tprims/blas.h`, `tprims_blas_*`), so documentation is easy to find from any side.
- Rust users who want shorter paths rename on import.

## 5. A C ABI per part, one library per build

Each part owns the C ABI for its own operations, next to the code whose semantics it exposes and tests. C ABI crates contain no algorithms.

- C ABI crates are `rlib`s. A single bundle crate links the parts a host selects into **one** shared or static library, `libtprims`.
- Everything a host builds together shares one execution runtime and one set of handle types, so an object created by one part is valid in every other part. The executor is the standard `TAPP_executor`, created and destroyed through the C API; the C host owns it and it owns its Rayon pool. Handles of another TAPP provider cannot be mixed in.
- Where a standard exists, the ABI is the standard: contraction is TAPP with the prototypes of the pinned upstream headers, and tprims adds only named extensions (`tprims_tapp_*`, BLAS and linear algebra with DLPack operands). Nonzero error codes are provider-defined, so callers test success with `TAPP_check_success`.
- Separate libraries per part are not supported: they would duplicate handle types, thread pools and, for static libraries, Rust runtime symbols.
- The ABI is a permanent contract: versioned from the start, panics caught at every entry point, errors returned as status codes, capabilities queryable at run time.

## 6. Zero copy at the boundary

Data crosses the C ABI as [DLPack](https://dmlc.github.io/dlpack/latest/) descriptors so that arrays from NumPy, PyTorch, JAX and Julia pass through without copying.

- Inputs are borrowed; outputs are either written into caller memory or handed over with an explicit deleter.
- Strides, offsets and column-major layouts are respected rather than normalized by copying.
- Anything DLPack cannot express, such as lazy conjugation, is an explicit argument, not a hidden conversion.

## 7. No hidden costs

A caller must be able to see and control what an operation costs.

- **No hidden copies.** If a layout forces materialization, the operation reports the chosen strategy, and a caller can forbid it.
- **No hidden threads.** Every expensive operation receives an explicit execution context. No part uses an ambient global pool, spawns threads on its own (including fallback teams when a pool is narrower than a plan), or keeps threads alive after the context is closed.
- **No hidden allocation in hot paths.** Scratch sizes are queryable and scratch is reusable across calls.
- **Explicit plans.** Validation and planning can be separated from execution so repeated calls pay only for the work.

## 8. The host owns execution

Whoever embeds the stack decides how many threads run and on which runtime.

- A Rust caller passes an execution context. A C, Julia or Python host can create, use and close a Rayon pool through the ABI, or inject its own scheduler through callbacks, or run serially.
- Closing a pool the library created joins its threads, so a host can shut down cleanly. Closing a context that borrows the host's pool never stops the host's threads.
- One thread budget governs both batch-level and inner parallelism so nested parallelism does not oversubscribe.

## 9. Correct first, then measured

- Numerical correctness gates every performance claim: reconstruction and residual checks, rank-deficient and indefinite inputs, extreme scales, complex values, noncontiguous views.
- Optimizations and provider choices require recorded, reproducible measurements under the [experiment protocol](experiments.md). A design hypothesis stays labelled as one until evidence exists.
- Unfavorable and inconclusive results are kept.

## 10. Credit and provenance

Published algorithms are implemented independently and cited. Translated code and imported tests keep their upstream notices and licenses, and are recorded under the [provenance policy](provenance.md). Scientific credit is due even when no license obligation exists.

## 11. A clear scope

Kept out of the stack, on purpose:

- N-ary einsum and contraction-order planning: frontends call `tprims-contract` per binary step.
- Automatic differentiation, tracing, device transfer and GPU backends.
- Array container types and ecosystem adapters.
- Iterative Krylov solvers, for now.

A narrow scope is what lets each part stay small enough to be reused on its own.
