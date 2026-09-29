# Phase 1f: C ABI slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A thin C ABI over `tprims-exec`, `tprims-blas` (gemm, batched gemm) and `tprims-contract` (plans), linked into one `libtprims` (`cdylib` + `staticlib`) by `tprims-bundle`, with DLPack zero-copy operands, synchronous pool close, a C test program and a C benchmark comparing per-call cost with direct Rust.

**Architecture:** Each C ABI crate is an `rlib` of `extern "C"` functions over its Rust part; only `crates/tprims-bundle` produces libraries, re-exporting the selected parts by feature (`blas`, `contract`). `tprims-core` owns the DLPack 1.x `#[repr(C)]` mirror, `tprims_tensor { DLTensor *view; uint64_t flags; }`, `tprims_status`, the thread-local last-error string, ABI version and part queries, and executor handles. Headers are hand-written under `crates/tprims-core/include/tprims/{core,blas,contract}.h` (checked by the C test compile). Every entry point catches panics (`catch_unwind`) and returns a status.

**Tech Stack:** Rust `extern "C"`, rayon `ThreadPoolBuilder::spawn_handler` (joinable owned pools), DLPack 1.x layout, C11 test/bench compiled with the system `cc`.

**Spec:** `docs/superpowers/specs/2026-09-29-phase1-cpu-backend-design.md` (section 1f); `docs/architecture.md` (Zero-copy data exchange, Execution context: close semantics).

## Global Constraints

- Branch `phase1f-capi` from `main` after 1c merges.
- Status codes: `TPRIMS_OK = 0`, `TPRIMS_ERR_INVALID_ARGUMENT`, `TPRIMS_ERR_SHAPE`, `TPRIMS_ERR_DTYPE`, `TPRIMS_ERR_DEVICE`, `TPRIMS_ERR_READ_ONLY`, `TPRIMS_ERR_ALIASED`, `TPRIMS_ERR_WOULD_MATERIALIZE`, `TPRIMS_ERR_NUMERICAL`, `TPRIMS_BUSY`, `TPRIMS_ERR_WOULD_DEADLOCK`, `TPRIMS_ERR_CLOSED`, `TPRIMS_ERR_PANIC`, `TPRIMS_ERR_INTERNAL`.
- DLPack: element strides (NULL = compact row-major, per DLPack), `byte_offset` a multiple of the element size, `lanes == 1`, devices `kDLCPU`/`kDLCUDAHost`, dtypes f32, f64, complex64, complex128. Views are built over the exact addressed span (no copy); a read-only output (`DLPACK_FLAG_BITMASK_READ_ONLY`) is rejected before any write.
- Executor handles are reference counted; `close` of an owned pool is synchronous (joins worker `JoinHandle`s kept from `spawn_handler`), returns `TPRIMS_BUSY` while calls are in flight, `TPRIMS_ERR_WOULD_DEADLOCK` from a worker of the same pool, `TPRIMS_OK` on repeat; calls on a closed handle return `TPRIMS_ERR_CLOSED`; `release` of the last reference closes.
- No new crates.io packages; `publish = false`.

## Review Focus

- Null pointers for every pointer argument; `ndim` / `shape` / `strides` pointer validity is the caller's precondition, but null is checked.
- Overflow in span computation for huge extents/strides (checked arithmetic, `TPRIMS_ERR_SHAPE`).
- Close racing with an in-flight call from another thread (BUSY, never use-after-free).
- A panic inside a kernel (e.g. provider assert) becomes `TPRIMS_ERR_PANIC` with a message, and the handle remains usable.
- Symbol visibility: only selected parts' `tprims_*` symbols exported from `libtprims.so` (checked with `nm -D`).

---

### Task 1: `tprims-exec` owned pools
Add `Pool::owned(tp: rayon::ThreadPool) -> Pool<'static>` (enum inside: borrowed or owned) so a C handle can own its pool; tests for owned-pool install/broadcast.

### Task 2: `tprims-core`
DLPack mirror + `tprims_tensor_borrow_versioned` / `tprims_tensor_borrow_raw`; `dl_view<T>` / `dl_view_mut<T>` (span, offset, dtype, device checks; zero copy); status and `tprims_last_error`; `tprims_abi_version`, `tprims_has_part`; executor handles (`tprims_exec_serial`, `tprims_exec_rayon_create(nthreads, opts)`, `tprims_exec_num_threads`, `tprims_exec_set_budget`, `tprims_exec_retain/release`, `tprims_exec_close`) with an in-flight counter guard used by other parts. Rust unit tests for DLPack conversion (row-major NULL strides, column-major, negative strides, byte_offset, read-only, bad dtype/device/lanes, overflow) and handle lifecycle (busy, closed, double close, self-worker close, TLS-destructor handshake proving close joined).

### Task 3: `tprims-blas-capi` and `tprims-contract-capi`
`tprims_blas_gemm(exec, const void *alpha, tprims_tensor a, int conj_a, tprims_tensor b, int conj_b, const void *beta, tprims_tensor c)`, `tprims_blas_gemm_batched(..., int strategy, int *selected)`; `tprims_contract_plan_create(const tprims_dot_general *cfg, tprims_tensor a, b, c, int conj_a, conj_b, int strategy, uint32_t flags, tprims_contract_plan **out)`, `tprims_contract_plan_execute(plan, exec, alpha, a, b, beta, c)`, `tprims_contract_plan_selected`, `tprims_contract_plan_destroy`. dtype dispatch at this boundary only. Rust tests calling the `extern "C"` functions directly.

### Task 4: `tprims-bundle`, headers, C test
`crates/tprims-bundle` (`crate-type = ["cdylib", "staticlib"]`, features `blas`, `contract`, default both), headers, `tests/c/abi_test.c` compiled and run by a Rust integration test (`cc` + link to the built `.so`, skipped with a message if no C compiler), checking zero copy by pointer identity (results written through the caller's DLTensor), column-major and strided inputs, pool create/use/close, cross-part handle use, error strings. `nm -D` symbol check in the same test.

### Task 5: C benchmark and docs
`benchmarks/c/bench.c` + `run.sh`: per-call cost of `tprims_abi_version` (empty call), 8x8 GEMM and a 2x2x2 contraction through C vs the same calls from a Rust bench binary, serial and with a 4-thread pool created from C, recorded under the protocol. Architecture/decision-log updates (C ABI execution row: explicit handle per call — decided by the prototype), README; gate; fresh review; PR; merge after green CI (CI gains a job building the bundle and running the C test).
