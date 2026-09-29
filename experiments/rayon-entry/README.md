# Rayon pool entry latency

**Question:** Where does the cost of entering a Rayon pool from a non-worker thread come from: waking sleeping workers, waking the blocked caller, or fanning out to many workers? How does it compare with an OpenMP parallel region?

This is a preliminary measurement for [Prototype 1](../../docs/experiments.md#prototype-1-execution-and-ffi-boundary). It times empty closures only, so there is no numerical correctness gate. Each configuration was run one to three times in one session; no paired statistics or noise estimate yet.

## Setup

| Item | Value |
| --- | --- |
| Date | 2026-09-29 |
| CPU | Apple M5 Max, 6 performance and 12 efficiency cores (`hw.ncpu` = 18), no affinity control |
| OS | macOS 26.5.1 (Darwin 25.5.0) |
| Rust | rustc 1.96.0, release profile, `opt-level = 3` |
| Rayon | rayon 1.11.0, rayon-core 1.13.0 (`Cargo.lock`) |
| OpenMP | Homebrew libomp 22.1.5, Apple clang 21.0.0, `-O2` |
| Timed boundary | From just before the entry call to its return on the calling thread. Pool construction and 1000 warm-up entries are outside. |
| Idle gap | The caller busy-waits for the gap (it never sleeps), so only the pool's workers can go to sleep. |
| Timer | `Instant` (Rust); `CLOCK_MONOTONIC_RAW` (C). The first OpenMP run used `CLOCK_MONOTONIC`, which rounds to 1 µs on macOS. |

```sh
cargo run --release                 # src/main.rs: install, par_iter, broadcast vs gap
cargo run --release --bin split     # src/bin/split.rs: caller wake vs worker wake
```

## Results (median µs; p10 to p90 in the raw output)

| Operation | Threads | Gap 0 | Gap 100 µs or more |
| --- | --- | --- | --- |
| `install(empty)` | 1 to 4 | 3.6 to 6 | 3.5 to 12.7 |
| `spawn`, caller spins on an atomic | 1 to 4 | 0.4 to 0.7 | 6 to 8 |
| `install` then `par_iter` over 4n items | 4 | 6.5 | 9 to 16 |
| `broadcast(empty)` | 4 | 8 | 12 to 17 |
| `install(empty)` | 18 | 23 | 8 to 12.5 |
| `install` then `par_iter` | 18 | 70 | 140 to 220 |
| `broadcast(empty)` | 18 | 55 | 120 to 200 |
| nested `install` inside the same pool | 1 to 18 | 0.008 to 0.033 | same |
| raw two-thread round trip, both spin | 2 | 0.12 | |
| raw two-thread round trip, `Mutex` + `Condvar` | 2 | 3.0 | |
| OpenMP `parallel` region, default settings | 4 | 12 | 10 to 12 |
| OpenMP, `OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite` | 4 | 0.6 | 0.5 to 0.7 |
| OpenMP, default settings | 18 | 53 | 56 to 100 |
| OpenMP, active wait | 18 | 2.1 | 2.4 to 5.1 (p90 80 to 190) |

## Reading

- **Worker wake-up is real.** Idle Rayon workers go to sleep within 100 µs (32 `yield_now` rounds, `rayon-core/src/sleep/mod.rs`). Handing a job to a sleeping worker and seeing it run costs 6 to 8 µs.
- **Caller wake-up is the other half.** Even with awake workers, `install` from outside costs 3 to 6 µs because the caller blocks on a lock latch and must be woken; this matches the raw `Condvar` round trip. A spinning caller with awake workers sees 0.4 to 0.7 µs.
- **Full-width fan-out dominates.** Waking all 18 workers costs 55 to 200 µs. The heterogeneous efficiency cores are a likely contributor; a homogeneous CPU should be measured before generalizing.
- **Already inside the pool, entry is nearly free** (8 to 33 ns).
- **OpenMP is not intrinsically cheaper.** With default libomp settings it is in the same range as Rayon. Its sub-microsecond figures come from an active wait policy, which Rayon does not offer. Active waiting with more spinning threads than cores produced p90 spikes of 80 to 190 µs.

## Decision taken from this result

Recorded in the [decision log](../../docs/decision-log.md): an entry cost of about 10 µs per parallel kernel is acceptable, because serial operations never pay it and parallel width is chosen from the amount of work. Spin-wait policies are deferred until a measurement shows a need.

## Not yet measured

Break-even of real small GEMM and contraction kernels against thread count; a homogeneous x86 CPU (the [tenferro #1945](https://github.com/tensor4all/tenferro-rs/issues/1945) numbers are from AMD EPYC); barrier cost inside one SPMD call on a borrowed pool.
