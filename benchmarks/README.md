# tprims-bench

Benchmarks for the tprims stack. The package started as an import of
[strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite)
(upstream `0550611`, with history); the strided-rs kernel benchmarks moved
back there when strided-rs became an external dependency again, and this
package now holds only the tprims benchmarks.

| Binary | Page |
| --- | --- |
| `exec_entry` | [exec_entry](benchmarks/tprims/exec_entry/README.md) |
| `blas` | [blas](benchmarks/tprims/blas/README.md) |
| `linalg` | [linalg](benchmarks/tprims/linalg/README.md) |
| `contract` | [contract](benchmarks/tprims/contract/README.md) |
| `capi_rust` | [C ABI comparison](c/README.md) |

## Build and run

The package is a member of the root workspace, so binaries land in the
repository's `target/`:

```bash
cargo build --release -p tprims-bench --bins   # jobs from CARGO_BUILD_JOBS
cpus=$(python3 scripts/idle_cpus.py pick 4)          # idle CPUs of one L3 domain
scripts/pinned.sh "${cpus%%,*}" -- ../target/release/blas --threads 1
scripts/pinned.sh "$cpus"       -- ../target/release/blas --threads 4
```

The full procedure is the `tprims-benchmark` skill
(`../.agents/skills/tprims-benchmark/SKILL.md`).

## Rules

Measure every tensor-sized case at 1 and 4 threads in the same run, verify the
effective thread count at startup, pin with `taskset` inside one L3 domain,
never run two benchmarks at once, and record the tprims-rs commit, CPU, core
set and profile beside every published table. See the root
[`PERFORMANCE_TIPS.md`](../PERFORMANCE_TIPS.md) and [`AGENTS.md`](AGENTS.md).
