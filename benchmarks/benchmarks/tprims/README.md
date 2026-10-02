# tprims benchmarks

Benchmarks for the new tprims parts. Every binary takes `--threads N`,
builds its `tprims_exec::Exec` from it (`Exec::Serial` for 1, a bounded pool
borrowed through `Exec` otherwise), rejects conflicting thread environment
variables, and prints the verified width. Measure 1T and 4T in the same
session, pinned with `taskset` inside one L3 domain on idle cores.

| Binary | Page |
| --- | --- |
| `exec_entry` | [exec_entry](exec_entry/README.md): tprims-exec entry cost, strided map and packed GEMM through `Exec` |
| `contract` | [contract](contract/README.md): binary contraction corpus, permute+GEMM vs TBLIS-style |
