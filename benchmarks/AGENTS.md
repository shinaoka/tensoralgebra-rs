@CLAUDE.md

## Critical Rules

- **NEVER run benchmarks in parallel.** Run them sequentially (1 thread first, then 4 threads, etc.); concurrent runs interfere and produce misleading results.
- **Record the exact tprims-rs commit for every published result table**, next to the table, together with the CPU, core set and profile.
- **Always pin CPU cores with `taskset` on Linux, including 1T**, inside one L3/CCD domain (`lscpu -e` maps cores to L3). Check `ps -eo pid,psr,%cpu,comm --sort=-%cpu | head` for busy cores first and avoid their CCD. On macOS, state that pinning was unavailable.
