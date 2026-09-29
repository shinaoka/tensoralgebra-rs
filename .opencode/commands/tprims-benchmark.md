---
description: Run, add or report a tprims-rs benchmark — build, idle cores in one L3 domain, pinned paired 1T/4T(/8T) runs, A/A noise, recorded results — following PERFORMANCE_TIPS.md.
---

Use `$ARGUMENTS` as the scope: the benchmark to run or add, or the change whose
performance is in question.

Follow `.agents/skills/tprims-benchmark/SKILL.md` step by step. Choose cores
with `python3 benchmarks/scripts/idle_cpus.py pick N`, run every measurement
through `benchmarks/scripts/pinned.sh CPUS -- CMD`, and record the commit, CPU,
core set, thread counts and A/A spread with the results. Set the build job
count with `CARGO_BUILD_JOBS` for the host; CPU pinning is Linux-only.

@.agents/skills/tprims-benchmark/SKILL.md
@PERFORMANCE_TIPS.md
