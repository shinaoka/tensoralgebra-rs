#!/usr/bin/env python3
"""Regenerate the instruction-only comparison from retained Callgrind windows."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
refs = {
    "baseline": "ec585b8a4bf0af96863a6136f0b1f8e9c1aeadba",
    "optimized": "17e05ffb168d0f826529ea2c3aceeb4ec851448b",
    "benchmark": "8ee3945ae90a1be77354c19011e99c648b7a5dcc",
}
cases = ["mul_2048", "mul_odd", "mul_strided", "axpby_1m", "tril_1024", "triu_rect", "diag_rank2"]
rows = []
for case in cases:
    row = {"case": case}
    for variant in ("baseline", "optimized"):
        stem = root / "raw" / f"{variant}-{case}"
        text = stem.with_suffix(".callgrind").read_text()
        log = stem.with_suffix(".log").read_text()
        assert "events: Ir" in text
        summaries = [int(line.split()[1]) for line in text.splitlines() if line.startswith("summary:")]
        assert len(summaries) == 1 and summaries[0] > 0
        assert f"CHECK {case} passed; threads=1 policy=Sequential samples=3" in log
        row[variant + "_total_ir"] = summaries[0]
        row[variant + "_ir_per_call"] = summaries[0] / 3
    row["ir_reduction_percent"] = 100 * (1 - row["optimized_total_ir"] / row["baseline_total_ir"])
    rows.append(row)
report = {"revisions": refs, "samples": 3, "kernel_threads": 1, "cpu_affinity": "16", "l3_domain": "16-23", "events": "Ir", "native_timing": False, "results": rows}
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
lines = [
    "# Dense CPU kernel instruction comparison", "",
    "**Instruction counts, not elapsed times, bandwidth, or predicted speedups.**", "",
    f"- Baseline strided-rs: `{refs['baseline']}` (migration completed, before optimization).",
    f"- Optimized strided-rs: `{refs['optimized']}`.",
    f"- Benchmark source: `{refs['benchmark']}`.",
    "- AMD EPYC 7713P; CPU 16 pinned (L3 domain 16–23); explicit Sequential/1T; no BLAS.",
    "- Rust 1.97.1, release; Valgrind 3.22.0; three collected calls after one excluded warmup.",
    "- Both multiply implementations ran pulp V3/AVX2; this is not a scalar-versus-SIMD comparison.",
    "- Other users' Julia jobs were active; **no native timing comparison was performed**.", "",
    "| Case | Baseline Ir/call | Optimized Ir/call | Ir reduction |",
    "|---|---:|---:|---:|",
]
for row in rows:
    lines.append(f"| {row['case']} | {row['baseline_ir_per_call']:,.0f} | {row['optimized_ir_per_call']:,.0f} | {row['ir_reduction_percent']:.2f}% |")
lines += ["", "Contiguous multiply now uses full-vector loads/stores, reserving partial accesses for its tail.",
          "Triangle masks copy only retained intervals and fill the rest, rather than copying and then masking.",
          "AXPBY and noncontiguous multiply are unchanged controls. Diagonal embedding differs by 28 Ir/call",
          "(0.002%, displayed as -0.00%); no improvement or meaningful regression is claimed for it.", "",
          "The baseline already includes migration. These numbers do not measure migration versus old tenferro,",
          "and do not include tenferro dispatch, input materialization, allocation or buffer-pool overhead.", "",
          "Every case checks all output values against nonzero scalar references outside the collection window.",
          "Memcheck reports zero errors for the seven benchmark cases and separate real/complex uninitialized",
          "SIMD output tests. Leak checking was disabled; this is not a leak-free claim.", "",
          "See raw/environment.txt for image/container IDs, compiler information and binary/source/lockfile hashes,",
          "raw/features.txt for effective crate features, and raw/*.callgrind plus *.log for primary evidence.",
          "Regenerate with `python3 result/amd-cpu/dense-kernels/summarize.py`."]
lines += ["", "## Validation and remaining work", ""]
for name in ("strided-optimized-suite", "tenferro-optimized-suite"):
    log = (root / "raw" / (name + ".log")).read_text()
    assert "test result: FAILED" not in log
    passed = sum(map(int, re.findall(r"test result: ok\. (\d+) passed", log)))
    assert passed > 0
    lines.append(f"- {name}: {passed} passed (including doctests), zero failures.")
for name in ("memcheck", "simd-memcheck"):
    assert "ERROR SUMMARY: 0 errors" in (root / "raw" / (name + ".log")).read_text()
lines += [
    "- Focused release SIMD/dense tests and SIMD-disabled dense tests also passed; logs retained.",
    "- Tenferro tests used its default CPU feature set (faer), with explicit local strided path overrides.",
    "- Strict Clippy failed on pre-existing lint errors in unchanged dependency/basic code under Rust 1.97.1; both logs are retained. No passing strict-lint gate is claimed.",
    "- Quiet-host native measurements remain pending.",
    "- Tenferro consumer base is `d8759f4320a337d2399f4a87dfec55af51d2ebf1`; its exact uncommitted migration diff is raw/tenferro-migration.patch and the tested override is raw/cpu-kernel-migration-cargo.toml (local absolute paths).",
    "- At measurement time the tenferro git dependency pin was unchanged; these historical local-override tests alone did not establish a ready-to-merge dependency chain. Subsequent integration is tracked by strided-rs PR259 (merged commit `5bc5ab75`) and tenferro-rs PR1807.",
]
(root / "README.md").write_text("\n".join(lines) + "\n")
