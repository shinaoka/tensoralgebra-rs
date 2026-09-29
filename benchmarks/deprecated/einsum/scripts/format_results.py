"""Parse benchmark log files and format results as a markdown table.

Usage:
    python scripts/format_results.py data/results/rust_*.log data/results/julia_*.log

Outputs a markdown table in the style of strided-rs/strided-opteinsum README.
"""

import re
import sys


def parse_log(filepath: str) -> tuple[dict[tuple[str, str, str], tuple[float, float | None]], dict[str, str]]:
    """Parse a benchmark log and return (results, metadata).

    results: {(instance, strategy, mode): (median_ms, iqr_ms or None)}
    metadata: thread environment variables found in the log
    """
    results: dict[tuple[str, str, str], tuple[float, float | None]] = {}
    current_mode = None
    current_strategy = None
    metadata: dict[str, str] = {}
    has_iqr = False  # whether current log has IQR column

    # Detect engine from header
    engine = None
    rust_backend = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()

            # Detect engine
            if "strided-opteinsum" in line.lower() and engine is None:
                engine = "strided-opteinsum"
            elif "julia einsum" in line.lower() and engine is None:
                engine = "julia"

            # Detect Rust backend variant: "Backend: strided-opteinsum(faer)" or "Backend: strided-opteinsum"
            m = re.match(r"^Backend:\s+(.+)", line)
            if m and rust_backend is None:
                rust_backend = m.group(1).strip()

            # Extract thread settings from log lines like:
            #   "RAYON_NUM_THREADS=4, OMP_NUM_THREADS=4"
            #   "OMP_NUM_THREADS=4, JULIA_NUM_THREADS=4"
            for var in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "JULIA_NUM_THREADS"):
                m2 = re.search(rf"{var}=(\d+)", line)
                if m2 and var not in metadata:
                    metadata[var] = m2.group(1)

            # Parse strategy header
            # Rust: "Strategy: opt_flops"
            m = re.match(r"^Strategy:\s+(\w+)", line)
            if m:
                current_strategy = m.group(1)
                current_mode = rust_backend or "strided-opteinsum"
                continue

            # Julia: "Mode: omeinsum_path / Strategy: opt_flops"
            m = re.match(r"^Mode:\s+(\w+)\s*/\s*Strategy:\s+(\w+)", line)
            if m:
                current_mode = m.group(1)
                current_strategy = m.group(2)
                continue

            # Detect IQR column from header
            if line.startswith("Instance") and "IQR" in line:
                has_iqr = True
                continue

            # Skip headers and separators
            if line.startswith("Instance") or line.startswith("-"):
                continue

            # Parse data line: name, tensors, log10flops, log2size, median_ms [, iqr_ms]
            parts = line.split()
            if len(parts) >= 5 and current_mode and current_strategy:
                try:
                    name = parts[0]
                    if has_iqr and len(parts) >= 6:
                        median_ms = float(parts[-2])
                        iqr_str = parts[-1]
                        iqr_ms = float(iqr_str) if iqr_str != "-" else None
                    else:
                        median_ms = float(parts[-1])
                        iqr_ms = None
                    key = (name, current_strategy, current_mode)
                    results[key] = (median_ms, iqr_ms)
                except (ValueError, IndexError):
                    continue

    return results, metadata


def format_markdown_table(all_results: dict, metadata: dict[str, str]) -> str:
    """Format results as a markdown table grouped by strategy."""
    # Collect all instances and modes
    instances = sorted(set(name for name, _, _ in all_results.keys()))
    strategies = sorted(set(strat for _, strat, _ in all_results.keys()))
    modes = sorted(set(mode for _, _, mode in all_results.keys()))

    # Check if any result has IQR data
    has_iqr = any(iqr is not None for _, (_, iqr) in all_results.items())

    # Modes to exclude from output (unfair comparison: different contraction path)
    excluded_modes = {"omeinsum_opt"}
    modes = [m for m in modes if m not in excluded_modes]

    # Determine column order
    preferred_order = [
        "strided-opteinsum(faer)",
        "strided-opteinsum(blas)",
        "strided-opteinsum",
        "omeinsum_path",
        "tensorops",
    ]
    mode_order = []
    for m in preferred_order:
        if m in modes:
            mode_order.append(m)
    for m in modes:
        if m not in mode_order:
            mode_order.append(m)

    mode_labels = {
        "strided-opteinsum": "strided-rs (ms)",
        "strided-opteinsum(faer)": "strided-rs faer (ms)",
        "strided-opteinsum(blas)": "strided-rs OpenBLAS (ms)",
        "omeinsum_path": "OMEinsum.jl OpenBLAS (ms)",
        "omeinsum_opt": "OMEinsum.jl opt (ms)",
        "tensorops": "TensorOperations.jl (ms)",
    }

    lines = []

    for strategy in strategies:
        lines.append(f"#### Strategy: {strategy}")
        lines.append("")
        thread_vars = ", ".join(
            f"{k}={metadata[k]}" for k in sorted(metadata) if k in metadata
        )
        if thread_vars:
            lines.append(f"Median ± IQR (ms). {thread_vars}." if has_iqr else f"Median time (ms). {thread_vars}.")
        else:
            lines.append("Median ± IQR (ms)." if has_iqr else "Median time (ms).")
        lines.append("")

        # Header
        cols = [mode_labels.get(m, m) for m in mode_order]
        header = "| Instance | " + " | ".join(cols) + " |"
        separator = "|---|" + "|".join("---:" for _ in cols) + "|"
        lines.append(header)
        lines.append(separator)

        # Data rows
        for name in instances:
            # Find the minimum median value across modes for this row
            medians = {}
            for mode in mode_order:
                key = (name, strategy, mode)
                if key in all_results:
                    medians[mode] = all_results[key][0]
            min_val = min(medians.values()) if medians else None

            row = [name]
            for mode in mode_order:
                key = (name, strategy, mode)
                if key in all_results:
                    median_ms, iqr_ms = all_results[key]
                    if has_iqr and iqr_ms is not None:
                        formatted = f"{median_ms:.3f} ± {iqr_ms:.3f}"
                    else:
                        formatted = f"{median_ms:.3f}"
                    if min_val is not None and median_ms == min_val:
                        formatted = f"**{formatted}**"
                    row.append(formatted)
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile1> [logfile2] ...")
        sys.exit(1)

    all_results = {}
    all_metadata: dict[str, str] = {}
    for filepath in sys.argv[1:]:
        results, metadata = parse_log(filepath)
        all_results.update(results)
        all_metadata.update(metadata)

    print(format_markdown_table(all_results, all_metadata))


if __name__ == "__main__":
    main()
