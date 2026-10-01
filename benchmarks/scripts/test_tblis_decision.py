#!/usr/bin/env python3
"""Tests for tblis_decision.py, against synthetic session directories."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "tblis_decision", Path(__file__).with_name("tblis_decision.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def session(root: Path, name: str, threads: int, rows: dict[tuple[str, str], float]) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"contract-{threads}t.csv", "w") as f:
        f.write("case,variant,threads,median_ns,samples\n")
        for (case, variant), ns in rows.items():
            f.write(f"{case},{variant},{threads},{ns},5\n")
    return d


def corpus(root: Path) -> Path:
    p = root / "corpus.json"
    p.write_text(
        json.dumps(
            {
                "source": {},
                "entries": [
                    {"name": "a", "calls": 2},
                    {"name": "b", "calls": 3},
                ],
            }
        )
    )
    return p


def rows(pg: float, tblis: float, scale: float = 1.0) -> dict[tuple[str, str], float]:
    return {
        ("a", "pg_exec"): pg * scale,
        ("b", "pg_exec"): pg * scale,
        ("a", "tblis_exec"): tblis * scale,
        ("b", "tblis_exec"): tblis * scale,
        ("a", "pg_plan"): 10.0,
        ("b", "pg_plan"): 10.0,
        ("a", "tblis_plan"): 10.0,
        ("b", "tblis_plan"): 10.0,
    }


def test_pair_mode_passes_when_the_new_build_is_not_slower():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        c = corpus(root)
        old = [session(root, "old1", 1, rows(100.0, 100.0)), session(root, "old2", 1, rows(100.0, 100.0, 1.01))]
        new = [session(root, "new1", 1, rows(100.0, 100.0)), session(root, "new2", 1, rows(100.0, 100.0, 0.99))]
        argv = ["--corpus", str(c), "--old", *map(str, old), "--new", *map(str, new), "--gate", "1"]
        assert m.main(argv) == 0


def test_pair_mode_fails_when_a_group_is_more_than_the_limit_slower():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        c = corpus(root)
        old = [session(root, "old1", 1, rows(100.0, 100.0))]
        new = [session(root, "new1", 1, rows(120.0, 100.0))]
        argv = ["--corpus", str(c), "--old", *map(str, old), "--new", *map(str, new), "--gate", "1"]
        assert m.main(argv) == 1


def test_pair_mode_accepts_a_slowdown_within_the_noise():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        c = corpus(root)
        # The same side varies by 10%, so a 6% difference is not a finding.
        old = [
            session(root, "old1", 1, rows(100.0, 100.0)),
            session(root, "old2", 1, rows(110.0, 110.0)),
        ]
        new = [
            session(root, "new1", 1, rows(106.0, 100.0)),
            session(root, "new2", 1, rows(106.0, 100.0)),
        ]
        argv = ["--corpus", str(c), "--old", *map(str, old), "--new", *map(str, new), "--gate", "1"]
        assert m.main(argv) == 0


def test_pair_mode_needs_both_sides_and_no_positional_sessions():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        c = corpus(root)
        old = session(root, "old1", 1, rows(100.0, 100.0))
        assert m.main(["--corpus", str(c), "--old", str(old), "--gate", "1"]) == 2
        assert m.main(["--corpus", str(c), "--gate", "1"]) == 2
        # `--new` is greedy, so the positional has to come first.
        assert (
            m.main([str(old), "--corpus", str(c), "--old", str(old), "--new", str(old), "--gate", "1"])
            == 2
        )


def test_incomplete_sessions_are_refused():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        c = corpus(root)
        partial = {("a", "pg_exec"): 1.0, ("a", "tblis_exec"): 1.0}
        old = session(root, "old1", 1, partial)
        new = session(root, "new1", 1, rows(100.0, 100.0))
        argv = ["--corpus", str(c), "--old", str(old), "--new", str(new), "--gate", "1"]
        assert m.main(argv) == 2


def test_strategy_mode_still_decides():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        c = corpus(root)
        # TBLIS an order of magnitude faster than pg in both sessions: switch.
        s = [session(root, f"s{i}", 1, rows(1000.0, 100.0)) for i in range(2)]
        assert m.main([str(p) for p in s] + ["--corpus", str(c), "--gate", "1"]) == 0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in tests:
        f()
    print(f"test_tblis_decision: {len(tests)} tests passed")
