#!/usr/bin/env python3
"""Check that agent skills are mirrored for every supported tool.

`.agents/skills/<name>/` is canonical (Codex and pi read it); Claude Code
reads a byte-identical copy under `.claude/skills/<name>/`, and OpenCode has a
command `.opencode/commands/<name>.md` that points at the canonical SKILL.md.
Same layout as tenferro-rs.
"""
from __future__ import annotations

import pathlib
import sys

MIRRORS = (pathlib.Path(".claude/skills"),)
OPENCODE = pathlib.Path(".opencode/commands")


def check(root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    canonical_root = root / ".agents/skills"
    skills = sorted(p for p in canonical_root.iterdir() if p.is_dir()) if canonical_root.is_dir() else []
    if not skills:
        errors.append("no skills under .agents/skills")
    for skill in skills:
        name = skill.name
        files = sorted(p.relative_to(skill) for p in skill.rglob("*") if p.is_file())
        if pathlib.Path("SKILL.md") not in files:
            errors.append(f"missing .agents/skills/{name}/SKILL.md")
        for mirror in MIRRORS:
            for rel in files:
                m = root / mirror / name / rel
                if not m.is_file():
                    errors.append(f"missing mirror file: {mirror / name / rel}")
                elif m.read_bytes() != (skill / rel).read_bytes():
                    errors.append(f"mirror differs from canonical: {mirror / name / rel}")
            mdir = root / mirror / name
            if mdir.is_dir():
                extra = sorted(p.relative_to(mdir) for p in mdir.rglob("*") if p.is_file())
                for rel in extra:
                    if rel not in files:
                        errors.append(f"unexpected mirror file: {mirror / name / rel}")
        entry = root / OPENCODE / f"{name}.md"
        if not entry.is_file():
            errors.append(f"missing OpenCode command: {OPENCODE / (name + '.md')}")
        elif f".agents/skills/{name}/SKILL.md" not in entry.read_text(encoding="utf-8"):
            errors.append(f"OpenCode command does not reference .agents/skills/{name}/SKILL.md")
    for mirror in MIRRORS:
        mroot = root / mirror
        if mroot.is_dir():
            for d in sorted(p for p in mroot.iterdir() if p.is_dir()):
                if not (canonical_root / d.name).is_dir():
                    errors.append(f"mirror skill without canonical: {mirror / d.name}")
    return errors


def main() -> int:
    errors = check(pathlib.Path(__file__).resolve().parents[1])
    for e in errors:
        print(f"check-agent-skills: {e}", file=sys.stderr)
    if not errors:
        print("check-agent-skills: ok")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
