#!/usr/bin/env python3
"""Check repository-local Markdown file links (external URLs are audited separately).

Also rejects `python3 ...` commands in inline code and fenced code blocks: inside the
course venv `python` is always right, while on Windows `python3` is the Microsoft Store
stub. Package names (python3-venv, python3-dev), python3.12 and shebang lines are fine.
A line, or the fenced block, directly after `<!-- allow-python3 -->` is exempt.
"""

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
SKIP = ("target", "build", ".git", "node_modules")
PYTHON3 = re.compile(r"(?<![\w./-])python3(?= )")
INLINE = re.compile(r"(`+)(.+?)\1")
FENCE = re.compile(r"^\s*(```+|~~~+)")
ALLOW = "<!-- allow-python3 -->"


def markdown_files():
    for f in sorted(ROOT.rglob("*.md")):
        parts = f.relative_to(ROOT).parts
        if any(x.startswith(".venv") or x in SKIP for x in parts):
            continue
        yield f


def python3_commands(text):
    """(line number, line) for every python3 command inside code."""
    found, fence, allow_next, allow_block = [], None, False, False
    for number, line in enumerate(text.splitlines(), 1):
        marker = FENCE.match(line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence, allow_block = None, False
            elif not allow_block and PYTHON3.search(line) and "#!" not in line:
                found.append((number, line.strip()))
            continue
        if marker:
            fence, allow_block, allow_next = marker[1], allow_next, False
            continue
        if line.strip() == ALLOW:
            allow_next = True
            continue
        if not allow_next and any(PYTHON3.search(m[2]) for m in INLINE.finditer(line)):
            found.append((number, line.strip()))
        allow_next = False
    return found


def main():
    errors, style, count = [], [], 0
    for f in markdown_files():
        text = f.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = target.strip("<>").split("#")[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            count += 1
            if not (f.parent / unquote(target)).exists():
                errors.append(f"{f.relative_to(ROOT).as_posix()} -> {target}")
        for number, line in python3_commands(text):
            where = f"{f.relative_to(ROOT).as_posix()}:{number}"
            style.append(f"{where}: use `python`, not `python3`: {line}")
    print(f"Checked {count} local links; {len(errors)} broken")
    print("\n".join(errors))
    print(f"python3 commands in code: {len(style)}")
    print("\n".join(style))
    return int(bool(errors or style))


if __name__ == "__main__":
    raise SystemExit(main())
