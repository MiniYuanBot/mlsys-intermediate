#!/usr/bin/env python3
"""Check repository-local Markdown file links (external URLs are audited separately)."""

import re
from pathlib import Path
from urllib.parse import unquote

root = Path(__file__).resolve().parents[2]
errors = []
count = 0
for f in root.rglob("*.md"):
    if any(
        x.startswith(".venv") or x in ("target", "build", ".git", "node_modules") for x in f.parts
    ):
        continue
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", f.read_text(encoding="utf-8")):
        target = target.strip("<>").split("#")[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        count += 1
        if not (f.parent / unquote(target)).exists():
            errors.append(f"{f.relative_to(root)} -> {target}")
print(f"Checked {count} local links; {len(errors)} broken")
print("\n".join(errors))
raise SystemExit(bool(errors))
