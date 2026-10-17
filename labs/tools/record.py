#!/usr/bin/env python3
"""Capture an extra experiment command in the same extractable record format."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from source_files import project_files

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--label", required=True)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
if not cmd:
    p.error("provide command after --")
root = Path(__file__).resolve().parents[2]
labs = root / "labs"
record = {
    "label": a.label,
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "platform": platform.platform(),
    "python": sys.version,
    "commands": [],
    "source_sha256": {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in project_files(labs)
        if f.suffix in {".py", ".cpp", ".hpp", ".cuh", ".cu", ".rs", ".toml"}
    },
}
record["packages"] = {}
for package in ["torch", "transformers", "numpy", "matplotlib"]:
    try:
        record["packages"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass
start = time.perf_counter()
try:
    result = subprocess.run(
        cmd,
        cwd=labs,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
except OSError as error:
    result = subprocess.CompletedProcess(cmd, 127, "", str(error) + "\n")
record["wall_seconds"] = time.perf_counter() - start
record["commands"].append(
    {"argv": cmd, "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
)
record["status"] = "passed" if result.returncode == 0 else "failed"
out = labs / "results" / f"{a.label}-{time.time_ns()}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
)
print(result.stdout, end="")
print(result.stderr, file=sys.stderr, end="")
print("RECORD:", out)
raise SystemExit(result.returncode)
