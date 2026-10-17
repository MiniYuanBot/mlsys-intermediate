#!/usr/bin/env python3
"""Run independent lab tests/benchmarks without overwriting student work."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from source_files import ignore_files, project_files

ROOT = Path(__file__).resolve().parents[2]
LABS = ROOT / "labs"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lab", type=int, choices=range(1, 9), required=True)
    p.add_argument("--impl", choices=["student", "reference"], default="student")
    p.add_argument("--mode", choices=["test", "bench"], default="test")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--part", choices=["all", "A", "B", "C"], default="all")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    record = {
        "lab": args.lab,
        "implementation": args.impl,
        "mode": args.mode,
        "device": args.device,
        "part": args.part,
        "host": platform.platform(),
        "python": sys.version,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commands": [],
        "status": "failed",
    }
    record["packages"] = {}
    for package in ["torch", "transformers", "numpy", "matplotlib"]:
        try:
            record["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    original = ROOT / "labs" / f"lab{args.lab}"
    sources = [original]
    if args.impl == "reference":
        sources.append(ROOT / "solutions" / f"lab{args.lab}")
    record["source_sha256"] = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for src in sources
        for f in project_files(src)
    }
    for tool in ["g++", "nvcc", "rustc", "nvidia-smi"]:
        if shutil.which(tool):
            cmd = (
                [tool, "--version"]
                if tool != "nvidia-smi"
                else [tool, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
            )
            q = subprocess.run(cmd, capture_output=True, text=True)
            record[tool] = q.stdout.strip()
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f"mlsys-lab{args.lab}-") as temp:
            work = Path(temp) / f"lab{args.lab}"
            shutil.copytree(original, work, ignore=ignore_files)
            if args.impl == "reference":
                shutil.copytree(sources[1], work, dirs_exist_ok=True, ignore=ignore_files)

            def run(cmd, env=None):
                print("+", " ".join(map(str, cmd)), flush=True)
                out = subprocess.run(
                    list(map(str, cmd)),
                    cwd=work,
                    capture_output=True,
                    text=True,
                    env=env or {**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                print(out.stdout, end="", flush=True)
                print(out.stderr, end="", file=sys.stderr, flush=True)
                record["commands"].append(
                    {
                        "argv": list(map(str, cmd)),
                        "returncode": out.returncode,
                        "stdout": out.stdout,
                        "stderr": out.stderr,
                    }
                )
                if out.returncode:
                    raise RuntimeError(f"command failed with exit {out.returncode}")

            if not (work / "exercise.py").is_file():
                raise RuntimeError(f"lab{args.lab} is missing its exercise.py driver")
            run(
                [
                    sys.executable,
                    "exercise.py",
                    "--mode",
                    args.mode,
                    "--device",
                    args.device,
                    "--part",
                    args.part,
                ]
            )
            record["status"] = "passed"
    except (RuntimeError, OSError) as e:
        record["error"] = str(e)
        print(str(e), file=sys.stderr)
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        out = (
            args.output
            or LABS
            / "results"
            / f"lab{args.lab}-{args.impl}-{args.part}-{args.mode}-{args.device}-{time.time_ns()}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"RECORD: {out}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
