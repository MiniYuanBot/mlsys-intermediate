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

import lab_meta
from platform_tools import (
    MISSING_EXIT,
    cargo_command,
    compiler_family,
    find_cxx,
    find_nvcc,
    host_platform,
    stage_helpers,
)
from source_files import ignore_files, project_files

ROOT = Path(__file__).resolve().parents[2]
LABS = ROOT / "labs"
GPU_LOCK = Path("/tmp/mlsys-gpu.lock")  # same file fault_inject.py and `flock` users take


def _ancestor_pids():
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            break
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
    return pids


def _lock_held_by_ancestor():
    """True when an outer `flock /tmp/mlsys-gpu.lock ...` already wraps this process."""
    try:
        inode = GPU_LOCK.stat().st_ino
        locks = Path("/proc/locks").read_text().splitlines()
    except OSError:
        return False
    holders = {
        int(f[4])
        for f in (line.split() for line in locks)
        if len(f) > 5 and f[1] == "FLOCK" and f[5].rsplit(":", 1)[-1] == str(inode)
    }
    return bool(holders & _ancestor_pids())


def acquire_gpu_lock(record):
    """Serialize CUDA runs on a shared GPU so timings never include another job's kernels.

    Returns the open lock file (keep it alive) or None. Only Linux/WSL have fcntl + /proc.
    """
    try:
        import fcntl
    except ImportError:
        record["gpu_lock"] = "unsupported"
        return None
    if _lock_held_by_ancestor():
        record["gpu_lock"] = "inherited"
        return None
    handle = open(GPU_LOCK, "a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"waiting for {GPU_LOCK} (another CUDA job is running)...", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
    record["gpu_lock"] = "acquired"
    return handle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lab", type=int, choices=lab_meta.LAB_NUMBERS, required=True)
    p.add_argument("--impl", choices=["student", "reference"], default="student")
    p.add_argument("--mode", choices=["test", "bench"], default="test")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--part", choices=["all", "A", "B", "C", "R"], default="all")
    p.add_argument("--output", type=Path)
    p.add_argument("--cuda-arch", default=os.environ.get("MLSYS_CUDA_ARCH", "native"))
    args = p.parse_args()
    meta = lab_meta.get(args.lab)
    if not meta.exists():
        print(lab_meta.missing_lab_message(args.lab), file=sys.stderr)
        return 2
    if args.part == "R" and ("R" not in meta.stages or not meta.r_implemented):
        reason = "no R stage" if "R" not in meta.stages else "not implemented yet in this lab"
        print(f"NOT RUN R ({reason}): lab{args.lab}", file=sys.stderr)
        return 2
    os.environ["MLSYS_CUDA_ARCH"] = args.cuda_arch
    # Read by platform_tools inside the temporary copy (WSL redirect hints).
    os.environ["MLSYS_LABS_DIR"] = str(LABS)
    os.environ["MLSYS_RERUN"] = " ".join(
        [f"--lab {args.lab} --impl {args.impl} --mode {args.mode}"]
        + [f"--device {args.device}", f"--part {args.part}"]
        + ([f"--cuda-arch {args.cuda_arch}"] if args.cuda_arch != "native" else [])
    )
    record = {
        "lab": args.lab,
        "implementation": args.impl,
        "mode": args.mode,
        "device": args.device,
        "part": args.part,
        "host": platform.platform(),
        "host_platform": host_platform(),
        "cuda_arch": args.cuda_arch,
        "python": sys.version,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commands": [],
        "status": "failed",
    }
    record["packages"] = {}
    for package in [
        "torch",
        "triton",
        "transformers",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "matplotlib",
        "pytest",
    ]:
        try:
            record["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    record["pins_sha256"] = lab_meta.pins_hash()
    stamp = Path(sys.prefix) / ".mlsys-pins"
    record["venv_pins_sha256"] = (
        stamp.read_text(encoding="utf-8").split()[0] if stamp.is_file() else None
    )
    original = ROOT / "labs" / f"lab{args.lab}"
    sources = [original]
    if args.impl == "reference":
        sources.append(ROOT / "solutions" / f"lab{args.lab}")
    record["source_sha256"] = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for src in sources
        for f in project_files(src)
    }
    helper = LABS / "tools/platform_tools.py"
    record["source_sha256"][str(helper.relative_to(ROOT))] = hashlib.sha256(
        helper.read_bytes()
    ).hexdigest()
    try:
        record["cxx"] = find_cxx()
        record["cxx_family"] = compiler_family(record["cxx"])
    except RuntimeError as error:
        record["cxx"] = str(error)
    record["nvcc_override"] = os.environ.get("NVCC")
    cargo = cargo_command()
    found = {"nvcc": find_nvcc(), "cargo": cargo}
    if cargo:
        found["rustc"] = shutil.which("rustc")
    for tool in ["c++", "g++", "clang++", "cl", "nvcc", "cargo", "rustc", "nvidia-smi"]:
        executable = found.get(tool) or shutil.which(tool)
        if executable:
            cmd = (
                [executable, "--version"]
                if tool != "nvidia-smi"
                else [
                    executable,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            )
            if tool == "cl":
                cmd = [executable]
            q = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            record[tool] = (q.stdout + q.stderr).strip()
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f"mlsys-lab{args.lab}-") as temp:
            work = Path(temp) / f"lab{args.lab}"
            shutil.copytree(original, work, ignore=ignore_files)
            if args.impl == "reference":
                shutil.copytree(sources[1], work, dirs_exist_ok=True, ignore=ignore_files)

            stage_helpers(work)

            def run(cmd, env=None):
                print("+", " ".join(map(str, cmd)), flush=True)
                out = subprocess.run(
                    list(map(str, cmd)),
                    cwd=work,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env
                    or {
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONUTF8": "1",
                        "PYTHONIOENCODING": "utf-8",
                    },
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
                if out.returncode == MISSING_EXIT:
                    record["status"] = "env-missing"
                    raise RuntimeError("missing dependency (see MISSING line above)")
                if out.returncode:
                    raise RuntimeError(f"command failed with exit {out.returncode}")

            if not (work / "exercise.py").is_file():
                raise RuntimeError(f"lab{args.lab} is missing its exercise.py driver")
            gpu_lock = acquire_gpu_lock(record) if args.device == "cuda" else None  # noqa: F841
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
        if record["status"] != "env-missing":  # the child already printed MISSING + fix
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
        out.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"RECORD: {out}")
    return {"passed": 0, "env-missing": MISSING_EXIT}.get(record["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
