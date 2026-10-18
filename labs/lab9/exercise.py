"""Build and run lab9: C++17 for A/B/C and the CPU half of R, CUDA for R1-R4 (WSL/Linux)."""

import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_tools"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from platform_tools import (
    cuda_command,
    cxx_command,
    executable_path,
    exit_on_missing,
    labs_directory,
    require_cuda,
    require_tool,
)

LAB = 9
WORK = Path(__file__).resolve().parent
BUILD = WORK / "build"
REFERENCE_PROFILE = "data/rtx4060_reference_profile.csv"


def run(command):
    subprocess.run(list(map(str, command)), cwd=WORK, check=True)


def build_cpp(name):
    exe = executable_path(BUILD / name)
    run(cxx_command(WORK / f"{name}.cpp", exe, [WORK / "src"]))
    return exe


def build_cuda():
    """profile.cu with cuBLAS if it links, otherwise without (gemm_cublas rows skipped)."""
    exe = executable_path(BUILD / "profile")
    command = cuda_command(WORK / "profile.cu", exe, [WORK / "src"])
    linked = subprocess.run(command + ["-lcublas"], cwd=WORK)
    if linked.returncode:
        print(
            "NOTE cuBLAS did not link; rebuilding profile.cu without the cuBLAS comparison",
            flush=True,
        )
        run(command + ["-DHS_NO_CUBLAS"])
    return exe


def host_note():
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    query = [
        smi,
        "--query-gpu=name,driver_version,clocks.max.sm,power.limit",
        "--format=csv,noheader",
    ]
    try:
        gpu = subprocess.run(query, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "nvidia-smi unavailable"
    return f"{gpu}; {platform.platform()}".replace(",", ";").replace("\n", " ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["test", "bench"], default="test")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--part", choices=["A", "B", "C", "R", "all"], default="all")
    a = p.parse_args()
    stage = "A/B/C" if a.part == "all" else a.part
    require_tool("c++", LAB, stage, f"compile {a.mode}.cpp")
    gpu = a.device == "cuda" and a.part in ("R", "all")
    if gpu:
        require_cuda(LAB, "R", "R1-R4 cudaEvent profiling (profile.cu)")
    BUILD.mkdir(exist_ok=True)
    exe = build_cpp(a.mode)
    if a.mode == "test":
        run([exe, a.part])  # 'all' includes the CPU half of R (R5/R6)
        if gpu:
            run([build_cuda(), "test"])
        elif a.part in ("R", "all"):
            print("NOT RUN R1-R4 (needs --device cuda in WSL/Linux); R5/R6 checked on CPU")
        return 0
    if a.part != "R":
        run([exe, a.part])
    if a.part in ("R", "all"):
        profile = REFERENCE_PROFILE
        if gpu:
            out = labs_directory() / "results" / "lab9" / "rtx4060_profile.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            run([build_cuda(), "bench", out, host_note()])
            profile = out
        else:
            print(f"NOTE R bench without --device cuda replays the shipped {REFERENCE_PROFILE}")
        run([exe, "R", profile])
    return 0


if __name__ == "__main__":
    exit_on_missing(main)
