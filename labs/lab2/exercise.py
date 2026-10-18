"""Compile and run lab 2: host stages A/C with a C++17 compiler; B, C's GPU integration
and R with nvcc in WSL2/Linux.

GPU tests run one group per process (see gpu_test.cu) so that a trapped kernel from an
unfinished TODO fails only its own group and every other check still reports.
"""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_tools"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from platform_tools import (
    cuda_command,
    cxx_command,
    executable_path,
    exit_on_missing,
    require_cuda,
    require_tool,
)

LAB = 2
GPU_GROUPS = {"B": ("B1", "B2", "B3", "B4", "B5"), "C": ("C",), "R": ("R1", "R2", "R3", "R4", "R5")}
CUBLAS = ["-lcublas", "-lcublasLt"]  # only gpu_bench.cu links them (strong baselines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "bench"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--part", choices=["all", "A", "B", "C", "R"], default="all")
    args = parser.parse_args()
    work = Path(__file__).resolve().parent
    build = work / "build"
    build.mkdir(exist_ok=True)
    part, cuda = args.part, args.device == "cuda"
    run_host = part in ("A", "C", "all")
    # B and R exist only on the GPU, so asking for them implies CUDA.
    gpu_parts = [p for p in ("B", "C", "R") if part == p or (part == "all" and cuda)]
    if part == "C" and not cuda:
        gpu_parts = []
    if gpu_parts:
        require_cuda(LAB, "/".join(gpu_parts), "W4/FP8 GEMV kernels (gpu_test.cu, gpu_bench.cu)")
    failed = []
    if run_host:
        require_tool(
            "c++", LAB, "A/C" if part == "all" else part, "compile quant_test.cpp (host algorithms)"
        )
        exe = executable_path(build / "quant_test")
        subprocess.run(cxx_command(work / "quant_test.cpp", exe, [work]), cwd=work, check=True)
        host_part = "all" if part == "all" else part
        extra = ["bench"] if args.mode == "bench" else []
        if subprocess.run([str(exe), host_part, *extra], cwd=work).returncode:
            failed.append("A/C host" if part == "all" else f"{part} host")
    if gpu_parts and args.mode == "test":
        exe = executable_path(build / "gpu_test")
        subprocess.run(cuda_command(work / "gpu_test.cu", exe, [work]), cwd=work, check=True)
        groups = [g for p in gpu_parts for g in GPU_GROUPS[p]]
        bad = [g for g in groups if subprocess.run([str(exe), g], cwd=work).returncode]
        print(
            f"SUMMARY GPU: {len(groups) - len(bad)}/{len(groups)} groups passed"
            + (f"; failed {' '.join(bad)}" if bad else "")
        )
        failed += [f"GPU:{g}" for g in bad]
    if gpu_parts and args.mode == "bench" and ("B" in gpu_parts or "R" in gpu_parts):
        exe = executable_path(build / "gpu_bench")
        subprocess.run(
            cuda_command(work / "gpu_bench.cu", exe, [work]) + CUBLAS, cwd=work, check=True
        )
        section = (
            "all" if ("B" in gpu_parts and "R" in gpu_parts) else ("B" if "B" in gpu_parts else "R")
        )
        if subprocess.run([str(exe), section], cwd=work).returncode:
            failed.append(f"GPU bench {section}")
    if part == "all" and not cuda:
        print("NOT RUN B, C-GPU, R (need --device cuda: nvcc + NVIDIA GPU, WSL2/Linux)")
    if failed:
        print(f"FAILED parts: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    exit_on_missing(main)
