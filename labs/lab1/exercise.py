"""Compile and run lab 1: C++17 stages A/B/C on the CPU, CUDA stage R in WSL2/Linux."""

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

LAB = 1
R_GROUPS = (
    "R3",
    "R2",
    "R1",
    "fused",
)  # each in its own process: a trapped kernel fails only its group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "bench"], required=True)
    parser.add_argument("--part", choices=["all", "A", "B", "C", "R"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.mode == "bench" and args.part not in ("all", "R"):
        parser.error(
            "the CPU benchmark combines A/B/C; use --part all (or --part R for the GPU bench)"
        )
    work = Path(__file__).resolve().parent
    build = work / "build"
    build.mkdir(exist_ok=True)
    run_cpu = args.part != "R"
    run_r = args.part == "R" or (args.part == "all" and args.device == "cuda")
    if run_r:  # R is CUDA-only; on native Windows/macOS this exits 3 with a WSL hint
        require_cuda(LAB, "R", "int4 KV fused decode-attention kernel (decode_test.cu)")
    failed = []
    if run_cpu:
        require_tool(
            "c++", LAB, "A/B/C" if args.part == "all" else args.part, f"compile {args.mode}.cpp"
        )
        exe = executable_path(build / f"cpu_{args.mode}")
        subprocess.run(
            cxx_command(work / f"{args.mode}.cpp", exe, [work / "src"]), cwd=work, check=True
        )
        if subprocess.run([str(exe), args.part], cwd=work).returncode:
            failed.append("A/B/C" if args.part == "all" else args.part)
    if run_r:
        name = "decode_test" if args.mode == "test" else "decode_bench"
        exe = executable_path(build / name)
        subprocess.run(
            cuda_command(work / f"{name}.cu", exe, [work / "src", work]), cwd=work, check=True
        )
        if args.mode == "bench":
            if subprocess.run([str(exe)], cwd=work).returncode:
                failed.append("R bench")
        else:
            bad = [g for g in R_GROUPS if subprocess.run([str(exe), g], cwd=work).returncode]
            print(
                f"SUMMARY R: {len(R_GROUPS) - len(bad)}/{len(R_GROUPS)} groups passed"
                + (f"; failed {' '.join(bad)}" if bad else "")
            )
            failed += [f"R:{g}" for g in bad]
    elif args.part == "all":
        print("NOT RUN R (needs --device cuda: nvcc + NVIDIA GPU, WSL2/Linux)")
    if failed:
        print(f"FAILED parts: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    exit_on_missing(main)
