"""Compile and run lab 7: C++17 stages A/B/C on the CPU, CUDA C3/C4/C6 and stage R in WSL2/Linux."""

import argparse
import subprocess
import sys
from pathlib import Path

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

LAB = 7
# Each group runs in its own process: a trapped kernel or a context kill fails only its group.
R_GROUPS = ("R1", "R2", "race", "R3", "R4", "R5")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["test", "bench"], default="test")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--part", choices=["A", "B", "C", "R", "all"], default="all")
    a = p.parse_args()
    work = Path(__file__).resolve().parent
    build = work / "build"
    run_cpu = a.part != "R"
    run_cuda_c = a.part in ("C", "all") and a.device == "cuda"
    run_r = a.part == "R" or (a.part == "all" and a.device == "cuda")
    if run_cuda_c:
        require_cuda(LAB, "C", "C3/C4/C6 offloaded expert pipeline (cuda_exercise.cu)")
    if run_r:
        require_cuda(LAB, "R", "MB-scale expert pipeline with cudaEvent timing (r_test.cu)")
    if run_cpu:
        require_tool("c++", LAB, "A/B/C" if a.part == "all" else a.part, f"compile {a.mode}.cpp")
    build.mkdir(exist_ok=True)
    failed = []

    def compile_and_run(source, executable, arguments, cuda=False):
        builder = cuda_command if cuda else cxx_command
        includes = [work / "src", work] if cuda else [work / "src"]
        subprocess.run(builder(work / source, executable, includes), cwd=work, check=True)
        return subprocess.run([str(executable), *arguments], cwd=work).returncode

    if run_cpu:
        exe = executable_path(build / f"cpu_{a.mode}")
        if compile_and_run(f"{a.mode}.cpp", exe, [a.part]):
            failed.append("A/B/C" if a.part == "all" else a.part)
    if run_cuda_c:
        exe = executable_path(build / "cuda_exercise")
        if compile_and_run("cuda_exercise.cu", exe, [a.mode], cuda=True):
            failed.append("C cuda")
    elif a.part in ("C", "all"):
        print(
            "CPU C1/C2/C5 checked; required CUDA C3/C4/C6 need --device cuda (no simulated GPU score)."
        )
    if run_r:
        name = "r_test" if a.mode == "test" else "r_bench"
        exe = executable_path(build / name)
        subprocess.run(
            cuda_command(work / f"{name}.cu", exe, [work / "src", work]), cwd=work, check=True
        )
        if a.mode == "bench":
            if subprocess.run([str(exe)], cwd=work).returncode:
                failed.append("R bench")
        else:
            bad = [g for g in R_GROUPS if subprocess.run([str(exe), g], cwd=work).returncode]
            print(
                f"SUMMARY R: {len(R_GROUPS) - len(bad)}/{len(R_GROUPS)} groups passed"
                + (f"; failed {' '.join(bad)}" if bad else "")
            )
            failed += [f"R:{g}" for g in bad]
    elif a.part == "all":
        print("NOT RUN R (needs --device cuda: nvcc + NVIDIA GPU, WSL2/Linux)")
    if failed:
        print(f"FAILED parts: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    exit_on_missing(main)
