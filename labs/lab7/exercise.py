"""Build/run each stage; CUDA C includes real offloaded expert execution."""

import argparse
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["test", "bench"], default="test")
p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
p.add_argument("--part", choices=["A", "B", "C", "all"], default="all")
a = p.parse_args()
work = Path(__file__).resolve().parent
build = work / "build"
build.mkdir(exist_ok=True)


def run(command):
    subprocess.run(list(map(str, command)), cwd=work, check=True)


exe = build / a.mode
run(["g++", "-std=c++17", "-O2", "-Isrc", f"{a.mode}.cpp", "-o", exe])
run([exe, a.part])
if a.part in ("C", "all") and a.device == "cuda":
    gpu = build / "cuda_exercise"
    run(["nvcc", "-std=c++17", "-O2", "-lineinfo", "cuda_exercise.cu", "-o", gpu])
    run([gpu, a.mode])
elif a.part in ("C", "all"):
    print("CPU C1/C2 checked; required CUDA C3/C4 need --device cuda (no simulated GPU score).")
