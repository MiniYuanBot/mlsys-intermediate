"""Build/run each stage; CUDA C includes real offloaded expert execution."""

import argparse
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_tools"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from platform_tools import cxx_command, cuda_command, executable_path

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


exe = executable_path(build / a.mode)
run(cxx_command(work / f"{a.mode}.cpp", exe, [work / "src"]))
run([exe, a.part])
if a.part in ("C", "all") and a.device == "cuda":
    gpu = executable_path(build / "cuda_exercise")
    run(cuda_command(work / "cuda_exercise.cu", gpu))
    run([gpu, a.mode])
elif a.part in ("C", "all"):
    print("CPU C1/C2 checked; required CUDA C3/C4 need --device cuda (no simulated GPU score).")
