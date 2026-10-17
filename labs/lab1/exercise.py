"""Compile the native exercise on Windows, Linux, macOS or WSL."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_tools"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from platform_tools import cxx_command, executable_path

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["test", "bench"], required=True)
parser.add_argument("--part", choices=["all", "A", "B", "C"], default="all")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
args = parser.parse_args()
if args.mode == "bench" and args.part != "all":
    parser.error("benchmark combines A/B/C; use --part all")
work = Path(__file__).resolve().parent
build = work / "build"
build.mkdir(exist_ok=True)
exe = executable_path(build / "exercise")
subprocess.run(cxx_command(work / f"{args.mode}.cpp", exe, [work / "src"]), cwd=work, check=True)
subprocess.run([str(exe), args.part], cwd=work, check=True)
