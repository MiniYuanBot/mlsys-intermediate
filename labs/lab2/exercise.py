"""Stage runner; compile only the files needed by the requested stage."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_tools"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from platform_tools import cxx_command, cuda_command, executable_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "bench"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--part", choices=["all", "A", "B", "C"], default="all")
    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    build = Path("build")
    build.mkdir(exist_ok=True)
    if args.device == "cpu" and args.part in ["all", "B"]:
        raise SystemExit(
            "Part B requires CUDA. Run --part A or --part C --device cpu for host tasks; CPU C excludes GPU integration."
        )
    suffix = ["bench"] if args.mode == "bench" else []
    if args.part in ["A", "C", "all"]:
        exe = executable_path(build / "quant_test")
        subprocess.run(cxx_command("quant_test.cpp", exe), check=True)
        subprocess.run([str(exe), args.part, *suffix], check=True)
    if args.device == "cuda" and args.part in ["B", "C", "all"]:
        exe = executable_path(build / "pipeline")
        subprocess.run(cuda_command("pipeline.cu", exe), check=True)
        subprocess.run([str(exe), args.part, *suffix], check=True)


if __name__ == "__main__":
    main()
