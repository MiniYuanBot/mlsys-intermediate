"""Stage runner; compile only the files needed by the requested stage."""

import argparse
import os
from pathlib import Path
import subprocess


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
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "quant_test.cpp", "-o", str(build / "quant_test")],
            check=True,
        )
        subprocess.run([str(build / "quant_test"), args.part, *suffix], check=True)
    if args.device == "cuda" and args.part in ["B", "C", "all"]:
        subprocess.run(
            [
                "nvcc",
                "-std=c++17",
                "-O2",
                "-arch=sm_89",
                "pipeline.cu",
                "-o",
                str(build / "pipeline"),
            ],
            check=True,
        )
        subprocess.run([str(build / "pipeline"), args.part, *suffix], check=True)


if __name__ == "__main__":
    main()
