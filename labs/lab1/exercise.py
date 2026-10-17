"""Compile the native exercise in the isolated directory created by tools/run.py."""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["test", "bench"], required=True)
parser.add_argument("--part", choices=["all", "A", "B", "C"], default="all")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
subprocess.run(
    [
        os.environ.get("CXX", "c++"),
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-I",
        "src",
        f"{args.mode}.cpp",
        "-o",
        "exercise",
    ],
    check=True,
)
if args.mode == "bench" and args.part != "all":
    parser.error("benchmark combines A/B/C; use --part all")
subprocess.run(["./exercise", args.part], check=True)
