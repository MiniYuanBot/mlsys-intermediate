"""Offline Rust exercise entry; the parent runner creates the solution overlay."""

import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["test", "bench"], required=True)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--part", choices=["all", "A", "B", "C"], default="all")
args = parser.parse_args()
if args.device == "cuda":
    print(
        "This lab uses a CPU discrete-event simulator; device=cuda does not create GPU measurements."
    )
if args.mode == "test":
    command = ["cargo", "test", "--offline", "--release", "--test", "exercises"]
    if args.part != "all":
        command += [args.part.lower() + "_"]
    command += ["--", "--nocapture"]
else:
    command = ["cargo", "run", "--offline", "--release", "--bin", "bench", "--", args.part]
raise SystemExit(subprocess.call(command, cwd=Path(__file__).resolve().parent))
