import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["test", "bench"], required=True)
parser.add_argument("--part", choices=["all", "A", "B", "C"], default="all")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
args = parser.parse_args()
command = [sys.executable, f"{args.mode}.py", "--device", args.device]
if args.mode == "test":
    command += ["--part", args.part]
elif args.part != "all":
    parser.error("benchmark combines A/B/C; use --part all")
subprocess.run(command, check=True)
