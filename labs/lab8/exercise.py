"""Run Rust contracts and a real small policy-learning feedback loop without crates."""

import argparse
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["test", "bench"], default="test")
p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
p.add_argument("--part", choices=["A", "B", "C", "all"], default="all")
a = p.parse_args()
work = Path(__file__).resolve().parent
if a.device == "cuda":
    print(
        "Lab8 runs a CPU event runtime and actual contextual-bandit learning; no GPU speed claim."
    )
if a.mode == "test":
    command = ["cargo", "test", "--offline", "--test", "contracts"]
    if a.part != "all":
        command += [a.part.lower() + "_"]
    command += ["--", "--nocapture"]
else:
    command = ["cargo", "run", "--offline", "--release", "--bin", "bench", "--", a.part]
subprocess.run(command, cwd=work, check=True)
