"""Lab 8 driver: Rust contract tests and benchmarks for A/B/C/R (CPU only, no crates).

  --part A|B|C|R   cargo tests named a_*/b_*/c_*/r_*
  --part all       every test
bench: A/B/C print synthetic / contextual-bandit tables; R replays the committed real
trace (lab8/real/data, Qwen3-0.6B sampling on an RTX 4060 Laptop) plus every
results/lab8-real/*.jsonl. With --mode bench --part R --device cuda it first samples a
fresh small trace with lab8/real/sample_trace.py (needs an NVIDIA GPU, torch,
transformers and ~/models/Qwen3-0.6B; about 2 minutes) and replays it as well.
The trace generator is provided code, not a TODO; all R TODOs are in Rust.
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_tools"))
sys.path.append(str(HERE.parents[0] / "tools"))
from platform_tools import (  # noqa: E402
    MissingDependency,
    exit_on_missing,
    require_cuda,
    require_module,
    require_tool,
    setup_hint,
)

LAB = 8
MODEL = "Qwen3-0.6B"


def results_dir() -> Path:
    base = Path(os.environ.get("MLSYS_LABS_DIR") or HERE.parent) / "results" / "lab8-real"
    base.mkdir(parents=True, exist_ok=True)
    return base


def collect_trace() -> Path:
    require_cuda(LAB, "R", "sample a real rollout-length trace with Qwen3-0.6B", nvcc=False)
    require_module("torch", LAB, "R", "trace sampling runs Qwen3-0.6B in PyTorch")
    require_module("transformers", LAB, "R", "trace sampling loads Qwen3-0.6B with transformers")
    models = Path(os.environ.get("MLSYS_MODELS") or Path.home() / "models").expanduser()
    if not (models / MODEL / "config.json").is_file():
        raise MissingDependency(
            f"model {MODEL} (looked in {models / MODEL}; set MLSYS_MODELS to override)",
            LAB,
            "R",
            "trace sampling",
            setup_hint(LAB),
        )
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = results_dir() / f"trace-{stamp}.jsonl"
    command = [
        sys.executable,
        "real/sample_trace.py",
        "--out",
        str(out),
        "--limit",
        "8",
        "--rounds",
        "2",
        "--models",
        str(models),
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["test", "bench"], default="test")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--part", choices=["A", "B", "C", "R", "all"], default="all")
    a = p.parse_args()
    stage = "A/B/C/R" if a.part == "all" else a.part
    fresh = None
    if a.mode == "bench" and a.device == "cuda" and a.part in ("R", "all"):
        fresh = collect_trace()
    elif a.device == "cuda":
        print(
            "Lab8 tests and A/B/C benchmarks run on the CPU; --device cuda only adds "
            "fresh trace sampling to --mode bench --part R.",
            flush=True,
        )
    cargo = require_tool("cargo", LAB, stage, "build and test the Rust runtime")
    if a.mode == "test":
        command = [cargo, "test", "--offline", "--release", "--test", "contracts"]
        if a.part != "all":
            command += [a.part.lower() + "_"]
        command += ["--", "--nocapture", "--test-threads=1"]
    else:
        command = [cargo, "run", "--offline", "--release", "--bin", "bench", "--", a.part]
        if a.part in ("R", "all"):
            extra = sorted(results_dir().glob("*.jsonl"))
            if fresh and fresh not in extra:
                extra.append(fresh)
            command += [str(x) for x in extra]
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


if __name__ == "__main__":
    exit_on_missing(main)
