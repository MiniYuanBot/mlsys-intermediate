"""Lab 5 driver: Rust tests/benchmarks (A/B/C/R) and the Python R1 fit checks.
run.py creates the reference overlay; this script only runs what it is told.

  --part A|B|C   cargo tests named a_*/b_*/c_* (CPU only)
  --part R       Rust r_* tests (Mooncake adapter, replay metric, committed trace) and the
                 Python R1 fit checks (standard library only); --device cuda adds a short
                 real Qwen3-0.6B prefill measurement (GPU + torch + transformers + model)
  --part all     every Rust test plus the Python R1 checks; with --device cuda also the
                 real measurement
bench: A/B/C print structural/modeled tables. R replays real/data/synthetic-mooncake.jsonl
(or $MOONCAKE_TRACE) under the committed 4060 fit; with --device cuda it first measures
T(p, n) on this GPU, fits it with your R1 and replays with that fit.
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

LAB = 5
MODEL = "Qwen3-0.6B"


def models_dir() -> Path:
    return Path(os.environ.get("MLSYS_MODELS") or Path.home() / "models").expanduser()


def require_gpu_stack(stage: str) -> Path:
    require_cuda(LAB, stage, "Qwen3-0.6B prefill timing on the GPU", nvcc=False)
    require_module("torch", LAB, stage, "stage R measures prefill with PyTorch")
    require_module("transformers", LAB, stage, "stage R loads Qwen3-0.6B with transformers")
    import torch

    if not torch.cuda.is_available():
        raise MissingDependency(
            f"CUDA-enabled PyTorch (torch {torch.__version__})",
            LAB,
            stage,
            "--device cuda",
            setup_hint(LAB),
        )
    d = models_dir() / MODEL
    if not (d / "config.json").is_file() or not any(d.glob("*.safetensors")):
        raise MissingDependency(
            f"model {MODEL} (looked in {d}; set MLSYS_MODELS to override)",
            LAB,
            stage,
            "real prefill measurement",
            setup_hint(LAB),
        )
    return models_dir()


def results_dir() -> Path:
    base = Path(os.environ.get("MLSYS_LABS_DIR") or HERE.parent) / "results" / "lab5-real"
    base.mkdir(parents=True, exist_ok=True)
    return base


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=HERE, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "bench"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--part", choices=["all", "A", "B", "C", "R"], default="all")
    args = parser.parse_args()
    stage = "A/B/C/R" if args.part == "all" else args.part
    with_r = args.part in ("R", "all")
    models = None
    if args.device == "cuda" and with_r:
        models = require_gpu_stack("R")
    elif args.device == "cuda":
        print(
            "Parts A/B/C are CPU data structures and a modeled replay; --device cuda adds nothing there."
        )
    cargo = require_tool("cargo", LAB, stage, "build and test the Rust prefix cache")

    if args.mode == "test":
        command = [cargo, "test", "--offline", "--release", "--test", "exercises"]
        if args.part != "all":
            command += [args.part.lower() + "_"]
        failed = subprocess.run(
            command + ["--", "--nocapture", "--test-threads=1"], cwd=HERE
        ).returncode
        if with_r:
            py = [sys.executable, "real/test_real.py", "--device", args.device]
            if models:
                py += ["--models", models]
            print("+", " ".join(map(str, py)), flush=True)
            failed |= subprocess.run(list(map(str, py)), cwd=HERE).returncode
        if failed:
            raise SystemExit(1)
        return
    bench = [cargo, "run", "--offline", "--release", "--bin", "bench", "--", args.part]
    if with_r:
        trace = Path(
            os.environ.get("MOONCAKE_TRACE") or HERE / "real/data/synthetic-mooncake.jsonl"
        )
        cost = HERE / "real/data/cost-qwen3-0.6b-rtx4060.json"
        if models:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            meas = results_dir() / f"prefill-{stamp}.json"
            cost = results_dir() / f"cost-{stamp}.json"
            run([sys.executable, "real/measure.py", "--model", models / MODEL, "--out", meas])
            run([sys.executable, "real/fit.py", "--in", meas, "--out", cost])
        bench += [trace, cost]
    run(bench)


if __name__ == "__main__":
    exit_on_missing(main)
