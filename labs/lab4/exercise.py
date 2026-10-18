"""Lab 4 driver: Rust simulator tests/benchmarks (A/B/C/R) and the Python+torch real
stage (R). run.py creates the reference overlay; this script only runs what it is told.

  --part A|B|C   cargo tests named a_*/b_*/c_* (CPU only)
  --part R       Rust replay tests r_* on the committed reference trace, then the Python
                 R1/R2 checks (needs torch); --device cuda adds real-model checks
                 (needs an NVIDIA GPU, transformers and both Qwen3 models)
  --part all     A/B/C/R Rust tests, plus the Python R checks when torch is present
bench: A/B/C print modeled tables; R replays the reference trace, and with
--device cuda first collects a fresh trace into results/lab4-real/.
"""

import argparse
import datetime
import importlib.util
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

LAB = 4
MODELS = ("Qwen3-0.6B", "Qwen3-1.7B")


def models_dir() -> Path:
    return Path(os.environ.get("MLSYS_MODELS") or Path.home() / "models").expanduser()


def require_models(stage: str) -> Path:
    root = models_dir()
    for name in MODELS:
        d = root / name
        if not (d / "config.json").is_file() or not any(d.glob("*.safetensors")):
            raise MissingDependency(
                f"model {name} (looked in {d}; set MLSYS_MODELS to override)",
                LAB,
                stage,
                "real acceptance and latency traces",
                setup_hint(LAB),
            )
    return root


def require_gpu_stack(stage: str) -> Path:
    require_cuda(LAB, stage, "Qwen3 draft/target on the GPU", nvcc=False)
    require_module("torch", LAB, stage, "stage R runs Qwen3 in PyTorch")
    require_module("transformers", LAB, stage, "stage R loads Qwen3 with transformers")
    import torch

    if not torch.cuda.is_available():
        raise MissingDependency(
            f"CUDA-enabled PyTorch (torch {torch.__version__})",
            LAB,
            stage,
            "--device cuda",
            setup_hint(LAB),
        )
    return require_models(stage)


def results_dir() -> Path:
    base = Path(os.environ.get("MLSYS_LABS_DIR") or HERE.parent) / "results" / "lab4-real"
    base.mkdir(parents=True, exist_ok=True)
    return base


def run(command, **kw):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=HERE, check=True, **kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "bench"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--part", choices=["all", "A", "B", "C", "R"], default="all")
    args = parser.parse_args()
    stage = "A/B/C/R" if args.part == "all" else args.part
    uses_python_r = args.part == "R" or (args.part == "all" and importlib.util.find_spec("torch"))
    models = None
    if args.device == "cuda" and args.part in ("R", "all"):
        models = require_gpu_stack("R")
    if args.part == "R":
        require_module(
            "torch", LAB, "R", "stage R checks use PyTorch (CPU is enough without --device cuda)"
        )
    elif args.device == "cuda":
        print("Parts A/B/C are a CPU discrete-event simulator; --device cuda adds nothing there.")
    cargo = require_tool("cargo", LAB, stage, "build and test the Rust simulator")

    if args.mode == "test":
        command = [cargo, "test", "--offline", "--release", "--test", "exercises"]
        if args.part != "all":
            command += [args.part.lower() + "_"]
        failed = subprocess.run(
            command + ["--", "--nocapture", "--test-threads=1"], cwd=HERE
        ).returncode
        if uses_python_r:
            py = [sys.executable, "real/test_real.py", "--device", args.device]
            if models:
                py += ["--models", models]
            print("+", " ".join(map(str, py)), flush=True)
            failed |= subprocess.run(list(map(str, py)), cwd=HERE).returncode
        elif args.part == "all":
            print("NOT RUN R python checks (needs torch); Rust r_* replay tests ran above")
        if failed:
            raise SystemExit(1)
        return
    bench = [cargo, "run", "--offline", "--release", "--bin", "bench", "--", args.part]
    if args.part in ("R", "all") and models:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        trace = results_dir() / f"trace-{stamp}.jsonl"
        run(
            [
                sys.executable,
                "real/collect.py",
                "--models",
                models,
                "--out",
                trace,
                "--hf-check",
                "2",
            ]
        )
        bench += [HERE / "real/data/qwen3-0.6b-1.7b-rtx4060-greedy.jsonl", trace]
    run(bench)


if __name__ == "__main__":
    exit_on_missing(main)
