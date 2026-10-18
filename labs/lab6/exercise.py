"""Run lab 6 tests/benchmarks: PyTorch on CPU or CUDA; stage R also samples MPS memory.

--mode test  --part A|B|C   test.py (CPU checks everywhere; CUDA adds transfer/peak checks)
--mode test  --part R       test_r.py (float64 gradcheck + saved bytes; CUDA adds peaks)
--mode bench --part all     bench.py (A-C policies + torch.compile budget baseline), then bench_r.py
--mode bench --part R       bench_r.py (R variants x sequence lengths)
--device mps                bench --part R only (MPS has no peak API; tests use --device cpu)
"""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_tools"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from platform_tools import (
    MissingDependency,
    exit_on_missing,
    require_cuda,
    require_module,
    setup_hint,
)

LAB = 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "bench"], required=True)
    parser.add_argument("--part", choices=["all", "A", "B", "C", "R"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    if args.mode == "bench" and args.part in ("A", "B", "C"):
        parser.error("the A-C benchmark combines all three stages; use --part all (or --part R)")
    stage = "A/B/C/R" if args.part == "all" else args.part
    if args.device == "cuda":
        require_cuda(LAB, stage, "offload transfers and CUDA peak-memory measurement", nvcc=False)
    require_module(
        "torch", LAB, stage, "LoRA training, autograd hooks and custom Functions use PyTorch"
    )
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise MissingDependency(
            f"CUDA-enabled PyTorch (torch {torch.__version__})",
            LAB,
            stage,
            "--device cuda",
            setup_hint(LAB),
        )
    if args.device == "mps":
        if args.mode == "test" or args.part != "R":
            raise MissingDependency(
                "MPS test path (not provided)",
                LAB,
                stage,
                "MPS lacks float64 gradcheck and a peak-memory API",
                "run the tests with --device cpu; use --mode bench --part R --device mps to sample MPS memory",
            )
        if not torch.backends.mps.is_available():
            raise MissingDependency(
                "Apple MPS backend", LAB, "R", "--device mps", "use --device cpu"
            )
    work = Path(__file__).resolve().parent
    jobs = []
    if args.mode == "test":
        if args.part != "R":
            jobs.append(
                ("A/B/C" if args.part == "all" else args.part, ["test.py", "--part", args.part])
            )
        if args.part in ("R", "all"):
            jobs.append(("R", ["test_r.py"]))
    else:
        if args.part == "all":
            jobs.append(("A/B/C bench", ["bench.py"]))
        jobs.append(("R bench", ["bench_r.py"]))
    failed = []
    for label, script in jobs:
        code = subprocess.run(
            [sys.executable, *script, "--device", args.device], cwd=work
        ).returncode
        if code == 3:
            raise SystemExit(3)
        if code:
            failed.append(label)
    if failed:
        print(f"FAILED parts: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    exit_on_missing(main)
