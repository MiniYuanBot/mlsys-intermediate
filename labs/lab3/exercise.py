"""CPU stages use actual PyTorch Inductor; CUDA C additionally runs native C5."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_tools"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from platform_tools import cuda_command, executable_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "bench"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--part", choices=["all", "A", "B", "C"], default="all")
    args = parser.parse_args()
    os.chdir(Path(__file__).parent)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
    from pipeline_test import run

    if args.part in ["A", "B", "all"]:
        run(args.part)
    if args.part in ["B", "all"]:
        sys.stdout.flush()
        subprocess.run([sys.executable, "policy_test.py"], check=True)
    if args.part in ["C", "all"]:
        from execution_test import run as run_execution

        run_execution(args.device, args.mode == "bench")
        if args.mode == "bench":
            from bench_scenarios import run as run_scenarios

            run_scenarios(args.device)
        sys.stdout.flush()
        if args.device == "cuda":
            Path("build").mkdir(exist_ok=True)
            exe = executable_path("build/fusion_test")
            subprocess.run(cuda_command("test.cu", exe, ["src"]), check=True)
            subprocess.run([str(exe)], check=True)
            if args.mode == "bench":
                exe = executable_path("build/fusion_bench")
                subprocess.run(cuda_command("bench.cu", exe, ["src"]), check=True)
                subprocess.run([str(exe)], check=True)
        else:
            print(
                "NOT RUN C5 native CUDA fusion (CPU selected); C1-C4 real CPU Inductor completed."
            )


if __name__ == "__main__":
    main()
