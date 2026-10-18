"""Lab 3 driver (run.py calls only this file).

  --part A|B   pipeline_test.py (standard library only)
  --part C     execution_test.py: C1-C4 with real torch.compile/Inductor on --device;
               --device cuda adds test.cu (C5-C7, no torch) and, with --mode bench,
               bench.cu + inductor_bench.py; --mode bench also runs bench_scenarios.py
  --part R     test_r.py / bench_r.py: CUDA-graph capture sizes (CPU: emulated runner +
               committed RTX 4060 profile; CUDA: real torch.cuda.CUDAGraph)
Each group prints PASS/FAIL per check and the driver keeps going after a failed group.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_tools"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from platform_tools import (  # noqa: E402
    MissingDependency,
    cuda_command,
    executable_path,
    exit_on_missing,
    host_platform,
    require_cuda,
    require_module,
    require_tool,
    setup_hint,
)

LAB = 3


def preflight(args):
    """All dependency checks happen before any output, so exit 3 never follows results."""
    parts = {"all": "ABCR", "A": "A", "B": "B", "C": "C", "R": "R"}[args.part]
    if "C" in parts:
        if args.device == "cuda":
            require_cuda(LAB, "C", "C5-C7 native CUDA kernels and GPU Inductor")
        require_module("torch", LAB, "C", "C1-C4 run real torch.compile (Inductor)")
        if host_platform() == "windows":
            require_tool(
                "cl",
                LAB,
                "C",
                "Inductor's CPU backend compiles generated C++ with MSVC on native Windows",
            )
    if "R" in parts:
        if args.device == "cuda":
            require_cuda(LAB, "R", "torch.cuda.CUDAGraph capture and replay", nvcc=False)
        require_module("torch", LAB, "R", "the decode block and graph runner use PyTorch")
    if args.device == "cuda" and set(parts) & set("CR"):
        import torch

        if not torch.cuda.is_available():
            raise MissingDependency(
                f"CUDA-enabled PyTorch (torch {torch.__version__})",
                LAB,
                parts,
                "--device cuda",
                setup_hint(LAB),
            )
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "bench"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--part", choices=["all", "A", "B", "C", "R"], default="all")
    args = parser.parse_args()
    parts = preflight(args)
    os.chdir(Path(__file__).parent)
    failed = []

    def step(label, cmd):
        sys.stdout.flush()
        code = subprocess.run(cmd).returncode
        if code == 3:
            raise SystemExit(3)
        if code:
            failed.append(label)

    py = sys.executable
    for stage in "AB":
        if stage in parts:
            step(stage, [py, "pipeline_test.py", stage])
    if "C" in parts:
        step(
            "C1-C4",
            [py, "execution_test.py", "--device", args.device]
            + (["--bench"] if args.mode == "bench" else []),
        )
        if args.mode == "bench":
            step("C bench_scenarios", [py, "bench_scenarios.py", "--device", args.device])
        if args.device == "cuda":
            Path("build").mkdir(exist_ok=True)
            exe = executable_path("build/fusion_test")
            subprocess.run(cuda_command("test.cu", exe, ["src", "."]), check=True)
            for group, label in [("reduce", "C5"), ("f32", "C6"), ("f16", "C7")]:
                step(label, [str(exe), "--only", group])
            if args.mode == "bench":
                bench = executable_path("build/fusion_bench")
                subprocess.run(cuda_command("bench.cu", bench, ["src", "."]), check=True)
                step("C bench.cu", [str(bench)])
                step("C inductor_bench", [py, "inductor_bench.py"])
        else:
            print(
                "NOT RUN C5-C7 native CUDA kernels (CPU selected); C1-C4 ran on CPU Inductor.",
                flush=True,
            )
    if "R" in parts:
        script = "bench_r.py" if args.mode == "bench" else "test_r.py"
        step("R", [py, script, "--device", args.device])
    if failed:
        print(f"FAILED parts: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    exit_on_missing(main)
