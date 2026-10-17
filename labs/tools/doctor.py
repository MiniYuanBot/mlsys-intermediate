#!/usr/bin/env python3
"""Show host tools and optionally probe the actual PyTorch Inductor backend."""

import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys

from platform_tools import find_cxx, host_platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-probe", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    print(host_platform(), platform.platform(), sys.version)
    try:
        print("C++:", find_cxx())
    except RuntimeError as error:
        print("C++:", error)
    for tool in ["cargo", "rustc", "nvcc", "nvidia-smi", "node", "npm"]:
        print(f"{tool}: {shutil.which(tool) or 'MISSING'}")
    if shutil.which("nvidia-smi"):
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]
        )
    if not importlib.util.find_spec("torch"):
        print("torch: MISSING (needed for lab3 C, lab6 and model capture)")
        return 1 if args.compile_probe else 0
    import torch

    mps = getattr(torch.backends, "mps", None)
    print(
        "torch",
        torch.__version__,
        "CUDA",
        torch.cuda.is_available(),
        "MPS",
        bool(mps and mps.is_available()),
    )
    if args.compile_probe:
        try:
            x = torch.arange(8, dtype=torch.float32, device=args.device)
            compiled = torch.compile(lambda x: (x + 1).relu(), backend="inductor", fullgraph=True)
            torch.testing.assert_close(compiled(x), (x + 1).relu())
        except Exception as error:
            print(f"Inductor probe FAILED: {error}")
            print(
                "Check the installed PyTorch/compiler dependencies. Windows users can also use WSL2."
            )
            return 1
        print(f"Inductor probe PASSED on {args.device}; lab3 still requires its full tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
