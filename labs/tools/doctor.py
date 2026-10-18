#!/usr/bin/env python3
"""Python-side probe used by check_env: host tools, compiler family, Inductor compile probe.

The per-lab pass/fail verdict lives in tools/check_env.sh / check_env.ps1; this script
only reports what the current interpreter sees (and --compile-probe really compiles).
"""

import argparse
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from platform_tools import (
    cargo_command,
    compiler_family,
    find_cxx,
    find_nvcc,
    host_platform,
    mingw_path_warning,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-probe", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    print(host_platform(), platform.platform(), sys.version)
    print("python:", sys.executable)
    try:
        cxx = find_cxx()
        family = compiler_family(cxx)
        print(f"C++: {cxx} ({family})")
        labs = Path(__file__).resolve().parents[1]
        if warning := mingw_path_warning(family, labs, tempfile.gettempdir()):
            print("WARNING:", warning)
    except RuntimeError as error:
        print("C++:", error)
    found = {"cargo": cargo_command(), "nvcc": find_nvcc()}
    for tool in ["cargo", "rustc", "nvcc", "nvidia-smi", "node", "npm"]:
        print(f"{tool}: {found.get(tool) or shutil.which(tool) or 'MISSING'}")
    if os.environ.get("VIRTUAL_ENV"):
        print("VIRTUAL_ENV:", os.environ["VIRTUAL_ENV"])
    if shutil.which("nvidia-smi"):
        sys.stdout.flush()
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
