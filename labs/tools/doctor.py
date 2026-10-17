#!/usr/bin/env python3
import importlib.util
import platform
import shutil
import subprocess
import sys

print(platform.platform(), sys.version)
for tool in ["c++", "cargo", "rustc", "nvcc", "nvidia-smi"]:
    print(f"{tool}: {shutil.which(tool) or 'MISSING'}")
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
if importlib.util.find_spec("torch"):
    import torch

    print(
        "torch",
        torch.__version__,
        "CUDA",
        torch.cuda.is_available(),
        "MPS",
        torch.backends.mps.is_available(),
    )
else:
    print("torch: MISSING (only required for lab6 and optional model experiments)")
