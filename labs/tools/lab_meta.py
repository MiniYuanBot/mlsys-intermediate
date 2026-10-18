#!/usr/bin/env python3
"""Single registry of labs, stages, per-stage dependencies and platform support.

run.py, materialize.py, workload_audit.py, test_platform.py and the docs read this file.
setup_env.sh/.ps1 and check_env.sh/.ps1 carry a copy of the per-lab install table
(between "BEGIN LAB TABLE" and "END LAB TABLE"); test_platform.py checks that the copies
match table_line() here. Standard library only.

Components (what setup_env installs / check_env checks per lab, non-cumulative):
  gpu    NVIDIA driver visible (checked only; never installed by the scripts)
  cuda   CUDA toolkit (nvcc) at CUDA in env-versions.env
  torch  PyTorch at TORCH
  hf     transformers + safetensors + huggingface_hub
  cargo  Rust at RUST
A trailing ":R" marks a component or model that only the lab's R stage needs; check_env
reports those as [WARN] instead of [FAIL] so A/B/C work is never blocked by them.

Need tokens:
  cxx           C++17 compiler (always part of the base install)
  nvcc          CUDA toolkit compiler (installed as component "cuda")
  gpu           an NVIDIA GPU with a working driver (not installable by the scripts)
  torch         PyTorch (CUDA wheel on WSL/Linux, CPU wheel on Windows/macOS)
  transformers  Hugging Face transformers + safetensors + huggingface_hub ("hf")
  cargo         Rust toolchain pinned to RUST in formatter-versions.env
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

LABS_DIR = Path(__file__).resolve().parents[1]
ROOT = LABS_DIR.parent
PLATFORMS = ("wsl", "linux", "windows", "macos")
CPU_ONLY_PLATFORMS = ("windows", "macos")
NEED_TO_COMPONENT = {
    "gpu": "gpu",
    "nvcc": "cuda",
    "torch": "torch",
    "transformers": "hf",
    "cargo": "cargo",
}
COMPONENT_ORDER = ("gpu", "cuda", "torch", "hf", "cargo")


@dataclass(frozen=True)
class Stage:
    cpu: tuple[str, ...] = ()  # needed to run the stage at all (CPU path)
    cuda: tuple[str, ...] = ()  # extra needs of the --device cuda / GPU part
    cpu_ok: bool = True  # False: the stage has no CPU path at all
    models: tuple[str, ...] = ()  # required local models (~/models/<name>)
    optional_models: tuple[str, ...] = ()
    guide_only: bool = False  # documentation only; nothing installed by setup_env
    note: str = ""

    def needs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.cpu + self.cuda))


@dataclass(frozen=True)
class Lab:
    number: int
    title: str
    stages: dict[str, Stage]
    r_implemented: bool = False  # flipped by the lab owner when the R stage lands in code
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return LABS_DIR / f"lab{self.number}"

    @property
    def solution_directory(self) -> Path:
        return ROOT / "solutions" / f"lab{self.number}"

    def exists(self) -> bool:
        return (self.directory / "exercise.py").is_file()

    def required_stages(self) -> tuple[str, ...]:
        """Stages whose TODO regions workload_audit must find."""
        names = [s for s in "ABC" if s in self.stages]
        if "R" in self.stages and self.r_implemented and not self.stages["R"].guide_only:
            names.append("R")
        return tuple(names)


CXX = ("cxx",)
CUDA = ("nvcc", "gpu")
LABS: dict[int, Lab] = {
    1: Lab(
        1,
        "KV cache: INT4, streaming cache, PQ retrieval",
        {
            "A": Stage(CXX),
            "B": Stage(CXX),
            "C": Stage(CXX),
            "R": Stage(
                cuda=CXX + CUDA,
                cpu_ok=False,
                optional_models=("Qwen3-0.6B",),
                note="int4 KV fused decode-attention CUDA kernel (split-K, GQA, Qwen3-1.7B shapes)",
            ),
        },
        r_implemented=True,
    ),
    2: Lab(
        2,
        "Low-bit operators",
        {
            "A": Stage(CXX),
            "B": Stage(cuda=CXX + CUDA, cpu_ok=False),
            "C": Stage(CXX, cuda=CUDA, note="host algorithms on CPU; kernel integration on GPU"),
            "R": Stage(
                cuda=CXX + CUDA,
                cpu_ok=False,
                note="FP8 E4M3 weight-only GEMV/GEMM on sm_89 vs INT4, cuBLAS FP16/FP32 and cuBLASLt FP8",
            ),
        },
        r_implemented=True,
    ),
    3: Lab(
        3,
        "Dynamic shapes and compilation",
        {
            "A": Stage(),
            "B": Stage(),
            "C": Stage(
                ("torch",),
                cuda=CUDA,
                note="C1-C4 real Inductor (CPU or GPU; native Windows CPU Inductor needs MSVC cl); "
                "C5-C7 warp-shuffle + float4/half2 fused add-RMSNorm CUDA kernels vs FP64 oracle "
                "(no torch) and vs the Inductor-generated kernel (bench)",
            ),
            "R": Stage(
                ("torch",),
                cuda=("gpu",),
                note="CUDA-graph capture sizes: GPU captures torch.cuda.CUDAGraph for a "
                "Qwen3-0.6B-shaped decode block at every batch size, plans sizes with the B1 DP "
                "under graph/memory budgets, replays held-out traces; CPU: emulated runner + "
                "planning from the committed RTX 4060 profile",
            ),
        },
        r_implemented=True,
    ),
    4: Lab(
        4,
        "Speculative decoding under SLOs",
        {
            "A": Stage(("cargo",)),
            "B": Stage(("cargo",)),
            "C": Stage(("cargo",)),
            "R": Stage(
                ("cargo", "torch"),
                cuda=("gpu", "transformers"),
                models=("Qwen3-0.6B", "Qwen3-1.7B"),
                note="CPU: replay the committed Qwen3 trace in Rust + R1/R2 unit checks; "
                "GPU: collect real acceptance and cudaEvent verify cost with Qwen3-0.6B -> 1.7B",
            ),
        },
        r_implemented=True,
    ),
    5: Lab(
        5,
        "Prefix reuse: cache identity, ownership and cache-aware routing",
        {
            "A": Stage(("cargo",)),
            "B": Stage(("cargo",)),
            "C": Stage(("cargo",)),
            "R": Stage(
                ("cargo",),
                cuda=("gpu", "torch", "transformers"),
                models=("Qwen3-0.6B",),
                note="CPU: R1 cost-model fit (Python stdlib) on the committed RTX 4060 measurement, "
                "R2/R3 Rust replay of a Mooncake-format trace; GPU: measure Qwen3-0.6B prefill "
                "T(p, n) with DynamicCache.crop and refit",
            ),
        },
        r_implemented=True,
    ),
    6: Lab(
        6,
        "Low-memory training",
        {
            "A": Stage(("torch",), cuda=("gpu",)),
            "B": Stage(("torch",), cuda=("gpu",)),
            "C": Stage(("torch",), cuda=("gpu",)),
            "R": Stage(
                ("torch",),
                cuda=("gpu",),
                note="custom autograd.Functions (LoRA-FA, fused LoRA MLP with GELU recompute, "
                "seed-only LoRA dropout): float64 gradcheck + saved bytes on CPU; "
                "CUDA adds allocator peaks and timing",
            ),
        },
        r_implemented=True,
    ),
    7: Lab(
        7,
        "MoE offloading",
        {
            "A": Stage(CXX),
            "B": Stage(CXX),
            "C": Stage(CXX, cuda=CUDA, note="C3/C4/C6 need CUDA"),
            "R": Stage(
                cuda=CXX + CUDA,
                cpu_ok=False,
                note="R1-R2 pinned pool and the event-ordered copy/compute pipeline over 4-32 MiB "
                "experts; R3-R4 overlap ratio and measured pinned vs pageable H2D; "
                "R5 CPU/GPU hybrid policy on a routing trace",
            ),
        },
        r_implemented=True,
    ),
    8: Lab(
        8,
        "Rollout scheduling",
        {
            "A": Stage(("cargo",)),
            "B": Stage(("cargo",)),
            "C": Stage(("cargo",)),
            "R": Stage(
                ("cargo",),
                optional_models=("Qwen3-0.6B",),
                note="Rust replay of the committed Qwen3-0.6B rollout-length trace (R1-R3); "
                "optional: bench --device cuda samples a fresh trace (GPU + torch + transformers); "
                "TRL GRPO bridge in real/GUIDE.md uses a separate ~/venvs/mlsys-rl, not installed",
            ),
        },
        r_implemented=True,
    ),
    9: Lab(
        9,
        "Heterogeneous multi-model serving: cost model and planning",
        {
            "A": Stage(CXX),
            "B": Stage(CXX),
            "C": Stage(CXX),
            "R": Stage(
                CXX,
                cuda=CUDA,
                note="R1-R4 cudaEvent bandwidth/GEMV/GEMM profiling on GPU; "
                "R5/R6 refit and re-plan on CPU from a profile CSV",
            ),
        },
        r_implemented=True,
    ),
}
LAB_NUMBERS = tuple(sorted(LABS))


def get(lab: int) -> Lab:
    if lab not in LABS:
        raise KeyError(f"lab{lab} is not registered in tools/lab_meta.py")
    return LABS[lab]


def missing_lab_message(lab: int) -> str:
    return (
        f"lab{lab} is registered in tools/lab_meta.py but labs/lab{lab}/exercise.py "
        "does not exist yet in this checkout."
    )


def install_components(lab: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(components, models) for one lab; ":R" suffix = needed only by stage R."""
    stages = {k: v for k, v in get(lab).stages.items() if not v.guide_only}
    core, r_only, models = set(), set(), []
    for name, stage in stages.items():
        found = {NEED_TO_COMPONENT[n] for n in stage.needs() if n in NEED_TO_COMPONENT}
        (r_only if name == "R" else core).update(found)
    core_models = [m for n, st in stages.items() if n != "R" for m in st.models]
    for name, stage in stages.items():
        for m in stage.models:
            token = m if m in core_models else f"{m}:R"
            if token not in models:
                models.append(token)
    components = [c if c in core else f"{c}:R" for c in COMPONENT_ORDER if c in core or c in r_only]
    return tuple(components), tuple(models)


def table_line(lab: int) -> str:
    """The LAB TABLE entry the setup/check scripts must carry for this lab."""
    components, models = install_components(lab)
    return f'comps="{" ".join(components)}" models="{" ".join(models)}"'


def feasibility(lab: int, stage: str, platform: str) -> str:
    """full | cpu-only | wsl | none | guide (see LABEL for the wording)."""
    info = get(lab).stages[stage]
    if info.guide_only:
        return "guide" if platform in ("wsl", "linux") else "none"
    if platform not in CPU_ONLY_PLATFORMS:
        return "full"
    gpu = any(n in ("nvcc", "gpu") for n in info.cuda)
    if not info.cpu_ok:
        return "wsl" if platform == "windows" else "none"
    return "cpu-only" if gpu else "full"


LABEL = {
    "full": "可做",
    "cpu-only": "仅 CPU 部分",
    "wsl": "不支持，转 WSL",
    "none": "不支持",
    "guide": "仅指南",
}
NEED_LABEL = {
    "cxx": "C++17",
    "nvcc": "nvcc",
    "gpu": "NVIDIA GPU",
    "torch": "PyTorch",
    "transformers": "transformers",
    "cargo": "Rust",
}


def describe_stage(lab: int, name: str) -> str:
    stage = get(lab).stages[name]
    if stage.guide_only:
        text = "独立的 `~/venvs/mlsys-rl`（默认不安装）"
    else:
        cpu = " + ".join(NEED_LABEL[n] for n in stage.cpu) or "Python 标准库"
        cuda = " + ".join(NEED_LABEL[n] for n in stage.cuda if n not in stage.cpu)
        if not stage.cpu_ok:
            text = cuda
        elif cuda:
            text = f"{cpu}；GPU 部分另需 {cuda}"
        else:
            text = cpu
    if stage.models:
        text += "；模型 " + "、".join(stage.models)
    if stage.optional_models:
        text += "；可选模型 " + "、".join(stage.optional_models)
    if name == "R" and not get(lab).r_implemented:
        text = "尚未实现；" + text
    return text


def markdown_table(lab: int) -> str:
    rows = [
        "| 阶段 | 需要 | WSL2 / Linux | Windows 原生 | macOS |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in get(lab).stages:
        cells = [LABEL[feasibility(lab, name, p)] for p in ("wsl", "windows", "macos")]
        rows.append(f"| {name} | {describe_stage(lab, name)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


PIN_LINE = re.compile(r"^[A-Z_][A-Z0-9_]*=")


def pin_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").replace("\r", "").splitlines()
    return [line for line in lines if PIN_LINE.match(line)]


def pins_hash(path: Path | None = None) -> str:
    """sha256 of the KEY=VALUE lines of env-versions.env, each followed by '\\n'."""
    import hashlib

    path = path or LABS_DIR / "tools" / "env-versions.env"
    data = "".join(line + "\n" for line in pin_lines(path)).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def as_dict() -> dict:
    return {
        str(n): {
            "title": lab.title,
            "exists": lab.exists(),
            "r_implemented": lab.r_implemented,
            "components": install_components(n)[0],
            "models": install_components(n)[1],
            "stages": {
                s: {
                    "cpu": st.cpu,
                    "cuda": st.cuda,
                    "cpu_ok": st.cpu_ok,
                    "models": st.models,
                    "optional_models": st.optional_models,
                    "guide_only": st.guide_only,
                    "note": st.note,
                    "platforms": {p: feasibility(n, s, p) for p in PLATFORMS},
                }
                for s, st in lab.stages.items()
            },
        }
        for n, lab in LABS.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", action="store_true", help="dump the registry")
    group.add_argument("--markdown", type=int, metavar="N", help="stage x platform table")
    group.add_argument("--components", type=int, metavar="N", help="what setup_env installs")
    group.add_argument("--pins-hash", action="store_true", help="hash of env-versions.env")
    group.add_argument("--table", action="store_true", help="LAB TABLE lines for the scripts")
    args = parser.parse_args()
    if args.json:
        print(json.dumps(as_dict(), indent=2, ensure_ascii=False))
    elif args.markdown is not None:
        print(markdown_table(args.markdown))
    elif args.components is not None:
        components, models = install_components(args.components)
        print(" ".join(components), "|", " ".join(models))
    elif args.table:
        for n in LAB_NUMBERS:
            print(f"        {n}) {table_line(n)} ;;")
    else:
        print(pins_hash())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
