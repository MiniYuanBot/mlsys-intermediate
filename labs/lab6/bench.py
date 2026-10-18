"""A-C benchmark: 3 seeds x 2 widths x 4 policies of real AdamW steps, then a strong
baseline: torch.compile with torch._functorch.config.activation_memory_budget."""

import argparse
import copy
import json
import math
import time

import torch
from model import Block
from src.train_student import (
    calibrate_blocks,
    measure_policy,
    parameter_bytes,
    plan_modes,
    run_policy,
)
from torch import nn

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
args = p.parse_args()
DEV = torch.device(args.device)
CUDA = DEV.type == "cuda"
torch.set_num_threads(2)
WIDTHS = (256, 1024) if CUDA else (64, 128)
TOKENS = 512 if CUDA else 32


def plan_for(base, x):
    calibration = calibrate_blocks(base, x[:2], repeats=2)
    # Planner input: measured retain/checkpoint only. Offload cost is not invented from
    # a bandwidth formula; the offload policy is measured as its own row.
    options = [
        {
            mode: (math.ceil(values["saved_bytes"] / 1024), values["ms"])
            for mode, values in row.items()
        }
        for row in calibration
    ]
    floor = sum(row["checkpoint"][0] for row in options)
    ceiling = sum(row["retain"][0] for row in options)
    budget = (floor + ceiling) // 2
    selected, used, estimated_ms = plan_modes(options, budget)
    return calibration, budget, selected, used, estimated_ms


def fwbw(fn, params, x, repeats=7):
    """Bench-side fw+bw metrics shared by planner plans and compiled baselines."""
    skip = {q.untyped_storage().data_ptr() for q in params}
    storages = {}

    def pack(t):
        if t.untyped_storage().data_ptr() not in skip:
            storages[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t.detach()

    def step():
        for q in params:
            q.grad = None
        fn(x).square().mean().backward()

    for _ in range(3):
        step()
    if CUDA:
        began = time.perf_counter()
        while time.perf_counter() - began < 0.3:
            step()
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEV)
        base = torch.cuda.memory_allocated(DEV)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(x).square().mean()
    out.backward()
    peak = torch.cuda.max_memory_allocated(DEV) - base if CUDA else None
    times = []
    for _ in range(repeats):
        if CUDA:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        else:
            began = time.perf_counter()
            step()
            times.append((time.perf_counter() - began) * 1000)
    q25, q50, q75 = (
        torch.tensor(times, dtype=torch.float64)
        .quantile(torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
        .tolist()
    )
    return {
        "saved_bytes": sum(storages.values()),
        "peak_delta_bytes": peak,
        "ms_median": q50,
        "ms_p25": q25,
        "ms_p75": q75,
        "stable": q75 <= 1.5 * q25,
    }


for seed in (17, 29, 43):
    for width in WIDTHS:
        torch.manual_seed(seed)
        base = nn.ModuleList([Block(width, dropout=0) for _ in range(4)]).to(DEV)
        x = torch.randn(3, TOKENS, width, device=DEV)
        target = torch.randn_like(x)
        calibration, budget, selected, used, estimated_ms = plan_for(base, x)
        for name, modes in [
            ("retain", ["retain"] * 4),
            ("checkpoint", ["checkpoint"] * 4),
            ("offload", ["offload"] * 4),
            ("calibrated", selected),
        ]:
            blocks = copy.deepcopy(base)
            batches = [(x[:2], target[:2]), (x[2:], target[2:])]
            result = measure_policy(blocks, batches, modes, repeats=5)
            print(
                json.dumps(
                    {
                        "kind": "actual_torch_training",
                        "seed": seed,
                        "width": width,
                        "tokens_per_sample": TOKENS,
                        "device": args.device,
                        "policy": name,
                        "modes": modes,
                        "parameter_bytes": parameter_bytes(blocks),
                        "budget_kib": budget,
                        "predicted_used_kib": used if name == "calibrated" else None,
                        "predicted_block_ms": estimated_ms if name == "calibrated" else None,
                        "calibration": calibration,
                        **result,
                    }
                ),
                flush=True,
            )
            del blocks

# Strong baseline (B4): PyTorch's partitioner solves a 0/1 knapsack over the joint graph
# under activation_memory_budget (0 = recompute everything allowed, 1 = default).
torch.manual_seed(17)
for width in WIDTHS:
    base = nn.ModuleList([Block(width, dropout=0) for _ in range(4)]).to(DEV)
    x = torch.randn(1, TOKENS, width, device=DEV)
    params = list(base.parameters())
    _, budget, selected, used, _ = plan_for(base, torch.cat([x, x, x]))
    for name, modes in (
        ("retain", ["retain"] * 4),
        ("checkpoint", ["checkpoint"] * 4),
        ("calibrated", selected),
    ):
        row = fwbw(lambda v, m=modes: run_policy(base, v, m), params, x)
        print(
            json.dumps(
                {
                    "kind": "budget_baseline",
                    "method": "lab_planner",
                    "policy": name,
                    "modes": modes,
                    "width": width,
                    "device": args.device,
                    **row,
                }
            ),
            flush=True,
        )

    def forward(v):
        for block in base:
            v = block(v)
        return v

    import torch._functorch.config as functorch_config

    for budget_fraction in (1.0, 0.5, 0.0):
        try:
            torch._dynamo.reset()
            functorch_config.activation_memory_budget = budget_fraction
            compiled = torch.compile(forward)
            began = time.perf_counter()
            compiled(x).square().mean().backward()
            compile_s = time.perf_counter() - began
            row = fwbw(compiled, params, x)
            print(
                json.dumps(
                    {
                        "kind": "budget_baseline",
                        "method": "torch.compile",
                        "activation_memory_budget": budget_fraction,
                        "compile_s": compile_s,
                        "width": width,
                        "device": args.device,
                        **row,
                    }
                ),
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - e.g. no C++ compiler for Inductor on this host
            print(
                json.dumps(
                    {
                        "kind": "budget_baseline",
                        "method": "torch.compile",
                        "activation_memory_budget": budget_fraction,
                        "width": width,
                        "skipped": " ".join(str(error).split())[:300],
                    }
                ),
                flush=True,
            )
        finally:
            functorch_config.activation_memory_budget = 1.0
torch._dynamo.reset()
