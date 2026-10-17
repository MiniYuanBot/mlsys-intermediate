import argparse
import copy
import json
import math

import torch
from model import Block
from src.train_student import calibrate_blocks, measure_policy, parameter_bytes, plan_modes
from torch import nn

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
args = p.parse_args()
torch.set_num_threads(2)
for seed in (17, 29, 43):
    for width in (64, 128):
        torch.manual_seed(seed)
        base = nn.ModuleList([Block(width, dropout=0) for _ in range(4)]).to(args.device)
        x = torch.randn(3, 32, width, device=args.device)
        target = torch.randn_like(x)
        calibration = calibrate_blocks(base, x[:2], repeats=2)
        # Planner uses actual retain/checkpoint calibration only. Offload cost must
        # not be invented from bandwidth; evaluate that policy in a separate run.
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
        for name, modes in [
            ("retain", ["retain"] * 4),
            ("checkpoint", ["checkpoint"] * 4),
            ("offload", ["offload"] * 4),
            ("calibrated", selected),
        ]:
            blocks = copy.deepcopy(base)
            batches = [(x[:2], target[:2]), (x[2:], target[2:])]
            result = measure_policy(blocks, batches, modes, repeats=3)
            print(
                json.dumps(
                    {
                        "kind": "actual_torch_training",
                        "seed": seed,
                        "width": width,
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
                )
            )
            del blocks
