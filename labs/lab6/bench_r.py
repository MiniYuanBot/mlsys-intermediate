"""R benchmark: saved bytes (measured vs hand-derived), allocator bytes, peak and step time
of every MLPBlock variant over several sequence lengths (Qwen3-0.6B widths on CUDA)."""

import argparse
import json

import torch
from model import R_VARIANTS, MLPBlock, predicted_saved_floats
from src.fused_student import measure_variant
from torch import nn

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
args = p.parse_args()
DEV = torch.device(args.device)
torch.set_num_threads(2)
if DEV.type == "cuda":
    width, inter, rank, layers, lengths = 1024, 3072, 16, 4, (1024, 4096, 8192)
elif DEV.type == "mps":
    width, inter, rank, layers, lengths = 1024, 3072, 16, 4, (512, 2048)
else:
    width, inter, rank, layers, lengths = 256, 768, 8, 2, (256, 1024)
mask_bytes = 1 if DEV.type == "cuda" else 4  # F.dropout mask: CUDA bool, CPU float32
for tokens in lengths:
    for variant in R_VARIANTS:
        torch.manual_seed(7)
        model = nn.Sequential(
            *[MLPBlock(width, inter, rank, variant, p=0.1) for _ in range(layers)]
        ).to(DEV)
        x = torch.randn(tokens, width, device=DEV, requires_grad=True)
        # x stays out of `parameters`: its storage is the first LayerNorm's saved input.
        row = measure_variant(
            lambda: model(x).square().mean(), list(model.parameters()), DEV, warmup=3, repeats=7
        )
        # blocks (hand-derived) + the loss's square(), which saves the model output (tokens x width)
        predicted = (
            (predicted_saved_floats(variant, width, inter, rank, mask_bytes) * layers + width)
            * 4
            * tokens
        )
        print(
            json.dumps(
                {
                    "kind": "r_variant",
                    "device": args.device,
                    "gpu": torch.cuda.get_device_name(DEV) if DEV.type == "cuda" else None,
                    "torch": torch.__version__,
                    "variant": variant,
                    "tokens": tokens,
                    "width": width,
                    "inter": inter,
                    "rank": rank,
                    "layers": layers,
                    "predicted_saved_bytes_modeled": int(predicted),
                    "saved_bytes_measured": row["saved_bytes"],
                    **{k: v for k, v in row.items() if k != "saved_bytes"},
                }
            ),
            flush=True,
        )
        del model, x
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
