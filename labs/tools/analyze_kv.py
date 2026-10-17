#!/usr/bin/env python3
"""Plot one captured KV head and compare grouping axes (fake quantization only)."""

import argparse
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("dump", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--layer", type=int, default=0)
p.add_argument("--head", type=int, default=0)
a = p.parse_args()
cache = torch.load(a.dump, map_location="cpu", weights_only=True)
layer = cache["layers"][a.layer]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
rows = []
for ax, name in zip(axes, ["k", "v"], strict=True):
    x = layer[name][0, a.head].float()
    im = ax.imshow(x.abs().numpy(), aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="absolute magnitude")
    ax.set(title=f"{name.upper()} absolute values", xlabel="channel", ylabel="token")
    for group in [16, 64]:
        for axis in ["token", "channel"]:
            restored = torch.empty_like(x)
            n = x.shape[1] if axis == "token" else x.shape[0]
            for begin in range(0, n, group):
                part = (
                    x[:, begin : begin + group] if axis == "token" else x[begin : begin + group, :]
                )
                dim = 1 if axis == "token" else 0
                lo = part.amin(dim=dim, keepdim=True)
                hi = part.amax(dim=dim, keepdim=True)
                scale = (hi - lo) / 15
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                dq = ((part - lo) / scale).round().clamp(0, 15) * scale + lo
                if axis == "token":
                    restored[:, begin : begin + group] = dq
                else:
                    restored[begin : begin + group, :] = dq
            rows.append(
                {
                    "tensor": name,
                    "layer": a.layer,
                    "head": a.head,
                    "axis": axis,
                    "group": group,
                    "relative_l2": float((restored - x).norm() / x.norm().clamp_min(1e-12)),
                    "scope": "offline fake quantization, not model quality or storage saving",
                }
            )
a.out.mkdir(parents=True, exist_ok=False)
fig.suptitle(f"Captured KV | layer {a.layer}, head {a.head} | separate color scales")
fig.tight_layout()
fig.savefig(a.out / "kv-distribution.png", dpi=160)
(a.out / "axis-error.json").write_text(json.dumps(rows, indent=2) + "\n")
print(a.out)
