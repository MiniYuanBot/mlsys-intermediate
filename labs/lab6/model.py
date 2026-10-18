"""Provided models. Block/Model: stages A-C. MLPBlock: stage R variants."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.train_student import LoRALinear, run_blocks


class Block(nn.Module):
    """x + dropout(down(GELU(up(LayerNorm(x))))), LoRA on up (h -> 2h) and down."""

    def __init__(self, width=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = LoRALinear(width, width * 2, rank=8)
        self.down = LoRALinear(width * 2, width, rank=8)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.down(F.gelu(self.up(self.norm(x)))))


class Model(nn.Module):
    def __init__(self, width=256, layers=8):
        super().__init__()
        self.blocks = nn.ModuleList([Block(width) for _ in range(layers)])

    def forward(self, x, keep):
        return run_blocks(self.blocks, x, keep)


def eager_lora_dropout(x, layer, p):
    """PEFT-style LoRA dropout in plain eager PyTorch: dropout only on the LoRA input."""
    base = F.linear(x, layer.weight, layer.bias)
    dropped = F.dropout(x, p, training=True)
    return base + layer.scale * F.linear(F.linear(dropped, layer.a), layer.b)


R_VARIANTS = (
    "eager",  # LoRALinear (A1) modules, A and B trainable
    "eager_fa",  # same modules with A frozen: eager autograd drops X by itself
    "fn_fa",  # R1 Function, A frozen
    "fused",  # R2 Function, GELU recomputed in backward
    "eager_drop",  # LoRA dropout via F.dropout (saves mask + dropped input)
    "fn_drop",  # R3 Function, seed only
    "checkpoint",  # eager block under non-reentrant checkpoint
)


class MLPBlock(nn.Module):
    """x + down(GELU(up(LayerNorm(x)))) with Qwen-like widths, executed as `variant`.

    Simplified from Qwen3: LayerNorm instead of RMSNorm, GELU instead of SwiGLU,
    no attention. All variants compute the same function (up to dropout masks).
    """

    def __init__(self, width, inter, rank=16, variant="eager", p=0.1):
        super().__init__()
        if variant not in R_VARIANTS:
            raise ValueError(f"variant {variant}")
        self.variant, self.p = variant, p
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.up = LoRALinear(width, inter, rank=rank, alpha=2 * rank)
        self.down = LoRALinear(inter, width, rank=rank, alpha=2 * rank)
        with torch.no_grad():
            self.up.b.normal_(std=0.02)
            self.down.b.normal_(std=0.02)
        if variant in ("eager_fa", "fn_fa"):
            self.up.a.requires_grad_(False)
            self.down.a.requires_grad_(False)

    def _eager(self, x):
        return x + self.down(F.gelu(self.up(self.norm(x))))

    def forward(self, x):
        from src.fused_student import lora_dropout_linear, lora_fa_linear, lora_mlp

        v = self.variant
        if v in ("eager", "eager_fa"):
            return self._eager(x)
        if v == "checkpoint":
            return checkpoint(self._eager, x, use_reentrant=False, preserve_rng_state=True)
        h = self.norm(x)
        if v == "fn_fa":
            return x + lora_fa_linear(F.gelu(lora_fa_linear(h, self.up)), self.down)
        if v == "fused":
            return x + lora_mlp(h, self.up, self.down)
        if v == "eager_drop":
            return x + eager_lora_dropout(
                F.gelu(eager_lora_dropout(h, self.up, self.p)), self.down, self.p
            )
        return x + lora_dropout_linear(
            F.gelu(lora_dropout_linear(h, self.up, self.p)), self.down, self.p
        )


def predicted_saved_floats(variant, width, inter, rank, mask_bytes):
    """Hand-derived saved floats per token for one MLPBlock (parameters excluded).

    LayerNorm saves its input (h) and mean/rstd (2). up/down LoRA with frozen W save
    their input only if A trains (h for up, f for down) and Z (r each). GELU saves u (f).
    F.dropout saves a mask of mask_bytes per element (CUDA bool: 1, CPU float: 4).
    Checkpoint keeps only the block input (h). Returns a float count (4-byte units).
    """
    h, f, r = width, inter, rank
    ln = h + 2
    table = {
        "eager": ln + h + r + f + f + r,
        "eager_fa": ln + r + f + r,
        "fn_fa": ln + r + f + r,
        "fused": ln + h + r + f + r,
        "eager_drop": ln + h + r + f + f + r + (h + f) * mask_bytes / 4,
        "fn_drop": ln + h + r + f + f + r,
        "checkpoint": h,
    }
    return table[variant]
