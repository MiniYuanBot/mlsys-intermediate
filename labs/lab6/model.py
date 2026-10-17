import torch
from src.train_student import LoRALinear, run_blocks
from torch import nn


class Block(nn.Module):
    def __init__(self, width=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = LoRALinear(width, width * 2, rank=8)
        self.down = LoRALinear(width * 2, width, rank=8)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.down(torch.nn.functional.gelu(self.up(self.norm(x)))))


class Model(nn.Module):
    def __init__(self, width=256, layers=8):
        super().__init__()
        self.blocks = nn.ModuleList([Block(width) for _ in range(layers)])

    def forward(self, x, keep):
        return run_blocks(self.blocks, x, keep)
