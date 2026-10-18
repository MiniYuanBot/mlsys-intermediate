"""Provided for stage R: a Qwen3-0.6B-shaped decode step and a synthetic serving trace.

DecodeModel is `layers` pre-norm decoder layers with Qwen3-0.6B widths (hidden 1024,
16 query heads, 8 KV heads, head_dim 128, MLP 3072, RMSNorm eps 1e-6) and random weights.
Each step takes one new token per sequence, x of shape [batch, hidden], and attends to a
fixed-length KV cache slice [:batch]. Every row is independent, so padding the batch with
zero rows and slicing the output back is exact (up to GEMM kernel choice).
"""

import math
import random

import torch
import torch.nn.functional as F

QWEN3_0_6B = dict(hidden=1024, heads=16, kv_heads=8, head_dim=128, mlp=3072)


class DecodeLayer(torch.nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, mlp):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.qkv = torch.nn.Linear(hidden, (heads + 2 * kv_heads) * head_dim, bias=False)
        self.o = torch.nn.Linear(heads * head_dim, hidden, bias=False)
        self.gate_up = torch.nn.Linear(hidden, 2 * mlp, bias=False)
        self.down = torch.nn.Linear(mlp, hidden, bias=False)
        self.norm1 = torch.nn.Parameter(torch.rand(hidden) + 0.5)
        self.norm2 = torch.nn.Parameter(torch.rand(hidden) + 0.5)
        for linear in [self.qkv, self.o, self.gate_up, self.down]:
            torch.nn.init.normal_(linear.weight, std=1 / math.sqrt(linear.in_features))

    def forward(self, x, k_cache, v_cache):
        b, hidden = x.shape
        h = F.rms_norm(x, (hidden,), self.norm1, 1e-6)
        q, k, v = self.qkv(h).split(
            [
                self.heads * self.head_dim,
                self.kv_heads * self.head_dim,
                self.kv_heads * self.head_dim,
            ],
            -1,
        )
        del k, v  # a real step appends them to the cache; the fixed cache keeps shapes static
        q = q.view(b, self.heads, 1, self.head_dim)
        attn = F.scaled_dot_product_attention(q, k_cache[:b], v_cache[:b], enable_gqa=True)
        x = x + self.o(attn.reshape(b, self.heads * self.head_dim))
        gate, up = self.gate_up(F.rms_norm(x, (hidden,), self.norm2, 1e-6)).chunk(2, -1)
        return x + self.down(F.silu(gate) * up)


class DecodeModel(torch.nn.Module):
    def __init__(
        self, layers=4, max_batch=64, context=256, device="cpu", dtype=torch.float32, seed=0, **dims
    ):
        super().__init__()
        dims = {**QWEN3_0_6B, **dims}
        torch.manual_seed(seed)
        self.hidden = dims["hidden"]
        self.layers = torch.nn.ModuleList(DecodeLayer(**dims) for _ in range(layers))
        shape = (max_batch, dims["kv_heads"], context, dims["head_dim"])
        g = torch.Generator().manual_seed(seed + 1)
        self.k = [torch.randn(shape, generator=g).to(device, dtype) for _ in range(layers)]
        self.v = [torch.randn(shape, generator=g).to(device, dtype) for _ in range(layers)]
        self.to(device, dtype)
        self.max_batch = max_batch

    @torch.no_grad()
    def forward(self, x):
        if x.shape[0] > self.max_batch:
            raise ValueError("batch larger than the KV cache")
        for layer, k, v in zip(self.layers, self.k, self.v):
            x = layer(x, k, v)
        return x


def serving_trace(steps, arrival_rate, max_batch=64, mean_output=160, seed=0):
    """Batch size per decode step of a continuous-batching server (deterministic).

    Each step: Poisson(arrival_rate) new requests join a FIFO queue; up to max_batch run;
    each running request emits one token and finishes after its output length
    (lognormal, mean about mean_output). Returns a list of batch sizes, idle steps
    (batch 0) removed. Little's law: mean batch is about arrival_rate * mean_output
    until the max_batch cap binds.
    """
    rng = random.Random(seed)
    sigma = 0.8
    mu = math.log(mean_output) - sigma * sigma / 2
    queue, running, sizes = 0, [], []
    for _ in range(steps):
        # Poisson sample by inversion (small rates)
        arrivals, p, u = 0, math.exp(-arrival_rate), rng.random()
        cumulative = p
        while u > cumulative:
            arrivals += 1
            p *= arrival_rate / arrivals
            cumulative += p
        queue += arrivals
        while queue and len(running) < max_batch:
            queue -= 1
            running.append(max(1, int(rng.lognormvariate(mu, sigma))))
        if running:
            sizes.append(len(running))
        running = [left - 1 for left in running if left > 1]
    return sizes
