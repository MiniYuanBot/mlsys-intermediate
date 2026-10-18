"""Stage R: custom autograd.Functions that save less than eager autograd.

Eager PyTorch already saves only what each op's backward needs: an mm operand is
saved only if the *other* operand needs a gradient. So a frozen-W LoRA linear already
saves only X and X A^T, and with A frozen only X A^T. The Functions below target
savings eager autograd cannot get on its own:

  R1 LoRAFALinearFn  frozen A: save only Z = X A^T (same set as eager LoRA-FA; the
                     exercise is writing the backward and proving the save set)
  R2 LoRAMLPFn       up -> GELU -> down with LoRA on both: recompute GELU(u) in
                     backward instead of saving it (eager saves u *and* GELU(u))
  R3 LoRADropoutFn   LoRA dropout: save an int seed and regenerate the mask in
                     backward instead of saving the mask and the dropped input
  R4 measure_variant saved bytes, allocator bytes after forward, CUDA peak, time
"""

import math
import time

import torch
import torch.nn.functional as F

from src.train_student import SavedInventory


def dropout_mask(shape, p, seed, device, dtype):
    """Provided. Inverted-dropout multiplier (0 or 1/(1-p)) from a private generator.

    The same (shape, p, seed, device) always gives the same mask; the global RNG is
    not touched. Forward and backward of R3 must both use this function.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    keep = torch.rand(shape, generator=generator, device=device) >= p
    return keep.to(dtype) / (1.0 - p)


def _flat(x):
    return x.reshape(-1, x.shape[-1])


class LoRAFALinearFn(torch.autograd.Function):
    """y = x W^T + bias + s (x A^T) B^T with W, bias and A frozen; trains B (and x).

    apply(x, weight, bias_or_None, a, b, scale). x has any leading dims.
    Save for backward: Z = x A^T and the parameters only, never x itself.
    Backward: dB = s dY^T Z, dx = dY W + s (dY B) A; None for weight/bias/a/scale.
    Use ctx.needs_input_grad to skip dx / dB when they are not needed.
    """

    # BEGIN TODO R1
    @staticmethod
    def forward(ctx, x, weight, bias, a, b, scale):
        raise NotImplementedError("R1")

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("R1")

    # END TODO R1


def lora_fa_linear(x, layer):
    """Provided wrapper for a LoRALinear whose A is frozen."""
    if layer.a.requires_grad:
        raise ValueError("LoRA-FA needs a frozen A: layer.a.requires_grad_(False)")
    if layer.merged:
        raise ValueError("unmerge first")
    return LoRAFALinearFn.apply(x, layer.weight, layer.bias, layer.a, layer.b, layer.scale)


class LoRAMLPFn(torch.autograd.Function):
    """Fused up -> GELU(exact, erf) -> down, LoRA on both projections, no bias.

    apply(x, w1, a1, b1, s1, w2, a2, b2, s2); W frozen, A/B trainable, x any leading dims.
      z1 = x A1^T, u = x W1^T + s1 z1 B1^T, g = GELU(u), z2 = g A2^T,
      y = g W2^T + s2 z2 B2^T.
    Save exactly x, z1, u, z2 (+ parameters). g is NOT saved: backward recomputes it
    from u. GELU'(u) = Phi(u) + u phi(u) with the standard normal cdf/pdf.
    Gradients for x, a1, b1, a2, b2 (respect ctx.needs_input_grad); None for the rest.
    """

    # BEGIN TODO R2
    @staticmethod
    def forward(ctx, x, w1, a1, b1, s1, w2, a2, b2, s2):
        raise NotImplementedError("R2")

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("R2")

    # END TODO R2


def lora_mlp(x, up, down):
    """Provided wrapper: fused MLP over two bias-free, unmerged LoRALinear layers."""
    if up.bias is not None or down.bias is not None or up.merged or down.merged:
        raise ValueError("fused MLP supports unmerged, bias-free layers only")
    return LoRAMLPFn.apply(
        x, up.weight, up.a, up.b, up.scale, down.weight, down.a, down.b, down.scale
    )


class LoRADropoutFn(torch.autograd.Function):
    """y = x W^T + s ((x * m) A^T) B^T, m = dropout_mask(x.shape, p, seed, ...).

    apply(x, weight, a, b, scale, p, seed); W frozen, A/B trainable, no bias.
    Save x, Z = (x*m) A^T and the parameters; keep p and the int seed on ctx. Do not
    save m or x*m: backward regenerates m with dropout_mask and the *same* seed.
      dB = s dY^T Z, dZ = s dY B, dA = dZ^T (x*m), dx = dY W + (dZ A) * m.
    """

    # BEGIN TODO R3
    @staticmethod
    def forward(ctx, x, weight, a, b, scale, p, seed):
        raise NotImplementedError("R3")

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("R3")

    # END TODO R3


def lora_dropout_linear(x, layer, p, seed=None):
    """Provided wrapper. seed=None draws one from the global CPU RNG (torch.manual_seed
    makes runs reproducible); an explicit seed makes the call deterministic."""
    if layer.bias is not None or layer.merged:
        raise ValueError("LoRA dropout Function supports unmerged, bias-free layers")
    if not 0.0 <= p < 1.0:
        raise ValueError("dropout p must be in [0, 1)")
    if seed is None:
        seed = int(torch.randint(0, 2**62, (1,)).item())
    return LoRADropoutFn.apply(x, layer.weight, layer.a, layer.b, layer.scale, p, seed)


def measure_variant(make_loss, parameters, device, warmup=3, repeats=7, warm_ms=300.0):
    """Measure one forward+backward variant. make_loss() builds a fresh graph and
    returns a scalar loss; parameters are excluded from saved bytes and their .grad
    is set to None before every pass.

    1. warmup passes (fw+bw), not recorded;
    2. one recorded pass: synchronize, reset CUDA peak stats, read allocated bytes
       `before`; forward under a SavedInventory(parameters); read allocated bytes
       again (after_forward_bytes = after - before); backward; synchronize;
       peak_delta_bytes = max_memory_allocated - before (CUDA only);
    3. CUDA only: keep the GPU busy with the same pass for >= warm_ms (laptop GPUs
       drop to a low memory clock between configs and time up to 8x slower);
    4. `repeats` timed passes (CUDA: torch.cuda.Event; otherwise perf_counter after
       synchronizing), reporting ms_median / ms_p25 / ms_p75 and
       stable = (ms_p75 <= 1.5 * ms_p25).
    Allocated bytes: CUDA torch.cuda.memory_allocated, MPS torch.mps.
    current_allocated_memory, CPU None. On return every parameter's .grad holds the
    gradient of exactly one pass.
    """
    # BEGIN TODO R4
    raise NotImplementedError("R4")
    # END TODO R4
