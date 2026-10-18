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
        x2 = _flat(x)
        z = x2 @ a.T
        y = x2 @ weight.T + scale * (z @ b.T)
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(z, weight, a, b)
        ctx.scale = scale
        ctx.x_shape = x.shape
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        z, weight, a, b = ctx.saved_tensors
        g = _flat(grad_out)
        grad_x = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = (g @ weight + ctx.scale * ((g @ b) @ a)).reshape(ctx.x_shape)
        if ctx.needs_input_grad[4]:
            grad_b = ctx.scale * (g.T @ z)
        return grad_x, None, None, None, grad_b, None

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
    def gelu_grad(u):
        cdf = 0.5 * (1.0 + torch.erf(u * (1.0 / math.sqrt(2.0))))
        pdf = torch.exp(-0.5 * u * u) * (1.0 / math.sqrt(2.0 * math.pi))
        return cdf + u * pdf

    @staticmethod
    def forward(ctx, x, w1, a1, b1, s1, w2, a2, b2, s2):
        x2 = _flat(x)
        z1 = x2 @ a1.T
        u = x2 @ w1.T + s1 * (z1 @ b1.T)
        g = F.gelu(u)
        z2 = g @ a2.T
        y = g @ w2.T + s2 * (z2 @ b2.T)
        ctx.save_for_backward(x2, z1, u, z2, w1, a1, b1, w2, a2, b2)
        ctx.scales = (s1, s2)
        ctx.x_shape = x.shape
        return y.reshape(*x.shape[:-1], w2.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x2, z1, u, z2, w1, a1, b1, w2, a2, b2 = ctx.saved_tensors
        s1, s2 = ctx.scales
        need = ctx.needs_input_grad
        gy = _flat(grad_out)
        g = F.gelu(u)
        dz2 = s2 * (gy @ b2)
        grad_b2 = s2 * (gy.T @ z2) if need[7] else None
        grad_a2 = dz2.T @ g if need[6] else None
        del g
        du = (gy @ w2 + dz2 @ a2) * LoRAMLPFn.gelu_grad(u)
        dz1 = s1 * (du @ b1)
        grad_b1 = s1 * (du.T @ z1) if need[3] else None
        grad_a1 = dz1.T @ x2 if need[2] else None
        grad_x = (du @ w1 + dz1 @ a1).reshape(ctx.x_shape) if need[0] else None
        return grad_x, None, grad_a1, grad_b1, None, None, grad_a2, grad_b2, None

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
        mask = dropout_mask(x.shape, p, seed, x.device, x.dtype)
        x2 = _flat(x)
        z = _flat(x * mask) @ a.T
        del mask
        y = x2 @ weight.T + scale * (z @ b.T)
        ctx.save_for_backward(x, z, weight, a, b)
        ctx.scale, ctx.p, ctx.seed = scale, p, seed
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x, z, weight, a, b = ctx.saved_tensors
        need = ctx.needs_input_grad
        mask = _flat(dropout_mask(x.shape, ctx.p, ctx.seed, x.device, x.dtype))
        gy = _flat(grad_out)
        dz = ctx.scale * (gy @ b)
        grad_b = ctx.scale * (gy.T @ z) if need[3] else None
        grad_a = dz.T @ (_flat(x) * mask) if need[2] else None
        grad_x = (gy @ weight + (dz @ a) * mask).reshape(x.shape) if need[0] else None
        return grad_x, None, grad_a, grad_b, None, None, None

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
    params = list(parameters)
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup >= 0 and repeats >= 1")
    device = torch.device(device)
    cuda, mps = device.type == "cuda", device.type == "mps"

    def clear():
        for p in params:
            p.grad = None

    def sync():
        if cuda:
            torch.cuda.synchronize(device)
        elif mps:
            torch.mps.synchronize()

    def allocated():
        if cuda:
            return torch.cuda.memory_allocated(device)
        if mps:
            return torch.mps.current_allocated_memory()
        return None

    for _ in range(warmup):
        clear()
        make_loss().backward()
    clear()
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    before = allocated()
    inventory = SavedInventory(params)
    with inventory.context():
        loss = make_loss()
    sync()
    after = allocated()
    loss.backward()
    sync()
    peak = torch.cuda.max_memory_allocated(device) - before if cuda else None
    del loss
    if cuda:
        began = time.perf_counter()
        while (time.perf_counter() - began) * 1000 < warm_ms:
            clear()
            make_loss().backward()
            torch.cuda.synchronize(device)
    times = []
    for _ in range(repeats):
        clear()
        if cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            make_loss().backward()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        else:
            sync()
            began = time.perf_counter()
            make_loss().backward()
            sync()
            times.append((time.perf_counter() - began) * 1000)
    q25, q50, q75 = (
        torch.tensor(times, dtype=torch.float64)
        .quantile(torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
        .tolist()
    )
    return {
        "saved_bytes": inventory.bytes,
        "after_forward_bytes": None if before is None else after - before,
        "peak_delta_bytes": peak,
        "ms_median": q50,
        "ms_p25": q25,
        "ms_p75": q75,
        "stable": q75 <= 1.5 * q25,
    }
    # END TODO R4
