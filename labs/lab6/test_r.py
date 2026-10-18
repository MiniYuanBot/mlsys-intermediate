"""Stage R checks: custom autograd.Functions against float64 eager oracles, double-
precision gradcheck, hand-derived saved-byte counts and (on CUDA) allocator deltas."""

import argparse
import math

import torch
import torch.nn.functional as F
from checks import Report
from model import MLPBlock, eager_lora_dropout, predicted_saved_floats
from src.fused_student import (
    LoRAFALinearFn,
    LoRAMLPFn,
    dropout_mask,
    lora_dropout_linear,
    lora_fa_linear,
    lora_mlp,
    measure_variant,
)
from src.train_student import LoRALinear
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
DEV = torch.device(args.device)
CUDA = DEV.type == "cuda"
F64 = torch.float64
torch.set_num_threads(2)


class ByteHook:
    """Test-side saved-storage counter; parameters (and their views) excluded."""

    def __init__(self, parameters=()):
        self.skip = {p.untyped_storage().data_ptr() for p in parameters}
        self.storages = {}

    def pack(self, t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in self.skip:
            self.storages[ptr] = t.untyped_storage().nbytes()
        return t.detach()

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self.pack, lambda t: t)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)

    @property
    def bytes(self):
        return sum(self.storages.values())


def layer64(d_in, d_out, rank, bias=False, seed=0):
    torch.manual_seed(seed)
    layer = LoRALinear(d_in, d_out, rank=rank, alpha=3 * rank, bias=bias).to(device=DEV, dtype=F64)
    with torch.no_grad():
        layer.b.normal_()  # B != 0 so dA and the LoRA part of dX are not trivially zero
        if bias:
            layer.bias.normal_()
    return layer


def dense(x, layer, x_lora=None):
    """Independent LoRA formula; x_lora is the (dropped) input of the low-rank branch."""
    x_lora = x if x_lora is None else x_lora
    out = x @ layer.weight.T + layer.scale * (x_lora @ layer.a.T) @ layer.b.T
    return out if layer.bias is None else out + layer.bias


def grads(tensors):
    return [None if t.grad is None else t.grad.detach().clone() for t in tensors]


def clear(tensors):
    for t in tensors:
        t.grad = None


def gradcheck(r, name, fn, inputs):
    try:
        ok = torch.autograd.gradcheck(
            fn, inputs, eps=1e-6, atol=1e-7, rtol=1e-5, raise_exception=True
        )
        return r.check(name, ok, True, ok)
    except RuntimeError as error:
        return r.check(name, False, "gradcheck passes", " ".join(str(error).split())[:180])


# ---------------------------------------------------------------- R1
def r1_lora_fa(r):
    layer = layer64(6, 5, 2, bias=True, seed=11)
    layer.a.requires_grad_(False)
    x = torch.randn(2, 3, 6, device=DEV, dtype=F64, requires_grad=True)
    g = torch.randn(2, 3, 5, device=DEV, dtype=F64)
    params = [layer.weight, layer.bias, layer.a, layer.b]
    with ByteHook(params) as hook:
        y = lora_fa_linear(x, layer)
    r.close(
        "R1 output = x W^T + bias + s (x A^T) B^T (leading dims kept)",
        y,
        dense(x, layer),
        rtol=1e-12,
        atol=1e-12,
    )
    (y * g).sum().backward()
    got = grads([x, layer.b])
    clear([x, layer.b])
    (dense(x, layer) * g).sum().backward()
    want = grads([x, layer.b])
    r.close("R1 dX = eager float64 reference", got[0], want[0], rtol=1e-12, atol=1e-12)
    r.close("R1 dB = eager float64 reference", got[1], want[1], rtol=1e-12, atol=1e-12)
    r.check(
        "R1 frozen A/W/bias get no grad",
        all(p.grad is None for p in (layer.a, layer.weight, layer.bias)),
        "None",
        [p.grad is None for p in (layer.a, layer.weight, layer.bias)],
    )
    n, r_ = 6, 2
    r.equal("R1 saves only Z = X A^T: N r 8 bytes", hook.bytes, n * r_ * 8)
    with ByteHook(params) as eager_fa:
        dense(x, layer)
    r.equal(
        "R1 lesson: eager with frozen A already saves the same N r 8 bytes",
        eager_fa.bytes,
        n * r_ * 8,
    )
    layer.a.requires_grad_(True)
    with ByteHook(params) as eager_full:
        dense(x, layer)
    r.equal(
        "R1 lesson: eager LoRA with trainable A saves X and Z: N (d_in + r) 8",
        eager_full.bytes,
        n * (6 + r_) * 8,
    )
    r.raises("R1 wrapper rejects a trainable A", ValueError, lambda: lora_fa_linear(x, layer))
    layer.a.requires_grad_(False)
    gradcheck(
        r,
        "R1 gradcheck (float64) over x and B",
        lambda xx, bb: LoRAFALinearFn.apply(xx, layer.weight, layer.bias, layer.a, bb, layer.scale),
        (x, layer.b),
    )
    clear([x, layer.b])
    plain = x.detach()
    (lora_fa_linear(plain, layer) * g).sum().backward()
    b_only = layer.b.grad.clone()
    clear([layer.b])
    (dense(plain, layer) * g).sum().backward()
    r.close(
        "R1 x without requires_grad: dB still correct", b_only, layer.b.grad, rtol=1e-12, atol=1e-12
    )


# ---------------------------------------------------------------- R2
def r2_fused_mlp(r):
    up, down = layer64(4, 5, 2, seed=21), layer64(5, 4, 2, seed=22)
    trainable = [up.a, up.b, down.a, down.b]
    params = [up.weight, down.weight] + trainable
    x = torch.randn(2, 3, 4, device=DEV, dtype=F64, requires_grad=True)
    g = torch.randn(2, 3, 4, device=DEV, dtype=F64)

    def eager(v):
        return dense(F.gelu(dense(v, up)), down)

    with ByteHook(params) as fused_hook:
        y = lora_mlp(x, up, down)
    r.close("R2 output = down(GELU(up(x))) dense float64", y, eager(x), rtol=1e-12, atol=1e-12)
    (y * g).sum().backward()
    got = grads([x] + trainable)
    clear([x] + trainable)
    with ByteHook(params) as eager_hook:
        reference = eager(x)
    (reference * g).sum().backward()
    want = grads([x] + trainable)
    for label, a, b in zip(("dX", "dA_up", "dB_up", "dA_down", "dB_down"), got, want, strict=True):
        r.close(f"R2 {label} = eager float64 reference", a, b, rtol=1e-10, atol=1e-12)
    n, h, f, rank = 6, 4, 5, 2
    r.equal(
        "R2 fused saves x, z1, u, z2: N (h + f + 2r) 8 bytes",
        fused_hook.bytes,
        n * (h + f + 2 * rank) * 8,
    )
    r.equal(
        "R2 eager saves x, z1, u, GELU(u), z2: N (h + 2f + 2r) 8 bytes",
        eager_hook.bytes,
        n * (h + 2 * f + 2 * rank) * 8,
    )
    xs = torch.randn(3, 4, device=DEV, dtype=F64, requires_grad=True)
    gradcheck(
        r,
        "R2 gradcheck (float64) over x, A_up, B_up, A_down, B_down",
        lambda xx, a1, b1, a2, b2: LoRAMLPFn.apply(
            xx, up.weight, a1, b1, up.scale, down.weight, a2, b2, down.scale
        ),
        (xs, up.a, up.b, down.a, down.b),
    )
    u = torch.linspace(-6, 6, 97, device=DEV, dtype=F64)
    try:
        derivative = LoRAMLPFn.gelu_grad(u)
    except AttributeError:
        derivative = None
    if derivative is not None:
        exact = torch.autograd.functional.jacobian(lambda t: F.gelu(t).sum(), u)
        r.close(
            "R2 gelu_grad = derivative of exact (erf) GELU",
            derivative,
            exact,
            rtol=1e-10,
            atol=1e-12,
        )
    up.a.requires_grad_(False)
    clear([x] + trainable)
    (lora_mlp(x.detach(), up, down) * g).sum().backward()
    partial = grads([up.b, down.a, down.b])
    clear(trainable)
    (eager(x.detach()) * g).sum().backward()
    r.close(
        "R2 frozen A_up and plain x: remaining grads still correct",
        partial,
        grads([up.b, down.a, down.b]),
        rtol=1e-10,
        atol=1e-12,
    )
    r.check("R2 frozen A_up gets no grad", up.a.grad is None, None, type(up.a.grad))
    up.a.requires_grad_(True)


# ---------------------------------------------------------------- R3
def r3_seed_dropout(r):
    layer = layer64(6, 5, 2, seed=31)
    p, seed = 0.3, 1234
    params = [layer.weight, layer.a, layer.b]
    x = torch.randn(2, 3, 6, device=DEV, dtype=F64, requires_grad=True)
    g = torch.randn(2, 3, 5, device=DEV, dtype=F64)
    mask = dropout_mask(x.shape, p, seed, DEV, F64)
    with ByteHook(params) as hook:
        y = lora_dropout_linear(x, layer, p, seed)
    r.close(
        "R3 output = dense LoRA with the seed's mask on the LoRA input",
        y,
        dense(x, layer, x * mask),
        rtol=1e-12,
        atol=1e-12,
    )
    (y * g).sum().backward()
    got = grads([x, layer.a, layer.b])
    clear([x, layer.a, layer.b])
    (dense(x, layer, x * mask) * g).sum().backward()
    want = grads([x, layer.a, layer.b])
    for label, a, b in zip(("dX", "dA", "dB"), got, want, strict=True):
        r.close(
            f"R3 {label} = eager float64 reference with the same mask", a, b, rtol=1e-12, atol=1e-12
        )
    n, d, rank = 6, 6, 2
    r.equal(
        "R3 saves x and Z only: N (d + r) 8 bytes (no mask, no dropped input)",
        hook.bytes,
        n * (d + rank) * 8,
    )
    torch.manual_seed(0)
    with ByteHook(params) as eager_hook:
        eager_lora_dropout(x, layer, p)
    mask_bytes = 1 if CUDA else 8  # CUDA keeps a bool mask, CPU a mask in the input dtype
    r.equal(
        "R3 lesson: eager F.dropout LoRA saves dropped input + mask + Z",
        eager_hook.bytes,
        n * (d + rank) * 8 + n * d * mask_bytes,
    )
    gradcheck(
        r,
        "R3 gradcheck (float64, fixed seed) over x, A, B",
        lambda xx, aa, bb: lora_dropout_linear_raw(xx, layer, aa, bb, p, seed),
        (x, layer.a, layer.b),
    )
    torch.manual_seed(5)
    first = lora_dropout_linear(x, layer, p)
    torch.manual_seed(5)
    second = lora_dropout_linear(x, layer, p)
    third = lora_dropout_linear(x, layer, p)
    r.check(
        "R3 seed=None is reproducible under torch.manual_seed and changes between calls",
        torch.equal(first, second) and not torch.equal(second, third),
        "same, then different",
        (torch.equal(first, second), torch.equal(second, third)),
    )
    torch.manual_seed(6)
    drawn = int(torch.randint(0, 2**62, (1,)).item())
    torch.manual_seed(6)
    clear([x, layer.a, layer.b])
    (lora_dropout_linear(x, layer, p) * g).sum().backward()
    got = grads([x, layer.a])
    clear([x, layer.a, layer.b])
    m2 = dropout_mask(x.shape, p, drawn, DEV, F64)
    (dense(x, layer, x * m2) * g).sum().backward()
    r.close(
        "R3 seed=None: backward regenerates the forward mask",
        got,
        grads([x, layer.a]),
        rtol=1e-12,
        atol=1e-12,
    )
    zero = lora_dropout_linear(x, layer, 0.0, seed)
    r.close("R3 p=0 is the plain LoRA layer", zero, dense(x, layer), rtol=1e-12, atol=1e-12)


def lora_dropout_linear_raw(x, layer, a, b, p, seed):
    from src.fused_student import LoRADropoutFn

    return LoRADropoutFn.apply(x, layer.weight, a, b, layer.scale, p, seed)


# ---------------------------------------------------------------- R4
def r4_measure(r):
    torch.manual_seed(41)
    weight = nn.Parameter(torch.randn(1024, device=DEV))
    payload_floats = 1 << 20  # 4 MiB saved by every pass
    calls = [0]
    temp_floats = 8 << 20  # 32 MiB transient allocation inside every pass

    def make_loss():
        calls[0] += 1
        if CUDA:
            temp = torch.ones(temp_floats, device=DEV)
            del temp
        payload = torch.randn(payload_floats, device=DEV)
        # mul saves only payload (weight needs the grad); square saves the parameter itself
        return (payload.view(1024, 1024) * weight).sum() + weight.square().sum()

    held = torch.ones(16 << 20, device=DEV) if CUDA else None  # 64 MiB alive throughout
    if CUDA:
        spike = torch.ones(32 << 20, device=DEV)  # 128 MiB freed before measuring
        del spike
    row = measure_variant(make_loss, [weight], DEV, warmup=2, repeats=5)
    r.equal(
        "R4 saved_bytes = the one 4 MiB payload (saved parameter excluded)",
        row["saved_bytes"],
        payload_floats * 4,
    )
    r.check(
        "R4 ms quantiles ordered and positive",
        0 < row["ms_p25"] <= row["ms_median"] <= row["ms_p75"],
        "0 < p25 <= median <= p75",
        (row["ms_p25"], row["ms_median"], row["ms_p75"]),
    )
    r.check(
        "R4 stable flag = p75 <= 1.5 p25",
        row["stable"] == (row["ms_p75"] <= 1.5 * row["ms_p25"]),
        row["ms_p75"] <= 1.5 * row["ms_p25"],
        row["stable"],
    )
    if CUDA:
        r.check(
            "R4 make_loss ran warmup + recorded + warm loop + repeats",
            calls[0] >= 2 + 1 + 5,
            ">= 8",
            calls[0],
        )
    else:
        r.equal("R4 make_loss calls = warmup + 1 recorded + repeats", calls[0], 2 + 1 + 5)
    fixed = torch.randn(1024, 1024, device=DEV)
    measure_variant(
        lambda: (fixed * weight).sum() + weight.square().sum(),
        [weight],
        DEV,
        warmup=2,
        repeats=3,
        warm_ms=50.0,
    )
    r.close(
        "R4 on return .grad holds exactly one pass (no accumulation)",
        weight.grad,
        fixed.sum(0) + 2 * weight.detach(),
        rtol=1e-5,
        atol=1e-4,
    )
    if not CUDA:
        r.check(
            "R4 off CUDA: after_forward/peak are None on CPU",
            row["after_forward_bytes"] is None and row["peak_delta_bytes"] is None,
            "None, None",
            (row["after_forward_bytes"], row["peak_delta_bytes"]),
        )
        return
    mib = 1 << 20
    r.check(
        "R4 after_forward_bytes = bytes held after forward (payload, not the 64 MiB held tensor)",
        payload_floats * 4 <= row["after_forward_bytes"] < payload_floats * 4 + mib,
        "[4 MiB, 5 MiB)",
        row["after_forward_bytes"],
    )
    r.check(
        "R4 peak_delta_bytes covers the 32 MiB transient but not earlier/held memory",
        temp_floats * 4 <= row["peak_delta_bytes"] < temp_floats * 4 + 16 * mib,
        "[32 MiB, 48 MiB)",
        row["peak_delta_bytes"],
    )
    del held


# ---------------------------------------------------------------- R variants
def stack(variant, width, inter, rank, layers, seed, p=0.0):
    torch.manual_seed(seed)
    return nn.Sequential(*[MLPBlock(width, inter, rank, variant, p) for _ in range(layers)]).to(DEV)


def r_variants(r):
    width, inter, rank, layers, tokens = 16, 48, 4, 2, 10
    x0 = torch.randn(tokens, width, device=DEV)
    results = {}
    for variant in ("eager", "eager_fa", "fn_fa", "fused", "fn_drop", "checkpoint"):
        model = stack(variant, width, inter, rank, layers, seed=51)
        x = x0.clone().requires_grad_(True)
        with ByteHook(model.parameters()) as hook:
            out = model(x)
        out.square().sum().backward()
        results[variant] = (
            out.detach(),
            x.grad.clone(),
            {n: q.grad.clone() for n, q in model.named_parameters() if q.grad is not None},
        )
        expected = (
            predicted_saved_floats(variant, width, inter, rank, mask_bytes=4) * 4 * tokens * layers
        )
        r.equal(f"R variant {variant}: saved bytes = hand-derived count", hook.bytes, int(expected))
    out0, gx0, gp0 = results["eager"]
    for variant in ("eager_fa", "fn_fa", "fused", "fn_drop", "checkpoint"):
        out, gx, gp = results[variant]
        worst = max(
            [float((out - out0).abs().max()), float((gx - gx0).abs().max())]
            + [float((g - gp0[n]).abs().max()) for n, g in gp.items()]
        )
        r.check(
            f"R variant {variant}: output and grads match eager (p=0)",
            worst <= 1e-5,
            "<= 1e-5",
            worst,
        )
    if not CUDA:
        return
    width, inter, rank, layers, tokens = 1024, 3072, 16, 4, 2048
    x0 = torch.randn(tokens, width, device=DEV)
    held, peak = {}, {}
    for variant in ("eager", "fused", "checkpoint"):
        model = stack(variant, width, inter, rank, layers, seed=52)
        x = x0.clone().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEV)
        start = torch.cuda.memory_allocated(DEV)
        out = model(x)
        torch.cuda.synchronize()
        held[variant] = torch.cuda.memory_allocated(DEV) - start
        out.sum().backward()
        torch.cuda.synchronize()
        peak[variant] = torch.cuda.max_memory_allocated(DEV) - start
        del out, model, x
    saving = layers * tokens * inter * 4
    r.check(
        "R CUDA 4x(1024,3072) N=2048: fused holds L N f 4 fewer bytes after forward (5% tol)",
        abs((held["eager"] - held["fused"]) - saving) <= 0.05 * saving,
        saving,
        held["eager"] - held["fused"],
    )
    r.check(
        "R CUDA: fused fw+bw peak below eager peak",
        peak["fused"] < peak["eager"],
        f"< {peak['eager']}",
        peak["fused"],
    )
    r.check(
        "R CUDA: checkpoint peak below fused peak",
        peak["checkpoint"] < peak["fused"],
        f"< {peak['fused']}",
        peak["checkpoint"],
    )


TESTS = [r1_lora_fa, r2_fused_mlp, r3_seed_dropout, r4_measure, r_variants]

if __name__ == "__main__":
    report = Report("R")
    for test in TESTS:
        report.group(test.__name__, test)
    raise SystemExit(report.summary())
