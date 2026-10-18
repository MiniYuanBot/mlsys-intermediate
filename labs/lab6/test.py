"""Stage A/B/C checks. Oracles are written here independently of src/ (dense formulas,
exhaustive search, hand-built hooks); a failing check prints FAIL and the run goes on."""

import argparse
import copy
import itertools
import math

import torch
from checks import Report
from model import Block, Model
from src.train_student import (
    CpuOffload,
    LoRALinear,
    SavedInventory,
    adapter_state,
    calibrate_blocks,
    inject_lora,
    load_adapters,
    measure_policy,
    parameter_bytes,
    plan_keep,
    plan_modes,
    run_blocks,
    run_policy,
    train_accumulated,
)
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
parser.add_argument("--part", choices=["all", "A", "B", "C"], default="all")
args = parser.parse_args()
DEV = torch.device(args.device)
CUDA = DEV.type == "cuda"
torch.set_num_threads(2)


# ---------------------------------------------------------------- oracle helpers
def lora_dense(x, layer):
    """Independent LoRA formula: x W^T + bias + s (x A^T) B^T."""
    out = x @ layer.weight.T + layer.scale * (x @ layer.a.T) @ layer.b.T
    return out if layer.bias is None else out + layer.bias


class ByteHook:
    """Test-side saved-tensor counter (does not use SavedInventory)."""

    def __init__(self, parameters=()):
        self.skip = {p.untyped_storage().data_ptr() for p in parameters}
        self.storages, self.calls, self.call_bytes = {}, 0, 0

    def pack(self, t):
        self.calls += 1
        ptr = t.untyped_storage().data_ptr()
        if ptr not in self.skip:
            self.storages[ptr] = t.untyped_storage().nbytes()
            self.call_bytes += t.numel() * t.element_size()
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


def count_calls(modules):
    counts = [0] * len(modules)
    handles = []
    for i, module in enumerate(modules):

        def hook(_module, _inputs, i=i):
            counts[i] += 1

        handles.append(module.register_forward_pre_hook(hook))
    return counts, handles


def rng_snapshot():
    state = [torch.get_rng_state()]
    if CUDA:
        state.append(torch.cuda.get_rng_state(DEV))
    return state


def rng_same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def offload_owner(parameters, **kw):
    """Copy path on whatever device the test runs (CPU tensors copy to fresh host memory)."""
    return CpuOffload(parameters, offload_devices=(DEV.type,), **kw)


# ---------------------------------------------------------------- A
def a1_forward_and_gradients(r):
    torch.manual_seed(1)
    layer = LoRALinear(7, 5, rank=2, bias=True).to(DEV)
    with torch.no_grad():
        layer.bias.normal_()
    x = torch.randn(3, 7, device=DEV, requires_grad=True)
    r.close(
        "A1 B=0: output equals the base x W^T + bias", layer(x), x @ layer.weight.T + layer.bias
    )
    layer(x).square().sum().backward()
    r.check(
        "A1 frozen W and bias get no .grad",
        layer.weight.grad is None and layer.bias.grad is None,
        "None, None",
        (layer.weight.grad is None, layer.bias.grad is None),
    )
    r.check(
        "A1 B=0: first grad of A is zero, of B nonzero",
        float(layer.a.grad.abs().sum()) == 0 and float(layer.b.grad.abs().sum()) > 0,
        "A=0, B>0",
        (float(layer.a.grad.abs().sum()), float(layer.b.grad.abs().sum())),
    )
    with torch.no_grad():
        layer.b.normal_()
    layer.zero_grad(set_to_none=True)
    x3 = torch.randn(2, 3, 7, device=DEV, requires_grad=True)
    out = layer(x3)
    r.equal("A1 keeps leading dims", tuple(out.shape), (2, 3, 5))
    r.close("A1 3-D output = dense LoRA formula (bias once)", out, lora_dense(x3, layer))
    g = torch.randn(2, 3, 5, device=DEV)
    (out * g).sum().backward()
    x2, g2 = x3.detach().reshape(6, 7), g.reshape(6, 5)
    s = layer.scale
    r.close(
        "A1 dX = G W + s G B A (input grad through frozen base)",
        x3.grad.reshape(6, 7),
        g2 @ layer.weight + s * g2 @ layer.b @ layer.a,
        rtol=1e-5,
        atol=1e-5,
    )
    r.close(
        "A1 dB = s G^T (X A^T)", layer.b.grad, s * g2.T @ (x2 @ layer.a.T), rtol=1e-5, atol=1e-5
    )
    r.close("A1 dA = s B^T G^T X", layer.a.grad, s * layer.b.T @ g2.T @ x2, rtol=1e-5, atol=1e-5)


def a2_a3_merge_restore(r):
    torch.manual_seed(2)
    layer = LoRALinear(7, 5, rank=2).to(DEV)
    r.raises("A2 merge in train mode raises ValueError", ValueError, layer.merge)
    r.check("A2 rejected merge leaves the layer unmerged", not layer.merged, False, layer.merged)
    layer.eval()
    with torch.no_grad():
        layer.b.normal_(std=10.0)  # large delta: W + d - d is not bitwise W
    x = torch.randn(4, 7, device=DEV)
    weight = layer.weight.detach().clone()
    merged_weight = weight + layer.scale * layer.b.detach() @ layer.a.detach()
    expected = x @ merged_weight.T
    for round_ in range(3):
        layer.merge()
        layer.merge()
        r.close(
            f"A2 round {round_}: double merge adds the delta once",
            layer(x),
            expected,
            rtol=1e-5,
            atol=1e-4,
        )
        r.close(
            f"A2 round {round_}: merged weight = W + s B A",
            layer.weight,
            merged_weight,
            rtol=1e-6,
            atol=1e-5,
        )
        layer.unmerge()
        layer.unmerge()
        r.check(
            f"A3 round {round_}: unmerge restores W bitwise",
            torch.equal(weight, layer.weight),
            "bitwise equal",
            float((weight - layer.weight).abs().max()),
        )
    r.check(
        "A3 backup released after unmerge",
        layer._base_backup is None,
        None,
        type(layer._base_backup),
    )
    r.close(
        "A3 unmerged forward uses the LoRA branch again",
        layer(x),
        lora_dense(x, layer),
        rtol=1e-5,
        atol=1e-4,
    )


def a4_injection(r):
    torch.manual_seed(3)

    def fresh():
        model = nn.Sequential(
            nn.Linear(7, 9), nn.Sequential(nn.ReLU(), nn.Linear(9, 5, bias=False)), nn.Linear(5, 5)
        )
        model = model.to(device=DEV, dtype=torch.float64)
        model[0].eval()
        return model

    model = fresh()
    x = torch.randn(4, 7, device=DEV, dtype=torch.float64)
    expected = model(x).detach()
    old = model[0]
    for bad in (["0", "missing"], ["0", "1.0"], ["0", "0"], []):
        r.raises(f"A4 rejects targets {bad}", ValueError, lambda bad=bad: inject_lora(model, bad))
    r.check(
        "A4 rejection leaves the model untouched",
        model[0] is old and all(p.requires_grad for p in model.parameters()),
        "same module, all trainable",
        (model[0] is old, [p.requires_grad for p in model.parameters()]),
    )
    r.equal("A4 returns the replaced names", inject_lora(model, ["0", "1.1"], 2), ["0", "1.1"])
    r.check(
        "A4 replaced modules are LoRALinear",
        isinstance(model[0], LoRALinear) and isinstance(model[1][1], LoRALinear),
        True,
        (type(model[0]).__name__, type(model[1][1]).__name__),
    )
    r.close("A4 B=0 model computes the original function", model(x), expected, rtol=0, atol=1e-12)
    r.equal(
        "A4 only adapters train (untouched Linear frozen too)",
        {n for n, p in model.named_parameters() if p.requires_grad},
        {"0.a", "0.b", "1.1.a", "1.1.b"},
    )
    r.check(
        "A4 dtype/device preserved (float64)",
        model[0].a.dtype == torch.float64
        and model[1][1].weight.dtype == torch.float64
        and model[0].a.device.type == DEV.type,
        "float64 on " + DEV.type,
        (model[0].a.dtype, model[1][1].weight.dtype, model[0].a.device),
    )
    r.check(
        "A4 train/eval flag preserved",
        (not model[0].training) and model[1][1].training,
        "(False, True)",
        (model[0].training, model[1][1].training),
    )
    r.check(
        "A4 bias kept exactly, absent bias stays absent",
        model[0].bias is not None and model[1][1].bias is None and not model[0].bias.requires_grad,
        True,
        (model[0].bias is None, model[1][1].bias is None),
    )


def a5_a6_state_transactions(r):
    torch.manual_seed(4)
    model = Model(width=8, layers=2).to(DEV)
    state = adapter_state(model)
    r.equal(
        "A5 one entry per LoRALinear",
        sorted(state),
        ["blocks.0.down", "blocks.0.up", "blocks.1.down", "blocks.1.up"],
    )
    entry = state["blocks.0.up"]
    r.check(
        "A5 entries are CPU tensors + scale only, no base weight",
        set(entry) == {"a", "b", "scale"} and entry["a"].device.type == "cpu",
        "{a,b,scale} on cpu",
        (sorted(entry), entry["a"].device),
    )
    snapshot = copy.deepcopy(state)
    with torch.no_grad():
        model.blocks[0].up.a.add_(1)
    r.check(
        "A5 saved tensors do not alias the model",
        torch.equal(state["blocks.0.up"]["a"], snapshot["blocks.0.up"]["a"]),
        "unchanged after model update",
        "changed",
    )
    load_adapters(model, state)
    r.close(
        "A6 load restores every adapter",
        adapter_state(model)["blocks.0.up"]["a"],
        state["blocks.0.up"]["a"],
    )

    def unchanged(label):
        r.close(
            f"A6 {label}: earlier modules unchanged",
            adapter_state(model)["blocks.0.up"]["a"],
            state["blocks.0.up"]["a"],
            rtol=0,
            atol=0,
        )

    bad = copy.deepcopy(state)
    bad["blocks.0.up"]["a"].add_(2)
    bad["blocks.1.down"]["b"] = torch.zeros(1)
    r.raises(
        "A6 wrong shape in the last entry raises", ValueError, lambda: load_adapters(model, bad)
    )
    unchanged("bad shape")
    bad = copy.deepcopy(state)
    bad["blocks.0.up"]["a"].add_(2)
    bad["blocks.1.down"]["a"].fill_(float("nan"))
    r.raises("A6 non-finite value raises", ValueError, lambda: load_adapters(model, bad))
    unchanged("NaN")
    bad = copy.deepcopy(state)
    bad["blocks.0.up"]["a"].add_(2)
    source = bad["blocks.1.down"]["b"]
    bad["blocks.1.down"]["b"] = torch.full(source.shape, 1e100, dtype=torch.float64)
    r.raises(
        "A6 finite float64 that overflows float32 raises",
        ValueError,
        lambda: load_adapters(model, bad),
    )
    unchanged("overflow")
    bad = copy.deepcopy(state)
    bad["blocks.0.up"]["scale"] = 3.0
    r.raises("A6 wrong scale raises", ValueError, lambda: load_adapters(model, bad))
    bad = copy.deepcopy(state)
    del bad["blocks.1.up"]
    r.raises("A6 missing module raises", ValueError, lambda: load_adapters(model, bad))


def a7_parameter_bytes(r):
    layer = LoRALinear(7, 5, rank=2, bias=True)
    r.equal(
        "A7 LoRALinear(7,5,r=2,bias)",
        parameter_bytes(layer),
        {"trainable": (14 + 10) * 4, "frozen": (35 + 5) * 4},
    )
    tied = nn.Module()
    tied.first = nn.Linear(3, 4, bias=False)
    tied.second = nn.Linear(3, 4, bias=False)
    tied.second.weight = tied.first.weight
    r.equal(
        "A7 tied Parameter object counted once",
        parameter_bytes(tied),
        {"trainable": 48, "frozen": 0},
    )
    w = torch.randn(4, 3)
    alias = nn.Module()
    alias.p = nn.Parameter(w)  # trainable, 48-byte storage
    alias.q = nn.Parameter(w, requires_grad=False)  # distinct object, same storage
    alias.v = nn.Parameter(w.view(-1)[:6], requires_grad=False)  # view of the same storage
    alias.r = nn.Parameter(torch.randn(5), requires_grad=False)
    r.equal(
        "A7 distinct Parameters sharing one storage: counted once, trainable wins",
        parameter_bytes(alias),
        {"trainable": 48, "frozen": 20},
    )
    order = nn.Module()
    order.q = nn.Parameter(w, requires_grad=False)
    order.p = nn.Parameter(w)
    r.equal(
        "A7 shared storage: frozen alias seen first, still trainable",
        parameter_bytes(order),
        {"trainable": 48, "frozen": 0},
    )
    view_only = nn.Module()
    view_only.v = nn.Parameter(torch.randn(4, 3).view(-1)[:6])
    r.equal(
        "A7 a view Parameter counts its whole 48-byte storage",
        parameter_bytes(view_only),
        {"trainable": 48, "frozen": 0},
    )


# ---------------------------------------------------------------- B
def b1_checkpoint(r):
    torch.manual_seed(19)
    base = Model(width=16, layers=3).to(DEV)
    x = torch.randn(2, 5, 16, device=DEV)
    target = torch.randn_like(x)
    expected, saved = None, {}
    for keep in ([True] * 3, [False] * 3, [True, False, True]):
        model = copy.deepcopy(base)
        counts, handles = count_calls(list(model.blocks))
        torch.manual_seed(27)
        with ByteHook(model.parameters()) as hook:
            out = model(x, keep)
        loss = (out - target).square().mean()
        loss.backward()
        for handle in handles:
            handle.remove()
        saved[tuple(keep)] = hook.bytes
        r.equal(
            f"B1 keep={keep}: forward calls per block (2 = recomputed)",
            counts,
            [1 if k else 2 for k in keep],
        )
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        if expected is None:
            expected = (loss.detach(), grads)
            continue
        r.close(f"B1 keep={keep}: same loss under dropout", loss.detach(), expected[0])
        worst = max(float((g - expected[1][n]).abs().max()) for n, g in grads.items())
        r.check(
            f"B1 keep={keep}: every trainable grad matches retain", worst <= 1e-5, "<= 1e-5", worst
        )
    boundary = x.numel() * 4
    r.equal("B1 all-checkpoint saves only the 3 block inputs", saved[(False,) * 3], 3 * boundary)
    r.check(
        "B1 retain saves more than mixed, mixed more than all-checkpoint",
        saved[(True,) * 3] > saved[(True, False, True)] > saved[(False,) * 3],
        "strictly decreasing",
        [saved[(True,) * 3], saved[(True, False, True)], saved[(False,) * 3]],
    )
    r.raises("B1 short keep list raises", ValueError, lambda: run_blocks(base.blocks, x, [True]))
    r.raises("B1 non-bool keep raises", ValueError, lambda: run_blocks(base.blocks, x, [1, 0, 1]))


def b2_knapsack(r):
    sizes, costs = [2, 5, 3, 7], [1, 9, 4, 6]
    bad = []
    for budget in range(18):
        chosen = plan_keep(sizes, costs, budget)
        used = sum(s for s, k in zip(sizes, chosen, strict=True) if k)
        gain = sum(c for c, k in zip(costs, chosen, strict=True) if k)
        optimal = max(
            sum(c * k for c, k in zip(costs, bits, strict=True))
            for bits in itertools.product([0, 1], repeat=4)
            if sum(s * k for s, k in zip(sizes, bits, strict=True)) <= budget
        )
        if used > budget or gain != optimal or len(chosen) != 4:
            bad.append((budget, chosen, gain, optimal))
    r.check(
        "B2 optimal for all 18 budgets (exhaustive 16-subset oracle)",
        not bad,
        "no mismatch",
        bad[:3],
    )
    r.equal(
        "B2 greedy counterexample [10,20,30]/[60,100,120]/50",
        plan_keep([10, 20, 30], [60, 100, 120], 50),
        [False, True, True],
    )
    r.equal("B2 empty input", plan_keep([], [], 0), [])
    r.raises("B2 zero size raises", ValueError, lambda: plan_keep([0], [1], 2))
    r.raises("B2 negative budget raises", ValueError, lambda: plan_keep([1], [1], -1))
    r.raises("B2 NaN cost raises", ValueError, lambda: plan_keep([1], [float("nan")], 1))


def b3_inventory(r):
    p = nn.Parameter(torch.ones(8, device=DEV))
    x = torch.ones(16, device=DEV, requires_grad=True)
    y = x * 2  # non-leaf with a grad_fn
    inventory = SavedInventory([p])
    for value in (p, p[:3], x, x[4:8], x[:8], y[1:5]):
        packed = inventory.pack(value)
        r.close(
            f"B3 unpack(pack(t)) == t for shape {tuple(value.shape)}",
            inventory.unpack(packed),
            value,
        )
        r.check(
            f"B3 packed object is detached (shape {tuple(value.shape)})",
            packed.grad_fn is None and not packed.requires_grad,
            "no grad_fn",
            (packed.grad_fn, packed.requires_grad),
        )
    r.equal("B3 every pack call counted", inventory.saves, 6)
    r.equal(
        "B3 views share storage; parameter views excluded; full storage bytes",
        inventory.bytes,
        x.untyped_storage().nbytes() + y.untyped_storage().nbytes(),
    )


def b4_multiple_choice(r):
    options = [
        {"retain": (5, 1), "checkpoint": (1, 4), "offload": (0, 6)},
        {"retain": (6, 2), "checkpoint": (2, 3), "offload": (0, 8)},
        {"retain": (4, 1), "checkpoint": (1, 5)},
    ]
    bad = []
    for budget in range(0, 16):
        feasible = [
            (sum(options[i][m][1] for i, m in enumerate(path)), path)
            for path in itertools.product(*[sorted(o) for o in options])
            if sum(options[i][m][0] for i, m in enumerate(path)) <= budget
        ]
        if not feasible:
            r.raises(
                f"B4 budget {budget} infeasible raises",
                ValueError,
                lambda b=budget: plan_modes(options, b),
            )
            continue
        modes, used, cost = plan_modes(options, budget)
        ok = (
            len(modes) == 3
            and all(m in o for m, o in zip(modes, options, strict=True))
            and used == sum(o[m][0] for o, m in zip(options, modes, strict=True))
            and math.isclose(cost, sum(o[m][1] for o, m in zip(options, modes, strict=True)))
            and used <= budget
            and math.isclose(cost, min(item[0] for item in feasible))
        )
        if not ok:
            bad.append((budget, modes, used, cost, min(item[0] for item in feasible)))
    r.check(
        "B4 consistent and optimal for budgets 1..15 (exhaustive oracle)",
        not bad,
        "no mismatch",
        bad[:3],
    )
    r.raises("B4 unknown mode raises", ValueError, lambda: plan_modes([{"swap": (1, 1)}], 5))
    r.raises("B4 empty block options raise", ValueError, lambda: plan_modes([{}], 5))


class Widen(nn.Module):
    """Width 8 -> 16, so a calibration that forgets to advance its input fails loudly."""

    def __init__(self):
        super().__init__()
        self.proj = LoRALinear(8, 16, rank=2)

    def forward(self, x):
        return self.proj(x)


def b5_calibration(r):
    torch.manual_seed(5)
    blocks = nn.ModuleList([Widen(), Block(16, dropout=0)]).to(DEV)
    sample = torch.randn(2, 3, 8, device=DEV)
    rows = calibrate_blocks(blocks, sample, repeats=2)
    r.equal("B5 one row per block", len(rows), 2)
    r.check(
        "B5 grads cleared on return",
        all(p.grad is None for p in blocks.parameters()),
        "all None",
        [n for n, p in blocks.named_parameters() if p.grad is not None],
    )
    second_input = blocks[0](sample).detach()
    for i, (block, value) in enumerate(((blocks[0], sample), (blocks[1], second_input))):
        for mode in ("retain", "checkpoint"):
            v = value.detach().requires_grad_(True)
            with ByteHook(block.parameters()) as hook:
                out = (
                    block(v)
                    if mode == "retain"
                    else torch.utils.checkpoint.checkpoint(block, v, use_reentrant=False)
                )
            out.sum().backward()
            r.equal(
                f"B5 block {i} {mode}: saved_bytes = independent hook count",
                rows[i][mode]["saved_bytes"],
                hook.bytes,
            )
            r.check(
                f"B5 block {i} {mode}: ms > 0", rows[i][mode]["ms"] > 0, "> 0", rows[i][mode]["ms"]
            )
    r.raises("B5 repeats=0 raises", ValueError, lambda: calibrate_blocks(blocks, sample, repeats=0))


# ---------------------------------------------------------------- C
def c1_c2_offload_hooks(r):
    torch.manual_seed(6)
    p = nn.Parameter(torch.ones(8, device=DEV))
    owner = offload_owner([p])
    tensor = torch.randn(4, 5, device=DEV)
    snapshot = tensor.clone()
    payload = owner.pack(tensor)
    r.equal("C1 one copy, payload bytes counted", (owner.d2h_copies, owner.packed_bytes), (1, 80))
    host = payload[2]
    r.check(
        "C1 payload is host memory that shares nothing with the input",
        host.device.type == "cpu"
        and host.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr(),
        "fresh cpu storage",
        (host.device, host.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()),
    )
    tensor.add_(100)
    first = owner.unpack(payload)
    r.close("C2 unpack returns the value at pack time", first, snapshot)
    r.check(
        "C2 restored to the original device",
        first.device == tensor.device,
        tensor.device,
        first.device,
    )
    first.add_(5)
    second = owner.unpack(payload)
    r.close("C2 repeated unpack still valid after the first result was modified", second, snapshot)
    r.equal(
        "C2 counters after 2 unpacks (h2d, restores)", (owner.h2d_copies, owner.restores), (2, 2)
    )
    for dtype, make in (
        (torch.float64, lambda: torch.randn(3, 3, dtype=torch.float64, device=DEV)),
        (torch.bool, lambda: torch.rand(64, device=DEV) > 0.5),
        (torch.int64, lambda: torch.arange(12, device=DEV)),
    ):
        value = make()
        back = owner.unpack(owner.pack(value))
        r.check(
            f"C2 round trip keeps dtype {dtype} and device",
            back.dtype == dtype and back.device == value.device and torch.equal(back, value),
            (dtype, value.device),
            (back.dtype, back.device),
        )
    before = (owner.d2h_copies, owner.packed_bytes)
    for label, value in (("parameter", p), ("parameter view", p[2:5])):
        payload = owner.pack(value)
        r.check(
            f"C1 {label} is never copied",
            payload[0] is False
            and payload[2].untyped_storage().data_ptr() == p.untyped_storage().data_ptr(),
            "identity payload",
            payload[0],
        )
    r.equal(
        "C1 parameter packs do not move counters", (owner.d2h_copies, owner.packed_bytes), before
    )
    small = offload_owner((), min_bytes=1024)
    payload = small.pack(torch.randn(16, device=DEV))
    r.check(
        "C1 tensors below min_bytes stay in place",
        payload[0] is False and small.d2h_copies == 0,
        "identity",
        (payload[0], small.d2h_copies),
    )
    if not CUDA:
        default = CpuOffload(())
        payload = default.pack(torch.randn(16))
        r.check(
            "C1 default owner leaves CPU tensors in place",
            payload[0] is False and default.d2h_copies == 0,
            "identity",
            (payload[0], default.d2h_copies),
        )
        return
    torch.manual_seed(7)
    blocks = nn.ModuleList([Block(1024, dropout=0), Block(1024, dropout=0)]).to(DEV)
    x = torch.randn(1, 2048, 1024, device=DEV, requires_grad=True)
    held = {}
    for mode in ("retain", "offload"):
        owner = CpuOffload(blocks.parameters())
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated(DEV)
        out = run_policy(blocks, x, [mode] * 2, owner)
        torch.cuda.synchronize()
        held[mode] = torch.cuda.memory_allocated(DEV) - start
        out.sum().backward()
        del out
    r.check(
        "C1 CUDA: offload frees device memory after forward (< 60% of retain)",
        held["offload"] < 0.6 * held["retain"],
        f"< {0.6 * held['retain']:.0f}",
        held["offload"],
    )


def c3_policy(r):
    torch.manual_seed(31)
    base = nn.ModuleList([Block(8), Block(8), Block(8)]).to(DEV)
    x = torch.randn(2, 4, 8, device=DEV)
    target = torch.randn_like(x)
    expected = None
    for modes in (
        ["retain"] * 3,
        ["checkpoint"] * 3,
        ["offload"] * 3,
        ["retain", "offload", "checkpoint"],
    ):
        blocks = copy.deepcopy(base)
        value = x.detach().clone().requires_grad_(True)
        owner = offload_owner(blocks.parameters())
        counts, handles = count_calls(list(blocks))
        torch.manual_seed(73)
        loss = (run_policy(blocks, value, modes, owner) - target).square().mean()
        loss.backward()
        for handle in handles:
            handle.remove()
        r.equal(
            f"C3 {modes}: checkpointed blocks run twice",
            counts,
            [2 if m == "checkpoint" else 1 for m in modes],
        )
        if "offload" in modes:
            r.check(
                f"C3 {modes}: offload blocks copied saved tensors and restored each once",
                owner.d2h_copies > 0 and owner.d2h_copies == owner.h2d_copies,
                "d2h == h2d > 0",
                (owner.d2h_copies, owner.h2d_copies),
            )
        else:
            r.equal(f"C3 {modes}: no offload copies", owner.d2h_copies, 0)
        gradients = [value.grad] + [q.grad for q in blocks.parameters() if q.requires_grad]
        if expected is None:
            expected = [g.clone() for g in gradients]
            continue
        worst = max(float((a - b).abs().max()) for a, b in zip(gradients, expected, strict=True))
        r.check(
            f"C3 {modes}: input and all trainable grads match retain (dropout on)",
            worst <= 1e-5,
            "<= 1e-5",
            worst,
        )
    blocks = copy.deepcopy(base)
    for label, modes, owner in (
        ("missing offload owner", ["retain", "offload", "retain"], None),
        ("unknown mode", ["retain", "swap", "retain"], offload_owner(())),
        ("wrong length", ["retain", "retain"], None),
    ):
        counts, handles = count_calls(list(blocks))
        state = rng_snapshot()
        r.raises(
            f"C3 {label} raises ValueError",
            ValueError,
            lambda m=modes, o=owner: run_policy(blocks, x, m, o),
        )
        for handle in handles:
            handle.remove()
        r.check(
            f"C3 {label}: rejected before any block ran or RNG was used",
            counts == [0, 0, 0] and rng_same(state, rng_snapshot()),
            "[0, 0, 0], same RNG",
            counts,
        )


def full_batch_oracle(blocks, x, y, lr, clip):
    """Plain eager full-batch MSE mean, manual clip and SGD (no src/ calls)."""
    ref = copy.deepcopy(blocks)
    out = x
    for block in ref:
        out = block(out)
    loss = (out - y).square().mean()
    loss.backward()
    params = [p for p in ref.parameters() if p.requires_grad]
    norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in params))
    factor = 1.0 if clip is None else min(1.0, clip / (norm + 1e-6))
    with torch.no_grad():
        for p in params:
            p -= lr * factor * p.grad
    return ref, loss.item(), norm


def c4_accumulation(r):
    torch.manual_seed(41)
    base = nn.ModuleList([Block(8, dropout=0), Block(8, dropout=0)]).to(DEV)
    with torch.no_grad():
        for block in base:
            block.up.b.normal_()
            block.down.b.normal_()
    x = torch.randn(5, 3, 8, device=DEV)
    y = torch.randn_like(x)
    # lr=1 with clip=1e-3: every clipped update has norm ~1e-3, so per-microbatch
    # clipping (a different direction) differs by ~1e-3 >> the 1e-5 tolerance.
    lr, clip = 1.0, 1e-3
    ref, ref_loss, norm = full_batch_oracle(base, x, y, lr, clip)
    r.check(
        "C4 setup: full-batch grad norm is far above max_grad_norm",
        norm > 10 * clip,
        f"> {10 * clip}",
        norm,
    )
    for label, splits, modes in (
        ("2+3 checkpoint/offload", [(0, 2), (2, 5)], ["checkpoint", "offload"]),
        ("1+1+3 retain", [(0, 1), (1, 2), (2, 5)], ["retain", "retain"]),
    ):
        blocks = copy.deepcopy(base)
        optimizer = torch.optim.SGD(blocks.parameters(), lr=lr)
        batches = [(x[a:b], y[a:b]) for a, b in splits]
        loss = train_accumulated(
            blocks,
            optimizer,
            batches,
            modes,
            offload_owner(blocks.parameters()),
            max_grad_norm=clip,
        )
        r.check(
            f"C4 {label}: returned loss = full-batch MSE before the step",
            math.isclose(loss, ref_loss, rel_tol=1e-5),
            ref_loss,
            loss,
        )
        worst = max(
            float((p.detach() - q.detach()).abs().max())
            for p, q in zip(blocks.parameters(), ref.parameters(), strict=True)
        )
        r.check(
            f"C4 {label}: update = full batch, clipped once after accumulation",
            worst <= 1e-5,
            "<= 1e-5",
            worst,
        )
    lr = 0.05
    ref_free, _, _ = full_batch_oracle(base, x, y, lr, None)
    blocks = copy.deepcopy(base)
    optimizer = torch.optim.SGD(blocks.parameters(), lr=lr)
    train_accumulated(blocks, optimizer, [(x[:1], y[:1]), (x[1:], y[1:])], ["retain", "retain"])
    worst = max(
        float((p.detach() - q.detach()).abs().max())
        for p, q in zip(blocks.parameters(), ref_free.parameters(), strict=True)
    )
    r.check(
        "C4 1+4 without clipping = unclipped full-batch update", worst <= 1e-5, "<= 1e-5", worst
    )
    blocks = copy.deepcopy(base)
    optimizer = torch.optim.SGD(blocks.parameters(), lr=lr)
    sentinel = [torch.full_like(p, 7.0) for p in blocks.parameters()]
    for p, s in zip(blocks.parameters(), sentinel, strict=True):
        p.grad = s.clone()
    before = [p.detach().clone() for p in blocks.parameters()]
    r.raises(
        "C4 offload mode without owner raises",
        ValueError,
        lambda: train_accumulated(blocks, optimizer, [(x, y)], ["offload", "retain"], None),
    )
    r.check(
        "C4 rejected call touched neither parameters nor grads",
        all(torch.equal(p, q) for p, q in zip(blocks.parameters(), before, strict=True))
        and all(torch.equal(p.grad, s) for p, s in zip(blocks.parameters(), sentinel, strict=True)),
        "unchanged",
        "modified",
    )
    r.raises(
        "C4 empty microbatch list raises",
        ValueError,
        lambda: train_accumulated(blocks, optimizer, [], ["retain"] * 2),
    )
    r.raises(
        "C4 shape mismatch raises",
        ValueError,
        lambda: train_accumulated(blocks, optimizer, [(x, y[:2])], ["retain"] * 2),
    )


def c5_measure(r):
    torch.manual_seed(8)
    blocks = nn.ModuleList([Block(8, dropout=0), Block(8, dropout=0)]).to(DEV)
    x = torch.randn(3, 3, 8, device=DEV)
    batches = [(x[:2], torch.zeros_like(x[:2])), (x[2:], torch.zeros_like(x[2:]))]
    with ByteHook(blocks.parameters()) as hook:
        for xb, _ in batches:
            out = xb
            for block in blocks:
                out = block(out)
    del out
    before = [p.detach().clone() for p in blocks.parameters()]
    row = measure_policy(
        blocks, batches, ["offload", "retain"], repeats=3, offload_devices=(DEV.type,)
    )
    r.check("C5 step_ms > 0", row["step_ms"] > 0, "> 0", row["step_ms"])
    r.equal("C5 one loss per measured step", len(row["losses"]), 3)
    r.check(
        "C5 lr=0: identical losses and parameters",
        len(set(row["losses"])) == 1
        and all(torch.equal(p, q) for p, q in zip(blocks.parameters(), before, strict=True)),
        "unchanged",
        row["losses"],
    )
    first_block_bytes = None
    with ByteHook(blocks[0].parameters()) as first:
        for xb, _ in batches:
            blocks[0](xb)
    first_block_bytes = first.call_bytes
    r.equal(
        "C5 d2h bytes per step = bytes saved by the offloaded block in one step",
        row["d2h_bytes_per_step"],
        first_block_bytes,
    )
    r.check(
        "C5 counters: d2h copies = h2d copies per step",
        row["d2h_copies_per_step"] == row["h2d_copies_per_step"] > 0,
        "equal, > 0",
        (row["d2h_copies_per_step"], row["h2d_copies_per_step"]),
    )
    r.check("C5 hook sanity: the oracle saw saved tensors", hook.calls > 0, "> 0", hook.calls)
    r.check(
        "C5 allocator peaks are None off CUDA and numbers on CUDA",
        (row["peak_allocated_bytes"] is None) != CUDA,
        "None" if not CUDA else "int",
        row["peak_allocated_bytes"],
    )
    if CUDA:
        torch.manual_seed(9)
        big = nn.ModuleList([Block(1024, dropout=0), Block(1024, dropout=0)]).to(DEV)
        xb = torch.randn(1, 2048, 1024, device=DEV)
        peaks = {}
        for mode in ("retain", "checkpoint"):
            peaks[mode] = measure_policy(
                copy.deepcopy(big), [(xb, torch.zeros_like(xb))], [mode] * 2, repeats=1
            )["peak_allocated_bytes"]
        r.check(
            "C5 CUDA: checkpoint peak < retain peak at width 1024",
            peaks["checkpoint"] < peaks["retain"],
            f"< {peaks['retain']}",
            peaks["checkpoint"],
        )


PARTS = {
    "A": [
        a1_forward_and_gradients,
        a2_a3_merge_restore,
        a4_injection,
        a5_a6_state_transactions,
        a7_parameter_bytes,
    ],
    "B": [b1_checkpoint, b2_knapsack, b3_inventory, b4_multiple_choice, b5_calibration],
    "C": [c1_c2_offload_hooks, c3_policy, c4_accumulation, c5_measure],
}

if __name__ == "__main__":
    status = 0
    for part, tests in PARTS.items():
        if args.part not in (part, "all"):
            continue
        report = Report(part)
        for test in tests:
            report.group(test.__name__, test)
        status |= report.summary()
    raise SystemExit(status)
