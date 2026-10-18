"""LoRA lifecycle, activation planning and saved-tensor offload on a small LoRA MLP.

Stages A/B/C of lab 6. Stage R (custom autograd.Function) lives in fused_student.py.
This is a teaching system: no quantized base, no attention, no PEFT file format.
"""

import contextlib
import math
import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

MODES = ("retain", "checkpoint", "offload")


def storage_key(tensor):
    # Provided: views share storage; storage bytes may exceed view tensor bytes.
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=8, bias=False):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank")
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None
        self.a = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))
        self.b = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank
        self.merged = False
        self.register_buffer("_base_backup", None, persistent=False)

    def forward(self, x):
        # Input  x[..., in]; output [..., out] = x W^T + bias + scale * (x A^T) B^T.
        # Invariants: gradients still flow to x through the frozen base (no no_grad);
        # bias is added exactly once; a merged layer skips the low-rank branch.
        # BEGIN TODO A1
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.merged:
            return base
        update = torch.nn.functional.linear(torch.nn.functional.linear(x, self.a), self.b)
        return base + self.scale * update
        # END TODO A1

    @torch.no_grad()
    def merge(self):
        # Eval-only (ValueError in train mode); idempotent. Keep an exact copy of the
        # base so unmerge restores it bitwise instead of subtracting the delta back.
        # BEGIN TODO A2
        if self.training:
            raise ValueError("merge requires eval mode")
        if self.merged:
            return
        self._base_backup = self.weight.detach().clone()
        self.weight.add_(self.b @ self.a, alpha=self.scale)
        self.merged = True
        # END TODO A2

    @torch.no_grad()
    def unmerge(self):
        # Restore the bitwise base and release the backup; calling it twice is a no-op.
        # BEGIN TODO A3
        if not self.merged:
            return
        if self._base_backup is None:
            raise RuntimeError("missing merge backup")
        self.weight.copy_(self._base_backup)
        self._base_backup = None
        self.merged = False
        # END TODO A3


def inject_lora(model, targets, rank=4, alpha=8):
    """Replace exact named nn.Linear targets, freeze the original model, return names.

    Validate every target before any mutation (ValueError, model untouched). Keep
    weight/bias values, device, dtype and the train/eval flag; only A/B train.
    """
    # BEGIN TODO A4
    modules = dict(model.named_modules())
    names = list(targets)
    if rank <= 0 or not names or len(names) != len(set(names)):
        raise ValueError("unique nonempty targets/rank")
    for name in names:
        if not name or name not in modules or not isinstance(modules[name], nn.Linear):
            raise ValueError(f"invalid linear target {name}")
    replacements = []
    for name in names:
        old = modules[name]
        new = LoRALinear(old.in_features, old.out_features, rank, alpha, old.bias is not None)
        new.to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        new.train(old.training)
        parent_name, _, child = name.rpartition(".")
        replacements.append((modules[parent_name], child, new))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parent, child, new in replacements:
        setattr(parent, child, new)
    return names
    # END TODO A4


def adapter_state(model):
    """{module name: {"a", "b", "scale"}} with cloned CPU tensors; no base weights.

    Merged layers are rejected; later changes to the model must not alter the result.
    """
    # BEGIN TODO A5
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if module.merged:
                raise ValueError("unmerge before saving adapters")
            state[name] = {
                "a": module.a.detach().cpu().clone(),
                "b": module.b.detach().cpu().clone(),
                "scale": module.scale,
            }
    return state
    # END TODO A5


@torch.no_grad()
def load_adapters(model, state):
    """Validate names/fields/shapes/finite values/scale, stage conversions, then copy.

    Conversion to the destination dtype may overflow (finite float64 -> inf float32);
    reject that before copying. Any failure leaves every module unchanged.
    """
    # BEGIN TODO A6
    modules = {
        name: layer for name, layer in model.named_modules() if isinstance(layer, LoRALinear)
    }
    if set(modules) != set(state):
        raise ValueError("adapter names")
    for name, layer in modules.items():
        entry = state[name]
        if layer.merged or set(entry) != {"a", "b", "scale"}:
            raise ValueError("adapter metadata/merged")
        if not math.isfinite(entry["scale"]) or entry["scale"] != layer.scale:
            raise ValueError("adapter scale")
        for key in ("a", "b"):
            value = entry[key]
            if not isinstance(value, torch.Tensor) or value.shape != getattr(layer, key).shape:
                raise ValueError("adapter shape")
            if not torch.isfinite(value).all():
                raise ValueError("adapter finite")
    staged = {}
    for name, layer in modules.items():
        staged[name] = {}
        for key in ("a", "b"):
            destination = getattr(layer, key)
            converted = state[name][key].to(device=destination.device, dtype=destination.dtype)
            if not torch.isfinite(converted).all():
                raise ValueError("adapter conversion overflow")
            staged[name][key] = converted
    for name, layer in modules.items():
        layer.a.copy_(staged[name]["a"])
        layer.b.copy_(staged[name]["b"])
    # END TODO A6


def parameter_bytes(model):
    """{"trainable": bytes, "frozen": bytes} over unique parameter *storages*.

    Distinct Parameter objects may alias one storage (tied weights built from .data,
    views); count each storage once, in full (untyped_storage().nbytes()). A storage
    is trainable if any parameter that aliases it requires grad. Not optimizer state,
    gradients, activations or allocator reserve.
    """
    # BEGIN TODO A7
    storages = {}
    for parameter in model.parameters():
        key = storage_key(parameter)
        nbytes, trainable = storages.get(key, (parameter.untyped_storage().nbytes(), False))
        storages[key] = (nbytes, trainable or parameter.requires_grad)
    result = {"trainable": 0, "frozen": 0}
    for nbytes, trainable in storages.values():
        result["trainable" if trainable else "frozen"] += nbytes
    return result
    # END TODO A7


def run_blocks(blocks, x, keep):
    """keep[i] True: plain call; False: non-reentrant checkpoint with RNG preserved.

    Validate the whole policy first (length and bool type) so a short list cannot
    silently drop the tail of the network.
    """
    # BEGIN TODO B1
    if len(blocks) != len(keep) or any(type(value) is not bool for value in keep):
        raise ValueError("keep policy")
    for block, retain in zip(blocks, keep, strict=True):
        if retain:
            x = block(x)
        else:
            x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
    return x
    # END TODO B1


def plan_keep(sizes, recompute_costs, budget):
    """0/1 knapsack: maximize the summed cost of blocks kept within budget units.

    sizes are positive ints, costs finite and >= 0, budget an int >= 0. Each block is
    taken at most once. Returns a bool list (True = keep activations).
    """
    # BEGIN TODO B2
    if len(sizes) != len(recompute_costs) or type(budget) is not int or budget < 0:
        raise ValueError("knapsack inputs")
    if any(type(s) is not int or s <= 0 for s in sizes):
        raise ValueError("positive sizes")
    if any(not math.isfinite(c) or c < 0 for c in recompute_costs):
        raise ValueError("costs")
    states = {0: (0.0, [])}
    for size, gain in zip(sizes, recompute_costs, strict=True):
        next_states = {}
        for used, (value, path) in states.items():
            for take in (False, True):
                new_size = used + size * take
                new_value = value + gain * take
                if new_size <= budget and (
                    new_size not in next_states or new_value > next_states[new_size][0]
                ):
                    next_states[new_size] = (new_value, path + [take])
        states = next_states
    return max(states.values(), key=lambda entry: entry[0])[1]
    # END TODO B2


class SavedInventory:
    def __init__(self, parameters=()):
        self.excluded = {storage_key(p) for p in parameters}
        self.storages = {}
        self.saves = 0

    def pack(self, tensor):
        # Count each non-parameter storage once (full storage bytes, keyed by
        # storage_key); parameter storages and their views are excluded. Count every
        # call in self.saves. Return a detached alias: the packed object must not
        # carry the autograd graph of the input (a reference cycle otherwise).
        # BEGIN TODO B3
        self.saves += 1
        key = storage_key(tensor)
        if key not in self.excluded:
            self.storages[key] = tensor.untyped_storage().nbytes()
        return tensor.detach()
        # END TODO B3

    @staticmethod
    def unpack(tensor):
        return tensor

    @property
    def bytes(self):
        return sum(self.storages.values())

    def context(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)


def plan_modes(options, budget):
    """Multiple-choice knapsack. options[i][mode] = (int retained units, measured ms).

    Choose exactly one mode per block, total units <= budget, minimal summed ms.
    Return (modes, used, cost); ValueError when infeasible or malformed. The additive
    model ignores lifetimes and workspaces: verify any plan with a measured step.
    """
    # BEGIN TODO B4
    if type(budget) is not int or budget < 0:
        raise ValueError("budget")
    states = {0: (0.0, [])}
    for choices in options:
        if not choices:
            raise ValueError("empty block options")
        next_states = {}
        for mode, (size, cost) in sorted(choices.items()):
            if mode not in MODES:
                raise ValueError("mode")
            if type(size) is not int or size < 0 or not math.isfinite(cost) or cost < 0:
                raise ValueError("option size/cost")
            for used, (total, path) in states.items():
                new_used, new_cost = used + size, total + cost
                if new_used <= budget and (
                    new_used not in next_states or new_cost < next_states[new_used][0]
                ):
                    next_states[new_used] = (new_cost, path + [mode])
        # Pareto pruning: a smaller state that is no more expensive dominates.
        states = {}
        best_cost = math.inf
        for used, value in sorted(next_states.items()):
            if value[0] < best_cost:
                states[used] = value
                best_cost = value[0]
        if not states:
            raise ValueError("infeasible budget")
    used, (cost, path) = min(states.items(), key=lambda item: (item[1][0], item[0]))
    return path, used, cost
    # END TODO B4


def calibrate_blocks(blocks, sample, repeats=3):
    """Per block: median fw+bw ms and saved storage bytes for retain and checkpoint.

    Block i is measured on the detached output of blocks[:i]. One warmup per mode is
    discarded. Synchronize CUDA around timing. Gradients are None on return.
    Returns [{"retain": {"ms", "saved_bytes"}, "checkpoint": {...}}, ...].
    """
    # BEGIN TODO B5
    if repeats < 1:
        raise ValueError("repeats")
    rows = []
    x = sample.detach()
    for block in blocks:
        row = {}
        for mode in ("retain", "checkpoint"):
            times, inventories = [], []
            for repeat in range(repeats + 1):
                block.zero_grad(set_to_none=True)
                value = x.detach().requires_grad_(True)
                inventory = SavedInventory(block.parameters())
                if value.is_cuda:
                    torch.cuda.synchronize(value.device)
                start = time.perf_counter()
                with inventory.context():
                    result = run_blocks([block], value, [mode == "retain"])
                result.sum().backward()
                if value.is_cuda:
                    torch.cuda.synchronize(value.device)
                if repeat:
                    times.append((time.perf_counter() - start) * 1000)
                    inventories.append(inventory.bytes)
            row[mode] = {"ms": sorted(times)[len(times) // 2], "saved_bytes": max(inventories)}
        rows.append(row)
        with torch.no_grad():
            x = block(x).detach()
        block.zero_grad(set_to_none=True)
    return rows
    # END TODO B5


class CpuOffload:
    """Synchronous saved-tensor offload to host memory. No transfer/compute overlap.

    offload_devices: device types whose tensors are copied to host. The default is
    ("cuda",); tests on a CPU-only machine pass ("cpu",) so the copy path (fresh host
    storage, counters, dtype round trip) is exercised there too.
    """

    def __init__(self, parameters=(), min_bytes=0, offload_devices=("cuda",)):
        self.excluded = {storage_key(p) for p in parameters}
        self.min_bytes = min_bytes
        self.offload_devices = tuple(offload_devices)
        self.packed_bytes = 0
        self.d2h_copies = 0
        self.h2d_copies = 0
        self.restores = 0

    def pack(self, tensor):
        # Payload = (copied, original device, value). Parameter storages (and views),
        # tensors on devices outside offload_devices and tensors smaller than
        # min_bytes: copied=False, value = tensor.detach(). Otherwise value is a fresh
        # host copy that shares nothing with the input (later in-place writes to the
        # input must not reach it) and no reference to the device tensor is kept.
        # Count packed_bytes (numel * element_size) and d2h_copies for copies only.
        # BEGIN TODO C1
        nbytes = tensor.numel() * tensor.element_size()
        excluded = storage_key(tensor) in self.excluded
        if tensor.device.type not in self.offload_devices or excluded or nbytes < self.min_bytes:
            return (False, tensor.device, tensor.detach())
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        host.copy_(tensor.detach())
        self.packed_bytes += nbytes
        self.d2h_copies += 1
        return (True, tensor.device, host)
        # END TODO C1

    def unpack(self, payload):
        # May be called more than once per payload; the payload stays valid. A copied
        # payload returns a *new* tensor on the original device with the original
        # dtype (h2d_copies += 1 each time); identity payloads return the value.
        # restores counts every call.
        # BEGIN TODO C2
        copied, device, value = payload
        self.restores += 1
        if copied:
            self.h2d_copies += 1
            return value.to(device=device, copy=True)
        return value
        # END TODO C2

    def context(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)


def run_policy(blocks, x, modes, offload=None):
    """Run blocks with per-block retain/checkpoint/offload.

    Validate before running anything: len(modes) == len(blocks), every mode in MODES,
    and an offload owner when any mode is "offload" (ValueError; no block may run and
    no RNG may be consumed). Offload blocks run inside offload.context(); the hooks
    stay attached to the graph after the context exits.
    """
    # BEGIN TODO C3
    if len(modes) != len(blocks) or any(m not in MODES for m in modes):
        raise ValueError("policy")
    if "offload" in modes and offload is None:
        raise ValueError("offload owner required")
    for block, mode in zip(blocks, modes, strict=True):
        if mode == "checkpoint":
            x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
        else:
            context = offload.context() if mode == "offload" else contextlib.nullcontext()
            with context:
                x = block(x)
    return x
    # END TODO C3


def train_accumulated(blocks, optimizer, microbatches, modes, offload=None, max_grad_norm=None):
    """One optimizer step over unequal-size MSE microbatches; returns the loss float.

    Objective: sum over all microbatches of (output - target)^2 divided by the total
    number of target elements (= full-batch MSE mean). Zero gradients once, backward
    each microbatch, optionally clip the *accumulated* gradient of all trainable
    parameters of blocks (clip_grad_norm_, L2) once, then step once. The returned
    value is the pre-step objective. Inputs/targets must have equal shapes.
    """
    # BEGIN TODO C4
    batches = list(microbatches)
    if not batches or any(x.shape != y.shape or y.numel() == 0 for x, y in batches):
        raise ValueError("nonempty matching microbatches")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("gradient norm")
    if len(modes) != len(blocks) or ("offload" in modes and offload is None):
        raise ValueError("policy")
    total = sum(y.numel() for _, y in batches)
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    for x, target in batches:
        output = run_policy(blocks, x, modes, offload)
        loss = (output - target).square().sum() / total
        loss.backward()
        loss_total += float(loss.detach())
    if max_grad_norm is not None:
        parameters = [p for block in blocks for p in block.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    optimizer.step()
    return loss_total
    # END TODO C4


def measure_policy(blocks, batches, modes, repeats=3, offload_devices=("cuda",), warm_ms=300.0):
    """Timed full optimizer steps (AdamW, lr=0) for one policy.

    One warmup step creates the optimizer state and is not measured. On CUDA the same
    step then runs for >= warm_ms more (laptop GPUs idle at a low memory clock), and
    each measured step is timed with torch.cuda.Event (perf_counter elsewhere).
    Counters and CUDA peak stats are reset right before the measured steps. lr=0
    keeps parameters fixed across policies while backward and AdamW still run.
    Returns step_ms (median), step_ms_p25/p75, stable (p75 <= 1.5 p25), losses (one
    per measured step), per-step transfer counters, CUDA allocator peaks (None off CUDA).
    """
    # BEGIN TODO C5
    if repeats < 1:
        raise ValueError("repeats")
    parameters = [p for block in blocks for p in block.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.0)
    device = batches[0][0].device
    cuda = device.type == "cuda"
    offload = CpuOffload(
        (p for block in blocks for p in block.parameters()), offload_devices=offload_devices
    )
    train_accumulated(blocks, optimizer, batches, modes, offload)
    if cuda:
        began = time.perf_counter()
        while (time.perf_counter() - began) * 1000 < warm_ms:
            train_accumulated(blocks, optimizer, batches, modes, offload)
            torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    offload.packed_bytes = offload.d2h_copies = offload.h2d_copies = offload.restores = 0
    losses, times = [], []
    for _ in range(repeats):
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            losses.append(train_accumulated(blocks, optimizer, batches, modes, offload))
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        else:
            began = time.perf_counter()
            losses.append(train_accumulated(blocks, optimizer, batches, modes, offload))
            times.append((time.perf_counter() - began) * 1000)
    q25, q50, q75 = (
        torch.tensor(times, dtype=torch.float64)
        .quantile(torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
        .tolist()
    )
    result = {
        "step_ms": q50,
        "step_ms_p25": q25,
        "step_ms_p75": q75,
        "stable": q75 <= 1.5 * q25,
        "losses": losses,
        "d2h_bytes_per_step": offload.packed_bytes / repeats,
        "d2h_copies_per_step": offload.d2h_copies / repeats,
        "h2d_copies_per_step": offload.h2d_copies / repeats,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
    }
    optimizer.zero_grad(set_to_none=True)
    return result
    # END TODO C5
