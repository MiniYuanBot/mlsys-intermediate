"""Small executable PEFT/activation-memory system; no claim of full QLoRA reproduction."""

import contextlib
import math
import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


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
        # Preserve input gradients through the frozen base. A merged layer skips
        # the adapter branch; bias, if present, is applied exactly once.
        # BEGIN TODO A1
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.merged:
            return base
        update = torch.nn.functional.linear(torch.nn.functional.linear(x, self.a), self.b)
        return base + self.scale * update
        # END TODO A1

    @torch.no_grad()
    def merge(self):
        # Idempotent inference merge. Save an exact base copy so unmerge does not
        # accumulate floating-point subtraction error. Reject merge in train mode.
        # BEGIN TODO A2
        if self.training:
            raise ValueError("merge requires eval mode")
        if self.merged:
            return
        self._base_backup = self.weight.detach().clone()
        delta = self.b @ self.a
        self.weight.add_(delta, alpha=self.scale)
        self.merged = True
        # END TODO A2

    @torch.no_grad()
    def unmerge(self):
        # Restore the bitwise base and release its backup; repeated calls are safe.
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
    """Replace exact named nn.Linear targets, freeze original model, return names.

    Validate all targets before mutation. Preserve weight/bias/device/dtype and
    training flags. Snapshot named_modules before replacement; only adapters train.
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
    """Portable cloned CPU tensors and scale metadata, no frozen base weights."""
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
    """Validate exact module names/shapes/finite tensors/scale before any copy.

    Stage conversion to each destination device/dtype and reject conversion
    overflow before copying; an invalid last entry must leave all
    earlier modules unchanged. Reject merged destinations.
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
    """Disjoint trainable/frozen tensor bytes (not optimizer state or activation)."""
    # BEGIN TODO A7
    result = {"trainable": 0, "frozen": 0}
    seen = set()
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        key = "trainable" if parameter.requires_grad else "frozen"
        result[key] += parameter.numel() * parameter.element_size()
    return result
    # END TODO A7


def run_blocks(blocks, x, keep):
    """Non-reentrant checkpoint; preserve dropout RNG, validate full policy first."""
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
    """0/1 knapsack, positive integer size units; ties keep the earlier path."""
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


def storage_key(tensor):
    # Provided: views share storage; storage bytes may exceed view tensor bytes.
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


class SavedInventory:
    def __init__(self, parameters=()):
        self.excluded = {storage_key(p) for p in parameters}
        self.storages = {}
        self.saves = 0

    def pack(self, tensor):
        # Count unique nonparameter storage retained across this forward. This is
        # a forward inventory, not time-varying live/peak memory. Return a detach
        # to avoid retaining the input's autograd graph through the packed object.
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


def plan_modes(options, budget):
    """Multiple-choice DP. options[i][mode]=(integer retained units, measured ms).

    Choose one of retain/checkpoint/offload per block to minimize summed cost.
    Return (modes, used, cost); infeasible budgets raise. Model excludes lifetime
    overlap/workspaces; verify its proposal with a full measured training step.
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
            if mode not in {"retain", "checkpoint", "offload"}:
                raise ValueError("mode")
            if type(size) is not int or size < 0 or not math.isfinite(cost) or cost < 0:
                raise ValueError("option size/cost")
            for used, (total, path) in states.items():
                new_used, new_cost = used + size, total + cost
                if new_used <= budget and (
                    new_used not in next_states or new_cost < next_states[new_used][0]
                ):
                    next_states[new_used] = (new_cost, path + [mode])
        # Pareto pruning: a cheaper smaller state dominates a larger expensive one.
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
    """Measure each block retain/checkpoint fw+bw, median ms and saved storage.

    Inputs are detached between blocks. This calibration is not an end-to-end
    memory predictor. Synchronize GPU around timing; clear grads each repetition.
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
                with torch.autograd.graph.saved_tensors_hooks(inventory.pack, inventory.unpack):
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
    """Correct synchronous saved-tensor offload. No asynchronous overlap claim."""

    def __init__(self, parameters=(), min_bytes=0):
        self.excluded = {storage_key(p) for p in parameters}
        self.min_bytes = min_bytes
        self.packed_bytes = 0
        self.d2h_copies = 0
        self.h2d_copies = 0
        self.restores = 0

    def pack(self, tensor):
        # Never offload model parameter storage or a view of it. CPU tensors and
        # small tensors use detached identity. Payload records original device;
        # actual CUDA tensors copy to independent host ownership synchronously.
        # BEGIN TODO C1
        nbytes = tensor.numel() * tensor.element_size()
        excluded = storage_key(tensor) in self.excluded
        if not tensor.is_cuda or excluded or nbytes < self.min_bytes:
            return (False, tensor.device, tensor.detach())
        host = tensor.detach().to(device="cpu", copy=True)
        self.packed_bytes += nbytes
        self.d2h_copies += 1
        return (True, tensor.device, host)
        # END TODO C1

    def unpack(self, payload):
        # Hooks may unpack more than once; payload must remain valid and immutable.
        # Return the original value/device/dtype, not a detached reconstruction of
        # another view. Host copy transfer is counted on every actual restoration.
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
    """Execute an already validated per-block mode list; backward uses saved hooks."""
    # BEGIN TODO C3
    if len(modes) != len(blocks) or any(
        m not in {"retain", "checkpoint", "offload"} for m in modes
    ):
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
    """One SGD/Adam step over unequal-size MSE microbatches.

    Weight each mean loss by target.numel()/total; zero gradients once and step
    once. Inputs/targets must have equal shapes. Compare to concatenation with
    dropout disabled: independently partitioned dropout consumes RNG differently.
    """
    # BEGIN TODO C4
    batches = list(microbatches)
    if not batches or any(x.shape != y.shape or y.numel() == 0 for x, y in batches):
        raise ValueError("nonempty matching microbatches")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("gradient norm")
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


def measure_policy(blocks, batches, modes, repeats=3):
    """Full optimizer-step timing + allocator peaks; cold optimizer init warmed up.

    Optimizer has lr=0 to keep parameters fixed for policy comparison. Each repeat
    still runs real backward/AdamW step. CUDA allocator stats are absolute device
    allocations for this process, CPU peak is None. Return transfer counters per
    measured step and sampled losses; don't infer GPU savings from logical bytes.
    """
    # BEGIN TODO C5
    if repeats < 1:
        raise ValueError("repeats")
    parameters = [p for block in blocks for p in block.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.0)
    device = batches[0][0].device
    offload = CpuOffload(p for block in blocks for p in block.parameters())
    train_accumulated(blocks, optimizer, batches, modes, offload)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    offload.packed_bytes = offload.d2h_copies = offload.h2d_copies = offload.restores = 0
    losses = []
    start = time.perf_counter()
    for _ in range(repeats):
        losses.append(train_accumulated(blocks, optimizer, batches, modes, offload))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000 / repeats
    result = {
        "step_ms": elapsed,
        "losses": losses,
        "d2h_bytes_per_step": offload.packed_bytes / repeats,
        "d2h_copies_per_step": offload.d2h_copies / repeats,
        "h2d_copies_per_step": offload.h2d_copies / repeats,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
    }
    optimizer.zero_grad(set_to_none=True)
    return result
    # END TODO C5
