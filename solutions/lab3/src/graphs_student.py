"""Stage R: CUDA-graph bucketing in miniature (the vLLM/SGLang capture-size problem).

A CUDA graph replays a fixed sequence of kernels on fixed buffers, so it only exists for
the batch sizes it was captured at. A server captures graphs at a few sizes and pads
each step's batch up to the next captured size; larger batches run eager.
"""

import bisect
import math
import statistics
import time

import torch
from src.pipeline_student import plan, route


# R1: smallest captured size >= batch (pad UP, never truncate); None when batch exceeds
# every captured size (eager). `sizes` must be sorted, unique and positive; batch >= 1.
# Same rule as vLLM's bs_to_padded_graph_size table and SGLang's bisect_left lookup.
# Test: test_r.py R1 select (boundaries, exact sizes, above the largest).
def select_graph(batch, sizes):
    # BEGIN TODO R1
    if batch < 1:
        raise ValueError("batch must be positive")
    if list(sizes) != sorted(set(sizes)) or (sizes and sizes[0] < 1):
        raise ValueError("sizes must be sorted, unique, positive")
    index = bisect.bisect_left(sizes, batch)
    return sizes[index] if index < len(sizes) else None
    # END TODO R1


class GraphRunner:
    """Graphs captured at `sizes` over ONE static input buffer and ONE shared memory pool.

    backend="cuda": real torch.cuda.CUDAGraph capture, largest size first, on one capture
    stream (a fresh stream per capture would allocate a new cuBLAS workspace each time).
    backend="emulate": no capture; _replay(s) recomputes fn(static_in[:s]) into the same
    static output buffer, so the padding/aliasing contract can be tested on CPU.
    After construction: capture_ms[s], and pool_mib = reserved memory that stays held after
    capture + empty_cache (the graphs' private pool; warmup blocks are released first).
    """

    def __init__(self, fn, sizes, hidden, device, dtype, backend="cuda"):
        if list(sizes) != sorted(set(sizes)) or not sizes or sizes[0] < 1:
            raise ValueError("sizes must be sorted, unique, positive")
        self.fn, self.sizes, self.backend = fn, list(sizes), backend
        self.static_in = torch.zeros(sizes[-1], hidden, device=device, dtype=dtype)
        self.outputs, self.graphs, self.capture_ms = {}, {}, {}
        self.pool_mib = 0.0
        if backend == "emulate":
            for s in self.sizes:
                self.outputs[s] = torch.empty(s, hidden, device=device, dtype=dtype)
                self.capture_ms[s] = 0.0
            return
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warm up cuBLAS/cuDNN on the capture stream
            for s in reversed(self.sizes):
                fn(self.static_in[:s])
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # drop warmup blocks so the delta below is the graph pool
        pool = torch.cuda.graph_pool_handle()
        before = torch.cuda.memory_reserved()
        for s in reversed(self.sizes):
            graph = torch.cuda.CUDAGraph()
            start = time.perf_counter()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                self.outputs[s] = fn(self.static_in[:s])
            torch.cuda.synchronize()
            self.capture_ms[s] = (time.perf_counter() - start) * 1000
            self.graphs[s] = graph
        torch.cuda.empty_cache()
        self.pool_mib = (torch.cuda.memory_reserved() - before) / 2**20

    def _replay(self, size):
        if self.backend == "emulate":
            self.outputs[size].copy_(self.fn(self.static_in[:size]))
        else:
            self.graphs[size].replay()

    # R2: run one decode step x [batch, hidden]. size = select_graph(batch, self.sizes);
    # None -> return (self.fn(x), ("eager", 0)). Otherwise copy x into static_in[:batch],
    # ZERO static_in[batch:size] (stale rows from an earlier larger step must not leak into
    # padded rows), self._replay(size), and return (a COPY of outputs[size][:batch],
    # ("graph", size)). Returning a view would alias the static buffer: the next replay
    # would silently overwrite this step's result.
    # Test: test_r.py R2 runner (CPU emulate + real CUDA graphs vs eager).
    def run(self, x):
        # BEGIN TODO R2
        batch = x.shape[0]
        size = select_graph(batch, self.sizes)
        if size is None:
            return self.fn(x), ("eager", 0)
        self.static_in[:batch].copy_(x)
        self.static_in[batch:size].zero_()
        self._replay(size)
        return self.outputs[size][:batch].clone(), ("graph", size)
        # END TODO R2


# R3: turn raw measurements into planner profiles (the same dict shape B1 consumes).
# records: [{"batch": b, "replay_ms": [...], "eager_ms": [...], "capture_ms": c,
#            "pool_mib": m}, ...]. Returns {b: {"steady_ms": median(replay_ms),
# "eager_ms": median(eager_ms), "first_premium_ms": capture_ms, "pool_mib": pool_mib,
# "mad_ms": median absolute deviation of replay_ms}}. Duplicate batches, empty sample
# lists, NaN/inf or negative values -> ValueError. Test: test_r.py R3 cost_model.
def cost_model(records):
    # BEGIN TODO R3
    model = {}
    for record in records:
        batch = record["batch"]
        replay, eager = list(record["replay_ms"]), list(record["eager_ms"])
        values = [*replay, *eager, record["capture_ms"], record["pool_mib"]]
        if batch in model or batch < 1 or not replay or not eager:
            raise ValueError("duplicate/invalid batch or missing samples")
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid measurement")
        steady = statistics.median(replay)
        model[batch] = {
            "steady_ms": steady,
            "eager_ms": statistics.median(eager),
            "first_premium_ms": record["capture_ms"],
            "pool_mib": record["pool_mib"],
            "mad_ms": statistics.median(abs(v - steady) for v in replay),
        }
    return model
    # END TODO R3


def plan_cost(histogram, model, sizes):
    """Provided objective J(S) in ms: one capture per graph + every step at its padded
    graph's replay latency, or at its own eager latency when it exceeds max(S)."""
    total = sum(model[s]["first_premium_ms"] for s in sizes)
    for batch, count in histogram:
        size = select_graph(batch, sizes) if sizes else None
        total += count * (model[size]["steady_ms"] if size else model[batch]["eager_ms"])
    return total


def plan_memory(model, sizes, overhead_mib):
    """Provided shared-pool memory model: the largest graph's pool + a per-graph overhead."""
    return (model[sizes[-1]]["pool_mib"] + overhead_mib * len(sizes)) if sizes else 0.0


# R4: choose capture sizes S (subset of the histogram's batch sizes) minimizing
# plan_cost(histogram, model, S) subject to len(S) <= max_graphs and
# plan_memory(model, S, overhead_mib) <= mem_budget_mib. S may be empty (all eager).
# Reduce to B1: for each candidate top size t (largest graph) and each k <= max_graphs,
# steps with batch <= t form a prefix histogram that plan(prefix, model, k) solves
# exactly; steps above t pay eager_ms. Keep the cheapest feasible plan; ties keep the
# earlier candidate (smaller t, then smaller k). Returns (sizes, cost_ms).
# Test: test_r.py R4 planner (brute force over all subsets, budgets, overhead).
def plan_capture_sizes(histogram, model, max_graphs, mem_budget_mib, overhead_mib=0.0):
    # BEGIN TODO R4
    if max_graphs < 0 or mem_budget_mib < 0:
        raise ValueError("nonnegative budgets required")
    histogram = sorted(histogram)
    best_sizes, best_cost = [], plan_cost(histogram, model, [])
    for top_index, (top, _) in enumerate(histogram):
        prefix = histogram[: top_index + 1]
        eager_rest = sum(count * model[b]["eager_ms"] for b, count in histogram[top_index + 1 :])
        for k in range(1, max_graphs + 1):
            sizes = plan(prefix, model, k)
            if plan_memory(model, sizes, overhead_mib) > mem_budget_mib:
                continue
            cost = plan_cost(prefix, model, sizes) + eager_rest
            if cost < best_cost:
                best_sizes, best_cost = sizes, cost
    return best_sizes, best_cost
    # END TODO R4
