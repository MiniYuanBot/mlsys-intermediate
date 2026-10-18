"""Stage R bench: measure CUDA-graph costs per batch size (R.1), plan capture sizes under
graph-count and memory budgets with the B1 DP (R.2), compare with vLLM/SGLang-style
default lists and uniform buckets on held-out serving traces (R.3).

--device cuda : profile a Qwen3-0.6B-shaped 4-layer FP16 decode step at every batch size
                1..64 on this GPU, plan, then also REPLAY the held-out traces through real
                graphs and report measured ms/step next to the modeled value.
--device cpu  : no GPU; plan and compare from the committed profile measured on an
                RTX 4060 Laptop GPU (real/data/profile_rtx4060.json). Costs are modeled
                from those measurements, nothing is measured on this machine.

Timing rules (labs/docs/authoring.md section 5): each configuration is spun >= 300 ms with
the same launch, then 25 timed reps with CUDA events; median/p25/p75 and stable column.
"""

import argparse
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real.decode_block import DecodeModel, serving_trace  # noqa: E402
from src.graphs_student import GraphRunner, cost_model, plan_capture_sizes, plan_cost, plan_memory  # noqa: E402

HERE = Path(__file__).resolve().parent
COMMITTED = HERE / "real" / "data" / "profile_rtx4060.json"
MAX_BATCH, SPIN_MS, REPS = 64, 300.0, 25
# vLLM @ 371745ebd6c12f314402c6c319de2e7ef82c2981 vllm/config/compilation.py docstring of
# max_cudagraph_capture_size: [1, 2, 4] + list(range(8, 256, 8)) + range(256, max+1, 16).
VLLM = [1, 2, 4] + list(range(8, 256, 8))
# SGLang @ 57db5089c207ae4ce189cc092a89a2027c8aea29 python/sglang/srt/arg_groups/
# cuda_graph_hook.py generate_decode_cuda_graph_batch_sizes: [1, 2, 4, 8, 12] + range(16, 257, 8) ...
SGLANG = [1, 2, 4, 8, 12] + list(range(16, 257, 8))


def timed(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    busy = 0.0
    while busy < SPIN_MS:
        for _ in range(4):
            fn()
        end.record()
        end.synchronize()
        busy = start.elapsed_time(end)
    samples = []
    for _ in range(REPS):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def quartiles(samples):
    s = sorted(samples)
    q = lambda p: s[int(p * (len(s) - 1) + 0.5)]  # noqa: E731
    return q(0.5), q(0.25), q(0.75)


def profile(model):
    """R.1: per batch size, eager latency, graph replay latency, capture ms, private pool MiB."""
    records = []
    x_all = torch.randn(MAX_BATCH, model.hidden, device="cuda", dtype=torch.float16)
    print("batch,eager_ms,replay_ms,replay_p25,replay_p75,capture_ms,pool_mib,stable", flush=True)
    for b in range(1, MAX_BATCH + 1):
        x = x_all[:b]
        eager = timed(lambda: model(x))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        runner = GraphRunner(model, [b], model.hidden, "cuda", torch.float16)
        runner.static_in.copy_(x)
        replay = timed(lambda: runner.graphs[b].replay())
        rec = {
            "batch": b,
            "eager_ms": eager,
            "replay_ms": replay,
            "capture_ms": runner.capture_ms[b],
            "pool_mib": runner.pool_mib,
        }
        records.append(rec)
        med, p25, p75 = quartiles(replay)
        stable = (
            "yes" if p75 <= 1.5 * p25 and quartiles(eager)[2] <= 1.5 * quartiles(eager)[1] else "no"
        )
        print(
            f"{b},{quartiles(eager)[0]:.4f},{med:.4f},{p25:.4f},{p75:.4f},"
            f"{rec['capture_ms']:.2f},{rec['pool_mib']:.1f},{stable}",
            flush=True,
        )
        del runner
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()  # one workspace per capture stream
    return records


def histogram(trace):
    return sorted(Counter(trace).items())


def policies(train_hist, model, k, budget, overhead):
    dp, _ = plan_capture_sizes(train_hist, model, k, budget, overhead)
    uniform = sorted({max(1, round(MAX_BATCH * i / k)) for i in range(1, k + 1)})
    return {
        "dp_train": dp,
        "vllm_default": [s for s in VLLM if s <= MAX_BATCH],
        f"vllm_first_{k}": [s for s in VLLM if s <= MAX_BATCH][:k],
        "sglang_default": [s for s in SGLANG if s <= MAX_BATCH],
        f"uniform_{k}": uniform,
        "all_eager": [],
    }


def measured_replay(model, sizes, trace):
    """Replay a held-out trace through real graphs; returns measured ms/step (median of 3)."""
    runner = GraphRunner(model, sizes, model.hidden, "cuda", torch.float16) if sizes else None
    x_all = torch.randn(MAX_BATCH, model.hidden, device="cuda", dtype=torch.float16)

    def loop():
        # A server reads the sampled tokens back every step, so each step ends with a
        # device-to-host copy (a sync); without it CPU launches would overlap GPU work.
        for b in trace:
            y = model(x_all[:b]) if runner is None else runner.run(x_all[:b])[0]
            y[0, 0].item()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    loop()  # the trace itself is longer than 300 ms; it is the warmup
    end.record()
    end.synchronize()
    runs = []
    for _ in range(3):
        start.record()
        loop()
        end.record()
        end.synchronize()
        runs.append(start.elapsed_time(end) / len(trace))
    return sorted(runs)[1], (runner.pool_mib if runner else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-graphs", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--overhead-mib", type=float, default=0.0)
    args = parser.parse_args()
    if args.device == "cuda":
        model = DecodeModel(
            layers=4, max_batch=MAX_BATCH, context=256, device="cuda", dtype=torch.float16
        )
        records = profile(model)
        source = {
            "measured_on": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "host": platform.platform(),
            "kind": "measured",
        }
        out = HERE / "build"
        out.mkdir(exist_ok=True)
        (out / "profile_measured.json").write_text(
            json.dumps({"source": source, "records": records}), encoding="utf-8"
        )
        print(
            f"# wrote build/profile_measured.json (copy to real/data/ to refresh the committed profile)"
        )
    else:
        data = json.loads(COMMITTED.read_text(encoding="utf-8"))
        records, source = (
            data["records"],
            {**data["source"], "kind": "committed profile; costs modeled"},
        )
        model = None
    costs = cost_model(records)
    print("# profile source: " + json.dumps(source), flush=True)
    train = serving_trace(4000, 0.2, MAX_BATCH, seed=1)
    tests = {
        "stable": serving_trace(4000, 0.2, MAX_BATCH, seed=2),
        "shift": serving_trace(4000, 0.32, MAX_BATCH, seed=3),
    }
    tight = costs[32]["pool_mib"] + 1e-9  # memory budget = what the batch-32 graph's pool needs
    train_hist = histogram(train)
    print(
        f"# train trace {len(train)} steps, mean batch {sum(train) / len(train):.1f}; "
        + "; ".join(f"{n} mean {sum(t) / len(t):.1f}" for n, t in tests.items()),
        flush=True,
    )
    for k in args.max_graphs:
        for budget_name, budget in [("loose", math.inf), ("tight", tight)]:
            plans = policies(train_hist, costs, k, budget, args.overhead_mib)
            for scenario, trace in tests.items():
                hist = histogram(trace)
                hindsight, best = plan_capture_sizes(hist, costs, k, budget, args.overhead_mib)
                for name, sizes in plans.items():
                    j = plan_cost(hist, costs, sizes)
                    row = {
                        "scenario": scenario,
                        "K": k,
                        "budget": budget_name,
                        "budget_mib": None if math.isinf(budget) else round(budget, 2),
                        "policy": name,
                        "sizes": sizes,
                        "graphs": len(sizes),
                        "memory_model_mib": round(plan_memory(costs, sizes, args.overhead_mib), 2),
                        "within_budget": len(sizes) <= k
                        and plan_memory(costs, sizes, args.overhead_mib) <= budget,
                        "modeled_ms_per_step": j / len(trace),
                        "regret_vs_hindsight_pct": 100 * (j - best) / best,
                        "hindsight_sizes": hindsight,
                        "label": "modeled from measured profile",
                    }
                    if (
                        model is not None
                        and budget_name == "loose"
                        and name in ("dp_train", "vllm_default", f"uniform_{k}", "all_eager")
                    ):
                        ms, pool = measured_replay(model, sizes, trace[:1500])
                        capture = sum(costs[s]["first_premium_ms"] for s in sizes)
                        modeled = (
                            plan_cost(histogram(trace[:1500]), costs, sizes) - capture
                        ) / 1500
                        row.update(
                            measured_ms_per_step=ms,
                            measured_pool_mib=round(pool, 1),
                            modeled_replay_ms_per_step_first1500=modeled,
                        )
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
