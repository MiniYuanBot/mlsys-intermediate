"""Stage R checks. CPU: selection, padded replay through the emulated runner, cost model,
planner vs brute force over every subset. CUDA adds real torch.cuda.CUDAGraph replay vs
eager. Oracles are written here; one PASS/FAIL line per check."""

import argparse
import itertools
import math
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checks import Report  # noqa: E402
from real.decode_block import DecodeModel  # noqa: E402
from src.graphs_student import GraphRunner, cost_model, plan_capture_sizes, select_graph  # noqa: E402

TINY = dict(hidden=64, heads=4, kv_heads=2, head_dim=16, mlp=128)


def first_fit(batch, sizes):
    for s in sizes:
        if s >= batch:
            return s
    return None


def objective(hist, model, sizes):
    """Test-side J(S): capture once per graph + padded replay or eager per step."""
    total = sum(model[s]["first_premium_ms"] for s in sizes)
    for b, f in hist:
        s = first_fit(b, sizes)
        total += f * (model[s]["steady_ms"] if s is not None else model[b]["eager_ms"])
    return total


def memory(model, sizes, overhead):
    return model[max(sizes)]["pool_mib"] + overhead * len(sizes) if sizes else 0.0


def r1(r):
    sizes = [1, 2, 4, 8, 16, 24, 32]
    for b in range(1, 41):
        r.equal(
            f"R1 select_graph({b}) pads up to the next captured size",
            select_graph(b, sizes),
            first_fit(b, sizes),
        )
    r.equal("R1 single size, exact", select_graph(5, [5]), 5)
    r.raises("R1 unsorted sizes rejected", ValueError, lambda: select_graph(3, [4, 2]))
    r.raises("R1 batch 0 rejected", ValueError, lambda: select_graph(0, [4]))


def check_runner(r, runner, model, batches, device, tag, tol):
    g = torch.Generator(device="cpu").manual_seed(5)
    kept = []
    for i, b in enumerate(batches):
        x = torch.randn(b, model.hidden, generator=g).to(device)
        out, event = runner.run(x)
        want = model(x)
        expected_kind = (
            ("graph", first_fit(b, runner.sizes)) if first_fit(b, runner.sizes) else ("eager", 0)
        )
        r.equal(f"R2 {tag} step {i} batch {b}: path", event, expected_kind)
        r.close(f"R2 {tag} step {i} batch {b}: output vs eager", out, want, tol, tol)
        if event[0] == "graph":
            size = event[1]
            pad = runner.static_in[b:size]
            r.check(
                f"R2 {tag} step {i}: padded rows zeroed",
                bool((pad == 0).all()),
                "zeros",
                pad.abs().max().item() if pad.numel() else 0,
            )
            kept.append((out, want))
    stale = [not torch.allclose(out, want, rtol=tol, atol=tol) for out, want in kept]
    r.check(
        f"R2 {tag} earlier results survive later replays (no aliasing)",
        not any(stale),
        "all kept outputs intact",
        stale,
    )


def r2_cpu(r):
    model = DecodeModel(layers=2, max_batch=16, context=8, device="cpu", **TINY)
    runner = GraphRunner(model, [2, 4, 8], model.hidden, "cpu", torch.float32, backend="emulate")
    check_runner(r, runner, model, [8, 3, 5, 1, 9, 3, 2], "cpu", "emulate", 1e-5)


def r2_cuda(r):
    model = DecodeModel(layers=2, max_batch=32, context=16, device="cuda", **TINY)
    runner = GraphRunner(model, [1, 4, 8, 16], model.hidden, "cuda", torch.float32, backend="cuda")
    r.check(
        "R2 cuda capture_ms recorded per size",
        sorted(runner.capture_ms) == [1, 4, 8, 16]
        and all(v > 0 for v in runner.capture_ms.values()),
        "4 positive values",
        runner.capture_ms,
    )
    check_runner(r, runner, model, [16, 13, 1, 5, 20, 3, 16], "cuda", "cuda graphs", 1e-4)


def r3(r):
    records = [
        {
            "batch": 4,
            "replay_ms": [1.0, 1.2, 9.0],
            "eager_ms": [3.0, 2.0, 4.0, 100.0],
            "capture_ms": 7.5,
            "pool_mib": 2.0,
        },
        {
            "batch": 8,
            "replay_ms": [2.0, 2.5, 3.0, 4.0],
            "eager_ms": [5.0],
            "capture_ms": 0.0,
            "pool_mib": 4.0,
        },
    ]
    model = cost_model(records)
    r.equal("R3 one profile per batch", sorted(model), [4, 8])
    for rec in records:
        b = rec["batch"]
        med = statistics.median(rec["replay_ms"])
        devs = sorted(abs(v - med) for v in rec["replay_ms"])
        mad = statistics.median(devs)
        got = model.get(b, {})
        r.near(f"R3 batch {b} steady = median replay", got.get("steady_ms"), med)
        r.near(
            f"R3 batch {b} eager = median eager",
            got.get("eager_ms"),
            statistics.median(rec["eager_ms"]),
        )
        r.near(f"R3 batch {b} premium = capture_ms", got.get("first_premium_ms"), rec["capture_ms"])
        r.near(f"R3 batch {b} pool_mib kept", got.get("pool_mib"), rec["pool_mib"])
        r.near(f"R3 batch {b} MAD around the median", got.get("mad_ms"), mad)
    bad = [
        ("duplicate batch", records + records[:1]),
        ("empty replay", [{**records[0], "replay_ms": []}]),
        ("NaN eager", [{**records[0], "eager_ms": [float("nan")]}]),
        ("negative pool", [{**records[0], "pool_mib": -1.0}]),
    ]
    for name, recs in bad:
        r.raises(f"R3 rejects {name}", ValueError, lambda recs=recs: cost_model(recs))


def r4(r):
    rng = random.Random(11)
    batches = list(range(1, 13))
    for trial in range(4):
        model = {}
        for b in batches:
            steady = 0.2 + 0.03 * b + rng.uniform(-0.02, 0.02)  # not monotone: noise
            model[b] = {
                "steady_ms": steady,
                "eager_ms": steady + rng.uniform(-0.05, 0.4),
                "first_premium_ms": rng.choice([0.0, 0.5, 3.0]),
                "pool_mib": 1.0 + 0.5 * b,
                "mad_ms": 0.0,
            }
        observed = sorted(rng.sample(batches, 7))
        hist = [(b, rng.randint(1, 30)) for b in observed]
        for k, budget, overhead in itertools.product(
            [0, 1, 2, 3], [0.0, 3.5, 5.0, math.inf], [0.0, 0.5]
        ):
            sizes, cost = plan_capture_sizes(hist, model, k, budget, overhead)
            best = min(
                objective(hist, model, list(c))
                for n in range(0, k + 1)
                for c in itertools.combinations(observed, n)
                if memory(model, list(c), overhead) <= budget
            )
            tag = f"R4 trial {trial} K={k} M={budget} o={overhead}"
            feasible = (
                list(sizes) == sorted(set(sizes))
                and set(sizes) <= set(observed)
                and len(sizes) <= k
                and memory(model, list(sizes), overhead) <= budget
            )
            r.check(
                f"{tag}: feasible", feasible, "observed sizes within K and memory budget", sizes
            )
            r.near(
                f"{tag}: reported cost = J(sizes)", cost, objective(hist, model, list(sizes)), 1e-9
            )
            r.near(f"{tag}: optimal vs brute force", cost, best, 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = Report("R")
    report.group("R1", r1)
    report.group("R2 emulate", r2_cpu)
    if args.device == "cuda":
        report.group("R2 cuda", r2_cuda)
    else:
        print(
            "NOT RUN R2 real CUDA-graph replay (CPU selected); emulated runner checked", flush=True
        )
    report.group("R3", r3)
    report.group("R4", r4)
    return report.summary()


if __name__ == "__main__":
    raise SystemExit(main())
