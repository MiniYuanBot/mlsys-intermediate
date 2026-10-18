"""Stage C1-C4 checks: real Inductor compilation, both execution paths, held-out outputs
compared one by one with an FP64 oracle written here. One PASS/FAIL line per check."""

import json
import time

import torch
from checks import Report
from src.execution_student import (
    HOLDOUT_SEED,
    calibrate,
    execute,
    measure,
    request_tensors,
    workflow,
)
from src.pipeline_student import ArtifactCache


def oracle(x, residual):
    """FP64 residual-add RMSNorm, independent of src/."""
    z = (x + residual).double()
    return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype)


def first_fit(length, bounds, max_padding=2.0):
    for b in bounds:
        if b >= length:
            return b if b <= max_padding * length else None
    return None


def c1(r, device):
    calls = [0]

    def slow_first():
        calls[0] += 1
        if calls[0] == 1:
            time.sleep(0.03)

    first, samples = measure(slow_first, device, 3, 2)
    r.equal("C1 fn called 1 + warmup + repetitions times", calls[0], 6)
    r.equal("C1 one sample per repetition", len(samples), 3)
    r.check("C1 first call timed alone (>= 30 ms)", first >= 29, ">= 29 ms", first)
    r.check("C1 slow first call not inside steady samples", max(samples) < 20, "< 20 ms", samples)
    r.raises("C1 repetitions=0 rejected", ValueError, lambda: measure(lambda: None, device, 0))
    if device == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(20_000_000)
        end.record()
        end.synchronize()
        gpu_ms = start.elapsed_time(end)
        _, samples = measure(lambda: torch.cuda._sleep(20_000_000), device, 3, 1)
        r.check(
            "C1 CUDA samples include the GPU work (synchronized)",
            min(samples) >= 0.5 * gpu_ms,
            f">= {0.5 * gpu_ms:.2f} ms",
            samples,
        )


def c2(r, device):
    profiles, artifacts, calibration_ms = calibrate([4, 8], 32, device, 3)
    r.equal("C2 one profile per candidate", sorted(profiles), [4, 8])
    r.equal("C2 one artifact per candidate", sorted(artifacts), [4, 8])
    r.check(
        "C2 artifacts are torch.compile wrappers (not eager)",
        all(hasattr(fn, "_torchdynamo_orig_callable") for fn in artifacts.values()),
        "Dynamo-wrapped callables",
        {b: type(fn).__name__ for b, fn in artifacts.items()},
    )
    for bound, fn in artifacts.items():
        x, residual = request_tensors(bound, 32, device, 500 + bound)
        r.close(
            f"C2 artifact[{bound}] matches FP64 oracle",
            fn(x, residual),
            oracle(x, residual),
            2e-4,
            2e-5,
        )
    premiums = sum(p["first_premium_ms"] for p in profiles.values())
    r.check(
        "C2 calibration_ms covers every candidate's first call",
        calibration_ms >= premiums > 0,
        f">= sum of premiums {premiums:.1f} > 0",
        calibration_ms,
    )
    return artifacts


def c3(r, device, artifacts):
    fixture = {
        n: {"eager_ms": 3, "steady_ms": 1, "mad_ms": 0, "first_premium_ms": 0} for n in [4, 8]
    }
    lengths = [3, 7, 13, 1, 8]
    requests = [request_tensors(n, 32, device, 900 + i) for i, n in enumerate(lengths)]
    cache = ArtifactCache(1)
    outputs, events = execute(requests, [4, 8], fixture, artifacts, {4: 10, 8: 10}, cache)
    r.equal(
        "C3 events (pad 3->4, 7->8, 13 too large, 1 over padding budget, exact 8)",
        events,
        [
            (3, "compiled", 4),
            (7, "compiled", 8),
            (13, "eager", 0),
            (1, "eager", 0),
            (8, "compiled", 8),
        ],
    )
    r.equal("C3 one output per request", len(outputs), len(requests))
    for i, (out, (x, residual)) in enumerate(zip(outputs, requests)):
        r.close(
            f"C3 request {i} (rows {lengths[i]}) output vs FP64 oracle",
            out,
            oracle(x, residual),
            2e-4,
            2e-5,
        )
    r.equal("C3 no lease left pinned", sum(cache.pins.values()), 0)
    r.equal("C3 capacity-1 cache holds the last artifact", [k[-1] for k in cache.entries], [8])
    cold = ArtifactCache(2)
    _, events = execute(requests[:2], [4, 8], fixture, artifacts, {4: 0, 8: 10}, cold)
    r.equal(
        "C3 admission: bound 4 with 0 predicted calls stays eager",
        [e[1] for e in events],
        ["eager", "compiled"],
    )

    def boom(*_):
        raise RuntimeError("artifact failure")

    broken = ArtifactCache(2)
    try:
        execute(requests[:1], [4, 8], fixture, {4: boom, 8: boom}, {4: 10, 8: 10}, broken)
        raised = False
    except RuntimeError:
        raised = True
    r.check("C3 artifact exception propagates", raised, "RuntimeError", "no exception")
    r.equal("C3 lease released on the exception path", sum(broken.pins.values()), 0)


def c4(r, device, bench):
    lengths = [4, 8, 4, 8, 4, 8, 3, 7, 13, 1]
    holdout = lengths[6:]
    result = workflow(lengths, 6, 32, device, capacity=2)
    keys = {
        "bounds",
        "profiles",
        "events",
        "outputs",
        "calibration_ms",
        "replay_ms",
        "eager_replay_median_ms",
        "eager_first_ms",
        "horizon_multiplier",
    }
    r.check("C4 result has every reported field", keys <= set(result), sorted(keys), sorted(result))
    r.equal("C4 calibration used training shapes only", sorted(result["profiles"]), [4, 8])
    bounds = result["bounds"]
    r.check(
        "C4 bounds come from training shapes",
        bounds and set(bounds) <= {4, 8},
        "subset of {4, 8}",
        bounds,
    )
    r.equal("C4 one output per held-out request", len(result["outputs"]), len(holdout))
    for i, n in enumerate(holdout):
        x, residual = request_tensors(n, 32, device, HOLDOUT_SEED + i)
        out = result["outputs"][i] if i < len(result["outputs"]) else torch.empty(0)
        r.close(
            f"C4 held-out request {i} (rows {n}) vs FP64 oracle",
            out,
            oracle(x, residual),
            2e-4,
            2e-5,
        )
    events = result["events"]
    r.equal("C4 events keep held-out order", [e[0] for e in events], holdout)
    r.check(
        "C4 unroutable requests ran eager",
        all(e[1] == "eager" for e, n in zip(events, holdout) if first_fit(n, bounds) is None),
        "eager where route() is None",
        events,
    )
    cal, rep = result["calibration_ms"], result["replay_ms"]
    r.check("C4 calibration_ms and replay_ms positive", cal > 0 and rep > 0, "> 0", (cal, rep))
    r.check("C4 replay_ms excludes calibration", rep < 0.5 * cal, f"< {0.5 * cal:.1f}", rep)
    r.check(
        "C4 eager replay measured",
        result["eager_replay_median_ms"] > 0,
        "> 0",
        result["eager_replay_median_ms"],
    )
    r.equal("C4 horizon recorded", result["horizon_multiplier"], 1)
    r.raises("C4 horizon 0 rejected", ValueError, lambda: workflow(lengths, 6, 32, device, 2, 0))
    if bench:
        shown = {k: v for k, v in result.items() if k != "outputs"}
        print(
            json.dumps(
                {
                    "device": device,
                    "scope": "real small RMSNorm block, no LLM",
                    "first_call_note": "may reuse compiler disk cache; not cold compilation",
                    **shown,
                }
            )
        )


def run(device, bench=False):
    report = Report("C1-C4")
    torch.compiler.reset()
    report.group("C1", lambda r: c1(r, device))
    holder = {}
    report.group("C2", lambda r: holder.update(artifacts=c2(r, device)))
    if "artifacts" in holder:
        report.group("C3", lambda r: c3(r, device, holder["artifacts"]))
    else:
        report.check("C3 needs C2 artifacts", False, "C2 artifacts", "C2 failed")
    report.group("C4", lambda r: c4(r, device, bench))
    return report.summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.device, args.bench))
