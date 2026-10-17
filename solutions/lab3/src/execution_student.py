"""Real PyTorch calibration and execution, not idealized latency simulation."""

import statistics
import time
import torch
from src.pipeline_student import ArtifactCache, Sample, admit, plan, route, split_trace, summarize


# Provided operation: each row is independent, so zero padding rows is valid.
def operation(x, residual):
    value = x + residual
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


# C1: measure synchronized first call separately, then warmups and repetitions.
# Return (first_ms, tuple steady_ms); validate output with eager before timing
# elsewhere. Test: timed_samples also proves all calls execute, not just enqueue.
def measure(fn, device, repetitions=5, warmup=2):
    # BEGIN TODO C1
    if repetitions < 1 or warmup < 0:
        raise ValueError("invalid repetitions")
    synchronize(device)
    start = time.perf_counter()
    fn()
    synchronize(device)
    first = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    return first, tuple(samples)
    # END TODO C1


# C2: for each candidate, actual torch.compile(inductor, fullgraph, static shape),
# deterministic inputs, eager equality, first/steady wall time, and retained
# callable. Calibration is allowed to compile candidates that planner discards;
# report its TOTAL elapsed cost separately. Test: real_calibration.
def calibrate(bounds, width, device, repetitions=5):
    # BEGIN TODO C2
    records = []
    artifacts = {}
    started = time.perf_counter()
    for bound in bounds:
        generator = torch.Generator(device=device).manual_seed(17 + bound)
        x = torch.randn(bound, width, generator=generator, device=device)
        residual = torch.randn(bound, width, generator=generator, device=device)
        compiled = torch.compile(operation, backend="inductor", dynamic=False, fullgraph=True)
        first, steady = measure(lambda: compiled(x, residual), device, repetitions)
        expected = operation(x, residual)
        torch.testing.assert_close(compiled(x, residual), expected, rtol=2e-4, atol=2e-5)
        _, eager = measure(lambda: operation(x, residual), device, repetitions)
        records.append(Sample(bound, first, steady, eager))
        artifacts[bound] = compiled
    synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000
    profiles = summarize(records)
    return profiles, artifacts, elapsed
    # END TODO C2


# C3: use only selected/admitted artifacts. Guard oversized/padding-heavy requests,
# zero-pad X and residual TOGETHER, slice back actual rows, and release lease on
# every path. Namespace (device,dtype,width,bound) prevents wrong artifact reuse.
# Test: executor_padding_fallback checks real compiled and eager paths.
def execute(requests, bounds, profiles, artifacts, predicted, capacity=3, max_padding=2.0):
    # BEGIN TODO C3
    cache = ArtifactCache(capacity)
    outputs = []
    events = []
    for x, residual in requests:
        if x.shape != residual.shape or x.ndim != 2:
            raise ValueError("matching rank-two inputs required")
        length, width = x.shape
        bound = route(length, bounds, max_padding)
        key = (str(x.device), str(x.dtype), width, bound)
        eligible = bound is not None and admit(profiles[bound], predicted.get(bound, 0))
        fn = cache.lease(key) if eligible else None
        if eligible and fn is None:
            if cache.put(key, artifacts[bound]):
                fn = cache.lease(key)
        if fn is None:
            outputs.append(operation(x, residual))
            events.append((length, "eager", 0))
            continue
        try:
            padding = bound - length
            padded_x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            padded_r = torch.nn.functional.pad(residual, (0, 0, 0, padding))
            result = fn(padded_x, padded_r)[:length]
            outputs.append(result)
            events.append((length, "compiled", bound))
        finally:
            cache.lease(key, release=True)
    return outputs, events
    # END TODO C3


# C4: closed loop. Split history BEFORE measurement/planning, forecast future
# bucket counts by a declared horizon multiplier, execute holdout, verify eager
# output and report calibration wall time separately from replay wall time.
# Test: closed_loop uses real tensors; no fabricated timing lookup is allowed.
def workflow(lengths, train_count, width, device, capacity=3, horizon_multiplier=1):
    # BEGIN TODO C4
    if horizon_multiplier < 1:
        raise ValueError("positive horizon multiplier")
    histogram, holdout = split_trace(lengths, train_count)
    candidates = [length for length, _ in histogram]
    profiles, artifacts, calibration_ms = calibrate(candidates, width, device)
    forecast = [(length, count * horizon_multiplier) for length, count in histogram]
    bounds = plan(forecast, profiles, capacity)
    predicted = dict.fromkeys(bounds, 0)
    for length, count in forecast:
        bound = route(length, bounds)
        if bound is not None:
            predicted[bound] += count
    requests = []
    for index, length in enumerate(holdout):
        generator = torch.Generator(device=device).manual_seed(101 + index)
        x = torch.randn(length, width, generator=generator, device=device)
        residual = torch.randn(length, width, generator=generator, device=device)
        requests.append((x, residual))
    synchronize(device)
    start = time.perf_counter()
    outputs, events = execute(requests, bounds, profiles, artifacts, predicted, capacity)
    synchronize(device)
    replay_ms = (time.perf_counter() - start) * 1000
    for output, (x, residual) in zip(outputs, requests, strict=True):
        torch.testing.assert_close(output, operation(x, residual), rtol=2e-4, atol=2e-5)
    eager_first, eager_samples = measure(lambda: [operation(x, r) for x, r in requests], device)
    return {
        "bounds": bounds,
        "profiles": profiles,
        "events": events,
        "calibration_ms": calibration_ms,
        "replay_ms": replay_ms,
        "eager_replay_median_ms": statistics.median(eager_samples),
        "eager_first_ms": eager_first,
        "horizon_multiplier": horizon_multiplier,
    }
    # END TODO C4
