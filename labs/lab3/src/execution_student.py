"""Real PyTorch calibration and execution, not idealized latency simulation."""

import statistics
import time

import torch
from src.pipeline_student import ArtifactCache, Sample, admit, plan, route, split_trace, summarize

EPS = 1e-5
HOLDOUT_SEED = 101  # held-out request i uses request_tensors(..., seed=HOLDOUT_SEED + i)


# Provided operation: each row is independent, so zero padding rows is valid.
def operation(x, residual):
    value = x + residual
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS)


# Provided: the ONE way request tensors are generated (C2 calibration and C4 held-out).
def request_tensors(length, width, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(length, width, generator=generator, device=device)
    residual = torch.randn(length, width, generator=generator, device=device)
    return x, residual


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


# C1: time fn() with a wall clock. Synchronize the device before starting and after
# EVERY timed call (GPU work is asynchronous). The first call is timed alone; then
# `warmup` untimed calls; then `repetitions` timed calls.
# Returns (first_ms, tuple of `repetitions` floats). fn is called exactly
# 1 + warmup + repetitions times. repetitions < 1 or warmup < 0 -> ValueError.
# Test: timed_samples (call count; on CUDA a known GPU sleep must appear in the samples).
def measure(fn, device, repetitions=5, warmup=2):
    # BEGIN TODO C1
    raise NotImplementedError("C1")
    # END TODO C1


# C2: for each candidate bound: inputs = request_tensors(bound, width, device, 17 + bound);
# compiled = torch.compile(operation, backend="inductor", dynamic=False, fullgraph=True);
# time compiled with C1 (its first call includes compilation), check it equals eager,
# time eager with C1, build Sample(bound, first, steady, eager) and keep `compiled`.
# Returns (summarize(samples), {bound: compiled}, calibration_ms) where calibration_ms is
# the wall time of the WHOLE loop over all candidates (it includes discarded ones).
# Test: real_calibration (artifacts are Dynamo-wrapped callables; calibration_ms covers
# every candidate's first call).
def calibrate(bounds, width, device, repetitions=5):
    # BEGIN TODO C2
    raise NotImplementedError("C2")
    # END TODO C2


# C3: execute requests [(x, residual), ...] in order with the caller's ArtifactCache.
# Per request: bound = route(rows, bounds, max_padding); compiled path only if bound is not
# None and admit(profiles[bound], predicted.get(bound, 0)); key = (device, dtype, width,
# bound); lease the key, or put(key, artifacts[bound]) then lease; if no artifact can be
# leased run eager. Compiled path: zero-pad x AND residual to `bound` rows, call, slice
# back to the real rows, and release the lease on EVERY path (try/finally).
# Returns (outputs, events) with events[i] = (rows, "compiled", bound) or (rows, "eager", 0).
# Test: executor_padding_fallback (outputs vs FP64 oracle, events, no lease left pinned).
def execute(requests, bounds, profiles, artifacts, predicted, cache, max_padding=2.0):
    # BEGIN TODO C3
    raise NotImplementedError("C3")
    # END TODO C3


# C4: closed loop. (1) split_trace(lengths, train_count) BEFORE anything is measured;
# (2) calibrate the training shapes only; (3) forecast counts = training counts *
# horizon_multiplier; bounds = plan(forecast, profiles, capacity); predicted calls per
# bound = sum of forecast counts that route to it; (4) build held-out requests with
# request_tensors(length, width, device, HOLDOUT_SEED + i); (5) time execute() over the
# held-out list (synchronized wall time = replay_ms; calibration is NOT inside it) with a
# fresh ArtifactCache(capacity); (6) time an eager replay of the same requests with C1.
# Returns dict with keys: bounds, profiles, events, outputs (one tensor per held-out
# request, in order), calibration_ms, replay_ms, eager_replay_median_ms, eager_first_ms,
# horizon_multiplier.
# Test: closed_loop (every held-out output vs an FP64 oracle; profiles only for training
# shapes; replay_ms excludes calibration_ms; events follow route()).
def workflow(lengths, train_count, width, device, capacity=3, horizon_multiplier=1):
    # BEGIN TODO C4
    raise NotImplementedError("C4")
    # END TODO C4
