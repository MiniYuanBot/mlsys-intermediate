"""Compatibility entrypoints for the earlier linear-cost diagnostic."""

from collections import Counter
from src.pipeline_student import plan, route


def plan_buckets(histogram, compile_ms, per_item_ms):
    if compile_ms < 0 or per_item_ms < 0 or any(n <= 0 or f <= 0 for n, f in histogram):
        raise ValueError("positive shapes/counts and nonnegative costs")
    counts = Counter()
    for n, f in histogram:
        counts[n] += f
    merged = sorted(counts.items())
    profiles = {
        n: {"first_premium_ms": compile_ms, "steady_ms": n * per_item_ms} for n, _ in merged
    }
    return plan(merged, profiles, max(1, len(merged)))


def bucket_for(length, bounds):
    return route(length, bounds, float("inf"))
