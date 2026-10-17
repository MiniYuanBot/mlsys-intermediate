"""Measured shape planning; no hold-out lengths or timings enter the plan."""

import bisect
import math
import statistics
from collections import Counter, OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    bound: int
    first_ms: float
    steady_ms: tuple[float, ...]
    eager_ms: tuple[float, ...]


# A1: chronological split, then histogram of TRAIN ONLY. Keep duplicate requests;
# sorting before the split leaks future popularity. Test: chronological_split.
def split_trace(lengths, train_count):
    # BEGIN TODO A1
    if not 0 < train_count < len(lengths):
        raise ValueError("nonempty training and holdout required")
    if any(not isinstance(n, int) or n <= 0 for n in lengths):
        raise ValueError("positive integer lengths required")
    training = list(lengths[:train_count])
    holdout = list(lengths[train_count:])
    frequency = Counter(training)
    histogram = sorted(frequency.items())
    assert sum(count for _, count in histogram) == len(training)
    return histogram, holdout
    # END TODO A1


# A2: median steady/eager samples; first-call premium=max(first-steady,0).
# Measurements already use synchronization. Reject duplicate bounds or NaNs.
# Store MAD so a noisy calibration remains visible. Test: robust_profiles.
def summarize(samples):
    # BEGIN TODO A2
    profiles = {}
    for sample in samples:
        if sample.bound <= 0 or sample.bound in profiles:
            raise ValueError("duplicate/invalid bound")
        values = [sample.first_ms, *sample.steady_ms, *sample.eager_ms]
        if not sample.steady_ms or not sample.eager_ms:
            raise ValueError("missing repetitions")
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("invalid timing")
        steady = statistics.median(sample.steady_ms)
        eager = statistics.median(sample.eager_ms)
        mad = statistics.median(abs(x - steady) for x in sample.steady_ms)
        profiles[sample.bound] = {
            "steady_ms": steady,
            "eager_ms": eager,
            "first_premium_ms": max(0.0, sample.first_ms - steady),
            "mad_ms": mad,
        }
    return profiles
    # END TODO A2


# A3: produce an interval-cost table for contiguous sorted training shapes.
# Bucket [i,j] uses its largest shape, pays ONE first-call premium plus all
# invocations at its measured cost. No linear per-item assumption. Test: interval_cost.
def segment_costs(histogram, profiles):
    # BEGIN TODO A3
    if histogram != sorted(histogram) or len({n for n, _ in histogram}) != len(histogram):
        raise ValueError("histogram must be sorted and unique")
    if any(n <= 0 or count <= 0 for n, count in histogram):
        raise ValueError("invalid histogram")
    prefix = [0]
    for _, count in histogram:
        prefix.append(prefix[-1] + count)
    costs = {}
    for end, (bound, _) in enumerate(histogram):
        if bound not in profiles:
            raise ValueError("missing measured bound")
        profile = profiles[bound]
        for begin in range(end + 1):
            calls = prefix[end + 1] - prefix[begin]
            runtime = calls * profile["steady_ms"]
            costs[begin, end] = profile["first_premium_ms"] + runtime
    return costs
    # END TODO A3


# A4: histogram total-variation distance on union support; counterexample must
# detect a new shape even when old mean length is unchanged. Test: distribution_shift.
def trace_shift(training, recent):
    # BEGIN TODO A4
    old = dict(training)
    new = Counter(recent)
    if any(n <= 0 or f <= 0 for n, f in training) or any(n <= 0 for n in recent):
        raise ValueError("invalid observation")
    old_total = sum(old.values())
    new_total = sum(new.values())
    if old_total == 0 or new_total == 0:
        return 0.0 if old_total == new_total else 1.0
    keys = set(old) | set(new)
    distance = 0.0
    for key in keys:
        distance += abs(old.get(key, 0) / old_total - new.get(key, 0) / new_total)
    return distance / 2
    # END TODO A4


# B1: exact DP over prefix length and number of buckets, constrained by capacity.
# Backtrack chosen upper bounds; deterministic tie: smaller bucket count, then
# earliest split. Test: dp_exhaustive enumerates ALL partitions independently.
def plan(histogram, profiles, capacity):
    # BEGIN TODO B1
    if capacity < 1:
        raise ValueError("capacity must be positive")
    if not histogram:
        return []
    costs = segment_costs(histogram, profiles)
    n = len(histogram)
    limit = min(capacity, n)
    dp = [[math.inf] * (n + 1) for _ in range(limit + 1)]
    previous = [[None] * (n + 1) for _ in range(limit + 1)]
    dp[0][0] = 0.0
    for used in range(1, limit + 1):
        for end in range(used, n + 1):
            for begin in range(used - 1, end):
                value = dp[used - 1][begin] + costs[begin, end - 1]
                if value < dp[used][end]:
                    dp[used][end] = value
                    previous[used][end] = begin
    used = min(range(1, limit + 1), key=lambda count: dp[count][n])
    end = n
    bounds = []
    while used:
        bounds.append(histogram[end - 1][0])
        end = previous[used][end]
        used -= 1
    bounds.reverse()
    return bounds
    # END TODO B1


# B2: upper-bound route; enforce padding ratio budget. Unknown large shapes
# return None for eager fallback. Zero is never a legal length. Test: guarded_route.
def route(length, bounds, max_padding=2.0):
    # BEGIN TODO B2
    if length <= 0 or max_padding < 1:
        raise ValueError("invalid request/padding limit")
    if bounds != sorted(set(bounds)) or any(n <= 0 for n in bounds):
        raise ValueError("sorted unique positive bounds required")
    index = bisect.bisect_left(bounds, length)
    if index == len(bounds):
        return None
    bound = bounds[index]
    if bound / length > max_padding:
        return None
    return bound
    # END TODO B2


class ArtifactCache:
    """Measured planner's bounded application cache; independent of Dynamo cache."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("positive cache capacity")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.pins = Counter()

    # B3: transactional LRU admission. Existing entry is refreshed; new entry
    # evicts oldest UNPINNED only. Failed insertion must leave state unchanged.
    # A pinned existing key cannot be replaced. Test: pinned_transactional_lru.
    def put(self, key, value):
        # BEGIN TODO B3
        if key in self.entries:
            if self.pins[key]:
                return False
            self.entries[key] = value
            self.entries.move_to_end(key)
            return True
        victim = None
        if len(self.entries) >= self.capacity:
            victim = next(
                (candidate for candidate in self.entries if not self.pins[candidate]), None
            )
            if victim is None:
                return False
        if victim is not None:
            del self.entries[victim]
            self.pins.pop(victim, None)
        self.entries[key] = value
        return True
        # END TODO B3

    # B4: acquire/release a lease. Touch LRU on acquire; invalid release errors.
    # Test: lease_lifetime; C executor uses try/finally so exceptions release.
    def lease(self, key, release=False):
        # BEGIN TODO B4
        if release:
            if self.pins[key] <= 0:
                raise ValueError("release without lease")
            self.pins[key] -= 1
            return None
        if key not in self.entries:
            return None
        self.pins[key] += 1
        self.entries.move_to_end(key)
        return self.entries[key]
        # END TODO B4


# B5: predicted eager vs compiled saving over a horizon must amortize measured
# first-call premium plus a MAD uncertainty margin PER CALL. Equality rejects.
# Never use observed test counts here. Test: admission_break_even.
def admit(profile, future_calls, margin=2.0):
    # BEGIN TODO B5
    if future_calls < 0 or margin < 0:
        raise ValueError("nonnegative prediction and margin")
    eager = profile["eager_ms"]
    conservative = profile["steady_ms"] + margin * profile["mad_ms"]
    saving = eager - conservative
    if saving <= 0:
        return False
    total_saving = saving * future_calls
    return total_saving > profile["first_premium_ms"]
    # END TODO B5


# A5: total real rows, padded rows, and eager-fallback calls for a histogram.
# Fallback contributes actual rows (no padding). Weighted waste differs from
# averaging per-shape ratios. Test: weighted_padding_accounting.
def padding_account(histogram, bounds, max_padding=2.0):
    # BEGIN TODO A5
    real_rows = 0
    executed_rows = 0
    fallback_calls = 0
    for length, frequency in histogram:
        if frequency <= 0:
            raise ValueError("positive frequency required")
        bound = route(length, bounds, max_padding)
        real_rows += length * frequency
        if bound is None:
            executed_rows += length * frequency
            fallback_calls += frequency
        else:
            executed_rows += bound * frequency
    waste = executed_rows - real_rows
    ratio = waste / executed_rows if executed_rows else 0.0
    return {
        "real_rows": real_rows,
        "executed_rows": executed_rows,
        "padding_fraction": ratio,
        "fallback_calls": fallback_calls,
    }
    # END TODO A5
