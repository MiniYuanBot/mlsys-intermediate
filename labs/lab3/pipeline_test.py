"""Stage A/B checks. Every oracle is written here (brute force, Fractions, hand sums);
none calls the function under test to produce its own expected value. One PASS/FAIL line
per check; the run continues past failures and exits nonzero at the end."""

import itertools
import random
from collections import Counter
from fractions import Fraction
from unittest.mock import patch

import src.pipeline_student as module
from checks import Report
from src.pipeline_student import (
    ArtifactCache,
    Sample,
    admit,
    padding_account,
    plan,
    route,
    segment_costs,
    split_trace,
    summarize,
    trace_shift,
)


# ---------------------------------------------------------------- oracles (test side)
def tv_oracle(training, recent):
    """Exact total variation with Fractions on the union of supports."""
    old = Counter()
    for n, f in training:
        old[n] += f
    new = Counter(recent)
    a, b = sum(old.values()), sum(new.values())
    if a == 0 and b == 0:
        return Fraction(0)
    if a == 0 or b == 0:
        return Fraction(1)
    keys = set(old) | set(new)
    return sum(abs(Fraction(old[k], a) - Fraction(new[k], b)) for k in keys) / 2


def first_fit(length, bounds, max_padding):
    for b in bounds:  # linear scan, not bisect
        if b >= length:
            return b if b <= max_padding * length else None
    return None


def plan_cost(hist, profiles, bounds):
    total = sum(profiles[b]["first_premium_ms"] for b in bounds)
    for n, f in hist:
        bound = next(b for b in bounds if b >= n)
        total += f * profiles[bound]["steady_ms"]
    return total


def all_plans(shapes, capacity):
    """Every contiguous partition = every subset of non-final split points + the max."""
    inner, top = shapes[:-1], shapes[-1]
    for k in range(capacity):
        for c in itertools.combinations(inner, k):
            yield list(c) + [top]


# ---------------------------------------------------------------- stage A
def a1(r):
    hist, held = split_trace([8, 4, 8, 99, 99], 3)
    r.equal("A1 train histogram uses only the first 3 requests", hist, [(4, 1), (8, 2)])
    r.equal("A1 holdout keeps arrival order", held, [99, 99])
    trace = [5, 1, 5, 2, 9, 1, 1, 7]
    hist, held = split_trace(trace, 5)
    r.equal("A1 duplicates kept and sorted by shape", hist, sorted(Counter(trace[:5]).items()))
    r.equal("A1 holdout is the exact suffix", held, trace[5:])
    r.raises("A1 zero length rejected", ValueError, lambda: split_trace([1, 0], 1))
    r.raises("A1 empty holdout rejected", ValueError, lambda: split_trace([1, 2], 2))


def a2(r):
    result = summarize([Sample(4, 10, (1, 1, 200), (3, 4, 5))])[4]
    r.equal(
        "A2 median/MAD hide one outlier but keep it in MAD input",
        result,
        {"first_premium_ms": 9, "steady_ms": 1, "eager_ms": 4, "mad_ms": 0},
    )
    steady = (2.0, 3.0, 4.0, 10.0)
    result = summarize([Sample(8, 0.5, steady, (5.0, 6.0))])[8]
    med = (steady[1] + steady[2]) / 2  # even count: mean of the two middle values
    mad = sorted(abs(x - med) for x in steady)
    r.near("A2 even-count median", result["steady_ms"], med)
    r.near("A2 MAD around the median", result["mad_ms"], (mad[1] + mad[2]) / 2)
    r.near("A2 premium clamps at zero when first < steady", result["first_premium_ms"], 0.0)
    for name, sample in [
        ("no steady samples", Sample(4, 1, (), (1,))),
        ("NaN first call", Sample(4, float("nan"), (1,), (1,))),
        ("negative eager", Sample(4, 1, (1,), (-1,))),
    ]:
        r.raises(f"A2 rejects {name}", ValueError, lambda s=sample: summarize([s]))
    r.raises(
        "A2 rejects duplicate bounds", ValueError, lambda: summarize([Sample(1, 1, (1,), (1,))] * 2)
    )


def a3(r):
    hist = [(4, 3), (8, 2), (16, 1)]
    profiles = {
        4: {"first_premium_ms": 3, "steady_ms": 1},
        8: {"first_premium_ms": 5, "steady_ms": 2},
        16: {"first_premium_ms": 7, "steady_ms": 4},
    }
    costs = segment_costs(hist, profiles)
    for i in range(3):
        for j in range(i, 3):
            calls = sum(f for _, f in hist[i : j + 1])
            top = profiles[hist[j][0]]
            expected = top["first_premium_ms"] + calls * top["steady_ms"]
            r.near(
                f"A3 cost[{i},{j}] = one premium + calls * steady(largest)",
                costs.get((i, j)),
                expected,
            )
    r.raises(
        "A3 duplicate shapes rejected",
        ValueError,
        lambda: segment_costs([(4, 1), (4, 2)], profiles),
    )
    r.raises(
        "A3 unsorted histogram rejected",
        ValueError,
        lambda: segment_costs([(8, 1), (4, 2)], profiles),
    )


def a4(r):
    cases = [
        ([(2, 2)], [2, 2, 2]),
        ([(2, 2)], [1, 3]),
        ([], []),
        ([(2, 2), (4, 2)], [2, 2, 2, 4]),  # 0.25: same support, shifted mass
        ([(1, 1), (2, 1)], [2, 3]),  # 0.5: one new shape, one vanished
        ([(3, 1), (5, 3)], [5, 7, 3, 3]),
        ([(4, 1)], []),
    ]
    rng = random.Random(7)
    for _ in range(6):
        training = sorted(
            Counter(rng.choice([1, 2, 3, 5, 8]) for _ in range(rng.randint(1, 9))).items()
        )
        recent = [rng.choice([1, 2, 3, 4, 8, 13]) for _ in range(rng.randint(1, 9))]
        cases.append((training, recent))
    for training, recent in cases:
        expected = tv_oracle(training, recent)
        r.near(
            f"A4 TV({training}, {recent}) = {expected}",
            trace_shift(training, recent),
            float(expected),
            1e-12,
        )
    r.raises("A4 nonpositive shape rejected", ValueError, lambda: trace_shift([(0, 1)], [1]))


def a5(r):
    with patch.object(module, "route", first_fit):  # A5 must not depend on B2 being done
        result = padding_account([(3, 10), (7, 1), (20, 2)], [4, 8])
        hist = [(1, 5), (2, 3), (5, 2), (9, 4), (30, 1)]
        other = padding_account(hist, [2, 8, 16], max_padding=2.0)
    r.equal("A5 real rows", result.get("real_rows"), 77)
    r.equal("A5 executed rows (fallback runs unpadded)", result.get("executed_rows"), 88)
    r.equal("A5 fallback calls weighted by frequency", result.get("fallback_calls"), 2)
    r.near("A5 weighted padding fraction 11/88", result.get("padding_fraction"), 11 / 88)
    real = sum(n * f for n, f in hist)
    executed = sum((first_fit(n, [2, 8, 16], 2.0) or n) * f for n, f in hist)
    fallback = sum(f for n, f in hist if first_fit(n, [2, 8, 16], 2.0) is None)
    r.equal("A5 second histogram executed rows", other.get("executed_rows"), executed)
    r.equal("A5 second histogram fallback calls", other.get("fallback_calls"), fallback)
    r.near(
        "A5 second histogram fraction", other.get("padding_fraction"), (executed - real) / executed
    )


# ---------------------------------------------------------------- stage B
def b1(r):
    hist = [(2, 8), (5, 3), (9, 2), (17, 1)]
    for premium in [0, 1, 10, 100]:
        profiles = {n: {"first_premium_ms": premium, "steady_ms": 0.1 * n} for n, _ in hist}
        for capacity in [1, 2, 3, 4, 6]:
            actual = plan(hist, profiles, capacity)
            shapes = [n for n, _ in hist]
            best = min(plan_cost(hist, profiles, c) for c in all_plans(shapes, min(capacity, 4)))
            ok = (
                isinstance(actual, list)
                and actual == sorted(set(actual))
                and set(actual) <= set(shapes)
                and actual[-1:] == [17]
                and len(actual) <= capacity
            )
            r.check(
                f"B1 premium={premium} cap={capacity}: bounds valid",
                ok,
                "sorted shapes ending at 17",
                actual,
            )
            if ok:
                r.near(
                    f"B1 premium={premium} cap={capacity}: optimal",
                    plan_cost(hist, profiles, actual),
                    best,
                )
    # non-linear measured costs: a large shape can be cheaper per call than linear
    rng = random.Random(3)
    for trial in range(4):
        shapes = sorted(rng.sample(range(1, 40), 6))
        hist = [(n, rng.randint(1, 9)) for n in shapes]
        profiles = {
            n: {"first_premium_ms": rng.choice([0, 2, 5, 20]), "steady_ms": rng.uniform(0.5, 3)}
            for n in shapes
        }
        for capacity in [2, 3]:
            actual = plan(hist, profiles, capacity)
            best = min(plan_cost(hist, profiles, c) for c in all_plans(shapes, capacity))
            valid = (
                actual[-1:] == [shapes[-1]]
                and set(actual) <= set(shapes)
                and len(actual) <= capacity
            )
            r.check(
                f"B1 random#{trial} cap={capacity}: valid", valid, f"ends at {shapes[-1]}", actual
            )
            if valid:
                r.near(
                    f"B1 random#{trial} cap={capacity}: optimal",
                    plan_cost(hist, profiles, actual),
                    best,
                    1e-9,
                )
    tie = {2: {"first_premium_ms": 1, "steady_ms": 1}, 4: {"first_premium_ms": 1, "steady_ms": 2}}
    r.equal("B1 tie: fewer buckets win (5 = 5)", plan([(2, 1), (4, 1)], tie, 2), [4])
    tie = {n: {"first_premium_ms": 0, "steady_ms": n} for n in [1, 2, 3]}
    r.equal("B1 tie: last bucket starts earliest", plan([(1, 1), (2, 1), (3, 1)], tie, 2), [1, 3])
    r.equal("B1 empty histogram", plan([], {}, 1), [])
    r.raises("B1 capacity 0 rejected", ValueError, lambda: plan([(1, 1)], {1: tie[1]}, 0))


def b2(r):
    r.equal("B2 smallest fitting bound", route(3, [4, 8]), 4)
    r.equal("B2 exact bound is not padded upward", route(4, [4, 8]), 4)
    r.equal("B2 ratio exactly 2 is allowed", route(4, [8]), 8)
    r.equal("B2 ratio exactly 2 (length 2 -> 4)", route(2, [4, 8]), 4)
    r.equal("B2 ratio 8/3 > 2 falls back", route(3, [8]), None)
    r.equal("B2 1 -> 4 exceeds padding budget", route(1, [4, 8]), None)
    r.equal("B2 larger than every bound falls back", route(9, [4, 8]), None)
    r.equal("B2 custom budget 1.5 rejects 5 -> 8", route(5, [8], 1.5), None)
    bounds = [3, 7, 12, 30]
    for n in range(1, 35):
        r.equal(
            f"B2 route({n}) matches linear first-fit", route(n, bounds), first_fit(n, bounds, 2.0)
        )
    r.raises("B2 unsorted bounds rejected", ValueError, lambda: route(1, [8, 4]))
    r.raises("B2 zero length rejected", ValueError, lambda: route(0, [4]))


def b3(r):
    cache = ArtifactCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    r.equal("B3 lease returns value", cache.lease("a"), 1)
    r.check(
        "B3 insert evicts the oldest unpinned",
        cache.put("c", 3) and "b" not in cache.entries,
        "b evicted",
        list(cache.entries),
    )
    r.equal("B3 pinned key cannot be replaced", cache.put("a", 4), False)
    r.equal("B3 pinned value unchanged", cache.entries.get("a"), 1)
    cache.lease("c")
    before = list(cache.entries.items())
    r.equal("B3 all pinned: insert refused", cache.put("d", 4), False)
    r.equal("B3 refused insert leaves keys, values, order", list(cache.entries.items()), before)
    cache.lease("a", release=True)
    r.check(
        "B3 after release the unpinned key is the victim",
        cache.put("d", 4) and "a" not in cache.entries,
        "a evicted",
        list(cache.entries),
    )
    fresh = ArtifactCache(3)
    for k in "xyz":
        fresh.put(k, k)
    fresh.put("x", "x2")  # refresh moves x to MRU
    fresh.put("w", "w")
    r.equal("B3 refreshed key moves to MRU", list(fresh.entries), ["z", "x", "w"])


def b4(r):
    cache = ArtifactCache(1)
    cache.put("a", 1)
    cache.lease("a")
    cache.lease("a")
    cache.lease("a", release=True)
    r.equal("B4 two leases, one release: still pinned", cache.put("b", 2), False)
    cache.lease("a", release=True)
    r.equal("B4 last release unpins", cache.put("b", 2), True)
    r.raises("B4 release without lease", ValueError, lambda: cache.lease("b", release=True))
    r.equal("B4 lease of a missing key is None", cache.lease("zzz"), None)
    lru = ArtifactCache(2)
    lru.put("p", 1)
    lru.put("q", 2)
    lru.lease("p")
    lru.lease("p", release=True)  # p is now MRU and unpinned
    lru.put("s", 3)
    r.equal("B4 lease touches LRU order", sorted(lru.entries), ["p", "s"])


def b5(r):
    profile = {"first_premium_ms": 10, "steady_ms": 1, "eager_ms": 3, "mad_ms": 0.5}
    # conservative saving per call = 3 - (1 + 2 * 0.5) = 1 ms
    r.equal("B5 break-even exactly (10 calls) rejects", admit(profile, 10), False)
    r.equal("B5 11 calls admits", admit(profile, 11), True)
    r.equal("B5 margin 4 makes saving <= 0", admit(profile, 10000, margin=4), False)
    r.equal("B5 margin 0: 6 calls (saving 2 ms each) admits", admit(profile, 6, margin=0), True)
    r.equal("B5 margin 0: 5 calls equals premium rejects", admit(profile, 5, margin=0), False)
    slower = {"first_premium_ms": 0, "steady_ms": 3, "eager_ms": 3, "mad_ms": 0}
    r.equal("B5 compiled not faster: never admit", admit(slower, 10**6), False)
    r.raises("B5 negative prediction rejected", ValueError, lambda: admit(profile, -1))


def run(part):
    report = Report(part)
    if part in ["A", "all"]:
        for name, fn in [("A1", a1), ("A2", a2), ("A3", a3), ("A4", a4), ("A5", a5)]:
            report.group(name, fn)
    if part in ["B", "all"]:
        for name, fn in [("B1", b1), ("B2", b2), ("B3", b3), ("B4", b4), ("B5", b5)]:
            report.group(name, fn)
    return report.summary()


if __name__ == "__main__":
    import sys

    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else "all"))
