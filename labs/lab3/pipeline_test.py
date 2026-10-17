"""Independent exhaustive oracles for trace/planning/cache tasks."""

import itertools
import unittest
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


class FoundationTests(unittest.TestCase):
    def test_chronological_split(self):
        hist, held = split_trace([8, 4, 8, 99, 99], 3)
        self.assertEqual(hist, [(4, 1), (8, 2)])
        self.assertEqual(held, [99, 99])
        with self.assertRaises(ValueError):
            split_trace([1, 0], 1)

    def test_robust_profiles(self):
        result = summarize([Sample(4, 10, (1, 1, 200), (3, 4, 5))])[4]
        self.assertEqual(
            result, {"first_premium_ms": 9, "steady_ms": 1, "eager_ms": 4, "mad_ms": 0}
        )
        for sample in [Sample(4, 1, (), (1,)), Sample(4, float("nan"), (1,), (1,))]:
            with self.assertRaises(ValueError):
                summarize([sample])
        with self.assertRaises(ValueError):
            summarize([Sample(1, 1, (1,), (1,))] * 2)

    def test_interval_cost(self):
        hist = [(4, 3), (8, 2), (16, 1)]
        profiles = {
            4: {"first_premium_ms": 3, "steady_ms": 1},
            8: {"first_premium_ms": 5, "steady_ms": 2},
            16: {"first_premium_ms": 7, "steady_ms": 4},
        }
        costs = segment_costs(hist, profiles)
        self.assertEqual(costs[0, 2], 31)
        self.assertEqual(costs[1, 1], 9)
        with self.assertRaises(ValueError):
            segment_costs([(4, 1), (4, 2)], profiles)

    def test_distribution_shift(self):
        self.assertEqual(trace_shift([(2, 2)], [2, 2, 2]), 0)
        self.assertEqual(trace_shift([(2, 2)], [1, 3]), 1)
        self.assertEqual(trace_shift([], []), 0)

    def test_weighted_padding_accounting(self):
        # A uses a provided direct upper-bound oracle; it does not depend on B2.
        import src.pipeline_student as module
        from unittest.mock import patch

        def oracle(length, bounds, max_padding):
            return next((b for b in bounds if b >= length and b / length <= max_padding), None)

        with patch.object(module, "route", oracle):
            result = padding_account([(3, 10), (7, 1), (20, 2)], [4, 8])
        self.assertEqual(result["real_rows"], 77)
        self.assertEqual(result["executed_rows"], 88)
        self.assertEqual(result["fallback_calls"], 2)
        self.assertAlmostEqual(result["padding_fraction"], 11 / 88)


class CoreTests(unittest.TestCase):
    def test_dp_exhaustive(self):
        for premium in [0, 1, 10, 100]:
            hist = [(2, 8), (5, 3), (9, 2), (17, 1)]
            profiles = {n: {"first_premium_ms": premium, "steady_ms": 0.1 * n} for n, _ in hist}
            for capacity in [1, 2, 3, 4]:
                actual = plan(hist, profiles, capacity)

                def cost(bounds):
                    total = premium * len(bounds)
                    for n, frequency in hist:
                        bound = next(b for b in bounds if b >= n)
                        total += frequency * profiles[bound]["steady_ms"]
                    return total

                candidates = [
                    list(c) + [17]
                    for k in range(capacity)
                    for c in itertools.combinations([2, 5, 9], k)
                ]
                self.assertAlmostEqual(cost(actual), min(cost(c) for c in candidates))
                self.assertLessEqual(len(actual), capacity)
        self.assertEqual(plan([], {}, 1), [])

    def test_guarded_route(self):
        self.assertEqual(route(3, [4, 8]), 4)
        self.assertIsNone(route(1, [4, 8]))
        self.assertIsNone(route(9, [4, 8]))
        with self.assertRaises(ValueError):
            route(1, [8, 4])

    def test_pinned_transactional_lru(self):
        cache = ArtifactCache(2)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(cache.lease("a"), 1)
        self.assertTrue(cache.put("c", 3))
        self.assertNotIn("b", cache.entries)
        self.assertFalse(cache.put("a", 4))
        cache.lease("c")
        before = list(cache.entries.items())
        self.assertFalse(cache.put("d", 4))
        self.assertEqual(list(cache.entries.items()), before)
        cache.lease("a", release=True)
        self.assertTrue(cache.put("d", 4))
        self.assertNotIn("a", cache.entries)

    def test_lease_lifetime(self):
        cache = ArtifactCache(1)
        cache.put("a", 1)
        cache.lease("a")
        cache.lease("a")
        cache.lease("a", release=True)
        self.assertFalse(cache.put("b", 2))
        cache.lease("a", release=True)
        self.assertTrue(cache.put("b", 2))
        with self.assertRaises(ValueError):
            cache.lease("b", release=True)

    def test_admission_break_even(self):
        profile = {"first_premium_ms": 10, "steady_ms": 1, "eager_ms": 3, "mad_ms": 0.5}
        self.assertFalse(admit(profile, 10))
        self.assertTrue(admit(profile, 11))
        self.assertFalse(admit(profile, 10000, margin=4))


def run(part):
    suite = unittest.TestSuite()
    if part in ["A", "all"]:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(FoundationTests))
    if part in ["B", "all"]:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
