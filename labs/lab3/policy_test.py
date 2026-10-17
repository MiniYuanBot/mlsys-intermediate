import unittest
from itertools import combinations

from src.policy_student import bucket_for, plan_buckets


class PolicyTests(unittest.TestCase):
    def test_dp_matches_exhaustive_partitions(self):
        items = [(1, 8), (3, 4), (7, 2), (11, 1)]
        for compile_ms in [0, 0.1, 2, 100]:
            bounds = plan_buckets(items, compile_ms, 0.05)

            def cost(bs, compile_ms=compile_ms):
                return compile_ms * len(bs) + sum(f * bucket_for(n, bs) * 0.05 for n, f in items)

            possible = [list(c) + [11] for k in range(4) for c in combinations([1, 3, 7], k)]
            self.assertAlmostEqual(cost(bounds), min(map(cost, possible)))

    def test_edges(self):
        self.assertEqual(plan_buckets([], 1, 1), [])
        self.assertEqual(plan_buckets([(4, 2), (4, 3)], 1, 1), [4])
        self.assertEqual(bucket_for(3, [4, 8]), 4)
        self.assertIsNone(bucket_for(9, [4, 8]))
        with self.assertRaises(ValueError):
            plan_buckets([(0, 1)], 1, 1)
        with self.assertRaises(ValueError):
            plan_buckets([(1, 1)], -1, 1)


if __name__ == "__main__":
    unittest.main()
