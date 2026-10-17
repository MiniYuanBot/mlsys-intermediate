import argparse
import copy
import itertools
import unittest

import torch
from model import Block, Model
from src.train_student import (
    CpuOffload,
    LoRALinear,
    SavedInventory,
    adapter_state,
    calibrate_blocks,
    inject_lora,
    load_adapters,
    measure_policy,
    parameter_bytes,
    plan_keep,
    plan_modes,
    run_blocks,
    run_policy,
    train_accumulated,
)
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
parser.add_argument("--part", choices=["all", "A", "B", "C"], default="all")
args = parser.parse_args()
torch.set_num_threads(2)


class ATests(unittest.TestCase):
    def test_a1_frozen_base_input_and_adapter_gradients(self):
        layer = LoRALinear(7, 5, rank=2, bias=True).to(args.device)
        x = torch.randn(3, 7, device=args.device, requires_grad=True)
        torch.testing.assert_close(layer(x), x @ layer.weight.T + layer.bias)
        layer(x).square().sum().backward()
        self.assertIsNone(layer.weight.grad)
        self.assertIsNone(layer.bias.grad)
        self.assertGreater(float(layer.b.grad.abs().sum()), 0)
        self.assertEqual(float(layer.a.grad.abs().sum()), 0)
        with torch.no_grad():
            layer.b.normal_()
        x.grad = None
        upstream = torch.randn(3, 5, device=args.device)
        (layer(x) * upstream).sum().backward()
        expected = upstream @ layer.weight + layer.scale * (upstream @ layer.b @ layer.a)
        torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-6)

    def test_a2_a3_merge_idempotent_restore(self):
        layer = LoRALinear(7, 5, rank=2).to(args.device)
        with self.assertRaises(ValueError):
            layer.merge()
        layer.eval()
        with torch.no_grad():
            layer.b.normal_()
        x = torch.randn(4, 7, device=args.device)
        expected, weight = layer(x), layer.weight.clone()
        for _ in range(3):
            layer.merge()
            layer.merge()
            torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)
            layer.unmerge()
            layer.unmerge()
            self.assertTrue(torch.equal(weight, layer.weight))

    def test_a4_injection_and_atomic_rejection(self):
        model = nn.Sequential(nn.Linear(7, 9), nn.Sequential(nn.ReLU(), nn.Linear(9, 5))).to(
            args.device
        )
        x = torch.randn(4, 7, device=args.device)
        expected = model(x).detach()
        old = model[0]
        with self.assertRaises(ValueError):
            inject_lora(model, ["0", "missing"])
        self.assertIs(model[0], old)
        self.assertTrue(model[0].weight.requires_grad)
        self.assertEqual(inject_lora(model, ["0", "1.1"], 2), ["0", "1.1"])
        torch.testing.assert_close(model(x), expected)
        self.assertEqual(
            {n for n, p in model.named_parameters() if p.requires_grad},
            {"0.a", "0.b", "1.1.a", "1.1.b"},
        )

    def test_a5_a6_save_restore_transaction(self):
        model = Model(width=8, layers=2).to(args.device)
        state = adapter_state(model)
        snapshot = copy.deepcopy(state)
        with torch.no_grad():
            model.blocks[0].up.a.add_(1)
        self.assertTrue(torch.equal(state["blocks.0.up"]["a"], snapshot["blocks.0.up"]["a"]))
        load_adapters(model, state)
        for name, entry in adapter_state(model).items():
            torch.testing.assert_close(entry["a"], state[name]["a"])
        bad = copy.deepcopy(state)
        bad["blocks.0.up"]["a"].add_(2)
        bad["blocks.1.down"]["b"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            load_adapters(model, bad)
        torch.testing.assert_close(
            adapter_state(model)["blocks.0.up"]["a"], state["blocks.0.up"]["a"]
        )
        bad = copy.deepcopy(state)
        bad["blocks.1.down"]["a"].fill_(float("nan"))
        with self.assertRaises(ValueError):
            load_adapters(model, bad)
        bad = copy.deepcopy(state)
        bad["blocks.0.up"]["a"].add_(2)
        source = bad["blocks.1.down"]["b"]
        bad["blocks.1.down"]["b"] = torch.full(source.shape, 1e100, dtype=torch.float64)
        with self.assertRaises(ValueError):
            load_adapters(model, bad)
        torch.testing.assert_close(
            adapter_state(model)["blocks.0.up"]["a"], state["blocks.0.up"]["a"]
        )

    def test_a7_bytes(self):
        layer = LoRALinear(7, 5, rank=2, bias=True)
        self.assertEqual(
            parameter_bytes(layer), {"trainable": (14 + 10) * 4, "frozen": (35 + 5) * 4}
        )


class BTests(unittest.TestCase):
    def test_b1_checkpoint_dropout(self):
        torch.manual_seed(19)
        base = Model(width=16, layers=3).to(args.device)
        x = torch.randn(2, 5, 16, device=args.device)
        target = torch.randn_like(x)
        expected = None
        for keep in [[True] * 3, [False] * 3, [True, False, True]]:
            model = copy.deepcopy(base)
            torch.manual_seed(27)
            loss = (model(x, keep) - target).square().mean()
            loss.backward()
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
            if expected is None:
                expected = (loss.detach(), grads)
            else:
                torch.testing.assert_close(loss, expected[0])
                for name, grad in grads.items():
                    torch.testing.assert_close(grad, expected[1][name], rtol=1e-4, atol=1e-6)
        with self.assertRaises(ValueError):
            run_blocks(base.blocks, x, [True])

    def test_b2_knapsack_exhaustive(self):
        sizes, costs = [2, 5, 3, 7], [1, 9, 4, 6]
        for budget in range(18):
            chosen = plan_keep(sizes, costs, budget)
            self.assertLessEqual(sum(s for s, k in zip(sizes, chosen, strict=True) if k), budget)
            gain = sum(c for c, k in zip(costs, chosen, strict=True) if k)
            optimal = max(
                sum(c * k for c, k in zip(costs, bits, strict=True))
                for bits in itertools.product([0, 1], repeat=4)
                if sum(s * k for s, k in zip(sizes, bits, strict=True)) <= budget
            )
            self.assertEqual(gain, optimal)
        self.assertEqual(plan_keep([], [], 0), [])
        with self.assertRaises(ValueError):
            plan_keep([0], [1], 2)

    def test_b3_storage_dedup_parameter_views(self):
        p = nn.Parameter(torch.ones(8, device=args.device))
        x = torch.ones(16, device=args.device, requires_grad=True)
        inventory = SavedInventory([p])
        for value in (p, p[:3], x, x[4:8], x[:8]):
            torch.testing.assert_close(inventory.unpack(inventory.pack(value)), value)
        self.assertEqual(inventory.saves, 5)
        self.assertEqual(inventory.bytes, x.untyped_storage().nbytes())

    def test_b4_modes_exhaustive_and_infeasible(self):
        options = [
            {"retain": (5, 1), "checkpoint": (1, 4), "offload": (0, 6)},
            {"retain": (6, 2), "checkpoint": (2, 3), "offload": (0, 8)},
            {"retain": (4, 1), "checkpoint": (1, 5)},
        ]
        for budget in range(1, 16):
            modes, used, cost = plan_modes(options, budget)
            feasible = [
                (sum(options[i][m][1] for i, m in enumerate(path)), path)
                for path in itertools.product(*[list(o) for o in options])
                if sum(options[i][m][0] for i, m in enumerate(path)) <= budget
            ]
            self.assertEqual(cost, min(item[0] for item in feasible))
            self.assertEqual(used, sum(o[m][0] for o, m in zip(options, modes, strict=True)))
            self.assertLessEqual(used, budget)
        with self.assertRaises(ValueError):
            plan_modes(options, 0)

    def test_b5_real_calibration(self):
        blocks = nn.ModuleList([Block(8, dropout=0), Block(8, dropout=0)]).to(args.device)
        rows = calibrate_blocks(blocks, torch.randn(2, 3, 8, device=args.device), repeats=1)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertGreater(row["retain"]["ms"], 0)
            self.assertGreater(row["retain"]["saved_bytes"], row["checkpoint"]["saved_bytes"])


class CTests(unittest.TestCase):
    def test_c1_c2_pack_ownership_and_repeated_unpack(self):
        p = nn.Parameter(torch.ones(8, device=args.device))
        owner = CpuOffload([p])
        tensor = torch.randn(4, 5, device=args.device)
        payload = owner.pack(tensor)
        for _ in range(2):
            restored = owner.unpack(payload)
            torch.testing.assert_close(restored, tensor)
            self.assertEqual(restored.device, tensor.device)
        owner.pack(p[:3])
        expected = 1 if args.device == "cuda" else 0
        self.assertEqual(owner.d2h_copies, expected)
        self.assertEqual(owner.h2d_copies, expected * 2)
        self.assertEqual(owner.packed_bytes, expected * tensor.numel() * 4)
        if expected:
            tensor.add_(100)
            self.assertFalse(torch.equal(owner.unpack(payload), tensor))

    def test_c3_all_modes_loss_gradients_dropout(self):
        torch.manual_seed(31)
        base = nn.ModuleList([Block(8), Block(8), Block(8)]).to(args.device)
        x = torch.randn(2, 4, 8, device=args.device, requires_grad=True)
        target = torch.randn_like(x)
        expected = None
        for modes in [
            ["retain"] * 3,
            ["checkpoint"] * 3,
            ["offload"] * 3,
            ["retain", "offload", "checkpoint"],
        ]:
            blocks = copy.deepcopy(base)
            value = x.detach().clone().requires_grad_(True)
            owner = CpuOffload(blocks.parameters())
            torch.manual_seed(73)
            loss = (run_policy(blocks, value, modes, owner) - target).square().mean()
            loss.backward()
            gradients = [value.grad] + [p.grad for p in blocks.parameters() if p.requires_grad]
            if expected is None:
                expected = [g.clone() for g in gradients]
            for actual, reference in zip(gradients, expected, strict=True):
                torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-6)
            if args.device == "cuda" and "offload" in modes:
                self.assertGreater(owner.d2h_copies, 0)
                self.assertEqual(owner.d2h_copies, owner.h2d_copies)

    def test_c4_unequal_microbatches_optimizer_equivalence(self):
        torch.manual_seed(41)
        a = nn.ModuleList([Block(8, dropout=0), Block(8, dropout=0)]).to(args.device)
        b = copy.deepcopy(a)
        x = torch.randn(5, 3, 8, device=args.device)
        y = torch.randn_like(x)
        oa = torch.optim.SGD(a.parameters(), lr=0.03)
        ob = torch.optim.SGD(b.parameters(), lr=0.03)
        first = train_accumulated(a, oa, [(x, y)], ["retain"] * 2, max_grad_norm=1)
        second = train_accumulated(
            b,
            ob,
            [(x[:2], y[:2]), (x[2:], y[2:])],
            ["checkpoint", "offload"],
            CpuOffload(b.parameters()),
            max_grad_norm=1,
        )
        self.assertAlmostEqual(first, second, places=5)
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-6)
        with self.assertRaises(ValueError):
            train_accumulated(a, oa, [], ["retain"] * 2)

    def test_c5_measured_step(self):
        blocks = nn.ModuleList([Block(8, dropout=0)]).to(args.device)
        x = torch.randn(2, 3, 8, device=args.device)
        row = measure_policy(blocks, [(x, torch.zeros_like(x))], ["offload"], repeats=2)
        self.assertGreater(row["step_ms"], 0)
        self.assertEqual(len(row["losses"]), 2)
        self.assertEqual(row["peak_allocated_bytes"] is None, args.device == "cpu")
        if args.device == "cuda":
            self.assertGreater(row["d2h_bytes_per_step"], 0)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for stage, cls in [("A", ATests), ("B", BTests), ("C", CTests)]:
        if args.part in (stage, "all"):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
