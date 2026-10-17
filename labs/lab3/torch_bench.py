"""Real eager/torch.compile benchmark. Run each mode in a fresh process."""

import argparse
import json
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["eager", "static", "dynamic", "bucket"], required=True)
p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
a = p.parse_args()
torch.manual_seed(17)


def model(x, r, g):
    z = x + r
    return z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5) * g


fn = model if a.mode == "eager" else torch.compile(model, dynamic=(a.mode == "dynamic"))


def sync():
    if a.device == "cuda":
        torch.cuda.synchronize()


for n in [1, 4, 8, 17, 32, 64, 9, 33, 80]:
    padded = next((b for b in [8, 32, 128] if b >= n), n) if a.mode == "bucket" else n
    # Pair the actual (unpadded) rows across all four modes.
    torch.manual_seed(17 + n)
    real_x = torch.randn(n, 1024, device=a.device)
    real_r = torch.randn_like(real_x)
    x = torch.nn.functional.pad(real_x, (0, 0, 0, padded - n))
    r = torch.nn.functional.pad(real_r, (0, 0, 0, padded - n))
    g = torch.ones(1024, device=a.device)
    sync()
    t = time.perf_counter()
    out = fn(x, r, g)
    sync()
    cold = (time.perf_counter() - t) * 1000
    torch.testing.assert_close(out[:n], model(real_x, real_r, g))
    for _ in range(5):
        fn(x, r, g)
    sync()
    t = time.perf_counter()
    for _ in range(100):
        fn(x, r, g)
    sync()
    warm = (time.perf_counter() - t) * 10
    print(
        json.dumps(
            {
                "mode": a.mode,
                "rows": n,
                "padded_rows": padded,
                "first_call_ms": cold,
                "warm_ms": warm,
                "torch": torch.__version__,
            }
        )
    )
