"""R1 driver (provided): measure prefill latency T(p, n) of Qwen3-0.6B on the local GPU.

p = cached prefix length (tokens whose KV is already in a DynamicCache), n = uncached
tokens fed in one forward. Protocol (labs/docs/authoring.md section 5):

* bf16, sdpa attention, model.eval(), torch.inference_mode(), batch 1.
* One DynamicCache is prefilled once at P_MAX tokens; for each p a snapshot is made with
  copy.deepcopy + crop(-(P_MAX - p)) (transformers 5.18: a negative argument removes that
  many tokens). Snapshots are never timed.
* The timed region is exactly one `model(ids[p:p+n], past_key_values=snap,
  logits_to_keep=1)` call between two CUDA events. logits_to_keep=1 computes the LM head
  only for the last position, as an engine does for prefill.
* After every call snap.crop(-n) rolls the cache back (a slice, no copy, not timed).
* Before every row the GPU is kept busy >= 300 ms with the same call (laptop P-state);
  then `--warmup` untimed and `--reps` timed calls. Rows with p75 > 1.5 * p25 are
  re-measured up to twice and keep stable=false if they never settle.
* n values are visited in a shuffled order per p so clock drift does not alias with n.
* Correctness gate: for a few (p, n) the last-position logits of the cached path must
  match a full forward over ids[:p+n] (same argmax or a near-tie, small max error).

HF DynamicCache appends with torch.cat, so every call also copies the p cached tokens
(112 KiB per token for Qwen3-0.6B in bf16). A paged engine (vLLM, SGLang) writes only
the n new tokens. The copy is reported as its own `copy` rows (same cat, no model) so the
fit can separate it; see README R1.

Usage (from labs/, in WSL with the GPU):
    flock /tmp/mlsys-gpu.lock python lab5/real/measure.py --out results/lab5-real/prefill.json
Default sweep: 7 values of p x 7 values of n, about 2 minutes on an RTX 4060 Laptop.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

P_DEFAULT = (0, 512, 1024, 2048, 3072, 4096, 6144)
N_DEFAULT = (16, 64, 128, 256, 512, 1024, 2048)


def models_dir() -> Path:
    return Path(os.environ.get("MLSYS_MODELS") or Path.home() / "models").expanduser()


def summarize(samples):
    """Nearest-rank p25/median/p75 and the stability flag (authoring.md section 5)."""
    ordered = sorted(samples)

    def rank(q):
        return ordered[max(1, math.ceil(q * len(ordered))) - 1]

    p25, p75 = rank(0.25), rank(0.75)
    return {
        "median_ms": rank(0.5),
        "p25_ms": p25,
        "p75_ms": p75,
        "n_samples": len(ordered),
        "stable": p75 <= 1.5 * p25,
    }


def environment(torch, transformers) -> dict:
    info = {
        "date": datetime.date.today().isoformat(),
        "host": platform.node(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,clocks.max.sm,clocks.max.mem",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        info["nvidia_smi"] = q.stdout.strip()
        info["driver"] = q.stdout.split(",")[0].strip()
    except (OSError, subprocess.SubprocessError):
        info["driver"] = "unknown"
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=str(models_dir() / "Qwen3-0.6B"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", default=",".join(map(str, P_DEFAULT)))
    ap.add_argument("--n", default=",".join(map(str, N_DEFAULT)))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--busy-ms", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, DynamicCache

    ps = sorted({int(x) for x in args.p.split(",")})
    ns = sorted({int(x) for x in args.n.split(",")})
    p_max = max(ps) + max(ns)
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = model.cuda().eval()
    cfg = model.config
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    ids = torch.randint(1000, 30000, (1, p_max), generator=gen).cuda()

    def busy(fn, reset):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) * 1000.0 < args.busy_ms:
            fn()
            reset()
        torch.cuda.synchronize()

    def timed(fn, reset):
        for attempt in range(3):
            busy(fn, reset)
            for _ in range(args.warmup):
                fn()
                reset()
            samples = []
            for _ in range(args.reps):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
                reset()
            stats = summarize(samples)
            if stats["stable"]:
                break
        return [round(x, 4) for x in samples], {**stats, "attempts": attempt + 1}

    rows, gate = [], []
    with torch.inference_mode():
        base = DynamicCache()
        model(ids[:, : max(ps)], past_key_values=base, use_cache=True, logits_to_keep=1)
        for p in sorted(ps, reverse=True):
            if p > 0:
                snap = copy.deepcopy(base)
                snap.crop(-(snap.get_seq_length() - p))
                assert snap.get_seq_length() == p
            order = list(ns)
            random.Random(p).shuffle(order)
            for n in order:
                x = ids[:, p : p + n]
                if p == 0:
                    holder = {"c": DynamicCache()}

                    def fn(x=x, holder=holder):
                        return model(
                            x, past_key_values=holder["c"], use_cache=True, logits_to_keep=1
                        )

                    def reset(holder=holder):
                        holder["c"] = DynamicCache()
                else:

                    def fn(x=x, snap=snap):
                        return model(x, past_key_values=snap, use_cache=True, logits_to_keep=1)

                    def reset(snap=snap, n=n, p=p):
                        snap.crop(-n)
                        assert snap.get_seq_length() == p

                samples, stats = timed(fn, reset)
                rows.append({"kind": "prefill", "p": p, "n": n, "samples_ms": samples, **stats})
                print(
                    json.dumps({k: rows[-1][k] for k in ("p", "n", "median_ms", "stable")}),
                    flush=True,
                )
                if n in (min(ns), max(ns)) and p in (min(ps), max(ps)):
                    cached = fn().logits[0, -1].float()
                    reset()
                    full = (
                        model(ids[:, : p + n], use_cache=False, logits_to_keep=1)
                        .logits[0, -1]
                        .float()
                    )
                    a, b = int(cached.argmax()), int(full.argmax())
                    err = float((cached - full).abs().max())
                    scale = float(full.abs().max())
                    gap = float(full[b] - full[a])
                    ok = (a == b or gap < 0.1) and err <= 0.05 * scale
                    gate.append(
                        {
                            "p": p,
                            "n": n,
                            "argmax_cached": a,
                            "argmax_full": b,
                            "max_abs_err": round(err, 4),
                            "logit_scale": round(scale, 3),
                            "ok": ok,
                        }
                    )
                    print(json.dumps(gate[-1]), flush=True)
            # Copy-only rows: the torch.cat that DynamicCache.update does per layer, without
            # the model. Lets the fit tell the HF-specific O(p) copy from attention work.
            if p > 0:
                k0 = snap.layers[0].keys
                new = torch.zeros(
                    k0.shape[0], k0.shape[1], 16, k0.shape[3], dtype=k0.dtype, device=k0.device
                )
                layers = [(layer.keys, layer.values) for layer in snap.layers]

                def copy_fn(layers=layers, new=new):
                    for k, v in layers:
                        torch.cat([k, new], dim=-2)
                        torch.cat([v, new], dim=-2)

                samples, stats = timed(copy_fn, lambda: None)
                rows.append({"kind": "copy", "p": p, "n": 16, "samples_ms": samples, **stats})
                print(json.dumps({"copy_p": p, "median_ms": stats["median_ms"]}), flush=True)
                del snap

    kv_bytes = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2
    out = {
        "meta": {
            **environment(torch, transformers),
            "model": Path(args.model).name,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "timer": "cudaEvent around one forward; snapshot/crop not timed",
            "busy_ms": args.busy_ms,
            "warmup": args.warmup,
            "reps": args.reps,
            "kv_bytes_per_token": kv_bytes,
            "layers": cfg.num_hidden_layers,
            "q_heads": cfg.num_attention_heads,
            "kv_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "label": "measured",
        },
        "gate": gate,
        "rows": rows,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    bad = [g for g in gate if not g["ok"]]
    unstable = [r for r in rows if not r["stable"]]
    print(
        f"wrote {path}: {len(rows)} rows, {len(unstable)} unstable, gate {len(gate) - len(bad)}/{len(gate)} ok"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
