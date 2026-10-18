"""Optional challenge: does the K/V int4 grouping axis matter on a real Qwen3 cache?

Prefills T tokens, fake-quantizes the whole cache (every layer, every KV head) with one
(axis, group) choice for K and one for V, then scores the next 32 tokens against the
unmodified model. Fake quantization = quantize + dequantize in FP32; it measures quality,
not storage or speed. Per-channel groups run along tokens, so the last T mod g tokens stay
in full precision, like a residual window.

    python lab1/axis_probe.py --model ~/models/Qwen3-0.6B --tokens 2048 --out results/lab1/axis.json

Needs torch + transformers and a local model; CUDA is used if available (about 2 GB for
Qwen3-0.6B at 2K tokens). The default text is this course's own lecture notes, so the
numbers are one document, not a benchmark: report them as a probe, then move to RULER/kvpress.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_tools"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from platform_tools import exit_on_missing, require_module

CONFIGS = [  # name, K (axis, group) or None = keep bf16, V (axis, group) or None
    ("bf16", None, None),
    ("K per-token g128", ("token", 128), None),
    ("K per-token g32", ("token", 32), None),
    ("K per-channel g64", ("channel", 64), None),
    ("V per-token g128", None, ("token", 128)),
    ("V per-channel g64", None, ("channel", 64)),
    ("KIVI layout: K channel g64 + V token g128", ("channel", 64), ("token", 128)),
    ("swapped: K token g128 + V channel g64", ("token", 128), ("channel", 64)),
]


def fake_quant(x, axis, g):
    """Asymmetric int4 over [B, H, T, D]; same rule as kv::quantize (offset=min, scale=range/15)."""
    x = x.float()
    b, h, n, d = x.shape
    if axis == "token":
        y = x.reshape(b, h, n, d // g, g)
        lo, hi = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    else:
        m = n // g * g
        y = x[:, :, :m].reshape(b, h, m // g, g, d)
        lo, hi = y.amin(3, keepdim=True), y.amax(3, keepdim=True)
    scale = (hi - lo) / 15
    scale = scale.where(hi > lo, scale.new_ones(()))
    q = ((y - lo) / scale).round().clamp(0, 15) * scale + lo
    if axis == "token":
        return q.reshape(b, h, n, d)
    out = x.clone()
    out[:, :, :m] = q.reshape(b, h, m, d)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument(
        "--text", type=Path, nargs="*", help="UTF-8 text files (default: course/modules/*.md)"
    )
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    for module in ("torch", "transformers"):
        require_module(module, 1, "challenge", "axis_probe.py loads a local Qwen3 model")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    files = a.text or sorted(
        (Path(__file__).resolve().parents[2] / "course" / "modules").glob("*.md")
    )
    text = "\n".join(f.read_text(encoding="utf-8") for f in files)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(device).eval()
    ids = tok(text, return_tensors="pt")["input_ids"][:, : a.tokens + 32].to(device)
    if ids.shape[1] < a.tokens + 32:
        p.error(f"text has only {ids.shape[1]} tokens; need --tokens + 32")
    rows = []
    with torch.inference_mode():
        cache = model(ids[:, : a.tokens], use_cache=True).past_key_values
        keys = [layer.keys.clone() for layer in cache.layers]
        values = [layer.values.clone() for layer in cache.layers]
        for name, kq, vq in CONFIGS:
            kerr = verr = 0.0
            for i, layer in enumerate(cache.layers):
                k = fake_quant(keys[i], *kq) if kq else keys[i].float()
                v = fake_quant(values[i], *vq) if vq else values[i].float()
                kerr += float((k - keys[i].float()).norm() / keys[i].float().norm()) / len(keys)
                verr += float((v - values[i].float()).norm() / values[i].float().norm()) / len(keys)
                layer.keys.copy_(k)
                layer.values.copy_(v)
            logits = model(ids[:, a.tokens :], past_key_values=cache, use_cache=True).logits.float()
            nll = (
                -torch.log_softmax(logits[0, :-1], -1)
                .gather(-1, ids[0, a.tokens + 1 :, None])
                .mean()
            )
            cache.crop(-(ids.shape[1] - a.tokens))  # drop the 32 scored tokens again
            rows.append(
                {
                    "config": name,
                    "next31_nll": round(float(nll), 4),
                    "k_rel_l2": round(kerr, 4),
                    "v_rel_l2": round(verr, 4),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        ratio = sum(
            float(
                (
                    k.float().abs().mean((0, 2)).amax(-1)
                    / k.float().abs().mean((0, 2)).median(-1).values
                ).mean()
            )
            for k in keys
        ) / len(keys)
    summary = {
        "model": str(a.model),
        "tokens": a.tokens,
        "device": device,
        "files": [str(f) for f in files],
        "k_channel_max_over_median_mean_abs": round(ratio, 2),
        "rows": rows,
        "scope": "fake quantization of one prefill cache; one text; measured, not a benchmark",
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    exit_on_missing(main)
