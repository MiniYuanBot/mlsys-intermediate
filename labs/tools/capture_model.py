#!/usr/bin/env python3
"""Local-only single-request model trace and prefill K/V capture.

Diagnostic instrumentation synchronizes every token. These numbers are NOT
online serving TTFT/ITL and do not measure concurrent batching.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def cache_pairs(cache):
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "to_legacy_cache"):
        return list(cache.to_legacy_cache())
    return list(cache)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument(
        "--prompts", type=Path, required=True, help="UTF-8 text file, one prompt per line"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--max-input", type=int, default=128)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--capture-kv", action="store_true")
    a = p.parse_args()
    if a.max_input < 1 or a.max_new < 1:
        p.error("token limits must be positive")
    if not a.model.is_dir():
        p.error("--model must be an existing local directory")
    try:
        prompt_bytes = a.prompts.read_bytes()
        prompts = prompt_bytes.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        p.error(f"cannot read --prompts as UTF-8: {error}")
    if not any(line.strip() for line in prompts):
        p.error("--prompts must contain at least one nonblank line")
    if a.output.exists():
        p.error("--output already exists; use a new directory to preserve records")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a.output.mkdir(parents=True)
    dtype = (
        torch.float32
        if a.device == "cpu"
        else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(a.model), local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(a.model),
            local_files_only=True,
            trust_remote_code=False,
            **(
                {"dtype": dtype}
                if int(transformers.__version__.split(".")[0]) >= 5
                else {"torch_dtype": dtype}
            ),
        )
        .to(a.device)
        .eval()
    )
    config = a.model / "config.json"
    manifest = {
        "model_path": str(a.model.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "weight_identity": "local files; config hash alone is NOT a weight checksum",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": str(dtype),
        "device": a.device,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "max_input": a.max_input,
        "max_new": a.max_new,
        "chat_template": False,
        "scope": "single request; per-token synchronization; KV capture outside token timing; cold first prompt included",
    }
    (a.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    def sync():
        if a.device == "cuda":
            torch.cuda.synchronize()

    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    with torch.inference_mode(), (a.output / "requests.jsonl").open("w") as out:
        for request_id, text in enumerate(prompts):
            if not text.strip():
                continue
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=a.max_input)[
                "input_ids"
            ].to(a.device)
            prompt_len = inputs.shape[1]
            cache = None
            generated = []
            times = []
            if a.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for step in range(a.max_new):
                mask = torch.ones((1, prompt_len + step), dtype=torch.long, device=a.device)
                sync()
                start = time.perf_counter()
                result = model(
                    input_ids=inputs,
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                token = result.logits[:, -1].argmax(-1, keepdim=True)
                sync()
                times.append((time.perf_counter() - start) * 1000)
                cache = result.past_key_values
                if step == 0 and a.capture_kv:
                    pairs = cache_pairs(cache)
                    torch.save(
                        {
                            "layers": [
                                {"k": k.detach().cpu(), "v": v.detach().cpu()} for k, v in pairs
                            ],
                            "prompt_tokens": inputs.cpu(),
                            "shape_note": "[B,Hkv,T,D] for supported decoder caches",
                        },
                        a.output / f"prefill-kv-{request_id}.pt",
                    )
                value = int(token.item())
                generated.append(value)
                inputs = token
                if value in eos:
                    break
            row = {
                "request_id": request_id,
                "prompt_tokens": prompt_len,
                "output_tokens": len(generated),
                "prefill_and_first_argmax_ms": times[0],
                "decode_step_ms": times[1:],
                "generated_ids": generated,
                "generated_text": tokenizer.decode(generated),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if a.device == "cuda"
                else None,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ["generated_ids", "generated_text"]}
                ),
                flush=True,
            )
            del result, cache, inputs, mask, token
    print(a.output)


if __name__ == "__main__":
    main()
