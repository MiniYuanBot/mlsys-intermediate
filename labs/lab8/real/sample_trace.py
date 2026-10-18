"""Provided (not a TODO): sample a REAL rollout-length trace with Qwen3-0.6B.

This is plain transformers sampling, NOT GRPO training: the weights never change, so
every row has policy_version = 0. The trace is what one synchronous GRPO rollout step
would see before its first update: for every prompt one group of G samples generated
in ONE `model.generate` call (batch = group), temperature 1.0, no top-k/top-p.

Timing: `gen_ms` is wall-clock time (time.perf_counter) from just before the group's
generate() call to the decode step at which that sample emitted EOS (or hit
max_new_tokens). It includes the group's prefill and is measured with batch = group, so
it is a per-sample completion time inside a batched call, not an isolated latency.
A StoppingCriteria hook records one timestamp per decode step; it adds one host
sync per step, which HF generate already does to test for finished sequences.
Before the first group the GPU is kept busy for >= 300 ms with the same generate call
(laptop GPUs idle in a low P-state); clocks are recorded before and after.

Reward: rule-based. The model is asked to end with "#### <integer>"; the integer after
the last "####" (else the last number in the text, flagged `fallback`) is compared with
the reference answer in prompts.jsonl (problems written for this course, MIT license).

Usage (from labs/, in WSL with the GPU; takes the shared GPU lock):
    flock /tmp/mlsys-gpu.lock python lab8/real/sample_trace.py \
        --out lab8/real/data/qwen3-0.6b-arith-g8.jsonl
Output schema (one JSON object per line):
  {"type": "header", ...environment, sampling settings, label...}
  {"type": "sample", "prompt_id", "group", "round", "sample_idx", "policy_version",
   "response_tokens", "finish_step", "truncated", "gen_ms", "reward", "parsed", "fallback"}
  {"type": "group", "group", "prompt_id", "round", "steps", "total_ms", "prefill_ms",
   "median_step_ms"}
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABEL = "sampled with Qwen3-0.6B, not from GRPO training"
INSTRUCTION = (
    "Solve the problem. Think step by step, then give the final answer as an integer "
    "on the last line in the form '#### <integer>'.\n\n"
)


def models_dir(explicit: str | None) -> Path:
    return Path(explicit or os.environ.get("MLSYS_MODELS") or Path.home() / "models").expanduser()


def load_prompts(limit: int | None) -> list[dict]:
    lines = (HERE / "prompts.jsonl").read_text("utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return rows[:limit] if limit else rows


def nvidia_smi() -> str:
    try:
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,clocks.sm,clocks.mem,pstate,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return q.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def score(text: str, answer: int) -> tuple[float, str | None, bool]:
    """(reward, parsed answer, used_fallback). Exact integer match after '####'."""
    tail = text.rsplit("####", 1)[1] if "####" in text else None
    pool = NUMBER.findall(tail if tail is not None else text)
    if not pool:
        return 0.0, None, tail is None
    token = (pool[0] if tail is not None else pool[-1]).replace(",", "")
    try:
        value = float(token)
    except ValueError:
        return 0.0, token, tail is None
    return float(value == answer), token, tail is None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models")
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

    if not torch.cuda.is_available():
        raise SystemExit("sample_trace.py needs a CUDA GPU (run it in WSL/Linux)")
    path = models_dir(args.models) / args.model
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    prompts = load_prompts(args.limit)

    def encode(question: str):
        text = tok.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION + question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tok(text, return_tensors="pt").input_ids.cuda()

    class StepClock(StoppingCriteria):
        """Called once per generated token: timestamp + first EOS step per sequence."""

        def __init__(self, width: int):
            self.stamps: list[float] = []
            self.done_at = [None] * width
            self.width = width

        def __call__(self, input_ids, scores, **kwargs):
            last = input_ids[:, -1].tolist()
            self.stamps.append(time.perf_counter())
            step = len(self.stamps)
            for i, t in enumerate(last):
                if self.done_at[i] is None and t in eos:
                    self.done_at[i] = step
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def generate(ids, clock, seed):
        torch.manual_seed(seed)
        return model.generate(
            ids,
            do_sample=True,
            temperature=args.temperature,
            top_k=0,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.group,
            stopping_criteria=[clock],
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    clocks_before = nvidia_smi()
    warm = encode(prompts[0]["question"])
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1000 < 300:  # spin-warm >= 300 ms, same launch
        generate(warm, StepClock(args.group), seed=0)
        torch.cuda.synchronize()
    header = {
        "type": "header",
        "label": LABEL,
        "model": args.model,
        "dtype": "bfloat16",
        "group_size": args.group,
        "rounds": args.rounds,
        "prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": 0,
        "top_p": 1.0,
        "seed": args.seed,
        "policy_version": 0,
        "chat_template": "enable_thinking=False",
        "prompts_file": "real/prompts.jsonl",
        "timer": "time.perf_counter wall-clock per sample, batch = group (one generate call per group)",
        "date": datetime.date.today().isoformat(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "nvidia_smi_before": clocks_before,
        "driver": clocks_before.split(",")[0].strip(),
    }
    rows = [header]
    group_id = 0
    started = time.perf_counter()
    for rnd in range(args.rounds):
        for p in prompts:
            ids = encode(p["question"])
            clock = StepClock(args.group)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            seqs = generate(ids, clock, seed=args.seed + 1000 * rnd + p["id"])
            torch.cuda.synchronize()
            end = time.perf_counter()
            new = seqs[:, ids.shape[1] :]
            steps = len(clock.stamps)
            for i in range(args.group):
                finish = clock.done_at[i]
                truncated = finish is None
                finish = steps if truncated else finish
                text = tok.decode(new[i, :finish], skip_special_tokens=True)
                reward, parsed, fallback = score(text, p["answer"])
                rows.append(
                    {
                        "type": "sample",
                        "prompt_id": p["id"],
                        "group": group_id,
                        "round": rnd,
                        "sample_idx": i,
                        "policy_version": 0,
                        "response_tokens": finish,
                        "finish_step": finish,
                        "truncated": truncated,
                        "gen_ms": round((clock.stamps[finish - 1] - begin) * 1000, 3),
                        "reward": reward,
                        "parsed": parsed,
                        "fallback": fallback,
                    }
                )
            gaps = [b - a for a, b in zip(clock.stamps, clock.stamps[1:])]
            rows.append(
                {
                    "type": "group",
                    "group": group_id,
                    "prompt_id": p["id"],
                    "round": rnd,
                    "steps": steps,
                    "total_ms": round((end - begin) * 1000, 3),
                    "prefill_ms": round((clock.stamps[0] - begin) * 1000, 3),
                    "median_step_ms": round(statistics.median(gaps) * 1000, 3) if gaps else None,
                }
            )
            print(
                f"round {rnd} prompt {p['id']:2d}: steps {steps:3d} "
                f"{(end - begin):6.2f}s reward {sum(r['reward'] for r in rows[-args.group - 1 : -1]):.0f}/{args.group}",
                flush=True,
            )
            group_id += 1
    rows.append(
        {
            "type": "footer",
            "nvidia_smi_after": nvidia_smi(),
            "wall_s": round(time.perf_counter() - started, 1),
        }
    )
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
