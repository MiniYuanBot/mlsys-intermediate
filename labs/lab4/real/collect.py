"""R1/R2 driver (provided): collect an exactly-replayable acceptance + cost trace.

For every prompt: one target greedy generation (ground truth y) and ONE teacher-forced
draft pass give greedy acceptance flags (R1 greedy_flags). The cost model is measured
separately with CUDA events (R2 measure/summarize): verify latency T_v(L, k) for
k = 0..kmax and the draft step T_d(L), each after warmup, in both the CUDA-graph and the
eager regime. A subset of prompts is decoded for real with the hand-written fixed-k loop
so that stage R can compare replayed rounds and modeled time against measurements.
Optional: coupled sampling traces (R1 coupled_step) and an HF assisted-generation
cross-check (batch 1, assistant_confidence_threshold = 0 for fixed k).

Usage (from labs/, in WSL with the GPU):
    flock /tmp/mlsys-gpu.lock python lab4/real/collect.py --out results/lab4-real/trace.jsonl
A full default run takes a few minutes on an RTX 4060 Laptop (8 GB). Before every timed
row the GPU is kept busy for >= 300 ms with the same call; rows whose p75 > 1.5 * p25 are
re-measured up to twice and keep `stable: false` if they never settle.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from accept_student import coupled_step, greedy_flags, position_rates  # noqa: E402
from engine import DRAFT, EOS, TARGET, Pair, environment, load_prompts, models_dir  # noqa: E402
from profile_student import measure, summarize  # noqa: E402


class CudaTimer:
    """cudaEventRecord around the timed call; stop() synchronizes on the end event."""

    def start(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, begin) -> float:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        return begin.elapsed_time(end)


def keep_busy(fn, ms: float = 300.0):
    """Run fn back to back for >= ms of wall time so the laptop GPU leaves its low
    P-state before timing (memory clock 810 vs 8000 MHz on an RTX 4060 Laptop)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1000.0 < ms:
        fn()
        torch.cuda.synchronize()


def timed(fn, timer, warmup, reps, retries=2):
    """keep_busy + R2 measure/summarize; re-measure rows that are not stable."""
    for attempt in range(retries + 1):
        keep_busy(fn)
        samples = measure(fn, timer, warmup, reps)
        stats = summarize(samples)
        if stats["stable"]:
            break
    return samples, {**stats, "attempts": attempt + 1}


def cost_profile(pair: Pair, contexts, kmax, warmup, reps, regime, emit):
    """T_v(L, k) and T_d(L) at each context L. The cache is filled once per context;
    each timed call writes positions L..L+k and is rolled back by the next call."""
    timer = CudaTimer()
    gen = torch.Generator(device="cpu").manual_seed(1234)
    for context in contexts:
        filler = torch.randint(1000, 30000, (1, context), generator=gen).cuda()
        probe = torch.randint(1000, 30000, (1, kmax + 1), generator=gen).cuda()
        pair.target.eager(filler, 0)
        pair.draft.eager(filler, 0)
        if regime == "graph":
            tf = lambda k: lambda: pair.target.forward(probe[:, : k + 1], context)
            df = lambda: pair.draft.forward(probe[:, :1], context)
        else:
            tf = lambda k: lambda: pair.target.eager(probe[:, : k + 1], context)
            df = lambda: pair.draft.eager(probe[:, :1], context)
        order = list(range(kmax + 1))
        random.Random(context).shuffle(order)  # do not sweep k monotonically (clock drift)
        for k in order:
            samples, stats = timed(tf(k), timer, warmup, reps)
            emit(
                {
                    "type": "verify",
                    "regime": regime,
                    "context": context,
                    "k": k,
                    "samples_ms": [round(x, 4) for x in samples],
                    **stats,
                }
            )
        samples, stats = timed(df, timer, warmup, reps)
        emit(
            {
                "type": "draft",
                "regime": regime,
                "context": context,
                "samples_ms": [round(x, 4) for x in samples],
                **stats,
            }
        )


def coupled_trace(
    pair: Pair, prompt: list[int], max_new: int, temperature: float, seed: int, pid: int
):
    """Sampling trace with common random numbers: at decode step s the draft proposal,
    the accept test and the residual draw use uniforms keyed by (seed, prompt, s), so any
    chain-SD policy replaying this trace makes the same decisions (A = min(L, k))."""
    ids = torch.tensor([prompt], device="cuda")
    p = torch.softmax(pair.target.eager(ids, 0)[0, -1].float() / temperature, -1)
    q = torch.softmax(pair.draft.eager(ids, 0)[0, -1].float() / temperature, -1)
    tokens, flags, betas = [], [], []
    for s in range(max_new):
        rng = random.Random(f"{seed}:{pid}:{s}")
        y, accepted, beta = coupled_step(p, q, rng.random(), rng.random(), rng.random())
        tokens.append(y)
        if s > 0:  # step 0 is the prefill token, never speculated
            flags.append(int(accepted))
            betas.append(round(beta, 4))
        if y in EOS or s == max_new - 1:
            break
        pos = len(prompt) + s
        p = pair.probs_step("target", y, pos, temperature)
        q = pair.probs_step("draft", y, pos, temperature)
    return tokens, flags, betas


def hf_check(pair: Pair, prompt: list[int], k: int, max_new: int) -> dict:
    """Cross-check only: HF assisted generation, batch 1, fixed k, no confidence stop."""
    g = pair.draft_model.generation_config
    g.num_assistant_tokens, g.num_assistant_tokens_schedule = k, "constant"
    g.assistant_confidence_threshold = 0.0  # HF default 0.4 stops drafting early (DISCO-style)
    ids = torch.tensor([prompt], device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pair.target_model.generate(
        ids,
        assistant_model=pair.draft_model,
        do_sample=False,
        max_new_tokens=max_new,
        top_p=None,
        top_k=None,
        temperature=None,
    )
    torch.cuda.synchronize()
    return {"tokens": out[0, len(prompt) :].tolist(), "wall_ms": 1000 * (time.perf_counter() - t0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--models",
        help="directory holding Qwen3-0.6B and Qwen3-1.7B (default $MLSYS_MODELS or ~/models)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=36)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--contexts", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--spec-prompts", type=int, default=8, help="prompts decoded for real per k")
    ap.add_argument("--spec-k", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--sample-prompts", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-eager", action="store_true", help="skip the eager-regime cost profile")
    ap.add_argument(
        "--hf-check", type=int, default=0, metavar="N", help="HF assisted cross-check on N prompts"
    )
    a = ap.parse_args(argv)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pair = Pair(models_dir(a.models), kmax=a.kmax)
    lines = []
    emit = lambda row: lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    header = {
        "type": "header",
        "draft": DRAFT,
        "target": TARGET,
        "dtype": "bfloat16",
        "mode": "greedy",
        "max_new_tokens": a.max_new,
        "kmax": a.kmax,
        "warmup": a.warmup,
        "reps": a.reps,
        "date": datetime.date.today().isoformat(),
        "timer": "cudaEvent",
        "chat_template": "enable_thinking=False",
        "prompts_file": "real/prompts.jsonl",
        **environment(),
    }
    emit(header)
    prompts = load_prompts(limit=a.prompts)
    warm = pair.chat_ids("Warm up.")
    pair.greedy(warm, 16)
    pair.speculative(warm, 2, 16)
    emit({"type": "clocks_before", "nvidia_smi": environment().get("nvidia_smi", "")})

    print(f"[R2] cost profile, contexts {a.contexts}, k=0..{a.kmax}", flush=True)
    cost_profile(pair, a.contexts, a.kmax, a.warmup, a.reps, "graph", emit)
    if not a.no_eager:
        cost_profile(pair, a.contexts, a.kmax, a.warmup, max(5, a.reps // 2), "eager", emit)

    print(f"[R1] greedy traces for {len(prompts)} prompts, max_new={a.max_new}", flush=True)
    greedy_out = {}
    for row in prompts:
        ids = pair.chat_ids(row["prompt"])
        g = pair.greedy(ids, a.max_new)
        seq = ids + g["tokens"]
        argmax, top1 = pair.draft_teacher_forced(seq)
        flags = greedy_flags(argmax, len(ids), g["tokens"])
        rec = {
            "type": "prompt",
            "id": row["id"],
            "task": row["task"],
            "prompt_tokens": len(ids),
            "tokens": len(g["tokens"]),
            "decode_ms": round(g["decode_ms"], 3),
            "flags": flags,
            "draft_p": [round(top1[len(ids) + i], 3) for i in range(len(flags))],
        }
        if (
            row["id"] < a.spec_prompts
        ):  # same flags from cached width-1 draft steps (bf16 near-ties)
            step_flags = greedy_flags(
                pair.draft_argmax_stepwise(seq, len(ids)), len(ids), g["tokens"]
            )
            rec["stepwise_flags"] = step_flags
            rec["stepwise_mismatch"] = sum(x != y for x, y in zip(flags, step_flags))
        emit(rec)
        greedy_out[row["id"]] = (ids, g)

    print(
        f"[R1] hand-written fixed-k loop on {a.spec_prompts} prompts, k in {a.spec_k}", flush=True
    )
    one_step = lambda: pair.target.forward(torch.zeros(1, 1, dtype=torch.long, device="cuda"), 0)
    for row in prompts[: a.spec_prompts]:
        ids, _ = greedy_out[row["id"]]
        keep_busy(one_step)
        g = pair.greedy(ids, a.max_new)  # re-timed right after the busy warmup
        same_path = g["tokens"] == greedy_out[row["id"]][1]["tokens"]
        emit(
            {
                "type": "plain",
                "prompt": row["id"],
                "decode_ms": round(g["decode_ms"], 3),
                "tokens": len(g["tokens"]),
                "deterministic": same_path,
            }
        )
        for k in a.spec_k:
            keep_busy(one_step)
            s = pair.speculative(ids, k, a.max_new)
            diff = next(
                (i for i, (x, y) in enumerate(zip(s["tokens"], g["tokens"])) if x != y), None
            )
            if diff is None and len(s["tokens"]) != len(g["tokens"]):
                diff = min(len(s["tokens"]), len(g["tokens"]))
            margin = None
            if diff is not None and diff < len(
                g["tokens"]
            ):  # top-2 target logit gap where they split
                with torch.no_grad():
                    lg = (
                        pair.target_model(
                            torch.tensor([ids + g["tokens"][:diff]], device="cuda"), use_cache=False
                        )
                        .logits[0, -1]
                        .float()
                    )
                top = lg.topk(2).values
                margin = round(float(top[0] - top[1]), 4)
            emit(
                {
                    "type": "spec",
                    "prompt": row["id"],
                    "k": k,
                    "rounds": [list(r) for r in s["rounds"]],
                    "decode_ms": round(s["decode_ms"], 3),
                    "tokens": len(s["tokens"]),
                    "same_as_greedy": diff is None,
                    "first_diff": diff,
                    "diff_top2_margin": margin,
                }
            )

    if a.sample_prompts:
        print(
            f"[R1] coupled sampling traces on {a.sample_prompts} prompts, T={a.temperature}",
            flush=True,
        )
        for row in prompts[: a.sample_prompts]:
            ids = pair.chat_ids(row["prompt"])
            tokens, flags, betas = coupled_trace(
                pair, ids, a.max_new, a.temperature, a.seed, row["id"]
            )
            emit(
                {
                    "type": "sample",
                    "id": row["id"],
                    "task": row["task"],
                    "temperature": a.temperature,
                    "seed": a.seed,
                    "prompt_tokens": len(ids),
                    "tokens": len(tokens),
                    "flags": flags,
                    "beta": betas,
                }
            )

    for row in prompts[: a.hf_check]:
        ids, g = greedy_out[row["id"]]
        for k in (2, 4):
            h = hf_check(pair, ids, k, a.max_new)
            emit(
                {
                    "type": "hf_check",
                    "prompt": row["id"],
                    "k": k,
                    "wall_ms": round(h["wall_ms"], 1),
                    "same_as_greedy": h["tokens"] == g["tokens"],
                    "note": "eager HF, DynamicCache; wall clock",
                }
            )

    # quick look (the Rust bench does the real analysis)
    flat = [json.loads(x) for x in lines if '"type":"prompt"' in x]
    surv, haz = zip(*(position_rates(r["flags"], 6) for r in flat))
    mean = lambda col: [
        round(
            sum(v[j] for v in col if v[j] is not None) / max(1, sum(v[j] is not None for v in col)),
            3,
        )
        for j in range(6)
    ]
    print(f"mean over prompts  S_j={mean(surv)}  a_j={mean(haz)}")
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote {a.out} ({a.out.stat().st_size / 1024:.0f} KiB, {len(lines)} records) "
        f"in {time.perf_counter() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
