"""Generate a small synthetic trace in the Mooncake FAST'25 schema (provided, stdlib only).

Each line: {"timestamp": ms, "input_length": tokens, "output_length": tokens,
"hash_ids": [...]}. As in the released traces, hash_ids are remapped chained hashes of
consecutive 512-token blocks: id j of a request depends on ids 0..j-1 and the block
content, so equal ids at equal positions mean an equal prefix, and the ids of a request
are its prefix-cache keys. The real traces (conversation / toolagent / synthetic,
12k-24k requests, average input 8.6k-12k tokens) are at
https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release (repository license
Apache-2.0; the README states no separate license for the trace files). They are not
downloaded by this lab; pass one to `bench R` if you fetch it yourself.

Workload mix (shares chosen to resemble the released traces' prefix structure, not fit
to them):
* chat (55%): a session picks one of 6 system prompts (2 blocks), then 1-6 turns; each
  turn's prompt is the previous prompt plus the previous answer and a new question.
* agent (30%): one of 3 tool specs (6 blocks) + a task context; 2-5 steps, each step
  appends the tool result.
* one-off (10%): a fresh 2-10 block document, no reuse.
* regenerate (5%): an earlier prompt re-sent verbatim a few seconds later (a "retry"
  button). Only these give FULL hits, where SGLang's and vLLM's full-hit rules differ.
Inputs are kept <= ~8k tokens, the range R1 measures on an 8 GB GPU.

    python lab5/real/make_trace.py --out lab5/real/data/synthetic-mooncake.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BLOCK = 512


class Chain:
    """Assigns remapped chained ids: (parent id, content) -> new int id."""

    def __init__(self):
        self.ids: dict[tuple[int, int], int] = {}
        self.next_content = 0

    def fresh_content(self, n: int) -> list[int]:
        out = list(range(self.next_content, self.next_content + n))
        self.next_content += n
        return out

    def hash_ids(self, contents: list[int]) -> list[int]:
        parent, out = -1, []
        for c in contents:
            key = (parent, c)
            if key not in self.ids:
                self.ids[key] = len(self.ids)
            parent = self.ids[key]
            out.append(parent)
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1200, help="number of requests")
    ap.add_argument("--rate", type=float, default=12.0, help="new sessions per second")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--max-blocks", type=int, default=16)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    chain = Chain()
    systems = [chain.fresh_content(2) for _ in range(6)]
    tools = [chain.fresh_content(6) for _ in range(3)]
    events = []  # (time_ms, contents, output_len)
    t = 0.0
    while len(events) < args.n:
        t += rng.expovariate(args.rate) * 1000.0
        kind = rng.random()
        at = t
        if kind < 0.55:
            prompt = list(systems[min(int(rng.paretovariate(1.2)) - 1, 5)])
            for _ in range(rng.randint(1, 6)):
                prompt += chain.fresh_content(rng.randint(1, 2))
                if len(prompt) > args.max_blocks:
                    break
                out_len = rng.randint(64, 600)
                events.append((at, list(prompt), out_len))
                prompt += chain.fresh_content(1)  # the answer joins the next prompt
                at += rng.uniform(5_000, 40_000)
        elif kind < 0.85:
            prompt = list(tools[rng.randrange(3)]) + chain.fresh_content(rng.randint(1, 3))
            for _ in range(rng.randint(2, 5)):
                if len(prompt) > args.max_blocks:
                    break
                events.append((at, list(prompt), rng.randint(16, 200)))
                prompt += chain.fresh_content(1)  # tool result
                at += rng.uniform(500, 4_000)
        elif kind < 0.95 or not events:
            events.append((at, chain.fresh_content(rng.randint(2, 10)), rng.randint(64, 400)))
        else:
            _, contents, out_len = events[rng.randrange(max(0, len(events) - 50), len(events))]
            events.append((at + rng.uniform(2_000, 10_000), list(contents), out_len))
    events.sort(key=lambda e: e[0])
    events = events[: args.n]
    t0 = events[0][0]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for at, contents, out_len in events:
            blocks = len(contents)
            # last block partial; a fixed function of the content so a re-sent prompt has the
            # same length
            input_length = (blocks - 1) * BLOCK + 1 + (contents[-1] * 7919) % BLOCK
            rec = {
                "timestamp": int(round(at - t0)),
                "input_length": input_length,
                "output_length": out_len,
                "hash_ids": chain.hash_ids(contents),
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print(
        f"wrote {path}: {len(events)} requests, {len(chain.ids)} distinct blocks, "
        f"span {(events[-1][0] - t0) / 1000:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
