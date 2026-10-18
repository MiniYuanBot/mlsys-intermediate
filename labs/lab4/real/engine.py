"""Provided: Qwen3 draft/target pair on CUDA with static KV caches and CUDA graphs.

Why graphs: in eager PyTorch a 28-layer decode step is launch-bound, so the 0.6B draft
costs almost as much as the 1.7B target (c = T_d / T_v near 0.85 on an RTX 4060 Laptop)
and speculation can never pay off. Replaying one captured graph per input width removes
most of the launch overhead and brings c down to the order foundations lab5 measured
with its Rust+CUDA engine (about 0.5). Both regimes are real; the lab uses the graphed
one and the README asks you to report the eager one as a negative result.

Everything here is provided. Your TODOs are in accept_student.py (R1) and
profile_student.py (R2). Run as a script for a two-prompt smoke test:
    python real/engine.py --models ~/models
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

DRAFT = "Qwen3-0.6B"
TARGET = "Qwen3-1.7B"
EOS = (151645, 151643)  # <|im_end|>, <|endoftext|> (generation_config of both models)
HERE = Path(__file__).resolve().parent


def models_dir(explicit: str | None = None) -> Path:
    return Path(explicit or os.environ.get("MLSYS_MODELS") or Path.home() / "models").expanduser()


def load_prompts(path: Path | None = None, limit: int | None = None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (path or HERE / "prompts.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


class Graphed:
    """One model + StaticCache + one CUDA graph per input width.

    State contract: the cache holds valid KV for positions [0, start) of the sequence
    the caller is decoding; `forward(ids, start)` writes positions start..start+n-1 and
    returns logits for those n positions. Rolling back is just calling forward with a
    smaller start: stale entries beyond it are masked out and later overwritten.
    transformers 5.18's StaticLayer ignores cache_position and appends at its own
    device-side `cumulative_length`, so we overwrite that counter before every call
    (inside the graph for replays).
    """

    def __init__(self, model, widths, max_len: int):
        from transformers.cache_utils import StaticCache

        self.model, self.max_len, self.widths = model, max_len, tuple(widths)
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        dev = model.device
        self.cols = torch.arange(max_len, device=dev)
        self.start_t = torch.zeros((), dtype=torch.long, device=dev)
        self.inp = {n: torch.zeros(1, n, dtype=torch.long, device=dev) for n in widths}
        self.pos = {n: torch.zeros(n, dtype=torch.long, device=dev) for n in widths}
        self.mask = {n: torch.zeros(1, 1, n, max_len, dtype=torch.bool, device=dev) for n in widths}
        self.out, self.graphs = {}, {}

    def _call(self, ids, pos, mask):
        return self.model(
            input_ids=ids,
            attention_mask=mask,
            cache_position=pos,
            position_ids=pos[None],
            past_key_values=self.cache,
            use_cache=True,
        ).logits

    def _mask(self, start: int, n: int):
        rows = torch.arange(start, start + n, device=self.cols.device)
        return rows, (self.cols[None, :] <= rows[:, None])[None, None]

    @torch.no_grad()
    def eager(self, ids: torch.Tensor, start: int) -> torch.Tensor:
        n = ids.shape[1]
        assert start + n <= self.max_len, (
            f"sequence {start + n} exceeds static cache {self.max_len}"
        )
        for layer in self.cache.layers:
            if torch.is_tensor(getattr(layer, "cumulative_length", None)):
                layer.cumulative_length.fill_(start)
        rows, mask = self._mask(start, n)
        return self._call(ids, rows, mask)

    @torch.no_grad()
    def capture(self):
        self.eager(
            torch.zeros(1, 1, dtype=torch.long, device=self.cols.device), 0
        )  # lazy cache init
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for n in self.widths:
                for _ in range(2):
                    for layer in self.cache.layers:
                        layer.cumulative_length.copy_(self.start_t)
                    self._call(self.inp[n], self.pos[n], self.mask[n])
        torch.cuda.current_stream().wait_stream(side)
        for n in self.widths:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):  # private pool per graph: replays happen in any order
                for layer in self.cache.layers:
                    layer.cumulative_length.copy_(self.start_t)
                self.out[n] = self._call(self.inp[n], self.pos[n], self.mask[n])
            self.graphs[n] = graph
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, start: int) -> torch.Tensor:
        n = ids.shape[1]
        if n not in self.graphs:
            return self.eager(ids, start)
        assert start + n <= self.max_len, (
            f"sequence {start + n} exceeds static cache {self.max_len}"
        )
        rows, mask = self._mask(start, n)
        self.inp[n].copy_(ids)
        self.pos[n].copy_(rows)
        self.mask[n].copy_(mask)
        self.start_t.fill_(start)
        self.graphs[n].replay()
        return self.out[n]


class Pair:
    """Draft and target models sharing one tokenizer (Qwen3 tokenizer.json is identical)."""

    def __init__(self, root: Path, kmax: int = 8, max_len: int = 768, graphs: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        root = Path(root)
        self.tokenizer = AutoTokenizer.from_pretrained(root / TARGET)
        load = lambda name: (
            AutoModelForCausalLM.from_pretrained(root / name, dtype=torch.bfloat16).cuda().eval()
        )
        self.target_model, self.draft_model = load(TARGET), load(DRAFT)
        self.kmax, self.max_len, self.graphs = kmax, max_len, graphs
        self.target = Graphed(self.target_model, range(1, kmax + 2), max_len)
        self.draft = Graphed(self.draft_model, (1, 2), max_len)
        if graphs:
            self.target.capture()
            self.draft.capture()
        else:  # eager regime: same code path, no replay
            self.target.graphs.clear()
            self.draft.graphs.clear()
            self.target.eager(torch.zeros(1, 1, dtype=torch.long, device="cuda"), 0)
            self.draft.eager(torch.zeros(1, 1, dtype=torch.long, device="cuda"), 0)

    def chat_ids(self, text: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3 thinks by default; thinking traces change acceptance
        )
        return self.tokenizer(rendered, add_special_tokens=False)["input_ids"]

    # ---- plain greedy decoding (the k=0 baseline and the trace's ground truth) ----
    @torch.no_grad()
    def greedy(self, prompt: list[int], max_new: int) -> dict:
        ids = torch.tensor([prompt], device="cuda")
        first = int(self.target.eager(ids, 0)[0, -1].argmax())
        seq = prompt + [first]
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        while len(seq) - len(prompt) < max_new and seq[-1] not in EOS:
            logits = self.target.forward(torch.tensor([[seq[-1]]], device="cuda"), len(seq) - 1)
            seq.append(int(logits[0, -1].argmax()))
        stop.record()
        torch.cuda.synchronize()
        return {"tokens": seq[len(prompt) :], "decode_ms": start.elapsed_time(stop)}

    @torch.no_grad()
    def draft_teacher_forced(self, sequence: list[int]) -> tuple[list[int], list[float]]:
        """argmax and top-1 probability of q(. | sequence[:i+1]) for every i, from ONE
        eager draft forward. Entry i is the draft's guess for sequence[i+1]; aligning it
        with the generated tokens is your job in R1 (greedy_flags)."""
        logits = self.draft_model(torch.tensor([sequence], device="cuda"), use_cache=False).logits[
            0
        ]
        top = torch.softmax(logits.float(), -1).max(-1)
        return top.indices.tolist(), top.values.tolist()

    @torch.no_grad()
    def probs_step(self, which: str, token: int, start: int, temperature: float) -> torch.Tensor:
        """Next-token distribution (float32, temperature-scaled) after feeding `token`
        at position `start` through the draft or target graph (width 1)."""
        model = self.draft if which == "draft" else self.target
        logits = model.forward(torch.tensor([[token]], device="cuda"), start)[0, -1].float()
        return torch.softmax(logits / temperature, -1)

    @torch.no_grad()
    def draft_argmax_stepwise(self, sequence: list[int], prompt_len: int) -> list[int]:
        """Same quantity computed the way the speculative loop computes it (cached steps).

        Entries before prompt_len-1 are -1; used only to measure teacher-forcing mismatch.
        """
        out = [-1] * (prompt_len - 1)
        logits = self.draft.eager(torch.tensor([sequence[:prompt_len]], device="cuda"), 0)
        out.append(int(logits[0, -1].argmax()))
        for i in range(prompt_len, len(sequence)):
            logits = self.draft.forward(torch.tensor([[sequence[i]]], device="cuda"), i)
            out.append(int(logits[0, -1].argmax()))
        return out

    # ---- hand-written chain speculative decoding, greedy acceptance ----
    @torch.no_grad()
    def speculative(self, prompt: list[int], k: int, max_new: int) -> dict:
        """Fixed-k greedy speculative decoding. Returns committed tokens and per-round
        (proposed, accepted). Proposals are capped by max_new only (the loop cannot know
        where EOS will fall). Output equals greedy() up to bf16 near-ties between the
        width-1 and width-(k+1) target kernels (measured by collect.py, not assumed)."""
        assert 1 <= k <= self.kmax
        dev = "cuda"
        ids = torch.tensor([prompt], device=dev)
        first = int(self.target.eager(ids, 0)[0, -1].argmax())
        self.draft.eager(ids, 0)
        seq, d_len, rounds = prompt + [first], len(prompt), []
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        while len(seq) - len(prompt) < max_new and seq[-1] not in EOS:
            kk = min(k, max_new - (len(seq) - len(prompt)) - 1)
            if kk == 0:
                logits = self.target.forward(torch.tensor([[seq[-1]]], device=dev), len(seq) - 1)
                seq.append(int(logits[0, -1].argmax()))
                rounds.append((0, 0))
                continue
            # The draft has not seen the last 1 (after a rejection) or 2 (after a fully
            # accepted round) committed tokens; feed them in one width-1 or width-2 call.
            # bf16 rounds differently at width 2, so near-tie guesses can depend on the
            # previous round's outcome: replay is exact up to such ties (measured).
            pending = torch.tensor([seq[d_len:]], device=dev)
            logits = (
                self.draft.forward(pending, d_len)
                if pending.shape[1] <= 2
                else self.draft.eager(pending, d_len)
            )
            d_len = len(seq)
            drafts = [logits[0, -1].argmax().view(1, 1)]
            for _ in range(kk - 1):  # stays on the GPU: no host sync inside the draft loop
                drafts.append(self.draft.forward(drafts[-1], d_len)[0, -1].argmax().view(1, 1))
                d_len += 1
            proposal = torch.cat(drafts, dim=1)
            verify_in = torch.cat([torch.tensor([[seq[-1]]], device=dev), proposal], dim=1)
            target_next = self.target.forward(verify_in, len(seq) - 1)[0].argmax(-1)
            proposal, target_next = proposal[0].tolist(), target_next.tolist()  # one sync per round
            accepted = 0
            while accepted < kk and proposal[accepted] == target_next[accepted]:
                accepted += 1
            base = len(seq)
            seq.extend(proposal[:accepted] + [target_next[accepted]])
            d_len = base + min(accepted, kk - 1)  # draft KV valid for seq[:d_len]
            rounds.append((kk, accepted))
        stop.record()
        torch.cuda.synchronize()
        gen = seq[len(prompt) :]
        for i, t in enumerate(gen):  # EOS inside an accepted prefix ends generation there
            if t in EOS:
                gen = gen[: i + 1]
                break
        return {"tokens": gen[:max_new], "rounds": rounds, "decode_ms": start.elapsed_time(stop)}


def environment() -> dict:
    import subprocess

    import transformers

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
    }
    info["device"] = torch.cuda.get_device_name(0)
    try:
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,clocks.sm,temperature.gpu,power.draw",
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="two-prompt smoke test")
    ap.add_argument("--models")
    ap.add_argument("--eager", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    pair = Pair(models_dir(a.models), graphs=not a.eager)
    print(
        f"loaded+captured in {time.perf_counter() - t0:.1f}s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB"
    )
    for row in load_prompts(limit=2):
        p = pair.chat_ids(row["prompt"])
        g = pair.greedy(p, 64)
        s = pair.speculative(p, 3, 64)
        print(
            row["task"],
            len(p),
            len(g["tokens"]),
            f"{g['decode_ms']:.0f}ms",
            f"{s['decode_ms']:.0f}ms",
            "same" if g["tokens"] == s["tokens"] else "DIFF",
            s["rounds"][:6],
        )
        print(repr(pair.tokenizer.decode(g["tokens"]))[:120])
