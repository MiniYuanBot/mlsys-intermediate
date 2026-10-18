"""Stage R tests for the Python TODOs (R1 acceptance extraction, R2 latency measurement).

CPU checks use hand-made inputs and independent oracles (enumeration, Monte Carlo
against the exact target distribution, fake timers). With --device cuda the script also
loads Qwen3-0.6B/1.7B and checks R1 against the hand-written speculative loop on real
prompts. Prints PASS/FAIL per check, never stops at the first failure, exits 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail=lambda: ""):
    if ok:
        print(f"PASS R/{name}", flush=True)
    else:
        line = f"FAIL R/{name}: {detail()}"
        print(line, flush=True)
        FAILURES.append(line)


def guarded(name, fn):
    """Run one group of checks; a NotImplementedError or crash is one FAIL, not an abort."""
    try:
        fn()
    except NotImplementedError as e:
        check(name, False, lambda: f"TODO {e} not implemented")
    except Exception as e:  # noqa: BLE001 - report and continue
        check(name, False, lambda: f"{type(e).__name__}: {e}")


def runs_of(flags):
    return [
        next((j for j, f in enumerate(flags[i:]) if not f), len(flags) - i)
        for i in range(len(flags))
    ]


def schedule(flags, k):
    """Independent fixed-k greedy replay (same contract as Rust R3 fixed_k_schedule)."""
    n, e, out = len(flags), 0, []
    runs = runs_of(flags)
    while e < n:
        kk = min(k, n - e - 1)
        a = min(runs[e], kk)
        out.append((kk, a))
        e += a + 1
    return out


# ------------------------------------------------------------------ R1 (CPU)
def r1_flags():
    from accept_student import greedy_flags

    # prompt [p0 p1 p2], generated [g0 g1 g2 g3]; sequence index of g_i is 3 + i.
    # draft_argmax[j] guesses sequence[j+1]; flags[i] compares the guess for g_{i+1},
    # which sits at draft_argmax[3 + i].
    prompt_len, gen = 3, [10, 11, 12, 13]
    argmax = [-7, -7, 10, 11, 99, 13]  # guesses for p1, p2, g0, g1, g2 (wrong), g3
    got = greedy_flags(argmax, prompt_len, gen)
    check("greedy_flags hand example", got == [1, 0, 1], lambda: f"expected [1, 0, 1] vs got {got}")
    got = greedy_flags([5, 5, 5], 2, [5, 5])
    check("greedy_flags all-equal", got == [1], lambda: f"expected [1] vs got {got}")
    check("greedy_flags single token", greedy_flags([1, 2], 2, [3]) == [], lambda: "expected []")
    rng = random.Random(3)
    bad = 0
    for _ in range(200):
        p = rng.randint(1, 6)
        gen = [rng.randint(0, 3) for _ in range(rng.randint(1, 12))]
        seq = [9] * p + gen
        guess = [seq[j + 1] if rng.random() < 0.7 else 100 + j for j in range(len(seq) - 1)] + [7]
        want = [int(guess[p + i] == gen[i + 1]) for i in range(len(gen) - 1)]
        bad += greedy_flags(guess, p, gen) != want
    check("greedy_flags random alignment", bad == 0, lambda: f"{bad}/200 wrong")


def r1_rates():
    from accept_student import hazard_from_survival, position_rates

    # vLLM PR #17010 example: accepted_per_pos [5,3,1] over 5 drafts -> S = [1, .6, .2]
    got = hazard_from_survival([1.0, 0.6, 0.2])
    want = [1.0, 0.6, 0.2 / 0.6]
    check(
        "hazard_from_survival vLLM example",
        all(abs(a - b) < 1e-12 for a, b in zip(got, want)) and len(got) == 3,
        lambda: f"expected {want} vs got {got}",
    )
    got = hazard_from_survival([0.5, 0.0, 0.0])
    check(
        "hazard_from_survival zero survival",
        got[:2] == [0.5, 0.0] and got[2] is None,
        lambda: f"got {got}",
    )
    flags = [1, 1, 0, 1, 1, 1, 0, 1]
    surv, haz = position_rates(flags, 3)
    runs = runs_of(flags)
    for j in (1, 2, 3):
        starts = [i for i in range(len(flags)) if i + j <= len(flags)]
        s = sum(runs[i] >= j for i in starts) / len(starts)
        h = sum(runs[i] >= j for i in starts) / sum(runs[i] >= j - 1 for i in starts)
        check(
            f"position_rates S_{j}",
            abs(surv[j - 1] - s) < 1e-12,
            lambda: f"expected {s} vs got {surv[j - 1]}",
        )
        check(
            f"position_rates a_{j}",
            abs(haz[j - 1] - h) < 1e-12,
            lambda: f"expected {h} vs got {haz[j - 1]}",
        )
    # IID flags: survival decays like alpha^j but the hazard stays near alpha.
    rng = random.Random(1)
    iid = [int(rng.random() < 0.8) for _ in range(20000)]
    surv, haz = position_rates(iid, 4)
    check("IID survival decays", abs(surv[3] - 0.8**4) < 0.02, lambda: f"S_4 {surv[3]}")
    check("IID hazard is flat", all(abs(h - 0.8) < 0.02 for h in haz), lambda: f"a_j {haz}")


def r1_coupled():
    import torch

    from accept_student import coupled_step

    p = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
    q = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    beta_exact = float(torch.minimum(p, q).sum())
    rng = random.Random(7)
    n = 20000
    counts, accepts, betas = [0] * 4, 0, set()
    for _ in range(n):
        y, acc, beta = coupled_step(p, q, rng.random(), rng.random(), rng.random())
        counts[y] += 1
        accepts += acc
        betas.add(round(beta, 9))
    freq = [c / n for c in counts]
    tol = 4 * math.sqrt(0.25 / n)  # 4 sigma
    check(
        "coupled output ~ target p",
        all(abs(f - float(x)) < tol for f, x in zip(freq, p)),
        lambda: f"expected {p.tolist()} vs got {freq}",
    )
    check(
        "coupled acceptance rate = sum min(p,q)",
        abs(accepts / n - beta_exact) < tol,
        lambda: f"expected {beta_exact} vs got {accepts / n}",
    )
    check("beta is exact", betas == {round(beta_exact, 9)}, lambda: f"got {sorted(betas)[:3]}")
    # Determinism under common random numbers and the documented inverse-CDF rule.
    y1 = coupled_step(p, q, 0.05, 0.99, 0.5)  # x = 0 (cdf_q 0.1 > 0.05); 0.99*0.1 < 0.5 -> accept
    check("inverse-CDF draft sample", y1[:2] == (0, True), lambda: f"got {y1}")
    y2 = coupled_step(p, q, 0.95, 0.9, 0.7)  # x = 3; 0.9*0.4 = 0.36 >= 0.05 -> reject
    # residual max(p-q,0) = [0.4, 0.1, 0, 0] / 0.5 -> cdf [0.8, 1.0]; u = 0.7 -> index 0
    check("residual inverse-CDF", y2[:2] == (0, False), lambda: f"got {y2}")
    y3 = coupled_step(p, q, 0.95, 0.9, 0.85)
    check("residual second bucket", y3[:2] == (1, False), lambda: f"got {y3}")
    check(
        "same uniforms, same result",
        coupled_step(p, q, 0.3, 0.4, 0.5) == coupled_step(p, q, 0.3, 0.4, 0.5),
    )
    # p == q: every proposal is accepted and beta = 1
    y = coupled_step(p, p.clone(), 0.42, 0.999, 0.1)
    check("p == q always accepts", y[1] is True and abs(y[2] - 1.0) < 1e-9, lambda: f"got {y}")


# ------------------------------------------------------------------ R2 (CPU)
class FakeTimer:
    """Each fn() call advances a fake clock by the next duration; stop() reads it."""

    def __init__(self, durations):
        self.durations, self.clock, self.calls = list(durations), 0.0, 0

    def work(self):
        self.clock += self.durations[self.calls]
        self.calls += 1

    def start(self):
        return self.clock

    def stop(self, token):
        return self.clock - token


def r2_measure():
    from profile_student import measure, summarize

    t = FakeTimer([100.0, 50.0, 40.0, 3.0, 1.0, 2.0, 5.0, 4.0])
    got = measure(t.work, t, warmup=3, reps=5)
    check(
        "measure excludes warmup",
        got == [3.0, 1.0, 2.0, 5.0, 4.0],
        lambda: f"expected [3, 1, 2, 5, 4] vs got {got}",
    )
    check("measure call count", t.calls == 8, lambda: f"expected 8 calls vs got {t.calls}")
    t = FakeTimer([7.0, 9.0])
    got = measure(t.work, t, warmup=0, reps=2)
    check("measure no warmup", got == [7.0, 9.0], lambda: f"got {got}")
    s = summarize([4.0, 1.0, 3.0, 2.0])
    check(
        "summarize nearest-rank",
        (s["median"], s["p25"], s["p75"], s["n"]) == (2.0, 1.0, 3.0, 4),
        lambda: f"got {s}",
    )
    check(
        "summarize stable flag (spread)", s.get("stable") is False, lambda: f"got {s}"
    )  # 3 > 1.5 * 1
    s = summarize([5.0])
    check(
        "summarize single", (s["median"], s["p25"], s["p75"]) == (5.0, 5.0, 5.0), lambda: f"got {s}"
    )
    # bimodal sample (half the reps at a low P-state): p25 = 2, p75 = 16 -> unstable
    s = summarize([2.0, 2.1, 2.0, 16.0, 15.5, 16.2, 2.05, 15.9])
    check("summarize stable flag (bimodal)", s.get("stable") is False, lambda: f"got {s}")
    s = summarize([10.0, 14.9, 15.0, 10.0])  # p25 10, p75 14.9 -> stable at the boundary side
    check("summarize stable flag (boundary)", s.get("stable") is True, lambda: f"got {s}")


# ------------------------------------------------------------------ CUDA
def cuda_checks(models: Path):
    import torch

    from accept_student import greedy_flags
    from collect import CudaTimer
    from engine import Pair, load_prompts
    from profile_student import measure, summarize

    pair = Pair(models, kmax=4, max_len=512)
    compared = equal = 0
    whole = whole_equal = 0
    for row in load_prompts(limit=4):
        ids = pair.chat_ids(row["prompt"])
        g = pair.greedy(ids, 64)
        flags = greedy_flags(
            pair.draft_argmax_stepwise(ids + g["tokens"], len(ids)), len(ids), g["tokens"]
        )
        check(
            f"prompt {row['id']}: flags length",
            len(flags) == len(g["tokens"]) - 1,
            lambda: f"{len(flags)} vs {len(g['tokens']) - 1}",
        )
        for k in (2, 4):
            s = pair.speculative(ids, k, 64)
            diff = next(
                (i for i, (x, y) in enumerate(zip(s["tokens"], g["tokens"])) if x != y), None
            )
            replay = schedule(flags, k)
            e = 0
            for i, (kk, a) in enumerate(s["rounds"]):
                if diff is not None and e + a + 1 >= diff:
                    break
                compared += 1
                if i < len(replay) and replay[i] == (kk, a):
                    equal += 1
                else:
                    break
                e += a + 1
            if diff is None:  # final round may differ when EOS ends generation early
                whole += 1
                m = len(s["rounds"])
                whole_equal += len(replay) == m and replay[: m - 1] == s["rounds"][: m - 1]
    print(f"  compared {compared} rounds before divergence, {whole} runs fully greedy")
    check(
        "real rounds replay from R1 flags",
        compared >= 20 and equal * 100 >= compared * 95,
        lambda: f"{equal}/{compared} equal",
    )
    print(
        f"  {whole_equal}/{whole} greedy-path runs replay exactly (width-2 draft near-ties can break the rest)"
    )
    ctx = torch.randint(1000, 30000, (1, 128), device="cuda")
    pair.target.eager(ctx, 0)
    x = ctx[:, :1]
    cold = CudaTimer()
    samples = measure(lambda: pair.target.forward(x, 128), cold, 3, 10)
    s = summarize(samples)
    check(
        "verify latency samples", len(samples) == 10 and 1.0 < s["median"] < 500.0, lambda: f"{s}"
    )
    check("median within IQR", s["p25"] <= s["median"] <= s["p75"], lambda: f"{s}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--models", type=Path)
    a = ap.parse_args()
    guarded("R1 greedy_flags", r1_flags)
    guarded("R1 rates", r1_rates)
    guarded("R1 coupled_step", r1_coupled)
    guarded("R2 measure", r2_measure)
    if a.device == "cuda":
        guarded("R1 on real models", lambda: cuda_checks(a.models))
    print(f"R python checks: {len(FAILURES)} failed", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
