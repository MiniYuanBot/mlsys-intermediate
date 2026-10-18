"""R1: turn model outputs into acceptance data (reference solution).

All functions are pure; collect.py and test_real.py call them. Tensors are torch
tensors on any device; lists are plain Python lists of ints.
"""

from __future__ import annotations

import torch


# BEGIN TODO R1
def greedy_flags(draft_argmax: list[int], prompt_len: int, generated: list[int]) -> list[int]:
    """Greedy acceptance flags along the target's greedy output.

    draft_argmax[i] is the draft's argmax given sequence[: i + 1], where
    sequence = prompt + generated (so it is the guess for sequence[i + 1]).
    generated[0] comes from prefill and is never speculated. Return flags of length
    len(generated) - 1 with flags[i] = 1 iff the draft's guess for generated[i + 1]
    equals generated[i + 1].
    """
    assert len(draft_argmax) >= prompt_len + len(generated) - 1
    return [
        int(draft_argmax[prompt_len + i] == generated[i + 1]) for i in range(len(generated) - 1)
    ]


def position_rates(flags: list[int], kmax: int) -> tuple[list, list]:
    """Survival S_j = P(L >= j) and conditional hazard a_j = P(L >= j | L >= j-1),
    j = 1..kmax, where L(i) is the run of 1s starting at decode step i.

    Only start positions where depth j is observable count: i + j <= n (n = len(flags)).
    S_j = #{i: L(i) >= j} / #{i}, a_j = #{i: L(i) >= j} / #{i: L(i) >= j-1}, both over
    those observable i. Undefined entries (empty denominators) are None.
    """
    n = len(flags)
    runs, nxt = [0] * n, 0
    for i in range(n - 1, -1, -1):
        nxt = nxt + 1 if flags[i] else 0
        runs[i] = nxt
    survival, hazard = [], []
    for j in range(1, kmax + 1):
        starts = range(0, n - j + 1)
        reach = sum(runs[i] >= j for i in starts)
        prior = sum(runs[i] >= j - 1 for i in starts)
        survival.append(reach / len(starts) if len(starts) else None)
        hazard.append(reach / prior if prior else None)
    return survival, hazard


def hazard_from_survival(survival: list[float]) -> list[float]:
    """Convert vLLM-style per-position rates S_j = P(A >= j) (j = 1..k) into
    conditional rates a_j = S_j / S_{j-1} with S_0 = 1. A zero S_{j-1} gives None."""
    out, previous = [], 1.0
    for s in survival:
        out.append(None if previous == 0 else s / previous)
        previous = s
    return out


def coupled_step(p: torch.Tensor, q: torch.Tensor, u_draft: float, u_accept: float, u_resid: float):
    """One speculative-sampling decision with common random numbers.

    p, q: 1-D target / draft probability vectors (same length, each sums to 1).
    x = inverse-CDF sample of q at u_draft (smallest index whose cumulative q exceeds
    u_draft, clamped to the last index). Accept iff u_accept * q[x] < p[x]. On
    rejection sample from the residual max(p - q, 0) / sum(max(p - q, 0)) by inverse
    CDF at u_resid. Return (token, accepted, beta) with beta = sum(min(p, q)).
    """

    def inverse_cdf(prob: torch.Tensor, u: float) -> int:
        cdf = torch.cumsum(prob.double(), 0)
        index = int(
            torch.searchsorted(
                cdf, torch.tensor([u], dtype=cdf.dtype, device=cdf.device), right=True
            )
        )
        return min(index, prob.numel() - 1)

    x = inverse_cdf(q, u_draft)
    beta = float(torch.minimum(p, q).double().sum())
    if u_accept * float(q[x]) < float(p[x]):
        return x, True, beta
    residual = torch.clamp(p.double() - q.double(), min=0.0)
    residual = residual / residual.sum()
    return inverse_cdf(residual, u_resid), False, beta


# END TODO R1
