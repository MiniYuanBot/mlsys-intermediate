"""R2: latency measurement for the cost model (reference solution)."""

from __future__ import annotations

import math


# BEGIN TODO R2
def measure(fn, timer, warmup: int, reps: int) -> list[float]:
    """Call fn() `warmup` times untimed, then `reps` times timed.

    timer.start() returns a token; timer.stop(token) blocks until the timed work is
    finished and returns elapsed milliseconds. Return exactly `reps` samples in call
    order; warmup calls never appear in the result.
    """
    raise NotImplementedError("R2")


def summarize(samples: list[float]) -> dict:
    """Nearest-rank median, p25 and p75 (rank = max(1, ceil(q * n)), as Rust A3), n, and
    stable = (p75 <= 1.5 * p25): a laptop GPU that dropped to a low P-state mid-run gives
    a bimodal sample whose IQR fails this test (labs/docs/authoring.md section 5)."""
    raise NotImplementedError("R2")


# END TODO R2
