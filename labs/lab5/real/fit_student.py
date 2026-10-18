"""R1: fit the prefill cost model T(p, n) to measured latencies (standard library only).

Model (labs/lab5/src/cost.rs, lecture section 5):

    T(p, n) = max(floor, a + b*n + c*n*(p + n/2) + d*p)       [ms]

p = cached (reused) tokens, n = uncached tokens computed in this forward.
"""

from __future__ import annotations

import math

COEFFICIENTS = ("a_ms", "b_ms", "c_ms", "d_ms")


def features(p: float, n: float) -> list[float]:
    """Provided: regressors for (a, b, c, d)."""
    return [1.0, float(n), float(n) * (float(p) + float(n) / 2.0), float(p)]


def predict(model: dict, p: float, n: float) -> float:
    """Provided: the fitted model, including the launch floor."""
    x = features(p, n)
    work = sum(model[k] * v for k, v in zip(COEFFICIENTS, x))
    return max(model["floor_ms"], work)


def fit_cost_model(points, floor_ratio: float = 1.3) -> dict:
    """Least-squares fit of the cost model.

    points: iterable of (p, n, t_ms) with t_ms > 0 (one median per configuration).
    1. floor_ms = the smallest t_ms (the launch-bound floor of an eager forward).
    2. GPU-bound subset: points with t_ms >= floor_ratio * floor_ms. Fewer than 4 such
       points -> raise ValueError (4 unknowns).
    3. Ordinary (unweighted) least squares of t_ms on features(p, n) over that subset:
       minimize sum (t - x.beta)^2 for beta = (a, b, c, d). Scale columns before
       solving; the n*(p+n/2) column is ~1e7 while the constant column is 1.
    4. Return {"floor_ms", "a_ms", "b_ms", "c_ms", "d_ms",
               "used": size of the subset,
               "max_rel_err": max over ALL points of |predict - t| / t,
               "crossover_p": b / c (context where per-token attention equals the linear
                              layers), or math.inf if c <= 0}.
    """
    # BEGIN TODO R1
    raise NotImplementedError("R1")
    # END TODO R1
