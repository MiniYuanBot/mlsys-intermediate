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
    pts = [(float(p), float(n), float(t)) for p, n, t in points]
    if not pts or any(t <= 0 for _, _, t in pts):
        raise ValueError("need positive latencies")
    floor = min(t for _, _, t in pts)
    used = [(p, n, t) for p, n, t in pts if t >= floor_ratio * floor]
    if len(used) < len(COEFFICIENTS):
        raise ValueError(f"only {len(used)} GPU-bound points; need >= {len(COEFFICIENTS)}")
    k = len(COEFFICIENTS)
    rows = [features(p, n) for p, n, _ in used]
    y = [t for _, _, t in used]
    scale = [max(abs(r[j]) for r in rows) or 1.0 for j in range(k)]
    xs = [[r[j] / scale[j] for j in range(k)] for r in rows]
    # Normal equations (X^T X) beta = X^T y on the scaled columns.
    a = [[sum(x[i] * x[j] for x in xs) for j in range(k)] for i in range(k)]
    b = [sum(x[i] * yy for x, yy in zip(xs, y)) for i in range(k)]
    for col in range(k):  # Gaussian elimination with partial pivoting
        piv = max(range(col, k), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            raise ValueError("singular design: vary both p and n")
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, k):
            f = a[r][col] / a[col][col]
            for j in range(col, k):
                a[r][j] -= f * a[col][j]
            b[r] -= f * b[col]
    beta = [0.0] * k
    for i in reversed(range(k)):
        beta[i] = (b[i] - sum(a[i][j] * beta[j] for j in range(i + 1, k))) / a[i][i]
    beta = [beta[j] / scale[j] for j in range(k)]
    model = {"floor_ms": floor, **dict(zip(COEFFICIENTS, beta))}
    model["used"] = len(used)
    model["max_rel_err"] = max(abs(predict(model, p, n) - t) / t for p, n, t in pts)
    model["crossover_p"] = model["b_ms"] / model["c_ms"] if model["c_ms"] > 0 else math.inf
    return model
    # END TODO R1
