"""Stage R Python checks for R1 (fit_cost_model). Standard library only.

CPU: synthetic points with known coefficients (exact recovery), launch-floor handling,
the least-squares optimality conditions on noisy data (normal equations and a
finite-difference check of the student's own objective), and a sanity fit of the
committed RTX 4060 measurement. --device cuda additionally runs a short real
measurement with measure.py (Qwen3-0.6B) and checks the correctness gate and stability.
Prints PASS/FAIL per check, never stops at the first failure, exits 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FAILURES: list[str] = []
TRUE = {"a_ms": 4.0, "b_ms": 0.06, "c_ms": 1.5e-5, "d_ms": 0.005}


def check(name: str, ok: bool, detail=lambda: ""):
    if ok:
        print(f"PASS R/{name}", flush=True)
    else:
        line = f"FAIL R/{name}: {detail()}"
        print(line, flush=True)
        FAILURES.append(line)


def guarded(name, fn):
    try:
        fn()
    except NotImplementedError as e:
        check(name, False, lambda: f"TODO {e} not implemented")
    except Exception as e:  # noqa: BLE001 - report and continue
        check(name, False, lambda: f"{type(e).__name__}: {e}")


def truth(p, n, coef=TRUE, floor=0.0):
    """Independent model evaluation (does not use fit_student.predict)."""
    return max(
        floor, coef["a_ms"] + coef["b_ms"] * n + coef["c_ms"] * n * (p + n / 2) + coef["d_ms"] * p
    )


GRID = [(p, n) for p in (0, 512, 1024, 2048, 4096, 6144) for n in (64, 256, 512, 1024, 2048)]


def r1_exact():
    from fit_student import fit_cost_model

    pts = [(p, n, truth(p, n)) for p, n in GRID]
    m = fit_cost_model(pts, floor_ratio=1.0)
    for k, v in TRUE.items():
        check(
            f"exact recovery {k}",
            math.isclose(m[k], v, rel_tol=1e-6, abs_tol=1e-9),
            lambda k=k, v=v: f"expected {v} vs got {m[k]}",
        )
    check(
        "used = all points at floor_ratio 1",
        m["used"] == len(pts),
        lambda: f"{m['used']} vs {len(pts)}",
    )
    check("max_rel_err ~ 0 on exact data", m["max_rel_err"] < 1e-6, lambda: f"{m['max_rel_err']}")
    check(
        "crossover_p = b/c",
        math.isclose(m["crossover_p"], TRUE["b_ms"] / TRUE["c_ms"], rel_tol=1e-6),
        lambda: f"{m['crossover_p']}",
    )


def r1_floor():
    from fit_student import fit_cost_model

    floor = 20.0
    pts = [(p, n, truth(p, n, floor=floor)) for p, n in GRID] + [(0, 16, floor), (512, 16, floor)]
    m = fit_cost_model(pts)  # default floor_ratio 1.3
    check("floor = smallest latency", m["floor_ms"] == floor, lambda: f"{m['floor_ms']}")
    gpu_bound = sum(1 for p, n in GRID if truth(p, n) >= 1.3 * floor)
    check(
        "only points >= 1.3 x floor used",
        m["used"] == gpu_bound,
        lambda: f"used {m['used']} vs {gpu_bound}",
    )
    for k, v in TRUE.items():
        check(
            f"floor points do not bias {k}",
            math.isclose(m[k], v, rel_tol=1e-6, abs_tol=1e-9),
            lambda k=k, v=v: f"expected {v} vs got {m[k]}",
        )
    # max_rel_err is over ALL points with the floor applied: exact here.
    check(
        "max_rel_err uses max(floor, .) on all points",
        m["max_rel_err"] < 1e-6,
        lambda: f"{m['max_rel_err']}",
    )
    # One excluded point measured at 24 ms (below 1.3 x floor) while the model says
    # max(20, ~5) = 20: it must not move the fit, but it counts in max_rel_err (4/24).
    m2 = fit_cost_model(pts + [(0, 16, 24.0)])
    check(
        "excluded point does not move the fit",
        math.isclose(m2["c_ms"], TRUE["c_ms"], rel_tol=1e-6),
        lambda: f"{m2['c_ms']}",
    )
    check(
        "max_rel_err counts excluded points too",
        math.isclose(m2["max_rel_err"], 4 / 24, rel_tol=1e-6),
        lambda: f"expected {4 / 24} vs got {m2['max_rel_err']}",
    )
    try:
        fit_cost_model([(0, 16, 10.0), (0, 64, 10.0), (0, 256, 30.0)])
        check("too few GPU-bound points -> ValueError", False, lambda: "no exception")
    except ValueError:
        check("too few GPU-bound points -> ValueError", True)


def sse(points, coef):
    return sum(
        (t - (coef["a_ms"] + coef["b_ms"] * n + coef["c_ms"] * n * (p + n / 2) + coef["d_ms"] * p))
        ** 2
        for p, n, t in points
    )


def r1_noisy_optimality():
    from fit_student import fit_cost_model

    rng = random.Random(7)
    pts = [(p, n, truth(p, n) * (1 + rng.gauss(0, 0.05))) for p, n in GRID]
    m = fit_cost_model(pts, floor_ratio=1.0)
    # Normal equations: the residual is orthogonal to every regressor column.
    cols = [lambda p, n: 1.0, lambda p, n: n, lambda p, n: n * (p + n / 2), lambda p, n: p]
    res = [
        t - (m["a_ms"] + m["b_ms"] * n + m["c_ms"] * n * (p + n / 2) + m["d_ms"] * p)
        for p, n, t in pts
    ]
    for name, f in zip(("1", "n", "n(p+n/2)", "p"), cols):
        dot = sum(r * f(p, n) for r, (p, n, _) in zip(res, pts))
        scale = math.sqrt(sum(f(p, n) ** 2 for p, n, _ in pts)) * math.sqrt(sum(r * r for r in res))
        check(
            f"residual orthogonal to column {name}",
            abs(dot) <= 1e-6 * scale,
            lambda dot=dot, scale=scale: f"{dot} (scale {scale})",
        )
    # Finite differences on the objective: no coordinate move lowers the SSE.
    base = sse(pts, m)
    worse = []
    for k in ("a_ms", "b_ms", "c_ms", "d_ms"):
        for sign in (-1, 1):
            h = 1e-4 * max(abs(m[k]), 1e-9)
            moved = dict(m)
            moved[k] += sign * h
            if sse(pts, moved) < base * (1 - 1e-12):
                worse.append(f"{k}{'+' if sign > 0 else '-'}")
    check(
        "SSE is a local minimum in every coefficient",
        not worse,
        lambda: f"moves that lower SSE: {worse}",
    )


def r1_reference_data():
    from fit_student import fit_cost_model, predict

    data = HERE / "data" / "prefill-qwen3-0.6b-rtx4060.json"
    meas = json.loads(data.read_text(encoding="utf-8"))
    pts = [(r["p"], r["n"], r["median_ms"]) for r in meas["rows"] if r["kind"] == "prefill"]
    m = fit_cost_model(pts)
    check(
        "4060 fit: all coefficients of the GPU work positive",
        all(m[k] > 0 for k in ("b_ms", "c_ms", "d_ms")),
        lambda: f"{m}",
    )
    check("4060 fit: max rel err < 0.35", m["max_rel_err"] < 0.35, lambda: f"{m['max_rel_err']}")
    check(
        "4060 fit: crossover in [1k, 10k] tokens (FLOP estimate ~3.8k)",
        1000 <= m["crossover_p"] <= 10000,
        lambda: f"{m['crossover_p']}",
    )
    committed = json.loads(
        (HERE / "data" / "cost-qwen3-0.6b-rtx4060.json").read_text(encoding="utf-8")
    )
    c = committed["coefficients"]
    worst = max(abs(predict(m, p, n) - predict(c, p, n)) / predict(c, p, n) for p, n, _ in pts)
    check(
        "your fit reproduces the committed reference fit (<1%)",
        worst < 0.01,
        lambda: f"worst {worst:.4f}",
    )


def cuda_measure(models: Path):
    model = models / "Qwen3-0.6B"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "quick.json"
        cmd = [
            sys.executable,
            str(HERE / "measure.py"),
            "--model",
            str(model),
            "--out",
            str(out),
            "--p",
            "0,2048",
            "--n",
            "16,1024",
            "--reps",
            "5",
        ]
        print("+", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        check("measure.py quick sweep exits 0 (gate ok)", rc == 0, lambda: f"exit {rc}")
        if not out.is_file():
            return
        meas = json.loads(out.read_text(encoding="utf-8"))
    rows = {(r["p"], r["n"]): r for r in meas["rows"] if r["kind"] == "prefill"}
    check(
        "cached path matches full forward (gate)",
        all(g["ok"] for g in meas["gate"]) and meas["gate"],
        lambda: f"{meas['gate']}",
    )
    check(
        "every row stable (p75 <= 1.5 p25)",
        all(r["stable"] for r in rows.values()),
        lambda: f"{[(k, r['p25_ms'], r['p75_ms']) for k, r in rows.items()]}",
    )
    check(
        "a cached prefix still costs attention: T(2048,1024) > T(0,1024)",
        rows[(2048, 1024)]["median_ms"] > rows[(0, 1024)]["median_ms"],
        lambda: f"{rows[(2048, 1024)]['median_ms']} vs {rows[(0, 1024)]['median_ms']}",
    )
    from fit_student import predict

    ref = json.loads((HERE / "data" / "cost-qwen3-0.6b-rtx4060.json").read_text(encoding="utf-8"))
    for (p, n), r in rows.items():
        pred = predict(ref["coefficients"], p, n)
        rel = abs(pred - r["median_ms"]) / r["median_ms"]
        tag = "INFO" if rel <= 0.35 else "WARN"
        print(
            f"{tag} R/reference fit vs this GPU at p={p} n={n}: predicted {pred:.1f} ms, measured "
            f"{r['median_ms']:.1f} ms ({rel:.0%}; a different GPU/driver/clock is expected to differ)"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument(
        "--models", default=os.environ.get("MLSYS_MODELS") or str(Path.home() / "models")
    )
    args = ap.parse_args()
    guarded("R1 exact recovery", r1_exact)
    guarded("R1 launch floor", r1_floor)
    guarded("R1 least-squares optimality", r1_noisy_optimality)
    guarded("R1 committed 4060 measurement", r1_reference_data)
    if args.device == "cuda":
        guarded("R1 real measurement", lambda: cuda_measure(Path(args.models)))
    print(f"R python: {len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
