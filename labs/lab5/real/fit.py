"""R1 driver (provided): fit the cost model to a measure.py file and write the fit JSON
that `bench R` reads (coefficients + the device/driver/date of the measurement).

    python lab5/real/fit.py --in results/lab5-real/prefill.json --out results/lab5-real/cost.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fit_student import COEFFICIENTS, fit_cost_model, predict  # noqa: E402


def load_points(measurement: dict):
    return [(r["p"], r["n"], r["median_ms"]) for r in measurement["rows"] if r["kind"] == "prefill"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor-ratio", type=float, default=1.3)
    args = ap.parse_args(argv)
    measurement = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    points = load_points(measurement)
    model = fit_cost_model(points, floor_ratio=args.floor_ratio)
    meta = measurement["meta"]
    copy_rows = [r for r in measurement["rows"] if r["kind"] == "copy"]
    out = {
        "meta": {
            k: meta.get(k)
            for k in (
                "gpu",
                "driver",
                "date",
                "torch",
                "transformers",
                "model",
                "dtype",
                "attn_implementation",
                "kv_bytes_per_token",
            )
        },
        "label": "fit of measured prefill latency (model: max(floor, a + b n + c n (p + n/2) + d p))",
        "source": Path(args.inp).name,
        "coefficients": {k: model[k] for k in ("floor_ms",) + COEFFICIENTS},
        "quality": {
            "used_points": model["used"],
            "all_points": len(points),
            "max_rel_err": model["max_rel_err"],
            "floor_ratio": args.floor_ratio,
        },
        "crossover_p": model["crossover_p"],
        "hf_copy_ms_per_cached_token": (
            sum(r["median_ms"] / r["p"] for r in copy_rows) / len(copy_rows)
        )
        if copy_rows
        else None,
        "residuals": [
            {"p": p, "n": n, "measured_ms": t, "predicted_ms": round(predict(model, p, n), 3)}
            for p, n, t in points
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    c = out["coefficients"]
    print(
        f"floor {c['floor_ms']:.2f} ms | a {c['a_ms']:.3f} ms | b {c['b_ms'] * 1e3:.2f} us/token | "
        f"c {c['c_ms'] * 1e6:.3f} ns/token^2 | d {c['d_ms'] * 1e3:.3f} us/cached token"
    )
    print(
        f"crossover p* = b/c = {out['crossover_p']:.0f} tokens; used {model['used']}/{len(points)} points; "
        f"max rel err {model['max_rel_err']:.3f}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
