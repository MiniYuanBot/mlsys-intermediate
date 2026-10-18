"""C bench: paired real-input comparison of the measured plan (horizon 1 and 10) against
eager, static, dynamic and fixed-bucket torch.compile on three deterministic scenarios
(stable / shift / tail). Dynamo is reset per strategy; the Inductor disk cache is kept,
so first calls are warm-cache numbers, not cold compilation."""

from collections import Counter
import json
import statistics
import torch
from src.execution_student import HOLDOUT_SEED, measure, operation, request_tensors, workflow
from src.pipeline_student import padding_account, split_trace, trace_shift


def inputs(lengths, width, device):
    # Same seeds as C4's held-out requests, so every strategy sees identical tensors.
    return [request_tensors(n, width, device, HOLDOUT_SEED + i) for i, n in enumerate(lengths)]


def baseline(requests, mode, device):
    torch.compiler.reset()
    fn = (
        operation
        if mode == "eager"
        else torch.compile(
            operation, backend="inductor", dynamic=(mode == "dynamic"), fullgraph=True
        )
    )

    def run():
        outputs = []
        for x, residual in requests:
            length = x.shape[0]
            if mode == "fixed_bucket":
                bound = next((n for n in [4, 8, 16] if n >= length), None)
                if bound is None:
                    outputs.append(operation(x, residual))
                    continue
                x = torch.nn.functional.pad(x, (0, 0, 0, bound - length))
                residual = torch.nn.functional.pad(residual, (0, 0, 0, bound - length))
            outputs.append(fn(x, residual)[:length])
        return outputs

    # First run is timed before correctness comparison to keep startup observable.
    first, times = measure(run, device, repetitions=5)
    outputs = run()
    for output, (x, residual) in zip(outputs, requests, strict=True):
        torch.testing.assert_close(output, operation(x, residual), rtol=2e-4, atol=2e-5)
    return {"first_replay_ms": first, "warm_replay_median_ms": statistics.median(times)}


def run(device):
    training = [4, 8, 16, 4, 8, 4] * 6
    scenarios = {
        "stable": [3, 7, 15, 3, 7, 3] * 10,
        "shift": [15, 15, 15, 7, 3, 10] * 10,
        "tail": [3, 7, 15, 3, 7, 33] * 10,
    }
    for name, heldout in scenarios.items():
        history, _ = split_trace(training + heldout, len(training))
        requests = inputs(heldout, 32, device)
        for horizon in [1, 10]:
            torch.compiler.reset()
            result = workflow(
                training + heldout,
                len(training),
                32,
                device,
                capacity=3,
                horizon_multiplier=horizon,
            )
            accounting = padding_account(sorted(Counter(heldout).items()), result["bounds"])
            print(
                json.dumps(
                    {
                        "scenario": name,
                        "mode": "measured_plan",
                        "device": device,
                        "trace_shift": trace_shift(history, heldout),
                        "cache_regime": "Dynamo reset per strategy; normal disk cache, not cold",
                        "heldout_calls": len(heldout),
                        **accounting,
                        **{k: v for k, v in result.items() if k != "outputs"},
                    }
                )
            )
        for mode in ["eager", "static", "dynamic", "fixed_bucket"]:
            print(
                json.dumps(
                    {
                        "scenario": name,
                        "mode": mode,
                        "device": device,
                        "heldout_calls": len(heldout),
                        "cache_regime": "Dynamo reset per strategy; normal disk cache, not cold",
                        **baseline(requests, mode, device),
                    }
                )
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args().device)
