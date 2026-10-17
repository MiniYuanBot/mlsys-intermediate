"""Execute real Inductor paths and compare every request with eager PyTorch."""

import json
import torch
from src.execution_student import calibrate, execute, measure, operation, workflow


def run(device, bench=False):
    count = [0]

    def tick():
        count[0] += 1

    first, samples = measure(tick, device, 3, 2)
    assert count[0] == 6 and first >= 0 and len(samples) == 3
    print("PASS C1 timed_samples")
    profiles, artifacts, calibration_ms = calibrate([4, 8], 32, device, 3)
    assert set(profiles) == {4, 8} and calibration_ms > 0
    print("PASS C2 real_calibration: torch.compile backend=inductor")
    # A deterministic admission fixture forces both paths regardless of machine
    # speed; ACTUAL compiled functions remain those measured above, not mocks.
    admission = {
        n: {"eager_ms": 3, "steady_ms": 1, "mad_ms": 0, "first_premium_ms": 0} for n in [4, 8]
    }
    generator = torch.Generator(device=device).manual_seed(73)
    requests = [
        (
            torch.randn(n, 32, generator=generator, device=device),
            torch.randn(n, 32, generator=generator, device=device),
        )
        for n in [3, 7, 13, 1]
    ]
    outputs, events = execute(requests, [4, 8], admission, artifacts, {4: 10, 8: 10}, capacity=1)
    assert [e[1] for e in events] == ["compiled", "compiled", "eager", "eager"]
    for result, (x, residual) in zip(outputs, requests, strict=True):
        torch.testing.assert_close(result, operation(x, residual), rtol=2e-4, atol=2e-5)
    print("PASS C3 executor_padding_fallback and bounded-cache eviction")
    result = workflow([4, 8, 4, 8, 4, 8, 3, 7, 13, 1], 6, 32, device, capacity=2)
    assert all(bound in [4, 8] for bound in result["bounds"])
    assert result["events"][2][1] == "eager"
    print("PASS C4 closed_loop: training-only calibration/planning; holdout eager oracle")
    if bench:
        print(
            json.dumps(
                {
                    "device": device,
                    "scope": "real small RMSNorm block, no LLM",
                    "first_call_note": "may reuse compiler disk cache; not cold compilation",
                    **result,
                }
            )
        )
