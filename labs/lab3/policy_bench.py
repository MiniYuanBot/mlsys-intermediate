import json

from src.policy_student import bucket_for, plan_buckets

train = [(1, 100), (4, 50), (8, 30), (17, 20), (32, 10), (64, 4)]
heldout = [(2, 40), (9, 20), (33, 10), (80, 3)]
for compile_ms in [0, 1, 10, 100]:
    bounds = plan_buckets(train, compile_ms, 0.02)
    # Explicit model units; this is not a measured torch.compile benchmark.
    print(
        json.dumps(
            {
                "kind": "cost_model",
                "compile_ms_assumed": compile_ms,
                "bounds": bounds,
                "train_cost_ms": compile_ms * len(bounds)
                + sum(f * bucket_for(n, bounds) * 0.02 for n, f in train),
                "heldout_fallback_requests": sum(
                    f for n, f in heldout if bucket_for(n, bounds) is None
                ),
            }
        )
    )
