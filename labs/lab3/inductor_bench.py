"""C bench: the Inductor-generated kernel for the SAME function as C6/C7, timed with the
same rules as bench.cu, plus eager PyTorch for reference.

    f(x, residual, gamma) -> (y, z) with z = x + residual, y = z * rsqrt(mean(z^2) + eps) * gamma

Inductor cannot mutate `residual` in place without a copy, so the compiled function returns
z; the DRAM traffic is the same four row-sized accesses as the hand kernel (read x and
residual, write z and y) plus gamma. Each method is captured as one CUDA graph of 20 calls
(no Python or launch overhead inside the timed region), spun for >= 300 ms, then 25 timed
reps (median/p25/p75/stable), matching bench.cu. The generated Triton source is written to
build/inductor_<dtype>_<width>.py so you can read the kernel Inductor chose.
"""

import argparse
from pathlib import Path

import torch
from torch._inductor.utils import run_and_get_code

LAUNCHES, SPIN_MS, EPS = 20, 300.0, 1e-5


def add_rmsnorm(x, residual, gamma):
    z = x + residual
    zf = z.float()
    y = (zf * torch.rsqrt(zf.square().mean(-1, keepdim=True) + EPS)).to(z.dtype) * gamma
    return y, z


def time_graph(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(LAUNCHES):
            fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    busy = 0.0
    while busy < SPIN_MS:
        for _ in range(8):
            graph.replay()
        end.record()
        end.synchronize()
        busy = start.elapsed_time(end)
    samples = []
    for rep in range(30):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        if rep >= 5:
            samples.append(start.elapsed_time(end) * 1000 / LAUNCHES)
    samples.sort()
    q = lambda p: samples[int(p * (len(samples) - 1) + 0.5)]  # noqa: E731
    return q(0.5), q(0.25), q(0.75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="build")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    print(
        f"# device {torch.cuda.get_device_name()}, torch {torch.__version__}; "
        f"{LAUNCHES} calls per CUDA graph, >= {SPIN_MS:.0f} ms spin, 25 reps"
    )
    print(
        "dtype,rows,width,method,median_us,p25_us,p75_us,ideal_bytes,gbps_ideal,stable,triton_kernels"
    )
    for rows in [1, 16, 128, 1024, 4096]:
        for width in [1024, 2048, 4096]:
            for dtype, tag in [(torch.float32, "f32"), (torch.float16, "f16")]:
                g = torch.Generator(device="cuda").manual_seed(7)
                x = torch.zeros(rows, width, device="cuda", dtype=dtype)
                residual = torch.randn(rows, width, generator=g, device="cuda").to(dtype)
                gamma = (torch.rand(width, generator=g, device="cuda") * 1.75 + 0.25).to(dtype)
                torch.compiler.reset()
                compiled = torch.compile(
                    add_rmsnorm, backend="inductor", dynamic=False, fullgraph=True
                )
                (y, z), codes = run_and_get_code(compiled, x, residual, gamma)
                ref_y, ref_z = add_rmsnorm(x.double(), residual.double(), gamma.double())
                tol = 2e-5 if dtype == torch.float32 else 4e-3
                torch.testing.assert_close(y.double(), ref_y, rtol=tol, atol=tol)
                torch.testing.assert_close(z.double(), ref_z, rtol=tol, atol=tol)
                source = "\n\n".join(codes)
                kernels = source.count("@triton.jit")
                if rows == 128:
                    (out / f"inductor_{tag}_{width}.py").write_text(source, encoding="utf-8")
                ideal = 4 * rows * width * x.element_size() + width * x.element_size()
                for method, fn in [
                    ("inductor", lambda: compiled(x, residual, gamma)),
                    ("eager_torch", lambda: add_rmsnorm(x, residual, gamma)),
                ]:
                    med, p25, p75 = time_graph(fn)
                    stable = "yes" if p75 <= 1.5 * p25 else "no"
                    count = kernels if method == "inductor" else ""
                    print(
                        f"{tag},{rows},{width},{method},{med:.3f},{p25:.3f},{p75:.3f},{ideal},"
                        f"{ideal / (med * 1e-6) / 1e9:.1f},{stable},{count}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
