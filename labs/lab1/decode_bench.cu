// Stage R benchmark: one decode step, one layer, Qwen3-1.7B attention shapes
// (16 query heads, 8 KV heads, head_dim 128), batch 1. Three paths:
//   fp16       KV stored densely in FP16, split-K attention (R2<Fp16Cache> + R3)
//   dequant    int4 cache -> dense FP16 (dequant_to_fp16), then the fp16 path
//   fused      int4 cache consumed directly (R2<Int4Cache> + R3)
// Timing: cudaEvent around the kernels of one path, L2 flushed (64 MiB memset) before
// every repetition and outside the timed region; 25 timed reps, median + IQR.
// Clock warmup: laptop GPUs drop to low P-states (memory clock 810 vs 8000 MHz on the
// RTX 4060 Laptop) while the host builds the next fixture, which made this memory-bound
// kernel up to ~8x slower in some runs. Each measurement therefore first keeps the GPU
// busy with the same launch for >= kSpinMs, and rows whose IQR is wide are flagged.
// Bytes: logical bytes each path must move (reads + writes), not profiler counters.
#include "r_fixture.cuh"
#include <iostream>

namespace {
using fixture::D;

struct Stats {
    double median, p25, p75;
};
Stats summarize(std::vector<float> ms) {
    std::sort(ms.begin(), ms.end());
    auto at = [&](double f) { return double(ms[size_t(f * (ms.size() - 1) + 0.5)]); };
    return {at(0.5) * 1e3, at(0.25) * 1e3, at(0.75) * 1e3}; // microseconds
}
constexpr float kSpinMs = 300; // long enough for the driver to raise SM and memory clocks
template <class F> Stats time_it(F launch, void *flush, size_t flush_bytes) {
    cudaEvent_t a, b;
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));
    // Spin until the GPU has been busy for kSpinMs (checked in batches of launches).
    CUDA_OK(cudaEventRecord(a));
    for (float busy = 0; busy < kSpinMs;) {
        for (int i = 0; i < 16; ++i) launch();
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        CUDA_OK(cudaEventElapsedTime(&busy, a, b));
    }
    std::vector<float> ms;
    for (int rep = 0; rep < 30; ++rep) {
        CUDA_OK(cudaMemsetAsync(flush, rep & 0xff, flush_bytes));
        CUDA_OK(cudaEventRecord(a));
        launch();
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float t = 0;
        CUDA_OK(cudaEventElapsedTime(&t, a, b));
        if (rep >= 5) ms.push_back(t);
    }
    CUDA_OK(cudaGetLastError());
    cudaEventDestroy(a), cudaEventDestroy(b);
    return summarize(ms);
}
__global__ void copy_kernel(const float4 *__restrict__ in, float4 *__restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = in[i];
}
int pick_split(int tokens, int kv_heads, int sms) {
    int target = std::max(1, sms * 16 / kv_heads); // about 16 warps (splits) per SM in total
    return std::max(32, (tokens + target - 1) / target);
}
} // namespace

int main() {
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver), cudaRuntimeGetVersion(&runtime);
    std::cout << "# device " << prop.name << ", sm_" << prop.major << prop.minor << ", "
              << prop.multiProcessorCount << " SMs, L2 " << prop.l2CacheSize / (1 << 20)
              << " MiB, driver " << driver << ", runtime " << runtime << '\n';
    const size_t flush_bytes = 64u << 20;
    void *flush = nullptr;
    CUDA_OK(cudaMalloc(&flush, flush_bytes));

    // Measured peak: a grid-stride float4 copy of 256 MiB (read + write counted).
    size_t n4 = (256u << 20) / sizeof(float4);
    float4 *src, *dst;
    CUDA_OK(cudaMalloc(&src, n4 * sizeof(float4)));
    CUDA_OK(cudaMalloc(&dst, n4 * sizeof(float4)));
    CUDA_OK(cudaMemset(src, 1, n4 * sizeof(float4)));
    auto copy = time_it([&] { copy_kernel<<<prop.multiProcessorCount * 8, 256>>>(src, dst, n4); },
                        flush, flush_bytes);
    double peak = 2.0 * n4 * sizeof(float4) / (copy.median * 1e-6) / 1e9;
    cudaFree(src), cudaFree(dst);
    std::cout << "# measured copy peak " << peak
              << " GB/s (median of 25, 256 MiB read + 256 MiB write)\n";
    std::cout << "path,tokens,split,n_splits,median_us,p25_us,p75_us,kv_bytes,total_bytes,gbps,"
                 "frac_of_peak,max_abs_vs_fp16,stable\n";

    auto run_length = [&](int tokens, int split, bool all_paths) {
        fixture::Config cfg;
        cfg.vgroup = 64;
        cfg.n_res = 64; // KIVI-style FP32 residual window; quantized prefix = tokens - 64
        cfg.n_chunks = (tokens - cfg.n_res) / cfg.block;
        auto raw = fixture::make_raw(cfg, 1234 + tokens);
        auto packed = fixture::quantize(raw);
        fixture::DeviceInt4 dev(packed);
        int T = cfg.tokens();
        if (split <= 0) split = pick_split(T, cfg.kv_heads, prop.multiProcessorCount);
        kvdec::DecodeShape shape{cfg.q_heads, cfg.kv_heads, T, split};
        int S = shape.n_splits();
        float *dq = fixture::to_device(raw.q), *pm, *pl, *pa, *out_fused, *out_fp16, *out_deq;
        size_t parts = (size_t)cfg.q_heads * S;
        CUDA_OK(cudaMalloc(&pm, parts * sizeof(float)));
        CUDA_OK(cudaMalloc(&pl, parts * sizeof(float)));
        CUDA_OK(cudaMalloc(&pa, parts * D * sizeof(float)));
        for (float **o : {&out_fused, &out_fp16, &out_deq})
            CUDA_OK(cudaMalloc(o, (size_t)cfg.q_heads * D * sizeof(float)));
        std::vector<__half> hk(raw.k.size()), hv(raw.v.size());
        for (size_t i = 0; i < hk.size(); ++i)
            hk[i] = __float2half(raw.k[i]), hv[i] = __float2half(raw.v[i]);
        __half *fk = fixture::to_device(hk), *fv = fixture::to_device(hv), *qk, *qv;
        CUDA_OK(cudaMalloc(&qk, hk.size() * sizeof(__half)));
        CUDA_OK(cudaMalloc(&qv, hv.size() * sizeof(__half)));
        kvdec::Fp16Cache dense{fk, fv, T}, dequantized{qk, qv, T};
        dim3 deq_grid((T + kvdec::kWarpsPerBlock - 1) / kvdec::kWarpsPerBlock, cfg.kv_heads);

        size_t small =
            cfg.q_heads * D * 4 * 2 + parts * (D + 2) * 4 * 2; // q + out + partials (write + read)
        size_t int4 = fixture::int4_kv_bytes(cfg), fp16 = fixture::fp16_kv_bytes(cfg);
        struct Path {
            const char *name;
            size_t kv, total;
            Stats st;
            float *out;
        };
        std::vector<Path> paths;
        paths.push_back(
            {"fused", int4, int4 + small,
             time_it(
                 [&] { kvdec::decode_attention<2>(dev.cache, shape, dq, pm, pl, pa, out_fused); },
                 flush, flush_bytes),
             out_fused});
        if (all_paths) {
            paths.push_back(
                {"fp16", fp16, fp16 + small,
                 time_it(
                     [&] { kvdec::decode_attention<2>(dense, shape, dq, pm, pl, pa, out_fp16); },
                     flush, flush_bytes),
                 out_fp16});
            paths.push_back(
                {"dequant", int4, int4 + 2 * fp16 + small,
                 time_it(
                     [&] {
                         kvdec::dequant_to_fp16<<<deq_grid, kvdec::kWarpsPerBlock * 32>>>(dev.cache,
                                                                                          qk, qv);
                         kvdec::decode_attention<2>(dequantized, shape, dq, pm, pl, pa, out_deq);
                     },
                     flush, flush_bytes),
                 out_deq});
        }
        CUDA_OK(cudaDeviceSynchronize());
        auto ref =
            all_paths ? fixture::to_host(out_fp16, (size_t)cfg.q_heads * D) : std::vector<float>();
        bool ok = true;
        for (auto &p : paths) {
            double diff = 0;
            if (all_paths) {
                auto o = fixture::to_host(p.out, ref.size());
                for (size_t i = 0; i < o.size(); ++i)
                    diff = std::max(diff, double(std::fabs(o[i] - ref[i])));
                ok &= std::isfinite(diff) &&
                      diff < 0.25; // int4 vs fp16 values; a broken kernel is far off
            }
            double gbps = p.total / (p.st.median * 1e-6) / 1e9;
            std::cout
                << p.name << ',' << T << ',' << split << ',' << S << ',' << p.st.median << ','
                << p.st.p25 << ',' << p.st.p75 << ',' << p.kv << ',' << p.total << ',' << gbps
                << ',' << gbps / peak << ',' << (all_paths ? diff : NAN)
                << ','
                // p75 > 1.5 * p25: clocks or a co-tenant changed during the reps; rerun before using it
                << (p.st.p75 <= 1.5 * p.st.p25 ? "yes" : "NO") << '\n';
        }
        if (all_paths) {
            auto f = fixture::to_host(out_fused, ref.size()),
                 d = fixture::to_host(out_deq, ref.size());
            double gap = 0;
            for (size_t i = 0; i < f.size(); ++i)
                gap = std::max(gap, double(std::fabs(f[i] - d[i])));
            ok &=
                gap < 2e-2; // same int4 values; only FP16 rounding of the dequantized cache differs
            if (!ok) std::cout << "FAIL paths disagree: max|fused - dequant| = " << gap << '\n';
        }
        for (void *p :
             {(void *)dq, (void *)pm, (void *)pl, (void *)pa, (void *)out_fused, (void *)out_fp16,
              (void *)out_deq, (void *)fk, (void *)fv, (void *)qk, (void *)qv})
            cudaFree(p);
        return ok;
    };
    bool ok = true;
    for (int tokens : {1024, 2048, 4096, 8192, 16384, 32768}) ok &= run_length(tokens, 0, true);
    std::cout << "# split sweep at 32768 tokens (fused only)\n";
    for (int split : {64, 128, 256, 512, 1024, 4096}) ok &= run_length(32768, split, false);
    cudaFree(flush);
    std::cout << (ok ? "RESULT: bench completed\n" : "RESULT: FAIL\n");
    return ok ? 0 : 1;
}
