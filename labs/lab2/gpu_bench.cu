// GPU benchmark for stages B and R. Usage: gpu_bench <B|R|all>.
//   B: the four INT4 SIMT kernels (B2 warp, B3 byte-pair, B4 4-token tile, B5 split-K)
//      against FP32 cuBLAS, FP16 cuBLAS (tensor cores) and "unpack INT4 to FP16, then
//      cuBLAS", at batch 1 and 4 on every shape of bench_plan.hpp.
//   R: INT4 vs FP8 vs FP16 vs FP32 weight paths (plus cuBLASLt FP8 W8A8) at batch
//      1..64 on the sweep shapes: latency, effective GB/s and output error vs FP64.
// Measurement (labs/docs/authoring.md section 5): before each configuration the same
// launch keeps the GPU busy for >= 300 ms (laptop P-states), then 30 reps, each after a
// 64 MiB L2 flush outside the timed region; the first 5 are dropped; median, p25, p75
// from cudaEvent; stable = p75 <= 1.5 * p25. Bytes are logical bytes the path moves
// (weights + metadata as many times as the path reads them, X, Y, intermediates), not
// profiler counters; gbps = bytes / median; frac_peak divides by the measured copy peak.
// floor_us = (weights once + X + Y) / copy peak is MODELED, not measured.
// rel_err = relative L2 error vs FP64 with the ORIGINAL FP32 weights on 256 sampled rows,
// so it contains the format's quantization error; each path must stay under a bound.
#include "baselines.cuh"
#include "bench_plan.hpp"
#include "oracle.hpp"
#include "src/fp8_gemv_student.cuh"
#include "src/fp8_student.hpp"
#include "src/gemv_student.cuh"
#include <functional>
#include <iostream>
#include <map>

namespace {
struct Stats {
    double median, p25, p75;
};
Stats summarize(std::vector<float> ms) {
    std::sort(ms.begin(), ms.end());
    auto at = [&](double f) { return double(ms[size_t(f * (ms.size() - 1) + 0.5)]); };
    return {at(0.5) * 1e3, at(0.25) * 1e3, at(0.75) * 1e3}; // microseconds
}
void *g_flush = nullptr;
Stats time_it(const std::function<void()> &launch) {
    cudaEvent_t a, b;
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));
    CUDA_OK(cudaEventRecord(a)); // spin until the GPU has been busy for kSpinMs
    for (float busy = 0; busy < plan::kSpinMs;) {
        for (int i = 0; i < 8; ++i) launch();
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        CUDA_OK(cudaEventElapsedTime(&busy, a, b));
    }
    std::vector<float> ms;
    for (int rep = 0; rep < plan::kReps; ++rep) {
        if (plan::kFlushL2) CUDA_OK(cudaMemsetAsync(g_flush, rep & 0xff, plan::kFlushBytes));
        CUDA_OK(cudaEventRecord(a));
        launch();
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float t = 0;
        CUDA_OK(cudaEventElapsedTime(&t, a, b));
        if (rep >= plan::kDiscard) ms.push_back(t);
    }
    CUDA_OK(cudaGetLastError());
    cudaEventDestroy(a), cudaEventDestroy(b);
    return summarize(ms);
}
__global__ void copy_kernel(const float4 *__restrict__ in, float4 *__restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x)
        out[i] = in[i];
}
template <class T> T *dev(size_t n) {
    T *p = nullptr;
    CUDA_OK(cudaMalloc(&p, n * sizeof(T) + 256));
    return p;
}
template <class T> T *upload(const std::vector<T> &h) {
    T *p = dev<T>(h.size());
    CUDA_OK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
using fp8::Format;
} // namespace

int main(int argc, char **argv) {
    try {
        std::string section = argc > 1 ? argv[1] : "all";
        bool run_b = section == "B" || section == "all", run_r = section == "R" || section == "all";
        cudaDeviceProp prop{};
        CUDA_OK(cudaGetDeviceProperties(&prop, 0));
        int driver = 0, runtime = 0;
        cudaDriverGetVersion(&driver), cudaRuntimeGetVersion(&runtime);
        std::cout << "# device " << prop.name << ", sm_" << prop.major << prop.minor << ", "
                  << prop.multiProcessorCount << " SMs, L2 " << (prop.l2CacheSize >> 20)
                  << " MiB, driver " << driver << ", runtime " << runtime << '\n';
        std::string warm =
            plan::cold_violations(size_t(prop.l2CacheSize), [](int f, int r, int c, int g) {
                return fp8::weight_bytes(Format(f), r, c, g);
            });
        if (!warm.empty()) {
            std::cout << "FAIL bench plan: weights stay in L2 between reps for" << warm << '\n';
            return 1;
        }
        CUDA_OK(cudaMalloc(&g_flush, plan::kFlushBytes));
        size_t n4 = (size_t(256) << 20) / sizeof(float4);
        float4 *src = dev<float4>(n4), *dst = dev<float4>(n4);
        CUDA_OK(cudaMemset(src, 1, n4 * sizeof(float4)));
        auto copy =
            time_it([&] { copy_kernel<<<prop.multiProcessorCount * 8, 256>>>(src, dst, n4); });
        const double peak = 2.0 * n4 * sizeof(float4) / (copy.median * 1e-6) / 1e9;
        cudaFree(src), cudaFree(dst);
        std::cout << "# measured copy peak " << peak
                  << " GB/s (256 MiB read + 256 MiB write, median of "
                  << plan::kReps - plan::kDiscard << ")\n";
        std::cout << "section,shape,rows,cols,batch,path,median_us,p25_us,p75_us,bytes,gbps,frac_"
                     "peak,tflops,"
                     "floor_us_modeled,rel_err,stable\n";
        base::Blas blas;
        bool all_ok = true;
        const int G = plan::kGroup, maxb = 64;
        for (const plan::Shape &s : plan::kShapes) {
            if (!run_b && !(run_r)) break;
            const int M = s.rows, K = s.cols;
            auto w = oracle::gaussian(size_t(M) * K, 2026 + M + K);
            auto x = oracle::uniform(maxb * K, 7 + K, -1, 1);
            // FP64 reference on 256 sampled rows for every token.
            const int step = std::max(1, M / 256);
            std::vector<int> sample;
            for (int r = 0; r < M; r += step) sample.push_back(r);
            std::vector<double> ref(size_t(maxb) * sample.size());
            for (int t = 0; t < maxb; ++t)
                for (size_t i = 0; i < sample.size(); ++i) {
                    double acc = 0;
                    const float *wr = &w[size_t(sample[i]) * K], *xt = &x[size_t(t) * K];
                    for (int k = 0; k < K; ++k) acc += double(wr[k]) * xt[k];
                    ref[t * sample.size() + i] = acc;
                }
            // Artifacts: the student's A2 packer and R2 FP8 packer, FP16/FP32 copies, and a
            // per-tensor FP8 W8A8 pair for cuBLASLt.
            auto q4 = lowbit::pack(w, M, K, G);
            auto q8 = fp8::fp8_pack(w, M, K, G);
            float wmax = 0, xmax = 0;
            for (float v : w) wmax = std::max(wmax, std::abs(v));
            for (float v : x) xmax = std::max(xmax, std::abs(v));
            const float sw = wmax / fp8::kMax, sx = xmax / fp8::kMax;
            std::vector<uint8_t> w8(w.size()), x8(x.size());
            for (size_t i = 0; i < w.size(); ++i) w8[i] = fp8::encode_e4m3(w[i] / sw);
            for (size_t i = 0; i < x.size(); ++i) x8[i] = fp8::encode_e4m3(x[i] / sx);
            std::vector<__half> wh(w.size()), xh(x.size());
            for (size_t i = 0; i < w.size(); ++i) wh[i] = __float2half(w[i]);
            for (size_t i = 0; i < x.size(); ++i) xh[i] = __float2half(x[i]);
            uint8_t *d_q4 = upload(q4.bytes), *d_q8 = upload(q8.codes), *d_w8 = upload(w8),
                    *d_x8 = upload(x8);
            float *d_s4 = upload(q4.scales), *d_s8 = upload(q8.scales), *d_w = upload(w),
                  *d_x = upload(x);
            __half *d_wh = upload(wh), *d_xh = upload(xh), *d_unpacked = dev<__half>(w.size());
            float *d_y = dev<float>(size_t(maxb) * M),
                  *d_part = dev<float>(size_t(M) * (K / plan::kSplitChunk + 1));
            std::vector<float> scale_pair = {sw, sx};
            float *d_lt_scales = upload(scale_pair);

            const size_t B4 = fp8::weight_bytes(Format::INT4, M, K, G),
                         B8 = fp8::weight_bytes(Format::FP8, M, K, G);
            const size_t B16 = fp8::weight_bytes(Format::FP16, M, K, G),
                         B32 = fp8::weight_bytes(Format::FP32, M, K, G);
            const int parts = (K + plan::kSplitChunk - 1) / plan::kSplitChunk;
            struct Path {
                const char *name;
                bool in_b, in_r, batch1_only;
                double bound; // max rel_err vs FP64 of the original weights
                std::function<size_t(int)> bytes, floor_bytes;
                std::function<void(int)> launch;
            };
            auto io = [&](int b, int xb) { return size_t(b) * K * xb + size_t(b) * M * 4; };
            std::vector<Path> paths = {
                {"int4_b2_warp", true, false, false, .25, [&](int b) { return b * B4 + io(b, 4); },
                 [&](int b) { return B4 + io(b, 4); },
                 [&](int b) {
                     for (int t = 0; t < b; ++t)
                         launch_fused(d_q4, d_s4, d_x + size_t(t) * K, d_y + size_t(t) * M, M, K,
                                      G);
                 }},
                {"int4_b3_pair", true, false, false, .25, [&](int b) { return b * B4 + io(b, 4); },
                 [&](int b) { return B4 + io(b, 4); },
                 [&](int b) {
                     for (int t = 0; t < b; ++t)
                         launch_pair(d_q4, d_s4, d_x + size_t(t) * K, d_y + size_t(t) * M, M, K, G);
                 }},
                {"int4_b5_splitk", true, false, false, .25,
                 [&](int b) { return b * (B4 + size_t(8) * M * parts) + io(b, 4); },
                 [&](int b) { return B4 + io(b, 4); },
                 [&](int b) {
                     for (int t = 0; t < b; ++t)
                         launch_split(d_q4, d_s4, d_x + size_t(t) * K, d_y + size_t(t) * M, d_part,
                                      M, K, G, plan::kSplitChunk);
                 }},
                {"int4_b4_tile", true, true, false, .25,
                 [&](int b) { return size_t((b + 3) / 4) * B4 + io(b, 4); },
                 [&](int b) { return B4 + io(b, 4); },
                 [&](int b) { launch_batch(d_q4, d_s4, d_x, d_y, M, K, G, b); }},
                {"fp8_r3_gemv", false, true, true, .1, [&](int b) { return b * B8 + io(b, 4); },
                 [&](int b) { return B8 + io(b, 4); },
                 [&](int b) {
                     for (int t = 0; t < b; ++t)
                         fp8::launch_fp8_gemv(d_q8, d_s8, d_x + size_t(t) * K, d_y + size_t(t) * M,
                                              M, K, G);
                 }},
                {"fp8_r4_tile", false, true, false, .1,
                 [&](int b) { return size_t((b + fp8::kTile - 1) / fp8::kTile) * B8 + io(b, 4); },
                 [&](int b) { return B8 + io(b, 4); },
                 [&](int b) { fp8::launch_fp8_batch(d_q8, d_s8, d_x, d_y, M, K, G, b); }},
                {"fp32_cublas_sgemm", true, true, false, 1e-4,
                 [&](int b) { return B32 + io(b, 4); }, [&](int b) { return B32 + io(b, 4); },
                 [&](int b) { blas.sgemm(d_w, d_x, d_y, M, K, b); }},
                {"fp16_cublas_gemmex", true, true, false, .01,
                 [&](int b) { return B16 + io(b, 2); }, [&](int b) { return B16 + io(b, 2); },
                 [&](int b) { blas.hgemm(d_wh, d_xh, d_y, M, K, b); }},
                {"int4_unpack_fp16_gemmex", true, true, false, .25,
                 [&](int b) { return B4 + 2 * B16 + io(b, 2); },
                 [&](int b) { return B4 + io(b, 2); },
                 [&](int b) {
                     base::unpack(d_q4, d_s4, d_unpacked, M, K, G);
                     blas.hgemm(d_unpacked, d_xh, d_y, M, K, b);
                 }},
            };
            auto run = [&](const char *sec, const Path &p, int b,
                           const std::function<void()> &launch, size_t bytes, size_t floor_bytes,
                           double bound) {
                CUDA_OK(cudaMemset(d_y, 0xff, size_t(b) * M * sizeof(float)));
                launch();
                CUDA_OK(cudaDeviceSynchronize());
                std::vector<float> y(size_t(b) * M);
                CUDA_OK(
                    cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
                double num = 0, den = 0;
                for (int t = 0; t < b; ++t)
                    for (size_t i = 0; i < sample.size(); ++i) {
                        double r = ref[t * sample.size() + i], d = y[size_t(t) * M + sample[i]] - r;
                        num += d * d, den += r * r;
                    }
                double err = std::sqrt(num / den);
                bool ok = err < bound; // also false for NaN
                all_ok &= ok;
                Stats st = time_it(launch);
                double gbps = bytes / (st.median * 1e-6) / 1e9;
                std::cout << sec << ',' << s.name << ',' << M << ',' << K << ',' << b << ','
                          << p.name << ',' << st.median << ',' << st.p25 << ',' << st.p75 << ','
                          << bytes << ',' << gbps << ',' << gbps / peak << ','
                          << 2.0 * M * K * b / (st.median * 1e-6) / 1e12 << ','
                          << floor_bytes / (peak * 1e9) * 1e6 << ',' << err << ','
                          << (st.p75 <= 1.5 * st.p25 ? "yes" : "NO") << (ok ? "" : ",WRONG_OUTPUT")
                          << '\n';
            };
            if (run_b)
                for (int b : {1, 4})
                    for (auto &p : paths)
                        if (p.in_b)
                            run(
                                "B", p, b, [&] { p.launch(b); }, p.bytes(b), p.floor_bytes(b),
                                p.bound);
            if (run_r) {
                std::vector<int> batches =
                    s.sweep ? std::vector<int>(std::begin(plan::kBatches), std::end(plan::kBatches))
                            : std::vector<int>{1};
                for (int b : batches) {
                    for (auto &p : paths)
                        if (p.in_r && !(p.batch1_only && b > 1))
                            run(
                                "R", p, b, [&] { p.launch(b); }, p.bytes(b), p.floor_bytes(b),
                                p.bound);
                    base::Fp8Lt lt(blas, M, K, b, d_lt_scales, d_lt_scales + 1);
                    Path p{"fp8_cublaslt_w8a8", false, true, false, .15, nullptr, nullptr, nullptr};
                    size_t bytes = size_t(M) * K + size_t(b) * K + size_t(b) * M * 4;
                    if (lt.ok)
                        run("R", p, b, [&] { lt.run(d_w8, d_x8, d_y); }, bytes, bytes, p.bound);
                    else
                        std::cout << "R," << s.name << ',' << M << ',' << K << ',' << b
                                  << ",fp8_cublaslt_w8a8,,,,,,,,,," << lt.why << '\n';
                }
            }
            for (void *ptr :
                 std::initializer_list<void *>{d_q4, d_q8, d_w8, d_x8, d_s4, d_s8, d_w, d_x, d_wh,
                                               d_xh, d_unpacked, d_y, d_part, d_lt_scales})
                cudaFree(ptr);
        }
        cudaFree(g_flush);
        if (!all_ok) {
            std::cout << "FAIL some path produced output outside its error bound (rows marked "
                         "WRONG_OUTPUT)\n";
            return 1;
        }
        return 0;
    } catch (const std::exception &e) {
        std::cout << "FAIL bench: " << e.what() << '\n';
        return 1;
    }
}
