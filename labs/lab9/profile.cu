// Stage R GPU driver. Usage:
//   profile test                 R1-R4 checks (independent CPU oracles, PASS/FAIL per check)
//   profile bench <out.csv>      measure copy / GEMV / WMMA GEMM (and cuBLAS if linked) and
//                                write a profile CSV that bench.cpp R3 reads (measured).
#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

#include "check.hpp"
#include "profile_student.cuh"
#ifndef HS_NO_CUBLAS
#include <cublas_v2.h>
#endif

using namespace hs;
using chk::Suite;

// ------------------------------------------------------------------ helpers (provided)
__global__ void spin_kernel(long long ns) {
    unsigned long long start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < static_cast<unsigned long long>(ns));
}

static void spin(double us) {
    spin_kernel<<<1, 1>>>(static_cast<long long>(us * 1000.0));
}

template <class T> struct DevBuf {
    T *p = nullptr;
    size_t n = 0;
    explicit DevBuf(size_t count) : n(count) {
        HS_CUDA(cudaMalloc(&p, std::max<size_t>(1, count) * sizeof(T)));
    }
    ~DevBuf() { cudaFree(p); }
    DevBuf(const DevBuf &) = delete;
    void upload(const std::vector<T> &h) {
        HS_CUDA(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> h(n);
        HS_CUDA(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

static std::vector<__half> random_half(size_t n, uint32_t seed) {
    std::vector<__half> v(n);
    uint32_t s = seed;
    for (auto &x : v) {
        s = s * 1664525u + 1013904223u;
        x = __float2half((static_cast<int>((s >> 8) % 2001u) - 1000) / 1000.0f); // [-1, 1]
    }
    return v;
}

// ------------------------------------------------------------------ test mode
static void test_r1(Suite &S) {
    S.group("R1 time_cuda", [&] {
        TimingStats a = time_cuda([] { spin(400); }, 3, 21);
        // A globaltimer spin never ends early; launch overhead and timer granularity add < 0.15 ms.
        S.expect("R1 400 us spin: 0.40 <= median <= 0.55 ms",
                 a.median_ms >= 0.399 && a.median_ms <= 0.55, "median " + chk::num(a.median_ms));
        S.expect("R1 reps recorded", a.reps == 21);
        S.expect("R1 ordering min <= q25 <= median <= q75 <= max",
                 a.min_ms <= a.q25_ms && a.q25_ms <= a.median_ms && a.median_ms <= a.q75_ms &&
                     a.q75_ms <= a.max_ms);
        int calls = 0;
        TimingStats w = time_cuda([&] { spin(calls++ < 4 ? 4000 : 200); }, 4, 20);
        S.expect("R1 launch called warmup + reps times", calls == 24,
                 "calls " + std::to_string(calls));
        S.expect("R1 warmup runs are excluded (max < 1 ms)", w.max_ms < 1.0,
                 "max " + chk::num(w.max_ms));
        // 20 timed runs of (1 + j) ms, j = 0..19, in a scrambled order. The test measures each
        // duration itself (own events, median of 5) to get the sorted values v_j, then
        // interpolates: h = 19 p, q(p) = v[floor h] + (h - floor h)(v[floor h + 1] - v[floor h]).
        // Nearest-rank or (n p) indexing differ from this by >= 0.25 ms; the spin kernel itself
        // jitters by up to ~0.1 ms on a laptop GPU, hence the 0.12 ms tolerance.
        std::vector<double> v(20);
        for (int j = 0; j < 20; ++j) {
            cudaEvent_t a0, a1;
            HS_CUDA(cudaEventCreate(&a0));
            HS_CUDA(cudaEventCreate(&a1));
            std::vector<double> m3;
            for (int r = 0; r < 5; ++r) {
                HS_CUDA(cudaEventRecord(a0));
                spin(1000.0 * (1 + j));
                HS_CUDA(cudaEventRecord(a1));
                HS_CUDA(cudaEventSynchronize(a1));
                float ms = 0;
                HS_CUDA(cudaEventElapsedTime(&ms, a0, a1));
                m3.push_back(ms);
            }
            std::sort(m3.begin(), m3.end());
            v[j] = m3[2];
            HS_CUDA(cudaEventDestroy(a0));
            HS_CUDA(cudaEventDestroy(a1));
        }
        std::sort(v.begin(), v.end());
        auto interp = [&](double p) {
            double h = 19 * p;
            int lo = static_cast<int>(h);
            return v[lo] + (h - lo) * (v[std::min(lo + 1, 19)] - v[lo]);
        };
        int k = 0;
        TimingStats q = time_cuda(
            [&] {
                int i = k++;
                int j =
                    i < 2 ? 0 : (7 * (i - 2)) % 20; // two warmup calls, then a permutation of 0..19
                spin(1000.0 * (1 + j));
            },
            2, 20);
        S.near("R1 interpolated q25 (about 5.75 ms)", interp(0.25), q.q25_ms, 0.002, 0.12);
        S.near("R1 interpolated median (about 10.5 ms)", interp(0.5), q.median_ms, 0.002, 0.12);
        S.near("R1 interpolated q75 (about 15.25 ms)", interp(0.75), q.q75_ms, 0.002, 0.12);
        S.near("R1 min (about 1 ms)", v.front(), q.min_ms, 0.002, 0.12);
        S.near("R1 max (about 20 ms)", v.back(), q.max_ms, 0.002, 0.12);
        S.throws("R1 rejects reps = 0", [] { time_cuda([] {}, 1, 0); });
    });
}

static void test_r2(Suite &S) {
    S.group("R2 copy_vec4_kernel", [&] {
        // 4,000,003 floats: beyond one grid pass (grid is capped) and with a 3-element tail.
        for (size_t n :
             {size_t(1), size_t(3), size_t(4), size_t(5), size_t(1023), size_t(4000003)}) {
            const size_t pad = 16;
            std::vector<float> in(n), out0(n + pad, -7.0f);
            for (size_t i = 0; i < n; ++i) in[i] = float(i % 9973) * 0.5f - 11.0f;
            DevBuf<float> din(n), dout(n + pad);
            din.upload(in);
            dout.upload(out0);
            launch_copy(din.p, dout.p, n);
            HS_CUDA(cudaGetLastError());
            HS_CUDA(cudaDeviceSynchronize());
            auto got = dout.download();
            size_t bad = 0, overrun = 0;
            for (size_t i = 0; i < n; ++i) bad += got[i] != in[i];
            for (size_t i = n; i < n + pad; ++i) overrun += got[i] != -7.0f;
            S.expect("R2 copy n=" + std::to_string(n) + " exact", bad == 0,
                     std::to_string(bad) + " wrong elements");
            S.expect("R2 copy n=" + std::to_string(n) + " no write past n", overrun == 0,
                     std::to_string(overrun) + " overwritten");
        }
    });
}

static void test_r3(Suite &S) {
    S.group("R3 gemv_fp16_kernel", [&] {
        struct Shape {
            int M, K;
        };
        for (Shape s :
             {Shape{1, 8}, Shape{7, 64}, Shape{33, 1032}, Shape{513, 8}, Shape{4096, 2048}}) {
            auto W = random_half(size_t(s.M) * s.K, 11 + s.M), x = random_half(s.K, 5 + s.K);
            DevBuf<__half> dW(W.size()), dx(x.size());
            DevBuf<float> dy(s.M + 8);
            dW.upload(W), dx.upload(x), dy.upload(std::vector<float>(s.M + 8, 123.0f));
            launch_gemv(dW.p, dx.p, dy.p, s.M, s.K);
            HS_CUDA(cudaGetLastError());
            HS_CUDA(cudaDeviceSynchronize());
            auto y = dy.download();
            double worst = 0;
            for (int r = 0; r < s.M; ++r) {
                double ref = 0, mag = 0;
                for (int k = 0; k < s.K; ++k) {
                    double a = __half2float(W[size_t(r) * s.K + k]), b = __half2float(x[k]);
                    ref += a * b, mag += std::abs(a * b);
                }
                double err = std::isfinite(y[r]) ? std::abs(y[r] - ref) / (mag + 1e-30) : 1e30;
                if (!(err <= worst)) worst = err;
            }
            bool untouched = true;
            for (int r = s.M; r < s.M + 8; ++r) untouched = untouched && y[r] == 123.0f;
            std::string tag = "R3 gemv " + std::to_string(s.M) + "x" + std::to_string(s.K);
            S.expect(tag + " matches FP64 reference (rel 1e-4 of sum |w x|)", worst < 1e-4,
                     "worst " + chk::num(worst));
            S.expect(tag + " rows >= M untouched", untouched);
        }
    });
}

static void test_r4(Suite &S) {
    S.group("R4 gemm_wmma_kernel", [&] {
        struct Shape {
            int M, N, K;
        };
        for (Shape s :
             {Shape{64, 64, 32}, Shape{128, 192, 96}, Shape{256, 128, 512}, Shape{64, 320, 64}}) {
            auto A = random_half(size_t(s.M) * s.K, 3 + s.M),
                 B = random_half(size_t(s.K) * s.N, 9 + s.N);
            DevBuf<__half> dA(A.size()), dB(B.size());
            DevBuf<float> dC(size_t(s.M) * s.N);
            dA.upload(A), dB.upload(B);
            HS_CUDA(
                cudaMemset(dC.p, 0xFF, dC.n * sizeof(float))); // NaN pattern: unwritten tiles fail
            launch_gemm(dA.p, dB.p, dC.p, s.M, s.N, s.K);
            HS_CUDA(cudaGetLastError());
            HS_CUDA(cudaDeviceSynchronize());
            auto C = dC.download();
            double worst = 0;
            for (int i = 0; i < s.M; ++i)
                for (int j = 0; j < s.N; ++j) {
                    double ref = 0, mag = 0;
                    for (int k = 0; k < s.K; ++k) {
                        double a = __half2float(A[size_t(i) * s.K + k]),
                               b = __half2float(B[size_t(k) * s.N + j]);
                        ref += a * b, mag += std::abs(a * b);
                    }
                    double c = C[size_t(i) * s.N + j];
                    double err = std::isfinite(c) ? std::abs(c - ref) / (mag + 1e-30) : 1e30;
                    if (!(err <= worst)) worst = err;
                }
            std::string tag = "R4 gemm " + std::to_string(s.M) + "x" + std::to_string(s.N) + "x" +
                              std::to_string(s.K);
            S.expect(tag + " matches FP64 reference (rel 1e-4 of sum |a b|)", worst < 1e-4,
                     "worst " + chk::num(worst));
        }
    });
}

// ------------------------------------------------------------------ bench mode
static std::string device_line() {
    cudaDeviceProp p{};
    int dev = 0, driver = 0, runtime = 0;
    HS_CUDA(cudaGetDevice(&dev));
    HS_CUDA(cudaGetDeviceProperties(&p, dev));
    HS_CUDA(cudaDriverGetVersion(&driver));
    HS_CUDA(cudaRuntimeGetVersion(&runtime));
    char buf[512];
    std::snprintf(buf, sizeof buf, "%s, sm_%d%d, %d SMs, %.2f GiB, CUDA driver API %d, runtime %d",
                  p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0,
                  driver, runtime);
    return buf;
}

// Keep the GPU busy with the same launch for at least `ms` before timing. Between configs the
// laptop GPU drops to a low P-state (memory clock 810 instead of 8000 MHz measured on this
// RTX 4060 Laptop); a few warmup launches do not bring it back, 300 ms of work does.
template <class Launch> static void keep_busy(Launch &&launch, float ms) {
    cudaEvent_t a, b;
    HS_CUDA(cudaEventCreate(&a));
    HS_CUDA(cudaEventCreate(&b));
    HS_CUDA(cudaEventRecord(a));
    for (float busy = 0; busy < ms;) {
        for (int i = 0; i < 8; ++i) launch();
        HS_CUDA(cudaEventRecord(b));
        HS_CUDA(cudaEventSynchronize(b));
        HS_CUDA(cudaEventElapsedTime(&busy, a, b));
    }
    HS_CUDA(cudaEventDestroy(a));
    HS_CUDA(cudaEventDestroy(b));
}

static bool stable(const TimingStats &t) {
    return t.q75_ms <= 1.5 * t.q25_ms;
}

// keep_busy + time_cuda (R1); an unstable row (q75 > 1.5 q25) is re-measured up to twice.
template <class Launch> static TimingStats measure(Launch &&launch, int warm, int reps) {
    TimingStats t;
    for (int attempt = 0; attempt < 3; ++attempt) {
        keep_busy(launch, 300.0f);
        t = time_cuda(launch, warm, reps);
        if (stable(t)) break;
    }
    return t;
}

static int bench(const std::string &out_path, const std::string &note) {
    std::FILE *f = std::fopen(out_path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 1;
    }
    std::time_t now = std::time(nullptr);
    char date[64];
    std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    std::fprintf(f, "# measured by labs/lab9/profile.cu bench (300 ms busy warmup, then median and "
                    "IQR of 30 runs after 5 warmup runs)\n");
    std::fprintf(f, "# device: %s\n# utc: %s\n", device_line().c_str(), date);
    if (!note.empty()) std::fprintf(f, "# host: %s\n", note.c_str());
    std::fprintf(f, "kernel,m,n,k,median_ms,q25_ms,q75_ms,achieved,stable\n");
    auto row = [&](const char *kernel, long long m, long long n, long long k, const TimingStats &t,
                   double achieved, const char *unit) {
        std::fprintf(f, "%s,%lld,%lld,%lld,%.6f,%.6f,%.6f,%.2f %s,%d\n", kernel, m, n, k,
                     t.median_ms, t.q25_ms, t.q75_ms, achieved, unit, stable(t) ? 1 : 0);
        std::printf("BENCH R2 kernel=%s m=%lld n=%lld k=%lld median_ms=%.4f iqr_ms=%.4f "
                    "achieved=%.2f %s stable=%d "
                    "(measured)\n",
                    kernel, m, n, k, t.median_ms, t.q75_ms - t.q25_ms, achieved, unit,
                    stable(t) ? 1 : 0);
        std::fflush(stdout);
    };
    const int warm = 5, reps = 30;
    // Sizes start above the 32 MB L2 of the RTX 4060 Laptop; smaller copies measure L2, not DRAM.
    for (double mib : {64.0, 128.0, 256.0, 512.0}) {
        size_t n = static_cast<size_t>(mib * 1048576 / 4);
        DevBuf<float> a(n), b(n);
        HS_CUDA(cudaMemset(a.p, 0, n * 4));
        TimingStats t = measure([&] { launch_copy(a.p, b.p, n); }, warm, reps);
        row("copy", static_cast<long long>(n * 4), 1, 1, t,
            2.0 * n * 4 / (t.median_ms * 1e-3) / 1e9, "GB/s");
    }
    struct Mk {
        int m, k;
    };
    // 64-128 MB weight matrices (Qwen3-1.7B: about 100 MB per layer), also beyond L2.
    for (Mk s : {Mk{8192, 4096}, Mk{12288, 4096}, Mk{6144, 8192}, Mk{16384, 4096}}) {
        DevBuf<__half> W(size_t(s.m) * s.k), x(s.k);
        DevBuf<float> y(s.m);
        HS_CUDA(cudaMemset(W.p, 0, W.n * 2));
        HS_CUDA(cudaMemset(x.p, 0, x.n * 2));
        TimingStats t = measure([&] { launch_gemv(W.p, x.p, y.p, s.m, s.k); }, warm, reps);
        row("gemv", s.m, 1, s.k, t,
            (2.0 * s.m * s.k + 2.0 * s.k + 4.0 * s.m) / (t.median_ms * 1e-3) / 1e9, "GB/s");
    }
    struct Mnk {
        int m, n, k;
    };
    // Square shapes are compute-bound; the skinny ones (batch 64/128 times a 96-128 MB weight)
    // are decode-like and memory-bound. Keep every weight operand above the 32 MB L2, or the
    // repeated runs read it from cache and the compulsory-traffic byte count is wrong.
    std::vector<Mnk> shapes = {{512, 512, 512},    {1024, 1024, 1024}, {2048, 2048, 2048},
                               {4096, 4096, 4096}, {2048, 6144, 2048}, {64, 16384, 4096},
                               {128, 12288, 4096}, {256, 12288, 4096}};
#ifndef HS_NO_CUBLAS
    cublasHandle_t handle;
    bool have_cublas = cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS;
#endif
    for (Mnk s : shapes) {
        DevBuf<__half> A(size_t(s.m) * s.k), B(size_t(s.k) * s.n);
        DevBuf<float> C(size_t(s.m) * s.n);
        A.upload(random_half(A.n, 1)), B.upload(random_half(B.n, 2));
        TimingStats t = measure([&] { launch_gemm(A.p, B.p, C.p, s.m, s.n, s.k); }, warm, reps);
        row("gemm", s.m, s.n, s.k, t, 2.0 * s.m * s.n * s.k / (t.median_ms * 1e-3) / 1e12,
            "TFLOPS");
#ifndef HS_NO_CUBLAS
        if (have_cublas) {
            const float one = 1.0f, zero = 0.0f;
            // Row-major C = A B is column-major C^T = B^T A^T: pass B first.
            auto call = [&] {
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, s.n, s.m, s.k, &one, B.p, CUDA_R_16F,
                             s.n, A.p, CUDA_R_16F, s.k, &zero, C.p, CUDA_R_32F, s.n,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            };
            TimingStats c = measure(call, warm, reps);
            row("gemm_cublas", s.m, s.n, s.k, c,
                2.0 * s.m * s.n * s.k / (c.median_ms * 1e-3) / 1e12, "TFLOPS");
        }
#endif
    }
#ifndef HS_NO_CUBLAS
    if (have_cublas) cublasDestroy(handle);
#else
    std::printf("NOTE cuBLAS not linked; gemm_cublas rows skipped\n");
#endif
    std::fclose(f);
    std::printf("WROTE %s (measured)\n", out_path.c_str());
    return 0;
}

int main(int argc, char **argv) {
    std::string mode = argc > 1 ? argv[1] : "test";
    try {
        std::printf("DEVICE %s\n", device_line().c_str());
        if (mode == "bench")
            return bench(argc > 2 ? argv[2] : "rtx4060_profile.csv", argc > 3 ? argv[3] : "");
        Suite S("R-gpu");
        test_r1(S);
        test_r2(S);
        test_r3(S);
        test_r4(S);
        return S.finish();
    } catch (const std::exception &e) {
        std::printf("FAIL R-gpu: %s\n", e.what());
        return 1;
    }
}
