// GPU tests for stages B (int4 SIMT kernels), C (frozen artifact on the GPU) and R
// (FP8 E4M3 weight-only GEMV). Usage: gpu_test <group>, one group per process so that
// a trapped kernel (an unfinished TODO) only fails its own group:
//   B1 nibble probe | B2 warp GEMV | B3 byte-pair | B4 4-token tile | B5 split-K+merge
//   C  frozen calibration artifact -> CUDA kernels
//   R1 E4M3 encoder | R2 FP8 packing | R3 FP8 GEMV | R4 FP8 8-token tile | R5 bytes + bench plan
// Kernel inputs are built by oracle.hpp (not by the student's A2/R2), and every output
// is compared with an FP64 decode of the same bytes, so B and R do not inherit A or R1/R2.
#include "bench_plan.hpp"
#include "check.hpp"
#include "oracle.hpp"
#include "src/fp8_gemv_student.cuh"
#include "src/fp8_student.hpp"
#include "src/gemv_student.cuh"
#include <cuda_fp8.h>
#include <string>

using chk::check;
using chk::guard;
using chk::num;
using lowbit::Packed;

#define CUDA_OK(x)                                                                                 \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess)                                                                     \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_));              \
    } while (0)

namespace {
template <class T> T *upload(const std::vector<T> &x, size_t extra = 0, T fill = T()) {
    T *p = nullptr;
    CUDA_OK(cudaMalloc(&p, (x.size() + extra) * sizeof(T) + 16));
    std::vector<T> host(x);
    host.resize(x.size() + extra, fill);
    CUDA_OK(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T> std::vector<T> download(const T *p, size_t n) {
    std::vector<T> h(n);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
// Tolerance from the magnitude of the terms: FP32 accumulation of n products.
struct Ref {
    std::vector<double> y, mag; // mag = sum |w||x|
};
Ref ref_int4(const Packed &p, const std::vector<float> &x, int batch) {
    auto w = oracle::decode_all(p);
    Ref r{std::vector<double>(size_t(batch) * p.rows), std::vector<double>(size_t(batch) * p.rows)};
    for (int t = 0; t < batch; ++t)
        for (int i = 0; i < p.rows; ++i) {
            double a = 0, m = 0;
            for (int c = 0; c < p.cols; ++c) {
                double v = w[size_t(i) * p.cols + c] * x[size_t(t) * p.cols + c];
                a += v, m += std::abs(v);
            }
            r.y[size_t(t) * p.rows + i] = a, r.mag[size_t(t) * p.rows + i] = m;
        }
    return r;
}
// Largest |got - want| / (1e-5 * (1 + mag)); <= 1 passes. NaN counts as a failure.
double worst_ratio(const std::vector<float> &got, const Ref &r) {
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double e = std::abs(double(got[i]) - r.y[i]) / (1e-5 * (1 + r.mag[i]));
        if (!(e == e)) return 1e300;
        worst = std::max(worst, e);
    }
    return worst;
}

// ------------------------------- B -----------------------------------------------
__global__ void nibble_probe(unsigned *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 512) out[i] = nibble(unsigned(i >> 1), 4u * (i & 1));
}
void group_b1() {
    guard("B1 nibble(byte, shift) = (byte >> shift) & 15 for all 256 bytes, shift 0/4", [] {
        unsigned *d = nullptr;
        CUDA_OK(cudaMalloc(&d, 512 * sizeof(unsigned)));
        nibble_probe<<<4, 128>>>(d);
        CUDA_OK(cudaGetLastError());
        auto h = download(d, 512);
        cudaFree(d);
        int bad = 0;
        for (int i = 0; i < 512; ++i) bad += h[i] != (unsigned(i >> 1) >> (4 * (i & 1)) & 15u);
        check(bad == 0,
              "B1 nibble(byte, shift) = (byte >> shift) & 15 for all 256 bytes, shift 0/4",
              std::to_string(bad) + " of 512 wrong (e.g. sign-extended or shift ignored)");
    });
}
enum class Kernel { Warp, Pair, Tile, Split };
// Runs one kernel on shapes rows {1,7,33} x cols {1,31,65,257} x group {3,32,128}.
void kernel_sweep(Kernel k, const std::string &name, std::vector<int> batches) {
    guard(name, [&] {
        double worst = 0;
        std::string where;
        for (int m : {1, 7, 33})
            for (int n : {1, 31, 65, 257})
                for (int g : {3, 32, 128})
                    for (int b : batches) {
                        auto p = oracle::pack(oracle::uniform(m * n, 17 + m * n + g), m, n, g);
                        auto x = oracle::uniform(b * n, 19 + b);
                        auto dp = upload(p.bytes);
                        auto ds = upload(p.scales);
                        auto dx = upload(x);
                        float *dy = nullptr, *tmp = nullptr;
                        CUDA_OK(cudaMalloc(&dy, size_t(b) * m * sizeof(float)));
                        CUDA_OK(cudaMemset(dy, 0xff,
                                           size_t(b) * m * sizeof(float))); // NaN until written
                        CUDA_OK(cudaMalloc(&tmp, size_t(m) * ((n + 16) / 17) * sizeof(float)));
                        for (int t = 0; t < (k == Kernel::Tile ? 1 : b); ++t) {
                            const float *in = dx + size_t(t) * n;
                            float *out = dy + size_t(t) * m;
                            if (k == Kernel::Warp) launch_fused(dp, ds, in, out, m, n, g);
                            if (k == Kernel::Pair) launch_pair(dp, ds, in, out, m, n, g);
                            if (k == Kernel::Split) launch_split(dp, ds, in, out, tmp, m, n, g, 17);
                            if (k == Kernel::Tile) launch_batch(dp, ds, dx, dy, m, n, g, b);
                        }
                        CUDA_OK(cudaGetLastError());
                        auto got = download(dy, size_t(b) * m);
                        cudaFree(dp), cudaFree(ds), cudaFree(dx), cudaFree(dy), cudaFree(tmp);
                        double e = worst_ratio(got, ref_int4(p, x, b));
                        if (e > worst) {
                            worst = e;
                            where = std::to_string(m) + "x" + std::to_string(n) + " g" +
                                    std::to_string(g) + " batch " + std::to_string(b);
                        }
                    }
        check(worst <= 1, name, "worst error " + num(worst) + "x tolerance at " + where);
    });
}
void group_b2() {
    kernel_sweep(Kernel::Warp, "B2 warp-per-row GEMV = FP64 decode (36 shapes, K tails, row tails)",
                 {1});
}
void group_b3() {
    kernel_sweep(Kernel::Pair, "B3 byte-pair GEMV = FP64 decode (pairs straddle groups at g=3)",
                 {1});
}
void group_b4() {
    kernel_sweep(Kernel::Tile,
                 "B4 4-token tile = FP64 decode for batch 1/3/4/7/8 (partial last tile)",
                 {1, 3, 4, 7, 8});
}
void group_b5() {
    kernel_sweep(Kernel::Split,
                 "B5/B6 split-K (chunk 17) + merge = FP64 decode, incl. cols < chunk", {1, 2});
    guard("B5/B6 split-K writes partial[part * rows + row] and merges every part", [] {
        // 3 rows x 100 cols, chunk 17 -> 6 parts; read the partial buffer back directly.
        int m = 3, n = 100, g = 32, parts = 6;
        auto p = oracle::pack(oracle::uniform(m * n, 5), m, n, g);
        auto x = oracle::uniform(n, 6);
        auto dp = upload(p.bytes);
        auto ds = upload(p.scales);
        auto dx = upload(x);
        float *dy = nullptr, *tmp = nullptr;
        CUDA_OK(cudaMalloc(&dy, m * sizeof(float)));
        CUDA_OK(cudaMalloc(&tmp, m * parts * sizeof(float)));
        launch_split(dp, ds, dx, dy, tmp, m, n, g, 17);
        CUDA_OK(cudaGetLastError());
        auto partial = download(tmp, size_t(m) * parts);
        auto y = download(dy, m);
        cudaFree(dp), cudaFree(ds), cudaFree(dx), cudaFree(dy), cudaFree(tmp);
        auto w = oracle::decode_all(p);
        double worst = 0, worst_y = 0;
        for (int r = 0; r < m; ++r) {
            double total = 0;
            for (int s = 0; s < parts; ++s) {
                double a = 0;
                for (int c = 17 * s; c < std::min(n, 17 * s + 17); ++c) a += w[r * n + c] * x[c];
                total += a;
                worst = std::max(worst, std::abs(partial[s * m + r] - a));
            }
            worst_y = std::max(worst_y, std::abs(y[r] - total));
        }
        check(worst < 1e-4 && worst_y < 1e-4,
              "B5/B6 split-K writes partial[part * rows + row] and merges every part",
              "partial err " + num(worst) + ", merged err " + num(worst_y));
    });
}

// ------------------------------- C -----------------------------------------------
void group_c() {
    guard("C frozen artifact (C3 search) -> inverse channel scales -> 4 CUDA paths = C4 and FP64",
          [] {
              int m = 31, n = 65, b = 7;
              auto w = oracle::uniform(m * n, 101);
              auto calib = oracle::uniform(13 * n, 103);
              for (int t = 0; t < 13; ++t) calib[t * n] *= 10;
              auto model = lowbit::search(w, calib, m, n, 13, 16, {0, .25f, .5f, .75f, 1});
              auto held = oracle::uniform(b * n, 107);
              auto scaled = held;
              for (int t = 0; t < b; ++t)
                  for (int k = 0; k < n; ++k) scaled[t * n + k] /= model.channels[k];
              auto cpu = lowbit::execute(model, held, b);
              auto ref = ref_int4(model.weight, scaled, b);
              double cpu_err = 0;
              for (size_t i = 0; i < cpu.size(); ++i)
                  cpu_err = std::max(cpu_err, std::abs(cpu[i] - ref.y[i]));
              const Packed &p = model.weight;
              auto dp = upload(p.bytes);
              auto ds = upload(p.scales);
              auto dx = upload(scaled);
              float *dy = nullptr, *tmp = nullptr;
              CUDA_OK(cudaMalloc(&dy, size_t(b) * m * sizeof(float)));
              CUDA_OK(cudaMalloc(&tmp, size_t(m) * ((n + 16) / 17) * sizeof(float)));
              double worst = 0;
              for (int mode = 0; mode < 4; ++mode) {
                  CUDA_OK(cudaMemset(dy, 0xff, size_t(b) * m * sizeof(float)));
                  if (mode == 2) launch_batch(dp, ds, dx, dy, m, n, p.group, b);
                  for (int t = 0; mode != 2 && t < b; ++t) {
                      const float *in = dx + size_t(t) * n;
                      float *out = dy + size_t(t) * m;
                      if (mode == 0) launch_fused(dp, ds, in, out, m, n, p.group);
                      if (mode == 1) launch_pair(dp, ds, in, out, m, n, p.group);
                      if (mode == 3) launch_split(dp, ds, in, out, tmp, m, n, p.group, 17);
                  }
                  CUDA_OK(cudaGetLastError());
                  worst = std::max(worst, worst_ratio(download(dy, size_t(b) * m), ref));
              }
              cudaFree(dp), cudaFree(ds), cudaFree(dx), cudaFree(dy), cudaFree(tmp);
              check(worst <= 1 && cpu_err < 1e-9,
                    "C frozen artifact (C3 search) -> inverse channel scales -> 4 CUDA paths = C4 "
                    "and FP64",
                    "GPU " + num(worst) + "x tolerance, C4 vs FP64 " + num(cpu_err));
          });
}

// ------------------------------- R -----------------------------------------------
__global__ void hw_e4m3(const float *x, uint8_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __nv_cvt_float_to_fp8(x[i], __NV_SATFINITE, __NV_E4M3);
}
// Inputs that exercise every rounding rule of E4M3.
std::vector<float> e4m3_inputs() {
    std::vector<float> v;
    for (int c = 0; c < 256; ++c) {
        double a = oracle::e4m3_value(uint8_t(c));
        if (a != a) continue;
        v.push_back(float(a)); // exact codes, incl. +-0 and subnormals
    }
    for (int c = 0; c < 0x7E; ++c) { // midpoints (ties -> even) and points just off them
        double lo = oracle::e4m3_value(uint8_t(c)), hi = oracle::e4m3_value(uint8_t(c + 1)),
               mid = (lo + hi) / 2;
        for (double t : {mid, std::nextafter(mid, 0.0), std::nextafter(mid, 1e9)})
            v.push_back(float(t)), v.push_back(-float(t));
    }
    for (float t : {448.f, 449.f, 463.99f, 464.f, 480.f, 1e6f, 3e38f, INFINITY, -INFINITY, -449.f,
                    -1e6f, std::ldexp(1.f, -10), std::ldexp(1.f, -10) * 1.001f,
                    std::ldexp(1.f, -11), 1e-30f, -1e-30f})
        v.push_back(t);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> e(-14, 10), sgn(-1, 1);
    for (int i = 0; i < 20000; ++i) v.push_back((sgn(rng) < 0 ? -1.f : 1.f) * std::exp2(e(rng)));
    return v;
}
bool same_code(uint8_t a, uint8_t b) {
    bool na = (a & 0x7F) == 0x7F, nb = (b & 0x7F) == 0x7F; // any NaN encoding
    return na || nb ? na == nb : a == b;
}
void group_r1() {
    auto inputs = e4m3_inputs();
    guard("R1 oracle sanity: brute-force nearest = sm_89 cvt.rn.satfinite.e4m3 on all test inputs",
          [&] {
              float *dx = upload(inputs);
              uint8_t *dc = nullptr;
              CUDA_OK(cudaMalloc(&dc, inputs.size()));
              hw_e4m3<<<int(inputs.size() + 255) / 256, 256>>>(dx, dc, int(inputs.size()));
              CUDA_OK(cudaGetLastError());
              auto hw = download(dc, inputs.size());
              cudaFree(dx), cudaFree(dc);
              int bad = 0;
              float first = 0;
              for (size_t i = 0; i < inputs.size(); ++i)
                  if (!same_code(hw[i], oracle::e4m3_nearest(inputs[i])))
                      first = bad++ ? first : inputs[i];
              check(bad == 0,
                    "R1 oracle sanity: brute-force nearest = sm_89 cvt.rn.satfinite.e4m3 on all "
                    "test inputs",
                    std::to_string(bad) + " disagreements, first x = " + num(first));
          });
    struct Case {
        const char *name;
        std::vector<float> xs;
    };
    std::vector<float> exact, ties, sat, tiny, randoms;
    for (int c = 0; c < 256; ++c)
        if (oracle::e4m3_value(uint8_t(c)) == oracle::e4m3_value(uint8_t(c)))
            exact.push_back(float(oracle::e4m3_value(uint8_t(c))));
    for (int c = 0; c < 0x7E; ++c) {
        double mid = (oracle::e4m3_value(uint8_t(c)) + oracle::e4m3_value(uint8_t(c + 1))) / 2;
        ties.push_back(float(mid)), ties.push_back(-float(mid));
    }
    sat = {448.f, 449.f, 464.f, 480.f, 1e6f, 3e38f, INFINITY, -INFINITY, -449.f, -1e6f};
    tiny = {std::ldexp(1.f, -9),
            std::ldexp(1.f, -10),
            std::ldexp(1.f, -10) * 1.001f,
            std::ldexp(3.f, -10),
            std::ldexp(1.f, -11),
            1e-30f,
            -1e-30f,
            -std::ldexp(5.f, -10)};
    for (size_t i = inputs.size() - 20000; i < inputs.size(); ++i) randoms.push_back(inputs[i]);
    std::vector<Case> cases = {
        {"R1 every finite code round-trips exactly (incl. -0 and subnormals)", exact},
        {"R1 midpoints round to the even mantissa (RNE)", ties},
        {"R1 |x| >= 448 and +-inf saturate to +-448 (0x7E/0xFE), never NaN", sat},
        {"R1 subnormal range: 2^-9 is the smallest code, 2^-10 ties to 0, sign kept", tiny},
        {"R1 20000 log-uniform values = brute-force nearest", randoms},
    };
    for (auto &c : cases)
        guard(c.name, [&] {
            int bad = 0;
            std::string first;
            for (float x : c.xs) {
                uint8_t want = oracle::e4m3_nearest(x), got = fp8::encode_e4m3(x);
                if (!same_code(got, want) && !bad++) {
                    char buf[64];
                    std::snprintf(buf, sizeof buf, ": want 0x%02X got 0x%02X", unsigned(want),
                                  unsigned(got));
                    first = "x=" + num(x) + buf;
                }
            }
            check(bad == 0, c.name, std::to_string(bad) + " wrong; first " + first);
        });
    guard("R1 NaN -> a NaN code (exponent and mantissa all ones)", [] {
        uint8_t a = fp8::encode_e4m3(NAN), b = fp8::encode_e4m3(-NAN);
        check((a & 0x7F) == 0x7F && (b & 0x7F) == 0x7F,
              "R1 NaN -> a NaN code (exponent and mantissa all ones)",
              "got " + std::to_string(a) + ", " + std::to_string(b));
    });
}
void group_r2() {
    guard("R2 hand fixture: groups of 16 get their own absmax/448 (1 and 0.01)", [] {
        std::vector<float> w(32, 0);
        w[0] = 448, w[1] = 1, w[2] = -2, w[16] = 4.48f, w[17] = -0.01f, w[20] = 1.12f;
        auto p = fp8::fp8_pack(w, 1, 32, 16);
        bool ok = p.scales.size() == 2 && chk::rel(p.scales[0], 1) < 1e-6 &&
                  chk::rel(p.scales[1], 0.01) < 1e-6;
        // codes: 448 -> 0x7E, 1 -> 0x38, -2 -> 0xC0, 4.48/0.01 = 448 -> 0x7E, -1 -> 0xB8, 112 -> 0x6E
        ok = ok && p.codes.size() == 32 && p.codes[0] == 0x7E && p.codes[1] == 0x38 &&
             p.codes[2] == 0xC0 && p.codes[16] == 0x7E && p.codes[17] == 0xB8 &&
             p.codes[20] == 0x6E && p.codes[3] == 0;
        check(ok, "R2 hand fixture: groups of 16 get their own absmax/448 (1 and 0.01)",
              p.scales.size() == 2 ? "scales " + num(p.scales[0]) + ", " + num(p.scales[1])
                                   : "wrong number of scales");
    });
    guard("R2 scales = FP64 per-group absmax/448, codes = nearest E4M3 of w/s, padding 0", [] {
        int bad = 0;
        double worst = 0;
        for (int m : {1, 5})
            for (int n : {1, 15, 16, 17, 100, 257})
                for (int g : {16, 32, 128}) {
                    auto w = oracle::gaussian(size_t(m) * n, 3 + m * n + g);
                    if (n > 3) w[2] = 0; // and an all-zero row group below
                    if (m > 1) std::fill(w.begin() + n, w.begin() + n + std::min(n, g), 0.f);
                    auto p = fp8::fp8_pack(w, m, n, g);
                    int stride = oracle::round_up16(n), ng = (n + g - 1) / g;
                    if (p.stride != stride || p.codes.size() != size_t(m) * stride ||
                        p.scales.size() != size_t(m) * ng) {
                        bad += 1000;
                        continue;
                    }
                    for (int r = 0; r < m; ++r)
                        for (int j = 0; j < ng; ++j) {
                            double a = 0;
                            for (int c = j * g; c < std::min(n, j * g + g); ++c)
                                a = std::max(a, std::abs(double(w[r * n + c])));
                            worst = std::max(worst,
                                             chk::rel(p.scales[r * ng + j], a > 0 ? a / 448 : 1));
                        }
                    for (int r = 0; r < m; ++r)
                        for (int c = 0; c < stride; ++c) {
                            uint8_t got = p.codes[size_t(r) * stride + c];
                            if (c >= n) {
                                bad += got != 0;
                                continue;
                            }
                            float s = p.scales[r * ng + c / g];
                            bad += !same_code(got, oracle::e4m3_nearest(w[r * n + c] / s));
                        }
                }
        check(bad == 0 && worst < 1e-6,
              "R2 scales = FP64 per-group absmax/448, codes = nearest E4M3 of w/s, padding 0",
              std::to_string(bad) + " wrong bytes, scale rel err " + num(worst));
    });
    guard("R2 rejects group not a multiple of 16, bad shapes and non-finite weights", [] {
        std::vector<float> w(32, 1);
        bool ok = chk::rejects([&] { fp8::fp8_pack(w, 1, 32, 8); }) &&
                  chk::rejects([&] { fp8::fp8_pack(w, 1, 32, 0); }) &&
                  chk::rejects([&] { fp8::fp8_pack(w, 2, 32, 16); });
        w[3] = NAN;
        ok = ok && chk::rejects([&] { fp8::fp8_pack(w, 1, 32, 16); });
        check(ok, "R2 rejects group not a multiple of 16, bad shapes and non-finite weights",
              "accepted");
    });
}
// Oracle FP8 artifact; padding codes are 0x7F (NaN) so that using them is visible.
fp8::Fp8Packed oracle_fp8(const std::vector<float> &w, int m, int n, int g) {
    int stride = oracle::round_up16(n), ng = (n + g - 1) / g;
    fp8::Fp8Packed p{m,
                     n,
                     g,
                     stride,
                     std::vector<uint8_t>(size_t(m) * stride, 0x7F),
                     std::vector<float>(size_t(m) * ng)};
    for (int r = 0; r < m; ++r)
        for (int j = 0; j < ng; ++j) {
            double a = 0;
            for (int c = j * g; c < std::min(n, j * g + g); ++c)
                a = std::max(a, std::abs(double(w[r * n + c])));
            float s = float(a > 0 ? a / 448 : 1);
            p.scales[r * ng + j] = s;
            for (int c = j * g; c < std::min(n, j * g + g); ++c)
                p.codes[size_t(r) * stride + c] = oracle::e4m3_nearest(w[r * n + c] / s);
        }
    return p;
}
Ref ref_fp8(const fp8::Fp8Packed &p, const std::vector<float> &x, int batch) {
    int ng = (p.cols + p.group - 1) / p.group;
    Ref r{std::vector<double>(size_t(batch) * p.rows), std::vector<double>(size_t(batch) * p.rows)};
    for (int t = 0; t < batch; ++t)
        for (int i = 0; i < p.rows; ++i) {
            double a = 0, m = 0;
            for (int c = 0; c < p.cols; ++c) {
                double v = oracle::e4m3_value(p.codes[size_t(i) * p.stride + c]) *
                           p.scales[i * ng + c / p.group] * x[size_t(t) * p.cols + c];
                a += v, m += std::abs(v);
            }
            r.y[size_t(t) * p.rows + i] = a, r.mag[size_t(t) * p.rows + i] = m;
        }
    return r;
}
void fp8_sweep(bool tiled, const std::string &name, std::vector<int> batches) {
    guard(name, [&] {
        double worst = 0;
        std::string where;
        for (int m : {1, 7, 33})
            for (int n : {1, 15, 16, 17, 100, 257, 1000})
                for (int g : {16, 32, 128})
                    for (int b : batches) {
                        auto p =
                            oracle_fp8(oracle::gaussian(size_t(m) * n, 9 + m * n + g), m, n, g);
                        auto x = oracle::uniform(b * n, 11 + b + n);
                        auto dc = upload(p.codes);
                        auto ds = upload(p.scales);
                        auto dx = upload(x, 64, NAN); // NaN right after the last token
                        float *dy = nullptr;
                        CUDA_OK(cudaMalloc(&dy, size_t(b) * m * sizeof(float)));
                        CUDA_OK(cudaMemset(dy, 0xff, size_t(b) * m * sizeof(float)));
                        if (tiled)
                            fp8::launch_fp8_batch(dc, ds, dx, dy, m, n, g, b);
                        else
                            for (int t = 0; t < b; ++t)
                                fp8::launch_fp8_gemv(dc, ds, dx + size_t(t) * n, dy + size_t(t) * m,
                                                     m, n, g);
                        CUDA_OK(cudaGetLastError());
                        auto got = download(dy, size_t(b) * m);
                        cudaFree(dc), cudaFree(ds), cudaFree(dx), cudaFree(dy);
                        double e = worst_ratio(got, ref_fp8(p, x, b));
                        if (e > worst) {
                            worst = e;
                            where = std::to_string(m) + "x" + std::to_string(n) + " g" +
                                    std::to_string(g) + " batch " + std::to_string(b);
                        }
                    }
        check(worst <= 1, name, "worst error " + num(worst) + "x tolerance at " + where);
    });
}
void group_r3() {
    fp8_sweep(
        false,
        "R3 FP8 GEMV = FP64 decode (63 shapes; NaN padding codes and NaN after x must be ignored)",
        {1, 2});
}
void group_r4() {
    fp8_sweep(true, "R4 FP8 8-token tile = FP64 decode for batch 1/3/8/9/17 (partial tiles)",
              {1, 3, 8, 9, 17});
}
void group_r5() {
    using fp8::Format;
    guard("R5 weight bytes: hand-computed sizes for 4 formats at 2048x2048 g128 and 3x17 g16", [] {
        bool ok =
            fp8::weight_bytes(Format::INT4, 2048, 2048, 128) == 2048u * 1024 + 4u * 2048 * 16 &&
            fp8::weight_bytes(Format::FP8, 2048, 2048, 128) == 2048u * 2048 + 4u * 2048 * 16 &&
            fp8::weight_bytes(Format::FP16, 2048, 2048, 128) == 2u * 2048 * 2048 &&
            fp8::weight_bytes(Format::FP32, 2048, 2048, 128) == 4u * 2048 * 2048 &&
            fp8::weight_bytes(Format::INT4, 3, 17, 16) == 3u * 9 + 4u * 3 * 2 &&
            fp8::weight_bytes(Format::FP8, 3, 17, 16) == 3u * 32 + 4u * 3 * 2;
        check(ok,
              "R5 weight bytes: hand-computed sizes for 4 formats at 2048x2048 g128 and 3x17 g16",
              "INT4 2048^2 g128 = " +
                  std::to_string(fp8::weight_bytes(Format::INT4, 2048, 2048, 128)) +
                  " (want 2228224), FP8 3x17 g16 = " +
                  std::to_string(fp8::weight_bytes(Format::FP8, 3, 17, 16)) + " (want 120)");
    });
    guard("R5 bench plan: every timed rep starts with the weights evicted from L2", [] {
        cudaDeviceProp prop{};
        CUDA_OK(cudaGetDeviceProperties(&prop, 0));
        std::string why =
            plan::cold_violations(size_t(prop.l2CacheSize), [](int f, int r, int c, int g) {
                return fp8::weight_bytes(Format(f), r, c, g);
            });
        check(why.empty(), "R5 bench plan: every timed rep starts with the weights evicted from L2",
              "L2 " + std::to_string(prop.l2CacheSize >> 20) + " MiB; warm configs:" + why);
    });
}
} // namespace

int main(int argc, char **argv) {
    std::string group = argc > 1 ? argv[1] : "";
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("FAIL %s: no CUDA device\n", group.c_str());
        return 1;
    }
    if (group == "B1")
        group_b1();
    else if (group == "B2")
        group_b2();
    else if (group == "B3")
        group_b3();
    else if (group == "B4")
        group_b4();
    else if (group == "B5")
        group_b5();
    else if (group == "C")
        group_c();
    else if (group == "R1")
        group_r1();
    else if (group == "R2")
        group_r2();
    else if (group == "R3")
        group_r3();
    else if (group == "R4")
        group_r4();
    else if (group == "R5")
        group_r5();
    else {
        std::printf("FAIL unknown group '%s'\n", group.c_str());
        return 1;
    }
    return chk::summary(group);
}
