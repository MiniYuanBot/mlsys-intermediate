// Host-side fixture for stage R (provided; shared by decode_test.cu and decode_bench.cu).
// Builds raw K/V with Qwen3-like structure, quantizes it into the A/B layouts with an
// independent host quantizer (it does not call kv::quantize or R1), decodes it back in
// FP64, and computes FP64 reference attention.
#pragma once
#include "decode_student.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_OK(call)                                                                              \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("FAIL CUDA %s at %s:%d: %s\n", #call, __FILE__, __LINE__,                  \
                        cudaGetErrorString(err_));                                                 \
            std::printf("  (a kernel that still contains asm(\"trap;\") ends here: finish the "    \
                        "TODO it calls)\n");                                                       \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace fixture {
constexpr int D = kvdec::kHeadDim;

struct Config {
    int q_heads = 16, kv_heads = 8, n_chunks = 5, block = 64, vgroup = 32, n_res = 37;
    int tokens() const { return n_chunks * block + n_res; }
};

// Raw history in FP32, [kv_heads][T][D]. K has a few outlier channels with a large
// per-channel offset (the structure per-channel K quantization is designed for); V has a
// per-token magnitude that drifts over time (the structure per-token V quantization fits).
struct Raw {
    Config cfg;
    std::vector<float> k, v, q; // q: [q_heads][D]
};
inline Raw make_raw(const Config &cfg, unsigned seed, float q_scale = 0.35f) {
    Raw r{cfg, {}, {}, {}};
    std::mt19937 rng(seed);
    std::normal_distribution<float> n01;
    size_t T = cfg.tokens();
    r.k.resize((size_t)cfg.kv_heads * T * D);
    r.v.resize(r.k.size());
    for (int h = 0; h < cfg.kv_heads; ++h)
        for (size_t t = 0; t < T; ++t)
            for (int c = 0; c < D; ++c) {
                size_t i = ((size_t)h * T + t) * D + c;
                bool outlier = (c * 7 + h) % 29 == 0;
                r.k[i] = (outlier ? 6.f + 0.5f * h : 0.3f * std::sin(0.1f * c + h)) +
                         (outlier ? 1.5f : 0.6f) * n01(rng);
                float mag = 0.5f + 0.5f * std::fabs(std::sin(0.013f * float(t) + h));
                r.v[i] = mag * n01(rng);
            }
    r.q.resize((size_t)cfg.q_heads * D);
    for (float &x : r.q) x = q_scale * n01(rng);
    return r;
}

// Host copy of the device layout (see Int4Cache in decode_student.cuh).
struct Packed {
    Config cfg;
    std::vector<uint8_t> k_codes, v_codes;
    std::vector<float> k_scale, k_offset, v_scale, v_offset, res_k, res_v;
};
inline void affine(const float *x, size_t n, size_t stride, float &scale, float &offset) {
    double lo = INFINITY, hi = -INFINITY;
    for (size_t i = 0; i < n; ++i)
        lo = std::min(lo, double(x[i * stride])), hi = std::max(hi, double(x[i * stride]));
    offset = float(lo);
    scale = hi == lo ? 1.f : float((hi - lo) / 15.0);
}
inline int code_of(float x, float scale, float offset) {
    return std::clamp(int(std::lround((double(x) - offset) / scale)), 0, 15);
}
inline Packed quantize(const Raw &r) {
    const Config &c = r.cfg;
    size_t T = c.tokens(), Tq = (size_t)c.n_chunks * c.block, gv = D / c.vgroup;
    Packed p{c, {}, {}, {}, {}, {}, {}, {}, {}};
    p.k_codes.assign((size_t)c.kv_heads * Tq * D / 2, 0);
    p.v_codes = p.k_codes;
    p.k_scale.resize((size_t)c.kv_heads * c.n_chunks * D);
    p.k_offset = p.k_scale;
    p.v_scale.resize((size_t)c.kv_heads * Tq * gv);
    p.v_offset = p.v_scale;
    for (int h = 0; h < c.kv_heads; ++h) {
        const float *K = r.k.data() + (size_t)h * T * D, *V = r.v.data() + (size_t)h * T * D;
        for (int b = 0; b < c.n_chunks; ++b)
            for (int ch = 0; ch < D;
                 ++ch) { // per-channel K: one group = block tokens of one channel
                size_t g = ((size_t)h * c.n_chunks + b) * D + ch;
                affine(K + (size_t)b * c.block * D + ch, c.block, D, p.k_scale[g], p.k_offset[g]);
            }
        for (size_t t = 0; t < Tq; ++t)
            for (size_t j = 0; j < gv;
                 ++j) { // per-token V: one group = vgroup channels of one token
                size_t g = ((size_t)h * Tq + t) * gv + j;
                affine(V + t * D + j * c.vgroup, c.vgroup, 1, p.v_scale[g], p.v_offset[g]);
            }
        for (size_t t = 0; t < Tq; ++t)
            for (int ch = 0; ch < D; ++ch) {
                size_t lin = ((size_t)h * Tq + t) * D + ch;
                size_t gk = ((size_t)h * c.n_chunks + t / c.block) * D + ch;
                size_t g = ((size_t)h * Tq + t) * gv + ch / c.vgroup;
                p.k_codes[lin / 2] |= uint8_t(code_of(K[t * D + ch], p.k_scale[gk], p.k_offset[gk])
                                              << (4 * (lin % 2)));
                p.v_codes[lin / 2] |=
                    uint8_t(code_of(V[t * D + ch], p.v_scale[g], p.v_offset[g]) << (4 * (lin % 2)));
            }
        p.res_k.insert(p.res_k.end(), K + Tq * D, K + T * D);
        p.res_v.insert(p.res_v.end(), V + Tq * D, V + T * D);
    }
    return p;
}
// FP64 decode of the host layout, [kv_heads][T][D] (independent of R1).
inline void decode(const Packed &p, std::vector<double> &k, std::vector<double> &v) {
    const Config &c = p.cfg;
    size_t T = c.tokens(), Tq = (size_t)c.n_chunks * c.block, gv = D / c.vgroup;
    k.assign((size_t)c.kv_heads * T * D, 0);
    v = k;
    for (int h = 0; h < c.kv_heads; ++h)
        for (size_t t = 0; t < T; ++t)
            for (int ch = 0; ch < D; ++ch) {
                size_t out = ((size_t)h * T + t) * D + ch;
                if (t >= Tq) {
                    k[out] = p.res_k[((size_t)h * c.n_res + t - Tq) * D + ch];
                    v[out] = p.res_v[((size_t)h * c.n_res + t - Tq) * D + ch];
                    continue;
                }
                size_t lin = ((size_t)h * Tq + t) * D + ch;
                int ck = (p.k_codes[lin / 2] >> (4 * (lin % 2))) & 15,
                    cv = (p.v_codes[lin / 2] >> (4 * (lin % 2))) & 15;
                size_t gk = ((size_t)h * c.n_chunks + t / c.block) * D + ch,
                       g = ((size_t)h * Tq + t) * gv + ch / c.vgroup;
                k[out] = double(p.k_offset[gk]) + double(p.k_scale[gk]) * ck;
                v[out] = double(p.v_offset[g]) + double(p.v_scale[g]) * cv;
            }
}
// FP64 attention for every query head; query head h reads KV head h / (q_heads/kv_heads).
template <class Vec>
std::vector<double> attention(const Config &c, const std::vector<float> &q, const Vec &k,
                              const Vec &v) {
    size_t T = c.tokens();
    int G = c.q_heads / c.kv_heads;
    std::vector<double> out((size_t)c.q_heads * D, 0);
    for (int h = 0; h < c.q_heads; ++h) {
        int kh = h / G;
        std::vector<double> s(T);
        double mx = -INFINITY, z = 0;
        for (size_t t = 0; t < T; ++t) {
            double dot = 0;
            for (int ch = 0; ch < D; ++ch)
                dot += double(q[(size_t)h * D + ch]) * double(k[((size_t)kh * T + t) * D + ch]);
            mx = std::max(mx, s[t] = dot / std::sqrt(double(D)));
        }
        for (size_t t = 0; t < T; ++t) z += s[t] = std::exp(s[t] - mx);
        for (size_t t = 0; t < T; ++t)
            for (int ch = 0; ch < D; ++ch)
                out[(size_t)h * D + ch] += s[t] / z * double(v[((size_t)kh * T + t) * D + ch]);
    }
    return out;
}
// Worst-case |o' - o| per (head, channel) from quantization alone (lecture §3.4):
//   eps_h = max_t sum_c |q_hc| sK_{c,b(t)} / 2 / sqrt(D)   (logit perturbation bound)
//   bound = max_t |dV_tc| + (e^{2 eps} - 1) * (max_t v_tc - min_t v_tc) / 2
inline std::vector<double> error_bound(const Raw &r, const Packed &p) {
    const Config &c = r.cfg;
    size_t T = c.tokens(), Tq = (size_t)c.n_chunks * c.block, gv = D / c.vgroup;
    int G = c.q_heads / c.kv_heads;
    std::vector<double> bound((size_t)c.q_heads * D);
    for (int h = 0; h < c.q_heads; ++h) {
        int kh = h / G;
        double eps = 0;
        for (int b = 0; b < c.n_chunks; ++b) {
            double e = 0;
            for (int ch = 0; ch < D; ++ch)
                e += std::fabs(r.q[(size_t)h * D + ch]) *
                     p.k_scale[((size_t)kh * c.n_chunks + b) * D + ch] / 2;
            eps = std::max(eps, e / std::sqrt(double(D)));
        }
        for (int ch = 0; ch < D; ++ch) {
            double dv = 0, lo = INFINITY, hi = -INFINITY;
            for (size_t t = 0; t < T; ++t) {
                double x = r.v[((size_t)kh * T + t) * D + ch];
                lo = std::min(lo, x), hi = std::max(hi, x);
                if (t < Tq)
                    dv = std::max(
                        dv, double(p.v_scale[((size_t)kh * Tq + t) * gv + ch / c.vgroup]) / 2);
            }
            bound[(size_t)h * D + ch] = dv + std::expm1(2 * eps) * (hi - lo) / 2;
        }
    }
    return bound;
}

template <class T> T *to_device(const std::vector<T> &x) {
    T *d = nullptr;
    CUDA_OK(cudaMalloc(&d, std::max<size_t>(1, x.size()) * sizeof(T)));
    if (!x.empty()) CUDA_OK(cudaMemcpy(d, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> std::vector<T> to_host(const T *d, size_t n) {
    std::vector<T> x(n);
    CUDA_OK(cudaMemcpy(x.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return x;
}
struct DeviceInt4 {
    kvdec::Int4Cache cache{};
    std::vector<void *> owned;
    explicit DeviceInt4(const Packed &p) {
        auto own = [&](auto *ptr) {
            owned.push_back((void *)ptr);
            return ptr;
        };
        cache.k_codes = own(to_device(p.k_codes));
        cache.v_codes = own(to_device(p.v_codes));
        cache.k_scale = own(to_device(p.k_scale));
        cache.k_offset = own(to_device(p.k_offset));
        cache.v_scale = own(to_device(p.v_scale));
        cache.v_offset = own(to_device(p.v_offset));
        cache.res_k = own(to_device(p.res_k));
        cache.res_v = own(to_device(p.res_v));
        cache.n_chunks = p.cfg.n_chunks, cache.block = p.cfg.block, cache.vgroup = p.cfg.vgroup,
        cache.n_res = p.cfg.n_res;
    }
    ~DeviceInt4() {
        for (void *ptr : owned) cudaFree(ptr);
    }
    DeviceInt4(const DeviceInt4 &) = delete;
    DeviceInt4 &operator=(const DeviceInt4 &) = delete;
};
// Logical bytes the fused int4 decode must read for one layer (codes + metadata + residual).
inline size_t int4_kv_bytes(const Config &c) {
    size_t Tq = (size_t)c.n_chunks * c.block;
    return (size_t)c.kv_heads *
           (2 * Tq * D / 2 + 2 * sizeof(float) * (size_t)c.n_chunks * D +
            2 * sizeof(float) * Tq * (D / c.vgroup) + 2 * sizeof(float) * (size_t)c.n_res * D);
}
inline size_t fp16_kv_bytes(const Config &c) {
    return (size_t)c.kv_heads * c.tokens() * D * 2 * 2;
}
} // namespace fixture
