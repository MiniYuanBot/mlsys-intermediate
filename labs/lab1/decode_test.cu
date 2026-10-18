// Stage R tests. Usage: decode_test R1|R2|R3|fused. exercise.py runs each in its own
// process, so a kernel that traps (an unfinished TODO) fails only its own group.
// Oracles: an independent host quantizer/decoder and FP64 attention (r_fixture.cuh),
// an FP64 per-split recomputation for R2, and an FP64 log-sum-exp merge for R3.
#include "r_fixture.cuh"
#include <iostream>
#include <sstream>

namespace {
int g_pass = 0, g_fail = 0;
void check(bool ok, const std::string &name, const std::string &detail = "") {
    (ok ? ++g_pass : ++g_fail);
    std::cout << (ok ? "PASS " : "FAIL ") << name << (ok ? "" : ": " + detail) << '\n';
}
std::string num(double x) {
    std::ostringstream s;
    s.precision(4);
    s << x;
    return s.str();
}
using fixture::D;

__global__ void probe_load4(kvdec::Int4Cache cache, float *k, float *v) {
    int lane = threadIdx.x, t = blockIdx.x, kh = blockIdx.y;
    float a[4], b[4];
    if (t < cache.prefix()) {
        cache.load4(kh, false, t, lane, a);
        cache.load4(kh, true, t, lane, b);
    } else {
        cache.load4_tail(kh, false, t, lane, a);
        cache.load4_tail(kh, true, t, lane, b);
    }
    size_t at = ((size_t)kh * cache.tokens() + t) * D + 4 * lane;
    for (int i = 0; i < 4; ++i) k[at + i] = a[i], v[at + i] = b[i];
}

void test_r1() {
    for (int vgroup : {32, 128}) {
        fixture::Config cfg;
        cfg.vgroup = vgroup;
        auto raw = fixture::make_raw(cfg, 11 + vgroup);
        auto packed = fixture::quantize(raw);
        std::vector<double> dk, dv;
        fixture::decode(packed, dk, dv);
        fixture::DeviceInt4 dev(packed);
        size_t n = (size_t)cfg.kv_heads * cfg.tokens() * D;
        float *gk, *gv;
        CUDA_OK(cudaMalloc(&gk, n * sizeof(float)));
        CUDA_OK(cudaMalloc(&gv, n * sizeof(float)));
        probe_load4<<<dim3(cfg.tokens(), cfg.kv_heads), 32>>>(dev.cache, gk, gv);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        auto hk = fixture::to_host(gk, n), hv = fixture::to_host(gv, n);
        cudaFree(gk), cudaFree(gv);
        size_t Tq = (size_t)cfg.n_chunks * cfg.block, T = cfg.tokens();
        double ek = 0, ev = 0, er = 0, bk = 0, bv = 0;
        for (int h = 0; h < cfg.kv_heads; ++h)
            for (size_t t = 0; t < T; ++t)
                for (int c = 0; c < D; ++c) {
                    size_t i = ((size_t)h * T + t) * D + c;
                    double xk = std::fabs(hk[i] - dk[i]) / (1 + std::fabs(dk[i]));
                    double xv = std::fabs(hv[i] - dv[i]) / (1 + std::fabs(dv[i]));
                    if (t >= Tq) {
                        er = std::max({er, xk, xv});
                        continue;
                    }
                    ek = std::max(ek, xk), ev = std::max(ev, xv);
                    // Semantics, not just layout: decoded values lie within half a step of raw.
                    double sk = packed.k_scale[((size_t)h * cfg.n_chunks + t / cfg.block) * D + c];
                    double sv = packed.v_scale[((size_t)h * Tq + t) * (D / vgroup) + c / vgroup];
                    bk = std::max(bk, std::fabs(hk[i] - raw.k[i]) - 0.5 * sk * (1 + 1e-4));
                    bv = std::max(bv, std::fabs(hv[i] - raw.v[i]) - 0.5 * sv * (1 + 1e-4));
                }
        std::string tag = " (vgroup=" + std::to_string(vgroup) + ")";
        check(ek < 1e-6,
              "R1 K prefix = host decode, per-channel scale/offset, low nibble first" + tag,
              "max rel err " + num(ek));
        check(ev < 1e-6, "R1 V prefix = host decode, per-token group scale/offset" + tag,
              "max rel err " + num(ev));
        check(er < 1e-7, "R1 fixture: residual rows via load4_tail are exact" + tag,
              "max rel err " + num(er));
        check(bk <= 1e-6 && bv <= 1e-6, "R1 decoded prefix within half a step of the raw K/V" + tag,
              "excess K " + num(bk) + ", V " + num(bv));
    }
}

// FP64 partial state of every (query head, split) from the dense FP16 values.
void test_r2() {
    struct Case {
        int n_chunks, n_res, split;
    };
    for (Case cs : {Case{5, 37, 100}, Case{5, 37, 64}, Case{0, 33, 8}, Case{2, 1, 129}}) {
        fixture::Config cfg;
        cfg.n_chunks = cs.n_chunks, cfg.n_res = cs.n_res;
        auto raw = fixture::make_raw(cfg, 7 + cs.split, 0.6f);
        int T = cfg.tokens();
        std::vector<__half> hk(raw.k.size()), hv(raw.v.size());
        for (size_t i = 0; i < hk.size(); ++i)
            hk[i] = __float2half(raw.k[i]), hv[i] = __float2half(raw.v[i]);
        kvdec::DecodeShape shape{cfg.q_heads, cfg.kv_heads, T, cs.split};
        int S = shape.n_splits();
        __half *dk = fixture::to_device(hk), *dv = fixture::to_device(hv);
        float *dq = fixture::to_device(raw.q), *pm, *pl, *pa;
        CUDA_OK(cudaMalloc(&pm, (size_t)cfg.q_heads * S * sizeof(float)));
        CUDA_OK(cudaMalloc(&pl, (size_t)cfg.q_heads * S * sizeof(float)));
        CUDA_OK(cudaMalloc(&pa, (size_t)cfg.q_heads * S * D * sizeof(float)));
        CUDA_OK(cudaMemset(pm, 0xff, (size_t)cfg.q_heads * S * sizeof(float))); // NaN: unwritten
        kvdec::Fp16Cache cache{dk, dv, T};
        dim3 grid((S + kvdec::kWarpsPerBlock - 1) / kvdec::kWarpsPerBlock, cfg.kv_heads);
        kvdec::split_partial<2><<<grid, kvdec::kWarpsPerBlock * 32>>>(cache, shape, dq, pm, pl, pa);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        auto m = fixture::to_host(pm, (size_t)cfg.q_heads * S), l = fixture::to_host(pl, m.size());
        auto acc = fixture::to_host(pa, m.size() * D);
        cudaFree(dk), cudaFree(dv), cudaFree(dq), cudaFree(pm), cudaFree(pl), cudaFree(pa);
        double em = 0, el = 0, ea = 0;
        for (int h = 0; h < cfg.q_heads; ++h)
            for (int s = 0; s < S; ++s) {
                int kh = h / 2, begin = s * cs.split, end = std::min(T, begin + cs.split);
                std::vector<double> score;
                double mx = -INFINITY;
                for (int t = begin; t < end; ++t) {
                    double dot = 0;
                    for (int c = 0; c < D; ++c)
                        dot += double(raw.q[(size_t)h * D + c]) *
                               double(__half2float(hk[((size_t)kh * T + t) * D + c]));
                    score.push_back(dot / std::sqrt(double(D)));
                    mx = std::max(mx, score.back());
                }
                double z = 0;
                std::vector<double> a(D, 0);
                for (int t = begin; t < end; ++t) {
                    double w = std::exp(score[t - begin] - mx);
                    z += w;
                    for (int c = 0; c < D; ++c)
                        a[c] += w * double(__half2float(hv[((size_t)kh * T + t) * D + c]));
                }
                size_t at = (size_t)h * S + s;
                em = std::max(em, std::isfinite(m[at]) ? std::fabs(m[at] - mx) : INFINITY);
                el = std::max(el, std::isfinite(l[at]) ? std::fabs(l[at] - z) / z : INFINITY);
                for (int c = 0; c < D; ++c) {
                    double got = acc[at * D + c] / l[at];
                    ea = std::max(ea, std::isfinite(got) ? std::fabs(got - a[c] / z) : INFINITY);
                }
            }
        std::string tag = " (T=" + std::to_string(T) + ", split=" + std::to_string(cs.split) +
                          ", S=" + std::to_string(S) + ")";
        check(em < 1e-4, "R2 part_m = max scaled score of exactly the split's tokens" + tag,
              "max abs err " + num(em));
        check(el < 1e-4, "R2 part_l = sum exp(score - m)" + tag, "max rel err " + num(el));
        check(ea < 1e-4, "R2 part_acc / part_l = split-local softmax-weighted V" + tag,
              "max abs err " + num(ea));
    }
}

void test_r3() {
    struct Case {
        int heads, splits;
        bool empty, huge;
    };
    for (Case cs : {Case{4, 6, true, true}, Case{3, 1, false, false}, Case{16, 37, false, true}}) {
        std::mt19937 rng(3 + cs.splits);
        std::uniform_real_distribution<float> u(-1, 1);
        size_t n = (size_t)cs.heads * cs.splits;
        std::vector<float> m(n), l(n), acc(n * D);
        for (size_t i = 0; i < n; ++i) {
            m[i] = 30 * u(rng) + (cs.huge && i % 3 == 0 ? 100.f : 0.f); // maxima 100+ apart
            l[i] = 1.5f + u(rng);
            for (int c = 0; c < D; ++c) acc[i * D + c] = l[i] * (u(rng) + 0.01f * c);
        }
        if (cs.empty)
            m[1] = -INFINITY, l[1] = 0, std::fill(acc.begin() + D, acc.begin() + 2 * D, 0.f);
        float *pm = fixture::to_device(m), *pl = fixture::to_device(l),
              *pa = fixture::to_device(acc), *out;
        CUDA_OK(cudaMalloc(&out, (size_t)cs.heads * D * sizeof(float)));
        kvdec::reduce_splits<<<cs.heads, D>>>(cs.splits, pm, pl, pa, out);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        auto got = fixture::to_host(out, (size_t)cs.heads * D);
        cudaFree(pm), cudaFree(pl), cudaFree(pa), cudaFree(out);
        double err = 0;
        for (int h = 0; h < cs.heads; ++h) {
            double M = -INFINITY;
            for (int s = 0; s < cs.splits; ++s)
                M = std::max(M, double(m[(size_t)h * cs.splits + s]));
            for (int c = 0; c < D; ++c) {
                double L = 0, A = 0;
                for (int s = 0; s < cs.splits; ++s) {
                    size_t i = (size_t)h * cs.splits + s;
                    double w = std::isinf(m[i]) ? 0 : std::exp(double(m[i]) - M);
                    L += w * l[i], A += w * acc[i * D + c];
                }
                double x = got[(size_t)h * D + c];
                err = std::max(err, std::isfinite(x) ? std::fabs(x - A / L) / (1 + std::fabs(A / L))
                                                     : INFINITY);
            }
        }
        std::string tag =
            " (heads=" + std::to_string(cs.heads) + ", splits=" + std::to_string(cs.splits) +
            (cs.empty ? ", one empty split" : "") + (cs.huge ? ", maxima 100 apart" : "") + ")";
        check(err < 1e-5, "R3 log-sum-exp merge = FP64 merge" + tag, "max rel err " + num(err));
    }
}

void test_fused() {
    for (int split : {100, 64, 357}) {
        fixture::Config cfg;
        auto raw = fixture::make_raw(cfg, 5);
        auto packed = fixture::quantize(raw);
        std::vector<double> dk, dv;
        fixture::decode(packed, dk, dv);
        auto oracle = fixture::attention(cfg, raw.q, dk, dv);      // same int4 values, FP64
        auto truth = fixture::attention(cfg, raw.q, raw.k, raw.v); // unquantized history
        auto bound = fixture::error_bound(raw, packed);
        fixture::DeviceInt4 dev(packed);
        kvdec::DecodeShape shape{cfg.q_heads, cfg.kv_heads, cfg.tokens(), split};
        int S = shape.n_splits();
        float *dq = fixture::to_device(raw.q), *pm, *pl, *pa, *out;
        CUDA_OK(cudaMalloc(&pm, (size_t)cfg.q_heads * S * sizeof(float)));
        CUDA_OK(cudaMalloc(&pl, (size_t)cfg.q_heads * S * sizeof(float)));
        CUDA_OK(cudaMalloc(&pa, (size_t)cfg.q_heads * S * D * sizeof(float)));
        CUDA_OK(cudaMalloc(&out, (size_t)cfg.q_heads * D * sizeof(float)));
        kvdec::decode_attention<2>(dev.cache, shape, dq, pm, pl, pa, out);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        auto got = fixture::to_host(out, (size_t)cfg.q_heads * D);
        cudaFree(dq), cudaFree(pm), cudaFree(pl), cudaFree(pa), cudaFree(out);
        double e1 = 0, worst = -INFINITY, err_truth = 0, mean_bound = 0;
        size_t bad_head = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            double x = std::isfinite(got[i]) ? got[i] : INFINITY;
            e1 = std::max(e1, std::fabs(x - oracle[i]));
            double excess = std::fabs(x - truth[i]) - (bound[i] + 1e-4);
            worst = std::max(worst, excess);
            err_truth = std::max(err_truth, std::fabs(x - truth[i]));
            mean_bound += bound[i] / got.size();
            if (excess > 0) bad_head = i / D;
        }
        std::string tag = " (T=" + std::to_string(cfg.tokens()) +
                          ", split=" + std::to_string(split) + ", S=" + std::to_string(S) + ")";
        check(e1 < 1e-4,
              "fused R1+R2+R3 = FP64 attention on the same int4 values, all 16 heads" + tag,
              "max abs err " + num(e1));
        check(worst <= 0, "fused vs unquantized FP64 within the derived quantization bound" + tag,
              "head " + std::to_string(bad_head) + " exceeds by " + num(worst));
        std::cout << "  info" << tag << ": max|fused - unquantized| = " << num(err_truth)
                  << ", mean bound = " << num(mean_bound) << '\n';
    }
}
} // namespace

int main(int argc, char **argv) {
    std::string what = argc > 1 ? argv[1] : "";
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "== R " << what << " on " << prop.name << " (sm_" << prop.major << prop.minor
              << ") ==\n";
    if (what == "R1")
        test_r1();
    else if (what == "R2")
        test_r2();
    else if (what == "R3")
        test_r3();
    else if (what == "fused")
        test_fused();
    else {
        std::cout << "usage: decode_test R1|R2|R3|fused\n";
        return 2;
    }
    std::cout << "SUMMARY R " << what << ": " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
