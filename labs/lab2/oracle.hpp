// Independent oracles for lab2. Nothing here calls a student function: every
// reference is recomputed from the raw inputs in FP64 (or by brute force), so a
// wrong A1 cannot make its own check pass through A2/A4/A5, and C3 cannot be
// validated with the student's C1/C2. Only the plain data structs come from src/.
#pragma once
#include "src/quant_student.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace oracle {
using lowbit::Packed;

inline std::vector<float> uniform(int n, int seed, float lo = -2, float hi = 2) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> x(n);
    for (auto &v : x) v = d(rng);
    return x;
}
inline std::vector<float> gaussian(size_t n, int seed, float sigma = 1) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0, sigma);
    std::vector<float> x(n);
    for (auto &v : x) v = d(rng);
    return x;
}
// y[t, r] = sum_k w[r, k] x[t, k] in FP64; w row-major [m, n], x row-major [b, n].
inline std::vector<double> dense(const std::vector<float> &w, const std::vector<float> &x, int m,
                                 int n, int b) {
    std::vector<double> y(size_t(b) * m, 0);
    for (int t = 0; t < b; ++t)
        for (int r = 0; r < m; ++r) {
            double acc = 0;
            for (int k = 0; k < n; ++k) acc += double(w[size_t(r) * n + k]) * x[size_t(t) * n + k];
            y[size_t(t) * m + r] = acc;
        }
    return y;
}
inline double mse(const std::vector<double> &a, const std::vector<double> &b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i) e += (a[i] - b[i]) * (a[i] - b[i]);
    return e / a.size();
}
inline double rel_l2(const std::vector<double> &a, const std::vector<double> &b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) num += (a[i] - b[i]) * (a[i] - b[i]), den += b[i] * b[i];
    return std::sqrt(num / (den + 1e-300));
}

// ---------- INT4 symmetric format (A) -------------------------------------------
// Scale of columns [c0, c1) of row r: absmax / 7 in FP64, 1 for an all-zero group.
inline double group_scale(const std::vector<float> &w, int cols, int r, int c0, int c1) {
    double m = 0;
    for (int c = c0; c < c1; ++c) m = std::max(m, std::abs(double(w[size_t(r) * cols + c])));
    return m > 0 ? m / 7 : 1;
}
inline std::vector<double> scales(const std::vector<float> &w, int rows, int cols, int group) {
    int ng = (cols + group - 1) / group;
    std::vector<double> s(size_t(rows) * ng);
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < ng; ++g)
            s[size_t(r) * ng + g] =
                group_scale(w, cols, r, g * group, std::min(cols, (g + 1) * group));
    return s;
}
// Contract rounding: q = clamp(lround(w / s), -7, 7) with FP32 w and FP32 s.
inline int code_of(float w, float s) {
    return std::clamp(int(std::lround(w / s)), -7, 7);
}
// Decode with the test's own rule: low nibble = even column, value = (u - 8) * scale.
inline int nibble_at(const Packed &p, int r, int c) {
    int stride = (p.cols + 1) / 2;
    return (p.bytes[size_t(r) * stride + c / 2] >> (4 * (c % 2))) & 15;
}
inline double decode(const Packed &p, int r, int c) {
    int ng = (p.cols + p.group - 1) / p.group;
    return double(nibble_at(p, r, c) - 8) * p.scales[size_t(r) * ng + c / p.group];
}
inline std::vector<double> decode_all(const Packed &p) {
    std::vector<double> w(size_t(p.rows) * p.cols);
    for (int r = 0; r < p.rows; ++r)
        for (int c = 0; c < p.cols; ++c) w[size_t(r) * p.cols + c] = decode(p, r, c);
    return w;
}
// Pack with scales given by the oracle (FP32-rounded); used to build kernel inputs
// that do not depend on the student's A1/A2.
inline Packed pack(const std::vector<float> &w, int rows, int cols, int group) {
    Packed p{rows, cols, group, {}, {}};
    auto s = scales(w, rows, cols, group);
    p.scales.assign(s.begin(), s.end());
    int stride = (cols + 1) / 2, ng = (cols + group - 1) / group;
    p.bytes.assign(size_t(rows) * stride, 0);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            int q = code_of(w[size_t(r) * cols + c], p.scales[size_t(r) * ng + c / group]);
            p.bytes[size_t(r) * stride + c / 2] |= uint8_t((q + 8) << (4 * (c % 2)));
        }
    return p;
}
inline std::vector<double> linear(const Packed &p, const std::vector<float> &x, int batch) {
    auto w = decode_all(p);
    std::vector<double> y(size_t(batch) * p.rows, 0);
    for (int t = 0; t < batch; ++t)
        for (int r = 0; r < p.rows; ++r) {
            double acc = 0;
            for (int c = 0; c < p.cols; ++c)
                acc += w[size_t(r) * p.cols + c] * x[size_t(t) * p.cols + c];
            y[size_t(t) * p.rows + r] = acc;
        }
    return y;
}

// ---------- AWQ-style channel scaling (C) ---------------------------------------
// s_k = max(mean_t |x_tk|, 1e-4)^alpha / sqrt(max_j s_j * min_j s_j), in FP64.
inline std::vector<double> channel_scales(const std::vector<float> &x, int batch, int cols,
                                          double alpha) {
    std::vector<double> s(cols);
    for (int k = 0; k < cols; ++k) {
        double m = 0;
        for (int t = 0; t < batch; ++t) m += std::abs(double(x[size_t(t) * cols + k]));
        s[k] = std::pow(std::max(m / batch, 1e-4), alpha);
    }
    auto mm = std::minmax_element(s.begin(), s.end());
    double norm = std::sqrt(*mm.first * *mm.second);
    for (auto &v : s) v /= norm;
    return s;
}
struct SearchResult {
    double alpha, mse;
    std::vector<double> mse_per_alpha;
};
// Exhaustive alpha search built only from this file: FP32 scale, W*s and X/s in FP32
// (the contract's arithmetic), oracle quantizer, FP64 output, MSE against original W.
inline SearchResult search(const std::vector<float> &w, const std::vector<float> &x, int rows,
                           int cols, int batch, int group, const std::vector<float> &alphas) {
    auto reference = dense(w, x, rows, cols, batch);
    SearchResult best{-1, std::numeric_limits<double>::infinity(), {}};
    for (float a : alphas) {
        auto s64 = channel_scales(x, batch, cols, a);
        std::vector<float> s(s64.begin(), s64.end()), ws(w), xs(x);
        for (int r = 0; r < rows; ++r)
            for (int k = 0; k < cols; ++k)
                ws[size_t(r) * cols + k] = w[size_t(r) * cols + k] * s[k];
        for (int t = 0; t < batch; ++t)
            for (int k = 0; k < cols; ++k)
                xs[size_t(t) * cols + k] = x[size_t(t) * cols + k] / s[k];
        double e = mse(linear(pack(ws, rows, cols, group), xs, batch), reference);
        best.mse_per_alpha.push_back(e);
        if (e < best.mse) best.mse = e, best.alpha = a;
    }
    return best;
}

// ---------- FP8 E4M3 (R) --------------------------------------------------------
// OCP / NVIDIA e4m3 ("fn"): bias 7, no infinities, S.1111.111 is NaN, max 448,
// smallest subnormal 2^-9. Decoded from the bit fields, not from a library.
inline double e4m3_value(uint8_t b) {
    int sign = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return std::numeric_limits<double>::quiet_NaN();
    double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1 + m / 8.0, e - 7);
    return sign ? -v : v;
}
// Brute-force nearest code over all 127 non-negative finite codes; ties to the code
// with an even mantissa LSB; |x| >= 448 (incl. inf) saturates to 448; NaN -> 0x7F.
inline uint8_t e4m3_nearest(double x) {
    if (std::isnan(x)) return 0x7F;
    uint8_t sign = std::signbit(x) ? 0x80 : 0;
    double a = std::abs(x);
    if (a >= 448) return sign | 0x7E;
    int best = 0;
    double err = std::numeric_limits<double>::infinity();
    for (int c = 0; c <= 0x7E; ++c) {
        double d = std::abs(e4m3_value(uint8_t(c)) - a);
        if (d < err || (d == err && (c & 1) == 0)) err = d, best = c;
    }
    return sign | uint8_t(best);
}
inline int round_up16(int n) {
    return (n + 15) / 16 * 16;
}
} // namespace oracle
