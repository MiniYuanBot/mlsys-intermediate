#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace kv {
enum class Axis { Token, Channel };
struct Packed {
    size_t rows, cols, group;
    Axis axis;
    std::vector<uint8_t> codes;
    std::vector<float> scales, offsets;
    size_t bytes() const { return codes.size() + 4 * (scales.size() + offsets.size()); }
};
inline size_t group_index(size_t r, size_t c, size_t rows, size_t cols, size_t g, Axis axis) {
    return axis == Axis::Token ? r * ((cols + g - 1) / g) + c / g
                               : c * ((rows + g - 1) / g) + r / g;
}
inline Packed quantize(const std::vector<float> &x, size_t rows, size_t cols, size_t g, Axis axis) {
    // Asymmetric INT4; store min as offset, tail groups supported. Use double
    // for x-min so opposite finite FP32 extrema do not overflow. Clamp nonzero
    // scale to at least denorm_min; reject nonfinite values, not valid extrema.
    // BEGIN TODO A1
    if (!rows || !cols || !g || x.size() != rows * cols) throw std::invalid_argument("shape/group");
    for (float v : x)
        if (!std::isfinite(v)) throw std::invalid_argument("nonfinite");
    size_t ng = axis == Axis::Token ? rows * ((cols + g - 1) / g) : cols * ((rows + g - 1) / g);
    Packed p{rows,
             cols,
             g,
             axis,
             std::vector<uint8_t>((x.size() + 1) / 2, 0),
             std::vector<float>(ng),
             std::vector<float>(ng)};
    std::vector<float> lo(ng, INFINITY), hi(ng, -INFINITY);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            size_t j = group_index(r, c, rows, cols, g, axis);
            lo[j] = std::min(lo[j], x[r * cols + c]);
            hi[j] = std::max(hi[j], x[r * cols + c]);
        }
    for (size_t j = 0; j < ng; ++j) {
        p.scales[j] = hi[j] == lo[j] ? 1.f
                                     : std::max(std::numeric_limits<float>::denorm_min(),
                                                float((double(hi[j]) - double(lo[j])) / 15.0));
        p.offsets[j] = lo[j];
    }
    for (size_t i = 0; i < x.size(); ++i) {
        size_t j = group_index(i / cols, i % cols, rows, cols, g, axis);
        int q = std::clamp(int(std::lround((double(x[i]) - double(lo[j])) / double(p.scales[j]))),
                           0, 15);
        p.codes[i / 2] |= uint8_t(q << (4 * (i % 2)));
    }
    return p;
    // END TODO A1
}
inline std::vector<float> dequantize(const Packed &p) {
    // Inverse layout; do not interpret nibble 8..15 as negative. Reconstruct in
    // double and saturate to finite FP32 range (rounded scale can overshoot).
    // BEGIN TODO A2
    std::vector<float> x(p.rows * p.cols);
    for (size_t i = 0; i < x.size(); ++i) {
        size_t j = group_index(i / p.cols, i % p.cols, p.rows, p.cols, p.group, p.axis);
        int q = (p.codes[i / 2] >> (4 * (i % 2))) & 15;
        double reconstructed = double(p.offsets[j]) + q * double(p.scales[j]);
        double limit = std::numeric_limits<float>::max();
        x[i] = float(std::clamp(reconstructed, -limit, limit));
    }
    return x;
    // END TODO A2
}
inline std::vector<size_t> select_tokens(const std::vector<float> &scores, size_t budget,
                                         size_t sink, size_t recent) {
    // select mandatory prefix/tail then highest past-only scores.
    // BEGIN TODO B1
    size_t n = scores.size();
    if (budget > n || sink > n || recent > n) throw std::invalid_argument("budget/range");
    for (float v : scores)
        if (!std::isfinite(v)) throw std::invalid_argument("nonfinite");
    std::vector<size_t> keep, candidates;
    for (size_t i = 0; i < n; ++i) {
        if (i < sink || i >= n - recent)
            keep.push_back(i);
        else
            candidates.push_back(i);
    }
    if (keep.size() > budget) throw std::invalid_argument("mandatory tokens exceed budget");
    std::stable_sort(candidates.begin(), candidates.end(),
                     [&](size_t a, size_t b) { return scores[a] > scores[b]; });
    keep.insert(keep.end(), candidates.begin(), candidates.begin() + (budget - keep.size()));
    std::sort(keep.begin(), keep.end());
    return keep;
    // END TODO B1
}
// Provided stable attention oracle. Inputs are [tokens, dim]; no GQA batching here.
inline std::vector<float> attention(const std::vector<float> &q, const std::vector<float> &k,
                                    const std::vector<float> &v, const std::vector<size_t> &keep) {
    const size_t d = q.size();
    if (keep.empty() || !d || k.size() != v.size() || k.size() % d)
        throw std::invalid_argument("attention shape");
    std::vector<double> scores;
    double mx = -INFINITY;
    for (size_t t : keep) {
        if (t >= k.size() / d) throw std::out_of_range("token");
        double s = 0;
        for (size_t j = 0; j < d; ++j) s += double(q[j]) * k[t * d + j];
        scores.push_back(s / std::sqrt(double(d)));
        mx = std::max(mx, scores.back());
    }
    double sum = 0;
    for (double &s : scores) {
        s = std::exp(s - mx);
        sum += s;
    }
    std::vector<float> out(d, 0);
    for (size_t i = 0; i < keep.size(); ++i)
        for (size_t j = 0; j < d; ++j) out[j] += float(scores[i] / sum * v[keep[i] * d + j]);
    return out;
}
} // namespace kv
