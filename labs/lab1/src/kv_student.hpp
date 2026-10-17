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
    throw std::logic_error("Complete A1");
    // END TODO A1
}
inline std::vector<float> dequantize(const Packed &p) {
    // Inverse layout; do not interpret nibble 8..15 as negative. Reconstruct in
    // double and saturate to finite FP32 range (rounded scale can overshoot).
    // BEGIN TODO A2
    throw std::logic_error("Complete A2");
    // END TODO A2
}
inline std::vector<size_t> select_tokens(const std::vector<float> &scores, size_t budget,
                                         size_t sink, size_t recent) {
    // select mandatory prefix/tail then highest past-only scores.
    // BEGIN TODO B1
    throw std::logic_error("Complete B1");
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
