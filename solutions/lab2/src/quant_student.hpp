#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lowbit {
struct Packed {
    int rows, cols, group;
    std::vector<uint8_t> bytes;
    std::vector<float> scales;
};
// A1: W is row-major [rows, cols]. A group is `group` consecutive columns of ONE
// row; return scales[r * ceil(cols/group) + g] = max |W[r, c]| / 7 over that group
// (FP32), 1 for an all-zero group. The last group of a row ends at cols. Throw
// std::invalid_argument for rows/cols/group <= 0, a size mismatch or a non-finite
// weight. Tests: A1 hand fixtures (group != row), FP64 per-group oracle.
inline std::vector<float> calibrate(const std::vector<float> &w, int rows, int cols, int group) {
    // BEGIN TODO A1
    if (rows <= 0 || cols <= 0 || group <= 0 || w.size() != size_t(rows) * cols)
        throw std::invalid_argument("shape");
    const int ng = (cols + group - 1) / group;
    std::vector<float> scales(rows * ng);
    for (int row = 0; row < rows; ++row) {
        for (int g = 0; g < ng; ++g) {
            float maximum = 0;
            const int end = std::min(cols, (g + 1) * group);
            for (int col = g * group; col < end; ++col) {
                float value = w[row * cols + col];
                if (!std::isfinite(value)) throw std::invalid_argument("nonfinite weight");
                maximum = std::max(maximum, std::abs(value));
            }
            scales[row * ng + g] = maximum > 0 ? maximum / 7 : 1;
        }
    }
    return scales;
    // END TODO A1
}
// A2: q = clamp(lround(w / s), -7, 7) with FP32 w and the A1 scale s (ties away
// from zero), stored as u = q + 8. Column c of row r lives in byte
// r * ceil(cols/2) + c/2, even c in the low nibble. An odd row's unused high nibble
// is 0. Tests: A2 hand codes, oracle codes for every element, row stride.
inline Packed pack(const std::vector<float> &w, int rows, int cols, int group) {
    // BEGIN TODO A2
    Packed out{rows, cols, group, {}, calibrate(w, rows, cols, group)};
    const int stride = (cols + 1) / 2;
    const int ng = (cols + group - 1) / group;
    out.bytes.assign(rows * stride, 0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float scale = out.scales[row * ng + col / group];
            int quant = int(std::lround(w[row * cols + col] / scale));
            quant = std::clamp(quant, -7, 7);
            const int offset = row * stride + col / 2;
            const int shift = 4 * (col % 2);
            out.bytes[offset] |= uint8_t((quant + 8) << shift);
        }
    }
    return out;
    // END TODO A2
}
// A3: W_hat[r, c] = (u - 8) * s. Throw std::invalid_argument when the shape is
// invalid, bytes/scales have the wrong length, or a scale is not finite and > 0.
// The padding nibble is not a weight. Tests: A3 hand-built artifact, rejections.
inline std::vector<float> unpack(const Packed &p) {
    // BEGIN TODO A3
    if (p.rows <= 0 || p.cols <= 0 || p.group <= 0) throw std::invalid_argument("shape");
    const int stride = (p.cols + 1) / 2;
    const int ng = (p.cols + p.group - 1) / p.group;
    if (p.bytes.size() != size_t(p.rows) * stride || p.scales.size() != size_t(p.rows) * ng)
        throw std::invalid_argument("artifact length");
    for (float scale : p.scales)
        if (!(scale > 0) || !std::isfinite(scale)) throw std::invalid_argument("scale");
    std::vector<float> out(p.rows * p.cols);
    for (int row = 0; row < p.rows; ++row) {
        for (int col = 0; col < p.cols; ++col) {
            int byte = p.bytes[row * stride + col / 2];
            int quant = ((byte >> (4 * (col % 2))) & 15) - 8;
            out[row * p.cols + col] = quant * p.scales[row * ng + col / p.group];
        }
    }
    return out;
    // END TODO A3
}
// A4: Y = X W_hat^T with X [batch, cols], result [batch, rows], accumulated in
// double straight from the packed bytes (do not call unpack()). Throw
// std::invalid_argument on an X size mismatch. Test: A4 vs the test's own decoder.
inline std::vector<double> packed_linear(const Packed &p, const std::vector<float> &x, int batch) {
    // BEGIN TODO A4
    if (batch <= 0 || x.size() != size_t(batch) * p.cols) throw std::invalid_argument("X shape");
    const int stride = (p.cols + 1) / 2;
    const int ng = (p.cols + p.group - 1) / p.group;
    std::vector<double> out(batch * p.rows, 0);
    for (int b = 0; b < batch; ++b) {
        for (int row = 0; row < p.rows; ++row) {
            double accumulator = 0;
            for (int col = 0; col < p.cols; ++col) {
                int code = p.bytes.at(row * stride + col / 2);
                int value = ((code >> (4 * (col % 2))) & 15) - 8;
                double weight = double(value) * p.scales.at(row * ng + col / p.group);
                accumulator += weight * x[b * p.cols + col];
            }
            out[b * p.rows + row] = accumulator;
        }
    }
    return out;
    // END TODO A4
}
// C1: AWQ-style per-input-channel statistic. m_k = mean_t |x[t, k]| over the
// calibration rows; s_k = max(m_k, 1e-4)^alpha; then divide every s_k by
// sqrt(max_j s_j * min_j s_j) so the scales are centred around 1 in log space.
// alpha = 0 gives all ones. Throw std::invalid_argument for alpha outside [0, 1],
// a size mismatch or a non-finite activation. Tests: C1 hand-computed 2x3 X.
inline std::vector<float> channel_scales(const std::vector<float> &x, int batch, int cols,
                                         float alpha) {
    // BEGIN TODO C1
    if (batch <= 0 || cols <= 0 || x.size() != size_t(batch) * cols || alpha < 0 || alpha > 1)
        throw std::invalid_argument("calibration shape/alpha");
    std::vector<float> scale(cols);
    for (int col = 0; col < cols; ++col) {
        double magnitude = 0;
        for (int b = 0; b < batch; ++b) {
            float value = x[b * cols + col];
            if (!std::isfinite(value)) throw std::invalid_argument("nonfinite activation");
            magnitude += std::abs(value);
        }
        scale[col] = std::pow(std::max(magnitude / batch, 1e-4), alpha);
    }
    auto limits = std::minmax_element(scale.begin(), scale.end());
    const float normalizer = std::sqrt(*limits.first * *limits.second);
    for (float &value : scale) value /= normalizer;
    return scale;
    // END TODO C1
}
struct Transformed {
    std::vector<float> weight, input;
};
// C2: W' = W diag(s) (column k of W times s_k) and X' = X diag(s)^-1, as new
// row-major copies; X' W'^T = X W^T in exact arithmetic. Throw
// std::invalid_argument for a non-positive scale or size mismatch. Test: C2.
inline Transformed transform(const std::vector<float> &w, const std::vector<float> &x, int rows,
                             int cols, int batch, const std::vector<float> &scale) {
    // BEGIN TODO C2
    if (scale.size() != size_t(cols) || w.size() != size_t(rows) * cols ||
        x.size() != size_t(batch) * cols)
        throw std::invalid_argument("transform shape");
    Transformed out{w, x};
    for (int col = 0; col < cols; ++col) {
        float s = scale[col];
        if (!(s > 0) || !std::isfinite(s)) throw std::invalid_argument("channel scale");
        for (int row = 0; row < rows; ++row) {
            const int index = row * cols + col;
            out.weight[index] = w[index] * s;
        }
        for (int b = 0; b < batch; ++b) {
            const int index = b * cols + col;
            out.input[index] = x[index] / s;
        }
    }
    return out;
    // END TODO C2
}
struct Calibrated {
    Packed weight;
    std::vector<float> channels;
    float alpha;
    double calibration_mse;
};
// C3: for every alpha: s = C1, (W', X') = C2, quantize W' with A2, output via A4 on
// X', MSE against X W^T with the ORIGINAL float W. Return the first minimizer.
// alphas must contain 0 (the plain-RTN baseline). Tests: C3 vs an in-test search,
// outlier-channel case where alpha > 0 must win.
inline Calibrated search(const std::vector<float> &w, const std::vector<float> &x, int rows,
                         int cols, int batch, int group, const std::vector<float> &alphas) {
    // BEGIN TODO C3
    if (alphas.empty() || std::find(alphas.begin(), alphas.end(), 0.f) == alphas.end())
        throw std::invalid_argument("alpha=0 baseline required");
    std::vector<double> reference(batch * rows, 0);
    for (int b = 0; b < batch; ++b)
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < cols; ++col)
                reference[b * rows + row] += double(w.at(row * cols + col)) * x.at(b * cols + col);
    Calibrated best;
    best.calibration_mse = std::numeric_limits<double>::infinity();
    for (float alpha : alphas) {
        auto channels = channel_scales(x, batch, cols, alpha);
        auto changed = transform(w, x, rows, cols, batch, channels);
        auto quant = pack(changed.weight, rows, cols, group);
        auto output = packed_linear(quant, changed.input, batch);
        double error = 0;
        for (size_t i = 0; i < output.size(); ++i) {
            double delta = output[i] - reference[i];
            error += delta * delta;
        }
        error /= output.size();
        if (error < best.calibration_mse) best = {quant, channels, alpha, error};
    }
    return best;
    // END TODO C3
}
// C4: run a FROZEN artifact on new X: divide X by the stored channel scales once,
// then the packed linear. Do not re-estimate scales. Test: C4 held-out oracle.
inline std::vector<double> execute(const Calibrated &model, const std::vector<float> &x,
                                   int batch) {
    // BEGIN TODO C4
    const int cols = model.weight.cols;
    if (batch <= 0 || x.size() != size_t(batch) * cols || model.channels.size() != size_t(cols))
        throw std::invalid_argument("execution shape");
    std::vector<float> scaled(x.size());
    for (int b = 0; b < batch; ++b) {
        for (int col = 0; col < cols; ++col) {
            float scale = model.channels[col];
            if (!(scale > 0)) throw std::invalid_argument("scale");
            scaled[b * cols + col] = x[b * cols + col] / scale;
        }
    }
    auto output = packed_linear(model.weight, scaled, batch);
    return output;
    // END TODO C4
}
// A5: max_half_step_ratio = max |W - W_hat| / (s / 2), mse = mean (W - W_hat)^2,
// bytes = payload bytes + 4 * number of scales (no object overhead). For an
// unclipped absmax artifact the ratio is <= 1. Test: A5 vs in-test recomputation.
struct Certificate {
    double max_half_step_ratio, mse;
    size_t bytes;
};
inline Certificate certificate(const std::vector<float> &w, const Packed &p) {
    // BEGIN TODO A5
    auto restored = unpack(p);
    if (w.size() != restored.size()) throw std::invalid_argument("certificate shape");
    const int ng = (p.cols + p.group - 1) / p.group;
    double worst = 0;
    double squared = 0;
    for (int row = 0; row < p.rows; ++row) {
        for (int col = 0; col < p.cols; ++col) {
            const int index = row * p.cols + col;
            const double error = std::abs(double(w[index]) - restored[index]);
            const double half_step = p.scales[row * ng + col / p.group] / 2;
            const double ratio = error / half_step;
            worst = std::max(worst, ratio);
            squared += error * error;
        }
    }
    const size_t bytes = p.bytes.size() + sizeof(float) * p.scales.size();
    return {worst, squared / w.size(), bytes};
    // END TODO A5
}
// C5: for each group size whose artifact fits the budget (payload + 4 B per group
// scale + 4 B per channel scale, counted even for alpha = 0), run C3; return the
// first min-MSE candidate; throw std::runtime_error if none fits. Test: C5.
inline Calibrated search_budget(const std::vector<float> &w, const std::vector<float> &x, int rows,
                                int cols, int batch, const std::vector<int> &groups, size_t budget,
                                const std::vector<float> &alphas) {
    // BEGIN TODO C5
    bool found = false;
    Calibrated best;
    best.calibration_mse = std::numeric_limits<double>::infinity();
    for (int group : groups) {
        if (group <= 0) throw std::invalid_argument("group");
        const size_t payload = size_t(rows) * ((cols + 1) / 2);
        const size_t metadata = sizeof(float) * size_t(rows) * ((cols + group - 1) / group);
        const size_t channels = sizeof(float) * size_t(cols);
        if (payload + metadata + channels > budget) continue;
        auto candidate = search(w, x, rows, cols, batch, group, alphas);
        if (!found || candidate.calibration_mse < best.calibration_mse) {
            best = candidate;
            found = true;
        }
    }
    if (!found) throw std::runtime_error("no feasible group size");
    return best;
    // END TODO C5
}
// A6: per (row, group) try s = ratio * absmax/7 for every ratio in (0, 1], keep the
// first ratio with the smallest weight SSE (codes clamped to [-7,7]), and store the
// winning scale AND the codes re-quantized with it. Throw std::invalid_argument for
// an empty list or a ratio outside (0, 1]. Test: A6 exhaustive per-group oracle.
inline Packed clip_pack(const std::vector<float> &w, int rows, int cols, int group,
                        const std::vector<float> &ratios) {
    // BEGIN TODO A6
    if (ratios.empty()) throw std::invalid_argument("empty clipping candidates");
    Packed out = pack(w, rows, cols, group);
    const int ng = (cols + group - 1) / group;
    const int stride = (cols + 1) / 2;
    for (float ratio : ratios)
        if (!(ratio > 0 && ratio <= 1)) throw std::invalid_argument("clip ratio");
    for (int row = 0; row < rows; ++row) {
        for (int g = 0; g < ng; ++g) {
            const float base = out.scales[row * ng + g];
            float selected = base;
            double best = std::numeric_limits<double>::infinity();
            for (float ratio : ratios) {
                const float scale = base * ratio;
                double error = 0;
                for (int col = g * group; col < std::min(cols, (g + 1) * group); ++col) {
                    const float value = w[row * cols + col];
                    const int q = std::clamp(int(std::lround(value / scale)), -7, 7);
                    const double delta = value - q * scale;
                    error += delta * delta;
                }
                if (error < best) {
                    best = error;
                    selected = scale;
                }
            }
            out.scales[row * ng + g] = selected;
            for (int col = g * group; col < std::min(cols, (g + 1) * group); ++col) {
                const int q = std::clamp(int(std::lround(w[row * cols + col] / selected)), -7, 7);
                const int index = row * stride + col / 2;
                const int shift = 4 * (col % 2);
                out.bytes[index] =
                    uint8_t((out.bytes[index] & ~(15 << shift)) | ((q + 8) << shift));
            }
        }
    }
    return out;
    // END TODO A6
}
} // namespace lowbit
