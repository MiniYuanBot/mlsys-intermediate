#include "src/quant_student.hpp"
#include <iostream>
#include <random>
#include <string>
using namespace lowbit;
inline void require(bool ok, const char *what) {
    if (!ok) throw std::runtime_error(what);
}
inline void close(double a, double b, double tol = 1e-5) {
    require(std::abs(a - b) <= tol * (1 + std::abs(b)), "numeric mismatch");
}
inline std::vector<float> random_values(int n, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-2, 2);
    std::vector<float> x(n);
    for (auto &v : x) v = d(rng);
    return x;
}
inline std::vector<double> dense(const std::vector<float> &w, const std::vector<float> &x, int m,
                                 int n, int b) {
    std::vector<double> y(b * m, 0);
    for (int t = 0; t < b; ++t)
        for (int r = 0; r < m; ++r)
            for (int k = 0; k < n; ++k) y[t * m + r] += double(w[r * n + k]) * x[t * n + k];
    return y;
}
inline double mse(const std::vector<double> &a, const std::vector<double> &b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i) e += (a[i] - b[i]) * (a[i] - b[i]);
    return e / a.size();
}
void test_a() {
    auto z = pack(std::vector<float>(14, 0), 2, 7, 3);
    for (float s : z.scales) close(s, 1);
    require(z.bytes.back() == 8, "unused high tail nibble must be zero");
    auto codes = pack({-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}, 1, 15, 15);
    for (int k = 0; k < 15; ++k)
        require(((codes.bytes[k / 2] >> (4 * (k % 2))) & 15) == k + 1, "packing_bits");
    for (int m : {1, 3, 7})
        for (int n : {1, 3, 17, 65})
            for (int g : {3, 8, 128}) {
                auto w = random_values(m * n, 17);
                auto p = pack(w, m, n, g);
                auto restored = unpack(p);
                auto cert = certificate(w, p);
                require(cert.max_half_step_ratio <= 1.00002, "quantization bound");
                require(cert.bytes == size_t(m * ((n + 1) / 2) + 4 * m * ((n + g - 1) / g)),
                        "byte accounting");
                for (int b : {1, 3, 7}) {
                    auto x = random_values(b * n, 23);
                    auto actual = packed_linear(p, x, b);
                    auto expected = dense(restored, x, m, n, b);
                    for (size_t i = 0; i < actual.size(); ++i) close(actual[i], expected[i]);
                }
            }
    auto cw = random_values(3 * 17, 117);
    cw[0] = 12;
    auto cp = clip_pack(cw, 3, 17, 8, {1, .75f, .5f});
    auto cr = unpack(cp);
    auto ordinary = unpack(pack(cw, 3, 17, 8));
    double clipped_error = 0, ordinary_error = 0;
    for (size_t i = 0; i < cw.size(); ++i) {
        clipped_error += std::pow(cw[i] - cr[i], 2);
        ordinary_error += std::pow(cw[i] - ordinary[i], 2);
    }
    require(clipped_error <= ordinary_error + 1e-5, "clipping baseline domination");
    for (int row = 0; row < 3; ++row)
        for (int group = 0; group < 3; ++group) {
            double expected = 1e100;
            float scale = 1;
            float base = calibrate(cw, 3, 17, 8)[row * 3 + group];
            for (float ratio : {1.f, .75f, .5f}) {
                double err = 0;
                for (int k = group * 8; k < std::min(17, (group + 1) * 8); ++k) {
                    float value = cw[row * 17 + k];
                    int q = std::clamp(int(std::lround(value / (base * ratio))), -7, 7);
                    err += std::pow(value - q * base * ratio, 2);
                }
                if (err < expected) {
                    expected = err;
                    scale = base * ratio;
                }
            }
            close(cp.scales[row * 3 + group], scale);
        }
    bool caught = false;
    try {
        calibrate({1}, 1, 1, 0);
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    require(caught, "invalid group");
    auto broken = z;
    broken.scales.clear();
    caught = false;
    try {
        unpack(broken);
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    require(caught, "artifact validation");
    caught = false;
    try {
        calibrate({NAN}, 1, 1, 1);
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    require(caught, "nonfinite weight");
    std::cout << "PASS A1 scales_and_zero_tail; A2 packing_bits; A3 "
                 "roundtrip_error_and_invalid_artifact; A4 independent_batch_oracle; A5 "
                 "certificate; A6 clipping_exhaustive\n";
}
void test_c(bool bench) {
    for (int n : {7, 31, 65}) {
        int m = 5, b = 9;
        auto w = random_values(m * n, 51);
        auto x = random_values(b * n, 67);
        for (int t = 0; t < b; ++t) x[t * n] *= 12;
        auto original = w;
        auto inputs = x;
        std::vector<float> alphas{0, .25, .5, .75, 1};
        auto identity = channel_scales(x, b, n, 0);
        for (float s : identity) close(s, 1);
        auto scales = channel_scales(x, b, n, .5);
        auto t = transform(w, x, m, n, b, scales);
        auto before = dense(w, x, m, n, b), after = dense(t.weight, t.input, m, n, b);
        for (size_t i = 0; i < before.size(); ++i) close(after[i], before[i]);
        auto chosen = search(w, x, m, n, b, 8, alphas);
        double best = 1e100;
        float alpha = -1;
        for (float a : alphas) {
            auto s = channel_scales(x, b, n, a);
            auto transformed = transform(w, x, m, n, b, s);
            auto restored = unpack(pack(transformed.weight, m, n, 8));
            auto output = dense(restored, transformed.input, m, n, b);
            double error = mse(output, before);
            if (error < best) {
                best = error;
                alpha = a;
            }
        }
        close(chosen.calibration_mse, best);
        close(chosen.alpha, alpha);
        require(w == original && x == inputs, "mutated source");
        auto held = random_values(3 * n, 101);
        auto output = execute(chosen, held, 3);
        auto wt = unpack(chosen.weight);
        for (int r = 0; r < m; ++r)
            for (int k = 0; k < n; ++k) wt[r * n + k] /= chosen.channels[k];
        auto expected = dense(wt, held, m, n, 3);
        for (size_t i = 0; i < output.size(); ++i) close(output[i], expected[i]);
        size_t budget = m * ((n + 1) / 2) + 4 * m * ((n + 8 - 1) / 8) + 4 * n;
        auto budgeted = search_budget(w, x, m, n, b, {2, 8, 32}, budget, alphas);
        require(budgeted.weight.group >= 8, "budget not respected");
        double oracle = std::min(search(w, x, m, n, b, 8, alphas).calibration_mse,
                                 search(w, x, m, n, b, 32, alphas).calibration_mse);
        close(budgeted.calibration_mse, oracle);
        bool caught = false;
        try {
            search_budget(w, x, m, n, b, {8}, 1, alphas);
        } catch (const std::runtime_error &) {
            caught = true;
        }
        require(caught, "infeasible budget");
        if (bench) {
            auto baseline = search(w, x, m, n, b, 8, {0});
            auto clipped = clip_pack(w, m, n, 8, {1, .75f, .5f});
            size_t artifact_bytes = budgeted.weight.bytes.size() +
                                    4 * budgeted.weight.scales.size() +
                                    4 * budgeted.channels.size();
            std::cout << n << ',' << chosen.alpha << ',' << best << ','
                      << mse(output, dense(w, held, m, n, 3)) << ',' << baseline.calibration_mse
                      << ',' << mse(execute(baseline, held, 3), dense(w, held, m, n, 3)) << ','
                      << mse(packed_linear(clipped, x, b), before) << ','
                      << mse(packed_linear(clipped, held, 3), dense(w, held, m, n, 3)) << ','
                      << budgeted.weight.group << ',' << artifact_bytes << ','
                      << budgeted.calibration_mse << '\n';
        }
    }
    if (!bench)
        std::cout
            << "PASS C1 channel_scales_identity; C2 equivalent_channel_transform; C3 "
               "exhaustive_search_and_no_mutation; C4 heldout_frozen_artifact; C5 budget_search\n";
}
#ifndef LOWBIT_NO_MAIN
int main(int argc, char **argv) {
    try {
        std::string part = argc > 1 ? argv[1] : "all";
        bool bench = argc > 2;
        if (part == "A" || part == "all") test_a();
        if (bench)
            std::cout << "cols,selected_alpha,calibration_mse,heldout_mse,baseline_calibration_mse,"
                         "baseline_heldout_mse,clipping_calibration_mse,clipping_heldout_mse,"
                         "budget_group,artifact_bytes,budget_calibration_mse\n";
        if (part == "C" || part == "all") test_c(bench);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
#endif
