// Host tests for stage A (int4 format) and stage C (AWQ-style channel scaling).
// Usage: quant_test <A|C|all> [bench]. Each check prints PASS/FAIL and the run goes on;
// the exit code is 1 if any check failed. References come from oracle.hpp, which never
// calls student code: A1 is checked against hand-computed group scales, A2-A6 against
// the test's own packer/decoder, C1/C3/C5 against an in-test FP64 AWQ search.
#include "check.hpp"
#include "oracle.hpp"
#include <iostream>
#include <string>

using namespace lowbit;
using chk::check;
using chk::guard;
using chk::num;

namespace {
// Largest relative difference of two equally sized arrays.
template <class A, class B> double max_rel(const A &a, const B &b) {
    if (a.size() != b.size()) return 1e300;
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(double(a[i]) - double(b[i])) / (1e-30 + std::abs(double(b[i]))));
    return e;
}
template <class A, class B> double max_abs(const A &a, const B &b) {
    if (a.size() != b.size()) return 1e300;
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(double(a[i]) - double(b[i])));
    return e;
}
std::string vec(const std::vector<float> &v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + num(v[i]);
    return s + "]";
}
// Every code of p must equal the oracle code under p's OWN group scales, except within
// 1e-4 of a rounding tie (FP32 vs FP64 division may round a near-tie either way).
int code_mismatches(const std::vector<float> &w, const Packed &p) {
    int bad = 0, ng = (p.cols + p.group - 1) / p.group;
    for (int r = 0; r < p.rows; ++r)
        for (int c = 0; c < p.cols; ++c) {
            float s = p.scales[size_t(r) * ng + c / p.group];
            float v = w[size_t(r) * p.cols + c];
            int want = oracle::code_of(v, s) + 8, got = oracle::nibble_at(p, r, c);
            double frac =
                std::abs(std::abs(double(v) / s) - std::floor(std::abs(double(v) / s)) - 0.5);
            if (got != want && frac > 1e-4) ++bad;
        }
    return bad;
}
// Hand-built artifact used by A3/A4: rows 2, cols 3, group 2 (two groups per row,
// the second one a 1-column tail), row stride ceil(3/2) = 2 bytes.
Packed hand_artifact() {
    // byte 0x9F: c0 = low F (q=7), c1 = high 9 (q=1); byte 0x02: c2 = low 2 (q=-6).
    // byte 0x7C: c0 = low C (q=4), c1 = high 7 (q=-1); byte 0x0E: c2 = low E (q=6).
    return Packed{2, 3, 2, {0x9F, 0x02, 0x7C, 0x0E}, {0.5f, 2.f, 1.5f, 0.25f}};
}
const std::vector<float> kHandW = {3.5f, 0.5f, -12.f, 6.f, -1.5f, 1.5f};

void test_a() {
    // ---------------- A1 calibrate ----------------
    guard("A1 group absmax, not row absmax: [7,1,-1 | 0.7,0.1,0], g=3 -> [1, 0.1]", [] {
        auto s = calibrate({7, 1, -1, 0.7f, 0.1f, 0}, 1, 6, 3);
        check(s.size() == 2 && std::abs(s[0] - 1) < 1e-6 && std::abs(s[1] - 0.1f) < 1e-7,
              "A1 group absmax, not row absmax: [7,1,-1 | 0.7,0.1,0], g=3 -> [1, 0.1]",
              "expected [1,0.1] vs " + vec(s));
    });
    guard("A1 two rows with different maxima, zero group = 1, short tail group", [] {
        // row 0: [3,-6,1 | 0.5,0.25] -> [6/7, 0.5/7]; row 1: [0,0,0 | -2.1,14] -> [1, 2]
        auto s = calibrate({3, -6, 1, 0.5f, 0.25f, 0, 0, 0, -2.1f, 14}, 2, 5, 3);
        std::vector<float> want = {6.f / 7, 0.5f / 7, 1, 2};
        check(max_rel(s, want) < 1e-6,
              "A1 two rows with different maxima, zero group = 1, short tail group",
              vec(want) + " vs " + vec(s));
    });
    guard("A1 tail group stops at cols (does not read the next row)", [] {
        // row 0 tail = cols 4..6 = 0.5 -> 0.5/7; row 1 starts with 70.
        auto s = calibrate({1, 1, 1, 1, .5f, .5f, .5f, 70, 0, 0, 0, 0, 0, 0}, 2, 7, 4);
        std::vector<float> want = {1.f / 7, .5f / 7, 10, 1};
        check(max_rel(s, want) < 1e-6, "A1 tail group stops at cols (does not read the next row)",
              vec(want) + " vs " + vec(s));
    });
    guard("A1 per-group scales = FP64 oracle on 36 random shapes", [] {
        double worst = 0;
        for (int m : {1, 3, 7})
            for (int n : {1, 3, 17, 65})
                for (int g : {3, 8, 128}) {
                    auto w = oracle::uniform(m * n, 17 + m * n + g);
                    worst =
                        std::max(worst, max_rel(calibrate(w, m, n, g), oracle::scales(w, m, n, g)));
                }
        check(worst < 1e-6, "A1 per-group scales = FP64 oracle on 36 random shapes",
              "max rel err " + num(worst));
    });
    guard("A1 rejects group 0, rows 0, size mismatch, NaN and inf", [] {
        bool ok = chk::rejects([] { calibrate({1}, 1, 1, 0); }) &&
                  chk::rejects([] { calibrate({}, 0, 1, 1); }) &&
                  chk::rejects([] { calibrate({1, 2}, 1, 3, 1); }) &&
                  chk::rejects([] { calibrate({NAN}, 1, 1, 1); }) &&
                  chk::rejects([] { calibrate({1, INFINITY}, 1, 2, 2); });
        check(ok, "A1 rejects group 0, rows 0, size mismatch, NaN and inf",
              "an invalid input was accepted");
    });

    // ---------------- A2 pack ----------------
    guard("A2 known bytes: ties away from zero, q+8, even column in the low nibble", [] {
        // s = 1; q = [7, 3, -3, 1, -1, 2] (lround: 2.5->3, 0.5->1, 1.5->2)
        auto p = pack({7, 2.5f, -2.5f, 0.5f, -0.5f, 1.5f}, 1, 6, 6);
        std::vector<uint8_t> want = {0xBF, 0x95, 0xA7};
        check(p.bytes == want,
              "A2 known bytes: ties away from zero, q+8, even column in the low nibble",
              "expected BF 95 A7");
    });
    guard("A2 all 15 codes: W = -7..7 with group = 15 encodes u = q + 8", [] {
        auto p = pack({-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}, 1, 15, 15);
        bool ok = p.bytes.size() == 8;
        for (int k = 0; ok && k < 15; ++k) ok = oracle::nibble_at(p, 0, k) == k + 1;
        check(ok && (p.bytes[7] >> 4) == 0,
              "A2 all 15 codes: W = -7..7 with group = 15 encodes u = q + 8",
              "codes or tail nibble wrong");
    });
    guard("A2 odd width: row stride ceil(cols/2), unused high nibble is 0", [] {
        auto p = pack({7, 0, -7, 7, 7, 7}, 2, 3, 3);
        std::vector<uint8_t> want = {0x8F, 0x01, 0xFF, 0x0F};
        check(p.bytes == want, "A2 odd width: row stride ceil(cols/2), unused high nibble is 0",
              "expected 8F 01 FF 0F, got " + std::to_string(p.bytes.size()) + " bytes");
    });
    guard("A2 every code = oracle rounding under group scales (36 shapes)", [] {
        int bad = 0;
        double worst = 0;
        for (int m : {1, 3, 7})
            for (int n : {1, 3, 17, 65})
                for (int g : {3, 8, 128}) {
                    auto w = oracle::uniform(m * n, 5 + m + n + g);
                    auto p = pack(w, m, n, g);
                    worst = std::max(worst, max_rel(p.scales, oracle::scales(w, m, n, g)));
                    bad += code_mismatches(w, p);
                    if (n % 2 && (p.bytes.back() >> 4)) ++bad;
                }
        check(bad == 0 && worst < 1e-6,
              "A2 every code = oracle rounding under group scales (36 shapes)",
              std::to_string(bad) + " wrong codes, scale rel err " + num(worst));
    });

    // ---------------- A3 unpack ----------------
    guard("A3 decodes a hand-built artifact: (u - 8) * s, low nibble first", [] {
        auto w = unpack(hand_artifact());
        check(max_abs(w, kHandW) == 0,
              "A3 decodes a hand-built artifact: (u - 8) * s, low nibble first",
              vec(kHandW) + " vs " + vec(w));
    });
    guard("A3 rejects wrong byte/scale length and scales that are 0, negative or NaN", [] {
        auto a = hand_artifact(), b = a, c = a, d = a, e = a;
        a.bytes.pop_back();
        b.scales.pop_back();
        c.scales[1] = 0;
        d.scales[2] = -1;
        e.scales[3] = NAN;
        bool ok = chk::rejects([&] { unpack(a); }) && chk::rejects([&] { unpack(b); }) &&
                  chk::rejects([&] { unpack(c); }) && chk::rejects([&] { unpack(d); }) &&
                  chk::rejects([&] { unpack(e); });
        check(ok, "A3 rejects wrong byte/scale length and scales that are 0, negative or NaN",
              "an invalid artifact was accepted");
    });
    guard("A3 round trip of oracle artifacts = oracle decoder, within s/2 of W", [] {
        double worst = 0, bound = 0;
        for (int m : {1, 3, 7})
            for (int n : {1, 17, 65})
                for (int g : {3, 128}) {
                    auto w = oracle::uniform(m * n, 31 + m * n + g);
                    auto p = oracle::pack(w, m, n, g);
                    auto got = unpack(p);
                    worst = std::max(worst, max_abs(got, oracle::decode_all(p)));
                    int ng = (n + g - 1) / g;
                    for (int r = 0; r < m; ++r)
                        for (int c = 0; c < n; ++c)
                            bound =
                                std::max(bound, double(std::abs(got[r * n + c] - w[r * n + c])) /
                                                    (p.scales[r * ng + c / g] / 2));
                }
        check(worst < 1e-6 && bound <= 1.00002,
              "A3 round trip of oracle artifacts = oracle decoder, within s/2 of W",
              "max abs " + num(worst) + ", half-step ratio " + num(bound));
    });

    // ---------------- A4 packed_linear ----------------
    guard("A4 hand artifact: Y = X W_hat^T for a 2-token batch", [] {
        // W_hat = [[3.5,0.5,-12],[6,-1.5,1.5]], X = [[1,2,3],[-1,0,1]]
        auto y = packed_linear(hand_artifact(), {1, 2, 3, -1, 0, 1}, 2);
        std::vector<double> want = {-31.5, 7.5, -15.5, -4.5};
        check(max_abs(y, want) < 1e-12, "A4 hand artifact: Y = X W_hat^T for a 2-token batch",
              "[-31.5,7.5,-15.5,-4.5] vs " +
                  (y.size() == 4 ? num(y[0]) + "," + num(y[1]) + "," + num(y[2]) + "," + num(y[3])
                                 : "size " + std::to_string(y.size())));
    });
    guard("A4 = FP64 oracle decoder on 27 shapes x batch 1/3/7", [] {
        double worst = 0;
        for (int m : {1, 3, 7})
            for (int n : {1, 17, 65})
                for (int g : {3, 8, 128})
                    for (int b : {1, 3, 7}) {
                        auto p = oracle::pack(oracle::uniform(m * n, 41 + m + n), m, n, g);
                        auto x = oracle::uniform(b * n, 43 + b);
                        worst = std::max(worst,
                                         max_abs(packed_linear(p, x, b), oracle::linear(p, x, b)));
                    }
        check(worst < 1e-9, "A4 = FP64 oracle decoder on 27 shapes x batch 1/3/7",
              "max abs err " + num(worst));
    });
    guard("A4 rejects an X of the wrong size", [] {
        check(chk::rejects([] { packed_linear(hand_artifact(), {1, 2}, 1); }),
              "A4 rejects an X of the wrong size", "accepted");
    });

    // ---------------- A5 certificate ----------------
    guard("A5 ratio, MSE and bytes = in-test recomputation; unclipped ratio <= 1", [] {
        bool ok = true;
        std::string why;
        for (int m : {1, 3, 7})
            for (int n : {1, 17, 65})
                for (int g : {3, 8, 128}) {
                    auto w = oracle::uniform(m * n, 61 + m * n + g);
                    auto p = oracle::pack(w, m, n, g);
                    auto cert = certificate(w, p);
                    auto hat = oracle::decode_all(p);
                    int ng = (n + g - 1) / g;
                    double ratio = 0, se = 0;
                    for (int r = 0; r < m; ++r)
                        for (int c = 0; c < n; ++c) {
                            double d = std::abs(w[r * n + c] - hat[r * n + c]);
                            ratio = std::max(ratio, d / (p.scales[r * ng + c / g] / 2.0));
                            se += d * d;
                        }
                    size_t bytes = size_t(m) * ((n + 1) / 2) + 4 * size_t(m) * ng;
                    if (std::abs(cert.max_half_step_ratio - ratio) > 1e-5 + 1e-4 * ratio ||
                        std::abs(cert.mse - se / (m * n)) > 1e-6 * se / (m * n) + 1e-12 ||
                        cert.bytes != bytes || cert.max_half_step_ratio > 1.00002) {
                        ok = false;
                        why = "shape " + std::to_string(m) + "x" + std::to_string(n) + " g" +
                              std::to_string(g) + ": ratio " + num(cert.max_half_step_ratio) +
                              " vs " + num(ratio) + ", bytes " + std::to_string(cert.bytes) +
                              " vs " + std::to_string(bytes);
                    }
                }
        check(ok, "A5 ratio, MSE and bytes = in-test recomputation; unclipped ratio <= 1", why);
    });

    // ---------------- A6 clip_pack ----------------
    guard("A6 per-group ratio = exhaustive weight-SSE oracle; codes re-quantized", [] {
        // 3 rows x 70 cols, group 32 (tail group of 6). Uniform groups prefer a mild
        // clip (0.9 or 0.8); row 0 / group 0 holds an outlier and must keep ratio 1.
        const int m = 3, n = 70, g = 32, ng = 3;
        auto w = oracle::uniform(m * n, 117);
        w[0] = 12;
        // Designed groups: 31 values of +-0.5 and one absmax value. At ratio 1 every
        // 0.5 sits on a rounding tie (3.5 steps) and rounds to 4; ratio 0.9 moves them
        // to 3.89 steps for a far smaller total SSE (0.016 vs 0.158 for absmax 1).
        for (int k = 0; k < 32; ++k)
            w[1 * n + k] = k % 2 ? -.5f : .5f, w[2 * n + 32 + k] = k % 2 ? 1.5f : -1.5f;
        w[1 * n + 5] = 1, w[2 * n + 40] = -3;
        std::vector<float> ratios = {1, .9f, .8f, .5f};
        auto cp = clip_pack(w, m, n, g, ratios);
        auto base = oracle::scales(w, m, n, g);
        double worst = 0;
        int clipped = 0;
        for (int r = 0; r < m; ++r)
            for (int j = 0; j < ng; ++j) {
                float b = float(base[r * ng + j]), pick = b;
                double best = 1e300;
                for (float ratio : ratios) {
                    float s = b * ratio;
                    double err = 0;
                    for (int k = j * g; k < std::min(n, (j + 1) * g); ++k) {
                        float v = w[r * n + k];
                        err += std::pow(v - oracle::code_of(v, s) * s, 2);
                    }
                    if (err < best) best = err, pick = s;
                }
                clipped += pick != b;
                worst = std::max(
                    worst,
                    chk::rel(cp.scales.size() == size_t(m * ng) ? cp.scales[r * ng + j] : 0, pick));
            }
        int bad = cp.scales.size() == size_t(m * ng) ? code_mismatches(w, cp) : 99;
        check(worst < 1e-6 && bad == 0 && clipped >= 2,
              "A6 per-group ratio = exhaustive weight-SSE oracle; codes re-quantized",
              "scale rel err " + num(worst) + ", " + std::to_string(bad) + " stale codes, " +
                  std::to_string(clipped) + " clipped groups in the oracle");
    });
    guard("A6 rejects an empty list and ratios outside (0, 1]", [] {
        auto w = oracle::uniform(8, 3);
        bool ok = chk::rejects([&] { clip_pack(w, 1, 8, 8, {}); }) &&
                  chk::rejects([&] { clip_pack(w, 1, 8, 8, {1, 0}); }) &&
                  chk::rejects([&] { clip_pack(w, 1, 8, 8, {1.5f}); });
        check(ok, "A6 rejects an empty list and ratios outside (0, 1]", "accepted");
    });
}

// Outlier-channel fixture for C3: two input channels carry 30x larger activations.
struct Fixture {
    int m, n, b, group;
    std::vector<float> w, x;
};
Fixture outlier_fixture() {
    Fixture f{
        8, 32, 16, 32, oracle::uniform(8 * 32, 701, -1, 1), oracle::uniform(16 * 32, 703, -1, 1)};
    for (int t = 0; t < f.b; ++t) f.x[t * f.n + 3] *= 30, f.x[t * f.n + 17] *= 30;
    return f;
}
const std::vector<float> kAlphas = {0, .25f, .5f, .75f, 1};

void test_c(bool bench) {
    // ---------------- C1 channel_scales ----------------
    guard("C1 hand 2x3 X: mean|x| = [2,2,4] -> alpha 0.5 gives 2^[-1/4,-1/4,1/4]", [] {
        std::vector<float> x = {1, -2, 4, 3, 2, -4};
        auto s0 = channel_scales(x, 2, 3, 0), s5 = channel_scales(x, 2, 3, .5f),
             s1 = channel_scales(x, 2, 3, 1);
        double q = std::pow(2.0, 0.25);
        std::vector<double> w5 = {1 / q, 1 / q, q},
                            w1 = {1 / std::sqrt(2.0), 1 / std::sqrt(2.0), std::sqrt(2.0)};
        check(max_rel(s0, std::vector<double>{1, 1, 1}) < 1e-6 && max_rel(s5, w5) < 1e-5 &&
                  max_rel(s1, w1) < 1e-5,
              "C1 hand 2x3 X: mean|x| = [2,2,4] -> alpha 0.5 gives 2^[-1/4,-1/4,1/4]",
              "alpha .5: " + vec(s5) + ", alpha 1: " + vec(s1));
    });
    guard("C1 clamp to 1e-4 BEFORE the power: dead channel at alpha 0.5 -> [0.1, 10]", [] {
        auto s = channel_scales({0, 1, 0, -1}, 2, 2, .5f);
        check(max_rel(s, std::vector<double>{0.1, 10}) < 1e-4,
              "C1 clamp to 1e-4 BEFORE the power: dead channel at alpha 0.5 -> [0.1, 10]", vec(s));
    });
    guard("C1 = FP64 oracle (mean |x|, power, geometric normalization) on random X", [] {
        double worst = 0;
        for (int n : {1, 7, 65})
            for (float a : kAlphas) {
                auto x = oracle::uniform(9 * n, 67 + n);
                for (int t = 0; t < 9; ++t) x[t * n] *= 12;
                worst = std::max(
                    worst, max_rel(channel_scales(x, 9, n, a), oracle::channel_scales(x, 9, n, a)));
            }
        check(worst < 1e-5,
              "C1 = FP64 oracle (mean |x|, power, geometric normalization) on random X",
              "max rel err " + num(worst));
    });
    guard("C1 rejects alpha outside [0,1] and non-finite activations", [] {
        bool ok = chk::rejects([] { channel_scales({1, 2}, 1, 2, -.1f); }) &&
                  chk::rejects([] { channel_scales({1, 2}, 1, 2, 1.5f); }) &&
                  chk::rejects([] { channel_scales({1, NAN}, 1, 2, .5f); });
        check(ok, "C1 rejects alpha outside [0,1] and non-finite activations", "accepted");
    });

    // ---------------- C2 transform ----------------
    guard("C2 hand case: W=[2,3], x=[4,5], s=[2,.5] -> W'=[4,1.5], x'=[2,10]", [] {
        auto t = transform({2, 3}, {4, 5}, 1, 2, 1, {2, .5f});
        check(t.weight == std::vector<float>{4, 1.5f} && t.input == std::vector<float>{2, 10},
              "C2 hand case: W=[2,3], x=[4,5], s=[2,.5] -> W'=[4,1.5], x'=[2,10]",
              "W' " + vec(t.weight) + ", x' " + vec(t.input));
    });
    guard("C2 X' W'^T = X W^T on random data; inputs not mutated; bad scale rejected", [] {
        auto w = oracle::uniform(5 * 31, 3), x = oracle::uniform(9 * 31, 4),
             s = oracle::uniform(31, 5, .2f, 3);
        auto w0 = w, x0 = x;
        auto t = transform(w, x, 5, 31, 9, s);
        double e =
            max_rel(oracle::dense(t.weight, t.input, 5, 31, 9), oracle::dense(w, x, 5, 31, 9));
        auto bad = s;
        bad[4] = 0;
        bool rej = chk::rejects([&] { transform(w, x, 5, 31, 9, bad); });
        check(e < 1e-4 && w == w0 && x == x0 && rej,
              "C2 X' W'^T = X W^T on random data; inputs not mutated; bad scale rejected",
              "rel err " + num(e) + (rej ? "" : ", zero scale accepted"));
    });

    // ---------------- C3 search ----------------
    guard("C3 alpha, MSE and channel scales = in-test exhaustive search (3 shapes)", [] {
        std::string why;
        for (int n : {7, 31, 65}) {
            int m = 5, b = 9;
            auto w = oracle::uniform(m * n, 51 + n), x = oracle::uniform(b * n, 67 + n);
            for (int t = 0; t < b; ++t) x[t * n] *= 12;
            auto got = search(w, x, m, n, b, 8, kAlphas);
            auto want = oracle::search(w, x, m, n, b, 8, kAlphas);
            double es = max_rel(got.channels, oracle::channel_scales(x, b, n, want.alpha));
            if (got.alpha != float(want.alpha) || chk::rel(got.calibration_mse, want.mse) > 1e-4 ||
                es > 1e-5)
                why += " n=" + std::to_string(n) + ": alpha " + num(got.alpha) + " vs " +
                       num(want.alpha) + ", mse " + num(got.calibration_mse) + " vs " +
                       num(want.mse) + ";";
        }
        check(why.empty(),
              "C3 alpha, MSE and channel scales = in-test exhaustive search (3 shapes)", why);
    });
    guard("C3 outlier channels: the search must pick alpha > 0", [] {
        auto f = outlier_fixture();
        auto want = oracle::search(f.w, f.x, f.m, f.n, f.b, f.group, kAlphas);
        bool fixture_ok = want.alpha > 0 && want.mse < 0.8 * want.mse_per_alpha[0];
        auto got = search(f.w, f.x, f.m, f.n, f.b, f.group, kAlphas);
        check(fixture_ok && got.alpha == float(want.alpha) &&
                  chk::rel(got.calibration_mse, want.mse) < 1e-4,
              "C3 outlier channels: the search must pick alpha > 0",
              "oracle alpha " + num(want.alpha) + " (mse " + num(want.mse) + " vs alpha0 " +
                  num(want.mse_per_alpha[0]) + "), got alpha " + num(got.alpha) + " mse " +
                  num(got.calibration_mse));
    });
    guard("C3 artifact codes = oracle quantization of W diag(s)", [] {
        auto f = outlier_fixture();
        auto got = search(f.w, f.x, f.m, f.n, f.b, f.group, kAlphas);
        std::vector<float> ws(f.w);
        for (int r = 0; r < f.m; ++r)
            for (int k = 0; k < f.n; ++k) ws[r * f.n + k] *= got.channels[k];
        int bad = code_mismatches(ws, got.weight);
        double es = max_rel(got.weight.scales, oracle::scales(ws, f.m, f.n, f.group));
        check(bad == 0 && es < 1e-5, "C3 artifact codes = oracle quantization of W diag(s)",
              std::to_string(bad) + " codes, scale err " + num(es));
    });
    guard("C3 rejects an alpha list without the alpha = 0 baseline", [] {
        auto f = outlier_fixture();
        check(chk::rejects([&] { search(f.w, f.x, f.m, f.n, f.b, f.group, {.5f, 1}); }),
              "C3 rejects an alpha list without the alpha = 0 baseline", "accepted");
    });

    // ---------------- C4 execute ----------------
    guard("C4 hand artifact: X / channel scales exactly once, then packed linear", [] {
        Calibrated model{hand_artifact(), {2, .5f, 4}, .5f, 0};
        auto y = execute(model, {2, 1, 8, -2, 0, 4}, 2); // X / s = [[1,2,2],[-1,0,1]]
        // W_hat = [[3.5,0.5,-12],[6,-1.5,1.5]]: [[3.5+1-24, 6-3+3], [-3.5-12, -6+1.5]]
        std::vector<double> want = {-19.5, 6, -15.5, -4.5};
        check(max_abs(y, want) < 1e-9,
              "C4 hand artifact: X / channel scales exactly once, then packed linear",
              y.size() == 4 ? num(y[0]) + "," + num(y[1]) + "," + num(y[2]) + "," + num(y[3])
                            : "wrong size");
    });
    guard("C4 held-out run of a frozen search artifact = oracle decode of the same artifact", [] {
        auto f = outlier_fixture();
        auto model = search(f.w, f.x, f.m, f.n, f.b, f.group, kAlphas);
        auto held = oracle::uniform(3 * f.n, 101);
        auto scaled = held;
        for (int t = 0; t < 3; ++t)
            for (int k = 0; k < f.n; ++k) scaled[t * f.n + k] /= model.channels[k];
        double e = max_abs(execute(model, held, 3), oracle::linear(model.weight, scaled, 3));
        check(e < 1e-9,
              "C4 held-out run of a frozen search artifact = oracle decode of the same artifact",
              "max abs err " + num(e));
    });

    // ---------------- C5 search_budget ----------------
    auto artifact_bytes = [](int m, int n, int g) {
        return size_t(m) * ((n + 1) / 2) + 4 * size_t(m) * ((n + g - 1) / g) + 4 * size_t(n);
    };
    guard("C5 = in-test budget search: feasible groups only, first min-MSE", [&] {
        std::string why;
        for (int n : {7, 31, 65}) {
            int m = 5, b = 9;
            auto w = oracle::uniform(m * n, 51 + n), x = oracle::uniform(b * n, 67 + n);
            for (int t = 0; t < b; ++t) x[t * n] *= 12;
            size_t budget = artifact_bytes(m, n, 8);
            int best_g = -1;
            double best = 1e300;
            for (int g : {2, 8, 32})
                if (artifact_bytes(m, n, g) <= budget) {
                    double e = oracle::search(w, x, m, n, b, g, kAlphas).mse;
                    if (e < best) best = e, best_g = g;
                }
            auto got = search_budget(w, x, m, n, b, {2, 8, 32}, budget, kAlphas);
            if (got.weight.group != best_g || chk::rel(got.calibration_mse, best) > 1e-4)
                why += " n=" + std::to_string(n) + ": group " + std::to_string(got.weight.group) +
                       " vs " + std::to_string(best_g) + ";";
        }
        check(why.empty(), "C5 = in-test budget search: feasible groups only, first min-MSE", why);
    });
    guard("C5 channel-scale bytes count: budget 1 B short of g=8 forces g=32", [&] {
        int n = 31;
        auto w = oracle::uniform(5 * n, 82), x = oracle::uniform(9 * n, 83);
        for (int t = 0; t < 9; ++t) x[t * n] *= 12;
        auto got = search_budget(w, x, 5, n, 9, {8, 32}, artifact_bytes(5, n, 8) - 1, kAlphas);
        check(got.weight.group == 32,
              "C5 channel-scale bytes count: budget 1 B short of g=8 forces g=32",
              "picked group " + std::to_string(got.weight.group));
    });
    guard("C5 infeasible budget throws std::runtime_error", [] {
        auto w = oracle::uniform(5 * 7, 1), x = oracle::uniform(9 * 7, 2);
        bool caught = false;
        try {
            search_budget(w, x, 5, 7, 9, {8}, 1, kAlphas);
        } catch (const std::runtime_error &) {
            caught = true;
        } catch (...) {
        }
        check(caught, "C5 infeasible budget throws std::runtime_error", "no runtime_error");
    });

    if (!bench) return;
    // Calibration table, one row per width; the same W, X and held-out sets feed every
    // method. Calibration X: channel 0 is 12x larger. heldout: new samples from the SAME
    // distribution; shifted: the large channel moved to channel 1 (distribution shift).
    std::cout
        << "cols,selected_alpha,calib_mse,heldout_mse,shifted_mse,rtn_calib_mse,rtn_heldout_mse,"
           "rtn_shifted_mse,clip_calib_mse,clip_heldout_mse,budget_group,artifact_bytes,budget_"
           "calib_mse\n";
    for (int n : {7, 31, 65, 257}) {
        int m = 5, b = 9, h = 16;
        auto w = oracle::uniform(m * n, 51 + n), x = oracle::uniform(b * n, 67 + n);
        auto held = oracle::uniform(h * n, 101 + n), shifted = oracle::uniform(h * n, 103 + n);
        for (int t = 0; t < b; ++t) x[t * n] *= 12;
        for (int t = 0; t < h; ++t) held[t * n] *= 12, shifted[t * n + std::min(1, n - 1)] *= 12;
        auto ref_c = oracle::dense(w, x, m, n, b), ref_h = oracle::dense(w, held, m, n, h);
        auto ref_s = oracle::dense(w, shifted, m, n, h);
        auto chosen = search(w, x, m, n, b, 8, kAlphas);
        auto rtn = search(w, x, m, n, b, 8, {0});
        auto clipped = clip_pack(w, m, n, 8, {1, .95f, .9f, .85f, .8f});
        auto budgeted = search_budget(w, x, m, n, b, {2, 8, 32}, artifact_bytes(m, n, 8), kAlphas);
        std::cout << n << ',' << chosen.alpha << ',' << chosen.calibration_mse << ','
                  << oracle::mse(execute(chosen, held, h), ref_h) << ','
                  << oracle::mse(execute(chosen, shifted, h), ref_s) << ',' << rtn.calibration_mse
                  << ',' << oracle::mse(execute(rtn, held, h), ref_h) << ','
                  << oracle::mse(execute(rtn, shifted, h), ref_s) << ','
                  << oracle::mse(packed_linear(clipped, x, b), ref_c) << ','
                  << oracle::mse(packed_linear(clipped, held, h), ref_h) << ','
                  << budgeted.weight.group << ',' << artifact_bytes(m, n, budgeted.weight.group)
                  << ',' << budgeted.calibration_mse << '\n';
    }
}
} // namespace

int main(int argc, char **argv) {
    std::string part = argc > 1 ? argv[1] : "all";
    bool bench = argc > 2 && std::string(argv[2]) == "bench";
    if (part == "A" || part == "all") test_a();
    if (part == "C" || part == "all") guard("C bench table", [&] { test_c(bench); });
    return chk::summary(part == "all" ? "A+C (host)" : part + " (host)");
}
