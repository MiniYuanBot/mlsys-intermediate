// Lab 1 CPU tests. Every check prints one PASS/FAIL line and the run continues past
// failures; a section that throws (for example "Complete A1") is reported as FAIL and
// the next section still runs. Oracles never call the function under test: quantization
// metadata, codes, Lloyd, nearest-center encoding, ADC scores, top-k selection and
// attention are recomputed here in FP64 from the raw inputs.
#include "pq_student.hpp"
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>

namespace {
int g_pass = 0, g_fail = 0;
void check(bool ok, const std::string &name, const std::string &detail = "") {
    if (ok) {
        ++g_pass;
        std::cout << "PASS " << name << '\n';
    } else {
        ++g_fail;
        std::cout << "FAIL " << name << ": " << detail << '\n';
    }
}
std::string fmt(double expected, double actual) {
    std::ostringstream s;
    s.precision(9);
    s << "expected " << expected << " vs actual " << actual;
    return s.str();
}
// Run one group of checks; an exception becomes a FAIL line instead of aborting.
void section(const std::string &name, const std::function<void()> &body) {
    try {
        body();
    } catch (const std::exception &e) {
        check(false, name, std::string("threw: ") + e.what());
    }
}
template <class F> bool throws_invalid(F f) {
    try {
        f();
    } catch (const std::invalid_argument &) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// ---------- independent oracles ----------
// Group membership from the axis definition (not kv::group_index):
//   per-token  (Axis::Token):   group = (row r, channel block c/g), ordered r-major
//   per-channel (Axis::Channel): group = (channel c, token block r/g), ordered c-major
size_t oracle_group(size_t r, size_t c, size_t rows, size_t cols, size_t g, kv::Axis axis) {
    if (axis == kv::Axis::Token) return r * ((cols + g - 1) / g) + c / g;
    return c * ((rows + g - 1) / g) + r / g;
}
struct OracleMeta {
    std::vector<double> lo, hi, scale;
};
OracleMeta oracle_meta(const std::vector<float> &x, size_t rows, size_t cols, size_t g,
                       kv::Axis axis) {
    size_t ng = axis == kv::Axis::Token ? rows * ((cols + g - 1) / g) : cols * ((rows + g - 1) / g);
    OracleMeta m{std::vector<double>(ng, INFINITY), std::vector<double>(ng, -INFINITY),
                 std::vector<double>(ng)};
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            size_t j = oracle_group(r, c, rows, cols, g, axis);
            m.lo[j] = std::min(m.lo[j], double(x[r * cols + c]));
            m.hi[j] = std::max(m.hi[j], double(x[r * cols + c]));
        }
    for (size_t j = 0; j < ng; ++j) m.scale[j] = (m.hi[j] - m.lo[j]) / 15.0;
    return m;
}
// Offset must equal the group min; scale must equal (max-min)/15, or be any finite positive
// value for a constant group (its codes are then all 0).
bool meta_ok(const kv::Packed &p, const OracleMeta &m, size_t j) {
    if (double(p.offsets[j]) != double(float(m.lo[j]))) return false;
    if (m.hi[j] == m.lo[j]) return std::isfinite(p.scales[j]) && p.scales[j] > 0;
    return std::abs(double(p.scales[j]) - m.scale[j]) <= 1e-6 * m.scale[j] + 1e-38;
}
int nibble(const std::vector<uint8_t> &codes, size_t i) {
    return (codes[i / 2] >> (4 * (i % 2))) & 15;
}
// Decode a Packed from its bytes with this file's own formulas (no kv::dequantize).
std::vector<double> oracle_decode(const kv::Packed &p) {
    std::vector<double> y(p.rows * p.cols);
    for (size_t i = 0; i < y.size(); ++i) {
        size_t j = oracle_group(i / p.cols, i % p.cols, p.rows, p.cols, p.group, p.axis);
        y[i] = double(p.offsets[j]) + double(p.scales[j]) * nibble(p.codes, i);
    }
    return y;
}
// Compare student metadata with the FP64 per-group oracle and check every code against
// the ORACLE scale (not the student's own), so a loose or shared scale cannot pass.
void check_packed(const std::string &name, const kv::Packed &p, const std::vector<float> &x,
                  size_t rows, size_t cols, size_t g, kv::Axis axis) {
    auto m = oracle_meta(x, rows, cols, g, axis);
    check(p.rows == rows && p.cols == cols && p.group == g && p.axis == axis, name + " header",
          "rows/cols/group/axis must describe the input");
    check(p.codes.size() == (rows * cols + 1) / 2, name + " packed byte count",
          fmt(double((rows * cols + 1) / 2), double(p.codes.size())));
    check(p.scales.size() == m.lo.size() && p.offsets.size() == m.lo.size(), name + " group count",
          fmt(double(m.lo.size()), double(p.scales.size())));
    if (p.scales.size() != m.lo.size() || p.offsets.size() != m.lo.size() ||
        p.codes.size() != (rows * cols + 1) / 2)
        return;
    size_t bad_meta = 0, bad_code = 0, first = 0;
    double worst = 0;
    for (size_t j = 0; j < m.lo.size(); ++j) {
        if (!meta_ok(p, m, j) && !bad_meta++) first = j;
    }
    check(!bad_meta, name + " per-group offset=min, scale=(max-min)/15",
          bad_meta ? "group " + std::to_string(first) + " " + fmt(m.lo[first], p.offsets[first]) +
                         " offset; " + fmt(m.scale[first], p.scales[first]) + " scale"
                   : "");
    for (size_t i = 0; i < x.size(); ++i) {
        size_t j = oracle_group(i / cols, i % cols, rows, cols, g, axis);
        double s = m.hi[j] == m.lo[j] ? 0 : m.scale[j];
        double y = m.lo[j] + s * nibble(p.codes, i);
        double err = std::abs(y - double(x[i]));
        double bound = 0.5 * s * (1 + 1e-5) + 1e-7 * std::abs(double(x[i]));
        if (err > bound && !bad_code++) first = i;
        worst = std::max(worst, err - bound);
    }
    check(!bad_code, name + " codes within half an oracle step (low nibble = even index)",
          bad_code ? std::to_string(bad_code) + " codes off, first linear index " +
                         std::to_string(first) + ", worst excess " + std::to_string(worst)
                   : "");
    if (rows * cols % 2) check((p.codes.back() & 0xf0) == 0, name + " odd tail high nibble is 0");
}

double dot(const std::vector<double> &a, const std::vector<double> &b, size_t ai, size_t bi,
           size_t n) {
    double s = 0;
    for (size_t j = 0; j < n; ++j) s += a[ai + j] * b[bi + j];
    return s;
}
// FP64 dense softmax attention over rows of k/v (no reuse of kv::attention).
std::vector<double> oracle_attention(const std::vector<double> &q, const std::vector<double> &k,
                                     const std::vector<double> &v) {
    size_t d = q.size(), n = k.size() / d;
    std::vector<double> s(n), out(d, 0);
    double mx = -INFINITY, z = 0;
    for (size_t t = 0; t < n; ++t)
        mx = std::max(mx, s[t] = dot(q, k, 0, t * d, d) / std::sqrt(double(d)));
    for (size_t t = 0; t < n; ++t) z += s[t] = std::exp(s[t] - mx);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j) out[j] += s[t] / z * v[t * d + j];
    return out;
}
std::vector<double> to_double(const std::vector<float> &x) {
    return {x.begin(), x.end()};
}
double max_abs_diff(const std::vector<double> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return INFINITY;
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - double(b[i])));
    return m;
}
// Mandatory sink/recent union, then highest score, ties to the smaller ID, sorted output.
// Validates `ids` against `scores` with tolerance `tol`, so float/double accumulation order
// cannot cause a false failure while a reversed ranking or tie-break still does.
std::string topk_violation(const std::vector<size_t> &ids, const std::vector<double> &scores,
                           size_t budget, size_t sink, size_t recent, double tol) {
    size_t n = scores.size();
    if (ids.size() != budget)
        return "size " + std::to_string(ids.size()) + " != budget " + std::to_string(budget);
    for (size_t i = 1; i < ids.size(); ++i)
        if (ids[i] <= ids[i - 1]) return "ids not strictly increasing";
    std::vector<bool> chosen(n, false);
    for (size_t id : ids) {
        if (id >= n) return "id out of range";
        chosen[id] = true;
    }
    for (size_t i = 0; i < n; ++i)
        if ((i < sink || i + recent >= n) && !chosen[i])
            return "mandatory id " + std::to_string(i) + " missing";
    for (size_t a = sink; a + recent < n; ++a) {
        if (!chosen[a]) continue;
        for (size_t b = sink; b + recent < n; ++b) {
            if (chosen[b]) continue;
            if (scores[b] > scores[a] + tol)
                return "dropped " + std::to_string(b) + " (score " + std::to_string(scores[b]) +
                       ") but kept " + std::to_string(a) + " (score " + std::to_string(scores[a]) +
                       ")";
            if (std::abs(scores[b] - scores[a]) <= tol * 1e-3 && b < a)
                return "tie between " + std::to_string(b) + " and " + std::to_string(a) +
                       " must keep the smaller ID";
        }
    }
    return "";
}

// ---------- A ----------
void test_a() {
    // Asymmetric data: rows != cols, every channel has its own positive offset (so a
    // per-token/per-channel swap, a global min/max, or zero-padding a tail group all
    // change the metadata), and values vary inside each row and column.
    for (auto axis : {kv::Axis::Token, kv::Axis::Channel})
        for (size_t g : {1, 2, 3, 4, 8}) {
            std::string name = std::string("A1 ") +
                               (axis == kv::Axis::Token ? "per-token" : "per-channel") +
                               " g=" + std::to_string(g);
            section(name, [&] {
                std::vector<float> x(5 * 7);
                for (size_t r = 0; r < 5; ++r)
                    for (size_t c = 0; c < 7; ++c)
                        x[r * 7 + c] =
                            3.f + 10.f * c + 2.5f * std::sin(float(r * 7 + c)) + 0.7f * r;
                auto p = kv::quantize(x, 5, 7, g, axis);
                check_packed(name, p, x, 5, 7, g, axis);
            });
        }
    section("A1 tail group of equal positive values", [] {
        // Channel axis, T=5, g=4: channel 0 has groups {rows 0..3} and the tail {row 4}.
        // Token axis, D=7, g=4: each row's tail group is channels 4..6 = [4,4,4].
        std::vector<float> x(35);
        for (size_t r = 0; r < 5; ++r)
            for (size_t c = 0; c < 7; ++c) x[r * 7 + c] = c >= 4 ? 4.f : 1.f + r + c;
        auto p = kv::quantize(x, 5, 7, 4, kv::Axis::Token);
        check_packed("A1 tail [4,4,4] per-token", p, x, 5, 7, 4, kv::Axis::Token);
        auto y = oracle_decode(p);
        bool exact = true;
        for (size_t r = 0; r < 5; ++r)
            for (size_t c = 4; c < 7; ++c) exact &= y[r * 7 + c] == 4.0;
        check(exact, "A1 constant tail group decodes exactly",
              "tail [4,4,4] must restore 4 (no zero padding)");
        auto pc = kv::quantize(x, 5, 7, 4, kv::Axis::Channel);
        check_packed("A1 tail rows per-channel", pc, x, 5, 7, 4, kv::Axis::Channel);
    });
    section("A1 constant tensor", [] {
        auto p = kv::quantize(std::vector<float>(35, -2.5f), 5, 7, 4, kv::Axis::Channel);
        auto y = oracle_decode(p);
        bool exact = true;
        for (double v : y) exact &= v == -2.5;
        check(exact, "A1 constant groups restore exactly from codes+metadata");
    });
    section("A1 FP32 extrema and subnormals", [] {
        float hi = std::numeric_limits<float>::max();
        auto p = kv::quantize({-hi, hi}, 1, 2, 2, kv::Axis::Token);
        check(std::isfinite(p.scales[0]) && p.offsets[0] == -hi,
              "A1 opposite extrema keep a finite scale");
        float tiny = std::numeric_limits<float>::denorm_min();
        auto s = kv::quantize({0.f, tiny}, 1, 2, 2, kv::Axis::Token);
        auto y = oracle_decode(s);
        check(s.scales[0] > 0 && y[0] == 0.0 && y[1] == double(tiny),
              "A1 subnormal range keeps a nonzero scale and restores [0, denorm_min]");
    });
    section("A1 rejects bad input", [] {
        check(throws_invalid([] { kv::quantize({NAN}, 1, 1, 1, kv::Axis::Token); }),
              "A1 rejects NaN");
        check(throws_invalid([] { kv::quantize({1}, 1, 1, 0, kv::Axis::Token); }),
              "A1 rejects g=0");
        check(throws_invalid([] { kv::quantize({1, 2}, 1, 1, 1, kv::Axis::Token); }),
              "A1 rejects shape mismatch");
    });
    section("A2 dequantize hand-built Packed", [] {
        // Hand-written codes include 8..15 (unsigned, never negative) and an odd tail.
        for (auto axis : {kv::Axis::Token, kv::Axis::Channel}) {
            kv::Packed p{3, 5, 2, axis, {}, {}, {}};
            size_t ng = axis == kv::Axis::Token ? 3 * 3 : 5 * 2;
            for (size_t i = 0; i < 15; ++i)
                if (i % 2 == 0)
                    p.codes.push_back(uint8_t((i + 7) % 16));
                else
                    p.codes.back() |= uint8_t(((i + 7) % 16) << 4);
            for (size_t j = 0; j < ng; ++j) {
                p.scales.push_back(0.25f + 0.5f * j);
                p.offsets.push_back(-3.f + j);
            }
            auto y = kv::dequantize(p);
            double err = INFINITY;
            if (y.size() == 15) {
                err = 0;
                for (size_t i = 0; i < 15; ++i) {
                    size_t j = oracle_group(i / 5, i % 5, 3, 5, 2, axis);
                    double expect = (-3.0 + j) + (0.25 + 0.5 * j) * double((i + 7) % 16);
                    err = std::max(err, std::abs(expect - y[i]));
                }
            }
            check(err < 1e-5,
                  std::string("A2 decodes from codes+metadata only (") +
                      (axis == kv::Axis::Token ? "per-token" : "per-channel") + ")",
                  "max error " + std::to_string(err));
        }
    });
    section("A2 roundtrip vs raw input", [] {
        std::vector<float> x(9 * 6);
        for (size_t i = 0; i < x.size(); ++i) x[i] = 5 * std::cos(0.37f * i) + 0.1f * i;
        for (auto axis : {kv::Axis::Token, kv::Axis::Channel}) {
            auto p = kv::quantize(x, 9, 6, 4, axis);
            auto m = oracle_meta(x, 9, 6, 4, axis);
            auto y = kv::dequantize(p);
            bool ok = y.size() == x.size();
            for (size_t i = 0; ok && i < x.size(); ++i)
                ok = std::abs(y[i] - x[i]) <=
                     0.501 * m.scale[oracle_group(i / 6, i % 6, 9, 6, 4, axis)] + 1e-6;
            check(ok, "A2 roundtrip within half oracle step", "decoded value outside half step");
        }
    });
    section("A3 gather_rows", [] {
        check(kv::gather_rows({1, 2, 3, 4, 5, 6}, 2, {0, 2}) == std::vector<float>{1, 2, 5, 6},
              "A3 gathers rows 0 and 2 of [1,2;3,4;5,6]");
        check(kv::gather_rows({1, 2, 3, 4, 5, 6}, 3, {1}) == std::vector<float>{4, 5, 6},
              "A3 row width follows cols");
        check(kv::gather_rows({1}, 1, {}).empty(), "A3 empty selection is legal");
        check(throws_invalid([] { kv::gather_rows({1, 2}, 1, {1, 0}); }),
              "A3 rejects unsorted ids");
        check(throws_invalid([] { kv::gather_rows({1, 2}, 1, {0, 0}); }),
              "A3 rejects duplicate ids");
        check(throws_invalid([] { kv::gather_rows({1, 2}, 1, {2}); }),
              "A3 rejects out-of-range id");
        check(throws_invalid([] { kv::gather_rows({1, 2, 3}, 2, {0}); }),
              "A3 rejects ragged input");
    });
    section("A4 errors", [] {
        // Hand computed: a=[1,2,2], b=[1,2,3]: diff=[0,0,-1], |a|=3, rel L2=1/3, max=1,
        // cos=(1+4+6)/(3*sqrt(14)).
        auto e = kv::errors({1, 2, 2}, {1, 2, 3});
        check(std::abs(e.relative_l2 - 1.0 / 3) < 1e-12, "A4 relative L2 uses the reference norm",
              fmt(1.0 / 3, e.relative_l2));
        check(e.max_abs == 1, "A4 max_abs", fmt(1, e.max_abs));
        check(std::abs(e.cosine - 11 / (3 * std::sqrt(14.0))) < 1e-12, "A4 cosine",
              fmt(11 / (3 * std::sqrt(14.0)), e.cosine));
        auto z = kv::errors({3, 4}, {0, 0});
        check(std::abs(z.relative_l2 - 1) < 1e-12 && z.max_abs == 4 && z.cosine == 0,
              "A4 one zero vector: rel=1, cos=0");
        check(kv::errors({0}, {0}).cosine == 1 && kv::errors({0}, {0}).relative_l2 == 0,
              "A4 two zero vectors: cos=1, rel=0");
        auto s = kv::errors({-1, 1}, {1, 1});
        check(s.max_abs == 2, "A4 max_abs is an absolute value", fmt(2, s.max_abs));
        check(throws_invalid([] { kv::errors({INFINITY}, {1}); }), "A4 rejects Inf");
        check(throws_invalid([] { kv::errors({1, 2}, {1}); }), "A4 rejects length mismatch");
    });
}

// ---------- B ----------
void test_b() {
    section("B1 select_tokens", [] {
        check(kv::select_tokens({0, 4, 4, 1, 0}, 4, 1, 1) == std::vector<size_t>{0, 1, 2, 4},
              "B1 mandatory {0,4} plus top scores");
        // Budget boundary tie: after {0,5} and id 2 (score 5), one slot is left for three
        // candidates tied at 3; the rule keeps the smallest, id 1.
        auto tie = kv::select_tokens({0, 3, 5, 3, 3, 0}, 4, 1, 1);
        check(tie == std::vector<size_t>{0, 1, 2, 5}, "B1 boundary tie keeps the smaller id",
              "expected [0,1,2,5]");
        check(kv::select_tokens({1, 2, 3}, 3, 2, 2) == std::vector<size_t>{0, 1, 2},
              "B1 overlapping sink/recent counted once");
        check(kv::select_tokens({9, 1, 1, 1, 1, 1, 1, 7}, 3, 0, 2) == std::vector<size_t>{0, 6, 7},
              "B1 recent=2 covers exactly the last two ids");
        check(throws_invalid([] { kv::select_tokens({1, 2, 3}, 1, 1, 1); }),
              "B1 rejects infeasible budget");
        check(throws_invalid([] { kv::select_tokens({1, NAN, 3}, 3, 0, 0); }),
              "B1 rejects NaN score");
        // Brute force: every subset that contains the mandatory set; best = max score sum,
        // then lexicographically smallest sorted ids (= ties go to smaller ids).
        std::mt19937 rng(5);
        size_t bad = 0;
        for (int trial = 0; trial < 300; ++trial) {
            size_t n = 1 + rng() % 8, sink = rng() % 3, recent = rng() % 3;
            std::vector<float> s(n);
            for (float &x : s) x = float(rng() % 4); // many ties
            std::vector<size_t> best;
            double best_sum = -INFINITY;
            size_t budget = rng() % (n + 1);
            for (unsigned mask = 0; mask < (1u << n); ++mask) {
                size_t bits = 0;
                for (unsigned m = mask; m; m >>= 1) bits += m & 1;
                if (bits != budget) continue;
                bool ok = true;
                for (size_t i = 0; i < n; ++i)
                    if ((i < sink || i + recent >= n) && !(mask >> i & 1)) ok = false;
                if (!ok) continue;
                std::vector<size_t> ids;
                double sum = 0;
                for (size_t i = 0; i < n; ++i)
                    if (mask >> i & 1)
                        ids.push_back(i), sum += (i < sink || i + recent >= n) ? 0 : s[i];
                if (sum > best_sum || (sum == best_sum && ids < best)) best_sum = sum, best = ids;
            }
            bool feasible = best_sum > -INFINITY;
            bool threw = false;
            std::vector<size_t> got;
            try {
                got = kv::select_tokens(s, budget, std::min(sink, n), std::min(recent, n));
            } catch (const std::invalid_argument &) {
                threw = true;
            }
            if (feasible ? (threw || got != best) : !threw) ++bad;
        }
        check(!bad, "B1 matches brute force on 300 random tie-heavy cases",
              std::to_string(bad) + " mismatches");
    });

    for (size_t residual : {0, 3, 8}) {
        std::string tag = " (residual=" + std::to_string(residual) + ")";
        section("B2-B5" + tag, [&] {
            const size_t dim = 6, block = 4, group = 3, steps = 37;
            kv::ResidualCache cache(dim, block, residual, group);
            check(throws_invalid([&] { cache.attend(std::vector<float>(dim)); }),
                  "B5 empty cache rejects query" + tag);
            std::vector<float> raw_k, raw_v;
            std::vector<double> q(dim);
            for (size_t j = 0; j < dim; ++j) q[j] = 0.15 * (j + 1) - 0.4;
            std::vector<float> qf(q.begin(), q.end());
            size_t bad_count = 0, bad_axis = 0, bad_meta = 0, bad_recent = 0, bad_mat = 0,
                   bad_bytes = 0, bad_attend = 0, bad_bound = 0, bad_order = 0;
            double worst_attend = 0, worst_excess = 0;
            for (size_t t = 0; t < steps; ++t) {
                std::vector<float> k(dim), v(dim);
                // K: channel-dependent offsets (outlier channel 5) so per-channel groups matter;
                // V: token-dependent magnitude so per-token groups matter.
                for (size_t j = 0; j < dim; ++j) {
                    k[j] = std::sin(float(t * dim + j)) + (j == 5 ? 6.f : 0.5f * j);
                    v[j] = (1.f + 0.1f * t) * std::cos(float(t + 2 * j));
                }
                cache.append(k, v);
                raw_k.insert(raw_k.end(), k.begin(), k.end());
                raw_v.insert(raw_v.end(), v.begin(), v.end());
                size_t packed = cache.chunks.size() * block, recent = cache.recent_k.size() / dim;
                bad_count += cache.tokens != t + 1 || packed + recent != t + 1 ||
                             recent >= residual + block ||
                             cache.recent_v.size() != cache.recent_k.size();
                // Expected state from first principles: a block flushes when recent reaches
                // residual+block, so packed blocks = floor((t+1-residual)/block) for t+1>=residual+block.
                size_t n = t + 1;
                size_t expect_chunks = n >= residual + block ? (n - residual) / block : 0;
                bad_count += cache.chunks.size() != expect_chunks;
                // Every chunk: K per-channel, V per-token, metadata equal to an FP64 oracle of
                // the RAW rows of that block (catches zeroed/newest/axis-swapped flushes).
                for (size_t b = 0; b < cache.chunks.size() && b * block + block <= n; ++b) {
                    const auto &c = cache.chunks[b];
                    bad_axis += c.k.axis != kv::Axis::Channel || c.v.axis != kv::Axis::Token ||
                                c.k.rows != block || c.k.cols != dim || c.v.rows != block ||
                                c.k.group != group || c.v.group != group;
                    std::vector<float> bk(raw_k.begin() + b * block * dim,
                                          raw_k.begin() + (b + 1) * block * dim);
                    std::vector<float> bv(raw_v.begin() + b * block * dim,
                                          raw_v.begin() + (b + 1) * block * dim);
                    auto mk = oracle_meta(bk, block, dim, group, kv::Axis::Channel);
                    auto mv = oracle_meta(bv, block, dim, group, kv::Axis::Token);
                    if (c.k.offsets.size() != mk.lo.size() || c.v.offsets.size() != mv.lo.size()) {
                        ++bad_meta;
                        continue;
                    }
                    for (size_t j = 0; j < mk.lo.size(); ++j) bad_meta += !meta_ok(c.k, mk, j);
                    for (size_t j = 0; j < mv.lo.size(); ++j) bad_meta += !meta_ok(c.v, mv, j);
                }
                // Recent rows are the exact raw tail, in order.
                bad_recent += !std::equal(cache.recent_k.begin(), cache.recent_k.end(),
                                          raw_k.begin() + packed * dim) ||
                              !std::equal(cache.recent_v.begin(), cache.recent_v.end(),
                                          raw_v.begin() + packed * dim);
                // B3: materialize == chunks decoded by THIS file's decoder + exact recent rows.
                auto [dk, dv] = cache.materialize();
                std::vector<double> ok_k, ok_v;
                for (const auto &c : cache.chunks) {
                    auto a = oracle_decode(c.k), b2 = oracle_decode(c.v);
                    ok_k.insert(ok_k.end(), a.begin(), a.end());
                    ok_v.insert(ok_v.end(), b2.begin(), b2.end());
                }
                ok_k.insert(ok_k.end(), cache.recent_k.begin(), cache.recent_k.end());
                ok_v.insert(ok_v.end(), cache.recent_v.begin(), cache.recent_v.end());
                bad_mat += max_abs_diff(ok_k, dk) > 1e-5 || max_abs_diff(ok_v, dv) > 1e-5;
                // Time order: decoded rows stay within half a step of the raw rows.
                bad_order += max_abs_diff(to_double(raw_k), dk) > 0.6 * 6.5 / 15 + 1e-5 ||
                             max_abs_diff(to_double(raw_v), dv) > 0.6 * 9.0 / 15 + 1e-5;
                // B4: closed-form logical payload.
                size_t gk = dim * ((block + group - 1) / group),
                       gv = block * ((dim + group - 1) / group);
                size_t expect_bytes =
                    cache.chunks.size() * (2 * ((block * dim + 1) / 2) + 8 * (gk + gv)) +
                    8 * recent * dim;
                bad_bytes += cache.bytes() != expect_bytes;
                // B5 (tight): attention over the cache's own content, decoded independently.
                auto got = cache.attend(qf);
                double e = max_abs_diff(oracle_attention(q, ok_k, ok_v), got);
                worst_attend = std::max(worst_attend, e);
                bad_attend += e > 2e-5;
                // B2+B5 vs the RAW history: |o'-o| <= max|dV| + (e^{2 eps}-1) * range(V)/2,
                // eps = max_t sum_c |q_c| sK_c/2 / sqrt(d) (derivation in the lecture §3.4).
                double eps = 0, dvmax = 0;
                std::vector<double> vmin(dim, INFINITY), vmax(dim, -INFINITY);
                for (size_t b = 0; b * block + block <= packed; ++b) {
                    std::vector<float> bk(raw_k.begin() + b * block * dim,
                                          raw_k.begin() + (b + 1) * block * dim);
                    std::vector<float> bv(raw_v.begin() + b * block * dim,
                                          raw_v.begin() + (b + 1) * block * dim);
                    auto mk = oracle_meta(bk, block, dim, group, kv::Axis::Channel);
                    auto mv = oracle_meta(bv, block, dim, group, kv::Axis::Token);
                    for (size_t r = 0; r < block; ++r) {
                        double et = 0;
                        for (size_t c = 0; c < dim; ++c)
                            et +=
                                std::abs(q[c]) *
                                mk.scale[oracle_group(r, c, block, dim, group, kv::Axis::Channel)] /
                                2;
                        eps = std::max(eps, et / std::sqrt(double(dim)));
                    }
                    for (double s : mv.scale) dvmax = std::max(dvmax, s / 2);
                }
                for (size_t r = 0; r < n; ++r)
                    for (size_t j = 0; j < dim; ++j) {
                        vmin[j] = std::min(vmin[j], double(raw_v[r * dim + j]));
                        vmax[j] = std::max(vmax[j], double(raw_v[r * dim + j]));
                    }
                auto truth = oracle_attention(q, to_double(raw_k), to_double(raw_v));
                for (size_t j = 0; j < dim && got.size() == dim; ++j) {
                    double bound = dvmax + std::expm1(2 * eps) * (vmax[j] - vmin[j]) / 2 + 1e-5;
                    double excess = std::abs(got[j] - truth[j]) - bound;
                    worst_excess = std::max(worst_excess, excess);
                    bad_bound += excess > 0;
                }
                if (got.size() != dim) ++bad_bound;
            }
            check(!bad_count, "B2 token count, flush schedule, recent < residual+block" + tag,
                  std::to_string(bad_count) + " bad states");
            check(!bad_axis, "B2 chunk K per-channel, V per-token, block rows" + tag,
                  std::to_string(bad_axis) + " bad chunks");
            check(!bad_meta, "B2 chunk metadata equals FP64 oracle of the raw oldest block" + tag,
                  std::to_string(bad_meta) + " mismatches");
            check(!bad_recent, "B2 recent rows are the exact raw tail" + tag,
                  std::to_string(bad_recent) + " mismatches");
            check(!bad_mat, "B3 materialize = decoded chunks + recent, in time order" + tag,
                  std::to_string(bad_mat) + " mismatches");
            check(!bad_order, "B3 materialized rows track the raw history" + tag,
                  std::to_string(bad_order) + " steps off");
            check(!bad_bytes, "B4 bytes = codes + 8B metadata per group + FP32 recent" + tag,
                  std::to_string(bad_bytes) + " mismatches");
            check(!bad_attend, "B5 online softmax = FP64 attention on decoded cache (<2e-5)" + tag,
                  "worst " + std::to_string(worst_attend));
            check(!bad_bound,
                  "B5 output vs raw-history FP64 within derived quantization bound" + tag,
                  "worst excess " + std::to_string(worst_excess));
            auto before = cache.materialize();
            check(throws_invalid([&] { cache.append({1}, {2}); }),
                  "B2 rejects wrong row width" + tag);
            check(throws_invalid(
                      [&] { cache.append(std::vector<float>(dim, NAN), std::vector<float>(dim)); }),
                  "B2 rejects NaN" + tag);
            check(before == cache.materialize() && cache.tokens == steps,
                  "B2 rejected append leaves state unchanged" + tag);
        });
    }
    section("B5 large scores", [] {
        kv::ResidualCache cache(2, 2, 0, 2);
        cache.append({100, 0}, {1, 0});
        cache.append({101, 0}, {0, 1});
        cache.append({0, 0}, {5, 5});
        auto out = cache.attend({100, 0});
        double w = 1 / (1 + std::exp(-100 / std::sqrt(2.0)));
        bool finite = out.size() == 2 && std::isfinite(out[0]) && std::isfinite(out[1]);
        check(finite && std::abs(out[1] - w) < 1e-3, "B5 no overflow with logits ~ 7000",
              finite ? fmt(w, out[1]) : "non-finite output");
    });
}

// ---------- C ----------
struct OracleLloyd {
    std::vector<float> book;
    std::vector<double> objective;
    double init_sse;
};
// Independent Lloyd with the stated rules: seeds = row c*N/C, nearest center by FP64
// squared distance with ties to the smaller ID, empty clusters keep their center,
// SSE recorded after each update.
OracleLloyd oracle_lloyd(const std::vector<float> &x, size_t dim, size_t parts, size_t cents,
                         size_t iters) {
    size_t n = x.size() / dim, sub = dim / parts;
    OracleLloyd o{std::vector<float>(parts * cents * sub), {}, 0};
    auto at = [&](size_t p, size_t c, size_t j) -> float & {
        return o.book[(p * cents + c) * sub + j];
    };
    for (size_t p = 0; p < parts; ++p)
        for (size_t c = 0; c < cents; ++c)
            for (size_t j = 0; j < sub; ++j) at(p, c, j) = x[(c * n / cents) * dim + p * sub + j];
    auto nearest = [&](size_t i, size_t p, double &best_d) {
        size_t best = 0;
        best_d = INFINITY;
        for (size_t c = 0; c < cents; ++c) {
            double d = 0;
            for (size_t j = 0; j < sub; ++j) {
                double t = double(x[i * dim + p * sub + j]) - at(p, c, j);
                d += t * t;
            }
            if (d < best_d) best_d = d, best = c;
        }
        return best;
    };
    for (size_t i = 0; i < n; ++i)
        for (size_t p = 0; p < parts; ++p) {
            double d;
            nearest(i, p, d);
            o.init_sse += d;
        }
    for (size_t it = 0; it < iters; ++it) {
        std::vector<size_t> assign(n * parts), count(parts * cents, 0);
        std::vector<double> sum(o.book.size(), 0);
        for (size_t i = 0; i < n; ++i)
            for (size_t p = 0; p < parts; ++p) {
                double d;
                size_t c = assign[i * parts + p] = nearest(i, p, d);
                ++count[p * cents + c];
                for (size_t j = 0; j < sub; ++j)
                    sum[(p * cents + c) * sub + j] += x[i * dim + p * sub + j];
            }
        for (size_t p = 0; p < parts; ++p)
            for (size_t c = 0; c < cents; ++c)
                if (count[p * cents + c])
                    for (size_t j = 0; j < sub; ++j)
                        at(p, c, j) = float(sum[(p * cents + c) * sub + j] / count[p * cents + c]);
        double sse = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t p = 0; p < parts; ++p)
                for (size_t j = 0; j < sub; ++j) {
                    double t = double(x[i * dim + p * sub + j]) - at(p, assign[i * parts + p], j);
                    sse += t * t;
                }
        o.objective.push_back(sse);
    }
    return o;
}
std::vector<uint16_t> oracle_encode(const std::vector<float> &book, const std::vector<float> &x,
                                    size_t dim, size_t parts, size_t cents) {
    size_t sub = dim / parts, n = x.size() / dim;
    std::vector<uint16_t> codes(n * parts);
    for (size_t i = 0; i < n; ++i)
        for (size_t p = 0; p < parts; ++p) {
            double best_d = INFINITY;
            size_t best = 0;
            for (size_t c = 0; c < cents; ++c) {
                double d = 0;
                for (size_t j = 0; j < sub; ++j) {
                    double t = double(x[i * dim + p * sub + j]) - book[(p * cents + c) * sub + j];
                    d += t * t;
                }
                if (d < best_d) best_d = d, best = c;
            }
            codes[i * parts + p] = uint16_t(best);
        }
    return codes;
}
// ADC score from the index's book/codes, FP64, without calling ProductIndex::scores.
std::vector<double> oracle_adc(const kv::ProductIndex &p, const std::vector<float> &q) {
    std::vector<double> s(p.codes.size() / p.parts, 0);
    for (size_t i = 0; i < s.size(); ++i)
        for (size_t part = 0; part < p.parts; ++part)
            for (size_t j = 0; j < p.subdim; ++j)
                s[i] += double(q[part * p.subdim + j]) *
                        p.book[(part * p.centroids + p.codes[i * p.parts + part]) * p.subdim + j];
    return s;
}

void test_c() {
    std::mt19937 rng(17);
    std::normal_distribution<float> normal;
    section("C1 fit vs independent Lloyd", [&] {
        std::vector<float> train(160 * 8);
        for (float &x : train) x = normal(rng);
        kv::ProductIndex p(8, 2, 8);
        p.fit(train, 6);
        auto o = oracle_lloyd(train, 8, 2, 8, 6);
        double worst = INFINITY;
        if (p.book.size() == o.book.size()) {
            worst = 0;
            for (size_t i = 0; i < o.book.size(); ++i)
                worst = std::max(worst, double(std::abs(p.book[i] - o.book[i])));
        }
        check(worst < 1e-4, "C1 codebook equals oracle Lloyd after 6 iterations",
              "max center diff " + std::to_string(worst));
        bool obj = p.objective.size() == o.objective.size();
        for (size_t i = 0; obj && i < o.objective.size(); ++i)
            obj = std::abs(p.objective[i] - o.objective[i]) <= 1e-6 * o.objective[i];
        check(obj, "C1 objective[i] = SSE after update i (oracle)",
              p.objective.empty() ? "empty objective" : fmt(o.objective[0], p.objective[0]));
        check(!p.objective.empty() && p.objective[0] < o.init_sse * (1 - 1e-6),
              "C1 first update strictly lowers SSE on random data",
              p.objective.empty() ? "empty" : fmt(o.init_sse, p.objective[0]));
        bool mono = true;
        for (size_t i = 1; i < p.objective.size(); ++i)
            mono &= p.objective[i] <= p.objective[i - 1] * (1 + 1e-9);
        check(mono, "C1 SSE non-increasing");
        p.encode(train);
        p.fit(train, 2);
        check(p.codes.empty(), "C1 refit invalidates old codes");
        p.fit(std::vector<float>(160 * 8, 1), 3);
        bool finite = !p.book.empty();
        for (float x : p.book) finite &= std::isfinite(x);
        check(finite, "C1 empty clusters keep finite centers");
        check(throws_invalid([&] { p.fit(std::vector<float>(7 * 8), 2); }),
              "C1 rejects fewer rows than centroids");
    });
    section("C2 encode vs brute-force nearest center", [&] {
        std::vector<float> train(96 * 8), held(40 * 8);
        for (float &x : train) x = normal(rng);
        for (float &x : held) x = normal(rng);
        kv::ProductIndex p(8, 4, 6);
        p.fit(train, 4);
        auto book = p.book;
        p.encode(held);
        check(p.codes == oracle_encode(book, held, 8, 4, 6),
              "C2 held-out codes = nearest center per subspace");
        check(p.book == book, "C2 encode does not retrain the book");
        // Tie: book centers 0 and 2 in 1-D subspace; key 1 is equidistant -> smaller ID 0.
        kv::ProductIndex t(1, 1, 3);
        t.fit({0, 5, 2}, 1); // seeds rows 0,1,2 -> centers 0,5,2 (each its own cluster)
        t.encode({1});
        check(t.codes.size() == 1 && t.codes[0] == 0,
              "C2 equidistant key picks the smaller center ID");
        kv::ProductIndex untrained(4, 2, 2);
        check(throws_invalid([&] { untrained.encode({1, 2, 3, 4}); }),
              "C2 rejects encode before fit");
        check(throws_invalid([] { kv::ProductIndex bad(7, 2, 3); }),
              "C2 rejects dim not divisible by parts");
    });
    std::vector<float> train(128 * 8);
    for (float &x : train) x = normal(rng);
    kv::ProductIndex p(8, 2, 8);
    section("C3 ADC scores", [&] {
        p.fit(train, 8);
        p.encode(train);
        auto book = p.book;
        auto codes = p.codes;
        double worst = 0;
        for (int i = 0; i < 16; ++i) {
            std::vector<float> q(8);
            for (float &x : q) x = normal(rng);
            auto s = p.scores(q);
            auto o = oracle_adc(p, q);
            worst = std::max(worst, s.size() == o.size() ? max_abs_diff(o, s) : INFINITY);
        }
        check(worst < 2e-6, "C3 scores = q . reconstructed key (16 held-out queries)",
              "worst " + std::to_string(worst));
        // Inner product, not distance: a key twice as long in the query direction scores higher.
        kv::ProductIndex e(2, 1, 4);
        std::vector<float> k = {1, 0, 0, 1, -1, 0, 2, 0};
        e.fit(k, 3);
        e.encode(k);
        check(e.scores({1, 0}) == std::vector<float>{1, 0, -1, 2},
              "C3 hand example: inner products [1,0,-1,2]");
        check(p.book == book && p.codes == codes, "C3 scoring does not modify the index");
    });
    section("C4 retrieve", [&] {
        // Clustered keys so that PQ ranking is informative.
        const size_t n = 256, d = 16;
        std::vector<float> centers(8 * d), k(n * d), v(n * d);
        for (float &x : centers) x = 2 * normal(rng);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j) {
                k[i * d + j] = centers[(i % 8) * d + j] + 0.3f * normal(rng);
                v[i * d + j] = normal(rng);
            }
        kv::ProductIndex idx(d, 4, 16);
        idx.fit(k, 10);
        idx.encode(k);
        size_t budget = 48, sink = 2, recent = 3, bad_ids = 0, bad_out = 0, bad_bytes = 0;
        double recall_sum = 0;
        for (int trial = 0; trial < 8; ++trial) {
            std::vector<float> q(d);
            for (size_t j = 0; j < d; ++j)
                q[j] = centers[(trial % 8) * d + j] * 0.5f + 0.2f * normal(rng);
            auto r = kv::retrieve(idx, q, k, v, budget, sink, recent);
            auto adc = oracle_adc(idx, q);
            auto why = topk_violation(r.ids, adc, budget, sink, recent, 1e-4);
            if (!why.empty() && !bad_ids++) std::cout << "  first C4 id problem: " << why << '\n';
            // Output: FP64 attention over exactly the returned rows of the ORIGINAL K/V.
            std::vector<double> qk = to_double(q), sk, sv;
            for (size_t id : r.ids)
                if (id < n)
                    for (size_t j = 0; j < d; ++j)
                        sk.push_back(k[id * d + j]), sv.push_back(v[id * d + j]);
            bad_out += r.ids.empty() || max_abs_diff(oracle_attention(qk, sk, sv), r.output) > 1e-5;
            bad_bytes += r.fetched_bytes != 2 * budget * d * sizeof(float);
            // Recall vs exact inner-product top-k under the same mandatory rule.
            std::vector<double> exact(n);
            for (size_t i = 0; i < n; ++i) exact[i] = dot(qk, to_double(k), 0, i * d, d);
            std::vector<size_t> cand;
            for (size_t i = sink; i + recent < n; ++i) cand.push_back(i);
            std::stable_sort(cand.begin(), cand.end(),
                             [&](size_t a, size_t b) { return exact[a] > exact[b]; });
            std::vector<bool> ideal(n, false);
            for (size_t i = 0; i < budget - sink - recent; ++i) ideal[cand[i]] = true;
            size_t hits = 0;
            for (size_t id : r.ids)
                if (id < n && id >= sink && id + recent < n) hits += ideal[id];
            recall_sum += double(hits) / double(budget - sink - recent);
        }
        double recall = recall_sum / 8,
               random_rate = double(budget - sink - recent) / double(n - sink - recent);
        check(!bad_ids, "C4 ids = mandatory + ADC top-k with smaller-ID ties (8 queries)",
              std::to_string(bad_ids) + " queries wrong");
        check(!bad_out, "C4 output = FP64 attention over the gathered original rows",
              std::to_string(bad_out) + " queries wrong");
        check(!bad_bytes, "C4 fetched_bytes = 2 * budget * d * 4",
              std::to_string(bad_bytes) + " queries wrong");
        check(recall > std::max(0.6, 3 * random_rate),
              "C4 recall vs exact top-k beats random by 3x",
              "recall " + std::to_string(recall) + ", random " + std::to_string(random_rate));
        auto full = kv::retrieve(idx, std::vector<float>(d, 0.1f), k, v, n, sink, recent);
        std::vector<double> qf(d, 0.1);
        check(full.ids.size() == n && max_abs_diff(oracle_attention(qf, to_double(k), to_double(v)),
                                                   full.output) < 1e-5,
              "C4 full budget recovers full attention");
        check(throws_invalid([&] { kv::retrieve(idx, std::vector<float>(d, 1), k, v, 4, 2, 3); }),
              "C4 rejects infeasible budget");
        check(throws_invalid([&] {
                  kv::retrieve(idx, std::vector<float>(d, 1), k, std::vector<float>(d), 8, 1, 1);
              }),
              "C4 rejects K/V length mismatch");
    });
}
} // namespace

int main(int argc, char **argv) {
    std::string part = argc > 1 ? argv[1] : "all";
    for (char stage : {'A', 'B', 'C'}) {
        if (part != "all" && part != std::string(1, stage)) continue;
        int before_pass = g_pass, before_fail = g_fail;
        std::cout << "== part " << stage << " ==\n";
        if (stage == 'A') test_a();
        if (stage == 'B') test_b();
        if (stage == 'C') test_c();
        std::cout << "SUMMARY " << stage << ": " << g_pass - before_pass << " passed, "
                  << g_fail - before_fail << " failed\n";
    }
    if (g_fail)
        std::cout << "RESULT: FAIL (" << g_fail << " checks failed)\n";
    else
        std::cout << "RESULT: PASS (" << g_pass << " checks)\n";
    return g_fail ? 1 : 0;
}
