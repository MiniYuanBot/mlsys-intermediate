#include "pq_student.hpp"
#include <iostream>
#include <random>
#include <string>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) throw std::runtime_error(#x);                                                    \
    } while (0)
template <class F> void rejects(F f) {
    bool threw = false;
    try {
        f();
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}
void test_a() {
    for (auto axis : {kv::Axis::Token, kv::Axis::Channel})
        for (size_t g : {1, 2, 4, 8}) {
            std::vector<float> x(35);
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i)) * 3;
            auto p = kv::quantize(x, 5, 7, g, axis);
            auto y = kv::dequantize(p);
            CHECK(p.codes.size() == 18 && (p.codes.back() & 0xf0) == 0);
            for (size_t i = 0; i < x.size(); ++i) {
                auto j = kv::group_index(i / 7, i % 7, 5, 7, g, axis);
                CHECK(std::abs(x[i] - y[i]) <= .501 * p.scales[j] + 1e-6);
            }
            auto z = kv::dequantize(kv::quantize(std::vector<float>(35, -2.5), 5, 7, g, axis));
            for (float value : z) CHECK(value == -2.5);
        }
    // Finite opposite extrema must not overflow subtraction; tiny ranges must
    // not create a zero scale. This is a correctness check, not a benchmark.
    float hi = std::numeric_limits<float>::max();
    auto extreme = kv::dequantize(kv::quantize({-hi, hi}, 1, 2, 2, kv::Axis::Token));
    CHECK(std::isfinite(extreme[0]) && std::isfinite(extreme[1]));
    CHECK(extreme[0] == -hi && std::abs(double(extreme[1]) / hi - 1.0) < 1e-6);
    float tiny = std::numeric_limits<float>::denorm_min();
    auto small = kv::dequantize(kv::quantize({0.f, tiny}, 1, 2, 2, kv::Axis::Token));
    CHECK(small[0] == 0.f && small[1] == tiny);
    rejects([] { kv::quantize({NAN}, 1, 1, 1, kv::Axis::Token); });
    rejects([] { kv::quantize({1}, 1, 1, 0, kv::Axis::Token); });
    CHECK((kv::gather_rows({1, 2, 3, 4, 5, 6}, 2, {0, 2}) == std::vector<float>{1, 2, 5, 6}));
    CHECK(kv::gather_rows({1}, 1, {}).empty());
    rejects([] { kv::gather_rows({1, 2}, 1, {1, 0}); });
    rejects([] { kv::gather_rows({1, 2}, 1, {0, 0}); });
    rejects([] { kv::gather_rows({1, 2}, 1, {2}); });
    auto e = kv::errors({3, 4}, {0, 0});
    CHECK(std::abs(e.relative_l2 - 1) < 1e-12 && e.max_abs == 4 && e.cosine == 0);
    CHECK(kv::errors({0}, {0}).cosine == 1);
    std::cout << "PASS A: axes/tails/constants/nibbles, gather contracts, error conventions\n";
}
void test_b() {
    CHECK((kv::select_tokens({0, 4, 4, 1, 0}, 4, 1, 1) == std::vector<size_t>{0, 1, 2, 4}));
    CHECK((kv::select_tokens({1, 2, 3}, 3, 2, 2) == std::vector<size_t>{0, 1, 2}));
    rejects([] { kv::select_tokens({1, 2, 3}, 1, 1, 1); });
    for (size_t residual : {0, 3, 8}) {
        kv::ResidualCache cache(7, 4, residual, 3);
        rejects([&] { cache.attend(std::vector<float>(7)); });
        for (size_t t = 0; t < 37; ++t) {
            std::vector<float> k(7), v(7), q(7);
            for (size_t j = 0; j < 7; ++j) {
                k[j] = std::sin(float(t * 7 + j));
                v[j] = std::cos(float(t + j));
                q[j] = float(j) * 20;
            }
            cache.append(k, v);
            CHECK(cache.tokens == t + 1);
            CHECK(cache.recent_k.size() / 7 < residual + 4);
            CHECK(cache.chunks.size() * 4 + cache.recent_k.size() / 7 == t + 1);
            auto [dk, dv] = cache.materialize();
            CHECK(dk.size() == (t + 1) * 7 && dv.size() == dk.size());
            std::vector<size_t> ids(t + 1);
            std::iota(ids.begin(), ids.end(), 0);
            auto expected = kv::attention(q, dk, dv, ids);
            auto actual = cache.attend(q);
            CHECK(kv::errors(expected, actual).max_abs < 2e-5);
            size_t bytes = 8 * cache.recent_k.size();
            for (auto &c : cache.chunks) bytes += c.k.bytes() + c.v.bytes();
            CHECK(cache.bytes() == bytes);
        }
        auto before = cache.materialize();
        rejects([&] { cache.append({1}, {2}); });
        rejects([&] { cache.append(std::vector<float>(7, NAN), std::vector<float>(7)); });
        CHECK(before == cache.materialize() && cache.tokens == 37);
    }
    std::cout << "PASS B: mandatory ties, 111 append boundaries, online softmax oracle, invalid "
                 "atomicity\n";
}
void test_c() {
    // Exact book: four training rows become four distinct centers, including a tie.
    std::vector<float> k = {1, 0, 0, 1, -1, 0, 0, -1}, v = {1, 2, 3, 4, 5, 6, 7, 8};
    kv::ProductIndex exact(2, 1, 4);
    exact.fit(k, 3);
    exact.encode(k);
    auto scores = exact.scores({2, 3});
    CHECK((scores == std::vector<float>{2, 3, -2, -3}));
    auto r = kv::retrieve(exact, {2, 3}, k, v, 4, 1, 1);
    CHECK(kv::errors(kv::attention({2, 3}, k, v, {0, 1, 2, 3}), r.output).max_abs < 1e-6);
    CHECK(r.fetched_bytes == 64);
    rejects([&] { kv::retrieve(exact, {2, 3}, k, v, 1, 1, 1); });
    kv::ProductIndex untrained(4, 2, 2);
    rejects([&] { untrained.encode({1, 2, 3, 4}); });
    rejects([] { kv::ProductIndex bad(7, 2, 3); });
    std::mt19937 rng(17);
    std::normal_distribution<float> normal;
    std::vector<float> train(128 * 8), queries(16 * 8);
    for (float &x : train) x = normal(rng);
    for (float &x : queries) x = normal(rng);
    kv::ProductIndex p(8, 2, 8);
    p.fit(train, 8);
    p.encode(train);
    for (size_t i = 1; i < p.objective.size(); ++i)
        CHECK(p.objective[i] <= p.objective[i - 1] + 1e-5);
    auto codes = p.codes;
    auto book = p.book;
    for (size_t i = 0; i < 16; ++i) {
        std::vector<float> q(queries.begin() + i * 8, queries.begin() + (i + 1) * 8);
        auto s = p.scores(q);
        // Independent reconstructed-center dot product tests the lookup layout.
        for (size_t row = 0; row < 128; ++row) {
            double expected = 0;
            for (size_t part = 0; part < 2; ++part)
                for (size_t j = 0; j < 4; ++j)
                    expected += double(q[part * 4 + j]) *
                                p.book[p.offset(part, p.codes[row * 2 + part], j)];
            CHECK(std::abs(s[row] - expected) < 2e-6);
        }
        auto out = kv::retrieve(p, q, train, train, 16, 2, 3);
        CHECK(out.ids.size() == 16 && out.ids.front() == 0 && out.ids.back() == 127);
        CHECK(out.fetched_bytes == 16 * 8 * 8);
    }
    CHECK(p.book == book && p.codes == codes);
    p.fit(std::vector<float>(128 * 8, 1), 3); // Empty clusters must remain finite.
    for (float x : p.book) CHECK(std::isfinite(x));
    CHECK(p.codes.empty());
    std::cout << "PASS C: exact PQ, monotone Lloyd SSE, held-out ADC oracle, retrieve/gather, "
                 "refit invalidation\n";
}
int main(int argc, char **argv) {
    std::string part = argc > 1 ? argv[1] : "all";
    if (part == "A" || part == "all") test_a();
    if (part == "B" || part == "all") test_b();
    if (part == "C" || part == "all") test_c();
}
