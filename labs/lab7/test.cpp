#include "fixture.hpp"
#include <iostream>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) throw std::runtime_error(#x);                                                    \
    } while (0)
void a_tests() {
    auto routes = moe::route_topk({1000, 1000, 0, -10, -9, -8}, 2, 3, 2);
    CHECK(routes[0].expert == 0 && routes[1].expert == 1);
    CHECK(std::abs(routes[0].weight - .5) < 1e-6 && routes[2].expert == 2);
    std::cout << "PASS a_router\n";
    auto d = moe::dispatch({{0, 2, .5}, {0, 0, .5}, {1, 2, 1}, {1, 2, 1}}, 2, 4);
    CHECK((d.offsets == std::vector<int>{0, 1, 1, 4, 4}));
    CHECK(d.ordered[1].token == 0 && d.ordered[2].token == 1);
    CHECK(moe::dispatch({}, 0, 3).ordered.empty());
    std::cout << "PASS a_dispatch\n";
    std::vector<float> w = {0, 0, 0, -2, .1, 1};
    auto q = moe::quantize(w, 2, 3);
    CHECK(q.scale[0] == 1);
    for (size_t i = 0; i < w.size(); ++i)
        CHECK(std::abs(w[i] - q.q[i] * q.scale[i / 3]) <= q.scale[i / 3] / 2 + 1e-6);
    std::cout << "PASS a_quant\n";
    auto y = moe::combine(d, {2, 4, 6, 8}, 2, 1);
    CHECK((y == std::vector<float>{3, 14}));
    std::cout << "PASS a_combine\n";
    auto p = moe::pack(d, {5, 6}, 1, 2);
    CHECK((p.route_index == std::vector<int>{0, -1, 1, 2, 3, -1}));
    CHECK((p.values == std::vector<float>{5, 0, 5, 6, 6, 0}));
    std::cout << "PASS a_packing\n";
    auto stats = moe::router_stats(d, 2, 2);
    CHECK(stats.padded == 6 && stats.overflow == 1 && stats.imbalance == 3);
    CHECK(moe::router_stats(moe::dispatch({}, 0, 3), 4, 2).entropy == 0);
    std::cout << "PASS a_stats\n";
}
void b_tests() {
    moe::Cache c(20, 2);
    CHECK(c.request(0, 12, 0) == moe::Access::Loaded);
    CHECK(c.request(1, 8, 0) == moe::Access::Loaded);
    CHECK(c.request(2, 10, 0) == moe::Access::Blocked);
    CHECK(c.used == 20 && c.transferred == 20);
    CHECK((c.advance(6) == std::vector<int>{0}));
    c.pin(0);
    CHECK(c.request(2, 10, 6) == moe::Access::Blocked && c.entries.size() == 2);
    CHECK((c.advance(10) == std::vector<int>{1}));
    CHECK(c.request(2, 10, 10) == moe::Access::Blocked && c.entries.size() == 2);
    c.release(0);
    CHECK(c.request(2, 10, 10) == moe::Access::Loaded);
    CHECK(c.used <= 20 && c.entries.count(0) == 0);
    std::cout << "PASS b_admission\n";
    bool bad = false;
    try {
        c.pin(2);
    } catch (const std::logic_error &) {
        bad = true;
    }
    CHECK(bad);
    c.advance(15);
    c.pin(2);
    c.pin(2);
    c.release(2);
    CHECK(c.entries.at(2).pins == 1);
    c.release(2);
    bad = false;
    try {
        c.release(2);
    } catch (const std::logic_error &) {
        bad = true;
    }
    CHECK(bad);
    std::cout << "PASS b_lifetime\n";
    moe::Predictor p;
    CHECK(p.predict({10, 10}, 20, .3).empty());
    for (int i = 0; i < 10; ++i) {
        p.observe(0);
        p.observe(1);
    }
    p.observe(0);
    CHECK((p.predict({10, 10, 10}, 10, .3) == std::vector<int>{1}));
    CHECK(p.predict({10, 10, 10}, 9, .3).empty());
    for (int i = 0; i < 30; ++i) {
        p.observe(2);
        p.observe(0);
    }
    CHECK((p.predict({10, 10, 10}, 10, .5) == std::vector<int>{2}));
    std::cout << "PASS b_predictor\n";
    auto r = moe::replay({0, 1, 0}, {10, 10}, 10, 2, 3, false);
    CHECK(r.finish == 24 && r.transferred == 30 && r.misses == 3);
    auto hit = moe::replay({0, 0, 0}, {10}, 10, 2, 3, true);
    CHECK(hit.finish == 14 && hit.transferred == 10);
    std::cout << "PASS b_replay\n";
    CHECK(moe::oracle_misses({0, 1, 2, 0, 1, 2}, 2) == 4);
    CHECK(moe::oracle_misses({0, 0}, 0) == 2);
    auto lru = moe::replay({0, 1, 2, 0, 1, 2}, {10, 10, 10}, 20, 2, 3, false);
    CHECK(moe::oracle_misses({0, 1, 2, 0, 1, 2}, 2) <= lru.misses);
    std::cout << "PASS b_oracle\n";
}
void c_tests() {
    // Independent dense FP32 oracle on exactly representable quantized integer weights.
    moe::Expert e{moe::quantize({1, -1, 2, 0}, 2, 2), moe::quantize({1, 0, 0, 1}, 2, 2)};
    auto y = moe::evaluate(e, {2, 1});
    CHECK(std::abs(y[0] - 1) < .02 && std::abs(y[1] - 4) < .02);
    std::cout << "PASS c_expert\n";
    Fixture f(7, 5, 9, 5, 11);
    auto out = moe::forward(f.experts, f.dispatch, f.x, f.tokens, f.width);
    std::vector<float> oracle(out.size(), 0);
    for (auto r : f.dispatch.ordered) {
        auto &ex = f.experts[r.expert];
        std::vector<double> h(ex.up.rows);
        for (int j = 0; j < ex.up.rows; ++j) {
            double sum = 0;
            for (int k = 0; k < f.width; ++k)
                sum +=
                    double(ex.up.q[j * f.width + k]) * ex.up.scale[j] * f.x[r.token * f.width + k];
            h[j] = std::max(0., sum);
        }
        for (int k = 0; k < f.width; ++k)
            for (int j = 0; j < ex.up.rows; ++j)
                oracle[r.token * f.width + k] +=
                    r.weight * ex.down.q[k * ex.up.rows + j] * ex.down.scale[k] * h[j];
    }
    for (size_t i = 0; i < out.size(); ++i) CHECK(std::abs(out[i] - oracle[i]) < 1e-5);
    std::cout << "PASS c_forward\n";
    auto z = moe::block({e, e}, {1, 0, 0, 1}, {2, 1}, 1, 2, 1);
    CHECK(std::abs(z[0] - 3) < .02 && std::abs(z[1] - 5) < .02);
    std::cout << "PASS c_block\n";
}
int main(int argc, char **argv) {
    std::string part = argc > 1 ? argv[1] : "all";
    if (part == "A" || part == "all") a_tests();
    if (part == "B" || part == "all") b_tests();
    if (part == "C" || part == "all") c_tests();
}
