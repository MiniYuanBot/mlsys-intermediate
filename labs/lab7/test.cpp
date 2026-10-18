// Lab 7 CPU tests. Every check prints one PASS/FAIL line and the run continues past
// failures: an unfinished TODO ("Complete A1") is reported as FAIL for that check and the
// remaining checks still run, so one run shows every stage that is still open.
// Oracles never call the function under test: router weights, bucket layout, quantization
// metadata, cache admission schedules, Belady misses and the INT8 MLP are recomputed here
// from the raw inputs (long double / FP64), or are hand-derived numbers with the derivation
// written next to them.
#include "fixture.hpp"
#include <functional>
#include <iostream>
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
    s.precision(10);
    s << "expected " << expected << " vs actual " << actual;
    return s.str();
}
// One group of checks; an exception (including an unfinished TODO) becomes a FAIL line.
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
template <class F> bool throws_logic(F f) {
    try {
        f();
    } catch (const std::logic_error &) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// ---------- independent oracles ----------

// Router oracle in long double, written straight from the three published conventions.
// Selection score: sigmoid(logit)+bias for SigmoidRenorm, logit+bias otherwise.
// Weights: TopkSoftmax = softmax over the k selected logits; SoftmaxNoRenorm = softmax over
// all experts, read at the selected positions; SigmoidRenorm = selected sigmoids divided by
// their own sum. The bias never appears in a weight. Everything times spec.scale.
std::vector<moe::Route> oracle_routes(const std::vector<float> &logits, int tokens, int experts,
                                      int k, const moe::RouterSpec &spec) {
    std::vector<moe::Route> out;
    for (int t = 0; t < tokens; ++t) {
        std::vector<long double> value(experts), score(experts);
        for (int e = 0; e < experts; ++e) {
            long double z = logits[size_t(t) * experts + e];
            value[e] = spec.norm == moe::Norm::SigmoidRenorm ? 1.0L / (1.0L + expl(-z)) : z;
            score[e] = value[e] + (spec.bias.empty() ? 0.0L : (long double)spec.bias[e]);
        }
        std::vector<int> order(experts);
        for (int e = 0; e < experts; ++e) order[e] = e;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return score[a] > score[b]; });
        std::vector<long double> w(k);
        long double denominator = 0;
        if (spec.norm == moe::Norm::SigmoidRenorm) {
            for (int j = 0; j < k; ++j) denominator += w[j] = value[order[j]];
        } else if (spec.norm == moe::Norm::TopkSoftmax) {
            long double top = value[order[0]];
            for (int j = 0; j < k; ++j) denominator += w[j] = expl(value[order[j]] - top);
        } else {
            long double top = *std::max_element(value.begin(), value.end());
            for (int e = 0; e < experts; ++e) denominator += expl(value[e] - top);
            for (int j = 0; j < k; ++j) w[j] = expl(value[order[j]] - top);
        }
        for (int j = 0; j < k; ++j)
            out.push_back({t, order[j], float(w[j] / denominator * (long double)spec.scale)});
    }
    return out;
}
bool routes_match(const std::vector<moe::Route> &got, const std::vector<moe::Route> &want,
                  double tolerance, std::string &detail) {
    if (got.size() != want.size()) {
        detail = "route count " + std::to_string(want.size()) + " vs " + std::to_string(got.size());
        return false;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        if (got[i].token != want[i].token || got[i].expert != want[i].expert) {
            detail = "route " + std::to_string(i) + " expert " + std::to_string(want[i].expert) +
                     " vs " + std::to_string(got[i].expert);
            return false;
        }
        if (std::abs(double(got[i].weight) - double(want[i].weight)) > tolerance) {
            detail = "weight " + std::to_string(i) + " " + fmt(want[i].weight, got[i].weight);
            return false;
        }
    }
    return true;
}
// Bucket layout recomputed by an O(R*E) scan over the raw routes (never the prefix sum).
bool layout_ok(const moe::Dispatch &d, const std::vector<moe::Route> &routes, int experts,
               std::string &detail) {
    if (d.offsets.size() != size_t(experts) + 1) {
        detail = "offsets size";
        return false;
    }
    size_t cursor = 0;
    for (int e = 0; e < experts; ++e) {
        std::vector<moe::Route> expected;
        for (auto r : routes)
            if (r.expert == e) expected.push_back(r);
        if (d.offsets[e] != int(cursor)) {
            detail = "offset[" + std::to_string(e) + "] " + fmt(double(cursor), d.offsets[e]);
            return false;
        }
        for (size_t j = 0; j < expected.size(); ++j) {
            const auto &g = d.ordered.at(cursor + j);
            if (g.token != expected[j].token || g.expert != e ||
                std::abs(g.weight - expected[j].weight) > 1e-7) {
                detail = "bucket " + std::to_string(e) + " entry " + std::to_string(j);
                return false;
            }
        }
        cursor += expected.size();
    }
    if (d.offsets[experts] != int(routes.size()) || d.ordered.size() != routes.size()) {
        detail = "last offset / route conservation";
        return false;
    }
    return true;
}
// Belady on equal-sized experts, written as a direct simulation with full future knowledge.
int oracle_belady(const std::vector<int> &trace, size_t slots) {
    if (slots == 0) return int(trace.size());
    std::vector<int> resident;
    int misses = 0;
    for (size_t i = 0; i < trace.size(); ++i) {
        if (std::find(resident.begin(), resident.end(), trace[i]) != resident.end()) continue;
        ++misses;
        if (resident.size() == slots) {
            size_t victim = 0, farthest = 0;
            for (size_t j = 0; j < resident.size(); ++j) {
                size_t next = trace.size() + 1;
                for (size_t m = i + 1; m < trace.size(); ++m)
                    if (trace[m] == resident[j]) {
                        next = m;
                        break;
                    }
                if (next >= farthest) {
                    farthest = next;
                    victim = j;
                }
            }
            resident.erase(resident.begin() + long(victim));
        }
        resident.push_back(trace[i]);
    }
    return misses;
}
// FP64 dense oracle of down(ReLU(up(x))) straight from the quantized integers and scales.
std::vector<double> oracle_mlp(const moe::Expert &e, const std::vector<float> &x) {
    std::vector<double> h(e.up.rows), y(e.down.rows);
    for (int j = 0; j < e.up.rows; ++j) {
        double sum = 0;
        for (int c = 0; c < e.up.cols; ++c)
            sum += double(e.up.q[size_t(j) * e.up.cols + c]) * double(x[c]);
        h[j] = std::max(0.0, sum * double(e.up.scale[j]));
    }
    for (int d = 0; d < e.down.rows; ++d) {
        double sum = 0;
        for (int j = 0; j < e.down.cols; ++j)
            sum += double(e.down.q[size_t(d) * e.down.cols + j]) * h[j];
        y[d] = sum * double(e.down.scale[d]);
    }
    return y;
}

// ---------- A ----------
void a_tests() {
    section("a_router", [] {
        // Row 0 {1000,1000,0} is the tie/overflow row: both conventions give 0.5 here, which
        // is exactly why it cannot tell them apart. Row 1 {2,1,0,-1} separates them:
        //   TopkSoftmax      e^1/(e^1+1) = 0.73105858, 0.26894142          (sums to 1)
        //   SoftmaxNoRenorm  e^2/sum = 0.64391426, e^1/sum = 0.23688284    (sums to 0.88)
        //   SigmoidRenorm    0.88079708/(0.88079708+0.73105858) = 0.54644910, then x2.5
        std::vector<float> wide = {2, 1, 0, -1};
        struct Case {
            const char *name;
            moe::Norm norm;
            float scale;
            double first, second;
        };
        const Case cases[] = {
            {"topk-softmax", moe::Norm::TopkSoftmax, 1.f, 0.7310585786, 0.2689414214},
            {"softmax-no-renorm", moe::Norm::SoftmaxNoRenorm, 1.f, 0.6439142599, 0.2368828441},
            {"sigmoid-renorm-x2.5", moe::Norm::SigmoidRenorm, 2.5f, 1.3661227579, 1.1338772421},
        };
        for (const auto &c : cases) {
            moe::RouterSpec spec{c.norm, {}, c.scale};
            auto got = moe::route_topk(wide, 1, 4, 2, spec);
            std::string detail;
            bool ok = routes_match(got, oracle_routes(wide, 1, 4, 2, spec), 1e-6, detail);
            check(ok, std::string("a_router ") + c.name + " matches the long double oracle",
                  detail);
            bool hand = got.size() == 2 && std::abs(double(got[0].weight) - c.first) < 1e-6 &&
                        std::abs(double(got[1].weight) - c.second) < 1e-6;
            check(hand, std::string("a_router ") + c.name + " hand-derived weights",
                  got.size() == 2 ? fmt(c.first, got[0].weight) : "route count");
        }
        auto ties = moe::route_topk({1000, 1000, 0}, 1, 3, 2);
        check(ties.size() == 2 && ties[0].expert == 0 && ties[1].expert == 1 &&
                  std::abs(ties[0].weight - .5f) < 1e-6 && std::abs(ties[1].weight - .5f) < 1e-6,
              "a_router equal logits prefer the smaller ID and do not overflow",
              ties.size() == 2 ? fmt(.5, ties[0].weight) : "route count");
        // DeepSeek-V3: e_score_correction_bias moves the SELECTION only. Sigmoids are
        // 0.52497919, 0.54983400, 0.57444252; the bias picks experts 0 and 1, and the
        // weights are the unbiased sigmoids normalised over those two:
        // 0.52497919/(0.52497919+0.54983400) = 0.48843761.
        moe::RouterSpec biased{moe::Norm::SigmoidRenorm, {0.9f, 0.5f, 0.f}, 1.f};
        auto got = moe::route_topk({0.1f, 0.2f, 0.3f}, 1, 3, 2, biased);
        std::string detail;
        bool ok =
            routes_match(got, oracle_routes({0.1f, 0.2f, 0.3f}, 1, 3, 2, biased), 1e-6, detail);
        check(ok && got.size() == 2 && got[0].expert == 0 && got[1].expert == 1,
              "a_router selection-only bias does not enter the weights", detail);
        check(got.size() == 2 && std::abs(double(got[0].weight) - 0.4884376140) < 1e-6,
              "a_router biased weights come from the unbiased sigmoids",
              got.size() == 2 ? fmt(0.4884376140, got[0].weight) : "route count");
        check(throws_invalid([] { return moe::route_topk({1, 2}, 1, 2, 3); }),
              "a_router rejects k larger than the expert count");
        check(throws_invalid([] { return moe::route_topk({1, 2, 3}, 1, 2, 1); }),
              "a_router rejects a logits buffer of the wrong size");
        check(moe::route_topk({}, 0, 3, 2).empty(), "a_router accepts zero tokens");
    });
    // Every section rebuilds what it needs inside its own try/catch, so an unfinished A2
    // shows up as a FAIL line for each dependent check instead of killing the process.
    const std::vector<moe::Route> routes = {{0, 2, .5f}, {0, 0, .5f}, {1, 2, 1.f}, {1, 2, 1.f}};
    section("a_dispatch", [&] {
        auto d = moe::dispatch(routes, 2, 4);
        std::string detail;
        check(layout_ok(d, routes, 4, detail), "a_dispatch layout equals the O(R*E) oracle",
              detail);
        check(moe::dispatch({}, 0, 3).ordered.empty() &&
                  moe::dispatch({}, 0, 3).offsets == std::vector<int>{0, 0, 0, 0},
              "a_dispatch empty routes still produce E+1 offsets");
        check(throws_invalid([] { return moe::dispatch({{0, 5, 1.f}}, 1, 4); }),
              "a_dispatch rejects an out-of-range expert ID");
        // Order inside a bucket must follow the input, not the token ID.
        std::vector<moe::Route> reversed = {{2, 0, .3f}, {0, 0, .7f}};
        auto r = moe::dispatch(reversed, 3, 1);
        check(r.ordered.size() == 2 && r.ordered[0].token == 2 && r.ordered[1].token == 0,
              "a_dispatch is stable in input order inside a bucket");
    });
    section("a_quant", [] {
        std::vector<float> w = {0, 0, 0, -2, .1f, 1};
        auto q = moe::quantize(w, 2, 3);
        check(q.scale[0] == 1, "a_quant all-zero row keeps scale 1", fmt(1, q.scale[0]));
        double expected = 2.0 / 127.0;
        check(std::abs(double(q.scale[1]) - expected) < 1e-9,
              "a_quant row scale is rowmax/127 from the raw row", fmt(expected, q.scale[1]));
        bool within = true, saturates = true;
        for (size_t i = 0; i < w.size(); ++i) {
            double s = q.scale[i / 3];
            within &= std::abs(double(w[i]) - q.q[i] * s) <= s / 2 + 1e-6;
            saturates &= std::abs(int(q.q[i])) <= 127;
        }
        check(within, "a_quant error stays within half a scale step of the FP64 oracle");
        check(saturates && q.q[3] == -127, "a_quant clamps to [-127,127] with the peak at -127",
              fmt(-127, q.q[3]));
        check(throws_invalid([] { return moe::quantize({1, 2, 3}, 2, 3); }),
              "a_quant rejects a weight buffer of the wrong size");
    });
    section("a_combine", [&] {
        auto d = moe::dispatch(routes, 2, 4);
        // Bucket order is r1(token0,w=.5), r0(token0,w=.5), r2(token1,w=1), r3(token1,w=1);
        // single-channel outputs 2,4,6,8 give token0 = .5*2+.5*4 = 3, token1 = 6+8 = 14.
        auto y = moe::combine(d, {2, 4, 6, 8}, 2, 1);
        check(y.size() == 2 && std::abs(y[0] - 3) < 1e-6 && std::abs(y[1] - 14) < 1e-6,
              "a_combine adds duplicate routes instead of overwriting",
              y.size() == 2 ? fmt(14, y[1]) : "size");
        auto zero = moe::combine(moe::dispatch({}, 0, 2), {}, 0, 3);
        check(zero.empty(), "a_combine handles an empty dispatch");
        // A token with no route keeps its zero row.
        auto sparse = moe::dispatch({{1, 0, 2.f}}, 3, 1);
        auto out = moe::combine(sparse, {5, 7}, 3, 2);
        check(out.size() == 6 && out[0] == 0 && out[1] == 0 && std::abs(out[2] - 10) < 1e-6 &&
                  std::abs(out[3] - 14) < 1e-6 && out[4] == 0,
              "a_combine leaves unrouted tokens at zero");
        check(throws_invalid([&] { return moe::combine(d, {1, 2, 3}, 2, 1); }),
              "a_combine rejects an expert-output buffer of the wrong size");
    });
    section("a_packing", [&] {
        auto d = moe::dispatch(routes, 2, 4);
        auto p = moe::pack(d, {5, 6}, 1, 2);
        check(p.route_index == std::vector<int>({0, -1, 1, 2, 3, -1}),
              "a_packing pads each expert to a tile and marks padding with -1");
        check(p.values == std::vector<float>({5, 0, 5, 6, 6, 0}),
              "a_packing gathers the routed token rows, zero only for padding");
        check(p.expert == std::vector<int>({0, 0, 2, 2, 2, 2}),
              "a_packing keeps an expert ID per padded row and skips empty experts");
        auto exact = moe::pack(moe::dispatch({{0, 0, 1.f}, {1, 0, 1.f}}, 2, 1), {1, 2}, 1, 2);
        check(exact.route_index == std::vector<int>({0, 1}),
              "a_packing adds no tile when the bucket is already full");
    });
    section("a_stats", [&] {
        auto d = moe::dispatch(routes, 2, 4);
        auto stats = moe::router_stats(d, 2, 2);
        // load = [1,0,3,0]: padded = 2+0+4+0 = 6, overflow = max(0,3-2) = 1,
        // imbalance = 3/(4/4) = 3, entropy = -(0.25 ln 0.25 + 0.75 ln 0.75) = 0.5623351446.
        check(stats.load == std::vector<int>({1, 0, 3, 0}), "a_stats counts routes per expert");
        check(stats.padded == 6, "a_stats padded rows match the packed layout",
              fmt(6, stats.padded));
        check(stats.overflow == 1, "a_stats reports capacity overflow without dropping routes",
              fmt(1, stats.overflow));
        check(std::abs(stats.imbalance - 3) < 1e-9, "a_stats imbalance is max/mean",
              fmt(3, stats.imbalance));
        check(std::abs(stats.entropy - 0.5623351446) < 1e-7,
              "a_stats entropy uses routed assignments, not distinct tokens",
              fmt(0.5623351446, stats.entropy));
        // 6 routes over 3 experts (load [1,2,3]) separates "divide by the routed assignments"
        // from "divide by the expert count": p = [1/6, 2/6, 3/6] gives
        // -(1/6 ln 1/6 + 2/6 ln 2/6 + 3/6 ln 3/6) = 1.0114042647, imbalance = 3/(6/3) = 1.5.
        auto wide =
            moe::dispatch({{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 2, 1}, {2, 2, 1}, {3, 2, 1}}, 4, 3);
        auto spread = moe::router_stats(wide, 2, 2);
        check(spread.load == std::vector<int>({1, 2, 3}) && spread.padded == 8 &&
                  spread.overflow == 1,
              "a_stats load, padding and overflow with 6 routes over 3 experts");
        check(std::abs(spread.entropy - 1.0114042647) < 1e-7,
              "a_stats entropy normalises by the routed assignments, not the expert count",
              fmt(1.0114042647, spread.entropy));
        check(std::abs(spread.imbalance - 1.5) < 1e-9, "a_stats imbalance on an uneven load",
              fmt(1.5, spread.imbalance));
        auto empty = moe::router_stats(moe::dispatch({}, 0, 3), 4, 2);
        check(empty.entropy == 0 && empty.padded == 0 && empty.imbalance == 0,
              "a_stats empty dispatch stays at zero");
    });
}

// ---------- B ----------
void b_tests() {
    section("b_admission", [] {
        // Capacity 20B, 2B/us, one serial copy engine. 12B is admitted at 0 and finishes at
        // 6us; 8B queues behind it on the link and finishes at 10us. A 10B request can only
        // be served by freeing 12B (pinned) or 8B, so it must stay Blocked until 0 is released.
        moe::Cache c(20, 2);
        check(c.request(0, 12, 0) == moe::Access::Loaded, "b_admission first load is admitted");
        check(c.request(1, 8, 0) == moe::Access::Loaded, "b_admission second load is admitted");
        check(c.entries.at(0).ready == 6 && c.entries.at(1).ready == 10,
              "b_admission serial copy engine queues the second load",
              fmt(10, c.entries.at(1).ready));
        check(c.request(2, 10, 0) == moe::Access::Blocked && c.entries.size() == 2 &&
                  c.used == 20 && c.transferred == 20,
              "b_admission a Blocked request changes nothing (loading entries hold bytes)");
        check(c.advance(6) == std::vector<int>{0}, "b_admission advance reports completions");
        c.pin(0);
        check(c.request(2, 10, 6) == moe::Access::Blocked && c.entries.size() == 2,
              "b_admission will not evict a pinned expert");
        check(c.advance(10) == std::vector<int>{1}, "b_admission reports the second completion");
        check(c.request(2, 10, 10) == moe::Access::Blocked && c.entries.count(1) == 1,
              "b_admission plans victims before committing: 8B alone is not enough");
        c.release(0);
        check(c.request(2, 10, 10) == moe::Access::Loaded && c.entries.count(0) == 0 &&
                  c.used <= 20,
              "b_admission admits once enough bytes can actually be freed");
        check(c.request(2, 10, 10) == moe::Access::Pending,
              "b_admission a second request for a loading expert is Pending, not a new copy");
        check(c.transferred == 30, "b_admission bytes are counted once per admitted load",
              fmt(30, double(c.transferred)));
        moe::Cache small(8, 2);
        check(small.request(0, 9, 0) == moe::Access::Blocked && small.entries.empty(),
              "b_admission an expert larger than the capacity is always Blocked");
        check(throws_invalid([&] { return small.request(0, 4, -1); }),
              "b_admission rejects a clock going backwards");
    });
    section("b_lifetime", [] {
        moe::Cache c(20, 2);
        c.request(0, 10, 0);
        check(throws_logic([&] { c.pin(0); }), "b_lifetime cannot pin a still-loading expert");
        check(throws_logic([&] { c.pin(9); }), "b_lifetime cannot pin an absent expert");
        c.advance(5);
        c.pin(0);
        c.pin(0);
        c.release(0);
        check(c.entries.at(0).pins == 1, "b_lifetime pins are reference counted",
              fmt(1, c.entries.at(0).pins));
        c.release(0);
        check(throws_logic([&] { c.release(0); }), "b_lifetime rejects an unowned release");
        moe::Cache q(40, 2);
        q.request(0, 4, 0);
        q.request(1, 8, 0);
        q.request(2, 2, 0);
        // 4B, 8B, 2B on one engine finish at 2, 6, 7us; advance(6) returns them in completion
        // order, advance(7) only the last one.
        check(q.advance(6) == std::vector<int>({0, 1}),
              "b_lifetime advance sorts by completion time");
        check(q.advance(7) == std::vector<int>{2}, "b_lifetime advance reports each load once");
        check(q.advance(99).empty(), "b_lifetime advance past the end reports nothing new");
    });
    section("b_policy", [] {
        // Three 10B experts fill a 30B cache, then the demands 0,0,1,2 run. Uses end at
        // 0:3, 1:2, 2:2 while the last-use order ends at 0,1,2. Admitting a fourth expert
        // therefore evicts 0 under LRU (oldest stamp) but 1 under LFU (fewest uses, tie
        // broken by the older stamp). The two policies must disagree on this trace.
        for (moe::Policy policy : {moe::Policy::LRU, moe::Policy::LFU}) {
            moe::Cache c(30, 10, policy);
            for (int id : {0, 1, 2}) c.request(id, 10, 0);
            c.advance(3);
            for (int id : {0, 0, 1, 2}) c.request(id, 10, 3);
            c.request(3, 10, 3);
            bool lfu = policy == moe::Policy::LFU;
            int evicted = lfu ? 1 : 0, kept = lfu ? 0 : 1;
            check(c.entries.count(3) == 1 && c.entries.count(evicted) == 0 &&
                      c.entries.count(kept) == 1 && c.entries.count(2) == 1,
                  std::string("b_policy ") +
                      (lfu ? "LFU evicts the least used" : "LRU evicts the least recently used"));
        }
        // A prefetch must not look like a use: it may neither refresh the stamp nor raise the
        // use count of an entry that is already resident.
        moe::Cache c(20, 10);
        c.request(0, 10, 0);
        c.request(1, 10, 0);
        c.advance(2);
        c.request(0, 10, 2);       // demand: 0 becomes the most recent
        c.request(1, 10, 2, true); // prefetch touch of a resident entry: no promotion
        c.request(2, 10, 2);
        check(c.entries.count(1) == 0 && c.entries.count(0) == 1,
              "b_policy a prefetch touch does not promote a resident entry");
    });
    section("b_predictor", [] {
        moe::Predictor p;
        check(p.predict({10, 10}, 20, .3).empty(), "b_predictor cold start predicts nothing");
        for (int i = 0; i < 10; ++i) {
            p.observe(0);
            p.observe(1);
        }
        p.observe(0);
        check(p.predict({10, 10, 10}, 10, .3) == std::vector<int>{1},
              "b_predictor returns the observed successor");
        check(p.predict({10, 10, 10}, 9, .3).empty(),
              "b_predictor respects the byte budget, not a count");
        check(p.predict({10, 10, 10}, 10, 1.01).empty(),
              "b_predictor drops candidates below the confidence threshold");
        // A long stale phase followed by a short recent one: 30 observations of 0 -> 1 and
        // only 11 of 0 -> 2. Counting without forgetting gives p(1) = 30/41 = 0.73 and picks
        // the stale successor; discounting the whole table each step makes the recent
        // transition dominate instead.
        moe::Predictor stale;
        for (int i = 0; i < 30; ++i) {
            stale.observe(0);
            stale.observe(1);
        }
        for (int i = 0; i < 11; ++i) {
            stale.observe(2);
            stale.observe(0);
        }
        check(stale.predict({10, 10, 10}, 10, .5) == std::vector<int>{2},
              "b_predictor forgets the old hot successor after a phase change");
        // A big first candidate must be skipped, not end the scan: 0 -> {1,2} both at p = 0.5
        // after this history, and only the 10B one fits a 10B budget.
        moe::Predictor two;
        for (int i = 0; i < 5; ++i) {
            two.observe(0);
            two.observe(1);
            two.observe(0);
            two.observe(2);
        }
        two.observe(0);
        // Row 0 is {1: 1.8621, 2: 2.2989}: expert 2 is ranked first. Making expert 2 the
        // oversized one forces the scan to skip it and keep looking at expert 1, instead of
        // stopping at the first candidate that does not fit.
        check(two.predict({10, 10, 50}, 10, .3) == std::vector<int>{1},
              "b_predictor skips an oversized candidate and keeps looking");
    });
    section("b_replay", [] {
        // 10B experts, 2B/us, 3us of compute, one slot.
        // 0: load 0..5, compute 5..8; 1: load 8..13, compute 13..16; 0 again: 16..21, 21..24.
        auto r = moe::replay({0, 1, 0}, {10, 10}, 10, 2, 3, false);
        check(std::abs(r.finish - 24) < 1e-9, "b_replay hand schedule without prefetch",
              fmt(24, r.finish));
        check(r.transferred == 30 && r.misses == 3 && r.prefetched == 0 && r.wasted == 0,
              "b_replay counts three compulsory misses and no speculation");
        // Three 10B experts cycling through a two-slot cache, 6us of compute: the budget
        // compute*bandwidth = 12B now covers one expert, so the successor is prefetched
        // during compute and the next demand finds it ready.
        std::vector<int> cycle;
        for (int i = 0; i < 12; ++i) cycle.push_back(i % 3);
        std::vector<size_t> ten(3, 10);
        auto pf = moe::replay(cycle, ten, 20, 2, 6, true);
        auto demand = moe::replay(cycle, ten, 20, 2, 6, false);
        check(pf.finish < demand.finish, "b_replay prefetch wins on a periodic trace",
              fmt(demand.finish, pf.finish));
        check(pf.prefetch_hits > 0 && pf.misses < demand.misses,
              "b_replay counts demand hits served by speculation", fmt(demand.misses, pf.misses));
        check(pf.transferred >= pf.prefetched && pf.transferred > demand.transferred,
              "b_replay speculation adds traffic on top of the demand loads",
              fmt(double(demand.transferred), double(pf.transferred)));
        check(pf.wasted + pf.stranded <= pf.prefetched,
              "b_replay every prefetched byte is used, evicted unused, or left stranded");
        auto hit = moe::replay({0, 0, 0}, {10}, 10, 2, 3, true);
        check(std::abs(hit.finish - 14) < 1e-9 && hit.transferred == 10,
              "b_replay a resident expert is not reloaded", fmt(14, hit.finish));
        // Capacity for one expert only: an alternating trace thrashes and every access misses.
        auto thrash = moe::replay({0, 1, 0, 1}, {10, 10}, 10, 2, 3, false);
        check(thrash.misses == 4, "b_replay single-slot thrashing misses every access",
              fmt(4, thrash.misses));
        // A phase change with 7 experts and 2 slots: some speculation is evicted before any
        // demand touches it (wasted) and some is still untouched at the end (stranded).
        std::vector<int> phase;
        for (int i = 0; i < 30; ++i) phase.push_back(i % 3);
        for (int i = 0; i < 30; ++i) phase.push_back(3 + i % 4);
        auto mixed = moe::replay(phase, std::vector<size_t>(7, 10), 20, 2, 20, true);
        check(mixed.wasted > 0 && mixed.stranded > 0,
              "b_replay charges for speculation evicted unused and for what is left stranded",
              "wasted=" + std::to_string(mixed.wasted) +
                  " stranded=" + std::to_string(mixed.stranded));
        check(mixed.wasted + mixed.stranded <= mixed.prefetched &&
                  mixed.prefetched <= mixed.transferred,
              "b_replay the three byte counters stay consistent");
        check(moe::replay({}, {10}, 10, 2, 3, true).finish == 0, "b_replay empty trace is free");
    });
    section("b_oracle", [] {
        check(moe::oracle_misses({0, 1, 2, 0, 1, 2}, 2) == 4,
              "b_oracle hand trace: 4 misses with two slots",
              fmt(4, moe::oracle_misses({0, 1, 2, 0, 1, 2}, 2)));
        check(moe::oracle_misses({0, 0}, 0) == 2, "b_oracle zero slots miss every access");
        check(moe::oracle_misses({3, 3, 3}, 2) == 1, "b_oracle repeated access misses once");
        std::vector<std::vector<int>> traces = {{0, 1, 2, 3, 0, 1, 2, 3},
                                                {0, 1, 0, 2, 0, 3, 0, 1},
                                                {0, 0, 1, 1, 2, 2, 0, 2, 1},
                                                {4, 3, 2, 1, 0, 1, 2, 3, 4}};
        bool matches = true, bounds = true;
        for (const auto &trace : traces)
            for (size_t slots = 1; slots <= 3; ++slots) {
                int got = moe::oracle_misses(trace, slots);
                matches &= got == oracle_belady(trace, slots);
                auto lru = moe::replay(trace, std::vector<size_t>(8, 10), slots * 10, 2, 3, false);
                bounds &= got <= lru.misses;
            }
        check(matches, "b_oracle matches an independent Belady simulation on 12 cases");
        check(bounds, "b_oracle is a lower bound on demand LRU misses");
    });
}

// ---------- C ----------
void c_tests() {
    // Integer-valued weights so the quantized matrices are exact and the oracle is unambiguous:
    // up = [[1,-1],[2,0]] with x = [2,1] gives ReLU([1,4]) = [1,4]; down = I keeps [1,4].
    auto hand_expert = [] {
        return moe::Expert{moe::quantize({1, -1, 2, 0}, 2, 2), moe::quantize({1, 0, 0, 1}, 2, 2)};
    };
    section("c_expert", [&] {
        auto e = hand_expert();
        auto y = moe::evaluate(e, {2, 1});
        auto want = oracle_mlp(e, {2, 1});
        check(y.size() == 2 && std::abs(y[0] - want[0]) < 1e-4 && std::abs(y[1] - want[1]) < 1e-4,
              "c_expert matches the FP64 dequantized oracle",
              y.size() == 2 ? fmt(want[0], y[0]) : "size");
        check(y.size() == 2 && std::abs(y[0] - 1) < .02 && std::abs(y[1] - 4) < .02,
              "c_expert hand example down(ReLU(up([2,1]))) = [1,4]",
              y.size() == 2 ? fmt(4, y[1]) : "size");
        // ReLU must clip: x = [-2,-1] drives both hidden rows negative, so the output is zero.
        auto clipped = moe::evaluate(e, {-2, -1});
        check(clipped.size() == 2 && std::abs(clipped[0]) < 1e-4 && std::abs(clipped[1]) < 1e-4,
              "c_expert clips negative hidden activations");
        check(throws_invalid([&] { return moe::evaluate(e, {1, 2, 3}); }),
              "c_expert rejects an input of the wrong width");
    });
    section("c_forward", [&] {
        Fixture f(7, 5, 9, 5, 11);
        auto out = moe::forward(f.experts, f.dispatch, f.x, f.tokens, f.width);
        std::vector<double> oracle(out.size(), 0);
        for (auto r : f.dispatch.ordered) {
            std::vector<float> row(f.x.begin() + size_t(r.token) * f.width,
                                   f.x.begin() + size_t(r.token + 1) * f.width);
            auto y = oracle_mlp(f.experts[r.expert], row);
            for (int c = 0; c < f.width; ++c)
                oracle[size_t(r.token) * f.width + c] += r.weight * y[c];
        }
        double worst = 0;
        for (size_t i = 0; i < out.size(); ++i)
            worst = std::max(worst, std::abs(out[i] - oracle[i]));
        check(worst < 1e-5, "c_forward matches the per-route FP64 oracle (odd shapes, 5 experts)",
              fmt(0, worst));
        auto none = moe::forward(f.experts, moe::dispatch({}, 0, 5), {}, 0, f.width);
        check(none.empty(), "c_forward with no routes launches no expert work");
    });
    section("c_block", [&] {
        auto e = hand_expert();
        // gate = I, x = [2,1]: logits [2,1], top-1 selects expert 0, weight 1, plus residual.
        auto z = moe::block({e, e}, {1, 0, 0, 1}, {2, 1}, 1, 2, 1);
        check(z.size() == 2 && std::abs(z[0] - 3) < .02 && std::abs(z[1] - 5) < .02,
              "c_block gate -> top-1 -> MLP -> residual equals the hand oracle",
              z.size() == 2 ? fmt(5, z[1]) : "size");
        // A non-square gate whose transpose changes the winner: with gate [experts=3, width=2]
        // = [[1,0],[0,3],[0,0]] and x = [2,1] the logits are [2,3,0], so expert 1 (up = 2I)
        // wins and the block returns 2*[2,1] + [2,1] = [6,3]. Reading the same buffer as
        // [width, experts] gives logits [5,0,0] and picks expert 0 instead, which returns
        // [4,2] - a square gate could never show this.
        moe::Expert twice{moe::quantize({2, 0, 0, 2}, 2, 2), moe::quantize({1, 0, 0, 1}, 2, 2)};
        moe::Expert swap{moe::quantize({0, 1, 1, 0}, 2, 2), moe::quantize({1, 0, 0, 1}, 2, 2)};
        auto wide = moe::block({e, twice, swap}, {1, 0, 0, 3, 0, 0}, {2, 1}, 1, 2, 1);
        check(wide.size() == 2 && std::abs(wide[0] - 6) < .02 && std::abs(wide[1] - 3) < .02,
              "c_block reads the dense gate as [experts, width]",
              wide.size() == 2 ? fmt(6, wide[0]) : "size");
        // top-2 over two identical experts: the weights sum to 1, so the MoE part is unchanged.
        auto both = moe::block({e, e}, {1, 0, 0, 1}, {2, 1}, 1, 2, 2);
        check(both.size() == 2 && std::abs(both[0] - 3) < .02 && std::abs(both[1] - 5) < .02,
              "c_block weights of identical experts sum to one under TopkSoftmax",
              both.size() == 2 ? fmt(5, both[1]) : "size");
    });
}
} // namespace

int main(int argc, char **argv) {
    std::string part = argc > 1 ? argv[1] : "all";
    if (part == "A" || part == "all") a_tests();
    if (part == "B" || part == "all") b_tests();
    if (part == "C" || part == "all") c_tests();
    std::cout << "SUMMARY " << part << ": " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
