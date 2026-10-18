// Lab 7 CPU benchmarks. Part A reports layout padding and the weight mass each router
// convention keeps; part B replays routing traces through the event model (modeled, not
// measured: the us come from the declared bandwidth and compute cost); part C measures real
// CPU wall time for the INT8 MoE block.
#include "fixture.hpp"
#include <chrono>
#include <iostream>
#include <random>

namespace {
// 120 accesses: 60 cycling over experts 0-2, then 60 cycling over 3-6. The phase change is
// what makes a stale transition table prefetch the wrong expert.
std::vector<int> phase_trace() {
    std::vector<int> t;
    for (int i = 0; i < 120; ++i) t.push_back(i < 60 ? i % 3 : 3 + i % 4);
    return t;
}
// 120 uniform draws over all 8 experts: no transition is predictable, so every prefetch is
// a gamble and the hit rate can only come from capacity.
std::vector<int> uniform_trace() {
    std::mt19937 rng(20261018);
    std::uniform_int_distribution<int> pick(0, 7);
    std::vector<int> t;
    for (int i = 0; i < 120; ++i) t.push_back(pick(rng));
    return t;
}
// Zipf-like popularity over 8 experts with a sticky run length of 2: skewed but not periodic.
std::vector<int> skewed_trace() {
    std::mt19937 rng(7);
    std::discrete_distribution<int> pick({40, 20, 14, 10, 6, 4, 3, 3});
    std::vector<int> t;
    while (t.size() < 120) {
        int e = pick(rng);
        t.push_back(e);
        t.push_back(e);
    }
    t.resize(120);
    return t;
}
} // namespace

int main(int argc, char **argv) {
    std::string part = argc > 1 ? argv[1] : "all";
    if (part == "A" || part == "all") {
        std::cout << "kind,part,tokens,experts,tile,routes,padded_rows,imbalance,entropy\n";
        for (int t : {13, 64, 127})
            for (int tile : {1, 4, 16}) {
                Fixture f(t, 16, 32, 8, 1);
                auto p = moe::pack(f.dispatch, f.x, 16, tile);
                auto stats = moe::router_stats(f.dispatch, tile, 16);
                if (stats.padded != int(p.route_index.size()))
                    throw std::runtime_error("padding diagnostic disagrees with pack");
                std::cout << "layout,A," << t << ",8," << tile << "," << f.dispatch.ordered.size()
                          << "," << p.route_index.size() << "," << stats.imbalance << ","
                          << stats.entropy << "\n";
            }
        // How much weight mass each convention keeps, on one fixed batch of logits.
        std::cout << "kind,part,convention,topk,weight_sum_mean,weight_sum_min\n";
        std::mt19937 rng(3);
        std::normal_distribution<float> normal(0.f, 2.f);
        std::vector<float> logits(64 * 8);
        for (auto &v : logits) v = normal(rng);
        const char *names[] = {"topk_softmax", "softmax_no_renorm", "sigmoid_renorm"};
        moe::Norm norms[] = {moe::Norm::TopkSoftmax, moe::Norm::SoftmaxNoRenorm,
                             moe::Norm::SigmoidRenorm};
        for (int k : {2, 4})
            for (int n = 0; n < 3; ++n) {
                auto routes = moe::route_topk(logits, 64, 8, k, {norms[n], {}, 1.f});
                double total = 0, worst = 1e9;
                for (int t = 0; t < 64; ++t) {
                    double sum = 0;
                    for (int j = 0; j < k; ++j) sum += routes[size_t(t) * k + j].weight;
                    total += sum;
                    worst = std::min(worst, sum);
                }
                std::cout << "router_mass,A," << names[n] << "," << k << "," << total / 64 << ","
                          << worst << "\n";
            }
    }
    if (part == "B" || part == "all") {
        std::cout << "kind,part,trace,policy,capacity,slots,prefetch,finish_us,transferred_bytes,"
                     "prefetched_bytes,wasted_bytes,stranded_bytes,misses,prefetch_hits,blocked,"
                     "oracle_misses\n";
        const size_t expert_bytes = 2048;
        const std::vector<size_t> sizes(8, expert_bytes);
        struct Named {
            const char *name;
            std::vector<int> trace;
        };
        const Named traces[] = {
            {"phase", phase_trace()}, {"uniform", uniform_trace()}, {"skewed", skewed_trace()}};
        for (const auto &named : traces)
            for (size_t capacity : {2048u, 4096u, 6144u, 8192u})
                for (moe::Policy policy : {moe::Policy::LRU, moe::Policy::LFU})
                    for (bool prefetch : {false, true}) {
                        auto r =
                            moe::replay(named.trace, sizes, capacity, 128, 20, prefetch, policy);
                        std::cout << "synthetic_event,B," << named.name << ","
                                  << (policy == moe::Policy::LRU ? "LRU" : "LFU") << "," << capacity
                                  << "," << capacity / expert_bytes << "," << prefetch << ","
                                  << r.finish << "," << r.transferred << "," << r.prefetched << ","
                                  << r.wasted << "," << r.stranded << "," << r.misses << ","
                                  << r.prefetch_hits << "," << r.blocked << ","
                                  << moe::oracle_misses(named.trace, capacity / expert_bytes)
                                  << "\n";
                    }
    }
    if (part == "C" || part == "all") {
        std::cout << "kind,part,seed,width,tokens,ms,checksum\n";
        for (int seed : {1, 2, 3})
            for (int w : {32, 64}) {
                Fixture f(32, w, 2 * w, 8, seed);
                std::vector<float> gate(8 * size_t(w), .01f);
                moe::block(f.experts, gate, f.x, 32, w, 2); // warm the caches, not timed
                std::vector<double> samples;
                std::vector<float> y;
                for (int rep = 0; rep < 5; ++rep) {
                    auto begin = std::chrono::steady_clock::now();
                    y = moe::block(f.experts, gate, f.x, 32, w, 2);
                    samples.push_back(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - begin)
                                          .count());
                }
                std::sort(samples.begin(), samples.end());
                std::cout << "measured_cpu,C," << seed << "," << w << ",32," << samples[2] << ","
                          << std::accumulate(y.begin(), y.end(), 0.) << "\n";
            }
    }
}
