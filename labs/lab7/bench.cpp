#include "fixture.hpp"
#include <chrono>
#include <iostream>
int main(int argc, char **argv) {
    std::string part = argc > 1 ? argv[1] : "all";
    if (part == "A" || part == "all") {
        std::cout << "kind,part,tokens,experts,tile,routes,padded_rows\n";
        for (int t : {13, 64, 127})
            for (int tile : {1, 4, 16}) {
                Fixture f(t, 16, 32, 8, 1);
                auto p = moe::pack(f.dispatch, f.x, 16, tile);
                auto stats = moe::router_stats(f.dispatch, tile, 16);
                if (stats.padded != int(p.route_index.size()))
                    throw std::runtime_error("padding diagnostic");
                std::cout << "layout,A," << t << ",8," << tile << "," << f.dispatch.ordered.size()
                          << "," << p.route_index.size() << "\n";
            }
    }
    if (part == "B" || part == "all") {
        std::cout << "kind,part,capacity,prefetch,finish_us,transferred_bytes,wasted_bytes,misses,"
                     "blocked,oracle_misses\n";
        std::vector<int> trace;
        for (int i = 0; i < 120; ++i) trace.push_back(i < 60 ? i % 3 : 3 + i % 4);
        for (size_t cap : {4096, 8192, 12288})
            for (bool prefetch : {false, true}) {
                auto r = moe::replay(trace, std::vector<size_t>(8, 2048), cap, 128, 20, prefetch);
                std::cout << "synthetic_event,B," << cap << "," << prefetch << "," << r.finish
                          << "," << r.transferred << "," << r.wasted << "," << r.misses << ","
                          << r.blocked << "," << moe::oracle_misses(trace, cap / 2048) << "\n";
            }
    }
    if (part == "C" || part == "all") {
        std::cout << "kind,part,seed,width,tokens,ms,checksum\n";
        for (int seed : {1, 2, 3})
            for (int w : {32, 64}) {
                Fixture f(32, w, 2 * w, 8, seed);
                auto begin = std::chrono::steady_clock::now();
                std::vector<float> gate(8 * w, .01f);
                auto y = moe::block(f.experts, gate, f.x, 32, w, 2);
                double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - begin)
                                .count();
                std::cout << "measured_cpu,C," << seed << "," << w << ",32," << ms << ","
                          << std::accumulate(y.begin(), y.end(), 0.) << "\n";
            }
    }
}
