#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace moe {
struct Route {
    int token, expert;
    float weight;
};
struct Dispatch {
    std::vector<int> offsets;
    std::vector<Route> ordered;
};
// A1: logits is row-major [tokens, experts]. Select top-k BEFORE normalization.
// Equal logits prefer smaller expert IDs; subtract max before exp. Test: a_router.
inline std::vector<Route> route_topk(const std::vector<float> &logits, int tokens, int experts,
                                     int k) {
    // BEGIN TODO A1
    if (tokens < 0 || experts < 1 || k < 1 || k > experts ||
        logits.size() != size_t(tokens * experts))
        throw std::invalid_argument("router shape");
    std::vector<Route> out;
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> ids(experts);
        std::iota(ids.begin(), ids.end(), 0);
        std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
            return logits[t * experts + a] > logits[t * experts + b];
        });
        float maximum = logits[t * experts + ids[0]], total = 0;
        std::vector<float> weights(k);
        for (int j = 0; j < k; ++j) {
            weights[j] = std::exp(logits[t * experts + ids[j]] - maximum);
            total += weights[j];
        }
        for (int j = 0; j < k; ++j) out.push_back({t, ids[j], weights[j] / total});
    }
    return out;
    // END TODO A1
}
// A2: counting-sort routes by expert, preserving their original order within a bin.
// offsets has experts+1 entries, including empty experts. Test: a_dispatch.
inline Dispatch dispatch(const std::vector<Route> &routes, int tokens, int experts) {
    // BEGIN TODO A2
    Dispatch d{std::vector<int>(experts + 1, 0), std::vector<Route>(routes.size())};
    for (auto r : routes) {
        if (r.token < 0 || r.token >= tokens || r.expert < 0 || r.expert >= experts)
            throw std::invalid_argument("route index");
        ++d.offsets[r.expert + 1];
    }
    for (int e = 1; e <= experts; ++e) d.offsets[e] += d.offsets[e - 1];
    auto cursor = d.offsets;
    for (auto r : routes) d.ordered[cursor[r.expert]++] = r;
    return d;
    // END TODO A2
}
struct Quantized {
    int rows, cols;
    std::vector<int8_t> q;
    std::vector<float> scale;
};
// A3: row-wise symmetric INT8, round-to-nearest, clamp [-127,127].
// All-zero rows use scale=1. Keep metadata bytes in your report. Test: a_quant.
inline Quantized quantize(const std::vector<float> &w, int rows, int cols) {
    // BEGIN TODO A3
    if (rows < 1 || cols < 1 || w.size() != size_t(rows * cols))
        throw std::invalid_argument("weight shape");
    Quantized z{rows, cols, std::vector<int8_t>(w.size()), std::vector<float>(rows)};
    for (int r = 0; r < rows; ++r) {
        float peak = 0;
        for (int c = 0; c < cols; ++c) peak = std::max(peak, std::abs(w[r * cols + c]));
        z.scale[r] = peak == 0 ? 1.f : peak / 127.f;
        for (int c = 0; c < cols; ++c) {
            int value = int(std::round(w[r * cols + c] / z.scale[r]));
            z.q[r * cols + c] = int8_t(std::clamp(value, -127, 127));
        }
    }
    return z;
    // END TODO A3
}
// A4: scatter [routes,width] expert outputs back to tokens with route weights.
// Repeated token/expert pairs ADD; no overwrite or hidden token dropping. Test: a_combine.
inline std::vector<float> combine(const Dispatch &d, const std::vector<float> &ys, int tokens,
                                  int width) {
    // BEGIN TODO A4
    if (ys.size() != d.ordered.size() * size_t(width)) throw std::invalid_argument("output shape");
    std::vector<float> out(tokens * width, 0);
    for (size_t i = 0; i < d.ordered.size(); ++i) {
        const auto r = d.ordered[i];
        for (int c = 0; c < width; ++c) {
            out[r.token * width + c] += r.weight * ys[i * width + c];
        }
    }
    return out;
    // END TODO A4
}
struct Packed {
    std::vector<int> expert, route_index;
    std::vector<float> values;
};
// A5: gather by expert into blocks of tile rows. Only padding rows carry route_index=-1.
// Preserve every original route; masks distinguish padding from legitimate zero tokens.
// Test: a_packing uses empty experts, duplicate routes, and a final partial tile.
inline Packed pack(const Dispatch &d, const std::vector<float> &x, int width, int tile) {
    // BEGIN TODO A5
    if (tile <= 0 || width <= 0) throw std::invalid_argument("tile shape");
    Packed p;
    for (size_t e = 0; e + 1 < d.offsets.size(); ++e) {
        int count = d.offsets[e + 1] - d.offsets[e];
        int rows = (count + tile - 1) / tile * tile;
        for (int row = 0; row < rows; ++row) {
            p.expert.push_back(int(e));
            bool valid = row < count;
            int route = valid ? d.offsets[e] + row : -1;
            p.route_index.push_back(route);
            for (int c = 0; c < width; ++c) {
                float value = valid ? x[d.ordered[route].token * width + c] : 0;
                p.values.push_back(value);
            }
        }
    }
    return p;
    // END TODO A5
}
struct RouterStats {
    std::vector<int> load;
    double entropy, imbalance;
    int padded, overflow;
};
// A6: diagnostics count actual routed assignments, not distinct tokens. Entropy is
// on empirical expert frequencies; imbalance is max/mean. Capacity is diagnostic:
// report overflow but NEVER drop routes in dispatch. Test: a_stats, including empty.
inline RouterStats router_stats(const Dispatch &d, int tile, int capacity) {
    // BEGIN TODO A6
    RouterStats s{{}, 0, 0, 0, 0};
    int experts = int(d.offsets.size()) - 1;
    double total = double(d.ordered.size());
    int maximum = 0;
    for (int e = 0; e < experts; ++e) {
        int n = d.offsets[e + 1] - d.offsets[e];
        s.load.push_back(n);
        maximum = std::max(maximum, n);
        s.padded += (n + tile - 1) / tile * tile;
        s.overflow += std::max(0, n - capacity);
        if (n > 0) {
            double probability = n / total;
            s.entropy -= probability * std::log(probability);
        }
    }
    if (total > 0) s.imbalance = maximum / (total / experts);
    return s;
    // END TODO A6
}
enum class Access { Hit, Pending, Loaded, Blocked };
struct Entry {
    size_t bytes;
    double ready;
    long stamp;
    int pins = 0;
    bool prefetched = false;
};
struct Cache {
    size_t capacity, used = 0, transferred = 0, wasted = 0;
    double bandwidth, link_free = 0, now = 0;
    long clock = 0;
    std::map<int, Entry> entries;
    Cache(size_t bytes, double bytes_per_us) : capacity(bytes), bandwidth(bytes_per_us) {
        if (bytes_per_us <= 0) throw std::invalid_argument("bandwidth");
    }
    // B1: monotonic clock and serial-copy-engine reservation. Loading entries occupy bytes.
    // Victims are ready, unpinned LRU. Plan all evictions first: Blocked must be atomic.
    // Demand access to an existing prefetch clears speculative status. Test: b_admission.
    Access request(int id, size_t bytes, double time, bool prefetch = false) {
        // BEGIN TODO B1
        if (time < now) throw std::invalid_argument("time reversal");
        now = time;
        auto found = entries.find(id);
        if (found != entries.end()) {
            found->second.stamp = ++clock;
            if (!prefetch) found->second.prefetched = false;
            return found->second.ready <= time ? Access::Hit : Access::Pending;
        }
        if (bytes > capacity) return Access::Blocked;
        std::vector<std::pair<long, int>> candidates;
        for (const auto &[key, e] : entries)
            if (!e.pins && e.ready <= time) candidates.emplace_back(e.stamp, key);
        std::sort(candidates.begin(), candidates.end());
        size_t available = capacity - used;
        std::vector<int> victims;
        for (auto [stamp, key] : candidates) {
            if (available >= bytes) break;
            available += entries.at(key).bytes;
            victims.push_back(key);
        }
        if (available < bytes) return Access::Blocked;
        for (int key : victims) {
            auto e = entries.at(key);
            used -= e.bytes;
            if (e.prefetched) wasted += e.bytes;
            entries.erase(key);
        }
        link_free = std::max(time, link_free) + double(bytes) / bandwidth;
        entries.emplace(id, Entry{bytes, link_free, ++clock, 0, prefetch});
        used += bytes;
        transferred += bytes;
        return Access::Loaded;
        // END TODO B1
    }
    // B2: wait for the actual load completion before pinning. Pins are reference counts.
    // Do not silently pin an absent/loading expert or release someone else's lease.
    // Test: b_lifetime; return ready IDs sorted by completion then ID for deterministic replay.
    std::vector<int> advance(double time) {
        // BEGIN TODO B2
        if (time < now) throw std::invalid_argument("time reversal");
        std::vector<std::pair<double, int>> events;
        for (const auto &[id, e] : entries)
            if (e.ready > now && e.ready <= time) events.emplace_back(e.ready, id);
        std::sort(events.begin(), events.end());
        now = time;
        std::vector<int> done;
        for (auto [at, id] : events) done.push_back(id);
        return done;
        // END TODO B2
    }
    // B3: acquire a ready expert lease; nested consumers each own one pin.
    void pin(int id) {
        // BEGIN TODO B3
        auto it = entries.find(id);
        if (it == entries.end() || it->second.ready > now) throw std::logic_error("not ready");
        ++it->second.pins;
        it->second.stamp = ++clock;
        // END TODO B3
    }
    // B4: release exactly one existing lease; reject absent or zero-count entries.
    void release(int id) {
        // BEGIN TODO B4
        auto it = entries.find(id);
        if (it == entries.end() || it->second.pins == 0) throw std::logic_error("unowned release");
        --it->second.pins;
        // END TODO B4
    }
};
struct Predictor {
    std::map<int, std::map<int, double>> counts;
    int previous = -1;
    // B5: exponentially discount the whole transition table then observe one transition.
    // predict budget is bytes of overlap, not number of experts. Never inspect future trace.
    // Test: b_predictor (cold start, confidence threshold, phase change, byte budget).
    void observe(int expert) {
        // BEGIN TODO B5
        for (auto &[from, row] : counts)
            for (auto &[to, count] : row) count *= 0.9;
        if (previous >= 0) counts[previous][expert] += 1;
        previous = expert;
        // END TODO B5
    }
    // B6: return confidence-ranked expert IDs fitting the overlap byte budget.
    std::vector<int> predict(const std::vector<size_t> &bytes, size_t budget,
                             double min_probability) const {
        // BEGIN TODO B6
        auto it = counts.find(previous);
        if (it == counts.end()) return {};
        double total = 0;
        for (auto [expert, n] : it->second) total += n;
        std::vector<std::pair<double, int>> ranked;
        for (auto [expert, n] : it->second)
            if (n / total >= min_probability) ranked.emplace_back(-n / total, expert);
        std::sort(ranked.begin(), ranked.end());
        std::vector<int> result;
        for (auto [score, expert] : ranked) {
            if (bytes.at(expert) > budget) continue;
            budget -= bytes[expert];
            result.push_back(expert);
        }
        return result;
        // END TODO B6
    }
};
struct TraceResult {
    double finish;
    size_t transferred, wasted;
    int misses, blocked;
};
// B7: causal event replay. Demand blocks until its transfer completes; prediction runs
// only after observing this demand and receives compute-time*bandwidth as byte budget.
// Prefetch uses the SAME copy engine/cache as demand, so a wrong prediction can hurt.
// Test: b_replay checks the no-prefetch hand schedule and byte conservation.
inline TraceResult replay(const std::vector<int> &trace, const std::vector<size_t> &sizes,
                          size_t capacity, double bandwidth, double compute, bool prefetch) {
    // BEGIN TODO B7
    Cache cache(capacity, bandwidth);
    Predictor predictor;
    double time = 0;
    int misses = 0, blocked = 0;
    for (int expert : trace) {
        auto status = cache.request(expert, sizes.at(expert), time);
        while (status == Access::Blocked) {
            if (cache.link_free <= time) throw std::logic_error("expert larger than capacity");
            ++blocked;
            time = cache.link_free;
            cache.advance(time);
            status = cache.request(expert, sizes.at(expert), time);
        }
        if (status != Access::Hit) {
            ++misses;
            time = cache.entries.at(expert).ready;
            cache.advance(time);
        }
        cache.pin(expert);
        predictor.observe(expert);
        if (prefetch) {
            auto candidates = predictor.predict(sizes, size_t(compute * bandwidth), .3);
            for (int next : candidates) cache.request(next, sizes.at(next), time, true);
        }
        time += compute;
        cache.advance(time);
        cache.release(expert);
    }
    return {time, cache.transferred, cache.wasted, misses, blocked};
    // END TODO B7
}
// B8: clairvoyant Belady lower bound for equal-sized experts. On a miss, evict
// the resident expert whose NEXT use is farthest (never used again = infinity).
// This function is an explicitly labeled oracle, never fed to online replay.
// Test: b_oracle checks a hand trace and lower-bound inequality against demand LRU.
inline int oracle_misses(const std::vector<int> &trace, size_t slots) {
    // BEGIN TODO B8
    if (slots == 0) return int(trace.size());
    std::vector<int> resident;
    int misses = 0;
    for (size_t i = 0; i < trace.size(); ++i) {
        int current = trace[i];
        if (std::find(resident.begin(), resident.end(), current) != resident.end()) continue;
        ++misses;
        if (resident.size() == slots) {
            size_t victim = 0, farthest = 0;
            for (size_t j = 0; j < resident.size(); ++j) {
                auto next = std::find(trace.begin() + i + 1, trace.end(), resident[j]);
                size_t distance = size_t(next - trace.begin());
                if (distance >= farthest) {
                    farthest = distance;
                    victim = j;
                }
            }
            resident.erase(resident.begin() + victim);
        }
        resident.push_back(current);
    }
    return misses;
    // END TODO B8
}
struct Expert {
    Quantized up, down;
};
// C1: a REAL two-layer MLP expert: down(ReLU(up(x))). INT8 matrices dequantize in the dot.
// x:[D], up:[H,D], down:[D,H], return [D]. Test: c_expert vs independent FP32 oracle.
inline std::vector<float> evaluate(const Expert &e, const std::vector<float> &x) {
    // BEGIN TODO C1
    if (int(x.size()) != e.up.cols || e.down.cols != e.up.rows)
        throw std::invalid_argument("MLP shape");
    std::vector<float> hidden(e.up.rows), out(e.down.rows);
    for (int h = 0; h < e.up.rows; ++h) {
        float sum = 0;
        for (int d = 0; d < e.up.cols; ++d) sum += float(e.up.q[h * e.up.cols + d]) * x[d];
        hidden[h] = std::max(0.f, sum * e.up.scale[h]);
    }
    for (int d = 0; d < e.down.rows; ++d) {
        float sum = 0;
        for (int h = 0; h < e.down.cols; ++h)
            sum += float(e.down.q[d * e.down.cols + h]) * hidden[h];
        out[d] = sum * e.down.scale[d];
    }
    return out;
    // END TODO C1
}
// C2: gather each dispatched token, execute the corresponding expert and combine.
// Empty experts must launch no work. Test: c_forward with duplicates and unused experts.
inline std::vector<float> forward(const std::vector<Expert> &experts, const Dispatch &d,
                                  const std::vector<float> &x, int tokens, int width) {
    // BEGIN TODO C2
    auto packed = pack(d, x, width, 4);
    std::vector<float> ys(d.ordered.size() * width);
    for (size_t row = 0; row < packed.route_index.size(); ++row) {
        int route = packed.route_index[row];
        if (route < 0) continue;
        std::vector<float> input(packed.values.begin() + row * width,
                                 packed.values.begin() + (row + 1) * width);
        auto out = evaluate(experts[packed.expert[row]], input);
        std::copy(out.begin(), out.end(), ys.begin() + route * width);
    }
    return combine(d, ys, tokens, width);
    // END TODO C2
}
// C5: execute a real residual MoE block, with a learned dense gate [experts,width].
// Route only from the current x. Output x+MoE(x), plus observed route statistics.
// Test: c_block compares the full gate->topk->MLP->residual path to a hand oracle.
inline std::vector<float> block(const std::vector<Expert> &experts, const std::vector<float> &gate,
                                const std::vector<float> &x, int tokens, int width, int topk) {
    // BEGIN TODO C5
    int count = int(experts.size());
    std::vector<float> logits(tokens * count, 0);
    for (int t = 0; t < tokens; ++t) {
        for (int e = 0; e < count; ++e) {
            double sum = 0;
            for (int c = 0; c < width; ++c) sum += double(x[t * width + c]) * gate[e * width + c];
            logits[t * count + e] = float(sum);
        }
    }
    auto routes = route_topk(logits, tokens, count, topk);
    auto layout = dispatch(routes, tokens, count);
    auto y = forward(experts, layout, x, tokens, width);
    for (size_t i = 0; i < y.size(); ++i) y[i] += x[i];
    return y;
    // END TODO C5
}
} // namespace moe
