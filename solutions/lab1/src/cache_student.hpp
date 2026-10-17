#pragma once
#include "kv_student.hpp"
#include <utility>

namespace kv {
// Return a physically compact row-major allocation. IDs must be strictly increasing;
// reject bad IDs before copying. Empty selection is legal; cols must be positive.
inline std::vector<float> gather_rows(const std::vector<float> &x, size_t cols,
                                      const std::vector<size_t> &ids) {
    // BEGIN TODO A3
    if (!cols || x.size() % cols) throw std::invalid_argument("gather shape");
    for (size_t j = 0; j < ids.size(); ++j) {
        if (ids[j] >= x.size() / cols || (j && ids[j] <= ids[j - 1]))
            throw std::invalid_argument("ordered unique row ids");
    }
    std::vector<float> out;
    out.reserve(ids.size() * cols);
    for (size_t id : ids) out.insert(out.end(), x.begin() + id * cols, x.begin() + (id + 1) * cols);
    return out;
    // END TODO A3
}

struct ErrorStats {
    double relative_l2, max_abs, cosine;
};
// Equal nonempty finite vectors. Define cosine=1 for two zero vectors, =0 if only
// one is zero. Accumulate in double; use a 1e-20 denominator for relative L2.
inline ErrorStats errors(const std::vector<float> &a, const std::vector<float> &b) {
    // BEGIN TODO A4
    if (a.empty() || a.size() != b.size()) throw std::invalid_argument("error shape");
    double aa = 0, bb = 0, ab = 0, sq = 0, mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::invalid_argument("finite");
        double d = double(a[i]) - b[i];
        aa += double(a[i]) * a[i];
        bb += double(b[i]) * b[i];
        ab += double(a[i]) * b[i];
        sq += d * d;
        mx = std::max(mx, std::abs(d));
    }
    double cosine = aa && bb ? ab / std::sqrt(aa * bb) : (aa == bb ? 1 : 0);
    return {std::sqrt(sq / std::max(aa, 1e-20)), mx, cosine};
    // END TODO A4
}

struct Chunk {
    Packed k, v;
};
class ResidualCache {
public:
    size_t dim, block, residual, group, tokens = 0;
    std::vector<Chunk> chunks;
    std::vector<float> recent_k, recent_v;
    ResidualCache(size_t d, size_t b, size_t r, size_t g)
        : dim(d), block(b), residual(r), group(g) {
        if (!d || !b || !g) throw std::invalid_argument("cache configuration");
    }
    // Append one finite K/V row. Flush oldest block only at residual+block rows.
    // K groups run along tokens; V groups run along channels. Validation must not
    // mutate cache. After return recent rows < residual+block, order is preserved.
    void append(const std::vector<float> &k, const std::vector<float> &v) {
        // BEGIN TODO B2
        if (k.size() != dim || v.size() != dim) throw std::invalid_argument("append shape");
        for (const auto *row : {&k, &v})
            for (float x : *row)
                if (!std::isfinite(x)) throw std::invalid_argument("append finite");
        recent_k.insert(recent_k.end(), k.begin(), k.end());
        recent_v.insert(recent_v.end(), v.begin(), v.end());
        ++tokens;
        if (recent_k.size() / dim >= residual + block) {
            std::vector<float> a(recent_k.begin(), recent_k.begin() + block * dim);
            std::vector<float> b(recent_v.begin(), recent_v.begin() + block * dim);
            auto pk = quantize(a, block, dim, group, Axis::Channel);
            auto pv = quantize(b, block, dim, group, Axis::Token);
            chunks.push_back({std::move(pk), std::move(pv)});
            recent_k.erase(recent_k.begin(), recent_k.begin() + block * dim);
            recent_v.erase(recent_v.begin(), recent_v.begin() + block * dim);
        }
        // END TODO B2
    }
    // Debug oracle only: concatenate decoded blocks with exact residual rows.
    std::pair<std::vector<float>, std::vector<float>> materialize() const {
        // BEGIN TODO B3
        std::vector<float> k, v;
        k.reserve(tokens * dim);
        v.reserve(tokens * dim);
        for (const auto &chunk : chunks) {
            auto a = dequantize(chunk.k), b = dequantize(chunk.v);
            k.insert(k.end(), a.begin(), a.end());
            v.insert(v.end(), b.begin(), b.end());
        }
        k.insert(k.end(), recent_k.begin(), recent_k.end());
        v.insert(v.end(), recent_v.begin(), recent_v.end());
        return {std::move(k), std::move(v)};
        // END TODO B3
    }
    // Logical payload: packed bytes+scale+offset and FP32 residuals. Separately
    // report vector capacity/allocator overhead in the report, never call this RSS.
    size_t bytes() const {
        // BEGIN TODO B4
        size_t result = sizeof(float) * (recent_k.size() + recent_v.size());
        for (const auto &c : chunks) result += c.k.bytes() + c.v.bytes();
        return result;
        // END TODO B4
    }
    // Online softmax across blocks: m'=max(m,s), l'=l*exp(m-m')+exp(s-m').
    // Rescale the running weighted V sum too. Decode at most one block at a time;
    // do not call materialize(). Empty cache is invalid, including zero residual.
    std::vector<float> attend(const std::vector<float> &q) const {
        // BEGIN TODO B5
        if (q.size() != dim || !tokens) throw std::invalid_argument("query shape/empty");
        for (float x : q)
            if (!std::isfinite(x)) throw std::invalid_argument("query finite");
        double m = -INFINITY, z = 0;
        std::vector<double> acc(dim, 0);
        auto consume = [&](const std::vector<float> &k, const std::vector<float> &v) {
            for (size_t i = 0; i < k.size() / dim; ++i) {
                double score = 0;
                for (size_t j = 0; j < dim; ++j) score += double(q[j]) * k[i * dim + j];
                score /= std::sqrt(double(dim));
                double next = std::max(m, score);
                double old_weight = std::exp(m - next), weight = std::exp(score - next);
                z = z * old_weight + weight;
                for (size_t j = 0; j < dim; ++j)
                    acc[j] = acc[j] * old_weight + weight * v[i * dim + j];
                m = next;
            }
        };
        for (const auto &chunk : chunks) consume(dequantize(chunk.k), dequantize(chunk.v));
        consume(recent_k, recent_v);
        std::vector<float> out(dim);
        for (size_t j = 0; j < dim; ++j) out[j] = float(acc[j] / z);
        return out;
        // END TODO B5
    }
};
} // namespace kv
