#pragma once
#include "kv_student.hpp"
#include <utility>

namespace kv {
// Return a physically compact row-major allocation. IDs must be strictly increasing;
// reject bad IDs before copying. Empty selection is legal; cols must be positive.
inline std::vector<float> gather_rows(const std::vector<float> &x, size_t cols,
                                      const std::vector<size_t> &ids) {
    // BEGIN TODO A3
    throw std::logic_error("Complete A3");
    // END TODO A3
}

struct ErrorStats {
    double relative_l2, max_abs, cosine;
};
// Equal nonempty finite vectors. Define cosine=1 for two zero vectors, =0 if only
// one is zero. Accumulate in double; use a 1e-20 denominator for relative L2.
inline ErrorStats errors(const std::vector<float> &a, const std::vector<float> &b) {
    // BEGIN TODO A4
    throw std::logic_error("Complete A4");
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
        throw std::logic_error("Complete B2");
        // END TODO B2
    }
    // Debug oracle only: concatenate decoded blocks with exact residual rows.
    std::pair<std::vector<float>, std::vector<float>> materialize() const {
        // BEGIN TODO B3
        throw std::logic_error("Complete B3");
        // END TODO B3
    }
    // Logical payload: packed bytes+scale+offset and FP32 residuals. Separately
    // report vector capacity/allocator overhead in the report, never call this RSS.
    size_t bytes() const {
        // BEGIN TODO B4
        throw std::logic_error("Complete B4");
        // END TODO B4
    }
    // Online softmax across blocks: m'=max(m,s), l'=l*exp(m-m')+exp(s-m').
    // Rescale the running weighted V sum too. Decode at most one block at a time;
    // do not call materialize(). Empty cache is invalid, including zero residual.
    std::vector<float> attend(const std::vector<float> &q) const {
        // BEGIN TODO B5
        throw std::logic_error("Complete B5");
        // END TODO B5
    }
};
} // namespace kv
