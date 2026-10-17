#pragma once
#include "cache_student.hpp"

namespace kv {
struct ProductIndex {
    size_t dim, parts, centroids, subdim;
    std::vector<float> book;
    std::vector<uint16_t> codes;
    std::vector<double> objective;
    ProductIndex(size_t d, size_t m, size_t c)
        : dim(d), parts(m), centroids(c), subdim(m ? d / m : 0) {
        if (!d || !m || d % m || !c || c > 65536) throw std::invalid_argument("PQ configuration");
    }
    size_t offset(size_t part, size_t centroid, size_t j) const {
        return (part * centroids + centroid) * subdim + j;
    }
    // Independent subspace Lloyd clustering. Deterministic seeds row c*N/C.
    // Empty clusters keep their old centers. Record SSE AFTER each center update.
    // train is immutable; reject nonfinite data and fewer rows than centroids.
    void fit(const std::vector<float> &train, size_t iterations) {
        // BEGIN TODO C1
        throw std::logic_error("Complete C1");
        // END TODO C1
    }
    // Encode rows against the frozen book, nearest Euclidean center per subspace.
    // Ties choose the smallest center ID; do not retrain on held-out queries.
    void encode(const std::vector<float> &keys) {
        // BEGIN TODO C2
        throw std::logic_error("Complete C2");
        // END TODO C2
    }
    // ADC for inner products: table[p,c]=dot(q_part,center); sum code lookups.
    // Attention ranking uses inner product, not Euclidean nearest neighbors.
    std::vector<float> scores(const std::vector<float> &q) const {
        // BEGIN TODO C3
        throw std::logic_error("Complete C3");
        // END TODO C3
    }
    size_t bytes() const { return book.size() * sizeof(float) + codes.size() * sizeof(uint16_t); }
};
struct RetrievalResult {
    std::vector<size_t> ids;
    std::vector<float> output;
    size_t fetched_bytes;
};
// Query-derived ranking is legitimate retrieval at decode time. It is not a
// historical eviction score. Gather original K/V to model selective fetch cost.
inline RetrievalResult retrieve(const ProductIndex &index, const std::vector<float> &q,
                                const std::vector<float> &k, const std::vector<float> &v,
                                size_t budget, size_t sink, size_t recent) {
    // BEGIN TODO C4
    throw std::logic_error("Complete C4");
    // END TODO C4
}
} // namespace kv
