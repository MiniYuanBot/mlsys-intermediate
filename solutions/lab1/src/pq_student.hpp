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
        if (train.size() % dim || train.size() / dim < centroids || !iterations)
            throw std::invalid_argument("PQ training shape/iterations");
        for (float x : train)
            if (!std::isfinite(x)) throw std::invalid_argument("PQ finite");
        size_t n = train.size() / dim;
        book.resize(parts * centroids * subdim);
        objective.clear();
        for (size_t p = 0; p < parts; ++p)
            for (size_t c = 0; c < centroids; ++c)
                for (size_t j = 0; j < subdim; ++j)
                    book[offset(p, c, j)] = train[(c * n / centroids) * dim + p * subdim + j];
        for (size_t it = 0; it < iterations; ++it) {
            std::vector<double> sums(book.size(), 0);
            std::vector<size_t> count(parts * centroids, 0), assignments(n * parts);
            for (size_t i = 0; i < n; ++i)
                for (size_t p = 0; p < parts; ++p) {
                    size_t best = 0;
                    double distance = INFINITY;
                    for (size_t c = 0; c < centroids; ++c) {
                        double value = 0;
                        for (size_t j = 0; j < subdim; ++j) {
                            double delta = train[i * dim + p * subdim + j] - book[offset(p, c, j)];
                            value += delta * delta;
                        }
                        if (value < distance) {
                            distance = value;
                            best = c;
                        }
                    }
                    assignments[i * parts + p] = best;
                    ++count[p * centroids + best];
                    for (size_t j = 0; j < subdim; ++j)
                        sums[offset(p, best, j)] += train[i * dim + p * subdim + j];
                }
            for (size_t p = 0; p < parts; ++p)
                for (size_t c = 0; c < centroids; ++c)
                    if (count[p * centroids + c])
                        for (size_t j = 0; j < subdim; ++j)
                            book[offset(p, c, j)] =
                                float(sums[offset(p, c, j)] / count[p * centroids + c]);
            double sse = 0;
            for (size_t i = 0; i < n; ++i)
                for (size_t p = 0; p < parts; ++p)
                    for (size_t j = 0; j < subdim; ++j) {
                        double delta = train[i * dim + p * subdim + j] -
                                       book[offset(p, assignments[i * parts + p], j)];
                        sse += delta * delta;
                    }
            objective.push_back(sse);
        }
        codes.clear(); // A new codebook invalidates old codes.
        // END TODO C1
    }
    // Encode rows against the frozen book, nearest Euclidean center per subspace.
    // Ties choose the smallest center ID; do not retrain on held-out queries.
    void encode(const std::vector<float> &keys) {
        // BEGIN TODO C2
        if (book.empty() || keys.size() % dim)
            throw std::invalid_argument("PQ encode shape/untrained");
        for (float x : keys)
            if (!std::isfinite(x)) throw std::invalid_argument("PQ finite");
        std::vector<uint16_t> encoded(keys.size() / dim * parts);
        for (size_t i = 0; i < keys.size() / dim; ++i)
            for (size_t p = 0; p < parts; ++p) {
                double distance = INFINITY;
                size_t best = 0;
                for (size_t c = 0; c < centroids; ++c) {
                    double value = 0;
                    for (size_t j = 0; j < subdim; ++j) {
                        double delta = keys[i * dim + p * subdim + j] - book[offset(p, c, j)];
                        value += delta * delta;
                    }
                    if (value < distance) {
                        distance = value;
                        best = c;
                    }
                }
                encoded[i * parts + p] = uint16_t(best);
            }
        codes = std::move(encoded);
        // END TODO C2
    }
    // ADC for inner products: table[p,c]=dot(q_part,center); sum code lookups.
    // Attention ranking uses inner product, not Euclidean nearest neighbors.
    std::vector<float> scores(const std::vector<float> &q) const {
        // BEGIN TODO C3
        if (q.size() != dim || book.empty()) throw std::invalid_argument("PQ query");
        for (float x : q)
            if (!std::isfinite(x)) throw std::invalid_argument("PQ finite");
        std::vector<double> lookup(parts * centroids, 0);
        for (size_t p = 0; p < parts; ++p)
            for (size_t c = 0; c < centroids; ++c)
                for (size_t j = 0; j < subdim; ++j)
                    lookup[p * centroids + c] += double(q[p * subdim + j]) * book[offset(p, c, j)];
        std::vector<float> result(codes.size() / parts, 0);
        for (size_t i = 0; i < result.size(); ++i) {
            double value = 0;
            for (size_t p = 0; p < parts; ++p)
                value += lookup[p * centroids + codes[i * parts + p]];
            result[i] = float(value);
        }
        return result;
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
    if (k.size() != v.size() || k.size() % index.dim ||
        k.size() / index.dim != index.codes.size() / index.parts)
        throw std::invalid_argument("retrieval shape");
    auto ids = select_tokens(index.scores(q), budget, sink, recent);
    if (ids.empty()) throw std::invalid_argument("empty retrieval");
    auto selected_k = gather_rows(k, index.dim, ids), selected_v = gather_rows(v, index.dim, ids);
    std::vector<size_t> local(ids.size());
    std::iota(local.begin(), local.end(), 0);
    auto out = attention(q, selected_k, selected_v, local);
    size_t bytes = (selected_k.size() + selected_v.size()) * sizeof(float);
    return {std::move(ids), std::move(out), bytes};
    // END TODO C4
}
} // namespace kv
