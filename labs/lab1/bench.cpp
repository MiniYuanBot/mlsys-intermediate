#include "pq_student.hpp"
#include <chrono>
#include <iostream>
#include <random>
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
int main() {
    std::cout << "seed,tokens,dim,keep,cache_bytes,pq_bytes,fetch_bytes,fp32_kv_bytes,append_us,"
                 "stream_us,pq_fit_us,pq_query_us,recall,stream_error,retrieval_error\n";
    for (unsigned seed : {17, 29, 43})
        for (size_t n : {128, 512}) {
            size_t d = 32;
            std::mt19937 rng(seed);
            std::normal_distribution<float> normal;
            std::vector<float> k(n * d), v(n * d), q(d);
            for (auto *a : {&k, &v, &q})
                for (float &x : *a) x = normal(rng);
            std::vector<size_t> all(n);
            std::iota(all.begin(), all.end(), 0);
            auto truth = kv::attention(q, k, v, all);
            kv::ResidualCache cache(d, 32, 16, 16);
            auto t = Clock::now();
            for (size_t i = 0; i < n; ++i)
                cache.append({k.begin() + i * d, k.begin() + (i + 1) * d},
                             {v.begin() + i * d, v.begin() + (i + 1) * d});
            double append = elapsed(t) / n;
            t = Clock::now();
            std::vector<float> stream;
            for (int rep = 0; rep < 20; ++rep) stream = cache.attend(q);
            double stream_us = elapsed(t) / 20;
            kv::ProductIndex index(d, 4, 16);
            t = Clock::now();
            index.fit(k, 8);
            index.encode(k);
            double fit = elapsed(t);
            std::vector<float> exact(n, 0);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < d; ++j) exact[i] += q[j] * k[i * d + j];
            for (size_t budget : {n / 4, n / 2, n}) {
                t = Clock::now();
                kv::RetrievalResult result;
                for (int rep = 0; rep < 20; ++rep)
                    result = kv::retrieve(index, q, k, v, budget, 2, 4);
                double query = elapsed(t) / 20;
                auto ideal = kv::select_tokens(exact, budget, 2, 4);
                size_t hits = 0;
                for (size_t id : result.ids)
                    hits += std::binary_search(ideal.begin(), ideal.end(), id);
                std::cout << seed << ',' << n << ',' << d << ',' << budget << ',' << cache.bytes()
                          << ',' << index.bytes() << ',' << result.fetched_bytes << ',' << 8 * n * d
                          << ',' << append << ',' << stream_us << ',' << fit << ',' << query << ','
                          << double(hits) / budget << ',' << kv::errors(truth, stream).relative_l2
                          << ',' << kv::errors(truth, result.output).relative_l2 << '\n';
            }
        }
}
