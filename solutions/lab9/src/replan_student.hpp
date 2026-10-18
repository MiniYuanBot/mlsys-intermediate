// Stage R (CPU half): turn measured microbenchmarks into roofline samples, refit eta, and
// check whether plan rankings survive eta uncertainty. Runs offline from a profile CSV.
#pragma once
#include "coloc_student.hpp"

namespace hs {

// ============================================================================ R5
// profile_to_samples: FLOPs and bytes of each measured kernel (seconds = median_ms / 1000).
//   copy        m = bytes of the source buffer:           F = 0,        B = 2m (read + write)
//   gemv        y[m] = W[m x k] x[k], FP16 W/x, FP32 y:   F = 2mk,      B = 2mk + 2k + 4m
//   gemm, gemm_cublas
//               C[m x n] = A[m x k] B[k x n], FP16 A/B, FP32 C:
//                                                         F = 2mnk,     B = 2(mk + kn) + 4mn
//   label = kernel name + "_" + m + "x" + n + "x" + k. Unknown kernel or median <= 0 ->
//   std::invalid_argument. Each operand is counted once: this is the compulsory traffic, a
//   lower bound on real DRAM traffic (tiles of A/B are re-read when they miss in L2).
inline std::vector<Sample> profile_to_samples(const std::vector<ProfileRow> &rows) {
    // BEGIN TODO R5
    std::vector<Sample> out;
    for (const auto &r : rows) {
        if (!(r.median_ms > 0)) throw std::invalid_argument("profile_to_samples: median <= 0");
        double m = double(r.m), n = double(r.n), k = double(r.k);
        Sample s;
        s.label = r.kernel + "_" + std::to_string(r.m) + "x" + std::to_string(r.n) + "x" +
                  std::to_string(r.k);
        s.seconds = r.median_ms / 1000.0;
        if (r.kernel == "copy") {
            s.flops = 0, s.bytes = 2 * m;
        } else if (r.kernel == "gemv") {
            s.flops = 2 * m * k, s.bytes = 2 * m * k + 2 * k + 4 * m;
        } else if (r.kernel == "gemm" || r.kernel == "gemm_cublas") {
            s.flops = 2 * m * n * k, s.bytes = 2 * (m * k + k * n) + 4 * m * n;
        } else {
            throw std::invalid_argument("profile_to_samples: unknown kernel " + r.kernel);
        }
        out.push_back(s);
    }
    return out;
    // END TODO R5
}

struct SweepRow {
    double scale = 1.0;
    std::vector<double> cost;  // cost[plan] at this scale (kInf = plan infeasible)
    std::vector<int> ranking;  // plan indices, best first
    bool top1_changed = false; // ranking[0] differs from the nominal ranking[0]
    int kendall = 0;           // discordant plan pairs versus the nominal ranking
};

// ============================================================================ R6
// sensitivity_sweep: evaluate `cost(plan, scale)` for every plan and scale, rank the plans
//   (ascending cost, ties by lower index, kInf last), and compare each ranking with the
//   nominal one (the row whose scale == 1.0): top1_changed, and the Kendall distance =
//   number of plan pairs (a, b) whose relative order differs between the two rankings.
//   Rows are returned in the order of `scales`. No scale equal to 1.0, or n_plans < 1 ->
//   std::invalid_argument.
// Oracle: test.cpp counts discordant pairs by brute force on cost functions with known
//   crossings.
inline std::vector<SweepRow> sensitivity_sweep(int n_plans, const std::vector<double> &scales,
                                               const std::function<double(int, double)> &cost) {
    // BEGIN TODO R6
    if (n_plans < 1) throw std::invalid_argument("sensitivity_sweep: no plans");
    auto nominal_it = std::find(scales.begin(), scales.end(), 1.0);
    if (nominal_it == scales.end())
        throw std::invalid_argument("sensitivity_sweep: need scale 1.0");
    std::vector<SweepRow> rows;
    for (double s : scales) {
        SweepRow row;
        row.scale = s;
        for (int p = 0; p < n_plans; ++p) row.cost.push_back(cost(p, s));
        row.ranking.resize(n_plans);
        std::iota(row.ranking.begin(), row.ranking.end(), 0);
        std::stable_sort(row.ranking.begin(), row.ranking.end(),
                         [&](int a, int b) { return row.cost[a] < row.cost[b]; });
        rows.push_back(row);
    }
    const SweepRow &nom = rows[nominal_it - scales.begin()];
    std::vector<int> nominal_pos(n_plans);
    for (int i = 0; i < n_plans; ++i) nominal_pos[nom.ranking[i]] = i;
    for (auto &row : rows) {
        row.top1_changed = row.ranking[0] != nom.ranking[0];
        row.kendall = 0;
        for (int i = 0; i < n_plans; ++i)
            for (int j = i + 1; j < n_plans; ++j)
                row.kendall += nominal_pos[row.ranking[i]] > nominal_pos[row.ranking[j]];
    }
    return rows;
    // END TODO R6
}

} // namespace hs
