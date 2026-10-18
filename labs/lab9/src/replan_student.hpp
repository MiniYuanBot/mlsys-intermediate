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
    throw std::logic_error("Complete R5");
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
    throw std::logic_error("Complete R6");
    // END TODO R6
}

} // namespace hs
