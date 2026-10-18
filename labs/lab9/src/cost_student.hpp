// Stage A: analytic cost model (FLOPs/bytes per layer, roofline, fitted efficiency, memory).
#pragma once
#include "types.hpp"

namespace hs {

// layer_params (provided): parameters of ONE decoder layer: q/k/v/o projections,
// gate/up/down of the SwiGLU MLP, two RMSNorm vectors (h each) and the Qwen3 per-head
// q_norm/k_norm vectors (d each). No biases.
inline double layer_params(const ModelConfig &c) {
    double h = c.hidden, q = static_cast<double>(c.q_dim()), kv = static_cast<double>(c.kv_dim());
    return h * q + 2 * h * kv + q * h + 3 * h * c.intermediate + 2 * h + 2.0 * c.head_dim;
}

// ============================================================================ A1
// layer_cost: FLOPs and bytes of one layer forward for `b` (see Batch in types.hpp).
//   FLOPs  = 2 * MACs of every matmul: projections, MLP, QK^T and PV. GQA shares K/V across
//            query heads but every query head still does its own QK^T and PV.
//            New token i (0-based) of a request attends to context + i + 1 positions.
//   bytes  = weights (layer_params, each parameter once) + cached KV read once + new KV
//            written once + layer input read + layer output written; element size is
//            cfg.bytes_per_elem.
//   Ignored on purpose: norm/softmax/activation FLOPs and intermediate activations.
// Invariants: requests >= 1, new_tokens >= 1, context >= 0, else std::invalid_argument.
// Oracle: test.cpp runs the tiny model as explicit loops and counts every MAC and byte.
inline LayerCost layer_cost(const ModelConfig &c, const Batch &b) {
    // BEGIN TODO A1
    throw std::logic_error("Complete A1");
    // END TODO A1
}

// ============================================================================ A2
// roofline_time = t0 + max(F / (eta_c * P), B / (eta_m * W)), P = gpu.peak_flops(),
// W = gpu.peak_bw(). Seconds. eta_c, eta_m must be > 0 (std::invalid_argument otherwise);
// flops/bytes must be >= 0. A kernel with F = B = 0 costs exactly t0.
inline double roofline_time(double flops, double bytes, const GpuSpec &gpu, const Efficiency &eff) {
    // BEGIN TODO A2
    throw std::logic_error("Complete A2");
    // END TODO A2
}

// ============================================================================ A3
// fit_efficiency: fit (t0, eta_c, eta_m) of the roofline model to measured samples.
//   Model per sample: t = t0 + u_c * F/P   if the sample is compute-bound,
//                     t = t0 + u_m * B/W   otherwise,          u_c = 1/eta_c, u_m = 1/eta_m.
//   Objective: weighted least squares on RELATIVE error, sum_i ((pred_i - t_i) / t_i)^2,
//   i.e. weights 1/t_i^2, solved in closed form from the 3x3 normal equations
//   (X^T W X) beta = X^T W y with your own Gaussian elimination (partial pivoting).
//   Regimes: label sample i compute-bound iff F/(eta_c P) >= B/(eta_m W) using the CURRENT
//   estimate (start with eta = 1, t0 = 0), refit, relabel; stop when labels do not change
//   (at most 50 rounds).
// Errors (std::invalid_argument): fewer than 3 samples, seconds <= 0, a regime with no
//   samples, a singular system, or a non-physical fit (u_c <= 0 or u_m <= 0).
// Oracles: exact recovery on noise-free data generated from known coefficients whose ridge
//   differs from the spec ridge; zero finite-difference gradient of the test's own objective.
inline Efficiency fit_efficiency(const std::vector<Sample> &samples, const GpuSpec &gpu) {
    // BEGIN TODO A3
    throw std::logic_error("Complete A3");
    // END TODO A3
}

// ============================================================================ A4
// memory_fit: does a GPU with `capacity_bytes` hold `layers` decoder layers plus KV for
// `kv_tokens` tokens and the activation peak of `act_tokens` tokens, after `reserve_bytes`
// (CUDA context, allocator slack)?
//   need.weights     = e * (layers * layer_params + (with_embedding ? embed : 0)),
//                      embed = vocab * h * (tied ? 1 : 2) + h   (embedding/LM head + final norm)
//   need.kv          = layers * kv_tokens * 2 * kv_dim * e      (K and V)
//   need.activations = act_tokens * (2h + 2I) * e               (peak of one layer forward)
//   fits             = reserve + need.total() <= capacity
//   max_kv_tokens    = largest integer N with reserve + need(N).total() <= capacity,
//                      -1 if even N = 0 does not fit. An exact fit must return N itself.
// layers < 1 or negative kv_tokens/act_tokens -> std::invalid_argument.
inline MemoryFit memory_fit(const ModelConfig &c, int layers, double kv_tokens, double act_tokens,
                            double capacity_bytes, double reserve_bytes, bool with_embedding) {
    // BEGIN TODO A4
    throw std::logic_error("Complete A4");
    // END TODO A4
}

// ============================================================================ A5
// heldout_error: fit on `train` with fit_efficiency, predict every `test` sample with
//   roofline_time and report |pred - t| / t statistics: mean, median (average of the two
//   middle values for even n), max, n, and the fitted coefficients. Empty test set ->
//   std::invalid_argument. The test set must never influence the fit.
inline ErrorStats heldout_error(const std::vector<Sample> &train, const std::vector<Sample> &test,
                                const GpuSpec &gpu) {
    // BEGIN TODO A5
    throw std::logic_error("Complete A5");
    // END TODO A5
}

// ---------------------------------------------------------------- provided helpers
// LM head (untied or tied, it is one GEMM/GEMV over the vocabulary) for `tokens` tokens.
inline LayerCost lm_head_cost(const ModelConfig &c, double tokens) {
    LayerCost out;
    double e = c.bytes_per_elem;
    out.flops = 2.0 * tokens * double(c.vocab) * c.hidden;
    out.weight_bytes = e * double(c.vocab) * c.hidden;
    out.act_bytes = e * tokens * c.hidden + 4.0 * tokens * double(c.vocab);
    return out;
}

inline double layer_time(const ModelConfig &c, const Batch &b, const GpuSpec &g,
                         const Efficiency &e) {
    LayerCost lc = layer_cost(c, b);
    return roofline_time(lc.flops, lc.bytes(), g, e);
}

// Whole-model latency functions used by stages B, C and R (modeled, not measured).
inline ServingModel serving_model(const ModelConfig &c, const GpuSpec &g, const Efficiency &e) {
    ServingModel m;
    m.name = c.name + "@" + g.name;
    m.prefill_s = [c, g, e](int prompt) {
        LayerCost head = lm_head_cost(c, 1);
        return c.layers * layer_time(c, prefill_batch(1, prompt), g, e) +
               roofline_time(head.flops, head.bytes(), g, e);
    };
    m.step_s = [c, g, e](int batch, double context) {
        int ctx = static_cast<int>(std::lround(context));
        LayerCost head = lm_head_cost(c, batch);
        return c.layers * layer_time(c, decode_batch(batch, ctx), g, e) +
               roofline_time(head.flops, head.bytes(), g, e);
    };
    m.kv_bytes_per_token = double(c.bytes_per_elem) * c.layers * 2.0 * double(c.kv_dim());
    m.weight_bytes = memory_fit(c, c.layers, 0, 0, kInf, 0, true).need.weights;
    return m;
}

} // namespace hs
