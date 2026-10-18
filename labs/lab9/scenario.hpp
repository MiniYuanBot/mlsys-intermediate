// Provided: simulated clusters and workloads shared by bench.cpp and the R replan step.
// Every number produced from these scenarios is MODELED, NOT MEASURED: spec-sheet peaks
// (data/gpus.csv) times an efficiency that was either assumed or fitted on the RTX 4060.
#pragma once
#include <cstdio>

#include "replan_student.hpp"

namespace hs {

// Placeholder efficiencies used by the A/B/C benchmarks until stage R refits them on a real
// GPU. They are assumptions, chosen inside the range typical for hand-written kernels.
inline Efficiency assumed_efficiency() {
    return {0.55, 0.80, 6e-6};
}

struct NodeSpec {
    std::string gpu;
    int count = 1;
};

// One machine per GPU (no tensor parallelism). rate_per_layer = decode tokens/s of ONE layer
// at batch `batch` and context `context`; max_layers from A4 with the KV of `batch` requests
// of `context + output` tokens and the activation peak of `batch` tokens. Links move one
// hidden-state vector (h * 2 bytes) per token: tokens/s = bandwidth / (2h).
struct ClusterInputs {
    ModelConfig model;
    std::vector<NodeSpec> nodes;
    int batch = 16;
    int context = 1024;
    int output = 256;
    double reserve_bytes = 1.5 * 1073741824.0;
    double network_gbit = 10.0; // between machines and to the coordinator
    double slow_gbit = 0.0;     // > 0: links between different GPU types use this
};

inline Cluster make_cluster(const std::vector<GpuSpec> &gpus, const ClusterInputs &in,
                            const std::function<Efficiency(const GpuSpec &)> &eff) {
    Cluster c;
    for (const auto &ns : in.nodes) {
        GpuSpec g = find_gpu(gpus, ns.gpu);
        double t = layer_time(in.model, decode_batch(in.batch, in.context), g, eff(g));
        MemoryFit one = memory_fit(in.model, 1, double(in.batch) * (in.context + in.output),
                                   in.batch, g.mem_bytes(), in.reserve_bytes, false);
        int max_layers = 0;
        for (int k = 1; k <= in.model.layers; ++k)
            if (memory_fit(in.model, k, double(in.batch) * (in.context + in.output), in.batch,
                           g.mem_bytes(), in.reserve_bytes, false)
                    .fits)
                max_layers = k;
        (void)one;
        for (int r = 0; r < ns.count; ++r)
            c.nodes.push_back(
                {ns.gpu + "#" + std::to_string(c.nodes.size()), ns.gpu, in.batch / t, max_layers});
    }
    int n = c.size();
    double per_token = 2.0 * in.model.hidden;
    double fast = in.network_gbit * 1e9 / 8 / per_token;
    double slow = in.slow_gbit > 0 ? in.slow_gbit * 1e9 / 8 / per_token : fast;
    c.link.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) c.link[i][j] = c.nodes[i].gpu == c.nodes[j].gpu ? fast : slow;
    c.from_source.assign(n, fast);
    c.to_sink.assign(n, fast);
    return c;
}

inline double price_of(const GpuSpec &g) {
    return g.price_aws > 0 ? g.price_aws : g.price_runpod;
}

inline double cluster_price(const std::vector<GpuSpec> &gpus, const Cluster &c,
                            const Placement &p) {
    double cost = 0;
    for (int i = 0; i < c.size(); ++i)
        if (p[i].used()) cost += price_of(find_gpu(gpus, c.nodes[i].gpu));
    return cost;
}

// Multi-model scenario for stage C (and R3): four Qwen3 sizes with different popularity.
struct MultiModelScenario {
    std::vector<ModelConfig> models{qwen3_0p6b(), qwen3_1p7b(), qwen3_4b(), qwen3_8b()};
    std::vector<Workload> work{{512, 128}, {512, 128}, {1024, 256}, {1024, 256}};
    std::vector<Slo> slo{{0.5, 0.05}, {0.5, 0.05}, {1.0, 0.08}, {1.5, 0.10}};
    std::vector<double> demand{6.0, 3.0, 1.2, 0.4}; // requests/s (power-law-like popularity)
    std::vector<std::string> gpu_names{"T4", "L4", "A10G~A10", "L40S", "A100-40G"};
};

inline std::vector<GpuSpec> pick_gpus(const std::vector<GpuSpec> &all,
                                      const std::vector<std::string> &names) {
    std::vector<GpuSpec> out;
    for (const auto &n : names) out.push_back(find_gpu(all, n));
    return out;
}

inline TemplateInputs template_inputs(const MultiModelScenario &sc,
                                      const std::vector<GpuSpec> &gpus,
                                      const std::function<Efficiency(const GpuSpec &)> &eff) {
    TemplateInputs in;
    in.models = sc.models, in.work = sc.work, in.slo = sc.slo, in.gpus = gpus;
    in.price = [](const GpuSpec &g) { return g.price_aws; };
    in.eff = eff;
    return in;
}

// Effective $/hour to meet `demand` with a fixed plan: if the plan's capacity at the current
// efficiencies falls short, its mix is scaled up fractionally until every model is covered.
// kInf if some model with demand has no capacity at all in the plan.
inline double scaled_plan_cost(const std::vector<Template> &tpl_now, const Assignment &a,
                               const std::vector<double> &demand) {
    if (!a.feasible) return kInf;
    double cost = 0, factor = 1;
    for (size_t t = 0; t < tpl_now.size(); ++t) cost += a.count[t] * tpl_now[t].price_hr;
    for (size_t m = 0; m < demand.size(); ++m) {
        if (demand[m] <= 0) continue;
        double cap = 0;
        for (size_t t = 0; t < tpl_now.size(); ++t) cap += a.count[t] * tpl_now[t].capacity[m];
        if (cap <= 0) return kInf;
        factor = std::max(factor, demand[m] / cap);
    }
    return cost * factor;
}

// Recompute the capacity of the SAME templates (same GPU, same model set, equal shares)
// under new efficiencies, with the rule of make_templates. A hosted model whose SLO can no
// longer be met gets capacity 0 (the plan may then need more instances: scaled_plan_cost).
inline std::vector<Template>
recompute_templates(const std::vector<Template> &base, const MultiModelScenario &sc,
                    const std::vector<GpuSpec> &gpus,
                    const std::function<Efficiency(const GpuSpec &)> &eff) {
    TemplateInputs in = template_inputs(sc, gpus, eff);
    std::vector<Template> out = base;
    for (auto &t : out) {
        GpuSpec g = find_gpu(gpus, t.gpu);
        double weights = 0, act = 0;
        for (int m : t.models) {
            weights +=
                memory_fit(sc.models[m], sc.models[m].layers, 0, 0, kInf, 0, true).need.weights;
            act = std::max(
                act,
                memory_fit(sc.models[m], 1, 0, sc.work[m].prompt, kInf, 0, false).need.activations);
        }
        double free = g.mem_bytes() - in.reserve_bytes - weights - act;
        for (int m : t.models) {
            Capacity cap =
                slo_capacity(serving_model(sc.models[m], g, eff(g)), sc.work[m], sc.slo[m],
                             1.0 / t.models.size(), free / t.models.size(), in.max_batch);
            t.capacity[m] = cap.feasible ? cap.rate : 0.0;
            t.batch[m] = cap.batch;
        }
    }
    return out;
}

inline std::string plan_string(const std::vector<Template> &tpl, const Assignment &a) {
    std::string s;
    if (!a.feasible) return "infeasible";
    for (size_t t = 0; t < tpl.size(); ++t) {
        if (!a.count[t]) continue;
        if (!s.empty()) s += " + ";
        s += std::to_string(a.count[t]) + "x" + tpl[t].gpu + "{";
        for (size_t k = 0; k < tpl[t].models.size(); ++k)
            s += (k ? "," : "") + std::to_string(tpl[t].models[k]);
        s += "}";
    }
    return s;
}

} // namespace hs
