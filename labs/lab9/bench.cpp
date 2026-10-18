// lab9 benchmarks. Usage: bench <A|B|C|all> | bench R <profile.csv>
// Output: one "BENCH <part> key=value ..." row per configuration. Every row is MODELED, NOT
// MEASURED (spec sheet x efficiency, or simulation), except rows produced by profile.cu.
#include <chrono>
#include <cstdio>

#include "scenario.hpp"

using namespace hs;
static const std::string kGpus = "data/gpus.csv";

static std::string fmt(double x) {
    char b[32];
    std::snprintf(b, sizeof b, "%.3f", x);
    return b;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ------------------------------------------------------------------------------- A
static void bench_a() {
    auto gpus = load_gpus(kGpus);
    Efficiency e = assumed_efficiency();
    std::printf("NOTE A uses ASSUMED efficiencies eta_c=%.2f eta_m=%.2f t0=%.1e s until stage R "
                "refits them\n",
                e.eta_c, e.eta_m, e.t0);
    ModelConfig m = qwen3_1p7b();
    std::vector<std::pair<std::string, Batch>> batches = {
        {"prefill", prefill_batch(1, 128)},  {"prefill", prefill_batch(1, 512)},
        {"prefill", prefill_batch(1, 2048)}, {"decode", decode_batch(1, 1024)},
        {"decode", decode_batch(8, 1024)},   {"decode", decode_batch(32, 1024)},
        {"decode", decode_batch(128, 1024)}, {"decode", decode_batch(32, 8192)}};
    for (const auto &[phase, b] : batches) {
        LayerCost lc = layer_cost(m, b);
        for (const char *name : {"RTX4060L", "L4", "A100-40G", "H100-SXM"}) {
            GpuSpec g = find_gpu(gpus, name);
            double t = roofline_time(lc.flops, lc.bytes(), g, e);
            bool compute =
                lc.flops / (e.eta_c * g.peak_flops()) >= lc.bytes() / (e.eta_m * g.peak_bw());
            std::printf("BENCH A1A2 model=%s phase=%s R=%d q=%d c=%d gpu=%s flops=%.4g bytes=%.4g "
                        "kv_bytes=%.4g "
                        "intensity=%.1f ridge=%.1f regime=%s layer_ms=%.4f model_tok_s=%.0f "
                        "(modeled, not measured)\n",
                        m.name.c_str(), phase.c_str(), b.requests, b.new_tokens, b.context, name,
                        lc.flops, lc.bytes(), lc.kv_read_bytes + lc.kv_write_bytes, lc.intensity(),
                        g.ridge() * e.eta_c / e.eta_m, compute ? "compute" : "memory", t * 1e3,
                        b.requests * b.new_tokens / (m.layers * t));
        }
    }
    for (const auto &cfg : {qwen3_0p6b(), qwen3_1p7b(), qwen3_4b(), qwen3_8b()})
        for (const char *name : {"RTX4060L", "T4", "L4", "A100-40G"}) {
            GpuSpec g = find_gpu(gpus, name);
            MemoryFit f =
                memory_fit(cfg, cfg.layers, 0, 512, g.mem_bytes(), 1.25 * 1073741824.0, true);
            std::printf(
                "BENCH A4 model=%s gpu=%s weights_gib=%.2f fits_weights=%d max_kv_tokens=%lld "
                "requests_of_1280_tokens=%lld (modeled, not measured)\n",
                cfg.name.c_str(), name, f.need.weights / 1073741824.0, f.max_kv_tokens >= 0 ? 1 : 0,
                f.max_kv_tokens, f.max_kv_tokens >= 0 ? f.max_kv_tokens / 1280 : -1);
        }
    // A3/A5 on synthetic profiles: how well does the estimator recover known efficiencies?
    GpuSpec g = find_gpu(gpus, "RTX4060L");
    for (double noise : {0.0, 0.02, 0.05, 0.10}) {
        Rng rng(42);
        auto make = [&](double ec, double em, double t0) {
            std::vector<Sample> s;
            auto add = [&](double F, double B) {
                double t = t0 + std::max(F / (ec * g.peak_flops()), B / (em * g.peak_bw()));
                double u1 = rng.uniform() + 1e-12, u2 = rng.uniform();
                t *= 1 + noise * std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2);
                s.push_back({"syn", F, B, t});
            };
            for (double mb : {64.0, 128.0, 256.0, 512.0}) add(0, 2 * mb * 1048576);
            for (double n : {512.0, 1024.0, 2048.0, 4096.0}) add(2 * n * n * n, 8 * n * n);
            for (double mk : {3.2e7, 5e7, 6.7e7}) add(2 * mk, 2 * mk);
            return s;
        };
        auto train = make(0.85, 0.78, 8e-6), test = make(0.85, 0.78, 8e-6);
        ErrorStats st = heldout_error(train, test, g);
        std::printf("BENCH A3A5 noise=%.2f true_eta_c=0.85 true_eta_m=0.78 fit_eta_c=%.4f "
                    "fit_eta_m=%.4f fit_t0_us=%.2f "
                    "heldout_mean=%.4f heldout_max=%.4f (synthetic)\n",
                    noise, st.fit.eta_c, st.fit.eta_m, st.fit.t0 * 1e6, st.mean_abs_rel,
                    st.max_abs_rel);
    }
}

// ------------------------------------------------------------------------------- B
// Placement at block granularity: `block` consecutive layers move together (rate and memory
// are rescaled), which keeps the local search fast for 36-layer models.
static Cluster to_blocks(Cluster c, int block) {
    for (auto &n : c.nodes) n.rate_per_layer /= block, n.max_layers /= block;
    return c;
}

static std::string bottleneck(const Cluster &c, const FlowGraph &g, const CutReport &cut) {
    std::string s;
    for (int k : cut.cut_edges) {
        const auto &e = g.edges[k];
        std::string item = std::string(kind_name(e.kind)) + ":";
        if (e.kind == EdgeKind::Compute)
            item += c.nodes[e.a].name;
        else if (e.kind == EdgeKind::Link)
            item += c.nodes[e.a].name + ">" + c.nodes[e.b].name;
        else if (e.kind == EdgeKind::Source)
            item += "coord>" + c.nodes[e.b].name;
        else
            item += c.nodes[e.a].name + ">coord";
        s += (s.empty() ? "" : "|") + item;
    }
    return s.empty() ? "none" : s;
}

static void report_placement(const char *scenario, const char *planner,
                             const std::vector<GpuSpec> &gpus, const Cluster &c, const Placement &p,
                             int L, double plan_ms, double block_layers) {
    FlowGraph g = build_flow_graph(c, p, L);
    Dinic d = make_dinic(g);
    double f = d.max_flow(0, 1);
    CutReport cut = min_cut(g, d);
    auto paths = decompose_flow(g, edge_flows(g, d));
    double price = cluster_price(gpus, c, p);
    int used = 0;
    for (const auto &r : p) used += r.used();
    std::printf(
        "BENCH B scenario=%s planner=%s gpus_used=%d maxflow_tok_s=%.0f cut_value=%.0f "
        "pipelines=%zu "
        "cost_hr=%.3f usd_per_1M_tok=%.4f plan_ms=%.1f bottleneck=%s (modeled, not measured)\n",
        scenario, planner, used, f, cut.capacity, paths.size(), price,
        dollars_per_million_tokens(price, f), plan_ms, bottleneck(c, g, cut).c_str());
    std::string ranges;
    for (int i = 0; i < c.size(); ++i)
        if (p[i].used())
            ranges += " " + c.nodes[i].name + "=[" +
                      std::to_string(int(p[i].start * block_layers)) + "," +
                      std::to_string(int(p[i].end * block_layers)) + ")";
    std::printf("PLACEMENT %s %s:%s\n", scenario, planner, ranges.c_str());
    if (!paths.empty()) {
        std::vector<double> w;
        for (const auto &pp : paths) w.push_back(pp.rate);
        Iwrr rr(w);
        std::vector<int> cnt(paths.size(), 0);
        for (int i = 0; i < 1000; ++i) ++cnt[rr.next()];
        std::string mix;
        for (size_t k = 0; k < paths.size() && k < 6; ++k) {
            std::string chain;
            for (int gi : paths[k].gpus) chain += (chain.empty() ? "" : ">") + c.nodes[gi].name;
            mix += " " + chain + ":" + std::to_string(cnt[k]) + "/1000";
        }
        std::printf("IWRR %s %s first 1000 requests:%s\n", scenario, planner, mix.c_str());
    }
}

static void bench_b_with(const std::function<Efficiency(const GpuSpec &)> &eff, const char *tag) {
    auto gpus = load_gpus(kGpus);
    const int block = 4;
    for (double slow : {0.0, 0.02}) {
        ClusterInputs in;
        in.model = qwen3_32b(); // about 61 GiB of FP16 weights: no single GPU below holds it
        in.nodes = {{"L4", 4}, {"T4", 4}, {"A10G~A10", 4}, {"L40S", 2}};
        in.slow_gbit = slow;
        Cluster full = make_cluster(gpus, in, eff);
        Cluster c = to_blocks(full, block);
        int L = in.model.layers / block;
        std::string scenario = std::string(tag) + (slow > 0 ? "-slow-cross-type-links" : "-10gbit");
        auto t0 = std::chrono::steady_clock::now();
        Placement helix = place_layers(c, L, 8, 2);
        report_placement(scenario.c_str(), "flow-local-search", gpus, c, helix, L, ms_since(t0),
                         block);
        // Single pipeline through every node (Petals-like chain), partitioned optimally by B4.
        std::vector<int> all(c.size());
        std::iota(all.begin(), all.end(), 0);
        std::stable_sort(all.begin(), all.end(), [&](int a, int b) {
            return c.nodes[a].rate_per_layer * c.nodes[a].max_layers >
                   c.nodes[b].rate_per_layer * c.nodes[b].max_layers;
        });
        t0 = std::chrono::steady_clock::now();
        ChainPlan cp = partition_chain(c, all, L);
        Placement chain(c.size());
        for (size_t k = 0, s = 0; cp.feasible && k < all.size(); ++k)
            chain[all[k]] = {int(s), int(s + cp.layers[k])}, s += cp.layers[k];
        report_placement(scenario.c_str(), "single-chain-dp", gpus, c, chain, L, ms_since(t0),
                         block);
        // Homogeneous sub-clusters: only one GPU type, same planner.
        for (const char *type : {"L4", "A10G~A10", "L40S"}) {
            Placement hom(c.size());
            Cluster sub;
            std::vector<int> idx;
            for (int i = 0; i < c.size(); ++i)
                if (c.nodes[i].gpu == type) idx.push_back(i);
            for (int i : idx)
                sub.nodes.push_back(c.nodes[i]), sub.from_source.push_back(c.from_source[i]),
                    sub.to_sink.push_back(c.to_sink[i]);
            sub.link.assign(idx.size(), std::vector<double>(idx.size(), 0));
            for (size_t a = 0; a < idx.size(); ++a)
                for (size_t b = 0; b < idx.size(); ++b) sub.link[a][b] = c.link[idx[a]][idx[b]];
            t0 = std::chrono::steady_clock::now();
            Placement sp = place_layers(sub, L, 8, 2);
            for (size_t a = 0; a < idx.size(); ++a) hom[idx[a]] = sp[a];
            report_placement(scenario.c_str(), (std::string("homogeneous-") + type).c_str(), gpus,
                             c, hom, L, ms_since(t0), block);
        }
    }
}

static void bench_b() {
    std::printf("NOTE B: Qwen3-32B, decode batch 16 at context 1024 per node, placement in blocks "
                "of 4 layers, "
                "assumed efficiencies, AWS on-demand prices from data/gpus.csv\n");
    Efficiency e = assumed_efficiency();
    bench_b_with([e](const GpuSpec &) { return e; }, "assumed-eta");
}

// ------------------------------------------------------------------------------- C
struct PlanSet {
    std::vector<std::string> name;
    std::vector<Assignment> plan;
};

static PlanSet c_plans(const std::vector<Template> &tpl, const std::vector<double> &demand,
                       bool print, const char *tag) {
    PlanSet ps;
    auto t0 = std::chrono::steady_clock::now();
    Assignment opt = min_cost_assignment(tpl, demand);
    double opt_ms = ms_since(t0);
    Baselines b = baseline_plans(tpl, demand);
    ps.name = {"bnb-coloc-hetero", "no-coloc", "homogeneous-" + b.homogeneous_gpu,
               "greedy-singleton"};
    ps.plan = {opt, b.no_coloc, b.homogeneous, b.greedy};
    if (print)
        std::printf("BENCH C2 tag=%s templates=%zu bnb_nodes=%lld bnb_ms=%.2f optimum_cost_hr=%.3f "
                    "(modeled, not measured)\n",
                    tag, tpl.size(), opt.nodes, opt_ms, opt.cost_hr);
    return ps;
}

static void bench_c_with(const std::function<Efficiency(const GpuSpec &)> &eff, const char *tag) {
    auto all = load_gpus(kGpus);
    MultiModelScenario sc;
    auto gpus = pick_gpus(all, sc.gpu_names);
    TemplateInputs in = template_inputs(sc, gpus, eff);
    auto tpl = make_templates(in);
    for (const auto &t : tpl)
        if (t.models.size() == 1) {
            int m = t.models[0];
            std::printf("BENCH C1 tag=%s gpu=%s model=%s batch=%d capacity_req_s=%.3f (modeled, "
                        "not measured)\n",
                        tag, t.gpu.c_str(), sc.models[m].name.c_str(), t.batch[m], t.capacity[m]);
        }
    PlanSet ps = c_plans(tpl, sc.demand, true, tag);
    // Same optimiser with 30% planning headroom: analytic capacity is a boundary, not a
    // comfortable operating point, so queueing (and bursts) push latency over the SLO.
    std::vector<double> padded = sc.demand;
    for (auto &d : padded) d *= 1.3;
    ps.name.push_back("bnb-coloc-hetero-headroom-1.3");
    ps.plan.push_back(min_cost_assignment(tpl, padded));
    double tokens_s = 0;
    for (size_t m = 0; m < sc.demand.size(); ++m) tokens_s += sc.demand[m] * sc.work[m].output;
    for (size_t k = 0; k < ps.plan.size(); ++k) {
        const Assignment &a = ps.plan[k];
        std::printf("BENCH C4 tag=%s plan=%s cost_hr=%.3f usd_per_1M_output_tok=%.4f instances=%s "
                    "(modeled, not measured)\n",
                    tag, ps.name[k].c_str(), a.feasible ? a.cost_hr : -1,
                    a.feasible ? dollars_per_million_tokens(a.cost_hr, tokens_s) : -1,
                    plan_string(tpl, a).c_str());
        if (!a.feasible) continue;
        Deployment d = make_deployment(tpl, a, static_cast<int>(sc.models.size()),
                                       [&](const std::string &gname, int m) {
                                           GpuSpec g = find_gpu(gpus, gname);
                                           return serving_model(sc.models[m], g, eff(g));
                                       });
        for (double cv : {1.0, 3.0}) {
            auto trace = make_trace(sc.demand, sc.work, 300.0, cv, 7);
            auto t0 = std::chrono::steady_clock::now();
            SimSummary s = simulate(d, trace, sc.slo);
            double offered = 0;
            for (const auto &r : trace) offered += r.output;
            offered /= 300.0;
            std::printf("BENCH C3 tag=%s plan=%s cv=%.0f requests=%d slo_attain=%.4f "
                        "mean_ttft_s=%.3f p99_ttft_s=%.3f "
                        "mean_tpot_ms=%.2f tok_s=%.1f offered_tok_s=%.1f little_L=%.3f "
                        "little_lambdaW=%.3f sim_ms=%.0f "
                        "(simulated, modeled latencies)\n",
                        tag, ps.name[k].c_str(), cv, s.requests, s.slo_attainment, s.mean_ttft,
                        s.p99_ttft, s.mean_tpot * 1e3, s.throughput_tok_s, offered, s.little_L,
                        s.little_lambda_W, ms_since(t0));
        }
    }
}

static void bench_c() {
    std::printf("NOTE C: Qwen3-0.6B/1.7B/4B/8B, demand 6/3/1.2/0.4 req/s, AWS on-demand prices, "
                "assumed efficiencies\n");
    Efficiency e = assumed_efficiency();
    bench_c_with([e](const GpuSpec &) { return e; }, "assumed-eta");
}

// ------------------------------------------------------------------------------- R
static int bench_r(const std::string &profile) {
    auto gpus = load_gpus(kGpus);
    GpuSpec g4060 = find_gpu(gpus, "RTX4060L");
    {
        std::FILE *f = std::fopen(profile.c_str(), "rb");
        char line[512];
        while (f && std::fgets(line, sizeof line, f) && line[0] == '#')
            std::printf("PROFILE %s", line);
        if (f) std::fclose(f);
    }
    auto rows = load_profile(profile);
    std::vector<ProfileRow> own, copy_gemm, gemv_only;
    for (const auto &r : rows) {
        if (r.kernel == "gemm_cublas") continue;
        own.push_back(r);
        (r.kernel == "gemv" ? gemv_only : copy_gemm).push_back(r);
    }
    auto samples = profile_to_samples(own);
    Efficiency fit = fit_efficiency(samples, g4060);
    std::printf("BENCH R3 fit source=%s samples=%zu eta_c=%.4f eta_m=%.4f t0_us=%.2f (fitted on "
                "measured RTX4060L data)\n",
                profile.c_str(), samples.size(), fit.eta_c, fit.eta_m, fit.t0 * 1e6);
    for (const auto &s : samples)
        std::printf("BENCH R3 sample=%s measured_ms=%.4f predicted_ms=%.4f rel_err=%.3f\n",
                    s.label.c_str(), s.seconds * 1e3,
                    roofline_time(s.flops, s.bytes, g4060, fit) * 1e3,
                    (roofline_time(s.flops, s.bytes, g4060, fit) - s.seconds) / s.seconds);
    try {
        ErrorStats st =
            heldout_error(profile_to_samples(copy_gemm), profile_to_samples(gemv_only), g4060);
        std::printf("BENCH R3 heldout train=copy+gemm test=gemv n=%d mean_rel=%.4f max_rel=%.4f "
                    "eta_c=%.4f eta_m=%.4f\n",
                    st.n, st.mean_abs_rel, st.max_abs_rel, st.fit.eta_c, st.fit.eta_m);
    } catch (const std::exception &e) {
        std::printf("BENCH R3 heldout skipped: %s\n", e.what());
    }
    // Replan B and C with the measured efficiencies (transferred to every other GPU: an
    // assumption, which is exactly what the sweep below stresses).
    bench_b_with([fit](const GpuSpec &) { return fit; }, "measured-eta");
    MultiModelScenario sc;
    auto pool = pick_gpus(gpus, sc.gpu_names);
    // Sweeps: scale the transferred efficiencies of every planned GPU (all-gpus), only eta_m
    // (decode is memory-bound), or only the GPU type the nominal optimum relies on most.
    struct Sweep {
        std::string name;
        bool scale_c, scale_m;
        std::string only_gpu; // empty = every GPU
    };
    TemplateInputs in = template_inputs(sc, pool, [fit](const GpuSpec &) { return fit; });
    auto base = make_templates(in);
    Assignment nominal_opt = min_cost_assignment(base, sc.demand);
    std::string heavy;
    double heavy_cost = 0;
    for (size_t t = 0; t < base.size(); ++t)
        if (nominal_opt.feasible && nominal_opt.count[t] * base[t].price_hr > heavy_cost)
            heavy_cost = nominal_opt.count[t] * base[t].price_hr, heavy = base[t].gpu;
    std::vector<Sweep> sweeps = {{"all-gpus", true, true, ""},
                                 {"eta_m-only", false, true, ""},
                                 {heavy + "-only", true, true, heavy}};
    std::vector<double> scales = {0.5, 0.625, 0.75, 0.875, 1.0};
    for (const auto &sw : sweeps) {
        auto eff_at = [&](double s) {
            return [fit, s, sw](const GpuSpec &g) {
                bool hit = sw.only_gpu.empty() || g.name == sw.only_gpu;
                return Efficiency{fit.eta_c * (hit && sw.scale_c ? s : 1.0),
                                  fit.eta_m * (hit && sw.scale_m ? s : 1.0), fit.t0};
            };
        };
        // Candidate plans: the nominal optimum and baselines, plus the optimum at every scale.
        PlanSet cand = c_plans(base, sc.demand, false, "nominal");
        std::vector<std::vector<Template>> at(scales.size());
        for (size_t i = 0; i < scales.size(); ++i) {
            at[i] = recompute_templates(base, sc, pool, eff_at(scales[i]));
            Assignment a = min_cost_assignment(at[i], sc.demand);
            bool dup = false;
            for (const auto &p : cand.plan) dup = dup || (p.count == a.count);
            if (a.feasible && !dup) {
                char nm[64];
                std::snprintf(nm, sizeof nm, "bnb-opt-at-%.3f", scales[i]);
                cand.name.push_back(nm), cand.plan.push_back(a);
            }
        }
        auto cost = [&](int p, double s) {
            size_t i = std::find(scales.begin(), scales.end(), s) - scales.begin();
            return scaled_plan_cost(at[i], cand.plan[p], sc.demand);
        };
        auto rows_out = sensitivity_sweep(static_cast<int>(cand.plan.size()), scales, cost);
        for (size_t k = 0; k < cand.plan.size(); ++k)
            std::printf("BENCH R3 sweep=%s plan_id=%zu plan=%s instances=%s\n", sw.name.c_str(), k,
                        cand.name[k].c_str(), plan_string(base, cand.plan[k]).c_str());
        for (const auto &r : rows_out) {
            std::string costs;
            for (double c : r.cost)
                costs += (costs.empty() ? "" : "/") + (std::isinf(c) ? std::string("inf") : fmt(c));
            std::printf("BENCH R3 sweep=%s eta_scale=%.3f best_plan=%s top1_changed=%d kendall=%d "
                        "costs_hr=%s "
                        "(modeled, not measured)\n",
                        sw.name.c_str(), r.scale, cand.name[r.ranking[0]].c_str(),
                        r.top1_changed ? 1 : 0, r.kendall, costs.c_str());
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    std::string part = argc > 1 ? argv[1] : "all";
    try {
        if (part == "R") return bench_r(argc > 2 ? argv[2] : "data/rtx4060_reference_profile.csv");
        if (part == "A" || part == "all") bench_a();
        if (part == "B" || part == "all") bench_b();
        if (part == "C" || part == "all") bench_c();
    } catch (const std::exception &e) {
        std::printf("FAIL bench %s: %s\n", part.c_str(), e.what());
        return 1;
    }
    return 0;
}
