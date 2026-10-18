// lab9 tests. Usage: test <A|B|C|R|all>. Every check prints PASS/FAIL; the run continues
// past failures and exits non-zero if any check failed. Oracles never call the function
// under test: they are explicit loops, enumerations or independent re-implementations.
#include <map>
#include <set>
#include <tuple>

#include "check.hpp"
#include "replan_student.hpp"

using namespace hs;
using chk::Suite;

static const std::string kGpus = "data/gpus.csv";
static const std::string kProfile = "data/rtx4060_reference_profile.csv";

static GpuSpec spec(double tflops, double bw_gbs, double mem_gib = 80) {
    GpuSpec g;
    g.name = "synthetic";
    g.tflops = tflops, g.bw_gbs = bw_gbs, g.mem_gib = mem_gib;
    return g;
}

// =============================================================================== part A
// Brute-force operation counter: walks the explicit layer graph element by element.
struct BruteCost {
    double macs = 0, params = 0, kv_read = 0, kv_write = 0, act = 0;
};

static BruteCost brute_layer(const ModelConfig &c, const Batch &b) {
    BruteCost out;
    const long long h = c.hidden, qd = c.q_dim(), kvd = c.kv_dim(), I = c.intermediate;
    struct Mat {
        long long in, out;
    };
    const Mat projections[] = {{h, qd}, {h, kvd}, {h, kvd}, {qd, h}, {h, I}, {h, I}, {I, h}};
    for (const auto &m : projections)
        for (long long r = 0; r < m.out; ++r)
            for (long long k = 0; k < m.in; ++k) out.params += 1;
    for (long long k = 0; k < h; ++k) out.params += 2;          // input + post-attention RMSNorm
    for (long long k = 0; k < c.head_dim; ++k) out.params += 2; // q_norm + k_norm
    const int group = c.heads / c.kv_heads;
    for (int r = 0; r < b.requests; ++r) {
        for (int p = 0; p < b.context; ++p)
            for (long long e = 0; e < kvd; ++e) out.kv_read += 2; // cached K and V
        for (int i = 0; i < b.new_tokens; ++i) {
            for (const auto &m : projections) out.macs += double(m.in) * double(m.out);
            for (int head = 0; head < c.heads; ++head) {
                int kv_head = head / group;
                (void)
                    kv_head; // each query head reads its shared KV head but computes its own scores
                long long positions = b.context + i + 1;
                for (long long p = 0; p < positions; ++p)
                    for (int e = 0; e < c.head_dim; ++e) out.macs += 2; // QK^T and PV
            }
            for (long long e = 0; e < kvd; ++e) out.kv_write += 2;
            for (long long e = 0; e < h; ++e) out.act += 2; // read input, write output
        }
    }
    return out;
}

// Test-side roofline (independent of A2) for synthetic data generation.
static double model_time(double F, double B, const GpuSpec &g, double ec, double em, double t0) {
    return t0 + std::max(F / (ec * g.peak_flops()), B / (em * g.peak_bw()));
}

static std::vector<Sample> synthetic_samples(const GpuSpec &g, double ec, double em, double t0,
                                             double noise, uint64_t seed) {
    Rng rng(seed);
    std::vector<Sample> s;
    auto add = [&](const std::string &label, double F, double B) {
        double t = model_time(F, B, g, ec, em, t0);
        if (noise > 0) {
            double u1 = rng.uniform() + 1e-12, u2 = rng.uniform();
            double z = std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2);
            t *= 1 + noise * z;
        }
        s.push_back({label, F, B, t});
    };
    for (double mb : {1.0, 4.0, 16.0, 64.0, 256.0}) add("copy", 0, 2 * mb * 1048576);
    for (double m : {2048.0, 4096.0, 12288.0})
        for (double k : {2048.0, 6144.0}) add("gemv", 2 * m * k, 2 * m * k + 2 * k + 4 * m);
    for (double n : {256.0, 512.0, 1024.0, 2048.0, 4096.0})
        add("gemm", 2 * n * n * n, 2 * (2 * n * n) + 4 * n * n);
    // Intensities between the true ridge (ec/em * spec ridge) and the spec ridge: labelled
    // memory-bound by the spec but compute-bound under the true efficiencies.
    double spec_ridge = g.ridge(), true_ridge = ec / em * spec_ridge;
    for (double f : {0.15, 0.35, 0.55, 0.75, 0.9}) {
        double I = true_ridge + f * (spec_ridge - true_ridge);
        double B = 64.0 * 1048576;
        add("ridge", I * B, B);
    }
    return s;
}

static double weighted_objective(const std::vector<Sample> &s, const GpuSpec &g, double t0,
                                 double uc, double um) {
    double J = 0;
    for (const auto &x : s) {
        double pred = t0 + std::max(uc * x.flops / g.peak_flops(), um * x.bytes / g.peak_bw());
        double r = (pred - x.seconds) / x.seconds;
        J += r * r;
    }
    return J;
}

static int test_a() {
    Suite S("A");
    // ---------------------------------------------------------------- A1
    S.group("A1 layer_cost", [&] {
        std::vector<std::pair<ModelConfig, Batch>> cases = {
            {tiny_model(), {1, 5, 0}}, {tiny_model(), {3, 4, 0}},  {tiny_model(), {4, 1, 7}},
            {tiny_model(), {2, 3, 6}}, {qwen3_1p7b(), {2, 1, 64}}, {qwen3_0p6b(), {1, 3, 5}}};
        for (const auto &[cfg, b] : cases) {
            std::string tag = "A1 " + cfg.name + " R=" + std::to_string(b.requests) +
                              " q=" + std::to_string(b.new_tokens) +
                              " c=" + std::to_string(b.context);
            BruteCost o = brute_layer(cfg, b);
            LayerCost lc = layer_cost(cfg, b);
            double e = cfg.bytes_per_elem;
            S.near(tag + " flops", 2 * o.macs, lc.flops, 1e-12);
            S.near(tag + " weight bytes", e * o.params, lc.weight_bytes, 1e-12);
            S.near(tag + " kv read bytes", e * o.kv_read, lc.kv_read_bytes, 1e-12, 1e-9);
            S.near(tag + " kv write bytes", e * o.kv_write, lc.kv_write_bytes, 1e-12);
            S.near(tag + " act bytes", e * o.act, lc.act_bytes, 1e-12);
        }
        // Hand numbers (Qwen3-1.7B): 50,336,000 parameters per layer; a decode token at
        // context 1024 reads 2 * 1024 * (8 * 128) * 2 bytes = 4 MiB of KV per layer.
        LayerCost d = layer_cost(qwen3_1p7b(), decode_batch(1, 1024));
        S.near("A1 Qwen3-1.7B weight bytes per layer (hand)", 2.0 * 50336000, d.weight_bytes,
               1e-12);
        S.near("A1 Qwen3-1.7B decode KV read at c=1024 (hand)", 4194304, d.kv_read_bytes, 1e-12);
        GpuSpec g4060 = find_gpu(load_gpus(kGpus), "RTX4060L");
        S.expect("A1 decode batch 1 is memory-bound (intensity < 2)", d.intensity() < 2,
                 "intensity " + chk::num(d.intensity()));
        LayerCost p = layer_cost(qwen3_1p7b(), prefill_batch(1, 2048));
        S.expect("A1 prefill 2048 is above the RTX4060L ridge", p.intensity() > g4060.ridge(),
                 chk::num(p.intensity()) + " vs ridge " + chk::num(g4060.ridge()));
        S.throws("A1 rejects requests = 0", [] { layer_cost(tiny_model(), {0, 1, 0}); });
        S.throws("A1 rejects new_tokens = 0", [] { layer_cost(tiny_model(), {1, 0, 0}); });
        S.throws("A1 rejects context < 0", [] { layer_cost(tiny_model(), {1, 1, -1}); });
    });
    // ---------------------------------------------------------------- A2
    S.group("A2 roofline_time", [&] {
        GpuSpec g = spec(100, 1000);
        Efficiency e{0.5, 0.8, 1e-5};
        // compute-bound: 1e12 / (0.5 * 1e14) = 0.02 s; memory term 1e9 / (0.8 * 1e12) = 1.25 ms
        S.near("A2 compute-bound hand value", 0.02 + 1e-5, roofline_time(1e12, 1e9, g, e), 1e-12);
        // memory-bound: 1e10 / 0.8e12 = 12.5 ms; compute term 1e9 / 0.5e14 = 20 us
        S.near("A2 memory-bound hand value", 0.0125 + 1e-5, roofline_time(1e9, 1e10, g, e), 1e-12);
        S.near("A2 zero work costs t0", 1e-5, roofline_time(0, 0, g, e), 1e-12);
        S.near("A2 max, not sum, at the ridge", 1e-5 + 0.01,
               roofline_time(0.01 * 0.5e14, 0.01 * 0.8e12, g, e), 1e-12);
        S.throws("A2 rejects eta_c = 0", [&] { roofline_time(1, 1, g, {0, 1, 0}); });
        S.throws("A2 rejects negative flops", [&] { roofline_time(-1, 1, g, e); });
    });
    // ---------------------------------------------------------------- A3
    S.group("A3 fit_efficiency", [&] {
        GpuSpec g = find_gpu(load_gpus(kGpus), "RTX4060L");
        const double ec = 0.45, em = 0.80, t0 = 6e-6;
        auto clean = synthetic_samples(g, ec, em, t0, 0.0, 1);
        Efficiency f = fit_efficiency(clean, g);
        S.near("A3 noise-free eta_c recovered", ec, f.eta_c, 1e-6);
        S.near("A3 noise-free eta_m recovered", em, f.eta_m, 1e-6);
        S.near("A3 noise-free t0 recovered", t0, f.t0, 1e-6, 1e-12);
        auto noisy = synthetic_samples(g, ec, em, t0, 0.03, 7);
        auto more = synthetic_samples(g, ec, em, t0, 0.03, 8);
        noisy.insert(noisy.end(), more.begin(), more.end());
        Efficiency n = fit_efficiency(noisy, g);
        S.near("A3 noisy eta_c within 5%", ec, n.eta_c, 0.05);
        S.near("A3 noisy eta_m within 5%", em, n.eta_m, 0.05);
        // Stationarity of the test's own weighted objective at the returned point.
        double th[3] = {n.t0, 1 / n.eta_c, 1 / n.eta_m};
        double J = weighted_objective(noisy, g, th[0], th[1], th[2]);
        for (int i = 0; i < 3; ++i) {
            double h = 1e-5 * std::max(std::abs(th[i]), 1e-7);
            double a[3] = {th[0], th[1], th[2]}, b[3] = {th[0], th[1], th[2]};
            a[i] += h, b[i] -= h;
            double grad = (weighted_objective(noisy, g, a[0], a[1], a[2]) -
                           weighted_objective(noisy, g, b[0], b[1], b[2])) /
                          (2 * h);
            double scaled = grad * std::abs(th[i]);
            S.expect("A3 weighted-LS stationarity d/dtheta" + std::to_string(i),
                     std::abs(scaled) < 1e-6 + 1e-4 * J,
                     "scaled gradient " + chk::num(scaled) + ", J " + chk::num(J));
        }
        S.throws("A3 rejects < 3 samples", [&] { fit_efficiency({clean[0], clean[1]}, g); });
        std::vector<Sample> copies(clean.begin(), clean.begin() + 5);
        S.throws("A3 rejects a single regime (copies only)", [&] { fit_efficiency(copies, g); });
        auto bad = clean;
        bad[3].seconds = 0;
        S.throws("A3 rejects seconds <= 0", [&] { fit_efficiency(bad, g); });
    });
    // ---------------------------------------------------------------- A4
    S.group("A4 memory_fit", [&] {
        for (const auto &cfg : {tiny_model(), qwen3_1p7b(), qwen3_8b()}) {
            BruteCost one =
                brute_layer(cfg, {1, 1, 1}); // kv_read of 1 cached token = KV per token per layer
            double e = cfg.bytes_per_elem;
            double embed =
                double(cfg.vocab) * cfg.hidden * (cfg.tied_embeddings ? 1 : 2) + cfg.hidden;
            int layers = std::max(1, cfg.layers / 2);
            double kv_tokens = 1000, act_tokens = 64;
            MemoryFit f = memory_fit(cfg, layers, kv_tokens, act_tokens, kInf, 0, true);
            S.near("A4 " + cfg.name + " weights", e * (layers * one.params + embed), f.need.weights,
                   1e-12);
            S.near("A4 " + cfg.name + " kv", e * layers * kv_tokens * one.kv_read, f.need.kv,
                   1e-12);
            S.near("A4 " + cfg.name + " activations",
                   e * act_tokens * (2.0 * cfg.hidden + 2.0 * cfg.intermediate), f.need.activations,
                   1e-12);
            MemoryFit nf = memory_fit(cfg, layers, 0, 0, kInf, 0, false);
            S.near("A4 " + cfg.name + " weights without embedding", e * layers * one.params,
                   nf.need.weights, 1e-12);
        }
        // Hand: full Qwen3-1.7B in FP16 = 2 * (28 * 50,336,000 + 151,936 * 2048 + 2048) bytes.
        MemoryFit full = memory_fit(qwen3_1p7b(), 28, 0, 0, kInf, 0, true);
        S.near("A4 Qwen3-1.7B weights 3,441,149,952 B (hand)", 3441149952.0, full.need.weights,
               1e-12);
        // Exact-fit boundary with N0 = 12345 tokens (114,688 B per token for 28 layers).
        ModelConfig q = qwen3_1p7b();
        double reserve = 1.25 * 1073741824.0, act = 512 * (2.0 * 2048 + 2.0 * 6144) * 2;
        double cap = reserve + 3441149952.0 + act + 12345.0 * 114688.0;
        MemoryFit exact = memory_fit(q, 28, 12345, 512, cap, reserve, true);
        S.expect("A4 exact fit reports fits", exact.fits);
        S.expect("A4 exact fit max_kv_tokens == 12345", exact.max_kv_tokens == 12345,
                 "got " + std::to_string(exact.max_kv_tokens));
        MemoryFit below = memory_fit(q, 28, 12345, 512, cap - 1, reserve, true);
        S.expect("A4 one byte short -> does not fit", !below.fits);
        S.expect("A4 one byte short -> 12344 tokens", below.max_kv_tokens == 12344,
                 "got " + std::to_string(below.max_kv_tokens));
        MemoryFit tiny = memory_fit(q, 28, 0, 512, 1e9, reserve, true);
        S.expect("A4 weights larger than capacity -> -1", tiny.max_kv_tokens == -1 && !tiny.fits,
                 "got " + std::to_string(tiny.max_kv_tokens));
        S.throws("A4 rejects layers = 0", [&] { memory_fit(q, 0, 1, 1, 1e12, 0, false); });
    });
    // ---------------------------------------------------------------- A5
    S.group("A5 heldout_error", [&] {
        GpuSpec g = find_gpu(load_gpus(kGpus), "RTX4060L");
        const double ec = 0.6, em = 0.85, t0 = 4e-6;
        auto train = synthetic_samples(g, ec, em, t0, 0.0, 3);
        // Held-out samples: true time = model * (1 + delta); |pred - t| / t = |delta / (1 + delta)|.
        const double delta[] = {0.10, -0.20, 0.05, -0.40, 0.30, 0.0};
        const double F[] = {0, 2.0 * 8192 * 4096, 2.0 * 3072 * 3072 * 3072, 0, 1e9, 5e12};
        const double B[] = {3e8, 2.0 * 8192 * 4096, 4.0 * 3072 * 3072, 1e6, 1e8, 1e9};
        std::vector<Sample> test;
        std::vector<double> err;
        for (int i = 0; i < 6; ++i) {
            double m = model_time(F[i], B[i], g, ec, em, t0);
            test.push_back({"heldout", F[i], B[i], m * (1 + delta[i])});
            err.push_back(std::abs(delta[i] / (1 + delta[i])));
        }
        std::vector<double> sorted = err;
        std::sort(sorted.begin(), sorted.end());
        ErrorStats st = heldout_error(train, test, g);
        S.expect("A5 n == 6", st.n == 6, "got " + std::to_string(st.n));
        S.near("A5 fit uses train only (eta_c)", ec, st.fit.eta_c, 1e-6);
        S.near("A5 mean |rel err|", std::accumulate(err.begin(), err.end(), 0.0) / 6,
               st.mean_abs_rel, 1e-6);
        S.near("A5 median of even n = mean of middle two", 0.5 * (sorted[2] + sorted[3]),
               st.median_abs_rel, 1e-6);
        S.near("A5 max |rel err|", sorted.back(), st.max_abs_rel, 1e-6);
        S.throws("A5 rejects an empty test set", [&] { heldout_error(train, {}, g); });
    });
    return S.finish();
}

// =============================================================================== part B
// Independent max-flow: Edmonds-Karp on a dense capacity matrix.
static double ek_maxflow(std::vector<std::vector<double>> cap, int s, int t) {
    int n = static_cast<int>(cap.size());
    double total = 0;
    for (;;) {
        std::vector<int> prev(n, -1);
        prev[s] = s;
        std::vector<int> q{s};
        for (size_t h = 0; h < q.size() && prev[t] < 0; ++h)
            for (int v = 0; v < n; ++v)
                if (prev[v] < 0 && cap[q[h]][v] > 1e-12) prev[v] = q[h], q.push_back(v);
        if (prev[t] < 0) return total;
        double f = kInf;
        for (int v = t; v != s; v = prev[v]) f = std::min(f, cap[prev[v]][v]);
        for (int v = t; v != s; v = prev[v]) cap[prev[v]][v] -= f, cap[v][prev[v]] += f;
        total += f;
    }
}

// Minimum over all S-T cuts (vertices 0 = S, 1 = T); n <= 12.
static double enumerate_min_cut(const std::vector<std::vector<double>> &cap) {
    int n = static_cast<int>(cap.size());
    double best = kInf;
    for (int mask = 0; mask < (1 << (n - 2)); ++mask) {
        auto side = [&](int v) {
            return v == 0 ? true : v == 1 ? false : bool(mask >> (v - 2) & 1);
        };
        double c = 0;
        for (int u = 0; u < n; ++u)
            for (int v = 0; v < n; ++v)
                if (side(u) && !side(v)) c += cap[u][v];
        best = std::min(best, c);
    }
    return best;
}

struct OEdge {
    int u, v;
    double cap;
    EdgeKind kind;
};

// The test's own graph rule (re-implementation of the B1 contract).
static std::vector<OEdge> oracle_edges(const Cluster &c, const Placement &p, int L) {
    std::vector<OEdge> e;
    int n = c.size();
    for (int i = 0; i < n; ++i) {
        if (!p[i].used()) continue;
        e.push_back(
            {2 + 2 * i, 3 + 2 * i, c.nodes[i].rate_per_layer / p[i].size(), EdgeKind::Compute});
        if (p[i].start == 0) e.push_back({0, 2 + 2 * i, c.from_source[i], EdgeKind::Source});
        if (p[i].end == L) e.push_back({3 + 2 * i, 1, c.to_sink[i], EdgeKind::Sink});
        for (int j = 0; j < n; ++j)
            if (j != i && p[j].used() && p[i].end == p[j].start && c.link[i][j] > 0)
                e.push_back({3 + 2 * i, 2 + 2 * j, c.link[i][j], EdgeKind::Link});
    }
    return e;
}

static std::vector<std::vector<double>> to_matrix(int nodes, const std::vector<OEdge> &e) {
    std::vector<std::vector<double>> m(nodes, std::vector<double>(nodes, 0.0));
    for (const auto &x : e) m[x.u][x.v] += x.cap;
    return m;
}

static double oracle_throughput(const Cluster &c, const Placement &p, int L) {
    return ek_maxflow(to_matrix(2 + 2 * c.size(), oracle_edges(c, p, L)), 0, 1);
}

static Cluster random_cluster(int n, int L, uint64_t seed, const std::vector<int> &caps = {}) {
    Rng rng(seed);
    Cluster c;
    for (int i = 0; i < n; ++i) {
        Node node;
        node.name = "g" + std::to_string(i);
        node.gpu = "synthetic";
        node.rate_per_layer = 100.0 * rng.range(2, 40);
        node.max_layers = caps.empty() ? rng.range(1, L) : caps[i];
        c.nodes.push_back(node);
    }
    c.link.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) c.link[i][j] = rng.uniform() < 0.15 ? 0.0 : 100.0 * rng.range(1, 60);
    for (int i = 0; i < n; ++i) {
        c.from_source.push_back(100.0 * rng.range(5, 80));
        c.to_sink.push_back(100.0 * rng.range(5, 80));
    }
    return c;
}

static Placement random_placement(const Cluster &c, int L, Rng &rng) {
    Placement p(c.size());
    for (int i = 0; i < c.size(); ++i) {
        if (rng.uniform() < 0.2) continue;
        int k = rng.range(1, std::min(L, c.nodes[i].max_layers));
        int s = rng.range(0, L - k);
        p[i] = {s, s + k};
    }
    return p;
}

static std::vector<std::vector<double>> random_graph(int n, double density, Rng &rng) {
    std::vector<std::vector<double>> m(n, std::vector<double>(n, 0.0));
    for (int u = 0; u < n; ++u)
        for (int v = 0; v < n; ++v)
            if (u != v && v != 0 && u != 1 && rng.uniform() < density)
                m[u][v] = double(rng.range(1, 20));
    return m;
}

// Every placement with ranges inside the node memory caps (for exhaustive oracles).
static double exhaustive_best(const Cluster &c, int L) {
    int n = c.size();
    std::vector<std::vector<Range>> options(n);
    for (int i = 0; i < n; ++i) {
        options[i].push_back({0, 0});
        for (int s = 0; s < L; ++s)
            for (int e = s + 1; e <= std::min(L, s + c.nodes[i].max_layers); ++e)
                options[i].push_back({s, e});
    }
    Placement p(n);
    double best = 0;
    std::function<void(int)> rec = [&](int i) {
        if (i == n) {
            bool starts = false, ends = false; // cheap necessary condition for positive flow
            for (const auto &r : p)
                starts |= r.used() && r.start == 0, ends |= r.used() && r.end == L;
            if (starts && ends) best = std::max(best, oracle_throughput(c, p, L));
            return;
        }
        for (const auto &r : options[i]) p[i] = r, rec(i + 1);
    };
    rec(0);
    return best;
}

static void check_flow_valid(Suite &S, const std::string &tag,
                             const std::vector<std::vector<double>> &m,
                             const std::vector<std::tuple<int, int, int>> &ids, const Dinic &d,
                             double value) {
    int n = static_cast<int>(m.size());
    std::vector<double> net(n, 0.0);
    bool caps_ok = true;
    for (const auto &[u, v, k] : ids) {
        double f = d.flow(k);
        if (f < -1e-9 || f > m[u][v] + 1e-9) caps_ok = false;
        net[u] -= f, net[v] += f;
    }
    bool cons = true;
    for (int v = 2; v < n; ++v) cons = cons && std::abs(net[v]) < 1e-7;
    S.expect(tag + " capacity respected", caps_ok);
    S.expect(tag + " conservation at inner vertices", cons);
    S.near(tag + " net inflow at T equals value", value, net[1], 1e-9, 1e-9);
}

static int test_b() {
    Suite S("B");
    // ---------------------------------------------------------------- B1
    S.group("B1 build_flow_graph", [&] {
        Cluster c;
        for (int i = 0; i < 3; ++i) c.nodes.push_back({"g" + std::to_string(i), "x", 1200.0, 4});
        c.link = {{0, 500, 900}, {700, 0, 300}, {100, 200, 0}};
        c.from_source = {1000, 2000, 3000};
        c.to_sink = {4000, 5000, 6000};
        // node 0 holds [0,2), node 1 holds [2,4), node 2 holds [0,4): L = 4
        Placement p = {{0, 2}, {2, 4}, {0, 4}};
        FlowGraph g = build_flow_graph(c, p, 4);
        auto key = [](int u, int v, double cap, EdgeKind k) {
            return std::make_tuple(u, v, cap, int(k));
        };
        std::multiset<std::tuple<int, int, double, int>> got, want;
        for (const auto &e : g.edges) got.insert(key(e.u, e.v, e.cap, e.kind));
        // hand-derived expected edges
        want.insert(key(2, 3, 600, EdgeKind::Compute)); // 1200 / 2 layers
        want.insert(key(4, 5, 600, EdgeKind::Compute));
        want.insert(key(6, 7, 300, EdgeKind::Compute)); // 1200 / 4 layers
        want.insert(key(0, 2, 1000, EdgeKind::Source));
        want.insert(key(0, 6, 3000, EdgeKind::Source));
        want.insert(key(5, 1, 5000, EdgeKind::Sink));
        want.insert(key(7, 1, 6000, EdgeKind::Sink));
        want.insert(key(3, 4, 500, EdgeKind::Link)); // g0 ends at 2 == g1 starts at 2
        S.expect("B1 hand example edge multiset", got == want,
                 "got " + std::to_string(got.size()) + " edges, want " +
                     std::to_string(want.size()));
        S.expect("B1 vertex count 2 + 2n", g.num_nodes == 8);
        bool attrs = true;
        for (const auto &e : g.edges) {
            if (e.kind == EdgeKind::Compute)
                attrs =
                    attrs && e.a >= 0 && e.u == FlowGraph::in(e.a) && e.v == FlowGraph::out(e.a);
            if (e.kind == EdgeKind::Link)
                attrs = attrs && e.u == FlowGraph::out(e.a) && e.v == FlowGraph::in(e.b);
            if (e.kind == EdgeKind::Source) attrs = attrs && e.v == FlowGraph::in(e.b);
            if (e.kind == EdgeKind::Sink) attrs = attrs && e.u == FlowGraph::out(e.a);
        }
        S.expect("B1 edge kind/a/b attributes consistent with vertices", attrs);
        Rng rng(11);
        int mism = 0;
        for (int trial = 0; trial < 40; ++trial) {
            Cluster rc = random_cluster(rng.range(3, 6), 6, 100 + trial);
            Placement rp = random_placement(rc, 6, rng);
            FlowGraph rg = build_flow_graph(rc, rp, 6);
            std::multiset<std::tuple<int, int, double, int>> a, b;
            for (const auto &e : rg.edges) a.insert(key(e.u, e.v, e.cap, e.kind));
            for (const auto &e : oracle_edges(rc, rp, 6)) b.insert(key(e.u, e.v, e.cap, e.kind));
            mism += a != b;
        }
        S.expect("B1 40 random placements match the oracle rule", mism == 0,
                 std::to_string(mism) + " mismatches");
        Placement too_big = {{0, 4}, {0, 0}, {0, 5}};
        S.throws("B1 rejects a range beyond L", [&] { build_flow_graph(c, too_big, 4); });
        Cluster small = c;
        small.nodes[0].max_layers = 1;
        S.throws("B1 rejects a range above max_layers", [&] { build_flow_graph(small, p, 4); });
        S.throws("B1 rejects a wrong placement size", [&] { build_flow_graph(c, {{0, 4}}, 4); });
    });
    // ---------------------------------------------------------------- B2
    S.group("B2 Dinic max_flow", [&] {
        Rng rng(5);
        int wrong = 0, trials = 0;
        for (int trial = 0; trial < 60; ++trial) {
            int n = rng.range(4, 12);
            auto m = random_graph(n, trial % 2 ? 0.35 : 0.6, rng);
            Dinic d(n);
            std::vector<std::tuple<int, int, int>> ids;
            for (int u = 0; u < n; ++u)
                for (int v = 0; v < n; ++v)
                    if (m[u][v] > 0) ids.emplace_back(u, v, d.add_edge(u, v, m[u][v]));
            double f = d.max_flow(0, 1);
            double cut = enumerate_min_cut(m);
            ++trials;
            if (std::abs(f - cut) > 1e-9 * std::max(1.0, cut)) ++wrong;
            if (trial < 8)
                check_flow_valid(S, "B2 random graph " + std::to_string(trial), m, ids, d, f);
        }
        S.expect("B2 max-flow == enumerated min-cut on 60 random graphs", wrong == 0,
                 std::to_string(wrong) + "/" + std::to_string(trials) + " wrong");
        // Antiparallel + parallel edges and the classic cancellation graph:
        // S->a 10, S->b 10, a->b 1, a->T 8 (x2 parallel 4+4), b->T 10, b->a 6.
        std::vector<std::vector<double>> m = {
            {0, 0, 10, 10}, {0, 0, 0, 0}, {0, 8, 0, 1}, {0, 10, 6, 0}};
        Dinic d(4);
        d.add_edge(0, 2, 10), d.add_edge(0, 3, 10), d.add_edge(2, 3, 1), d.add_edge(2, 1, 4);
        d.add_edge(2, 1, 4), d.add_edge(3, 1, 10), d.add_edge(3, 2, 6);
        double f = d.max_flow(0, 1);
        S.near("B2 parallel/antiparallel edges = 18 (hand)", 18, f, 1e-12);
        S.near("B2 second call returns 0", 0, d.max_flow(0, 1), 0, 1e-12);
        S.near("B2 enumerated min-cut of the same graph", 18, enumerate_min_cut(m), 1e-12);
        // Cancellation graph: S->a, S->b, a->c, b->d, c->T, d->T all 1 and a->d 1, inserted so
        // that the first augmenting path found is S-a-d-T (a->d added before a->c). The
        // maximum 2 needs S-b-d-a-c-T, which travels a->d BACKWARDS: without reverse arcs a
        // solver stops at 1. Vertices: S=0, T=1, a=2, b=3, c=4, d=5.
        Dinic r(6);
        r.add_edge(0, 2, 1), r.add_edge(2, 5, 1), r.add_edge(0, 3, 1), r.add_edge(2, 4, 1);
        r.add_edge(3, 5, 1), r.add_edge(5, 1, 1), r.add_edge(4, 1, 1);
        double rv = r.max_flow(0, 1);
        S.near("B2 cancellation graph needs a reverse arc: flow 2", 2, rv, 1e-12);
    });
    // ---------------------------------------------------------------- B3
    S.group("B3 min_cut", [&] {
        Rng rng(9);
        int bad_value = 0, bad_side = 0, bad_sat = 0;
        for (int trial = 0; trial < 30; ++trial) {
            Cluster c = random_cluster(rng.range(3, 5), 5, 300 + trial);
            Placement p = random_placement(c, 5, rng);
            FlowGraph g = build_flow_graph(c, p, 5);
            Dinic d = make_dinic(g);
            double f = d.max_flow(0, 1);
            CutReport cut = min_cut(g, d);
            double enumerated = enumerate_min_cut(to_matrix(g.num_nodes, oracle_edges(c, p, 5)));
            if (std::abs(cut.capacity - f) > 1e-7 * std::max(1.0, f) ||
                std::abs(enumerated - f) > 1e-7 * std::max(1.0, f))
                ++bad_value;
            // Residual reachability recomputed by the test.
            std::vector<bool> reach(g.num_nodes, false);
            reach[0] = true;
            for (bool grew = true; grew;) {
                grew = false;
                for (size_t k = 0; k < g.edges.size(); ++k) {
                    const auto &e = g.edges[k];
                    double fl = d.flow(static_cast<int>(k));
                    if (reach[e.u] && !reach[e.v] && e.cap - fl > 1e-9) reach[e.v] = grew = true;
                    if (reach[e.v] && !reach[e.u] && fl > 1e-9) reach[e.u] = grew = true;
                }
            }
            if (cut.source_side != reach || cut.source_side[1]) ++bad_side;
            std::vector<int> want;
            for (size_t k = 0; k < g.edges.size(); ++k)
                if (reach[g.edges[k].u] && !reach[g.edges[k].v])
                    want.push_back(static_cast<int>(k));
            bool sat = cut.cut_edges == want;
            for (int k : cut.cut_edges) sat = sat && std::abs(d.flow(k) - g.edges[k].cap) < 1e-7;
            bad_sat += !sat;
        }
        S.expect("B3 cut capacity == max-flow == enumerated min-cut (30 placements)",
                 bad_value == 0, std::to_string(bad_value) + " bad");
        S.expect("B3 source side == residual-reachable set, T excluded", bad_side == 0,
                 std::to_string(bad_side) + " bad");
        S.expect("B3 cut edges = S-side to T-side edges, all saturated", bad_sat == 0,
                 std::to_string(bad_sat) + " bad");
        // Bottleneck naming: two fast GPUs joined by a slow link.
        Cluster c;
        c.nodes = {{"fast0", "x", 8000, 2}, {"fast1", "x", 8000, 2}};
        c.link = {{0, 300}, {300, 0}};
        c.from_source = {9000, 9000};
        c.to_sink = {9000, 9000};
        FlowGraph g = build_flow_graph(c, {{0, 2}, {2, 4}}, 4);
        Dinic d = make_dinic(g);
        double f = d.max_flow(0, 1);
        CutReport cut = min_cut(g, d);
        bool link_only =
            cut.cut_edges.size() == 1 && g.edges[cut.cut_edges[0]].kind == EdgeKind::Link;
        S.near("B3 slow-link example flow = 300", 300, f, 1e-12);
        S.expect("B3 slow-link example: the cut is exactly the link", link_only);
    });
    // ---------------------------------------------------------------- B4
    S.group("B4 partition_chain", [&] {
        Cluster h;
        h.nodes = {{"a", "x", 10, 4}, {"b", "x", 30, 4}};
        h.link = {{0, 1e9}, {1e9, 0}};
        h.from_source = {1e9, 1e9};
        h.to_sink = {1e9, 1e9};
        ChainPlan hp = partition_chain(h, {0, 1}, 4);
        // k = (1,3): min(10/1, 30/3) = 10 beats (2,2): 5 and (3,1): 3.33
        S.expect("B4 hand example layers (1,3)",
                 hp.feasible && hp.layers == std::vector<int>({1, 3}));
        S.near("B4 hand example bottleneck 10", 10, hp.bottleneck, 1e-12);
        Rng rng(21);
        int wrong = 0, invalid = 0, total = 0;
        for (int trial = 0; trial < 150; ++trial) {
            int L = rng.range(3, 9);
            Cluster c = random_cluster(5, L, 500 + trial);
            int m = rng.range(1, 4);
            std::vector<int> chain;
            for (int i = 0; i < 5 && static_cast<int>(chain.size()) < m; ++i)
                if (rng.uniform() < 0.8) chain.push_back(i);
            if (chain.empty()) chain.push_back(0);
            // Oracle: every composition of L into chain.size() positive parts within caps.
            double best = -1;
            std::vector<int> k(chain.size());
            std::function<void(size_t, int)> rec = [&](size_t j, int left) {
                if (j == chain.size()) {
                    if (left) return;
                    double v = std::min(c.from_source[chain[0]], c.to_sink[chain.back()]);
                    for (size_t t = 0; t < chain.size(); ++t) {
                        v = std::min(v, c.nodes[chain[t]].rate_per_layer / k[t]);
                        if (t + 1 < chain.size()) v = std::min(v, c.link[chain[t]][chain[t + 1]]);
                    }
                    best = std::max(best, v);
                    return;
                }
                for (int x = 1; x <= std::min(left, c.nodes[chain[j]].max_layers); ++x)
                    k[j] = x, rec(j + 1, left - x);
            };
            rec(0, L);
            ChainPlan cp = partition_chain(c, chain, L);
            ++total;
            if ((best < 0) != !cp.feasible) {
                ++wrong;
                continue;
            }
            if (best < 0) continue;
            if (std::abs(cp.bottleneck - best) > 1e-9 * std::max(1.0, best)) ++wrong;
            // The returned layers must realise the reported bottleneck.
            int sum = 0;
            double v = std::min(c.from_source[chain[0]], c.to_sink[chain.back()]);
            bool ok = cp.layers.size() == chain.size();
            for (size_t t = 0; ok && t < chain.size(); ++t) {
                ok = cp.layers[t] >= 1 && cp.layers[t] <= c.nodes[chain[t]].max_layers;
                sum += cp.layers[t];
                v = std::min(v, c.nodes[chain[t]].rate_per_layer / std::max(1, cp.layers[t]));
                if (t + 1 < chain.size()) v = std::min(v, c.link[chain[t]][chain[t + 1]]);
            }
            if (!ok || sum != L || std::abs(v - cp.bottleneck) > 1e-9 * std::max(1.0, v)) ++invalid;
        }
        S.expect("B4 DP bottleneck == composition enumeration (150 chains)", wrong == 0,
                 std::to_string(wrong) + "/" + std::to_string(total) + " wrong");
        S.expect("B4 returned layers are valid and realise the bottleneck", invalid == 0,
                 std::to_string(invalid) + " invalid");
        S.expect("B4 more nodes than layers -> infeasible",
                 !partition_chain(h, {0, 1}, 1).feasible);
        Cluster tight = h;
        tight.nodes[0].max_layers = tight.nodes[1].max_layers = 1;
        S.expect("B4 memory too small -> infeasible", !partition_chain(tight, {0, 1}, 3).feasible);
    });
    // ---------------------------------------------------------------- B5
    S.group("B5 place_layers", [&] {
        struct Case {
            int n, L;
            std::vector<int> caps;
            uint64_t seed;
        };
        // Sizes keep the exhaustive oracle below ~2M max-flow evaluations in total. The
        // first case needs replicas (two nodes holding the same range); the second needs a
        // node to fan out to two successors.
        std::vector<Case> cases = {{5, 8, {3, 2, 2, 2, 2}, 43}, {3, 6, {6, 3, 3}, 44},
                                   {4, 8, {4, 4, 3, 3}, 42},    {4, 6, {6, 3, 3, 2}, 41},
                                   {4, 7, {3, 3, 3, 2}, 45},    {5, 6, {2, 2, 2, 2, 2}, 46}};
        double ratio_sum = 0;
        int counted = 0;
        for (const auto &cs : cases) {
            Cluster c = random_cluster(cs.n, cs.L, cs.seed, cs.caps);
            std::string tag = "B5 " + std::to_string(cs.n) + " nodes x " + std::to_string(cs.L) +
                              " layers seed " + std::to_string(cs.seed);
            Placement p = place_layers(c, cs.L);
            bool feasible = static_cast<int>(p.size()) == cs.n;
            for (int i = 0; feasible && i < cs.n; ++i)
                feasible = p[i].start >= 0 && p[i].end <= cs.L && p[i].end >= p[i].start &&
                           p[i].size() <= cs.caps[i];
            S.expect(tag + " memory-feasible placement", feasible);
            if (!feasible) continue;
            double got = oracle_throughput(c, p, cs.L), best = exhaustive_best(c, cs.L);
            ratio_sum += best > 0 ? got / best : 1, ++counted;
            S.expect(tag + " >= 90% of the exhaustive optimum", got >= 0.90 * best - 1e-9,
                     "got " + chk::num(got) + " vs optimum " + chk::num(best));
            // Local optimality under single-node moves, re-derived with the oracle max-flow.
            bool local = true;
            for (int i = 0; local && i < cs.n; ++i) {
                Placement q = p;
                for (int s = 0; s <= cs.L && local; ++s)
                    for (int e = s; e <= std::min(cs.L, s + cs.caps[i]) && local; ++e) {
                        if ((e == s && s > 0)) continue;
                        q[i] = {s, e};
                        local = oracle_throughput(c, q, cs.L) <= got * (1 + 1e-7) + 1e-9;
                    }
            }
            S.expect(tag + " no single-node move improves the result", local);
        }
        S.expect("B5 mean ratio to the optimum >= 0.97 over the fixed cases",
                 counted > 0 && ratio_sum / counted >= 0.97,
                 "mean ratio " + chk::num(counted ? ratio_sum / counted : 0));
        // The heuristic must beat the best single chain (one pipeline through all nodes in
        // the B4 order) on a case where replicas pay off: three slow equal GPUs.
        Cluster rep;
        for (int i = 0; i < 3; ++i) rep.nodes.push_back({"r" + std::to_string(i), "x", 600.0, 4});
        rep.link.assign(3, std::vector<double>(3, 1e9));
        rep.from_source.assign(3, 1e9);
        rep.to_sink.assign(3, 1e9);
        for (int i = 0; i < 3; ++i) rep.link[i][i] = 0;
        Placement rp = place_layers(rep, 4);
        // Best: all three hold [0,4) in parallel: 3 * 600 / 4 = 450 tokens/s; best chain 300.
        S.near("B5 three replicas of the whole model: 450 tokens/s", 450,
               oracle_throughput(rep, rp, 4), 1e-9);
    });
    // ---------------------------------------------------------------- B6
    S.group("B6 decompose_flow", [&] {
        Rng rng(31);
        int bad = 0, count = 0;
        for (int trial = 0; trial < 30; ++trial) {
            int L = 6;
            Cluster c = random_cluster(rng.range(3, 6), L, 700 + trial);
            Placement p = random_placement(c, L, rng);
            FlowGraph g = build_flow_graph(c, p, L);
            Dinic d = make_dinic(g);
            double f = d.max_flow(0, 1);
            std::vector<double> ef = edge_flows(g, d);
            auto paths = decompose_flow(g, ef);
            double sum = 0;
            std::vector<double> recon(g.edges.size(), 0.0);
            bool valid = true;
            for (const auto &path : paths) {
                sum += path.rate;
                valid = valid && path.rate > 0 && !path.gpus.empty();
                // pipeline must cover [0, L) with exact hand-offs and use existing edges
                int layer = 0;
                int prev = -1;
                for (int gpu : path.gpus) {
                    valid = valid && p[gpu].start == layer;
                    layer = p[gpu].end;
                    for (size_t k = 0; k < g.edges.size(); ++k) {
                        const auto &e = g.edges[k];
                        if (e.kind == EdgeKind::Compute && e.a == gpu) recon[k] += path.rate;
                        if (prev < 0 && e.kind == EdgeKind::Source && e.b == gpu)
                            recon[k] += path.rate;
                        if (prev >= 0 && e.kind == EdgeKind::Link && e.a == prev && e.b == gpu)
                            recon[k] += path.rate;
                    }
                    prev = gpu;
                }
                valid = valid && layer == L;
                for (size_t k = 0; k < g.edges.size(); ++k)
                    if (prev >= 0 && g.edges[k].kind == EdgeKind::Sink && g.edges[k].a == prev)
                        recon[k] += path.rate;
            }
            for (size_t k = 0; k < g.edges.size(); ++k)
                valid = valid && std::abs(recon[k] - ef[k]) < 1e-6 * std::max(1.0, f);
            int positive = 0;
            for (double x : ef) positive += x > 1e-9;
            valid = valid && static_cast<int>(paths.size()) <= positive;
            valid = valid && std::abs(sum - f) < 1e-6 * std::max(1.0, f);
            bad += !valid;
            count += static_cast<int>(paths.size());
        }
        S.expect("B6 30 decompositions: rates sum to flow, per-edge recomposition, valid "
                 "pipelines, <= |E+| paths",
                 bad == 0, std::to_string(bad) + " bad");
        S.expect("B6 produced pipelines", count > 0);
        // Fan-out by hand: A holds [0,2) (600/2 = 300 tokens/s) and feeds B and C, which hold
        // [2,4) at 200/2 = 100 and 300/2 = 150. Max-flow 250 = two pipelines A>B (100) and
        // A>C (150), found in edge order. A path rate taken from the first edge (300) is wrong.
        Cluster fan;
        fan.nodes = {{"A", "x", 600, 4}, {"B", "x", 200, 4}, {"C", "x", 300, 4}};
        fan.link = {{0, 1e6, 1e6}, {1e6, 0, 1e6}, {1e6, 1e6, 0}};
        fan.from_source = {1e6, 1e6, 1e6};
        fan.to_sink = {1e6, 1e6, 1e6};
        FlowGraph fg = build_flow_graph(fan, {{0, 2}, {2, 4}, {2, 4}}, 4);
        Dinic fd = make_dinic(fg);
        double fv = fd.max_flow(0, 1);
        auto fp = decompose_flow(fg, edge_flows(fg, fd));
        std::map<std::vector<int>, double> got;
        for (const auto &path : fp) got[path.gpus] += path.rate;
        std::map<std::vector<int>, double> want = {{{0, 1}, 100.0}, {{0, 2}, 150.0}};
        bool same = got.size() == want.size();
        for (const auto &[k, v] : want) same = same && got.count(k) && std::abs(got[k] - v) < 1e-9;
        S.near("B6 fan-out example max-flow 250", 250, fv, 1e-12);
        S.expect("B6 fan-out example: pipelines A>B at 100 and A>C at 150", same);
        Cluster c;
        c.nodes = {{"a", "x", 100, 4}, {"b", "x", 100, 4}};
        c.link = {{0, 100}, {100, 0}};
        c.from_source = {100, 100};
        c.to_sink = {100, 100};
        FlowGraph g = build_flow_graph(c, {{0, 2}, {2, 4}}, 4);
        std::vector<double> broken(g.edges.size(), 0.0);
        for (size_t k = 0; k < g.edges.size(); ++k)
            if (g.edges[k].kind == EdgeKind::Source) broken[k] = 10; // flow enters but never leaves
        bool threw = false;
        try {
            decompose_flow(g, broken);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        S.expect("B6 non-conserved flow -> std::runtime_error", threw);
    });
    // ---------------------------------------------------------------- B7
    S.group("B7 Iwrr", [&] {
        Iwrr a({5, 1, 1});
        std::vector<int> seq;
        for (int i = 0; i < 7; ++i) seq.push_back(a.next());
        S.expect("B7 weights (5,1,1) give a a b a c a a",
                 seq == std::vector<int>({0, 0, 1, 0, 2, 0, 0}));
        Rng rng(77);
        double worst = 0;
        bool zero_picked = false;
        for (int trial = 0; trial < 20; ++trial) {
            int k = rng.range(2, 6);
            std::vector<double> w(k);
            double W = 0;
            for (auto &x : w) x = rng.uniform() < 0.15 ? 0.0 : 0.1 + 10 * rng.uniform(), W += x;
            if (W <= 0) w[0] = 1, W = 1;
            Iwrr it(w);
            std::vector<int> cnt(k, 0);
            for (int t = 1; t <= 1000; ++t) {
                int p = it.next();
                ++cnt[p];
                zero_picked = zero_picked || w[p] == 0;
                for (int i = 0; i < k; ++i)
                    worst = std::max(worst, std::abs(cnt[i] - t * w[i] / W));
            }
        }
        S.expect("B7 prefix deviation |count_i - t w_i / W| < 1 over 20 random weight sets",
                 worst < 1 + 1e-9, "worst " + chk::num(worst));
        S.expect("B7 zero-weight entries are never picked", !zero_picked);
        Iwrr per({3, 2});
        std::vector<int> first, second;
        for (int i = 0; i < 5; ++i) first.push_back(per.next());
        for (int i = 0; i < 5; ++i) second.push_back(per.next());
        S.expect("B7 integer weights (3,2) give a period-5 sequence", first == second);
        S.throws("B7 rejects negative weights", [] { Iwrr bad({1, -1}); });
        S.throws("B7 rejects all-zero weights", [] { Iwrr bad({0, 0}); });
    });
    return S.finish();
}

// =============================================================================== part C
static ServingModel closed_form(double a, double b0, double b1, double b2, double kv) {
    ServingModel m;
    m.name = "closed-form";
    m.prefill_s = [a](int p) { return 0.01 + a * p; };
    m.step_s = [b0, b1, b2](int b, double c) { return b0 + b1 * b + b2 * b * c; };
    m.kv_bytes_per_token = kv;
    return m;
}

struct ORate {
    bool ok = false;
    int b = 0;
    double rate = 0, tpot = 0;
};

// Test-side re-implementation of the C1 rule.
static ORate oracle_capacity(double a, double b0, double b1, double b2, double kv,
                             const Workload &w, const Slo &s, double share, double budget,
                             int max_b) {
    ORate best;
    double tp = (0.01 + a * w.prompt) / share;
    if (tp > s.ttft_s) return best;
    if (w.output == 1) {
        if ((w.prompt + 1) * kv <= budget) best = {true, 1, 1 / tp, 0};
        return best;
    }
    double c = w.prompt + w.output / 2.0;
    for (int b = 1; b <= max_b; ++b) {
        if (b * (w.prompt + w.output) * kv > budget) break;
        double tpot = (b0 + b1 * b + b2 * b * c) / share;
        double rate = 1 / (tp + (w.output - 1) * tpot / b);
        double obs = tpot / (1 - rate * tp);
        if (obs <= s.tpot_s && rate > best.rate * (1 + 1e-12)) best = {true, b, rate, obs};
    }
    return best;
}

static Template make_tpl(const std::string &gpu, double price, std::vector<int> models,
                         std::vector<double> cap) {
    Template t;
    t.gpu = gpu, t.price_hr = price, t.models = std::move(models), t.capacity = std::move(cap);
    t.batch.assign(t.capacity.size(), 1);
    return t;
}

static double brute_cost(const std::vector<Template> &tpl, const std::vector<double> &demand) {
    size_t T = tpl.size(), M = demand.size();
    std::vector<int> upper(T, 0), cnt(T, 0);
    for (size_t t = 0; t < T; ++t)
        for (size_t m = 0; m < M; ++m)
            if (tpl[t].capacity[m] > 0 && demand[m] > 0)
                upper[t] =
                    std::max(upper[t], int(std::ceil(demand[m] / tpl[t].capacity[m] - 1e-9)));
    double best = kInf;
    std::function<void(size_t)> rec = [&](size_t t) {
        if (t == T) {
            double cost = 0;
            for (size_t m = 0; m < M; ++m) {
                double cov = 0;
                for (size_t u = 0; u < T; ++u) cov += cnt[u] * tpl[u].capacity[m];
                if (cov < demand[m] - 1e-9) return;
            }
            for (size_t u = 0; u < T; ++u) cost += cnt[u] * tpl[u].price_hr;
            best = std::min(best, cost);
            return;
        }
        for (int k = 0; k <= upper[t]; ++k) cnt[t] = k, rec(t + 1);
    };
    rec(0);
    return best;
}

static bool covers(const std::vector<Template> &tpl, const Assignment &a,
                   const std::vector<double> &d) {
    if (a.count.size() != tpl.size()) return false;
    for (size_t m = 0; m < d.size(); ++m) {
        double cov = 0;
        for (size_t t = 0; t < tpl.size(); ++t) cov += a.count[t] * tpl[t].capacity[m];
        if (cov < d[m] - 1e-9) return false;
    }
    return true;
}

static double plan_cost(const std::vector<Template> &tpl, const Assignment &a) {
    double c = 0;
    for (size_t t = 0; t < tpl.size() && t < a.count.size(); ++t) c += a.count[t] * tpl[t].price_hr;
    return c;
}

static std::vector<Template> random_templates(Rng &rng, int M, int G) {
    std::vector<Template> out;
    for (int g = 0; g < G; ++g) {
        double price = 0.5 + 2.5 * rng.uniform();
        for (int mask = 1; mask < (1 << M); ++mask) {
            int bits = __builtin_popcount(mask);
            if (bits > 2 || rng.uniform() < 0.25) continue;
            std::vector<int> ms;
            std::vector<double> cap(M, 0.0);
            for (int m = 0; m < M; ++m)
                if (mask >> m & 1) ms.push_back(m), cap[m] = (0.5 + 2.0 * rng.uniform()) / bits;
            out.push_back(make_tpl("gpu" + std::to_string(g), price, ms, cap));
        }
    }
    return out;
}

static int test_c() {
    Suite S("C");
    // ---------------------------------------------------------------- C1
    S.group("C1 slo_capacity", [&] {
        struct Case {
            double a, b0, b1, b2, kv;
            Workload w;
            Slo s;
            double share, budget;
            int maxb;
            std::string name;
        };
        std::vector<Case> cases = {
            {1e-4,
             0.02,
             1e-4,
             1e-7,
             1000,
             {512, 128},
             {1.0, 0.05},
             1.0,
             1e9,
             256,
             "full GPU, loose memory"},
            {1e-4, 0.02, 1e-4, 1e-7, 1000, {512, 128}, {1.0, 0.08}, 0.5, 1e9, 256, "half share"},
            {1e-4,
             0.02,
             1e-4,
             1e-7,
             1000,
             {512, 128},
             {1.0, 0.05},
             1.0,
             640 * 1000.0 * 9.5,
             256,
             "memory-bound batch"},
            {1e-4,
             0.02,
             1e-4,
             1e-7,
             1000,
             {512, 128},
             {0.03, 0.05},
             0.5,
             1e9,
             256,
             "TTFT infeasible"},
            {1e-4, 0.02, 1e-4, 1e-7, 1000, {512, 1}, {1.0, 0.05}, 1.0, 1e9, 256, "output = 1"},
            {2e-4, 0.01, 5e-4, 2e-7, 500, {256, 512}, {2.0, 0.03}, 0.75, 1e8, 64, "tight TPOT"},
        };
        for (const auto &c : cases) {
            ORate o =
                oracle_capacity(c.a, c.b0, c.b1, c.b2, c.kv, c.w, c.s, c.share, c.budget, c.maxb);
            Capacity got = slo_capacity(closed_form(c.a, c.b0, c.b1, c.b2, c.kv), c.w, c.s, c.share,
                                        c.budget, c.maxb);
            S.expect("C1 " + c.name + " feasibility", got.feasible == o.ok);
            if (!o.ok) continue;
            S.expect("C1 " + c.name + " batch", got.batch == o.b,
                     "expected " + std::to_string(o.b) + " vs " + std::to_string(got.batch));
            S.near("C1 " + c.name + " rate", o.rate, got.rate, 1e-9);
            S.near("C1 " + c.name + " observed TPOT", o.tpot, got.tpot_s, 1e-9, 1e-12);
        }
        S.throws("C1 rejects share 0", [] {
            slo_capacity(closed_form(1e-4, .02, 1e-4, 1e-7, 1), {1, 2}, {}, 0, 1e9, 4);
        });
        S.throws("C1 rejects share > 1", [] {
            slo_capacity(closed_form(1e-4, .02, 1e-4, 1e-7, 1), {1, 2}, {}, 1.5, 1e9, 4);
        });
    });
    // ---------------------------------------------------------------- C2
    S.group("C2 min_cost_assignment", [&] {
        // Hand: one shared instance (price 1, 1 req/s for both models) beats two singletons
        // (price 0.8 each). Optimum 1.0; a SUM-of-models bound (2.0) would wrongly prune it.
        std::vector<Template> h = {make_tpl("x", 0.8, {0}, {1, 0}), make_tpl("x", 0.8, {1}, {0, 1}),
                                   make_tpl("x", 1.0, {0, 1}, {1, 1})};
        Assignment a = min_cost_assignment(h, {1, 1});
        S.expect("C2 hand: co-located template chosen",
                 a.feasible && a.count == std::vector<int>({0, 0, 1}));
        S.near("C2 hand: optimum cost 1.0", 1.0, a.cost_hr, 1e-12);
        Rng rng(13);
        int wrong = 0, bad_plan = 0;
        long long nodes = 0;
        for (int trial = 0; trial < 12; ++trial) {
            auto tpl = random_templates(rng, 3, 2);
            std::vector<double> d = {0.3 + 2 * rng.uniform(), 0.2 + 1.5 * rng.uniform(),
                                     rng.uniform() < 0.2 ? 0 : 1 + rng.uniform()};
            double best = brute_cost(tpl, d);
            Assignment got = min_cost_assignment(tpl, d);
            nodes += got.nodes;
            if (std::isinf(best) != !got.feasible ||
                (!std::isinf(best) && std::abs(got.cost_hr - best) > 1e-9))
                ++wrong;
            if (got.feasible &&
                (!covers(tpl, got, d) || std::abs(plan_cost(tpl, got) - got.cost_hr) > 1e-9))
                ++bad_plan;
        }
        S.expect("C2 B&B optimum == brute force on 12 random instances", wrong == 0,
                 std::to_string(wrong) + " wrong");
        S.expect("C2 returned counts cover demand and match cost", bad_plan == 0,
                 std::to_string(bad_plan) + " bad");
        S.expect("C2 counts search nodes", nodes > 0);
        Assignment none = min_cost_assignment({make_tpl("x", 1, {0}, {1, 0})}, {1, 1});
        S.expect("C2 uncoverable model -> infeasible", !none.feasible);
        S.throws("C2 rejects capacity size mismatch",
                 [] { min_cost_assignment({make_tpl("x", 1, {0}, {1})}, {1, 1}); });
    });
    // ---------------------------------------------------------------- C3
    S.group("C3 simulate_instance", [&] {
        ServingModel fixed;
        fixed.prefill_s = [](int) { return 1.0; };
        fixed.step_s = [](int, double) { return 0.5; };
        // Hand trace (worked out in the README): batch 2, three requests.
        std::vector<Request> reqs = {{0.0, 0, 1, 3}, {0.2, 0, 1, 2}, {0.3, 0, 1, 1}};
        InstanceStats st = simulate_instance({{0, 2, &fixed}}, reqs);
        std::map<int, RequestRecord> rec;
        for (const auto &r : st.records) rec[r.id] = r;
        S.expect("C3 hand: first tokens 1.0 / 2.0 / 3.5",
                 rec.size() == 3 && rec[0].first_token == 1.0 && rec[1].first_token == 2.0 &&
                     rec[2].first_token == 3.5);
        S.expect("C3 hand: finishes 4.0 / 2.5 / 3.5",
                 rec[0].finish == 4.0 && rec[1].finish == 2.5 && rec[2].finish == 3.5);
        S.near("C3 hand: area 9.5", 9.5, st.area, 1e-12);
        S.near("C3 hand: busy 4.0", 4.0, st.busy, 1e-12);
        S.near("C3 hand: end_time 4.0", 4.0, st.end_time, 1e-12);
        // Round robin between two models (each batch 1, prefill-only requests at t = 0).
        std::vector<Request> rr = {{0, 0, 1, 1}, {0, 0, 1, 1}, {0, 1, 1, 1}};
        InstanceStats st2 = simulate_instance({{0, 1, &fixed}, {1, 1, &fixed}}, rr);
        std::map<int, double> ft;
        for (const auto &r : st2.records) ft[r.id] = r.first_token;
        S.expect("C3 round robin across models: model 1 served second",
                 ft[0] == 1 && ft[2] == 2 && ft[1] == 3);
        // M/D/1: service S = 10 ms, rho = 0.7, Poisson arrivals.
        const double svc = 0.01, rho = 0.7;
        ServingModel det;
        det.prefill_s = [svc](int) { return svc; };
        det.step_s = [](int, double) { return 1.0; };
        Rng rng(2024);
        std::vector<Request> md1;
        double t = 0;
        for (int i = 0; i < 100000; ++i)
            t += rng.exponential(rho / svc), md1.push_back({t, 0, 1, 1});
        InstanceStats q = simulate_instance({{0, 1, &det}}, md1);
        double wait = 0, sojourn = 0;
        for (const auto &r : q.records) {
            wait += r.first_token - md1[r.id].arrival - svc;
            sojourn += r.finish - md1[r.id].arrival;
        }
        wait /= md1.size();
        S.near("C3 M/D/1 mean wait = rho S / (2 (1 - rho))", rho * svc / (2 * (1 - rho)), wait,
               0.05);
        S.near("C3 Little's law: area == sum of sojourn times", sojourn, q.area, 1e-9);
        // Throughput bound under overload: all requests at t = 0.
        ServingModel lin = closed_form(1e-3, 0.01, 0.002, 1e-6, 1);
        const int b = 8, p = 100, o = 50;
        std::vector<Request> burst(200, Request{0.0, 0, p, o});
        InstanceStats ov = simulate_instance({{0, b, &lin}}, burst);
        double bound = 1 / ((0.01 + 1e-3 * p) + (o - 1) * (0.01 + 0.002 * b + 1e-6 * b * p) / b);
        double thr = burst.size() / ov.end_time;
        S.expect("C3 completed req/s <= per-request GPU-time bound", thr <= bound * (1 + 1e-9),
                 chk::num(thr) + " vs bound " + chk::num(bound));
        S.expect("C3 overload throughput >= 80% of the bound", thr >= 0.8 * bound,
                 chk::num(thr) + " vs " + chk::num(bound));
        S.near("C3 busy == end_time when requests never stop arriving", ov.end_time, ov.busy, 1e-9);
        S.throws("C3 rejects an unsorted trace",
                 [&] { simulate_instance({{0, 1, &fixed}}, {{1, 0, 1, 1}, {0, 0, 1, 1}}); });
        S.throws("C3 rejects a model without a slot",
                 [&] { simulate_instance({{0, 1, &fixed}}, {{0, 3, 1, 1}}); });
    });
    // ---------------------------------------------------------------- C4
    S.group("C4 baseline_plans", [&] {
        Rng rng(17);
        int bad = 0;
        for (int trial = 0; trial < 8; ++trial) {
            auto tpl = random_templates(rng, 3, 2);
            for (int m = 0; m < 3; ++m) { // guarantee one singleton per model per GPU
                std::vector<double> cap(3, 0.0);
                cap[m] = 0.4 + rng.uniform();
                tpl.push_back(
                    make_tpl("gpu" + std::to_string(m % 2), 0.7 + rng.uniform(), {m}, cap));
            }
            std::vector<double> d = {0.5 + 2 * rng.uniform(), 0.3 + rng.uniform(),
                                     0.2 + rng.uniform()};
            Baselines b = baseline_plans(tpl, d);
            // greedy by hand
            double greedy = 0;
            std::vector<int> gc(tpl.size(), 0);
            for (int m = 0; m < 3; ++m) {
                int pick = -1;
                for (size_t t = 0; t < tpl.size(); ++t)
                    if (tpl[t].models.size() == 1 && tpl[t].models[0] == m &&
                        tpl[t].capacity[m] > 0 &&
                        (pick < 0 || tpl[t].price_hr / tpl[t].capacity[m] <
                                         tpl[pick].price_hr / tpl[pick].capacity[m]))
                        pick = static_cast<int>(t);
                int k = int(std::ceil(d[m] / tpl[pick].capacity[m] - 1e-9));
                gc[pick] += k, greedy += k * tpl[pick].price_hr;
            }
            bool ok = b.greedy.feasible && b.greedy.count == gc &&
                      std::abs(b.greedy.cost_hr - greedy) < 1e-9;
            std::vector<Template> single;
            for (const auto &t : tpl)
                if (t.models.size() == 1) single.push_back(t);
            ok = ok && std::abs(b.no_coloc.cost_hr - brute_cost(single, d)) < 1e-9 &&
                 covers(tpl, b.no_coloc, d);
            double hom = kInf;
            std::string hom_gpu;
            for (std::string g : {"gpu0", "gpu1"}) {
                std::vector<Template> sub;
                for (const auto &t : tpl)
                    if (t.gpu == g) sub.push_back(t);
                double c = brute_cost(sub, d);
                if (c < hom - 1e-9) hom = c, hom_gpu = g;
            }
            ok = ok && std::abs(b.homogeneous.cost_hr - hom) < 1e-9 &&
                 b.homogeneous_gpu == hom_gpu && covers(tpl, b.homogeneous, d);
            bool type_ok = true;
            for (size_t t = 0; t < tpl.size(); ++t)
                type_ok = type_ok && (b.homogeneous.count[t] == 0 || tpl[t].gpu == hom_gpu);
            ok = ok && type_ok;
            double opt = brute_cost(tpl, d);
            ok = ok && opt <= b.no_coloc.cost_hr + 1e-9 && opt <= b.homogeneous.cost_hr + 1e-9 &&
                 opt <= greedy + 1e-9;
            bad += !ok;
        }
        S.expect("C4 greedy / no-co-location / homogeneous match independent recomputation (8 "
                 "instances)",
                 bad == 0, std::to_string(bad) + " bad");
        S.near("C4 $/1M tokens: $3.6/h at 1000 tok/s = $1.00 (provided helper)", 1.0,
               dollars_per_million_tokens(3.6, 1000), 1e-12);
    });
    return S.finish();
}

// =============================================================================== part R (CPU)
static int test_r() {
    Suite S("R");
    S.group("R5 profile_to_samples", [&] {
        std::vector<ProfileRow> rows = {{"copy", 1 << 20, 1, 1, 0.01, 0.009, 0.011},
                                        {"gemv", 4096, 1, 2048, 0.05, 0.049, 0.051},
                                        {"gemm", 128, 256, 64, 0.02, 0.019, 0.021},
                                        {"gemm_cublas", 64, 64, 32, 0.004, 0.004, 0.004}};
        auto s = profile_to_samples(rows);
        S.expect("R5 one sample per row", s.size() == 4);
        if (s.size() != 4) return;
        S.near("R5 copy bytes = 2m", 2.0 * (1 << 20), s[0].bytes, 1e-15);
        S.near("R5 copy flops = 0", 0, s[0].flops, 0, 0);
        S.near("R5 copy seconds", 1e-5, s[0].seconds, 1e-12);
        S.near("R5 gemv flops = 2mk", 2.0 * 4096 * 2048, s[1].flops, 1e-15);
        S.near("R5 gemv bytes = 2mk + 2k + 4m", 2.0 * 4096 * 2048 + 2 * 2048 + 4 * 4096, s[1].bytes,
               1e-15);
        S.near("R5 gemm flops = 2mnk", 2.0 * 128 * 256 * 64, s[2].flops, 1e-15);
        S.near("R5 gemm bytes = 2(mk + kn) + 4mn", 2.0 * (128 * 64 + 64 * 256) + 4.0 * 128 * 256,
               s[2].bytes, 1e-15);
        S.near("R5 cuBLAS gemm counted like gemm", 2.0 * 64 * 64 * 32, s[3].flops, 1e-15);
        S.expect("R5 label format", s[2].label == "gemm_128x256x64", "got " + s[2].label);
        S.throws("R5 rejects an unknown kernel",
                 [] { profile_to_samples({{"fft", 1, 1, 1, 1, 1, 1}}); });
        S.throws("R5 rejects median <= 0",
                 [] { profile_to_samples({{"copy", 1, 1, 1, 0, 0, 0}}); });
    });
    S.group("R5 reference profile refit", [&] {
        auto rows = load_profile(kProfile);
        GpuSpec g = find_gpu(load_gpus(kGpus), "RTX4060L");
        std::vector<ProfileRow> mine;
        for (const auto &r : rows)
            if (r.kernel != "gemm_cublas") mine.push_back(r);
        auto s = profile_to_samples(mine);
        Efficiency e = fit_efficiency(s, g);
        S.expect("R5 reference eta_c in (0, 1.5)", e.eta_c > 0 && e.eta_c < 1.5, chk::num(e.eta_c));
        S.expect("R5 reference eta_m in (0.3, 1.2)", e.eta_m > 0.3 && e.eta_m < 1.2,
                 chk::num(e.eta_m));
        double worst = 0;
        for (const auto &x : s)
            worst = std::max(worst, std::abs(roofline_time(x.flops, x.bytes, g, e) - x.seconds) /
                                        x.seconds);
        S.expect("R5 in-sample max relative error < 60%", worst < 0.6, chk::num(worst));
    });
    S.group("R6 sensitivity_sweep", [&] {
        // cost_p(s) = a_p + b_p / s: plan 0 cheap but eta-sensitive, plan 1 robust, plan 2
        // dominated, plan 3 infeasible below s = 0.7.
        auto cost = [](int p, double s) {
            const double a[] = {1.0, 1.6, 3.0, 1.2}, b[] = {1.0, 0.3, 0.1, 0.5};
            if (p == 3 && s < 0.7) return kInf;
            return a[p] + b[p] / s;
        };
        std::vector<double> scales = {0.5, 0.625, 0.75, 0.875, 1.0};
        auto rows = sensitivity_sweep(4, scales, cost);
        S.expect("R6 one row per scale in input order",
                 rows.size() == 5 && rows[0].scale == 0.5 && rows[4].scale == 1.0);
        if (rows.size() != 5) return;
        // Oracle: rank by (cost, index) and count discordant pairs by brute force.
        auto rank = [&](double s) {
            std::vector<int> r = {0, 1, 2, 3};
            std::stable_sort(r.begin(), r.end(),
                             [&](int x, int y) { return cost(x, s) < cost(y, s); });
            return r;
        };
        auto nominal = rank(1.0);
        bool ok = true;
        for (size_t i = 0; i < scales.size(); ++i) {
            auto r = rank(scales[i]);
            int disc = 0;
            for (int x = 0; x < 4; ++x)
                for (int y = 0; y < 4; ++y) {
                    if (x == y) continue;
                    auto px = std::find(r.begin(), r.end(), x) - r.begin(),
                         py = std::find(r.begin(), r.end(), y) - r.begin();
                    auto nx = std::find(nominal.begin(), nominal.end(), x) - nominal.begin();
                    auto ny = std::find(nominal.begin(), nominal.end(), y) - nominal.begin();
                    disc += (px < py) && (nx > ny);
                }
            ok = ok && rows[i].ranking == r && rows[i].kendall == disc &&
                 rows[i].top1_changed == (r[0] != nominal[0]);
            for (int p = 0; p < 4; ++p) ok = ok && (rows[i].cost[p] == cost(p, scales[i]));
        }
        S.expect("R6 rankings, Kendall distances and top-1 flags match brute force", ok);
        S.expect("R6 this example flips the top plan at s = 0.5", rows[0].top1_changed);
        S.expect("R6 infeasible plan ranked last at s = 0.5", rows[0].ranking.back() == 3);
        S.expect("R6 nominal row has distance 0", rows[4].kendall == 0 && !rows[4].top1_changed);
        S.throws("R6 rejects scales without 1.0", [&] { sensitivity_sweep(2, {0.5}, cost); });
        S.throws("R6 rejects n_plans < 1", [&] { sensitivity_sweep(0, {1.0}, cost); });
    });
    return S.finish();
}

int main(int argc, char **argv) {
    std::string part = argc > 1 ? argv[1] : "all";
    int rc = 0;
    if (part == "A" || part == "all") rc |= test_a();
    if (part == "B" || part == "all") rc |= test_b();
    if (part == "C" || part == "all") rc |= test_c();
    if (part == "R" || part == "all") rc |= test_r();
    if (part != "A" && part != "B" && part != "C" && part != "R" && part != "all") {
        std::printf("unknown part %s\n", part.c_str());
        return 2;
    }
    return rc;
}
