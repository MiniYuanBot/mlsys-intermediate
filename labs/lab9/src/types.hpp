// Shared, provided types for lab9 (no TODOs here). Every student header includes this file.
//
// Units used everywhere in the lab:
//   time      seconds (double)            FLOPs     floating-point operations (2 per MAC)
//   bytes     bytes (double)              rates     tokens/s (B) or requests/s (C)
//   memory    GPU capacity in GiB (2^30)  bandwidth GB/s (1e9 bytes/s), compute TFLOPS (1e12)
#pragma once
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-parameter" // student stubs ignore their inputs
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace hs {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------- model configs
// Decoder-only transformer with GQA attention and a SwiGLU MLP (Qwen3 layout).
struct ModelConfig {
    std::string name;
    int layers = 0;
    int hidden = 0;       // h
    int heads = 0;        // H (query heads)
    int kv_heads = 0;     // H_kv
    int head_dim = 0;     // d
    int intermediate = 0; // I (SwiGLU width)
    long long vocab = 0;
    bool tied_embeddings = true;
    int bytes_per_elem = 2; // FP16 / BF16 weights, activations and KV
    long long q_dim() const { return 1LL * heads * head_dim; }
    long long kv_dim() const { return 1LL * kv_heads * head_dim; }
};

// Values copied from the Hugging Face config.json files (accessed 2026-10-18).
inline ModelConfig qwen3_0p6b() {
    return {"Qwen3-0.6B", 28, 1024, 16, 8, 128, 3072, 151936, true, 2};
}
inline ModelConfig qwen3_1p7b() {
    return {"Qwen3-1.7B", 28, 2048, 16, 8, 128, 6144, 151936, true, 2};
}
inline ModelConfig qwen3_4b() {
    return {"Qwen3-4B", 36, 2560, 32, 8, 128, 9728, 151936, true, 2};
}
inline ModelConfig qwen3_8b() {
    return {"Qwen3-8B", 36, 4096, 32, 8, 128, 12288, 151936, false, 2};
}
inline ModelConfig qwen3_32b() {
    return {"Qwen3-32B", 64, 5120, 64, 8, 128, 25600, 151936, false, 2};
}
// Tiny asymmetric config for brute-force oracles (H != H_kv, d*H != h, I not a multiple of h).
inline ModelConfig tiny_model() {
    return {"tiny", 2, 6, 3, 1, 4, 10, 17, true, 2};
}

// ---------------------------------------------------------------- A: cost model types
// One forward pass of ONE decoder layer over a batch of `requests` requests. Each request
// computes `new_tokens` tokens and already has `context` tokens in its KV cache.
// Prefill of a prompt of length s: {R, s, 0}. Decode step at context c: {R, 1, c}.
struct Batch {
    int requests = 1;
    int new_tokens = 1;
    int context = 0;
};
inline Batch prefill_batch(int requests, int prompt) {
    return {requests, prompt, 0};
}
inline Batch decode_batch(int requests, int context) {
    return {requests, 1, context};
}

struct LayerCost {
    double flops = 0;          // 2 * MACs of the projections, MLP and attention (QK^T, PV)
    double weight_bytes = 0;   // every layer parameter read once
    double kv_read_bytes = 0;  // cached K and V rows read once
    double kv_write_bytes = 0; // K and V rows of the new tokens written once
    double act_bytes = 0;      // layer input read + layer output written
    double bytes() const { return weight_bytes + kv_read_bytes + kv_write_bytes + act_bytes; }
    double intensity() const { return bytes() > 0 ? flops / bytes() : 0; }
};

struct GpuSpec {
    std::string name;
    double tflops = 0;  // dense FP16 tensor TFLOPS (never the "with sparsity" number)
    double mem_gib = 0; // capacity in GiB
    double bw_gbs = 0;  // memory bandwidth, GB/s
    std::string interconnect;
    double link_gbs = 0;
    double price_aws = 0;    // $/GPU-hour, 0 = not listed
    double price_runpod = 0; // $/GPU-hour, 0 = not listed
    std::string status, spec_url, price_url, accessed;
    double peak_flops() const { return tflops * 1e12; }
    double peak_bw() const { return bw_gbs * 1e9; }
    double mem_bytes() const { return mem_gib * 1073741824.0; }
    double ridge() const { return peak_flops() / peak_bw(); } // FLOP/byte
};

// Roofline efficiency: achieved = eta * peak; t0 is a fixed per-kernel overhead (launch etc.).
struct Efficiency {
    double eta_c = 1.0;
    double eta_m = 1.0;
    double t0 = 0.0;
};

struct Sample {
    std::string label;
    double flops = 0;
    double bytes = 0;
    double seconds = 0;
};

struct ErrorStats {
    int n = 0;
    double mean_abs_rel = 0;
    double median_abs_rel = 0;
    double max_abs_rel = 0;
    Efficiency fit;
};

struct MemoryBreakdown {
    double weights = 0;
    double kv = 0;
    double activations = 0;
    double total() const { return weights + kv + activations; }
};

struct MemoryFit {
    MemoryBreakdown need;         // at the requested kv_tokens
    bool fits = false;            // reserve + need.total() <= capacity
    long long max_kv_tokens = -1; // largest N that fits, -1 if even N = 0 does not
};

// ---------------------------------------------------------------- B: cluster and flow types
// One GPU machine. rate_per_layer = tokens/s it sustains when it holds exactly one layer
// (holding k layers gives rate_per_layer / k). max_layers = memory limit on k.
struct Node {
    std::string name;
    std::string gpu;
    double rate_per_layer = 0;
    int max_layers = 0;
};

// Link capacities are in tokens/s (bandwidth / bytes moved per token).
// link[i][j]: node i -> node j; from_source[i]: coordinator -> i; to_sink[i]: i -> coordinator.
struct Cluster {
    std::vector<Node> nodes;
    std::vector<std::vector<double>> link;
    std::vector<double> from_source;
    std::vector<double> to_sink;
    int size() const { return static_cast<int>(nodes.size()); }
};

// Layer range [start, end). end == start means the node is unused.
struct Range {
    int start = 0;
    int end = 0;
    bool used() const { return end > start; }
    int size() const { return end - start; }
};
using Placement = std::vector<Range>;

enum class EdgeKind { Source, Compute, Link, Sink };
inline const char *kind_name(EdgeKind k) {
    switch (k) {
    case EdgeKind::Source:
        return "source";
    case EdgeKind::Compute:
        return "compute";
    case EdgeKind::Link:
        return "link";
    default:
        return "sink";
    }
}

struct FlowEdge {
    int u = 0, v = 0;
    double cap = 0;
    EdgeKind kind = EdgeKind::Link;
    int a = -1, b = -1; // GPU indices: Compute (a), Link (a -> b), Source (b), Sink (a)
};

// Vertex numbering is fixed so that tests can inspect graphs:
//   0 = S (coordinator out), 1 = T (coordinator in), 2 + 2i = in(i), 3 + 2i = out(i).
struct FlowGraph {
    int num_nodes = 2;
    std::vector<FlowEdge> edges;
    static int source() { return 0; }
    static int sink() { return 1; }
    static int in(int i) { return 2 + 2 * i; }
    static int out(int i) { return 3 + 2 * i; }
};

struct CutReport {
    std::vector<bool> source_side; // per vertex: reachable from S in the residual graph
    std::vector<int> cut_edges;    // indices into FlowGraph::edges, ascending
    double capacity = 0;
};

struct ChainPlan {
    bool feasible = false;
    std::vector<int> layers; // layers given to chain[k], each >= 1, sum = L
    double bottleneck = 0;   // tokens/s of the chain
};

struct PathFlow {
    std::vector<int> gpus; // GPU indices in pipeline order
    double rate = 0;       // tokens/s routed along this pipeline
};

// ---------------------------------------------------------------- C: co-location types
// Latency model of one model on one GPU type at full GPU time (share = 1).
struct ServingModel {
    std::string name;
    std::function<double(int prompt)> prefill_s;             // whole model, one request
    std::function<double(int batch, double context)> step_s; // one decode step for the batch
    double kv_bytes_per_token = 0;                           // all layers
    double weight_bytes = 0;
};

struct Workload {
    int prompt = 512;
    int output = 128;
};

struct Slo {
    double ttft_s = 1.0;
    double tpot_s = 0.1;
};

struct Capacity {
    bool feasible = false;
    int batch = 0;
    double rate = 0;   // requests/s
    double ttft_s = 0; // prefill latency at this share (no queueing)
    double tpot_s = 0; // observed TPOT including prefill interruptions
};

// A serving template (Coral-style): one GPU of type `gpu` hosting `models` with equal time
// shares; capacity[m] = requests/s it provides for model m (0 if not hosted).
struct Template {
    std::string gpu;
    double price_hr = 0;
    std::vector<int> models;
    std::vector<double> capacity;
    std::vector<int> batch; // decode batch chosen by C1 per model (0 if not hosted)
};

struct Assignment {
    bool feasible = false;
    std::vector<int> count; // instances per template
    double cost_hr = kInf;
    long long nodes = 0; // search nodes visited (B&B statistics)
};

struct Request {
    double arrival = 0;
    int model = 0;
    int prompt = 0;
    int output = 1;
};

struct RequestRecord {
    int id = -1; // index into the request list given to the simulator
    double first_token = 0;
    double finish = 0;
};

// One model hosted on a simulated instance. Latencies are the full-GPU ServingModel ones;
// time sharing between models emerges from the scheduler, not from a fixed share.
struct SimSlot {
    int model = 0;
    int max_batch = 1;
    const ServingModel *lat = nullptr;
};

struct InstanceStats {
    std::vector<RequestRecord> records; // one per request, any order
    double area = 0;                    // integral of (requests arrived and unfinished) dt
    double busy = 0;                    // total time spent in prefill or decode iterations
    double end_time = 0;                // time when the last request finished
};

// ---------------------------------------------------------------- R: profiling types
struct ProfileRow {
    std::string kernel; // copy | gemv | gemm | gemm_cublas
    long long m = 0, n = 0, k = 0;
    double median_ms = 0, q25_ms = 0, q75_ms = 0;
};

// ---------------------------------------------------------------- CSV helpers
inline std::vector<std::string> split_csv(const std::string &line) {
    std::vector<std::string> out(1);
    for (char ch : line) {
        if (ch == ',')
            out.emplace_back();
        else
            out.back().push_back(ch);
    }
    return out;
}

// C stdio on purpose: on Windows, mixed MinGW runtimes on PATH can break std::ifstream.
inline std::vector<std::vector<std::string>> read_csv(const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    int ch;
    auto flush = [&] {
        while (!line.empty() && (line.back() == 13 || line.back() == ' '))
            line.pop_back(); // 13 = CR
        if (!line.empty() && line[0] != '#') rows.push_back(split_csv(line));
        line.clear();
    };
    while ((ch = std::fgetc(f)) != EOF) {
        if (ch == 10)
            flush(); // 10 = LF
        else
            line.push_back(static_cast<char>(ch));
    }
    flush();
    std::fclose(f);
    return rows;
}

inline int column(const std::vector<std::string> &header, const std::string &name) {
    for (size_t i = 0; i < header.size(); ++i)
        if (header[i] == name) return static_cast<int>(i);
    throw std::runtime_error("missing CSV column " + name);
}

inline std::vector<GpuSpec> load_gpus(const std::string &path) {
    auto rows = read_csv(path);
    if (rows.empty()) throw std::runtime_error("empty " + path);
    const auto &h = rows[0];
    std::vector<GpuSpec> out;
    for (size_t r = 1; r < rows.size(); ++r) {
        const auto &c = rows[r];
        GpuSpec g;
        g.name = c.at(column(h, "name"));
        g.tflops = std::stod(c.at(column(h, "dense_fp16_tflops")));
        g.mem_gib = std::stod(c.at(column(h, "mem_gib")));
        g.bw_gbs = std::stod(c.at(column(h, "mem_bw_gbs")));
        g.interconnect = c.at(column(h, "interconnect"));
        g.link_gbs = std::stod(c.at(column(h, "link_gbs")));
        g.price_aws = std::stod(c.at(column(h, "price_aws_hr")));
        g.price_runpod = std::stod(c.at(column(h, "price_runpod_hr")));
        g.status = c.at(column(h, "status"));
        g.spec_url = c.at(column(h, "spec_source_url"));
        g.price_url = c.at(column(h, "price_source_url"));
        g.accessed = c.at(column(h, "accessed"));
        out.push_back(g);
    }
    return out;
}

inline GpuSpec find_gpu(const std::vector<GpuSpec> &gpus, const std::string &name) {
    for (const auto &g : gpus)
        if (g.name == name) return g;
    throw std::runtime_error("unknown GPU " + name);
}

inline std::vector<ProfileRow> load_profile(const std::string &path) {
    auto rows = read_csv(path);
    if (rows.empty()) throw std::runtime_error("empty " + path);
    const auto &h = rows[0];
    std::vector<ProfileRow> out;
    for (size_t r = 1; r < rows.size(); ++r) {
        const auto &c = rows[r];
        ProfileRow p;
        p.kernel = c.at(column(h, "kernel"));
        p.m = std::stoll(c.at(column(h, "m")));
        p.n = std::stoll(c.at(column(h, "n")));
        p.k = std::stoll(c.at(column(h, "k")));
        p.median_ms = std::stod(c.at(column(h, "median_ms")));
        p.q25_ms = std::stod(c.at(column(h, "q25_ms")));
        p.q75_ms = std::stod(c.at(column(h, "q75_ms")));
        out.push_back(p);
    }
    return out;
}

// Deterministic RNG (splitmix64) so that tests behave identically with every compiler.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); } // [0, 1)
    double exponential(double rate) { return -std::log(1.0 - uniform()) / rate; }
    int range(int lo, int hi) { return lo + static_cast<int>(next() % uint64_t(hi - lo + 1)); }
    // Gamma(shape, 1) by Marsaglia-Tsang (shape < 1 boosted).
    double gamma(double shape) {
        if (shape < 1.0) return gamma(shape + 1.0) * std::pow(uniform() + 1e-300, 1.0 / shape);
        double d = shape - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
        for (;;) {
            double x, v;
            do {
                // Box-Muller normal
                double u1 = uniform() + 1e-300, u2 = uniform();
                x = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
                v = 1.0 + c * x;
            } while (v <= 0);
            v = v * v * v;
            double u = uniform();
            if (u < 1 - 0.0331 * x * x * x * x) return d * v;
            if (std::log(u + 1e-300) < 0.5 * x * x + d * (1 - v + std::log(v))) return d * v;
        }
    }
};

} // namespace hs
