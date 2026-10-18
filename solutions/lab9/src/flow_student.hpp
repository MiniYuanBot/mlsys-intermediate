// Stage B: Helix-style layer placement over a heterogeneous cluster via max-flow.
#pragma once
#include <deque>

#include "types.hpp"

namespace hs {

constexpr double kFlowEps = 1e-9;

// ============================================================================ B1
// build_flow_graph: the token-flow network of a placement (vertex numbering in types.hpp).
//   For every used node i (range [s_i, e_i), k_i = e_i - s_i):
//     Compute edge in(i) -> out(i), cap = rate_per_layer_i / k_i   (node splitting)
//     Source  edge S -> in(i),  cap = from_source[i]   iff s_i == 0
//     Sink    edge out(i) -> T, cap = to_sink[i]       iff e_i == L
//   For every ordered pair of used nodes i != j with e_i == s_j and link[i][j] > 0:
//     Link    edge out(i) -> in(j), cap = link[i][j]
//   Unused nodes get no edges (their two vertices still exist). Fill FlowEdge::kind, a, b.
// Errors (std::invalid_argument): placement size != cluster size, a range outside [0, L],
//   end < start, or k_i > max_layers (does not fit in memory).
// Oracle: test.cpp rebuilds the graph as a capacity matrix and compares edge multisets.
inline FlowGraph build_flow_graph(const Cluster &c, const Placement &p, int L) {
    // BEGIN TODO B1
    int n = c.size();
    if (static_cast<int>(p.size()) != n) throw std::invalid_argument("placement size");
    for (int i = 0; i < n; ++i) {
        if (p[i].start < 0 || p[i].end > L || p[i].end < p[i].start)
            throw std::invalid_argument("bad range");
        if (p[i].size() > c.nodes[i].max_layers)
            throw std::invalid_argument("range exceeds memory");
    }
    FlowGraph g;
    g.num_nodes = 2 + 2 * n;
    for (int i = 0; i < n; ++i) {
        if (!p[i].used()) continue;
        g.edges.push_back({FlowGraph::in(i), FlowGraph::out(i),
                           c.nodes[i].rate_per_layer / p[i].size(), EdgeKind::Compute, i, -1});
        if (p[i].start == 0)
            g.edges.push_back(
                {FlowGraph::source(), FlowGraph::in(i), c.from_source[i], EdgeKind::Source, -1, i});
        if (p[i].end == L)
            g.edges.push_back(
                {FlowGraph::out(i), FlowGraph::sink(), c.to_sink[i], EdgeKind::Sink, i, -1});
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j && p[i].used() && p[j].used() && p[i].end == p[j].start && c.link[i][j] > 0)
                g.edges.push_back(
                    {FlowGraph::out(i), FlowGraph::in(j), c.link[i][j], EdgeKind::Link, i, j});
    return g;
    // END TODO B1
}

// ============================================================================ B2
// Dinic max-flow on a residual graph. add_edge (provided) stores arc 2k (u -> v, residual
// = cap) and its reverse arc 2k+1 (v -> u, residual 0) and returns k.
// max_flow(s, t): repeat { BFS levels from s over arcs with residual > kFlowEps; stop if t
//   is unreachable; push a blocking flow with DFS that only follows level[v] == level[u]+1
//   and keeps a per-vertex arc iterator } and return the total flow pushed. Every push of f
//   on arc a must decrease res[a] and increase res[a ^ 1] by f. Calling max_flow twice
//   returns 0 the second time (the flow is already maximal).
// Oracle: test.cpp enumerates all 2^(n-2) s-t cuts (n <= 12) and checks max-flow = min-cut,
//   plus capacity and conservation of flow(k) on every edge.
class Dinic {
public:
    explicit Dinic(int n) : adj_(n) {}
    int add_edge(int u, int v, double cap) {
        if (cap < 0) throw std::invalid_argument("negative capacity");
        int k = static_cast<int>(cap_.size());
        to_.push_back(v), res_.push_back(cap), adj_[u].push_back(2 * k);
        to_.push_back(u), res_.push_back(0.0), adj_[v].push_back(2 * k + 1);
        cap_.push_back(cap);
        return k;
    }
    double flow(int k) const { return cap_[k] - res_[2 * k]; }
    int vertices() const { return static_cast<int>(adj_.size()); }

    double max_flow(int s, int t) {
        // BEGIN TODO B2
        int n = vertices();
        double total = 0;
        std::vector<int> level(n), it(n);
        std::function<double(int, double)> dfs = [&](int u, double pushed) -> double {
            if (u == t) return pushed;
            for (int &i = it[u]; i < static_cast<int>(adj_[u].size()); ++i) {
                int a = adj_[u][i], v = to_[a];
                if (res_[a] > kFlowEps && level[v] == level[u] + 1) {
                    double got = dfs(v, std::min(pushed, res_[a]));
                    if (got > kFlowEps) {
                        res_[a] -= got;
                        res_[a ^ 1] += got;
                        return got;
                    }
                }
            }
            return 0.0;
        };
        for (;;) {
            std::fill(level.begin(), level.end(), -1);
            std::deque<int> q{s};
            level[s] = 0;
            while (!q.empty()) {
                int u = q.front();
                q.pop_front();
                for (int a : adj_[u])
                    if (res_[a] > kFlowEps && level[to_[a]] < 0) {
                        level[to_[a]] = level[u] + 1;
                        q.push_back(to_[a]);
                    }
            }
            if (level[t] < 0) break;
            std::fill(it.begin(), it.end(), 0);
            while (double f = dfs(s, kInf)) total += f;
        }
        return total;
        // END TODO B2
    }

private:
    std::vector<std::vector<int>> adj_;
    std::vector<int> to_;
    std::vector<double> res_, cap_;
};

// Provided: load a FlowGraph into a Dinic object (edge k of the graph = Dinic edge k).
inline Dinic make_dinic(const FlowGraph &g) {
    Dinic d(g.num_nodes);
    for (const auto &e : g.edges) d.add_edge(e.u, e.v, e.cap);
    return d;
}

// ============================================================================ B3
// min_cut: after `solved.max_flow(S, T)`, find the vertices reachable from S in the residual
//   graph (forward residual cap - flow > kFlowEps, backward residual flow > kFlowEps) and
//   report the edges going from the reachable side to the other side (ascending index) and
//   the sum of their capacities. With a maximum flow this sum equals the flow value and
//   names the bottleneck (compute edges = GPUs, link edges = network).
// Oracle: test.cpp checks the cut separates S from T, its value equals the enumerated
//   minimum cut, and each reported edge is saturated.
inline CutReport min_cut(const FlowGraph &g, const Dinic &solved) {
    // BEGIN TODO B3
    CutReport r;
    r.source_side.assign(g.num_nodes, false);
    std::deque<int> q{FlowGraph::source()};
    r.source_side[FlowGraph::source()] = true;
    while (!q.empty()) {
        int u = q.front();
        q.pop_front();
        for (size_t k = 0; k < g.edges.size(); ++k) {
            const auto &e = g.edges[k];
            double f = solved.flow(static_cast<int>(k));
            int next = -1;
            if (e.u == u && e.cap - f > kFlowEps) next = e.v;
            if (e.v == u && f > kFlowEps) next = e.u;
            if (next >= 0 && !r.source_side[next]) {
                r.source_side[next] = true;
                q.push_back(next);
            }
        }
    }
    for (size_t k = 0; k < g.edges.size(); ++k)
        if (r.source_side[g.edges[k].u] && !r.source_side[g.edges[k].v]) {
            r.cut_edges.push_back(static_cast<int>(k));
            r.capacity += g.edges[k].cap;
        }
    return r;
    // END TODO B3
}

// Provided: throughput upper bound of a placement = max-flow value (0 if infeasible).
inline double placement_throughput(const Cluster &c, const Placement &p, int L) {
    FlowGraph g = build_flow_graph(c, p, L);
    Dinic d = make_dinic(g);
    return d.max_flow(FlowGraph::source(), FlowGraph::sink());
}

// ============================================================================ B4
// partition_chain: one pipeline through the nodes `chain` (in this order). Give chain[j]
//   k_j >= 1 consecutive layers (k_j <= max_layers, sum k_j = L) to maximise the chain
//   bottleneck
//     min( from_source[chain[0]], to_sink[chain.back()],
//          min_j rate_per_layer[chain[j]] / k_j,  min_j link[chain[j]][chain[j+1]] ).
//   Use DP: f[j][l] = best bottleneck when the first j nodes hold exactly the first l layers,
//   f[j][l] = max_k min(f[j-1][l-k], rate_j / k, link into node j), k = 1..max_layers_j.
//   Ties: prefer the smaller k for the LAST node considered (scan k ascending, replace only
//   on strict improvement). Infeasible (empty chain, more nodes than layers, memory too
//   small) -> feasible = false. A feasible chain may still have bottleneck 0 (dead link).
// Oracle: test.cpp enumerates every composition of L.
inline ChainPlan partition_chain(const Cluster &c, const std::vector<int> &chain, int L) {
    // BEGIN TODO B4
    ChainPlan plan;
    int m = static_cast<int>(chain.size());
    if (m == 0 || m > L) return plan;
    const double NEG = -1.0;
    std::vector<std::vector<double>> f(m + 1, std::vector<double>(L + 1, NEG));
    std::vector<std::vector<int>> pick(m + 1, std::vector<int>(L + 1, 0));
    f[0][0] = c.from_source[chain[0]];
    for (int j = 1; j <= m; ++j) {
        const Node &node = c.nodes[chain[j - 1]];
        double in_link = j == 1 ? kInf : c.link[chain[j - 2]][chain[j - 1]];
        for (int l = j; l <= L; ++l)
            for (int k = 1; k <= std::min(node.max_layers, l); ++k) {
                double prev = f[j - 1][l - k];
                if (prev < 0) continue;
                double v = std::min({prev, in_link, node.rate_per_layer / k});
                if (v > f[j][l]) f[j][l] = v, pick[j][l] = k;
            }
    }
    if (f[m][L] < 0) return plan;
    plan.feasible = true;
    plan.bottleneck = std::min(f[m][L], c.to_sink[chain.back()]);
    plan.layers.assign(m, 0);
    for (int j = m, l = L; j >= 1; --j) {
        plan.layers[j - 1] = pick[j][l];
        l -= pick[j][l];
    }
    return plan;
    // END TODO B4
}

// ============================================================================ B5
// place_layers: a placement (one Range per node) maximising placement_throughput.
//   1. Seeds (each is a set of disjoint chains partitioned with B4; infeasible chains are
//      dropped):
//      - deterministic: sort nodes by rate_per_layer * max_layers (descending, stable) and
//        for p = 1..n deal them in snake order into p chains;
//      - randomised: `restarts` random permutations (Rng(1234 + r)), each dealt round-robin
//        into p = 1 + rng.next() % n chains.
//   2. Local search from the `starts` best seeds (ties: earlier seed), best improvement
//      (> 1e-9 relative), at most 200 sweeps each. Each sweep first tries
//      - single moves: node i takes any other range (unused, or [s, e) with
//        1 <= e - s <= max_layers);
//      and only if none improves,
//      - pair moves: node i takes a used range r AND one other node j takes either the same
//        range r (a replica) or a range that abuts r ([r.end, r.end + len) or
//        [r.start - len, r.start), len <= max_layers_j).
//      Pair moves matter: replicas that feed one node (a DAG, not disjoint chains) are often
//      optimal, and reaching them needs two coordinated changes.
//   Return the best placement found (all-unused if nothing carries flow).
// Oracle: exhaustive search over all placements for <= 5 nodes x 8 layers; the test asks for
//   >= 97% of the optimum (the search is a heuristic; Helix solves a MILP instead).
inline Placement place_layers(const Cluster &c, int L, int restarts = 24, int starts = 6) {
    // BEGIN TODO B5
    const int n = c.size();
    auto chains_to_placement = [&](const std::vector<std::vector<int>> &chains) {
        Placement cand(n);
        for (const auto &ch : chains) {
            if (ch.empty()) continue;
            ChainPlan cp = partition_chain(c, ch, L);
            if (!cp.feasible) continue;
            int s = 0;
            for (size_t k = 0; k < ch.size(); ++k)
                cand[ch[k]] = {s, s + cp.layers[k]}, s += cp.layers[k];
        }
        return cand;
    };
    std::vector<std::pair<double, Placement>> seeds;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return c.nodes[a].rate_per_layer * c.nodes[a].max_layers >
               c.nodes[b].rate_per_layer * c.nodes[b].max_layers;
    });
    for (int p = 1; p <= n; ++p) {
        std::vector<std::vector<int>> chains(p);
        for (int i = 0; i < n; ++i) {
            int round = i / p, slot = i % p;
            chains[round % 2 ? p - 1 - slot : slot].push_back(order[i]);
        }
        Placement cand = chains_to_placement(chains);
        seeds.push_back({placement_throughput(c, cand, L), cand});
    }
    for (int r = 0; r < restarts; ++r) {
        Rng rng(1234 + r);
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        for (int i = n - 1; i > 0; --i) std::swap(perm[i], perm[rng.next() % (i + 1)]);
        int p = 1 + static_cast<int>(rng.next() % n);
        std::vector<std::vector<int>> chains(p);
        for (int i = 0; i < n; ++i) chains[i % p].push_back(perm[i]);
        Placement cand = chains_to_placement(chains);
        seeds.push_back({placement_throughput(c, cand, L), cand});
    }
    std::stable_sort(seeds.begin(), seeds.end(),
                     [](const auto &a, const auto &b) { return a.first > b.first; });
    auto options = [&](int i) {
        std::vector<Range> out{{0, 0}};
        for (int s = 0; s < L; ++s)
            for (int e = s + 1; e <= std::min(L, s + c.nodes[i].max_layers); ++e)
                out.push_back({s, e});
        return out;
    };
    Placement best(n);
    double best_v = 0;
    for (int st = 0; st < std::min<int>(starts, static_cast<int>(seeds.size())); ++st) {
        Placement cur = seeds[st].second;
        double cur_v = seeds[st].first;
        for (int sweep = 0; sweep < 200; ++sweep) {
            Placement pick = cur;
            double pick_v = cur_v;
            auto consider = [&](const Placement &cand) {
                double v = placement_throughput(c, cand, L);
                if (v > pick_v * (1 + 1e-9) + 1e-12) pick = cand, pick_v = v;
            };
            for (int i = 0; i < n; ++i)
                for (const Range &r : options(i)) {
                    if (r.start == cur[i].start && r.end == cur[i].end) continue;
                    Placement cand = cur;
                    cand[i] = r;
                    consider(cand);
                }
            if (!(pick_v > cur_v)) // single moves are stuck: try coordinated pair moves
                for (int i = 0; i < n; ++i)
                    for (const Range &r : options(i)) {
                        if (!r.used()) continue;
                        Placement cand = cur;
                        cand[i] = r;
                        for (int j = 0; j < n; ++j) {
                            if (j == i) continue;
                            Placement replica = cand;
                            if (r.size() <= c.nodes[j].max_layers)
                                replica[j] = r, consider(replica);
                            for (int len = 1; len <= c.nodes[j].max_layers; ++len) {
                                Placement pair = cand;
                                if (r.end + len <= L)
                                    pair[j] = {r.end, r.end + len}, consider(pair);
                                if (r.start - len >= 0)
                                    pair[j] = {r.start - len, r.start}, consider(pair);
                            }
                        }
                    }
            if (!(pick_v > cur_v)) break;
            cur = pick, cur_v = pick_v;
        }
        if (cur_v > best_v * (1 + 1e-9) + 1e-12) best = cur, best_v = cur_v;
    }
    return best;
    // END TODO B5
}

// ============================================================================ B6
// decompose_flow: split a feasible S-T flow (edge_flow[k] on g.edges[k]) into pipelines.
//   Repeat while S still has an outgoing edge with flow > kFlowEps: walk from S, always
//   taking the lowest-index outgoing edge with remaining flow > kFlowEps, until T; the
//   path rate is the minimum remaining flow on the walk; subtract it on every edge of the
//   walk; record the GPUs of the Compute edges in order. A walk that gets stuck before T
//   (flow not conserved) -> std::runtime_error.
// Properties: rates sum to the flow value; summing path rates per edge reproduces
//   edge_flow; at most |E| paths (each subtraction zeroes at least one edge).
inline std::vector<PathFlow> decompose_flow(const FlowGraph &g, std::vector<double> edge_flow) {
    // BEGIN TODO B6
    std::vector<PathFlow> paths;
    auto next_edge = [&](int u) {
        for (size_t k = 0; k < g.edges.size(); ++k)
            if (g.edges[k].u == u && edge_flow[k] > kFlowEps) return static_cast<int>(k);
        return -1;
    };
    while (next_edge(FlowGraph::source()) >= 0) {
        std::vector<int> walk;
        int u = FlowGraph::source();
        while (u != FlowGraph::sink()) {
            int k = next_edge(u);
            if (k < 0 || static_cast<int>(walk.size()) > static_cast<int>(g.edges.size()))
                throw std::runtime_error("decompose_flow: flow not conserved");
            walk.push_back(k);
            u = g.edges[k].v;
        }
        double rate = kInf;
        for (int k : walk) rate = std::min(rate, edge_flow[k]);
        PathFlow p;
        p.rate = rate;
        for (int k : walk) {
            edge_flow[k] -= rate;
            if (g.edges[k].kind == EdgeKind::Compute) p.gpus.push_back(g.edges[k].a);
        }
        paths.push_back(p);
    }
    return paths;
    // END TODO B6
}

// ============================================================================ B7
// Iwrr: interleaved (smooth) weighted round robin over pipelines, used to dispatch each
// new request to a pipeline. Weights >= 0 with a positive sum (constructor checks).
// next(): add w_i to every current_i, pick the largest current (ties: lowest index),
//   subtract the total weight from the picked one, return its index.
// Properties checked by test.cpp: over any prefix of length t, count_i stays within 1 of
//   t * w_i / sum(w); zero-weight entries are never picked; the sequence is periodic for
//   integer weights.
class Iwrr {
public:
    explicit Iwrr(std::vector<double> weights) : w_(std::move(weights)), cur_(w_.size(), 0.0) {
        total_ = 0;
        for (double x : w_) {
            if (x < 0) throw std::invalid_argument("Iwrr: negative weight");
            total_ += x;
        }
        if (!(total_ > 0)) throw std::invalid_argument("Iwrr: zero total weight");
    }
    int next() {
        // BEGIN TODO B7
        int pick = 0;
        for (size_t i = 0; i < w_.size(); ++i) {
            cur_[i] += w_[i];
            if (cur_[i] > cur_[pick]) pick = static_cast<int>(i);
        }
        cur_[pick] -= total_;
        return pick;
        // END TODO B7
    }

private:
    std::vector<double> w_, cur_;
    double total_ = 0;
};

// Provided: the per-edge flow of a solved graph, in FlowGraph edge order.
inline std::vector<double> edge_flows(const FlowGraph &g, const Dinic &d) {
    std::vector<double> f(g.edges.size());
    for (size_t k = 0; k < f.size(); ++k) f[k] = d.flow(static_cast<int>(k));
    return f;
}

} // namespace hs
