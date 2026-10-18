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
    throw std::logic_error("Complete B1");
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
        throw std::logic_error("Complete B2");
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
    throw std::logic_error("Complete B3");
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
    throw std::logic_error("Complete B4");
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
    throw std::logic_error("Complete B5");
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
    throw std::logic_error("Complete B6");
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
        throw std::logic_error("Complete B7");
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
