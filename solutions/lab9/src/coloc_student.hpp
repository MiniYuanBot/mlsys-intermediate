// Stage C: multi-model co-location under SLOs: capacity, cost-min assignment, simulation.
#pragma once
#include <deque>
#include <map>

#include "cost_student.hpp"
#include "flow_student.hpp"

namespace hs {

// ============================================================================ C1
// slo_capacity: best decode batch b for one model that gets a fraction `share` of a GPU.
//   T_p  = prefill_s(prompt) / share                      (time-shared GPU: everything / share)
//   TPOT(b) = step_s(b, cbar) / share,  cbar = prompt + output / 2   (average context)
//   rate(b) = 1 / (T_p + (output - 1) * TPOT(b) / b)       requests/s (first token = prefill)
//   observed TPOT = TPOT(b) / (1 - rate(b) * T_p)           (decode pauses during prefills)
//   constraints: T_p <= slo.ttft_s; observed TPOT <= slo.tpot_s;
//                b * (prompt + output) * kv_bytes_per_token <= kv_budget_bytes
//   Scan b = 1..max_batch (stop at the first b that breaks the memory constraint) and return
//   the feasible b with the largest rate (ties: smaller b). output == 1: no decode, rate =
//   1 / T_p, batch = 1, tpot = 0 (memory still checked for one request).
// Errors (std::invalid_argument): share outside (0, 1], max_batch < 1, prompt/output < 1.
// Oracle: test.cpp brute-forces b with closed-form latency functions.
inline Capacity slo_capacity(const ServingModel &m, const Workload &w, const Slo &slo, double share,
                             double kv_budget_bytes, int max_batch) {
    // BEGIN TODO C1
    if (!(share > 0) || share > 1 || max_batch < 1 || w.prompt < 1 || w.output < 1)
        throw std::invalid_argument("slo_capacity: bad input");
    Capacity best;
    double tp = m.prefill_s(w.prompt) / share;
    best.ttft_s = tp;
    if (tp > slo.ttft_s) return best;
    double per_req = double(w.prompt + w.output) * m.kv_bytes_per_token;
    if (w.output == 1) {
        if (per_req <= kv_budget_bytes) best = {true, 1, 1.0 / tp, tp, 0.0};
        return best;
    }
    double cbar = w.prompt + w.output / 2.0;
    for (int b = 1; b <= max_batch; ++b) {
        if (b * per_req > kv_budget_bytes) break;
        double tpot = m.step_s(b, cbar) / share;
        double rate = 1.0 / (tp + (w.output - 1) * tpot / b);
        double observed = tpot / (1.0 - rate * tp);
        if (observed <= slo.tpot_s && rate > best.rate * (1 + 1e-12))
            best = {true, b, rate, tp, observed};
    }
    return best;
    // END TODO C1
}

// ============================================================================ C2
// min_cost_assignment: integer covering problem (Coral/Melange style)
//     minimise sum_t price_t * n_t   s.t.  sum_t n_t * capacity_t[m] >= demand[m]  for all m,
//     n_t integer >= 0.
//   Branch and bound over templates in index order. At template t with residual demand r:
//   * lower bound on the extra cost: LB(r) = max_m r_m * min_{t' >= t, cap>0} price_t'/cap_t'[m]
//     (each remaining template covers model m at no less than that unit price; a SUM over
//     models is NOT a valid bound because one instance can serve several models);
//     prune if cost + LB >= best - 1e-9, or if some r_m > 1e-9 has no remaining template;
//   * branch n_t = u, u-1, ..., 0 with u = max_m ceil(r_m / cap_t[m] - 1e-9) over models
//     with r_m > 1e-9 and cap_t[m] > 0 (u = 0 if none).
//   Count every visited search node in `nodes`. Ties: keep the first optimum found.
//   No template can cover some positive demand -> feasible = false (cost kInf, empty count).
// Errors: capacity vector sizes != demand size -> std::invalid_argument.
// Oracle: test.cpp enumerates every count vector on small instances.
inline Assignment min_cost_assignment(const std::vector<Template> &tpl,
                                      const std::vector<double> &demand) {
    // BEGIN TODO C2
    const int T = static_cast<int>(tpl.size()), M = static_cast<int>(demand.size());
    for (const auto &t : tpl)
        if (static_cast<int>(t.capacity.size()) != M) throw std::invalid_argument("capacity size");
    Assignment best;
    std::vector<int> cur(T, 0);
    std::function<void(int, std::vector<double> &, double)> go = [&](int t, std::vector<double> &r,
                                                                     double cost) {
        ++best.nodes;
        bool done = true;
        double lb = 0;
        for (int m = 0; m < M; ++m) {
            if (r[m] <= 1e-9) continue;
            done = false;
            double unit = kInf;
            for (int u = t; u < T; ++u)
                if (tpl[u].capacity[m] > 0)
                    unit = std::min(unit, tpl[u].price_hr / tpl[u].capacity[m]);
            if (unit == kInf) return;
            lb = std::max(lb, r[m] * unit);
        }
        if (done) {
            if (cost < best.cost_hr - 1e-9)
                best.cost_hr = cost, best.count = cur, best.feasible = true;
            return;
        }
        if (t == T || cost + lb >= best.cost_hr - 1e-9) return;
        int upper = 0;
        for (int m = 0; m < M; ++m)
            if (r[m] > 1e-9 && tpl[t].capacity[m] > 0)
                upper =
                    std::max(upper, static_cast<int>(std::ceil(r[m] / tpl[t].capacity[m] - 1e-9)));
        for (int k = upper; k >= 0; --k) {
            std::vector<double> next = r;
            for (int m = 0; m < M; ++m) next[m] -= k * tpl[t].capacity[m];
            cur[t] = k;
            go(t + 1, next, cost + k * tpl[t].price_hr);
        }
        cur[t] = 0;
    };
    std::vector<double> r = demand;
    go(0, r, 0.0);
    return best;
    // END TODO C2
}

// ============================================================================ C3
// simulate_instance: iteration-level discrete-event simulation of ONE GPU instance that
// hosts `slots` (one per model). `reqs` must be sorted by arrival (else invalid_argument);
// every request's model must have exactly one slot.
//   Loop until every request finished:
//   * admit all requests with arrival <= t into their slot's FIFO waiting queue;
//   * choose the next slot round-robin (starting after the previously chosen one) among
//     slots that have work: (waiting non-empty and active < max_batch) or active non-empty;
//     if none has work, jump t to the next arrival;
//   * chosen slot: if it can admit, PREFILL the oldest waiting request: dt = prefill_s(prompt),
//     its first token appears at t + dt (it finishes there if output == 1, else it becomes
//     active with 1 generated token); otherwise one DECODE step for all its active requests:
//     dt = step_s(n_active, mean context), context of a request = prompt + generated, every
//     active request gains one token and finishes at t + dt once generated == output;
//   * area += (requests in system during [t, t + dt]) integrated exactly, including requests
//     that arrive inside the iteration (they are only admitted at its end); busy += dt.
//   records[i] belongs to reqs[i]; end_time = finish time of the last request.
// Oracles: M/D/1 Pollaczek-Khinchine mean wait; Little's law area = sum(finish - arrival);
//   completed requests/s <= 1 / (T_p + (o - 1) * step(b, p) / b).
inline InstanceStats simulate_instance(const std::vector<SimSlot> &slots,
                                       const std::vector<Request> &reqs) {
    // BEGIN TODO C3
    const size_t S = slots.size(), n = reqs.size();
    if (S == 0) throw std::invalid_argument("simulate_instance: no slots");
    std::map<int, size_t> slot_of;
    for (size_t s = 0; s < S; ++s) {
        if (!slots[s].lat || slots[s].max_batch < 1) throw std::invalid_argument("bad slot");
        if (!slot_of.emplace(slots[s].model, s).second)
            throw std::invalid_argument("duplicate model");
    }
    for (size_t i = 0; i < n; ++i) {
        if (i && reqs[i].arrival < reqs[i - 1].arrival)
            throw std::invalid_argument("unsorted trace");
        if (!slot_of.count(reqs[i].model) || reqs[i].output < 1 || reqs[i].prompt < 1)
            throw std::invalid_argument("bad request");
    }
    struct Act {
        size_t id;
        int generated;
    };
    std::vector<std::deque<size_t>> waiting(S);
    std::vector<std::vector<Act>> active(S);
    InstanceStats st;
    st.records.resize(n);
    for (size_t i = 0; i < n; ++i) st.records[i].id = static_cast<int>(i);
    size_t next = 0, done = 0, rr = 0;
    long long in_system = 0;
    double t = 0;
    while (done < n) {
        while (next < n && reqs[next].arrival <= t) {
            waiting[slot_of[reqs[next].model]].push_back(next);
            ++in_system, ++next;
        }
        int chosen = -1;
        for (size_t k = 0; k < S; ++k) {
            size_t s = (rr + k) % S;
            bool can_prefill =
                !waiting[s].empty() && static_cast<int>(active[s].size()) < slots[s].max_batch;
            if (can_prefill || !active[s].empty()) {
                chosen = static_cast<int>(s);
                break;
            }
        }
        if (chosen < 0) {
            if (next >= n) throw std::logic_error("simulate_instance: stuck");
            t = std::max(t, reqs[next].arrival);
            continue;
        }
        rr = (chosen + 1) % S;
        const SimSlot &sl = slots[chosen];
        std::vector<size_t> finished;
        double dt;
        if (!waiting[chosen].empty() && static_cast<int>(active[chosen].size()) < sl.max_batch) {
            size_t id = waiting[chosen].front();
            waiting[chosen].pop_front();
            dt = sl.lat->prefill_s(reqs[id].prompt);
            st.records[id].first_token = t + dt;
            if (reqs[id].output <= 1)
                finished.push_back(id);
            else
                active[chosen].push_back({id, 1});
        } else {
            auto &act = active[chosen];
            double ctx = 0;
            for (const auto &a : act) ctx += reqs[a.id].prompt + a.generated;
            ctx /= act.size();
            dt = sl.lat->step_s(static_cast<int>(act.size()), ctx);
            std::vector<Act> keep;
            for (auto a : act) {
                ++a.generated;
                if (a.generated >= reqs[a.id].output)
                    finished.push_back(a.id);
                else
                    keep.push_back(a);
            }
            act.swap(keep);
        }
        double end = t + dt;
        st.area += in_system * dt;
        for (size_t j = next; j < n && reqs[j].arrival < end; ++j) st.area += end - reqs[j].arrival;
        st.busy += dt;
        t = end;
        for (size_t id : finished) {
            st.records[id].finish = t;
            --in_system, ++done;
        }
        st.end_time = t;
    }
    return st;
    // END TODO C3
}

// ============================================================================ C4
struct Baselines {
    Assignment greedy;      // per model, cheapest $/capacity singleton template, ceil count
    Assignment no_coloc;    // optimum over singleton templates only
    Assignment homogeneous; // optimum over templates of ONE GPU type (best type)
    std::string homogeneous_gpu;
};

// baseline_plans: the three baselines every co-location result must be compared with.
//   greedy:      for each model m with demand > 0 independently, among templates hosting
//                only m with capacity > 0 pick the lowest price/capacity (ties: lower
//                index) and add ceil(demand / capacity - 1e-9) instances; any model without
//                such a template -> greedy infeasible.
//   no_coloc:    min_cost_assignment restricted to single-model templates.
//   homogeneous: for each GPU name in order of first appearance, min_cost_assignment
//                restricted to that GPU's templates; keep the cheapest feasible (ties: first).
//   Counts are always reported in the FULL template index space (0 for excluded ones).
// Oracle: test.cpp recomputes greedy by hand and brute-forces the restricted problems.
inline Baselines baseline_plans(const std::vector<Template> &tpl,
                                const std::vector<double> &demand) {
    // BEGIN TODO C4
    const size_t T = tpl.size(), M = demand.size();
    auto restricted = [&](const std::function<bool(const Template &)> &keep) {
        std::vector<Template> sub;
        std::vector<size_t> idx;
        for (size_t t = 0; t < T; ++t)
            if (keep(tpl[t])) sub.push_back(tpl[t]), idx.push_back(t);
        Assignment a = min_cost_assignment(sub, demand);
        Assignment full = a;
        full.count.assign(T, 0);
        if (a.feasible)
            for (size_t k = 0; k < idx.size(); ++k) full.count[idx[k]] = a.count[k];
        return full;
    };
    Baselines b;
    b.greedy.count.assign(T, 0);
    b.greedy.feasible = true;
    b.greedy.cost_hr = 0;
    for (size_t m = 0; m < M; ++m) {
        if (demand[m] <= 1e-9) continue;
        int pick = -1;
        for (size_t t = 0; t < T; ++t) {
            if (tpl[t].models.size() != 1 || tpl[t].models[0] != static_cast<int>(m) ||
                tpl[t].capacity[m] <= 0)
                continue;
            if (pick < 0 ||
                tpl[t].price_hr / tpl[t].capacity[m] < tpl[pick].price_hr / tpl[pick].capacity[m])
                pick = static_cast<int>(t);
        }
        if (pick < 0) {
            b.greedy = Assignment{};
            break;
        }
        int k = static_cast<int>(std::ceil(demand[m] / tpl[pick].capacity[m] - 1e-9));
        b.greedy.count[pick] += k;
        b.greedy.cost_hr += k * tpl[pick].price_hr;
    }
    b.no_coloc = restricted([](const Template &t) { return t.models.size() == 1; });
    std::vector<std::string> names;
    for (const auto &t : tpl)
        if (std::find(names.begin(), names.end(), t.gpu) == names.end()) names.push_back(t.gpu);
    for (const auto &name : names) {
        Assignment a = restricted([&](const Template &t) { return t.gpu == name; });
        if (a.feasible && (!b.homogeneous.feasible || a.cost_hr < b.homogeneous.cost_hr - 1e-9))
            b.homogeneous = a, b.homogeneous_gpu = name;
    }
    if (!b.homogeneous.feasible) b.homogeneous.count.assign(T, 0);
    return b;
    // END TODO C4
}

// ---------------------------------------------------------------- provided helpers
inline double dollars_per_million_tokens(double cost_per_hour, double tokens_per_second) {
    if (!(tokens_per_second > 0)) return kInf;
    return cost_per_hour / (tokens_per_second * 3600.0) * 1e6;
}

// Coral-style templates: every subset of <= max_group models on every GPU type, equal time
// shares, KV budget split equally, capacity from C1. Modeled, not measured.
struct TemplateInputs {
    std::vector<ModelConfig> models;
    std::vector<Workload> work;
    std::vector<Slo> slo;
    std::vector<GpuSpec> gpus;
    std::function<double(const GpuSpec &)> price;
    std::function<Efficiency(const GpuSpec &)> eff;
    int max_group = 2;
    int max_batch = 256;
    double reserve_bytes = 1.5 * 1073741824.0;
};

inline std::vector<Template> make_templates(const TemplateInputs &in) {
    std::vector<Template> out;
    const int M = static_cast<int>(in.models.size());
    for (const auto &g : in.gpus) {
        double price = in.price(g);
        if (!(price > 0)) continue;
        Efficiency e = in.eff(g);
        for (int mask = 1; mask < (1 << M); ++mask) {
            std::vector<int> ms;
            for (int m = 0; m < M; ++m)
                if (mask >> m & 1) ms.push_back(m);
            if (static_cast<int>(ms.size()) > in.max_group) continue;
            double weights = 0, act = 0;
            for (int m : ms) {
                weights +=
                    memory_fit(in.models[m], in.models[m].layers, 0, 0, kInf, 0, true).need.weights;
                act =
                    std::max(act, memory_fit(in.models[m], 1, 0, in.work[m].prompt, kInf, 0, false)
                                      .need.activations);
            }
            double free = g.mem_bytes() - in.reserve_bytes - weights - act;
            if (free <= 0) continue;
            Template t{g.name, price, ms, std::vector<double>(M, 0.0), std::vector<int>(M, 0)};
            bool ok = true;
            for (int m : ms) {
                ServingModel sm = serving_model(in.models[m], g, e);
                Capacity cap = slo_capacity(sm, in.work[m], in.slo[m], 1.0 / ms.size(),
                                            free / ms.size(), in.max_batch);
                if (!cap.feasible) ok = false;
                t.capacity[m] = cap.rate, t.batch[m] = cap.batch;
            }
            if (ok) out.push_back(t);
        }
    }
    return out;
}

// Request traces. Poisson: exponential gaps. Bursty: Gamma gaps with coefficient of
// variation cv (shape 1/cv^2), same mean rate.
inline std::vector<Request> make_trace(const std::vector<double> &rate,
                                       const std::vector<Workload> &work, double duration,
                                       double cv, uint64_t seed) {
    std::vector<Request> out;
    Rng rng(seed);
    for (size_t m = 0; m < rate.size(); ++m) {
        if (!(rate[m] > 0)) continue;
        double t = 0, shape = 1.0 / (cv * cv);
        for (;;) {
            t += cv == 1.0 ? rng.exponential(rate[m]) : rng.gamma(shape) / (shape * rate[m]);
            if (t > duration) break;
            out.push_back({t, static_cast<int>(m), work[m].prompt, work[m].output});
        }
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const Request &a, const Request &b) { return a.arrival < b.arrival; });
    return out;
}

struct Deployment {
    std::vector<ServingModel> lat; // owned latency models
    std::vector<std::vector<SimSlot>> instances;
    std::vector<std::vector<double>> weight; // routing weight [instance][model]
    double cost_hr = 0;
};

// Build simulated instances for an assignment. `lat_of(gpu, model)` returns the latency model.
inline Deployment
make_deployment(const std::vector<Template> &tpl, const Assignment &a, int M,
                const std::function<ServingModel(const std::string &, int)> &lat_of) {
    Deployment d;
    std::vector<std::pair<std::string, int>> keys;
    for (size_t t = 0; t < tpl.size(); ++t)
        if (a.count[t] > 0)
            for (int m : tpl[t].models)
                if (std::find(keys.begin(), keys.end(), std::make_pair(tpl[t].gpu, m)) ==
                    keys.end())
                    keys.push_back({tpl[t].gpu, m});
    for (const auto &k : keys) d.lat.push_back(lat_of(k.first, k.second));
    for (size_t t = 0; t < tpl.size(); ++t)
        for (int c = 0; c < a.count[t]; ++c) {
            std::vector<SimSlot> slots;
            std::vector<double> w(M, 0.0);
            for (int m : tpl[t].models) {
                size_t k = std::find(keys.begin(), keys.end(), std::make_pair(tpl[t].gpu, m)) -
                           keys.begin();
                slots.push_back({m, std::max(1, tpl[t].batch[m]), &d.lat[k]});
                w[m] = tpl[t].capacity[m];
            }
            d.instances.push_back(slots);
            d.weight.push_back(w);
            d.cost_hr += tpl[t].price_hr;
        }
    return d;
}

struct SimSummary {
    int requests = 0;
    double makespan = 0, tokens = 0, throughput_tok_s = 0;
    double slo_attainment = 0, mean_ttft = 0, p99_ttft = 0, mean_tpot = 0;
    double little_L = 0, little_lambda_W = 0;
};

// Route each request with one Iwrr per model (weights = instance capacity), simulate every
// instance with C3 and aggregate.
inline SimSummary simulate(const Deployment &d, const std::vector<Request> &trace,
                           const std::vector<Slo> &slo) {
    const size_t I = d.instances.size();
    SimSummary out;
    if (I == 0 || trace.empty()) return out;
    size_t M = d.weight[0].size();
    std::vector<std::vector<size_t>> per(I);
    for (size_t m = 0; m < M; ++m) {
        std::vector<double> w(I);
        for (size_t i = 0; i < I; ++i) w[i] = d.weight[i][m];
        bool any = std::any_of(w.begin(), w.end(), [](double x) { return x > 0; });
        if (!any) continue;
        Iwrr rr(w);
        for (size_t r = 0; r < trace.size(); ++r)
            if (trace[r].model == static_cast<int>(m)) per[rr.next()].push_back(r);
    }
    std::vector<double> ttft, tpot;
    double area = 0, sojourn = 0;
    int ok = 0;
    for (size_t i = 0; i < I; ++i) {
        std::sort(per[i].begin(), per[i].end());
        std::vector<Request> sub;
        for (size_t r : per[i]) sub.push_back(trace[r]);
        if (sub.empty()) continue;
        InstanceStats st = simulate_instance(d.instances[i], sub);
        area += st.area;
        out.makespan = std::max(out.makespan, st.end_time);
        for (const auto &rec : st.records) {
            const Request &q = sub[rec.id];
            double a = rec.first_token - q.arrival;
            double b = q.output > 1 ? (rec.finish - rec.first_token) / (q.output - 1) : 0.0;
            ttft.push_back(a), tpot.push_back(b);
            sojourn += rec.finish - q.arrival;
            out.tokens += q.output;
            ok += a <= slo[q.model].ttft_s && b <= slo[q.model].tpot_s;
        }
    }
    out.requests = static_cast<int>(ttft.size());
    if (out.requests == 0) return out;
    out.throughput_tok_s = out.tokens / out.makespan;
    out.slo_attainment = double(ok) / out.requests;
    out.mean_ttft = std::accumulate(ttft.begin(), ttft.end(), 0.0) / out.requests;
    out.mean_tpot = std::accumulate(tpot.begin(), tpot.end(), 0.0) / out.requests;
    std::sort(ttft.begin(), ttft.end());
    out.p99_ttft = ttft[std::min(ttft.size() - 1, static_cast<size_t>(0.99 * ttft.size()))];
    out.little_L = area / out.makespan;
    out.little_lambda_W = (out.requests / out.makespan) * (sojourn / out.requests);
    return out;
}

} // namespace hs
