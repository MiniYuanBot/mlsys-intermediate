//! C1-C5: admission, the full-hit rule, cache-aware routing across replicas, the
//! multi-replica event replay and its metrics.
use crate::cost::{CostModel, FullHitRule};
use crate::{block_keys, Cache, Key, Lease, Namespace, Policy};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionPolicy {
    /// Cache every full block a finished request computed.
    Always,
    /// Cache-on-second-hit (Akamai, Maggs & Sitaraman, SIGCOMM CCR 2015): only blocks
    /// whose key was observed at least twice.
    TwoHit,
}
/// Bounded access history for TwoHit. `capacity` is part of the contract: every caller
/// passes it explicitly (ClusterConfig::history_capacity).
#[derive(Debug, Clone)]
pub struct Admission {
    /// key -> (times observed, clock of the last observation)
    pub history: BTreeMap<Key, (usize, u64)>,
    pub capacity: usize,
    pub clock: u64,
}
impl Admission {
    pub fn new(capacity: usize) -> Self {
        Self {
            history: BTreeMap::new(),
            capacity,
            clock: 0,
        }
    }
    /// C1: Record one request and return how many of its tokens may be inserted.
    /// 1. clock += 1; for every full-block key of (ns, tokens): count += 1, last = clock.
    /// 2. While history.len() > capacity, forget the entry with the smallest
    ///    (last, key): least RECENTLY observed, not least recently inserted.
    /// 3. Always: all full blocks (keys.len() * block). TwoHit: the longest leading run
    ///    of keys whose count is >= 2 AFTER steps 1-2, times block (stop at the first
    ///    key with count < 2 or already forgotten). capacity = 0 => TwoHit admits 0.
    /// Tests: c_admission_lru_history.
    pub fn observe(
        &mut self,
        ns: &Namespace,
        tokens: &[u32],
        block: usize,
        policy: AdmissionPolicy,
    ) -> usize {
        // BEGIN TODO C1
        self.clock += 1;
        let keys = block_keys(ns, tokens, block);
        for key in &keys {
            let value = self.history.entry(key.clone()).or_insert((0, self.clock));
            value.0 += 1;
            value.1 = self.clock;
        }
        while self.history.len() > self.capacity {
            let oldest = self
                .history
                .iter()
                .min_by(|(ka, a), (kb, b)| a.1.cmp(&b.1).then(ka.cmp(kb)))
                .map(|(key, _)| key.clone())
                .unwrap();
            self.history.remove(&oldest);
        }
        match policy {
            AdmissionPolicy::Always => keys.len() * block,
            AdmissionPolicy::TwoHit => {
                keys.iter()
                    .take_while(|key| self.history.get(*key).is_some_and(|v| v.0 >= 2))
                    .count()
                    * block
            }
        }
        // END TODO C1
    }
}

/// C2: How many of `matched` cached tokens a prefill may actually skip.
/// A forward pass needs at least one input token to produce next-token logits, so a
/// full hit still recomputes something:
///   Sglang: min(matched, total - 1)                  (token granular, page_size 1)
///   Vllm:   floor(min(matched, total - 1) / block) * block   (whole blocks only)
/// i.e. the largest r <= matched with r < total (and, for Vllm, r a multiple of block).
/// `block` is the cache block size in TOKENS. total = 0 gives 0. Requires matched <=
/// total and block >= 1. Example: total 32, block 16, fully cached: Sglang 31, Vllm 16.
/// Tests: c_full_hit_rule.
pub fn reusable_tokens(matched: usize, total: usize, block: usize, rule: FullHitRule) -> usize {
    // BEGIN TODO C2
    assert!(block > 0 && matched <= total);
    if total == 0 {
        return 0;
    }
    let capped = matched.min(total - 1);
    match rule {
        FullHitRule::Sglang => capped,
        FullHitRule::Vllm => capped / block * block,
    }
    // END TODO C2
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoutePolicy {
    RoundRobin,
    LeastLoad,
    /// Pure affinity: the longest cached prefix wins, ignoring load.
    PrefixAffinity,
    /// SGLang-router style: balance first when the cluster is imbalanced, otherwise
    /// follow the prefix if it covers more than `cache_threshold` of the prompt.
    Hybrid {
        abs_threshold: usize,
        rel_threshold: f64,
        cache_threshold: f64,
    },
}
/// What the router may see about one replica when a request arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplicaView {
    /// Requests routed to the replica and not finished (waiting + running).
    pub load: usize,
    /// Tokens of this prompt already resident in the replica's cache (read-only query).
    pub matched: usize,
}
#[derive(Clone, Debug)]
pub struct Router {
    pub policy: RoutePolicy,
    /// Round-robin cursor: number of requests routed so far under RoundRobin.
    pub next: usize,
}
impl Router {
    pub fn new(policy: RoutePolicy) -> Self {
        Self { policy, next: 0 }
    }
    /// C3: Return the replica index for one request (views non-empty).
    ///   RoundRobin: next % len, then next += 1 (the only policy that mutates self).
    ///   LeastLoad: min (load, index).
    ///   PrefixAffinity: max matched; ties -> min (load, index).
    ///   Hybrid: let hi/lo = max/min load. If hi - lo > abs_threshold AND
    ///     hi as f64 > rel_threshold * lo as f64, the cluster is imbalanced: LeastLoad.
    ///     Otherwise take the PrefixAffinity choice c; if prompt_tokens > 0 and
    ///     matched(c) / prompt_tokens > cache_threshold return c, else LeastLoad.
    /// Tests: c_routing_policies.
    pub fn route(&mut self, views: &[ReplicaView], prompt_tokens: usize) -> usize {
        // BEGIN TODO C3
        assert!(!views.is_empty());
        let least_load = || {
            (0..views.len())
                .min_by_key(|&i| (views[i].load, i))
                .unwrap()
        };
        let affinity = || {
            (0..views.len())
                .min_by_key(|&i| (std::cmp::Reverse(views[i].matched), views[i].load, i))
                .unwrap()
        };
        match self.policy {
            RoutePolicy::RoundRobin => {
                let r = self.next % views.len();
                self.next += 1;
                r
            }
            RoutePolicy::LeastLoad => least_load(),
            RoutePolicy::PrefixAffinity => affinity(),
            RoutePolicy::Hybrid {
                abs_threshold,
                rel_threshold,
                cache_threshold,
            } => {
                let hi = views.iter().map(|v| v.load).max().unwrap();
                let lo = views.iter().map(|v| v.load).min().unwrap();
                if hi - lo > abs_threshold && hi as f64 > rel_threshold * lo as f64 {
                    return least_load();
                }
                let c = affinity();
                if prompt_tokens > 0
                    && views[c].matched as f64 / prompt_tokens as f64 > cache_threshold
                {
                    c
                } else {
                    least_load()
                }
            }
        }
        // END TODO C3
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    pub id: usize,
    pub arrival: f64,
    pub ns: Namespace,
    /// Cache units: tokens for synthetic traces, 512-token block hash ids for Mooncake.
    pub tokens: Vec<u32>,
    /// Prompt length in model tokens (= tokens.len() * unit_tokens for full units).
    pub prompt_tokens: usize,
    pub decode_ms: f64,
}
#[derive(Clone, Debug)]
pub struct ClusterConfig {
    pub replicas: usize,
    /// Concurrent requests per replica (abstract worker slots, FIFO admission).
    pub slots: usize,
    /// Cache capacity per replica, in cache blocks.
    pub capacity_blocks: usize,
    /// Cache block size in cache units.
    pub block: usize,
    /// Model tokens per cache unit (1 for token traces, 512 for Mooncake hash ids).
    pub unit_tokens: usize,
    pub eviction: Policy,
    pub admission: AdmissionPolicy,
    /// TwoHit history entries per replica.
    pub history_capacity: usize,
    pub rule: FullHitRule,
    pub cost: CostModel,
    pub route: RoutePolicy,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ResultRow {
    pub id: usize,
    pub tenant: String,
    pub replica: usize,
    pub arrival: f64,
    pub start: f64,
    /// start + prefill
    pub first_token: f64,
    pub finish: f64,
    /// Tokens resident at start (capped at prompt_tokens).
    pub matched: usize,
    /// Tokens actually skipped after the full-hit rule (C2).
    pub reused: usize,
    pub total: usize,
    /// Whether the completion-time insert succeeded (false = capacity/pins, never TwoHit).
    pub inserted: bool,
}
#[derive(Debug)]
struct Active {
    index: usize,
    replica: usize,
    start: f64,
    first_token: f64,
    finish: f64,
    lease: Lease,
    admit_units: usize,
    matched: usize,
    reused: usize,
}
#[derive(Debug)]
pub struct ClusterReplay {
    pub rows: Vec<ResultRow>,
    pub caches: Vec<Cache>,
    pub peak_pins: usize,
}
/// C4: Event-driven replay over `cfg.replicas` replicas, each with its own Cache
/// (capacity_blocks, block, eviction), Admission(history_capacity), FIFO queue and
/// `slots` worker slots. Requests are processed in (arrival, id) order; ids are unique.
/// At each event time `now`, in this order:
///   1. Completions with finish <= now, sorted by (finish, replica, id): release the
///      lease, then insert tokens[..admit_units] into that replica's cache; emit a row.
///   2. Arrivals with arrival <= now, in (arrival, id) order: build one ReplicaView per
///      replica (load = waiting + running there, matched = min(matched_tokens *
///      unit_tokens, prompt_tokens)) and route with ONE Router for the whole replay; push
///      to that replica's queue. Later arrivals at the same `now` see the updated loads.
///   3. For replicas in index order, while a slot is free and the queue is non-empty,
///      start the queue head: matched (as above), reused = reusable_tokens(matched,
///      prompt_tokens, block * unit_tokens, rule), lease = acquire(...), admit_units =
///      observe(...) (frozen now), first_token = now + cost.prefill_ms(reused,
///      prompt_tokens - reused), finish = first_token + decode_ms.
///   4. now = min(next arrival, earliest finish).
/// A request's KV is published only at its completion, so two concurrent misses on the
/// same prefix both pay full prefill. Rows are in completion order. At the end every
/// lease is released and every cache validates. Tests: c_replay_hand_timeline,
/// c_replay_routing_oracle, c_replay_conservation_and_isolation.
pub fn replay(mut requests: Vec<Request>, cfg: &ClusterConfig) -> ClusterReplay {
    // BEGIN TODO C4
    assert!(cfg.replicas > 0 && cfg.slots > 0 && cfg.unit_tokens > 0);
    requests.sort_by(|a, b| a.arrival.total_cmp(&b.arrival).then(a.id.cmp(&b.id)));
    let ids: std::collections::BTreeSet<_> = requests.iter().map(|r| r.id).collect();
    assert_eq!(ids.len(), requests.len(), "duplicate request id");
    let k = cfg.replicas;
    let mut caches: Vec<Cache> = (0..k)
        .map(|_| Cache::new(cfg.capacity_blocks, cfg.block, cfg.eviction))
        .collect();
    let mut history: Vec<Admission> = (0..k)
        .map(|_| Admission::new(cfg.history_capacity))
        .collect();
    let mut queues: Vec<VecDeque<usize>> = vec![VecDeque::new(); k];
    let mut active: Vec<Active> = Vec::new();
    let mut router = Router::new(cfg.route);
    let mut rows = Vec::with_capacity(requests.len());
    let (mut now, mut cursor, mut peak_pins) = (0.0_f64, 0usize, 0usize);
    let matched_of = |cache: &Cache, req: &Request| {
        (cache.matched_tokens(&req.ns, &req.tokens) * cfg.unit_tokens).min(req.prompt_tokens)
    };
    while rows.len() < requests.len() {
        let (mut done, running): (Vec<_>, Vec<_>) = active.drain(..).partition(|a| a.finish <= now);
        active = running;
        done.sort_by(|a, b| {
            a.finish
                .total_cmp(&b.finish)
                .then(a.replica.cmp(&b.replica))
                .then(requests[a.index].id.cmp(&requests[b.index].id))
        });
        for job in done {
            let req = &requests[job.index];
            let cache = &mut caches[job.replica];
            cache.release(job.lease);
            let inserted = cache.insert(&req.ns, &req.tokens[..job.admit_units]);
            rows.push(ResultRow {
                id: req.id,
                tenant: req.ns.tenant.clone(),
                replica: job.replica,
                arrival: req.arrival,
                start: job.start,
                first_token: job.first_token,
                finish: job.finish,
                matched: job.matched,
                reused: job.reused,
                total: req.prompt_tokens,
                inserted,
            });
        }
        while cursor < requests.len() && requests[cursor].arrival <= now {
            let req = &requests[cursor];
            let views: Vec<ReplicaView> = (0..k)
                .map(|r| ReplicaView {
                    load: queues[r].len() + active.iter().filter(|a| a.replica == r).count(),
                    matched: matched_of(&caches[r], req),
                })
                .collect();
            let r = router.route(&views, req.prompt_tokens);
            queues[r].push_back(cursor);
            cursor += 1;
        }
        for r in 0..k {
            while active.iter().filter(|a| a.replica == r).count() < cfg.slots {
                let Some(index) = queues[r].pop_front() else {
                    break;
                };
                let req = &requests[index];
                let matched = matched_of(&caches[r], req);
                let reused = reusable_tokens(
                    matched,
                    req.prompt_tokens,
                    cfg.block * cfg.unit_tokens,
                    cfg.rule,
                );
                let lease = caches[r].acquire(&req.ns, &req.tokens);
                let admit_units =
                    history[r].observe(&req.ns, &req.tokens, cfg.block, cfg.admission);
                let first_token = now + cfg.cost.prefill_ms(reused, req.prompt_tokens - reused);
                active.push(Active {
                    index,
                    replica: r,
                    start: now,
                    first_token,
                    finish: first_token + req.decode_ms,
                    lease,
                    admit_units,
                    matched,
                    reused,
                });
            }
        }
        let pins: usize = caches
            .iter()
            .map(|c| c.nodes.values().map(|e| e.refs).sum::<usize>())
            .sum();
        peak_pins = peak_pins.max(pins);
        if rows.len() == requests.len() {
            break;
        }
        let arrival = requests.get(cursor).map_or(f64::INFINITY, |r| r.arrival);
        let completion = active
            .iter()
            .map(|a| a.finish)
            .fold(f64::INFINITY, f64::min);
        let next = arrival.min(completion);
        assert!(next.is_finite() && next >= now, "no progress at t={now}");
        now = next;
    }
    for cache in &caches {
        assert!(cache.nodes.values().all(|e| e.refs == 0), "lease leaked");
        assert!(cache.validate().is_ok());
    }
    ClusterReplay {
        rows,
        caches,
        peak_pins,
    }
    // END TODO C4
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Metrics {
    pub requests: usize,
    /// sum(reused) / sum(total): token-weighted, after the full-hit rule.
    pub hit_ratio: f64,
    /// mean of start - arrival
    pub mean_queue: f64,
    /// mean and nearest-rank p95 of first_token - arrival
    pub mean_ttft: f64,
    pub p95_ttft: f64,
    /// sum(total - reused)
    pub recomputed: usize,
    /// rows with inserted == false
    pub insert_failures: usize,
    /// max_r busy_r / mean_r busy_r over ALL `replicas` (busy = sum of finish - start of
    /// the selected rows on r); 1.0 = perfectly even; 0.0 when nothing was busy.
    pub imbalance: f64,
}
/// C5: Aggregate the selected rows (all rows, or one tenant). Empty selection ->
/// Metrics::default(). Nearest rank: the ceil(0.95 * n)-th smallest value.
/// Tests: c_metrics_and_duplication.
pub fn summarize(rows: &[ResultRow], tenant: Option<&str>, replicas: usize) -> Metrics {
    // BEGIN TODO C5
    let selected: Vec<_> = rows
        .iter()
        .filter(|r| tenant.is_none_or(|t| r.tenant == t))
        .collect();
    if selected.is_empty() {
        return Metrics::default();
    }
    let (mut reused, mut total, mut queue, mut failures) = (0usize, 0usize, 0.0, 0usize);
    let mut ttft = Vec::with_capacity(selected.len());
    let mut busy = vec![0.0; replicas];
    for row in &selected {
        reused += row.reused;
        total += row.total;
        queue += row.start - row.arrival;
        failures += usize::from(!row.inserted);
        ttft.push(row.first_token - row.arrival);
        busy[row.replica] += row.finish - row.start;
    }
    let n = selected.len() as f64;
    let mean_ttft = ttft.iter().sum::<f64>() / n;
    ttft.sort_by(f64::total_cmp);
    let rank = ((0.95 * n).ceil() as usize).max(1);
    let mean_busy = busy.iter().sum::<f64>() / replicas as f64;
    let max_busy = busy.iter().cloned().fold(0.0, f64::max);
    Metrics {
        requests: selected.len(),
        hit_ratio: if total == 0 {
            0.0
        } else {
            reused as f64 / total as f64
        },
        mean_queue: queue / n,
        mean_ttft,
        p95_ttft: ttft[rank - 1],
        recomputed: total - reused,
        insert_failures: failures,
        imbalance: if mean_busy > 0.0 {
            max_busy / mean_busy
        } else {
            0.0
        },
    }
    // END TODO C5
}
/// C6: Blocks stored more than once across replicas: sum over distinct keys of
/// (number of caches holding it - 1). Tests: c_metrics_and_duplication.
pub fn duplicated_blocks(caches: &[Cache]) -> usize {
    // BEGIN TODO C6
    let mut copies: BTreeMap<&Key, usize> = BTreeMap::new();
    for cache in caches {
        for key in cache.nodes.keys() {
            *copies.entry(key).or_insert(0) += 1;
        }
    }
    copies.values().map(|c| c - 1).sum()
    // END TODO C6
}
