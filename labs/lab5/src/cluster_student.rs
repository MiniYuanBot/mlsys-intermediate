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
        todo!("TODO C1")
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
    todo!("TODO C2")
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
        todo!("TODO C3")
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
    todo!("TODO C4")
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
    todo!("TODO C5")
    // END TODO C5
}
/// C6: Blocks stored more than once across replicas: sum over distinct keys of
/// (number of caches holding it - 1). Tests: c_metrics_and_duplication.
pub fn duplicated_blocks(caches: &[Cache]) -> usize {
    // BEGIN TODO C6
    todo!("TODO C6")
    // END TODO C6
}
