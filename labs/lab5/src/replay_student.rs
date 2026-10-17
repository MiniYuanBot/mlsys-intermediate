use crate::{block_keys, Cache, Key, Lease, Namespace};
use std::collections::BTreeMap;
#[derive(Clone, Copy, Debug)]
pub enum AdmissionPolicy {
    Always,
    TwoHit,
}
#[derive(Debug)]
pub struct Admission {
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
    /// C1: Observe each completed-block key once per request; bounded LRU history.
    /// TwoHit admits the longest contiguous chain observed >=2 times (current
    /// request included), Always admits all. Compute after updates; history
    /// eviction can make admission conservative. capacity=0 => TwoHit admits zero.
    /// Tests: c_admission_pollution_and_bound.
    pub fn observe(
        &mut self,
        ns: &Namespace,
        tokens: &[u32],
        block: usize,
        policy: AdmissionPolicy,
    ) -> usize {
        // BEGIN TODO C1
        todo!("C1: implement the contract above");
        // END TODO C1
    }
}
#[derive(Clone, Debug)]
pub struct Request {
    pub id: usize,
    pub arrival: f64,
    pub ns: Namespace,
    pub tokens: Vec<u32>,
    pub decode_ms: f64,
}
#[derive(Clone, Copy, Debug)]
pub enum Schedule {
    Fifo,
    LongestPrefix,
}
/// C2: Return index INTO waiting (not request ID). Among requests waiting at
/// least age_limit choose earliest arrival then ID; otherwise FIFO or longest
/// cached prefix, with arrival/ID ties. No future arrivals are in waiting.
/// Tests: c_scheduler_age_and_prefix.
pub fn select(
    waiting: &[usize],
    requests: &[Request],
    cache: &Cache,
    now: f64,
    age_limit: f64,
    policy: Schedule,
) -> Option<usize> {
    // BEGIN TODO C2
    todo!("C2: implement the contract above");
    // END TODO C2
}
#[derive(Clone, Debug)]
pub struct ResultRow {
    pub id: usize,
    pub tenant: String,
    pub arrival: f64,
    pub start: f64,
    pub finish: f64,
    pub hit: usize,
    pub total: usize,
    pub admitted: bool,
}
#[derive(Debug)]
struct Active {
    index: usize,
    finish: f64,
    lease: Lease,
    admit_tokens: usize,
    start: f64,
    hit: usize,
}
#[derive(Debug)]
pub struct Replay {
    pub rows: Vec<ResultRow>,
    pub cache: Cache,
    pub peak_pins: usize,
}
/// C3: Event-driven replay with abstract parallel worker slots. At each timestamp
/// finish active jobs first (release lease then insert newly computed prefix),
/// enqueue arrivals, then schedule. On start: record hit, acquire it, snapshot
/// admission decision; duration=(total-hit)*prefill_ms+decode_ms. Never publish
/// an in-flight miss as a cache hit. Jump to min(next arrival,next completion).
/// Tests: c_replay_hand_timeline, c_conservation_isolation_and_capacity.
pub fn replay(
    mut requests: Vec<Request>,
    mut cache: Cache,
    slots: usize,
    prefill_ms: f64,
    schedule: Schedule,
    admission_policy: AdmissionPolicy,
    age_limit: f64,
) -> Replay {
    // BEGIN TODO C3
    todo!("C3: implement the contract above");
    // END TODO C3
}
#[derive(Debug, Default)]
pub struct Metrics {
    pub requests: usize,
    pub hit_ratio: f64,
    pub mean_wait: f64,
    pub p95_e2e: f64,
    pub recomputed: usize,
    pub admission_failures: usize,
}
/// C4: Aggregate globally or for one tenant; token-weighted hit ratio (not mean
/// request hit ratio), nearest-rank p95 E2E, true queue wait start-arrival.
/// Empty tenant -> all zero. Tests: c_metrics_weighting_and_empty.
pub fn summarize(rows: &[ResultRow], tenant: Option<&str>) -> Metrics {
    // BEGIN TODO C4
    todo!("C4: implement the contract above");
    // END TODO C4
}
