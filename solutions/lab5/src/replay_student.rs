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
    let mut candidates: Vec<_> = waiting.iter().enumerate().collect();
    let has_aged = candidates
        .iter()
        .any(|(_, id)| now - requests[**id].arrival >= age_limit);
    if has_aged {
        candidates.retain(|(_, id)| now - requests[**id].arrival >= age_limit);
    }
    candidates.sort_by(|(_, a), (_, b)| {
        let ra = &requests[**a];
        let rb = &requests[**b];
        let order = if !has_aged && matches!(policy, Schedule::LongestPrefix) {
            cache
                .matched_tokens(&rb.ns, &rb.tokens)
                .cmp(&cache.matched_tokens(&ra.ns, &ra.tokens))
        } else {
            std::cmp::Ordering::Equal
        };
        order
            .then(ra.arrival.total_cmp(&rb.arrival))
            .then(ra.id.cmp(&rb.id))
    });
    candidates.first().map(|(position, _)| *position)
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
    assert!(slots > 0 && prefill_ms > 0.0);
    requests.sort_by(|a, b| a.arrival.total_cmp(&b.arrival).then(a.id.cmp(&b.id)));
    let ids: std::collections::BTreeSet<_> = requests.iter().map(|r| r.id).collect();
    assert_eq!(ids.len(), requests.len());
    assert!(requests
        .iter()
        .all(|r| r.arrival >= 0.0 && r.decode_ms > 0.0));
    let mut history = Admission::new(cache.capacity.max(1) * 8);
    let (mut now, mut cursor, mut peak_pins) = (0.0_f64, 0, 0);
    let mut waiting = Vec::new();
    let mut active: Vec<Active> = Vec::new();
    let mut rows = Vec::new();
    while rows.len() < requests.len() {
        let mut completed = Vec::new();
        let mut running = Vec::new();
        for job in active.drain(..) {
            if job.finish <= now {
                completed.push(job);
            } else {
                running.push(job);
            }
        }
        active = running;
        completed.sort_by(|a, b| a.finish.total_cmp(&b.finish).then(a.index.cmp(&b.index)));
        for job in completed {
            let req = &requests[job.index];
            cache.release(job.lease);
            let admitted = cache.insert(&req.ns, &req.tokens[..job.admit_tokens]);
            rows.push(ResultRow {
                id: req.id,
                tenant: req.ns.tenant.clone(),
                arrival: req.arrival,
                start: job.start,
                finish: job.finish,
                hit: job.hit,
                total: req.tokens.len(),
                admitted,
            });
        }
        while cursor < requests.len() && requests[cursor].arrival <= now {
            waiting.push(cursor);
            cursor += 1;
        }
        while active.len() < slots && !waiting.is_empty() {
            let position = select(&waiting, &requests, &cache, now, age_limit, schedule).unwrap();
            let index = waiting.remove(position);
            let req = &requests[index];
            let hit = cache.matched_tokens(&req.ns, &req.tokens);
            let lease = cache.acquire(&req.ns, &req.tokens);
            let admit_tokens = history.observe(&req.ns, &req.tokens, cache.block, admission_policy);
            let finish = now + (req.tokens.len() - hit) as f64 * prefill_ms + req.decode_ms;
            active.push(Active {
                index,
                finish,
                lease,
                admit_tokens,
                start: now,
                hit,
            });
            let pins = cache.nodes.values().map(|e| e.refs).sum();
            peak_pins = peak_pins.max(pins);
            assert!(cache.validate().is_ok());
        }
        if rows.len() == requests.len() {
            break;
        }
        let arrival = requests.get(cursor).map_or(f64::INFINITY, |r| r.arrival);
        let completion = active
            .iter()
            .map(|a| a.finish)
            .fold(f64::INFINITY, f64::min);
        let next = arrival.min(completion);
        assert!(next.is_finite() && next > now, "no progress");
        now = next;
    }
    assert!(cache.nodes.values().all(|e| e.refs == 0));
    Replay {
        rows,
        cache,
        peak_pins,
    }
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
    let selected: Vec<_> = rows
        .iter()
        .filter(|r| tenant.is_none_or(|t| r.tenant == t))
        .collect();
    if selected.is_empty() {
        return Metrics::default();
    }
    let (mut hit, mut total, mut wait, mut failures) = (0, 0, 0.0, 0);
    let mut latency = Vec::new();
    for row in &selected {
        assert!(row.hit <= row.total && row.start >= row.arrival && row.finish >= row.start);
        hit += row.hit;
        total += row.total;
        wait += row.start - row.arrival;
        failures += usize::from(!row.admitted);
        latency.push(row.finish - row.arrival);
    }
    latency.sort_by(f64::total_cmp);
    let rank = (0.95 * latency.len() as f64).ceil() as usize;
    Metrics {
        requests: selected.len(),
        hit_ratio: if total == 0 {
            0.0
        } else {
            hit as f64 / total as f64
        },
        mean_wait: wait / selected.len() as f64,
        p95_e2e: latency[rank - 1],
        recomputed: total - hit,
        admission_failures: failures,
    }
    // END TODO C4
}
