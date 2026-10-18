//! C: request state, arrivals, a single-server discrete-event engine and SLO metrics.
//! Policies and acceptance environments are pluggable so stage R can replay real traces.

use crate::{choose_k, interpolate, percentile, Drift, Estimator, ProfilePoint};
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Request {
    pub id: usize,
    /// Arrival time, ms.
    pub arrival: f64,
    /// Tokens to commit (stage R: decode steps after the prefill token).
    pub output: usize,
    pub context: usize,
    /// Absolute deadline, ms.
    pub deadline: f64,
    /// Synthetic environment only: conditional acceptance a_j at depth j (0-based).
    /// Controllers never read it.
    pub accept: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct State {
    pub request: Request,
    pub emitted: usize,
    pub first: Option<f64>,
    pub finish: Option<f64>,
    pub rounds: usize,
    pub committed_at: Vec<f64>,
}
impl State {
    pub fn new(request: Request) -> Self {
        Self {
            request,
            emitted: 0,
            first: None,
            finish: None,
            rounds: 0,
            committed_at: Vec::new(),
        }
    }
    /// C1: Commit min(accepted+1,remaining) tokens atomically at round completion `end`.
    /// Record first/finish once, one timestamp per committed token, no overflow.
    /// Returns the number of committed tokens. Panics if already finished or time goes back.
    /// Tests: c_commit_cap_and_timestamps.
    pub fn commit(&mut self, accepted: usize, end: f64) -> usize {
        // BEGIN TODO C1
        assert!(self.finish.is_none() && end >= self.request.arrival);
        if let Some(&previous) = self.committed_at.last() {
            assert!(end >= previous);
        }
        let remaining = self.request.output - self.emitted;
        let committed = (accepted + 1).min(remaining);
        assert!(committed > 0);
        if self.first.is_none() {
            self.first = Some(end);
        }
        for _ in 0..committed {
            self.committed_at.push(end);
        }
        self.emitted += committed;
        self.rounds += 1;
        if self.emitted == self.request.output {
            self.finish = Some(end);
        }
        committed
        // END TODO C1
    }
}

/// C2: Move sorted arrivals <= now into the FIFO ready queue exactly once.
/// Advance cursor even when several arrivals have the same timestamp.
/// Tests: c_arrivals_once.
pub fn enqueue(states: &[State], cursor: &mut usize, now: f64, ready: &mut VecDeque<usize>) {
    // BEGIN TODO C2
    while *cursor < states.len() {
        let state = &states[*cursor];
        if state.request.arrival > now {
            break;
        }
        assert_eq!(state.emitted, 0);
        ready.push_back(*cursor);
        *cursor += 1;
    }
    // END TODO C2
}

/// Provided: where accepted prefixes come from. Implementations must return a value
/// in 0..=k and must not depend on which policy is running (paired comparison).
pub trait Acceptance {
    fn accepted(&self, request: &Request, emitted: usize, k: usize) -> usize;
}

/// Provided: deterministic draw keyed by seed, request, committed position, and
/// proposed depth. Policies cannot consume each other's random stream.
pub fn draw(seed: u64, id: usize, emitted: usize, depth: usize) -> f64 {
    let mut x = seed
        ^ (id as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (emitted as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ depth as u64;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((x ^ (x >> 31)) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Provided synthetic environment with NON-IID depth-conditional acceptance:
/// depth j is accepted iff every depth < j was and draw(...) < request.accept[j].
pub struct Synthetic {
    pub seed: u64,
}
impl Acceptance for Synthetic {
    fn accepted(&self, r: &Request, emitted: usize, k: usize) -> usize {
        (0..k)
            .take_while(|&depth| draw(self.seed, r.id, emitted, depth) < r.accept[depth])
            .count()
    }
}

/// Provided: a controller sees only costs, the budget, the remaining output and its
/// own past observations. `choose` returns None when nothing fits the budget.
pub trait Policy {
    fn name(&self) -> String;
    fn choose(&mut self, costs: &ProfilePoint, budget: f64, remaining: usize) -> Option<usize>;
    fn observe(&mut self, accepted: usize, proposed: usize);
    fn reset(&mut self) {}
}

/// Provided: always proposes min(k, remaining-1, kmax), even when it does not fit.
pub struct Fixed(pub usize);
impl Policy for Fixed {
    fn name(&self) -> String {
        format!("k{}", self.0)
    }
    fn choose(&mut self, costs: &ProfilePoint, _: f64, remaining: usize) -> Option<usize> {
        Some(
            self.0
                .min(costs.verify.len() - 1)
                .min(remaining.saturating_sub(1)),
        )
    }
    fn observe(&mut self, _: usize, _: usize) {}
}

/// Provided: per-position estimator (B1) + B3 choice. z = 0 gives posterior means
/// ("mean"); z > 0 gives the Wilson lower bound without any probing ("no_probe").
pub struct PerPosition {
    pub estimator: Estimator,
    pub z: f64,
}
impl PerPosition {
    pub fn new(kmax: usize, decay: f64, z: f64) -> Self {
        Self {
            estimator: Estimator::new(kmax, decay),
            z,
        }
    }
}
impl Policy for PerPosition {
    fn name(&self) -> String {
        if self.z == 0.0 {
            "mean".into()
        } else {
            format!("no_probe_z{}", self.z)
        }
    }
    fn choose(&mut self, costs: &ProfilePoint, budget: f64, remaining: usize) -> Option<usize> {
        choose_k(
            &self.estimator.probabilities(self.z),
            costs,
            budget,
            remaining,
        )
    }
    fn observe(&mut self, accepted: usize, proposed: usize) {
        self.estimator.observe(accepted, proposed);
    }
    fn reset(&mut self) {
        self.estimator = Estimator::new(self.estimator.success.len(), self.estimator.decay);
    }
}

/// Provided: IID baseline. Same censored counts, but one pooled alpha for all depths.
pub struct Pooled {
    pub estimator: Estimator,
}
impl Policy for Pooled {
    fn name(&self) -> String {
        "pooled_iid".into()
    }
    fn choose(&mut self, costs: &ProfilePoint, budget: f64, remaining: usize) -> Option<usize> {
        choose_k(&self.estimator.pooled(), costs, budget, remaining)
    }
    fn observe(&mut self, accepted: usize, proposed: usize) {
        self.estimator.observe(accepted, proposed);
    }
    fn reset(&mut self) {
        self.estimator = Estimator::new(self.estimator.success.len(), self.estimator.decay);
    }
}

/// One executed verification round (the engine's audit log).
#[derive(Clone, Debug, PartialEq)]
pub struct RoundRecord {
    pub request: usize,
    pub start: f64,
    pub end: f64,
    pub k: usize,
    pub accepted: usize,
    pub committed: usize,
    pub budget: f64,
    pub infeasible: bool,
    pub reset: bool,
}

#[derive(Debug)]
pub struct Simulation {
    pub states: Vec<State>,
    pub busy: f64,
    pub infeasible: usize,
    pub resets: usize,
    pub rounds: Vec<RoundRecord>,
}

/// Provided drift-detector parameters used by C3.
pub const DRIFT_DELTA: f64 = 0.05;
pub const DRIFT_THRESHOLD: f64 = 3.0;

/// C3: Single-server round-robin discrete-event service.
/// * Sort requests by (arrival, id); states[i] is the i-th request in that order.
/// * Loop until all finish: if ready is empty jump `now` to the next arrival; enqueue
///   arrivals <= now; pop the front request.
/// * costs = interpolate(profile, context + emitted); remaining = output - emitted;
///   budget = min(round_bound, deadline - now).
/// * k = policy.choose(...); None -> execute k = 0 and mark the round infeasible.
///   EVERY policy (Fixed included) marks a round infeasible when round_cost(k) exceeds
///   budget + 1e-9. `infeasible` counts such rounds.
/// * accepted = env.accepted(request, emitted, k); now += round_cost(k); busy too.
/// * policy.observe(accepted, k). If k > 0, feed accepted/k to the Drift detector
///   (DRIFT_DELTA, DRIFT_THRESHOLD); on a trigger call policy.reset() and mark the round.
/// * commit at `now`; enqueue arrivals that happened during the round BEFORE requeueing
///   the running request; push one RoundRecord per round.
/// Never drop late work, never read request.accept. Tests: c_hand_timeline,
/// c_infeasible_counts_all_policies, c_mean_policy_shadow_replay, c_drift_reset_replay,
/// c_conservation_empty_and_overload.
pub fn simulate(
    mut requests: Vec<Request>,
    profile: &[ProfilePoint],
    policy: &mut dyn Policy,
    env: &dyn Acceptance,
    round_bound: f64,
) -> Simulation {
    // BEGIN TODO C3
    requests.sort_by(|a, b| a.arrival.total_cmp(&b.arrival).then(a.id.cmp(&b.id)));
    let unique: std::collections::BTreeSet<_> = requests.iter().map(|r| r.id).collect();
    assert_eq!(unique.len(), requests.len());
    assert!(requests
        .iter()
        .all(|r| r.output > 0 && r.arrival >= 0.0 && r.deadline >= r.arrival));
    let mut states: Vec<_> = requests.into_iter().map(State::new).collect();
    let mut ready = VecDeque::new();
    let (mut cursor, mut finished, mut now) = (0, 0, 0.0_f64);
    let (mut busy, mut infeasible) = (0.0, 0);
    let mut drift = Drift::default();
    let mut rounds = Vec::new();
    while finished < states.len() {
        if ready.is_empty() && cursor < states.len() {
            now = now.max(states[cursor].request.arrival);
        }
        enqueue(&states, &mut cursor, now, &mut ready);
        let index = ready.pop_front().expect("unfinished work must be queued");
        let state = &states[index];
        let costs = interpolate(profile, state.request.context + state.emitted);
        let remaining = state.request.output - state.emitted;
        let budget = round_bound.min(state.request.deadline - now);
        let choice = policy.choose(&costs, budget, remaining);
        let k = choice.unwrap_or(0);
        let cost = costs.round_cost(k);
        let is_infeasible = choice.is_none() || cost > budget + 1e-9;
        if is_infeasible {
            infeasible += 1;
        }
        let accepted = env.accepted(&state.request, state.emitted, k);
        assert!(accepted <= k);
        let start = now;
        now += cost;
        busy += cost;
        policy.observe(accepted, k);
        let reset = k > 0 && drift.update(accepted as f64 / k as f64, DRIFT_DELTA, DRIFT_THRESHOLD);
        if reset {
            policy.reset();
        }
        let committed = states[index].commit(accepted, now);
        rounds.push(RoundRecord {
            request: states[index].request.id,
            start,
            end: now,
            k,
            accepted,
            committed,
            budget,
            infeasible: is_infeasible,
            reset,
        });
        enqueue(&states, &mut cursor, now, &mut ready);
        if states[index].finish.is_some() {
            finished += 1;
        } else {
            ready.push_back(index);
        }
    }
    Simulation {
        states,
        busy,
        infeasible,
        resets: drift.resets,
        rounds,
    }
    // END TODO C3
}

#[derive(Debug, Default, Clone)]
pub struct Metrics {
    pub requests: usize,
    pub tokens: usize,
    pub p95_ttft: f64,
    pub p95_e2e: f64,
    /// p95 over requests of mean TPOT = (finish - first) / (tokens - 1); 1-token requests skipped.
    pub p95_tpot: f64,
    pub goodput: f64,
    pub makespan: f64,
    pub met: usize,
    pub utilization: f64,
}
/// C4: Nearest-rank p95 (A3); TTFT=first-arrival, E2E=finish-arrival,
/// mean TPOT=(finish-first)/(output-1) for output>1. A request is met if
/// finish<=absolute deadline. goodput = met / (last finish - first arrival) in
/// requests per second (input ms). Check token conservation per request.
/// Empty simulation -> all zero. Tests: c_metrics_independent and conservation tests.
pub fn summarize(sim: &Simulation) -> Metrics {
    // BEGIN TODO C4
    if sim.states.is_empty() {
        return Metrics::default();
    }
    let start = sim
        .states
        .iter()
        .map(|s| s.request.arrival)
        .fold(f64::INFINITY, f64::min);
    let end = sim
        .states
        .iter()
        .map(|s| s.finish.unwrap())
        .fold(0.0_f64, f64::max);
    let makespan = end - start;
    let (mut ttft, mut e2e, mut tpot) = (Vec::new(), Vec::new(), Vec::new());
    let (mut tokens, mut met) = (0, 0);
    for state in &sim.states {
        assert_eq!(state.emitted, state.request.output);
        assert_eq!(state.committed_at.len(), state.emitted);
        let finish = state.finish.unwrap();
        let first = state.first.unwrap();
        ttft.push(first - state.request.arrival);
        e2e.push(finish - state.request.arrival);
        if state.emitted > 1 {
            tpot.push((finish - first) / (state.emitted - 1) as f64);
        }
        tokens += state.emitted;
        if finish <= state.request.deadline {
            met += 1;
        }
    }
    Metrics {
        requests: sim.states.len(),
        tokens,
        p95_ttft: percentile(&ttft, 0.95).unwrap(),
        p95_e2e: percentile(&e2e, 0.95).unwrap(),
        p95_tpot: percentile(&tpot, 0.95).unwrap_or(0.0),
        goodput: met as f64 * 1000.0 / makespan,
        makespan,
        met,
        utilization: sim.busy / makespan,
    }
    // END TODO C4
}
