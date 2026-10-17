use crate::{choose_k, interpolate, percentile, Drift, Estimator, ProfilePoint};
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Request {
    pub id: usize,
    pub arrival: f64,
    pub output: usize,
    pub context: usize,
    pub deadline: f64,
    pub alpha: f64,
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
    /// C1: Commit min(accepted+1,remaining) tokens atomically at round completion.
    /// Record first/finish once, one timestamp per committed token, no overflow.
    /// Tests: c_commit_cap_and_timestamps.
    pub fn commit(&mut self, accepted: usize, end: f64) {
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

#[derive(Clone, Copy, Debug)]
pub enum Controller {
    Fixed(usize),
    Mean,
    ConservativeNoProbe,
}
#[derive(Debug)]
pub struct Simulation {
    pub states: Vec<State>,
    pub busy: f64,
    pub infeasible: usize,
    pub resets: usize,
}
/// Provided deterministic draw keyed by seed, request, committed position, and
/// proposed depth. Policies cannot consume each other's RNG stream.
fn draw(seed: u64, id: usize, emitted: usize, depth: usize) -> f64 {
    let mut x = seed
        ^ (id as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (emitted as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ depth as u64;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((x ^ (x >> 31)) >> 11) as f64 / ((1u64 << 53) as f64)
}
/// C3: Single-server round-robin discrete-event service. New arrivals during a
/// round enter BEFORE the running request is requeued. Idle time jumps to the
/// next arrival. Budget is min(round_bound,deadline-now), with explicit k=0
/// fallback on infeasibility; do not drop late work or read future acceptance.
/// Tests: c_hand_timeline, c_conservation_empty_and_overload.
pub fn simulate(
    mut requests: Vec<Request>,
    profile: &[ProfilePoint],
    controller: Controller,
    round_bound: f64,
    seed: u64,
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
    let mut estimator = Estimator::new(profile[0].verify.len() - 1, 0.97);
    let mut drift = Drift::default();
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
        let p = estimator.probabilities(if matches!(controller, Controller::ConservativeNoProbe) {
            1.0
        } else {
            0.0
        });
        let k = match controller {
            Controller::Fixed(k) => k.min(p.len()).min(remaining - 1),
            _ => match choose_k(&p, &costs, budget, remaining) {
                Some(k) => k,
                None => {
                    infeasible += 1;
                    0
                }
            },
        };
        let accepted = (0..k)
            .take_while(|&depth| {
                draw(seed, state.request.id, state.emitted, depth) < state.request.alpha
            })
            .count();
        let cost = costs.draft * k as f64 + costs.verify[k];
        now += cost;
        busy += cost;
        estimator.observe(accepted, k);
        if k > 0 && drift.update(accepted as f64 / k as f64, 0.05, 3.0) {
            estimator = Estimator::new(p.len(), 0.97);
        }
        states[index].commit(accepted, now);
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
    }
    // END TODO C3
}

#[derive(Debug, Default)]
pub struct Metrics {
    pub requests: usize,
    pub tokens: usize,
    pub p95_ttft: f64,
    pub p95_e2e: f64,
    pub goodput: f64,
    pub makespan: f64,
    pub met: usize,
    pub utilization: f64,
}
/// C4: Nearest-rank p95; TTFT=first-arrival, E2E=finish-arrival. A request
/// contributes to goodput only if finish<=absolute deadline. Divide by span
/// from first arrival to last completion, in seconds (input ms). Empty is zero.
/// Tests: c_metrics_independent and c_conservation_empty_and_overload.
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
    let mut ttft = Vec::new();
    let mut e2e = Vec::new();
    let (mut tokens, mut met) = (0, 0);
    for state in &sim.states {
        assert_eq!(state.emitted, state.request.output);
        assert_eq!(state.committed_at.len(), state.emitted);
        let finish = state.finish.unwrap();
        ttft.push(state.first.unwrap() - state.request.arrival);
        e2e.push(finish - state.request.arrival);
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
        goodput: met as f64 * 1000.0 / makespan,
        makespan,
        met,
        utilization: sim.busy / makespan,
    }
    // END TODO C4
}
