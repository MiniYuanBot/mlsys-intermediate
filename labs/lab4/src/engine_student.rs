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
        todo!("C1: implement the contract above");
        // END TODO C1
    }
}

/// C2: Move sorted arrivals <= now into the FIFO ready queue exactly once.
/// Advance cursor even when several arrivals have the same timestamp.
/// Tests: c_arrivals_once.
pub fn enqueue(states: &[State], cursor: &mut usize, now: f64, ready: &mut VecDeque<usize>) {
    // BEGIN TODO C2
    todo!("C2: implement the contract above");
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
    todo!("C3: implement the contract above");
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
    todo!("C4: implement the contract above");
    // END TODO C4
}
