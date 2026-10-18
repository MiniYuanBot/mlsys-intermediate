//! Lab 8 A/B/C: length history, list scheduling, group advantages, segment ledgers,
//! a quantum/work-stealing executor, keep-first-n selection, a bounded group queue,
//! the clipped surrogate and its gradient, an AReaL-style staleness bound, decoupled
//! importance terms for partial rollouts, and a real contextual-bandit learning loop.
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Debug)]
pub struct PredictedJob {
    pub id: usize,
    pub estimated_tokens: u64,
}
#[derive(Default)]
pub struct LengthModel {
    pub history: BTreeMap<usize, VecDeque<u64>>,
    pub window: usize,
}
impl LengthModel {
    // A1: append one COMPLETED length to the history of `class`; keep only the newest
    // `max(window, 1)` values of that class. Other classes are untouched.
    // Test: a_length_history (window eviction, class isolation).
    pub fn observe(&mut self, class: usize, length: u64) {
        // BEGIN TODO A1
        let row = self.history.entry(class).or_default();
        row.push_back(length);
        while row.len() > self.window.max(1) {
            row.pop_front();
        }
        // END TODO A1
    }
    // A2: upper empirical median (sorted[len / 2]) of the class history; a class with
    // no completed history predicts 32. Must not change the history.
    // Test: a_length_history.
    pub fn predict(&self, class: usize) -> u64 {
        // BEGIN TODO A2
        let Some(row) = self.history.get(&class) else {
            return 32;
        };
        let mut lengths: Vec<_> = row.iter().copied().collect();
        lengths.sort_unstable();
        lengths[lengths.len() / 2]
        // END TODO A2
    }
}
// A3: list scheduling on ESTIMATED lengths. lpt=true visits jobs by estimate descending
// (ties: smaller id first); lpt=false keeps the input order. Each job goes to the worker
// with the smallest estimated load (ties: smaller worker index). Returns one queue of job
// ids per worker, in assignment order. workers > 0.
// Test: a_assign (unsorted input, exact queues for both flags, id tie-break).
pub fn assign(jobs: &[PredictedJob], workers: usize, lpt: bool) -> Vec<Vec<usize>> {
    // BEGIN TODO A3
    assert!(workers > 0);
    let mut order = jobs.to_vec();
    if lpt {
        order.sort_by(|a, b| {
            b.estimated_tokens
                .cmp(&a.estimated_tokens)
                .then(a.id.cmp(&b.id))
        });
    }
    let mut queues = vec![Vec::new(); workers];
    let mut load = vec![0u64; workers];
    for job in order {
        let worker = (0..workers).min_by_key(|&w| (load[w], w)).unwrap();
        queues[worker].push(job.id);
        load[worker] += job.estimated_tokens;
    }
    queues
    // END TODO A3
}
// A4: group-relative advantages (r_i - mean) / std. bessel=false: population std
// (divide by G); bessel=true: sample std (divide by G-1, what verl/TRL/CS336 use).
// Empty input -> empty; a group whose std is < 1e-10 (constant rewards, or G=1 with
// bessel=true) -> all zeros. No epsilon in the denominator.
// Test: a_advantage.
pub fn advantages(rewards: &[f64], bessel: bool) -> Vec<f64> {
    // BEGIN TODO A4
    let n = rewards.len();
    if n == 0 {
        return vec![];
    }
    let denom = if bessel { n as f64 - 1.0 } else { n as f64 };
    let mean = rewards.iter().sum::<f64>() / n as f64;
    let ss = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>();
    if denom <= 0.0 || (ss / denom).sqrt() < 1e-10 {
        return vec![0.; n];
    }
    let std = (ss / denom).sqrt();
    rewards.iter().map(|r| (r - mean) / std).collect()
    // END TODO A4
}
// A5: histogram with buckets (-inf, b0], (b0, b1], ..., (b_last, +inf) and the empirical
// quantile sorted[ceil((N-1) q)]. Bounds strictly increasing, q in [0, 1]; empty -> 0.
// Test: a_histogram.
pub fn histogram(lengths: &[u64], bounds: &[u64], quantile: f64) -> (Vec<usize>, u64) {
    // BEGIN TODO A5
    assert!(bounds.windows(2).all(|p| p[0] < p[1]));
    assert!((0.0..=1.0).contains(&quantile));
    let mut counts = vec![0; bounds.len() + 1];
    for &length in lengths {
        let index = bounds
            .iter()
            .position(|&bound| length <= bound)
            .unwrap_or(bounds.len());
        counts[index] += 1;
    }
    if lengths.is_empty() {
        return (counts, 0);
    }
    let mut sorted = lengths.to_vec();
    sorted.sort_unstable();
    let rank = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    (counts, sorted[rank])
    // END TODO A5
}
#[derive(Clone, Debug)]
pub struct Segment {
    pub id: usize,
    pub group: usize,
    pub version: u64,
    pub start: usize,
    pub tokens: Vec<u32>,
    pub terminal: bool,
}
#[derive(Clone, Debug)]
pub struct Completed {
    pub id: usize,
    pub group: usize,
    pub version: u64,
    pub tokens: Vec<u32>,
}
// B1: rebuild one trajectory from segments received in any order. Sort by start; the
// first starts at 0, each next start equals the tokens collected so far, no segment is
// empty, id/group/version are identical, exactly the last segment is terminal.
// Anything else -> Err (never a partial trajectory). Test: b_reassemble.
pub fn reassemble(segments: &[Segment]) -> Result<Completed, String> {
    // BEGIN TODO B1
    let first = segments.first().ok_or("empty trajectory")?;
    let mut ordered = segments.to_vec();
    ordered.sort_by_key(|s| s.start);
    let mut tokens = Vec::new();
    let mut terminal = false;
    for segment in ordered {
        if (segment.id, segment.group, segment.version) != (first.id, first.group, first.version) {
            return Err("mixed identity/version".into());
        }
        if terminal || segment.start != tokens.len() || segment.tokens.is_empty() {
            return Err("gap/overlap/after-terminal".into());
        }
        tokens.extend(segment.tokens);
        terminal = segment.terminal;
    }
    if !terminal {
        return Err("unfinished trajectory".into());
    }
    Ok(Completed {
        id: first.id,
        group: first.group,
        version: first.version,
        tokens,
    })
    // END TODO B1
}
// B2: the completed set must equal the preselected id set exactly (no duplicates, no
// extras, nothing missing), every sample has `version`, every group has group_size
// members. Test: b_batch_gate.
pub fn training_ready(
    completed: &[Completed],
    expected_ids: &[usize],
    group_size: usize,
    version: u64,
) -> Result<(), String> {
    // BEGIN TODO B2
    if group_size == 0 {
        return Err("zero group".into());
    }
    let expected: BTreeSet<_> = expected_ids.iter().copied().collect();
    let actual: BTreeSet<_> = completed.iter().map(|s| s.id).collect();
    if expected.len() != expected_ids.len() || actual.len() != completed.len() || actual != expected
    {
        return Err("batch identity mismatch".into());
    }
    let mut groups = BTreeMap::new();
    for sample in completed {
        if sample.version != version {
            return Err("stale version".into());
        }
        *groups.entry(sample.group).or_insert(0) += 1;
    }
    if groups.values().any(|&n| n != group_size) {
        return Err("incomplete group".into());
    }
    Ok(())
    // END TODO B2
}
#[derive(Clone, Debug)]
pub struct Job {
    pub id: usize,
    pub group: usize,
    pub version: u64,
    pub estimate: u64,
    pub actual: u64,
}
pub struct Schedule {
    pub finish: u64,
    pub busy: u64,
    pub steals: usize,
    pub segments: Vec<Segment>,
}
// B3: event-driven quantum scheduler. Queues start from assign(estimates, lpt=true).
// Loop: every idle worker takes the head of its own queue; if empty and steal=true it
// takes the TAIL of the longest other queue (ties: smaller worker index) and counts a
// steal. A started job runs min(quantum, actual - progress) units. Jump to the earliest
// completion time; settle ALL workers finishing then (in worker order), emit one Segment
// each (token value id*10000+position), and push unfinished jobs to the back of the
// worker's own queue. Only the executor reads `actual`. Test: b_scheduler.
pub fn schedule(jobs: &[Job], workers: usize, quantum: u64, steal: bool) -> Schedule {
    // BEGIN TODO B3
    assert!(workers > 0 && quantum > 0);
    let predicted: Vec<_> = jobs
        .iter()
        .map(|j| PredictedJob {
            id: j.id,
            estimated_tokens: j.estimate,
        })
        .collect();
    let assigned = assign(&predicted, workers, true);
    let by_id: BTreeMap<_, _> = jobs.iter().map(|j| (j.id, j)).collect();
    let mut queues: Vec<VecDeque<usize>> = assigned.into_iter().map(VecDeque::from).collect();
    let mut progress: BTreeMap<usize, u64> = BTreeMap::new();
    let mut active: Vec<Option<(usize, u64, u64)>> = vec![None; workers];
    let mut time = 0;
    let mut result = Schedule {
        finish: 0,
        busy: 0,
        steals: 0,
        segments: vec![],
    };
    loop {
        for worker in 0..workers {
            if active[worker].is_some() {
                continue;
            }
            let mut next = queues[worker].pop_front();
            if next.is_none() && steal {
                let victim = (0..workers)
                    .filter(|&w| w != worker && !queues[w].is_empty())
                    .max_by_key(|&w| (queues[w].len(), std::cmp::Reverse(w)));
                if let Some(victim) = victim {
                    next = queues[victim].pop_back();
                    result.steals += 1;
                }
            }
            if let Some(id) = next {
                let done = *progress.get(&id).unwrap_or(&0);
                let count = quantum.min(by_id[&id].actual - done);
                active[worker] = Some((id, time + count, count));
                result.busy += count;
            }
        }
        let Some(event) = active.iter().flatten().map(|s| s.1).min() else {
            break;
        };
        time = event;
        for worker in 0..workers {
            let Some((id, end, count)) = active[worker] else {
                continue;
            };
            if end != time {
                continue;
            }
            let start = *progress.get(&id).unwrap_or(&0);
            let job = by_id[&id];
            let terminal = start + count == job.actual;
            result.segments.push(Segment {
                id,
                group: job.group,
                version: job.version,
                start: start as usize,
                tokens: (start..start + count)
                    .map(|v| (id as u32) * 10000 + v as u32)
                    .collect(),
                terminal,
            });
            progress.insert(id, start + count);
            active[worker] = None;
            if !terminal {
                queues[worker].push_back(id);
            }
        }
    }
    result.finish = time;
    result
    // END TODO B3
}
#[derive(Clone, Debug)]
pub struct Finished {
    pub sample_idx: usize,
    pub finish: f64,
    pub tokens: u64,
    pub reward: f64,
}
// B4: redundancy with "keep the first n finishers" (the selection rule in DARTS'
// RedundancyCounter). Return the positions (indices into `group`, NOT sample_idx) of the
// min(n, len) samples with the smallest (finish, sample_idx), in that order. The input
// may be in any order. This changes the training distribution: B4's test measures the
// shift of mean length and reward against an exact order-statistic oracle.
// Test: b_keep_first_n.
pub fn keep_first_n(group: &[Finished], n: usize) -> Vec<usize> {
    // BEGIN TODO B4
    let mut order: Vec<usize> = (0..group.len()).collect();
    order.sort_by(|&a, &b| {
        group[a]
            .finish
            .total_cmp(&group[b].finish)
            .then(group[a].sample_idx.cmp(&group[b].sample_idx))
    });
    order.truncate(n.min(group.len()));
    order
    // END TODO B4
}
#[derive(Clone, Debug)]
pub struct Sample {
    pub id: usize,
    pub group: usize,
    pub version: u64,
    pub context: usize,
    pub action: usize,
    pub reward: f64,
    pub behavior_p: f64,
}
pub struct GroupQueue {
    pub capacity: usize,
    pub group_size: usize,
    pub max_lag: u64,
    pub version: u64,
    pub pending: BTreeMap<usize, BTreeMap<usize, Sample>>,
    pub consumed: BTreeSet<usize>,
    pub backpressure: usize,
    pub stale: usize,
}
impl GroupQueue {
    pub fn new(capacity: usize, group_size: usize, max_lag: u64) -> Self {
        assert!(capacity > 0 && group_size > 0);
        Self {
            capacity,
            group_size,
            max_lag,
            version: 0,
            pending: BTreeMap::new(),
            consumed: BTreeSet::new(),
            backpressure: 0,
            stale: 0,
        }
    }
    // C1: admission. Reject (return false, state unchanged except counters) when:
    // version is in the future or older than version - max_lag (stale += 1); id/group
    // mismatch (id / group_size != group); group already consumed; duplicate id; version
    // differs from the group's already admitted members. Capacity counts GROUPS: a new
    // group needs a free slot (else backpressure += 1, return false); members of an
    // admitted group always fit. Test: c_queue_admission.
    pub fn push(&mut self, sample: Sample) -> bool {
        // BEGIN TODO C1
        if sample.version > self.version || self.version - sample.version > self.max_lag {
            self.stale += 1;
            return false;
        }
        if sample.id / self.group_size != sample.group || self.consumed.contains(&sample.group) {
            return false;
        }
        if !self.pending.contains_key(&sample.group) && self.pending.len() == self.capacity {
            self.backpressure += 1;
            return false;
        }
        let group = self.pending.entry(sample.group).or_default();
        if group.contains_key(&sample.id) || group.values().any(|s| s.version != sample.version) {
            return false;
        }
        group.insert(sample.id, sample);
        true
        // END TODO C1
    }
    // C2: first evict every pending group whose version is older than version - max_lag
    // (stale += its size, mark consumed), then return the complete group with the
    // smallest group id (members in id order) and mark it consumed. Never return a
    // partial group. Test: c_queue_drain.
    pub fn pop(&mut self) -> Option<Vec<Sample>> {
        // BEGIN TODO C2
        let expired: Vec<_> = self
            .pending
            .iter()
            .filter_map(|(&id, row)| {
                let v = row.values().next().unwrap().version;
                (self.version.saturating_sub(v) > self.max_lag).then_some(id)
            })
            .collect();
        for id in expired {
            self.stale += self.pending.remove(&id).unwrap().len();
            self.consumed.insert(id);
        }
        let ready = self
            .pending
            .iter()
            .find(|(_, row)| row.len() == self.group_size)
            .map(|(&id, _)| id)?;
        let samples = self.pending.remove(&ready).unwrap().into_values().collect();
        self.consumed.insert(ready);
        Some(samples)
        // END TODO C2
    }
}
#[derive(Clone, Debug)]
pub struct Policy {
    pub logits: Vec<f64>,
    pub version: u64,
}
impl Policy {
    pub fn probability(&self, context: usize) -> f64 {
        1.0 / (1.0 + (-self.logits[context]).exp())
    }
    /// Probability of `action` (0 or 1) in `context` under this policy.
    pub fn action_probability(&self, context: usize, action: usize) -> f64 {
        let p = self.probability(context);
        if action == 1 {
            p
        } else {
            1.0 - p
        }
    }
    // A6: sample one group from THIS snapshot with the given uniforms: action = 1 iff
    // u < p(context); reward = 1 iff action == context % 2; id = group * len + offset;
    // version = self.version; behavior_p = p(context) (probability of action 1).
    // Test: a_sample.
    pub fn sample(&self, group: usize, context: usize, uniforms: &[f64]) -> Vec<Sample> {
        // BEGIN TODO A6
        let p = self.probability(context);
        let mut samples = Vec::new();
        for (offset, &u) in uniforms.iter().enumerate() {
            let action = usize::from(u < p);
            let reward = if action == context % 2 { 1. } else { 0. };
            samples.push(Sample {
                id: group * uniforms.len() + offset,
                group,
                version: self.version,
                context,
                action,
                reward,
                behavior_p: p,
            });
        }
        samples
        // END TODO A6
    }
    // C3: the clipped surrogate this policy would maximize on `samples`:
    //   J = (1/G) sum_i min(rho_i A_i, clip(rho_i, 1-eps, 1+eps) A_i),
    //   rho_i = pi(a_i | c_i) / mu(a_i | c_i), A = advantages(rewards, false),
    // where pi uses the CURRENT logits and mu the recorded behavior_p (1 - behavior_p for
    // action 0). Empty samples -> 0. Pure: must not change self.
    // Test: c_objective (independent closed form) and c_gradient (finite differences).
    pub fn objective(&self, samples: &[Sample], epsilon: f64) -> f64 {
        // BEGIN TODO C3
        if samples.is_empty() {
            return 0.0;
        }
        let rewards: Vec<_> = samples.iter().map(|s| s.reward).collect();
        let adv = advantages(&rewards, false);
        let mut total = 0.0;
        for (s, &a) in samples.iter().zip(&adv) {
            let behavior = if s.action == 1 {
                s.behavior_p
            } else {
                1.0 - s.behavior_p
            };
            let ratio = self.action_probability(s.context, s.action) / behavior;
            let clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon);
            total += (ratio * a).min(clipped * a);
        }
        total / samples.len() as f64
        // END TODO C3
    }
    // C4: one gradient-ascent step on objective(): logits[c] += lr * dJ/dlogits[c], with
    // the gradient taken at the logits BEFORE the step (whole group, then one update).
    // Per sample, d/dz of the active branch is rho * A * (a - p); the branch is inactive
    // (gradient 0) only when A > 0 and rho > 1+eps, or A < 0 and rho < 1-eps.
    // Returns objective() evaluated before the step; increments version by one.
    // Test: c_gradient (finite differences of YOUR objective at rho != 1, all branches).
    pub fn update(&mut self, samples: &[Sample], lr: f64, epsilon: f64) -> f64 {
        // BEGIN TODO C4
        let before = self.objective(samples, epsilon);
        let rewards: Vec<_> = samples.iter().map(|s| s.reward).collect();
        let adv = advantages(&rewards, false);
        let mut gradient = vec![0.; self.logits.len()];
        for (sample, &a) in samples.iter().zip(&adv) {
            let p = self.probability(sample.context);
            let behavior = if sample.action == 1 {
                sample.behavior_p
            } else {
                1. - sample.behavior_p
            };
            let ratio = self.action_probability(sample.context, sample.action) / behavior;
            let inactive = (a > 0. && ratio > 1. + epsilon) || (a < 0. && ratio < 1. - epsilon);
            if !inactive {
                gradient[sample.context] += ratio * a * (sample.action as f64 - p);
            }
        }
        let n = samples.len().max(1) as f64;
        for (value, g) in self.logits.iter_mut().zip(gradient) {
            *value += lr * g / n;
        }
        self.version += 1;
        before
        // END TODO C4
    }
}
// C5: AReaL-style bounded staleness on the GENERATION side. The trainer consumes
// trajectories in submission order, `batch` per update, and the k-th batch (0-based) is
// trained when the policy is at version k. A rollout controller that has already
// submitted `generated` trajectories and holds weights of version `version` may submit
// the next one iff that trajectory will be at most `eta` versions old when trained.
// eta = 0 is synchronous RL. batch > 0.
// Test: c_staleness_bound (simulated producer/trainer: bound holds AND is reached).
pub fn may_submit(generated: u64, batch: u64, version: u64, eta: u64) -> bool {
    // BEGIN TODO C5
    assert!(batch > 0);
    generated / batch <= version + eta
    // END TODO C5
}
#[derive(Clone, Debug)]
pub struct TokenRecord {
    pub context: usize,
    pub action: usize,
    /// Version of the weights that actually sampled this token.
    pub version: u64,
    /// Probability of action 1 under that version, recorded at sampling time.
    pub behavior_p: f64,
}
// C6: decoupled PPO terms (AReaL Eq. 5) for a trajectory produced in segments by several
// versions (partial rollout: interrupted, then resumed with newer weights). For each
// token return (w, u) with w = pi_prox(a) / pi_behav(a) and u = pi_theta(a) / pi_prox(a):
// behav is the version that ACTUALLY sampled the token (its recorded behavior_p, per
// token, not per trajectory), prox is the policy `prox` (trust-region center), theta is
// `current`. Probabilities of action 0 are 1 - p. Test: c_partial_ratios.
pub fn decoupled_terms(tokens: &[TokenRecord], prox: &Policy, current: &Policy) -> Vec<(f64, f64)> {
    // BEGIN TODO C6
    tokens
        .iter()
        .map(|t| {
            let behav = if t.action == 1 {
                t.behavior_p
            } else {
                1.0 - t.behavior_p
            };
            let p_prox = prox.action_probability(t.context, t.action);
            let p_cur = current.action_probability(t.context, t.action);
            (p_prox / behav, p_cur / p_prox)
        })
        .collect()
    // END TODO C6
}
pub struct Learning {
    pub snapshots: Vec<[u64; 5]>,
    pub rollout_tokens: u64,
    pub rollout_ticks: u64,
    pub policy: Policy,
    pub updates: usize,
    pub reward: Vec<f64>,
    pub pressure: usize,
    pub stale: usize,
}
pub fn uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / (9007199254740992.0)
}
/// Exact expected reward of a policy over the four contexts (not a noisy batch mean).
pub fn expected_reward(policy: &Policy) -> f64 {
    (0..4)
        .map(|c| policy.action_probability(c, c % 2))
        .sum::<f64>()
        / 4.0
}
// C7: rollout/trainer loop. Each wave: snapshot the policy, sample 4 groups x 8 (group
// id = wave*4 + k, context = group % 4, uniforms from `uniform(&mut state)` with
// state = seed), execute their synthetic token work with B3 (estimate 8 + 4c, actual
// 4 + 6c + 3a, 4 workers, quantum 8, steal on), and hand groups to a GroupQueue
// (capacity, group_size 8, max_lag) in order of their last terminal segment. Before each
// push set queue.version = policy.version; a group older than max_lag is dropped whole
// (stale += 8); when the queue has no free slot, pop one group, update (lr 0.3, eps 0.2),
// and retry the blocked group. After the wave, drain and update. Record
// expected_reward and the cumulative [updates, backpressure, stale, tokens, ticks].
// Test: c_learning_closed_loop.
pub fn learn(seed: u64, waves: usize, capacity: usize, max_lag: u64) -> Learning {
    // BEGIN TODO C7
    let mut policy = Policy {
        logits: vec![0.; 4],
        version: 0,
    };
    let mut queue = GroupQueue::new(capacity, 8, max_lag);
    let mut state = seed;
    let mut reward = Vec::new();
    let mut snapshots = Vec::new();
    let mut updates = 0;
    let mut rollout_tokens = 0;
    let mut rollout_ticks = 0;
    for wave in 0..waves {
        let snapshot = policy.clone();
        let mut produced = VecDeque::new();
        for offset in 0..4 {
            let group = wave * 4 + offset;
            let uniforms: Vec<_> = (0..8).map(|_| uniform(&mut state)).collect();
            produced.push_back(snapshot.sample(group, group % 4, &uniforms));
        }
        let jobs: Vec<_> = produced
            .iter()
            .flatten()
            .map(|sample| Job {
                id: sample.id,
                group: sample.group,
                version: sample.version,
                estimate: 8 + sample.context as u64 * 4,
                actual: 4 + sample.context as u64 * 6 + sample.action as u64 * 3,
            })
            .collect();
        let timeline = schedule(&jobs, 4, 8, true);
        rollout_tokens += timeline.busy;
        rollout_ticks += timeline.finish;
        let mut finished_at = BTreeMap::new();
        for (order, segment) in timeline.segments.iter().enumerate() {
            if segment.terminal {
                finished_at.insert(segment.group, order);
            }
        }
        let mut groups: Vec<_> = produced.drain(..).collect();
        groups.sort_by_key(|group| finished_at[&group[0].group]);
        produced.extend(groups);
        while let Some(group) = produced.pop_front() {
            queue.version = policy.version;
            if queue.version - group[0].version > queue.max_lag {
                queue.stale += group.len();
                continue;
            }
            if queue.pending.len() == capacity {
                queue.backpressure += 1;
                if let Some(batch) = queue.pop() {
                    policy.update(&batch, 0.3, 0.2);
                    updates += 1;
                }
                produced.push_front(group);
                continue;
            }
            for sample in group {
                assert!(queue.push(sample));
            }
        }
        loop {
            queue.version = policy.version;
            let Some(batch) = queue.pop() else {
                break;
            };
            policy.update(&batch, 0.3, 0.2);
            updates += 1;
        }
        reward.push(expected_reward(&policy));
        snapshots.push([
            updates as u64,
            queue.backpressure as u64,
            queue.stale as u64,
            rollout_tokens,
            rollout_ticks,
        ]);
    }
    Learning {
        snapshots,
        rollout_tokens,
        rollout_ticks,
        policy,
        updates,
        reward,
        pressure: queue.backpressure,
        stale: queue.stale,
    }
    // END TODO C7
}
