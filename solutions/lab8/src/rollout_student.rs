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
    // A1: retain a bounded per-prompt-class history; prediction is empirical upper
    // median of PAST completed lengths (cold start=32). No unfinished length input.
    // Test: a_length_history validates window eviction and class isolation.
    pub fn observe(&mut self, class: usize, length: u64) {
        // BEGIN TODO A1
        let row = self.history.entry(class).or_default();
        row.push_back(length);
        while row.len() > self.window.max(1) {
            row.pop_front();
        }
        // END TODO A1
    }
    // A2: upper empirical median of the retained class history; cold start is 32.
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
// A3: LPT list scheduling uses predicted lengths only, ties by id then worker id.
// Test: a_assign checks exact hand partition and conservation for empty/uneven jobs.
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
// A4: population standard deviation, group-local rewards. Constant/empty groups ->0.
// This is a teaching GRPO-style normalization convention, not all trainers' default.
// Test: a_advantage checks translation/scale invariance and zero group.
pub fn advantages(rewards: &[f64]) -> Vec<f64> {
    // BEGIN TODO A4
    if rewards.is_empty() {
        return vec![];
    }
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rewards.len() as f64;
    if var < 1e-20 {
        return vec![0.; rewards.len()];
    }
    rewards.iter().map(|r| (r - mean) / var.sqrt()).collect()
    // END TODO A4
}
// A5: build a histogram with explicit overflow bucket and return empirical quantile.
// Bounds sorted increasing; bins include values <= bound. Test: a_histogram.
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
// B1: accept out-of-order segments but reject gaps/overlap, mixed identity/version,
// multiple terminal segments, and trailing tokens after terminal. Test: b_reassemble.
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
// B2: all preselected IDs must appear exactly once and every group must be complete.
// Test: b_batch_gate catches selection bias from training only fast-completing groups.
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
// B3: event-driven quantum scheduler. Worker queues start from predicted LPT.
// On idle, steal the tail of the longest QUEUE (ties by worker ID). Only the executor
// uses actual length to terminate a quantum; policy must not inspect actual to steal.
// A job resumes with the same policy version and exact next token position.
// Test: b_scheduler validates work conservation, segment reassembly, deterministic time.
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
    // C1: reserve capacity per group, admit samples atomically, reject stale/future,
    // duplicate, mixed-version and mismatched id/group. Returns false for backpressure:
    // producer retains that sample and retries AFTER a dequeue. Test: c_queue_admission.
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
    // C2: evict entire stale groups, then return the oldest complete group in ID order.
    // Never return a partial group or combine versions. Test: c_queue_drain.
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
#[derive(Clone)]
pub struct Policy {
    pub logits: Vec<f64>,
    pub version: u64,
}
impl Policy {
    pub fn probability(&self, context: usize) -> f64 {
        1.0 / (1.0 + (-self.logits[context]).exp())
    }
    // A6: sample a two-action contextual policy with supplied uniforms. Keep behavior
    // probabilities from the SAME snapshot as actions; reward=1 iff action=context%2.
    // This is a real stochastic policy, not a fabricated learning curve. Test: a_sample.
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
    // C3: one clipped importance-weighted policy-gradient update, group advantages.
    // Differentiate the clipped surrogate: clipped branch has ZERO gradient.
    // Use the OLD current logits for all members, then apply one averaged update.
    // Test: c_gradient finite differences + off-policy clipped branch + zero rewards.
    pub fn update(&mut self, samples: &[Sample], lr: f64, epsilon: f64) -> f64 {
        // BEGIN TODO C3
        let rewards: Vec<_> = samples.iter().map(|s| s.reward).collect();
        let adv = advantages(&rewards);
        let mut gradient = vec![0.; self.logits.len()];
        let mut objective = 0.;
        for (sample, &a) in samples.iter().zip(&adv) {
            let p = self.probability(sample.context);
            let (current, behavior) = if sample.action == 1 {
                (p, sample.behavior_p)
            } else {
                (1. - p, 1. - sample.behavior_p)
            };
            let ratio = current / behavior;
            let clipped = ratio.clamp(1. - epsilon, 1. + epsilon);
            objective += (ratio * a).min(clipped * a);
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
        objective / n
        // END TODO C3
    }
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
// C4: rollouter/trainer feedback loop. Generate groups in waves from one policy
// snapshot, push through a bounded queue, drain/update under backpressure, broadcast
// the new version, reject stale groups atomically. Test: c_learning_closed_loop.
// The token executor is B3; this routine measures actual policy learning, NOT GPU speed.
pub fn learn(seed: u64, waves: usize, capacity: usize, max_lag: u64) -> Learning {
    // BEGIN TODO C4
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
        // Execute synthetic token workloads for the actual sampled trajectories.
        // Only the executor knows realized work; estimates use prompt context.
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
        let mean = (0..4)
            .map(|c| {
                if c % 2 == 1 {
                    policy.probability(c)
                } else {
                    1. - policy.probability(c)
                }
            })
            .sum::<f64>()
            / 4.;
        reward.push(mean);
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
    // END TODO C4
}
