//! Lab 8 contract tests. Every test prints one PASS/FAIL line per check and fails at the
//! end, so one run lists everything that is wrong. Oracles are written here, independent
//! of the code under test: hand-derived values (derivations in comments), closed-form
//! order statistics, test-side simulators, and finite differences of YOUR objective.
use rollout_scheduling::*;
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Debug;

struct Checks {
    test: &'static str,
    failures: Vec<String>,
}
impl Checks {
    fn new(test: &'static str) -> Self {
        Self {
            test,
            failures: Vec::new(),
        }
    }
    fn check(&mut self, name: &str, ok: bool, detail: impl FnOnce() -> String) {
        if ok {
            println!("PASS {}/{}", self.test, name);
        } else {
            let line = format!("FAIL {}/{}: {}", self.test, name, detail());
            println!("{line}");
            self.failures.push(line);
        }
    }
    fn close(&mut self, name: &str, got: f64, want: f64, tol: f64) {
        self.check(name, (got - want).abs() <= tol, || {
            format!("expected {want} vs got {got} (tol {tol})")
        });
    }
    fn eq<T: PartialEq + Debug>(&mut self, name: &str, got: T, want: T) {
        let ok = got == want;
        self.check(name, ok, || format!("expected {want:?} vs got {got:?}"));
    }
    fn done(self) {
        if !self.failures.is_empty() {
            panic!(
                "{} check(s) failed in {}:\n{}",
                self.failures.len(),
                self.test,
                self.failures.join("\n")
            );
        }
    }
}
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}
fn samples(group: usize, version: u64) -> Vec<Sample> {
    // Hand-built (not via A6): context 1, logit 0 => p = 0.5; uniforms 0.1,0.9,0.2,0.8.
    [1usize, 0, 1, 0]
        .iter()
        .enumerate()
        .map(|(k, &action)| Sample {
            id: group * 4 + k,
            group,
            version,
            context: 1,
            action,
            reward: if action == 1 { 1.0 } else { 0.0 },
            behavior_p: 0.5,
        })
        .collect()
}

// ------------------------------------------------------------------- A
#[test]
fn a_length_history() {
    let mut c = Checks::new("a_length_history");
    let mut m = LengthModel {
        window: 3,
        ..Default::default()
    };
    c.eq("cold start", m.predict(9), 32);
    for length in [5, 1, 9, 7] {
        m.observe(0, length);
    }
    // window 3 keeps [1,9,7] -> sorted [1,7,9], upper median sorted[1] = 7
    c.eq("window eviction + median", m.predict(0), 7);
    c.eq(
        "retained length",
        m.history.get(&0).map(|r| r.len()),
        Some(3),
    );
    c.eq("other class still cold", m.predict(1), 32);
    let before: Vec<u64> = m.history[&0].iter().copied().collect();
    let _ = m.predict(0);
    c.eq(
        "predict is pure",
        m.history[&0].iter().copied().collect::<Vec<_>>(),
        before,
    );
    let mut even = LengthModel {
        window: 4,
        ..Default::default()
    };
    for length in [4, 1, 3, 2] {
        even.observe(2, length);
    }
    c.eq(
        "upper (not lower) median of even history",
        even.predict(2),
        3,
    );
    let mut zero = LengthModel::default();
    zero.observe(0, 10);
    zero.observe(0, 20);
    c.eq("window 0 behaves as 1", zero.predict(0), 20);
    c.done();
}

fn jobs_of(estimates: &[u64], ids: &[usize]) -> Vec<PredictedJob> {
    ids.iter()
        .zip(estimates)
        .map(|(&id, &estimated_tokens)| PredictedJob {
            id,
            estimated_tokens,
        })
        .collect()
}
/// Test-side list scheduler (independent of A3).
fn list_oracle(jobs: &[PredictedJob], workers: usize, lpt: bool) -> Vec<Vec<usize>> {
    let mut order: Vec<&PredictedJob> = jobs.iter().collect();
    if lpt {
        order.sort_by_key(|j| (std::cmp::Reverse(j.estimated_tokens), j.id));
    }
    let mut q = vec![vec![]; workers];
    let mut load = vec![0u64; workers];
    for j in order {
        let mut best = 0;
        for w in 1..workers {
            if load[w] < load[best] {
                best = w;
            }
        }
        q[best].push(j.id);
        load[best] += j.estimated_tokens;
    }
    q
}
#[test]
fn a_assign() {
    let mut c = Checks::new("a_assign");
    // Lecture D8-2: estimates [2,3,2,3,2] (NOT pre-sorted), two workers.
    let jobs = jobs_of(&[2, 3, 2, 3, 2], &[0, 1, 2, 3, 4]);
    // input order: 0->w0 [2,0]; 1->w1 [2,3]; 2->w0 [4,3]; 3->w1 [4,6]; 4->w0 [6,6]
    c.eq(
        "lpt=false keeps input order",
        assign(&jobs, 2, false),
        vec![vec![0, 2, 4], vec![1, 3]],
    );
    // LPT visits 1,3,0,2,4: w0 [3], w1 [3], tie -> w0 [5,3], w1 [5,5], w0 [7,5]
    c.eq(
        "lpt=true exact queues",
        assign(&jobs, 2, true),
        vec![vec![1, 0, 4], vec![3, 2]],
    );
    // Equal estimates, shuffled ids: LPT orders by id, input order does not.
    let tie = jobs_of(&[5; 5], &[4, 2, 0, 3, 1]);
    c.eq(
        "lpt tie-break by id",
        assign(&tie, 2, true),
        vec![vec![0, 2, 4], vec![1, 3]],
    );
    c.eq(
        "lpt=false ignores ids",
        assign(&tie, 2, false),
        vec![vec![4, 0, 1], vec![2, 3]],
    );
    c.eq(
        "empty input",
        assign(&[], 2, true),
        vec![Vec::<usize>::new(), vec![]],
    );
    c.eq(
        "more workers than jobs",
        assign(&jobs_of(&[1, 9], &[0, 1]), 3, true),
        vec![vec![1], vec![0], vec![]],
    );
    let mut rng = Lcg(11);
    let mut bad = 0;
    for case in 0..60 {
        let n = (rng.next() % 9) as usize;
        let ids: Vec<usize> = {
            let mut v: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                v.swap(i, (rng.next() as usize) % (i + 1));
            }
            v
        };
        let est: Vec<u64> = (0..n).map(|_| 1 + rng.next() % 6).collect();
        let jobs = jobs_of(&est, &ids);
        let workers = 1 + (rng.next() % 4) as usize;
        let lpt = case % 2 == 0;
        if assign(&jobs, workers, lpt) != list_oracle(&jobs, workers, lpt) {
            bad += 1;
        }
    }
    c.eq("60 random cases vs test-side list scheduler", bad, 0);
    c.done();
}
#[test]
fn a_advantage() {
    let mut c = Checks::new("a_advantage");
    let vec_close = |c: &mut Checks, name: &str, got: Vec<f64>, want: &[f64]| {
        let ok = got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-9);
        c.check(name, ok, || format!("expected {want:?} vs got {got:?}"));
    };
    vec_close(
        &mut c,
        "population [0,1]",
        advantages(&[0., 1.], false),
        &[-1., 1.],
    );
    vec_close(
        &mut c,
        "population [10,12]",
        advantages(&[10., 12.], false),
        &[-1., 1.],
    );
    // sample std of [0,1] = sqrt(0.5) -> +-1/sqrt(2)
    let h = 0.5f64.sqrt();
    vec_close(
        &mut c,
        "bessel [0,1]",
        advantages(&[0., 1.], true),
        &[-h, h],
    );
    // [0,0,1,1]: population var 0.25 -> +-1; sample var 1/3 -> +-0.5*sqrt(3)
    vec_close(
        &mut c,
        "population [0,0,1,1]",
        advantages(&[0., 0., 1., 1.], false),
        &[-1., -1., 1., 1.],
    );
    let s = 0.5 * 3f64.sqrt();
    vec_close(
        &mut c,
        "bessel [0,0,1,1]",
        advantages(&[0., 0., 1., 1.], true),
        &[-s, -s, s, s],
    );
    vec_close(
        &mut c,
        "constant group",
        advantages(&[1., 1., 1.], true),
        &[0., 0., 0.],
    );
    vec_close(
        &mut c,
        "single, population",
        advantages(&[3.], false),
        &[0.],
    );
    vec_close(&mut c, "single, bessel", advantages(&[3.], true), &[0.]);
    c.eq("empty", advantages(&[], true).len(), 0);
    let a = advantages(&[0., 3., 1., 1., 5.], false);
    let mean = a.iter().sum::<f64>() / 5.;
    let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 5.;
    c.close("population: mean 0", mean, 0., 1e-12);
    c.close("population: variance 1", var, 1., 1e-12);
    c.done();
}
#[test]
fn a_histogram() {
    let mut c = Checks::new("a_histogram");
    // buckets (-inf,4], (4,10], (10,inf); q=0.5 -> ceil(4*0.5)=2 -> sorted[2]=5
    c.eq(
        "counts + median",
        histogram(&[1, 4, 5, 10, 11], &[4, 10], 0.5),
        (vec![2, 2, 1], 5),
    );
    // q=0.3 -> ceil(1.2)=2 -> 5 (floor would give 4)
    c.eq(
        "ceil rank",
        histogram(&[11, 1, 10, 4, 5], &[4, 10], 0.3).1,
        5,
    );
    c.eq("q=0 is min", histogram(&[7, 3, 9], &[5], 0.0).1, 3);
    c.eq("q=1 is max", histogram(&[7, 3, 9], &[5], 1.0).1, 9);
    c.eq(
        "bound is inclusive",
        histogram(&[5, 6], &[5], 0.0).0,
        vec![1, 1],
    );
    c.eq("empty", histogram(&[], &[4], 0.9), (vec![0, 0], 0));
    c.done();
}
#[test]
fn a_sample() {
    let mut c = Checks::new("a_sample");
    let p = Policy {
        logits: vec![0.; 4],
        version: 3,
    };
    let s = p.sample(2, 1, &[0.1, 0.9]);
    c.eq(
        "actions (u < p)",
        s.iter().map(|s| s.action).collect::<Vec<_>>(),
        vec![1, 0],
    );
    c.eq(
        "ids = group*len+offset",
        s.iter().map(|s| s.id).collect::<Vec<_>>(),
        vec![4, 5],
    );
    c.eq(
        "version from snapshot",
        s.iter().map(|s| s.version).collect::<Vec<_>>(),
        vec![3, 3],
    );
    c.eq(
        "behavior_p = p(action 1)",
        s.iter().map(|s| s.behavior_p).collect::<Vec<_>>(),
        vec![0.5, 0.5],
    );
    c.eq(
        "reward iff action == context % 2",
        s.iter().map(|s| s.reward).collect::<Vec<_>>(),
        vec![1., 0.],
    );
    let q = Policy {
        logits: vec![2.0, 0., 0., 0.],
        version: 0,
    };
    let p0 = sigmoid(2.0);
    let s = q.sample(0, 0, &[p0, p0 - 1e-9, 0.99]);
    c.eq("u == p gives action 0 (strict <)", s[0].action, 0);
    c.eq(
        "context 0 rewards action 0",
        s.iter().map(|s| s.reward).collect::<Vec<_>>(),
        vec![1., 0., 1.],
    );
    c.check(
        "behavior_p = P(action 1) for every sample, also action 0",
        s.iter().all(|x| (x.behavior_p - p0).abs() < 1e-15),
        || format!("{:?}", s.iter().map(|x| x.behavior_p).collect::<Vec<_>>()),
    );
    c.done();
}

// ------------------------------------------------------------------- B
fn segment(start: usize, tokens: Vec<u32>, terminal: bool) -> Segment {
    Segment {
        id: 0,
        group: 0,
        version: 7,
        start,
        tokens,
        terminal,
    }
}
#[test]
fn b_reassemble() {
    let mut c = Checks::new("b_reassemble");
    let ok = reassemble(&[
        segment(3, vec![13], true),
        segment(0, vec![10, 11], false),
        segment(2, vec![12], false),
    ]);
    c.eq(
        "out-of-order input restored",
        ok.map(|x| x.tokens),
        Ok(vec![10, 11, 12, 13]),
    );
    let cases: Vec<(&str, Vec<Segment>)> = vec![
        ("empty", vec![]),
        ("unfinished", vec![segment(0, vec![1, 2], false)]),
        (
            "two terminals",
            vec![segment(0, vec![1, 2], true), segment(2, vec![3], true)],
        ),
        (
            "terminal before last",
            vec![segment(0, vec![1, 2], true), segment(2, vec![3], false)],
        ),
        (
            "overlap",
            vec![segment(0, vec![1, 2], false), segment(1, vec![3], true)],
        ),
        (
            "gap",
            vec![segment(0, vec![1, 2], false), segment(3, vec![3], true)],
        ),
        ("not starting at 0", vec![segment(1, vec![1], true)]),
        (
            "empty segment",
            vec![
                segment(0, vec![1], false),
                segment(1, vec![], false),
                segment(1, vec![2], true),
            ],
        ),
    ];
    for (name, bad) in cases {
        c.check(
            &format!("rejects {name}"),
            reassemble(&bad).is_err(),
            || "accepted".into(),
        );
    }
    let mut other = segment(2, vec![3], true);
    other.version = 8;
    c.check(
        "rejects mixed version",
        reassemble(&[segment(0, vec![1, 2], false), other]).is_err(),
        || "accepted".into(),
    );
    let mut other = segment(2, vec![3], true);
    other.id = 1;
    c.check(
        "rejects mixed id",
        reassemble(&[segment(0, vec![1, 2], false), other]).is_err(),
        || "accepted".into(),
    );
    c.done();
}
#[test]
fn b_batch_gate() {
    let mut c = Checks::new("b_batch_gate");
    let completed: Vec<_> = (0..4)
        .map(|id| Completed {
            id,
            group: id / 2,
            version: 2,
            tokens: vec![1],
        })
        .collect();
    c.check(
        "full batch accepted",
        training_ready(&completed, &[0, 1, 2, 3], 2, 2).is_ok(),
        || "rejected".into(),
    );
    c.check(
        "only the fast group",
        training_ready(&completed[..2], &[0, 1, 2, 3], 2, 2).is_err(),
        || "accepted".into(),
    );
    c.check(
        "wrong version",
        training_ready(&completed, &[0, 1, 2, 3], 2, 3).is_err(),
        || "accepted".into(),
    );
    let dup = [completed[0].clone(), completed[0].clone()];
    c.check(
        "duplicate id",
        training_ready(&dup, &[0, 1], 2, 2).is_err(),
        || "accepted".into(),
    );
    let mut extra = completed.clone();
    extra.push(Completed {
        id: 9,
        group: 4,
        version: 2,
        tokens: vec![1],
    });
    c.check(
        "unexpected id",
        training_ready(&extra, &[0, 1, 2, 3], 2, 2).is_err(),
        || "accepted".into(),
    );
    c.check(
        "group size mismatch",
        training_ready(&completed, &[0, 1, 2, 3], 4, 2).is_err(),
        || "accepted".into(),
    );
    c.done();
}
#[test]
fn b_scheduler() {
    let mut c = Checks::new("b_scheduler");
    let jobs: Vec<_> = [20, 1, 3, 2, 15, 2]
        .iter()
        .enumerate()
        .map(|(id, &actual)| Job {
            id,
            group: id / 2,
            version: 4,
            estimate: 4,
            actual,
        })
        .collect();
    for steal in [false, true] {
        let r = schedule(&jobs, 3, 3, steal);
        c.eq(&format!("steal={steal}: token work conserved"), r.busy, 43);
        c.check(
            &format!("steal={steal}: lower/upper bound"),
            r.finish >= 20 && r.finish <= 43,
            || format!("finish {}", r.finish),
        );
        let mut completed = vec![];
        let mut good = true;
        for job in &jobs {
            let segments: Vec<_> = r
                .segments
                .iter()
                .filter(|s| s.id == job.id)
                .cloned()
                .collect();
            match reassemble(&segments) {
                Ok(done) => {
                    let want: Vec<u32> = (0..job.actual)
                        .map(|v| job.id as u32 * 10000 + v as u32)
                        .collect();
                    good &= done.tokens == want;
                    completed.push(done);
                }
                Err(_) => good = false,
            }
        }
        c.check(
            &format!("steal={steal}: every job reassembles in order"),
            good,
            || "segments broken".into(),
        );
        c.check(
            &format!("steal={steal}: batch gate accepts"),
            training_ready(&completed, &[0, 1, 2, 3, 4, 5], 2, 4).is_ok(),
            || "rejected".into(),
        );
    }
    c.eq(
        "single worker = total work",
        schedule(&jobs, 1, 3, true).finish,
        43,
    );
    // Hand timeline: actual [6,1,1,1], estimates all 2, quantum 3, 2 workers.
    // LPT queues w0 [0,2], w1 [1,3]. No steal: w0 runs 0(3),2(1),0(3) -> 7.
    // Steal: w1 finishes 1 at t=1, 3 at t=2, then steals 2 from w0's tail -> w0 6.
    let hand: Vec<_> = [6, 1, 1, 1]
        .iter()
        .enumerate()
        .map(|(id, &actual)| Job {
            id,
            group: id / 2,
            version: 0,
            estimate: 2,
            actual,
        })
        .collect();
    c.eq(
        "hand: no steal finish",
        schedule(&hand, 2, 3, false).finish,
        7,
    );
    let stolen = schedule(&hand, 2, 3, true);
    c.eq("hand: steal finish", stolen.finish, 6);
    c.eq("hand: one steal", stolen.steals, 1);
    // Steal from the TAIL of a 2-deep queue. Estimates [100,10,10,10] put job 0 alone on
    // w0 and jobs 1,2,3 on w1; actual [1,5,2,4], quantum 10.
    // t=1 w0 idle, steals the tail (job 3) -> runs to t=5. At t=5 both finish; idle
    // workers pick in index order, so w0 steals job 2 from w1's queue before w1 looks at
    // it -> t=7 (two steals). Completion order of segments: 0 (t=1), 3 and 1 (t=5, worker
    // order), 2 (t=7). Stealing the head (job 2 first) would give [0,2,1,3].
    let tail: Vec<_> = [(100, 1), (10, 5), (10, 2), (10, 4)]
        .iter()
        .enumerate()
        .map(|(id, &(estimate, actual))| Job {
            id,
            group: 0,
            version: 0,
            estimate,
            actual,
        })
        .collect();
    let r = schedule(&tail, 2, 10, true);
    c.eq(
        "tail steal: segment completion order",
        r.segments.iter().map(|s| s.id).collect::<Vec<_>>(),
        vec![0, 3, 1, 2],
    );
    c.eq("tail steal: steals", r.steals, 2);
    c.eq("tail steal: finish", r.finish, 7);
    c.done();
}
#[test]
fn b_keep_first_n() {
    let mut c = Checks::new("b_keep_first_n");
    // positions 0..3 carry sample_idx [3,1,2,0] and finish [5,1,3,1]: the two finish=1
    // samples tie and are ordered by sample_idx (pos3 has idx 0, pos1 has idx 1), which is
    // the opposite of their positions.
    let group: Vec<Finished> = [(3, 5.0), (1, 1.0), (2, 3.0), (0, 1.0)]
        .iter()
        .map(|&(sample_idx, finish)| Finished {
            sample_idx,
            finish,
            tokens: finish as u64,
            reward: 0.0,
        })
        .collect();
    c.eq(
        "keep 2: positions, tie by sample_idx",
        keep_first_n(&group, 2),
        vec![3, 1],
    );
    c.eq("keep 3", keep_first_n(&group, 3), vec![3, 1, 2]);
    c.eq(
        "n > len keeps all, ordered",
        keep_first_n(&group, 9),
        vec![3, 1, 2, 0],
    );
    c.eq("n = 0", keep_first_n(&group, 0), Vec::<usize>::new());
    // Exact oracle: m = 3 i.i.d. lengths uniform on {1,2,3}, finish time = length,
    // reward = [length >= 2]; keep the first n = 2.
    //   E[max] = sum_k P(max >= k) = 1 + 26/27 + 19/27 = 8/3, so the mean kept length is
    //   (E[sum] - E[max]) / 2 = (6 - 8/3) / 2 = 5/3, versus 2 for all samples.
    //   The dropped sample has length >= 2 unless all three are 1 (prob 1/27), so
    //   E[#kept rewarded] = 3 * 2/3 - 26/27 = 28/27 and the kept mean reward is 14/27,
    //   versus 2/3 for all samples.
    let (mut len, mut rew, mut count) = (0.0, 0.0, 0.0);
    for code in 0..27usize {
        let lengths = [code % 3 + 1, code / 3 % 3 + 1, code / 9 + 1];
        let group: Vec<Finished> = lengths
            .iter()
            .enumerate()
            .map(|(i, &l)| Finished {
                sample_idx: i,
                finish: l as f64,
                tokens: l as u64,
                reward: f64::from(l >= 2),
            })
            .collect();
        for i in keep_first_n(&group, 2) {
            len += group[i].tokens as f64;
            rew += group[i].reward;
            count += 1.0;
        }
    }
    c.eq("kept count over 27 tuples", count as usize, 54);
    c.close(
        "mean kept length = 5/3 (all: 2)",
        len / count,
        5.0 / 3.0,
        1e-12,
    );
    c.close(
        "mean kept reward = 14/27 (all: 2/3)",
        rew / count,
        14.0 / 27.0,
        1e-12,
    );
    c.done();
}

// ------------------------------------------------------------------- C
#[test]
fn c_queue_admission() {
    let mut c = Checks::new("c_queue_admission");
    let mut q = GroupQueue::new(1, 4, 1);
    let s = samples(0, 0);
    c.check("first member admitted", q.push(s[0].clone()), || {
        "rejected".into()
    });
    c.check("duplicate rejected", !q.push(s[0].clone()), || {
        "accepted".into()
    });
    c.check(
        "new group blocked when full",
        !q.push(samples(1, 0)[0].clone()),
        || "accepted".into(),
    );
    c.eq("backpressure counted", q.backpressure, 1);
    c.check("partial group not returned", q.pop().is_none(), || {
        "returned".into()
    });
    let mut rest = true;
    for sample in s.into_iter().skip(1) {
        rest &= q.push(sample);
    }
    c.check("members of a reserved group always fit", rest, || {
        "rejected".into()
    });
    c.eq("complete group returned", q.pop().map(|g| g.len()), Some(4));
    c.check(
        "consumed group rejected",
        !q.push(samples(0, 0)[0].clone()),
        || "accepted".into(),
    );
    c.check(
        "slot freed after pop",
        q.push(samples(1, 0)[0].clone()),
        || "rejected".into(),
    );
    let mut q = GroupQueue::new(2, 4, 1);
    q.version = 1;
    c.check(
        "lag 1 admitted (max_lag 1)",
        q.push(samples(0, 0)[0].clone()),
        || "rejected".into(),
    );
    let mut future = samples(0, 2)[1].clone();
    c.check("future version rejected", !q.push(future.clone()), || {
        "accepted".into()
    });
    future.version = 1;
    c.check(
        "mixed version inside a group rejected",
        !q.push(future),
        || "accepted".into(),
    );
    let mut mismatched = samples(1, 0)[0].clone(); // id 4, same version as group 0
    mismatched.group = 0;
    c.check("id/group mismatch rejected", !q.push(mismatched), || {
        "accepted".into()
    });
    q.version = 3;
    let before = q.stale;
    c.check(
        "lag 2 rejected (max_lag 1)",
        !q.push(samples(5, 1)[0].clone()),
        || "accepted".into(),
    );
    c.eq("stale counted", q.stale, before + 1);
    c.eq(
        "state unchanged by rejections",
        (q.pending.len(), q.pending[&0].len()),
        (1, 1),
    );
    c.done();
}
#[test]
fn c_queue_drain() {
    let mut c = Checks::new("c_queue_drain");
    let mut q = GroupQueue::new(3, 4, 1);
    for s in samples(0, 0) {
        q.push(s);
    }
    q.version = 1;
    for s in samples(1, 1) {
        q.push(s);
    }
    q.version = 2;
    let got = q.pop();
    c.eq(
        "fresh group returned",
        got.as_ref().map(|g| g[0].group),
        Some(1),
    );
    c.eq(
        "members in id order",
        got.map(|g| g.iter().map(|s| s.id).collect::<Vec<_>>()),
        Some(vec![4, 5, 6, 7]),
    );
    c.eq("whole stale group evicted", q.stale, 4);
    c.check(
        "evicted group cannot come back",
        !q.push(samples(0, 2)[0].clone()),
        || "accepted".into(),
    );
    c.check("nothing else ready", q.pop().is_none(), || {
        "returned".into()
    });
    c.done();
}

/// Independent closed form of the clipped surrogate for two-action samples.
fn surrogate_oracle(logits: &[f64], s: &[Sample], eps: f64, adv: &[f64]) -> f64 {
    let mut total = 0.0;
    for (x, &a) in s.iter().zip(adv) {
        let p = sigmoid(logits[x.context]);
        let (pi, mu) = if x.action == 1 {
            (p, x.behavior_p)
        } else {
            (1.0 - p, 1.0 - x.behavior_p)
        };
        let rho = pi / mu;
        total += (rho * a).min(rho.clamp(1.0 - eps, 1.0 + eps) * a);
    }
    total / s.len() as f64
}
fn two(actions: [usize; 2], rewards: [f64; 2]) -> Vec<Sample> {
    (0..2)
        .map(|k| Sample {
            id: k,
            group: 0,
            version: 0,
            context: 1,
            action: actions[k],
            reward: rewards[k],
            behavior_p: 0.5,
        })
        .collect()
}
#[test]
fn c_objective() {
    let mut c = Checks::new("c_objective");
    // (logit, actions, rewards); behavior p = 0.5, eps = 0.2, A = [+1,-1] or [-1,+1].
    let cases = [
        ("rho=1", 0.0, [1, 0], [1.0, 0.0]),
        ("rho!=1 unclipped", 0.3, [1, 0], [1.0, 0.0]),
        ("A<0 rho>1+eps active", 3.0, [1, 0], [0.0, 1.0]),
        ("both clipped", 3.0, [1, 0], [1.0, 0.0]),
    ];
    for (name, z, actions, rewards) in cases {
        let s = two(actions, rewards);
        let adv = if rewards[0] > rewards[1] {
            [1.0, -1.0]
        } else {
            [-1.0, 1.0]
        };
        let p = Policy {
            logits: vec![0.0, z, 0.0, 0.0],
            version: 0,
        };
        c.close(
            name,
            p.objective(&s, 0.2),
            surrogate_oracle(&p.logits, &s, 0.2, &adv),
            1e-12,
        );
    }
    // logit 3, A=[-1,+1]: (1/2)(-1.905148 + 0.094852) = -0.905148
    let p = Policy {
        logits: vec![0.0, 3.0, 0.0, 0.0],
        version: 5,
    };
    c.close(
        "hand value, logit 3",
        p.objective(&two([1, 0], [0.0, 1.0]), 0.2),
        -0.9051482536,
        1e-9,
    );
    c.eq("empty -> 0", p.objective(&[], 0.2), 0.0);
    c.eq(
        "objective is pure",
        (p.logits.clone(), p.version),
        (vec![0.0, 3.0, 0.0, 0.0], 5),
    );
    c.done();
}
#[test]
fn c_gradient() {
    let mut c = Checks::new("c_gradient");
    let (lr, eps, h) = (0.2, 0.2, 1e-6);
    // Cases from the lecture (behavior p = 0.5):
    //  rho=1     : z=0,   A=[+1,-1], mean grad = 0.5
    //  rho!=1    : z=0.3, A=[+1,-1], both ratios in [0.8,1.2]; grad = 2p(1-p) = 0.48892
    //              (forgetting the ratio factor gives 0.5)
    //  case 3+6  : z=3,   A=[-1,+1], rho1=1.905 (A<0, active) rho0=0.095 (A>0, active);
    //              grad = -2p(1-p) = -0.090353 ("zero when out of range" gives 0)
    //  case 4+5  : z=3,   A=[+1,-1], both on the clipped constant -> grad 0
    let p3 = sigmoid(3.0);
    let p03 = sigmoid(0.3);
    let cases = [
        ("rho=1", 0.0, [1, 0], [1.0, 0.0], 0.5),
        (
            "rho!=1 unclipped",
            0.3,
            [1, 0],
            [1.0, 0.0],
            2.0 * p03 * (1.0 - p03),
        ),
        (
            "A<0 rho>1+eps and A>0 rho<1-eps (active)",
            3.0,
            [1, 0],
            [0.0, 1.0],
            -2.0 * p3 * (1.0 - p3),
        ),
        (
            "A>0 rho>1+eps and A<0 rho<1-eps (clipped)",
            3.0,
            [1, 0],
            [1.0, 0.0],
            0.0,
        ),
    ];
    for (name, z, actions, rewards, want) in cases {
        let s = two(actions, rewards);
        let mut p = Policy {
            logits: vec![0.0, z, 0.0, 0.0],
            version: 7,
        };
        let at = |dz: f64| {
            Policy {
                logits: vec![0.0, z + dz, 0.0, 0.0],
                version: 7,
            }
            .objective(&s, eps)
        };
        let fd = (at(h) - at(-h)) / (2.0 * h);
        let j0 = p.objective(&s, eps);
        let returned = p.update(&s, lr, eps);
        let step = (p.logits[1] - z) / lr;
        c.close(
            &format!("{name}: step/lr = finite difference of your objective"),
            step,
            fd,
            1e-6,
        );
        c.close(
            &format!("{name}: step/lr = hand gradient"),
            step,
            want,
            1e-9,
        );
        c.close(
            &format!("{name}: returns objective before the step"),
            returned,
            j0,
            1e-12,
        );
        c.eq(&format!("{name}: version incremented"), p.version, 8);
        c.eq(
            &format!("{name}: other logits untouched"),
            (p.logits[0], p.logits[2], p.logits[3]),
            (0.0, 0.0, 0.0),
        );
    }
    let s = two([1, 0], [1.0, 1.0]);
    let mut p = Policy {
        logits: vec![0.0, 0.3, 0.0, 0.0],
        version: 0,
    };
    p.update(&s, lr, eps);
    c.eq("constant rewards: no change", p.logits[1], 0.3);
    // Two contexts in one group: each logit only gets its own samples' gradient.
    let mut mixed = two([1, 0], [1.0, 0.0]);
    mixed[1].context = 2;
    let mut p = Policy {
        logits: vec![0.0, 0.3, -0.4, 0.0],
        version: 0,
    };
    let base = p.logits.clone();
    let fd = |k: usize| {
        let mut up = base.clone();
        up[k] += h;
        let mut dn = base.clone();
        dn[k] -= h;
        (Policy {
            logits: up,
            version: 0,
        }
        .objective(&mixed, eps)
            - Policy {
                logits: dn,
                version: 0,
            }
            .objective(&mixed, eps))
            / (2.0 * h)
    };
    let (g1, g2) = (fd(1), fd(2));
    p.update(&mixed, lr, eps);
    c.close(
        "mixed contexts: logit 1",
        (p.logits[1] - base[1]) / lr,
        g1,
        1e-6,
    );
    c.close(
        "mixed contexts: logit 2",
        (p.logits[2] - base[2]) / lr,
        g2,
        1e-6,
    );
    c.done();
}
#[test]
fn c_staleness_bound() {
    let mut c = Checks::new("c_staleness_bound");
    // Direct values: floor(generated / batch) <= version + eta.
    c.eq("g=3,B=4,v=0,eta=0", may_submit(3, 4, 0, 0), true);
    c.eq("g=4,B=4,v=0,eta=0", may_submit(4, 4, 0, 0), false);
    c.eq("g=7,B=4,v=0,eta=1", may_submit(7, 4, 0, 1), true);
    c.eq("g=8,B=4,v=0,eta=1", may_submit(8, 4, 0, 1), false);
    c.eq("g=8,B=4,v=1,eta=0", may_submit(8, 4, 1, 0), false);
    // Simulator: a fast producer submits while allowed; the trainer consumes `batch`
    // oldest trajectories per tick and bumps the version. Staleness of a trajectory =
    // trainer version at consumption - version that generated it. The bound must hold
    // AND be reached (a stricter rule wastes overlap; a looser one breaks the bound).
    for batch in [1u64, 4] {
        for eta in 0u64..4 {
            let (mut version, mut generated) = (0u64, 0u64);
            let mut buffer: VecDeque<u64> = VecDeque::new();
            let mut max_stale = 0;
            for _tick in 0..60 {
                let mut submitted = 0;
                while submitted < 64 && may_submit(generated, batch, version, eta) {
                    buffer.push_back(version);
                    generated += 1;
                    submitted += 1;
                }
                if buffer.len() as u64 >= batch {
                    for _ in 0..batch {
                        let v = buffer.pop_front().unwrap();
                        max_stale = max_stale.max(version - v);
                    }
                    version += 1;
                }
            }
            c.check(
                &format!("B={batch} eta={eta}: no deadlock"),
                version >= 50,
                || format!("only {version} updates in 60 ticks"),
            );
            c.eq(
                &format!("B={batch} eta={eta}: max staleness == eta"),
                max_stale,
                eta,
            );
        }
    }
    c.done();
}
#[test]
fn c_partial_ratios() {
    let mut c = Checks::new("c_partial_ratios");
    // A trajectory sampled by v5 (tokens 0,1), interrupted, resumed by v6 (tokens 2,3).
    // prox = v7 (weights before this update), current = theta after one step.
    let history: BTreeMap<u64, Vec<f64>> = [
        (5, vec![0.0, 0.4, -0.2, 0.0]),
        (6, vec![0.0, 0.9, 0.1, 0.0]),
    ]
    .into_iter()
    .collect();
    let prox = Policy {
        logits: vec![0.0, 1.2, 0.3, 0.0],
        version: 7,
    };
    let current = Policy {
        logits: vec![0.0, 1.5, 0.2, 0.0],
        version: 7,
    };
    let spec = [(1usize, 1usize, 5u64), (2, 0, 5), (1, 0, 6), (2, 1, 6)];
    let tokens: Vec<TokenRecord> = spec
        .iter()
        .map(|&(context, action, version)| TokenRecord {
            context,
            action,
            version,
            behavior_p: sigmoid(history[&version][context]),
        })
        .collect();
    let prob = |logits: &[f64], ctx: usize, a: usize| {
        if a == 1 {
            sigmoid(logits[ctx])
        } else {
            1.0 - sigmoid(logits[ctx])
        }
    };
    let got = decoupled_terms(&tokens, &prox, &current);
    c.eq("one term per token", got.len(), 4);
    for (k, &(ctx, a, v)) in spec.iter().enumerate() {
        let behav = prob(&history[&v], ctx, a);
        let w = prob(&prox.logits, ctx, a) / behav;
        let u = prob(&current.logits, ctx, a) / prob(&prox.logits, ctx, a);
        if let Some(&(gw, gu)) = got.get(k) {
            c.close(
                &format!("token {k} (sampled by v{v}): w = pi_prox/pi_behav"),
                gw,
                w,
                1e-12,
            );
            c.close(&format!("token {k}: u = pi_theta/pi_prox"), gu, u, 1e-12);
            c.close(
                &format!("token {k}: w*u = pi_theta/pi_behav"),
                gw * gu,
                prob(&current.logits, ctx, a) / behav,
                1e-12,
            );
        }
    }
    // On-policy special case: prox = the sampling policy -> w = 1.
    let v6 = Policy {
        logits: history[&6].clone(),
        version: 6,
    };
    let resumed = decoupled_terms(&tokens[2..], &v6, &v6);
    c.check(
        "prox = behavior gives w = 1, u = 1",
        resumed
            .iter()
            .all(|&(w, u)| (w - 1.0).abs() < 1e-12 && (u - 1.0).abs() < 1e-12),
        || format!("{resumed:?}"),
    );
    c.done();
}
#[test]
fn c_learning_closed_loop() {
    let mut c = Checks::new("c_learning_closed_loop");
    for seed in [1, 2, 3] {
        let run = learn(seed, 100, 2, 4);
        c.check(
            &format!("seed {seed}: final expected reward > 0.85"),
            run.reward.last().copied().unwrap_or(0.0) > 0.85,
            || format!("{:?}", run.reward.last()),
        );
        c.eq(&format!("seed {seed}: updates"), run.updates, 400);
        c.eq(
            &format!("seed {seed}: policy version"),
            run.policy.version,
            400,
        );
        c.check(
            &format!("seed {seed}: backpressure happened"),
            run.pressure > 0,
            || "0".into(),
        );
        c.eq(
            &format!("seed {seed}: no stale drops at max_lag 4"),
            run.stale,
            0,
        );
        c.close(
            &format!("seed {seed}: reward is the exact expectation"),
            *run.reward.last().unwrap_or(&0.0),
            expected_reward(&run.policy),
            1e-12,
        );
    }
    // One wave, capacity 2, max_lag 0 (exact, any seed): groups a,b are admitted; c hits
    // backpressure, one complete group is trained (version 1) and c is retried; c and d
    // are now 1 version old -> dropped (16); the drain evicts the other admitted group
    // (8). updates 1, stale 24, backpressure 1. A loop that does not tell the queue about
    // the new version admits c and d and trains twice.
    for seed in [1, 2] {
        let one = learn(seed, 1, 2, 0);
        c.eq(
            &format!("seed {seed}: one strict wave (updates, stale, backpressure)"),
            (one.updates, one.stale, one.pressure),
            (1, 24, 1),
        );
    }
    let strict = learn(1, 5, 2, 0);
    c.check("max_lag 0 drops stale groups", strict.stale > 0, || {
        "no drops".into()
    });
    c.check(
        "max_lag 0 trains on fewer groups",
        strict.updates < 20,
        || format!("{}", strict.updates),
    );
    c.done();
}

// ------------------------------------------------------------------- R
/// Test-side nearest-rank quantile (independent of R3).
fn nearest_rank(values: &[u64], q: f64) -> u64 {
    let mut v = values.to_vec();
    v.sort();
    let mut rank = 1;
    while (rank as f64) < q * v.len() as f64 {
        rank += 1;
    }
    v[rank - 1]
}
#[test]
fn r_predictor() {
    let mut c = Checks::new("r_predictor");
    let mut p = QuantilePredictor::new(0.5, 3, 256);
    c.eq("default when nothing observed", p.predict(7), 256);
    p.observe(1, &[10, 30, 20]);
    c.eq("per-prompt median (rank ceil(1.5)=2)", p.predict(1), 20);
    c.eq("unseen prompt uses global history", p.predict(2), 20);
    p.observe(1, &[40]);
    // per-prompt window 3 -> [30,20,40] -> 30; global [10,30,20,40] -> rank 2 -> 20
    c.eq("per-prompt window evicts oldest", p.predict(1), 30);
    c.eq("global keeps 4 <= 48", p.predict(2), 20);
    let snapshot = (p.per_prompt.clone(), p.global.clone());
    let _ = p.predict(1);
    c.eq(
        "predict is pure",
        (p.per_prompt.clone(), p.global.clone()),
        snapshot,
    );
    let mut one = QuantilePredictor::new(1.0, 1, 0);
    one.observe(0, &(1..=20).collect::<Vec<u64>>());
    c.eq("global bounded by 16*window", one.global.len(), 16);
    c.eq("q=1 is max of window 1", one.predict(0), 20);
    c.eq("global q=1 sees newest 16", one.predict(3), 20);
    let mut zero = QuantilePredictor::new(0.0, 4, 0);
    zero.observe(0, &[9, 4, 7]);
    c.eq("q=0 is min", zero.predict(0), 4);
    // Randomized: compare with a test-side history and nearest rank.
    let mut rng = Lcg(5);
    let mut bad = 0;
    for _case in 0..40 {
        let q = [0.25, 0.5, 0.9, 1.0][(rng.next() % 4) as usize];
        let window = 1 + (rng.next() % 5) as usize;
        let mut p = QuantilePredictor::new(q, window, 99);
        let mut mine: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
        let mut global: Vec<u64> = vec![];
        for _ in 0..12 {
            let prompt = (rng.next() % 3) as usize;
            let lengths: Vec<u64> = (0..1 + rng.next() % 4).map(|_| rng.next() % 500).collect();
            p.observe(prompt, &lengths);
            let row = mine.entry(prompt).or_default();
            for &l in &lengths {
                row.push(l);
                global.push(l);
            }
            let keep = row.len().saturating_sub(window);
            row.drain(..keep);
            let keep = global.len().saturating_sub(16 * window);
            global.drain(..keep);
            for probe in 0..4 {
                let want = match mine.get(&probe) {
                    Some(r) if !r.is_empty() => nearest_rank(r, q),
                    _ if !global.is_empty() => nearest_rank(&global, q),
                    _ => 99,
                };
                bad += (p.predict(probe) != want) as usize;
            }
        }
    }
    c.eq("random cases vs test-side quantile", bad, 0);
    c.done();
}

/// Hand-computed synthetic trace (rows deliberately out of order). Durations in ms.
/// round 0: g0 p0 (100,1000,1) (120,1200,0) | g1 p1 (80,800,1) (90,900,1)
///          g2 p2 (300,3000,0) (400,4000,1)
/// round 1: g3 p0 (110,1100,1) (130,1300,1) | g4 p1 (100,1000,0) (70,700,1)
///          g5 p2 (350,3500,1) (380,3800,0)
const SYNTH: &str = r#"{"type":"header","label":"synthetic"}
{"prompt_id":2,"group":5,"round":1,"sample_idx":1,"policy_version":1,"response_tokens":380,"gen_ms":3800,"reward":0}
{"prompt_id":0,"group":0,"round":0,"sample_idx":1,"policy_version":0,"response_tokens":120,"gen_ms":1200,"reward":0}
{"type":"group","group":0}
{"prompt_id":0,"group":0,"round":0,"sample_idx":0,"policy_version":0,"response_tokens":100,"gen_ms":1000,"reward":1}
{"prompt_id":1,"group":1,"round":0,"sample_idx":0,"policy_version":0,"response_tokens":80,"gen_ms":800,"reward":1}
{"prompt_id":1,"group":1,"round":0,"sample_idx":1,"policy_version":0,"response_tokens":90,"gen_ms":900,"reward":1}
{"prompt_id":2,"group":2,"round":0,"sample_idx":0,"policy_version":0,"response_tokens":300,"gen_ms":3000,"reward":0}
{"prompt_id":2,"group":2,"round":0,"sample_idx":1,"policy_version":0,"response_tokens":400,"gen_ms":4000,"reward":1}
{"prompt_id":0,"group":3,"round":1,"sample_idx":0,"policy_version":1,"response_tokens":110,"gen_ms":1100,"reward":1}
{"prompt_id":0,"group":3,"round":1,"sample_idx":1,"policy_version":1,"response_tokens":130,"gen_ms":1300,"reward":1}
{"prompt_id":1,"group":4,"round":1,"sample_idx":0,"policy_version":1,"response_tokens":100,"gen_ms":1000,"reward":0}
{"prompt_id":1,"group":4,"round":1,"sample_idx":1,"policy_version":1,"response_tokens":70,"gen_ms":700,"reward":1}
{"prompt_id":2,"group":5,"round":1,"sample_idx":0,"policy_version":1,"response_tokens":350,"gen_ms":3500,"reward":1}
"#;
#[test]
fn r_replay_synthetic() {
    let mut c = Checks::new("r_replay_synthetic");
    let trace = RolloutTrace::parse(SYNTH).expect("synthetic trace parses");
    let run = |s: Strategy| replay(&trace, 2, s, &mut QuantilePredictor::new(1.0, 8, 256));
    // Hand makespans (2 workers). Round 0 durations g0 1200, g1 900, g2 4000; round 1
    // g3 1300, g4 1000, g5 3800. Predictions (q=1): round 0 all 256 (cold), round 1
    // p0 120, p1 90, p2 400 (max of round 0 only).
    //  fifo          r0: w0 1200, w1 900+4000=4900          r1: w0 1300, w1 1000+3800=4800
    //  lpt_oracle    r0: 4000 | 1200+900                     r1: 3800 | 1300+1000
    //  lpt_predicted r0: equal estimates -> w0 g0,g2 = 5200  r1: g5 | g3,g4 -> 3800
    //  lpt_pull      r0: order g0,g1,g2 = fifo 4900          r1: order g5,g3,g4 -> 3800
    //  work_stealing r0: w1 steals g2 at t=900 -> 4900       r1: 3800, no steal
    let expect = [
        (Strategy::Fifo, vec![4900, 4800], 0),
        (Strategy::LptOracle, vec![4000, 3800], 0),
        (Strategy::LptPredicted, vec![5200, 3800], 0),
        (Strategy::LptPull, vec![4900, 3800], 0),
        (Strategy::WorkStealing, vec![4900, 3800], 1),
    ];
    for (s, rounds, steals) in expect {
        let r = run(s);
        c.eq(
            &format!("{}: round makespans", s.name()),
            r.round_makespan_ms.clone(),
            rounds.clone(),
        );
        c.eq(
            &format!("{}: total makespan", s.name()),
            r.makespan_ms,
            rounds.iter().sum::<u64>(),
        );
        c.eq(&format!("{}: steals", s.name()), r.steals, steals);
        c.eq(
            &format!("{}: keeps every sample", s.name()),
            r.kept_samples,
            12,
        );
    }
    let fifo = run(Strategy::Fifo);
    c.eq(
        "predictions use only earlier rounds",
        fifo.predictions.clone(),
        vec![
            (0, 256, 120),
            (1, 256, 90),
            (2, 256, 400),
            (3, 120, 130),
            (4, 90, 100),
            (5, 400, 380),
        ],
    );
    c.eq(
        "busy = sum of durations",
        fifo.busy_ms,
        1200 + 900 + 4000 + 1300 + 1000 + 3800,
    );
    c.close(
        "all: mean length 2230/12",
        fifo.mean_len_all,
        2230.0 / 12.0,
        1e-9,
    );
    c.close(
        "all: mean reward 8/12",
        fifo.mean_reward_all,
        8.0 / 12.0,
        1e-12,
    );
    c.eq(
        "all: zero-advantage groups (g1, g3)",
        fifo.zero_adv_groups_all,
        2,
    );
    c.close(
        "fifo: no shift",
        fifo.mean_len_kept,
        fifo.mean_len_all,
        1e-12,
    );
    // keep_first_1: kept g0 s0, g1 s0, g2 s0, g3 s0, g4 s1, g5 s0.
    //   r0 durations 1000, 800, 3000 -> 1000 | 800+3000 = 3800
    //   r1 durations 1100, 700, 3500 -> 1100 | 700+3500 = 4200
    let k1 = run(Strategy::KeepFirstN(1));
    c.eq(
        "keep_first_1: round makespans",
        k1.round_makespan_ms.clone(),
        vec![3800, 4200],
    );
    c.eq("keep_first_1: kept", k1.kept_samples, 6);
    c.close(
        "keep_first_1: mean kept length 1010/6",
        k1.mean_len_kept,
        1010.0 / 6.0,
        1e-9,
    );
    c.close(
        "keep_first_1: mean kept reward 5/6",
        k1.mean_reward_kept,
        5.0 / 6.0,
        1e-12,
    );
    c.close(
        "keep_first_1: mean length of ALL samples still 2230/12",
        k1.mean_len_all,
        2230.0 / 12.0,
        1e-9,
    );
    c.eq(
        "keep_first_1: every kept group has zero advantage",
        k1.zero_adv_groups_kept,
        6,
    );
    // Censoring: only kept (finished) samples are observed -> round-1 predictions
    // p0 100, p1 80, p2 300 instead of 120, 90, 400.
    c.eq(
        "keep_first_1: predictor sees only finished samples",
        k1.predictions.iter().map(|x| x.1).collect::<Vec<_>>(),
        vec![256, 256, 256, 100, 80, 300],
    );
    // first_idx_1 (unbiased subset): g4 keeps s0 (1000 ms) -> r1 1100 | 1000+3500 = 4500
    let f1 = run(Strategy::FirstIdx(1));
    c.eq(
        "first_idx_1: round makespans",
        f1.round_makespan_ms.clone(),
        vec![3800, 4500],
    );
    c.close(
        "first_idx_1: mean kept length 1040/6",
        f1.mean_len_kept,
        1040.0 / 6.0,
        1e-9,
    );
    c.close(
        "first_idx_1: mean kept reward 4/6",
        f1.mean_reward_kept,
        4.0 / 6.0,
        1e-12,
    );
    let k2 = run(Strategy::KeepFirstN(2));
    c.eq(
        "keep_first_2 of 2 = fifo",
        (k2.makespan_ms, k2.kept_samples),
        (9700, 12),
    );
    c.close(
        "throughput = kept / makespan",
        fifo.throughput_samples_per_s(),
        12.0 / 9.7,
        1e-9,
    );
    c.done();
}

/// Test-side greedy pull simulation: jobs in the given order, each to the earliest-idle
/// worker (ties: lower index). Returns the makespan.
fn pull_makespan(durations: &[u64], workers: usize) -> u64 {
    let mut free = vec![0u64; workers];
    for &d in durations {
        let w = (0..workers).min_by_key(|&w| (free[w], w)).unwrap();
        free[w] += d;
    }
    free.into_iter().max().unwrap_or(0)
}
#[test]
fn r_replay_reference() {
    let mut c = Checks::new("r_replay_reference");
    let text = std::fs::read_to_string(reference_trace_path()).expect("reference trace present");
    let trace = RolloutTrace::parse(&text).expect("reference trace parses");
    c.check(
        "label says it is not GRPO training",
        trace
            .header
            .get("label")
            .map_or(false, |l| l.contains("not from GRPO training")),
        || format!("{:?}", trace.header.get("label")),
    );
    c.eq("rounds", trace.rounds(), vec![0, 1]);
    c.eq("groups", trace.groups.len(), 64);
    c.eq("samples", trace.samples().count(), 512);
    c.check(
        "8 samples per group",
        trace.groups.iter().all(|g| g.samples.len() == 8),
        || "uneven".into(),
    );
    let workers = 4;
    // Independent per-round durations (ms) computed here from the raw rows.
    let mut lower = 0u64;
    let mut fifo = 0u64;
    let mut oracle = 0u64;
    let mut k4_len = 0.0;
    let mut k4_rew = 0.0;
    let mut k4_zero = 0;
    let mut k4_fifo = 0u64;
    for round in trace.rounds() {
        let gs: Vec<_> = trace.groups.iter().filter(|g| g.round == round).collect();
        let full: Vec<u64> = gs
            .iter()
            .map(|g| {
                g.samples
                    .iter()
                    .map(|s| s.gen_ms)
                    .fold(0.0, f64::max)
                    .round() as u64
            })
            .collect();
        let sum: u64 = full.iter().sum();
        lower += (*full.iter().max().unwrap()).max((sum + workers as u64 - 1) / workers as u64);
        fifo += pull_makespan(&full, workers);
        let mut sorted = full.clone();
        sorted.sort_by(|a, b| b.cmp(a));
        oracle += pull_makespan(&sorted, workers);
        let mut cut = vec![];
        for g in &gs {
            let mut s: Vec<_> = g.samples.iter().collect();
            s.sort_by(|a, b| {
                a.gen_ms
                    .total_cmp(&b.gen_ms)
                    .then(a.sample_idx.cmp(&b.sample_idx))
            });
            let kept = &s[..4];
            k4_len += kept.iter().map(|x| x.response_tokens as f64).sum::<f64>();
            k4_rew += kept.iter().map(|x| x.reward).sum::<f64>();
            k4_zero += kept.iter().all(|x| x.reward == kept[0].reward) as usize;
            cut.push(kept[3].gen_ms.round() as u64);
        }
        k4_fifo += pull_makespan(&cut, workers);
    }
    let all_len = trace
        .samples()
        .map(|s| s.response_tokens as f64)
        .sum::<f64>()
        / 512.0;
    let all_rew = trace.samples().map(|s| s.reward).sum::<f64>() / 512.0;
    let new = || QuantilePredictor::new(1.0, 8, 256);
    let r = replay(&trace, workers, Strategy::Fifo, &mut new());
    c.eq(
        "fifo makespan = test-side pull simulation",
        r.makespan_ms,
        fifo,
    );
    let r = replay(&trace, workers, Strategy::LptOracle, &mut new());
    c.eq(
        "lpt_oracle makespan = test-side sorted pull",
        r.makespan_ms,
        oracle,
    );
    for s in [
        Strategy::Fifo,
        Strategy::LptPredicted,
        Strategy::LptPull,
        Strategy::WorkStealing,
        Strategy::LptOracle,
        Strategy::KeepFirstN(8),
    ] {
        let r = replay(&trace, workers, s, &mut new());
        c.check(
            &format!("{}: makespan >= lower bound {lower}", s.name()),
            r.makespan_ms >= lower,
            || format!("{}", r.makespan_ms),
        );
        c.eq(&format!("{}: kept all 512", s.name()), r.kept_samples, 512);
        c.close(
            &format!("{}: mean length of all samples", s.name()),
            r.mean_len_all,
            all_len,
            1e-9,
        );
    }
    let k4 = replay(&trace, workers, Strategy::KeepFirstN(4), &mut new());
    c.eq("keep_first_4: kept", k4.kept_samples, 256);
    c.eq(
        "keep_first_4: makespan = test-side pull on 4th finish",
        k4.makespan_ms,
        k4_fifo,
    );
    c.close(
        "keep_first_4: mean kept length (independent)",
        k4.mean_len_kept,
        k4_len / 256.0,
        1e-9,
    );
    c.close(
        "keep_first_4: mean kept reward (independent)",
        k4.mean_reward_kept,
        k4_rew / 256.0,
        1e-12,
    );
    c.close(
        "keep_first_4: mean reward of all samples",
        k4.mean_reward_all,
        all_rew,
        1e-12,
    );
    c.eq(
        "keep_first_4: zero-advantage kept groups (independent)",
        k4.zero_adv_groups_kept,
        k4_zero,
    );
    c.check(
        "keep_first_4: kept responses are shorter on this trace (measured)",
        k4.mean_len_kept < k4.mean_len_all,
        || format!("{} vs {}", k4.mean_len_kept, k4.mean_len_all),
    );
    println!(
        "INFO keep_first_4 mean length {:.1} vs all {:.1}",
        k4.mean_len_kept, k4.mean_len_all
    );
    let f4 = replay(&trace, workers, Strategy::FirstIdx(4), &mut new());
    c.eq("first_idx_4: kept", f4.kept_samples, 256);
    // Leakage: round-1 predictions must equal the max of round-0 lengths of that prompt.
    let p = replay(&trace, workers, Strategy::LptPredicted, &mut new());
    let mut round0: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    for g in trace.groups.iter().filter(|g| g.round == 0) {
        round0
            .entry(g.prompt_id)
            .or_default()
            .extend(g.samples.iter().map(|s| s.response_tokens));
    }
    let mut wrong = 0;
    for (g, &(id, pred, truth)) in trace.groups.iter().zip(&p.predictions) {
        let want = if g.round == 0 {
            256
        } else {
            nearest_rank(&round0[&g.prompt_id], 1.0)
        };
        let true_max = g.samples.iter().map(|s| s.response_tokens).max().unwrap();
        wrong += (id != g.group || pred != want || truth != true_max) as usize;
    }
    c.eq(
        "predictions: cold round 0, round 1 from round 0 only",
        wrong,
        0,
    );
    c.done();
}
