use rollout_scheduling::*;
fn samples(group: usize, version: u64) -> Vec<Sample> {
    Policy {
        logits: vec![0.; 4],
        version,
    }
    .sample(group, 1, &[0.1, 0.9, 0.2, 0.8])
}
#[test]
fn a_length_history() {
    let mut m = LengthModel {
        window: 3,
        ..Default::default()
    };
    assert_eq!(m.predict(9), 32);
    for length in [1, 2, 3, 100] {
        m.observe(0, length);
    }
    assert_eq!(m.predict(0), 3);
    assert_eq!(m.predict(1), 32);
    assert_eq!(m.history[&0].len(), 3);
}
#[test]
fn a_assign() {
    let jobs: Vec<_> = [3, 3, 2, 2, 2]
        .iter()
        .enumerate()
        .map(|(id, &estimated_tokens)| PredictedJob {
            id,
            estimated_tokens,
        })
        .collect();
    let q = assign(&jobs, 2, true);
    let loads: Vec<u64> = q
        .iter()
        .map(|ids| ids.iter().map(|&id| jobs[id].estimated_tokens).sum())
        .collect();
    assert_eq!(loads, vec![7, 5]);
    let mut all: Vec<_> = q.into_iter().flatten().collect();
    all.sort_unstable();
    assert_eq!(all, vec![0, 1, 2, 3, 4]);
    assert_eq!(assign(&[], 2, true), vec![vec![], vec![]]);
}
#[test]
fn a_advantage() {
    assert_eq!(advantages(&[1., 1.]), vec![0., 0.]);
    assert_eq!(advantages(&[0., 1.]), vec![-1., 1.]);
    assert_eq!(advantages(&[10., 12.]), vec![-1., 1.]);
    assert!(advantages(&[]).is_empty());
}
#[test]
fn a_histogram() {
    assert_eq!(
        histogram(&[1, 4, 5, 10, 11], &[4, 10], 0.5),
        (vec![2, 2, 1], 5)
    );
    assert_eq!(histogram(&[], &[4], 0.9), (vec![0, 0], 0));
}
#[test]
fn a_sample() {
    let p = Policy {
        logits: vec![0.; 4],
        version: 3,
    };
    let s = p.sample(2, 1, &[0.1, 0.9]);
    assert_eq!(s.iter().map(|s| s.action).collect::<Vec<_>>(), vec![1, 0]);
    assert_eq!(s[0].id, 4);
    assert_eq!(s[1].version, 3);
    assert_eq!(s[0].behavior_p, 0.5);
    assert_eq!(s[0].reward, 1.);
}
fn segment(start: usize, terminal: bool) -> Segment {
    Segment {
        id: 0,
        group: 0,
        version: 7,
        start,
        tokens: vec![10, 11],
        terminal,
    }
}
#[test]
fn b_reassemble() {
    assert_eq!(
        reassemble(&[segment(2, true), segment(0, false)])
            .unwrap()
            .tokens,
        vec![10, 11, 10, 11]
    );
    for bad in [
        vec![],
        vec![segment(0, false)],
        vec![segment(0, true), segment(2, true)],
        vec![segment(0, false), segment(1, true)],
        vec![segment(0, false), segment(3, true)],
    ] {
        assert!(reassemble(&bad).is_err());
    }
    let mut other = segment(2, true);
    other.version = 8;
    assert!(reassemble(&[segment(0, false), other]).is_err());
}
#[test]
fn b_batch_gate() {
    let completed: Vec<_> = (0..4)
        .map(|id| Completed {
            id,
            group: id / 2,
            version: 2,
            tokens: vec![1],
        })
        .collect();
    assert!(training_ready(&completed, &[0, 1, 2, 3], 2, 2).is_ok());
    assert!(training_ready(&completed[..2], &[0, 1, 2, 3], 2, 2).is_err());
    assert!(training_ready(&completed, &[0, 1, 2, 3], 2, 3).is_err());
    assert!(training_ready(&[completed[0].clone(), completed[0].clone()], &[0, 1], 2, 2).is_err());
}
#[test]
fn b_scheduler() {
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
        assert_eq!(r.busy, 43);
        assert!(r.finish >= 15 && r.finish <= 43);
        let mut completed = vec![];
        for job in &jobs {
            let segments: Vec<_> = r
                .segments
                .iter()
                .filter(|s| s.id == job.id)
                .cloned()
                .collect();
            let c = reassemble(&segments).unwrap();
            assert_eq!(c.tokens.len(), job.actual as usize);
            assert_eq!(
                c.tokens,
                (0..job.actual)
                    .map(|v| job.id as u32 * 10000 + v as u32)
                    .collect::<Vec<_>>()
            );
            completed.push(c);
        }
        assert!(training_ready(&completed, &[0, 1, 2, 3, 4, 5], 2, 4).is_ok());
    }
    assert_eq!(schedule(&jobs, 1, 3, true).finish, 43);
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
    assert_eq!(schedule(&hand, 2, 3, false).finish, 7);
    let stolen = schedule(&hand, 2, 3, true);
    assert_eq!(stolen.finish, 6);
    assert_eq!(stolen.steals, 1);
}
#[test]
fn c_queue_admission() {
    let mut q = GroupQueue::new(1, 4, 1);
    let s = samples(0, 0);
    assert!(q.push(s[0].clone()));
    assert!(!q.push(s[0].clone()));
    assert!(!q.push(samples(1, 0)[0].clone()));
    assert_eq!(q.backpressure, 1);
    assert!(q.pop().is_none());
    for sample in s.into_iter().skip(1) {
        assert!(q.push(sample));
    }
    assert_eq!(q.pop().unwrap().len(), 4);
    assert!(!q.push(samples(0, 0)[0].clone()));
    assert!(q.push(samples(1, 0)[0].clone()));
    let mut q = GroupQueue::new(2, 4, 1);
    q.version = 1;
    assert!(q.push(samples(0, 0)[0].clone()));
    let mut future = samples(0, 2)[1].clone();
    assert!(!q.push(future.clone()));
    future.version = 1;
    assert!(!q.push(future)); // mixed version inside already reserved group
    let mut mismatched = samples(1, 1)[0].clone();
    mismatched.group = 0;
    assert!(!q.push(mismatched));
    assert_eq!(q.pending.len(), 1);
    assert_eq!(q.pending[&0].len(), 1);
    assert_eq!(q.pending[&0][&0].version, 0);
}
#[test]
fn c_queue_drain() {
    let mut q = GroupQueue::new(3, 4, 1);
    for s in samples(0, 0) {
        assert!(q.push(s));
    }
    q.version = 1;
    for s in samples(1, 1) {
        assert!(q.push(s));
    }
    q.version = 2;
    let got = q.pop().unwrap();
    assert_eq!(got[0].group, 1);
    assert_eq!(q.stale, 4);
    assert!(!q.push(samples(2, 0)[0].clone()));
    assert!(q.pop().is_none());
}
#[test]
fn c_gradient() {
    let mut p = Policy {
        logits: vec![0.; 4],
        version: 0,
    };
    let s = p.sample(0, 1, &[0.1, 0.9]);
    let h = 1e-5;
    let objective = |theta: f64| {
        let prob = 1. / (1. + (-theta).exp());
        prob - (1. - prob)
    };
    let fd = (objective(h) - objective(-h)) / (2. * h);
    p.update(&s, 0.2, 0.2);
    assert!((p.logits[1] / 0.2 - fd).abs() < 1e-8);
    let before = p.logits[1];
    let mut same = s.clone();
    for x in &mut same {
        x.reward = 1.;
    }
    p.update(&same, 0.2, 0.2);
    assert_eq!(p.logits[1], before);
    let mut clipped = Policy {
        logits: vec![0., 3., 0., 0.],
        version: 0,
    };
    clipped.update(&s, 0.2, 0.2);
    assert_eq!(clipped.logits[1], 3.);
}
#[test]
fn c_learning_closed_loop() {
    for seed in [1, 2, 3] {
        let run = learn(seed, 100, 2, 4);
        assert!(run.reward.last().unwrap() > &0.85);
        assert_eq!(run.updates, 400);
        assert_eq!(run.policy.version, 400);
        assert!(run.pressure > 0);
        assert_eq!(run.stale, 0);
    }
    let strict = learn(1, 5, 2, 0);
    assert!(strict.stale > 0);
    assert!(strict.updates < 20);
}
