use slo_speculation::*;
use std::collections::VecDeque;
fn profile() -> Vec<ProfilePoint> {
    vec![ProfilePoint {
        context: 0,
        draft: 0.1,
        verify: vec![1.0, 1.1, 1.5, 2.4],
    }]
}
fn req(id: usize, arrival: f64, output: usize, deadline: f64, alpha: f64) -> Request {
    Request {
        id,
        arrival,
        output,
        context: 0,
        deadline,
        alpha,
    }
}
#[test]
fn a_distribution_oracle() {
    // Independent enumeration of all 2^k binary vectors, including unobserved suffix.
    let p = [0.2, 0.7, 0.4];
    let mut expected = vec![0.0; 4];
    for mask in 0..8 {
        let mut weight = 1.0;
        let mut prefix = 0;
        for (j, &prob) in p.iter().enumerate() {
            weight *= if mask & (1 << j) != 0 {
                prob
            } else {
                1.0 - prob
            };
        }
        while prefix < 3 && mask & (1 << prefix) != 0 {
            prefix += 1;
        }
        expected[prefix] += weight;
    }
    for (a, b) in committed_distribution(&p, 3).iter().zip(expected) {
        assert!((a - b).abs() < 1e-12);
    }
    assert_eq!(committed_distribution(&[], 0), vec![1.0]);
}
#[test]
fn a_moments_and_quantiles() {
    let (mean, var) = committed_moments(&[0.5, 0.25], 2);
    assert!((mean - 1.625).abs() < 1e-12);
    assert!((var - 0.484375).abs() < 1e-12);
    assert_eq!(percentile(&[], 0.5), None);
    assert_eq!(percentile(&[4., 1., 3., 2.], 0.75), Some(3.));
    assert_eq!(percentile(&[4., 1.], 0.), Some(1.));
    assert_eq!(percentile(&[4., 1.], 1.), Some(4.));
}
#[test]
fn a_profile_interpolation() {
    let p = vec![
        ProfilePoint {
            context: 100,
            draft: 1.,
            verify: vec![2., 4.],
        },
        ProfilePoint {
            context: 300,
            draft: 3.,
            verify: vec![6., 10.],
        },
    ];
    assert_eq!(interpolate(&p, 200).verify, vec![4., 7.]);
    assert_eq!(interpolate(&p, 200).draft, 2.);
    assert_eq!(interpolate(&p, 1).verify, vec![2., 4.]);
    assert_eq!(interpolate(&p, 999).verify, vec![6., 10.]);
}
#[test]
fn b_censored_observation_and_decay() {
    let mut e = Estimator::new(4, 0.5);
    e.observe(1, 4);
    assert_eq!(e.success, vec![1., 0., 0., 0.]);
    assert_eq!(e.failure, vec![0., 1., 0., 0.]);
    e.observe(4, 4);
    assert_eq!(e.success, vec![1.5, 1., 1., 1.]);
    assert_eq!(e.failure, vec![0., 0.5, 0., 0.]);
    e.observe(0, 0);
    assert_eq!(e.failure[1], 0.25);
}
#[test]
fn b_lower_bound_and_ranking() {
    let e = Estimator {
        success: vec![0., 1., 100.],
        failure: vec![0., 0., 0.],
        decay: 1.,
    };
    let p = e.probabilities(1.96);
    assert_eq!(p[0], 0.);
    assert!(p[1] < p[2] && p[2] < 1.);
    let p = e.probabilities(0.);
    assert_eq!(p[0], 0.5);
    assert!((p[1] - 2. / 3.).abs() < 1e-12);
}
#[test]
fn b_choice_exhaustive_and_infeasible() {
    let row = &profile()[0];
    for prob in [0., 0.4, 1.] {
        for remaining in 1..=5 {
            for bound in [0.1, 1.0, 1.2, 2., 10.] {
                let mut oracle = None;
                let mut rate = -1.;
                for k in 0..=3.min(remaining - 1) {
                    let cost = row.draft * k as f64 + row.verify[k];
                    let mut e = 1.;
                    for j in 1..=k {
                        e += f64::powi(prob, j as i32);
                    }
                    if cost <= bound && e / cost > rate + 1e-12 {
                        oracle = Some(k);
                        rate = e / cost;
                    }
                }
                assert_eq!(choose_k(&[prob; 3], row, bound, remaining), oracle);
            }
        }
    }
}
#[test]
fn b_drift_drop_only() {
    let mut d = Drift::default();
    for _ in 0..20 {
        assert!(!d.update(0.9, 0.05, 2.));
    }
    let events = (0..20).filter(|_| d.update(0.1, 0.05, 2.)).count();
    assert!(events >= 1);
    let mut rising = Drift::default();
    for p in [0.1, 0.2, 0.4, 0.8, 0.9] {
        assert!(!rising.update(p, 0.05, 2.));
    }
}
#[test]
fn c_commit_cap_and_timestamps() {
    let mut s = State::new(req(0, 1., 3, 10., 1.));
    s.commit(0, 2.);
    s.commit(9, 4.);
    assert_eq!(s.committed_at, vec![2., 4., 4.]);
    assert_eq!(s.first, Some(2.));
    assert_eq!(s.finish, Some(4.));
}
#[test]
fn c_arrivals_once() {
    let s = vec![
        State::new(req(0, 0., 1, 5., 1.)),
        State::new(req(1, 0., 1, 5., 1.)),
        State::new(req(2, 3., 1, 5., 1.)),
    ];
    let mut q = VecDeque::new();
    let mut cursor = 0;
    enqueue(&s, &mut cursor, 0., &mut q);
    enqueue(&s, &mut cursor, 0., &mut q);
    assert_eq!(q, VecDeque::from(vec![0, 1]));
    enqueue(&s, &mut cursor, 3., &mut q);
    assert_eq!(q.len(), 3);
}
#[test]
fn c_hand_timeline() {
    let s = simulate(
        vec![req(0, 0., 2, 10., 1.), req(1, 0.5, 1, 10., 1.)],
        &profile(),
        Controller::Fixed(0),
        2.,
        1,
    );
    assert_eq!(s.states[0].committed_at, vec![1., 3.]);
    assert_eq!(s.states[1].committed_at, vec![2.]);
    assert_eq!(s.busy, 3.);
}
#[test]
fn c_metrics_independent() {
    let sim = Simulation {
        states: vec![
            State {
                request: req(0, 1., 2, 5., 1.),
                emitted: 2,
                first: Some(2.),
                finish: Some(4.),
                rounds: 2,
                committed_at: vec![2., 4.],
            },
            State {
                request: req(1, 2., 1, 3., 1.),
                emitted: 1,
                first: Some(5.),
                finish: Some(5.),
                rounds: 1,
                committed_at: vec![5.],
            },
        ],
        busy: 3.,
        infeasible: 0,
        resets: 0,
    };
    let m = summarize(&sim);
    assert_eq!(m.p95_ttft, 3.);
    assert_eq!(m.p95_e2e, 3.);
    assert_eq!(m.met, 1);
    assert_eq!(m.goodput, 250.);
    assert_eq!(m.utilization, 0.75);
}
#[test]
fn c_conservation_empty_and_overload() {
    // Explicit negative baseline: a zero-evidence lower bound never explores.
    // For this increasing-cost profile it is exactly ordinary k=0 decoding.
    let fresh = vec![req(0, 0., 5, 100., 1.), req(1, 0., 4, 100., 1.)];
    let plain = simulate(fresh.clone(), &profile(), Controller::Fixed(0), 2., 3);
    let no_probe = simulate(fresh, &profile(), Controller::ConservativeNoProbe, 2., 3);
    for (a, b) in plain.states.iter().zip(&no_probe.states) {
        assert_eq!(a.committed_at, b.committed_at);
    }

    assert_eq!(
        summarize(&simulate(vec![], &profile(), Controller::Mean, 1., 1)).requests,
        0
    );
    for policy in [
        Controller::Fixed(0),
        Controller::Fixed(3),
        Controller::Mean,
        Controller::ConservativeNoProbe,
    ] {
        let requests = (0..40)
            .map(|i| req(i, (i / 5) as f64, 1 + i % 7, (i / 5) as f64 + 2., 0.4))
            .collect();
        let sim = simulate(requests, &profile(), policy, 1.4, 2);
        let m = summarize(&sim);
        assert_eq!(m.requests, 40);
        assert_eq!(m.tokens, (0..40).map(|i| 1 + i % 7).sum());
        assert!(m.utilization <= 1. + 1e-12);
        assert!(m.met < 40);
    }
}
#[test]
fn a_calibration_grouping_and_missing() {
    let samples = vec![
        TimingSample {
            context: 32,
            k: 1,
            elapsed: 4.,
        },
        TimingSample {
            context: 32,
            k: 0,
            elapsed: 3.,
        },
        TimingSample {
            context: 32,
            k: 0,
            elapsed: 1.,
        },
        TimingSample {
            context: 32,
            k: 0,
            elapsed: 2.,
        },
    ];
    let table = calibrate(&samples, 1, 0.2).unwrap();
    assert_eq!(table[0].verify, vec![2., 4.]);
    assert!(calibrate(&samples, 2, 0.2).is_err());
    assert!(calibrate(&[], 1, 0.2).is_err());
}
#[test]
fn b_joint_budget_and_nonoptimal_counterexample() {
    let rows = vec![
        ProfilePoint {
            context: 0,
            draft: 0.,
            verify: vec![1., 2.]
        };
        2
    ];
    assert_eq!(
        allocate_round(&[vec![0.9], vec![0.2]], &rows, 3.),
        Some(vec![1, 0])
    );
    assert!(allocate_round(&[vec![0.9], vec![0.2]], &rows, 1.).is_none());
    assert_eq!(allocate_round(&[], &[], 0.), Some(vec![]));
    // Greedy takes 0.6 gain for 1.0 cost; optimum takes 0.9 for 1.6.
    let rows = vec![
        ProfilePoint {
            context: 0,
            draft: 0.,
            verify: vec![1., 2.],
        },
        ProfilePoint {
            context: 0,
            draft: 0.,
            verify: vec![1., 2.6],
        },
    ];
    let k = allocate_round(&[vec![0.6], vec![0.9]], &rows, 3.6).unwrap();
    assert_eq!(k, vec![1, 0]);
    assert!(committed_moments(&[0.6], 1).0 + 1. < 1. + committed_moments(&[0.9], 1).0);
}
