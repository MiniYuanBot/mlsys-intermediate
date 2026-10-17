use slo_speculation::*;
fn main() {
    let part = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let profile = vec![
        ProfilePoint {
            context: 0,
            draft: 0.08,
            verify: vec![1., 1.05, 1.15, 1.35, 1.7],
        },
        ProfilePoint {
            context: 512,
            draft: 0.12,
            verify: vec![1.5, 1.6, 1.8, 2.2, 2.8],
        },
    ];
    if part == "A" || part == "all" {
        println!("kind,stage,context,k,expected_tokens,variance,simulated_round_ms");
        for context in [0, 128, 256, 512] {
            let row = interpolate(&profile, context);
            for k in 0..=4 {
                let (e, v) = committed_moments(&[0.8; 4], k);
                println!(
                    "synthetic,A,{context},{k},{e},{v},{}",
                    row.draft * k as f64 + row.verify[k]
                );
            }
        }
    }
    if part == "B" || part == "all" {
        println!("kind,stage,phase,step,mean_k,conservative_k,lower_p0");
        let mut e = Estimator::new(4, 0.97);
        for (phase, accepted) in [("high", 4), ("low", 0), ("recovery", 3)] {
            for step in 0..40 {
                e.observe(accepted, 4);
                let mean = e.probabilities(0.);
                let low = e.probabilities(1.);
                println!(
                    "synthetic,B,{phase},{step},{},{},{}",
                    choose_k(&mean, &profile[0], 2., 20).unwrap(),
                    choose_k(&low, &profile[0], 2., 20).unwrap(),
                    low[0]
                );
            }
        }
    }
    if part == "C" || part == "all" {
        println!("kind,stage,policy,seed,arrival_gap_sim_ms,requests,tokens,p95_ttft_sim_ms,p95_e2e_sim_ms,goodput_req_per_sim_s,met,infeasible,resets");
        for gap in [0.2, 2., 8.] {
            for seed in 1..=5 {
                for (name, policy) in [
                    ("k0", Controller::Fixed(0)),
                    ("k3", Controller::Fixed(3)),
                    ("mean", Controller::Mean),
                    ("conservative_no_probe", Controller::ConservativeNoProbe),
                ] {
                    let requests = (0..90)
                        .map(|id| Request {
                            id,
                            arrival: id as f64 * gap,
                            output: 8 + id % 9,
                            context: (id % 4) * 128,
                            deadline: id as f64 * gap + 40.,
                            alpha: if (30..60).contains(&id) { 0.2 } else { 0.85 },
                        })
                        .collect();
                    let sim = simulate(requests, &profile, policy, 2., seed);
                    let m = summarize(&sim);
                    println!(
                        "synthetic_queue,C,{name},{seed},{gap},{},{},{},{},{},{},{},{}",
                        m.requests,
                        m.tokens,
                        m.p95_ttft,
                        m.p95_e2e,
                        m.goodput,
                        m.met,
                        sim.infeasible,
                        sim.resets
                    );
                }
            }
        }
    }
}
