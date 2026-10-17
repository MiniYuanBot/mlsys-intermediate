use rollout_scheduling::*;
fn main() {
    let part = std::env::args().nth(1).unwrap_or("all".into());
    if part == "A" || part == "all" {
        println!("kind,part,window,mae,p50,p95");
        for window in [4, 16, 64] {
            let mut m = LengthModel {
                window,
                ..Default::default()
            };
            let mut error = 0.;
            let mut lengths = vec![];
            for id in 0..200 {
                let class = id % 4;
                let length = if id < 100 {
                    10 + class as u64 * 10
                } else {
                    40 + class as u64 * 15
                };
                error += (m.predict(class) as f64 - length as f64).abs();
                m.observe(class, length);
                lengths.push(length);
            }
            println!(
                "synthetic_history,A,{window},{},{},{}",
                error / 200.,
                histogram(&lengths, &[16, 32, 64], 0.5).1,
                histogram(&lengths, &[16, 32, 64], 0.95).1
            );
        }
    }
    if part == "B" || part == "all" {
        println!(
            "kind,part,workers,quantum,steal,makespan_token_units,utilization,segments,steals"
        );
        let jobs: Vec<_> = (0..64)
            .map(|id| Job {
                id,
                group: id / 8,
                version: 1,
                estimate: 32,
                actual: if id % 13 == 0 {
                    300
                } else {
                    10 + (id * 17 % 50) as u64
                },
            })
            .collect();
        for workers in [2, 4, 8] {
            for quantum in [16, 64] {
                for steal in [false, true] {
                    let r = schedule(&jobs, workers, quantum, steal);
                    println!(
                        "synthetic_events,B,{workers},{quantum},{steal},{},{},{},{}",
                        r.finish,
                        r.busy as f64 / (workers as f64 * r.finish as f64),
                        r.segments.len(),
                        r.steals
                    );
                }
            }
        }
    }
    if part == "C" || part == "all" {
        println!("kind,part,seed,capacity,max_lag,wave,expected_reward,updates,backpressure,stale_samples,rollout_token_units,rollout_makespan_units");
        for seed in [1, 2, 3] {
            for capacity in [1, 2, 4] {
                for lag in [0, 1, 4] {
                    let run = learn(seed, 100, capacity, lag);
                    for wave in [0, 9, 49, 99] {
                        println!(
                            "actual_bandit_learning,C,{seed},{capacity},{lag},{wave},{},{},{},{},{},{}",
                            run.reward[wave], run.snapshots[wave][0], run.snapshots[wave][1], run.snapshots[wave][2], run.snapshots[wave][3], run.snapshots[wave][4]
                        );
                    }
                }
            }
        }
    }
}
