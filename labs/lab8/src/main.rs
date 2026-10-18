//! Lab 8 benchmarks. A/B/C print synthetic or bandit tables; R replays real traces.
//! Usage: bench <A|B|C|R|all> [extra trace.jsonl ...]
use rollout_scheduling::*;

fn part_a() {
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

fn part_b() {
    println!("kind,part,workers,quantum,steal,makespan_token_units,utilization,segments,steals");
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
                    "synthetic_events,B,{workers},{quantum},{steal},{},{:.4},{},{}",
                    r.finish,
                    r.busy as f64 / (workers as f64 * r.finish as f64),
                    r.segments.len(),
                    r.steals
                );
            }
        }
    }
    // B4: keep-first-n on i.i.d. synthetic groups whose length decides reward.
    println!("kind,part,m,n,mean_len_all,mean_len_kept,mean_reward_all,mean_reward_kept");
    let mut state = 7u64;
    for (m, n) in [(8, 8), (8, 6), (8, 4), (16, 8)] {
        let (mut la, mut lk, mut ra, mut rk, mut na, mut nk) = (0., 0., 0., 0., 0., 0.);
        for _ in 0..2000 {
            let group: Vec<Finished> = (0..m)
                .map(|i| {
                    let tokens = 50 + (uniform(&mut state) * 450.0) as u64;
                    // longer answers are more often correct in this toy (reasoning helps)
                    let reward = f64::from(uniform(&mut state) < tokens as f64 / 600.0);
                    Finished {
                        sample_idx: i,
                        finish: tokens as f64,
                        tokens,
                        reward,
                    }
                })
                .collect();
            for f in &group {
                la += f.tokens as f64;
                ra += f.reward;
                na += 1.;
            }
            for i in keep_first_n(&group, n) {
                lk += group[i].tokens as f64;
                rk += group[i].reward;
                nk += 1.;
            }
        }
        println!(
            "synthetic_keep_first_n,B,{m},{n},{:.2},{:.2},{:.4},{:.4}",
            la / na,
            lk / nk,
            ra / na,
            rk / nk
        );
    }
}

fn part_c() {
    println!("kind,part,seed,capacity,max_lag,wave,expected_reward,updates,backpressure,stale_samples,rollout_token_units,rollout_makespan_units");
    for seed in [1, 2, 3] {
        for capacity in [1, 2, 4] {
            for lag in [0, 1, 4] {
                let run = learn(seed, 100, capacity, lag);
                for wave in [0, 9, 49, 99] {
                    let s = run.snapshots[wave];
                    println!(
                        "actual_bandit_learning,C,{seed},{capacity},{lag},{wave},{:.4},{},{},{},{},{}",
                        run.reward[wave], s[0], s[1], s[2], s[3], s[4]
                    );
                }
            }
        }
    }
}

fn strategies() -> Vec<(Strategy, Option<f64>)> {
    let mut v = vec![(Strategy::Fifo, None), (Strategy::LptOracle, None)];
    for q in [0.5, 0.9, 1.0] {
        v.push((Strategy::LptPredicted, Some(q)));
        v.push((Strategy::LptPull, Some(q)));
        v.push((Strategy::WorkStealing, Some(q)));
    }
    for n in [8, 6, 4] {
        v.push((Strategy::KeepFirstN(n), None));
    }
    for n in [6, 4] {
        v.push((Strategy::FirstIdx(n), None));
    }
    v
}

fn part_r(extra: &[String]) {
    let mut paths = vec![reference_trace_path().display().to_string()];
    paths.extend(extra.iter().cloned());
    println!("kind,part,trace,workers,strategy,q,makespan_s,throughput_samples_per_s,kept_tokens_per_s,kept,groups,steals,mean_len_kept,mean_len_all,len_shift_pct,mean_reward_kept,mean_reward_all,zero_adv_kept,zero_adv_all,pred_mae_after_round0,source");
    for path in paths {
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("cannot read {path}: {e}");
                std::process::exit(1);
            }
        };
        let trace = RolloutTrace::parse(&text).unwrap_or_else(|e| {
            eprintln!("{path}: {e}");
            std::process::exit(1)
        });
        let name = std::path::Path::new(&path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        eprintln!(
            "# {name}: {} | {} | driver {} | {} | {} groups",
            trace
                .header
                .get("label")
                .map(String::as_str)
                .unwrap_or("no label"),
            trace
                .header
                .get("device")
                .map(String::as_str)
                .unwrap_or("?"),
            trace
                .header
                .get("driver")
                .map(String::as_str)
                .unwrap_or("?"),
            trace.header.get("date").map(String::as_str).unwrap_or("?"),
            trace.groups.len()
        );
        let first_round = trace.rounds().first().copied().unwrap_or(0);
        for workers in [2, 4, 8] {
            for (strategy, q) in strategies() {
                let mut predictor = QuantilePredictor::new(q.unwrap_or(1.0), 8, 256);
                let r = replay(&trace, workers, strategy, &mut predictor);
                let late: Vec<_> = trace
                    .groups
                    .iter()
                    .zip(&r.predictions)
                    .filter(|(g, _)| g.round != first_round)
                    .map(|(_, &(_, p, t))| (p as f64 - t as f64).abs())
                    .collect();
                let mae = if late.is_empty() {
                    f64::NAN
                } else {
                    late.iter().sum::<f64>() / late.len() as f64
                };
                println!(
                    "real_trace_replay,R,{name},{workers},{},{},{:.3},{:.3},{:.1},{},{},{},{:.2},{:.2},{:.2},{:.4},{:.4},{},{},{:.2},modeled from measured gen_ms",
                    strategy.name(),
                    q.map(|x| x.to_string()).unwrap_or("-".into()),
                    r.makespan_ms as f64 / 1000.0,
                    r.throughput_samples_per_s(),
                    r.kept_tokens as f64 / (r.makespan_ms as f64 / 1000.0),
                    r.kept_samples,
                    r.groups,
                    r.steals,
                    r.mean_len_kept,
                    r.mean_len_all,
                    100.0 * (r.mean_len_kept / r.mean_len_all - 1.0),
                    r.mean_reward_kept,
                    r.mean_reward_all,
                    r.zero_adv_groups_kept,
                    r.zero_adv_groups_all,
                    mae
                );
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let part = args.get(1).cloned().unwrap_or("all".into());
    let all = part == "all";
    if part == "A" || all {
        part_a();
    }
    if part == "B" || all {
        part_b();
    }
    if part == "C" || all {
        part_c();
    }
    if part == "R" || all {
        part_r(&args[2.min(args.len())..]);
    }
}
