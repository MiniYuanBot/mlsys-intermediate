//! Benchmarks: `bench A|B|C|R|all [trace.jsonl ...]`. Every table starts with a header
//! line; the `kind` column says `modeled` (formula or simulation) or `measured`.
use slo_speculation::*;
use std::collections::BTreeMap;

/// Modeled cost tables (ms). "optimistic" is the old lecture table (c = 0.08);
/// "rtx4060_graph" has the order of the stage-R measurements (CUDA graphs, bf16,
/// Qwen3-0.6B draft / Qwen3-1.7B target): c = T_d / T_v(0) near 0.5.
fn profiles() -> Vec<(&'static str, Vec<ProfilePoint>)> {
    vec![
        (
            "optimistic",
            vec![
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
            ],
        ),
        (
            "rtx4060_graph",
            vec![
                ProfilePoint {
                    context: 128,
                    draft: 9.4,
                    verify: vec![18.7, 19.8, 19.9, 20.0, 20.0],
                },
                ProfilePoint {
                    context: 512,
                    draft: 9.6,
                    verify: vec![19.0, 20.1, 20.2, 20.3, 20.4],
                },
            ],
        ),
    ]
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn part_a() {
    println!(
        "kind,stage,profile,context,k,alpha,expected_tokens,variance,round_ms,tokens_per_ms,best_k"
    );
    for (name, profile) in profiles() {
        for context in [profile[0].context, profile[1].context] {
            let row = interpolate(&profile, context);
            for alpha in [0.6, 0.8, 0.9] {
                let rate = |k: usize| committed_moments(&[alpha; 4], k).0 / row.round_cost(k);
                let best = (0..=4)
                    .max_by(|&a, &b| rate(a).total_cmp(&rate(b)))
                    .unwrap();
                for k in 0..=4 {
                    let (e, v) = committed_moments(&[alpha; 4], k);
                    println!(
                        "modeled,A,{name},{context},{k},{alpha},{e:.4},{v:.4},{:.4},{:.5},{best}",
                        row.round_cost(k),
                        rate(k)
                    );
                }
            }
        }
    }
}

fn part_b() {
    let profile = &profiles()[0].1;
    println!("kind,stage,phase,step,mean_k,conservative_k,lower_p0");
    let mut e = Estimator::new(4, 0.97);
    for (phase, accepted) in [("high", 4), ("low", 0), ("recovery", 3)] {
        for step in 0..40 {
            e.observe(accepted, 4);
            let mean = e.probabilities(0.);
            let low = e.probabilities(1.);
            println!(
                "modeled,B,{phase},{step},{},{},{:.4}",
                choose_k(&mean, &profile[0], 2., 20).unwrap(),
                choose_k(&low, &profile[0], 2., 20).unwrap(),
                low[0]
            );
        }
    }
    println!("kind,stage,requests,instances,greedy_suboptimal,worst_greedy_over_exact,mean_greedy_over_exact");
    let mut rng = Lcg(42);
    for n in [2, 4, 8] {
        let (mut sub, mut worst, mut sum) = (0, 1.0_f64, 0.0);
        let trials = 200;
        for _ in 0..trials {
            let ps: Vec<Vec<f64>> = (0..n)
                .map(|_| (0..4).map(|_| (rng.next() * 10.).round() / 10.).collect())
                .collect();
            let rows: Vec<ProfilePoint> = (0..n)
                .map(|_| {
                    let mut v = vec![1.0];
                    for _ in 0..4 {
                        v.push(v.last().unwrap() + (1. + (rng.next() * 8.).floor()) / 10.);
                    }
                    ProfilePoint {
                        context: 0,
                        draft: (rng.next() * 3.).floor() / 10.,
                        verify: v,
                    }
                })
                .collect();
            let budget = n as f64 + (rng.next() * 2.0 * n as f64 * 10.).floor() / 10.;
            let value = |ks: &[usize]| -> f64 {
                ks.iter()
                    .enumerate()
                    .map(|(i, &k)| committed_moments(&ps[i], k).0)
                    .sum()
            };
            let g = value(&allocate_round(&ps, &rows, budget).unwrap());
            let x = value(&allocate_round_exact(&ps, &rows, budget, 0.1).unwrap());
            if g < x - 1e-9 {
                sub += 1;
            }
            worst = worst.min(g / x);
            sum += g / x;
        }
        println!(
            "modeled,B,{n},{trials},{sub},{worst:.4},{:.5}",
            sum / trials as f64
        );
    }
}

fn phase_accept(id: usize) -> Vec<f64> {
    // Non-IID, depth-dependent: high phase decays with depth, low phase is flat, the
    // recovery phase has a bursty (increasing) hazard like the real Qwen3 trace.
    match id {
        0..=29 => (0..8).map(|j| 0.92 * 0.85f64.powi(j)).collect(),
        30..=59 => vec![0.25; 8],
        _ => (0..8).map(|j| (0.7 + 0.04 * j as f64).min(0.95)).collect(),
    }
}

fn part_c() {
    // Loads are expressed in units of T_v(0) of each profile: mean output is 12 tokens,
    // so an arrival gap of 14 T_v(0) keeps plain decoding near 86% utilization.
    println!("kind,stage,profile,policy,seed,arrival_gap_ms,requests,tokens,p95_ttft_ms,p95_e2e_ms,p95_tpot_ms,goodput_req_per_s,met,infeasible,resets,busy_ms");
    for (pname, profile) in profiles() {
        let t0 = profile[0].verify[0];
        for gap in [8., 14., 30.] {
            for seed in 1..=5 {
                let mut policies: Vec<Box<dyn Policy>> = vec![
                    Box::new(Fixed(0)),
                    Box::new(Fixed(2)),
                    Box::new(Fixed(4)),
                    Box::new(PerPosition::new(4, 0.97, 0.)),
                    Box::new(Pooled {
                        estimator: Estimator::new(4, 0.97),
                    }),
                    Box::new(PerPosition::new(4, 0.97, 1.)),
                ];
                for policy in policies.iter_mut() {
                    let requests = (0..90)
                        .map(|id| {
                            let output = 8 + id % 9;
                            let arrival = id as f64 * gap * t0;
                            Request {
                                id,
                                arrival,
                                output,
                                context: (id % 4) * 128,
                                deadline: arrival + (1.5 * output as f64 + 10.) * t0,
                                accept: phase_accept(id),
                            }
                        })
                        .collect();
                    let sim = simulate(
                        requests,
                        &profile,
                        policy.as_mut(),
                        &Synthetic { seed },
                        3. * t0,
                    );
                    let m = summarize(&sim);
                    println!(
                        "modeled,C,{pname},{},{seed},{:.2},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{},{:.2}",
                        policy.name(), gap * t0, m.requests, m.tokens, m.p95_ttft, m.p95_e2e, m.p95_tpot, m.goodput, m.met, sim.infeasible, sim.resets, sim.busy
                    );
                }
            }
        }
    }
}

fn part_r(path: &std::path::Path) {
    let trace = match Trace::load(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read trace {}: {e}", path.display());
            std::process::exit(1);
        }
    };
    let label = path.file_name().unwrap().to_string_lossy();
    println!(
        "# trace {label}: device={} driver={} date={} torch={} prompts={} unstable_timing_rows={} (measured inputs; replay outputs are modeled)",
        trace.header.get("device").map(String::as_str).unwrap_or("?"),
        trace.header.get("driver").map(String::as_str).unwrap_or("?"),
        trace.header.get("date").map(String::as_str).unwrap_or("?"),
        trace.header.get("torch").map(String::as_str).unwrap_or("?"),
        trace.prompts.len(),
        trace.unstable_rows
    );
    let rows = trace
        .profile()
        .expect("calibrate the measured cost samples (A5)");
    println!("kind,stage,table,regime,context,k,verify_ms,draft_ms,round_ms,c");
    let eager = trace.profile_eager().unwrap_or_default();
    for (regime, table) in [("graph", &rows), ("eager", &eager)] {
        for row in table.iter() {
            for k in 0..row.verify.len() {
                println!(
                    "measured,R,cost,{regime},{},{k},{:.3},{:.3},{:.3},{:.3}",
                    row.context,
                    row.verify[k],
                    row.draft,
                    row.round_cost(k),
                    row.draft / row.verify[0]
                );
            }
        }
    }
    // Per-position survival vs hazard, aggregated over prompts by counts.
    println!("kind,stage,table,task,j,survival_P(L>=j),hazard_a_j,observable_starts");
    let mut tasks: BTreeMap<String, Vec<&TracePrompt>> = BTreeMap::new();
    for p in &trace.prompts {
        tasks.entry("all".into()).or_default().push(p);
        tasks.entry(p.task.clone()).or_default().push(p);
    }
    for (task, prompts) in &tasks {
        for j in 1..=6 {
            let (mut starts, mut reach, mut prior) = (0usize, 0usize, 0usize);
            for p in prompts {
                let runs = run_lengths(&p.flags);
                for i in 0..p.flags.len() {
                    if i + j <= p.flags.len() {
                        starts += 1;
                        reach += (runs[i] >= j) as usize;
                        prior += (runs[i] >= j - 1) as usize;
                    }
                }
            }
            println!(
                "measured,R,rates,{task},{j},{:.4},{:.4},{starts}",
                reach as f64 / starts.max(1) as f64,
                reach as f64 / prior.max(1) as f64
            );
        }
    }
    // R4: fixed-k speedup, measured by the hand-written loop vs modeled by the replay.
    println!("kind,stage,table,k,runs,same_as_greedy,rounds_compared,rounds_equal,measured_speedup,modeled_speedup_replay,modeled_speedup_iid,modeled_ms_per_measured_ms");
    let by_id: BTreeMap<usize, &TracePrompt> = trace.prompts.iter().map(|p| (p.id, p)).collect();
    let mut ks: Vec<usize> = trace.spec.iter().map(|s| s.k).collect();
    ks.sort();
    ks.dedup();
    let alpha = {
        let (s, n): (usize, usize) = trace.prompts.iter().fold((0, 0), |a, p| {
            (
                a.0 + p.flags.iter().filter(|&&f| f == 1).count(),
                a.1 + p.flags.len(),
            )
        });
        s as f64 / n.max(1) as f64
    };
    for k in ks {
        let runs: Vec<&SpecRun> = trace.spec.iter().filter(|s| s.k == k).collect();
        let (mut plain_m, mut spec_m, mut plain_x, mut spec_x, mut compared, mut equal, mut same) =
            (0.0, 0.0, 0.0, 0.0, 0, 0, 0);
        for run in &runs {
            let p = by_id[&run.prompt];
            let r = run_lengths(&p.flags);
            let schedule = fixed_k_schedule(&r, k);
            let flags = p.stepwise_flags.as_ref().unwrap_or(&p.flags);
            let (n, ok) = compare_before_divergence(
                &run.rounds,
                &fixed_k_schedule(&run_lengths(flags), k),
                run.first_diff,
            );
            compared += n;
            equal += ok;
            same += run.same_as_greedy as usize;
            plain_m += trace.plain_ms[&run.prompt];
            spec_m += run.decode_ms;
            plain_x += schedule_ms(&fixed_k_schedule(&r, 0), &rows, p.prompt_tokens);
            spec_x += schedule_ms(&schedule, &rows, p.prompt_tokens);
        }
        let row = interpolate(&rows, 256);
        let iid = committed_moments(&[alpha; 16], k).0 * row.verify[0] / row.round_cost(k);
        println!(
            "measured,R,fixed_k,{k},{},{same},{compared},{equal},{:.4},{:.4},{:.4},{:.4}",
            runs.len(),
            plain_m / spec_m,
            plain_x / spec_x,
            iid,
            spec_x / spec_m
        );
    }
    // Controllers on the real trace: prompts in task blocks (drift), round-robin server.
    println!("kind,stage,table,load,policy,requests,tokens,busy_ms,tokens_per_s,p95_tpot_ms,met,infeasible,resets,mean_k,regret_vs_best_fixed");
    let mut order: Vec<&TracePrompt> = trace.prompts.iter().filter(|p| p.flags.len() > 1).collect();
    order.sort_by_key(|p| (p.task.clone(), p.id));
    let flags: Vec<Vec<u8>> = order.iter().map(|p| p.flags.clone()).collect();
    let env = TraceEnv::from_flags(&flags);
    let t0 = rows[0].verify[0];
    // arrival gap in units of T_v(0): "loaded" keeps plain decoding near 95% utilization
    for (load, gap_tokens) in [("sequential", 400.0), ("loaded", 100.0)] {
        let tpot_slo = 0.9 * t0; // tighter than plain decoding: only speculation can meet it
        let requests: Vec<Request> = order
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let arrival = i as f64 * gap_tokens * t0;
                Request {
                    id: i,
                    arrival,
                    output: p.flags.len(),
                    context: p.prompt_tokens,
                    deadline: arrival + tpot_slo * p.flags.len() as f64 + 2.0 * t0,
                    accept: vec![],
                }
            })
            .collect();
        let kmax = trace.kmax;
        let mut results = Vec::new();
        let mut policies: Vec<Box<dyn Policy>> = (0..=kmax.min(6))
            .map(|k| Box::new(Fixed(k)) as Box<dyn Policy>)
            .collect();
        policies.push(Box::new(PerPosition::new(kmax, 0.97, 0.)));
        policies.push(Box::new(Pooled {
            estimator: Estimator::new(kmax, 0.97),
        }));
        policies.push(Box::new(PerPosition::new(kmax, 0.97, 1.)));
        policies.push(Box::new(HfHeuristic { k: 2, kmax }));
        policies.push(Box::new(ArmUcb::new(kmax, 0.97, 0.5)));
        policies.push(Box::new(CensoredUcb::new(kmax, 0.97, 1.0)));
        for policy in policies.iter_mut() {
            let sim = simulate(requests.clone(), &rows, policy.as_mut(), &env, 1e9);
            let m = summarize(&sim);
            let mean_k =
                sim.rounds.iter().map(|r| r.k as f64).sum::<f64>() / sim.rounds.len() as f64;
            results.push((
                policy.name(),
                m,
                sim.busy,
                sim.infeasible,
                sim.resets,
                mean_k,
            ));
        }
        let best_fixed = results
            .iter()
            .filter(|r| r.0.starts_with('k'))
            .map(|r| r.2)
            .fold(f64::INFINITY, f64::min);
        for (name, m, busy, infeasible, resets, mean_k) in results {
            println!(
                "modeled,R,policy,{load},{name},{},{},{busy:.1},{:.2},{:.3},{},{infeasible},{resets},{mean_k:.3},{:.4}",
                m.requests, m.tokens, 1000.0 * m.tokens as f64 / busy, m.p95_tpot, m.met, busy / best_fixed - 1.0
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let part = args.get(1).cloned().unwrap_or_else(|| "all".into());
    if part == "A" || part == "all" {
        part_a();
    }
    if part == "B" || part == "all" {
        part_b();
    }
    if part == "C" || part == "all" {
        part_c();
    }
    if part == "R" || part == "all" {
        let traces: Vec<std::path::PathBuf> = if args.len() > 2 {
            args[2..].iter().map(Into::into).collect()
        } else {
            vec![reference_trace_path()]
        };
        for t in traces {
            part_r(&t);
        }
    }
}
