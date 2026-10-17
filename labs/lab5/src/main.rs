use prefix_cache::*;
fn ns(tenant: &str) -> Namespace {
    Namespace {
        model: "fixed@sha".into(),
        adapter: "none".into(),
        tokenizer: "tok@sha".into(),
        tenant: tenant.into(),
        position_offset: 0,
    }
}
fn main() {
    let part = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if part == "A" || part == "all" {
        println!("kind,stage,prompt_tokens,block,keys,key_payload_bytes");
        for n in [16, 64, 256, 1024] {
            for block in [2, 4, 8] {
                let tokens = vec![1; n];
                let keys = block_keys(&ns("a"), &tokens, block);
                println!(
                    "cpu_key_structure,A,{n},{block},{},{}",
                    keys.len(),
                    keys.iter().map(|k| k.prefix.len() * 4).sum::<usize>()
                );
            }
        }
    }
    if part == "B" || part == "all" {
        println!("kind,stage,policy,capacity,hits,total,failed,stored");
        for cap in [2, 4, 8, 16] {
            for (name, policy) in [("lru", Policy::Lru), ("frequency", Policy::Frequency)] {
                let mut c = Cache::new(cap, 2, policy);
                let (mut hits, mut failed) = (0, 0);
                for i in 0..100 {
                    let tokens = vec![1, 1, (i % 5) as u32, 2];
                    hits += c.matched_tokens(&ns("a"), &tokens);
                    let lease = c.acquire(&ns("a"), &tokens);
                    c.release(lease);
                    failed += usize::from(!c.insert(&ns("a"), &tokens));
                }
                println!(
                    "synthetic_replay,B,{name},{cap},{hits},400,{failed},{}",
                    c.nodes.len()
                );
            }
        }
    }
    if part == "C" || part == "all" {
        println!("kind,stage,admission,schedule,capacity,tenant,requests,hit_ratio,mean_queue_sim_ms,p95_e2e_sim_ms,recomputed,capacity_failures,peak_pins");
        for cap in [4, 8, 16] {
            for (aname, admission) in [
                ("always", AdmissionPolicy::Always),
                ("two_hit", AdmissionPolicy::TwoHit),
            ] {
                for (sname, schedule) in [
                    ("fifo", Schedule::Fifo),
                    ("lpm_age", Schedule::LongestPrefix),
                ] {
                    let requests = (0..180)
                        .map(|i| {
                            let phase = i / 60;
                            let hot = if phase == 1 { i as u32 } else { (i % 2) as u32 };
                            Request {
                                id: i,
                                arrival: (i / 6) as f64 * 2.,
                                ns: ns(if i % 4 == 0 { "b" } else { "a" }),
                                tokens: vec![hot; 4]
                                    .into_iter()
                                    .chain([i as u32, 9, 8, 7])
                                    .collect(),
                                decode_ms: 2. + (i % 3) as f64,
                            }
                        })
                        .collect();
                    let result = replay(
                        requests,
                        Cache::new(cap, 2, Policy::Lru),
                        3,
                        0.3,
                        schedule,
                        admission,
                        12.,
                    );
                    for tenant in [None, Some("a"), Some("b")] {
                        let m = summarize(&result.rows, tenant);
                        println!(
                            "synthetic_queue,C,{aname},{sname},{cap},{},{},{},{},{},{},{},{}",
                            tenant.unwrap_or("all"),
                            m.requests,
                            m.hit_ratio,
                            m.mean_wait,
                            m.p95_e2e,
                            m.recomputed,
                            m.admission_failures,
                            result.peak_pins
                        );
                    }
                }
            }
        }
    }
}
