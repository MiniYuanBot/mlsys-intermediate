//! Lab 5 benchmarks (provided). Every number printed here is MODELED by a cost model or
//! is a structural count; only R's cost coefficients come from a GPU measurement.
//!   cargo run --release --bin bench -- A|B|C|R|all [trace.jsonl] [cost.json]
use prefix_cache::*;
use std::path::PathBuf;

fn ns(tenant: &str) -> Namespace {
    Namespace {
        model: "Qwen3-0.6B@rev".into(),
        adapter: "none".into(),
        tokenizer: "qwen3-tok@rev".into(),
        tenant: tenant.into(),
        position_offset: 0,
    }
}

const C_UNIT: usize = 32;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() * n as f64) as usize % n.max(1)
    }
}

/// Multi-tenant chat workload in 32-token units (one unit = one cache block of 32 tokens,
/// like Mooncake's 512-token hash ids): per tenant two system prompts (16 units = 512
/// tokens); sessions of up to 6 turns whose prompt is system + all earlier turns + a new
/// 4-unit question (the answer adds 4 more). Sessions start as a Poisson process at
/// `rate` per ms. One tenant ("bulk") sends one-shot requests with a shared 64-unit
/// (2048-token) template: a hot prefix.
fn chat_workload(rate: f64, seed: u64) -> Vec<Request> {
    let mut rng = Lcg(seed);
    let mut reqs = Vec::new();
    let mut next_token = 1000u32;
    let mut fresh = |n: usize| {
        let v: Vec<u32> = (0..n as u32).map(|i| next_token + i).collect();
        next_token += n as u32;
        v
    };
    let tenants = ["a", "b", "c"];
    let systems: Vec<Vec<Vec<u32>>> = tenants.iter().map(|_| vec![fresh(16), fresh(16)]).collect();
    let template = fresh(64);
    let mut t = 0.0;
    for s in 0..160usize {
        t += -(1.0 - rng.next()).ln() / rate;
        if s % 4 == 3 {
            let mut tokens = template.clone();
            tokens.extend(fresh(4));
            reqs.push((t, "bulk", tokens, 20.0));
            continue;
        }
        let ti = rng.below(3);
        let mut history = systems[ti][rng.below(2)].clone();
        let turns = 1 + rng.below(6);
        let mut at = t;
        for _ in 0..turns {
            history.extend(fresh(4));
            reqs.push((
                at,
                tenants[ti],
                history.clone(),
                20.0 + rng.below(40) as f64,
            ));
            history.extend(fresh(4)); // the answer joins the next turn's prompt
            at += 200.0 + rng.below(400) as f64;
        }
    }
    reqs.sort_by(|a, b| a.0.total_cmp(&b.0));
    reqs.into_iter()
        .enumerate()
        .map(|(id, (arrival, tenant, tokens, decode_ms))| Request {
            id,
            arrival,
            ns: ns(tenant),
            prompt_tokens: tokens.len() * C_UNIT,
            tokens,
            decode_ms,
        })
        .collect()
}

fn routes() -> Vec<(&'static str, RoutePolicy)> {
    vec![
        ("round_robin", RoutePolicy::RoundRobin),
        ("least_load", RoutePolicy::LeastLoad),
        ("prefix_affinity", RoutePolicy::PrefixAffinity),
        (
            "hybrid",
            RoutePolicy::Hybrid {
                abs_threshold: 4,
                rel_threshold: 1.5,
                cache_threshold: 0.3,
            },
        ),
    ]
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("real/data")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let part = args.get(1).cloned().unwrap_or_else(|| "all".into());
    if part == "A" || part == "all" {
        println!("kind,stage,prompt_tokens,block,keys,vec_key_payload_bytes,chained_hash_bytes");
        for n in [16, 64, 256, 1024] {
            for block in [2, 4, 8] {
                let tokens = vec![1; n];
                let keys = block_keys(&ns("a"), &tokens, block);
                let hashes = block_hashes(
                    &HashRequest {
                        tokens: tokens.clone(),
                        ..Default::default()
                    },
                    block,
                );
                println!(
                    "structure,A,{n},{block},{},{},{}",
                    keys.len(),
                    keys.iter().map(|k| k.prefix.len() * 4).sum::<usize>(),
                    hashes.len() * 8
                );
            }
        }
    }
    if part == "B" || part == "all" {
        println!("kind,stage,policy,capacity,hit_tokens,total_tokens,failed_inserts,stored");
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
                    "serial_replay,B,{name},{cap},{hits},400,{failed},{}",
                    c.nodes.len()
                );
            }
        }
    }
    if part == "C" || part == "all" {
        // Modeled cost (rounded from the committed 4060 fit, kept fixed so C results do
        // not move when R is re-measured).
        let cost = CostModel {
            floor_ms: 23.0,
            a_ms: 0.0,
            b_ms: 0.067,
            c_ms: 1.6e-5,
            d_ms: 0.0057,
        };
        println!("kind,stage,route,admission,rate_per_s,requests,hit_ratio,mean_ttft_ms,p95_ttft_ms,imbalance,duplicated_blocks,insert_failures,label");
        for rate_per_s in [2.0, 8.0] {
            let trace = chat_workload(rate_per_s / 1000.0, 7);
            for (rname, route) in routes() {
                for (aname, admission) in [
                    ("always", AdmissionPolicy::Always),
                    ("two_hit", AdmissionPolicy::TwoHit),
                ] {
                    let cfg = ClusterConfig {
                        replicas: 4,
                        slots: 4,
                        capacity_blocks: 256,
                        block: 1,
                        unit_tokens: C_UNIT,
                        eviction: Policy::Lru,
                        admission,
                        history_capacity: 512,
                        rule: FullHitRule::Sglang,
                        cost,
                        route,
                    };
                    let out = replay(trace.clone(), &cfg);
                    let m = summarize(&out.rows, None, cfg.replicas);
                    println!(
                        "cluster,C,{rname},{aname},{rate_per_s},{},{:.4},{:.2},{:.2},{:.3},{},{},modeled",
                        m.requests,
                        m.hit_ratio,
                        m.mean_ttft,
                        m.p95_ttft,
                        m.imbalance,
                        duplicated_blocks(&out.caches),
                        m.insert_failures
                    );
                }
            }
        }
    }
    if part == "R" || part == "all" {
        let trace_path = args
            .get(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir().join("synthetic-mooncake.jsonl"));
        let cost_path = args
            .get(3)
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir().join("cost-qwen3-0.6b-rtx4060.json"));
        let (cost, label) =
            load_cost_model(&std::fs::read_to_string(&cost_path).expect("read cost model"))
                .unwrap_or_else(|e| panic!("{}: {e}", cost_path.display()));
        let text = std::fs::read_to_string(&trace_path).expect("read trace");
        let tc = TraceConfig {
            time_scale: 1.0,
            decode_ms_per_token: 0.0,
            ns: ns("mooncake"),
        };
        let reqs = load_mooncake(&text, &tc)
            .unwrap_or_else(|(line, e)| panic!("{}:{line}: {e:?}", trace_path.display()));
        println!(
            "# trace {} ({} requests); cost model {} [{}]; decode not modeled (TTFT only)",
            trace_path.file_name().unwrap().to_string_lossy(),
            reqs.len(),
            cost_path.file_name().unwrap().to_string_lossy(),
            label
        );
        println!("kind,stage,route,rule,replicas,token_hit,compute_saved,mean_ttft_ms,p95_ttft_ms,imbalance,duplicated_blocks,label");
        for replicas in [2, 4] {
            for (rname, route) in routes() {
                for (fname, rule) in [("sglang", FullHitRule::Sglang), ("vllm", FullHitRule::Vllm)]
                {
                    let cfg = ClusterConfig {
                        replicas,
                        slots: 2,
                        capacity_blocks: 128,
                        block: 1,
                        unit_tokens: MOONCAKE_BLOCK_TOKENS,
                        eviction: Policy::Lru,
                        admission: AdmissionPolicy::Always,
                        history_capacity: 4096,
                        rule,
                        cost,
                        route,
                    };
                    let out = replay(reqs.clone(), &cfg);
                    let m = summarize(&out.rows, None, replicas);
                    let s = compute_saved(&out.rows, &cost);
                    println!(
                        "mooncake_replay,R,{rname},{fname},{replicas},{:.4},{:.4},{:.1},{:.1},{:.3},{},modeled_with_measured_cost",
                        s.token_hit,
                        s.compute_saved,
                        m.mean_ttft,
                        m.p95_ttft,
                        m.imbalance,
                        duplicated_blocks(&out.caches)
                    );
                }
            }
        }
    }
}
