use prefix_cache::*;
use std::collections::BTreeSet;
fn ns(tenant: &str) -> Namespace {
    Namespace {
        model: "model@sha".into(),
        adapter: "none".into(),
        tokenizer: "tok@sha".into(),
        tenant: tenant.into(),
        position_offset: 0,
    }
}
fn fixture() -> Cache {
    let mut c = Cache::new(4, 2, Policy::Lru);
    c.clock = 5;
    for (tokens, last) in [(vec![1, 2], 1), (vec![1, 2, 3, 4], 2), (vec![9, 8], 3)] {
        c.nodes.insert(
            Key {
                ns: ns("a"),
                prefix: tokens,
            },
            Entry {
                refs: 0,
                last,
                hits: last,
            },
        );
    }
    c
}
fn req(id: usize, arrival: f64, tenant: &str, tokens: Vec<u32>) -> Request {
    Request {
        id,
        arrival,
        ns: ns(tenant),
        tokens,
        decode_ms: 1.,
    }
}
#[test]
fn a_keys_namespace_and_tail() {
    let keys = block_keys(&ns("a"), &[1, 2, 3, 4, 5], 2);
    assert_eq!(
        keys.iter().map(|k| k.prefix.clone()).collect::<Vec<_>>(),
        vec![vec![1, 2], vec![1, 2, 3, 4]]
    );
    assert_ne!(keys[0], block_keys(&ns("b"), &[1, 2], 2)[0]);
    for field in 0..4 {
        let mut n = ns("a");
        match field {
            0 => n.model = "other".into(),
            1 => n.adapter = "other".into(),
            2 => n.tokenizer = "other".into(),
            _ => n.position_offset = 8,
        };
        assert_ne!(keys[0], block_keys(&n, &[1, 2], 2)[0]);
    }
    assert!(block_keys(&ns("a"), &[1], 2).is_empty());
}
#[test]
fn a_match_stops_at_gap() {
    let mut c = fixture();
    // Both roots exist, but equal local suffix [3,4] has a different ancestor.
    assert_eq!(c.matched_tokens(&ns("a"), &[9, 8, 3, 4]), 2);
    assert_eq!(c.matched_tokens(&ns("a"), &[1, 2, 3, 4, 9]), 4);
    assert_eq!(c.matched_tokens(&ns("a"), &[1, 2, 9, 4]), 2);
    assert_eq!(c.matched_tokens(&ns("b"), &[1, 2, 3, 4]), 0);
    c.nodes.remove(&Key {
        ns: ns("a"),
        prefix: vec![1, 2],
    });
    assert_eq!(c.matched_tokens(&ns("a"), &[1, 2, 3, 4]), 0);
}
#[test]
fn a_leaf_victims() {
    let mut c = fixture();
    let protected = BTreeSet::new();
    assert_eq!(c.victim(&protected).unwrap().prefix, vec![1, 2, 3, 4]);
    c.nodes
        .get_mut(&Key {
            ns: ns("a"),
            prefix: vec![1, 2, 3, 4],
        })
        .unwrap()
        .refs = 1;
    assert_eq!(c.victim(&protected).unwrap().prefix, vec![9, 8]);
    let protected = BTreeSet::from([Key {
        ns: ns("a"),
        prefix: vec![9, 8],
    }]);
    assert!(c.victim(&protected).is_none());
    let mut c = fixture();
    c.policy = Policy::Frequency;
    c.nodes
        .get_mut(&Key {
            ns: ns("a"),
            prefix: vec![9, 8],
        })
        .unwrap()
        .hits = 0;
    assert_eq!(c.victim(&BTreeSet::new()).unwrap().prefix, vec![9, 8]);
}
#[test]
fn a_invariants_detect_corruption() {
    assert!(fixture().validate().is_ok());
    let mut c = fixture();
    c.capacity = 2;
    assert_eq!(c.validate(), Err("capacity".into()));
    let mut c = fixture();
    c.nodes.remove(&Key {
        ns: ns("a"),
        prefix: vec![1, 2],
    });
    assert_eq!(c.validate(), Err("orphan".into()));
    let mut c = fixture();
    c.nodes
        .get_mut(&Key {
            ns: ns("a"),
            prefix: vec![1, 2, 3, 4],
        })
        .unwrap()
        .refs = 1;
    assert_eq!(c.validate(), Err("lease ancestry".into()));
}
#[test]
fn b_transaction_rollback_and_leaf_order() {
    let mut c = Cache::new(2, 2, Policy::Lru);
    assert!(c.insert(&ns("a"), &[1, 2, 3, 4]));
    let lease = c.acquire(&ns("a"), &[1, 2]);
    let before = c.clone();
    assert!(!c.insert(&ns("b"), &[5, 6, 7, 8]));
    assert_eq!(c.nodes, before.nodes);
    assert_eq!(c.clock, before.clock);
    c.release(lease);
    assert!(c.insert(&ns("b"), &[5, 6]));
    assert_eq!(c.matched_tokens(&ns("a"), &[1, 2, 3, 4]), 2);
    let before = c.clone();
    assert!(!c.insert(&ns("a"), &[1, 2, 3, 4, 5, 6]));
    assert_eq!(c.nodes, before.nodes);
}
#[test]
fn b_shared_leases() {
    let mut c = Cache::new(1, 2, Policy::Lru);
    c.insert(&ns("a"), &[1, 2]);
    let one = c.acquire(&ns("a"), &[1, 2]);
    let two = c.acquire(&ns("a"), &[1, 2]);
    c.release(one);
    assert!(!c.insert(&ns("a"), &[9, 8]));
    c.release(two);
    assert!(c.insert(&ns("a"), &[9, 8]));
}
#[test]
fn b_release_preserves_reuse() {
    let mut c = fixture();
    let lease = c.acquire(&ns("a"), &[1, 2, 3, 4]);
    assert_eq!(c.nodes.values().map(|e| e.refs).sum::<usize>(), 2);
    c.release(lease);
    assert_eq!(c.matched_tokens(&ns("a"), &[1, 2, 3, 4]), 4);
    assert!(c.nodes.values().all(|e| e.refs == 0));
    let mut empty = Cache::new(0, 2, Policy::Lru);
    assert!(empty.insert(&ns("a"), &[]));
    assert!(!empty.insert(&ns("a"), &[1, 2]));
}
#[test]
fn b_namespace_invalidation() {
    let mut c = fixture();
    c.insert(&ns("b"), &[6, 6]);
    let lease = c.acquire(&ns("a"), &[1, 2]);
    let before = c.clone();
    assert_eq!(c.invalidate(&ns("a")), None);
    assert_eq!(c.nodes, before.nodes);
    c.release(lease);
    assert_eq!(c.invalidate(&ns("a")), Some(3));
    assert_eq!(c.matched_tokens(&ns("b"), &[6, 6]), 2);
}
#[test]
fn c_admission_pollution_and_bound() {
    let mut a = Admission::new(8);
    assert_eq!(
        a.observe(&ns("a"), &[1, 2, 3, 4], 2, AdmissionPolicy::TwoHit),
        0
    );
    assert_eq!(
        a.observe(&ns("a"), &[1, 2, 9, 9], 2, AdmissionPolicy::TwoHit),
        2
    );
    assert_eq!(
        a.observe(&ns("b"), &[1, 2, 9, 9], 2, AdmissionPolicy::TwoHit),
        0
    );
    for i in 0..30 {
        a.observe(&ns("a"), &[i, i], 2, AdmissionPolicy::TwoHit);
        assert!(a.history.len() <= 8);
    }
    assert_eq!(
        Admission::new(0).observe(&ns("a"), &[1, 2], 2, AdmissionPolicy::TwoHit),
        0
    );
}
#[test]
fn c_scheduler_age_and_prefix() {
    let c = fixture();
    let r = vec![
        req(0, 0., "a", vec![9, 9]),
        req(1, 1., "a", vec![1, 2, 3, 4]),
        req(2, 2., "a", vec![1, 2]),
    ];
    assert_eq!(
        select(&[0, 1, 2], &r, &c, 3., 10., Schedule::LongestPrefix),
        Some(1)
    );
    assert_eq!(
        select(&[0, 1, 2], &r, &c, 12., 10., Schedule::LongestPrefix),
        Some(0)
    );
    assert_eq!(select(&[], &r, &c, 3., 10., Schedule::Fifo), None);
}
#[test]
fn c_replay_hand_timeline() {
    let r = vec![
        req(0, 0., "a", vec![1, 2]),
        req(1, 0., "a", vec![1, 2]),
        req(2, 4., "a", vec![1, 2]),
    ];
    let out = replay(
        r,
        Cache::new(2, 2, Policy::Lru),
        2,
        1.,
        Schedule::Fifo,
        AdmissionPolicy::Always,
        10.,
    );
    assert_eq!(
        out.rows.iter().map(|r| r.hit).collect::<Vec<_>>(),
        vec![0, 0, 2]
    );
    assert_eq!(
        out.rows.iter().map(|r| r.finish).collect::<Vec<_>>(),
        vec![3., 3., 5.]
    );
    assert_eq!(out.peak_pins, 1);
}
#[test]
fn c_conservation_isolation_and_capacity() {
    let empty = replay(
        vec![],
        Cache::new(0, 2, Policy::Lru),
        1,
        1.,
        Schedule::Fifo,
        AdmissionPolicy::Always,
        5.,
    );
    assert!(empty.rows.is_empty());
    for policy in [AdmissionPolicy::Always, AdmissionPolicy::TwoHit] {
        for schedule in [Schedule::Fifo, Schedule::LongestPrefix] {
            let r = (0..100)
                .map(|i| {
                    req(
                        i,
                        (i / 10) as f64,
                        if i % 3 == 0 { "a" } else { "b" },
                        vec![1, 2, (i % 4) as u32, 9],
                    )
                })
                .collect();
            let out = replay(
                r,
                Cache::new(3, 2, Policy::Lru),
                3,
                0.5,
                schedule,
                policy,
                5.,
            );
            let ids: BTreeSet<_> = out.rows.iter().map(|r| r.id).collect();
            assert_eq!(ids.len(), 100);
            assert_eq!(out.rows.len(), 100);
            assert!(out.cache.validate().is_ok());
            assert!(out.cache.nodes.values().all(|e| e.refs == 0));
            assert_eq!(
                out.rows
                    .iter()
                    .filter(|r| r.id < 2)
                    .map(|r| r.hit)
                    .sum::<usize>(),
                0
            );
        }
    }
}
#[test]
fn c_metrics_weighting_and_empty() {
    let rows = vec![
        ResultRow {
            id: 0,
            tenant: "a".into(),
            arrival: 0.,
            start: 1.,
            finish: 3.,
            hit: 2,
            total: 2,
            admitted: true,
        },
        ResultRow {
            id: 1,
            tenant: "b".into(),
            arrival: 0.,
            start: 2.,
            finish: 5.,
            hit: 0,
            total: 8,
            admitted: false,
        },
    ];
    let m = summarize(&rows, None);
    assert_eq!(m.hit_ratio, 0.2);
    assert_eq!(m.mean_wait, 1.5);
    assert_eq!(m.p95_e2e, 5.);
    assert_eq!(m.recomputed, 8);
    assert_eq!(m.admission_failures, 1);
    assert_eq!(summarize(&rows, Some("a")).hit_ratio, 1.);
    assert_eq!(summarize(&rows, Some("missing")).requests, 0);
}
#[test]
fn a_accounting_multiple_leases() {
    let mut c = fixture();
    c.nodes
        .get_mut(&Key {
            ns: ns("a"),
            prefix: vec![1, 2],
        })
        .unwrap()
        .refs = 3;
    let s = c.stats();
    assert_eq!(s.blocks, 3);
    assert_eq!(s.pinned_blocks, 1);
    assert_eq!(s.references, 3);
    assert_eq!(s.key_token_bytes, 32);
    assert_eq!(s.max_depth, 2);
}
#[test]
fn b_resize_transaction_with_pins() {
    let mut c = fixture();
    let lease = c.acquire(&ns("a"), &[1, 2, 3, 4]);
    let before = c.clone();
    assert!(!c.resize(1));
    assert_eq!(c.nodes, before.nodes);
    assert_eq!(c.capacity, before.capacity);
    assert_eq!(c.clock, before.clock);
    assert!(c.resize(8));
    c.release(lease);
    assert!(c.resize(0));
    assert!(c.nodes.is_empty());
    assert_eq!(c.capacity, 0);
}
