//! Lab 5 tests. Every test prints one PASS/FAIL line per check and only fails at the
//! end, so one run lists everything that is wrong. Oracles live in this file and do not
//! call the function under test: hand-computed timelines (derivations in comments),
//! brute-force enumeration, spec re-implementations written differently, and invariant
//! checks over randomized replays.
use prefix_cache::*;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

struct Checks {
    test: &'static str,
    failures: Vec<String>,
}
impl Checks {
    fn new(test: &'static str) -> Self {
        // Panics inside guard() become FAIL lines with their message; keep stderr quiet.
        std::panic::set_hook(Box::new(|_| {}));
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
    fn eq<T: PartialEq + Debug>(&mut self, name: &str, got: T, want: T) {
        let ok = got == want;
        self.check(name, ok, || format!("expected {want:?} vs got {got:?}"));
    }
    fn close(&mut self, name: &str, got: f64, want: f64, tol: f64) {
        self.check(name, (got - want).abs() <= tol, || {
            format!("expected {want} vs got {got} (tol {tol})")
        });
    }
    /// Run `f`; a panic (todo!, assert) becomes one FAIL line instead of an abort.
    fn guard<T>(
        &mut self,
        name: &str,
        f: impl FnOnce() -> T + std::panic::UnwindSafe,
    ) -> Option<T> {
        match std::panic::catch_unwind(f) {
            Ok(v) => Some(v),
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic".into());
                self.check(name, false, || format!("panicked: {msg}"));
                None
            }
        }
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

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn ns(tenant: &str) -> Namespace {
    Namespace {
        model: "Qwen3-0.6B@rev".into(),
        adapter: "none".into(),
        tokenizer: "qwen3-tok@rev+template-v1".into(),
        tenant: tenant.into(),
        position_offset: 0,
    }
}
fn key(tenant: &str, prefix: &[u32]) -> Key {
    Key {
        ns: ns(tenant),
        prefix: prefix.to_vec(),
    }
}
/// capacity 4, block 2: [1,2] -> [1,2,3,4] chain plus a separate root [9,8].
fn fixture() -> Cache {
    let mut c = Cache::new(4, 2, Policy::Lru);
    c.clock = 5;
    for (tokens, last) in [(vec![1, 2], 1), (vec![1, 2, 3, 4], 2), (vec![9, 8], 3)] {
        c.nodes.insert(
            key("a", &tokens),
            Entry {
                refs: 0,
                last,
                hits: last,
            },
        );
    }
    c
}
fn snapshot(c: &Cache) -> (BTreeMap<Key, Entry>, u64, usize) {
    (c.nodes.clone(), c.clock, c.capacity)
}

// ===================================================================== A
#[test]
fn a_hash_chain_extra_keys() {
    let mut c = Checks::new("a_hash_chain_extra_keys");
    let toks: Vec<u32> = (1..=10).collect(); // block 4: [1..4] [5..8], tail [9,10] unhashed
    let b0 = [1u32, 2, 3, 4];
    let b1 = [5u32, 6, 7, 8];
    let base = HashRequest {
        tokens: toks.clone(),
        ..Default::default()
    };
    let lora = ExtraKey::Lora {
        name: "sql".into(),
        path: "/adapters/sql-v1".into(),
    };
    let salt = ExtraKey::Salt("tenant-a".into());
    // Expected values come straight from the provided primitive and the A1 contract.
    let h0 = hash_block(NONE_HASH, &b0, &[]);
    let plain = vec![h0, hash_block(h0, &b1, &[])];
    let l0 = hash_block(NONE_HASH, &b0, &[lora.clone()]);
    let with_lora = vec![l0, hash_block(l0, &b1, &[lora.clone()])];
    let s0 = hash_block(NONE_HASH, &b0, &[salt.clone()]);
    let with_salt = vec![s0, hash_block(s0, &b1, &[])];
    let ls0 = hash_block(NONE_HASH, &b0, &[lora.clone(), salt.clone()]);
    let with_both = vec![ls0, hash_block(ls0, &b1, &[lora.clone()])];
    let run =
        |c: &mut Checks, name: &str, req: HashRequest| c.guard(name, move || block_hashes(&req, 4));
    if let Some(got) = run(&mut c, "plain", base.clone()) {
        c.eq("plain: two full blocks, tail dropped", got, plain.clone());
    }
    let mut r = base.clone();
    r.lora = Some(("sql".into(), "/adapters/sql-v1".into()));
    if let Some(got) = run(&mut c, "lora", r.clone()) {
        c.eq("lora (name,path) on every block", got, with_lora.clone());
    }
    let mut r2 = r.clone();
    r2.lora = Some(("sql".into(), "/adapters/sql-v2".into()));
    if let (Some(a), Some(b)) = (
        run(&mut c, "lora v1", r.clone()),
        run(&mut c, "lora v2", r2),
    ) {
        c.check(
            "same LoRA name, new path: every block differs",
            a.iter().zip(&b).all(|(x, y)| x != y),
            || format!("{a:?} vs {b:?}"),
        );
    }
    let mut s = base.clone();
    s.cache_salt = Some("tenant-a".into());
    if let Some(got) = run(&mut c, "salt", s.clone()) {
        c.eq(
            "salt only on block 0, carried by the chain",
            got.clone(),
            with_salt.clone(),
        );
        c.check(
            "salt changes every block (vs plain)",
            got.iter().zip(&plain).all(|(x, y)| x != y),
            || format!("{got:?}"),
        );
    }
    let mut both = r.clone();
    both.cache_salt = Some("tenant-a".into());
    if let Some(got) = run(&mut c, "lora+salt", both) {
        c.eq("order: lora then salt on block 0", got, with_both);
    }
    // Multimodal: item A covers tokens 3..6 (overlaps blocks 0 and 1), item B covers
    // 8..10 (only the unhashed tail), item C covers 0..4 (ends exactly at block 1's start).
    let mut m = base.clone();
    m.mm = vec![
        MmItem {
            identifier: "imgA".into(),
            start: 3,
            len: 3,
        },
        MmItem {
            identifier: "imgB".into(),
            start: 8,
            len: 2,
        },
        MmItem {
            identifier: "imgC".into(),
            start: 0,
            len: 4,
        },
    ];
    let a0 = ExtraKey::Mm {
        identifier: "imgA".into(),
        offset: 3,
    };
    let a1 = ExtraKey::Mm {
        identifier: "imgA".into(),
        offset: -1,
    };
    let c0 = ExtraKey::Mm {
        identifier: "imgC".into(),
        offset: 0,
    };
    let m0 = hash_block(NONE_HASH, &b0, &[a0, c0]);
    let want = vec![m0, hash_block(m0, &b1, &[a1])];
    if let Some(got) = run(&mut c, "mm", m) {
        c.eq(
            "mm: overlap test, signed offset, item order, no touch-only overlap",
            got,
            want,
        );
    }
    // Chain property: editing block 1 keeps hash 0; editing block 0 changes both.
    let mut e1 = base.clone();
    e1.tokens[5] = 99;
    let mut e0 = base.clone();
    e0.tokens[0] = 99;
    if let (Some(p), Some(x1), Some(x0)) = (
        run(&mut c, "p", base.clone()),
        run(&mut c, "e1", e1),
        run(&mut c, "e0", e0),
    ) {
        c.check(
            "edit in block 1 keeps block 0",
            p[0] == x1[0] && p[1] != x1[1],
            || format!("{p:?} vs {x1:?}"),
        );
        c.check(
            "edit in block 0 changes every later block",
            p.iter().zip(&x0).all(|(x, y)| x != y),
            || format!("{p:?} vs {x0:?}"),
        );
    }
    // Same last block after different first blocks: equal local tokens, different hash.
    let x = HashRequest {
        tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
        ..Default::default()
    };
    let y = HashRequest {
        tokens: vec![9, 9, 9, 9, 5, 6, 7, 8],
        ..Default::default()
    };
    if let (Some(hx), Some(hy)) = (run(&mut c, "x", x), run(&mut c, "y", y)) {
        c.check(
            "same block tokens, different parent",
            hx[1] != hy[1],
            || format!("{hx:?} {hy:?}"),
        );
    }
    let short = HashRequest {
        tokens: vec![1, 2, 3],
        ..Default::default()
    };
    if let Some(got) = run(&mut c, "short", short) {
        c.eq("shorter than one block -> no hashes", got, vec![]);
    }
    c.done();
}
#[test]
fn a_keys_namespace_and_tail() {
    let mut c = Checks::new("a_keys_namespace_and_tail");
    let Some(keys) = c.guard("block_keys", || block_keys(&ns("a"), &[1, 2, 3, 4, 5], 2)) else {
        return c.done();
    };
    c.eq(
        "full ancestry prefixes, tail dropped",
        keys.iter().map(|k| k.prefix.clone()).collect::<Vec<_>>(),
        vec![vec![1, 2], vec![1, 2, 3, 4]],
    );
    c.check(
        "namespace copied",
        keys.iter().all(|k| k.ns == ns("a")),
        || format!("{keys:?}"),
    );
    for (name, field) in [
        ("model", 0),
        ("adapter", 1),
        ("tokenizer", 2),
        ("tenant", 3),
        ("position_offset", 4),
    ] {
        let mut n = ns("a");
        match field {
            0 => n.model = "other".into(),
            1 => n.adapter = "other".into(),
            2 => n.tokenizer = "other".into(),
            3 => n.tenant = "b".into(),
            _ => n.position_offset = 8,
        };
        let other = block_keys(&n, &[1, 2], 2);
        c.check(
            &format!("{name} changes identity"),
            other.first() != keys.first(),
            || format!("{other:?}"),
        );
    }
    c.eq("one token, block 2", block_keys(&ns("a"), &[1], 2), vec![]);
    c.eq(
        "block 1 gives one key per token",
        block_keys(&ns("a"), &[7, 7], 1).len(),
        2,
    );
    c.done();
}
#[test]
fn a_match_stops_at_gap() {
    let mut c = Checks::new("a_match_stops_at_gap");
    let mut f = fixture();
    let before = snapshot(&f);
    let cases: [(&str, Vec<u32>, usize); 5] = [
        ("equal suffix [3,4] under another root", vec![9, 8, 3, 4], 2),
        ("full chain plus tail", vec![1, 2, 3, 4, 9], 4),
        ("diverge in block 2", vec![1, 2, 9, 4], 2),
        ("partial first block", vec![1], 0),
        ("empty", vec![], 0),
    ];
    for (name, tokens, want) in cases {
        let t = tokens.clone();
        let fc = f.clone();
        if let Some(got) = c.guard(name, move || fc.matched_tokens(&ns("a"), &t)) {
            c.eq(name, got, want);
        }
    }
    let fc = f.clone();
    if let Some(got) = c.guard("other tenant", move || {
        fc.matched_tokens(&ns("b"), &[1, 2, 3, 4])
    }) {
        c.eq("identical tokens, other tenant -> 0", got, 0);
    }
    let _ = c.guard(
        "read-only",
        std::panic::AssertUnwindSafe(|| f.matched_tokens(&ns("a"), &[1, 2, 3, 4])),
    );
    c.eq("query is read-only (nodes, clock)", snapshot(&f), before);
    f.nodes.remove(&key("a", &[1, 2]));
    let fc = f.clone();
    if let Some(got) = c.guard("gap", move || fc.matched_tokens(&ns("a"), &[1, 2, 3, 4])) {
        c.eq(
            "resident child behind a missing parent is unreachable",
            got,
            0,
        );
    }
    c.done();
}
/// Independent victim oracle: filter by the definition, then sort a key tuple.
fn victim_oracle(cache: &Cache, protected: &BTreeSet<Key>) -> Option<Key> {
    let mut legal: Vec<(&Key, &Entry)> = cache
        .nodes
        .iter()
        .filter(|(k, e)| e.refs == 0 && !protected.contains(*k))
        .filter(|(k, _)| {
            !cache.nodes.keys().any(|o| {
                o.ns == k.ns
                    && o.prefix.len() > k.prefix.len()
                    && o.prefix[..k.prefix.len()] == k.prefix[..]
            })
        })
        .collect();
    legal.sort_by_key(|(k, e)| match cache.policy {
        Policy::Lru => (0, e.last, (*k).clone()),
        Policy::Frequency => (e.hits, e.last, (*k).clone()),
    });
    legal.first().map(|(k, _)| (*k).clone())
}
#[test]
fn a_leaf_victims() {
    let mut c = Checks::new("a_leaf_victims");
    let f = fixture();
    // [1,2] is the oldest (last=1) but has a child, so the oldest LEAF [1,2,3,4] goes.
    if let Some(v) = c.guard("lru", || fixture().victim(&BTreeSet::new())) {
        c.eq(
            "oldest leaf, not oldest node",
            v.map(|k| k.prefix),
            Some(vec![1, 2, 3, 4]),
        );
    }
    let mut pinned = f.clone();
    pinned.nodes.get_mut(&key("a", &[1, 2, 3, 4])).unwrap().refs = 1;
    pinned.nodes.get_mut(&key("a", &[1, 2])).unwrap().refs = 1;
    let p2 = pinned.clone();
    if let Some(v) = c.guard("pinned", move || p2.victim(&BTreeSet::new())) {
        c.eq("pinned leaf skipped", v.map(|k| k.prefix), Some(vec![9, 8]));
    }
    let p3 = pinned.clone();
    if let Some(v) = c.guard("protected", move || {
        p3.victim(&BTreeSet::from([key("a", &[9, 8])]))
    }) {
        c.eq("protected skipped -> none", v, None);
    }
    let mut child_gone = f.clone();
    child_gone.nodes.remove(&key("a", &[1, 2, 3, 4]));
    if let Some(v) = c.guard("parent becomes leaf", move || {
        child_gone.victim(&BTreeSet::new())
    }) {
        c.eq(
            "parent is a candidate once its child is gone",
            v.map(|k| k.prefix),
            Some(vec![1, 2]),
        );
    }
    let mut freq = f.clone();
    freq.policy = Policy::Frequency;
    freq.nodes.get_mut(&key("a", &[9, 8])).unwrap().hits = 0;
    if let Some(v) = c.guard("frequency", move || freq.victim(&BTreeSet::new())) {
        c.eq(
            "frequency: fewest hits first",
            v.map(|k| k.prefix),
            Some(vec![9, 8]),
        );
    }
    let mut tie = Cache::new(4, 2, Policy::Lru);
    tie.clock = 3;
    for p in [[5u32, 5], [3, 3], [4, 4]] {
        tie.nodes.insert(
            key("a", &p),
            Entry {
                refs: 0,
                last: 2,
                hits: 1,
            },
        );
    }
    if let Some(v) = c.guard("tie", move || tie.victim(&BTreeSet::new())) {
        c.eq(
            "equal last -> smallest key",
            v.map(|k| k.prefix),
            Some(vec![3, 3]),
        );
    }
    // Cross-namespace: a key of tenant b with the same tokens is not a child.
    let mut x = Cache::new(4, 2, Policy::Lru);
    x.clock = 9;
    x.nodes.insert(
        key("a", &[1, 2]),
        Entry {
            refs: 0,
            last: 1,
            hits: 0,
        },
    );
    x.nodes.insert(
        key("b", &[1, 2, 3, 4]),
        Entry {
            refs: 0,
            last: 5,
            hits: 0,
        },
    );
    x.nodes.insert(
        key("b", &[1, 2]),
        Entry {
            refs: 0,
            last: 6,
            hits: 0,
        },
    );
    if let Some(v) = c.guard("cross namespace", move || x.victim(&BTreeSet::new())) {
        c.eq(
            "other tenant's longer key is not a descendant",
            v.map(|k| k.prefix),
            Some(vec![1, 2]),
        );
    }
    // Randomized: compare with the definition-based oracle.
    let mut rng = Lcg(5);
    let mut bad = Vec::new();
    for case in 0..300 {
        let mut cache = Cache::new(
            64,
            1,
            if case % 2 == 0 {
                Policy::Lru
            } else {
                Policy::Frequency
            },
        );
        cache.clock = 50;
        for _ in 0..rng.below(8) + 1 {
            let len = rng.below(3) as usize + 1;
            let tokens: Vec<u32> = (0..len).map(|_| rng.below(2) as u32).collect();
            let tenant = if rng.below(4) == 0 { "b" } else { "a" };
            for d in 1..=len {
                let e = cache
                    .nodes
                    .entry(key(tenant, &tokens[..d]))
                    .or_insert(Entry {
                        refs: 0,
                        last: 0,
                        hits: 0,
                    });
                e.last = rng.below(4);
                e.hits = rng.below(3);
                if rng.below(5) == 0 {
                    e.refs = 1;
                }
            }
        }
        let protected: BTreeSet<Key> = cache
            .nodes
            .keys()
            .filter(|_| rng.below(6) == 0)
            .cloned()
            .collect();
        let want = victim_oracle(&cache, &protected);
        let cc = cache.clone();
        let pp = protected.clone();
        match std::panic::catch_unwind(move || cc.victim(&pp)) {
            Ok(got) if got == want => {}
            Ok(got) => bad.push(format!("case {case}: expected {want:?} vs got {got:?}")),
            Err(_) => bad.push(format!("case {case}: panicked")),
        }
    }
    c.check(
        "300 random caches vs definition oracle",
        bad.is_empty(),
        || bad.iter().take(2).cloned().collect::<Vec<_>>().join("; "),
    );
    c.done();
}
#[test]
fn a_invariants_detect_corruption() {
    let mut c = Checks::new("a_invariants_detect_corruption");
    let run = |c: &mut Checks, name: &str, cache: Cache| c.guard(name, move || cache.validate());
    if let Some(r) = run(&mut c, "valid", fixture()) {
        c.eq("valid fixture", r, Ok(()));
    }
    let mut x = fixture();
    x.capacity = 2;
    if let Some(r) = run(&mut c, "capacity", x) {
        c.eq(
            "over capacity",
            r,
            Err(InvariantError::OverCapacity {
                nodes: 3,
                capacity: 2,
            }),
        );
    }
    let mut x = fixture();
    x.nodes.remove(&key("a", &[1, 2]));
    if let Some(r) = run(&mut c, "orphan", x) {
        c.eq(
            "missing parent",
            r,
            Err(InvariantError::Orphan(key("a", &[1, 2, 3, 4]))),
        );
    }
    let mut x = fixture();
    x.nodes.get_mut(&key("a", &[1, 2, 3, 4])).unwrap().refs = 1;
    if let Some(r) = run(&mut c, "pin", x) {
        c.eq(
            "child pinned above parent",
            r,
            Err(InvariantError::PinAboveParent(key("a", &[1, 2, 3, 4]))),
        );
    }
    let mut x = fixture();
    x.nodes.get_mut(&key("a", &[1, 2])).unwrap().refs = 2;
    x.nodes.get_mut(&key("a", &[1, 2, 3, 4])).unwrap().refs = 1;
    if let Some(r) = run(&mut c, "pins ok", x) {
        c.eq("parent pinned more than child is fine", r, Ok(()));
    }
    let mut x = fixture();
    x.nodes.insert(
        key("a", &[9, 8, 7]),
        Entry {
            refs: 0,
            last: 1,
            hits: 0,
        },
    );
    if let Some(r) = run(&mut c, "misaligned", x) {
        c.eq(
            "prefix not a multiple of block",
            r,
            Err(InvariantError::Misaligned(key("a", &[9, 8, 7]))),
        );
    }
    let mut x = fixture();
    x.nodes.get_mut(&key("a", &[9, 8])).unwrap().last = 6;
    if let Some(r) = run(&mut c, "future", x) {
        c.eq(
            "last from the future",
            r,
            Err(InvariantError::FutureClock(key("a", &[9, 8]))),
        );
    }
    // Two violations: the orphan at [1,2,3,4] precedes [9,8] in key order.
    let mut x = fixture();
    x.nodes.remove(&key("a", &[1, 2]));
    x.nodes.get_mut(&key("a", &[9, 8])).unwrap().last = 6;
    if let Some(r) = run(&mut c, "first", x) {
        c.eq(
            "first violation in key order",
            r,
            Err(InvariantError::Orphan(key("a", &[1, 2, 3, 4]))),
        );
    }
    let mut x = fixture();
    x.nodes.insert(
        key("b", &[1, 2, 3, 4]),
        Entry {
            refs: 0,
            last: 1,
            hits: 0,
        },
    );
    if let Some(r) = run(&mut c, "cross-tenant parent", x) {
        c.eq(
            "parent must be in the SAME namespace",
            r,
            Err(InvariantError::Orphan(key("b", &[1, 2, 3, 4]))),
        );
    }
    c.done();
}
#[test]
fn a_accounting_multiple_leases() {
    let mut c = Checks::new("a_accounting_multiple_leases");
    let mut f = fixture();
    f.nodes.get_mut(&key("a", &[1, 2])).unwrap().refs = 3;
    if let Some(s) = c.guard("stats", move || f.stats()) {
        c.eq("blocks", s.blocks, 3);
        c.eq("pinned blocks counted once", s.pinned_blocks, 1);
        c.eq("references counted per lease", s.references, 3);
        c.eq(
            "key payload 4 B per stored token: (2+4+2)*4",
            s.key_token_bytes,
            32,
        );
        c.eq("max depth in blocks", s.max_depth, 2);
    }
    if let Some(s) = c.guard("empty", || Cache::new(0, 4, Policy::Lru).stats()) {
        c.eq("empty cache", s, CacheStats::default());
    }
    c.done();
}

// ===================================================================== B
#[test]
fn b_transaction_rollback_and_leaf_order() {
    let mut c = Checks::new("b_transaction_rollback_and_leaf_order");
    let r = std::panic::catch_unwind(|| {
        let mut out = Vec::new();
        let mut x = Cache::new(2, 2, Policy::Lru);
        out.push(("insert fits", x.insert(&ns("a"), &[1, 2, 3, 4]), true));
        let lease = x.acquire(&ns("a"), &[1, 2]);
        // [1,2,3,4] is an unpinned leaf and can go, but then [1,2] is pinned: the
        // insert of 2 blocks must fail and leave no trace.
        let before = snapshot(&x);
        out.push((
            "blocked insert fails",
            x.insert(&ns("b"), &[5, 6, 7, 8]),
            false,
        ));
        let same = snapshot(&x) == before;
        out.push(("failed insert leaves nodes/metadata/clock", same, true));
        x.release(lease);
        out.push(("insert after release", x.insert(&ns("b"), &[5, 6]), true));
        out.push((
            "evicted the leaf, kept the parent",
            x.matched_tokens(&ns("a"), &[1, 2, 3, 4]) == 2,
            true,
        ));
        let before = snapshot(&x);
        out.push((
            "path longer than capacity fails fast",
            x.insert(&ns("a"), &[1, 2, 3, 4, 5, 6]),
            false,
        ));
        out.push(("fast failure leaves state", snapshot(&x) == before, true));
        let before = snapshot(&x);
        out.push(("empty path succeeds", x.insert(&ns("a"), &[1]), true));
        out.push((
            "empty path is a no-op (clock too)",
            snapshot(&x) == before,
            true,
        ));
        // Re-inserting an existing path does not count as a read.
        let mut y = Cache::new(4, 2, Policy::Lru);
        y.insert(&ns("a"), &[1, 2]);
        let e = y.nodes[&key("a", &[1, 2])].clone();
        y.insert(&ns("a"), &[1, 2, 3, 4]);
        let e2 = y.nodes[&key("a", &[1, 2])].clone();
        out.push(("existing key untouched by insert", e == e2, true));
        out.push((
            "new key stamped with the new clock",
            y.nodes[&key("a", &[1, 2, 3, 4])].last == y.clock && y.clock == 2,
            true,
        ));
        // The path's own ancestor is the LRU leaf: it must be protected, so the other
        // leaf [9,9] goes and the result is exactly the new path.
        let mut w = Cache::new(2, 2, Policy::Lru);
        w.insert(&ns("a"), &[1, 2]);
        w.insert(&ns("a"), &[9, 9]);
        out.push((
            "insert next to an older own ancestor",
            w.insert(&ns("a"), &[1, 2, 3, 4]),
            true,
        ));
        let kept: Vec<_> = w.nodes.keys().map(|k| k.prefix.clone()).collect();
        out.push((
            "own path protected from eviction",
            kept == vec![vec![1, 2], vec![1, 2, 3, 4]] && w.validate().is_ok(),
            true,
        ));
        // LRU order through acquire: [1,2] touched after [3,4] -> [3,4] is evicted.
        let mut z = Cache::new(2, 2, Policy::Lru);
        z.insert(&ns("a"), &[1, 2]);
        z.insert(&ns("a"), &[3, 4]);
        let l = z.acquire(&ns("a"), &[1, 2]);
        z.release(l);
        z.insert(&ns("a"), &[5, 6]);
        let kept: Vec<_> = z.nodes.keys().map(|k| k.prefix.clone()).collect();
        out.push((
            "LRU evicts the least recently ACQUIRED leaf",
            kept == vec![vec![1, 2], vec![5, 6]],
            true,
        ));
        out
    });
    match r {
        Ok(rows) => {
            for (name, got, want) in rows {
                c.eq(name, got, want);
            }
        }
        Err(_) => c.check("sequence", false, || {
            "panicked (TODO B1/B2/B3 or earlier)".into()
        }),
    }
    c.done();
}
#[test]
fn b_shared_leases() {
    let mut c = Checks::new("b_shared_leases");
    let r = std::panic::catch_unwind(|| {
        let mut out: Vec<(&str, String, String)> = Vec::new();
        let mut x = Cache::new(1, 2, Policy::Lru);
        x.insert(&ns("a"), &[1, 2]);
        let one = x.acquire(&ns("a"), &[1, 2]);
        let two = x.acquire(&ns("a"), &[1, 2]);
        let e = x.nodes[&key("a", &[1, 2])].clone();
        out.push((
            "two leases -> refs 2, hits 2, last = clock",
            format!("{:?}", (e.refs, e.hits, e.last)),
            format!("{:?}", (2, 2, x.clock)),
        ));
        x.release(one);
        out.push((
            "one release keeps the pin",
            format!("{}", x.insert(&ns("a"), &[9, 8])),
            "false".into(),
        ));
        x.release(two);
        out.push((
            "second release frees it",
            format!("{}", x.insert(&ns("a"), &[9, 8])),
            "true".into(),
        ));
        // Partial miss: shorter lease, nothing past the first miss is touched.
        let mut y = fixture();
        let clock = y.clock;
        let lease = y.acquire(&ns("a"), &[1, 2, 7, 7, 9, 8]);
        out.push((
            "partial hit pins only the chain",
            format!("{}", lease.len()),
            "1".into(),
        ));
        out.push((
            "[9,8] not touched (not contiguous)",
            format!("{:?}", y.nodes[&key("a", &[9, 8])]),
            format!(
                "{:?}",
                Entry {
                    refs: 0,
                    last: 3,
                    hits: 3
                }
            ),
        ));
        out.push((
            "clock +1 per acquire",
            format!("{}", y.clock),
            format!("{}", clock + 1),
        ));
        y.release(lease);
        // A resident child behind a missing parent is not pinned.
        let mut g = fixture();
        g.nodes.remove(&key("a", &[1, 2]));
        let gl = g.acquire(&ns("a"), &[1, 2, 3, 4]);
        out.push((
            "gap: nothing pinned past the first miss",
            format!("{:?}", (gl.len(), g.nodes[&key("a", &[1, 2, 3, 4])].refs)),
            "(0, 0)".into(),
        ));
        let miss = y.acquire(&ns("b"), &[1, 2]);
        out.push((
            "total miss -> empty lease, clock still advances",
            format!("{:?}", (miss.is_empty(), y.clock)),
            format!("{:?}", (true, clock + 2)),
        ));
        y.release(miss);
        out
    });
    match r {
        Ok(rows) => {
            for (name, got, want) in rows {
                c.eq(name, got, want);
            }
        }
        Err(_) => c.check("sequence", false, || {
            "panicked (TODO B1/B2/B3 or earlier)".into()
        }),
    }
    c.done();
}
#[test]
fn b_release_preserves_reuse() {
    let mut c = Checks::new("b_release_preserves_reuse");
    let r = std::panic::catch_unwind(|| {
        let mut x = fixture();
        let lease = x.acquire(&ns("a"), &[1, 2, 3, 4]);
        let refs: usize = x.nodes.values().map(|e| e.refs).sum();
        let clock = x.clock;
        x.release(lease);
        (
            refs,
            x.matched_tokens(&ns("a"), &[1, 2, 3, 4]),
            x.nodes.values().all(|e| e.refs == 0),
            x.clock == clock,
            x.nodes.len(),
        )
    });
    match r {
        Ok((refs, matched, zero, clock_same, len)) => {
            c.eq("lease pins two blocks", refs, 2);
            c.eq("released blocks stay cached", matched, 4);
            c.eq("all refs back to zero", zero, true);
            c.eq("release does not move the clock", clock_same, true);
            c.eq("nothing removed", len, 3);
        }
        Err(_) => c.check("sequence", false, || "panicked".into()),
    }
    let r = std::panic::catch_unwind(|| {
        let mut x = fixture();
        let lease = x.acquire(&ns("a"), &[1, 2]);
        x.nodes.get_mut(&key("a", &[1, 2])).unwrap().refs = 0; // corrupt on purpose
        x.release(lease);
    });
    c.check("refcount underflow panics", r.is_err(), || {
        "release accepted a lease on a zero-ref entry".into()
    });
    let r = std::panic::catch_unwind(|| {
        let mut e = Cache::new(0, 2, Policy::Lru);
        (e.insert(&ns("a"), &[]), e.insert(&ns("a"), &[1, 2]))
    });
    match r {
        Ok(v) => c.eq(
            "capacity 0: empty insert ok, one block fails",
            v,
            (true, false),
        ),
        Err(_) => c.check("capacity 0", false, || "panicked".into()),
    }
    c.done();
}
#[test]
fn b_namespace_invalidation() {
    let mut c = Checks::new("b_namespace_invalidation");
    let r = std::panic::catch_unwind(|| {
        let mut out: Vec<(&str, String, String)> = Vec::new();
        let mut x = fixture();
        x.capacity = 6;
        x.insert(&ns("b"), &[1, 2, 3, 4]); // identical tokens, other tenant
        let lease = x.acquire(&ns("a"), &[1, 2]);
        let before = snapshot(&x);
        out.push((
            "pinned namespace -> None",
            format!("{:?}", x.invalidate(&ns("a"))),
            "None".into(),
        ));
        out.push((
            "blocked invalidation changes nothing (clock too)",
            format!("{}", snapshot(&x) == before),
            "true".into(),
        ));
        x.release(lease);
        let clock = x.clock;
        out.push((
            "removes exactly tenant a's 3 nodes",
            format!("{:?}", x.invalidate(&ns("a"))),
            "Some(3)".into(),
        ));
        out.push((
            "clock +1 on success",
            format!("{}", x.clock),
            format!("{}", clock + 1),
        ));
        out.push((
            "tenant b's identical tokens survive",
            format!("{}", x.matched_tokens(&ns("b"), &[1, 2, 3, 4])),
            "4".into(),
        ));
        out.push((
            "only b remains",
            format!("{}", x.nodes.keys().all(|k| k.ns == ns("b"))),
            "true".into(),
        ));
        out.push((
            "empty namespace -> Some(0)",
            format!("{:?}", x.invalidate(&ns("zzz"))),
            "Some(0)".into(),
        ));
        out
    });
    match r {
        Ok(rows) => {
            for (name, got, want) in rows {
                c.eq(name, got, want);
            }
        }
        Err(_) => c.check("sequence", false, || "panicked".into()),
    }
    c.done();
}
#[test]
fn b_resize_transaction_with_pins() {
    let mut c = Checks::new("b_resize_transaction_with_pins");
    let r = std::panic::catch_unwind(|| {
        let mut out: Vec<(&str, String, String)> = Vec::new();
        let mut x = fixture();
        let lease = x.acquire(&ns("a"), &[1, 2, 3, 4]);
        let before = snapshot(&x);
        out.push((
            "shrink blocked by pins",
            format!("{}", x.resize(1)),
            "false".into(),
        ));
        out.push((
            "failed shrink: nodes, capacity, clock unchanged",
            format!("{}", snapshot(&x) == before),
            "true".into(),
        ));
        let nodes = x.nodes.clone();
        let clock = x.clock;
        out.push(("grow", format!("{}", x.resize(8)), "true".into()));
        out.push((
            "grow keeps nodes and clock",
            format!(
                "{}",
                x.nodes == nodes && x.clock == clock && x.capacity == 8
            ),
            "true".into(),
        ));
        out.push((
            "shrink to 2 evicts the unpinned leaf only",
            format!("{}", x.resize(2)),
            "true".into(),
        ));
        out.push((
            "pinned chain kept",
            format!(
                "{:?}",
                x.nodes.keys().map(|k| k.prefix.clone()).collect::<Vec<_>>()
            ),
            format!("{:?}", vec![vec![1u32, 2], vec![1, 2, 3, 4]]),
        ));
        out.push((
            "successful shrink advances clock",
            format!("{}", x.clock),
            format!("{}", clock + 1),
        ));
        x.release(lease);
        out.push((
            "shrink to 0 after release",
            format!("{}", x.resize(0)),
            "true".into(),
        ));
        out.push((
            "empty, capacity 0",
            format!("{:?}", (x.nodes.len(), x.capacity)),
            "(0, 0)".into(),
        ));
        out
    });
    match r {
        Ok(rows) => {
            for (name, got, want) in rows {
                c.eq(name, got, want);
            }
        }
        Err(_) => c.check("sequence", false, || "panicked".into()),
    }
    c.done();
}

// ===================================================================== C
#[test]
fn c_admission_lru_history() {
    let mut c = Checks::new("c_admission_lru_history");
    let r = std::panic::catch_unwind(|| {
        let mut out: Vec<(&str, usize, usize)> = Vec::new();
        let tw = AdmissionPolicy::TwoHit;
        let mut h = Admission::new(8);
        out.push((
            "first sight admits nothing",
            h.observe(&ns("a"), &[1, 2, 3, 4], 2, tw),
            0,
        ));
        out.push((
            "shared first block seen twice -> 2 tokens",
            h.observe(&ns("a"), &[1, 2, 9, 9], 2, tw),
            2,
        ));
        out.push((
            "other tenant has its own counts",
            h.observe(&ns("b"), &[1, 2, 9, 9], 2, tw),
            0,
        ));
        out.push((
            "third sight of the full path [1,2,3,4]",
            h.observe(&ns("a"), &[1, 2, 3, 4, 5], 2, tw),
            4,
        ));
        out.push((
            "Always admits every full block",
            h.observe(&ns("a"), &[7, 7, 7], 2, AdmissionPolicy::Always),
            2,
        ));
        out.push((
            "capacity 0 admits 0 under TwoHit",
            Admission::new(0).observe(&ns("a"), &[1, 2], 2, tw),
            0,
        ));
        // LRU, not FIFO: capacity 2, block 2.
        //   t1 [1,2]  t2 [3,4]  t3 [1,2] (touch)  t4 [5,6] -> forget [3,4] (last=2)
        //   t5 [3,4] -> count restarts at 1 -> admits 0. FIFO would have forgotten
        //   [1,2] at t4 and admitted [3,4] at t5.
        let mut l = Admission::new(2);
        l.observe(&ns("a"), &[1, 2], 2, tw);
        l.observe(&ns("a"), &[3, 4], 2, tw);
        out.push((
            "re-observed [1,2] admitted",
            l.observe(&ns("a"), &[1, 2], 2, tw),
            2,
        ));
        l.observe(&ns("a"), &[5, 6], 2, tw);
        out.push(("history bounded", l.history.len(), 2));
        out.push((
            "[3,4] was the least recently observed and is forgotten",
            l.history.contains_key(&key("a", &[3, 4])) as usize,
            0,
        ));
        out.push((
            "[3,4] restarts at count 1",
            l.observe(&ns("a"), &[3, 4], 2, tw),
            0,
        ));
        out.push((
            "[1,2] (last=3) evicted at t5, before [5,6] (last=4)",
            l.history.contains_key(&key("a", &[1, 2])) as usize,
            0,
        ));
        // A request longer than the history: its own deeper keys sort after the root
        // at equal stamps, so the shortest prefixes are forgotten first.
        let mut s = Admission::new(1);
        s.observe(&ns("a"), &[1, 2, 3, 4], 2, tw);
        out.push((
            "one slot keeps the larger key at equal stamps",
            s.history.contains_key(&key("a", &[1, 2, 3, 4])) as usize,
            1,
        ));
        out.push((
            "chain broken at the forgotten root -> 0",
            s.observe(&ns("a"), &[1, 2, 3, 4], 2, tw),
            0,
        ));
        out.push((
            "counts are per request: stamp advanced twice",
            s.clock as usize,
            2,
        ));
        out
    });
    match r {
        Ok(rows) => {
            for (name, got, want) in rows {
                c.eq(name, got, want);
            }
        }
        Err(_) => c.check("sequence", false, || "panicked (TODO C1)".into()),
    }
    c.done();
}
/// Brute force: the largest r <= matched with r < total and (Vllm) r % block == 0.
fn rule_oracle(matched: usize, total: usize, block: usize, rule: FullHitRule) -> usize {
    (0..=matched)
        .rev()
        .find(|&r| r < total.max(1) && (rule == FullHitRule::Sglang || r % block == 0))
        .unwrap_or(0)
}
#[test]
fn c_full_hit_rule() {
    let mut c = Checks::new("c_full_hit_rule");
    let ex = c.guard("example", || {
        (
            reusable_tokens(32, 32, 16, FullHitRule::Sglang),
            reusable_tokens(32, 32, 16, FullHitRule::Vllm),
            reusable_tokens(16, 33, 16, FullHitRule::Vllm),
            reusable_tokens(0, 0, 4, FullHitRule::Sglang),
        )
    });
    if let Some((s, v, v2, z)) = ex {
        c.eq("32 cached of 32, SGLang recomputes 1", s, 31);
        c.eq(
            "32 cached of 32, B=16, vLLM recomputes a whole block",
            v,
            16,
        );
        c.eq("partial match below the cap is kept", v2, 16);
        c.eq("empty prompt", z, 0);
    }
    let mut bad = Vec::new();
    for rule in [FullHitRule::Sglang, FullHitRule::Vllm] {
        for block in 1..=6 {
            for total in 0..=40 {
                for matched in 0..=total {
                    let want = rule_oracle(matched, total, block, rule);
                    match std::panic::catch_unwind(move || reusable_tokens(matched, total, block, rule)) {
                        Ok(got) if got == want => {}
                        Ok(got) => bad.push(format!("{rule:?} m={matched} t={total} B={block}: expected {want} vs got {got}")),
                        Err(_) => bad.push(format!("{rule:?} m={matched} t={total} B={block}: panicked")),
                    }
                }
            }
        }
    }
    c.check(
        "exhaustive vs brute force (both rules)",
        bad.is_empty(),
        || {
            format!(
                "{} wrong, e.g. {}",
                bad.len(),
                bad.iter().take(2).cloned().collect::<Vec<_>>().join("; ")
            )
        },
    );
    if let Some(cost) = c.guard("cost of a full hit", || {
        let m = CostModel::per_token(1.0);
        let r = reusable_tokens(8, 8, 4, FullHitRule::Vllm);
        m.prefill_ms(r, 8 - r)
    }) {
        c.check("a full hit is never free", cost > 0.0, || {
            format!("prefill {cost}")
        });
    }
    c.done();
}
/// Independent router oracle: sort candidate tuples instead of min_by_key closures.
fn route_oracle(policy: RoutePolicy, next: usize, views: &[ReplicaView], prompt: usize) -> usize {
    let by_load = || {
        let mut v: Vec<(usize, usize)> =
            views.iter().enumerate().map(|(i, x)| (x.load, i)).collect();
        v.sort();
        v[0].1
    };
    let by_match = || {
        let mut v: Vec<(i64, usize, usize)> = views
            .iter()
            .enumerate()
            .map(|(i, x)| (-(x.matched as i64), x.load, i))
            .collect();
        v.sort();
        v[0].2
    };
    match policy {
        RoutePolicy::RoundRobin => next % views.len(),
        RoutePolicy::LeastLoad => by_load(),
        RoutePolicy::PrefixAffinity => by_match(),
        RoutePolicy::Hybrid {
            abs_threshold,
            rel_threshold,
            cache_threshold,
        } => {
            let loads: Vec<usize> = views.iter().map(|x| x.load).collect();
            let (hi, lo) = (*loads.iter().max().unwrap(), *loads.iter().min().unwrap());
            let imbalanced = hi - lo > abs_threshold && (hi as f64) > rel_threshold * (lo as f64);
            if imbalanced {
                return by_load();
            }
            let best = by_match();
            if prompt > 0 && (views[best].matched as f64) > cache_threshold * prompt as f64 {
                best
            } else {
                by_load()
            }
        }
    }
}
fn hybrid() -> RoutePolicy {
    RoutePolicy::Hybrid {
        abs_threshold: 2,
        rel_threshold: 1.5,
        cache_threshold: 0.3,
    }
}
#[test]
fn c_routing_policies() {
    let mut c = Checks::new("c_routing_policies");
    let v = |pairs: &[(usize, usize)]| {
        pairs
            .iter()
            .map(|&(load, matched)| ReplicaView { load, matched })
            .collect::<Vec<_>>()
    };
    let cases: Vec<(&str, RoutePolicy, Vec<ReplicaView>, usize, usize)> = vec![
        (
            "least load",
            RoutePolicy::LeastLoad,
            v(&[(3, 0), (1, 90), (1, 0)]),
            100,
            1,
        ),
        (
            "affinity: longest match",
            RoutePolicy::PrefixAffinity,
            v(&[(9, 80), (0, 10), (0, 0)]),
            100,
            0,
        ),
        (
            "affinity tie -> lower load",
            RoutePolicy::PrefixAffinity,
            v(&[(4, 50), (2, 50), (2, 0)]),
            100,
            1,
        ),
        (
            "affinity all-miss -> least load",
            RoutePolicy::PrefixAffinity,
            v(&[(4, 0), (2, 0), (3, 0)]),
            100,
            1,
        ),
        // hi-lo = 9 > 2 and 9 > 1.5*0: imbalanced -> least load even with an 80% match.
        (
            "hybrid imbalanced -> balance",
            hybrid(),
            v(&[(9, 80), (0, 0), (1, 0)]),
            100,
            1,
        ),
        // hi-lo = 2, not > 2: balanced; 80/100 > 0.3 -> follow the prefix.
        (
            "hybrid balanced, big match -> prefix",
            hybrid(),
            v(&[(2, 80), (0, 0), (1, 0)]),
            100,
            0,
        ),
        // abs gap 3 > 2 but 6 <= 1.5*4: relative test says balanced.
        (
            "hybrid relative threshold",
            hybrid(),
            v(&[(6, 80), (4, 0), (5, 0)]),
            100,
            0,
        ),
        // 30/100 is not > 0.3: least load.
        (
            "hybrid small match -> least load",
            hybrid(),
            v(&[(1, 30), (0, 0), (1, 0)]),
            100,
            1,
        ),
        (
            "hybrid empty prompt -> least load",
            hybrid(),
            v(&[(1, 0), (0, 0), (1, 0)]),
            0,
            1,
        ),
    ];
    for (name, policy, views, prompt, want) in cases {
        let vv = views.clone();
        if let Some(got) = c.guard(name, move || Router::new(policy).route(&vv, prompt)) {
            c.eq(name, got, want);
        }
    }
    if let Some(seq) = c.guard("round robin", || {
        let mut r = Router::new(RoutePolicy::RoundRobin);
        let views = vec![
            ReplicaView {
                load: 0,
                matched: 0
            };
            3
        ];
        (0..5).map(|_| r.route(&views, 10)).collect::<Vec<_>>()
    }) {
        c.eq(
            "round robin cycles and ignores state",
            seq,
            vec![0, 1, 2, 0, 1],
        );
    }
    let mut rng = Lcg(11);
    let mut bad = Vec::new();
    for case in 0..2000 {
        let k = rng.below(4) as usize + 1;
        let views: Vec<ReplicaView> = (0..k)
            .map(|_| ReplicaView {
                load: rng.below(6) as usize,
                matched: rng.below(3) as usize * 16,
            })
            .collect();
        let prompt = rng.below(4) as usize * 16;
        let policy = match case % 4 {
            0 => RoutePolicy::LeastLoad,
            1 => RoutePolicy::PrefixAffinity,
            2 => hybrid(),
            _ => RoutePolicy::Hybrid {
                abs_threshold: rng.below(3) as usize,
                rel_threshold: 1.0 + rng.below(3) as f64 * 0.5,
                cache_threshold: rng.below(4) as f64 * 0.25,
            },
        };
        let want = route_oracle(policy, 0, &views, prompt);
        let vv = views.clone();
        match std::panic::catch_unwind(move || Router::new(policy).route(&vv, prompt)) {
            Ok(got) if got == want => {}
            Ok(got) => bad.push(format!(
                "{policy:?} {views:?} prompt {prompt}: expected {want} vs got {got}"
            )),
            Err(_) => bad.push(format!("{policy:?}: panicked")),
        }
    }
    c.check("2000 random views vs oracle", bad.is_empty(), || {
        format!(
            "{} wrong, e.g. {}",
            bad.len(),
            bad.first().cloned().unwrap_or_default()
        )
    });
    c.done();
}

fn rq(id: usize, arrival: f64, tenant: &str, tokens: Vec<u32>, decode_ms: f64) -> Request {
    Request {
        id,
        arrival,
        ns: ns(tenant),
        prompt_tokens: tokens.len(),
        tokens,
        decode_ms,
    }
}
fn cfg(
    replicas: usize,
    slots: usize,
    capacity: usize,
    route: RoutePolicy,
    rule: FullHitRule,
) -> ClusterConfig {
    ClusterConfig {
        replicas,
        slots,
        capacity_blocks: capacity,
        block: 2,
        unit_tokens: 1,
        eviction: Policy::Lru,
        admission: AdmissionPolicy::Always,
        history_capacity: 64,
        rule,
        cost: CostModel::per_token(1.0),
        route,
    }
}
fn by_id(mut rows: Vec<ResultRow>) -> Vec<ResultRow> {
    rows.sort_by_key(|r| r.id);
    rows
}
#[test]
fn c_replay_hand_timeline() {
    let mut c = Checks::new("c_replay_hand_timeline");
    // B=2, one replica, 2 slots, 1 ms per uncached token, decode 1 ms.
    // t=0: r0 and r1 ([1,2]) both start on an empty cache: matched 0, prefill 2 ->
    //      first token 2, finish 3. Neither can reuse the other's in-flight KV.
    // t=3: both finish; [1,2] is published.
    // t=4: r2 ([1,2]) matches 2 tokens. SGLang reuses min(2, 2-1) = 1 -> prefill 1 ->
    //      first token 5, finish 6. vLLM reuses floor((2-1)/2)*2 = 0 -> prefill 2 ->
    //      first token 6, finish 7. r2 pins [1,2] while running: peak pins 1.
    let trace = || {
        vec![
            rq(0, 0., "a", vec![1, 2], 1.),
            rq(1, 0., "a", vec![1, 2], 1.),
            rq(2, 4., "a", vec![1, 2], 1.),
        ]
    };
    for (rule, reused, first, finish) in [
        (
            FullHitRule::Sglang,
            vec![0, 0, 1],
            vec![2., 2., 5.],
            vec![3., 3., 6.],
        ),
        (
            FullHitRule::Vllm,
            vec![0, 0, 0],
            vec![2., 2., 6.],
            vec![3., 3., 7.],
        ),
    ] {
        let config = cfg(1, 2, 2, RoutePolicy::LeastLoad, rule);
        let Some(out) = c.guard(&format!("{rule:?} replay"), move || {
            replay(trace(), &config)
        }) else {
            continue;
        };
        let rows = by_id(out.rows);
        c.eq(
            &format!("{rule:?} matched (in-flight miss is not a hit)"),
            rows.iter().map(|r| r.matched).collect::<Vec<_>>(),
            vec![0, 0, 2],
        );
        c.eq(
            &format!("{rule:?} reused after the full-hit rule"),
            rows.iter().map(|r| r.reused).collect::<Vec<_>>(),
            reused,
        );
        c.eq(
            &format!("{rule:?} first token"),
            rows.iter().map(|r| r.first_token).collect::<Vec<_>>(),
            first,
        );
        c.eq(
            &format!("{rule:?} finish"),
            rows.iter().map(|r| r.finish).collect::<Vec<_>>(),
            finish,
        );
        c.eq(&format!("{rule:?} peak pins"), out.peak_pins, 1);
    }
    // Queueing with 1 slot: r1 waits for r0. Nonlinear cost:
    // T(p,n) = max(3, 1 + 0.5 n + 0.01 n (p + n/2) + 0.1 p).
    // r0: 8 tokens cold: 1 + 4 + 0.01*8*4 = 5.32 -> first 5.32, finish 6.32.
    // r1 arrives at 1, waits until 6.32 (r0's [1..8] published at 6.32), matched 8 of 10,
    //   SGLang reuses 8, n=2: 1 + 1 + 0.01*2*9 + 0.8 = 2.98 -> floor 3 -> first 9.32.
    let mut config = cfg(1, 1, 8, RoutePolicy::LeastLoad, FullHitRule::Sglang);
    config.cost = CostModel {
        floor_ms: 3.0,
        a_ms: 1.0,
        b_ms: 0.5,
        c_ms: 0.01,
        d_ms: 0.1,
    };
    let trace2 = vec![
        rq(0, 0., "a", (1..=8).collect(), 1.),
        rq(1, 1., "a", (1..=10).collect(), 1.),
    ];
    if let Some(out) = c.guard("cost model replay", move || replay(trace2, &config)) {
        let rows = by_id(out.rows);
        c.close("cold 8 tokens", rows[0].first_token, 5.32, 1e-9);
        c.close("queued until the slot frees", rows[1].start, 6.32, 1e-9);
        c.eq(
            "waited request sees the published prefix",
            rows[1].reused,
            8,
        );
        c.close(
            "floor applies to the short suffix",
            rows[1].first_token,
            9.32,
            1e-9,
        );
    }
    c.done();
}
#[test]
fn c_replay_routing_oracle() {
    let mut c = Checks::new("c_replay_routing_oracle");
    // Two replicas, 1 slot each, B=2, 1 ms/token, decode 1 ms.
    // t=0: r0 [1,2,3,4] and r1 [5,6,7,8]; t=5 both published (r0 on replica 0, r1 on 1
    // under every policy: loads (0,0) then (1,0)). t=10: r2 [5,6,7,8,9,9], r3 [1,2,3,4,7,7].
    // LeastLoad/RoundRobin: r2 -> 0, r3 -> 1, both miss (prefill 6, first token 16).
    // PrefixAffinity: r2 -> 1, r3 -> 0, both reuse 4 (prefill 2, first token 12).
    let trace = || {
        vec![
            rq(0, 0., "a", vec![1, 2, 3, 4], 1.),
            rq(1, 0., "a", vec![5, 6, 7, 8], 1.),
            rq(2, 10., "a", vec![5, 6, 7, 8, 9, 9], 1.),
            rq(3, 10., "a", vec![1, 2, 3, 4, 7, 7], 1.),
        ]
    };
    for (name, route, replicas, reused, first) in [
        (
            "least load",
            RoutePolicy::LeastLoad,
            vec![0, 1, 0, 1],
            vec![0, 0, 0, 0],
            vec![4., 4., 16., 16.],
        ),
        (
            "round robin",
            RoutePolicy::RoundRobin,
            vec![0, 1, 0, 1],
            vec![0, 0, 0, 0],
            vec![4., 4., 16., 16.],
        ),
        (
            "affinity",
            RoutePolicy::PrefixAffinity,
            vec![0, 1, 1, 0],
            vec![0, 0, 4, 4],
            vec![4., 4., 12., 12.],
        ),
    ] {
        let config = cfg(2, 1, 8, route, FullHitRule::Sglang);
        if let Some(out) = c.guard(name, move || replay(trace(), &config)) {
            let rows = by_id(out.rows);
            c.eq(
                &format!("{name}: replicas"),
                rows.iter().map(|r| r.replica).collect::<Vec<_>>(),
                replicas,
            );
            c.eq(
                &format!("{name}: reused"),
                rows.iter().map(|r| r.reused).collect::<Vec<_>>(),
                reused,
            );
            c.eq(
                &format!("{name}: first token"),
                rows.iter().map(|r| r.first_token).collect::<Vec<_>>(),
                first,
            );
        }
    }
    // Hot prefix and a load threshold. r0 [1,2,3,4] lands on replica 0 (all-miss ->
    // least load) and is published at t=5. At t=10, r1..r4 = [1,2,3,4,k,k], decode 100:
    //   Hybrid(abs 1, rel 1.0, cache 0.3), loads seen at arrival (waiting + running):
    //   r1 (0,0) balanced, 4/6 > 0.3 -> 0;  r2 (1,0): gap 1 not > 1 -> 0;
    //   r3 (2,0): gap 2 > 1 and 2 > 0 -> imbalanced -> least load 1;  r4 (2,1): -> 0.
    //   Pure affinity ignores load: all on 0. LeastLoad: 0,0,1,0,1.
    let hot = || {
        let mut v = vec![rq(0, 0., "a", vec![1, 2, 3, 4], 1.)];
        for k in 1..=4u32 {
            v.push(rq(
                k as usize,
                10.,
                "a",
                vec![1, 2, 3, 4, 10 + k, 10 + k],
                100.,
            ));
        }
        v
    };
    let hyb = RoutePolicy::Hybrid {
        abs_threshold: 1,
        rel_threshold: 1.0,
        cache_threshold: 0.3,
    };
    for (name, route, replicas) in [
        ("hot hybrid", hyb, vec![0, 0, 0, 1, 0]),
        (
            "hot affinity",
            RoutePolicy::PrefixAffinity,
            vec![0, 0, 0, 0, 0],
        ),
        (
            "hot least load",
            RoutePolicy::LeastLoad,
            vec![0, 0, 1, 0, 1],
        ),
    ] {
        let config = cfg(2, 1, 16, route, FullHitRule::Sglang);
        if let Some(out) = c.guard(name, move || replay(hot(), &config)) {
            let rows = by_id(out.rows);
            c.eq(
                &format!("{name}: replicas"),
                rows.iter().map(|r| r.replica).collect::<Vec<_>>(),
                replicas,
            );
        }
    }
    c.done();
}
/// Common full-block prefix length (in tokens) of two token lists.
fn common_blocks(a: &[u32], b: &[u32], block: usize) -> usize {
    let mut n = 0;
    while n + block <= a.len().min(b.len()) && a[n..n + block] == b[n..n + block] {
        n += block;
    }
    n
}
#[test]
fn c_replay_conservation_and_isolation() {
    let mut c = Checks::new("c_replay_conservation_and_isolation");
    let empty = cfg(2, 1, 0, RoutePolicy::LeastLoad, FullHitRule::Sglang);
    if let Some(out) = c.guard("empty trace", move || replay(vec![], &empty)) {
        c.eq("empty trace -> no rows", out.rows.len(), 0);
    }
    let mut rng = Lcg(42);
    // Tenants a and b draw from the SAME token pool, so any cross-tenant reuse shows up.
    let systems: Vec<Vec<u32>> = (0..3)
        .map(|s| (0..4).map(|j| 100 * s + j).collect())
        .collect();
    let mut trace = Vec::new();
    for i in 0..90usize {
        let mut t = systems[rng.below(3) as usize].clone();
        if rng.below(3) > 0 {
            t.extend([7, 7]);
        }
        t.extend((0..rng.below(5)).map(|_| rng.below(3) as u32));
        let tenant = ["a", "b", "c"][rng.below(3) as usize];
        trace.push(rq(
            i,
            (i / 4) as f64 * 1.5 + rng.below(2) as f64 * 0.5,
            tenant,
            t,
            0.5 + rng.below(4) as f64,
        ));
    }
    let mut problems: Vec<String> = Vec::new();
    let mut runs = 0;
    for route in [
        RoutePolicy::RoundRobin,
        RoutePolicy::LeastLoad,
        RoutePolicy::PrefixAffinity,
        hybrid(),
    ] {
        for rule in [FullHitRule::Sglang, FullHitRule::Vllm] {
            for admission in [AdmissionPolicy::Always, AdmissionPolicy::TwoHit] {
                let mut config = cfg(3, 2, 5, route, rule);
                config.admission = admission;
                config.history_capacity = 16;
                config.cost = CostModel {
                    floor_ms: 0.4,
                    a_ms: 0.1,
                    b_ms: 0.25,
                    c_ms: 0.002,
                    d_ms: 0.01,
                };
                let tag = format!("{route:?}/{rule:?}/{admission:?}");
                let (t, cf) = (trace.clone(), config.clone());
                let Some(out) = c.guard(&tag, move || replay(t, &cf)) else {
                    continue;
                };
                runs += 1;
                let mut p = |ok: bool, msg: String| {
                    if !ok {
                        problems.push(format!("{tag}: {msg}"));
                    }
                };
                let ids: BTreeSet<usize> = out.rows.iter().map(|r| r.id).collect();
                p(
                    out.rows.len() == trace.len() && ids.len() == trace.len(),
                    format!("{} rows, {} ids", out.rows.len(), ids.len()),
                );
                p(
                    out.caches.len() == 3,
                    format!("{} caches", out.caches.len()),
                );
                for (i, cache) in out.caches.iter().enumerate() {
                    p(
                        cache.validate().is_ok(),
                        format!("replica {i} invalid: {:?}", cache.validate()),
                    );
                    p(
                        cache.nodes.values().all(|e| e.refs == 0),
                        format!("replica {i}: lease leaked"),
                    );
                    p(
                        cache.capacity == 5 && cache.block == 2,
                        format!("replica {i} geometry"),
                    );
                }
                let reqs: BTreeMap<usize, &Request> = trace.iter().map(|r| (r.id, r)).collect();
                for row in &out.rows {
                    let req = reqs[&row.id];
                    p(
                        row.tenant == req.ns.tenant && row.total == req.prompt_tokens,
                        format!("row {} fields", row.id),
                    );
                    p(
                        row.start >= row.arrival,
                        format!("row {} starts before arrival", row.id),
                    );
                    p(
                        row.reused == rule_oracle(row.matched, row.total, 2, rule),
                        format!(
                            "row {} reused {} for matched {}",
                            row.id, row.reused, row.matched
                        ),
                    );
                    let prefill = config.cost.prefill_ms(row.reused, row.total - row.reused);
                    p(
                        (row.first_token - row.start - prefill).abs() < 1e-9,
                        format!(
                            "row {} prefill {} vs {}",
                            row.id,
                            row.first_token - row.start,
                            prefill
                        ),
                    );
                    p(
                        (row.finish - row.first_token - req.decode_ms).abs() < 1e-9,
                        format!("row {} decode", row.id),
                    );
                    // Isolation + no in-flight reuse: the match can only come from
                    // same-tenant requests that FINISHED on the same replica before start.
                    let bound = out
                        .rows
                        .iter()
                        .filter(|o| {
                            o.replica == row.replica
                                && o.tenant == row.tenant
                                && o.finish <= row.start
                                && o.id != row.id
                        })
                        .map(|o| common_blocks(&reqs[&o.id].tokens, &req.tokens, 2))
                        .max()
                        .unwrap_or(0);
                    p(
                        row.matched <= bound,
                        format!(
                            "row {} (tenant {}) matched {} > same-tenant bound {}",
                            row.id, row.tenant, row.matched, bound
                        ),
                    );
                }
                for r in 0..3 {
                    let mut mine: Vec<&ResultRow> =
                        out.rows.iter().filter(|x| x.replica == r).collect();
                    // slots: at every start time at most 2 requests are running.
                    for x in &mine {
                        let running = mine
                            .iter()
                            .filter(|y| y.start <= x.start && x.start < y.finish)
                            .count();
                        p(
                            running <= 2,
                            format!("replica {r}: {running} running at {}", x.start),
                        );
                    }
                    // FIFO per replica: arrival order implies start order.
                    mine.sort_by(|a, b| a.arrival.total_cmp(&b.arrival).then(a.id.cmp(&b.id)));
                    p(
                        mine.windows(2).all(|w| w[0].start <= w[1].start),
                        format!("replica {r}: not FIFO"),
                    );
                }
                if matches!(route, RoutePolicy::RoundRobin) {
                    let mut sorted: Vec<&Request> = trace.iter().collect();
                    sorted.sort_by(|a, b| a.arrival.total_cmp(&b.arrival).then(a.id.cmp(&b.id)));
                    let want: BTreeMap<usize, usize> = sorted
                        .iter()
                        .enumerate()
                        .map(|(i, r)| (r.id, i % 3))
                        .collect();
                    p(
                        out.rows.iter().all(|r| want[&r.id] == r.replica),
                        "round robin order over (arrival, id)".into(),
                    );
                }
            }
        }
    }
    c.eq("all 16 configurations ran", runs, 16);
    c.check(
        "conservation, timing, FIFO, slots, isolation",
        problems.is_empty(),
        || {
            format!(
                "{} problems, e.g. {}",
                problems.len(),
                problems
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" | ")
            )
        },
    );
    c.done();
}
#[test]
fn c_metrics_and_duplication() {
    let mut c = Checks::new("c_metrics_and_duplication");
    let row =
        |id, tenant: &str, replica, arrival, start, first, finish, reused, total, inserted| {
            ResultRow {
                id,
                tenant: tenant.into(),
                replica,
                arrival,
                start,
                first_token: first,
                finish,
                matched: reused,
                reused,
                total,
                inserted,
            }
        };
    let rows = vec![
        row(0, "a", 0, 0., 1., 2., 3., 2, 2, true),
        row(1, "b", 1, 0., 2., 4., 5., 0, 8, false),
        row(2, "a", 0, 1., 1., 7., 9., 5, 10, true),
    ];
    // hit = (2+0+5)/(2+8+10) = 0.35; queue = (1+2+0)/3 = 1; ttft = {2,4,6}: mean 4,
    // p95 = 3rd of 3 = 6; recomputed = 13; busy replica0 = 2+8 = 10, replica1 = 3,
    // replica2 = 0 -> mean 13/3, imbalance = 10/(13/3) = 30/13.
    if let Some(m) = c.guard("all", || summarize(&rows, None, 3)) {
        c.eq("requests", m.requests, 3);
        c.close(
            "token-weighted hit, not mean per request",
            m.hit_ratio,
            0.35,
            1e-12,
        );
        c.close("mean queue start-arrival", m.mean_queue, 1.0, 1e-12);
        c.close("mean ttft first_token-arrival", m.mean_ttft, 4.0, 1e-12);
        c.close("p95 ttft nearest rank", m.p95_ttft, 6.0, 1e-12);
        c.eq("recomputed tokens", m.recomputed, 13);
        c.eq("insert failures", m.insert_failures, 1);
        c.close(
            "imbalance over all 3 replicas (idle one included)",
            m.imbalance,
            30.0 / 13.0,
            1e-12,
        );
    }
    let rows2 = rows.clone();
    if let Some(m) = c.guard("tenant a", move || summarize(&rows2, Some("a"), 3)) {
        c.close("tenant a hit", m.hit_ratio, 7.0 / 12.0, 1e-12);
        c.eq("tenant a requests", m.requests, 2);
    }
    let rows3 = rows.clone();
    if let Some(m) = c.guard("missing", move || summarize(&rows3, Some("zzz"), 3)) {
        c.eq("missing tenant -> default", m, Metrics::default());
    }
    // p95 nearest rank on 20 values 1..20 is the 19th.
    let many: Vec<ResultRow> = (0..20)
        .map(|i| {
            row(
                i,
                "a",
                0,
                0.,
                0.,
                (i + 1) as f64,
                (i + 1) as f64,
                0,
                1,
                true,
            )
        })
        .collect();
    if let Some(m) = c.guard("p95 of 20", move || summarize(&many, None, 1)) {
        c.close("p95 of 1..20 = 19", m.p95_ttft, 19.0, 1e-12);
        c.close("single replica imbalance 1", m.imbalance, 1.0, 1e-12);
    }
    // Duplication: replica 0 {a:[1,2], a:[1,2,3,4]}, replica 1 {a:[1,2], b:[1,2]},
    // replica 2 {a:[1,2], a:[1,2,3,4]}: a:[1,2] x3 -> 2, a:[1,2,3,4] x2 -> 1, b x1 -> 0.
    let mk = |keys: &[(&str, Vec<u32>)]| {
        let mut cache = Cache::new(8, 2, Policy::Lru);
        cache.clock = 1;
        for (t, p) in keys {
            cache.nodes.insert(
                key(t, p),
                Entry {
                    refs: 0,
                    last: 1,
                    hits: 0,
                },
            );
        }
        cache
    };
    let caches = vec![
        mk(&[("a", vec![1, 2]), ("a", vec![1, 2, 3, 4])]),
        mk(&[("a", vec![1, 2]), ("b", vec![1, 2])]),
        mk(&[("a", vec![1, 2]), ("a", vec![1, 2, 3, 4])]),
    ];
    if let Some(d) = c.guard("duplication", move || duplicated_blocks(&caches)) {
        c.eq("extra copies across replicas (tenant b is distinct)", d, 3);
    }
    if let Some(d) = c.guard("no caches", || duplicated_blocks(&[])) {
        c.eq("no caches", d, 0);
    }
    c.done();
}

// ===================================================================== R
fn tcfg() -> TraceConfig {
    TraceConfig {
        time_scale: 0.5,
        decode_ms_per_token: 2.0,
        ns: ns("mooncake"),
    }
}
#[test]
fn r_mooncake_adapter() {
    let mut c = Checks::new("r_mooncake_adapter");
    let line =
        r#"{"timestamp": 1200, "input_length": 1100, "output_length": 7, "hash_ids": [46, 47, 7]}"#;
    if let Some(r) = c.guard("valid", move || parse_mooncake_line(line, 5, &tcfg())) {
        match r {
            Ok(req) => {
                c.eq("id", req.id, 5);
                c.close(
                    "arrival = timestamp * time_scale",
                    req.arrival,
                    600.0,
                    1e-12,
                );
                c.eq("tokens = hash ids", req.tokens, vec![46, 47, 7]);
                c.eq(
                    "prompt tokens = input_length (last block partial)",
                    req.prompt_tokens,
                    1100,
                );
                c.close(
                    "decode = output_length * ms/token",
                    req.decode_ms,
                    14.0,
                    1e-12,
                );
                c.eq("namespace", req.ns, ns("mooncake"));
            }
            Err(e) => c.check("valid line", false, || format!("{e:?}")),
        }
    }
    let repeated = r#"{"timestamp":0,"input_length":1024,"output_length":0,"hash_ids":[7,7]}"#;
    if let Some(r) = c.guard("repeated", move || {
        parse_mooncake_line(repeated, 0, &tcfg())
    }) {
        c.check(
            "repeated ids and zero output are fine",
            r.as_ref()
                .is_ok_and(|q| q.tokens == vec![7, 7] && q.decode_ms == 0.0),
            || format!("{r:?}"),
        );
    }
    let mismatch = r#"{"timestamp":3,"input_length":5000,"output_length":1,"hash_ids":[1,2]}"#;
    if let Some(r) = c.guard("mismatch", move || {
        parse_mooncake_line(mismatch, 0, &tcfg())
    }) {
        c.check(
            "length mismatch is accepted, prompt_tokens kept",
            r.as_ref().is_ok_and(|q| q.prompt_tokens == 5000),
            || format!("{r:?}"),
        );
    }
    let errors: Vec<(&str, &str, TraceError)> = vec![
        (
            "not json",
            "{\"timestamp\": 1,",
            TraceError::Json(String::new()),
        ),
        (
            "array, not object",
            "[1,2]",
            TraceError::Json(String::new()),
        ),
        (
            "missing timestamp",
            r#"{"input_length":1,"output_length":1,"hash_ids":[1]}"#,
            TraceError::Missing("timestamp"),
        ),
        (
            "missing hash_ids",
            r#"{"timestamp":1,"input_length":1,"output_length":1}"#,
            TraceError::Missing("hash_ids"),
        ),
        (
            "negative input",
            r#"{"timestamp":1,"input_length":-5,"output_length":1,"hash_ids":[1]}"#,
            TraceError::NotInteger("input_length"),
        ),
        (
            "fractional output",
            r#"{"timestamp":1,"input_length":5,"output_length":1.5,"hash_ids":[1]}"#,
            TraceError::NotInteger("output_length"),
        ),
        (
            "string timestamp",
            r#"{"timestamp":"1","input_length":5,"output_length":1,"hash_ids":[1]}"#,
            TraceError::NotInteger("timestamp"),
        ),
        (
            "hash ids not array",
            r#"{"timestamp":1,"input_length":5,"output_length":1,"hash_ids":3}"#,
            TraceError::NotInteger("hash_ids"),
        ),
        (
            "negative id",
            r#"{"timestamp":1,"input_length":5,"output_length":1,"hash_ids":[1,-2]}"#,
            TraceError::NotInteger("hash_ids"),
        ),
        (
            "zero input",
            r#"{"timestamp":1,"input_length":0,"output_length":1,"hash_ids":[1]}"#,
            TraceError::Empty,
        ),
        (
            "empty ids",
            r#"{"timestamp":1,"input_length":9,"output_length":1,"hash_ids":[]}"#,
            TraceError::Empty,
        ),
        (
            "id overflow",
            r#"{"timestamp":1,"input_length":9,"output_length":1,"hash_ids":[4294967296]}"#,
            TraceError::IdOverflow(4294967296),
        ),
    ];
    for (name, text, want) in errors {
        let Some(got) = c.guard(name, move || parse_mooncake_line(text, 0, &tcfg())) else {
            continue;
        };
        let ok = match (&got, &want) {
            (Err(TraceError::Json(_)), TraceError::Json(_)) => true,
            (Err(e), w) => e == w,
            _ => false,
        };
        c.check(name, ok, || {
            format!("expected Err({want:?}) vs got {got:?}")
        });
    }
    let text = "\n{\"timestamp\":0,\"input_length\":512,\"output_length\":1,\"hash_ids\":[1]}\n\n{\"timestamp\":4,\"input_length\":600,\"output_length\":2,\"hash_ids\":[1,2]}\n";
    if let Some(r) = c.guard("load", move || load_mooncake(text, &tcfg())) {
        c.check(
            "load skips blank lines, ids 0..",
            r.as_ref()
                .is_ok_and(|v| v.len() == 2 && v[1].id == 1 && v[1].arrival == 2.0),
            || format!("{r:?}"),
        );
    }
    c.done();
}
#[test]
fn r_compute_saved() {
    let mut c = Checks::new("r_compute_saved");
    let row = |reused, total| ResultRow {
        id: 0,
        tenant: "x".into(),
        replica: 0,
        arrival: 0.,
        start: 0.,
        first_token: 0.,
        finish: 0.,
        matched: reused,
        reused,
        total,
        inserted: true,
    };
    // T(p,n) = 2 + n + 0.01 n (p + n/2), no floor, no d.
    // row A: 100 of 100 tokens... reused 99: T(99,1) = 2 + 1 + 0.01*(99.5) = 3.995;
    //        T(0,100) = 2 + 100 + 0.01*100*50 = 152.
    // row B: reused 0 of 50: T(0,50) = 2 + 50 + 12.5 = 64.5 both ways.
    // token hit = 99/150 = 0.66; saved = 1 - (3.995 + 64.5)/(152 + 64.5).
    let m = CostModel {
        floor_ms: 0.0,
        a_ms: 2.0,
        b_ms: 1.0,
        c_ms: 0.01,
        d_ms: 0.0,
    };
    let rows = vec![row(99, 100), row(0, 50)];
    if let Some(s) = c.guard("hand", move || compute_saved(&rows, &m)) {
        c.close("token hit", s.token_hit, 0.66, 1e-12);
        c.close("prefill ms with cache", s.prefill_ms, 3.995 + 64.5, 1e-9);
        c.close(
            "compute saved",
            s.compute_saved,
            1.0 - 68.495 / 216.5,
            1e-12,
        );
    }
    // A floor hides small savings: T = max(30, n): 10 of 20 reused -> both 30 ms.
    let floor = CostModel {
        floor_ms: 30.0,
        a_ms: 0.0,
        b_ms: 1.0,
        c_ms: 0.0,
        d_ms: 0.0,
    };
    if let Some(s) = c.guard("floor", move || compute_saved(&[row(10, 20)], &floor)) {
        c.close("token hit 0.5", s.token_hit, 0.5, 1e-12);
        c.close(
            "but zero time saved under the floor",
            s.compute_saved,
            0.0,
            1e-12,
        );
    }
    // The d*p term: reusing still pays per cached token.
    let dterm = CostModel {
        floor_ms: 0.0,
        a_ms: 0.0,
        b_ms: 1.0,
        c_ms: 0.0,
        d_ms: 0.25,
    };
    if let Some(s) = c.guard("d term", move || compute_saved(&[row(8, 10)], &dterm)) {
        // with: 2 + 0.25*8 = 4; without: 10 -> saved 0.6 < hit 0.8
        c.close("per-cached-token term", s.compute_saved, 0.6, 1e-12);
    }
    if let Some(s) = c.guard("empty", || compute_saved(&[], &CostModel::per_token(1.0))) {
        c.eq("empty -> zeros", s, Saved::default());
    }
    c.done();
}
#[test]
fn r_reference_trace_replay() {
    let mut c = Checks::new("r_reference_trace_replay");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("real/data");
    let trace =
        std::fs::read_to_string(dir.join("synthetic-mooncake.jsonl")).expect("committed trace");
    let fit =
        std::fs::read_to_string(dir.join("cost-qwen3-0.6b-rtx4060.json")).expect("committed fit");
    let (cost, label) = load_cost_model(&fit).expect("cost model parses");
    c.check(
        "fit is labeled with device/driver/date",
        label.matches(" / ").count() == 2,
        || label.clone(),
    );
    let lines = trace.lines().filter(|l| !l.trim().is_empty()).count();
    let tc = TraceConfig {
        time_scale: 1.0,
        decode_ms_per_token: 0.0,
        ns: ns("mooncake"),
    };
    let Some(Ok(reqs)) = c.guard("load trace", move || load_mooncake(&trace, &tc)) else {
        c.check("trace loads", false, || {
            "adapter rejected the committed trace".into()
        });
        return c.done();
    };
    c.eq("every line becomes a request", reqs.len(), lines);
    let mut results = Vec::new();
    for route in [RoutePolicy::RoundRobin, RoutePolicy::PrefixAffinity] {
        let config = ClusterConfig {
            replicas: 4,
            slots: 4,
            capacity_blocks: 64,
            block: 1,
            unit_tokens: MOONCAKE_BLOCK_TOKENS,
            eviction: Policy::Lru,
            admission: AdmissionPolicy::Always,
            history_capacity: 1024,
            rule: FullHitRule::Vllm,
            cost,
            route,
        };
        let r = reqs.clone();
        if let Some(out) = c.guard(&format!("{route:?}"), move || replay(r, &config)) {
            c.eq(
                &format!("{route:?} conserves requests"),
                out.rows.len(),
                reqs.len(),
            );
            let reused_ok = out
                .rows
                .iter()
                .all(|x| x.reused % 512 == 0 && x.reused < x.total);
            c.check(
                &format!("{route:?} vLLM rule: block-aligned, never the whole prompt"),
                reused_ok,
                || "bad reused".into(),
            );
            let s = compute_saved(&out.rows, &cost);
            results.push(s);
        }
    }
    if results.len() == 2 {
        // Regression on the committed trace (not a theorem): affinity keeps sessions
        // on the replica that holds their history, round robin scatters them.
        c.check(
            "affinity hit > round-robin hit on the committed trace",
            results[1].token_hit > results[0].token_hit,
            || format!("{results:?}"),
        );
        c.check(
            "compute saved in (0, 1)",
            results
                .iter()
                .all(|s| s.compute_saved > 0.0 && s.compute_saved < 1.0),
            || format!("{results:?}"),
        );
    }
    c.done();
}
