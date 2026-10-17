use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Namespace {
    pub model: String,
    pub adapter: String,
    pub tokenizer: String,
    pub tenant: String,
    pub position_offset: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    pub ns: Namespace,
    pub prefix: Vec<u32>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub refs: usize,
    pub last: u64,
    pub hits: u64,
}
#[derive(Clone, Copy, Debug)]
pub enum Policy {
    Lru,
    Frequency,
}
#[derive(Clone, Debug)]
pub struct Cache {
    pub capacity: usize,
    pub block: usize,
    pub nodes: BTreeMap<Key, Entry>,
    pub clock: u64,
    pub policy: Policy,
}
#[derive(Debug)]
pub struct Lease {
    keys: Vec<Key>,
}

/// A1: Materialize full ancestry keys for COMPLETE blocks only, with all namespace
/// fields. Include the first block and exclude the partial tail. Full token keys
/// deliberately avoid hash collisions; production code uses chained hashes.
/// Tests: a_keys_namespace_and_tail.
pub fn block_keys(ns: &Namespace, tokens: &[u32], block: usize) -> Vec<Key> {
    // BEGIN TODO A1
    assert!(block > 0);
    let mut keys = Vec::with_capacity(tokens.len() / block);
    let mut prefix = Vec::new();
    for chunk in tokens.chunks_exact(block) {
        prefix.extend_from_slice(chunk);
        keys.push(Key {
            ns: ns.clone(),
            prefix: prefix.clone(),
        });
    }
    keys
    // END TODO A1
}
impl Cache {
    pub fn new(capacity: usize, block: usize, policy: Policy) -> Self {
        assert!(block > 0);
        Self {
            capacity,
            block,
            nodes: BTreeMap::new(),
            clock: 0,
            policy,
        }
    }
    /// A2: Stop at the FIRST absent ancestor. A later matching block is not reusable
    /// if an earlier prefix differs. Query must not alter LRU/hits/refcounts.
    /// Tests: a_match_stops_at_gap.
    pub fn matched_tokens(&self, ns: &Namespace, tokens: &[u32]) -> usize {
        // BEGIN TODO A2
        let keys = block_keys(ns, tokens, self.block);
        let mut matched = 0;
        for key in keys {
            if !self.nodes.contains_key(&key) {
                break;
            }
            matched += self.block;
        }
        matched
        // END TODO A2
    }
    /// A3: Return an unpinned LEAF outside protected keys. Never remove an ancestor
    /// of any resident node. LRU minimizes (last,key); Frequency minimizes
    /// (hits,last,key). Deterministic key order breaks ties. Tests: a_leaf_victims.
    pub fn victim(&self, protected: &BTreeSet<Key>) -> Option<Key> {
        // BEGIN TODO A3
        let mut candidates = Vec::new();
        for (key, entry) in &self.nodes {
            if entry.refs > 0 || protected.contains(key) {
                continue;
            }
            let has_child = self.nodes.keys().any(|other| {
                other.ns == key.ns
                    && other.prefix.len() > key.prefix.len()
                    && other.prefix.starts_with(&key.prefix)
            });
            if has_child {
                continue;
            }
            candidates.push((key, entry));
        }
        candidates.sort_by(|(ka, a), (kb, b)| match self.policy {
            Policy::Lru => a.last.cmp(&b.last).then(ka.cmp(kb)),
            Policy::Frequency => a
                .hits
                .cmp(&b.hits)
                .then(a.last.cmp(&b.last))
                .then(ka.cmp(kb)),
        });
        candidates.first().map(|(key, _)| (*key).clone())
        // END TODO A3
    }
    /// A4: Structural audit: capacity, block alignment, all immediate parents,
    /// and refs(child)<=refs(parent). Return descriptive first failure. This
    /// invariant follows because leases pin every matched ancestor.
    /// Tests: a_invariants_detect_corruption.
    pub fn validate(&self) -> Result<(), String> {
        // BEGIN TODO A4
        if self.nodes.len() > self.capacity {
            return Err("capacity".into());
        }
        for (key, entry) in &self.nodes {
            let n = key.prefix.len();
            if n == 0 || n % self.block != 0 {
                return Err("alignment".into());
            }
            if n > self.block {
                let parent = Key {
                    ns: key.ns.clone(),
                    prefix: key.prefix[..n - self.block].to_vec(),
                };
                let parent_entry = self.nodes.get(&parent).ok_or("orphan")?;
                if entry.refs > parent_entry.refs {
                    return Err("lease ancestry".into());
                }
            }
            if entry.last > self.clock {
                return Err("future clock".into());
            }
        }
        Ok(())
        // END TODO A4
    }
    /// B1: Atomic all-prefix insertion. Stage mutations on a clone, protect every
    /// requested ancestor, repeatedly evict eligible leaves, then publish once.
    /// Failure must preserve nodes, metadata AND clock. Existing hits are not
    /// counted as reads here. Tests: b_transaction_rollback_and_leaf_order.
    pub fn insert(&mut self, ns: &Namespace, tokens: &[u32]) -> bool {
        // BEGIN TODO B1
        let keys = block_keys(ns, tokens, self.block);
        if keys.len() > self.capacity {
            return false;
        }
        let protected: BTreeSet<_> = keys.iter().cloned().collect();
        let missing = keys.iter().filter(|k| !self.nodes.contains_key(*k)).count();
        let mut staged = self.clone();
        while staged.nodes.len() + missing > staged.capacity {
            match staged.victim(&protected) {
                Some(key) => {
                    staged.nodes.remove(&key);
                }
                None => return false,
            }
        }
        staged.clock += 1;
        for key in keys {
            staged.nodes.entry(key).or_insert(Entry {
                refs: 0,
                last: staged.clock,
                hits: 0,
            });
        }
        assert!(staged.validate().is_ok());
        *self = staged;
        true
        // END TODO B1
    }
    /// B2: Pin the contiguous hit chain, touching each node once. Partial miss
    /// returns a shorter lease. Lease keys are private and release consumes it,
    /// preventing double release in safe code. Tests: b_shared_leases.
    pub fn acquire(&mut self, ns: &Namespace, tokens: &[u32]) -> Lease {
        // BEGIN TODO B2
        let keys = block_keys(ns, tokens, self.block);
        let mut pinned = Vec::new();
        self.clock += 1;
        for key in keys {
            match self.nodes.get_mut(&key) {
                None => break,
                Some(entry) => {
                    entry.refs += 1;
                    entry.last = self.clock;
                    entry.hits += 1;
                    pinned.push(key);
                }
            }
        }
        Lease { keys: pinned }
        // END TODO B2
    }
    /// B3: Release deepest-first. Pin counts never underflow; all leased entries
    /// must still exist. Do not remove zero-ref entries: they become reusable.
    /// Tests: b_shared_leases and b_release_preserves_reuse.
    pub fn release(&mut self, lease: Lease) {
        // BEGIN TODO B3
        for key in lease.keys.into_iter().rev() {
            let entry = self.nodes.get_mut(&key).expect("leased key was evicted");
            assert!(entry.refs > 0);
            entry.refs -= 1;
        }
        assert!(self.validate().is_ok());
        // END TODO B3
    }
    /// B4: Namespace invalidation is atomic: refuse if ANY matching entry is
    /// pinned; otherwise remove all matching nodes and leave other namespaces.
    /// Return removed count, or None for blocked. Tests: b_namespace_invalidation.
    pub fn invalidate(&mut self, ns: &Namespace) -> Option<usize> {
        // BEGIN TODO B4
        let keys: Vec<_> = self.nodes.keys().filter(|k| &k.ns == ns).cloned().collect();
        if keys.iter().any(|key| self.nodes[key].refs > 0) {
            return None;
        }
        let count = keys.len();
        for key in keys {
            self.nodes.remove(&key);
        }
        self.clock += 1;
        assert!(self.validate().is_ok());
        Some(count)
        // END TODO B4
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub blocks: usize,
    pub pinned_blocks: usize,
    pub references: usize,
    pub key_token_bytes: usize,
    pub max_depth: usize,
}
impl Cache {
    /// A5: Account actual full-prefix token-key payload and pin multiplicity.
    /// key_token_bytes excludes Vec/String/BTreeMap overhead and KV tensors;
    /// pinned_blocks counts nodes once, references counts every live lease.
    /// Tests: a_accounting_multiple_leases.
    pub fn stats(&self) -> CacheStats {
        // BEGIN TODO A5
        let mut stats = CacheStats::default();
        for (key, entry) in &self.nodes {
            stats.blocks += 1;
            stats.key_token_bytes += key.prefix.len() * std::mem::size_of::<u32>();
            stats.references += entry.refs;
            stats.max_depth = stats.max_depth.max(key.prefix.len() / self.block);
            if entry.refs > 0 {
                stats.pinned_blocks += 1;
            }
        }
        stats
        // END TODO A5
    }
    /// B5: Atomically resize the block budget. Shrink by eligible leaves until
    /// len<=target; if pins prevent it leave capacity, clock and nodes untouched.
    /// Growing must preserve LRU ordering; a successful shrink advances clock.
    /// Tests: b_resize_transaction_with_pins.
    pub fn resize(&mut self, target: usize) -> bool {
        // BEGIN TODO B5
        if target >= self.capacity {
            self.capacity = target;
            return true;
        }
        let mut staged = self.clone();
        let protected = BTreeSet::new();
        while staged.nodes.len() > target {
            let victim = match staged.victim(&protected) {
                Some(key) => key,
                None => return false,
            };
            staged.nodes.remove(&victim);
        }
        staged.capacity = target;
        staged.clock += 1;
        assert!(staged.validate().is_ok());
        *self = staged;
        true
        // END TODO B5
    }
}
