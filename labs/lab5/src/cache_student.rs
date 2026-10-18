//! A1-A6, B1-B5: block identity, the prefix-cache structure and its ownership rules.
use crate::hash::{hash_block, ExtraKey, HashRequest, NONE_HASH};
use std::collections::{BTreeMap, BTreeSet};

/// A1: vLLM-style chained block hashes for the FULL blocks of `req.tokens`
/// (a partial tail block gets no hash; block >= 1).
///
/// Block i covers tokens [s, e) = [i*block, (i+1)*block) and its hash is
/// `hash_block(parent, &tokens[s..e], &extra_i)` with parent = NONE_HASH for i = 0 and the
/// previous block's hash otherwise. `extra_i` lists, in this order:
///   1. `Lora{name, path}` on EVERY block when `req.lora` is Some;
///   2. one `Mm{identifier, offset}` per item of `req.mm` (in `req.mm` order) that
///      overlaps [s, e), i.e. item.start < e && item.start + item.len > s, with
///      offset = item.start as i64 - s as i64 (negative when the item began earlier);
///   3. `Salt(salt)` ONLY on block 0 when `req.cache_salt` is Some (the parent chain
///      carries it to every later block, as vLLM does).
/// Sampling parameters are deliberately absent: they do not change prefill KV.
/// Tests: a_hash_chain_extra_keys.
pub fn block_hashes(req: &HashRequest, block: usize) -> Vec<u64> {
    // BEGIN TODO A1
    todo!("TODO A1")
    // END TODO A1
}

/// Everything besides the tokens that must match before KV may be shared.
/// `tenant` plays the role of vLLM's `cache_salt`; `position_offset` stands in for
/// "the same chunk placed at another position" (only position-independent caching such
/// as EPIC/CacheBlend could reuse that; exact-prefix engines always start at 0).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Namespace {
    pub model: String,
    pub adapter: String,
    pub tokenizer: String,
    pub tenant: String,
    pub position_offset: usize,
}
/// Teaching key: the namespace plus the FULL token prefix up to the end of the block.
/// Collision-free by construction (costs O(T^2/B) memory; production uses A1's hashes).
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
/// Eviction order among legal victims. `Frequency` is SGLang's LFUStrategy order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
/// Pins on one contiguous hit chain. Fields are private: only `acquire` creates one and
/// `release` consumes it.
#[derive(Debug)]
pub struct Lease {
    keys: Vec<Key>,
}
impl Lease {
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// First broken invariant found by `Cache::validate` (A5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvariantError {
    /// More nodes than `capacity`.
    OverCapacity { nodes: usize, capacity: usize },
    /// Prefix length is 0 or not a multiple of `block`.
    Misaligned(Key),
    /// The immediate parent (prefix minus its last block) is not resident.
    Orphan(Key),
    /// refs(child) > refs(parent): a lease pinned a child without its ancestors.
    PinAboveParent(Key),
    /// entry.last > cache.clock.
    FutureClock(Key),
}

/// A2: Full-ancestry keys for the COMPLETE blocks of `tokens`, each carrying `ns`.
/// block=2, tokens=[1,2,3,4,5] gives prefixes [1,2] and [1,2,3,4]; the tail [5] is
/// dropped. Tests: a_keys_namespace_and_tail.
pub fn block_keys(ns: &Namespace, tokens: &[u32], block: usize) -> Vec<Key> {
    // BEGIN TODO A2
    todo!("TODO A2")
    // END TODO A2
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
    /// A3: Number of tokens covered by the contiguous run of resident full blocks,
    /// stopping at the FIRST absent block even if a deeper key is present (a resident
    /// child behind a gap is unreachable, exactly like vLLM's chained lookup). Read-only:
    /// no change to refs, last, hits or clock. Tests: a_match_stops_at_gap.
    pub fn matched_tokens(&self, ns: &Namespace, tokens: &[u32]) -> usize {
        // BEGIN TODO A3
        todo!("TODO A3")
        // END TODO A3
    }
    /// A4: Choose the eviction victim among LEGAL candidates only: refs == 0, not in
    /// `protected`, and a leaf (no other resident key in the same namespace extends its
    /// prefix). Lru minimizes (last, key); Frequency minimizes (hits, last, key).
    /// None when no legal candidate exists. Leaf-only eviction is SGLang semantics
    /// (vLLM may evict a parent and strand its children; see the lecture).
    /// Tests: a_leaf_victims.
    pub fn victim(&self, protected: &BTreeSet<Key>) -> Option<Key> {
        // BEGIN TODO A4
        todo!("TODO A4")
        // END TODO A4
    }
    /// A5: Structural audit. Check `nodes.len() <= capacity` first; then visit nodes in
    /// key order and, per node, check in this order: alignment (prefix length > 0 and a
    /// multiple of block), parent presence (for prefixes longer than one block),
    /// refs(node) <= refs(parent), and last <= clock. Return the FIRST violation.
    /// Tests: a_invariants_detect_corruption.
    pub fn validate(&self) -> Result<(), InvariantError> {
        // BEGIN TODO A5
        todo!("TODO A5")
        // END TODO A5
    }
    /// B1: All-or-nothing insertion of every full-block prefix of `tokens`.
    /// Stage on a clone: count missing keys, protect every key of this path, evict
    /// legal victims (A4) until len + missing <= capacity, then advance clock by 1 and
    /// add missing keys with refs 0, last = new clock, hits 0; publish by assignment.
    /// Existing keys on the path are NOT touched (insert is not a read).
    /// Return false, leaving nodes, metadata AND clock unchanged, if the path alone
    /// exceeds capacity or pins block eviction. An empty path succeeds as a no-op.
    /// Tests: b_transaction_rollback_and_leaf_order.
    pub fn insert(&mut self, ns: &Namespace, tokens: &[u32]) -> bool {
        // BEGIN TODO B1
        todo!("TODO B1")
        // END TODO B1
    }
    /// B2: Advance clock by 1, then pin the contiguous resident chain of `tokens`:
    /// for each hit key refs += 1, last = clock, hits += 1; stop at the first miss.
    /// A total miss still advances the clock and returns an empty lease.
    /// Tests: b_shared_leases.
    pub fn acquire(&mut self, ns: &Namespace, tokens: &[u32]) -> Lease {
        // BEGIN TODO B2
        todo!("TODO B2")
        // END TODO B2
    }
    /// B3: Consume the lease and decrement refs deepest-first. Panic if a leased key
    /// is missing or already at 0 (that is a bug, not a recoverable state). Zero-ref
    /// entries stay cached for future hits. Does not touch the clock.
    /// Tests: b_shared_leases, b_release_preserves_reuse.
    pub fn release(&mut self, lease: Lease) {
        // BEGIN TODO B3
        todo!("TODO B3")
        // END TODO B3
    }
    /// B4: Remove every node of namespace `ns` atomically. If ANY of them is pinned
    /// return None and change nothing (clock included). Otherwise remove them, advance
    /// clock by 1 and return Some(count). Other namespaces are untouched.
    /// Tests: b_namespace_invalidation.
    pub fn invalidate(&mut self, ns: &Namespace) -> Option<usize> {
        // BEGIN TODO B4
        todo!("TODO B4")
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
    /// A6: blocks = nodes; pinned_blocks = nodes with refs > 0 (each counted once);
    /// references = sum of refs; key_token_bytes = 4 bytes per u32 in every stored
    /// prefix (no container overhead, no KV tensors); max_depth = longest prefix in
    /// blocks. Tests: a_accounting_multiple_leases.
    pub fn stats(&self) -> CacheStats {
        // BEGIN TODO A6
        todo!("TODO A6")
        // END TODO A6
    }
    /// B5: Atomically change the block budget. Growing (target >= capacity) only sets
    /// capacity: nodes, metadata and clock unchanged. Shrinking evicts legal victims
    /// (A4, nothing protected) on a clone until len <= target, then sets capacity and
    /// advances clock by 1; if pins prevent reaching target, change nothing and return
    /// false. Tests: b_resize_transaction_with_pins.
    pub fn resize(&mut self, target: usize) -> bool {
        // BEGIN TODO B5
        todo!("TODO B5")
        // END TODO B5
    }
}
