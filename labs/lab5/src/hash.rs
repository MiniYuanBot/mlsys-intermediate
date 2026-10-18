//! Provided: the block-hash primitive for A1 (do not edit).
//!
//! vLLM v0.32.0 hashes each FULL block as H(parent_hash, block_tokens, extra_keys)
//! (`vllm/v1/core/kv_cache_utils.py::hash_block_tokens`). This module fixes a
//! deterministic, std-only H so tests can recompute expected hashes: 64-bit FNV-1a over
//! a length-prefixed byte encoding. It is NOT collision resistant and NOT keyed; real
//! engines use sha256 or a seeded xxhash (see the lecture section on hash seeds).

/// Hash of the "parent" of the first block (vLLM: NONE_HASH, a fixed seed for sha256).
pub const NONE_HASH: u64 = 0x6e6f_6e65_2d68_6173; // "none-has"

/// One extra key that joins a block's identity besides its tokens.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExtraKey {
    /// LoRA adapter, keyed by (name, path) so re-pointing a name at new weights misses.
    Lora { name: String, path: String },
    /// A multimodal item overlapping this block: content identifier plus the item's
    /// start position relative to the block start (negative if it began earlier).
    Mm { identifier: String, offset: i64 },
    /// Per-request isolation salt (vLLM `cache_salt`; the lab's tenant).
    Salt(String),
}

/// A multimodal placeholder run inside the prompt: tokens [start, start + len).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmItem {
    pub identifier: String,
    pub start: usize,
    pub len: usize,
}

/// Everything that decides block identity for one request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HashRequest {
    pub tokens: Vec<u32>,
    /// (adapter name, adapter path) when a LoRA adapter is active.
    pub lora: Option<(String, String)>,
    pub mm: Vec<MmItem>,
    pub cache_salt: Option<String>,
}

fn fnv(state: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *state ^= b as u64;
        *state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn put_str(state: &mut u64, tag: u8, s: &str) {
    fnv(state, &[tag]);
    fnv(state, &(s.len() as u64).to_le_bytes());
    fnv(state, s.as_bytes());
}

/// H(parent, tokens, extra). Every field is length-prefixed, so ([1,2],[3]) and
/// ([1],[2,3]) never encode to the same bytes.
pub fn hash_block(parent: u64, tokens: &[u32], extra: &[ExtraKey]) -> u64 {
    let mut s: u64 = 0xcbf2_9ce4_8422_2325;
    fnv(&mut s, &parent.to_le_bytes());
    fnv(&mut s, &(tokens.len() as u64).to_le_bytes());
    for t in tokens {
        fnv(&mut s, &t.to_le_bytes());
    }
    fnv(&mut s, &(extra.len() as u64).to_le_bytes());
    for key in extra {
        match key {
            ExtraKey::Lora { name, path } => {
                put_str(&mut s, b'L', name);
                put_str(&mut s, b'l', path);
            }
            ExtraKey::Mm { identifier, offset } => {
                put_str(&mut s, b'M', identifier);
                fnv(&mut s, &offset.to_le_bytes());
            }
            ExtraKey::Salt(salt) => put_str(&mut s, b'S', salt),
        }
    }
    s
}
