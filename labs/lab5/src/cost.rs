//! Provided: the prefill cost model shared by C and R (do not edit).
//!
//! For a request with p reused (cached) tokens and n uncached tokens,
//!
//!   T(p, n) = max(floor, a + b*n + c*n*(p + n/2) + d*p)      [ms]
//!
//! b*n   : per-token linear layers (2 * N_nonembedding FLOPs per token),
//! c*... : attention of the n new queries over p + (their own causal half) keys,
//! d*p   : per-call traffic over the cached KV (reading it; HF DynamicCache also copies it),
//! floor : the launch-bound minimum of an eager forward (CPU dispatch hides small GPU work).
//! R1 fits (floor, a, b, c, d) on the GPU; C tests use hand constants.

/// Which full-hit rule decides how many matched tokens may be reused (C2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FullHitRule {
    /// SGLang (page_size 1): at most total-1 tokens, so >= 1 token is recomputed.
    Sglang,
    /// vLLM: at most total-1 tokens rounded DOWN to a block boundary.
    Vllm,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    pub floor_ms: f64,
    pub a_ms: f64,
    pub b_ms: f64,
    pub c_ms: f64,
    pub d_ms: f64,
}

impl CostModel {
    /// Pure per-token cost (hand oracles): T = ms_per_token * n.
    pub fn per_token(ms_per_token: f64) -> Self {
        Self {
            floor_ms: 0.0,
            a_ms: 0.0,
            b_ms: ms_per_token,
            c_ms: 0.0,
            d_ms: 0.0,
        }
    }
    /// Prefill latency for p reused tokens and n uncached tokens.
    pub fn prefill_ms(&self, p: usize, n: usize) -> f64 {
        let (p, n) = (p as f64, n as f64);
        let work = self.a_ms + self.b_ms * n + self.c_ms * n * (p + n / 2.0) + self.d_ms * p;
        work.max(self.floor_ms)
    }
}
