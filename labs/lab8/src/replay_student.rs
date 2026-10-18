//! R: replay REAL rollout-length traces through the A/B schedulers.
//!
//! Trace schema (JSONL, one object per line; lines with "type" other than "sample" are
//! metadata; lines without "type" are samples, so a TRL callback can append bare rows):
//!   prompt_id, group, sample_idx, policy_version, response_tokens, gen_ms, reward
//!   optional: round (default 0), truncated (default false)
//! The committed reference trace was sampled with Qwen3-0.6B (plain transformers
//! sampling, batch = group, NOT GRPO training) on an RTX 4060 Laptop; see its header.
//!
//! Replay model (modeled, not measured): one synchronous RL step per `round`; a round's
//! groups run on `workers` identical workers, one group (one batched generate call) at a
//! time per worker; a group occupies its worker for max(gen_ms) over the samples it keeps
//! (measured on one GPU). The next round starts when the slowest worker is done.
use crate::{assign, keep_first_n, schedule, Finished, Job, PredictedJob};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

// ---------------------------------------------------------------------------------
// Provided: a minimal JSON reader (no external crates; the lab builds offline).
// ---------------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}
impl Json {
    pub fn parse(text: &str) -> Result<Json, String> {
        let bytes = text.as_bytes();
        let mut pos = 0;
        let value = parse_value(bytes, &mut pos)?;
        skip_ws(bytes, &mut pos);
        if pos != bytes.len() {
            return Err(format!("trailing characters at byte {pos}"));
        }
        Ok(value)
    }
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(map) => map.get(key),
            _ => None,
        }
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }
    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(v) => Some(v),
            _ => None,
        }
    }
    pub fn arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }
    pub fn nums(&self) -> Option<Vec<f64>> {
        self.arr()?.iter().map(Json::num).collect()
    }
}
fn skip_ws(b: &[u8], pos: &mut usize) {
    while *pos < b.len() && (b[*pos] as char).is_ascii_whitespace() {
        *pos += 1;
    }
}
fn parse_value(b: &[u8], pos: &mut usize) -> Result<Json, String> {
    skip_ws(b, pos);
    let c = *b.get(*pos).ok_or("unexpected end")?;
    match c {
        b'{' => {
            *pos += 1;
            let mut map = BTreeMap::new();
            skip_ws(b, pos);
            if b.get(*pos) == Some(&b'}') {
                *pos += 1;
                return Ok(Json::Obj(map));
            }
            loop {
                skip_ws(b, pos);
                let key = match parse_value(b, pos)? {
                    Json::Str(s) => s,
                    _ => return Err("object key must be a string".into()),
                };
                skip_ws(b, pos);
                if b.get(*pos) != Some(&b':') {
                    return Err(format!("expected ':' at byte {pos}"));
                }
                *pos += 1;
                map.insert(key, parse_value(b, pos)?);
                skip_ws(b, pos);
                match b.get(*pos) {
                    Some(b',') => *pos += 1,
                    Some(b'}') => {
                        *pos += 1;
                        return Ok(Json::Obj(map));
                    }
                    _ => return Err(format!("expected ',' or '}}' at byte {pos}")),
                }
            }
        }
        b'[' => {
            *pos += 1;
            let mut items = Vec::new();
            skip_ws(b, pos);
            if b.get(*pos) == Some(&b']') {
                *pos += 1;
                return Ok(Json::Arr(items));
            }
            loop {
                items.push(parse_value(b, pos)?);
                skip_ws(b, pos);
                match b.get(*pos) {
                    Some(b',') => *pos += 1,
                    Some(b']') => {
                        *pos += 1;
                        return Ok(Json::Arr(items));
                    }
                    _ => return Err(format!("expected ',' or ']' at byte {pos}")),
                }
            }
        }
        b'"' => {
            *pos += 1;
            let mut out = String::new();
            loop {
                let c = *b.get(*pos).ok_or("unterminated string")?;
                *pos += 1;
                match c {
                    b'"' => return Ok(Json::Str(out)),
                    b'\\' => {
                        let e = *b.get(*pos).ok_or("bad escape")?;
                        *pos += 1;
                        match e {
                            b'n' => out.push('\n'),
                            b't' => out.push('\t'),
                            b'r' => out.push('\r'),
                            b'b' | b'f' => {}
                            b'u' => {
                                let hex =
                                    std::str::from_utf8(b.get(*pos..*pos + 4).ok_or("bad \\u")?)
                                        .map_err(|e| e.to_string())?;
                                let code =
                                    u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                                out.push(char::from_u32(code).unwrap_or('?'));
                                *pos += 4;
                            }
                            other => out.push(other as char),
                        }
                    }
                    _ => {
                        // copy one UTF-8 sequence verbatim
                        let start = *pos - 1;
                        let len = match c {
                            0x00..=0x7f => 1,
                            0xc0..=0xdf => 2,
                            0xe0..=0xef => 3,
                            _ => 4,
                        };
                        *pos = start + len;
                        out.push_str(
                            std::str::from_utf8(&b[start..*pos]).map_err(|e| e.to_string())?,
                        );
                    }
                }
            }
        }
        b't' if b[*pos..].starts_with(b"true") => {
            *pos += 4;
            Ok(Json::Bool(true))
        }
        b'f' if b[*pos..].starts_with(b"false") => {
            *pos += 5;
            Ok(Json::Bool(false))
        }
        b'n' if b[*pos..].starts_with(b"null") => {
            *pos += 4;
            Ok(Json::Null)
        }
        _ => {
            let start = *pos;
            while *pos < b.len() && b"+-0123456789.eE".contains(&b[*pos]) {
                *pos += 1;
            }
            let s = std::str::from_utf8(&b[start..*pos]).map_err(|e| e.to_string())?;
            s.parse::<f64>()
                .map(Json::Num)
                .map_err(|_| format!("bad number {s:?} at byte {start}"))
        }
    }
}

// ---------------------------------------------------------------------------------
// Provided: trace model and parser.
// ---------------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq)]
pub struct TraceSample {
    pub prompt_id: usize,
    pub group: usize,
    pub round: usize,
    pub sample_idx: usize,
    pub policy_version: u64,
    pub response_tokens: u64,
    pub gen_ms: f64,
    pub reward: f64,
    pub truncated: bool,
}
#[derive(Clone, Debug)]
pub struct TraceGroup {
    pub group: usize,
    pub prompt_id: usize,
    pub round: usize,
    /// Sorted by sample_idx.
    pub samples: Vec<TraceSample>,
}
#[derive(Clone, Debug, Default)]
pub struct RolloutTrace {
    pub header: BTreeMap<String, String>,
    /// Sorted by (round, group).
    pub groups: Vec<TraceGroup>,
}
impl RolloutTrace {
    pub fn parse(text: &str) -> Result<RolloutTrace, String> {
        let mut t = RolloutTrace::default();
        let mut by_group: BTreeMap<usize, TraceGroup> = BTreeMap::new();
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let v = Json::parse(line).map_err(|e| format!("line {}: {e}", n + 1))?;
            let kind = v.get("type").and_then(Json::str).unwrap_or("sample");
            if kind == "header" {
                if let Json::Obj(map) = &v {
                    for (key, value) in map {
                        let text = match value {
                            Json::Str(s) => s.clone(),
                            Json::Num(x) => x.to_string(),
                            other => format!("{other:?}"),
                        };
                        t.header.insert(key.clone(), text);
                    }
                }
                continue;
            }
            if kind != "sample" {
                continue;
            }
            let num = |key: &str| -> Result<f64, String> {
                v.get(key)
                    .and_then(Json::num)
                    .ok_or(format!("line {}: missing number {key}", n + 1))
            };
            let s = TraceSample {
                prompt_id: num("prompt_id")? as usize,
                group: num("group")? as usize,
                round: v.get("round").and_then(Json::num).unwrap_or(0.0) as usize,
                sample_idx: num("sample_idx")? as usize,
                policy_version: num("policy_version")? as u64,
                response_tokens: num("response_tokens")? as u64,
                gen_ms: num("gen_ms")?,
                reward: num("reward")?,
                truncated: v.get("truncated") == Some(&Json::Bool(true)),
            };
            if !(s.gen_ms.is_finite() && s.gen_ms >= 0.0) {
                return Err(format!("line {}: gen_ms must be finite and >= 0", n + 1));
            }
            let entry = by_group.entry(s.group).or_insert_with(|| TraceGroup {
                group: s.group,
                prompt_id: s.prompt_id,
                round: s.round,
                samples: vec![],
            });
            if entry.prompt_id != s.prompt_id || entry.round != s.round {
                return Err(format!(
                    "line {}: group {} mixes prompts/rounds",
                    n + 1,
                    s.group
                ));
            }
            if entry.samples.iter().any(|x| x.sample_idx == s.sample_idx) {
                return Err(format!(
                    "line {}: duplicate sample {} in group {}",
                    n + 1,
                    s.sample_idx,
                    s.group
                ));
            }
            entry.samples.push(s);
        }
        let mut groups: Vec<_> = by_group.into_values().collect();
        for g in &mut groups {
            g.samples.sort_by_key(|s| s.sample_idx);
        }
        groups.sort_by_key(|g| (g.round, g.group));
        t.groups = groups;
        Ok(t)
    }
    pub fn rounds(&self) -> Vec<usize> {
        let set: BTreeSet<usize> = self.groups.iter().map(|g| g.round).collect();
        set.into_iter().collect()
    }
    pub fn samples(&self) -> impl Iterator<Item = &TraceSample> {
        self.groups.iter().flat_map(|g| g.samples.iter())
    }
}

/// Provided path of the committed reference trace (see its header for device/date).
pub fn reference_trace_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("real/data/qwen3-0.6b-arith-g8-rtx4060.jsonl")
}

// ---------------------------------------------------------------------------------
// R2/R3: a per-prompt quantile length predictor.
// ---------------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct QuantilePredictor {
    /// Nearest-rank quantile in [0, 1]; 1.0 predicts the largest remembered length.
    pub q: f64,
    /// Per-prompt history keeps at most `window` lengths; the global one 16 * window.
    pub window: usize,
    /// Prediction when nothing at all has been observed.
    pub default: u64,
    pub per_prompt: BTreeMap<usize, VecDeque<u64>>,
    pub global: VecDeque<u64>,
}
impl QuantilePredictor {
    pub fn new(q: f64, window: usize, default: u64) -> Self {
        assert!((0.0..=1.0).contains(&q) && window > 0);
        Self {
            q,
            window,
            default,
            per_prompt: BTreeMap::new(),
            global: VecDeque::new(),
        }
    }
    // R2: record the lengths of FINISHED samples of one group of `prompt_id`, in order.
    // Append each to the prompt's history (keep the newest `window`) and to the global
    // history (keep the newest 16 * window). Test: r_predictor.
    pub fn observe(&mut self, prompt_id: usize, lengths: &[u64]) {
        // BEGIN TODO R2
        todo!("TODO R2")
        // END TODO R2
    }
    // R3: nearest-rank q-quantile, sorted[max(ceil(q * n), 1) - 1], of the prompt's
    // history; if the prompt has none, of the global history; if that is empty too,
    // `default`. Pure (must not change the histories). Test: r_predictor.
    pub fn predict(&self, prompt_id: usize) -> u64 {
        // BEGIN TODO R3
        todo!("TODO R3")
        // END TODO R3
    }
}

// ---------------------------------------------------------------------------------
// R1: replay one strategy over every round of a trace.
// ---------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Strategy {
    /// Shared queue in group order; an idle worker pulls the next group.
    Fifo,
    /// Static queues from assign(predicted, lpt = true); no rebalancing.
    LptPredicted,
    /// Shared queue sorted by predicted length (desc, ties by position); idle workers pull.
    LptPull,
    /// Queues from assign(predicted, lpt = true), then B3 with stealing, no preemption.
    WorkStealing,
    /// Static LPT on the TRUE durations: a diagnostic bound, not deployable.
    LptOracle,
    /// Redundancy: keep the first n finishers of each group, abort the rest; Fifo order.
    KeepFirstN(usize),
    /// No redundancy, unbiased: keep sample_idx < n (time = their max gen_ms); Fifo order.
    FirstIdx(usize),
}
impl Strategy {
    pub fn name(&self) -> String {
        match self {
            Strategy::Fifo => "fifo".into(),
            Strategy::LptPredicted => "lpt_predicted".into(),
            Strategy::LptPull => "lpt_pull".into(),
            Strategy::WorkStealing => "work_stealing".into(),
            Strategy::LptOracle => "lpt_oracle".into(),
            Strategy::KeepFirstN(n) => format!("keep_first_{n}"),
            Strategy::FirstIdx(n) => format!("first_idx_{n}"),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct ReplayReport {
    /// Sum over rounds of the round makespan, in integer milliseconds.
    pub makespan_ms: u64,
    pub round_makespan_ms: Vec<u64>,
    /// Sum of all group durations (ms) actually executed.
    pub busy_ms: u64,
    pub groups: usize,
    pub kept_samples: usize,
    pub kept_tokens: u64,
    pub mean_len_all: f64,
    pub mean_len_kept: f64,
    pub mean_reward_all: f64,
    pub mean_reward_kept: f64,
    /// Groups whose rewards are all equal (GRPO advantage identically 0).
    pub zero_adv_groups_all: usize,
    pub zero_adv_groups_kept: usize,
    /// (group id, predicted length at dispatch time, true max response_tokens of the
    /// whole group) for every group, in trace order.
    pub predictions: Vec<(usize, u64, u64)>,
    pub steals: usize,
}
impl ReplayReport {
    pub fn throughput_samples_per_s(&self) -> f64 {
        self.kept_samples as f64 / (self.makespan_ms as f64 / 1000.0)
    }
}
/// Provided: true when every reward is equal (the group's GRPO advantages are all 0).
pub fn all_equal(xs: &[f64]) -> bool {
    xs.windows(2).all(|w| w[0] == w[1])
}
// R1: replay `strategy` over the trace, round by round in increasing order.
// For each round:
//   1. predictions: predictor.predict(prompt_id) for EVERY group of the round, before
//      anything of this round is observed (no future lengths); record them.
//   2. kept samples: all, or keep_first_n (B4) on (gen_ms, sample_idx) for
//      KeepFirstN(n), or sample_idx < n for FirstIdx(n). Duration of a group =
//      round(max gen_ms over its kept samples) in ms.
//   3. round makespan by the strategy (see Strategy docs); jobs use the group's position
//      in the round as id; WorkStealing calls schedule(.., quantum = u64::MAX, true).
//   4. then predictor.observe(prompt_id, response_tokens of the KEPT samples in
//      sample_idx order): aborted samples never finished, so they are not observed.
// Report totals and the distribution shift: means over all trace samples vs over kept
// samples, and zero-advantage groups (all rewards equal) over all vs kept samples.
// Test: r_replay_synthetic (hand-computed), r_replay_reference (independent oracles).
pub fn replay(
    trace: &RolloutTrace,
    workers: usize,
    strategy: Strategy,
    predictor: &mut QuantilePredictor,
) -> ReplayReport {
    // BEGIN TODO R1
    todo!("TODO R1")
    // END TODO R1
}
