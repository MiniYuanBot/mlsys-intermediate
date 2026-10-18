//! R: replay real acceptance traces (Qwen3-0.6B -> Qwen3-1.7B, collected by
//! labs/lab4/real/collect.py) through the C engine and compare controllers.
//!
//! Trace semantics (greedy): the target's greedy output is y_0..y_{T-1}; y_0 comes from
//! prefill and is never speculated. flags[i] = 1 iff the draft's argmax given y_{<=i}
//! (teacher-forced) equals y_{i+1}, for i = 0..T-2. A replayed request therefore has
//! output = flags.len() decode steps, and a round that starts after `emitted` committed
//! decode steps proposes guesses for flags[emitted], flags[emitted+1], ...

use crate::{choose_k, Acceptance, Estimator, Policy, ProfilePoint, Request, TimingSample};
use std::collections::BTreeMap;

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
// Provided: trace file model.
// ---------------------------------------------------------------------------------
#[derive(Clone, Debug, Default)]
pub struct TracePrompt {
    pub id: usize,
    pub task: String,
    pub prompt_tokens: usize,
    pub flags: Vec<u8>,
    /// Same flags from cached width-1 draft steps (only for prompts decoded for real).
    pub stepwise_flags: Option<Vec<u8>>,
}
#[derive(Clone, Debug, Default)]
pub struct SpecRun {
    pub prompt: usize,
    pub k: usize,
    /// (proposed, accepted) per round, as measured by the hand-written loop.
    pub rounds: Vec<(usize, usize)>,
    pub decode_ms: f64,
    pub same_as_greedy: bool,
    /// First generated-token index where the loop's output left the greedy output.
    pub first_diff: Option<usize>,
}
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub header: BTreeMap<String, String>,
    pub kmax: usize,
    /// CUDA-graph regime (what the hand-written loop runs).
    pub verify: Vec<TimingSample>,
    pub draft_ms: Vec<f64>,
    /// Eager PyTorch regime (launch-bound), kept as the negative result.
    pub verify_eager: Vec<TimingSample>,
    pub draft_eager_ms: Vec<f64>,
    /// Timing rows (graph regime) whose p75 > 1.5 * p25 even after re-measurement.
    pub unstable_rows: usize,
    pub prompts: Vec<TracePrompt>,
    pub spec: Vec<SpecRun>,
    /// plain greedy decode time per prompt id (same prompts as `spec`)
    pub plain_ms: BTreeMap<usize, f64>,
}
impl Trace {
    pub fn parse(text: &str) -> Result<Trace, String> {
        let mut t = Trace::default();
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let v = Json::parse(line).map_err(|e| format!("line {}: {e}", n + 1))?;
            let kind = v
                .get("type")
                .and_then(Json::str)
                .ok_or(format!("line {}: no type", n + 1))?;
            let num = |key: &str| -> Result<f64, String> {
                v.get(key)
                    .and_then(Json::num)
                    .ok_or(format!("line {}: missing {key}", n + 1))
            };
            match kind {
                "header" => {
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
                    t.kmax = num("kmax")? as usize;
                }
                "verify" => {
                    let (context, k) = (num("context")? as usize, num("k")? as usize);
                    let eager = v.get("regime").and_then(Json::str) == Some("eager");
                    if !eager && v.get("stable") == Some(&Json::Bool(false)) {
                        t.unstable_rows += 1;
                    }
                    for elapsed in v
                        .get("samples_ms")
                        .and_then(Json::nums)
                        .ok_or("samples_ms")?
                    {
                        let sample = TimingSample {
                            context,
                            k,
                            elapsed,
                        };
                        if eager {
                            t.verify_eager.push(sample)
                        } else {
                            t.verify.push(sample)
                        }
                    }
                }
                "draft" => {
                    let samples = v
                        .get("samples_ms")
                        .and_then(Json::nums)
                        .ok_or("samples_ms")?;
                    if v.get("regime").and_then(Json::str) != Some("eager")
                        && v.get("stable") == Some(&Json::Bool(false))
                    {
                        t.unstable_rows += 1;
                    }
                    if v.get("regime").and_then(Json::str) == Some("eager") {
                        t.draft_eager_ms.extend(samples)
                    } else {
                        t.draft_ms.extend(samples)
                    }
                }
                "prompt" => t.prompts.push(TracePrompt {
                    id: num("id")? as usize,
                    task: v.get("task").and_then(Json::str).unwrap_or("").to_string(),
                    prompt_tokens: num("prompt_tokens")? as usize,
                    flags: v
                        .get("flags")
                        .and_then(Json::nums)
                        .ok_or("flags")?
                        .into_iter()
                        .map(|x| x as u8)
                        .collect(),
                    stepwise_flags: v
                        .get("stepwise_flags")
                        .and_then(Json::nums)
                        .map(|f| f.into_iter().map(|x| x as u8).collect()),
                }),
                "spec" => t.spec.push(SpecRun {
                    prompt: num("prompt")? as usize,
                    k: num("k")? as usize,
                    rounds: v
                        .get("rounds")
                        .and_then(Json::arr)
                        .ok_or("rounds")?
                        .iter()
                        .map(|r| {
                            let x = r.nums().unwrap_or_default();
                            (x[0] as usize, x[1] as usize)
                        })
                        .collect(),
                    decode_ms: num("decode_ms")?,
                    same_as_greedy: v.get("same_as_greedy") == Some(&Json::Bool(true)),
                    first_diff: v.get("first_diff").and_then(Json::num).map(|x| x as usize),
                }),
                "plain" => {
                    t.plain_ms
                        .insert(num("prompt")? as usize, num("decode_ms")?);
                }
                _ => {} // forward compatible: ignore unknown record types
            }
        }
        if t.kmax == 0 || t.prompts.is_empty() {
            return Err("trace has no header/kmax or no prompt records".into());
        }
        Ok(t)
    }
    pub fn load(path: &std::path::Path) -> Result<Trace, String> {
        Trace::parse(
            &std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?,
        )
    }
    /// Cost model from the measured CUDA-graph samples: A5 calibrate with the median
    /// draft step (pooled over contexts; the draft step barely depends on context here).
    pub fn profile(&self) -> Result<Vec<ProfilePoint>, String> {
        let draft = crate::percentile(&self.draft_ms, 0.5).ok_or("no draft samples")?;
        crate::calibrate(&self.verify, self.kmax, draft)
    }
    /// Same for the eager regime (None if the trace has no eager samples).
    pub fn profile_eager(&self) -> Option<Vec<ProfilePoint>> {
        let draft = crate::percentile(&self.draft_eager_ms, 0.5)?;
        crate::calibrate(&self.verify_eager, self.kmax, draft).ok()
    }
}

/// Provided path of the committed reference trace (see its header for device/date).
pub fn reference_trace_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("real/data/qwen3-0.6b-1.7b-rtx4060-greedy.jsonl")
}

// ---------------------------------------------------------------------------------
// R3: censored replay of a trace.
// ---------------------------------------------------------------------------------
/// R3 (three small functions, one TODO):
/// * run_lengths: runs[i] = number of consecutive 1 flags starting at i (0 when
///   flags[i] == 0); runs.len() == flags.len(); O(n) with one backward pass.
/// * censored_accept: a round of k proposals starting after `emitted` committed decode
///   steps accepts min(runs[emitted], k). The trace knows how far the draft would have
///   matched; the controller only learns the part it paid for, and k = 0 reveals
///   nothing. Panics if emitted >= runs.len().
/// * fixed_k_schedule: closed-form fixed-k replay without the event engine. Returns
///   (proposed, accepted) per round with proposed = min(k, n - e - 1) (never propose
///   past the last decode step); the next round starts at e + accepted + 1. It must
///   reproduce the measured rounds of the hand-written loop whenever that loop's output
///   equals greedy.
/// Tests: r_runs_bruteforce, r_trace_env_censors, r_fixed_schedule_matches_engine,
/// r_reference_rounds_match_measured.
// BEGIN TODO R3
pub fn run_lengths(flags: &[u8]) -> Vec<usize> {
    let mut runs = vec![0; flags.len()];
    let mut next = 0;
    for i in (0..flags.len()).rev() {
        assert!(flags[i] <= 1, "flags must be 0/1");
        next = if flags[i] == 1 { next + 1 } else { 0 };
        runs[i] = next;
    }
    runs
}

pub fn censored_accept(runs: &[usize], emitted: usize, k: usize) -> usize {
    assert!(
        emitted < runs.len(),
        "round starts past the end of the trace"
    );
    runs[emitted].min(k)
}

pub fn fixed_k_schedule(runs: &[usize], k: usize) -> Vec<(usize, usize)> {
    let n = runs.len();
    let mut e = 0;
    let mut out = Vec::new();
    while e < n {
        let proposed = k.min(n - e - 1);
        let accepted = censored_accept(runs, e, proposed);
        out.push((proposed, accepted));
        e += accepted + 1;
    }
    out
}
// END TODO R3

/// Provided: the trace as an Acceptance environment for the C engine.
pub struct TraceEnv {
    /// runs[request id][i], from R3 run_lengths.
    pub runs: Vec<Vec<usize>>,
}
impl TraceEnv {
    pub fn from_flags(flags: &[Vec<u8>]) -> Self {
        Self {
            runs: flags.iter().map(|f| run_lengths(f)).collect(),
        }
    }
}
impl Acceptance for TraceEnv {
    fn accepted(&self, request: &Request, emitted: usize, k: usize) -> usize {
        censored_accept(&self.runs[request.id], emitted, k)
    }
}

/// Provided: compare a measured round log with a replayed schedule over the span where
/// the measured run was still on the greedy path. A round starting at decode step e and
/// committing c tokens covers generated tokens e+1..=e+c; it is compared only if
/// e + c < first_diff (strictly before the divergent token). Returns (compared, equal).
pub fn compare_before_divergence(
    measured: &[(usize, usize)],
    replay: &[(usize, usize)],
    first_diff: Option<usize>,
) -> (usize, usize) {
    let limit = first_diff.unwrap_or(usize::MAX);
    let (mut e, mut compared, mut equal) = (0usize, 0, 0);
    for (i, &(k, a)) in measured.iter().enumerate() {
        if e + a + 1 >= limit {
            break;
        }
        compared += 1;
        if replay.get(i) == Some(&(k, a)) {
            equal += 1;
        } else {
            break; // after one mismatch the two logs start at different positions
        }
        e += a + 1;
    }
    (compared, equal)
}

/// Provided: modeled decode time of a schedule under a cost profile (ms), context
/// measured from the start of decoding (prompt length + y_0 + emitted). Matches the
/// hand-written loop: each round runs k draft calls (the first one also catches up on
/// the 1-2 committed tokens) and one target call of width k+1.
pub fn schedule_ms(
    schedule: &[(usize, usize)],
    profile: &[ProfilePoint],
    prompt_tokens: usize,
) -> f64 {
    let mut emitted = 0;
    let mut total = 0.0;
    for &(k, a) in schedule {
        total += crate::interpolate(profile, prompt_tokens + 1 + emitted).round_cost(k);
        emitted += a + 1;
    }
    total
}

// ---------------------------------------------------------------------------------
// R4: censored-feedback optimism (your estimator).
// ---------------------------------------------------------------------------------
/// Per-position counts as in B1 (discounted, censored), but the decision uses an
/// OPTIMISTIC hazard per depth. Playing k reveals min(L, k), i.e. the outcome every
/// smaller k would have had, so optimism about depth j is resolved by any round with
/// k > j. That is the structural difference from an arm-independent bandit.
pub struct CensoredUcb {
    pub estimator: Estimator,
    pub z: f64,
}
impl CensoredUcb {
    pub fn new(kmax: usize, decay: f64, z: f64) -> Self {
        Self {
            estimator: Estimator::new(kmax, decay),
            z,
        }
    }
}
impl CensoredUcb {
    /// R4: upper end of the Wilson score interval per position,
    ///   (p + z^2/(2n) + z*sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n),  p = s/n, n = s+f,
    /// and 1.0 when n == 0 (unexplored depths are optimistic). Clamp to [0,1].
    /// Tests: r_ucb_upper_bisection.
    pub fn upper(&self) -> Vec<f64> {
        // BEGIN TODO R4
        let z = self.z;
        self.estimator
            .success
            .iter()
            .zip(&self.estimator.failure)
            .map(|(&s, &f)| {
                let n = s + f;
                if n == 0.0 {
                    return 1.0;
                }
                let p = s / n;
                let center = p + z * z / (2.0 * n);
                let margin = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
                ((center + margin) / (1.0 + z * z / n)).clamp(0.0, 1.0)
            })
            .collect()
        // END TODO R4
    }
}
impl Policy for CensoredUcb {
    fn name(&self) -> String {
        format!("censored_ucb_z{}", self.z)
    }
    /// Provided glue: B3 on the optimistic hazards.
    fn choose(&mut self, costs: &ProfilePoint, budget: f64, remaining: usize) -> Option<usize> {
        choose_k(&self.upper(), costs, budget, remaining)
    }
    /// Provided glue: B1 records the accepted prefix and the first rejection only.
    fn observe(&mut self, accepted: usize, proposed: usize) {
        self.estimator.observe(accepted, proposed);
    }
    fn reset(&mut self) {
        self.estimator = Estimator::new(self.estimator.success.len(), self.estimator.decay);
    }
}

/// Provided bandit baseline (UCB1 over arms k = 0..kmax, BanditSpec-style
/// simplification). Each arm keeps its own discounted mean of committed tokens per
/// ms and only learns when it is played: no sharing across k, no censoring structure.
pub struct ArmUcb {
    pub reward: Vec<f64>,
    pub plays: Vec<f64>,
    pub decay: f64,
    pub c: f64,
    last: Option<(usize, f64)>,
}
impl ArmUcb {
    pub fn new(kmax: usize, decay: f64, c: f64) -> Self {
        Self {
            reward: vec![0.0; kmax + 1],
            plays: vec![0.0; kmax + 1],
            decay,
            c,
            last: None,
        }
    }
}
impl Policy for ArmUcb {
    fn name(&self) -> String {
        format!("arm_ucb_c{}", self.c)
    }
    fn choose(&mut self, costs: &ProfilePoint, budget: f64, remaining: usize) -> Option<usize> {
        let max_k = (costs.verify.len() - 1)
            .min(self.plays.len() - 1)
            .min(remaining.saturating_sub(1));
        let total: f64 = self.plays.iter().sum::<f64>().max(1.0);
        let mut best: Option<(usize, f64)> = None;
        for k in 0..=max_k {
            if costs.round_cost(k) > budget {
                continue;
            }
            let score = if self.plays[k] < 1e-9 {
                f64::INFINITY
            } else {
                // reward is tokens/ms, scale bonus by the arm's own mean magnitude
                let mean = self.reward[k] / self.plays[k];
                mean + self.c * mean.max(1e-6) * (total.ln() / self.plays[k]).sqrt()
            };
            if best.is_none_or(|(_, s)| score > s + 1e-12) {
                best = Some((k, score));
            }
        }
        let k = best.map(|b| b.0)?;
        self.last = Some((k, costs.round_cost(k)));
        Some(k)
    }
    fn observe(&mut self, accepted: usize, proposed: usize) {
        for i in 0..self.plays.len() {
            self.plays[i] *= self.decay;
            self.reward[i] *= self.decay;
        }
        if let Some((k, cost)) = self.last.take() {
            if k == proposed {
                self.plays[k] += 1.0;
                self.reward[k] += (accepted + 1) as f64 / cost;
            }
        }
    }
    fn reset(&mut self) {
        *self = ArmUcb::new(self.plays.len() - 1, self.decay, self.c);
    }
}

/// Provided baseline: the Hugging Face "heuristic" assistant schedule (transformers
/// `num_assistant_tokens_schedule="heuristic"`): +2 after a fully accepted round,
/// -1 otherwise, never below 1. Ignores costs except for the budget filter.
pub struct HfHeuristic {
    pub k: usize,
    pub kmax: usize,
}
impl Policy for HfHeuristic {
    fn name(&self) -> String {
        "hf_heuristic".into()
    }
    fn choose(&mut self, costs: &ProfilePoint, budget: f64, remaining: usize) -> Option<usize> {
        let mut k = self
            .k
            .min(costs.verify.len() - 1)
            .min(remaining.saturating_sub(1));
        while k > 0 && costs.round_cost(k) > budget {
            k -= 1;
        }
        (costs.round_cost(k) <= budget).then_some(k)
    }
    fn observe(&mut self, accepted: usize, proposed: usize) {
        if proposed > 0 {
            self.k = if accepted == proposed {
                (self.k + 2).min(self.kmax)
            } else {
                self.k.saturating_sub(1).max(1)
            };
        }
    }
}

/// Provided: survival S_j = P(L >= j) and hazard a_j = P(L >= j | L >= j-1) over every
/// start position whose depth j is observable (i + j <= n). Used by the bench only.
pub fn position_rates(flags: &[u8], kmax: usize) -> (Vec<f64>, Vec<f64>) {
    let runs = run_lengths(flags);
    let n = flags.len();
    let (mut survival, mut hazard) = (Vec::new(), Vec::new());
    for j in 1..=kmax {
        let starts: Vec<usize> = (0..n).filter(|&i| i + j <= n).collect();
        let reach = starts.iter().filter(|&&i| runs[i] >= j).count();
        let prior = starts.iter().filter(|&&i| runs[i] >= j - 1).count();
        survival.push(if starts.is_empty() {
            f64::NAN
        } else {
            reach as f64 / starts.len() as f64
        });
        hazard.push(if prior == 0 {
            f64::NAN
        } else {
            reach as f64 / prior as f64
        });
    }
    (survival, hazard)
}
