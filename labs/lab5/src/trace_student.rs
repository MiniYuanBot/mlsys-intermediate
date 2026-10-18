//! R2-R3: Mooncake-format trace adapter and the replay metric under the measured cost
//! model. Loading helpers around the TODOs are provided.
use crate::cost::CostModel;
use crate::json::Json;
use crate::{Namespace, Request, ResultRow};

/// Mooncake FAST'25 traces: one JSON object per line,
/// {"timestamp": ms, "input_length": tokens, "output_length": tokens, "hash_ids": [..]}
/// where hash_ids are remapped CHAINED hashes of consecutive 512-token blocks, so equal
/// ids at equal positions mean equal prefixes (github.com/kvcache-ai/Mooncake,
/// FAST25-release/README.md).
pub const MOONCAKE_BLOCK_TOKENS: usize = 512;

#[derive(Clone, Debug)]
pub struct TraceConfig {
    /// Multiply timestamps by this to get arrival ms (< 1 compresses the trace = more load).
    pub time_scale: f64,
    /// Modeled decode cost per output token, ms.
    pub decode_ms_per_token: f64,
    /// Namespace given to every request (Mooncake has no tenant field).
    pub ns: Namespace,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TraceError {
    /// The line is not valid JSON or not an object.
    Json(String),
    /// A required field is absent.
    Missing(&'static str),
    /// A field is present but not a non-negative integer (or not an array of them).
    NotInteger(&'static str),
    /// input_length == 0 or hash_ids empty.
    Empty,
    /// A hash id does not fit in u32.
    IdOverflow(u64),
}

/// R2: Turn one Mooncake trace line into a Request with the given `id`.
/// * Parse with `Json::parse`; a parse error or a non-object -> Json(message).
/// * timestamp, input_length, output_length: required (Missing(name)); each must be a
///   finite, non-negative, integral number (NotInteger(name)).
/// * hash_ids: required array of non-negative integers (NotInteger("hash_ids") for
///   anything else); a value > u32::MAX -> IdOverflow(value).
/// * input_length == 0 or an empty hash_ids -> Empty.
/// * Request: arrival = timestamp * time_scale; ns = cfg.ns; tokens = hash_ids as u32
///   (one cache unit = one 512-token block, so repeated ids like [7, 7] are fine: the
///   full-ancestry keys [7] and [7, 7] differ); prompt_tokens = input_length (it need
///   not equal 512 * hash_ids.len(): the last block may be partial and the trace has
///   mismatches); decode_ms = output_length * decode_ms_per_token.
/// Tests: r_mooncake_adapter.
pub fn parse_mooncake_line(
    line: &str,
    id: usize,
    cfg: &TraceConfig,
) -> Result<Request, TraceError> {
    // BEGIN TODO R2
    todo!("TODO R2")
    // END TODO R2
}

/// Provided: parse every non-blank line (ids 0, 1, ... in file order); report the
/// 1-based line number of the first bad line.
pub fn load_mooncake(text: &str, cfg: &TraceConfig) -> Result<Vec<Request>, (usize, TraceError)> {
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let req = parse_mooncake_line(line, out.len(), cfg).map_err(|e| (lineno + 1, e))?;
        out.push(req);
    }
    Ok(out)
}

/// Provided: read the R1 fit file written by real/fit.py.
pub fn load_cost_model(text: &str) -> Result<(CostModel, String), String> {
    let json = Json::parse(text)?;
    let coef = json.get("coefficients").ok_or("missing coefficients")?;
    let get = |k: &str| {
        coef.get(k)
            .and_then(Json::num)
            .ok_or(format!("missing coefficients.{k}"))
    };
    let model = CostModel {
        floor_ms: get("floor_ms")?,
        a_ms: get("a_ms")?,
        b_ms: get("b_ms")?,
        c_ms: get("c_ms")?,
        d_ms: get("d_ms")?,
    };
    let label = json
        .get("meta")
        .map(|m| {
            ["gpu", "driver", "date"]
                .iter()
                .filter_map(|k| m.get(k).and_then(Json::str))
                .collect::<Vec<_>>()
                .join(" / ")
        })
        .unwrap_or_default();
    Ok((model, label))
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Saved {
    /// sum(reused) / sum(total)
    pub token_hit: f64,
    /// 1 - sum_r T(reused_r, total_r - reused_r) / sum_r T(0, total_r): the share of
    /// prefill TIME the cache removed under `cost`.
    pub compute_saved: f64,
    /// sum_r T(reused_r, total_r - reused_r), ms
    pub prefill_ms: f64,
}
/// R3: Replay metric. Token hit ratio and the prefill time saved, both over `rows`
/// (empty rows or zero baseline -> all zero). With the attention and per-call terms of
/// the cost model the two differ: a reused prefix still costs attention for the suffix,
/// and the floor hides small savings. Tests: r_compute_saved.
pub fn compute_saved(rows: &[ResultRow], cost: &CostModel) -> Saved {
    // BEGIN TODO R3
    todo!("TODO R3")
    // END TODO R3
}
