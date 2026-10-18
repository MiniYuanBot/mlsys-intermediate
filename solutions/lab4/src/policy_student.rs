//! Conditional acceptance, calibrated costs, estimators and budget allocation (A/B).

#[derive(Clone, Debug, PartialEq)]
pub struct ProfilePoint {
    pub context: usize,
    /// Cost of one draft step, ms.
    pub draft: f64,
    /// verify[k] = cost of verifying k proposals (k+1 target positions), ms.
    pub verify: Vec<f64>,
}

impl ProfilePoint {
    /// Provided: full round cost T(k) = k * T_d + T_v(k).
    pub fn round_cost(&self, k: usize) -> f64 {
        self.draft * k as f64 + self.verify[k]
    }
}

/// A1: Return P(committed=n), n=1..k+1. A failure ends observation; the
/// final bucket means every proposal was accepted. Check normalization and
/// support before using this distribution. Tests: a_distribution_oracle.
pub fn committed_distribution(conditional: &[f64], k: usize) -> Vec<f64> {
    // BEGIN TODO A1
    assert!(k <= conditional.len());
    let mut mass = Vec::with_capacity(k + 1);
    let mut survival = 1.0;
    for &p in &conditional[..k] {
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
        mass.push(survival * (1.0 - p));
        survival *= p;
    }
    mass.push(survival);
    let total: f64 = mass.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
    mass
    // END TODO A1
}

/// A2: Compute the mean and variance of committed tokens from A1, not a constant
/// acceptance shortcut. k=0 yields one correction token and zero variance.
/// Tests: a_moments_and_quantiles.
pub fn committed_moments(conditional: &[f64], k: usize) -> (f64, f64) {
    // BEGIN TODO A2
    let mass = committed_distribution(conditional, k);
    let mut mean = 0.0;
    let mut second = 0.0;
    for (i, p) in mass.iter().enumerate() {
        let n = (i + 1) as f64;
        mean += p * n;
        second += p * n * n;
    }
    (mean, (second - mean * mean).max(0.0))
    // END TODO A2
}

/// A3: Nearest-rank percentile, rank=max(1,ceil(q*n)); empty -> None.
/// Reject nonfinite samples; do not mutate caller data. q=0 is minimum.
/// Tests: a_moments_and_quantiles.
pub fn percentile(samples: &[f64], q: f64) -> Option<f64> {
    // BEGIN TODO A3
    assert!((0.0..=1.0).contains(&q));
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    assert!(sorted.iter().all(|v| v.is_finite()));
    sorted.sort_by(f64::total_cmp);
    let rank = ((q * sorted.len() as f64).ceil() as usize).max(1);
    sorted.get(rank - 1).copied()
    // END TODO A3
}

/// A4: Linear interpolation of per-k costs between strictly increasing context
/// samples. Clamp outside the measured range (report extrapolation separately).
/// All rows have the same nonempty verify width; costs are positive ms.
/// Tests: a_profile_interpolation.
pub fn interpolate(profile: &[ProfilePoint], context: usize) -> ProfilePoint {
    // BEGIN TODO A4
    assert!(!profile.is_empty());
    let width = profile[0].verify.len();
    assert!(width > 0);
    for row in profile {
        assert_eq!(row.verify.len(), width);
        assert!(row.draft >= 0.0 && row.draft.is_finite());
        assert!(row.verify.iter().all(|v| *v > 0.0 && v.is_finite()));
    }
    assert!(profile.windows(2).all(|p| p[0].context < p[1].context));
    if context <= profile[0].context {
        return profile[0].clone();
    }
    if context >= profile.last().unwrap().context {
        return profile.last().unwrap().clone();
    }
    let upper = profile.iter().position(|p| p.context >= context).unwrap();
    let (left, right) = (&profile[upper - 1], &profile[upper]);
    let ratio = (context - left.context) as f64 / (right.context - left.context) as f64;
    let verify = left
        .verify
        .iter()
        .zip(&right.verify)
        .map(|(a, b)| a + ratio * (b - a))
        .collect();
    ProfilePoint {
        context,
        draft: left.draft + ratio * (right.draft - left.draft),
        verify,
    }
    // END TODO A4
}

#[derive(Clone, Debug)]
pub struct TimingSample {
    pub context: usize,
    pub k: usize,
    pub elapsed: f64,
}
/// A5: Build a dense per-context verify table from repeated timing observations.
/// Group equal (context,k), use the A3 nearest-rank median, require every k=0..kmax.
/// Reject incomplete calibration instead of silently manufacturing a cost.
/// All samples here time the verifier only; draft_ms is calibrated separately.
/// Stage R feeds real cudaEvent samples from the 4060 into this function.
/// Tests: a_calibration_grouping_and_missing.
pub fn calibrate(
    samples: &[TimingSample],
    kmax: usize,
    draft_ms: f64,
) -> Result<Vec<ProfilePoint>, String> {
    // BEGIN TODO A5
    let mut grouped: std::collections::BTreeMap<usize, Vec<Vec<f64>>> =
        std::collections::BTreeMap::new();
    if !draft_ms.is_finite() || draft_ms < 0.0 {
        return Err("draft cost".into());
    }
    for sample in samples {
        if sample.k > kmax || sample.elapsed <= 0.0 || !sample.elapsed.is_finite() {
            return Err("invalid timing sample".into());
        }
        let buckets = grouped
            .entry(sample.context)
            .or_insert_with(|| vec![Vec::new(); kmax + 1]);
        buckets[sample.k].push(sample.elapsed);
    }
    if grouped.is_empty() {
        return Err("empty calibration".into());
    }
    let mut profile = Vec::new();
    for (context, buckets) in grouped {
        let mut verify = Vec::new();
        for values in buckets {
            verify.push(percentile(&values, 0.5).ok_or("missing k in calibration")?);
        }
        profile.push(ProfilePoint {
            context,
            draft: draft_ms,
            verify,
        });
    }
    Ok(profile)
    // END TODO A5
}

#[derive(Clone, Debug)]
pub struct Estimator {
    pub success: Vec<f64>,
    pub failure: Vec<f64>,
    pub decay: f64,
}
impl Estimator {
    pub fn new(kmax: usize, decay: f64) -> Self {
        assert!((0.0..=1.0).contains(&decay) && decay > 0.0);
        Self {
            success: vec![0.0; kmax],
            failure: vec![0.0; kmax],
            decay,
        }
    }
    /// B1: Discount EVERY historical position, then update only the accepted
    /// prefix and first rejected proposal. Never label an unobserved suffix.
    /// No pseudo-counts here: the confidence method handles zero evidence.
    /// Tests: b_censored_observation_and_decay.
    pub fn observe(&mut self, accepted: usize, proposed: usize) {
        // BEGIN TODO B1
        assert!(accepted <= proposed && proposed <= self.success.len());
        for j in 0..self.success.len() {
            self.success[j] *= self.decay;
            self.failure[j] *= self.decay;
        }
        for j in 0..accepted {
            self.success[j] += 1.0;
        }
        if accepted < proposed {
            self.failure[accepted] += 1.0;
        }
        // END TODO B1
    }
    /// B2: z == 0: posterior means (s+1)/(n+2) under Beta(1,1).
    /// z > 0: the LOWER end of the Wilson score interval on the (discounted) counts,
    ///   (p + z^2/(2n) - z*sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n),  p = s/n, n = s+f,
    /// and 0 when n == 0. With discounting and drift this is a conservative control
    /// score, NOT a calibrated confidence bound. Tests: b_wilson_numeric_and_ranking.
    pub fn probabilities(&self, z: f64) -> Vec<f64> {
        // BEGIN TODO B2
        assert!(z >= 0.0 && z.is_finite());
        let mut out = Vec::with_capacity(self.success.len());
        for (&s, &f) in self.success.iter().zip(&self.failure) {
            let n = s + f;
            if z == 0.0 {
                out.push((s + 1.0) / (n + 2.0));
                continue;
            }
            if n == 0.0 {
                out.push(0.0);
                continue;
            }
            let p = s / n;
            let denominator = 1.0 + z * z / n;
            let center = p + z * z / (2.0 * n);
            let margin = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
            out.push(((center - margin) / denominator).clamp(0.0, 1.0));
        }
        out
        // END TODO B2
    }
    /// Provided: pooled IID estimate alpha = sum(s) / sum(s+f) (the censored MLE when
    /// every position shares one acceptance probability), Beta(1,1)-smoothed and
    /// broadcast to every position. Used by the `Pooled` policy as the IID baseline.
    pub fn pooled(&self) -> Vec<f64> {
        let s: f64 = self.success.iter().sum();
        let f: f64 = self.failure.iter().sum();
        vec![(s + 1.0) / (s + f + 2.0); self.success.len()]
    }
}

/// B3: Feasible expected-tokens / round-cost maximization. Cost = round_cost(k).
/// Remaining output limits k to remaining-1; rate ties (within 1e-12) choose shorter k.
/// None means no k fits the bound (or remaining == 0); the caller counts it.
/// Tests: b_choice_exhaustive_and_infeasible.
pub fn choose_k(p: &[f64], profile: &ProfilePoint, bound: f64, remaining: usize) -> Option<usize> {
    // BEGIN TODO B3
    if remaining == 0 {
        return None;
    }
    let max_k = p.len().min(profile.verify.len() - 1).min(remaining - 1);
    let mut best = None;
    let mut best_rate = f64::NEG_INFINITY;
    for k in 0..=max_k {
        let cost = profile.round_cost(k);
        if cost > bound {
            continue;
        }
        let (tokens, _) = committed_moments(p, k);
        let rate = tokens / cost;
        if rate > best_rate + 1e-12 {
            best_rate = rate;
            best = Some(k);
        }
    }
    best
    // END TODO B3
}

#[derive(Debug, Default, Clone)]
pub struct Drift {
    pub mean: f64,
    pub count: usize,
    pub cumulative: f64,
    pub resets: usize,
}
impl Drift {
    /// B4: Downward Page-Hinkley-style detector on observed accepted/proposed.
    /// If count == 0, set mean = value first. Then
    ///   cumulative = max(0, cumulative + mean_old - value - delta), count += 1,
    ///   mean = running mean including value.
    /// If cumulative > threshold: resets += 1, count = 0, cumulative = 0, mean = value,
    /// return true. Tests: b_drift_exact_events. Educational detector, not AdaServe code.
    pub fn update(&mut self, value: f64, delta: f64, threshold: f64) -> bool {
        // BEGIN TODO B4
        assert!((0.0..=1.0).contains(&value));
        if self.count == 0 {
            self.mean = value;
        }
        self.cumulative = (self.cumulative + self.mean - value - delta).max(0.0);
        self.count += 1;
        self.mean += (value - self.mean) / self.count as f64;
        if self.cumulative > threshold {
            self.resets += 1;
            self.count = 0;
            self.cumulative = 0.0;
            self.mean = value;
            return true;
        }
        false
        // END TODO B4
    }
}

/// Provided: kmax usable for one request in B5/B6.
fn usable_k(p: &[f64], row: &ProfilePoint) -> usize {
    p.len().min(row.verify.len() - 1)
}

/// B5: Greedy shared-budget allocation. Start every request at k=0; if the sum of
/// round_cost(0) exceeds the budget return None. Then repeatedly apply the single
/// feasible upgrade (request i: current k -> any larger k) with the greatest
/// marginal E[tokens] gain / extra cost; only upgrades with gain > 0 and cost > 0
/// that keep the total within budget (+1e-12) qualify. Ties (within 1e-12) go to the
/// lower request index, then the shorter k. Stop when nothing qualifies.
/// This is a HEURISTIC: it can be beaten by B6 (see the counterexample in the README).
/// Tests: b_greedy_trace_feasible_and_bounded.
pub fn allocate_round(
    probabilities: &[Vec<f64>],
    profiles: &[ProfilePoint],
    budget: f64,
) -> Option<Vec<usize>> {
    // BEGIN TODO B5
    assert_eq!(probabilities.len(), profiles.len());
    let mut choices = vec![0; profiles.len()];
    let mut used: f64 = profiles.iter().map(|p| p.round_cost(0)).sum();
    if used > budget + 1e-12 {
        return None;
    }
    loop {
        let mut best: Option<(usize, usize, f64, f64)> = None;
        for (i, row) in profiles.iter().enumerate() {
            let current = choices[i];
            let old_cost = row.round_cost(current);
            let old_gain = committed_moments(&probabilities[i], current).0;
            for k in current + 1..=usable_k(&probabilities[i], row) {
                let cost = row.round_cost(k) - old_cost;
                let gain = committed_moments(&probabilities[i], k).0 - old_gain;
                if cost <= 0.0 || gain <= 0.0 || used + cost > budget + 1e-12 {
                    continue;
                }
                let score = gain / cost;
                if best.as_ref().is_none_or(|old| score > old.3 + 1e-12) {
                    best = Some((i, k, cost, score));
                }
            }
        }
        match best {
            Some((i, k, cost, _)) => {
                choices[i] = k;
                used += cost;
            }
            None => break,
        }
    }
    Some(choices)
    // END TODO B5
}

/// B6: Exact allocation as a multiple-choice knapsack on a quantized budget.
/// Each request picks exactly one k in 0..=min(p.len(), verify.len()-1).
/// Integer ticks: cost_ik = ceil(round_cost(k)/quantum - 1e-9), capacity
/// B = floor(budget/quantum + 1e-9) (the 1e-9 absorbs binary-float noise such as
/// 1.6/0.1 = 16.000000000000004). Maximize the sum of E[N_ik] subject to sum of ticks <= B.
/// None if all-zero does not fit. Ties: total value within 1e-9 counts as equal;
/// among equal-value optima return the lexicographically smallest choice vector.
/// Expected O(n * K * B) time. Tests: b_exact_matches_exhaustive.
pub fn allocate_round_exact(
    probabilities: &[Vec<f64>],
    profiles: &[ProfilePoint],
    budget: f64,
    quantum: f64,
) -> Option<Vec<usize>> {
    // BEGIN TODO B6
    assert_eq!(probabilities.len(), profiles.len());
    assert!(quantum > 0.0 && budget.is_finite());
    let n = profiles.len();
    if budget < 0.0 {
        return None;
    }
    let capacity = (budget / quantum + 1e-9).floor() as usize;
    let options: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|i| {
            (0..=usable_k(&probabilities[i], &profiles[i]))
                .map(|k| {
                    let ticks = (profiles[i].round_cost(k) / quantum - 1e-9).ceil().max(0.0);
                    (ticks as usize, committed_moments(&probabilities[i], k).0)
                })
                .collect()
        })
        .collect();
    // best[i][b]: max value of requests i..n with b ticks left (NEG_INFINITY = infeasible).
    let mut best = vec![vec![f64::NEG_INFINITY; capacity + 1]; n + 1];
    best[n] = vec![0.0; capacity + 1];
    for i in (0..n).rev() {
        for b in 0..=capacity {
            for &(ticks, value) in &options[i] {
                if ticks <= b && best[i + 1][b - ticks] > f64::NEG_INFINITY {
                    best[i][b] = best[i][b].max(value + best[i + 1][b - ticks]);
                }
            }
        }
    }
    if best[0][capacity] == f64::NEG_INFINITY {
        return None;
    }
    let mut choices = Vec::with_capacity(n);
    let mut b = capacity;
    for i in 0..n {
        let target = best[i][b];
        let (k, &(ticks, _)) = options[i]
            .iter()
            .enumerate()
            .find(|(_, &(ticks, value))| {
                ticks <= b && value + best[i + 1][b - ticks] >= target - 1e-9
            })
            .unwrap();
        choices.push(k);
        b -= ticks;
    }
    Some(choices)
    // END TODO B6
}
