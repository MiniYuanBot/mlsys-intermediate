//! Conditional acceptance, calibrated costs, and a constrained controller.

#[derive(Clone, Debug)]
pub struct ProfilePoint {
    pub context: usize,
    pub draft: f64,
    pub verify: Vec<f64>,
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

/// A2: Compute the first and second central moments from A1, not a constant
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
/// All rows have the same nonempty verify width; costs are positive simulated ms.
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
    /// B2: Return posterior means with Beta(1,1), or a Wilson-style conservative
    /// score on discounted counts when z>0. This score is a controller heuristic,
    /// NOT a calibrated confidence interval under nonstationarity. n=0 -> 0.
    /// Tests: b_lower_bound_and_ranking.
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
}

/// B3: Feasible expected tokens / round cost maximization. Cost=draft*k+verify[k].
/// Remaining output limits k to remaining-1; rate ties choose shorter k.
/// None means infeasible; the simulator explicitly falls back and counts it.
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
        let cost = profile.draft * k as f64 + profile.verify[k];
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

#[derive(Debug, Default)]
pub struct Drift {
    pub mean: f64,
    pub count: usize,
    pub cumulative: f64,
    pub resets: usize,
}
impl Drift {
    /// B4: Downward Page-Hinkley-like detector on observed accepted/proposed.
    /// Update cumulative=max(0,cumulative+old_mean-value-delta), then running mean.
    /// Trigger above threshold; reset window but preserve reset count.
    /// Tests: b_drift_drop_only. This is an educational detector, not AdaServe code.
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

#[derive(Clone, Debug)]
pub struct TimingSample {
    pub context: usize,
    pub k: usize,
    pub elapsed: f64,
}
/// A5: Build a dense per-context verify table from repeated timing observations.
/// Group equal (context,k), use nearest-rank median, require every k=0..kmax.
/// Reject incomplete calibration instead of silently manufacturing a cost.
/// All samples here time verifier only; draft_ms is calibrated separately.
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

/// B5: Allocate a shared round budget across multiple requests. Reserve k=0 for
/// every request first; if infeasible return None. Repeatedly choose the feasible
/// upgrade with greatest marginal E[tokens]/extra cost; ties use request index
/// then shorter k. Stop when no positive-gain upgrade fits. This is a GREEDY
/// heuristic for nonlinear tables, not an optimal knapsack solver or AdaServe.
/// Tests: b_joint_budget_and_nonoptimal_counterexample.
pub fn allocate_round(
    probabilities: &[Vec<f64>],
    profiles: &[ProfilePoint],
    budget: f64,
) -> Option<Vec<usize>> {
    // BEGIN TODO B5
    assert_eq!(probabilities.len(), profiles.len());
    let mut choices = vec![0; profiles.len()];
    let mut used: f64 = profiles.iter().map(|p| p.verify[0]).sum();
    if used > budget {
        return None;
    }
    loop {
        let mut best: Option<(usize, usize, f64, f64)> = None;
        for (i, row) in profiles.iter().enumerate() {
            let current = choices[i];
            let old_cost = row.verify[current] + row.draft * current as f64;
            let old_gain = committed_moments(&probabilities[i], current).0;
            for k in current + 1..row.verify.len().min(probabilities[i].len() + 1) {
                let cost = row.verify[k] + row.draft * k as f64 - old_cost;
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
