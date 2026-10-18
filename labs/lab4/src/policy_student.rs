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
    todo!("TODO A1")
    // END TODO A1
}

/// A2: Compute the mean and variance of committed tokens from A1, not a constant
/// acceptance shortcut. k=0 yields one correction token and zero variance.
/// Tests: a_moments_and_quantiles.
pub fn committed_moments(conditional: &[f64], k: usize) -> (f64, f64) {
    // BEGIN TODO A2
    todo!("TODO A2")
    // END TODO A2
}

/// A3: Nearest-rank percentile, rank=max(1,ceil(q*n)); empty -> None.
/// Reject nonfinite samples; do not mutate caller data. q=0 is minimum.
/// Tests: a_moments_and_quantiles.
pub fn percentile(samples: &[f64], q: f64) -> Option<f64> {
    // BEGIN TODO A3
    todo!("TODO A3")
    // END TODO A3
}

/// A4: Linear interpolation of per-k costs between strictly increasing context
/// samples. Clamp outside the measured range (report extrapolation separately).
/// All rows have the same nonempty verify width; costs are positive ms.
/// Tests: a_profile_interpolation.
pub fn interpolate(profile: &[ProfilePoint], context: usize) -> ProfilePoint {
    // BEGIN TODO A4
    todo!("TODO A4")
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
    todo!("TODO A5")
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
        todo!("TODO B1")
        // END TODO B1
    }
    /// B2: z == 0: posterior means (s+1)/(n+2) under Beta(1,1).
    /// z > 0: the LOWER end of the Wilson score interval on the (discounted) counts,
    ///   (p + z^2/(2n) - z*sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n),  p = s/n, n = s+f,
    /// and 0 when n == 0. With discounting and drift this is a conservative control
    /// score, NOT a calibrated confidence bound. Tests: b_wilson_numeric_and_ranking.
    pub fn probabilities(&self, z: f64) -> Vec<f64> {
        // BEGIN TODO B2
        todo!("TODO B2")
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
    todo!("TODO B3")
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
        todo!("TODO B4")
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
    todo!("TODO B5")
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
    todo!("TODO B6")
    // END TODO B6
}
