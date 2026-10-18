//! Lab 4 tests. Every test prints one PASS/FAIL line per check and fails at the end,
//! so a single run lists everything that is wrong. Oracles are written here
//! independently of the code under test (enumeration, bisection, shadow replays).
use slo_speculation::*;
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Debug;

struct Checks {
    test: &'static str,
    failures: Vec<String>,
}
impl Checks {
    fn new(test: &'static str) -> Self {
        Self {
            test,
            failures: Vec::new(),
        }
    }
    fn check(&mut self, name: &str, ok: bool, detail: impl FnOnce() -> String) {
        if ok {
            println!("PASS {}/{}", self.test, name);
        } else {
            let line = format!("FAIL {}/{}: {}", self.test, name, detail());
            println!("{line}");
            self.failures.push(line);
        }
    }
    fn close(&mut self, name: &str, got: f64, want: f64, tol: f64) {
        self.check(name, (got - want).abs() <= tol, || {
            format!("expected {want} vs got {got} (tol {tol})")
        });
    }
    fn eq<T: PartialEq + Debug>(&mut self, name: &str, got: T, want: T) {
        let ok = got == want;
        self.check(name, ok, || format!("expected {want:?} vs got {got:?}"));
    }
    fn done(self) {
        if !self.failures.is_empty() {
            panic!(
                "{} check(s) failed in {}:\n{}",
                self.failures.len(),
                self.test,
                self.failures.join("\n")
            );
        }
    }
}

fn profile() -> Vec<ProfilePoint> {
    vec![ProfilePoint {
        context: 0,
        draft: 0.1,
        verify: vec![1.0, 1.1, 1.5, 2.4],
    }]
}
fn req(id: usize, arrival: f64, output: usize, deadline: f64, accept: &[f64]) -> Request {
    Request {
        id,
        arrival,
        output,
        context: 0,
        deadline,
        accept: accept.to_vec(),
    }
}
/// Independent E[N] for conditional acceptance p and k proposals: 1 + sum of prefix products.
fn expected_tokens(p: &[f64], k: usize) -> f64 {
    let (mut e, mut prod) = (1.0, 1.0);
    for &a in &p[..k] {
        prod *= a;
        e += prod;
    }
    e
}
/// Tiny deterministic generator for randomized oracle cases.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ------------------------------------------------------------------- A
#[test]
fn a_distribution_oracle() {
    let mut c = Checks::new("a_distribution_oracle");
    // Enumerate all 2^k accept/reject vectors (including the unobserved suffix) and
    // aggregate by the position of the first rejection.
    for p in [
        vec![0.2, 0.7, 0.4],
        vec![0.9, 0.1, 0.0],
        vec![1.0, 1.0, 0.5],
    ] {
        let k = p.len();
        let mut expected = vec![0.0; k + 1];
        for mask in 0..(1usize << k) {
            let mut weight = 1.0;
            for (j, &prob) in p.iter().enumerate() {
                weight *= if mask & (1 << j) != 0 {
                    prob
                } else {
                    1.0 - prob
                };
            }
            let prefix = (0..k).take_while(|&j| mask & (1 << j) != 0).count();
            expected[prefix] += weight;
        }
        let got = committed_distribution(&p, k);
        c.eq(&format!("len p={p:?}"), got.len(), k + 1);
        for (n, (a, b)) in got.iter().zip(&expected).enumerate() {
            c.close(&format!("P(N={}) p={p:?}", n + 1), *a, *b, 1e-12);
        }
    }
    c.eq("k=0", committed_distribution(&[], 0), vec![1.0]);
    c.eq(
        "prefix k=1 of longer vector",
        committed_distribution(&[0.3, 0.9], 1).len(),
        2,
    );
    c.done();
}
#[test]
fn a_moments_and_quantiles() {
    let mut c = Checks::new("a_moments_and_quantiles");
    let (mean, var) = committed_moments(&[0.5, 0.25], 2);
    c.close("mean [0.5,0.25]", mean, 1.625, 1e-12); // 0.5*1 + 0.375*2 + 0.125*3
    c.close("var [0.5,0.25]", var, 0.484375, 1e-12); // 3.125 - 1.625^2
    let (mean, var) = committed_moments(&[0.8, 0.5, 0.25], 3);
    c.close("mean lecture example", mean, 2.3, 1e-12);
    c.close("var lecture example", var, 0.81, 1e-12);
    let (mean, var) = committed_moments(&[0.7], 0);
    c.close("k=0 mean", mean, 1.0, 0.0);
    c.close("k=0 var", var, 0.0, 0.0);
    c.eq("empty", percentile(&[], 0.5), None);
    c.eq("p75 of 4", percentile(&[4., 1., 3., 2.], 0.75), Some(3.));
    c.eq(
        "p50 of 4 (nearest rank)",
        percentile(&[4., 1., 3., 2.], 0.5),
        Some(2.),
    );
    c.eq("q=0 is min", percentile(&[4., 1.], 0.), Some(1.));
    c.eq("q=1 is max", percentile(&[4., 1.], 1.), Some(4.));
    let data = vec![5., 3., 9.];
    let _ = percentile(&data, 0.5);
    c.eq("input not mutated", data, vec![5., 3., 9.]);
    c.done();
}
#[test]
fn a_profile_interpolation() {
    let mut c = Checks::new("a_profile_interpolation");
    let p = vec![
        ProfilePoint {
            context: 100,
            draft: 1.,
            verify: vec![2., 4.],
        },
        ProfilePoint {
            context: 300,
            draft: 3.,
            verify: vec![6., 10.],
        },
    ];
    c.eq("mid verify", interpolate(&p, 200).verify, vec![4., 7.]);
    c.eq("mid draft", interpolate(&p, 200).draft, 2.);
    c.eq("quarter verify", interpolate(&p, 150).verify, vec![3., 5.5]);
    c.eq("clamp low", interpolate(&p, 1).verify, vec![2., 4.]);
    c.eq("clamp high", interpolate(&p, 999).verify, vec![6., 10.]);
    c.eq("exact knot", interpolate(&p, 300).draft, 3.);
    c.done();
}
#[test]
fn a_calibration_grouping_and_missing() {
    let mut c = Checks::new("a_calibration_grouping_and_missing");
    let s = |context, k, elapsed| TimingSample {
        context,
        k,
        elapsed,
    };
    let samples = vec![
        s(64, 0, 9.),
        s(32, 1, 4.),
        s(32, 0, 3.),
        s(32, 0, 1.),
        s(32, 0, 2.),
        s(32, 1, 6.),
        s(64, 1, 11.),
    ];
    match calibrate(&samples, 1, 0.2) {
        Ok(table) => {
            c.eq(
                "rows sorted by context",
                table.iter().map(|r| r.context).collect::<Vec<_>>(),
                vec![32, 64],
            );
            c.eq(
                "median per k (nearest rank)",
                table[0].verify.clone(),
                vec![2., 4.],
            );
            c.eq("second context", table[1].verify.clone(), vec![9., 11.]);
            c.eq("draft copied", table[0].draft, 0.2);
        }
        Err(e) => c.check("calibrate ok", false, || format!("unexpected error {e}")),
    }
    c.check(
        "missing k is an error",
        calibrate(&samples, 2, 0.2).is_err(),
        || "Ok for kmax=2".into(),
    );
    c.check("empty is an error", calibrate(&[], 1, 0.2).is_err(), || {
        "Ok".into()
    });
    c.check(
        "nonpositive sample rejected",
        calibrate(&[s(0, 0, 0.)], 0, 0.2).is_err(),
        || "Ok".into(),
    );
    c.done();
}

// ------------------------------------------------------------------- B
#[test]
fn b_censored_observation_and_decay() {
    let mut c = Checks::new("b_censored_observation_and_decay");
    let mut e = Estimator::new(4, 0.5);
    e.observe(1, 4);
    c.eq(
        "success after (1,4)",
        e.success.clone(),
        vec![1., 0., 0., 0.],
    );
    c.eq(
        "failure after (1,4): suffix unobserved",
        e.failure.clone(),
        vec![0., 1., 0., 0.],
    );
    e.observe(4, 4);
    c.eq(
        "success after (4,4)",
        e.success.clone(),
        vec![1.5, 1., 1., 1.],
    );
    c.eq(
        "failure after (4,4): no failure when all accepted",
        e.failure.clone(),
        vec![0., 0.5, 0., 0.],
    );
    e.observe(0, 0);
    c.eq("k=0 still decays", e.failure[1], 0.25);
    c.eq(
        "k=0 adds nothing",
        e.success.clone(),
        vec![0.75, 0.5, 0.5, 0.5],
    );
    e.observe(2, 3);
    c.eq(
        "(2,3) failure at depth 2",
        e.failure.clone(),
        vec![0., 0.125, 1., 0.],
    );
    c.done();
}

/// Wilson score interval end by bisection on (p_hat - x)^2 = z^2 x (1-x) / n.
fn wilson_bisect(s: f64, f: f64, z: f64, upper: bool) -> f64 {
    let n = s + f;
    let ph = s / n;
    let g = |x: f64| (ph - x) * (ph - x) - z * z * x * (1.0 - x) / n;
    let (mut lo, mut hi) = if upper { (ph, 1.0) } else { (0.0, ph) };
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        // lower end: g >= 0 left of the root; upper end: g <= 0 left of the root
        if (g(mid) >= 0.0) != upper {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}
#[test]
fn b_wilson_numeric_and_ranking() {
    let mut c = Checks::new("b_wilson_numeric_and_ranking");
    let cases = [
        (1., 0.),
        (0., 1.),
        (3., 1.),
        (2.5, 1.5),
        (0.97, 0.03),
        (100., 0.),
        (45.3, 12.7),
        (7., 7.),
    ];
    for z in [1.0, 1.96] {
        let e = Estimator {
            success: cases.iter().map(|c| c.0).collect(),
            failure: cases.iter().map(|c| c.1).collect(),
            decay: 1.0,
        };
        let got = e.probabilities(z);
        for (i, &(s, f)) in cases.iter().enumerate() {
            c.close(
                &format!("lower z={z} s={s} f={f}"),
                got[i],
                wilson_bisect(s, f, z, false),
                1e-9,
            );
        }
    }
    let e = Estimator {
        success: vec![0., 1., 100., 2.],
        failure: vec![0., 0., 0., 1.],
        decay: 1.,
    };
    let low = e.probabilities(1.96);
    c.eq("zero evidence -> 0", low[0], 0.);
    c.check("1/1 below 100/100", low[1] < low[2] && low[2] < 1., || {
        format!("{low:?}")
    });
    let mean = e.probabilities(0.);
    c.close("Beta(1,1) zero evidence", mean[0], 0.5, 1e-12);
    c.close("Beta(1,1) 1/1", mean[1], 2. / 3., 1e-12);
    c.close("Beta(1,1) 2/3", mean[3], 3. / 5., 1e-12);
    c.done();
}
#[test]
fn b_choice_exhaustive_and_infeasible() {
    let mut c = Checks::new("b_choice_exhaustive_and_infeasible");
    let row = &profile()[0];
    let mut rng = Lcg(7);
    let mut vectors = vec![
        vec![0.; 3],
        vec![0.4; 3],
        vec![1.; 3],
        vec![0.95, 0.6, 0.2],
        vec![0.3, 0.9, 0.99],
    ];
    for _ in 0..20 {
        vectors.push((0..3).map(|_| rng.next()).collect());
    }
    let mut mismatches = 0;
    let mut first = String::new();
    for p in &vectors {
        for remaining in 0..=5 {
            for bound in [0.1, 1.0, 1.2, 1.6, 2., 10.] {
                let mut oracle = None;
                let mut rate = -1.;
                if remaining > 0 {
                    for k in 0..=3.min(remaining - 1) {
                        let cost = row.draft * k as f64 + row.verify[k];
                        let r = expected_tokens(p, k) / cost;
                        if cost <= bound && r > rate + 1e-12 {
                            oracle = Some(k);
                            rate = r;
                        }
                    }
                }
                let got = choose_k(p, row, bound, remaining);
                if got != oracle {
                    mismatches += 1;
                    if first.is_empty() {
                        first = format!("p={p:?} remaining={remaining} bound={bound}: expected {oracle:?} vs got {got:?}");
                    }
                }
            }
        }
    }
    c.check(
        "matches exhaustive oracle on 900 cases",
        mismatches == 0,
        || format!("{mismatches} mismatches, first {first}"),
    );
    // Exact ties: draft 0 and verify[k] = k+1 with p = 1 give rate 1 for every k.
    let flat = ProfilePoint {
        context: 0,
        draft: 0.,
        verify: vec![1., 2., 3., 4.],
    };
    c.eq(
        "exact tie -> shortest k",
        choose_k(&[1.; 3], &flat, 10., 10),
        Some(0),
    );
    let half = ProfilePoint {
        context: 0,
        draft: 0.5,
        verify: vec![1., 1., 1.5, 3.],
    };
    // rates: k0 1/1, k1 2/1.5, k2 3/2.5 = 1.2, k3 4/4.5 -> k1 strictly best
    c.eq("strict best", choose_k(&[1.; 3], &half, 10., 10), Some(1));
    c.eq("infeasible bound", choose_k(&[0.9; 3], row, 0.9, 10), None);
    c.eq("remaining=0", choose_k(&[0.9; 3], row, 10., 0), None);
    c.done();
}

/// Independent Page-Hinkley oracle with the contract's update order.
fn ph_oracle(values: &[f64], delta: f64, threshold: f64) -> Vec<usize> {
    let (mut mean, mut count, mut cum) = (0.0, 0usize, 0.0);
    let mut hits = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if count == 0 {
            mean = v;
        }
        cum = f64::max(0.0, cum + mean - v - delta);
        count += 1;
        mean = mean + (v - mean) / count as f64;
        if cum > threshold {
            hits.push(i);
            count = 0;
            cum = 0.0;
            mean = v;
        }
    }
    hits
}
#[test]
fn b_drift_exact_events() {
    let mut c = Checks::new("b_drift_exact_events");
    // Hand trace (delta 0.05, threshold 2.1): 20 x 0.9 keeps cum = 0 and mean = 0.9.
    // Low values 0.1 use the OLD mean: cum 0.75 (mean -> 0.8619), 1.4619 (mean -> 0.8273),
    // 2.1392 > 2.1 -> exactly one trigger at the 3rd low value (step 22); afterwards
    // mean = 0.1 and nothing more. Updating the mean first would give 2.035 and fire at 23.
    let mut d = Drift::default();
    let mut hits = Vec::new();
    let values: Vec<f64> = std::iter::repeat_n(0.9, 20)
        .chain(std::iter::repeat_n(0.1, 20))
        .collect();
    for (i, &v) in values.iter().enumerate() {
        if d.update(v, 0.05, 2.1) {
            hits.push(i);
        }
    }
    c.eq("trigger steps (hand)", hits.clone(), vec![22]);
    c.eq("reset count", d.resets, 1);
    c.close("mean after reset window", d.mean, 0.1, 1e-12);
    // alternating blocks against the independent oracle
    let mut rng = Lcg(11);
    let mut values = Vec::new();
    for block in 0..8 {
        let level = if block % 2 == 0 { 0.85 } else { 0.15 };
        for _ in 0..15 {
            values.push((level + 0.2 * (rng.next() - 0.5)).clamp(0.0, 1.0));
        }
    }
    let mut d = Drift::default();
    let got: Vec<usize> = (0..values.len())
        .filter(|&i| d.update(values[i], 0.05, 1.5))
        .collect();
    let want = ph_oracle(&values, 0.05, 1.5);
    c.check("oracle has events", want.len() >= 3, || format!("{want:?}"));
    c.eq("trigger steps (oracle)", got, want.clone());
    c.eq("resets == events", d.resets, want.len());
    let mut rising = Drift::default();
    let fired = [0.1, 0.2, 0.4, 0.8, 0.9, 1.0]
        .iter()
        .filter(|&&p| rising.update(p, 0.05, 2.))
        .count();
    c.eq("rising never triggers", fired, 0);
    c.done();
}

fn total_cost(rows: &[ProfilePoint], ks: &[usize]) -> f64 {
    rows.iter()
        .zip(ks)
        .map(|(r, &k)| r.draft * k as f64 + r.verify[k])
        .sum()
}
fn total_value(ps: &[Vec<f64>], ks: &[usize]) -> f64 {
    ps.iter().zip(ks).map(|(p, &k)| expected_tokens(p, k)).sum()
}
/// Exhaustive optimum: lexicographically first vector whose value beats the best by > 1e-9.
fn exhaustive(ps: &[Vec<f64>], rows: &[ProfilePoint], budget: f64) -> Option<(Vec<usize>, f64)> {
    let limits: Vec<usize> = ps
        .iter()
        .zip(rows)
        .map(|(p, r)| p.len().min(r.verify.len() - 1))
        .collect();
    let mut ks = vec![0; ps.len()];
    let mut best: Option<(Vec<usize>, f64)> = None;
    loop {
        if total_cost(rows, &ks) <= budget + 1e-9 {
            let v = total_value(ps, &ks);
            if best.as_ref().is_none_or(|b| v > b.1 + 1e-9) {
                best = Some((ks.clone(), v));
            }
        }
        let mut i = ps.len();
        loop {
            if i == 0 {
                return best;
            }
            i -= 1;
            if ks[i] < limits[i] {
                ks[i] += 1;
                for k in ks.iter_mut().skip(i + 1) {
                    *k = 0;
                }
                break;
            }
        }
    }
}
fn random_instance(rng: &mut Lcg) -> (Vec<Vec<f64>>, Vec<ProfilePoint>, f64) {
    let n = 1 + (rng.next() * 3.0) as usize;
    let mut ps = Vec::new();
    let mut rows = Vec::new();
    for _ in 0..n {
        let k = 1 + (rng.next() * 3.0) as usize;
        ps.push((0..k).map(|_| (rng.next() * 10.0).round() / 10.0).collect());
        let mut verify = vec![1.0];
        for _ in 0..k {
            let last = *verify.last().unwrap();
            verify.push(last + (1.0 + (rng.next() * 8.0).floor()) / 10.0);
        }
        rows.push(ProfilePoint {
            context: 0,
            draft: (rng.next() * 3.0).floor() / 10.0,
            verify,
        });
    }
    let budget = n as f64 + (rng.next() * 30.0).floor() / 10.0;
    (ps, rows, budget)
}
/// Test-side re-implementation of the B5 contract (independent code, same rules).
fn greedy_oracle(ps: &[Vec<f64>], rows: &[ProfilePoint], budget: f64) -> Option<Vec<usize>> {
    let mut ks = vec![0; ps.len()];
    let mut used = total_cost(rows, &ks);
    if used > budget + 1e-12 {
        return None;
    }
    loop {
        let mut best: Option<(usize, usize, f64, f64)> = None;
        for i in 0..ps.len() {
            for k in ks[i] + 1..=ps[i].len().min(rows[i].verify.len() - 1) {
                let extra =
                    rows[i].draft * (k - ks[i]) as f64 + rows[i].verify[k] - rows[i].verify[ks[i]];
                let gain = expected_tokens(&ps[i], k) - expected_tokens(&ps[i], ks[i]);
                if extra > 0. && gain > 0. && used + extra <= budget + 1e-12 {
                    let r = gain / extra;
                    if best.is_none_or(|b| r > b.3 + 1e-12) {
                        best = Some((i, k, extra, r));
                    }
                }
            }
        }
        let Some((i, k, extra, _)) = best else {
            return Some(ks);
        };
        ks[i] = k;
        used += extra;
    }
}
#[test]
fn b_greedy_trace_feasible_and_bounded() {
    let mut c = Checks::new("b_greedy_trace_feasible_and_bounded");
    // Hand trace. r0: verify [1,2], p [0.9] -> 0->1 gain 0.9 / cost 1.
    // r1: verify [1,1.5,2.5], p [0.6,0.5] -> 0->1 ratio 0.6/0.5 = 1.2, 0->2 ratio 0.9/1.5.
    // Budget 3.5: step 1 r1->1 (used 2.5), step 2 r0->1 (used 3.5), r1 1->2 no longer fits.
    let rows = vec![
        ProfilePoint {
            context: 0,
            draft: 0.,
            verify: vec![1., 2.],
        },
        ProfilePoint {
            context: 0,
            draft: 0.,
            verify: vec![1., 1.5, 2.5],
        },
    ];
    let ps = vec![vec![0.9], vec![0.6, 0.5]];
    c.eq(
        "hand trace",
        allocate_round(&ps, &rows, 3.5),
        Some(vec![1, 1]),
    );
    c.eq(
        "cheapest upgrade only",
        allocate_round(&ps, &rows, 2.6),
        Some(vec![0, 1]),
    );
    c.eq(
        "only base fits",
        allocate_round(&ps, &rows, 2.4),
        Some(vec![0, 0]),
    );
    c.eq("base infeasible", allocate_round(&ps, &rows, 1.9), None);
    c.eq("empty", allocate_round(&[], &[], 0.), Some(vec![]));
    // Ties: all ratios are 1. Index first, then shorter k: r0->1, r0->2, then budget is gone.
    let rows = vec![
        ProfilePoint {
            context: 0,
            draft: 0.,
            verify: vec![1., 2., 3.],
        },
        ProfilePoint {
            context: 0,
            draft: 0.,
            verify: vec![1., 2.],
        },
    ];
    c.eq(
        "tie: lower index, shorter k",
        allocate_round(&[vec![1., 1.], vec![1.]], &rows, 4.),
        Some(vec![2, 0]),
    );
    // Best ratio, not first qualifying: r0 ratio 0.3/0.5, r1 ratio 0.9/0.5; one fits.
    let rows2 = vec![
        ProfilePoint {
            context: 0,
            draft: 0.,
            verify: vec![1., 1.5]
        };
        2
    ];
    c.eq(
        "best ratio wins",
        allocate_round(&[vec![0.3], vec![0.9]], &rows2, 2.5),
        Some(vec![0, 1]),
    );
    // Randomized: feasible, never above the exhaustive optimum, and maximal (no single
    // positive-gain upgrade still fits). Greedy is NOT required to be optimal.
    let mut rng = Lcg(3);
    let (mut infeasible, mut above, mut not_maximal, mut gaps, mut contract) = (0, 0, 0, 0, 0);
    for _ in 0..300 {
        let (ps, rows, budget) = random_instance(&mut rng);
        let opt = exhaustive(&ps, &rows, budget);
        contract +=
            (allocate_round(&ps, &rows, budget) != greedy_oracle(&ps, &rows, budget)) as usize;
        match (allocate_round(&ps, &rows, budget), opt) {
            (None, None) => {}
            (Some(ks), Some((_, best))) => {
                let cost = total_cost(&rows, &ks);
                if cost > budget + 1e-9 {
                    infeasible += 1;
                }
                let v = total_value(&ps, &ks);
                if v > best + 1e-9 {
                    above += 1;
                }
                if v < best - 1e-9 {
                    gaps += 1;
                }
                for i in 0..ks.len() {
                    for k in ks[i] + 1..=ps[i].len().min(rows[i].verify.len() - 1) {
                        let extra = rows[i].draft * (k - ks[i]) as f64 + rows[i].verify[k]
                            - rows[i].verify[ks[i]];
                        let gain = expected_tokens(&ps[i], k) - expected_tokens(&ps[i], ks[i]);
                        if gain > 1e-12 && extra > 0. && cost + extra <= budget - 1e-9 {
                            not_maximal += 1;
                        }
                    }
                }
            }
            (got, want) => {
                infeasible += 1;
                println!("  feasibility mismatch: greedy {got:?} vs exhaustive {want:?}");
            }
        }
    }
    c.eq(
        "random: equals the contract greedy (test-side re-implementation)",
        contract,
        0,
    );
    c.eq(
        "random: within budget and None iff infeasible",
        infeasible,
        0,
    );
    c.eq("random: never above optimum", above, 0);
    c.eq("random: maximal (no fitting upgrade left)", not_maximal, 0);
    println!(
        "  note: greedy was strictly suboptimal on {gaps}/300 random instances (expected > 0)"
    );
    c.done();
}
#[test]
fn b_exact_matches_exhaustive() {
    let mut c = Checks::new("b_exact_matches_exhaustive");
    let mut rng = Lcg(5);
    let (mut value_mismatch, mut vector_mismatch, mut first) = (0, 0, String::new());
    for _ in 0..300 {
        let (ps, rows, budget) = random_instance(&mut rng);
        let got = allocate_round_exact(&ps, &rows, budget, 0.1);
        let want = exhaustive(&ps, &rows, budget);
        match (&got, &want) {
            (None, None) => {}
            (Some(ks), Some((wk, wv))) => {
                if (total_value(&ps, ks) - wv).abs() > 1e-9 || total_cost(&rows, ks) > budget + 1e-9
                {
                    value_mismatch += 1;
                } else if ks != wk {
                    vector_mismatch += 1;
                }
                if (ks != wk) && first.is_empty() {
                    first = format!("budget {budget}: expected {wk:?} vs got {ks:?}");
                }
            }
            _ => {
                value_mismatch += 1;
                if first.is_empty() {
                    first = format!("budget {budget}: expected {want:?} vs got {got:?}");
                }
            }
        }
    }
    c.check(
        "optimal value on 300 random instances",
        value_mismatch == 0,
        || format!("{value_mismatch} wrong, first {first}"),
    );
    c.check("lexicographic tie rule", vector_mismatch == 0, || {
        format!("{vector_mismatch} wrong, first {first}")
    });
    // Counterexample for the greedy (documented in the README): after the base, 1.6 is
    // left. Upgrade A: gain 0.6 for 1.0; upgrade B: gain 0.9 for 1.6. B alone is optimal.
    let rows = vec![
        ProfilePoint {
            context: 0,
            draft: 0.,
            verify: vec![1., 2.],
        },
        ProfilePoint {
            context: 0,
            draft: 0.,
            verify: vec![1., 2.6],
        },
    ];
    let ps = vec![vec![0.6], vec![0.9]];
    // 2.6 / 0.1 = 26.000000000000004 in binary floats: without the tolerance the exact
    // optimum would not fit 36 ticks.
    c.eq(
        "counterexample exact",
        allocate_round_exact(&ps, &rows, 3.6, 0.1),
        Some(vec![0, 1]),
    );
    let greedy = allocate_round(&ps, &rows, 3.6)
        .map(|k| total_value(&ps, &k))
        .unwrap_or(0.);
    c.check(
        "greedy <= exact on the counterexample",
        greedy <= 2.9 + 1e-9,
        || format!("greedy value {greedy}"),
    );
    c.eq(
        "infeasible base",
        allocate_round_exact(&ps, &rows, 1.5, 0.1),
        None,
    );
    c.eq(
        "tie -> lexicographically smallest",
        allocate_round_exact(
            &[vec![1.], vec![1.]],
            &[rows[0].clone(), rows[0].clone()],
            3.,
            0.5,
        ),
        Some(vec![0, 1]),
    );
    c.done();
}

// ------------------------------------------------------------------- C
#[test]
fn c_commit_cap_and_timestamps() {
    let mut c = Checks::new("c_commit_cap_and_timestamps");
    let mut s = State::new(req(0, 1., 3, 10., &[1.]));
    c.eq("first commit size", s.commit(0, 2.), 1);
    c.eq("capped commit size", s.commit(9, 4.), 2);
    c.eq("timestamps", s.committed_at.clone(), vec![2., 4., 4.]);
    c.eq("first", s.first, Some(2.));
    c.eq("finish", s.finish, Some(4.));
    c.eq("rounds", s.rounds, 2);
    let mut t = State::new(req(1, 0., 5, 10., &[1.]));
    t.commit(2, 1.);
    c.eq("accepted+1 tokens", t.emitted, 3);
    c.eq("not finished", t.finish, None);
    c.done();
}
#[test]
fn c_arrivals_once() {
    let mut c = Checks::new("c_arrivals_once");
    let s = vec![
        State::new(req(0, 0., 1, 5., &[1.])),
        State::new(req(1, 0., 1, 5., &[1.])),
        State::new(req(2, 3., 1, 5., &[1.])),
    ];
    let mut q = VecDeque::new();
    let mut cursor = 0;
    enqueue(&s, &mut cursor, 0., &mut q);
    enqueue(&s, &mut cursor, 0., &mut q);
    c.eq(
        "same-time arrivals once",
        q.clone(),
        VecDeque::from(vec![0, 1]),
    );
    c.eq("cursor", cursor, 2);
    enqueue(&s, &mut cursor, 2.9, &mut q);
    c.eq("future arrival waits", q.len(), 2);
    enqueue(&s, &mut cursor, 3., &mut q);
    c.eq("arrival at now", q.clone(), VecDeque::from(vec![0, 1, 2]));
    c.done();
}
#[test]
fn c_hand_timeline() {
    let mut c = Checks::new("c_hand_timeline");
    let s = simulate(
        vec![req(0, 0., 2, 10., &[1.; 3]), req(1, 0.5, 1, 10., &[1.; 3])],
        &profile(),
        &mut Fixed(0),
        &Synthetic { seed: 1 },
        2.,
    );
    c.eq("r0 commits", s.states[0].committed_at.clone(), vec![1., 3.]);
    c.eq("r1 commits", s.states[1].committed_at.clone(), vec![2.]);
    c.eq("busy", s.busy, 3.);
    c.eq(
        "round order",
        s.rounds.iter().map(|r| r.request).collect::<Vec<_>>(),
        vec![0, 1, 0],
    );
    c.eq(
        "idle jump",
        simulate(
            vec![req(0, 5., 1, 10., &[1.])],
            &profile(),
            &mut Fixed(0),
            &Synthetic { seed: 1 },
            2.,
        )
        .states[0]
            .finish,
        Some(6.),
    );
    c.done();
}
#[test]
fn c_infeasible_counts_all_policies() {
    let mut c = Checks::new("c_infeasible_counts_all_policies");
    let env = Synthetic { seed: 9 };
    // Fixed(2) with bound 1.2: round 1 costs 0.2+1.5 = 1.7 > 1.2 (counted although Fixed
    // ignores the bound), commits 3; round 2 has remaining 2 -> k=1 costs 1.2 (fits).
    let s = simulate(
        vec![req(0, 0., 5, 99., &[1.; 3])],
        &profile(),
        &mut Fixed(2),
        &env,
        1.2,
    );
    c.eq(
        "Fixed(2) ks",
        s.rounds.iter().map(|r| r.k).collect::<Vec<_>>(),
        vec![2, 1],
    );
    c.eq("Fixed(2) infeasible", s.infeasible, 1);
    let s = simulate(
        vec![req(0, 0., 5, 99., &[1.; 3])],
        &profile(),
        &mut Fixed(0),
        &env,
        1.2,
    );
    c.eq("Fixed(0) feasible", s.infeasible, 0);
    // Deadline-driven: two 2-token requests at t=0, deadline 2.5, k=0, 1 ms rounds.
    // Rounds start at 0,1,2,3 with budgets min(5, 2.5-now) = 2.5, 2.5, 0.5, -0.5.
    let pair = || vec![req(0, 0., 2, 2.5, &[0.; 3]), req(1, 0., 2, 2.5, &[0.; 3])];
    let s = simulate(pair(), &profile(), &mut Fixed(0), &env, 5.);
    c.eq(
        "deadline infeasible rounds",
        s.rounds.iter().map(|r| r.infeasible).collect::<Vec<_>>(),
        vec![false, false, true, true],
    );
    c.eq("deadline infeasible count", s.infeasible, 2);
    let s = simulate(
        pair(),
        &profile(),
        &mut PerPosition::new(3, 0.97, 0.),
        &env,
        5.,
    );
    c.eq(
        "Mean: None -> k=0 and counted",
        (
            s.infeasible,
            s.rounds.iter().all(|r| r.k == 0 || !r.infeasible),
        ),
        (2, true),
    );
    let s = simulate(
        vec![req(0, 0., 3, 99., &[1.; 3])],
        &profile(),
        &mut PerPosition::new(3, 0.97, 0.),
        &env,
        0.5,
    );
    c.eq(
        "bound below T(0): every round infeasible",
        (s.infeasible, s.rounds.len()),
        (3, 3),
    );
    for r in &s.rounds {
        c.eq("fallback executes k=0", r.k, 0);
    }
    c.done();
}

/// Shadow replay of a Mean-policy simulation from its round log, using only test-side
/// code: discounted counts, Beta(1,1) means, exhaustive k choice, CRN draws, PH resets.
fn shadow_mean(
    c: &mut Checks,
    label: &str,
    requests: &[Request],
    sim: &Simulation,
    rows: &[ProfilePoint],
    bound: f64,
    seed: u64,
) {
    let kmax = rows[0].verify.len() - 1;
    let (mut s, mut f) = (vec![0.0; kmax], vec![0.0; kmax]);
    let mut emitted: BTreeMap<usize, usize> = BTreeMap::new();
    let by_id: BTreeMap<usize, &Request> = requests.iter().map(|r| (r.id, r)).collect();
    let (mut ph_mean, mut ph_count, mut ph_cum, mut resets) = (0.0, 0usize, 0.0, 0);
    let mut bad = Vec::new();
    let mut previous_end = 0.0;
    for (n, r) in sim.rounds.iter().enumerate() {
        let q = by_id[&r.request];
        let e = *emitted.get(&q.id).unwrap_or(&0);
        let remaining = q.output - e;
        let budget = bound.min(q.deadline - r.start);
        let p: Vec<f64> = (0..kmax)
            .map(|j| (s[j] + 1.0) / (s[j] + f[j] + 2.0))
            .collect();
        let mut best: Option<(usize, f64)> = None;
        for k in 0..=kmax.min(remaining - 1) {
            let cost = rows[0].draft * k as f64 + rows[0].verify[k];
            let rate = expected_tokens(&p, k) / cost;
            if cost <= budget && best.is_none_or(|b| rate > b.1 + 1e-12) {
                best = Some((k, rate));
            }
        }
        let k = best.map(|b| b.0).unwrap_or(0);
        let accepted = (0..k)
            .take_while(|&d| draw(seed, q.id, e, d) < q.accept[d])
            .count();
        let cost = rows[0].draft * k as f64 + rows[0].verify[k];
        let checks = [
            (
                r.k == k,
                format!("round {n}: k expected {k} vs got {}", r.k),
            ),
            (
                r.accepted == accepted,
                format!(
                    "round {n}: accepted expected {accepted} vs got {}",
                    r.accepted
                ),
            ),
            (
                (r.end - r.start - cost).abs() < 1e-9,
                format!("round {n}: duration {} vs cost {cost}", r.end - r.start),
            ),
            (
                r.start >= previous_end - 1e-9 && r.start >= q.arrival,
                format!("round {n}: start {} before {previous_end}/arrival", r.start),
            ),
            (
                r.infeasible == (best.is_none() || cost > budget + 1e-9),
                format!("round {n}: infeasible flag {}", r.infeasible),
            ),
            (
                r.committed == (accepted + 1).min(remaining),
                format!("round {n}: committed {}", r.committed),
            ),
        ];
        for (ok, msg) in checks {
            if !ok {
                bad.push(msg);
            }
        }
        previous_end = r.end;
        for j in 0..kmax {
            s[j] *= 0.97;
            f[j] *= 0.97;
        }
        for x in s.iter_mut().take(accepted) {
            *x += 1.0;
        }
        if accepted < k {
            f[accepted] += 1.0;
        }
        let mut reset = false;
        if k > 0 {
            let v = accepted as f64 / k as f64;
            if ph_count == 0 {
                ph_mean = v;
            }
            ph_cum = f64::max(0.0, ph_cum + ph_mean - v - DRIFT_DELTA);
            ph_count += 1;
            ph_mean += (v - ph_mean) / ph_count as f64;
            if ph_cum > DRIFT_THRESHOLD {
                reset = true;
                resets += 1;
                ph_count = 0;
                ph_cum = 0.0;
                ph_mean = v;
                s = vec![0.0; kmax];
                f = vec![0.0; kmax];
            }
        }
        if r.reset != reset {
            bad.push(format!(
                "round {n}: reset expected {reset} vs got {}",
                r.reset
            ));
        }
        emitted.insert(q.id, e + r.committed);
    }
    c.check(
        &format!("{label}: every round matches the shadow controller"),
        bad.is_empty(),
        || {
            format!(
                "{} mismatches, first: {}",
                bad.len(),
                bad.first().cloned().unwrap_or_default()
            )
        },
    );
    c.eq(&format!("{label}: resets"), sim.resets, resets);
    c.eq(
        &format!("{label}: infeasible"),
        sim.infeasible,
        sim.rounds.iter().filter(|r| r.infeasible).count(),
    );
}
fn decaying(a0: f64, ratio: f64) -> Vec<f64> {
    (0..8).map(|j| a0 * ratio.powi(j)).collect()
}
#[test]
fn c_mean_policy_shadow_replay() {
    let mut c = Checks::new("c_mean_policy_shadow_replay");
    let requests: Vec<Request> = (0..12)
        .map(|id| Request {
            id,
            arrival: id as f64 * 1.5,
            output: 6 + id % 5,
            context: 0,
            deadline: id as f64 * 1.5 + 30.0,
            accept: decaying(0.95, 0.8),
        })
        .collect();
    let sim = simulate(
        requests.clone(),
        &profile(),
        &mut PerPosition::new(3, 0.97, 0.),
        &Synthetic { seed: 4 },
        3.0,
    );
    c.check(
        "several k values used",
        sim.rounds
            .iter()
            .map(|r| r.k)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            >= 2,
        || "only one k".into(),
    );
    shadow_mean(&mut c, "decaying", &requests, &sim, &profile(), 3.0, 4);
    c.done();
}
#[test]
fn c_drift_reset_replay() {
    let mut c = Checks::new("c_drift_reset_replay");
    // 6 requests that always accept, then 10 that never accept: the per-round accepted/k
    // ratio drops from 1 to 0 and Page-Hinkley must reset the policy.
    let requests: Vec<Request> = (0..16)
        .map(|id| Request {
            id,
            arrival: id as f64 * 20.0,
            output: 12,
            context: 0,
            deadline: id as f64 * 20.0 + 100.0,
            accept: if id < 6 { vec![1.0; 8] } else { vec![0.0; 8] },
        })
        .collect();
    let sim = simulate(
        requests.clone(),
        &profile(),
        &mut PerPosition::new(3, 0.97, 0.),
        &Synthetic { seed: 2 },
        5.0,
    );
    c.check("at least one reset", sim.resets >= 1, || {
        format!("resets {}", sim.resets)
    });
    c.eq(
        "reset flags == resets",
        sim.rounds.iter().filter(|r| r.reset).count(),
        sim.resets,
    );
    shadow_mean(&mut c, "phase change", &requests, &sim, &profile(), 5.0, 2);
    c.done();
}
#[test]
fn c_per_position_beats_pooled_iid() {
    let mut c = Checks::new("c_per_position_beats_pooled_iid");
    // Every round accepts exactly 2 (a = [1, 1, 0, ...]): non-IID by construction.
    // Per-position estimates learn a_3 = 0 and settle at k = 2; the pooled IID alpha
    // never sees a failure at k = 2 and keeps paying for longer proposals.
    let rows = vec![ProfilePoint {
        context: 0,
        draft: 0.1,
        verify: vec![1.0, 1.05, 1.1, 1.2, 1.4],
    }];
    let requests: Vec<Request> = (0..10)
        .map(|id| Request {
            id,
            arrival: id as f64 * 40.0,
            output: 40,
            context: 0,
            deadline: 1e9,
            accept: vec![1.0, 1.0, 0.0, 0.0, 0.0],
        })
        .collect();
    let env = Synthetic { seed: 1 };
    let per = simulate(
        requests.clone(),
        &rows,
        &mut PerPosition::new(4, 0.97, 0.),
        &env,
        10.0,
    );
    let pooled = simulate(
        requests.clone(),
        &rows,
        &mut Pooled {
            estimator: Estimator::new(4, 0.97),
        },
        &env,
        10.0,
    );
    let tail = |s: &Simulation| {
        s.rounds
            .iter()
            .rev()
            .take(40)
            .filter(|r| r.k == 2 || r.committed < 3)
            .count()
    };
    c.check("per-position settles at k=2", tail(&per) >= 36, || {
        format!("{} of the last 40 rounds", tail(&per))
    });
    c.check("per-position is faster", per.busy < pooled.busy, || {
        format!("per {} vs pooled {}", per.busy, pooled.busy)
    });
    c.done();
}
#[test]
fn c_conservation_empty_and_overload() {
    let mut c = Checks::new("c_conservation_empty_and_overload");
    // Explicit negative baseline: a zero-evidence lower bound never explores. For this
    // increasing-cost profile it is exactly ordinary k=0 decoding.
    let fresh = vec![req(0, 0., 5, 100., &[1.; 3]), req(1, 0., 4, 100., &[1.; 3])];
    let env = Synthetic { seed: 3 };
    let plain = simulate(fresh.clone(), &profile(), &mut Fixed(0), &env, 2.);
    let no_probe = simulate(
        fresh,
        &profile(),
        &mut PerPosition::new(3, 0.97, 1.),
        &env,
        2.,
    );
    for (a, b) in plain.states.iter().zip(&no_probe.states) {
        c.eq(
            "no_probe == k0 timeline",
            a.committed_at.clone(),
            b.committed_at.clone(),
        );
    }
    c.eq(
        "empty",
        summarize(&simulate(vec![], &profile(), &mut Fixed(1), &env, 1.)).requests,
        0,
    );
    let mut policies: Vec<Box<dyn Policy>> = vec![
        Box::new(Fixed(0)),
        Box::new(Fixed(3)),
        Box::new(PerPosition::new(3, 0.97, 0.)),
        Box::new(PerPosition::new(3, 0.97, 1.)),
        Box::new(Pooled {
            estimator: Estimator::new(3, 0.97),
        }),
    ];
    for policy in policies.iter_mut() {
        let name = policy.name();
        let requests = (0..40)
            .map(|i| {
                req(
                    i,
                    (i / 5) as f64,
                    1 + i % 7,
                    (i / 5) as f64 + 2.,
                    &decaying(0.7, 0.8),
                )
            })
            .collect();
        let sim = simulate(requests, &profile(), policy.as_mut(), &env, 1.4);
        let m = summarize(&sim);
        c.eq(&format!("{name}: requests"), m.requests, 40);
        c.eq(
            &format!("{name}: tokens"),
            m.tokens,
            (0..40).map(|i| 1 + i % 7).sum::<usize>(),
        );
        c.eq(
            &format!("{name}: round log commits"),
            sim.rounds.iter().map(|r| r.committed).sum::<usize>(),
            m.tokens,
        );
        c.check(
            &format!("{name}: utilization <= 1"),
            m.utilization <= 1. + 1e-12,
            || format!("{}", m.utilization),
        );
        c.check(
            &format!("{name}: overload misses deadlines"),
            m.met < 40,
            || "all met".into(),
        );
        c.eq(
            &format!("{name}: infeasible == flagged rounds"),
            sim.infeasible,
            sim.rounds.iter().filter(|r| r.infeasible).count(),
        );
    }
    c.done();
}
#[test]
fn c_metrics_independent() {
    let mut c = Checks::new("c_metrics_independent");
    let st = |r: Request, emitted, first, finish, at: Vec<f64>| State {
        request: r,
        emitted,
        first: Some(first),
        finish: Some(finish),
        rounds: at.len(),
        committed_at: at,
    };
    let sim = Simulation {
        states: vec![
            st(req(0, 1., 2, 5., &[]), 2, 2., 4., vec![2., 4.]),
            st(req(1, 2., 1, 3., &[]), 1, 5., 5., vec![5.]),
            st(req(2, 2., 3, 9., &[]), 3, 3., 6., vec![3., 6., 6.]),
        ],
        busy: 4.,
        infeasible: 0,
        resets: 0,
        rounds: vec![],
    };
    let m = summarize(&sim);
    // TTFT = [1, 3, 1], E2E = [3, 3, 4], TPOT = [2, -, 1.5]; makespan 6 - 1 = 5 ms.
    c.eq("p95 ttft", m.p95_ttft, 3.);
    c.eq("p95 e2e", m.p95_e2e, 4.);
    c.eq("p95 tpot", m.p95_tpot, 2.);
    c.eq("met", m.met, 2);
    c.eq("makespan", m.makespan, 5.);
    c.close("goodput req/s", m.goodput, 400., 1e-9);
    c.close("utilization", m.utilization, 0.8, 1e-12);
    c.eq("tokens", m.tokens, 6);
    c.done();
}

// ------------------------------------------------------------------- R
fn trace_request(id: usize, n: usize) -> Request {
    Request {
        id,
        arrival: 0.,
        output: n,
        context: 0,
        deadline: 1e9,
        accept: vec![],
    }
}
#[test]
fn r_runs_bruteforce() {
    let mut c = Checks::new("r_runs_bruteforce");
    let mut rng = Lcg(21);
    let mut cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![1],
        vec![1, 1, 0, 1],
        vec![0, 0, 1, 1, 1],
    ];
    for _ in 0..30 {
        let n = (rng.next() * 20.0) as usize;
        cases.push((0..n).map(|_| (rng.next() < 0.7) as u8).collect());
    }
    for flags in &cases {
        let want: Vec<usize> = (0..flags.len())
            .map(|i| flags[i..].iter().take_while(|&&f| f == 1).count())
            .collect();
        c.eq(&format!("runs {flags:?}"), run_lengths(flags), want);
    }
    c.done();
}
#[test]
fn r_trace_env_censors() {
    let mut c = Checks::new("r_trace_env_censors");
    let flags = vec![vec![1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1]];
    let runs = run_lengths(&flags[0]);
    let env = TraceEnv::from_flags(&flags);
    let r = trace_request(0, flags[0].len());
    for e in 0..flags[0].len() {
        for k in 0..=5 {
            let want = flags[0][e..].iter().take_while(|&&f| f == 1).count().min(k);
            c.eq(
                &format!("accepted e={e} k={k}"),
                env.accepted(&r, e, k),
                want,
            );
            c.eq(
                &format!("censored_accept e={e} k={k}"),
                censored_accept(&runs, e, k),
                want,
            );
        }
    }
    // k = 0 reveals nothing: the zero-evidence lower bound must stay at k = 0 forever,
    // exactly like plain decoding, even though the trace has long accepted runs.
    let rows = profile();
    let a = simulate(vec![trace_request(0, 12)], &rows, &mut Fixed(0), &env, 5.);
    let b = simulate(
        vec![trace_request(0, 12)],
        &rows,
        &mut PerPosition::new(3, 0.97, 1.),
        &env,
        5.,
    );
    c.eq(
        "no_probe on a trace == k0",
        b.states[0].committed_at.clone(),
        a.states[0].committed_at.clone(),
    );
    c.done();
}
fn schedule_oracle(flags: &[u8], k: usize) -> Vec<(usize, usize)> {
    let n = flags.len();
    let (mut e, mut out) = (0, Vec::new());
    while e < n {
        let kk = k.min(n - e - 1);
        let mut a = 0;
        while a < kk && flags[e + a] == 1 {
            a += 1;
        }
        out.push((kk, a));
        e += a + 1;
    }
    out
}
#[test]
fn r_fixed_schedule_matches_engine() {
    let mut c = Checks::new("r_fixed_schedule_matches_engine");
    let mut rng = Lcg(33);
    for case in 0..12 {
        let n = 1 + (rng.next() * 40.0) as usize;
        let flags: Vec<u8> = (0..n).map(|_| (rng.next() < 0.75) as u8).collect();
        let runs = run_lengths(&flags);
        for k in [0, 1, 2, 3, 5] {
            let want = schedule_oracle(&flags, k);
            c.eq(
                &format!("case {case} k={k} oracle"),
                fixed_k_schedule(&runs, k),
                want.clone(),
            );
            let rows = vec![ProfilePoint {
                context: 0,
                draft: 0.1,
                verify: vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            }];
            let sim = simulate(
                vec![trace_request(0, n)],
                &rows,
                &mut Fixed(k),
                &TraceEnv::from_flags(&[flags.clone()]),
                1e9,
            );
            let engine: Vec<(usize, usize)> =
                sim.rounds.iter().map(|r| (r.k, r.accepted)).collect();
            c.eq(&format!("case {case} k={k} engine"), engine, want);
        }
    }
    c.done();
}
#[test]
fn r_reference_rounds_match_measured() {
    let mut c = Checks::new("r_reference_rounds_match_measured");
    let trace = match Trace::load(&reference_trace_path()) {
        Ok(t) => t,
        Err(e) => {
            c.check("reference trace loads", false, || e);
            return c.done();
        }
    };
    c.check(
        "header names device",
        trace.header.contains_key("device") && trace.header.contains_key("date"),
        || format!("{:?}", trace.header.keys()),
    );
    c.check("cost profile calibrates", trace.profile().is_ok(), || {
        format!("{:?}", trace.profile().err())
    });
    c.eq(
        "every graph-regime timing row is stable (p75 <= 1.5 p25)",
        trace.unstable_rows,
        0,
    );
    let by_id: BTreeMap<usize, &TracePrompt> = trace.prompts.iter().map(|p| (p.id, p)).collect();
    // The hand-written loop leaves the greedy path at bf16 near-ties (first_diff), so
    // rounds are compared only before that point, with flags from cached draft steps.
    let (mut compared, mut equal, mut whole, mut whole_equal) = (0, 0, 0, 0);
    let mut first = String::new();
    for run in &trace.spec {
        let p = by_id[&run.prompt];
        let Some(flags) = p.stepwise_flags.as_ref() else {
            continue;
        };
        let schedule = fixed_k_schedule(&run_lengths(flags), run.k);
        let (n, ok) = compare_before_divergence(&run.rounds, &schedule, run.first_diff);
        compared += n;
        equal += ok;
        if n != ok && first.is_empty() {
            first = format!(
                "prompt {} k={}: diverged after {ok} of {n} comparable rounds",
                run.prompt, run.k
            );
        }
        if run.same_as_greedy {
            // The loop caps proposals by max_new, the replay by the trace end, so only the
            // final round may differ when EOS ends generation early.
            whole += 1;
            let m = run.rounds.len();
            whole_equal +=
                (schedule.len() == m && m > 0 && schedule[..m - 1] == run.rounds[..m - 1]) as usize;
        }
    }
    println!("  {equal}/{compared} leading rounds equal before divergence; {whole_equal}/{whole} greedy-path runs replay exactly");
    c.check("enough comparable rounds", compared >= 100, || {
        format!("only {compared}")
    });
    // Not 100%: the loop feeds the draft 2 tokens at once after a fully accepted round,
    // and bf16 width-2 rounding flips near-tie guesses (measured ~1.6% of rounds on the
    // reference trace). An off-by-one replay agrees on almost no leading rounds.
    c.check(
        "replayed rounds equal measured rounds before divergence",
        equal * 100 >= compared * 95,
        || format!("{equal}/{compared} equal; {first}"),
    );
    c.check(
        "at least one greedy-path run replays exactly",
        whole_equal >= 1,
        || format!("{whole_equal}/{whole}"),
    );
    c.done();
}
#[test]
fn r_ucb_upper_bisection() {
    let mut c = Checks::new("r_ucb_upper_bisection");
    let cases = [
        (1., 0.),
        (0., 1.),
        (3., 1.),
        (2.5, 1.5),
        (0.97, 0.03),
        (0., 4.2),
        (45.3, 12.7),
    ];
    for z in [0.5, 1.0, 2.0] {
        let mut u = CensoredUcb::new(cases.len() + 1, 1.0, z);
        for (i, &(s, f)) in cases.iter().enumerate() {
            u.estimator.success[i] = s;
            u.estimator.failure[i] = f;
        }
        let got = u.upper();
        for (i, &(s, f)) in cases.iter().enumerate() {
            c.close(
                &format!("upper z={z} s={s} f={f}"),
                got[i],
                wilson_bisect(s, f, z, true),
                1e-9,
            );
        }
        c.eq(
            &format!("unexplored depth is optimistic z={z}"),
            got[cases.len()],
            1.0,
        );
    }
    c.done();
}
#[test]
fn r_ucb_policy_behaviour() {
    let mut c = Checks::new("r_ucb_policy_behaviour");
    let rows = vec![ProfilePoint {
        context: 0,
        draft: 0.1,
        verify: vec![1.0, 1.05, 1.1, 1.2, 1.4],
    }];
    let n = 200;
    let all = TraceEnv::from_flags(&[vec![1; n]]);
    let none = TraceEnv::from_flags(&[vec![0; n]]);
    let up = simulate(
        vec![trace_request(0, n)],
        &rows,
        &mut CensoredUcb::new(4, 0.97, 1.0),
        &all,
        1e9,
    );
    c.check(
        "all-accept trace: proposes kmax",
        up.rounds
            .iter()
            .skip(1)
            .filter(|r| r.k == 4 || r.committed < 5)
            .count()
            + 1
            >= up.rounds.len(),
        || format!("ks {:?}", up.rounds.iter().map(|r| r.k).collect::<Vec<_>>()),
    );
    let down = simulate(
        vec![trace_request(0, n)],
        &rows,
        &mut CensoredUcb::new(4, 0.97, 1.0),
        &none,
        1e9,
    );
    let probes = down.rounds.iter().filter(|r| r.k > 0).count();
    c.check(
        "all-reject trace: probes are rare",
        probes * 3 < down.rounds.len(),
        || format!("{probes} of {}", down.rounds.len()),
    );
    c.check(
        "all-reject trace: keeps probing after evidence decays",
        probes >= 3,
        || format!("{probes} probes"),
    );
    let no_probe = simulate(
        vec![trace_request(0, n)],
        &rows,
        &mut PerPosition::new(4, 0.97, 1.0),
        &all,
        1e9,
    );
    c.check(
        "optimism beats the no-probe lower bound on an easy trace",
        up.busy < no_probe.busy,
        || format!("{} vs {}", up.busy, no_probe.busy),
    );
    // Decision equals brute force over the optimistic hazards.
    let mut u = CensoredUcb::new(4, 1.0, 1.0);
    u.estimator.success = vec![9., 5., 1., 0.];
    u.estimator.failure = vec![1., 3., 1., 0.];
    let hz = u.upper();
    let mut best = (0, -1.0);
    for k in 0..=4 {
        let rate = expected_tokens(&hz, k) / rows[0].round_cost(k);
        if rate > best.1 + 1e-12 {
            best = (k, rate);
        }
    }
    c.eq(
        "choice == brute force on upper bounds",
        u.choose(&rows[0], 1e9, 100),
        Some(best.0),
    );
    c.done();
}
#[test]
fn r_reference_trace_replay_conservation() {
    let mut c = Checks::new("r_reference_trace_replay_conservation");
    let Ok(trace) = Trace::load(&reference_trace_path()) else {
        c.check("reference trace loads", false, || "missing".into());
        return c.done();
    };
    let Ok(rows) = trace.profile() else {
        c.check("profile", false, || "calibration failed".into());
        return c.done();
    };
    let flags: Vec<Vec<u8>> = trace.prompts.iter().map(|p| p.flags.clone()).collect();
    let env = TraceEnv::from_flags(&flags);
    let requests: Vec<Request> = trace
        .prompts
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.flags.is_empty())
        .map(|(i, p)| Request {
            id: i,
            arrival: i as f64 * 1000.0,
            output: p.flags.len(),
            context: p.prompt_tokens,
            deadline: 1e12,
            accept: vec![],
        })
        .collect();
    let total: usize = requests.iter().map(|r| r.output).sum();
    let kmax = trace.kmax;
    let mut policies: Vec<Box<dyn Policy>> = vec![
        Box::new(Fixed(0)),
        Box::new(Fixed(2)),
        Box::new(PerPosition::new(kmax, 0.97, 0.)),
        Box::new(CensoredUcb::new(kmax, 0.97, 1.0)),
        Box::new(ArmUcb::new(kmax, 0.97, 0.5)),
    ];
    for p in policies.iter_mut() {
        let name = p.name();
        let sim = simulate(requests.clone(), &rows, p.as_mut(), &env, 1e9);
        let m = summarize(&sim);
        c.eq(&format!("{name}: tokens conserved"), m.tokens, total);
        c.check(
            &format!("{name}: accepted <= proposed"),
            sim.rounds.iter().all(|r| r.accepted <= r.k),
            || "violation".into(),
        );
    }
    c.done();
}
