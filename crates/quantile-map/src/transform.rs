//! Core quantile mapping transform.

use std::collections::{BTreeMap, BTreeSet};

use statrs::distribution::ContinuousCDF;

use crate::config::QmConfig;
use crate::error::QuantileMapError;
use crate::fit::BaselineFit;
use crate::gamma::{GammaParams, gamma_dist};

/// Epsilon constant used for clamping CDF probabilities away from 0 and 1.
const EPS: f64 = 1e-12;

/// Amplify the upper tail of a CDF probability.
///
/// If `u > threshold`, maps `u` via `1 - (1-u)^k` and clamps to
/// `[EPS, 1 - EPS]`. Otherwise returns `u` unchanged.
pub(crate) fn amplify_tail(u: f64, threshold: f64, k: f64) -> f64 {
    if u > threshold {
        let amplified = 1.0 - (1.0 - u).powf(k);
        amplified.clamp(EPS, 1.0 - EPS)
    } else {
        u
    }
}

/// Group wet-day indices by `(month, year)`.
///
/// A single O(n) pass collects indices where the precipitation value is
/// non-NaN and exceeds `config.intensity_threshold()`. The returned
/// `BTreeMap` gives deterministic iteration order sorted by (month, year).
fn build_wet_day_groups(
    precip: &[f64],
    month: &[u8],
    year: &[u32],
    config: &QmConfig,
) -> BTreeMap<(u8, u32), Vec<usize>> {
    let mut groups: BTreeMap<(u8, u32), Vec<usize>> = BTreeMap::new();
    for i in 0..precip.len() {
        if !precip[i].is_nan() && precip[i] > config.intensity_threshold() {
            groups.entry((month[i], year[i])).or_default().push(i);
        }
    }
    groups
}

/// Apply the quantile mapping transform to a single month/year group.
///
/// For each index, the baseline CDF value is computed, optionally
/// tail-amplified, then inverted through the target distribution.
/// If `config.enforce_target_mean()` is set and the group is large enough,
/// the mapped values are rescaled so their mean matches `target_mean`.
fn transform_group(
    indices: &[usize],
    precip: &[f64],
    output: &mut [f64],
    base_dist: &statrs::distribution::Gamma,
    target_dist: &statrs::distribution::Gamma,
    target_mean: f64,
    config: &QmConfig,
) {
    for &i in indices {
        let mut u = base_dist.cdf(precip[i]);

        // Clamp to avoid exact 0 or 1 which can produce infinite quantiles.
        u = u.clamp(EPS, 1.0 - EPS);

        if config.exaggerate_extremes() {
            u = amplify_tail(u, config.extreme_prob_threshold(), config.extreme_k());
        }

        let mut mapped = target_dist.inverse_cdf(u);

        if config.validate_output() {
            if !mapped.is_finite() {
                mapped = precip[i];
            }
            if mapped < 0.0 {
                mapped = 0.0;
            }
        }

        output[i] = mapped;
    }

    // Enforce the target mean by rescaling if configured and the group is
    // large enough.
    if config.enforce_target_mean() && indices.len() >= config.min_group_for_enforcement() {
        let sum_mapped: f64 = indices.iter().map(|&i| output[i]).sum();
        let mean_mapped = sum_mapped / indices.len() as f64;

        if mean_mapped > EPS {
            let ratio = target_mean / mean_mapped;
            for &i in indices {
                output[i] *= ratio;
                if config.validate_output() && output[i] < 0.0 {
                    output[i] = 0.0;
                }
            }
        }
    }
}

/// Apply quantile mapping to a precipitation time-series.
///
/// Transforms `precip` from the baseline climate (described by `baseline`)
/// to each year's target climate (described by `target_params`).
///
/// # Returns
///
/// A tuple `(adjusted, perturbed_months, skipped_months)` where:
/// - `adjusted` has the same length as `precip` (dry days and NaN pass
///   through unchanged),
/// - `perturbed_months` lists the 1-indexed calendar months that were
///   successfully mapped, and
/// - `skipped_months` lists months that could not be mapped (missing
///   parameters or failed distribution construction).
#[allow(clippy::type_complexity)]
pub(crate) fn apply_quantile_mapping(
    precip: &[f64],
    month: &[u8],
    year: &[u32],
    baseline: &BaselineFit,
    target_params: &[[Option<GammaParams>; 12]],
    config: &QmConfig,
) -> Result<(Vec<f64>, Vec<u8>, Vec<u8>), QuantileMapError> {
    // Start with a copy of the input; dry days and NaN pass through.
    let mut output = precip.to_vec();

    let groups = build_wet_day_groups(precip, month, year, config);

    let mut perturbed_set = BTreeSet::<u8>::new();
    let mut skipped_set = BTreeSet::<u8>::new();

    for (&(m, y), indices) in &groups {
        // Baseline parameters for this calendar month.
        let base_params = match baseline.params_for_month(m) {
            Some(p) => p,
            None => {
                skipped_set.insert(m);
                continue;
            }
        };

        // Target parameters: year is 1-indexed in the data.
        let tgt_params = match target_params[(y - 1) as usize][(m - 1) as usize] {
            Some(p) => p,
            None => {
                skipped_set.insert(m);
                continue;
            }
        };

        // Build statrs distributions; skip the group on failure.
        let base_dist = match gamma_dist(&base_params) {
            Ok(d) => d,
            Err(_) => {
                skipped_set.insert(m);
                continue;
            }
        };
        let target_dist = match gamma_dist(&tgt_params) {
            Ok(d) => d,
            Err(_) => {
                skipped_set.insert(m);
                continue;
            }
        };

        transform_group(
            indices,
            precip,
            &mut output,
            &base_dist,
            &target_dist,
            tgt_params.mean(),
            config,
        );

        perturbed_set.insert(m);
    }

    Ok((
        output,
        perturbed_set.into_iter().collect(),
        skipped_set.into_iter().collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Gamma as GammaDist};

    // ---------------------------------------------------------------
    // Helper: generate synthetic test data
    // ---------------------------------------------------------------

    struct TestData {
        precip: Vec<f64>,
        months: Vec<u8>,
        years: Vec<u32>,
    }

    fn generate_test_data(shape: f64, scale: f64, n_years: u32, days_per_month: usize) -> TestData {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dist = GammaDist::new(shape, scale).unwrap();
        let n_days = n_years as usize * 12 * days_per_month;

        let mut precip = Vec::with_capacity(n_days);
        let mut months = Vec::with_capacity(n_days);
        let mut years = Vec::with_capacity(n_days);

        for y in 1..=n_years {
            for m in 1..=12u8 {
                for _ in 0..days_per_month {
                    precip.push(dist.sample(&mut rng));
                    months.push(m);
                    years.push(y);
                }
            }
        }

        TestData {
            precip,
            months,
            years,
        }
    }

    fn make_uniform_baseline(gp: GammaParams) -> BaselineFit {
        BaselineFit::new([Some(gp); 12], vec![], 0)
    }

    // ---------------------------------------------------------------
    // amplify_tail tests
    // ---------------------------------------------------------------

    #[test]
    fn tail_below_threshold_unchanged() {
        assert_eq!(amplify_tail(0.5, 0.95, 1.2), 0.5);
    }

    #[test]
    fn tail_above_threshold_increases() {
        let result = amplify_tail(0.98, 0.95, 1.2);
        assert!(
            result > 0.98,
            "expected amplified value > 0.98, got {result}"
        );
    }

    #[test]
    fn tail_k_one_identity() {
        let result = amplify_tail(0.97, 0.95, 1.0);
        assert_relative_eq!(result, 0.97, epsilon = 1e-14);
    }

    #[test]
    fn tail_order_preserved() {
        let u1 = 0.96;
        let u2 = 0.98;
        let a1 = amplify_tail(u1, 0.95, 1.2);
        let a2 = amplify_tail(u2, 0.95, 1.2);
        assert!(a1 < a2, "expected {a1} < {a2}");
    }

    // ---------------------------------------------------------------
    // apply_quantile_mapping tests
    // ---------------------------------------------------------------

    #[test]
    fn identity_mapping() {
        let gp = GammaParams::new(2.0, 3.0).unwrap();
        let data = generate_test_data(2.0, 3.0, 2, 30);

        let baseline = make_uniform_baseline(gp);
        let target_params = vec![[Some(gp); 12]; 2];

        let config = QmConfig::new()
            .with_enforce_target_mean(false)
            .with_min_events(1);

        let (adjusted, perturbed, skipped) = apply_quantile_mapping(
            &data.precip,
            &data.months,
            &data.years,
            &baseline,
            &target_params,
            &config,
        )
        .unwrap();

        assert!(skipped.is_empty());
        assert!(!perturbed.is_empty());

        let mae: f64 = adjusted
            .iter()
            .zip(data.precip.iter())
            .map(|(a, p)| (a - p).abs())
            .sum::<f64>()
            / adjusted.len() as f64;

        assert!(
            mae < 1e-6,
            "identity mapping should produce near-zero MAE, got {mae}"
        );
    }

    #[test]
    fn mean_scaling_ratio() {
        let base_gp = GammaParams::new(2.0, 3.0).unwrap(); // mean = 6
        let mean_factor = 0.7;
        // Target mean = 6 * 0.7 = 4.2 → shape=2, scale=2.1 gives mean=4.2
        let target_gp = GammaParams::new(2.0, 2.1).unwrap();

        let data = generate_test_data(2.0, 3.0, 2, 30);

        let baseline = make_uniform_baseline(base_gp);
        let target_params = vec![[Some(target_gp); 12]; 2];

        let config = QmConfig::new()
            .with_enforce_target_mean(true)
            .with_min_events(1);

        let (adjusted, _perturbed, _skipped) = apply_quantile_mapping(
            &data.precip,
            &data.months,
            &data.years,
            &baseline,
            &target_params,
            &config,
        )
        .unwrap();

        // Compute wet-day means for input and output.
        let wet_input: Vec<f64> = data
            .precip
            .iter()
            .copied()
            .filter(|&x| !x.is_nan() && x > 0.0)
            .collect();
        let wet_output: Vec<f64> = adjusted
            .iter()
            .copied()
            .zip(data.precip.iter().copied())
            .filter(|&(_a, p)| !p.is_nan() && p > 0.0)
            .map(|(a, _)| a)
            .collect();

        let mean_in = wet_input.iter().sum::<f64>() / wet_input.len() as f64;
        let mean_out = wet_output.iter().sum::<f64>() / wet_output.len() as f64;

        let ratio = mean_out / mean_in;
        assert_relative_eq!(ratio, mean_factor, epsilon = 0.1);
    }

    #[test]
    fn dry_days_pass_through() {
        let gp = GammaParams::new(2.0, 3.0).unwrap();
        let mut data = generate_test_data(2.0, 3.0, 2, 30);

        // Set some days to exactly 0.0 (dry).
        let dry_indices = [0, 10, 50, 100, 200];
        for &i in &dry_indices {
            data.precip[i] = 0.0;
        }

        let baseline = make_uniform_baseline(gp);
        let target_params = vec![[Some(gp); 12]; 2];

        let config = QmConfig::new().with_min_events(1);

        let (adjusted, _perturbed, _skipped) = apply_quantile_mapping(
            &data.precip,
            &data.months,
            &data.years,
            &baseline,
            &target_params,
            &config,
        )
        .unwrap();

        for &i in &dry_indices {
            assert_eq!(
                adjusted[i], 0.0,
                "dry day at index {i} should be exactly 0.0"
            );
        }
    }

    #[test]
    fn nan_pass_through() {
        let gp = GammaParams::new(2.0, 3.0).unwrap();
        let mut data = generate_test_data(2.0, 3.0, 2, 30);

        // Set some days to NaN.
        let nan_indices = [5, 15, 55, 105, 205];
        for &i in &nan_indices {
            data.precip[i] = f64::NAN;
        }

        let baseline = make_uniform_baseline(gp);
        let target_params = vec![[Some(gp); 12]; 2];

        let config = QmConfig::new().with_min_events(1);

        let (adjusted, _perturbed, _skipped) = apply_quantile_mapping(
            &data.precip,
            &data.months,
            &data.years,
            &baseline,
            &target_params,
            &config,
        )
        .unwrap();

        for &i in &nan_indices {
            assert!(
                adjusted[i].is_nan(),
                "NaN at index {i} should remain NaN, got {}",
                adjusted[i]
            );
        }
    }

    #[test]
    fn order_preservation() {
        let gp = GammaParams::new(2.0, 3.0).unwrap();
        // Use a target with a different mean to ensure non-trivial mapping.
        let target_gp = GammaParams::new(3.0, 2.0).unwrap();

        let data = generate_test_data(2.0, 3.0, 2, 30);

        let baseline = make_uniform_baseline(gp);
        let target_params = vec![[Some(target_gp); 12]; 2];

        let config = QmConfig::new()
            .with_enforce_target_mean(false)
            .with_min_events(1);

        let (adjusted, _perturbed, _skipped) = apply_quantile_mapping(
            &data.precip,
            &data.months,
            &data.years,
            &baseline,
            &target_params,
            &config,
        )
        .unwrap();

        // Build the same groups and check within-group order preservation.
        let groups = build_wet_day_groups(&data.precip, &data.months, &data.years, &config);
        for indices in groups.values() {
            for pair in indices.windows(2) {
                let (i, j) = (pair[0], pair[1]);
                if data.precip[i] < data.precip[j] {
                    assert!(
                        adjusted[i] <= adjusted[j],
                        "order violation: adjusted[{i}]={} > adjusted[{j}]={}",
                        adjusted[i],
                        adjusted[j]
                    );
                } else if data.precip[i] > data.precip[j] {
                    assert!(
                        adjusted[i] >= adjusted[j],
                        "order violation: adjusted[{i}]={} < adjusted[{j}]={}",
                        adjusted[i],
                        adjusted[j]
                    );
                }
            }
        }
    }

    #[test]
    fn all_values_non_negative() {
        let gp = GammaParams::new(2.0, 3.0).unwrap();
        let target_gp = GammaParams::new(1.5, 4.0).unwrap();

        let data = generate_test_data(2.0, 3.0, 2, 30);

        let baseline = make_uniform_baseline(gp);
        let target_params = vec![[Some(target_gp); 12]; 2];

        let config = QmConfig::new().with_min_events(1);

        let (adjusted, _perturbed, _skipped) = apply_quantile_mapping(
            &data.precip,
            &data.months,
            &data.years,
            &baseline,
            &target_params,
            &config,
        )
        .unwrap();

        for (i, &v) in adjusted.iter().enumerate() {
            if !v.is_nan() {
                assert!(v >= 0.0, "negative value at index {i}: {v}");
            }
        }
    }

    #[test]
    fn small_group_skips_enforcement() {
        // Create a group with only 3 wet days. With min_group_for_enforcement=5
        // the mean enforcement should be skipped.
        let gp = GammaParams::new(2.0, 3.0).unwrap();
        // Target with a very different mean so enforcement would be visible.
        let target_gp = GammaParams::new(2.0, 6.0).unwrap(); // mean = 12

        // We build a tiny dataset: 1 year, 1 month, 3 wet days.
        let precip = vec![4.0, 5.0, 6.0];
        let months = vec![1u8, 1, 1];
        let years = vec![1u32, 1, 1];

        let baseline = make_uniform_baseline(gp);
        let target_params = vec![[Some(target_gp); 12]; 1];

        let config_with_enforcement = QmConfig::new()
            .with_enforce_target_mean(true)
            .with_min_group_for_enforcement(5)
            .with_min_events(1);

        let config_without_enforcement = QmConfig::new()
            .with_enforce_target_mean(false)
            .with_min_events(1);

        let (adj_enforced, _, _) = apply_quantile_mapping(
            &precip,
            &months,
            &years,
            &baseline,
            &target_params,
            &config_with_enforcement,
        )
        .unwrap();

        let (adj_no_enforce, _, _) = apply_quantile_mapping(
            &precip,
            &months,
            &years,
            &baseline,
            &target_params,
            &config_without_enforcement,
        )
        .unwrap();

        // With only 3 values in the group (< min_group_for_enforcement=5),
        // the enforcement should have been skipped, yielding identical results.
        for i in 0..precip.len() {
            assert_relative_eq!(adj_enforced[i], adj_no_enforce[i], epsilon = 1e-14);
        }
    }
}
