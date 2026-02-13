//! Precipitation occurrence adjustment.
//!
//! Adjusts the number of wet days per month to match scenario-derived
//! target frequencies, adding or removing wet days as needed.

use crate::config::OccurrenceConfig;
use crate::error::PerturbError;
use crate::ramp::{RampMode, build_ramp};
use crate::result::OccurrenceResult;
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use zeus_quantile_map::BaselineFit;

/// Randomly selects `k` indices from `indices` without replacement.
fn sample_without_replacement(indices: &[usize], k: usize, rng: &mut impl Rng) -> Vec<usize> {
    // Fisher-Yates partial shuffle
    let mut pool = indices.to_vec();
    let k = k.min(pool.len());
    for i in 0..k {
        let j = rng.random_range(i..pool.len());
        pool.swap(i, j);
    }
    pool[..k].to_vec()
}

/// Adjusts precipitation occurrence to match scenario target frequencies.
///
/// For each (month, year) group:
/// 1. Compute the occurrence factor (possibly ramped if transient)
/// 2. Count current wet days (> `config.intensity_threshold()`)
/// 3. Compute `baseline_freq` = fraction of wet days in that month across all data
/// 4. Compute `target_wet = round(factor * baseline_freq * n_days_in_group)`
/// 5. If target < current: randomly zero out excess wet days
/// 6. If target > current: randomly pick dry days, sample amounts from baseline Gamma
///
/// # Arguments
///
/// * `precip` — Daily precipitation values.
/// * `month` — Calendar month (1--12) for each day.
/// * `year` — Year index (1-based) for each day.
/// * `baseline` — Fitted monthly Gamma parameters.
/// * `config` — Occurrence adjustment configuration.
/// * `rng` — Random number generator.
///
/// # Errors
///
/// Returns errors on empty data, length mismatches, or invalid months.
pub fn adjust_occurrence(
    precip: &[f64],
    month: &[u8],
    year: &[u32],
    baseline: &BaselineFit,
    config: &OccurrenceConfig,
    rng: &mut impl Rng,
) -> Result<OccurrenceResult, PerturbError> {
    // 1. Validate inputs
    if precip.is_empty() {
        return Err(PerturbError::EmptyData);
    }
    let n = precip.len();
    if month.len() != n {
        return Err(PerturbError::LengthMismatch {
            expected: n,
            got: month.len(),
            field: "month".to_string(),
        });
    }
    if year.len() != n {
        return Err(PerturbError::LengthMismatch {
            expected: n,
            got: year.len(),
            field: "year".to_string(),
        });
    }
    for &m in month {
        if !(1..=12).contains(&m) {
            return Err(PerturbError::InvalidMonth { month: m });
        }
    }

    // 2. Determine n_years from max year value
    let n_years = *year.iter().max().unwrap() as usize;

    // 3. Build ramp
    let ramp = build_ramp(
        config.factors(),
        n_years,
        config.transient(),
        RampMode::Multiplicative,
    )?;

    // 4. Compute baseline wet-day frequency per month
    let threshold = config.intensity_threshold();
    let mut baseline_freq = [0.0_f64; 12];
    for m in 1u8..=12 {
        let mut total_days = 0usize;
        let mut wet_days = 0usize;
        for i in 0..n {
            if month[i] == m {
                total_days += 1;
                if !precip[i].is_nan() && precip[i] > threshold {
                    wet_days += 1;
                }
            }
        }
        if total_days > 0 {
            baseline_freq[(m - 1) as usize] = wet_days as f64 / total_days as f64;
        }
    }

    // 5. Clone precip into mutable output vec
    let mut output = precip.to_vec();
    let mut total_days_added = 0usize;
    let mut total_days_removed = 0usize;

    // 6. For each year y (1..=n_years), for each month m (1..=12):
    for y in 1..=n_years {
        for m in 1u8..=12 {
            // Find indices where year[i] == y && month[i] == m
            let group_indices: Vec<usize> = (0..n)
                .filter(|&i| year[i] == y as u32 && month[i] == m)
                .collect();

            let group_size = group_indices.len();
            if group_size == 0 {
                continue;
            }

            // Count wet days (precip > threshold, not NaN)
            let wet_indices: Vec<usize> = group_indices
                .iter()
                .copied()
                .filter(|&i| !output[i].is_nan() && output[i] > threshold)
                .collect();
            let current_wet = wet_indices.len();

            // Get factor from ramp
            let factor = ramp[y - 1][(m - 1) as usize];

            // Compute target_wet
            let target_wet_raw =
                (factor * baseline_freq[(m - 1) as usize] * group_size as f64).round() as isize;
            let target_wet = target_wet_raw.clamp(0, group_size as isize) as usize;

            if target_wet < current_wet {
                // Remove excess wet days
                let to_remove = current_wet - target_wet;
                let chosen = sample_without_replacement(&wet_indices, to_remove, rng);
                for idx in chosen {
                    output[idx] = 0.0;
                }
                total_days_removed += to_remove;
            } else if target_wet > current_wet {
                // Add wet days: need baseline params for this month
                let params = match baseline.params_for_month(m) {
                    Some(p) => p,
                    None => continue, // skip month without params
                };

                let gamma = Gamma::new(params.shape(), params.scale()).map_err(|e| {
                    PerturbError::GammaConstruction {
                        shape: params.shape(),
                        scale: params.scale(),
                        message: e.to_string(),
                    }
                })?;

                // Find dry-day indices
                let dry_indices: Vec<usize> = group_indices
                    .iter()
                    .copied()
                    .filter(|&i| output[i].is_nan() || output[i] <= threshold)
                    .collect();

                let to_add = (target_wet - current_wet).min(dry_indices.len());
                let chosen = sample_without_replacement(&dry_indices, to_add, rng);
                for idx in chosen {
                    output[idx] = gamma.sample(rng);
                }
                total_days_added += to_add;
            }
        }
    }

    // 7. Return result
    Ok(OccurrenceResult::new(
        output,
        total_days_added,
        total_days_removed,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use zeus_quantile_map::{QmConfig, fit_monthly};

    /// Helper: build synthetic daily precip for `n_years` years, 30 days/month.
    /// About 50% of days are wet with Gamma-distributed amounts.
    fn make_synthetic(n_years: usize, wet_frac: f64, seed: u64) -> (Vec<f64>, Vec<u8>, Vec<u32>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let gamma = Gamma::new(2.0, 3.0).unwrap();
        let days_per_month = 30;
        let total = n_years * 12 * days_per_month;
        let mut precip = Vec::with_capacity(total);
        let mut month = Vec::with_capacity(total);
        let mut year = Vec::with_capacity(total);

        for y in 1..=n_years {
            for m in 1u8..=12 {
                for _ in 0..days_per_month {
                    let is_wet: bool = rng.random_bool(wet_frac);
                    let val = if is_wet { gamma.sample(&mut rng) } else { 0.0 };
                    precip.push(val);
                    month.push(m);
                    year.push(y as u32);
                }
            }
        }
        (precip, month, year)
    }

    /// Helper: fit baseline from synthetic data.
    fn fit_baseline(precip: &[f64], month: &[u8]) -> BaselineFit {
        let qm_config = QmConfig::new().with_min_events(3);
        fit_monthly(precip, month, &qm_config)
    }

    /// Helper: count wet days (> threshold).
    fn count_wet(precip: &[f64], threshold: f64) -> usize {
        precip
            .iter()
            .filter(|&&p| !p.is_nan() && p > threshold)
            .count()
    }

    #[test]
    fn factor_one_identity() {
        let (precip, month, year) = make_synthetic(3, 0.5, 100);
        let baseline = fit_baseline(&precip, &month);
        let config = OccurrenceConfig::new([1.0; 12], false, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result = adjust_occurrence(&precip, &month, &year, &baseline, &config, &mut rng)
            .expect("should succeed");

        let original_wet = count_wet(&precip, 0.0);
        let adjusted_wet = count_wet(result.precip(), 0.0);

        // With factor=1.0, the target is baseline_freq * group_size per group.
        // Because baseline_freq is computed globally, individual year-month
        // groups may round slightly differently.  The total change should be
        // very small (at most 1 per group due to rounding).
        let max_rounding_error = 12 * 3; // 12 months * 3 years
        let diff = (original_wet as isize - adjusted_wet as isize).unsigned_abs();
        assert!(
            diff <= max_rounding_error,
            "expected near-identity with factor 1.0: original={original_wet}, adjusted={adjusted_wet}, diff={diff}"
        );
    }

    #[test]
    fn reduce_wet_days() {
        let (precip, month, year) = make_synthetic(3, 0.5, 200);
        let baseline = fit_baseline(&precip, &month);
        let config = OccurrenceConfig::new([0.0001; 12], false, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result = adjust_occurrence(&precip, &month, &year, &baseline, &config, &mut rng)
            .expect("should succeed");

        let original_wet = count_wet(&precip, 0.0);
        let adjusted_wet = count_wet(result.precip(), 0.0);

        // Should have removed most wet days
        assert!(
            adjusted_wet < original_wet / 2,
            "expected significant reduction"
        );
        assert!(result.days_removed() > 0);
    }

    #[test]
    fn increase_wet_days() {
        let (precip, month, year) = make_synthetic(3, 0.3, 300);
        let baseline = fit_baseline(&precip, &month);
        let config = OccurrenceConfig::new([2.0; 12], false, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result = adjust_occurrence(&precip, &month, &year, &baseline, &config, &mut rng)
            .expect("should succeed");

        let original_wet = count_wet(&precip, 0.0);
        let adjusted_wet = count_wet(result.precip(), 0.0);

        // Should have added wet days
        assert!(adjusted_wet > original_wet, "expected more wet days");
        assert!(result.days_added() > 0);
    }

    #[test]
    fn deterministic_with_seed() {
        let (precip, month, year) = make_synthetic(2, 0.4, 400);
        let baseline = fit_baseline(&precip, &month);
        let config = OccurrenceConfig::new([1.5; 12], false, 0.0);

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(999);
        let result1 = adjust_occurrence(&precip, &month, &year, &baseline, &config, &mut rng1)
            .expect("should succeed");

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(999);
        let result2 = adjust_occurrence(&precip, &month, &year, &baseline, &config, &mut rng2)
            .expect("should succeed");

        assert_eq!(result1.precip(), result2.precip());
        assert_eq!(result1.days_added(), result2.days_added());
        assert_eq!(result1.days_removed(), result2.days_removed());
    }

    #[test]
    fn all_dry_edge_case() {
        // Input with no wet days
        let n_years = 2;
        let days_per_month = 30;
        let total = n_years * 12 * days_per_month;
        let precip = vec![0.0; total];
        let mut month = Vec::with_capacity(total);
        let mut year = Vec::with_capacity(total);
        for y in 1..=n_years {
            for m in 1u8..=12 {
                for _ in 0..days_per_month {
                    month.push(m);
                    year.push(y as u32);
                }
            }
        }

        // Baseline will have no params for any month (all dry, no wet values)
        let baseline = fit_baseline(&precip, &month);
        let config = OccurrenceConfig::new([2.0; 12], false, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result = adjust_occurrence(&precip, &month, &year, &baseline, &config, &mut rng)
            .expect("should succeed");

        // No params means no wet days can be added
        let adjusted_wet = count_wet(result.precip(), 0.0);
        assert_eq!(adjusted_wet, 0);
        assert_eq!(result.days_added(), 0);
        assert_eq!(result.days_removed(), 0);
    }

    #[test]
    fn skip_month_without_params() {
        // Create data where month 6 has no wet days (so baseline has None for it)
        // but other months have wet days
        let n_years = 2;
        let days_per_month = 30;
        let mut rng_data = rand::rngs::StdRng::seed_from_u64(500);
        let gamma = Gamma::new(2.0, 3.0).unwrap();
        let total = n_years * 12 * days_per_month;
        let mut precip = Vec::with_capacity(total);
        let mut month_vec = Vec::with_capacity(total);
        let mut year_vec = Vec::with_capacity(total);

        for y in 1..=n_years {
            for m in 1u8..=12 {
                for _ in 0..days_per_month {
                    let val = if m == 6 {
                        0.0 // always dry for month 6
                    } else {
                        let is_wet: bool = rng_data.random_bool(0.5);
                        if is_wet {
                            gamma.sample(&mut rng_data)
                        } else {
                            0.0
                        }
                    };
                    precip.push(val);
                    month_vec.push(m);
                    year_vec.push(y as u32);
                }
            }
        }

        let baseline = fit_baseline(&precip, &month_vec);
        // Month 6 should have no params
        assert!(baseline.params_for_month(6).is_none());

        let config = OccurrenceConfig::new([2.0; 12], false, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result =
            adjust_occurrence(&precip, &month_vec, &year_vec, &baseline, &config, &mut rng)
                .expect("should succeed");

        // Month 6 should remain unchanged (all zeros)
        for i in 0..precip.len() {
            if month_vec[i] == 6 {
                assert!(
                    (result.precip()[i] - precip[i]).abs() < f64::EPSILON,
                    "month 6 day should be unchanged"
                );
            }
        }
    }

    #[test]
    fn empty_input_error() {
        let baseline = fit_baseline(&[], &[]);
        let config = OccurrenceConfig::new([1.0; 12], false, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result = adjust_occurrence(&[], &[], &[], &baseline, &config, &mut rng);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PerturbError::EmptyData));
    }
}
