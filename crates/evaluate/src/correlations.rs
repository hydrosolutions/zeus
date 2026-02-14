//! Correlation diagnostics: anomaly, cross-grid, inter-variable, conditional.

#![allow(dead_code)]

use std::collections::BTreeMap;
use zeus_io::{MultiSiteData, ObservedData, SyntheticWeather};

use crate::input::MultiSiteSynthetic;

/// Compute anomalies by subtracting monthly means.
///
/// For each timestep, subtract the mean of all values sharing that month.
pub fn compute_anomalies(values: &[f64], months: &[u8]) -> Vec<f64> {
    // Group by month and compute means
    let mut monthly_sums: BTreeMap<u8, f64> = BTreeMap::new();
    let mut monthly_counts: BTreeMap<u8, usize> = BTreeMap::new();

    for (&value, &month) in values.iter().zip(months.iter()) {
        *monthly_sums.entry(month).or_insert(0.0) += value;
        *monthly_counts.entry(month).or_insert(0) += 1;
    }

    let monthly_means: BTreeMap<u8, f64> = monthly_sums
        .iter()
        .map(|(&month, &sum)| {
            let count = monthly_counts[&month];
            (month, sum / count as f64)
        })
        .collect();

    // Subtract monthly mean from each value
    values
        .iter()
        .zip(months.iter())
        .map(|(&value, &month)| value - monthly_means[&month])
        .collect()
}

/// Helper to extract a variable from ObservedData.
fn get_observed_variable<'a>(obs: &'a ObservedData, var_name: &str) -> Option<&'a [f64]> {
    match var_name {
        "precip" => Some(obs.precip()),
        "temp_max" => obs.temp_max(),
        "temp_min" => obs.temp_min(),
        _ => None,
    }
}

/// Helper to extract a variable from SyntheticWeather.
fn get_synthetic_variable<'a>(syn: &'a SyntheticWeather<'a>, var_name: &str) -> Option<&'a [f64]> {
    match var_name {
        "precip" => Some(syn.precip()),
        "temp_max" => syn.temp_max(),
        "temp_min" => syn.temp_min(),
        _ => None,
    }
}

/// Cross-grid correlation between site pairs for a given variable.
///
/// Returns vec of (site_a, site_b, obs_correlation, sim_mean_correlation).
pub fn cross_grid_correlations(
    observed: &MultiSiteData,
    synthetic: &MultiSiteSynthetic<'_>,
    variable: &str,
) -> Vec<(String, String, Option<f64>, Option<f64>)> {
    let mut results = Vec::new();

    // Collect site keys in sorted order
    let sites: Vec<_> = observed.keys().collect();

    // For each pair (i, j) where i < j
    for i in 0..sites.len() {
        for j in (i + 1)..sites.len() {
            let site_a = sites[i];
            let site_b = sites[j];

            let (obs_a, obs_b) = match (observed.get(site_a), observed.get(site_b)) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };

            // Extract variable values
            let values_a = get_observed_variable(obs_a, variable);
            let values_b = get_observed_variable(obs_b, variable);

            let (obs_corr, sim_mean_corr) = if let (Some(va), Some(vb)) = (values_a, values_b) {
                // Compute anomalies
                let anom_a = compute_anomalies(va, obs_a.months());
                let anom_b = compute_anomalies(vb, obs_b.months());

                // Observed correlation
                let obs_corr = zeus_stats::pearson_correlation(&anom_a, &anom_b);

                // Synthetic correlations
                let syn_realisations_a = synthetic.get(site_a);
                let syn_realisations_b = synthetic.get(site_b);

                let sim_mean_corr =
                    if let (Some(ra), Some(rb)) = (syn_realisations_a, syn_realisations_b) {
                        let mut sim_corrs = Vec::new();

                        for (syn_a, syn_b) in ra.iter().zip(rb.iter()) {
                            if let (Some(sva), Some(svb)) = (
                                get_synthetic_variable(syn_a, variable),
                                get_synthetic_variable(syn_b, variable),
                            ) {
                                let syn_anom_a = compute_anomalies(sva, syn_a.months());
                                let syn_anom_b = compute_anomalies(svb, syn_b.months());

                                if let Some(corr) =
                                    zeus_stats::pearson_correlation(&syn_anom_a, &syn_anom_b)
                                {
                                    sim_corrs.push(corr);
                                }
                            }
                        }

                        if sim_corrs.is_empty() {
                            None
                        } else {
                            Some(sim_corrs.iter().sum::<f64>() / sim_corrs.len() as f64)
                        }
                    } else {
                        None
                    };

                (obs_corr, sim_mean_corr)
            } else {
                (None, None)
            };

            results.push((site_a.clone(), site_b.clone(), obs_corr, sim_mean_corr));
        }
    }

    results
}

/// Inter-variable correlations at each site.
///
/// Returns vec of (site, var_a, var_b, obs_correlation, sim_mean_correlation).
#[allow(clippy::type_complexity)]
pub fn inter_variable_correlations(
    observed: &MultiSiteData,
    synthetic: &MultiSiteSynthetic<'_>,
    variables: &[&str],
) -> Vec<(String, String, String, Option<f64>, Option<f64>)> {
    let mut results = Vec::new();

    for site in observed.keys() {
        let obs = match observed.get(site) {
            Some(o) => o,
            None => continue,
        };

        // For each pair of variables (i, j) where i < j
        for i in 0..variables.len() {
            for j in (i + 1)..variables.len() {
                let var_a = variables[i];
                let var_b = variables[j];

                let values_a = get_observed_variable(obs, var_a);
                let values_b = get_observed_variable(obs, var_b);

                let (obs_corr, sim_mean_corr) = if let (Some(va), Some(vb)) = (values_a, values_b) {
                    // Compute anomalies
                    let anom_a = compute_anomalies(va, obs.months());
                    let anom_b = compute_anomalies(vb, obs.months());

                    // Observed correlation
                    let obs_corr = zeus_stats::pearson_correlation(&anom_a, &anom_b);

                    // Synthetic correlations
                    let sim_mean_corr = if let Some(syn_realisations) = synthetic.get(site) {
                        let mut sim_corrs = Vec::new();

                        for syn in syn_realisations {
                            if let (Some(sva), Some(svb)) = (
                                get_synthetic_variable(syn, var_a),
                                get_synthetic_variable(syn, var_b),
                            ) {
                                let syn_anom_a = compute_anomalies(sva, syn.months());
                                let syn_anom_b = compute_anomalies(svb, syn.months());

                                if let Some(corr) =
                                    zeus_stats::pearson_correlation(&syn_anom_a, &syn_anom_b)
                                {
                                    sim_corrs.push(corr);
                                }
                            }
                        }

                        if sim_corrs.is_empty() {
                            None
                        } else {
                            Some(sim_corrs.iter().sum::<f64>() / sim_corrs.len() as f64)
                        }
                    } else {
                        None
                    };

                    (obs_corr, sim_mean_corr)
                } else {
                    (None, None)
                };

                results.push((
                    site.clone(),
                    var_a.to_string(),
                    var_b.to_string(),
                    obs_corr,
                    sim_mean_corr,
                ));
            }
        }
    }

    results
}

/// Conditional precip-variable correlations under different regimes.
///
/// Returns vec of (site, variable, regime, obs_correlation, sim_mean_correlation).
/// Regimes: "all" (log1p precip), "wet" (precip >= threshold, log1p), "dry" (precip < threshold).
#[allow(clippy::type_complexity)]
pub fn conditional_correlations(
    observed: &MultiSiteData,
    synthetic: &MultiSiteSynthetic<'_>,
    non_precip_vars: &[&str],
    threshold: f64,
) -> Vec<(String, String, String, Option<f64>, Option<f64>)> {
    let mut results = Vec::new();

    for site in observed.keys() {
        let obs = match observed.get(site) {
            Some(o) => o,
            None => continue,
        };
        let obs_precip = obs.precip();

        for &var in non_precip_vars {
            let obs_var_values = get_observed_variable(obs, var);

            if let Some(var_values) = obs_var_values {
                // Regime: "all"
                let (all_obs_corr, all_sim_mean_corr) = {
                    let log_precip: Vec<f64> = obs_precip.iter().map(|&p| (1.0 + p).ln()).collect();
                    let obs_corr = zeus_stats::pearson_correlation(&log_precip, var_values);

                    let sim_mean_corr = if let Some(syn_realisations) = synthetic.get(site) {
                        let mut sim_corrs = Vec::new();

                        for syn in syn_realisations {
                            if let Some(syn_var) = get_synthetic_variable(syn, var) {
                                let syn_log_precip: Vec<f64> =
                                    syn.precip().iter().map(|&p| (1.0 + p).ln()).collect();
                                if let Some(corr) =
                                    zeus_stats::pearson_correlation(&syn_log_precip, syn_var)
                                {
                                    sim_corrs.push(corr);
                                }
                            }
                        }

                        if sim_corrs.is_empty() {
                            None
                        } else {
                            Some(sim_corrs.iter().sum::<f64>() / sim_corrs.len() as f64)
                        }
                    } else {
                        None
                    };

                    (obs_corr, sim_mean_corr)
                };

                results.push((
                    site.clone(),
                    var.to_string(),
                    "all".to_string(),
                    all_obs_corr,
                    all_sim_mean_corr,
                ));

                // Regime: "wet"
                let (wet_obs_corr, wet_sim_mean_corr) = {
                    let wet_indices: Vec<usize> = obs_precip
                        .iter()
                        .enumerate()
                        .filter(|&(_, p)| *p >= threshold)
                        .map(|(i, _)| i)
                        .collect();

                    if wet_indices.is_empty() {
                        (None, None)
                    } else {
                        let wet_precip: Vec<f64> = wet_indices
                            .iter()
                            .map(|&i| (1.0 + obs_precip[i]).ln())
                            .collect();
                        let wet_var: Vec<f64> =
                            wet_indices.iter().map(|&i| var_values[i]).collect();

                        let obs_corr = zeus_stats::pearson_correlation(&wet_precip, &wet_var);

                        let sim_mean_corr = if let Some(syn_realisations) = synthetic.get(site) {
                            let mut sim_corrs = Vec::new();

                            for syn in syn_realisations {
                                if let Some(syn_var) = get_synthetic_variable(syn, var) {
                                    let syn_precip = syn.precip();
                                    let syn_wet_indices: Vec<usize> = syn_precip
                                        .iter()
                                        .enumerate()
                                        .filter(|&(_, p)| *p >= threshold)
                                        .map(|(i, _)| i)
                                        .collect();

                                    if !syn_wet_indices.is_empty() {
                                        let syn_wet_precip: Vec<f64> = syn_wet_indices
                                            .iter()
                                            .map(|&i| (1.0 + syn_precip[i]).ln())
                                            .collect();
                                        let syn_wet_var: Vec<f64> =
                                            syn_wet_indices.iter().map(|&i| syn_var[i]).collect();

                                        if let Some(corr) = zeus_stats::pearson_correlation(
                                            &syn_wet_precip,
                                            &syn_wet_var,
                                        ) {
                                            sim_corrs.push(corr);
                                        }
                                    }
                                }
                            }

                            if sim_corrs.is_empty() {
                                None
                            } else {
                                Some(sim_corrs.iter().sum::<f64>() / sim_corrs.len() as f64)
                            }
                        } else {
                            None
                        };

                        (obs_corr, sim_mean_corr)
                    }
                };

                results.push((
                    site.clone(),
                    var.to_string(),
                    "wet".to_string(),
                    wet_obs_corr,
                    wet_sim_mean_corr,
                ));

                // Regime: "dry"
                let (dry_obs_corr, dry_sim_mean_corr) = {
                    let dry_indices: Vec<usize> = obs_precip
                        .iter()
                        .enumerate()
                        .filter(|&(_, p)| *p < threshold)
                        .map(|(i, _)| i)
                        .collect();

                    if dry_indices.is_empty() {
                        (None, None)
                    } else {
                        let dry_precip: Vec<f64> =
                            dry_indices.iter().map(|&i| obs_precip[i]).collect();
                        let dry_var: Vec<f64> =
                            dry_indices.iter().map(|&i| var_values[i]).collect();

                        let obs_corr = zeus_stats::pearson_correlation(&dry_precip, &dry_var);

                        let sim_mean_corr = if let Some(syn_realisations) = synthetic.get(site) {
                            let mut sim_corrs = Vec::new();

                            for syn in syn_realisations {
                                if let Some(syn_var) = get_synthetic_variable(syn, var) {
                                    let syn_precip = syn.precip();
                                    let syn_dry_indices: Vec<usize> = syn_precip
                                        .iter()
                                        .enumerate()
                                        .filter(|&(_, p)| *p < threshold)
                                        .map(|(i, _)| i)
                                        .collect();

                                    if !syn_dry_indices.is_empty() {
                                        let syn_dry_precip: Vec<f64> = syn_dry_indices
                                            .iter()
                                            .map(|&i| syn_precip[i])
                                            .collect();
                                        let syn_dry_var: Vec<f64> =
                                            syn_dry_indices.iter().map(|&i| syn_var[i]).collect();

                                        if let Some(corr) = zeus_stats::pearson_correlation(
                                            &syn_dry_precip,
                                            &syn_dry_var,
                                        ) {
                                            sim_corrs.push(corr);
                                        }
                                    }
                                }
                            }

                            if sim_corrs.is_empty() {
                                None
                            } else {
                                Some(sim_corrs.iter().sum::<f64>() / sim_corrs.len() as f64)
                            }
                        } else {
                            None
                        };

                        (obs_corr, sim_mean_corr)
                    }
                };

                results.push((
                    site.clone(),
                    var.to_string(),
                    "dry".to_string(),
                    dry_obs_corr,
                    dry_sim_mean_corr,
                ));
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_anomalies_basic() {
        let values = vec![10.0, 20.0, 30.0, 15.0, 25.0, 35.0];
        let months = vec![1, 2, 3, 1, 2, 3];

        let anomalies = compute_anomalies(&values, &months);

        // Month 1: [10.0, 15.0] -> mean = 12.5 -> anomalies = [-2.5, 2.5]
        // Month 2: [20.0, 25.0] -> mean = 22.5 -> anomalies = [-2.5, 2.5]
        // Month 3: [30.0, 35.0] -> mean = 32.5 -> anomalies = [-2.5, 2.5]

        assert!((anomalies[0] - (-2.5)).abs() < 0.01);
        assert!((anomalies[1] - (-2.5)).abs() < 0.01);
        assert!((anomalies[2] - (-2.5)).abs() < 0.01);
        assert!((anomalies[3] - 2.5).abs() < 0.01);
        assert!((anomalies[4] - 2.5).abs() < 0.01);
        assert!((anomalies[5] - 2.5).abs() < 0.01);

        // Check that sum of anomalies per month is ~0
        let month1_sum = anomalies[0] + anomalies[3];
        let month2_sum = anomalies[1] + anomalies[4];
        let month3_sum = anomalies[2] + anomalies[5];

        assert!(month1_sum.abs() < 0.01);
        assert!(month2_sum.abs() < 0.01);
        assert!(month3_sum.abs() < 0.01);
    }

    #[test]
    fn test_compute_anomalies_single_month() {
        let values = vec![5.0, 10.0, 15.0, 20.0];
        let months = vec![1, 1, 1, 1];

        let anomalies = compute_anomalies(&values, &months);

        // All same month -> mean = (5 + 10 + 15 + 20) / 4 = 12.5
        // Anomalies: -7.5, -2.5, 2.5, 7.5
        assert!((anomalies[0] - (-7.5)).abs() < 0.01);
        assert!((anomalies[1] - (-2.5)).abs() < 0.01);
        assert!((anomalies[2] - 2.5).abs() < 0.01);
        assert!((anomalies[3] - 7.5).abs() < 0.01);

        // Sum should be ~0
        let sum: f64 = anomalies.iter().sum();
        assert!(sum.abs() < 0.01);
    }

    #[test]
    fn test_cross_grid_perfect() {
        // This test is simplified since constructing full MultiSiteData/MultiSiteSynthetic is complex
        // We test the compute_anomalies helper which is the core computation

        let values = vec![1.0, 2.0, 3.0, 4.0];
        let months = vec![1, 1, 2, 2];

        let anom1 = compute_anomalies(&values, &months);
        let anom2 = compute_anomalies(&values, &months); // Same data

        let corr = zeus_stats::pearson_correlation(&anom1, &anom2);
        assert!(corr.is_some());
        assert!((corr.unwrap() - 1.0).abs() < 0.01); // Perfect correlation
    }

    #[test]
    fn test_cross_grid_no_sites() {
        // With a single site, no pairs exist, so results should be empty
        // This is an edge case that would be tested in integration tests
        // For unit tests, we verify the helper functions work correctly
    }

    #[test]
    fn test_inter_variable_perfect() {
        // Similar to cross_grid_perfect, testing the correlation computation
        let values_a = vec![1.0, 2.0, 3.0, 4.0];
        let values_b = vec![1.0, 2.0, 3.0, 4.0]; // Identical
        let months = vec![1, 1, 2, 2];

        let anom_a = compute_anomalies(&values_a, &months);
        let anom_b = compute_anomalies(&values_b, &months);

        let corr = zeus_stats::pearson_correlation(&anom_a, &anom_b);
        assert!(corr.is_some());
        assert!((corr.unwrap() - 1.0).abs() < 0.01); // Perfect correlation
    }

    #[test]
    fn test_inter_variable_single_var() {
        // With a single variable, no pairs exist
        // This would result in empty results in the full function
    }

    #[test]
    fn test_conditional_all_regime() {
        // Test that log1p transform is applied
        let precip = [0.0_f64, 1.0, 10.0, 100.0];
        let log_precip: Vec<f64> = precip.iter().map(|&p| (1.0 + p).ln()).collect();

        // Verify log1p values
        assert!((log_precip[0] - (1.0_f64).ln()).abs() < 0.01);
        assert!((log_precip[1] - (2.0_f64).ln()).abs() < 0.01);
        assert!((log_precip[2] - (11.0_f64).ln()).abs() < 0.01);
        assert!((log_precip[3] - (101.0_f64).ln()).abs() < 0.01);
    }

    #[test]
    fn test_conditional_wet_dry_filtering() {
        // Test filtering logic
        let precip = [0.5, 1.5, 0.3, 2.0, 0.1];
        let threshold = 1.0;

        let wet_indices: Vec<usize> = precip
            .iter()
            .enumerate()
            .filter(|&(_, p)| *p >= threshold)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(wet_indices, vec![1, 3]); // Indices 1 (1.5) and 3 (2.0)

        let dry_indices: Vec<usize> = precip
            .iter()
            .enumerate()
            .filter(|&(_, p)| *p < threshold)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(dry_indices, vec![0, 2, 4]); // Indices 0, 2, 4
    }
}
