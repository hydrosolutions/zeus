//! Zeus evaluation: compare synthetic weather realisations against observed data.

mod config;
mod correlations;
mod error;
mod input;
mod output;
mod scoring;
mod timeseries;

use std::collections::BTreeMap;
use zeus_io::MultiSiteData;

pub use config::EvaluateConfig;
pub use error::EvaluateError;
pub use input::MultiSiteSynthetic;
pub use output::{
    ConditionalEntry, ConfigSummary, CrossGridEntry, Diagnostics, EvaluationOutput, InterVarEntry,
    RealisationScore, TimeseriesComparison, TimeseriesStatsOutput, to_json,
};

/// Evaluate synthetic weather against observed data.
///
/// Compares synthetic realisations against observed data across all sites,
/// computing timeseries statistics, correlation diagnostics, and a MAE-based
/// scorecard. Returns a JSON string containing the full evaluation output.
///
/// # Errors
///
/// Returns [`EvaluateError::Validation`] if site keys don't match between
/// observed and synthetic data.
/// Returns [`EvaluateError::Serialization`] if JSON serialization fails.
pub fn evaluate(
    observed: &MultiSiteData,
    synthetic: &MultiSiteSynthetic<'_>,
    config: &EvaluateConfig,
) -> Result<String, EvaluateError> {
    // Step 1: Validate site keys match
    for site in observed.keys() {
        if synthetic.get(site).is_none() {
            return Err(EvaluateError::MissingSite {
                site: site.clone(),
                location: "synthetic".to_string(),
            });
        }
    }
    for site in synthetic.keys() {
        if observed.get(site).is_none() {
            return Err(EvaluateError::MissingSite {
                site: site.clone(),
                location: "observed".to_string(),
            });
        }
    }

    // Step 2: Determine available variables
    let first_site = observed
        .keys()
        .next()
        .ok_or_else(|| EvaluateError::Validation {
            count: 1,
            details: "observed data contains no sites".to_string(),
        })?;
    let first_obs = observed
        .get(first_site)
        .ok_or_else(|| EvaluateError::MissingSite {
            site: first_site.clone(),
            location: "observed".to_string(),
        })?;

    let mut variables = vec!["precip".to_string()];
    if first_obs.temp_max().is_some() {
        variables.push("temp_max".to_string());
    }
    if first_obs.temp_min().is_some() {
        variables.push("temp_min".to_string());
    }

    // Step 3: Compute timeseries stats
    let n_realisations = synthetic.n_realisations();
    let threshold = config.precip_threshold();

    let mut obs_flat = Vec::new();
    let mut sim_flat: Vec<Vec<f64>> = vec![Vec::new(); n_realisations];
    let mut timeseries_diagnostics: BTreeMap<
        String,
        BTreeMap<String, BTreeMap<String, output::TimeseriesComparison>>,
    > = BTreeMap::new();

    for site in observed.keys() {
        let obs = match observed.get(site) {
            Some(o) => o,
            None => continue,
        };
        let syn_realisations = match synthetic.get(site) {
            Some(s) => s,
            None => continue,
        };

        let mut site_map: BTreeMap<String, BTreeMap<String, output::TimeseriesComparison>> =
            BTreeMap::new();

        for month in 1..=12 {
            let month_str = format!("{:02}", month);
            let mut month_map: BTreeMap<String, output::TimeseriesComparison> = BTreeMap::new();

            for var in &variables {
                let obs_data = get_obs_var(obs, var);
                if let Some(obs_values) = obs_data {
                    let obs_stats = timeseries::compute_timeseries_stats(
                        obs_values,
                        obs.months(),
                        month,
                        threshold,
                    );

                    let mut syn_stats_list = Vec::new();
                    for syn in syn_realisations {
                        let syn_data = get_syn_var(syn, var);
                        if let Some(syn_values) = syn_data {
                            let syn_stats = timeseries::compute_timeseries_stats(
                                syn_values,
                                syn.months(),
                                month,
                                threshold,
                            );
                            syn_stats_list.push(syn_stats);
                        }
                    }

                    // Flatten stats for scoring
                    let obs_flat_stats = scoring::flatten_stats(&obs_stats);
                    obs_flat.extend(obs_flat_stats);

                    for (r, syn_stats) in syn_stats_list.iter().enumerate() {
                        let syn_flat_stats = scoring::flatten_stats(syn_stats);
                        sim_flat[r].extend(syn_flat_stats);
                    }

                    // Compute mean across realisations for diagnostics
                    let simulated_mean = average_stats(&syn_stats_list);
                    month_map.insert(
                        var.clone(),
                        output::TimeseriesComparison {
                            observed: stats_to_output(&obs_stats),
                            simulated_mean,
                        },
                    );
                }
            }

            site_map.insert(month_str, month_map);
        }

        timeseries_diagnostics.insert(site.clone(), site_map);
    }

    // Step 4: Compute correlations
    let variables_str: Vec<&str> = variables.iter().map(|s| s.as_str()).collect();
    let non_precip_vars: Vec<&str> = variables_str
        .iter()
        .filter(|&&v| v != "precip")
        .copied()
        .collect();

    // Cross-grid: one call per variable
    let mut cross_grid = Vec::new();
    for &var in &variables_str {
        let results = correlations::cross_grid_correlations(observed, synthetic, var);
        for (site_a, site_b, obs_corr, sim_corr) in results {
            cross_grid.push(output::CrossGridEntry {
                site_a,
                site_b,
                variable: var.to_string(),
                observed_correlation: obs_corr,
                simulated_correlation: sim_corr,
            });
        }
    }

    // Inter-variable
    let inter_var_results =
        correlations::inter_variable_correlations(observed, synthetic, &variables_str);
    let inter_var: Vec<output::InterVarEntry> = inter_var_results
        .into_iter()
        .map(
            |(site, var_a, var_b, obs_corr, sim_corr)| output::InterVarEntry {
                site,
                variable_a: var_a,
                variable_b: var_b,
                observed_correlation: obs_corr,
                simulated_correlation: sim_corr,
            },
        )
        .collect();

    // Conditional
    let cond_results =
        correlations::conditional_correlations(observed, synthetic, &non_precip_vars, threshold);
    let conditional: Vec<output::ConditionalEntry> = cond_results
        .into_iter()
        .map(
            |(site, var, regime, obs_corr, sim_corr)| output::ConditionalEntry {
                site,
                variable: var,
                regime,
                observed_correlation: obs_corr,
                simulated_correlation: sim_corr,
            },
        )
        .collect();

    // Step 5: Build scorecard
    let realisation_ids: Vec<u32> = (0..n_realisations as u32).collect();
    let scorecard = scoring::score_realisations(&obs_flat, &sim_flat, &realisation_ids);

    // Step 6: Assemble and serialize
    let eval_output = output::EvaluationOutput {
        config: output::ConfigSummary {
            precip_threshold: threshold,
            n_sites: observed.n_sites(),
            n_realisations,
            variables,
        },
        scorecard,
        diagnostics: output::Diagnostics {
            timeseries: timeseries_diagnostics,
            cross_grid_correlations: cross_grid,
            inter_variable_correlations: inter_var,
            conditional_correlations: conditional,
        },
    };

    output::to_json(&eval_output)
}

/// Extract a variable from ObservedData.
fn get_obs_var<'a>(obs: &'a zeus_io::ObservedData, var: &str) -> Option<&'a [f64]> {
    match var {
        "precip" => Some(obs.precip()),
        "temp_max" => obs.temp_max(),
        "temp_min" => obs.temp_min(),
        _ => None,
    }
}

/// Extract a variable from SyntheticWeather.
fn get_syn_var<'a>(syn: &'a zeus_io::SyntheticWeather<'a>, var: &str) -> Option<&'a [f64]> {
    match var {
        "precip" => Some(syn.precip()),
        "temp_max" => syn.temp_max(),
        "temp_min" => syn.temp_min(),
        _ => None,
    }
}

/// Convert TimeseriesStats to TimeseriesStatsOutput.
fn stats_to_output(stats: &timeseries::TimeseriesStats) -> output::TimeseriesStatsOutput {
    output::TimeseriesStatsOutput {
        mean: stats.mean,
        sd: stats.sd,
        skewness: stats.skewness,
        wet_days: stats.wet_days,
        dry_days: stats.dry_days,
        mean_wet_spell: stats.mean_wet_spell,
        mean_dry_spell: stats.mean_dry_spell,
    }
}

/// Average timeseries stats across realisations.
fn average_stats(stats_list: &[timeseries::TimeseriesStats]) -> output::TimeseriesStatsOutput {
    let n = stats_list.len() as f64;
    output::TimeseriesStatsOutput {
        mean: stats_list.iter().map(|s| s.mean).sum::<f64>() / n,
        sd: stats_list.iter().map(|s| s.sd).sum::<f64>() / n,
        skewness: {
            let valid: Vec<f64> = stats_list.iter().filter_map(|s| s.skewness).collect();
            if valid.is_empty() {
                None
            } else {
                Some(valid.iter().sum::<f64>() / valid.len() as f64)
            }
        },
        wet_days: (stats_list.iter().map(|s| s.wet_days).sum::<usize>() as f64 / n).round()
            as usize,
        dry_days: (stats_list.iter().map(|s| s.dry_days).sum::<usize>() as f64 / n).round()
            as usize,
        mean_wet_spell: stats_list.iter().map(|s| s.mean_wet_spell).sum::<f64>() / n,
        mean_dry_spell: stats_list.iter().map(|s| s.mean_dry_spell).sum::<f64>() / n,
    }
}
