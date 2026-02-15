//! Climate perturbation pipeline for the Zeus weather generator.
//!
//! This crate applies climate change scenarios to simulated weather data
//! through temperature scaling, precipitation quantile mapping,
//! occurrence adjustment, and safety rails.
//!
//! # Pipeline Order
//!
//! 1. **Temperature** — additive deltas (optionally ramped)
//! 2. **QM Intensity** — Gamma-to-Gamma quantile mapping via `zeus-quantile-map`
//! 3. **Occurrence** — adjust wet-day frequency
//! 4. **Safety Rails** — clamp NaN/Inf/negative, enforce floor/cap

mod config;
mod error;
mod occurrence;
mod ramp;
mod result;
mod safety;
mod temperature;

pub use config::{OccurrenceConfig, PerturbConfig, TempConfig};
pub use error::PerturbError;
pub use occurrence::adjust_occurrence;
pub use ramp::{RampMode, build_ramp};
pub use result::{ModulesApplied, OccurrenceResult, PerturbResult, TempResult};
pub use safety::apply_safety_rails;
pub use temperature::{TempDeltas, adjust_temperature};

// Re-export commonly needed types from zeus-quantile-map.
pub use zeus_quantile_map::{
    BaselineFit, GammaParams, QmConfig, QmResult, QuantileMapError, ScenarioFactors,
};

use rand::SeedableRng;
use std::collections::BTreeSet;
use tracing::debug;

/// Builds a seeded or OS-sourced RNG.
fn make_rng(seed: Option<u64>) -> rand::rngs::StdRng {
    match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_os_rng(),
    }
}

/// Returns the number of distinct year values in a year slice.
/// Assumes years are 1-based contiguous.
fn count_years(year: &[u32]) -> usize {
    let year_set: BTreeSet<u32> = year.iter().copied().collect();
    year_set.len()
}

/// Gets or fits a baseline for occurrence adjustment.
///
/// If QM was run, uses the QM result's baseline. Otherwise fits directly
/// from the (possibly QM-adjusted) precipitation.
fn get_or_fit_baseline(
    qm_result: Option<&QmResult>,
    precip: &[f64],
    month: &[u8],
    occurrence_config: &OccurrenceConfig,
) -> BaselineFit {
    if let Some(qm) = qm_result {
        return qm.baseline().clone();
    }
    // Build a QmConfig with matching threshold for fit_monthly
    let qm_cfg = QmConfig::new().with_intensity_threshold(occurrence_config.intensity_threshold());
    zeus_quantile_map::fit_monthly(precip, month, &qm_cfg)
}

/// Applies the full climate perturbation pipeline.
///
/// Pipeline order: **temperature -> QM intensity -> occurrence -> safety rails**.
///
/// Each step is optional and controlled by the corresponding field in
/// [`PerturbConfig`]. When all modules are `None`, the function returns
/// the input data unchanged (with safety rails still applied).
///
/// # Arguments
///
/// * `precip` — Daily precipitation values (mm/day).
/// * `temp` — Daily mean temperature values.
/// * `temp_min` — Daily minimum temperature values.
/// * `temp_max` — Daily maximum temperature values.
/// * `month` — Calendar month (1–12) for each day.
/// * `year` — Simulation year index (1-based, contiguous) for each day.
/// * `config` — Pipeline configuration.
///
/// # Errors
///
/// Returns [`PerturbError`] on invalid inputs, configuration problems,
/// or failures from underlying modules.
#[tracing::instrument(skip(precip, temp, temp_min, temp_max, month, year, config))]
pub fn apply_perturbations(
    precip: &[f64],
    temp: &[f64],
    temp_min: &[f64],
    temp_max: &[f64],
    month: &[u8],
    year: &[u32],
    config: &PerturbConfig,
) -> Result<PerturbResult, PerturbError> {
    config.validate()?;

    // --- Input validation ---
    let n = precip.len();
    if n == 0 {
        return Err(PerturbError::EmptyData);
    }
    for (slice, name) in [
        (temp.len(), "temp"),
        (temp_min.len(), "temp_min"),
        (temp_max.len(), "temp_max"),
        (month.len(), "month"),
        (year.len(), "year"),
    ] {
        if slice != n {
            return Err(PerturbError::LengthMismatch {
                expected: n,
                got: slice,
                field: name.to_string(),
            });
        }
    }
    for &m in month {
        if !(1..=12).contains(&m) {
            return Err(PerturbError::InvalidMonth { month: m });
        }
    }

    let n_years = count_years(year);
    let mut rng = make_rng(config.seed());

    let mut modules = ModulesApplied {
        temperature: false,
        quantile_map: false,
        occurrence: false,
        safety_rails: true, // always applied
    };

    // --- Step 1: Temperature ---
    let (adj_temp, adj_tmin, adj_tmax) = if let Some(temp_config) = config.temp() {
        let deltas = TempDeltas::from_config(temp_config, n_years)?;
        let tr = adjust_temperature(temp, Some(temp_min), Some(temp_max), month, year, &deltas)?;
        modules.temperature = true;
        debug!(module = "temperature", "applied");
        let tmin = tr.temp_min().map(|s| s.to_vec());
        let tmax = tr.temp_max().map(|s| s.to_vec());
        (tr.into_temp(), tmin, tmax)
    } else {
        (
            temp.to_vec(),
            Some(temp_min.to_vec()),
            Some(temp_max.to_vec()),
        )
    };

    // --- Step 2: QM Intensity ---
    let mut adj_precip;
    let qm_result;
    if let (Some(qm_cfg), Some(mean_f), Some(var_f)) = (
        config.qm_config(),
        config.mean_factors(),
        config.var_factors(),
    ) {
        let qr =
            zeus_quantile_map::adjust_precipitation(precip, month, year, mean_f, var_f, qm_cfg)?;
        adj_precip = qr.adjusted().to_vec();
        qm_result = Some(qr);
        modules.quantile_map = true;
        debug!(module = "quantile_map", "applied");
    } else {
        adj_precip = precip.to_vec();
        qm_result = None;
    }

    // --- Step 3: Occurrence ---
    if let Some(occ_config) = config.occurrence() {
        let baseline = get_or_fit_baseline(qm_result.as_ref(), &adj_precip, month, occ_config);
        let occ_result =
            adjust_occurrence(&adj_precip, month, year, &baseline, occ_config, &mut rng)?;
        adj_precip = occ_result.into_precip();
        modules.occurrence = true;
        debug!(module = "occurrence", "applied");
    }

    // --- Step 4: Safety Rails ---
    apply_safety_rails(&mut adj_precip, config.precip_floor(), config.precip_cap());

    Ok(PerturbResult::new(
        adj_precip, adj_temp, adj_tmin, adj_tmax, modules,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: builds minimal daily data for `n_years` years, 30 days per month.
    #[allow(clippy::type_complexity)]
    fn make_daily_data(
        n_years: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>, Vec<u32>) {
        let days_per_month = 30;
        let total = n_years * 12 * days_per_month;
        let mut precip = Vec::with_capacity(total);
        let mut temp = Vec::with_capacity(total);
        let mut temp_min = Vec::with_capacity(total);
        let mut temp_max = Vec::with_capacity(total);
        let mut month = Vec::with_capacity(total);
        let mut year = Vec::with_capacity(total);

        for y in 1..=n_years {
            for m in 1u8..=12 {
                for d in 0..days_per_month {
                    // Simple alternating wet/dry pattern
                    let p = if d % 2 == 0 { 5.0 } else { 0.0 };
                    precip.push(p);
                    temp.push(20.0 + m as f64);
                    temp_min.push(15.0 + m as f64);
                    temp_max.push(25.0 + m as f64);
                    month.push(m);
                    year.push(y as u32);
                }
            }
        }
        (precip, temp, temp_min, temp_max, month, year)
    }

    #[test]
    fn all_disabled_passthrough() {
        let (precip, temp, temp_min, temp_max, month, year) = make_daily_data(1);
        let config = PerturbConfig::new().with_seed(42);

        let result =
            apply_perturbations(&precip, &temp, &temp_min, &temp_max, &month, &year, &config)
                .expect("should succeed");

        // Precip should pass through safety rails unchanged (already clean)
        assert_eq!(result.precip(), &precip[..]);
        // Temp should be unchanged
        assert_eq!(result.temp(), &temp[..]);

        let ma = result.modules_applied();
        assert!(!ma.temperature);
        assert!(!ma.quantile_map);
        assert!(!ma.occurrence);
        assert!(ma.safety_rails);
    }

    #[test]
    fn temp_only() {
        let (precip, temp, temp_min, temp_max, month, year) = make_daily_data(1);
        let delta = 2.0;
        let config = PerturbConfig::new()
            .with_temp(TempConfig::new([delta; 12], false))
            .with_seed(42);

        let result =
            apply_perturbations(&precip, &temp, &temp_min, &temp_max, &month, &year, &config)
                .expect("should succeed");

        // Precip should be unchanged (after safety rails, but data is clean)
        assert_eq!(result.precip(), &precip[..]);

        // Temp should be shifted by delta
        for (i, &t) in result.temp().iter().enumerate() {
            assert!(
                (t - (temp[i] + delta)).abs() < 1e-10,
                "temp mismatch at index {i}: expected {}, got {t}",
                temp[i] + delta,
            );
        }

        let ma = result.modules_applied();
        assert!(ma.temperature);
        assert!(!ma.quantile_map);
        assert!(!ma.occurrence);
        assert!(ma.safety_rails);
    }

    #[test]
    fn validates_empty_input() {
        let config = PerturbConfig::new();
        let result = apply_perturbations(&[], &[], &[], &[], &[], &[], &config);
        assert!(matches!(result, Err(PerturbError::EmptyData)));
    }

    #[test]
    fn validates_length_mismatch() {
        let config = PerturbConfig::new();
        let result = apply_perturbations(
            &[1.0, 2.0],
            &[20.0], // wrong length
            &[15.0, 16.0],
            &[25.0, 26.0],
            &[1, 2],
            &[1, 1],
            &config,
        );
        assert!(
            matches!(result, Err(PerturbError::LengthMismatch { .. })),
            "expected LengthMismatch, got {result:?}",
        );
    }
}
