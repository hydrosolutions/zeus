//! Gamma-to-Gamma parametric quantile mapping for precipitation adjustment.
//!
//! This crate adjusts daily precipitation intensities under climate scenarios
//! using parametric quantile mapping with Gamma distributions.
//!
//! # Pipeline
//!
//! 1. **Fit** baseline Gamma distributions per calendar month (MME)
//! 2. **Scale** baseline moments by scenario factors to derive target parameters
//! 3. **Map** each wet-day value: pgamma → clamp → \[tail amplify\] → qgamma → \[enforce mean\]
//!
//! Dry days and NaN values pass through unchanged.
//!
//! # Glossary
//!
//! - **MME**: Method of Moments Estimation (shape = mean²/var, scale = var/mean)
//! - **pgamma**: Gamma CDF (cumulative distribution function)
//! - **qgamma**: Gamma quantile function (inverse CDF)
//! - **CV**: Coefficient of variation (σ/μ)
//!
//! # Quick Start
//!
//! ```no_run
//! use zeus_quantile_map::{QmConfig, ScenarioFactors, adjust_precipitation};
//!
//! // Daily precipitation, months (1-12), years (1-based index)
//! let precip = vec![0.0, 5.2, 0.0, 12.1, 3.4]; // mm/day
//! let month = vec![1u8, 1, 1, 1, 1];
//! let year = vec![1u32, 1, 1, 1, 1];
//!
//! let factors = ScenarioFactors::uniform(1, [0.9; 12]);
//! let config = QmConfig::new();
//!
//! let result = adjust_precipitation(&precip, &month, &year, &factors, &factors, &config);
//! ```

mod config;
mod error;
mod factors;
pub(crate) mod fit;
pub(crate) mod gamma;
mod result;
mod target;
pub(crate) mod transform;

pub use config::{FitMethod, QmConfig};
pub use error::QuantileMapError;
pub use factors::ScenarioFactors;
pub use fit::BaselineFit;
pub use gamma::GammaParams;
pub use result::QmResult;

use std::collections::BTreeSet;

/// Validates the inputs to [`adjust_precipitation`].
fn validate_inputs(
    precip: &[f64],
    month: &[u8],
    year: &[u32],
    mean_factors: &ScenarioFactors,
    var_factors: &ScenarioFactors,
) -> Result<(), QuantileMapError> {
    // 1. precip must not be empty.
    if precip.is_empty() {
        return Err(QuantileMapError::EmptyData);
    }

    // 2. All three slices must have the same length.
    if precip.len() != month.len() || precip.len() != year.len() {
        return Err(QuantileMapError::LengthMismatch {
            precip_len: precip.len(),
            months_len: month.len(),
            years_len: year.len(),
        });
    }

    // 3. All months must be in 1..=12.
    for &m in month {
        if !(1..=12).contains(&m) {
            return Err(QuantileMapError::InvalidMonth { month: m });
        }
    }

    // 4. Years must be contiguous 1..=n.
    let year_set: BTreeSet<u32> = year.iter().copied().collect();
    let min_year = *year_set.iter().next().unwrap(); // safe: non-empty
    let max_year = *year_set.iter().next_back().unwrap();

    if min_year != 1 || max_year != year_set.len() as u32 {
        return Err(QuantileMapError::NonContiguousYears {
            expected_max: max_year,
            reason: "year indices must be 1..=n with no gaps".to_string(),
        });
    }

    // 5. mean_factors year count must match data year count.
    if mean_factors.n_years() != max_year as usize {
        return Err(QuantileMapError::FactorYearMismatch {
            expected: max_year as usize,
            got: mean_factors.n_years(),
        });
    }

    // 6. var_factors year count must match data year count.
    if var_factors.n_years() != max_year as usize {
        return Err(QuantileMapError::FactorYearMismatch {
            expected: max_year as usize,
            got: var_factors.n_years(),
        });
    }

    Ok(())
}

/// Adjusts precipitation intensities using Gamma-to-Gamma quantile mapping.
///
/// # Arguments
///
/// * `precip` — Daily precipitation values (mm/day). Values ≤ `config.intensity_threshold()` are dry.
/// * `month` — Calendar month (1–12) for each day.
/// * `year` — Simulation year index (1-based, contiguous) for each day.
/// * `mean_factors` — Monthly mean scaling factors per year.
/// * `var_factors` — Monthly variance scaling factors per year.
/// * `config` — Quantile mapping configuration.
///
/// # Errors
///
/// Returns [`QuantileMapError`] on invalid inputs or if no months can be fitted.
pub fn adjust_precipitation(
    precip: &[f64],
    month: &[u8],
    year: &[u32],
    mean_factors: &ScenarioFactors,
    var_factors: &ScenarioFactors,
    config: &QmConfig,
) -> Result<QmResult, QuantileMapError> {
    config.validate()?;
    validate_inputs(precip, month, year, mean_factors, var_factors)?;

    let baseline = fit::fit_monthly(precip, month, config);

    if baseline.is_empty() {
        return Err(QuantileMapError::NoFittableMonths);
    }

    let target_params = target::compute_target_params(&baseline, mean_factors, var_factors, config);

    let (adjusted, perturbed_months, skipped_months) =
        transform::apply_quantile_mapping(precip, month, year, &baseline, &target_params, config)?;

    Ok(QmResult::new(
        adjusted,
        baseline,
        target_params,
        perturbed_months,
        skipped_months,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_empty_input() {
        let factors = ScenarioFactors::uniform(1, [1.0; 12]);
        let config = QmConfig::new();
        let result = adjust_precipitation(&[], &[], &[], &factors, &factors, &config);
        assert!(matches!(result, Err(QuantileMapError::EmptyData)));
    }

    #[test]
    fn validate_length_mismatch() {
        let factors = ScenarioFactors::uniform(1, [1.0; 12]);
        let config = QmConfig::new();
        let result = adjust_precipitation(&[1.0, 2.0], &[1], &[1], &factors, &factors, &config);
        assert!(matches!(
            result,
            Err(QuantileMapError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn validate_invalid_month() {
        let factors = ScenarioFactors::uniform(1, [1.0; 12]);
        let config = QmConfig::new();
        let result = adjust_precipitation(&[1.0], &[13], &[1], &factors, &factors, &config);
        assert!(matches!(
            result,
            Err(QuantileMapError::InvalidMonth { month: 13 })
        ));
    }

    #[test]
    fn validate_month_zero() {
        let factors = ScenarioFactors::uniform(1, [1.0; 12]);
        let config = QmConfig::new();
        let result = adjust_precipitation(&[1.0], &[0], &[1], &factors, &factors, &config);
        assert!(matches!(
            result,
            Err(QuantileMapError::InvalidMonth { month: 0 })
        ));
    }

    #[test]
    fn validate_non_contiguous_years() {
        let factors = ScenarioFactors::uniform(2, [1.0; 12]);
        let config = QmConfig::new();
        // years [1, 3] -- gap at 2
        let result =
            adjust_precipitation(&[1.0, 2.0], &[1, 1], &[1, 3], &factors, &factors, &config);
        assert!(matches!(
            result,
            Err(QuantileMapError::NonContiguousYears { .. })
        ));
    }

    #[test]
    fn validate_factor_year_mismatch() {
        let factors = ScenarioFactors::uniform(2, [1.0; 12]); // 2 years
        let config = QmConfig::new();
        // data has 1 year but factors have 2
        let result = adjust_precipitation(&[1.0], &[1], &[1], &factors, &factors, &config);
        assert!(matches!(
            result,
            Err(QuantileMapError::FactorYearMismatch { .. })
        ));
    }
}
