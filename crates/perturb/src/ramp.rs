//! Transient ramp utilities for scaling factors and deltas over time.

use crate::error::PerturbError;

/// How a monthly value should be ramped across simulation years.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RampMode {
    /// Ramp from 1.0 (no change) to `2 * factor - 1` (full change).
    /// Used for multiplicative factors like precipitation occurrence.
    Multiplicative,
    /// Ramp from 0.0 (no change) to `2 * delta` (full change).
    /// Used for additive deltas like temperature shifts.
    Additive,
}

/// Builds a per-year ramp of monthly values.
///
/// When `transient` is `false`, every year gets the raw `monthly` values unchanged.
///
/// When `transient` is `true`, values are linearly interpolated across years:
///
/// - **Multiplicative**: year `y` gets `1.0 + (2 * monthly[m] - 2) * frac`
///   where `frac = (y + 1) / n_years`. At year 0, frac ≈ 1/n; at the last year, frac = 1.0.
///   The midpoint reproduces the original `monthly` value.
///
/// - **Additive**: year `y` gets `2 * monthly[m] * frac`.
///   The midpoint reproduces the original `monthly` value.
///
/// # Errors
///
/// Returns [`PerturbError::InvalidConfig`] if `n_years` is zero.
pub fn build_ramp(
    monthly: &[f64; 12],
    n_years: usize,
    transient: bool,
    mode: RampMode,
) -> Result<Vec<[f64; 12]>, PerturbError> {
    if n_years == 0 {
        return Err(PerturbError::InvalidConfig {
            reason: "n_years must be > 0".to_string(),
        });
    }

    if !transient {
        return Ok(vec![*monthly; n_years]);
    }

    let mut result = Vec::with_capacity(n_years);
    for y in 0..n_years {
        let frac = (y + 1) as f64 / n_years as f64;
        let mut row = [0.0; 12];
        for m in 0..12 {
            row[m] = match mode {
                RampMode::Multiplicative => 1.0 + (2.0 * monthly[m] - 2.0) * frac,
                RampMode::Additive => 2.0 * monthly[m] * frac,
            };
        }
        result.push(row);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn non_transient_returns_constant() {
        let monthly = [1.5; 12];
        let ramp = build_ramp(&monthly, 5, false, RampMode::Multiplicative).unwrap();
        assert_eq!(ramp.len(), 5);
        for row in &ramp {
            for &v in row {
                assert_relative_eq!(v, 1.5);
            }
        }
    }

    #[test]
    fn multiplicative_endpoints() {
        let monthly = [1.2; 12]; // factor = 1.2
        let n = 100;
        let ramp = build_ramp(&monthly, n, true, RampMode::Multiplicative).unwrap();

        // First year: frac = 1/100 → 1 + (2*1.2 - 2) * 0.01 = 1 + 0.4*0.01 = 1.004
        assert_relative_eq!(ramp[0][0], 1.0 + 0.4 * (1.0 / 100.0), epsilon = 1e-10);

        // Last year: frac = 1.0 → 1 + (2*1.2 - 2)*1.0 = 1.4
        assert_relative_eq!(ramp[n - 1][0], 1.4, epsilon = 1e-10);
    }

    #[test]
    fn multiplicative_midpoint_recovers_original() {
        // At the midpoint frac=0.5: 1 + (2*f - 2)*0.5 = 1 + f - 1 = f
        let monthly = [0.8; 12];
        let n = 100;
        let ramp = build_ramp(&monthly, n, true, RampMode::Multiplicative).unwrap();
        // Year 49 has frac = 50/100 = 0.5
        assert_relative_eq!(ramp[49][0], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn additive_endpoints() {
        let monthly = [2.0; 12]; // delta = 2.0
        let n = 100;
        let ramp = build_ramp(&monthly, n, true, RampMode::Additive).unwrap();

        // First year: frac = 1/100 → 2*2.0 * 0.01 = 0.04
        assert_relative_eq!(ramp[0][0], 0.04, epsilon = 1e-10);

        // Last year: frac = 1.0 → 2*2.0 = 4.0
        assert_relative_eq!(ramp[n - 1][0], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn additive_midpoint_recovers_original() {
        // At midpoint frac=0.5: 2*delta*0.5 = delta
        let monthly = [3.0; 12];
        let n = 100;
        let ramp = build_ramp(&monthly, n, true, RampMode::Additive).unwrap();
        // Year 49 has frac = 50/100 = 0.5
        assert_relative_eq!(ramp[49][0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn single_year_transient() {
        let monthly = [1.5; 12];
        let ramp = build_ramp(&monthly, 1, true, RampMode::Multiplicative).unwrap();
        // frac = 1/1 = 1.0, so value = 1 + (2*1.5 - 2)*1 = 2.0
        assert_relative_eq!(ramp[0][0], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn zero_years_returns_error() {
        let monthly = [1.0; 12];
        let result = build_ramp(&monthly, 0, false, RampMode::Additive);
        assert!(result.is_err());
    }

    #[test]
    fn varying_monthly_values() {
        let mut monthly = [0.0; 12];
        for (i, val) in monthly.iter_mut().enumerate() {
            *val = (i + 1) as f64;
        }
        let ramp = build_ramp(&monthly, 10, true, RampMode::Additive).unwrap();
        // Last year frac=1.0: value[m] = 2 * monthly[m]
        for m in 0..12 {
            assert_relative_eq!(ramp[9][m], 2.0 * monthly[m], epsilon = 1e-10);
        }
    }
}
