//! Temperature adjustment module.
//!
//! Applies additive per-year monthly deltas to temperature series,
//! optionally ramped linearly over the simulation period.

use crate::config::TempConfig;
use crate::error::PerturbError;
use crate::ramp::{RampMode, build_ramp};
use crate::result::TempResult;

/// Pre-computed temperature deltas per year.
#[derive(Debug, Clone)]
pub struct TempDeltas {
    inner: Vec<[f64; 12]>,
}

impl TempDeltas {
    /// Build deltas from a `TempConfig` and the number of simulation years.
    ///
    /// If `config.transient()` is true, ramps additively from 0 to 2*delta.
    /// Otherwise, every year uses the raw deltas.
    pub fn from_config(config: &TempConfig, n_years: usize) -> Result<Self, PerturbError> {
        let ramp = build_ramp(
            config.deltas(),
            n_years,
            config.transient(),
            RampMode::Additive,
        )?;
        Ok(Self { inner: ramp })
    }

    /// Returns the delta for a 1-indexed year and 1-indexed month.
    ///
    /// Year indices in the data are 1-based (year 1 maps to inner\[0\]).
    pub fn get(&self, year: u32, month: u8) -> f64 {
        self.inner[(year - 1) as usize][(month - 1) as usize]
    }

    /// Returns the number of years.
    pub fn n_years(&self) -> usize {
        self.inner.len()
    }
}

/// Adjusts temperature series by adding per-year monthly deltas.
///
/// For each day, adds `deltas.get(year, month)` to the temperature value.
/// `temp_min` and `temp_max`, if provided, are shifted by the same delta.
/// NaN values pass through unchanged.
///
/// # Errors
///
/// Returns `PerturbError::EmptyData` if `temp` is empty.
/// Returns `PerturbError::LengthMismatch` if `month` or `year` lengths differ from `temp`.
/// Returns `PerturbError::InvalidMonth` if any month is outside 1..=12.
/// Returns `PerturbError::YearMismatch` if any year index is out of range for the deltas.
pub fn adjust_temperature(
    temp: &[f64],
    temp_min: Option<&[f64]>,
    temp_max: Option<&[f64]>,
    month: &[u8],
    year: &[u32],
    deltas: &TempDeltas,
) -> Result<TempResult, PerturbError> {
    let n = temp.len();

    // Validate non-empty
    if n == 0 {
        return Err(PerturbError::EmptyData);
    }

    // Validate lengths match
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
    if let Some(tmin) = temp_min
        && tmin.len() != n
    {
        return Err(PerturbError::LengthMismatch {
            expected: n,
            got: tmin.len(),
            field: "temp_min".to_string(),
        });
    }
    if let Some(tmax) = temp_max
        && tmax.len() != n
    {
        return Err(PerturbError::LengthMismatch {
            expected: n,
            got: tmax.len(),
            field: "temp_max".to_string(),
        });
    }

    // Validate all months in 1..=12
    for &m in month {
        if !(1..=12).contains(&m) {
            return Err(PerturbError::InvalidMonth { month: m });
        }
    }

    // Validate all year indices in 1..=deltas.n_years()
    let ny = deltas.n_years();
    for &y in year {
        if y == 0 || y as usize > ny {
            return Err(PerturbError::YearMismatch {
                expected: ny,
                got: y as usize,
            });
        }
    }

    // Apply deltas
    let mut out_temp = Vec::with_capacity(n);
    let mut out_min = temp_min.map(|_| Vec::with_capacity(n));
    let mut out_max = temp_max.map(|_| Vec::with_capacity(n));

    for i in 0..n {
        let delta = deltas.get(year[i], month[i]);

        // Mean temperature
        if temp[i].is_nan() {
            out_temp.push(f64::NAN);
        } else {
            out_temp.push(temp[i] + delta);
        }

        // Min temperature
        if let (Some(v), Some(tmin)) = (&mut out_min, temp_min) {
            if tmin[i].is_nan() {
                v.push(f64::NAN);
            } else {
                v.push(tmin[i] + delta);
            }
        }

        // Max temperature
        if let (Some(v), Some(tmax)) = (&mut out_max, temp_max) {
            if tmax[i].is_nan() {
                v.push(f64::NAN);
            } else {
                v.push(tmax[i] + delta);
            }
        }
    }

    Ok(TempResult::new(out_temp, out_min, out_max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Helper: build non-transient deltas from a uniform array.
    fn uniform_deltas(delta: f64, n_years: usize) -> TempDeltas {
        let config = TempConfig::new([delta; 12], false);
        TempDeltas::from_config(&config, n_years).unwrap()
    }

    #[test]
    fn zero_delta_identity() {
        let temp = vec![10.0, 20.0, 30.0];
        let month = vec![1u8, 2, 3];
        let year = vec![1u32, 1, 1];
        let deltas = uniform_deltas(0.0, 1);

        let result = adjust_temperature(&temp, None, None, &month, &year, &deltas).unwrap();
        for (a, b) in result.temp().iter().zip(temp.iter()) {
            assert_relative_eq!(a, b);
        }
    }

    #[test]
    fn uniform_delta() {
        let temp = vec![10.0, 20.0, 30.0];
        let month = vec![1u8, 2, 3];
        let year = vec![1u32, 1, 1];
        let deltas = uniform_deltas(2.0, 1);

        let result = adjust_temperature(&temp, None, None, &month, &year, &deltas).unwrap();
        for (a, b) in result.temp().iter().zip(temp.iter()) {
            assert_relative_eq!(*a, *b + 2.0);
        }
    }

    #[test]
    fn monthly_varying_delta() {
        // Deltas that differ by month: month m gets delta = m as f64
        let mut month_deltas = [0.0; 12];
        for (i, d) in month_deltas.iter_mut().enumerate() {
            *d = (i + 1) as f64;
        }
        let config = TempConfig::new(month_deltas, false);
        let deltas = TempDeltas::from_config(&config, 1).unwrap();

        // 12 days, one per month
        let temp: Vec<f64> = vec![100.0; 12];
        let month: Vec<u8> = (1..=12).collect();
        let year: Vec<u32> = vec![1; 12];

        let result = adjust_temperature(&temp, None, None, &month, &year, &deltas).unwrap();
        for i in 0..12 {
            let expected = 100.0 + (i + 1) as f64;
            assert_relative_eq!(result.temp()[i], expected);
        }
    }

    #[test]
    fn nan_passthrough() {
        let temp = vec![f64::NAN, 20.0, f64::NAN];
        let month = vec![1u8, 2, 3];
        let year = vec![1u32, 1, 1];
        let deltas = uniform_deltas(5.0, 1);

        let result = adjust_temperature(&temp, None, None, &month, &year, &deltas).unwrap();
        assert!(result.temp()[0].is_nan());
        assert_relative_eq!(result.temp()[1], 25.0);
        assert!(result.temp()[2].is_nan());
    }

    #[test]
    fn temp_min_max_shifted() {
        let temp = vec![20.0, 25.0];
        let temp_min = vec![15.0, 20.0];
        let temp_max = vec![25.0, 30.0];
        let month = vec![1u8, 2];
        let year = vec![1u32, 1];
        let deltas = uniform_deltas(3.0, 1);

        let result = adjust_temperature(
            &temp,
            Some(&temp_min),
            Some(&temp_max),
            &month,
            &year,
            &deltas,
        )
        .unwrap();

        assert_relative_eq!(result.temp()[0], 23.0);
        assert_relative_eq!(result.temp()[1], 28.0);

        let rmin = result.temp_min().unwrap();
        assert_relative_eq!(rmin[0], 18.0);
        assert_relative_eq!(rmin[1], 23.0);

        let rmax = result.temp_max().unwrap();
        assert_relative_eq!(rmax[0], 28.0);
        assert_relative_eq!(rmax[1], 33.0);
    }

    #[test]
    fn transient_ramp() {
        // With transient=true and delta=2.0 over 10 years:
        // Year 1: frac=1/10 => 2*2.0*0.1 = 0.4
        // Year 5: frac=5/10 => 2*2.0*0.5 = 2.0 (the original delta)
        // Year 10: frac=10/10 => 2*2.0*1.0 = 4.0
        let config = TempConfig::new([2.0; 12], true);
        let deltas = TempDeltas::from_config(&config, 10).unwrap();

        // Verify early years have smaller deltas
        let early = deltas.get(1, 1);
        let mid = deltas.get(5, 1);
        let late = deltas.get(10, 1);

        assert_relative_eq!(early, 0.4, epsilon = 1e-10);
        assert_relative_eq!(mid, 2.0, epsilon = 1e-10);
        assert_relative_eq!(late, 4.0, epsilon = 1e-10);

        assert!(early < mid);
        assert!(mid < late);

        // Apply to data: one day in year 1 vs year 10
        let temp = vec![10.0, 10.0];
        let month = vec![1u8, 1];
        let year = vec![1u32, 10];

        let result = adjust_temperature(&temp, None, None, &month, &year, &deltas).unwrap();
        assert_relative_eq!(result.temp()[0], 10.4, epsilon = 1e-10);
        assert_relative_eq!(result.temp()[1], 14.0, epsilon = 1e-10);
    }

    #[test]
    fn empty_input_error() {
        let deltas = uniform_deltas(1.0, 1);
        let result = adjust_temperature(&[], None, None, &[], &[], &deltas);
        assert!(matches!(result, Err(PerturbError::EmptyData)));
    }

    #[test]
    fn length_mismatch_error() {
        let temp = vec![10.0, 20.0];
        let month = vec![1u8]; // wrong length
        let year = vec![1u32, 1];
        let deltas = uniform_deltas(1.0, 1);

        let result = adjust_temperature(&temp, None, None, &month, &year, &deltas);
        assert!(matches!(result, Err(PerturbError::LengthMismatch { .. })));

        // Also test year length mismatch
        let month2 = vec![1u8, 2];
        let year2 = vec![1u32];
        let result2 = adjust_temperature(&temp, None, None, &month2, &year2, &deltas);
        assert!(matches!(result2, Err(PerturbError::LengthMismatch { .. })));
    }

    #[test]
    fn invalid_month_error() {
        let temp = vec![10.0];
        let deltas = uniform_deltas(1.0, 1);

        // Month 0
        let result = adjust_temperature(&temp, None, None, &[0u8], &[1u32], &deltas);
        assert!(matches!(
            result,
            Err(PerturbError::InvalidMonth { month: 0 })
        ));

        // Month 13
        let result = adjust_temperature(&temp, None, None, &[13u8], &[1u32], &deltas);
        assert!(matches!(
            result,
            Err(PerturbError::InvalidMonth { month: 13 })
        ));
    }
}
