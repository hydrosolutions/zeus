//! Scenario factor matrices for quantile mapping.

use crate::error::QuantileMapError;

/// A validated matrix of monthly scaling factors indexed by year.
///
/// Internally stored as `Vec<[f64; 12]>`, which makes the month-width
/// invariant a compile-time guarantee. Access pattern is always
/// `factors[year_idx][month_idx]`.
#[derive(Debug, Clone)]
pub struct ScenarioFactors {
    inner: Vec<[f64; 12]>,
}

impl ScenarioFactors {
    /// Creates a new `ScenarioFactors` from a vector of monthly factor rows.
    ///
    /// # Errors
    ///
    /// Returns [`QuantileMapError::EmptyData`] if `inner` is empty.
    ///
    /// Returns [`QuantileMapError::InvalidConfig`] if any value is not finite
    /// or is not strictly positive.
    pub fn new(inner: Vec<[f64; 12]>) -> Result<Self, QuantileMapError> {
        if inner.is_empty() {
            return Err(QuantileMapError::EmptyData);
        }

        for (year_idx, row) in inner.iter().enumerate() {
            for (month_idx, &val) in row.iter().enumerate() {
                if !val.is_finite() || val <= 0.0 {
                    return Err(QuantileMapError::InvalidConfig {
                        reason: format!(
                            "factor at year_idx={year_idx}, month_idx={month_idx} is {val}; \
                             all factors must be finite and > 0"
                        ),
                    });
                }
            }
        }

        Ok(Self { inner })
    }

    /// Creates factors where every year has the same monthly values.
    ///
    /// This is a convenience constructor that skips validation. The caller is
    /// responsible for ensuring all values in `monthly` are finite and positive.
    pub fn uniform(n_years: usize, monthly: [f64; 12]) -> Self {
        Self {
            inner: vec![monthly; n_years],
        }
    }

    /// Returns the number of years in the factor matrix.
    pub fn n_years(&self) -> usize {
        self.inner.len()
    }

    /// Returns the factor for a given year index and 1-indexed month.
    ///
    /// # Panics
    ///
    /// Panics if `year_idx` is out of range or `month` is not in `1..=12`.
    pub fn get(&self, year_idx: u32, month: u8) -> f64 {
        self.inner[year_idx as usize][month as usize - 1]
    }

    /// Returns the inner data as a slice of `[f64; 12]` rows.
    pub fn as_slice(&self) -> &[[f64; 12]] {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        let rows = vec![
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
            [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        ];
        let factors = ScenarioFactors::new(rows).unwrap();
        assert_eq!(factors.n_years(), 3);
        assert!((factors.get(0, 1) - 1.0).abs() < f64::EPSILON);
        assert!((factors.get(1, 12) - 3.1).abs() < f64::EPSILON);
        assert!((factors.get(2, 6) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn new_invalid_nan() {
        let mut row = [1.0; 12];
        row[5] = f64::NAN;
        let result = ScenarioFactors::new(vec![row]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("month_idx=5"));
    }

    #[test]
    fn new_invalid_negative() {
        let mut row = [1.0; 12];
        row[3] = -0.5;
        let result = ScenarioFactors::new(vec![row]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("month_idx=3"));
        assert!(err.contains("-0.5"));
    }

    #[test]
    fn new_invalid_zero() {
        let mut row = [1.0; 12];
        row[0] = 0.0;
        let result = ScenarioFactors::new(vec![row]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("month_idx=0"));
    }

    #[test]
    fn new_empty() {
        let result = ScenarioFactors::new(vec![]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), QuantileMapError::EmptyData));
    }

    #[test]
    fn uniform_construction() {
        let factors = ScenarioFactors::uniform(5, [1.0; 12]);
        assert_eq!(factors.n_years(), 5);
        for year in 0..5u32 {
            for month in 1..=12u8 {
                assert!((factors.get(year, month) - 1.0).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn get_accessors() {
        let row_a = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let row_b = [
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ];
        let factors = ScenarioFactors::new(vec![row_a, row_b]).unwrap();

        // Verify months 1..=12 for year 0.
        for month in 1..=12u8 {
            let expected = month as f64;
            assert!(
                (factors.get(0, month) - expected).abs() < f64::EPSILON,
                "year 0, month {month}: expected {expected}, got {}",
                factors.get(0, month)
            );
        }

        // Verify different year gives different values.
        assert!((factors.get(1, 1) - 10.0).abs() < f64::EPSILON);
        assert!((factors.get(1, 12) - 120.0).abs() < f64::EPSILON);
    }

    #[test]
    fn as_slice() {
        let rows = vec![[1.0; 12], [2.0; 12], [3.0; 12]];
        let factors = ScenarioFactors::new(rows).unwrap();
        assert_eq!(factors.as_slice().len(), factors.n_years());
    }
}
