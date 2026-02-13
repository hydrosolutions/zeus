//! Observed daily data with precomputed lookup tables.

use crate::error::ResampleError;

/// Owned observation data with precomputed lookup tables.
///
/// Built once from the observation record. Reused across all simulated
/// years and realizations. Stores owned copies of all input arrays.
pub struct ObsData {
    precip: Vec<f64>,
    temp: Vec<f64>,
    months: Vec<u8>,
    days: Vec<u8>,
    water_years: Vec<i32>,

    /// Per month-day: indices into the observation arrays.
    /// Index = `(month - 1) * 31 + (day - 1)`, invalid slots are empty.
    month_day_indices: [Vec<usize>; 372],

    /// Per water-year: indices into the observation arrays.
    /// Indexed by `(water_year - min_water_year)`.
    year_day_indices: Vec<Vec<usize>>,

    /// Observed annual precipitation per water year (for annual KNN).
    annual_precip: Vec<f64>,

    /// Unique water years in order.
    unique_water_years: Vec<i32>,

    /// Minimum water year (for indexing year_day_indices).
    min_water_year: i32,
}

impl ObsData {
    /// Build from observed daily data.
    ///
    /// All slices must have the same length (one entry per observed day).
    ///
    /// # Errors
    ///
    /// Returns [`ResampleError`] if inputs are empty, have mismatched
    /// lengths, or contain non-finite values.
    pub fn new(
        obs_precip: &[f64],
        obs_temp: &[f64],
        obs_months: &[u8],
        obs_days: &[u8],
        obs_water_years: &[i32],
    ) -> Result<Self, ResampleError> {
        let n = obs_precip.len();
        if n == 0 {
            return Err(ResampleError::EmptyData);
        }
        if obs_temp.len() != n {
            return Err(ResampleError::LengthMismatch {
                field: "temp",
                expected: n,
                got: obs_temp.len(),
            });
        }
        if obs_months.len() != n {
            return Err(ResampleError::LengthMismatch {
                field: "months",
                expected: n,
                got: obs_months.len(),
            });
        }
        if obs_days.len() != n {
            return Err(ResampleError::LengthMismatch {
                field: "days",
                expected: n,
                got: obs_days.len(),
            });
        }
        if obs_water_years.len() != n {
            return Err(ResampleError::LengthMismatch {
                field: "water_years",
                expected: n,
                got: obs_water_years.len(),
            });
        }
        if obs_precip.iter().any(|v| !v.is_finite()) {
            return Err(ResampleError::NonFiniteInput { field: "precip" });
        }
        if obs_temp.iter().any(|v| !v.is_finite()) {
            return Err(ResampleError::NonFiniteInput { field: "temp" });
        }
        for &m in obs_months {
            if !(1..=12).contains(&m) {
                return Err(ResampleError::InvalidMonth { month: m });
            }
        }

        // Build month-day lookup
        let mut month_day_indices: [Vec<usize>; 372] = std::array::from_fn(|_| Vec::new());
        for (i, (&m, &d)) in obs_months.iter().zip(obs_days.iter()).enumerate() {
            let slot = (m as usize - 1) * 31 + (d as usize - 1);
            month_day_indices[slot].push(i);
        }

        // Build year-day lookup and annual precip
        let min_wy = *obs_water_years.iter().min().expect("non-empty");
        let max_wy = *obs_water_years.iter().max().expect("non-empty");
        let n_years = (max_wy - min_wy + 1) as usize;
        let mut year_day_indices = vec![Vec::new(); n_years];
        for (i, &wy) in obs_water_years.iter().enumerate() {
            year_day_indices[(wy - min_wy) as usize].push(i);
        }

        // Unique water years and annual precip
        let mut unique_water_years = Vec::new();
        let mut annual_precip = Vec::new();
        for (offset, days) in year_day_indices.iter().enumerate() {
            if !days.is_empty() {
                let wy = min_wy + offset as i32;
                unique_water_years.push(wy);
                let total: f64 = days.iter().map(|&i| obs_precip[i]).sum();
                annual_precip.push(total);
            }
        }

        Ok(Self {
            precip: obs_precip.to_vec(),
            temp: obs_temp.to_vec(),
            months: obs_months.to_vec(),
            days: obs_days.to_vec(),
            water_years: obs_water_years.to_vec(),
            month_day_indices,
            year_day_indices,
            annual_precip,
            unique_water_years,
            min_water_year: min_wy,
        })
    }

    /// Returns the precipitation values.
    pub fn precip(&self) -> &[f64] {
        &self.precip
    }

    /// Returns the temperature values.
    pub fn temp(&self) -> &[f64] {
        &self.temp
    }

    /// Returns the month values.
    pub fn months(&self) -> &[u8] {
        &self.months
    }

    /// Returns the day-of-month values.
    pub fn days(&self) -> &[u8] {
        &self.days
    }

    /// Returns the water year values.
    pub fn water_years(&self) -> &[i32] {
        &self.water_years
    }

    /// Returns the total number of observed days.
    pub fn len(&self) -> usize {
        self.precip.len()
    }

    /// Returns `true` if there are no observed days.
    pub fn is_empty(&self) -> bool {
        self.precip.is_empty()
    }

    /// Returns indices for a specific month-day combination.
    pub fn month_day_candidates(&self, month: u8, day: u8) -> &[usize] {
        let slot = (month as usize - 1) * 31 + (day as usize - 1);
        &self.month_day_indices[slot]
    }

    /// Returns indices for a specific water year.
    pub fn year_day_candidates(&self, wy: i32) -> &[usize] {
        let offset = (wy - self.min_water_year) as usize;
        if offset < self.year_day_indices.len() {
            &self.year_day_indices[offset]
        } else {
            &[]
        }
    }

    /// Returns annual precipitation totals per water year.
    pub fn annual_precip(&self) -> &[f64] {
        &self.annual_precip
    }

    /// Returns the unique water years.
    pub fn unique_water_years(&self) -> &[i32] {
        &self.unique_water_years
    }

    /// Returns the number of unique observed water years.
    pub fn n_obs_years(&self) -> usize {
        self.unique_water_years.len()
    }

    /// Returns the minimum water year.
    pub fn min_water_year(&self) -> i32 {
        self.min_water_year
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_obs() -> ObsData {
        // 10 days: Jan 1-5, Feb 1-5, all WY 2000
        let precip = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let temp = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        let months = vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
        let days = vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5];
        let water_years = vec![2000; 10];
        ObsData::new(&precip, &temp, &months, &days, &water_years).unwrap()
    }

    #[test]
    fn basic_construction() {
        let obs = sample_obs();
        assert_eq!(obs.len(), 10);
        assert!(!obs.is_empty());
        assert_eq!(obs.n_obs_years(), 1);
        assert_eq!(obs.unique_water_years(), &[2000]);
        assert!((obs.annual_precip()[0] - 45.0).abs() < 1e-10);
    }

    #[test]
    fn month_day_lookup() {
        let obs = sample_obs();
        assert_eq!(obs.month_day_candidates(1, 1), &[0]);
        assert_eq!(obs.month_day_candidates(1, 3), &[2]);
        assert_eq!(obs.month_day_candidates(2, 1), &[5]);
        assert!(obs.month_day_candidates(3, 1).is_empty());
    }

    #[test]
    fn year_day_lookup() {
        let obs = sample_obs();
        let days = obs.year_day_candidates(2000);
        assert_eq!(days.len(), 10);
        assert!(obs.year_day_candidates(1999).is_empty());
    }

    #[test]
    fn multi_year() {
        let precip = vec![1.0, 2.0, 3.0, 4.0];
        let temp = vec![10.0, 20.0, 30.0, 40.0];
        let months = vec![1, 1, 1, 1];
        let days = vec![1, 2, 1, 2];
        let water_years = vec![2000, 2000, 2001, 2001];
        let obs = ObsData::new(&precip, &temp, &months, &days, &water_years).unwrap();
        assert_eq!(obs.n_obs_years(), 2);
        assert!((obs.annual_precip()[0] - 3.0).abs() < 1e-10); // WY 2000
        assert!((obs.annual_precip()[1] - 7.0).abs() < 1e-10); // WY 2001
    }

    #[test]
    fn error_empty() {
        let result = ObsData::new(&[], &[], &[], &[], &[]);
        assert!(matches!(result, Err(ResampleError::EmptyData)));
    }

    #[test]
    fn error_length_mismatch() {
        let result = ObsData::new(&[1.0], &[1.0, 2.0], &[1], &[1], &[2000]);
        assert!(matches!(
            result,
            Err(ResampleError::LengthMismatch { field: "temp", .. })
        ));
    }

    #[test]
    fn error_non_finite_precip() {
        let result = ObsData::new(&[f64::NAN], &[1.0], &[1], &[1], &[2000]);
        assert!(matches!(
            result,
            Err(ResampleError::NonFiniteInput { field: "precip" })
        ));
    }

    #[test]
    fn error_non_finite_temp() {
        let result = ObsData::new(&[1.0], &[f64::INFINITY], &[1], &[1], &[2000]);
        assert!(matches!(
            result,
            Err(ResampleError::NonFiniteInput { field: "temp" })
        ));
    }

    #[test]
    fn error_invalid_month() {
        let result = ObsData::new(&[1.0], &[1.0], &[13], &[1], &[2000]);
        assert!(matches!(
            result,
            Err(ResampleError::InvalidMonth { month: 13 })
        ));
    }

    #[test]
    fn accessors() {
        let obs = sample_obs();
        assert_eq!(obs.precip().len(), 10);
        assert_eq!(obs.temp().len(), 10);
        assert_eq!(obs.months().len(), 10);
        assert_eq!(obs.days().len(), 10);
        assert_eq!(obs.water_years().len(), 10);
        assert_eq!(obs.min_water_year(), 2000);
    }
}
