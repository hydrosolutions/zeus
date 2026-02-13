//! Observed climate data container.

use zeus_calendar::{NoLeapDate, water_year};

use crate::error::IoError;
use crate::validate;

/// Container for observed climate data at a single site.
///
/// Holds precipitation (required), optional temperature extremes, and
/// pre-computed calendar metadata derived from the date sequence.
#[derive(Debug, Clone)]
pub struct ObservedData {
    /// Precipitation time series (required).
    precip: Vec<f64>,
    /// Optional maximum temperature time series.
    temp_max: Option<Vec<f64>>,
    /// Optional minimum temperature time series.
    temp_min: Option<Vec<f64>>,
    /// Date for each time step.
    dates: Vec<NoLeapDate>,
    /// Month of each time step (1..=12).
    months: Vec<u8>,
    /// Water year of each time step.
    water_years: Vec<i32>,
    /// Day-of-year of each time step (1..=365).
    days_of_year: Vec<u16>,
}

impl ObservedData {
    /// Creates a new `ObservedData` after validating inputs.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Validation`] if any of the following checks fail:
    /// - Array lengths do not match
    /// - Precipitation contains negative values
    /// - `temp_min > temp_max` at any index
    ///
    /// Returns [`IoError::Calendar`] if water-year computation fails.
    pub fn new(
        precip: Vec<f64>,
        temp_max: Option<Vec<f64>>,
        temp_min: Option<Vec<f64>>,
        dates: Vec<NoLeapDate>,
        start_month: u8,
    ) -> Result<Self, IoError> {
        // Validate lengths.
        validate::validate_lengths(
            precip.len(),
            temp_max.as_ref().map(|v| v.len()),
            temp_min.as_ref().map(|v| v.len()),
            dates.len(),
        )
        .finish()?;

        // Validate precipitation is non-negative.
        validate::validate_precip_non_negative(&precip).finish()?;

        // Validate temperature ordering when both are present.
        if let (Some(tmax), Some(tmin)) = (&temp_max, &temp_min) {
            validate::validate_temp_ordering(tmax, tmin).finish()?;
        }

        // Compute calendar metadata.
        let months: Vec<u8> = dates.iter().map(|d| d.month()).collect();

        let water_years: Vec<i32> = dates
            .iter()
            .map(|d| water_year(d.year(), d.month(), start_month).map_err(IoError::from))
            .collect::<Result<Vec<_>, _>>()?;

        let days_of_year: Vec<u16> = dates.iter().map(|d| d.doy().get()).collect();

        Ok(Self {
            precip,
            temp_max,
            temp_min,
            dates,
            months,
            water_years,
            days_of_year,
        })
    }

    /// Returns the precipitation time series.
    pub fn precip(&self) -> &[f64] {
        &self.precip
    }

    /// Returns the maximum temperature time series, if present.
    pub fn temp_max(&self) -> Option<&[f64]> {
        self.temp_max.as_deref()
    }

    /// Returns the minimum temperature time series, if present.
    pub fn temp_min(&self) -> Option<&[f64]> {
        self.temp_min.as_deref()
    }

    /// Returns the date sequence.
    pub fn dates(&self) -> &[NoLeapDate] {
        &self.dates
    }

    /// Returns the month of each time step.
    pub fn months(&self) -> &[u8] {
        &self.months
    }

    /// Returns the water year of each time step.
    pub fn water_years(&self) -> &[i32] {
        &self.water_years
    }

    /// Returns the day-of-year of each time step.
    pub fn days_of_year(&self) -> &[u16] {
        &self.days_of_year
    }

    /// Returns the number of time steps.
    pub fn len(&self) -> usize {
        self.precip.len()
    }

    /// Returns `true` if the data contains no time steps.
    pub fn is_empty(&self) -> bool {
        self.precip.is_empty()
    }

    /// Consumes self and returns the precipitation vector.
    pub fn into_precip(self) -> Vec<f64> {
        self.precip
    }

    /// Consumes self and returns the optional maximum temperature vector.
    pub fn into_temp_max(self) -> Option<Vec<f64>> {
        self.temp_max
    }

    /// Consumes self and returns the optional minimum temperature vector.
    pub fn into_temp_min(self) -> Option<Vec<f64>> {
        self.temp_min
    }

    /// Consumes self and returns the date vector.
    pub fn into_dates(self) -> Vec<NoLeapDate> {
        self.dates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a short date sequence starting at the given (year, month, day).
    fn make_dates(year: i32, month: u8, day: u8, n: usize) -> Vec<NoLeapDate> {
        let mut dates = Vec::with_capacity(n);
        let mut d = NoLeapDate::new(year, month, day).unwrap();
        for _ in 0..n {
            dates.push(d);
            d = d.next();
        }
        dates
    }

    #[test]
    fn new_with_all_fields() {
        let dates = make_dates(2000, 1, 1, 3);
        let precip = vec![0.0, 1.5, 3.0];
        let tmax = vec![30.0, 25.0, 20.0];
        let tmin = vec![15.0, 10.0, 5.0];

        let obs = ObservedData::new(
            precip,
            Some(tmax),
            Some(tmin),
            dates,
            10, // October water year start
        )
        .unwrap();

        assert_eq!(obs.len(), 3);
        assert!(!obs.is_empty());
    }

    #[test]
    fn new_without_temperatures() {
        let dates = make_dates(2000, 6, 15, 2);
        let precip = vec![0.0, 2.0];

        let obs = ObservedData::new(precip, None, None, dates, 1).unwrap();

        assert_eq!(obs.len(), 2);
        assert!(obs.temp_max().is_none());
        assert!(obs.temp_min().is_none());
    }

    #[test]
    fn new_length_mismatch_returns_error() {
        let dates = make_dates(2000, 1, 1, 3);
        let precip = vec![0.0, 1.0]; // length 2 vs 3 dates

        let result = ObservedData::new(precip, None, None, dates, 1);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, IoError::Validation { .. }));
    }

    #[test]
    fn new_negative_precip_returns_error() {
        let dates = make_dates(2000, 1, 1, 3);
        let precip = vec![1.0, -0.5, 2.0];

        let result = ObservedData::new(precip, None, None, dates, 1);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, IoError::Validation { .. }));
    }

    #[test]
    fn new_temp_min_gt_temp_max_returns_error() {
        let dates = make_dates(2000, 1, 1, 3);
        let precip = vec![0.0, 1.0, 2.0];
        let tmax = vec![20.0, 15.0, 10.0];
        let tmin = vec![10.0, 25.0, 5.0]; // index 1: min > max

        let result = ObservedData::new(precip, Some(tmax), Some(tmin), dates, 1);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, IoError::Validation { .. }));
    }

    #[test]
    fn accessors_return_correct_values() {
        let dates = make_dates(2000, 3, 14, 3);
        let precip = vec![0.5, 1.0, 1.5];
        let tmax = vec![25.0, 26.0, 27.0];
        let tmin = vec![10.0, 11.0, 12.0];

        let obs = ObservedData::new(
            precip.clone(),
            Some(tmax.clone()),
            Some(tmin.clone()),
            dates.clone(),
            1,
        )
        .unwrap();

        assert_eq!(obs.precip(), &[0.5, 1.0, 1.5]);
        assert_eq!(obs.temp_max(), Some([25.0, 26.0, 27.0].as_slice()));
        assert_eq!(obs.temp_min(), Some([10.0, 11.0, 12.0].as_slice()));
        assert_eq!(obs.dates(), &dates[..]);
        assert_eq!(obs.len(), 3);
        assert!(!obs.is_empty());
    }

    #[test]
    fn calendar_metadata_computed_correctly() {
        // Build dates: Oct 1 2000, Oct 2 2000, Oct 3 2000
        let dates = make_dates(2000, 10, 1, 3);
        let precip = vec![0.0, 0.0, 0.0];

        let obs = ObservedData::new(precip, None, None, dates.clone(), 10).unwrap();

        // Months should all be October (10).
        assert_eq!(obs.months(), &[10, 10, 10]);

        // Water years: Oct 2000 with start_month=10 => WY 2001.
        assert_eq!(obs.water_years(), &[2001, 2001, 2001]);

        // Days of year: Oct 1 = DOY 274, Oct 2 = 275, Oct 3 = 276.
        assert_eq!(obs.days_of_year(), &[274, 275, 276]);
    }

    #[test]
    fn calendar_metadata_across_months() {
        // Build dates spanning a month boundary: Jan 31, Feb 1
        let dates = make_dates(2000, 1, 31, 2);
        let precip = vec![0.0, 0.0];

        let obs = ObservedData::new(precip, None, None, dates, 1).unwrap();

        assert_eq!(obs.months(), &[1, 2]);
        assert_eq!(obs.water_years(), &[2000, 2000]);
        assert_eq!(obs.days_of_year(), &[31, 32]);
    }

    #[test]
    fn into_consumers() {
        let dates = make_dates(2000, 1, 1, 2);
        let precip = vec![1.0, 2.0];
        let tmax = vec![30.0, 25.0];
        let tmin = vec![10.0, 5.0];

        let obs = ObservedData::new(
            precip.clone(),
            Some(tmax.clone()),
            Some(tmin.clone()),
            dates.clone(),
            1,
        )
        .unwrap();

        // Test into_precip.
        let obs2 = obs.clone();
        assert_eq!(obs2.into_precip(), precip);

        // Test into_temp_max.
        let obs3 = obs.clone();
        assert_eq!(obs3.into_temp_max(), Some(tmax));

        // Test into_temp_min.
        let obs4 = obs.clone();
        assert_eq!(obs4.into_temp_min(), Some(tmin));

        // Test into_dates.
        assert_eq!(obs.into_dates(), dates);
    }

    #[test]
    fn empty_data() {
        let obs = ObservedData::new(vec![], None, None, vec![], 1).unwrap();

        assert_eq!(obs.len(), 0);
        assert!(obs.is_empty());
        assert!(obs.precip().is_empty());
        assert!(obs.months().is_empty());
        assert!(obs.water_years().is_empty());
        assert!(obs.days_of_year().is_empty());
    }
}
