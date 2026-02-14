//! Owned storage for synthetic weather data.

use crate::error::IoError;
use crate::synthetic::SyntheticWeather;

/// Owned storage for a single realisation of synthetic weather output.
///
/// This is the owned counterpart of [`SyntheticWeather`], used when data is
/// read from Parquet and must outlive the Arrow record batches.
#[derive(Debug, Clone)]
pub struct OwnedSyntheticWeather {
    /// Synthetic precipitation values.
    pub precip: Vec<f64>,
    /// Optional synthetic maximum temperature values.
    pub temp_max: Option<Vec<f64>>,
    /// Optional synthetic minimum temperature values.
    pub temp_min: Option<Vec<f64>>,
    /// Month for each time step.
    pub months: Vec<u8>,
    /// Water year for each time step.
    pub water_years: Vec<i32>,
    /// Day-of-year for each time step.
    pub days_of_year: Vec<u16>,
    /// Realisation index.
    pub realisation: u32,
}

impl OwnedSyntheticWeather {
    /// Creates a new `OwnedSyntheticWeather` after validating that all present
    /// vectors share the same length as `precip`.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Validation`] if any vector length differs from
    /// `precip.len()`.
    pub fn new(
        precip: Vec<f64>,
        temp_max: Option<Vec<f64>>,
        temp_min: Option<Vec<f64>>,
        months: Vec<u8>,
        water_years: Vec<i32>,
        days_of_year: Vec<u16>,
        realisation: u32,
    ) -> Result<Self, IoError> {
        let expected = precip.len();
        let mut mismatches: Vec<String> = Vec::new();

        if let Some(ref tmax) = temp_max
            && tmax.len() != expected
        {
            mismatches.push(format!(
                "temp_max length {} != precip length {}",
                tmax.len(),
                expected,
            ));
        }

        if let Some(ref tmin) = temp_min
            && tmin.len() != expected
        {
            mismatches.push(format!(
                "temp_min length {} != precip length {}",
                tmin.len(),
                expected,
            ));
        }

        if months.len() != expected {
            mismatches.push(format!(
                "months length {} != precip length {}",
                months.len(),
                expected,
            ));
        }

        if water_years.len() != expected {
            mismatches.push(format!(
                "water_years length {} != precip length {}",
                water_years.len(),
                expected,
            ));
        }

        if days_of_year.len() != expected {
            mismatches.push(format!(
                "days_of_year length {} != precip length {}",
                days_of_year.len(),
                expected,
            ));
        }

        if !mismatches.is_empty() {
            return Err(IoError::Validation {
                count: mismatches.len(),
                details: mismatches.join("; "),
            });
        }

        Ok(Self {
            precip,
            temp_max,
            temp_min,
            months,
            water_years,
            days_of_year,
            realisation,
        })
    }

    /// Returns a borrowed [`SyntheticWeather`] view over this owned data.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Validation`] if the underlying
    /// [`SyntheticWeather::new`] validation fails.
    pub fn as_view(&self) -> Result<SyntheticWeather<'_>, IoError> {
        SyntheticWeather::new(
            &self.precip,
            self.temp_max.as_deref(),
            self.temp_min.as_deref(),
            &self.months,
            &self.water_years,
            &self.days_of_year,
            self.realisation,
        )
    }

    /// Returns the synthetic precipitation slice.
    pub fn precip(&self) -> &[f64] {
        &self.precip
    }

    /// Returns the optional synthetic maximum temperature slice.
    pub fn temp_max(&self) -> Option<&[f64]> {
        self.temp_max.as_deref()
    }

    /// Returns the optional synthetic minimum temperature slice.
    pub fn temp_min(&self) -> Option<&[f64]> {
        self.temp_min.as_deref()
    }

    /// Returns the month slice for each time step.
    pub fn months(&self) -> &[u8] {
        &self.months
    }

    /// Returns the water year slice for each time step.
    pub fn water_years(&self) -> &[i32] {
        &self.water_years
    }

    /// Returns the day-of-year slice for each time step.
    pub fn days_of_year(&self) -> &[u16] {
        &self.days_of_year
    }

    /// Returns the realisation index.
    pub fn realisation(&self) -> u32 {
        self.realisation
    }

    /// Returns the number of time steps (length of all vectors).
    pub fn len(&self) -> usize {
        self.precip.len()
    }

    /// Returns `true` if the weather data has zero time steps.
    pub fn is_empty(&self) -> bool {
        self.precip.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_with_all_fields() {
        let osw = OwnedSyntheticWeather::new(
            vec![1.0, 2.0, 3.0],
            Some(vec![30.0, 31.0, 32.0]),
            Some(vec![10.0, 11.0, 12.0]),
            vec![1, 1, 2],
            vec![2020, 2020, 2020],
            vec![1, 2, 3],
            0,
        );

        assert!(osw.is_ok());
        let osw = osw.unwrap();
        assert_eq!(osw.len(), 3);
        assert!(!osw.is_empty());
    }

    #[test]
    fn construction_without_temperatures() {
        let osw = OwnedSyntheticWeather::new(
            vec![0.5, 1.5],
            None,
            None,
            vec![6, 7],
            vec![2021, 2021],
            vec![180, 181],
            5,
        );

        assert!(osw.is_ok());
        let osw = osw.unwrap();
        assert_eq!(osw.len(), 2);
        assert!(osw.temp_max().is_none());
        assert!(osw.temp_min().is_none());
        assert_eq!(osw.realisation(), 5);
    }

    #[test]
    fn length_mismatch_returns_error() {
        let result = OwnedSyntheticWeather::new(
            vec![1.0, 2.0, 3.0],
            Some(vec![30.0, 31.0]), // wrong length
            None,
            vec![1, 1, 2],
            vec![2020, 2020, 2020],
            vec![1, 2, 3],
            0,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            IoError::Validation { count, details } => {
                assert_eq!(count, 1);
                assert!(details.contains("temp_max"));
            }
            _ => panic!("expected Validation error"),
        }
    }

    #[test]
    fn accessor_values() {
        let osw = OwnedSyntheticWeather::new(
            vec![5.0, 10.0],
            Some(vec![25.0, 26.0]),
            Some(vec![8.0, 9.0]),
            vec![3, 4],
            vec![2022, 2022],
            vec![60, 91],
            42,
        )
        .unwrap();

        assert_eq!(osw.precip(), &[5.0, 10.0]);
        assert_eq!(osw.temp_max(), Some([25.0, 26.0].as_slice()));
        assert_eq!(osw.temp_min(), Some([8.0, 9.0].as_slice()));
        assert_eq!(osw.months(), &[3u8, 4]);
        assert_eq!(osw.water_years(), &[2022i32, 2022]);
        assert_eq!(osw.days_of_year(), &[60u16, 91]);
        assert_eq!(osw.realisation(), 42);
        assert_eq!(osw.len(), 2);
        assert!(!osw.is_empty());
    }

    #[test]
    fn as_view_delegates_to_synthetic_weather() {
        let osw = OwnedSyntheticWeather::new(
            vec![1.0, 2.0],
            Some(vec![25.0, 26.0]),
            Some(vec![8.0, 9.0]),
            vec![3, 4],
            vec![2022, 2022],
            vec![60, 91],
            7,
        )
        .unwrap();

        let view = osw.as_view().unwrap();
        assert_eq!(view.precip(), osw.precip());
        assert_eq!(view.temp_max(), osw.temp_max());
        assert_eq!(view.temp_min(), osw.temp_min());
        assert_eq!(view.months(), osw.months());
        assert_eq!(view.water_years(), osw.water_years());
        assert_eq!(view.days_of_year(), osw.days_of_year());
        assert_eq!(view.realisation(), osw.realisation());
        assert_eq!(view.len(), osw.len());
    }

    #[test]
    fn empty_owned_synthetic_weather() {
        let osw =
            OwnedSyntheticWeather::new(vec![], None, None, vec![], vec![], vec![], 0).unwrap();

        assert!(osw.is_empty());
        assert_eq!(osw.len(), 0);
    }

    #[test]
    fn multiple_mismatches_reported() {
        let result = OwnedSyntheticWeather::new(
            vec![1.0, 2.0, 3.0],
            Some(vec![30.0, 31.0]), // wrong length
            Some(vec![10.0]),       // wrong length
            vec![1, 1],             // wrong length
            vec![2020, 2020, 2020],
            vec![1, 2, 3],
            0,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            IoError::Validation { count, details } => {
                assert_eq!(count, 3);
                assert!(details.contains("temp_max"));
                assert!(details.contains("temp_min"));
                assert!(details.contains("months"));
            }
            _ => panic!("expected Validation error"),
        }
    }
}
