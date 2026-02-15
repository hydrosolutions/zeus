//! Borrowed view over synthetic weather output.

use crate::error::IoError;

/// Borrowed view over a single realisation of synthetic weather output.
///
/// All slices must have the same length. This struct does not own its data;
/// it borrows arrays typically held by the generator.
#[derive(Debug, Clone, Copy)]
pub struct SyntheticWeather<'a> {
    /// Site identifier.
    site: &'a str,
    /// Synthetic precipitation values.
    precip: &'a [f64],
    /// Optional synthetic maximum temperature values.
    temp_max: Option<&'a [f64]>,
    /// Optional synthetic minimum temperature values.
    temp_min: Option<&'a [f64]>,
    /// Month for each time step.
    months: &'a [u8],
    /// Water year for each time step.
    water_years: &'a [i32],
    /// Day-of-year for each time step.
    days_of_year: &'a [u16],
    /// Realisation index.
    realisation: u32,
}

impl<'a> SyntheticWeather<'a> {
    /// Creates a new `SyntheticWeather` view after validating that all present
    /// slices share the same length as `precip`.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Validation`] if any slice length differs from
    /// `precip.len()`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        site: &'a str,
        precip: &'a [f64],
        temp_max: Option<&'a [f64]>,
        temp_min: Option<&'a [f64]>,
        months: &'a [u8],
        water_years: &'a [i32],
        days_of_year: &'a [u16],
        realisation: u32,
    ) -> Result<Self, IoError> {
        let expected = precip.len();
        let mut mismatches: Vec<String> = Vec::new();

        if let Some(tmax) = temp_max
            && tmax.len() != expected
        {
            mismatches.push(format!(
                "temp_max length {} != precip length {}",
                tmax.len(),
                expected,
            ));
        }

        if let Some(tmin) = temp_min
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
            site,
            precip,
            temp_max,
            temp_min,
            months,
            water_years,
            days_of_year,
            realisation,
        })
    }

    /// Returns the site identifier.
    pub fn site(&self) -> &'a str {
        self.site
    }

    /// Returns the synthetic precipitation slice.
    pub fn precip(&self) -> &'a [f64] {
        self.precip
    }

    /// Returns the optional synthetic maximum temperature slice.
    pub fn temp_max(&self) -> Option<&'a [f64]> {
        self.temp_max
    }

    /// Returns the optional synthetic minimum temperature slice.
    pub fn temp_min(&self) -> Option<&'a [f64]> {
        self.temp_min
    }

    /// Returns the month slice for each time step.
    pub fn months(&self) -> &'a [u8] {
        self.months
    }

    /// Returns the water year slice for each time step.
    pub fn water_years(&self) -> &'a [i32] {
        self.water_years
    }

    /// Returns the day-of-year slice for each time step.
    pub fn days_of_year(&self) -> &'a [u16] {
        self.days_of_year
    }

    /// Returns the realisation index.
    pub fn realisation(&self) -> u32 {
        self.realisation
    }

    /// Returns the number of time steps (length of all slices).
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
        let precip = [1.0, 2.0, 3.0];
        let tmax = [30.0, 31.0, 32.0];
        let tmin = [10.0, 11.0, 12.0];
        let months = [1u8, 1, 2];
        let water_years = [2020i32, 2020, 2020];
        let days_of_year = [1u16, 2, 3];

        let sw = SyntheticWeather::new(
            "test_site",
            &precip,
            Some(&tmax),
            Some(&tmin),
            &months,
            &water_years,
            &days_of_year,
            0,
        );

        assert!(sw.is_ok());
        let sw = sw.unwrap();
        assert_eq!(sw.len(), 3);
        assert!(!sw.is_empty());
    }

    #[test]
    fn construction_without_temperatures() {
        let precip = [0.5, 1.5];
        let months = [6u8, 7];
        let water_years = [2021i32, 2021];
        let days_of_year = [180u16, 181];

        let sw = SyntheticWeather::new(
            "test_site",
            &precip,
            None,
            None,
            &months,
            &water_years,
            &days_of_year,
            5,
        );

        assert!(sw.is_ok());
        let sw = sw.unwrap();
        assert_eq!(sw.len(), 2);
        assert!(sw.temp_max().is_none());
        assert!(sw.temp_min().is_none());
        assert_eq!(sw.realisation(), 5);
    }

    #[test]
    fn length_mismatch_returns_error() {
        let precip = [1.0, 2.0, 3.0];
        let tmax = [30.0, 31.0]; // wrong length
        let months = [1u8, 1, 2];
        let water_years = [2020i32, 2020, 2020];
        let days_of_year = [1u16, 2, 3];

        let result = SyntheticWeather::new(
            "test_site",
            &precip,
            Some(&tmax),
            None,
            &months,
            &water_years,
            &days_of_year,
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
        let precip = [5.0, 10.0];
        let tmax = [25.0, 26.0];
        let tmin = [8.0, 9.0];
        let months = [3u8, 4];
        let water_years = [2022i32, 2022];
        let days_of_year = [60u16, 91];

        let sw = SyntheticWeather::new(
            "test_site",
            &precip,
            Some(&tmax),
            Some(&tmin),
            &months,
            &water_years,
            &days_of_year,
            42,
        )
        .unwrap();

        assert_eq!(sw.precip(), &[5.0, 10.0]);
        assert_eq!(sw.temp_max(), Some([25.0, 26.0].as_slice()));
        assert_eq!(sw.temp_min(), Some([8.0, 9.0].as_slice()));
        assert_eq!(sw.months(), &[3u8, 4]);
        assert_eq!(sw.water_years(), &[2022i32, 2022]);
        assert_eq!(sw.days_of_year(), &[60u16, 91]);
        assert_eq!(sw.realisation(), 42);
        assert_eq!(sw.len(), 2);
        assert!(!sw.is_empty());
    }

    #[test]
    fn struct_is_copy() {
        let precip = [1.0];
        let months = [1u8];
        let water_years = [2020i32];
        let days_of_year = [1u16];

        let sw = SyntheticWeather::new(
            "test_site",
            &precip,
            None,
            None,
            &months,
            &water_years,
            &days_of_year,
            0,
        )
        .unwrap();

        // Assign to another variable; this only compiles if the type is Copy.
        let sw2 = sw;
        assert_eq!(sw.len(), sw2.len());
        assert_eq!(sw.realisation(), sw2.realisation());
    }
}
