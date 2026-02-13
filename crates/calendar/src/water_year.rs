//! Water year computation.

use crate::error::CalendarError;

/// Computes the water year for a given calendar year and month.
///
/// A water year is defined by its `start_month`. All months from `start_month`
/// onward belong to the *next* calendar year's water year.
///
/// # Errors
///
/// Returns [`CalendarError::InvalidMonth`] if `month` or `start_month` is
/// outside 1..=12.
///
/// # Examples
///
/// ```ignore
/// // October-start water year (standard US hydrological convention):
/// assert_eq!(water_year(2000, 10, 10).unwrap(), 2001); // Oct 2000 -> WY 2001
/// assert_eq!(water_year(2001, 9, 10).unwrap(), 2001);  // Sep 2001 -> WY 2001
///
/// // Calendar year (start_month = 1):
/// assert_eq!(water_year(2000, 6, 1).unwrap(), 2000);
/// ```
pub fn water_year(year: i32, month: u8, start_month: u8) -> Result<i32, CalendarError> {
    if !(1..=12).contains(&month) {
        return Err(CalendarError::InvalidMonth { month });
    }
    if !(1..=12).contains(&start_month) {
        return Err(CalendarError::InvalidMonth { month: start_month });
    }
    if start_month == 1 {
        return Ok(year);
    }
    if month >= start_month {
        Ok(year + 1)
    } else {
        Ok(year)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CalendarError;

    #[test]
    fn start_month_1_returns_year() {
        assert_eq!(water_year(2000, 6, 1).unwrap(), 2000);
    }

    #[test]
    fn start_month_1_all_months() {
        for m in 1..=12 {
            assert_eq!(water_year(2000, m, 1).unwrap(), 2000);
        }
    }

    #[test]
    fn october_start_standard() {
        assert_eq!(water_year(2000, 10, 10).unwrap(), 2001);
    }

    #[test]
    fn month_before_start() {
        assert_eq!(water_year(2001, 9, 10).unwrap(), 2001);
    }

    #[test]
    fn month_at_start() {
        assert_eq!(water_year(2000, 10, 10).unwrap(), 2001);
    }

    #[test]
    fn month_after_start() {
        assert_eq!(water_year(2000, 11, 10).unwrap(), 2001);
    }

    #[test]
    fn january_with_october_start() {
        assert_eq!(water_year(2001, 1, 10).unwrap(), 2001);
    }

    #[test]
    fn invalid_month_zero() {
        assert_eq!(
            water_year(2000, 0, 10).unwrap_err(),
            CalendarError::InvalidMonth { month: 0 }
        );
    }

    #[test]
    fn invalid_month_13() {
        assert_eq!(
            water_year(2000, 13, 10).unwrap_err(),
            CalendarError::InvalidMonth { month: 13 }
        );
    }

    #[test]
    fn invalid_start_month_zero() {
        assert_eq!(
            water_year(2000, 6, 0).unwrap_err(),
            CalendarError::InvalidMonth { month: 0 }
        );
    }

    #[test]
    fn invalid_start_month_13() {
        assert_eq!(
            water_year(2000, 6, 13).unwrap_err(),
            CalendarError::InvalidMonth { month: 13 }
        );
    }

    #[test]
    fn all_start_months() {
        let year = 2000;
        let month = 6_u8;
        for sm in 1..=12 {
            let wy = water_year(year, month, sm).unwrap();
            if sm == 1 {
                assert_eq!(wy, year);
            } else if month >= sm {
                assert_eq!(wy, year + 1);
            } else {
                assert_eq!(wy, year);
            }
        }
    }

    #[test]
    fn negative_year() {
        assert_eq!(water_year(-1, 10, 10).unwrap(), 0);
    }
}
