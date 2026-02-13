//! No-leap date with year context.

use crate::doy::Doy;
use crate::error::CalendarError;

/// A date in the 365-day no-leap calendar with year context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NoLeapDate {
    year: i32,
    month: u8,
    day: u8,
    doy: u16,
}

impl PartialOrd for NoLeapDate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NoLeapDate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.year, self.doy).cmp(&(other.year, other.doy))
    }
}

impl NoLeapDate {
    /// Creates a new `NoLeapDate` from year, month, and day.
    ///
    /// # Errors
    ///
    /// Returns [`CalendarError`] if the month or day is invalid for the
    /// 365-day no-leap calendar.
    pub fn new(year: i32, month: u8, day: u8) -> Result<Self, CalendarError> {
        let doy = Doy::from_month_day(month, day)?;
        Ok(Self {
            year,
            month,
            day,
            doy: doy.get(),
        })
    }

    /// Creates a `NoLeapDate` from a year and an already-validated [`Doy`].
    ///
    /// This constructor is infallible because `Doy` guarantees a valid
    /// day-of-year value.
    pub fn from_year_doy(year: i32, doy: Doy) -> Self {
        let (month, day) = doy.month_day();
        Self {
            year,
            month,
            day,
            doy: doy.get(),
        }
    }

    /// Returns the year.
    pub fn year(self) -> i32 {
        self.year
    }

    /// Returns the month (1..=12).
    pub fn month(self) -> u8 {
        self.month
    }

    /// Returns the day within the month (1..=31).
    pub fn day(self) -> u8 {
        self.day
    }

    /// Returns the day-of-year as a [`Doy`].
    ///
    /// This cannot fail because `NoLeapDate` always holds a valid doy.
    pub fn doy(self) -> Doy {
        // Safety: NoLeapDate always holds a valid doy in 1..=365,
        // guaranteed by the constructors.
        Doy::new(self.doy).expect("NoLeapDate always holds a valid doy")
    }

    /// Returns `(month, day)` as a tuple.
    pub fn month_day(self) -> (u8, u8) {
        (self.month, self.day)
    }

    /// Returns the next date in the no-leap calendar.
    ///
    /// If the current date is December 31, the result wraps to January 1
    /// of the following year.
    pub fn next(self) -> Self {
        if self.doy == 365 {
            Self::from_year_doy(self.year + 1, Doy::new(1).expect("doy 1 is always valid"))
        } else {
            Self::from_year_doy(self.year, Doy::new(self.doy + 1).expect("doy + 1 <= 365"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        let date = NoLeapDate::new(2000, 1, 1).unwrap();
        assert_eq!(date.year(), 2000);
        assert_eq!(date.month(), 1);
        assert_eq!(date.day(), 1);
        assert_eq!(date.doy().get(), 1);
    }

    #[test]
    fn new_invalid_month() {
        assert_eq!(
            NoLeapDate::new(2000, 0, 1).unwrap_err(),
            CalendarError::InvalidMonth { month: 0 }
        );
    }

    #[test]
    fn new_invalid_day() {
        assert_eq!(
            NoLeapDate::new(2000, 2, 29).unwrap_err(),
            CalendarError::InvalidDay {
                day: 29,
                month: 2,
                max_day: 28,
            }
        );
    }

    #[test]
    fn from_year_doy() {
        let doy = Doy::new(59).unwrap(); // Feb 28
        let date = NoLeapDate::from_year_doy(2000, doy);
        assert_eq!(date.month(), 2);
        assert_eq!(date.day(), 28);
    }

    #[test]
    fn accessors() {
        let date = NoLeapDate::new(2024, 3, 15).unwrap();
        assert_eq!(date.year(), 2024);
        assert_eq!(date.month(), 3);
        assert_eq!(date.day(), 15);
        assert_eq!(date.doy().get(), 74); // 31 (Jan) + 28 (Feb) + 15 = 74
        assert_eq!(date.month_day(), (3, 15));
    }

    #[test]
    fn next_within_month() {
        let date = NoLeapDate::new(2000, 1, 15).unwrap();
        let next = date.next();
        assert_eq!(next.month(), 1);
        assert_eq!(next.day(), 16);
        assert_eq!(next.year(), 2000);
    }

    #[test]
    fn next_month_boundary() {
        let date = NoLeapDate::new(2000, 1, 31).unwrap();
        let next = date.next();
        assert_eq!(next.month(), 2);
        assert_eq!(next.day(), 1);
        assert_eq!(next.year(), 2000);
    }

    #[test]
    fn next_feb_28_to_mar_1() {
        let date = NoLeapDate::new(2000, 2, 28).unwrap();
        assert_eq!(date.doy().get(), 59);
        let next = date.next();
        assert_eq!(next.month(), 3);
        assert_eq!(next.day(), 1);
        assert_eq!(next.doy().get(), 60);
    }

    #[test]
    fn next_dec_31_year_wrap() {
        let date = NoLeapDate::new(2000, 12, 31).unwrap();
        let next = date.next();
        assert_eq!(next.year(), 2001);
        assert_eq!(next.month(), 1);
        assert_eq!(next.day(), 1);
        assert_eq!(next.doy().get(), 1);
    }

    #[test]
    fn next_negative_year_boundary() {
        let date = NoLeapDate::new(-1, 12, 31).unwrap();
        let next = date.next();
        assert_eq!(next.year(), 0);
        assert_eq!(next.month(), 1);
        assert_eq!(next.day(), 1);
    }

    #[test]
    fn copy_trait() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<NoLeapDate>();
    }

    #[test]
    fn ord_same_year() {
        let jan1 = NoLeapDate::new(2000, 1, 1).unwrap();
        let dec31 = NoLeapDate::new(2000, 12, 31).unwrap();
        assert!(jan1 < dec31);
    }

    #[test]
    fn ord_different_years() {
        let dec31 = NoLeapDate::new(1999, 12, 31).unwrap();
        let jan1 = NoLeapDate::new(2000, 1, 1).unwrap();
        assert!(dec31 < jan1);
    }

    #[test]
    fn eq_trait() {
        let a = NoLeapDate::new(2000, 6, 15).unwrap();
        let b = NoLeapDate::new(2000, 6, 15).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn hash_trait() {
        fn assert_hash<T: std::hash::Hash>() {}
        assert_hash::<NoLeapDate>();
    }
}
