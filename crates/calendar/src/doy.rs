//! Day-of-year newtype and conversion tables for the 365-day no-leap calendar.

use crate::error::CalendarError;

/// Day-of-year in the 365-day no-leap calendar (1..=365).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Doy(u16);

/// Mapping from 0-based day-of-year index to `(month, day)` pairs.
///
/// Index 0 corresponds to January 1 and index 364 corresponds to December 31.
/// The calendar has no leap day (February always has 28 days).
#[rustfmt::skip]
pub(crate) const MONTH_DAY_TABLE: [(u8, u8); 365] = [
    // January (31 days)
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
    (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20),
    (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30),
    (1, 31),
    // February (28 days)
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
    (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20),
    (2, 21), (2, 22), (2, 23), (2, 24), (2, 25), (2, 26), (2, 27), (2, 28),
    // March (31 days)
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),
    (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (3, 19), (3, 20),
    (3, 21), (3, 22), (3, 23), (3, 24), (3, 25), (3, 26), (3, 27), (3, 28), (3, 29), (3, 30),
    (3, 31),
    // April (30 days)
    (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10),
    (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19), (4, 20),
    (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (4, 26), (4, 27), (4, 28), (4, 29), (4, 30),
    // May (31 days)
    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
    (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (5, 19), (5, 20),
    (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (5, 27), (5, 28), (5, 29), (5, 30),
    (5, 31),
    // June (30 days)
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
    (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20),
    (6, 21), (6, 22), (6, 23), (6, 24), (6, 25), (6, 26), (6, 27), (6, 28), (6, 29), (6, 30),
    // July (31 days)
    (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10),
    (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (7, 20),
    (7, 21), (7, 22), (7, 23), (7, 24), (7, 25), (7, 26), (7, 27), (7, 28), (7, 29), (7, 30),
    (7, 31),
    // August (31 days)
    (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10),
    (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20),
    (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28), (8, 29), (8, 30),
    (8, 31),
    // September (30 days)
    (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10),
    (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20),
    (9, 21), (9, 22), (9, 23), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (9, 30),
    // October (31 days)
    (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
    (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20),
    (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26), (10, 27), (10, 28), (10, 29), (10, 30),
    (10, 31),
    // November (30 days)
    (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10),
    (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20),
    (11, 21), (11, 22), (11, 23), (11, 24), (11, 25), (11, 26), (11, 27), (11, 28), (11, 29), (11, 30),
    // December (31 days)
    (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10),
    (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20),
    (12, 21), (12, 22), (12, 23), (12, 24), (12, 25), (12, 26), (12, 27), (12, 28), (12, 29), (12, 30),
    (12, 31),
];

/// Number of days in each month (index 0 unused, index 1 = January, ..., index 12 = December).
pub(crate) const DAYS_PER_MONTH: [u8; 13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Day-of-year on which each month starts (index 0 unused, index 1 = January starts at DOY 1, ...).
pub(crate) const MONTH_START_DOY: [u16; 13] =
    [0, 1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];

impl Doy {
    /// Creates a new `Doy` from a day-of-year value.
    ///
    /// # Errors
    ///
    /// Returns [`CalendarError::InvalidDoy`] if `doy` is not in 1..=365.
    pub fn new(doy: u16) -> Result<Self, CalendarError> {
        if !(1..=365).contains(&doy) {
            return Err(CalendarError::InvalidDoy { doy });
        }
        Ok(Self(doy))
    }

    /// Creates a new `Doy` from a (month, day) pair.
    ///
    /// # Errors
    ///
    /// Returns [`CalendarError::InvalidMonth`] if `month` is not in 1..=12.
    /// Returns [`CalendarError::InvalidDay`] if `day` is not valid for the given month.
    pub fn from_month_day(month: u8, day: u8) -> Result<Self, CalendarError> {
        if !(1..=12).contains(&month) {
            return Err(CalendarError::InvalidMonth { month });
        }
        let max_day = DAYS_PER_MONTH[month as usize];
        if !(1..=max_day).contains(&day) {
            return Err(CalendarError::InvalidDay {
                day,
                month,
                max_day,
            });
        }
        let doy = MONTH_START_DOY[month as usize] + day as u16 - 1;
        Ok(Self(doy))
    }

    /// Returns the inner day-of-year value (1..=365).
    pub fn get(self) -> u16 {
        self.0
    }

    /// Returns the 0-based index suitable for array indexing (0..=364).
    pub fn index(self) -> usize {
        (self.0 - 1) as usize
    }

    /// Returns the `(month, day)` pair for this day-of-year.
    pub fn month_day(self) -> (u8, u8) {
        MONTH_DAY_TABLE[self.index()]
    }

    /// Returns the month (1..=12) for this day-of-year.
    pub fn month(self) -> u8 {
        self.month_day().0
    }

    /// Returns the day within the month (1..=31) for this day-of-year.
    pub fn day(self) -> u8 {
        self.month_day().1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        assert!(Doy::new(1).is_ok());
        assert_eq!(Doy::new(1).unwrap().get(), 1);
        assert!(Doy::new(365).is_ok());
        assert_eq!(Doy::new(365).unwrap().get(), 365);
    }

    #[test]
    fn new_invalid_zero() {
        assert_eq!(
            Doy::new(0).unwrap_err(),
            CalendarError::InvalidDoy { doy: 0 }
        );
    }

    #[test]
    fn new_invalid_366() {
        assert_eq!(
            Doy::new(366).unwrap_err(),
            CalendarError::InvalidDoy { doy: 366 }
        );
    }

    #[test]
    fn from_month_day_valid() {
        // Jan 1 = doy 1
        assert_eq!(Doy::from_month_day(1, 1).unwrap().get(), 1);
        // Dec 31 = doy 365
        assert_eq!(Doy::from_month_day(12, 31).unwrap().get(), 365);
        // Feb 28 = doy 59
        assert_eq!(Doy::from_month_day(2, 28).unwrap().get(), 59);
    }

    #[test]
    fn from_month_day_invalid_month_zero() {
        assert_eq!(
            Doy::from_month_day(0, 1).unwrap_err(),
            CalendarError::InvalidMonth { month: 0 }
        );
    }

    #[test]
    fn from_month_day_invalid_month_13() {
        assert_eq!(
            Doy::from_month_day(13, 1).unwrap_err(),
            CalendarError::InvalidMonth { month: 13 }
        );
    }

    #[test]
    fn from_month_day_invalid_day_zero() {
        assert_eq!(
            Doy::from_month_day(1, 0).unwrap_err(),
            CalendarError::InvalidDay {
                day: 0,
                month: 1,
                max_day: 31,
            }
        );
    }

    #[test]
    fn from_month_day_feb_29() {
        assert_eq!(
            Doy::from_month_day(2, 29).unwrap_err(),
            CalendarError::InvalidDay {
                day: 29,
                month: 2,
                max_day: 28,
            }
        );
    }

    #[test]
    fn from_month_day_invalid_day_32() {
        assert_eq!(
            Doy::from_month_day(1, 32).unwrap_err(),
            CalendarError::InvalidDay {
                day: 32,
                month: 1,
                max_day: 31,
            }
        );
    }

    #[test]
    fn roundtrip_all_365() {
        for d in 1..=365u16 {
            let doy = Doy::new(d).unwrap();
            let (m, day) = doy.month_day();
            let roundtripped = Doy::from_month_day(m, day).unwrap();
            assert_eq!(
                doy, roundtripped,
                "roundtrip failed for doy {d}: month_day=({m}, {day})"
            );
        }
    }

    #[test]
    fn accessors() {
        let doy = Doy::new(59).unwrap(); // Feb 28
        assert_eq!(doy.get(), 59);
        assert_eq!(doy.index(), 58);
        assert_eq!(doy.month(), 2);
        assert_eq!(doy.day(), 28);
    }

    #[test]
    fn copy_trait() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<Doy>();
    }

    #[test]
    fn ord_trait() {
        let first = Doy::new(1).unwrap();
        let last = Doy::new(365).unwrap();
        assert!(first < last);
    }

    #[test]
    fn table_integrity_days_per_month() {
        let total: u16 = DAYS_PER_MONTH[1..=12].iter().copied().map(u16::from).sum();
        assert_eq!(total, 365);
    }

    #[test]
    fn table_integrity_month_start() {
        for m in 1..12usize {
            assert_eq!(
                MONTH_START_DOY[m] + DAYS_PER_MONTH[m] as u16,
                MONTH_START_DOY[m + 1],
                "MONTH_START_DOY mismatch at month {m}"
            );
        }
    }
}
