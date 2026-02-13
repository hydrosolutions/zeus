//! Error types for the zeus-calendar crate.

/// Error type for all fallible operations in the zeus-calendar crate.
///
/// This enum covers validation failures for day-of-year values,
/// month numbers, and day-within-month values in the 365-day no-leap
/// calendar.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[allow(clippy::enum_variant_names)]
pub enum CalendarError {
    /// Returned when a day-of-year value is outside the valid range 1..=365.
    #[error("invalid day of year: {doy} (must be 1..=365)")]
    InvalidDoy {
        /// The invalid day-of-year value that was provided.
        doy: u16,
    },

    /// Returned when a month number is outside the valid range 1..=12.
    #[error("invalid month: {month} (must be 1..=12)")]
    InvalidMonth {
        /// The invalid month number that was provided.
        month: u8,
    },

    /// Returned when a day number exceeds the number of days in the given month.
    #[error("invalid day: {day} for month {month} (max {max_day})")]
    InvalidDay {
        /// The invalid day number that was provided.
        day: u8,
        /// The month for which the day is invalid.
        month: u8,
        /// The maximum valid day for the given month.
        max_day: u8,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_invalid_doy() {
        let err = CalendarError::InvalidDoy { doy: 0 };
        assert_eq!(err.to_string(), "invalid day of year: 0 (must be 1..=365)");
    }

    #[test]
    fn error_invalid_month() {
        let err = CalendarError::InvalidMonth { month: 13 };
        assert_eq!(err.to_string(), "invalid month: 13 (must be 1..=12)");
    }

    #[test]
    fn error_invalid_day() {
        let err = CalendarError::InvalidDay {
            day: 29,
            month: 2,
            max_day: 28,
        };
        assert_eq!(err.to_string(), "invalid day: 29 for month 2 (max 28)");
    }

    #[test]
    fn error_is_std_error() {
        fn assert_impl<T: std::error::Error>() {}
        assert_impl::<CalendarError>();
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<CalendarError>();
    }

    #[test]
    fn error_is_clone() {
        let err = CalendarError::InvalidDoy { doy: 400 };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn error_is_partial_eq() {
        let a = CalendarError::InvalidMonth { month: 0 };
        let b = CalendarError::InvalidMonth { month: 0 };
        assert_eq!(a, b);

        let c = CalendarError::InvalidMonth { month: 13 };
        assert_ne!(a, c);
    }
}
