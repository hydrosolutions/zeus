//! Error types for the zeus-resample crate.

/// Error type for all fallible operations in the zeus-resample crate.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ResampleError {
    /// Returned when input data is empty.
    #[error("input data is empty")]
    EmptyData,

    /// Returned when array lengths don't match.
    #[error("{field}: expected {expected} elements, got {got}")]
    LengthMismatch {
        /// Name of the mismatched field.
        field: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },

    /// Returned when there are too few observed years.
    #[error("insufficient observed data: got {n} years, need at least {min}")]
    InsufficientData {
        /// Number of observed years.
        n: usize,
        /// Minimum required.
        min: usize,
    },

    /// Returned when input contains NaN or infinity.
    #[error("non-finite value in {field}")]
    NonFiniteInput {
        /// Name of the field containing the non-finite value.
        field: &'static str,
    },

    /// Returned when configuration is invalid.
    #[error("invalid configuration: {reason}")]
    InvalidConfig {
        /// Description of the problem.
        reason: String,
    },

    /// Returned when no candidates found after all fallbacks.
    #[error("no candidates for day {day} in month {month}")]
    NoCandidates {
        /// Day index in the year (0-based).
        day: usize,
        /// Calendar month (1-indexed).
        month: u8,
    },

    /// Returned when a month value is invalid.
    #[error("invalid month: {month} (must be 1..=12)")]
    InvalidMonth {
        /// The invalid month value.
        month: u8,
    },

    /// Markov error.
    #[error(transparent)]
    Markov(#[from] zeus_markov::MarkovError),

    /// KNN error.
    #[error(transparent)]
    Knn(#[from] zeus_knn::KnnError),

    /// Calendar error.
    #[error(transparent)]
    Calendar(#[from] zeus_calendar::CalendarError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_empty_data() {
        let e = ResampleError::EmptyData;
        assert_eq!(e.to_string(), "input data is empty");
    }

    #[test]
    fn display_length_mismatch() {
        let e = ResampleError::LengthMismatch {
            field: "temp",
            expected: 100,
            got: 99,
        };
        assert_eq!(e.to_string(), "temp: expected 100 elements, got 99");
    }

    #[test]
    fn display_insufficient_data() {
        let e = ResampleError::InsufficientData { n: 1, min: 2 };
        assert_eq!(
            e.to_string(),
            "insufficient observed data: got 1 years, need at least 2"
        );
    }

    #[test]
    fn display_non_finite() {
        let e = ResampleError::NonFiniteInput { field: "precip" };
        assert_eq!(e.to_string(), "non-finite value in precip");
    }

    #[test]
    fn display_invalid_config() {
        let e = ResampleError::InvalidConfig {
            reason: "bad".to_string(),
        };
        assert_eq!(e.to_string(), "invalid configuration: bad");
    }

    #[test]
    fn display_no_candidates() {
        let e = ResampleError::NoCandidates { day: 5, month: 3 };
        assert_eq!(e.to_string(), "no candidates for day 5 in month 3");
    }

    #[test]
    fn display_invalid_month() {
        let e = ResampleError::InvalidMonth { month: 13 };
        assert_eq!(e.to_string(), "invalid month: 13 (must be 1..=12)");
    }

    #[test]
    fn from_markov_error() {
        let me = zeus_markov::MarkovError::EmptyData;
        let re: ResampleError = me.into();
        assert!(matches!(re, ResampleError::Markov(_)));
    }

    #[test]
    fn from_knn_error() {
        let ke = zeus_knn::KnnError::EmptyCandidates;
        let re: ResampleError = ke.into();
        assert!(matches!(re, ResampleError::Knn(_)));
    }

    #[test]
    fn from_calendar_error() {
        let ce = zeus_calendar::CalendarError::InvalidMonth { month: 0 };
        let re: ResampleError = ce.into();
        assert!(matches!(re, ResampleError::Calendar(_)));
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<ResampleError>();
    }
}
