//! Error types for the zeus-markov crate.

/// Error type for all fallible operations in the zeus-markov crate.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MarkovError {
    /// Returned when input data is empty.
    #[error("input data is empty")]
    EmptyData,

    /// Returned when there are too few observations.
    #[error("insufficient data: got {n} observations, need at least {min}")]
    InsufficientData {
        /// Number of observations provided.
        n: usize,
        /// Minimum required.
        min: usize,
    },

    /// Returned when input contains NaN or infinity.
    #[error("input data contains non-finite values")]
    NonFiniteData,

    /// Returned when a month value is outside 1..=12.
    #[error("invalid month: {month} (must be 1..=12)")]
    InvalidMonth {
        /// The invalid month value.
        month: u8,
    },

    /// Returned when a threshold specification is invalid.
    #[error("invalid threshold: {reason}")]
    InvalidThreshold {
        /// Description of the problem.
        reason: String,
    },

    /// Returned when a spell factor is non-finite or non-positive.
    #[error("invalid spell factor for month {month}: {value} (must be finite and > 0)")]
    InvalidSpellFactor {
        /// 1-indexed month.
        month: u8,
        /// The invalid value.
        value: f64,
    },

    /// Returned when precip and month slices differ in length.
    #[error("length mismatch: precip has {precip_len} elements, months has {months_len}")]
    LengthMismatch {
        /// Length of the precipitation slice.
        precip_len: usize,
        /// Length of the months slice.
        months_len: usize,
    },

    /// Returned when a pre-allocated buffer has the wrong length.
    #[error("buffer length mismatch: expected {expected}, got {got}")]
    BufferLengthMismatch {
        /// Expected buffer length.
        expected: usize,
        /// Actual buffer length.
        got: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_empty_data() {
        let e = MarkovError::EmptyData;
        assert_eq!(e.to_string(), "input data is empty");
    }

    #[test]
    fn error_insufficient_data() {
        let e = MarkovError::InsufficientData { n: 5, min: 10 };
        assert_eq!(
            e.to_string(),
            "insufficient data: got 5 observations, need at least 10"
        );
    }

    #[test]
    fn error_non_finite_data() {
        let e = MarkovError::NonFiniteData;
        assert_eq!(e.to_string(), "input data contains non-finite values");
    }

    #[test]
    fn error_invalid_month() {
        let e = MarkovError::InvalidMonth { month: 13 };
        assert_eq!(e.to_string(), "invalid month: 13 (must be 1..=12)");
    }

    #[test]
    fn error_invalid_threshold() {
        let e = MarkovError::InvalidThreshold {
            reason: "too high".to_string(),
        };
        assert_eq!(e.to_string(), "invalid threshold: too high");
    }

    #[test]
    fn error_invalid_spell_factor() {
        let e = MarkovError::InvalidSpellFactor {
            month: 3,
            value: -1.0,
        };
        assert_eq!(
            e.to_string(),
            "invalid spell factor for month 3: -1 (must be finite and > 0)"
        );
    }

    #[test]
    fn error_length_mismatch() {
        let e = MarkovError::LengthMismatch {
            precip_len: 100,
            months_len: 99,
        };
        assert_eq!(
            e.to_string(),
            "length mismatch: precip has 100 elements, months has 99"
        );
    }

    #[test]
    fn error_buffer_length_mismatch() {
        let e = MarkovError::BufferLengthMismatch {
            expected: 365,
            got: 366,
        };
        assert_eq!(
            e.to_string(),
            "buffer length mismatch: expected 365, got 366"
        );
    }

    #[test]
    fn error_is_std_error() {
        fn assert_impl<T: std::error::Error>() {}
        assert_impl::<MarkovError>();
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<MarkovError>();
    }
}
