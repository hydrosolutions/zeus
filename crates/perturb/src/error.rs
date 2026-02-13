//! Error types for the zeus-perturb crate.

use zeus_quantile_map::QuantileMapError;

/// Error type for all fallible operations in the zeus-perturb crate.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PerturbError {
    /// Returned when input data is empty.
    #[error("input data is empty")]
    EmptyData,

    /// Returned when array lengths do not match.
    #[error("length mismatch: expected {expected}, got {got} for {field}")]
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
        /// Name of the mismatched field.
        field: String,
    },

    /// Returned when a month value is outside 1..=12.
    #[error("invalid month: {month} (must be 1..=12)")]
    InvalidMonth {
        /// The invalid month value.
        month: u8,
    },

    /// Returned when year counts do not match.
    #[error("year count mismatch: expected {expected}, got {got}")]
    YearMismatch {
        /// Expected number of years.
        expected: usize,
        /// Actual number of years.
        got: usize,
    },

    /// Returned when a configuration parameter is invalid.
    #[error("invalid configuration: {reason}")]
    InvalidConfig {
        /// Description of the problem.
        reason: String,
    },

    /// Wrapped error from the quantile-map crate.
    #[error(transparent)]
    QuantileMap(#[from] QuantileMapError),

    /// Returned when a Gamma distribution cannot be constructed.
    #[error("gamma construction failed (shape={shape}, scale={scale}): {message}")]
    GammaConstruction {
        /// Shape parameter.
        shape: f64,
        /// Scale parameter.
        scale: f64,
        /// Description of the failure.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_empty_data() {
        let e = PerturbError::EmptyData;
        assert_eq!(e.to_string(), "input data is empty");
    }

    #[test]
    fn display_length_mismatch() {
        let e = PerturbError::LengthMismatch {
            expected: 100,
            got: 50,
            field: "month".to_string(),
        };
        assert!(e.to_string().contains("100"));
        assert!(e.to_string().contains("50"));
        assert!(e.to_string().contains("month"));
    }

    #[test]
    fn display_invalid_month() {
        let e = PerturbError::InvalidMonth { month: 13 };
        assert!(e.to_string().contains("13"));
    }

    #[test]
    fn display_year_mismatch() {
        let e = PerturbError::YearMismatch {
            expected: 30,
            got: 25,
        };
        assert!(e.to_string().contains("30"));
        assert!(e.to_string().contains("25"));
    }

    #[test]
    fn display_invalid_config() {
        let e = PerturbError::InvalidConfig {
            reason: "bad value".to_string(),
        };
        assert!(e.to_string().contains("bad value"));
    }

    #[test]
    fn from_quantile_map_error() {
        let qm_err = QuantileMapError::EmptyData;
        let e: PerturbError = qm_err.into();
        assert!(matches!(e, PerturbError::QuantileMap(_)));
    }

    #[test]
    fn display_gamma_construction() {
        let e = PerturbError::GammaConstruction {
            shape: -1.0,
            scale: 2.0,
            message: "shape must be positive".to_string(),
        };
        assert!(e.to_string().contains("-1"));
        assert!(e.to_string().contains("2"));
    }

    #[test]
    fn error_is_std_error() {
        fn assert_impl<T: std::error::Error>() {}
        assert_impl::<PerturbError>();
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<PerturbError>();
    }
}
