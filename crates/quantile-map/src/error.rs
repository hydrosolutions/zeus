//! Error types for the zeus-quantile-map crate.

/// Error type for all fallible operations in the zeus-quantile-map crate.
#[derive(Debug, Clone, thiserror::Error)]
pub enum QuantileMapError {
    /// Returned when input data is empty.
    #[error("input data is empty")]
    EmptyData,

    /// Returned when precip, months, and years slices differ in length.
    #[error(
        "length mismatch: precip has {precip_len} elements, months has {months_len}, years has {years_len}"
    )]
    LengthMismatch {
        /// Length of the precipitation slice.
        precip_len: usize,
        /// Length of the months slice.
        months_len: usize,
        /// Length of the years slice.
        years_len: usize,
    },

    /// Returned when a month value is outside 1..=12.
    #[error("invalid month: {month} (must be 1..=12)")]
    InvalidMonth {
        /// The invalid month value.
        month: u8,
    },

    /// Returned when year indices are not contiguous.
    #[error("non-contiguous year indices (expected max {expected_max}): {reason}")]
    NonContiguousYears {
        /// The expected maximum year index.
        expected_max: u32,
        /// Description of the problem.
        reason: String,
    },

    /// Returned when the factor year count does not match the expected count.
    #[error("factor year count mismatch: expected {expected}, got {got}")]
    FactorYearMismatch {
        /// Expected number of factor years.
        expected: usize,
        /// Actual number of factor years.
        got: usize,
    },

    /// Returned when a configuration parameter is invalid.
    #[error("invalid configuration: {reason}")]
    InvalidConfig {
        /// Description of the problem.
        reason: String,
    },

    /// Returned when a gamma distribution cannot be constructed.
    ///
    /// The `message` field is a `String` (not a statrs error type) because
    /// statrs errors do not implement `Clone`.
    #[error("gamma construction failed (shape={shape}, scale={scale}): {message}")]
    GammaConstruction {
        /// Shape parameter that caused the failure.
        shape: f64,
        /// Scale parameter that caused the failure.
        scale: f64,
        /// Description of the failure.
        message: String,
    },

    /// Returned when no months had sufficient data for gamma fitting.
    #[error("no months had sufficient data for gamma fitting (skipped: {skipped_months:?})")]
    NoFittableMonths {
        /// Calendar months (1-indexed) that were skipped during fitting.
        skipped_months: Vec<u8>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_empty_data() {
        let e = QuantileMapError::EmptyData;
        assert_eq!(e.to_string(), "input data is empty");
    }

    #[test]
    fn error_length_mismatch() {
        let e = QuantileMapError::LengthMismatch {
            precip_len: 100,
            months_len: 99,
            years_len: 98,
        };
        assert_eq!(
            e.to_string(),
            "length mismatch: precip has 100 elements, months has 99, years has 98"
        );
    }

    #[test]
    fn error_invalid_month() {
        let e = QuantileMapError::InvalidMonth { month: 13 };
        assert_eq!(e.to_string(), "invalid month: 13 (must be 1..=12)");
    }

    #[test]
    fn error_non_contiguous_years() {
        let e = QuantileMapError::NonContiguousYears {
            expected_max: 10,
            reason: "gap at index 5".to_string(),
        };
        assert_eq!(
            e.to_string(),
            "non-contiguous year indices (expected max 10): gap at index 5"
        );
    }

    #[test]
    fn error_factor_year_mismatch() {
        let e = QuantileMapError::FactorYearMismatch {
            expected: 30,
            got: 25,
        };
        assert_eq!(
            e.to_string(),
            "factor year count mismatch: expected 30, got 25"
        );
    }

    #[test]
    fn error_invalid_config() {
        let e = QuantileMapError::InvalidConfig {
            reason: "threshold must be positive".to_string(),
        };
        assert_eq!(
            e.to_string(),
            "invalid configuration: threshold must be positive"
        );
    }

    #[test]
    fn error_gamma_construction() {
        let e = QuantileMapError::GammaConstruction {
            shape: -1.0,
            scale: 2.0,
            message: "shape must be positive".to_string(),
        };
        assert_eq!(
            e.to_string(),
            "gamma construction failed (shape=-1, scale=2): shape must be positive"
        );
    }

    #[test]
    fn error_no_fittable_months() {
        let e = QuantileMapError::NoFittableMonths {
            skipped_months: vec![1, 2, 3],
        };
        assert_eq!(
            e.to_string(),
            "no months had sufficient data for gamma fitting (skipped: [1, 2, 3])"
        );
    }

    #[test]
    fn error_is_std_error() {
        fn assert_impl<T: std::error::Error>() {}
        assert_impl::<QuantileMapError>();
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<QuantileMapError>();
    }
}
