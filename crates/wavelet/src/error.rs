//! Error types for the zeus-wavelet crate.

/// Error type for all fallible operations in the zeus-wavelet crate.
///
/// Covers validation failures, numerical issues, and transform problems
/// that may occur during wavelet analysis.
#[derive(Debug, Clone, thiserror::Error)]
pub enum WaveletError {
    /// Returned when the input series is shorter than the minimum required length.
    #[error("series too short: got {len} observations, need at least {min}")]
    SeriesTooShort {
        /// Number of observations provided.
        len: usize,
        /// Minimum number of observations required.
        min: usize,
    },

    /// Returned when the input data contains non-finite values (NaN or infinity).
    #[error("input data contains non-finite values")]
    NonFiniteData,

    /// Returned when the requested decomposition level exceeds the maximum.
    #[error("level too high: requested {requested}, max for length {len} is {max}")]
    LevelTooHigh {
        /// Level that was requested.
        requested: usize,
        /// Maximum feasible level.
        max: usize,
        /// Length of the input series.
        len: usize,
    },

    /// Returned when the MRA reconstruction error exceeds tolerance.
    #[error("reconstruction error {0} exceeds tolerance")]
    ReconstructionError(f64),

    /// Returned when an unsupported wavelet filter name is provided.
    #[error("unsupported wavelet filter: {0}")]
    UnsupportedFilter(String),

    /// Returned when the MODWT computation fails.
    #[error("MODWT failed: {0}")]
    ModwtFailed(String),

    /// Returned when the MRA computation fails.
    #[error("MRA failed: {0}")]
    MraFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_series_too_short() {
        let err = WaveletError::SeriesTooShort { len: 3, min: 8 };
        assert_eq!(
            err.to_string(),
            "series too short: got 3 observations, need at least 8"
        );
    }

    #[test]
    fn error_non_finite_data() {
        let err = WaveletError::NonFiniteData;
        assert_eq!(err.to_string(), "input data contains non-finite values");
    }

    #[test]
    fn error_level_too_high() {
        let err = WaveletError::LevelTooHigh {
            requested: 10,
            max: 7,
            len: 256,
        };
        assert_eq!(
            err.to_string(),
            "level too high: requested 10, max for length 256 is 7"
        );
    }

    #[test]
    fn error_reconstruction() {
        let err = WaveletError::ReconstructionError(0.05);
        assert_eq!(
            err.to_string(),
            "reconstruction error 0.05 exceeds tolerance"
        );
    }

    #[test]
    fn error_unsupported_filter() {
        let err = WaveletError::UnsupportedFilter("coif4".into());
        assert_eq!(err.to_string(), "unsupported wavelet filter: coif4");
    }

    #[test]
    fn error_modwt_failed() {
        let err = WaveletError::ModwtFailed("singular matrix".into());
        assert_eq!(err.to_string(), "MODWT failed: singular matrix");
    }

    #[test]
    fn error_mra_failed() {
        let err = WaveletError::MraFailed("level mismatch".into());
        assert_eq!(err.to_string(), "MRA failed: level mismatch");
    }

    #[test]
    fn error_is_std_error() {
        fn assert_impl<T: std::error::Error>() {}
        assert_impl::<WaveletError>();
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<WaveletError>();
    }
}
