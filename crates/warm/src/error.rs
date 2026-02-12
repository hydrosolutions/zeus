//! Error types for the zeus-warm crate.

use zeus_arma::ArmaError;
use zeus_wavelet::WaveletError;

/// Error type for all fallible operations in the zeus-warm crate.
///
/// Covers wavelet decomposition errors, ARMA fitting errors, and
/// WARM simulation/filtering failures.
#[derive(Debug, Clone, thiserror::Error)]
pub enum WarmError {
    /// Wavelet transform error.
    #[error(transparent)]
    Wavelet(#[from] WaveletError),

    /// ARMA model error.
    #[error(transparent)]
    Arma(#[from] ArmaError),

    /// Returned when there are not enough simulations available.
    #[error("insufficient simulations: requested {requested}, available {available}")]
    InsufficientSimulations {
        /// Number of simulations requested.
        requested: usize,
        /// Number of simulations available.
        available: usize,
    },

    /// Returned when the WARM pool filtering fails.
    #[error("filtering failed: {0}")]
    FilteringFailed(String),

    /// Returned when there are no MRA components to simulate.
    #[error("no components to simulate")]
    NoComponentsToSimulate,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_wavelet_transparent() {
        let inner = WaveletError::NonFiniteData;
        let err = WarmError::from(inner);
        assert_eq!(err.to_string(), "input data contains non-finite values");
    }

    #[test]
    fn error_arma_transparent() {
        let inner = ArmaError::EmptyData;
        let err = WarmError::from(inner);
        assert_eq!(err.to_string(), "input data is empty");
    }

    #[test]
    fn error_insufficient_simulations() {
        let err = WarmError::InsufficientSimulations {
            requested: 100,
            available: 50,
        };
        assert_eq!(
            err.to_string(),
            "insufficient simulations: requested 100, available 50"
        );
    }

    #[test]
    fn error_filtering_failed() {
        let err = WarmError::FilteringFailed("bad bounds".into());
        assert_eq!(err.to_string(), "filtering failed: bad bounds");
    }

    #[test]
    fn error_no_components() {
        let err = WarmError::NoComponentsToSimulate;
        assert_eq!(err.to_string(), "no components to simulate");
    }

    #[test]
    fn error_is_std_error() {
        fn assert_impl<T: std::error::Error>() {}
        assert_impl::<WarmError>();
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<WarmError>();
    }
}
