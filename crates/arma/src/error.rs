//! Error types for the zeus-arma crate.

/// Error type for all fallible operations in the zeus-arma crate.
///
/// This enum covers validation failures, numerical issues, and optimization
/// problems that may occur during ARMA model fitting and simulation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ArmaError {
    /// Returned when the input data is empty.
    #[error("input data is empty")]
    EmptyData,

    /// Returned when the input data has fewer observations than required.
    #[error("insufficient data: got {n} observations, need at least {min}")]
    InsufficientData {
        /// Number of observations provided.
        n: usize,
        /// Minimum number of observations required.
        min: usize,
    },

    /// Returned when the input data contains non-finite values (NaN or infinity).
    #[error("input data contains non-finite values")]
    NonFiniteData,

    /// Returned when the input data has zero variance.
    #[error("input data is constant (zero variance)")]
    ConstantData,

    /// Returned when the fitted ARMA model violates stationarity constraints.
    #[error("fitted model is non-stationary")]
    NonStationary,

    /// Returned when the optimization algorithm fails to converge.
    #[error("optimisation failed to converge")]
    OptimizationFailed,

    /// Returned when all candidate ARMA models fail to fit.
    #[error("all ARMA candidates failed (max_p={max_p}, max_q={max_q})")]
    AllCandidatesFailed {
        /// Maximum AR order attempted.
        max_p: usize,
        /// Maximum MA order attempted.
        max_q: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_empty_data() {
        let err = ArmaError::EmptyData;
        assert_eq!(err.to_string(), "input data is empty");
    }

    #[test]
    fn error_insufficient_data() {
        let err = ArmaError::InsufficientData { n: 5, min: 10 };
        assert_eq!(
            err.to_string(),
            "insufficient data: got 5 observations, need at least 10"
        );
    }

    #[test]
    fn error_non_finite_data() {
        let err = ArmaError::NonFiniteData;
        assert_eq!(err.to_string(), "input data contains non-finite values");
    }

    #[test]
    fn error_constant_data() {
        let err = ArmaError::ConstantData;
        assert_eq!(err.to_string(), "input data is constant (zero variance)");
    }

    #[test]
    fn error_non_stationary() {
        let err = ArmaError::NonStationary;
        assert_eq!(err.to_string(), "fitted model is non-stationary");
    }

    #[test]
    fn error_optimization_failed() {
        let err = ArmaError::OptimizationFailed;
        assert_eq!(err.to_string(), "optimisation failed to converge");
    }

    #[test]
    fn error_all_candidates_failed() {
        let err = ArmaError::AllCandidatesFailed { max_p: 3, max_q: 2 };
        assert_eq!(
            err.to_string(),
            "all ARMA candidates failed (max_p=3, max_q=2)"
        );
    }

    #[test]
    fn error_is_std_error() {
        fn assert_impl<T: std::error::Error>() {}
        assert_impl::<ArmaError>();
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<ArmaError>();
    }
}
