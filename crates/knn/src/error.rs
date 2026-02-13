//! Error types for the zeus-knn crate.

/// Error type for all fallible operations in the zeus-knn crate.
#[derive(Debug, Clone, thiserror::Error)]
pub enum KnnError {
    /// Returned when the candidates slice is empty.
    #[error("no candidates provided")]
    EmptyCandidates,

    /// Returned when k is zero.
    #[error("k must be >= 1, got {k}")]
    InvalidK {
        /// The invalid k value.
        k: usize,
    },

    /// Returned when n (number of samples) is zero.
    #[error("n must be >= 1, got {n}")]
    InvalidN {
        /// The invalid n value.
        n: usize,
    },

    /// Returned when epsilon is non-finite or non-positive.
    #[error("epsilon must be finite and positive, got {epsilon}")]
    InvalidEpsilon {
        /// The invalid epsilon value.
        epsilon: f64,
    },

    /// Returned when the target slice length does not match n_vars.
    #[error("target length {target} does not match n_vars {n_vars}")]
    TargetDimensionMismatch {
        /// Length of the target slice.
        target: usize,
        /// Expected number of variables.
        n_vars: usize,
    },

    /// Returned when the weights slice length does not match n_vars.
    #[error("weights length {weights} does not match n_vars {n_vars}")]
    WeightsDimensionMismatch {
        /// Length of the weights slice.
        weights: usize,
        /// Expected number of variables.
        n_vars: usize,
    },

    /// Returned when the candidates slice length is not divisible by n_vars.
    #[error("candidates length {len} is not divisible by n_vars {n_vars}")]
    CandidatesShapeMismatch {
        /// Length of the candidates slice.
        len: usize,
        /// Expected number of variables.
        n_vars: usize,
    },

    /// Returned when a required input contains NaN or infinity.
    #[error("non-finite value in {input}")]
    NonFiniteInput {
        /// Name of the input containing the non-finite value.
        input: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_empty_candidates() {
        let e = KnnError::EmptyCandidates;
        assert_eq!(e.to_string(), "no candidates provided");
    }

    #[test]
    fn error_invalid_k() {
        let e = KnnError::InvalidK { k: 0 };
        assert_eq!(e.to_string(), "k must be >= 1, got 0");
    }

    #[test]
    fn error_invalid_n() {
        let e = KnnError::InvalidN { n: 0 };
        assert_eq!(e.to_string(), "n must be >= 1, got 0");
    }

    #[test]
    fn error_invalid_epsilon() {
        let e = KnnError::InvalidEpsilon { epsilon: -0.5 };
        assert_eq!(
            e.to_string(),
            "epsilon must be finite and positive, got -0.5"
        );
    }

    #[test]
    fn error_target_dimension_mismatch() {
        let e = KnnError::TargetDimensionMismatch {
            target: 3,
            n_vars: 5,
        };
        assert_eq!(e.to_string(), "target length 3 does not match n_vars 5");
    }

    #[test]
    fn error_weights_dimension_mismatch() {
        let e = KnnError::WeightsDimensionMismatch {
            weights: 2,
            n_vars: 4,
        };
        assert_eq!(e.to_string(), "weights length 2 does not match n_vars 4");
    }

    #[test]
    fn error_candidates_shape_mismatch() {
        let e = KnnError::CandidatesShapeMismatch { len: 10, n_vars: 3 };
        assert_eq!(
            e.to_string(),
            "candidates length 10 is not divisible by n_vars 3"
        );
    }

    #[test]
    fn error_non_finite_input() {
        let e = KnnError::NonFiniteInput { input: "weights" };
        assert_eq!(e.to_string(), "non-finite value in weights");
    }

    #[test]
    fn error_is_std_error() {
        fn assert_impl<T: std::error::Error>() {}
        assert_impl::<KnnError>();
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<KnnError>();
    }
}
