//! Configuration for KNN sampling queries.

use crate::error::KnnError;

/// Probability weighting scheme for neighbor selection.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Sampling {
    /// All k neighbors have equal probability 1/k.
    Uniform,
    /// Rank-based: closest neighbor gets weight 1/1, second 1/2, third 1/3, etc.
    /// Normalized to sum to 1. (Lall & Sharma 1996)
    #[default]
    Rank,
    /// Gaussian kernel: `exp(-dist^2 / (2 * bandwidth^2)) + epsilon`.
    ///
    /// If `bandwidth` is `None`, uses the median of k-nearest distances (adaptive).
    Gaussian {
        /// Optional fixed bandwidth. `None` means adaptive (median of k-nearest distances).
        bandwidth: Option<f64>,
    },
}

/// Configuration for a KNN sampling query.
///
/// Use the builder methods to customise parameters.
///
/// # Example
///
/// ```
/// use zeus_knn::{KnnConfig, Sampling};
///
/// let config = KnnConfig::new(5)
///     .with_n(100)
///     .with_sampling(Sampling::Rank);
///
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct KnnConfig {
    /// Number of nearest neighbors to consider.
    k: usize,
    /// Number of samples to draw (with replacement).
    n: usize,
    /// Probability weighting scheme.
    sampling: Sampling,
    /// Floor for Gaussian mode zero-prevention.
    epsilon: f64,
}

impl KnnConfig {
    /// Creates a new configuration with the given k.
    ///
    /// Defaults: `n = 1`, `sampling = Rank`, `epsilon = 1e-8`.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            n: 1,
            sampling: Sampling::Rank,
            epsilon: 1e-8,
        }
    }

    /// Sets the number of samples to draw.
    pub fn with_n(mut self, n: usize) -> Self {
        self.n = n;
        self
    }

    /// Sets the probability weighting scheme.
    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// Sets the epsilon floor for Gaussian mode.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Returns the number of nearest neighbors.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the number of samples to draw.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the probability weighting scheme.
    pub fn sampling(&self) -> &Sampling {
        &self.sampling
    }

    /// Returns the epsilon floor.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Validates this configuration.
    ///
    /// Returns an error if k < 1, n < 1, or epsilon is non-finite / non-positive.
    pub fn validate(&self) -> Result<(), KnnError> {
        if self.k < 1 {
            return Err(KnnError::InvalidK { k: self.k });
        }
        if self.n < 1 {
            return Err(KnnError::InvalidN { n: self.n });
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(KnnError::InvalidEpsilon {
                epsilon: self.epsilon,
            });
        }
        Ok(())
    }
}

impl Default for KnnConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let cfg = KnnConfig::default();
        assert_eq!(cfg.k(), 1);
        assert_eq!(cfg.n(), 1);
        assert_eq!(cfg.sampling(), &Sampling::Rank);
        assert!((cfg.epsilon() - 1e-8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_new() {
        let cfg = KnnConfig::new(5);
        assert_eq!(cfg.k(), 5);
        assert_eq!(cfg.n(), 1);
        assert_eq!(cfg.sampling(), &Sampling::Rank);
        assert!((cfg.epsilon() - 1e-8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_chaining() {
        let cfg = KnnConfig::new(10)
            .with_n(50)
            .with_sampling(Sampling::Gaussian {
                bandwidth: Some(2.5),
            })
            .with_epsilon(1e-6);

        assert_eq!(cfg.k(), 10);
        assert_eq!(cfg.n(), 50);
        assert_eq!(
            cfg.sampling(),
            &Sampling::Gaussian {
                bandwidth: Some(2.5)
            }
        );
        assert!((cfg.epsilon() - 1e-6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sampling_default() {
        assert_eq!(Sampling::default(), Sampling::Rank);
    }

    #[test]
    fn test_sampling_eq() {
        // Uniform == Uniform
        assert_eq!(Sampling::Uniform, Sampling::Uniform);
        // Rank == Rank
        assert_eq!(Sampling::Rank, Sampling::Rank);
        // Gaussian with same bandwidth
        assert_eq!(
            Sampling::Gaussian {
                bandwidth: Some(1.0)
            },
            Sampling::Gaussian {
                bandwidth: Some(1.0)
            }
        );
        // Gaussian with None bandwidth
        assert_eq!(
            Sampling::Gaussian { bandwidth: None },
            Sampling::Gaussian { bandwidth: None }
        );
        // Different variants are not equal
        assert_ne!(Sampling::Uniform, Sampling::Rank);
        assert_ne!(Sampling::Uniform, Sampling::Gaussian { bandwidth: None });
        assert_ne!(
            Sampling::Rank,
            Sampling::Gaussian {
                bandwidth: Some(1.0)
            }
        );
        // Different bandwidths are not equal
        assert_ne!(
            Sampling::Gaussian {
                bandwidth: Some(1.0)
            },
            Sampling::Gaussian {
                bandwidth: Some(2.0)
            }
        );
        assert_ne!(
            Sampling::Gaussian {
                bandwidth: Some(1.0)
            },
            Sampling::Gaussian { bandwidth: None }
        );
    }

    #[test]
    fn test_validate_ok() {
        // Default config validates
        assert!(KnnConfig::default().validate().is_ok());
        // Custom valid config validates
        let cfg = KnnConfig::new(10)
            .with_n(100)
            .with_sampling(Sampling::Uniform)
            .with_epsilon(1e-12);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_k() {
        let result = KnnConfig::new(0).validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, KnnError::InvalidK { k: 0 }),
            "expected InvalidK, got {err:?}"
        );
    }

    #[test]
    fn test_validate_invalid_n() {
        let result = KnnConfig::new(1).with_n(0).validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, KnnError::InvalidN { n: 0 }),
            "expected InvalidN, got {err:?}"
        );
    }

    #[test]
    fn test_validate_invalid_epsilon() {
        // Zero
        let result = KnnConfig::new(1).with_epsilon(0.0).validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            KnnError::InvalidEpsilon { epsilon } if epsilon == 0.0
        ));

        // Negative
        let result = KnnConfig::new(1).with_epsilon(-1.0).validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            KnnError::InvalidEpsilon { epsilon } if epsilon == -1.0
        ));

        // NaN
        let result = KnnConfig::new(1).with_epsilon(f64::NAN).validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            KnnError::InvalidEpsilon { .. }
        ));

        // Infinity
        let result = KnnConfig::new(1).with_epsilon(f64::INFINITY).validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            KnnError::InvalidEpsilon { .. }
        ));
    }

    #[test]
    fn test_validate_error_priority() {
        // Both k=0 and n=0 -- should return InvalidK first since k is checked first.
        let result = KnnConfig::new(0).with_n(0).validate();
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), KnnError::InvalidK { k: 0 }),
            "expected InvalidK to be returned first when both k and n are invalid"
        );
    }
}
