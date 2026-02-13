//! Gamma distribution parameter type and statrs bridge.

use crate::error::QuantileMapError;
use statrs::distribution::Gamma;

/// Validated parameters for a Gamma distribution (shape/scale convention).
///
/// Both `shape` (k) and `scale` (theta) must be finite and positive.
/// Use [`GammaParams::new`] for direct construction or
/// [`GammaParams::from_moments`] for method-of-moments estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaParams {
    shape: f64,
    scale: f64,
}

impl GammaParams {
    /// Create new gamma parameters after validating that both `shape` and
    /// `scale` are finite and strictly positive.
    pub fn new(shape: f64, scale: f64) -> Option<Self> {
        if shape.is_finite() && shape > 0.0 && scale.is_finite() && scale > 0.0 {
            Some(Self { shape, scale })
        } else {
            None
        }
    }

    /// Estimate gamma parameters from sample mean and variance using the
    /// method of moments (MME).
    ///
    /// - shape = mean² / var
    /// - scale = var / mean
    ///
    /// Returns `None` if `mean` or `var` are not finite and positive.
    pub fn from_moments(mean: f64, var: f64) -> Option<Self> {
        if !mean.is_finite() || mean <= 0.0 || !var.is_finite() || var <= 0.0 {
            return None;
        }
        let shape = (mean * mean) / var;
        let scale = var / mean;
        Self::new(shape, scale)
    }

    /// Shape parameter (k).
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Scale parameter (theta).
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Distribution mean (shape * scale).
    pub fn mean(&self) -> f64 {
        self.shape * self.scale
    }

    /// Distribution variance (shape * scale²).
    pub fn var(&self) -> f64 {
        self.shape * self.scale * self.scale
    }

    /// Rate parameter (1 / scale), used by statrs which parameterises Gamma
    /// by (shape, rate) rather than (shape, scale).
    #[allow(dead_code)]
    pub(crate) fn rate(&self) -> f64 {
        1.0 / self.scale
    }
}

/// Build a [`statrs::distribution::Gamma`] from validated [`GammaParams`].
///
/// Note: `statrs::distribution::Gamma::new` takes `(shape, rate)` where
/// `rate = 1 / scale`.
#[allow(dead_code)]
pub(crate) fn gamma_dist(params: &GammaParams) -> Result<Gamma, QuantileMapError> {
    Gamma::new(params.shape(), params.rate()).map_err(|e| QuantileMapError::GammaConstruction {
        shape: params.shape(),
        scale: params.scale(),
        message: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use statrs::distribution::ContinuousCDF;

    #[test]
    fn new_valid() {
        let p = GammaParams::new(2.0, 3.0).unwrap();
        assert_relative_eq!(p.shape(), 2.0);
        assert_relative_eq!(p.scale(), 3.0);
        assert_relative_eq!(p.mean(), 6.0);
        assert_relative_eq!(p.var(), 18.0);
    }

    #[test]
    fn new_invalid_zero_shape() {
        assert!(GammaParams::new(0.0, 1.0).is_none());
    }

    #[test]
    fn new_invalid_negative_scale() {
        assert!(GammaParams::new(1.0, -1.0).is_none());
    }

    #[test]
    fn new_invalid_nan() {
        assert!(GammaParams::new(f64::NAN, 1.0).is_none());
    }

    #[test]
    fn new_invalid_inf() {
        assert!(GammaParams::new(f64::INFINITY, 1.0).is_none());
    }

    #[test]
    fn from_moments_known() {
        let p = GammaParams::from_moments(6.0, 18.0).unwrap();
        assert_relative_eq!(p.shape(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(p.scale(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn from_moments_invalid() {
        assert!(GammaParams::from_moments(0.0, 1.0).is_none());
        assert!(GammaParams::from_moments(-1.0, 1.0).is_none());
    }

    #[test]
    fn gamma_dist_correct_mean() {
        use statrs::statistics::Distribution as StatrsDistribution;
        let params = GammaParams::new(2.0, 3.0).unwrap();
        let dist = gamma_dist(&params).unwrap();
        let dist_mean = StatrsDistribution::mean(&dist).unwrap();
        assert_relative_eq!(dist_mean, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn pgamma_qgamma_round_trip() {
        let params = GammaParams::new(2.5, 4.0).unwrap();
        let dist = gamma_dist(&params).unwrap();
        let xs = [0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0];
        for &x in &xs {
            let p = dist.cdf(x);
            let x_back = dist.inverse_cdf(p);
            assert_relative_eq!(x_back, x, epsilon = 1e-10);
        }
    }

    #[test]
    fn cdf_boundaries() {
        let params = GammaParams::new(2.0, 3.0).unwrap();
        let dist = gamma_dist(&params).unwrap();
        assert_relative_eq!(dist.cdf(0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(dist.cdf(1e6), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn cdf_monotonicity() {
        let params = GammaParams::new(2.0, 3.0).unwrap();
        let dist = gamma_dist(&params).unwrap();
        let xs = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0];
        let mut prev = f64::NEG_INFINITY;
        for &x in &xs {
            let p = dist.cdf(x);
            assert!(p >= prev, "CDF not monotone at x={x}: {p} < {prev}");
            prev = p;
        }
    }

    #[test]
    fn gamma_params_is_copy_clone_send_sync() {
        fn assert_impl<T: Copy + Clone + Send + Sync>() {}
        assert_impl::<GammaParams>();
    }
}
