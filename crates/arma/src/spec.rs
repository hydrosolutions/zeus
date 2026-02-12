//! ARMA model specification (unfitted).

use crate::error::ArmaError;
use crate::fit::ArmaFit;

/// An unfitted ARMA(p,q) model specification.
///
/// This is the entry point of the typestate workflow. Create a spec with
/// [`ArmaSpec::new()`], then call [`ArmaSpec::fit()`] to obtain an [`ArmaFit`].
///
/// # Typestate Workflow
///
/// ```mermaid
/// graph LR
///     A["ArmaSpec::new(p, q)"] -->|".fit(&data)?"| B["ArmaFit"]
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArmaSpec {
    p: usize,
    q: usize,
}

impl ArmaSpec {
    /// Creates a new ARMA(p,q) specification with AR order `p` and MA order `q`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeus_arma::ArmaSpec;
    ///
    /// let spec = ArmaSpec::new(2, 1);
    /// assert_eq!(spec.p(), 2);
    /// assert_eq!(spec.q(), 1);
    /// ```
    pub fn new(p: usize, q: usize) -> Self {
        Self { p, q }
    }

    /// Returns the AR order (`p`).
    pub fn p(&self) -> usize {
        self.p
    }

    /// Returns the MA order (`q`).
    pub fn q(&self) -> usize {
        self.q
    }

    /// Fits this ARMA(p,q) specification to observed data via exact
    /// maximum-likelihood (Kalman filter).
    ///
    /// Returns an [`ArmaFit`] containing estimated coefficients,
    /// innovation variance (`sigma2`), residuals, and log-likelihood.
    ///
    /// # Errors
    ///
    /// | Variant | Trigger |
    /// |---------|---------|
    /// | [`ArmaError::EmptyData`] | `data` is empty |
    /// | [`ArmaError::InsufficientData`] | `data.len() < max(p, q) + 1` |
    /// | [`ArmaError::NonFiniteData`] | any element is NaN or infinite |
    /// | [`ArmaError::ConstantData`] | all elements are identical |
    /// | [`ArmaError::NonStationary`] | estimated AR roots inside unit circle |
    /// | [`ArmaError::OptimizationFailed`] | optimizer fails to converge |
    pub fn fit(&self, data: &[f64]) -> Result<ArmaFit, ArmaError> {
        crate::optimizer::fit_arma(self.p, self.q, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_round_trip() {
        let spec = ArmaSpec::new(2, 1);
        assert_eq!(spec.p(), 2);
        assert_eq!(spec.q(), 1);
    }

    #[test]
    fn spec_zero_order() {
        let spec = ArmaSpec::new(0, 0);
        assert_eq!(spec.p(), 0);
        assert_eq!(spec.q(), 0);
    }

    #[test]
    fn spec_is_copy() {
        let a = ArmaSpec::new(1, 1);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn spec_partial_eq() {
        assert_eq!(ArmaSpec::new(1, 1), ArmaSpec::new(1, 1));
        assert_ne!(ArmaSpec::new(1, 1), ArmaSpec::new(2, 0));
    }

    #[test]
    fn spec_debug_format() {
        let debug_str = format!("{:?}", ArmaSpec::new(1, 2));
        assert!(debug_str.contains("ArmaSpec"));
    }

    #[test]
    fn fit_empty_data() {
        let err = ArmaSpec::new(1, 0).fit(&[]).unwrap_err();
        assert!(matches!(err, ArmaError::EmptyData));
    }

    #[test]
    fn fit_insufficient_data() {
        let err = ArmaSpec::new(2, 0).fit(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, ArmaError::InsufficientData { .. }));
    }

    #[test]
    fn fit_nan_data() {
        let err = ArmaSpec::new(1, 0).fit(&[1.0, f64::NAN, 3.0]).unwrap_err();
        assert!(matches!(err, ArmaError::NonFiniteData));
    }

    #[test]
    fn fit_inf_data() {
        let err = ArmaSpec::new(1, 0)
            .fit(&[1.0, f64::INFINITY, 3.0])
            .unwrap_err();
        assert!(matches!(err, ArmaError::NonFiniteData));
    }

    #[test]
    fn fit_constant_data() {
        let err = ArmaSpec::new(1, 0)
            .fit(&[5.0, 5.0, 5.0, 5.0, 5.0])
            .unwrap_err();
        assert!(matches!(err, ArmaError::ConstantData));
    }

    #[test]
    fn fit_valid_data() {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..200).map(|_| normal.sample(&mut rng)).collect();

        let fit = ArmaSpec::new(1, 0).fit(&data).unwrap();
        assert_eq!(fit.order(), (1, 0));
        assert!(fit.sigma2() > 0.0);
        assert!(fit.log_likelihood().is_finite());
    }
}
