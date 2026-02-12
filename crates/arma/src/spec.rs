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
        let _ = data;
        todo!()
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
}
