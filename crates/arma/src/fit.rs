//! Fitted ARMA model results.

use ndarray::Array2;
use rand::Rng;

use crate::spec::ArmaSpec;

/// A fitted ARMA(p,q) model produced by [`ArmaSpec::fit()`].
///
/// Contains estimated AR (`phi`) and MA (`theta`) coefficients,
/// innovation variance (`sigma2`), residuals, and log-likelihood.
/// Use accessors to inspect results or call [`ArmaFit::simulate()`]
/// to generate synthetic realisations.
///
/// # Typestate Workflow
///
/// ```mermaid
/// graph LR
///     B["ArmaFit"] --> C[".ar() — AR coefficients"]
///     B --> D[".ma() — MA coefficients"]
///     B --> E[".sigma2() — innovation variance"]
///     B --> F[".aic() — Akaike Information Criterion"]
///     B --> G[".simulate(n, n_sim, &mut rng)"]
/// ```
#[derive(Clone, Debug)]
pub struct ArmaFit {
    spec: ArmaSpec,
    ar: Vec<f64>,
    ma: Vec<f64>,
    sigma2: f64,
    residuals: Vec<f64>,
    log_likelihood: f64,
}

impl ArmaFit {
    /// Creates a new `ArmaFit` (crate-internal constructor).
    #[allow(dead_code)]
    pub(crate) fn new(
        spec: ArmaSpec,
        ar: Vec<f64>,
        ma: Vec<f64>,
        sigma2: f64,
        residuals: Vec<f64>,
        log_likelihood: f64,
    ) -> Self {
        Self {
            spec,
            ar,
            ma,
            sigma2,
            residuals,
            log_likelihood,
        }
    }

    /// Returns the [`ArmaSpec`] that produced this fit.
    pub fn spec(&self) -> ArmaSpec {
        self.spec
    }

    /// Returns the `(p, q)` order of the fitted model.
    pub fn order(&self) -> (usize, usize) {
        (self.spec.p(), self.spec.q())
    }

    /// Returns the AR coefficients (`phi`).
    pub fn ar(&self) -> &[f64] {
        &self.ar
    }

    /// Returns the MA coefficients (`theta`).
    pub fn ma(&self) -> &[f64] {
        &self.ma
    }

    /// Returns the innovation variance (`sigma2`).
    pub fn sigma2(&self) -> f64 {
        self.sigma2
    }

    /// Returns the one-step-ahead prediction residuals.
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// Returns the maximised log-likelihood.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Computes the Akaike Information Criterion (AIC) for this fit.
    ///
    /// AIC = 2k - 2 * log_likelihood, where k = p + q + 1 (number of
    /// estimated parameters: AR coefficients + MA coefficients + variance).
    /// Lower AIC indicates a better trade-off between fit and complexity.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let fit = ArmaSpec::new(1, 0).fit(&data)?;
    /// println!("AIC = {}", fit.aic());
    /// ```
    pub fn aic(&self) -> f64 {
        let k = (self.spec.p() + self.spec.q() + 1) as f64;
        2.0 * k - 2.0 * self.log_likelihood
    }

    /// Generates synthetic realisations from this fitted ARMA model.
    ///
    /// Draws `n_sim` independent sample paths, each of length `n`,
    /// by driving the model with Gaussian white noise.
    ///
    /// Returns an [`Array2<f64>`] with shape `(n, n_sim)` — each column
    /// is one realisation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rand::thread_rng;
    /// let paths = fit.simulate(365, 100, &mut thread_rng());
    /// assert_eq!(paths.shape(), &[365, 100]);
    /// ```
    pub fn simulate<R: Rng>(&self, n: usize, n_sim: usize, rng: &mut R) -> Array2<f64> {
        let _ = (n, n_sim, rng);
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_accessors_round_trip() {
        let spec = ArmaSpec::new(2, 1);
        let fit = ArmaFit::new(
            spec,
            vec![0.5, -0.3],
            vec![0.4],
            1.0,
            vec![0.1, -0.2, 0.3],
            -50.0,
        );

        assert_eq!(fit.spec().p(), 2);
        assert_eq!(fit.spec().q(), 1);
        assert_eq!(fit.order(), (2, 1));
        assert_eq!(fit.ar(), &[0.5, -0.3]);
        assert_eq!(fit.ma(), &[0.4]);
        assert_eq!(fit.sigma2(), 1.0);
        assert_eq!(fit.residuals(), &[0.1, -0.2, 0.3]);
        assert_eq!(fit.log_likelihood(), -50.0);
    }

    #[test]
    fn fit_aic_computation() {
        let spec = ArmaSpec::new(1, 1);
        let fit = ArmaFit::new(spec, vec![0.5], vec![0.3], 1.0, vec![], -100.0);

        // k = p + q + 1 = 1 + 1 + 1 = 3
        // AIC = 2*3 - 2*(-100) = 6 + 200 = 206
        let expected_aic = 206.0;
        assert!((fit.aic() - expected_aic).abs() < f64::EPSILON);
    }

    #[test]
    fn fit_is_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<ArmaFit>();
    }

    #[test]
    fn fit_empty_coefficients() {
        let spec = ArmaSpec::new(0, 0);
        let fit = ArmaFit::new(spec, vec![], vec![], 1.0, vec![], 0.0);

        assert!(fit.ar().is_empty());
        assert!(fit.ma().is_empty());
    }
}
