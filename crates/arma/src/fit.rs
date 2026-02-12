//! Fitted ARMA model results.

use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};

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
    mean: f64,
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
        mean: f64,
    ) -> Self {
        Self {
            spec,
            ar,
            ma,
            sigma2,
            residuals,
            log_likelihood,
            mean,
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

    /// Returns the estimated mean of the data used to fit this model.
    ///
    /// The ARMA model is fitted to the centred data (observations minus mean).
    /// Simulated paths from [`ArmaFit::simulate()`] have this mean added back.
    pub fn mean(&self) -> f64 {
        self.mean
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
        const BURN_IN: usize = 100;

        // Edge cases
        if n == 0 || n_sim == 0 {
            return Array2::zeros((n, n_sim));
        }

        // Guard sigma2
        if !self.sigma2.is_finite() || self.sigma2 <= 0.0 {
            return Array2::zeros((n, n_sim));
        }

        let p = self.ar.len();
        let q = self.ma.len();
        let n_tot = BURN_IN + n;

        let normal = Normal::new(0.0, self.sigma2.sqrt()).unwrap();
        let mut output = Array2::zeros((n, n_sim));

        for sim in 0..n_sim {
            // Draw all innovations for this realization
            let eps: Vec<f64> = (0..n_tot).map(|_| normal.sample(rng)).collect();
            let mut y = vec![0.0; n_tot];

            for t in 0..n_tot {
                let mut val = eps[t];
                // AR part
                for i in 0..p {
                    if t > i {
                        val += self.ar[i] * y[t - 1 - i];
                    }
                }
                // MA part
                for j in 0..q {
                    if t > j {
                        val += self.ma[j] * eps[t - 1 - j];
                    }
                }
                y[t] = val;
            }

            // Copy post-burn-in into output column, adding mean
            for (i, &val) in y[BURN_IN..].iter().enumerate() {
                output[[i, sim]] = val + self.mean;
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

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
            0.0,
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
        let fit = ArmaFit::new(spec, vec![0.5], vec![0.3], 1.0, vec![], -100.0, 0.0);

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
        let fit = ArmaFit::new(spec, vec![], vec![], 1.0, vec![], 0.0, 0.0);

        assert!(fit.ar().is_empty());
        assert!(fit.ma().is_empty());
    }

    #[test]
    fn simulate_shape() {
        let spec = ArmaSpec::new(1, 0);
        let fit = ArmaFit::new(spec, vec![0.5], vec![], 1.0, vec![], -50.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = fit.simulate(100, 5, &mut rng);
        assert_eq!(result.shape(), &[100, 5]);
    }

    #[test]
    fn simulate_zero_length() {
        let spec = ArmaSpec::new(0, 0);
        let fit = ArmaFit::new(spec, vec![], vec![], 1.0, vec![], 0.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        assert_eq!(fit.simulate(0, 10, &mut rng).shape(), &[0, 10]);
        assert_eq!(fit.simulate(10, 0, &mut rng).shape(), &[10, 0]);
    }

    #[test]
    fn simulate_white_noise_stats() {
        let spec = ArmaSpec::new(0, 0);
        let sigma2 = 2.0;
        let fit = ArmaFit::new(spec, vec![], vec![], sigma2, vec![], 0.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let result = fit.simulate(10000, 1, &mut rng);
        let col = result.column(0);
        let mean: f64 = col.mean().unwrap();
        let var: f64 = col.mapv(|x| (x - mean).powi(2)).mean().unwrap();
        assert!(mean.abs() < 0.1, "mean = {}", mean);
        assert!((var - sigma2).abs() < 0.3, "var = {}", var);
    }

    #[test]
    fn simulate_ar1_stats() {
        let phi = 0.7;
        let sigma2 = 1.0;
        let spec = ArmaSpec::new(1, 0);
        let fit = ArmaFit::new(spec, vec![phi], vec![], sigma2, vec![], 0.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(456);
        let result = fit.simulate(10000, 1, &mut rng);
        let col = result.column(0);
        let mean: f64 = col.mean().unwrap();
        let n = col.len() as f64;
        let var: f64 = col.mapv(|x| (x - mean).powi(2)).sum() / n;
        let theoretical_var = sigma2 / (1.0 - phi * phi);
        assert!(mean.abs() < 0.2, "mean = {}", mean);
        assert!(
            (var - theoretical_var).abs() < 0.5,
            "var = {}, expected = {}",
            var,
            theoretical_var
        );

        // Lag-1 autocorrelation ≈ phi
        let shifted: Vec<f64> = col.iter().skip(1).copied().collect();
        let base: Vec<f64> = col.iter().take(col.len() - 1).copied().collect();
        let cov: f64 = shifted
            .iter()
            .zip(base.iter())
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / n;
        let acf1 = cov / var;
        assert!(
            (acf1 - phi).abs() < 0.1,
            "acf1 = {}, expected = {}",
            acf1,
            phi
        );
    }

    #[test]
    fn simulate_ma1_variance() {
        let theta = 0.6;
        let sigma2 = 1.5;
        let spec = ArmaSpec::new(0, 1);
        let fit = ArmaFit::new(spec, vec![], vec![theta], sigma2, vec![], 0.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(789);
        let result = fit.simulate(10000, 1, &mut rng);
        let col = result.column(0);
        let mean: f64 = col.mean().unwrap();
        let n = col.len() as f64;
        let var: f64 = col.mapv(|x| (x - mean).powi(2)).sum() / n;
        let theoretical_var = sigma2 * (1.0 + theta * theta);
        assert!(
            (var - theoretical_var).abs() < 0.5,
            "var = {}, expected = {}",
            var,
            theoretical_var
        );
    }

    #[test]
    fn simulate_deterministic_with_seed() {
        let spec = ArmaSpec::new(1, 1);
        let fit = ArmaFit::new(spec, vec![0.5], vec![0.3], 1.0, vec![], -50.0, 0.0);
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let a = fit.simulate(50, 3, &mut rng1);
        let b = fit.simulate(50, 3, &mut rng2);
        assert_eq!(a, b);
    }

    #[test]
    fn simulate_multiple_realizations_independent() {
        let spec = ArmaSpec::new(1, 0);
        let fit = ArmaFit::new(spec, vec![0.5], vec![], 1.0, vec![], -50.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        let result = fit.simulate(100, 2, &mut rng);
        // Columns should be different
        let col0: Vec<f64> = result.column(0).to_vec();
        let col1: Vec<f64> = result.column(1).to_vec();
        assert_ne!(col0, col1);
    }

    #[test]
    fn simulate_all_values_finite() {
        let spec = ArmaSpec::new(2, 1);
        let fit = ArmaFit::new(spec, vec![0.5, -0.3], vec![0.4], 1.0, vec![], -50.0, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = fit.simulate(500, 10, &mut rng);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn simulate_with_mean() {
        let mean = 5.0;
        let spec = ArmaSpec::new(0, 0);
        let fit = ArmaFit::new(spec, vec![], vec![], 0.01, vec![], 0.0, mean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = fit.simulate(1000, 1, &mut rng);
        let sample_mean: f64 = result.column(0).mean().unwrap();
        assert!(
            (sample_mean - mean).abs() < 0.1,
            "sample_mean = {}, expected ≈ {}",
            sample_mean,
            mean
        );
    }

    #[test]
    fn mean_accessor() {
        let spec = ArmaSpec::new(0, 0);
        let fit = ArmaFit::new(spec, vec![], vec![], 1.0, vec![], 0.0, 3.125);
        assert_eq!(fit.mean(), 3.125);
    }
}
