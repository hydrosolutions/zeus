//! Nelder-Mead optimizer for ARMA maximum-likelihood estimation.
//!
//! Wraps the `argmin` crate to minimize the negative concentrated
//! log-likelihood over unconstrained PACF parameters.
//!
//! **Not part of the public API.**

use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;

use crate::error::ArmaError;
use crate::fit::ArmaFit;
use crate::kalman;
use crate::params;
use crate::spec::ArmaSpec;
use crate::state_space::StateSpace;

/// Fits an ARMA(p,q) model to data via exact MLE.
///
/// This is the full pipeline:
/// 1. Validate data
/// 2. Center (subtract mean)
/// 3. Optimize concentrated log-likelihood via Nelder-Mead
/// 4. Extract final parameters via full Kalman pass
pub(crate) fn fit_arma(p: usize, q: usize, data: &[f64]) -> Result<ArmaFit, ArmaError> {
    // 1. Validate
    if data.is_empty() {
        return Err(ArmaError::EmptyData);
    }
    if data.iter().any(|x| !x.is_finite()) {
        return Err(ArmaError::NonFiniteData);
    }
    let min_len = p.max(q).max(1) + 1;
    if data.len() < min_len {
        return Err(ArmaError::InsufficientData {
            n: data.len(),
            min: min_len,
        });
    }
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max_val - min_val).abs() < f64::EPSILON {
        return Err(ArmaError::ConstantData);
    }

    // 2. Center — subtract sample mean
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();

    // 3. ARMA(0,0) fast path — no optimization needed
    if p == 0 && q == 0 {
        let sigma2 = centered.iter().map(|x| x * x).sum::<f64>() / n;
        let log_likelihood =
            -0.5 * n * (2.0 * std::f64::consts::PI).ln() - 0.5 * n * sigma2.ln() - 0.5 * n;
        return Ok(ArmaFit::new(
            ArmaSpec::new(0, 0),
            vec![],
            vec![],
            sigma2,
            centered,
            log_likelihood,
            mean,
        ));
    }

    // 4. Build simplex for Nelder-Mead
    let dim = p + q;
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(dim + 1);
    simplex.push(vec![0.0; dim]); // origin
    for i in 0..dim {
        let mut vertex = vec![0.0; dim];
        vertex[i] = 0.5;
        simplex.push(vertex);
    }

    // 5. Set up cost function
    let cost = ArmaCost { data: &centered, p };

    // 6. Run Nelder-Mead
    let solver = NelderMead::new(simplex)
        .with_sd_tolerance(1e-8)
        .map_err(|_| ArmaError::OptimizationFailed)?;
    let result = Executor::new(cost, solver)
        .configure(|state| state.max_iters(1000))
        .run()
        .map_err(|_| ArmaError::OptimizationFailed)?;

    let best_params = result
        .state()
        .best_param
        .as_ref()
        .ok_or(ArmaError::OptimizationFailed)?;

    // 7. Extract final coefficients
    let (alpha, beta) = best_params.split_at(p);
    let ar = params::unconstrained_to_coeffs(alpha);
    let ma = params::unconstrained_to_coeffs(beta);

    // 8. Run full Kalman for sigma2, residuals, log-likelihood
    let ss = StateSpace::new(&ar, &ma);
    let output = kalman::kalman_full(&ss, &centered)?;

    Ok(ArmaFit::new(
        ArmaSpec::new(p, q),
        ar,
        ma,
        output.sigma2,
        output.residuals,
        output.log_likelihood,
        mean,
    ))
}

/// Cost function for argmin: negative concentrated log-likelihood.
struct ArmaCost<'a> {
    data: &'a [f64],
    p: usize,
}

impl CostFunction for ArmaCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let (alpha, beta) = params.split_at(self.p);
        let ar = params::unconstrained_to_coeffs(alpha);
        let ma = params::unconstrained_to_coeffs(beta);
        let ss = StateSpace::new(&ar, &ma);

        match kalman::kalman_concentrated_loglik(&ss, self.data) {
            Ok(loglik) if loglik.is_finite() => Ok(-loglik),
            _ => Ok(f64::MAX),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    #[test]
    fn validation_empty() {
        let result = fit_arma(1, 0, &[]);
        assert!(matches!(result, Err(ArmaError::EmptyData)));
    }

    #[test]
    fn validation_insufficient() {
        let result = fit_arma(2, 0, &[1.0, 2.0]);
        assert!(matches!(result, Err(ArmaError::InsufficientData { .. })));
    }

    #[test]
    fn validation_non_finite() {
        let result = fit_arma(1, 0, &[1.0, f64::NAN, 3.0]);
        assert!(matches!(result, Err(ArmaError::NonFiniteData)));

        let result = fit_arma(1, 0, &[1.0, f64::INFINITY, 3.0]);
        assert!(matches!(result, Err(ArmaError::NonFiniteData)));
    }

    #[test]
    fn validation_constant() {
        let result = fit_arma(1, 0, &[5.0, 5.0, 5.0, 5.0, 5.0]);
        assert!(matches!(result, Err(ArmaError::ConstantData)));
    }

    #[test]
    fn arma00_white_noise() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng)).collect();
        let fit = fit_arma(0, 0, &data).unwrap();
        assert_eq!(fit.order(), (0, 0));
        assert!(
            fit.sigma2() > 0.5 && fit.sigma2() < 1.5,
            "sigma2 = {}",
            fit.sigma2()
        );
        assert!(fit.mean().abs() < 0.2, "mean = {}", fit.mean());
    }

    #[test]
    fn ar1_coefficient_recovery() {
        let phi = 0.7;
        let sigma2: f64 = 1.0;
        let n = 1000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let normal = Normal::new(0.0, sigma2.sqrt()).unwrap();

        let mut data = vec![0.0; n];
        for t in 1..n {
            data[t] = phi * data[t - 1] + normal.sample(&mut rng);
        }

        let fit = fit_arma(1, 0, &data).unwrap();
        assert_eq!(fit.order(), (1, 0));
        assert!(
            (fit.ar()[0] - phi).abs() < 0.15,
            "AR(1) phi: expected ~{}, got {}",
            phi,
            fit.ar()[0]
        );
    }

    #[test]
    fn ma1_recovery() {
        let theta = 0.5;
        let sigma2: f64 = 1.0;
        let n = 1000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(456);
        let normal = Normal::new(0.0, sigma2.sqrt()).unwrap();

        let mut data = vec![0.0; n];
        let mut eps = vec![0.0; n];
        for t in 0..n {
            eps[t] = normal.sample(&mut rng);
            data[t] = eps[t] + if t > 0 { theta * eps[t - 1] } else { 0.0 };
        }

        let fit = fit_arma(0, 1, &data).unwrap();
        assert_eq!(fit.order(), (0, 1));
        assert!(
            (fit.ma()[0] - theta).abs() < 0.15,
            "MA(1) theta: expected ~{}, got {}",
            theta,
            fit.ma()[0]
        );
    }

    #[test]
    fn white_noise_ar1_spec_gives_small_phi() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(789);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng)).collect();

        let fit = fit_arma(1, 0, &data).unwrap();
        assert!(
            fit.ar()[0].abs() < 0.15,
            "Expected phi ≈ 0 for white noise, got {}",
            fit.ar()[0]
        );
    }
}
