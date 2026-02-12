//! BFGS optimizer with Nelder-Mead fallback for ARMA maximum-likelihood estimation.
//!
//! Wraps the `argmin` crate to minimize the negative concentrated
//! log-likelihood over unconstrained PACF parameters.  Gradients are
//! approximated via central finite differences.
//!
//! **Not part of the public API.**

use argmin::core::{CostFunction, Executor, Gradient};
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::quasinewton::BFGS;
use argmin_math::ArgminEye;

use crate::error::ArmaError;
use crate::fit::ArmaFit;
use crate::kalman;
use crate::params;
use crate::spec::ArmaSpec;
use crate::state_space::StateSpace;

/// Penalty cost for infeasible parameter configurations.
const PENALTY_COST: f64 = 1e18;

/// Finite-difference step for gradient approximation, matching R's `ndeps`.
const FD_STEP: f64 = 1e-3;

/// CSS (Conditional Sum of Squares) objective — no Kalman filter needed.
///
/// Computes residuals by direct AR/MA recursion, skips the first
/// `ncond = max(p, q)` values, returns `0.5 * ln(mean(e²))`.
fn css_cost(params: &[f64], data: &[f64], p: usize, q: usize) -> f64 {
    let (alpha, beta) = params.split_at(p);
    let ar = params::unconstrained_to_coeffs(alpha);
    let ma = params::unconstrained_to_coeffs(beta);
    if ar.iter().chain(ma.iter()).any(|c| !c.is_finite()) {
        return PENALTY_COST;
    }
    let n = data.len();
    let ncond = p.max(q);
    let mut resid = vec![0.0; n];
    for t in 0..n {
        let mut e_t = data[t];
        for i in 0..p {
            if t > i {
                e_t -= ar[i] * data[t - 1 - i];
            }
        }
        for j in 0..q {
            if t > j {
                e_t -= ma[j] * resid[t - 1 - j];
            }
        }
        resid[t] = e_t;
    }
    let effective_n = n - ncond;
    if effective_n == 0 {
        return PENALTY_COST;
    }
    let msr: f64 = resid[ncond..].iter().map(|e| e * e).sum::<f64>() / effective_n as f64;
    if msr <= 0.0 || !msr.is_finite() {
        return PENALTY_COST;
    }
    0.5 * msr.ln()
}

/// Argmin wrapper for CSS optimization via Nelder-Mead.
struct CssCost<'a> {
    data: &'a [f64],
    p: usize,
    q: usize,
}

impl CostFunction for CssCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(css_cost(params, self.data, self.p, self.q))
    }
}

/// Runs Nelder-Mead on CSS to produce warm-start parameters for ML.
fn css_optimize(data: &[f64], p: usize, q: usize) -> Result<Vec<f64>, ArmaError> {
    let dim = p + q;
    if dim == 0 {
        return Ok(vec![]);
    }
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(dim + 1);
    simplex.push(vec![0.0; dim]);
    for i in 0..dim {
        let mut v = vec![0.0; dim];
        v[i] = 0.5;
        simplex.push(v);
    }
    let cost = CssCost { data, p, q };
    let solver = NelderMead::new(simplex)
        .with_sd_tolerance(1e-6)
        .map_err(|_| ArmaError::OptimizationFailed)?;
    let result = Executor::new(cost, solver)
        .configure(|state| state.max_iters(500))
        .run()
        .map_err(|_| ArmaError::OptimizationFailed)?;
    let best = result
        .state()
        .best_param
        .clone()
        .ok_or(ArmaError::OptimizationFailed)?;
    if !result.state().best_cost.is_finite() || result.state().best_cost >= PENALTY_COST {
        return Err(ArmaError::OptimizationFailed);
    }
    Ok(best)
}

/// Fits an ARMA(p,q) model to data via exact MLE.
///
/// This is the full pipeline:
/// 1. Validate data
/// 2. Center (subtract mean)
/// 3. Optimize via BFGS (with Nelder-Mead fallback)
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

    // 4. Optimize: BFGS with CSS warm-start and Nelder-Mead fallback
    let dim = p + q;
    let css_init = css_optimize(&centered, p, q).unwrap_or_else(|_| vec![0.0; dim]);
    let best_params = bfgs_optimize(&centered, p, &css_init)
        .or_else(|_| nelder_mead_optimize(&centered, p, dim))?;

    // 5. Extract final coefficients
    let (alpha, beta) = best_params.split_at(p);
    let ar = params::unconstrained_to_coeffs(alpha);
    let ma = params::unconstrained_to_coeffs(beta);

    // 6. Run full Kalman for sigma2, residuals, log-likelihood
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

/// Attempts BFGS optimization of the ARMA concentrated log-likelihood.
fn bfgs_optimize(data: &[f64], p: usize, init: &[f64]) -> Result<Vec<f64>, ArmaError> {
    let dim = init.len();
    let cost = ArmaCost { data, p };
    let armijo = ArmijoCondition::new(1e-4).map_err(|_| ArmaError::OptimizationFailed)?;
    let linesearch = BacktrackingLineSearch::new(armijo)
        .rho(0.2)
        .map_err(|_| ArmaError::OptimizationFailed)?;
    let solver = BFGS::new(linesearch)
        .with_tolerance_grad(f64::EPSILON.sqrt())
        .map_err(|_| ArmaError::OptimizationFailed)?
        .with_tolerance_cost(f64::EPSILON.sqrt())
        .map_err(|_| ArmaError::OptimizationFailed)?;
    let init_inv_hessian: Vec<Vec<f64>> = ArgminEye::eye(dim);
    let result = Executor::new(cost, solver)
        .configure(|state| {
            state
                .param(init.to_vec())
                .inv_hessian(init_inv_hessian)
                .max_iters(100)
        })
        .run()
        .map_err(|_| ArmaError::OptimizationFailed)?;
    let best = result
        .state()
        .best_param
        .clone()
        .ok_or(ArmaError::OptimizationFailed)?;
    let best_cost = result.state().best_cost;
    if !best_cost.is_finite() || best_cost >= PENALTY_COST {
        return Err(ArmaError::OptimizationFailed);
    }
    Ok(best)
}

/// Falls back to Nelder-Mead if BFGS fails.
fn nelder_mead_optimize(data: &[f64], p: usize, dim: usize) -> Result<Vec<f64>, ArmaError> {
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(dim + 1);
    simplex.push(vec![0.0; dim]);
    for i in 0..dim {
        let mut vertex = vec![0.0; dim];
        vertex[i] = 0.5;
        simplex.push(vertex);
    }
    let cost = ArmaCost { data, p };
    let solver = NelderMead::new(simplex)
        .with_sd_tolerance(1e-8)
        .map_err(|_| ArmaError::OptimizationFailed)?;
    let result = Executor::new(cost, solver)
        .configure(|state| state.max_iters(1000))
        .run()
        .map_err(|_| ArmaError::OptimizationFailed)?;
    let best = result
        .state()
        .best_param
        .clone()
        .ok_or(ArmaError::OptimizationFailed)?;
    let best_cost = result.state().best_cost;
    if !best_cost.is_finite() || best_cost >= PENALTY_COST {
        return Err(ArmaError::OptimizationFailed);
    }
    Ok(best)
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
            _ => Ok(PENALTY_COST),
        }
    }
}

impl Gradient for ArmaCost<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let dim = params.len();
        let f_center = self.cost(params)?;
        let mut grad = vec![0.0; dim];
        for i in 0..dim {
            let h = FD_STEP;
            let mut forward = params.clone();
            forward[i] += h;
            let mut backward = params.clone();
            backward[i] -= h;
            let f_plus = self.cost(&forward)?;
            let f_minus = self.cost(&backward)?;
            let plus_ok = f_plus < PENALTY_COST;
            let minus_ok = f_minus < PENALTY_COST;
            grad[i] = match (plus_ok, minus_ok) {
                (true, true) => (f_plus - f_minus) / (2.0 * h),
                (true, false) => (f_plus - f_center) / h,
                (false, true) => (f_center - f_minus) / h,
                (false, false) => 0.0,
            };
        }
        Ok(grad)
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
            (fit.ar()[0] - phi).abs() < 0.10,
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

    #[test]
    fn gradient_is_finite() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..200).map(|_| normal.sample(&mut rng)).collect();
        let cost = ArmaCost { data: &data, p: 1 };
        let params = vec![0.0, 0.0]; // ARMA(1,1) at origin
        let grad = cost.gradient(&params).unwrap();
        assert_eq!(grad.len(), 2);
        for (i, g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "gradient[{}] = {}", i, g);
        }
    }

    #[test]
    fn bfgs_converges_for_ar1() {
        let phi = 0.7;
        let n = 1000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(100);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut data = vec![0.0; n];
        for t in 1..n {
            data[t] = phi * data[t - 1] + normal.sample(&mut rng);
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();
        let init = css_optimize(&centered, 1, 0).unwrap_or_else(|_| vec![0.0]);
        let result = bfgs_optimize(&centered, 1, &init);
        assert!(result.is_ok(), "BFGS failed for AR(1): {:?}", result.err());
        let best = result.unwrap();
        let ar = params::unconstrained_to_coeffs(&best);
        assert!(
            (ar[0] - phi).abs() < 0.15,
            "BFGS AR(1) phi: expected ~{}, got {}",
            phi,
            ar[0]
        );
    }

    #[test]
    fn bfgs_converges_for_ma1() {
        let theta = 0.5;
        let n = 1000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(200);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut data = vec![0.0; n];
        let mut eps = vec![0.0; n];
        for t in 0..n {
            eps[t] = normal.sample(&mut rng);
            data[t] = eps[t] + if t > 0 { theta * eps[t - 1] } else { 0.0 };
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();
        let init = css_optimize(&centered, 0, 1).unwrap_or_else(|_| vec![0.0]);
        let result = bfgs_optimize(&centered, 0, &init);
        assert!(result.is_ok(), "BFGS failed for MA(1): {:?}", result.err());
    }

    #[test]
    fn bfgs_converges_for_arma11() {
        let phi = 0.5;
        let theta = 0.3;
        let n = 2000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(300);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut data = vec![0.0; n];
        let mut eps = vec![0.0; n];
        for t in 0..n {
            eps[t] = normal.sample(&mut rng);
            data[t] = if t > 0 {
                phi * data[t - 1] + theta * eps[t - 1]
            } else {
                0.0
            } + eps[t];
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();
        let init = css_optimize(&centered, 1, 1).unwrap_or_else(|_| vec![0.0, 0.0]);
        let result = bfgs_optimize(&centered, 1, &init);
        assert!(
            result.is_ok(),
            "BFGS failed for ARMA(1,1): {:?}",
            result.err()
        );
    }

    #[test]
    fn bfgs_converges_for_ar2() {
        let phi = [0.5, -0.3];
        let n = 1000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(400);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut data = vec![0.0; n];
        for t in 2..n {
            data[t] = phi[0] * data[t - 1] + phi[1] * data[t - 2] + normal.sample(&mut rng);
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();
        let init = css_optimize(&centered, 2, 0).unwrap_or_else(|_| vec![0.0, 0.0]);
        let result = bfgs_optimize(&centered, 2, &init);
        assert!(result.is_ok(), "BFGS failed for AR(2): {:?}", result.err());
        let best = result.unwrap();
        let ar = params::unconstrained_to_coeffs(&best);
        for (i, &expected) in phi.iter().enumerate() {
            assert!(
                (ar[i] - expected).abs() < 0.15,
                "AR(2) phi[{}]: expected ~{}, got {}",
                i,
                expected,
                ar[i]
            );
        }
    }

    #[test]
    fn bfgs_converges_for_arma21() {
        let phi = [0.5, -0.3];
        let theta = 0.4;
        let n = 2000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(500);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut data = vec![0.0; n];
        let mut eps = vec![0.0; n];
        for t in 0..n {
            eps[t] = normal.sample(&mut rng);
            data[t] = eps[t];
            if t >= 1 {
                data[t] += phi[0] * data[t - 1] + theta * eps[t - 1];
            }
            if t >= 2 {
                data[t] += phi[1] * data[t - 2];
            }
        }
        let fit = fit_arma(2, 1, &data);
        assert!(fit.is_ok(), "ARMA(2,1) fit failed: {:?}", fit.err());
    }

    #[test]
    fn css_produces_reasonable_init() {
        let phi = 0.7;
        let n = 1000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(600);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut data = vec![0.0; n];
        for t in 1..n {
            data[t] = phi * data[t - 1] + normal.sample(&mut rng);
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();
        let css_params = css_optimize(&centered, 1, 0).unwrap();
        let ar = params::unconstrained_to_coeffs(&css_params);
        assert!(
            (ar[0] - phi).abs() < 0.25,
            "CSS AR(1) init: expected ~{}, got {}",
            phi,
            ar[0]
        );
    }

    #[test]
    fn css_fallback_on_failure() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(700);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng)).collect();
        let fit = fit_arma(1, 0, &data);
        assert!(fit.is_ok(), "fit_arma failed on white noise as AR(1)");
        let fit = fit.unwrap();
        assert!(
            fit.ar()[0].abs() < 0.15,
            "Expected phi ≈ 0 for white noise, got {}",
            fit.ar()[0]
        );
    }
}
