//! BFGS optimizer with Nelder-Mead fallback for ARMA maximum-likelihood estimation.
//!
//! Wraps the `argmin` crate to minimize the negative concentrated
//! log-likelihood over unconstrained PACF parameters.  Gradients are
//! approximated via forward finite differences.
//!
//! **Not part of the public API.**

use std::cell::RefCell;

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

/// Pre-allocated scratch buffers for CSS optimization.
struct CssScratch {
    ar: Vec<f64>,
    ma: Vec<f64>,
    resid: Vec<f64>,
}

/// Argmin wrapper for CSS optimization via Nelder-Mead.
///
/// Uses [`RefCell`]-wrapped scratch buffers to eliminate heap allocations
/// during the Nelder-Mead optimization loop.
struct CssCost<'a> {
    data: &'a [f64],
    p: usize,
    q: usize,
    scratch: RefCell<CssScratch>,
}

impl<'a> CssCost<'a> {
    fn new(data: &'a [f64], p: usize, q: usize) -> Self {
        Self {
            data,
            p,
            q,
            scratch: RefCell::new(CssScratch {
                ar: vec![0.0; p],
                ma: vec![0.0; q],
                resid: vec![0.0; data.len()],
            }),
        }
    }
}

impl CostFunction for CssCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut scratch = self.scratch.borrow_mut();
        let (alpha, beta) = params.split_at(self.p);
        params::unconstrained_to_coeffs_buf(alpha, &mut scratch.ar);
        params::unconstrained_to_coeffs_buf(beta, &mut scratch.ma);
        if scratch
            .ar
            .iter()
            .chain(scratch.ma.iter())
            .any(|c| !c.is_finite())
        {
            return Ok(PENALTY_COST);
        }
        let n = self.data.len();
        let ncond = self.p.max(self.q);
        // Zero residuals
        for e in scratch.resid.iter_mut() {
            *e = 0.0;
        }
        for t in 0..n {
            let mut e_t = self.data[t];
            for i in 0..self.p {
                if t > i {
                    e_t -= scratch.ar[i] * self.data[t - 1 - i];
                }
            }
            for j in 0..self.q {
                if t > j {
                    e_t -= scratch.ma[j] * scratch.resid[t - 1 - j];
                }
            }
            scratch.resid[t] = e_t;
        }
        let effective_n = n - ncond;
        if effective_n == 0 {
            return Ok(PENALTY_COST);
        }
        let msr: f64 =
            scratch.resid[ncond..].iter().map(|e| e * e).sum::<f64>() / effective_n as f64;
        if msr <= 0.0 || !msr.is_finite() {
            return Ok(PENALTY_COST);
        }
        Ok(0.5 * msr.ln())
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
    let cost = CssCost::new(data, p, q);
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
    let q = init.len() - p;
    let cost = ArmaCost::new(data, p, q);
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
    let q = dim - p;
    let cost = ArmaCost::new(data, p, q);
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

/// Pre-allocated scratch buffers for the cost/gradient hot loop.
struct CostScratch {
    ar: Vec<f64>,
    ma: Vec<f64>,
    ss: StateSpace,
}

/// Cost function for argmin: negative concentrated log-likelihood.
///
/// Uses [`RefCell`]-wrapped scratch buffers to eliminate heap allocations
/// during the BFGS optimization hot loop.
struct ArmaCost<'a> {
    data: &'a [f64],
    p: usize,
    scratch: RefCell<CostScratch>,
}

impl<'a> ArmaCost<'a> {
    fn new(data: &'a [f64], p: usize, q: usize) -> Self {
        let r = p.max(q + 1).max(1);
        Self {
            data,
            p,
            scratch: RefCell::new(CostScratch {
                ar: vec![0.0; p],
                ma: vec![0.0; q],
                ss: StateSpace::new_zeroed(r),
            }),
        }
    }
}

impl CostFunction for ArmaCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut scratch = self.scratch.borrow_mut();
        let (alpha, beta) = params.split_at(self.p);
        let CostScratch { ar, ma, ss } = &mut *scratch;
        params::unconstrained_to_coeffs_buf(alpha, ar);
        params::unconstrained_to_coeffs_buf(beta, ma);

        if ar.iter().chain(ma.iter()).any(|c| !c.is_finite()) {
            return Ok(PENALTY_COST);
        }

        ss.update(ar, ma);

        match kalman::kalman_concentrated_loglik(ss, self.data) {
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
        let mut perturbed = params.clone();

        for i in 0..dim {
            perturbed[i] = params[i] + FD_STEP;
            let f_plus = self.cost(&perturbed)?;
            perturbed[i] = params[i]; // restore

            if f_plus < PENALTY_COST {
                grad[i] = (f_plus - f_center) / FD_STEP;
            } else {
                // Rare fallback: backward difference
                perturbed[i] = params[i] - FD_STEP;
                let f_minus = self.cost(&perturbed)?;
                perturbed[i] = params[i]; // restore
                grad[i] = if f_minus < PENALTY_COST {
                    (f_center - f_minus) / FD_STEP
                } else {
                    0.0
                };
            }
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
        let cost = ArmaCost::new(&data, 1, 1);
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
