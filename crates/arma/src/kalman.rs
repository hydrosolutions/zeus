//! Kalman filter for ARMA likelihood evaluation.
//!
//! Implements a univariate Kalman filter operating on the state-space
//! representation from [`crate::state_space`]. Used internally by
//! [`ArmaSpec::fit()`](crate::ArmaSpec::fit) to evaluate the exact
//! Gaussian log-likelihood via prediction error decomposition.
//!
//! **Not part of the public API.**

use ndarray::{Array1, Array2};

use crate::error::ArmaError;
use crate::state_space::StateSpace;

const F_MIN: f64 = 1e-12;
const COND_MAX: f64 = 1e12;
const DIFFUSE_KAPPA: f64 = 1e6;

/// Output of a full Kalman filter pass.
#[allow(dead_code)]
pub(crate) struct KalmanOutput {
    /// Full log-likelihood (with estimated sigma2).
    pub(crate) log_likelihood: f64,
    /// MLE estimate of sigma2.
    pub(crate) sigma2: f64,
    /// One-step-ahead prediction residuals.
    pub(crate) residuals: Vec<f64>,
}

/// Computes the concentrated log-likelihood for the optimizer hot loop.
///
/// Runs the Kalman filter with sigma2=1 and returns the concentrated
/// log-likelihood scalar. Called hundreds of times during optimization.
#[allow(dead_code)]
pub(crate) fn kalman_concentrated_loglik(ss: &StateSpace, data: &[f64]) -> Result<f64, ArmaError> {
    let n = data.len() as f64;
    let (sum_ln_f, sum_v2_f, _) = kalman_core(ss, data)?;
    let sigma2_hat = sum_v2_f / n;
    if sigma2_hat <= 0.0 || !sigma2_hat.is_finite() {
        return Err(ArmaError::OptimizationFailed);
    }
    let loglik = -0.5 * n * (2.0_f64 * std::f64::consts::PI).ln()
        - 0.5 * n * sigma2_hat.ln()
        - 0.5 * n
        - 0.5 * sum_ln_f;
    Ok(loglik)
}

/// Full Kalman filter pass: computes sigma2, residuals, and full log-likelihood.
///
/// Used once after optimization converges to extract final results.
#[allow(dead_code)]
pub(crate) fn kalman_full(ss: &StateSpace, data: &[f64]) -> Result<KalmanOutput, ArmaError> {
    let n = data.len() as f64;
    let (sum_ln_f, sum_v2_f, innovations) = kalman_core(ss, data)?;
    let sigma2 = sum_v2_f / n;
    if sigma2 <= 0.0 || !sigma2.is_finite() {
        return Err(ArmaError::OptimizationFailed);
    }
    let log_likelihood = -0.5 * n * (2.0_f64 * std::f64::consts::PI).ln()
        - 0.5 * n * sigma2.ln()
        - 0.5 * n
        - 0.5 * sum_ln_f;
    Ok(KalmanOutput {
        log_likelihood,
        sigma2,
        residuals: innovations,
    })
}

/// Core Kalman filter recursion.
///
/// Runs with sigma2=1. Returns (sum_ln_f, sum_v2_over_f, innovations).
fn kalman_core(ss: &StateSpace, data: &[f64]) -> Result<(f64, f64, Vec<f64>), ArmaError> {
    let r = ss.r();
    let n = data.len();

    // Initial state: zero mean
    let mut x = Array1::zeros(r);
    // Initial covariance: solve Lyapunov equation (with sigma2=1)
    let mut p = solve_lyapunov(ss.t(), ss.rrt());

    let mut sum_ln_f = 0.0;
    let mut sum_v2_f = 0.0;
    let mut innovations = Vec::with_capacity(n);

    for &y_t in data {
        // Prediction step
        // x_pred = T * x
        let x_pred = ss.t().dot(&x);
        // P_pred = T * P * T' + RRt (sigma2=1)
        let p_pred = ss.t().dot(&p).dot(&ss.t().t()) + ss.rrt();

        // Innovation
        let f_t = p_pred[[0, 0]].max(F_MIN);
        let v_t = y_t - x_pred[0];

        // Accumulate
        sum_ln_f += f_t.ln();
        sum_v2_f += v_t * v_t / f_t;
        innovations.push(v_t);

        // Kalman gain: K = P_pred[:, 0] / F_t
        let k: Array1<f64> = p_pred.column(0).mapv(|val| val / f_t);

        // State update: x = x_pred + K * v_t
        x = &x_pred + &k * v_t;

        // Covariance update: P = P_pred - K * P_pred[0, :]
        let p_row0 = p_pred.row(0).to_owned();
        let k_col = k.view().insert_axis(ndarray::Axis(1));
        let p_row0_row = p_row0.view().insert_axis(ndarray::Axis(0));
        p = &p_pred - &k_col.dot(&p_row0_row);
    }

    Ok((sum_ln_f, sum_v2_f, innovations))
}

/// Solves the discrete Lyapunov equation P0 = T*P0*T' + Q
/// via vectorization with Kronecker product.
///
/// Falls back to diffuse initialization (P0 = kappa*I) if the system
/// is ill-conditioned (near unit root).
fn solve_lyapunov(t_mat: &Array2<f64>, q_mat: &Array2<f64>) -> Array2<f64> {
    let r = t_mat.nrows();
    let r2 = r * r;

    // Build I_{r^2} - T kron T
    let kron = kron_product(t_mat, t_mat);
    let mut lhs = Array2::zeros((r2, r2));
    for i in 0..r2 {
        lhs[[i, i]] = 1.0;
    }
    lhs = &lhs - &kron;

    // vec(Q) in column-major order
    let mut q_vec = Array1::zeros(r2);
    for col in 0..r {
        for row in 0..r {
            q_vec[col * r + row] = q_mat[[row, col]];
        }
    }

    // Solve via LU decomposition with partial pivoting
    match lu_solve(&lhs, &q_vec) {
        Some(p_vec) => {
            // Reshape vec(P) back to r x r (column-major)
            let mut p = Array2::zeros((r, r));
            for col in 0..r {
                for row in 0..r {
                    p[[row, col]] = p_vec[col * r + row];
                }
            }
            p
        }
        None => {
            // Ill-conditioned -- fall back to diffuse initialization
            let mut p = Array2::zeros((r, r));
            for i in 0..r {
                p[[i, i]] = DIFFUSE_KAPPA;
            }
            p
        }
    }
}

/// Computes the Kronecker product A (kron) B.
fn kron_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (ar, ac) = (a.nrows(), a.ncols());
    let (br, bc) = (b.nrows(), b.ncols());
    let mut result = Array2::zeros((ar * br, ac * bc));
    for i in 0..ar {
        for j in 0..ac {
            let a_ij = a[[i, j]];
            for k in 0..br {
                for l in 0..bc {
                    result[[i * br + k, j * bc + l]] = a_ij * b[[k, l]];
                }
            }
        }
    }
    result
}

/// LU decomposition with partial pivoting.
///
/// Solves A*x = b. Returns None if the matrix is singular or
/// ill-conditioned (condition estimate > COND_MAX).
fn lu_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());
    assert_eq!(n, b.len());

    // Copy A into working matrix, build permutation
    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();

    let mut max_pivot = 0.0_f64;
    let mut min_pivot = f64::MAX;

    for col in 0..n {
        // Partial pivoting: find max absolute value in column
        let mut max_val = lu[[perm[col], col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = lu[[perm[row], col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // Singular
        }

        // Track pivots for condition estimate
        if max_val > max_pivot {
            max_pivot = max_val;
        }
        if max_val < min_pivot {
            min_pivot = max_val;
        }

        // Swap rows in permutation
        perm.swap(col, max_row);

        // Eliminate below
        let pivot = lu[[perm[col], col]];
        for row in (col + 1)..n {
            let factor = lu[[perm[row], col]] / pivot;
            lu[[perm[row], col]] = factor; // Store L factor
            for k in (col + 1)..n {
                let val = lu[[perm[col], k]];
                lu[[perm[row], k]] -= factor * val;
            }
        }
    }

    // Check condition estimate
    if min_pivot <= 0.0 || max_pivot / min_pivot > COND_MAX {
        return None;
    }

    // Forward substitution (L*y = P*b)
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[perm[i]];
        for j in 0..i {
            sum -= lu[[perm[i], j]] * y[j];
        }
        y[i] = sum;
    }

    // Back substitution (U*x = y)
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= lu[[perm[i], j]] * x[j];
        }
        x[i] = sum / lu[[perm[i], i]];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_space::StateSpace;
    use approx::assert_abs_diff_eq;

    #[test]
    fn lyapunov_ar1_analytical() {
        // For AR(1) with phi=0.5, sigma2=1:
        // P0 = sigma2 / (1 - phi^2) = 1 / (1 - 0.25) = 4/3
        let ss = StateSpace::new(&[0.5], &[]);
        let p0 = solve_lyapunov(ss.t(), ss.rrt());
        let expected = 1.0 / (1.0 - 0.25);
        assert_abs_diff_eq!(p0[[0, 0]], expected, epsilon = 1e-10);
    }

    #[test]
    fn lyapunov_ar2_verify() {
        // For AR(2), verify P = T*P*T' + Q (the Lyapunov equation itself)
        let ss = StateSpace::new(&[0.5, -0.3], &[]);
        let p = solve_lyapunov(ss.t(), ss.rrt());
        let tpt = ss.t().dot(&p).dot(&ss.t().t());
        let result = &tpt + ss.rrt();
        for i in 0..ss.r() {
            for j in 0..ss.r() {
                assert_abs_diff_eq!(p[[i, j]], result[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn lyapunov_near_unit_root_diffuse() {
        // phi very close to 1.0 should trigger diffuse fallback
        let ss = StateSpace::new(&[0.9999999999], &[]);
        let p0 = solve_lyapunov(ss.t(), ss.rrt());
        // Should be diffuse: P0 = DIFFUSE_KAPPA * I
        // OR very large value from solving (both are acceptable)
        assert!(
            p0[[0, 0]] > 1e4,
            "P0 should be very large for near-unit-root"
        );
    }

    #[test]
    fn lu_solve_identity() {
        let a = Array2::eye(3);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let x = lu_solve(&a, &b).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(x[i], b[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn lu_solve_known_3x3() {
        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]  b = [1, 2, 3]
        // Use: A*x = b, solve and verify A*result ~ b
        let a = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
            .unwrap();
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let x = lu_solve(&a, &b).unwrap();
        let ax = a.dot(&x);
        for i in 0..3 {
            assert_abs_diff_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn lu_solve_singular() {
        // Singular matrix: second row = 2 * first row
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let b = Array1::from_vec(vec![1.0, 2.0]);
        assert!(lu_solve(&a, &b).is_none());
    }

    #[test]
    fn white_noise_kalman() {
        // ARMA(0,0): white noise, sigma2=1
        // Log-likelihood of N(0,1) for data [0.5, -0.3, 1.2]
        let ss = StateSpace::new(&[], &[]);
        let data = [0.5, -0.3, 1.2];
        let output = kalman_full(&ss, &data).unwrap();
        assert!(output.sigma2 > 0.0);
        assert!(output.log_likelihood.is_finite());
        assert_eq!(output.residuals.len(), 3);
    }

    #[test]
    fn ar1_kalman() {
        // Generate AR(1) data and verify Kalman runs without error
        let ss = StateSpace::new(&[0.5], &[]);
        // Simple synthetic data
        let data: Vec<f64> = {
            let mut y = vec![0.0; 100];
            let mut rng = 42u64;
            for t in 1..100 {
                // Simple LCG pseudo-random
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (rng >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
                y[t] = 0.5 * y[t - 1] + u * 0.5;
            }
            y
        };
        let output = kalman_full(&ss, &data).unwrap();
        assert!(output.sigma2 > 0.0);
        assert!(output.log_likelihood.is_finite());
        assert_eq!(output.residuals.len(), 100);
    }

    #[test]
    fn concentrated_matches_full() {
        // The concentrated and full log-likelihood should match
        let ss = StateSpace::new(&[0.3], &[0.2]);
        let data = [0.5, -0.3, 1.2, 0.1, -0.8, 0.4, 0.7, -0.2, 0.9, -0.1];
        let conc = kalman_concentrated_loglik(&ss, &data).unwrap();
        let full = kalman_full(&ss, &data).unwrap();
        assert_abs_diff_eq!(conc, full.log_likelihood, epsilon = 1e-10);
    }

    #[test]
    fn tiny_series() {
        // Series of length 1
        let ss = StateSpace::new(&[0.5], &[]);
        let data = [1.0];
        let output = kalman_full(&ss, &data).unwrap();
        assert!(output.sigma2 > 0.0);
        assert_eq!(output.residuals.len(), 1);
        assert_abs_diff_eq!(output.residuals[0], 1.0, epsilon = 1e-10);
    }
}
