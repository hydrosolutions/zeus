//! Kalman filter for ARMA likelihood evaluation.
//!
//! Implements a univariate Kalman filter operating on the state-space
//! representation from [`crate::state_space`]. Used internally by
//! [`ArmaSpec::fit()`](crate::ArmaSpec::fit) to evaluate the exact
//! Gaussian log-likelihood via prediction error decomposition.
//!
//! For state dimensions r <= 4 (covering all practical ARMA models),
//! the filter uses stack-allocated fixed-size types from
//! [`crate::small_linalg`] to avoid heap allocation in the hot loop.
//!
//! **Not part of the public API.**

use ndarray::{Array1, Array2};

use crate::error::ArmaError;
use crate::small_linalg::{SmallMat, SmallVec};
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
///
/// For r <= 4, this dispatches to a zero-allocation path that avoids
/// collecting innovations into a `Vec`.
#[allow(dead_code)]
pub(crate) fn kalman_concentrated_loglik(ss: &StateSpace, data: &[f64]) -> Result<f64, ArmaError> {
    let n = data.len() as f64;
    let (sum_ln_f, sum_v2_f) = match ss.r() {
        1 => kalman_loglik_fixed::<1>(ss, data)?,
        2 => kalman_loglik_fixed::<2>(ss, data)?,
        3 => kalman_loglik_fixed::<3>(ss, data)?,
        4 => kalman_loglik_fixed::<4>(ss, data)?,
        _ => {
            let (slf, svf, _) = kalman_core_dynamic(ss, data)?;
            (slf, svf)
        }
    };
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

// ---------------------------------------------------------------------------
// Dispatch: fixed (r<=4) vs dynamic (r>4)
// ---------------------------------------------------------------------------

/// Core Kalman filter recursion with const-generic dispatch.
///
/// Returns (sum_ln_f, sum_v2_over_f, innovations).
fn kalman_core(ss: &StateSpace, data: &[f64]) -> Result<(f64, f64, Vec<f64>), ArmaError> {
    match ss.r() {
        1 => kalman_core_fixed::<1>(ss, data),
        2 => kalman_core_fixed::<2>(ss, data),
        3 => kalman_core_fixed::<3>(ss, data),
        4 => kalman_core_fixed::<4>(ss, data),
        _ => kalman_core_dynamic(ss, data),
    }
}

// ---------------------------------------------------------------------------
// Fixed-size state-space
// ---------------------------------------------------------------------------

/// Fixed-size state-space for the Kalman hot loop.
struct StateSpaceFixed<const R: usize> {
    t: SmallMat<R>,
    rrt: SmallMat<R>,
}

impl<const R: usize> StateSpaceFixed<R> {
    /// Converts from the dynamic ndarray `StateSpace`.
    fn from_dynamic(ss: &StateSpace) -> Self {
        let mut t = SmallMat::zeros();
        let mut rrt = SmallMat::zeros();
        for c in 0..R {
            for r in 0..R {
                t.set(r, c, ss.t()[[r, c]]);
                rrt.set(r, c, ss.rrt()[[r, c]]);
            }
        }
        Self { t, rrt }
    }
}

// ---------------------------------------------------------------------------
// Fixed-size Lyapunov solvers
// ---------------------------------------------------------------------------

/// Solves the discrete Lyapunov equation P = T*P*T' + Q for r=1.
///
/// Analytical formula: `P = q / (1 - t^2)`. Falls back to diffuse
/// initialization if the denominator is near zero.
#[inline(always)]
fn solve_lyapunov_r1(ssf: &StateSpaceFixed<1>) -> SmallMat<1> {
    let t00 = ssf.t.get(0, 0);
    let q00 = ssf.rrt.get(0, 0);
    let denom = 1.0 - t00 * t00;
    let mut p = SmallMat::zeros();
    if denom.abs() < 1e-12 {
        p.set(0, 0, DIFFUSE_KAPPA);
    } else {
        p.set(0, 0, q00 / denom);
    }
    p
}

/// Solves the discrete Lyapunov equation P = T*P*T' + Q for r=2.
///
/// The symmetric 2x2 case yields a 3x3 linear system in (p00, p01, p11).
/// Solved with inline Gaussian elimination; falls back to diffuse if singular.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn solve_lyapunov_r2(ssf: &StateSpaceFixed<2>) -> SmallMat<2> {
    let t00 = ssf.t.get(0, 0);
    let t01 = ssf.t.get(0, 1);
    let t10 = ssf.t.get(1, 0);
    let t11 = ssf.t.get(1, 1);
    let q00 = ssf.rrt.get(0, 0);
    let q01 = ssf.rrt.get(0, 1);
    let q11 = ssf.rrt.get(1, 1);

    // Lyapunov: P = T*P*T' + Q for 2x2 symmetric P gives:
    //   p00 = t00^2*p00 + 2*t00*t01*p01 + t01^2*p11 + q00
    //   p01 = t00*t10*p00 + (t00*t11+t01*t10)*p01 + t01*t11*p11 + q01
    //   p11 = t10^2*p00 + 2*t10*t11*p01 + t11^2*p11 + q11
    //
    // Rearranged as A*[p00, p01, p11]' = [q00, q01, q11]'
    // where A = I - coefficient_matrix

    let mut a = [[0.0_f64; 3]; 3];
    let mut b = [q00, q01, q11];

    // Row 0: (1 - t00^2)*p00 - 2*t00*t01*p01 - t01^2*p11 = q00
    a[0][0] = 1.0 - t00 * t00;
    a[0][1] = -2.0 * t00 * t01;
    a[0][2] = -(t01 * t01);

    // Row 1: -t00*t10*p00 + (1 - t00*t11 - t01*t10)*p01 - t01*t11*p11 = q01
    a[1][0] = -(t00 * t10);
    a[1][1] = 1.0 - (t00 * t11 + t01 * t10);
    a[1][2] = -(t01 * t11);

    // Row 2: -t10^2*p00 - 2*t10*t11*p01 + (1 - t11^2)*p11 = q11
    a[2][0] = -(t10 * t10);
    a[2][1] = -2.0 * t10 * t11;
    a[2][2] = 1.0 - t11 * t11;

    // Gaussian elimination with partial pivoting on 3x3
    let mut perm = [0usize, 1, 2];

    for col in 0..3 {
        // Find pivot
        let mut max_val = a[perm[col]][col].abs();
        let mut max_row = col;
        for row in (col + 1)..3 {
            let val = a[perm[row]][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            // Singular: fall back to diffuse
            let mut p = SmallMat::zeros();
            p.set(0, 0, DIFFUSE_KAPPA);
            p.set(1, 1, DIFFUSE_KAPPA);
            return p;
        }
        perm.swap(col, max_row);

        // Eliminate
        let pivot = a[perm[col]][col];
        for row in (col + 1)..3 {
            let factor = a[perm[row]][col] / pivot;
            a[perm[row]][col] = 0.0;
            for k in (col + 1)..3 {
                a[perm[row]][k] -= factor * a[perm[col]][k];
            }
            b[perm[row]] -= factor * b[perm[col]];
        }
    }

    // Back substitution
    let mut sol = [0.0_f64; 3];
    for i in (0..3).rev() {
        let mut sum = b[perm[i]];
        for j in (i + 1)..3 {
            sum -= a[perm[i]][j] * sol[j];
        }
        sol[i] = sum / a[perm[i]][i];
    }

    let mut p = SmallMat::zeros();
    p.set(0, 0, sol[0]);
    p.set(0, 1, sol[1]);
    p.set(1, 0, sol[1]); // symmetry
    p.set(1, 1, sol[2]);
    p
}

/// Solves the discrete Lyapunov equation P = T*P*T' + Q for r=3 or r=4.
///
/// Uses the Kronecker + LU approach with stack-allocated buffers.
/// Max r^2 = 16, so fixed-size arrays suffice.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn solve_lyapunov_stack<const R: usize>(ssf: &StateSpaceFixed<R>) -> SmallMat<R> {
    const MAX_R2: usize = 16;
    let r2 = R * R;

    // Build I_{r^2} - T kron T as a fixed-size matrix
    let mut lhs = [[0.0_f64; MAX_R2]; MAX_R2];

    // Identity
    for i in 0..r2 {
        lhs[i][i] = 1.0;
    }

    // Subtract T kron T
    for i in 0..R {
        for j in 0..R {
            let t_ij = ssf.t.get(i, j);
            for k in 0..R {
                for l in 0..R {
                    lhs[i * R + k][j * R + l] -= t_ij * ssf.t.get(k, l);
                }
            }
        }
    }

    // vec(Q) in column-major order
    let mut q_vec = [0.0_f64; MAX_R2];
    for col in 0..R {
        for row in 0..R {
            q_vec[col * R + row] = ssf.rrt.get(row, col);
        }
    }

    // LU decomposition with partial pivoting on stack arrays
    let mut perm = [0usize; MAX_R2];
    for i in 0..r2 {
        perm[i] = i;
    }

    let mut max_pivot = 0.0_f64;
    let mut min_pivot = f64::MAX;
    let mut singular = false;

    for col in 0..r2 {
        let mut max_val = lhs[perm[col]][col].abs();
        let mut max_row = col;
        for row in (col + 1)..r2 {
            let val = lhs[perm[row]][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            singular = true;
            break;
        }

        if max_val > max_pivot {
            max_pivot = max_val;
        }
        if max_val < min_pivot {
            min_pivot = max_val;
        }

        perm.swap(col, max_row);

        let pivot = lhs[perm[col]][col];
        for row in (col + 1)..r2 {
            let factor = lhs[perm[row]][col] / pivot;
            lhs[perm[row]][col] = factor; // Store L factor
            for k in (col + 1)..r2 {
                let val = lhs[perm[col]][k];
                lhs[perm[row]][k] -= factor * val;
            }
        }
    }

    if singular || min_pivot <= 0.0 || max_pivot / min_pivot > COND_MAX {
        // Fall back to diffuse
        let mut p = SmallMat::zeros();
        for i in 0..R {
            p.set(i, i, DIFFUSE_KAPPA);
        }
        return p;
    }

    // Forward substitution (L*y = P*b)
    let mut y = [0.0_f64; MAX_R2];
    for i in 0..r2 {
        let mut sum = q_vec[perm[i]];
        for j in 0..i {
            sum -= lhs[perm[i]][j] * y[j];
        }
        y[i] = sum;
    }

    // Back substitution (U*x = y)
    let mut x = [0.0_f64; MAX_R2];
    for i in (0..r2).rev() {
        let mut sum = y[i];
        for j in (i + 1)..r2 {
            sum -= lhs[perm[i]][j] * x[j];
        }
        x[i] = sum / lhs[perm[i]][i];
    }

    // Reshape vec(P) back to SmallMat (column-major)
    let mut p = SmallMat::zeros();
    for col in 0..R {
        for row in 0..R {
            p.set(row, col, x[col * R + row]);
        }
    }
    p
}

/// Dispatches to the appropriate fixed-size Lyapunov solver.
#[inline(always)]
fn solve_lyapunov_fixed<const R: usize>(ssf: &StateSpaceFixed<R>) -> SmallMat<R> {
    // For R=1 and R=2 we have specialized solvers; for R=3,4 use the
    // stack-based Kronecker+LU. The const-generic R is known at compile time
    // so the compiler will eliminate dead branches.
    if R == 1 {
        // SAFETY: We know R==1 at compile time. The compiler will monomorphize
        // this correctly. We use transmute to convert between SmallMat<1> and SmallMat<R>.
        // This is safe because R==1.
        let ssf1 = unsafe { &*(ssf as *const StateSpaceFixed<R>).cast::<StateSpaceFixed<1>>() };
        let p1 = solve_lyapunov_r1(ssf1);
        unsafe { *(&p1 as *const SmallMat<1>).cast::<SmallMat<R>>() }
    } else if R == 2 {
        let ssf2 = unsafe { &*(ssf as *const StateSpaceFixed<R>).cast::<StateSpaceFixed<2>>() };
        let p2 = solve_lyapunov_r2(ssf2);
        unsafe { *(&p2 as *const SmallMat<2>).cast::<SmallMat<R>>() }
    } else {
        solve_lyapunov_stack(ssf)
    }
}

// ---------------------------------------------------------------------------
// Fixed-size Kalman core (collects innovations)
// ---------------------------------------------------------------------------

/// Stack-allocated Kalman filter core for state dimension `R`.
///
/// Returns (sum_ln_f, sum_v2_over_f, innovations).
fn kalman_core_fixed<const R: usize>(
    ss: &StateSpace,
    data: &[f64],
) -> Result<(f64, f64, Vec<f64>), ArmaError> {
    let ssf = StateSpaceFixed::<R>::from_dynamic(ss);
    let n = data.len();

    // Initial state: zero mean
    let mut x = SmallVec::<R>::zeros();
    // Initial covariance: solve Lyapunov equation
    let mut p = solve_lyapunov_fixed(&ssf);

    let mut sum_ln_f = 0.0;
    let mut sum_v2_f = 0.0;
    let mut innovations = Vec::with_capacity(n);

    // Scratch buffers (reused each iteration, on stack)
    let mut p_pred = SmallMat::<R>::zeros();
    let mut k = SmallVec::<R>::zeros();
    let mut temp = SmallMat::<R>::zeros();

    for &y_t in data {
        // Prediction: x_pred = T * x
        let x_pred = ssf.t.mul_vec(&x);

        // Prediction covariance: P_pred = T * P * T' + RRt
        // temp = T * P
        for i in 0..R {
            for j in 0..R {
                let mut sum = 0.0;
                for kk in 0..R {
                    sum += ssf.t.get(i, kk) * p.get(kk, j);
                }
                temp.set(i, j, sum);
            }
        }
        // p_pred = temp * T' + rrt
        for i in 0..R {
            for j in 0..R {
                let mut sum = ssf.rrt.get(i, j);
                for kk in 0..R {
                    sum += temp.get(i, kk) * ssf.t.get(j, kk); // T' means swap indices
                }
                p_pred.set(i, j, sum);
            }
        }

        // Innovation
        let f_t = p_pred.get(0, 0).max(F_MIN);
        let v_t = y_t - x_pred.data[0];

        // Accumulate
        sum_ln_f += f_t.ln();
        sum_v2_f += v_t * v_t / f_t;
        innovations.push(v_t);

        // Kalman gain: K = P_pred[:, 0] / f_t
        for i in 0..R {
            k.data[i] = p_pred.get(i, 0) / f_t;
        }

        // State update: x = x_pred + K * v_t
        for i in 0..R {
            x.data[i] = x_pred.data[i] + k.data[i] * v_t;
        }

        // Covariance update: P = P_pred - K * P_pred[0, :]
        for i in 0..R {
            for j in 0..R {
                p.set(i, j, p_pred.get(i, j) - k.data[i] * p_pred.get(0, j));
            }
        }
    }

    Ok((sum_ln_f, sum_v2_f, innovations))
}

// ---------------------------------------------------------------------------
// Fixed-size Kalman loglik (zero-alloc, no innovations collection)
// ---------------------------------------------------------------------------

/// Zero-allocation Kalman log-likelihood for state dimension `R`.
///
/// Returns only (sum_ln_f, sum_v2_over_f) without collecting innovations,
/// making it truly heap-free for the concentrated log-likelihood hot path.
fn kalman_loglik_fixed<const R: usize>(
    ss: &StateSpace,
    data: &[f64],
) -> Result<(f64, f64), ArmaError> {
    let ssf = StateSpaceFixed::<R>::from_dynamic(ss);

    // Initial state: zero mean
    let mut x = SmallVec::<R>::zeros();
    // Initial covariance: solve Lyapunov equation
    let mut p = solve_lyapunov_fixed(&ssf);

    let mut sum_ln_f = 0.0;
    let mut sum_v2_f = 0.0;

    // Scratch buffers
    let mut p_pred = SmallMat::<R>::zeros();
    let mut k = SmallVec::<R>::zeros();
    let mut temp = SmallMat::<R>::zeros();

    for &y_t in data {
        // Prediction: x_pred = T * x
        let x_pred = ssf.t.mul_vec(&x);

        // Prediction covariance: P_pred = T * P * T' + RRt
        for i in 0..R {
            for j in 0..R {
                let mut sum = 0.0;
                for kk in 0..R {
                    sum += ssf.t.get(i, kk) * p.get(kk, j);
                }
                temp.set(i, j, sum);
            }
        }
        for i in 0..R {
            for j in 0..R {
                let mut sum = ssf.rrt.get(i, j);
                for kk in 0..R {
                    sum += temp.get(i, kk) * ssf.t.get(j, kk);
                }
                p_pred.set(i, j, sum);
            }
        }

        // Innovation
        let f_t = p_pred.get(0, 0).max(F_MIN);
        let v_t = y_t - x_pred.data[0];

        // Accumulate (no innovations Vec)
        sum_ln_f += f_t.ln();
        sum_v2_f += v_t * v_t / f_t;

        // Kalman gain
        for i in 0..R {
            k.data[i] = p_pred.get(i, 0) / f_t;
        }

        // State update
        for i in 0..R {
            x.data[i] = x_pred.data[i] + k.data[i] * v_t;
        }

        // Covariance update
        for i in 0..R {
            for j in 0..R {
                p.set(i, j, p_pred.get(i, j) - k.data[i] * p_pred.get(0, j));
            }
        }
    }

    Ok((sum_ln_f, sum_v2_f))
}

// ---------------------------------------------------------------------------
// Dynamic (ndarray) fallback for r > 4
// ---------------------------------------------------------------------------

/// Core Kalman filter recursion using heap-allocated ndarray types.
///
/// Used as fallback for state dimensions r > 4.
/// Returns (sum_ln_f, sum_v2_over_f, innovations).
fn kalman_core_dynamic(ss: &StateSpace, data: &[f64]) -> Result<(f64, f64, Vec<f64>), ArmaError> {
    let r = ss.r();
    let n = data.len();

    // Initial state: zero mean
    let mut x = Array1::zeros(r);
    // Initial covariance: solve Lyapunov equation (with sigma2=1)
    let mut p = solve_lyapunov_dynamic(ss.t(), ss.rrt());

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
/// via vectorization with Kronecker product (heap-allocated).
///
/// Falls back to diffuse initialization (P0 = kappa*I) if the system
/// is ill-conditioned (near unit root).
fn solve_lyapunov_dynamic(t_mat: &Array2<f64>, q_mat: &Array2<f64>) -> Array2<f64> {
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
        let p0 = solve_lyapunov_dynamic(ss.t(), ss.rrt());
        let expected = 1.0 / (1.0 - 0.25);
        assert_abs_diff_eq!(p0[[0, 0]], expected, epsilon = 1e-10);
    }

    #[test]
    fn lyapunov_ar2_verify() {
        // For AR(2), verify P = T*P*T' + Q (the Lyapunov equation itself)
        let ss = StateSpace::new(&[0.5, -0.3], &[]);
        let p = solve_lyapunov_dynamic(ss.t(), ss.rrt());
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
        let p0 = solve_lyapunov_dynamic(ss.t(), ss.rrt());
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

    // -------------------------------------------------------------------
    // Regression tests: fixed paths must match dynamic fallback exactly
    // -------------------------------------------------------------------

    #[test]
    fn fixed_matches_dynamic_r1() {
        let ss = StateSpace::new(&[0.5], &[]);
        let data = [0.5, -0.3, 1.2, 0.1, -0.8, 0.4, 0.7, -0.2, 0.9, -0.1];
        let (sum_ln_f_fixed, sum_v2_f_fixed, innovations_fixed) =
            kalman_core_fixed::<1>(&ss, &data).unwrap();
        let (sum_ln_f_dyn, sum_v2_f_dyn, innovations_dyn) =
            kalman_core_dynamic(&ss, &data).unwrap();
        assert_abs_diff_eq!(sum_ln_f_fixed, sum_ln_f_dyn, epsilon = 1e-10);
        assert_abs_diff_eq!(sum_v2_f_fixed, sum_v2_f_dyn, epsilon = 1e-10);
        for (a, b) in innovations_fixed.iter().zip(innovations_dyn.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn fixed_matches_dynamic() {
        // ARMA(1,1) -> r=2
        let ss = StateSpace::new(&[0.3], &[0.2]);
        let data = [0.5, -0.3, 1.2, 0.1, -0.8, 0.4, 0.7, -0.2, 0.9, -0.1];
        let (sum_ln_f_fixed, sum_v2_f_fixed, innovations_fixed) =
            kalman_core_fixed::<2>(&ss, &data).unwrap();
        let (sum_ln_f_dyn, sum_v2_f_dyn, innovations_dyn) =
            kalman_core_dynamic(&ss, &data).unwrap();
        assert_abs_diff_eq!(sum_ln_f_fixed, sum_ln_f_dyn, epsilon = 1e-10);
        assert_abs_diff_eq!(sum_v2_f_fixed, sum_v2_f_dyn, epsilon = 1e-10);
        for (a, b) in innovations_fixed.iter().zip(innovations_dyn.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn fixed_matches_dynamic_r3() {
        // ARMA(2,2) -> r=3
        let ss = StateSpace::new(&[0.5, -0.3], &[0.4, 0.2]);
        let data = [0.5, -0.3, 1.2, 0.1, -0.8, 0.4, 0.7, -0.2, 0.9, -0.1];
        let (sum_ln_f_fixed, sum_v2_f_fixed, innovations_fixed) =
            kalman_core_fixed::<3>(&ss, &data).unwrap();
        let (sum_ln_f_dyn, sum_v2_f_dyn, innovations_dyn) =
            kalman_core_dynamic(&ss, &data).unwrap();
        assert_abs_diff_eq!(sum_ln_f_fixed, sum_ln_f_dyn, epsilon = 1e-10);
        assert_abs_diff_eq!(sum_v2_f_fixed, sum_v2_f_dyn, epsilon = 1e-10);
        for (a, b) in innovations_fixed.iter().zip(innovations_dyn.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn loglik_fixed_matches_core_r1() {
        // Verify zero-alloc loglik path matches core path for r=1
        let ss = StateSpace::new(&[0.5], &[]);
        let data = [0.5, -0.3, 1.2, 0.1, -0.8, 0.4, 0.7, -0.2, 0.9, -0.1];
        let (sum_ln_f_core, sum_v2_f_core, _) = kalman_core_fixed::<1>(&ss, &data).unwrap();
        let (sum_ln_f_loglik, sum_v2_f_loglik) = kalman_loglik_fixed::<1>(&ss, &data).unwrap();
        assert_abs_diff_eq!(sum_ln_f_core, sum_ln_f_loglik, epsilon = 1e-14);
        assert_abs_diff_eq!(sum_v2_f_core, sum_v2_f_loglik, epsilon = 1e-14);
    }

    #[test]
    fn loglik_fixed_matches_core_r2() {
        // Verify zero-alloc loglik path matches core path for r=2
        let ss = StateSpace::new(&[0.3], &[0.2]);
        let data = [0.5, -0.3, 1.2, 0.1, -0.8, 0.4, 0.7, -0.2, 0.9, -0.1];
        let (sum_ln_f_core, sum_v2_f_core, _) = kalman_core_fixed::<2>(&ss, &data).unwrap();
        let (sum_ln_f_loglik, sum_v2_f_loglik) = kalman_loglik_fixed::<2>(&ss, &data).unwrap();
        assert_abs_diff_eq!(sum_ln_f_core, sum_ln_f_loglik, epsilon = 1e-14);
        assert_abs_diff_eq!(sum_v2_f_core, sum_v2_f_loglik, epsilon = 1e-14);
    }

    #[test]
    fn lyapunov_fixed_matches_dynamic_r1() {
        let ss = StateSpace::new(&[0.5], &[]);
        let ssf = StateSpaceFixed::<1>::from_dynamic(&ss);
        let p_fixed = solve_lyapunov_fixed(&ssf);
        let p_dyn = solve_lyapunov_dynamic(ss.t(), ss.rrt());
        assert_abs_diff_eq!(p_fixed.get(0, 0), p_dyn[[0, 0]], epsilon = 1e-10);
    }

    #[test]
    fn lyapunov_fixed_matches_dynamic_r2() {
        let ss = StateSpace::new(&[0.5, -0.3], &[]);
        let ssf = StateSpaceFixed::<2>::from_dynamic(&ss);
        let p_fixed = solve_lyapunov_fixed(&ssf);
        let p_dyn = solve_lyapunov_dynamic(ss.t(), ss.rrt());
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(p_fixed.get(i, j), p_dyn[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn lyapunov_fixed_matches_dynamic_r3() {
        let ss = StateSpace::new(&[0.5, -0.3], &[0.4, 0.2]);
        let ssf = StateSpaceFixed::<3>::from_dynamic(&ss);
        let p_fixed = solve_lyapunov_fixed(&ssf);
        let p_dyn = solve_lyapunov_dynamic(ss.t(), ss.rrt());
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(p_fixed.get(i, j), p_dyn[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn fixed_matches_dynamic_r4() {
        // ARMA(3,3) -> r=4
        let ss = StateSpace::new(&[0.3, -0.2, 0.1], &[0.2, 0.1, 0.05]);
        let data = [0.5, -0.3, 1.2, 0.1, -0.8, 0.4, 0.7, -0.2, 0.9, -0.1];
        let (sum_ln_f_fixed, sum_v2_f_fixed, innovations_fixed) =
            kalman_core_fixed::<4>(&ss, &data).unwrap();
        let (sum_ln_f_dyn, sum_v2_f_dyn, innovations_dyn) =
            kalman_core_dynamic(&ss, &data).unwrap();
        assert_abs_diff_eq!(sum_ln_f_fixed, sum_ln_f_dyn, epsilon = 1e-10);
        assert_abs_diff_eq!(sum_v2_f_fixed, sum_v2_f_dyn, epsilon = 1e-10);
        for (a, b) in innovations_fixed.iter().zip(innovations_dyn.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn loglik_fixed_matches_core_r3() {
        let ss = StateSpace::new(&[0.5, -0.3], &[0.4, 0.2]);
        let data = [0.5, -0.3, 1.2, 0.1, -0.8, 0.4, 0.7, -0.2, 0.9, -0.1];
        let (sum_ln_f_core, sum_v2_f_core, _) = kalman_core_fixed::<3>(&ss, &data).unwrap();
        let (sum_ln_f_loglik, sum_v2_f_loglik) = kalman_loglik_fixed::<3>(&ss, &data).unwrap();
        assert_abs_diff_eq!(sum_ln_f_core, sum_ln_f_loglik, epsilon = 1e-14);
        assert_abs_diff_eq!(sum_v2_f_core, sum_v2_f_loglik, epsilon = 1e-14);
    }

    #[test]
    fn loglik_fixed_matches_core_r4() {
        let ss = StateSpace::new(&[0.3, -0.2, 0.1], &[0.2, 0.1, 0.05]);
        let data = [0.5, -0.3, 1.2, 0.1, -0.8, 0.4, 0.7, -0.2, 0.9, -0.1];
        let (sum_ln_f_core, sum_v2_f_core, _) = kalman_core_fixed::<4>(&ss, &data).unwrap();
        let (sum_ln_f_loglik, sum_v2_f_loglik) = kalman_loglik_fixed::<4>(&ss, &data).unwrap();
        assert_abs_diff_eq!(sum_ln_f_core, sum_ln_f_loglik, epsilon = 1e-14);
        assert_abs_diff_eq!(sum_v2_f_core, sum_v2_f_loglik, epsilon = 1e-14);
    }

    #[test]
    fn lyapunov_fixed_matches_dynamic_r4() {
        let ss = StateSpace::new(&[0.3, -0.2, 0.1], &[0.2, 0.1, 0.05]);
        let ssf = StateSpaceFixed::<4>::from_dynamic(&ss);
        let p_fixed = solve_lyapunov_fixed(&ssf);
        let p_dyn = solve_lyapunov_dynamic(ss.t(), ss.rrt());
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(p_fixed.get(i, j), p_dyn[[i, j]], epsilon = 1e-10);
            }
        }
    }
}
