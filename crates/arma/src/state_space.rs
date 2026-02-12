//! ARMA state-space representation.
//!
//! Converts ARMA(p,q) coefficients into state-space form:
//!
//! ```text
//! x[t+1] = T * x[t] + R * e[t]     (state transition)
//! y[t]   = Z' * x[t]                (observation)
//! ```
//!
//! where `T` is the transition matrix, `R` the noise-input vector,
//! `Z` the observation vector, and `e[t] ~ N(0, sigma2)`.
//!
//! **Not part of the public API.**

use ndarray::{Array1, Array2, Axis};

/// State-space representation of an ARMA(p,q) model.
///
/// Holds the transition matrix `T`, noise input vector `R`, and
/// precomputed `R·Rᵀ` for the Kalman filter.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct StateSpace {
    r: usize,
    t: Array2<f64>,
    r_vec: Array1<f64>,
    rrt: Array2<f64>,
}

#[allow(dead_code)]
impl StateSpace {
    /// Builds a state-space representation from AR and MA coefficients.
    ///
    /// # Panics
    ///
    /// Does not panic — validation happens upstream in `ArmaSpec`.
    pub(crate) fn new(ar: &[f64], ma: &[f64]) -> Self {
        let p = ar.len();
        let q = ma.len();
        let r = p.max(q + 1).max(1);

        // Build transition matrix T (r×r) in companion form.
        // First column: T[i, 0] = ar[i] for i < p.
        // Super-diagonal: T[i, i+1] = 1.0 for i = 0..r-2.
        let mut t = Array2::zeros((r, r));
        for i in 0..p {
            t[[i, 0]] = ar[i];
        }
        for i in 0..r.saturating_sub(1) {
            t[[i, i + 1]] = 1.0;
        }

        // Build noise input vector R (length r).
        // R[0] = 1.0, R[j+1] = ma[j] for j = 0..q-1.
        let mut r_vec = Array1::zeros(r);
        r_vec[0] = 1.0;
        for j in 0..q {
            r_vec[j + 1] = ma[j];
        }

        // Compute RRᵀ as the outer product of R with itself.
        let r_col = r_vec.view().insert_axis(Axis(1)); // (r, 1)
        let r_row = r_vec.view().insert_axis(Axis(0)); // (1, r)
        let rrt = r_col.dot(&r_row);

        Self { r, t, r_vec, rrt }
    }

    /// State dimension `r = max(p, q+1)`, minimum 1.
    pub(crate) fn r(&self) -> usize {
        self.r
    }

    /// Transition matrix `T` (r×r) in companion form.
    pub(crate) fn t(&self) -> &Array2<f64> {
        &self.t
    }

    /// Noise input vector `R = [1, θ₁, …, θ_q, 0, …, 0]` (length r).
    pub(crate) fn r_vec(&self) -> &Array1<f64> {
        &self.r_vec
    }

    /// Precomputed `R·Rᵀ` (r×r), the rank-1 outer product.
    pub(crate) fn rrt(&self) -> &Array2<f64> {
        &self.rrt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn ar1() {
        let ss = StateSpace::new(&[0.5], &[]);
        assert_eq!(ss.r(), 1);
        assert_abs_diff_eq!(ss.t()[[0, 0]], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.r_vec()[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.rrt()[[0, 0]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn ma1() {
        let ss = StateSpace::new(&[], &[0.8]);
        assert_eq!(ss.r(), 2);
        assert_eq!(ss.t().shape(), &[2, 2]);
        assert_eq!(ss.r_vec().len(), 2);

        // T = [[0, 1], [0, 0]]
        assert_abs_diff_eq!(ss.t()[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[0, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 1]], 0.0, epsilon = 1e-12);

        // R = [1.0, 0.8]
        assert_abs_diff_eq!(ss.r_vec()[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.r_vec()[1], 0.8, epsilon = 1e-12);
    }

    #[test]
    fn arma11() {
        let ss = StateSpace::new(&[0.7], &[0.3]);
        assert_eq!(ss.r(), 2);

        // T = [[0.7, 1.0], [0.0, 0.0]]
        assert_abs_diff_eq!(ss.t()[[0, 0]], 0.7, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[0, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 1]], 0.0, epsilon = 1e-12);

        // R = [1.0, 0.3]
        assert_abs_diff_eq!(ss.r_vec()[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.r_vec()[1], 0.3, epsilon = 1e-12);
    }

    #[test]
    fn arma21() {
        let ss = StateSpace::new(&[0.5, -0.3], &[0.4]);
        assert_eq!(ss.r(), 2);

        // T = [[0.5, 1.0], [-0.3, 0.0]]
        assert_abs_diff_eq!(ss.t()[[0, 0]], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[0, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 0]], -0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 1]], 0.0, epsilon = 1e-12);

        // R = [1.0, 0.4]
        assert_abs_diff_eq!(ss.r_vec()[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.r_vec()[1], 0.4, epsilon = 1e-12);
    }

    #[test]
    fn arma22() {
        let ss = StateSpace::new(&[0.5, -0.3], &[0.4, 0.2]);
        assert_eq!(ss.r(), 3);

        // T = [[0.5, 1, 0], [-0.3, 0, 1], [0, 0, 0]]
        assert_abs_diff_eq!(ss.t()[[0, 0]], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[0, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[0, 2]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 0]], -0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 2]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[2, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[2, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[2, 2]], 0.0, epsilon = 1e-12);

        // R = [1.0, 0.4, 0.2]
        assert_abs_diff_eq!(ss.r_vec()[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.r_vec()[1], 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.r_vec()[2], 0.2, epsilon = 1e-12);
    }

    #[test]
    fn ar2() {
        let ss = StateSpace::new(&[0.6, -0.2], &[]);
        assert_eq!(ss.r(), 2);

        // T = [[0.6, 1.0], [-0.2, 0.0]]
        assert_abs_diff_eq!(ss.t()[[0, 0]], 0.6, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[0, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 0]], -0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.t()[[1, 1]], 0.0, epsilon = 1e-12);

        // R = [1.0, 0.0]
        assert_abs_diff_eq!(ss.r_vec()[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.r_vec()[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn arma00() {
        let ss = StateSpace::new(&[], &[]);
        assert_eq!(ss.r(), 1);
        assert_abs_diff_eq!(ss.t()[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.r_vec()[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ss.rrt()[[0, 0]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn rrt_symmetry() {
        let ss = StateSpace::new(&[0.7], &[0.3]);
        let rrt = ss.rrt();
        let rrt_t = rrt.t();
        for i in 0..ss.r() {
            for j in 0..ss.r() {
                assert_abs_diff_eq!(rrt[[i, j]], rrt_t[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn rrt_rank1() {
        let ss = StateSpace::new(&[0.5, -0.3], &[0.4]);
        let rrt = ss.rrt();
        let r_vec = ss.r_vec();
        for i in 0..ss.r() {
            for j in 0..ss.r() {
                assert_abs_diff_eq!(rrt[[i, j]], r_vec[i] * r_vec[j], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn clone_trait() {
        let ss = StateSpace::new(&[0.7], &[0.3]);
        let ss2 = ss.clone();
        assert_eq!(ss.r(), ss2.r());
        assert_eq!(ss.t(), ss2.t());
        assert_eq!(ss.r_vec(), ss2.r_vec());
        assert_eq!(ss.rrt(), ss2.rrt());
    }

    #[test]
    fn send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StateSpace>();
    }
}
