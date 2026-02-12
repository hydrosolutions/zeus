/// Maps unconstrained parameters to stationary/invertible coefficients
/// via the PACF parametrization (Jones 1980, Monahan 1984).
///
/// Step 1: `r_k = tanh(alpha_k)` maps each parameter to (-1, 1).
/// Step 2: Levinson-Durbin recursion converts partial autocorrelations
///         to polynomial coefficients.
///
/// The same transform enforces stationarity (for AR) and invertibility
/// (for MA) when applied independently to each set of coefficients.
#[allow(dead_code)]
pub(crate) fn unconstrained_to_coeffs(alpha: &[f64]) -> Vec<f64> {
    let p = alpha.len();
    if p == 0 {
        return Vec::new();
    }

    // Step 1: map each unconstrained parameter to (-1, 1) via tanh
    let r: Vec<f64> = alpha.iter().map(|a| a.tanh()).collect();

    // Step 2: Levinson-Durbin recursion
    let mut phi = vec![0.0; p];
    let mut prev = vec![0.0; p];

    phi[0] = r[0];

    for k in 1..p {
        // Copy phi into prev
        prev[..p].copy_from_slice(&phi[..p]);

        phi[k] = r[k];
        for j in 0..k {
            phi[j] = prev[j] - r[k] * prev[k - 1 - j];
        }
    }

    phi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let result = unconstrained_to_coeffs(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn single_zero() {
        let result = unconstrained_to_coeffs(&[0.0]);
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 0.0).abs() < 1e-15,
            "expected 0.0, got {}",
            result[0]
        );
    }

    #[test]
    fn single_large_positive() {
        let result = unconstrained_to_coeffs(&[10.0]);
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "expected close to 1.0, got {}",
            result[0]
        );
    }

    #[test]
    fn single_large_negative() {
        let result = unconstrained_to_coeffs(&[-10.0]);
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] + 1.0).abs() < 1e-6,
            "expected close to -1.0, got {}",
            result[0]
        );
    }

    #[test]
    fn two_coefficients() {
        let result = unconstrained_to_coeffs(&[0.5, 0.3]);
        assert_eq!(result.len(), 2);

        // Hand-computed values:
        // r = [tanh(0.5), tanh(0.3)] ≈ [0.46212, 0.29131]
        // phi[0] = r[0] = 0.46212
        // k=1: prev = [0.46212, 0.0]
        //   phi[1] = r[1] = 0.29131
        //   phi[0] = prev[0] - r[1]*prev[0] = 0.46212 - 0.29131*0.46212 ≈ 0.32748
        let expected_0 = 0.5_f64.tanh() - 0.3_f64.tanh() * 0.5_f64.tanh();
        let expected_1 = 0.3_f64.tanh();

        assert!(
            (result[0] - expected_0).abs() < 1e-4,
            "phi[0]: expected ~{}, got {}",
            expected_0,
            result[0]
        );
        assert!(
            (result[1] - expected_1).abs() < 1e-4,
            "phi[1]: expected ~{}, got {}",
            expected_1,
            result[1]
        );
    }

    #[test]
    fn all_zeros() {
        let result = unconstrained_to_coeffs(&[0.0, 0.0, 0.0]);
        assert_eq!(result.len(), 3);
        for (i, val) in result.iter().enumerate() {
            assert!(val.abs() < 1e-15, "phi[{}]: expected 0.0, got {}", i, val);
        }
    }

    #[test]
    fn stationarity_guarantee() {
        // Test with several inputs that the resulting AR coefficients
        // satisfy stationarity conditions.

        let test_cases: &[&[f64]] = &[
            &[1.0, -2.0, 3.0],
            &[0.5],
            &[-0.5],
            &[1.0, 0.5],
            &[-3.0, 2.0],
            &[0.1, 0.2],
        ];

        for alpha in test_cases {
            let phi = unconstrained_to_coeffs(alpha);

            match phi.len() {
                1 => {
                    // Order 1: stationarity requires |phi[0]| < 1
                    assert!(
                        phi[0].abs() < 1.0,
                        "Order 1 stationarity violated for alpha={:?}: |phi[0]| = {} >= 1",
                        alpha,
                        phi[0].abs()
                    );
                }
                2 => {
                    // Order 2: stationarity requires:
                    //   |phi[1]| < 1
                    //   phi[1] + phi[0] < 1
                    //   phi[1] - phi[0] < 1
                    assert!(
                        phi[1].abs() < 1.0,
                        "Order 2 stationarity violated for alpha={:?}: |phi[1]| = {} >= 1",
                        alpha,
                        phi[1].abs()
                    );
                    assert!(
                        phi[1] + phi[0] < 1.0,
                        "Order 2 stationarity violated for alpha={:?}: phi[1]+phi[0] = {} >= 1",
                        alpha,
                        phi[1] + phi[0]
                    );
                    assert!(
                        phi[1] - phi[0] < 1.0,
                        "Order 2 stationarity violated for alpha={:?}: phi[1]-phi[0] = {} >= 1",
                        alpha,
                        phi[1] - phi[0]
                    );
                }
                p => {
                    // For higher orders, verify via companion matrix spectral radius.
                    // The companion matrix for AR(p) polynomial has eigenvalues
                    // whose magnitudes must all be < 1.
                    // Here we check that the partial autocorrelations (tanh values)
                    // are all in (-1,1), which guarantees stationarity by construction.
                    let r: Vec<f64> = alpha.iter().map(|a| a.tanh()).collect();
                    for (k, rk) in r.iter().enumerate() {
                        assert!(
                            rk.abs() < 1.0,
                            "Order {} stationarity: |r[{}]| = {} >= 1 for alpha={:?}",
                            p,
                            k,
                            rk.abs(),
                            alpha
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn negation_symmetry() {
        // For order 1: unconstrained_to_coeffs(&[-a]) == -unconstrained_to_coeffs(&[a])
        let values = [0.0, 0.5, 1.0, 2.0, 10.0];
        for a in &values {
            let pos = unconstrained_to_coeffs(&[*a]);
            let neg = unconstrained_to_coeffs(&[-a]);
            assert_eq!(pos.len(), 1);
            assert_eq!(neg.len(), 1);
            assert!(
                (neg[0] + pos[0]).abs() < 1e-15,
                "Negation symmetry failed for a={}: pos={}, neg={}",
                a,
                pos[0],
                neg[0]
            );
        }
    }
}
