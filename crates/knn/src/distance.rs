//! Weighted squared Euclidean distance computation.

/// Computes weighted squared Euclidean distances from `target` to all candidates.
///
/// For each candidate row `i`:
/// ```text
/// out[i] = Σⱼ weights[j] × (candidates[i × n_vars + j] − target[j])²
/// ```
///
/// Dispatches to specialised implementations for 1D and 2D cases.
///
/// # Panics
///
/// Debug-asserts that `candidates.len() % n_vars == 0`, `target.len() == n_vars`,
/// `weights.len() == n_vars`, and `out.len() == candidates.len() / n_vars`.
pub(crate) fn weighted_sq_distances(
    candidates: &[f64],
    n_vars: usize,
    target: &[f64],
    weights: &[f64],
    out: &mut [f64],
) {
    let n_candidates = candidates.len() / n_vars;
    debug_assert_eq!(candidates.len() % n_vars, 0);
    debug_assert_eq!(target.len(), n_vars);
    debug_assert_eq!(weights.len(), n_vars);
    debug_assert_eq!(out.len(), n_candidates);

    match n_vars {
        1 => weighted_sq_dist_1d(candidates, target[0], weights[0], out),
        2 => weighted_sq_dist_2d(candidates, target, weights, out),
        _ => weighted_sq_dist_nd(candidates, n_vars, target, weights, out),
    }
}

#[inline]
fn weighted_sq_dist_1d(candidates: &[f64], target: f64, weight: f64, out: &mut [f64]) {
    for (o, &c) in out.iter_mut().zip(candidates.iter()) {
        let d = c - target;
        *o = weight * d * d;
    }
}

#[inline]
fn weighted_sq_dist_2d(candidates: &[f64], target: &[f64], weights: &[f64], out: &mut [f64]) {
    let t0 = target[0];
    let t1 = target[1];
    let w0 = weights[0];
    let w1 = weights[1];
    for (i, o) in out.iter_mut().enumerate() {
        let d0 = candidates[i * 2] - t0;
        let d1 = candidates[i * 2 + 1] - t1;
        *o = w0 * d0 * d0 + w1 * d1 * d1;
    }
}

#[inline]
fn weighted_sq_dist_nd(
    candidates: &[f64],
    n_vars: usize,
    target: &[f64],
    weights: &[f64],
    out: &mut [f64],
) {
    for (i, o) in out.iter_mut().enumerate() {
        let row = &candidates[i * n_vars..(i + 1) * n_vars];
        let mut acc = 0.0;
        for j in 0..n_vars {
            let d = row[j] - target[j];
            acc += weights[j] * d * d;
        }
        *o = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_1d_hand_computed() {
        let candidates = [1.0, 3.0, 5.0];
        let target = [2.0];
        let weights = [1.0];
        let mut out = [0.0; 3];
        weighted_sq_distances(&candidates, 1, &target, &weights, &mut out);
        assert_abs_diff_eq!(out[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[2], 9.0, epsilon = 1e-12);
    }

    #[test]
    fn test_2d_matches_nd() {
        // 5 candidates in 2D
        let candidates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let target = [2.5, 3.5];
        let weights = [1.3, 0.7];

        // Compute via dispatch (routes to 2D specialisation)
        let mut out_2d = [0.0; 5];
        weighted_sq_distances(&candidates, 2, &target, &weights, &mut out_2d);

        // Compute directly via ND path
        let mut out_nd = [0.0; 5];
        weighted_sq_dist_nd(&candidates, 2, &target, &weights, &mut out_nd);

        for i in 0..5 {
            assert_abs_diff_eq!(out_2d[i], out_nd[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_uniform_weights_is_euclidean() {
        // 3D, weights all 1.0 => plain squared Euclidean distance
        let candidates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let target = [0.0, 0.0, 0.0];
        let weights = [1.0, 1.0, 1.0];
        let mut out = [0.0; 2];
        weighted_sq_distances(&candidates, 3, &target, &weights, &mut out);

        // candidate 0: (1-0)^2 + (2-0)^2 + (3-0)^2 = 1 + 4 + 9 = 14
        assert_abs_diff_eq!(out[0], 14.0, epsilon = 1e-12);
        // candidate 1: (4-0)^2 + (5-0)^2 + (6-0)^2 = 16 + 25 + 36 = 77
        assert_abs_diff_eq!(out[1], 77.0, epsilon = 1e-12);
    }

    #[test]
    fn test_weights_effect() {
        // 2D, weights=[2.0, 1.0]
        // candidates: (1,0), (0,1), target=(0,0)
        // expected: [2*1^2 + 1*0^2, 2*0^2 + 1*1^2] = [2.0, 1.0]
        let candidates = [1.0, 0.0, 0.0, 1.0];
        let target = [0.0, 0.0];
        let weights = [2.0, 1.0];
        let mut out = [0.0; 2];
        weighted_sq_distances(&candidates, 2, &target, &weights, &mut out);
        assert_abs_diff_eq!(out[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_zero_distance() {
        // Target matches a candidate exactly => distance = 0.0
        let candidates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let target = [4.0, 5.0, 6.0];
        let weights = [1.5, 2.5, 3.5];
        let mut out = [0.0; 2];
        weighted_sq_distances(&candidates, 3, &target, &weights, &mut out);
        assert_abs_diff_eq!(out[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_single_candidate() {
        let candidates = [3.0, 7.0];
        let target = [1.0, 2.0];
        let weights = [1.0, 1.0];
        let mut out = [0.0; 1];
        weighted_sq_distances(&candidates, 2, &target, &weights, &mut out);
        // (3-1)^2 + (7-2)^2 = 4 + 25 = 29
        assert_abs_diff_eq!(out[0], 29.0, epsilon = 1e-12);
    }

    #[test]
    fn test_dispatch_routing() {
        // n_vars=1 routes to 1D
        {
            let candidates = [10.0];
            let target = [7.0];
            let weights = [2.0];
            let mut out = [0.0; 1];
            weighted_sq_distances(&candidates, 1, &target, &weights, &mut out);
            // 2*(10-7)^2 = 2*9 = 18
            assert_abs_diff_eq!(out[0], 18.0, epsilon = 1e-12);
        }
        // n_vars=2 routes to 2D
        {
            let candidates = [1.0, 2.0, 3.0, 4.0];
            let target = [0.0, 0.0];
            let weights = [1.0, 1.0];
            let mut out = [0.0; 2];
            weighted_sq_distances(&candidates, 2, &target, &weights, &mut out);
            // candidate 0: 1+4=5, candidate 1: 9+16=25
            assert_abs_diff_eq!(out[0], 5.0, epsilon = 1e-12);
            assert_abs_diff_eq!(out[1], 25.0, epsilon = 1e-12);
        }
        // n_vars=4 routes to ND
        {
            let candidates = [1.0, 1.0, 1.0, 1.0];
            let target = [0.0, 0.0, 0.0, 0.0];
            let weights = [1.0, 1.0, 1.0, 1.0];
            let mut out = [0.0; 1];
            weighted_sq_distances(&candidates, 4, &target, &weights, &mut out);
            // 1+1+1+1 = 4
            assert_abs_diff_eq!(out[0], 4.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_50_candidates() {
        let n_vars = 2;
        let n_candidates = 50;
        // candidates: (i as f64, i as f64 * 2) for i in 0..50
        let mut candidates = vec![0.0; n_candidates * n_vars];
        for i in 0..n_candidates {
            candidates[i * 2] = i as f64;
            candidates[i * 2 + 1] = (i as f64) * 2.0;
        }
        let target = [0.0, 0.0];
        let weights = [1.0, 1.0];
        let mut out = vec![0.0; n_candidates];
        weighted_sq_distances(&candidates, n_vars, &target, &weights, &mut out);

        assert_eq!(out.len(), 50);

        // Spot-check candidate 10: (10, 20), target (0,0)
        // dist = 10^2 + 20^2 = 100 + 400 = 500
        assert_abs_diff_eq!(out[10], 500.0, epsilon = 1e-12);

        // Spot-check candidate 0: (0, 0) => 0
        assert_abs_diff_eq!(out[0], 0.0, epsilon = 1e-12);
    }
}
