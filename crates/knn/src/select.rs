//! Top-k nearest neighbor selection.

use std::cmp::Ordering;

/// Selects the `k_eff` nearest neighbors from squared distances.
///
/// Uses full sort on (distance, index) pairs — efficient and cache-friendly for
/// the typical 10–200 candidate range. A partial-sort (`select_nth_unstable`)
/// optimisation can be added later if profiling shows it worthwhile.
///
/// Writes results into caller-provided buffers:
/// - `pairs`: scratch buffer for (distance, index) pairs
/// - `nn_indices`: indices of the k nearest neighbors (sorted by ascending distance)
/// - `nn_dists`: Euclidean distances (sqrt of squared distances) of the k nearest neighbors
///
/// # Panics
///
/// Debug-asserts that `k_eff >= 1` and `k_eff <= d2_sq.len()`.
pub(crate) fn select_k_nearest(
    d2_sq: &[f64],
    k_eff: usize,
    pairs: &mut Vec<(f64, usize)>,
    nn_indices: &mut Vec<usize>,
    nn_dists: &mut Vec<f64>,
) {
    debug_assert!(k_eff >= 1);
    debug_assert!(k_eff <= d2_sq.len());

    // Build (distance, original_index) pairs
    pairs.clear();
    pairs.extend(d2_sq.iter().copied().enumerate().map(|(i, d)| (d, i)));

    // Full sort — NaN-safe via Ordering::Equal fallback
    pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    pairs.truncate(k_eff);

    // Extract indices and sqrt distances
    nn_indices.clear();
    nn_dists.clear();
    for &(d2, idx) in pairs.iter() {
        nn_indices.push(idx);
        nn_dists.push(d2.sqrt());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Helper to avoid repeating buffer setup in every test.
    fn run(d2_sq: &[f64], k_eff: usize) -> (Vec<usize>, Vec<f64>) {
        let mut pairs = Vec::new();
        let mut nn_indices = Vec::new();
        let mut nn_dists = Vec::new();
        select_k_nearest(d2_sq, k_eff, &mut pairs, &mut nn_indices, &mut nn_dists);
        (nn_indices, nn_dists)
    }

    #[test]
    fn test_k1_closest() {
        let (indices, dists) = run(&[9.0, 1.0, 4.0], 1);
        assert_eq!(indices, vec![1]);
        assert_abs_diff_eq!(dists[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_k_equals_n_all_sorted() {
        let (indices, dists) = run(&[4.0, 1.0, 9.0, 0.0], 4);
        assert_eq!(indices, vec![3, 1, 0, 2]);
        let expected_dists = [0.0, 1.0, 2.0, 3.0];
        for (got, want) in dists.iter().zip(expected_dists.iter()) {
            assert_abs_diff_eq!(got, want, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_ties() {
        let (indices, dists) = run(&[4.0, 4.0, 1.0], 2);
        // Closest must be index 2 with distance 1.0
        assert_eq!(indices[0], 2);
        assert_abs_diff_eq!(dists[0], 1.0, epsilon = 1e-12);
        // Second element is one of the tied pair (index 0 or 1) with distance 2.0
        assert!(indices[1] == 0 || indices[1] == 1);
        assert_abs_diff_eq!(dists[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sqrt_correctness() {
        let (indices, dists) = run(&[16.0, 25.0], 2);
        assert_eq!(indices, vec![0, 1]);
        assert_abs_diff_eq!(dists[0], 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dists[1], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_single_candidate() {
        let (indices, dists) = run(&[7.0], 1);
        assert_eq!(indices, vec![0]);
        assert_abs_diff_eq!(dists[0], 7.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_buffer_clearing() {
        let mut pairs = Vec::new();
        let mut nn_indices = Vec::new();
        let mut nn_dists = Vec::new();

        // First call
        select_k_nearest(
            &[9.0, 1.0, 4.0],
            2,
            &mut pairs,
            &mut nn_indices,
            &mut nn_dists,
        );
        assert_eq!(nn_indices.len(), 2);

        // Second call with different data — buffers must be cleared internally
        select_k_nearest(&[25.0, 16.0], 1, &mut pairs, &mut nn_indices, &mut nn_dists);
        assert_eq!(nn_indices, vec![1]);
        assert_abs_diff_eq!(nn_dists[0], 4.0, epsilon = 1e-12);
        assert_eq!(nn_dists.len(), 1);
    }
}
