//! KNN sampling entry point and scratch buffer management.

use rand::Rng;

use crate::config::KnnConfig;
use crate::distance::weighted_sq_distances;
use crate::error::KnnError;
use crate::result::KnnResult;
use crate::sample::{compute_probs, weighted_sample};
use crate::select::select_k_nearest;

/// Pre-allocated scratch buffers for KNN sampling.
///
/// Reuse across multiple calls to [`knn_sample_with_scratch`] to avoid
/// repeated heap allocation in hot loops (e.g. 365 daily KNN calls per year).
///
/// # Example
///
/// ```
/// use zeus_knn::KnnScratch;
///
/// let mut scratch = KnnScratch::new(200);
/// // Use with knn_sample_with_scratch() in a loop...
/// ```
#[derive(Debug, Clone)]
pub struct KnnScratch {
    /// Squared distances buffer.
    pub(crate) d2_sq: Vec<f64>,
    /// (distance, index) pairs for sorting.
    pub(crate) pairs: Vec<(f64, usize)>,
    /// Indices of k nearest neighbors.
    pub(crate) nn_indices: Vec<usize>,
    /// Euclidean distances of k nearest neighbors.
    pub(crate) nn_dists: Vec<f64>,
    /// Sampling probabilities.
    pub(crate) probs: Vec<f64>,
    /// CDF buffer for weighted sampling.
    pub(crate) cdf: Vec<f64>,
    /// Sampled local indices (before mapping back to candidate indices).
    pub(crate) sampled: Vec<usize>,
}

impl KnnScratch {
    /// Creates a new scratch buffer with capacity for `max_candidates` candidates.
    pub fn new(max_candidates: usize) -> Self {
        Self {
            d2_sq: Vec::with_capacity(max_candidates),
            pairs: Vec::with_capacity(max_candidates),
            nn_indices: Vec::with_capacity(max_candidates),
            nn_dists: Vec::with_capacity(max_candidates),
            probs: Vec::with_capacity(max_candidates),
            cdf: Vec::with_capacity(max_candidates),
            sampled: Vec::with_capacity(max_candidates),
        }
    }
}

/// Validates all inputs and returns the derived `n_candidates`.
fn validate_inputs(
    candidates: &[f64],
    n_vars: usize,
    target: &[f64],
    weights: &[f64],
    config: &KnnConfig,
) -> Result<usize, KnnError> {
    // Config validation first
    config.validate()?;

    // n_vars must be >= 1 to avoid division by zero
    if n_vars == 0 {
        return Err(KnnError::CandidatesShapeMismatch {
            len: candidates.len(),
            n_vars,
        });
    }

    // Candidates shape check
    if candidates.is_empty() {
        return Err(KnnError::EmptyCandidates);
    }
    if !candidates.len().is_multiple_of(n_vars) {
        return Err(KnnError::CandidatesShapeMismatch {
            len: candidates.len(),
            n_vars,
        });
    }

    let n_candidates = candidates.len() / n_vars;

    // Dimension checks
    if target.len() != n_vars {
        return Err(KnnError::TargetDimensionMismatch {
            target: target.len(),
            n_vars,
        });
    }
    if weights.len() != n_vars {
        return Err(KnnError::WeightsDimensionMismatch {
            weights: weights.len(),
            n_vars,
        });
    }

    // NaN guards on target and weights (cheap — small arrays)
    if target.iter().any(|v| !v.is_finite()) {
        return Err(KnnError::NonFiniteInput { input: "target" });
    }
    if weights.iter().any(|v| !v.is_finite()) {
        return Err(KnnError::NonFiniteInput { input: "weights" });
    }

    Ok(n_candidates)
}

/// Internal implementation that assumes all inputs are validated.
#[allow(clippy::too_many_arguments)]
fn knn_sample_inner(
    candidates: &[f64],
    n_candidates: usize,
    n_vars: usize,
    target: &[f64],
    weights: &[f64],
    config: &KnnConfig,
    rng: &mut impl Rng,
    scratch: &mut KnnScratch,
) -> KnnResult {
    // Step 1: Compute k_eff
    let k_eff = config.k().min(n_candidates);

    // Step 2: Compute weighted squared distances
    scratch.d2_sq.clear();
    scratch.d2_sq.resize(n_candidates, 0.0);
    weighted_sq_distances(candidates, n_vars, target, weights, &mut scratch.d2_sq);

    // Step 3: Select k nearest neighbors
    select_k_nearest(
        &scratch.d2_sq,
        k_eff,
        &mut scratch.pairs,
        &mut scratch.nn_indices,
        &mut scratch.nn_dists,
    );

    // Step 4: Compute sampling probabilities
    compute_probs(
        &scratch.nn_dists,
        config.sampling(),
        config.epsilon(),
        &mut scratch.probs,
    );

    // Step 5: Sample from probabilities
    weighted_sample(
        &scratch.probs,
        config.n(),
        rng,
        &mut scratch.cdf,
        &mut scratch.sampled,
    );

    // Step 6: Map local indices back to original candidate indices
    let indices: Vec<usize> = scratch
        .sampled
        .iter()
        .map(|&local| scratch.nn_indices[local])
        .collect();

    // Step 7: Build result
    KnnResult::new(indices, scratch.nn_dists.clone())
}

/// Performs KNN sampling, allocating scratch buffers internally.
///
/// This is the simple entry point. For hot loops, use
/// [`knn_sample_with_scratch`] to reuse allocations.
///
/// # Arguments
///
/// * `candidates` — flat row-major candidate matrix `[n_candidates × n_vars]`
/// * `n_vars` — number of variables per candidate (1 for annual, 2 for daily)
/// * `target` — query point `[n_vars]`
/// * `weights` — per-variable distance weights `[n_vars]`
/// * `config` — KNN configuration (k, n, sampling scheme, epsilon)
/// * `rng` — random number generator
///
/// # Errors
///
/// Returns [`KnnError`] if inputs are invalid (empty candidates, dimension
/// mismatches, non-finite target/weights, invalid config).
pub fn knn_sample(
    candidates: &[f64],
    n_vars: usize,
    target: &[f64],
    weights: &[f64],
    config: &KnnConfig,
    rng: &mut impl Rng,
) -> Result<KnnResult, KnnError> {
    let n_candidates = validate_inputs(candidates, n_vars, target, weights, config)?;
    let mut scratch = KnnScratch::new(n_candidates);
    Ok(knn_sample_inner(
        candidates,
        n_candidates,
        n_vars,
        target,
        weights,
        config,
        rng,
        &mut scratch,
    ))
}

/// Performs KNN sampling, reusing pre-allocated scratch buffers.
///
/// Identical to [`knn_sample`] but avoids heap allocation by reusing `scratch`.
/// Buffers grow as needed and never shrink, making this ideal for hot loops
/// where candidate counts vary.
///
/// # Arguments
///
/// * `candidates` — flat row-major candidate matrix `[n_candidates × n_vars]`
/// * `n_vars` — number of variables per candidate
/// * `target` — query point `[n_vars]`
/// * `weights` — per-variable distance weights `[n_vars]`
/// * `config` — KNN configuration
/// * `rng` — random number generator
/// * `scratch` — reusable scratch buffers (see [`KnnScratch::new`])
///
/// # Errors
///
/// Returns [`KnnError`] if inputs are invalid.
pub fn knn_sample_with_scratch(
    candidates: &[f64],
    n_vars: usize,
    target: &[f64],
    weights: &[f64],
    config: &KnnConfig,
    rng: &mut impl Rng,
    scratch: &mut KnnScratch,
) -> Result<KnnResult, KnnError> {
    let n_candidates = validate_inputs(candidates, n_vars, target, weights, config)?;
    Ok(knn_sample_inner(
        candidates,
        n_candidates,
        n_vars,
        target,
        weights,
        config,
        rng,
        scratch,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    // Test: seeded reproducibility — same seed → same result
    #[test]
    fn test_seeded_reproducibility() {
        let candidates: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let target = [25.0];
        let weights = [1.0];
        let config = KnnConfig::new(7).with_n(10);

        let mut rng1 = StdRng::seed_from_u64(42);
        let r1 = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng1).unwrap();

        let mut rng2 = StdRng::seed_from_u64(42);
        let r2 = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng2).unwrap();

        assert_eq!(r1.indices(), r2.indices());
        assert_eq!(r1.nn_distances(), r2.nn_distances());
    }

    // Test: scratch matches allocating
    #[test]
    fn test_scratch_matches_allocating() {
        let candidates: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let target = [25.0];
        let weights = [1.0];
        let config = KnnConfig::new(7).with_n(10);

        let mut rng1 = StdRng::seed_from_u64(99);
        let r1 = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng1).unwrap();

        let mut rng2 = StdRng::seed_from_u64(99);
        let mut scratch = KnnScratch::new(50);
        let r2 = knn_sample_with_scratch(
            &candidates,
            1,
            &target,
            &weights,
            &config,
            &mut rng2,
            &mut scratch,
        )
        .unwrap();

        assert_eq!(r1.indices(), r2.indices());
        assert_eq!(r1.nn_distances(), r2.nn_distances());
    }

    // Test: single candidate — always returns that candidate
    #[test]
    fn test_single_candidate() {
        let candidates = [42.0];
        let target = [0.0];
        let weights = [1.0];
        let config = KnnConfig::new(5).with_n(3);
        let mut rng = StdRng::seed_from_u64(0);

        let result = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng).unwrap();
        assert_eq!(result.indices().len(), 3);
        for &idx in result.indices() {
            assert_eq!(idx, 0);
        }
        assert_eq!(result.nn_distances().len(), 1); // k_eff = min(5, 1) = 1
    }

    // Test: k > n_candidates is clamped
    #[test]
    fn test_k_greater_than_n_clamped() {
        let candidates = [1.0, 2.0, 3.0]; // 3 candidates, 1D
        let target = [0.0];
        let weights = [1.0];
        let config = KnnConfig::new(100); // k=100 >> 3
        let mut rng = StdRng::seed_from_u64(0);

        let result = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng).unwrap();
        assert_eq!(result.nn_distances().len(), 3); // clamped to 3
    }

    // Test: k=1, n=1 — simplest case
    #[test]
    fn test_k1_n1() {
        let candidates = [10.0, 20.0, 5.0]; // 3 candidates, 1D
        let target = [6.0];
        let weights = [1.0];
        let config = KnnConfig::new(1);
        let mut rng = StdRng::seed_from_u64(0);

        let result = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng).unwrap();
        // Closest to 6.0 is 5.0 (index 2)
        assert_eq!(result.index(), 2);
        assert_eq!(result.nn_distances().len(), 1);
    }

    // Test: all error cases
    #[test]
    fn test_error_empty_candidates() {
        let config = KnnConfig::new(1);
        let mut rng = StdRng::seed_from_u64(0);
        let result = knn_sample(&[], 1, &[0.0], &[1.0], &config, &mut rng);
        assert!(matches!(result, Err(KnnError::EmptyCandidates)));
    }

    #[test]
    fn test_error_candidates_shape() {
        let config = KnnConfig::new(1);
        let mut rng = StdRng::seed_from_u64(0);
        // 5 elements with n_vars=2 doesn't divide evenly
        let result = knn_sample(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            2,
            &[0.0, 0.0],
            &[1.0, 1.0],
            &config,
            &mut rng,
        );
        assert!(matches!(
            result,
            Err(KnnError::CandidatesShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_error_target_dim() {
        let config = KnnConfig::new(1);
        let mut rng = StdRng::seed_from_u64(0);
        let result = knn_sample(&[1.0, 2.0], 2, &[0.0], &[1.0, 1.0], &config, &mut rng);
        assert!(matches!(
            result,
            Err(KnnError::TargetDimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_error_weights_dim() {
        let config = KnnConfig::new(1);
        let mut rng = StdRng::seed_from_u64(0);
        let result = knn_sample(&[1.0, 2.0], 2, &[0.0, 0.0], &[1.0], &config, &mut rng);
        assert!(matches!(
            result,
            Err(KnnError::WeightsDimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_error_nan_target() {
        let config = KnnConfig::new(1);
        let mut rng = StdRng::seed_from_u64(0);
        let result = knn_sample(&[1.0], 1, &[f64::NAN], &[1.0], &config, &mut rng);
        assert!(matches!(
            result,
            Err(KnnError::NonFiniteInput { input: "target" })
        ));
    }

    #[test]
    fn test_error_inf_weights() {
        let config = KnnConfig::new(1);
        let mut rng = StdRng::seed_from_u64(0);
        let result = knn_sample(&[1.0], 1, &[0.0], &[f64::INFINITY], &config, &mut rng);
        assert!(matches!(
            result,
            Err(KnnError::NonFiniteInput { input: "weights" })
        ));
    }

    // Test: 1D annual-style
    #[test]
    fn test_1d_annual_style() {
        // 50 candidates (simulating 50 historical years of annual precip)
        let candidates: Vec<f64> = (0..50).map(|i| (i as f64) * 10.0).collect();
        let target = [250.0]; // target annual precip
        let weights = [1.0];
        let k = (50.0_f64).sqrt().ceil() as usize; // ceil(sqrt(50)) = 8
        let config = KnnConfig::new(k).with_n(100);
        let mut rng = StdRng::seed_from_u64(42);

        let result = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng).unwrap();
        assert_eq!(result.indices().len(), 100);
        assert_eq!(result.nn_distances().len(), k.min(50));
        // All indices should be in [0, 50)
        for &idx in result.indices() {
            assert!(idx < 50);
        }
    }

    // Test: 2D daily-style
    #[test]
    fn test_2d_daily_style() {
        // 20 candidates in 2D (precip_anom, temp_anom)
        let mut candidates = Vec::new();
        for i in 0..20 {
            candidates.push(i as f64 * 0.5); // precip anom
            candidates.push(i as f64 * 0.3); // temp anom
        }
        let target = [3.0, 2.0];
        let weights = [100.0 / 2.0, 10.0 / 1.5]; // typical KNN weights
        let k = (20.0_f64).sqrt().round() as usize; // round(sqrt(20)) = 4
        let config = KnnConfig::new(k);
        let mut rng = StdRng::seed_from_u64(42);

        let result = knn_sample(&candidates, 2, &target, &weights, &config, &mut rng).unwrap();
        assert_eq!(result.indices().len(), 1);
        assert!(result.index() < 20);
        assert_eq!(result.nn_distances().len(), k.min(20));
    }

    // Test: scratch reuse with varying sizes
    #[test]
    fn test_scratch_reuse_varying_sizes() {
        let mut scratch = KnnScratch::new(10);
        let weights = [1.0];
        let config = KnnConfig::new(3);

        // First call: 10 candidates
        let candidates1: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut rng = StdRng::seed_from_u64(1);
        let r1 = knn_sample_with_scratch(
            &candidates1,
            1,
            &[5.0],
            &weights,
            &config,
            &mut rng,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(r1.nn_distances().len(), 3);

        // Second call: 50 candidates (bigger — scratch should grow)
        let candidates2: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut rng = StdRng::seed_from_u64(2);
        let r2 = knn_sample_with_scratch(
            &candidates2,
            1,
            &[25.0],
            &weights,
            &config,
            &mut rng,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(r2.nn_distances().len(), 3);

        // Third call: 5 candidates (smaller — scratch doesn't shrink)
        let candidates3: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let mut rng = StdRng::seed_from_u64(3);
        let r3 = knn_sample_with_scratch(
            &candidates3,
            1,
            &[2.0],
            &weights,
            &config,
            &mut rng,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(r3.nn_distances().len(), 3);

        // Verify capacity didn't shrink (d2_sq was grown to 50)
        assert!(scratch.d2_sq.capacity() >= 50);
    }
}
