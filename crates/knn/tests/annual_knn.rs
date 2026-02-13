//! Integration tests for annual KNN (1D) usage pattern.

use approx::assert_abs_diff_eq;
use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_knn::{KnnConfig, Sampling, knn_sample};

/// Simulates the annual KNN use case: 1D candidates (annual precip),
/// k = ceil(sqrt(n_years)), n = 100 samples, Rank weighting.
#[test]
fn annual_knn_basic() {
    let n_years = 50;
    let candidates: Vec<f64> = (0..n_years).map(|i| 200.0 + (i as f64) * 10.0).collect();
    let target = [450.0]; // target annual precip
    let weights = [1.0];
    let k = (n_years as f64).sqrt().ceil() as usize; // ceil(sqrt(50)) = 8
    let config = KnnConfig::new(k).with_n(100).with_sampling(Sampling::Rank);
    let mut rng = StdRng::seed_from_u64(42);

    let result = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng).unwrap();

    // 100 samples drawn
    assert_eq!(result.indices().len(), 100);
    // 8 nearest-neighbor distances
    assert_eq!(result.nn_distances().len(), 8);
    // All indices valid
    for &idx in result.indices() {
        assert!(idx < n_years, "index {} out of bounds", idx);
    }
    // Distances are sorted ascending
    for w in result.nn_distances().windows(2) {
        assert!(w[0] <= w[1], "distances not sorted: {} > {}", w[0], w[1]);
    }
}

/// Rank bias: the closest neighbor should be sampled most often.
#[test]
fn annual_knn_rank_bias() {
    let candidates: Vec<f64> = (0..30).map(|i| i as f64 * 10.0).collect();
    let target = [150.0]; // closest to candidate 15
    let weights = [1.0];
    let k = (30.0_f64).sqrt().ceil() as usize; // 6
    let config = KnnConfig::new(k).with_n(5000).with_sampling(Sampling::Rank);
    let mut rng = StdRng::seed_from_u64(123);

    let result = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng).unwrap();

    // Count occurrences of the closest candidate (index 15, distance=0)
    let closest_count = result.indices().iter().filter(|&&idx| idx == 15).count();

    // With rank weighting and k=6, closest gets weight 1/H_6 ~ 0.408
    // Expected ~2040 out of 5000. Allow wide range.
    assert!(
        closest_count > 1500,
        "expected closest to appear >1500 times in 5000 draws, got {}",
        closest_count
    );
}

/// Seeded reproducibility for annual KNN.
#[test]
fn annual_knn_reproducibility() {
    let candidates: Vec<f64> = (0..40).map(|i| i as f64 * 5.0).collect();
    let target = [100.0];
    let weights = [1.0];
    let config = KnnConfig::new(6).with_n(50);

    let mut rng1 = StdRng::seed_from_u64(777);
    let r1 = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng1).unwrap();

    let mut rng2 = StdRng::seed_from_u64(777);
    let r2 = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng2).unwrap();

    assert_eq!(r1.indices(), r2.indices());
    for (a, b) in r1.nn_distances().iter().zip(r2.nn_distances().iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }
}
