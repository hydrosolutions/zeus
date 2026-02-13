//! Edge case integration tests.

use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_knn::{KnnConfig, knn_sample};

/// Single candidate: always selected.
#[test]
fn single_candidate_1d() {
    let config = KnnConfig::new(10).with_n(5); // k >> 1
    let mut rng = StdRng::seed_from_u64(0);
    let result = knn_sample(&[99.0], 1, &[0.0], &[1.0], &config, &mut rng).unwrap();
    assert_eq!(result.indices().len(), 5);
    for &idx in result.indices() {
        assert_eq!(idx, 0);
    }
    assert_eq!(result.nn_distances().len(), 1); // k_eff = min(10, 1) = 1
}

/// k > n_candidates: clamped to n_candidates.
#[test]
fn k_greater_than_n() {
    let candidates = [1.0, 2.0, 3.0];
    let config = KnnConfig::new(50);
    let mut rng = StdRng::seed_from_u64(0);
    let result = knn_sample(&candidates, 1, &[0.0], &[1.0], &config, &mut rng).unwrap();
    assert_eq!(result.nn_distances().len(), 3);
}

/// k = n_candidates: all candidates are neighbors.
#[test]
fn k_equals_n() {
    let candidates = [1.0, 2.0, 3.0, 4.0, 5.0];
    let config = KnnConfig::new(5).with_n(10);
    let mut rng = StdRng::seed_from_u64(42);
    let result = knn_sample(&candidates, 1, &[3.0], &[1.0], &config, &mut rng).unwrap();
    assert_eq!(result.nn_distances().len(), 5);
    assert_eq!(result.indices().len(), 10);
}

/// 1D with many candidates (100).
#[test]
fn many_candidates_1d() {
    let candidates: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let config = KnnConfig::new(10).with_n(50);
    let mut rng = StdRng::seed_from_u64(0);
    let result = knn_sample(&candidates, 1, &[50.0], &[1.0], &config, &mut rng).unwrap();
    assert_eq!(result.indices().len(), 50);
    assert_eq!(result.nn_distances().len(), 10);
    for &idx in result.indices() {
        assert!(idx < 100);
    }
}

/// 5D dispatch (exercises the ND path).
#[test]
fn five_dimensional() {
    let n_cand = 20;
    let n_vars = 5;
    let candidates: Vec<f64> = (0..n_cand * n_vars).map(|i| i as f64).collect();
    let target = vec![10.0; n_vars];
    let weights = vec![1.0; n_vars];
    let config = KnnConfig::new(4);
    let mut rng = StdRng::seed_from_u64(0);
    let result = knn_sample(&candidates, n_vars, &target, &weights, &config, &mut rng).unwrap();
    assert_eq!(result.indices().len(), 1);
    assert!(result.index() < n_cand);
}

/// Large k and n.
#[test]
fn large_k_and_n() {
    let n_cand = 200;
    let candidates: Vec<f64> = (0..n_cand).map(|i| i as f64).collect();
    let config = KnnConfig::new(50).with_n(500);
    let mut rng = StdRng::seed_from_u64(0);
    let result = knn_sample(&candidates, 1, &[100.0], &[1.0], &config, &mut rng).unwrap();
    assert_eq!(result.indices().len(), 500);
    assert_eq!(result.nn_distances().len(), 50);
}

/// All identical candidates: distances all zero, all candidates are equally likely.
#[test]
fn identical_candidates() {
    let candidates = vec![5.0; 10]; // 10 identical candidates
    let config = KnnConfig::new(5).with_n(100);
    let mut rng = StdRng::seed_from_u64(42);
    let result = knn_sample(&candidates, 1, &[5.0], &[1.0], &config, &mut rng).unwrap();
    assert_eq!(result.indices().len(), 100);
    // All distances should be 0
    for &d in result.nn_distances() {
        assert!(d.abs() < 1e-12, "expected 0 distance, got {}", d);
    }
}

/// Target equals a candidate exactly.
#[test]
fn target_equals_candidate() {
    let candidates = [1.0, 2.0, 3.0, 4.0, 5.0];
    let config = KnnConfig::new(1);
    let mut rng = StdRng::seed_from_u64(0);
    let result = knn_sample(&candidates, 1, &[3.0], &[1.0], &config, &mut rng).unwrap();
    assert_eq!(result.index(), 2); // index 2 has value 3.0
    assert!(result.nn_distances()[0].abs() < 1e-12);
}
