//! Integration tests for KnnScratch reuse.

use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_knn::{KnnConfig, KnnScratch, knn_sample, knn_sample_with_scratch};

/// Scratch version produces identical results to non-scratch.
#[test]
fn scratch_matches_non_scratch() {
    let candidates: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let target = [15.0];
    let weights = [1.0];
    let config = KnnConfig::new(5).with_n(20);

    let mut rng1 = StdRng::seed_from_u64(42);
    let r1 = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng1).unwrap();

    let mut rng2 = StdRng::seed_from_u64(42);
    let mut scratch = KnnScratch::new(30);
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

/// Simulate a 365-day loop with varying candidate counts.
#[test]
fn daily_loop_365() {
    let mut scratch = KnnScratch::new(50);
    let config = KnnConfig::new(3);

    for day in 0..365u64 {
        // Vary candidate count: between 10 and 60
        let n_cand = 10 + (day % 51) as usize;
        let candidates: Vec<f64> = (0..n_cand).map(|i| i as f64).collect();
        let target = [(n_cand as f64) / 2.0];
        let weights = [1.0];
        let mut rng = StdRng::seed_from_u64(day);

        let result = knn_sample_with_scratch(
            &candidates,
            1,
            &target,
            &weights,
            &config,
            &mut rng,
            &mut scratch,
        )
        .unwrap();

        assert_eq!(result.indices().len(), 1);
        assert!(result.index() < n_cand);
    }
}

/// Scratch works correctly after large -> small candidate transitions.
#[test]
fn scratch_large_then_small() {
    let mut scratch = KnnScratch::new(5);
    let config = KnnConfig::new(2);
    let weights = [1.0];

    // Large call
    let cands_large: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let mut rng = StdRng::seed_from_u64(1);
    let r1 = knn_sample_with_scratch(
        &cands_large,
        1,
        &[100.0],
        &weights,
        &config,
        &mut rng,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(r1.indices().len(), 1);
    assert!(r1.index() < 200);

    // Small call after large -- must still work correctly
    let cands_small: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let mut rng = StdRng::seed_from_u64(2);
    let r2 = knn_sample_with_scratch(
        &cands_small,
        1,
        &[2.0],
        &weights,
        &config,
        &mut rng,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(r2.indices().len(), 1);
    assert!(r2.index() < 5);
    assert_eq!(r2.nn_distances().len(), 2);
}
