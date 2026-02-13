//! Integration tests for daily KNN (2D) usage pattern.

use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_knn::{KnnConfig, Sampling, knn_sample};

/// Simulates the daily KNN: 2D candidates (precip_anom, temp_anom),
/// k ~ sqrt(n_candidates), n=1, Rank, with feature weights.
#[test]
fn daily_knn_basic() {
    let n_candidates = 25;
    let mut candidates = Vec::with_capacity(n_candidates * 2);
    for i in 0..n_candidates {
        candidates.push((i as f64) * 0.5 - 6.0); // precip anomaly
        candidates.push((i as f64) * 0.3 - 3.5); // temp anomaly
    }
    let target = [0.0, 0.0];
    let sd_precip = 2.0;
    let sd_temp = 1.5;
    let weights = [100.0 / sd_precip, 10.0 / sd_temp]; // typical KNN weights
    let k = (n_candidates as f64).sqrt().round() as usize; // 5
    let config = KnnConfig::new(k).with_sampling(Sampling::Rank);
    let mut rng = StdRng::seed_from_u64(42);

    let result = knn_sample(&candidates, 2, &target, &weights, &config, &mut rng).unwrap();

    assert_eq!(result.indices().len(), 1); // n=1
    assert!(result.index() < n_candidates);
    assert_eq!(result.nn_distances().len(), k);
    // Distances sorted
    for w in result.nn_distances().windows(2) {
        assert!(w[0] <= w[1]);
    }
}

/// The closest candidate should be selected most often across many draws.
#[test]
fn daily_knn_closest_selected_most() {
    let n_candidates = 20;
    let mut candidates = Vec::with_capacity(n_candidates * 2);
    for i in 0..n_candidates {
        candidates.push(i as f64);
        candidates.push(i as f64);
    }
    let target = [5.0, 5.0]; // closest to candidate 5
    let weights = [1.0, 1.0];
    let config = KnnConfig::new(4); // k=4, n=1

    let n_trials = 3000;
    let mut counts = vec![0usize; n_candidates];
    for trial in 0..n_trials {
        let mut rng = StdRng::seed_from_u64(trial as u64);
        let result = knn_sample(&candidates, 2, &target, &weights, &config, &mut rng).unwrap();
        counts[result.index()] += 1;
    }

    // Candidate 5 (distance=0) should have the highest count
    let max_idx = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| *c)
        .unwrap()
        .0;
    assert_eq!(
        max_idx, 5,
        "expected candidate 5 to be most frequent, got {}",
        max_idx
    );
}

/// Weights affect which candidate is closest.
#[test]
fn daily_knn_weights_affect_distance() {
    // Two candidates: (1, 0) and (0, 1), target = (0, 0)
    // With weights [1, 1]: both equidistant (d=1)
    // With weights [100, 1]: candidate (0,1) is much closer
    let candidates = [1.0, 0.0, 0.0, 1.0];
    let target = [0.0, 0.0];
    let weights = [100.0, 1.0];
    let config = KnnConfig::new(1); // k=1 -> always picks closest
    let mut rng = StdRng::seed_from_u64(0);

    let result = knn_sample(&candidates, 2, &target, &weights, &config, &mut rng).unwrap();
    // Candidate 1 (0,1) should be closer: dist^2 = 100*0 + 1*1 = 1 vs 100*1 + 1*0 = 100
    assert_eq!(result.index(), 1);
}
