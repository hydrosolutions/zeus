//! Integration tests for the three sampling modes.

use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_knn::{KnnConfig, Sampling, knn_sample};

/// Helper: run many trials and count how often each neighbor is selected.
fn sample_counts(
    candidates: &[f64],
    n_vars: usize,
    target: &[f64],
    weights: &[f64],
    config: &KnnConfig,
    n_trials: usize,
) -> Vec<usize> {
    let n_cand = candidates.len() / n_vars;
    let mut counts = vec![0usize; n_cand];
    for trial in 0..n_trials {
        let mut rng = StdRng::seed_from_u64(trial as u64);
        let result = knn_sample(candidates, n_vars, target, weights, config, &mut rng).unwrap();
        counts[result.index()] += 1;
    }
    counts
}

/// Uniform sampling: all k neighbors have roughly equal probability.
#[test]
fn uniform_roughly_equal() {
    // 10 candidates, k=5, uniform
    let candidates: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let target = [5.0];
    let weights = [1.0];
    let config = KnnConfig::new(5).with_sampling(Sampling::Uniform);
    let n_trials = 5000;

    let counts = sample_counts(&candidates, 1, &target, &weights, &config, n_trials);

    // The 5 nearest neighbors should be selected; the other 5 should have count 0
    // Nearest to 5.0 are: 5(d=0), 4(d=1), 6(d=1), 3(d=2), 7(d=2)
    // Each should get ~1000 ± tolerance
    let active_counts: Vec<usize> = counts.iter().copied().filter(|&c| c > 0).collect();
    assert_eq!(
        active_counts.len(),
        5,
        "expected 5 active neighbors, got {}",
        active_counts.len()
    );

    for &c in &active_counts {
        // Each should be in 700..1300 (expected ~1000)
        assert!(
            (700..=1300).contains(&c),
            "uniform count {} not in expected range 700..1300",
            c
        );
    }
}

/// Rank sampling: closest neighbors selected more often.
#[test]
fn rank_monotonically_decreasing() {
    // Use candidates at increasing distances from target
    // Candidates: 0, 1, 2, 3, 4 with target = 0.
    // Nearest: 0(d=0), 1(d=1), 2(d=2), 3(d=3), 4(d=4)
    let candidates: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let target = [0.0];
    let weights = [1.0];
    let config = KnnConfig::new(5).with_sampling(Sampling::Rank);
    let n_trials = 10_000;

    let counts = sample_counts(&candidates, 1, &target, &weights, &config, n_trials);

    // Rank weighting: closest (idx 0) should have highest count, furthest (idx 4) lowest
    // Counts should be monotonically decreasing for idx 0..5
    for i in 1..5 {
        assert!(
            counts[i - 1] >= counts[i],
            "rank counts not decreasing: counts[{}]={} < counts[{}]={}",
            i - 1,
            counts[i - 1],
            i,
            counts[i]
        );
    }
    // Closest should have significantly more than furthest
    assert!(
        counts[0] > counts[4] * 2,
        "expected counts[0]={} > 2 * counts[4]={}",
        counts[0],
        counts[4]
    );
}

/// Gaussian with small bandwidth: concentrates on closest.
#[test]
fn gaussian_small_bandwidth() {
    let candidates: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let target = [0.0];
    let weights = [1.0];
    let config = KnnConfig::new(5).with_sampling(Sampling::Gaussian {
        bandwidth: Some(0.1),
    });
    let n_trials = 3000;

    let counts = sample_counts(&candidates, 1, &target, &weights, &config, n_trials);

    // Very small bandwidth → almost all selections should be the closest neighbor
    assert!(
        counts[0] > n_trials * 8 / 10,
        "expected >80% for closest, got {} out of {}",
        counts[0],
        n_trials
    );
}

/// Gaussian with large bandwidth: approaches uniform.
#[test]
fn gaussian_large_bandwidth() {
    let candidates: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let target = [2.0];
    let weights = [1.0];
    let config = KnnConfig::new(5).with_sampling(Sampling::Gaussian {
        bandwidth: Some(1000.0),
    });
    let n_trials = 5000;

    let counts = sample_counts(&candidates, 1, &target, &weights, &config, n_trials);

    // Very large bandwidth → all neighbors roughly equally likely
    for &c in &counts {
        assert!(
            (600..=1400).contains(&c),
            "gaussian large bw count {} not in expected range",
            c
        );
    }
}
