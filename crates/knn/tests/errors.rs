//! Integration tests for KnnError variants.

use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_knn::{KnnConfig, KnnError, knn_sample};

fn rng() -> StdRng {
    StdRng::seed_from_u64(0)
}

#[test]
fn error_empty_candidates() {
    let config = KnnConfig::new(1);
    let result = knn_sample(&[], 1, &[0.0], &[1.0], &config, &mut rng());
    assert!(matches!(result, Err(KnnError::EmptyCandidates)));
}

#[test]
fn error_candidates_shape_mismatch() {
    // 5 elements, n_vars=2 → not divisible
    let config = KnnConfig::new(1);
    let result = knn_sample(
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        2,
        &[0.0, 0.0],
        &[1.0, 1.0],
        &config,
        &mut rng(),
    );
    assert!(matches!(
        result,
        Err(KnnError::CandidatesShapeMismatch { len: 5, n_vars: 2 })
    ));
}

#[test]
fn error_target_dimension_mismatch() {
    let config = KnnConfig::new(1);
    // candidates are 2D (1 candidate × 2 vars), but target is 1D
    let result = knn_sample(&[1.0, 2.0], 2, &[0.0], &[1.0, 1.0], &config, &mut rng());
    assert!(matches!(
        result,
        Err(KnnError::TargetDimensionMismatch {
            target: 1,
            n_vars: 2
        })
    ));
}

#[test]
fn error_weights_dimension_mismatch() {
    let config = KnnConfig::new(1);
    // candidates are 2D but weights is 1D
    let result = knn_sample(&[1.0, 2.0], 2, &[0.0, 0.0], &[1.0], &config, &mut rng());
    assert!(matches!(
        result,
        Err(KnnError::WeightsDimensionMismatch {
            weights: 1,
            n_vars: 2
        })
    ));
}

#[test]
fn error_nan_in_target() {
    let config = KnnConfig::new(1);
    let result = knn_sample(&[1.0], 1, &[f64::NAN], &[1.0], &config, &mut rng());
    assert!(matches!(
        result,
        Err(KnnError::NonFiniteInput { input: "target" })
    ));
}

#[test]
fn error_inf_in_target() {
    let config = KnnConfig::new(1);
    let result = knn_sample(&[1.0], 1, &[f64::INFINITY], &[1.0], &config, &mut rng());
    assert!(matches!(
        result,
        Err(KnnError::NonFiniteInput { input: "target" })
    ));
}

#[test]
fn error_nan_in_weights() {
    let config = KnnConfig::new(1);
    let result = knn_sample(&[1.0], 1, &[0.0], &[f64::NAN], &config, &mut rng());
    assert!(matches!(
        result,
        Err(KnnError::NonFiniteInput { input: "weights" })
    ));
}

#[test]
fn error_inf_in_weights() {
    let config = KnnConfig::new(1);
    let result = knn_sample(&[1.0], 1, &[0.0], &[f64::INFINITY], &config, &mut rng());
    assert!(matches!(
        result,
        Err(KnnError::NonFiniteInput { input: "weights" })
    ));
}

#[test]
fn error_invalid_k() {
    let config = KnnConfig::new(0);
    let result = knn_sample(&[1.0], 1, &[0.0], &[1.0], &config, &mut rng());
    assert!(matches!(result, Err(KnnError::InvalidK { k: 0 })));
}

#[test]
fn error_invalid_n() {
    let config = KnnConfig::new(1).with_n(0);
    let result = knn_sample(&[1.0], 1, &[0.0], &[1.0], &config, &mut rng());
    assert!(matches!(result, Err(KnnError::InvalidN { n: 0 })));
}

#[test]
fn error_invalid_epsilon() {
    let config = KnnConfig::new(1).with_epsilon(-1.0);
    let result = knn_sample(&[1.0], 1, &[0.0], &[1.0], &config, &mut rng());
    assert!(matches!(result, Err(KnnError::InvalidEpsilon { .. })));
}
