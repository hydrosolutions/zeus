//! Weighted K-nearest neighbor sampling.
//!
//! This crate provides a generic KNN implementation with probability-weighted
//! sampling, supporting three weighting schemes:
//!
//! | Scheme | Formula | Use case |
//! |--------|---------|----------|
//! | Uniform | `1/k` for all neighbors | Baseline |
//! | Rank | `(1/i) / Hₖ` (harmonic) | Default — Lall & Sharma (1996) |
//! | Gaussian | `exp(-d²/2bw²) + ε` | Distance-sensitive |
//!
//! # Quick start
//!
//! ```
//! use zeus_knn::{KnnConfig, Sampling, knn_sample};
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! let candidates = vec![1.0, 3.0, 5.0, 7.0, 9.0];
//! let target = [4.0];
//! let weights = [1.0];
//! let config = KnnConfig::new(3).with_n(2).with_sampling(Sampling::Rank);
//! let mut rng = StdRng::seed_from_u64(42);
//!
//! let result = knn_sample(&candidates, 1, &target, &weights, &config, &mut rng).unwrap();
//! assert_eq!(result.indices().len(), 2);
//! ```
//!
//! # Architecture
//!
//! ```text
//! knn_sample()
//!   ├─ validate inputs
//!   ├─ weighted_sq_distances()   (distance.rs)
//!   ├─ select_k_nearest()        (select.rs)
//!   ├─ compute_probs()           (sample.rs)
//!   └─ weighted_sample()         (sample.rs)
//! ```
//!
//! For hot loops, use [`knn_sample_with_scratch`] with a reusable
//! [`KnnScratch`] to avoid per-call heap allocation.

pub mod config;
pub mod error;
pub mod knn;
pub mod result;

pub(crate) mod distance;
pub(crate) mod sample;
pub(crate) mod select;

pub use config::{KnnConfig, Sampling};
pub use error::KnnError;
pub use knn::{KnnScratch, knn_sample, knn_sample_with_scratch};
pub use result::KnnResult;

/// Computes the Lall & Sharma (1996) heuristic for k.
///
/// Returns `floor(sqrt(n_candidates)).max(1)`.
///
/// Callers adjust for context: annual KNN typically uses `ceil(sqrt(n))`,
/// daily KNN uses `round(sqrt(n))`. This function provides the base `floor`
/// variant.
pub fn k_lall_sharma(n_candidates: usize) -> usize {
    (n_candidates as f64).sqrt().floor().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_lall_sharma_known_values() {
        assert_eq!(k_lall_sharma(1), 1);
        assert_eq!(k_lall_sharma(4), 2);
        assert_eq!(k_lall_sharma(9), 3);
        assert_eq!(k_lall_sharma(10), 3); // floor(3.16) = 3
        assert_eq!(k_lall_sharma(16), 4);
        assert_eq!(k_lall_sharma(25), 5);
        assert_eq!(k_lall_sharma(50), 7); // floor(7.07) = 7
        assert_eq!(k_lall_sharma(100), 10);
    }

    #[test]
    fn test_k_lall_sharma_zero() {
        // 0 candidates: sqrt(0)=0, max(1) => 1
        assert_eq!(k_lall_sharma(0), 1);
    }

    #[test]
    fn test_k_lall_sharma_large() {
        assert_eq!(k_lall_sharma(10000), 100);
        assert_eq!(k_lall_sharma(10001), 100); // floor(100.005) = 100
    }
}
