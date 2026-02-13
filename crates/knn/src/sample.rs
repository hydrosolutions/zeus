//! Probability computation and weighted sampling for KNN.

use crate::config::Sampling;
use rand::Rng;

/// Computes sampling probabilities for the k nearest neighbors.
///
/// # Schemes
///
/// | Mode | Formula |
/// |------|---------|
/// | Uniform | `1 / k` for all |
/// | Rank | `(1/i) / H_k` where `H_k = Σ 1/j` (harmonic) |
/// | Gaussian | `exp(-d² / 2bw²) + ε`, normalised; auto-bandwidth via median |
///
/// Gaussian falls back to uniform if bandwidth is non-finite or <= 0.
pub(crate) fn compute_probs(
    nn_dists: &[f64],
    sampling: &Sampling,
    epsilon: f64,
    probs: &mut Vec<f64>,
) {
    let k = nn_dists.len();
    probs.clear();

    match sampling {
        Sampling::Uniform => {
            let p = 1.0 / k as f64;
            probs.resize(k, p);
        }
        Sampling::Rank => {
            let harmonic_sum: f64 = (1..=k).map(|j| 1.0 / j as f64).sum();
            for j in 1..=k {
                probs.push((1.0 / j as f64) / harmonic_sum);
            }
        }
        Sampling::Gaussian { bandwidth } => {
            let bw = bandwidth.unwrap_or_else(|| median_of(nn_dists));
            if !bw.is_finite() || bw <= 0.0 {
                // Degenerate: fall back to uniform
                let p = 1.0 / k as f64;
                probs.resize(k, p);
            } else {
                let two_bw_sq = 2.0 * bw * bw;
                let mut sum = 0.0;
                for &d in nn_dists {
                    let p = (-d * d / two_bw_sq).exp() + epsilon;
                    probs.push(p);
                    sum += p;
                }
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }
    }
}

/// Samples `n` indices from `[0..probs.len())` with replacement using given probabilities.
///
/// Builds a CDF and uses binary search (`partition_point`) for each draw.
/// The last CDF entry is forced to 1.0 to eliminate floating-point edge cases.
pub(crate) fn weighted_sample(
    probs: &[f64],
    n: usize,
    rng: &mut impl Rng,
    cdf: &mut Vec<f64>,
    out: &mut Vec<usize>,
) {
    out.clear();
    cdf.clear();

    // Build cumulative distribution
    let mut acc = 0.0;
    for &p in probs {
        acc += p;
        cdf.push(acc);
    }
    // Force last entry to exactly 1.0 to handle floating-point accumulation
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }

    // Sample n times with replacement
    for _ in 0..n {
        let u: f64 = rng.random();
        let idx = cdf.partition_point(|&c| c < u).min(probs.len() - 1);
        out.push(idx);
    }
}

/// Computes the median of a slice.
///
/// Allocates a sorted clone. Returns 0.0 for empty input.
fn median_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_uniform_equal_probs() {
        let dists = [1.0, 2.0, 3.0, 4.0];
        let mut probs = Vec::new();
        compute_probs(&dists, &Sampling::Uniform, 1e-8, &mut probs);
        assert_eq!(probs.len(), 4);
        for &p in &probs {
            assert_abs_diff_eq!(p, 0.25, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_rank_harmonic_k1() {
        let dists = [1.0];
        let mut probs = Vec::new();
        compute_probs(&dists, &Sampling::Rank, 1e-8, &mut probs);
        assert_eq!(probs.len(), 1);
        assert_abs_diff_eq!(probs[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_harmonic_k2() {
        let dists = [1.0, 2.0];
        let mut probs = Vec::new();
        compute_probs(&dists, &Sampling::Rank, 1e-8, &mut probs);
        assert_eq!(probs.len(), 2);
        let h2 = 1.0 + 0.5;
        assert_abs_diff_eq!(probs[0], 1.0 / h2, epsilon = 1e-12);
        assert_abs_diff_eq!(probs[1], 0.5 / h2, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_harmonic_k3() {
        let dists = [1.0, 2.0, 3.0];
        let mut probs = Vec::new();
        compute_probs(&dists, &Sampling::Rank, 1e-8, &mut probs);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_harmonic_k5() {
        let dists = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut probs = Vec::new();
        compute_probs(&dists, &Sampling::Rank, 1e-8, &mut probs);
        assert_eq!(probs.len(), 5);
        let sum: f64 = probs.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
        // Verify decreasing
        for i in 1..probs.len() {
            assert!(
                probs[i] < probs[i - 1],
                "probs should be decreasing: probs[{}]={} >= probs[{}]={}",
                i,
                probs[i],
                i - 1,
                probs[i - 1]
            );
        }
    }

    #[test]
    fn test_gaussian_kernel_shape() {
        let dists = [0.0, 1.0, 2.0, 3.0];
        let sampling = Sampling::Gaussian {
            bandwidth: Some(1.0),
        };
        let mut probs = Vec::new();
        compute_probs(&dists, &sampling, 1e-8, &mut probs);
        assert_eq!(probs.len(), 4);
        // Verify decreasing (closer distance = higher prob)
        for i in 1..probs.len() {
            assert!(
                probs[i] < probs[i - 1],
                "probs should be decreasing: probs[{}]={} >= probs[{}]={}",
                i,
                probs[i],
                i - 1,
                probs[i - 1]
            );
        }
        // Verify sum to 1.0
        let sum: f64 = probs.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gaussian_auto_bandwidth() {
        let dists = [1.0, 2.0, 3.0, 4.0];
        let sampling = Sampling::Gaussian { bandwidth: None };
        let mut probs = Vec::new();
        compute_probs(&dists, &sampling, 1e-8, &mut probs);
        assert_eq!(probs.len(), 4);
        // median of [1,2,3,4] = 2.5
        // Verify sum to 1.0
        let sum: f64 = probs.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
        // Verify computed correctly with bw=2.5
        let bw = 2.5;
        let two_bw_sq = 2.0 * bw * bw;
        let eps = 1e-8;
        let raw: Vec<f64> = dists
            .iter()
            .map(|&d| (-d * d / two_bw_sq).exp() + eps)
            .collect();
        let raw_sum: f64 = raw.iter().sum();
        for (i, &p) in probs.iter().enumerate() {
            assert_abs_diff_eq!(p, raw[i] / raw_sum, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_gaussian_degenerate_fallback() {
        // All distances zero -> median = 0 -> fallback to uniform
        let dists = [0.0, 0.0, 0.0, 0.0];
        let sampling = Sampling::Gaussian { bandwidth: None };
        let mut probs = Vec::new();
        compute_probs(&dists, &sampling, 1e-8, &mut probs);
        assert_eq!(probs.len(), 4);
        for &p in &probs {
            assert_abs_diff_eq!(p, 0.25, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_seeded_reproducibility() {
        let probs = [0.5, 0.3, 0.2];
        let n = 100;

        let mut rng1 = StdRng::seed_from_u64(42);
        let mut cdf1 = Vec::new();
        let mut out1 = Vec::new();
        weighted_sample(&probs, n, &mut rng1, &mut cdf1, &mut out1);

        let mut rng2 = StdRng::seed_from_u64(42);
        let mut cdf2 = Vec::new();
        let mut out2 = Vec::new();
        weighted_sample(&probs, n, &mut rng2, &mut cdf2, &mut out2);

        assert_eq!(out1, out2);
    }

    #[test]
    fn test_statistical_distribution() {
        let probs = [0.5, 0.3, 0.2];
        let n = 10_000;
        let mut rng = StdRng::seed_from_u64(12345);
        let mut cdf = Vec::new();
        let mut out = Vec::new();
        weighted_sample(&probs, n, &mut rng, &mut cdf, &mut out);

        let mut counts = [0usize; 3];
        for &idx in &out {
            counts[idx] += 1;
        }

        // Index 0 should have the highest count, index 2 the lowest
        assert!(
            counts[0] > counts[1],
            "expected counts[0]={} > counts[1]={}",
            counts[0],
            counts[1]
        );
        assert!(
            counts[1] > counts[2],
            "expected counts[1]={} > counts[2]={}",
            counts[1],
            counts[2]
        );

        // Index 0 should be around 5000 (in 4500..5500)
        assert!(
            (4500..=5500).contains(&counts[0]),
            "expected counts[0] in 4500..5500, got {}",
            counts[0]
        );
    }

    #[test]
    fn test_cdf_edge_u_zero() {
        // When u is very close to 0, partition_point should return index 0
        // since cdf[0] > 0 for any nonzero prob.
        // We test this with a seeded rng and verify no out-of-bounds.
        let probs = [0.25, 0.25, 0.25, 0.25];
        let mut rng = StdRng::seed_from_u64(0);
        let mut cdf = Vec::new();
        let mut out = Vec::new();
        weighted_sample(&probs, 1000, &mut rng, &mut cdf, &mut out);

        // All indices should be in [0, 3]
        for &idx in &out {
            assert!(idx < 4, "index {} out of bounds", idx);
        }
    }

    #[test]
    fn test_median_helper() {
        // Odd length: [3, 1, 2] -> sorted [1, 2, 3] -> median = 2.0
        assert_abs_diff_eq!(median_of(&[3.0, 1.0, 2.0]), 2.0, epsilon = 1e-12);
        // Even length: [4, 1, 3, 2] -> sorted [1, 2, 3, 4] -> median = 2.5
        assert_abs_diff_eq!(median_of(&[4.0, 1.0, 3.0, 2.0]), 2.5, epsilon = 1e-12);
        // Single element: [5] -> 5.0
        assert_abs_diff_eq!(median_of(&[5.0]), 5.0, epsilon = 1e-12);
        // Empty: [] -> 0.0
        assert_abs_diff_eq!(median_of(&[]), 0.0, epsilon = 1e-12);
    }
}
