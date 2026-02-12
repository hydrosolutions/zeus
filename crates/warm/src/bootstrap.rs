//! Block bootstrap and ARMA viability checks for the WARM pipeline.

use rand::Rng;

/// Returns `true` if the data slice `x` is suitable for ARMA fitting.
///
/// The checks performed are:
/// 1. All elements are finite (no NaN or infinity).
/// 2. Length is at least `max(8, max_p + max_q + 2)`.
/// 3. Sample standard deviation is at least 1e-8 (data is not constant).
/// 4. At least 3 unique values are present.
pub(crate) fn is_arma_viable(x: &[f64], max_p: usize, max_q: usize) -> bool {
    // 1. All elements must be finite.
    if !x.iter().all(|v| v.is_finite()) {
        return false;
    }

    // 2. Minimum length requirement.
    let min_len = 8_usize.max(max_p + max_q + 2);
    if x.len() < min_len {
        return false;
    }

    // 3. Sample standard deviation >= 1e-8.
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();
    if sd < 1e-8 {
        return false;
    }

    // 4. At least 3 unique values.
    let mut unique_bits: Vec<u64> = x.iter().map(|v| v.to_bits()).collect();
    unique_bits.sort_unstable();
    unique_bits.dedup();
    if unique_bits.len() < 3 {
        return false;
    }

    true
}

/// Generates block bootstrap resamples from the input series `x`.
///
/// Produces `n_sim` realisations, each of length `n`, by repeatedly sampling
/// contiguous blocks of data from `x` and concatenating them.
///
/// # Arguments
///
/// * `x`     - The original time series to resample from.
/// * `n`     - Desired length of each bootstrap replicate.
/// * `n_sim` - Number of replicates to generate.
/// * `rng`   - Random number generator.
///
/// # Block length
///
/// The block length `b` is `floor(sqrt(m))` clamped to `[3, m]`, where
/// `m = x.len()`. For very short series (`m < 3`), `b = m`.
pub(crate) fn block_bootstrap(
    x: &[f64],
    n: usize,
    n_sim: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<f64>> {
    let m = x.len();

    if m == 0 || n == 0 {
        return vec![vec![]; n_sim];
    }

    // Determine block length.
    let b = if m < 3 {
        m
    } else {
        let raw = (m as f64).sqrt().floor() as usize;
        raw.clamp(3, m)
    };

    let mut result = Vec::with_capacity(n_sim);

    for _ in 0..n_sim {
        let mut series = Vec::with_capacity(n);

        while series.len() < n {
            let start = if m == b {
                0
            } else {
                rng.random_range(0..=(m - b))
            };
            series.extend_from_slice(&x[start..start + b]);
        }

        series.truncate(n);
        result.push(series);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_is_arma_viable_basic() {
        let data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        assert!(is_arma_viable(&data, 2, 2));
    }

    #[test]
    fn test_is_arma_viable_too_short() {
        let data: Vec<f64> = (0..5).map(|i| i as f64).collect();
        // Needs >= max(8, 2+2+2) = 8, but only 5 elements.
        assert!(!is_arma_viable(&data, 2, 2));
    }

    #[test]
    fn test_is_arma_viable_constant() {
        let data = vec![3.125; 20];
        assert!(!is_arma_viable(&data, 2, 2));
    }

    #[test]
    fn test_is_arma_viable_nan() {
        let mut data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        data[5] = f64::NAN;
        assert!(!is_arma_viable(&data, 2, 2));
    }

    #[test]
    fn test_is_arma_viable_two_unique() {
        // Only 2 unique values.
        let data: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 1.0 } else { 2.0 })
            .collect();
        assert!(!is_arma_viable(&data, 2, 2));
    }

    #[test]
    fn test_block_bootstrap_shape() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut rng = StdRng::seed_from_u64(42);
        let result = block_bootstrap(&x, 50, 10, &mut rng);
        assert_eq!(result.len(), 10);
        for v in &result {
            assert_eq!(v.len(), 50);
        }
    }

    #[test]
    fn test_block_bootstrap_values_from_input() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut rng = StdRng::seed_from_u64(99);
        let result = block_bootstrap(&x, 50, 5, &mut rng);
        for series in &result {
            for val in series {
                assert!(x.contains(val), "value {} not found in input", val);
            }
        }
    }

    #[test]
    fn test_block_bootstrap_deterministic() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let r1 = block_bootstrap(&x, 50, 5, &mut rng1);
        let r2 = block_bootstrap(&x, 50, 5, &mut rng2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_block_bootstrap_edge_m1() {
        let x = vec![7.0];
        let mut rng = StdRng::seed_from_u64(42);
        let result = block_bootstrap(&x, 5, 3, &mut rng);
        assert_eq!(result.len(), 3);
        for v in &result {
            assert_eq!(v.len(), 5);
            assert!(v.iter().all(|&val| val == 7.0));
        }
    }

    #[test]
    fn test_block_bootstrap_edge_m2() {
        let x = vec![1.0, 2.0];
        let mut rng = StdRng::seed_from_u64(42);
        let result = block_bootstrap(&x, 6, 2, &mut rng);
        assert_eq!(result.len(), 2);
        for v in &result {
            assert_eq!(v.len(), 6);
            for val in v {
                assert!(*val == 1.0 || *val == 2.0);
            }
        }
    }

    #[test]
    fn test_block_bootstrap_n_zero() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut rng = StdRng::seed_from_u64(42);
        let result = block_bootstrap(&x, 0, 5, &mut rng);
        assert_eq!(result.len(), 5);
        for v in &result {
            assert!(v.is_empty());
        }
    }

    #[test]
    fn test_block_bootstrap_contiguous() {
        // Verify that blocks are contiguous slices of the input.
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut rng = StdRng::seed_from_u64(42);
        let result = block_bootstrap(&x, 50, 1, &mut rng);
        let series = &result[0];

        // Block length for m=100 is floor(sqrt(100)) = 10.
        let b = 10;
        // Check that each block of length b starting at block boundaries is contiguous.
        let mut i = 0;
        while i + b <= series.len() {
            let first = series[i];
            // Find this value in the original.
            if let Some(pos) = x.iter().position(|&v| v == first)
                && pos + b <= x.len()
            {
                assert_eq!(&series[i..i + b], &x[pos..pos + b]);
            }
            i += b;
        }
    }
}
