// TODO: kurtosis, tail-mass metrics, area averages, distribution fitting

//! Statistical helper functions for the Zeus weather generator.

/// Arithmetic mean of a slice. Returns 0.0 if empty.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

/// Sample variance with N-1 denominator (matching R's `var()`).
/// Returns 0.0 if fewer than 2 elements.
pub fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = data.iter().sum::<f64>() / nf;
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (nf - 1.0)
}

/// Sample standard deviation with N-1 denominator (matching R's `sd()`).
/// Returns 0.0 if fewer than 2 elements.
pub fn sd(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// R's default quantile algorithm (type=7).
///
/// **Expects pre-sorted input** (caller's responsibility).
///
/// # Panics
///
/// Panics if `sorted` is empty.
pub fn quantile_type7(sorted: &[f64], p: f64) -> f64 {
    assert!(
        !sorted.is_empty(),
        "quantile_type7: input must not be empty"
    );
    let n = sorted.len();
    let h = (n - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (h - h.floor()) * (sorted[hi] - sorted[lo])
}

/// Median of pre-sorted data. For even length, averages the middle two values.
///
/// # Panics
///
/// Panics if `sorted` is empty.
pub fn median(sorted: &[f64]) -> f64 {
    assert!(!sorted.is_empty(), "median: input must not be empty");
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Cascade measure of spread:
///
/// 1. IQR (Q75 - Q25). If IQR > 1e-10, return IQR.
/// 2. MAD with constant=1 (median of |x - median(x)|). If MAD > 1e-10, return MAD.
/// 3. SD. If SD > 1e-10, return SD.
/// 4. Fallback: 1.0.
pub fn robust_scale(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // 1. IQR
    let q25 = quantile_type7(&sorted, 0.25);
    let q75 = quantile_type7(&sorted, 0.75);
    let iqr = q75 - q25;
    if iqr > 1e-10 {
        return iqr;
    }

    // 2. MAD (constant = 1)
    let med = median(&sorted);
    let mut abs_devs: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = median(&abs_devs);
    if mad > 1e-10 {
        return mad;
    }

    // 3. SD
    let s = sd(data);
    if s > 1e-10 {
        return s;
    }

    // 4. Fallback
    1.0
}

/// Pearson correlation coefficient.
///
/// Filters to indices where both `x[i]` and `y[i]` are finite.
/// Returns `None` if fewer than 3 finite pairs or if the denominator is zero
/// (constant input).
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
        .map(|(xi, yi)| (*xi, *yi))
        .collect();

    if pairs.len() < 3 {
        return None;
    }

    let n = pairs.len() as f64;
    let mx: f64 = pairs.iter().map(|(xi, _)| xi).sum::<f64>() / n;
    let my: f64 = pairs.iter().map(|(_, yi)| yi).sum::<f64>() / n;

    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    for &(xi, yi) in &pairs {
        let dx = xi - mx;
        let dy = yi - my;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    let denom = (sum_xx * sum_yy).sqrt();
    if denom == 0.0 {
        return None;
    }

    Some(sum_xy / denom)
}

/// Skewness using R's `e1071::skewness` type 1 formula.
///
/// Computes `m3 / m2^1.5` where `m2` and `m3` are population moments
/// (dividing by `n`, not `n-1`).
///
/// Returns `None` if fewer than 3 elements or if `m2 < 1e-20`.
pub fn skewness(data: &[f64]) -> Option<f64> {
    let n = data.len();
    if n < 3 {
        return None;
    }

    let nf = n as f64;
    let mean: f64 = data.iter().sum::<f64>() / nf;

    let m2: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nf;
    if m2 < 1e-20 {
        return None;
    }

    let m3: f64 = data.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / nf;

    Some(m3 / m2.powf(1.5))
}

/// Spell lengths (run-length encoding of wet/dry sequences).
///
/// Returns `(wet_spells, dry_spells)` where each vector contains the lengths
/// of consecutive runs of `true` (wet) or `false` (dry) values.
///
/// Empty input returns two empty vectors.
pub fn spell_lengths(wet: &[bool]) -> (Vec<usize>, Vec<usize>) {
    if wet.is_empty() {
        return (vec![], vec![]);
    }

    let mut wet_spells = Vec::new();
    let mut dry_spells = Vec::new();

    let mut current_state = wet[0];
    let mut current_len = 1;

    for &w in &wet[1..] {
        if w == current_state {
            current_len += 1;
        } else {
            // Run ended, record it
            if current_state {
                wet_spells.push(current_len);
            } else {
                dry_spells.push(current_len);
            }
            current_state = w;
            current_len = 1;
        }
    }

    // Record the final run
    if current_state {
        wet_spells.push(current_len);
    } else {
        dry_spells.push(current_len);
    }

    (wet_spells, dry_spells)
}

/// Mean absolute error between observed and simulated values.
///
/// Computes `sum(|obs[i] - sim[i]|) / n`.
///
/// # Panics
///
/// Panics if the slices are empty or have different lengths.
pub fn mae(observed: &[f64], simulated: &[f64]) -> f64 {
    assert!(
        !observed.is_empty(),
        "mae: observed slice must not be empty"
    );
    assert!(
        observed.len() == simulated.len(),
        "mae: observed and simulated slices must have the same length"
    );

    let sum_abs_diff: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| (o - s).abs())
        .sum();

    sum_abs_diff / observed.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mean() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_relative_eq!(mean(&data), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mean_empty() {
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_sd() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_relative_eq!(sd(&data), 2.138090, epsilon = 1e-6);
    }

    #[test]
    fn test_sd_single() {
        assert_eq!(sd(&[5.0]), 0.0);
    }

    #[test]
    fn test_quantile_type7() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(quantile_type7(&sorted, 0.25), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_quantile_type7_median() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(quantile_type7(&sorted, 0.5), 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_median_odd() {
        assert_relative_eq!(median(&[1.0, 2.0, 3.0]), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_median_even() {
        assert_relative_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5, epsilon = 1e-6);
    }

    #[test]
    fn test_robust_scale_iqr() {
        // Data with a clear IQR spread
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let scale = robust_scale(&data);
        // IQR = Q75 - Q25; for 1..=10, Q25=3.25, Q75=7.75 => IQR=4.5
        assert_relative_eq!(scale, 4.5, epsilon = 1e-6);
    }

    #[test]
    fn test_robust_scale_constant() {
        assert_relative_eq!(robust_scale(&[5.0, 5.0, 5.0]), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pearson_correlation_perfect() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert_relative_eq!(r.unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pearson_correlation_insufficient() {
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];
        assert!(pearson_correlation(&x, &y).is_none());
    }

    #[test]
    fn test_pearson_correlation_with_nan() {
        let x = [1.0, f64::NAN, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, f64::NAN, 8.0, 10.0];
        // Finite pairs: (1,2), (4,8), (5,10) — 3 pairs, perfect linear
        let r = pearson_correlation(&x, &y);
        assert_relative_eq!(r.unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_variance_basic() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // variance = sd^2 = 2.138090^2 ≈ 4.571429
        assert_relative_eq!(variance(&data), 4.571429, epsilon = 1e-4);
    }

    #[test]
    fn test_variance_empty() {
        assert_eq!(variance(&[]), 0.0);
    }

    #[test]
    fn test_variance_single() {
        assert_eq!(variance(&[5.0]), 0.0);
    }

    #[test]
    fn test_variance_two() {
        // [3.0, 7.0]: mean=5, sum_sq=8, var=8/1=8
        assert_relative_eq!(variance(&[3.0, 7.0]), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_type7_p0() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(quantile_type7(&sorted, 0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_type7_p1() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(quantile_type7(&sorted, 1.0), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_type7_interpolation() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        // p=0.1 → h=0.4, lo=0, hi=1 → 1 + 0.4*(2-1) = 1.4
        assert_relative_eq!(quantile_type7(&sorted, 0.1), 1.4, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_type7_r_crossvalidation() {
        // R: quantile(1:10, 0.3, type=7) = 3.7
        let sorted: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        assert_relative_eq!(quantile_type7(&sorted, 0.3), 3.7, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "quantile_type7: input must not be empty")]
    fn test_quantile_type7_empty_panics() {
        quantile_type7(&[], 0.5);
    }

    #[test]
    #[should_panic(expected = "median: input must not be empty")]
    fn test_median_empty_panics() {
        median(&[]);
    }

    #[test]
    fn test_skewness_symmetric() {
        let data = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let sk = skewness(&data).unwrap();
        assert_relative_eq!(sk, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_skewness_right_skewed() {
        let data = [1.0, 1.0, 1.0, 1.0, 10.0];
        let sk = skewness(&data).unwrap();
        assert!(sk > 0.0, "Expected positive skewness, got {}", sk);
    }

    #[test]
    fn test_skewness_insufficient() {
        assert!(skewness(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_skewness_constant() {
        let data = [5.0, 5.0, 5.0, 5.0];
        assert!(skewness(&data).is_none());
    }

    #[test]
    fn test_spell_lengths_basic() {
        let wet = [true, true, false, true, false, false, false];
        let (wet_spells, dry_spells) = spell_lengths(&wet);
        assert_eq!(wet_spells, vec![2, 1]);
        assert_eq!(dry_spells, vec![1, 3]);
    }

    #[test]
    fn test_spell_lengths_empty() {
        let (wet_spells, dry_spells) = spell_lengths(&[]);
        assert_eq!(wet_spells, Vec::<usize>::new());
        assert_eq!(dry_spells, Vec::<usize>::new());
    }

    #[test]
    fn test_spell_lengths_all_wet() {
        let wet = [true, true, true];
        let (wet_spells, dry_spells) = spell_lengths(&wet);
        assert_eq!(wet_spells, vec![3]);
        assert_eq!(dry_spells, Vec::<usize>::new());
    }

    #[test]
    fn test_spell_lengths_all_dry() {
        let wet = [false, false];
        let (wet_spells, dry_spells) = spell_lengths(&wet);
        assert_eq!(wet_spells, Vec::<usize>::new());
        assert_eq!(dry_spells, vec![2]);
    }

    #[test]
    fn test_mae_basic() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [1.5, 2.5, 3.5];
        let error = mae(&obs, &sim);
        assert_relative_eq!(error, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_mae_identical() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [1.0, 2.0, 3.0];
        let error = mae(&obs, &sim);
        assert_relative_eq!(error, 0.0, epsilon = 1e-10);
    }
}
