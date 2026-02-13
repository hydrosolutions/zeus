//! Baseline Gamma distribution fitting for monthly precipitation.

use crate::config::QmConfig;
use crate::gamma::GammaParams;

/// Result of fitting Gamma distributions to monthly precipitation data.
///
/// Contains one optional [`GammaParams`] per calendar month (12 total),
/// along with metadata about which months were skipped and how many fits
/// failed.
#[derive(Debug, Clone)]
pub struct BaselineFit {
    params: [Option<GammaParams>; 12],
    skipped_months: Vec<u8>,
    n_failed_fits: usize,
}

impl BaselineFit {
    /// Creates a new `BaselineFit` from pre-computed arrays.
    pub(crate) fn new(
        params: [Option<GammaParams>; 12],
        skipped_months: Vec<u8>,
        n_failed_fits: usize,
    ) -> Self {
        Self {
            params,
            skipped_months,
            n_failed_fits,
        }
    }

    /// Returns the fitted Gamma parameters for a 1-indexed calendar month.
    ///
    /// # Panics
    ///
    /// Panics if `month` is 0 or greater than 12.
    pub fn params_for_month(&self, month: u8) -> Option<GammaParams> {
        assert!(
            (1..=12).contains(&month),
            "month must be in 1..=12, got {month}"
        );
        self.params[(month - 1) as usize]
    }

    /// Returns a reference to the full 12-element parameter array (0-indexed).
    pub fn params(&self) -> &[Option<GammaParams>; 12] {
        &self.params
    }

    /// Returns 1-indexed months that have successfully fitted parameters.
    pub fn fitted_months(&self) -> Vec<u8> {
        self.params
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.as_ref().map(|_| (i + 1) as u8))
            .collect()
    }

    /// Returns the 1-indexed months that were skipped during fitting.
    pub fn skipped_months(&self) -> &[u8] {
        &self.skipped_months
    }

    /// Returns the number of months where fitting was attempted but failed.
    pub fn n_failed_fits(&self) -> usize {
        self.n_failed_fits
    }

    /// Returns `true` if all 12 months are `None` (no successful fits).
    pub fn is_empty(&self) -> bool {
        self.params.iter().all(|p| p.is_none())
    }
}

/// Fit a Gamma distribution to a slice of (wet-day) values using the
/// method of moments.
///
/// Returns `None` if there are fewer than 3 unique values, the sample
/// variance is near zero, or the moment estimates produce invalid parameters.
pub(crate) fn fit_gamma_mme(values: &[f64]) -> Option<GammaParams> {
    let n = values.len();
    if n < 2 {
        return None;
    }

    let mean = values.iter().copied().sum::<f64>() / n as f64;

    let variance = values
        .iter()
        .copied()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;

    if count_unique(values) < 3 {
        return None;
    }

    if variance <= 1e-10 {
        return None;
    }

    GammaParams::from_moments(mean, variance)
}

/// Counts the number of unique values in a slice, using an epsilon
/// tolerance of 1e-10 for floating-point comparison.
fn count_unique(values: &[f64]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    sorted
        .windows(2)
        .filter(|w| (w[1] - w[0]).abs() > 1e-10)
        .count()
        + 1
}

/// Fit Gamma distributions to monthly precipitation data.
///
/// For each calendar month (1..=12), collects wet-day values that exceed
/// `config.intensity_threshold()`, then fits a Gamma distribution via
/// method-of-moments if enough events are present.
pub(crate) fn fit_monthly(precip: &[f64], month: &[u8], config: &QmConfig) -> BaselineFit {
    let mut params: [Option<GammaParams>; 12] = [None; 12];
    let mut skipped_months = Vec::new();
    let mut n_failed_fits = 0usize;

    for m in 1u8..=12 {
        let wet_values: Vec<f64> = precip
            .iter()
            .zip(month.iter())
            .filter(|&(&p, &mo)| mo == m && p > config.intensity_threshold() && !p.is_nan())
            .map(|(&p, _)| p)
            .collect();

        if wet_values.len() < config.min_events() {
            skipped_months.push(m);
            params[(m - 1) as usize] = None;
        } else {
            match fit_gamma_mme(&wet_values) {
                Some(gp) => {
                    params[(m - 1) as usize] = Some(gp);
                }
                None => {
                    n_failed_fits += 1;
                    skipped_months.push(m);
                    params[(m - 1) as usize] = None;
                }
            }
        }
    }

    BaselineFit::new(params, skipped_months, n_failed_fits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Gamma as GammaDist};

    #[test]
    fn fit_12_months_synthetic() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let years = 10;
        let days_per_month = 30;
        let total_days = years * 12 * days_per_month;

        let mut precip = Vec::with_capacity(total_days);
        let mut months = Vec::with_capacity(total_days);

        for _year in 0..years {
            for m in 1u8..=12 {
                let shape = 1.0 + m as f64 * 0.3;
                let scale = 2.0 + m as f64 * 0.2;
                let dist = GammaDist::new(shape, scale).unwrap();
                for _ in 0..days_per_month {
                    let value: f64 = dist.sample(&mut rng);
                    precip.push(value);
                    months.push(m);
                }
            }
        }

        let config = QmConfig::new().with_min_events(5);
        let fit = fit_monthly(&precip, &months, &config);

        assert!(!fit.is_empty());
        assert!(fit.skipped_months().is_empty());
        assert_eq!(fit.n_failed_fits(), 0);
        assert_eq!(fit.fitted_months().len(), 12);

        for m in 1u8..=12 {
            assert!(
                fit.params_for_month(m).is_some(),
                "Expected Some for month {m}"
            );
        }
    }

    #[test]
    fn seasonal_mean_recovery() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let shape = 2.0;
        let scale = 3.0;
        let expected_mean = shape * scale; // 6.0
        let dist = GammaDist::new(shape, scale).unwrap();

        let n = 500;
        let values: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();

        let fitted = fit_gamma_mme(&values).expect("fit should succeed");
        let recovered_mean = fitted.mean();

        assert_relative_eq!(
            recovered_mean,
            expected_mean,
            epsilon = expected_mean * 0.15
        );
    }

    #[test]
    fn skip_insufficient_data() {
        // Month 5 gets only 3 wet-day values (below min_events=10 default).
        // Other months get plenty of data.
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let dist = GammaDist::new(2.0, 3.0).unwrap();

        let mut precip = Vec::new();
        let mut months = Vec::new();

        for m in 1u8..=12 {
            let count = if m == 5 { 3 } else { 50 };
            for _ in 0..count {
                precip.push(dist.sample(&mut rng));
                months.push(m);
            }
        }

        let config = QmConfig::new().with_min_events(10);
        let fit = fit_monthly(&precip, &months, &config);

        assert!(fit.skipped_months().contains(&5));
        assert!(fit.params_for_month(5).is_none());
        // Other months should be fitted
        for m in 1u8..=12 {
            if m != 5 {
                assert!(
                    fit.params_for_month(m).is_some(),
                    "Expected Some for month {m}"
                );
            }
        }
    }

    #[test]
    fn skip_zero_variance() {
        // Month 3 has all identical wet values (5.0). fit_gamma_mme should
        // return None because there are fewer than 3 unique values.
        let mut rng = rand::rngs::StdRng::seed_from_u64(55);
        let dist = GammaDist::new(2.0, 3.0).unwrap();

        let mut precip = Vec::new();
        let mut months = Vec::new();

        for m in 1u8..=12 {
            if m == 3 {
                for _ in 0..50 {
                    precip.push(5.0);
                    months.push(m);
                }
            } else {
                for _ in 0..50 {
                    precip.push(dist.sample(&mut rng));
                    months.push(m);
                }
            }
        }

        let config = QmConfig::new().with_min_events(10);
        let fit = fit_monthly(&precip, &months, &config);

        assert!(fit.skipped_months().contains(&3));
        assert!(fit.params_for_month(3).is_none());
        assert!(fit.n_failed_fits() >= 1);
    }

    #[test]
    fn respects_intensity_threshold() {
        // With threshold = 1.0, values <= 1.0 should be excluded.
        let mut rng = rand::rngs::StdRng::seed_from_u64(33);
        let dist = GammaDist::new(2.0, 3.0).unwrap();

        let mut precip = Vec::new();
        let mut months = Vec::new();

        // Month 7: mix of sub-threshold (0.5) and above-threshold values.
        // Add exactly 5 sub-threshold values and 50 real values.
        for _ in 0..5 {
            precip.push(0.5);
            months.push(7u8);
        }
        for _ in 0..50 {
            // Ensure values are above the threshold by adding offset.
            let v: f64 = dist.sample(&mut rng) + 1.1;
            precip.push(v);
            months.push(7u8);
        }

        // Fill other months with valid data.
        for m in 1u8..=12 {
            if m == 7 {
                continue;
            }
            for _ in 0..50 {
                let v: f64 = dist.sample(&mut rng) + 1.1;
                precip.push(v);
                months.push(m);
            }
        }

        let config = QmConfig::new()
            .with_intensity_threshold(1.0)
            .with_min_events(10);
        let fit = fit_monthly(&precip, &months, &config);

        // Month 7 should still be fitted because there are 50 values > 1.0.
        assert!(fit.params_for_month(7).is_some());

        // The 0.5 values should not influence the fit.
        // Verify the mean is above the threshold (since all wet values > 1.1).
        let gp = fit.params_for_month(7).unwrap();
        assert!(gp.mean() > 1.0, "mean should be well above threshold");
    }

    #[test]
    fn nan_excluded() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(88);
        let dist = GammaDist::new(2.0, 3.0).unwrap();

        let mut precip = Vec::new();
        let mut months = Vec::new();

        for m in 1u8..=12 {
            for _ in 0..50 {
                precip.push(dist.sample(&mut rng));
                months.push(m);
            }
            // Inject NaN values for every month
            for _ in 0..5 {
                precip.push(f64::NAN);
                months.push(m);
            }
        }

        let config = QmConfig::new().with_min_events(10);
        let fit = fit_monthly(&precip, &months, &config);

        // All months should still fit successfully despite NaN values.
        assert!(fit.skipped_months().is_empty());
        assert_eq!(fit.n_failed_fits(), 0);
        for m in 1u8..=12 {
            assert!(
                fit.params_for_month(m).is_some(),
                "Expected Some for month {m} (NaN should be excluded)"
            );
        }
    }
}
