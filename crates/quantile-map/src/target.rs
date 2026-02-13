//! Target Gamma parameter computation from baseline fits and scenario factors.

use crate::config::QmConfig;
use crate::factors::ScenarioFactors;
use crate::fit::BaselineFit;
use crate::gamma::GammaParams;

/// Compute target Gamma parameters for each (year, month) by scaling baseline
/// moments with scenario factors.
///
/// For every year/month combination the baseline fit is combined with the
/// corresponding mean and variance factors.  When
/// [`QmConfig::scale_var_with_mean`] is `true` the variance factor is further
/// multiplied by `mean_factor²` so that the coefficient of variation is
/// preserved under pure mean scaling.
///
/// Returns a `Vec` of length `mean_factors.n_years()`, where each element is
/// a 12-element array (0-indexed by month) of `Option<GammaParams>`.
pub(crate) fn compute_target_params(
    baseline: &BaselineFit,
    mean_factors: &ScenarioFactors,
    var_factors: &ScenarioFactors,
    config: &QmConfig,
) -> Vec<[Option<GammaParams>; 12]> {
    let n_years = mean_factors.n_years();
    let mut result: Vec<[Option<GammaParams>; 12]> = Vec::with_capacity(n_years);

    for y in 0..n_years {
        let mut row: [Option<GammaParams>; 12] = [None; 12];

        for m in 1u8..=12 {
            // 1. Get baseline params; skip if None.
            let baseline_params = match baseline.params_for_month(m) {
                Some(p) => p,
                None => continue,
            };

            // 2. Look up the scenario factors for this year and month.
            let mean_factor = mean_factors.get(y as u32, m);
            let var_factor = var_factors.get(y as u32, m);

            // 3. Compute target moments.
            let target_mean = baseline_params.mean() * mean_factor;

            let effective_var_factor = if config.scale_var_with_mean() {
                var_factor * mean_factor * mean_factor
            } else {
                var_factor
            };
            let target_var = baseline_params.var() * effective_var_factor;

            // 4. Guard: both moments must be finite and positive.
            if !target_mean.is_finite()
                || target_mean <= 0.0
                || !target_var.is_finite()
                || target_var <= 0.0
            {
                continue;
            }

            // 5. Convert moments to Gamma parameters (may return None).
            row[(m - 1) as usize] = GammaParams::from_moments(target_mean, target_var);
        }

        result.push(row);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QmConfig;
    use crate::factors::ScenarioFactors;
    use crate::fit::BaselineFit;
    use crate::gamma::GammaParams;
    use approx::assert_relative_eq;

    /// Create a `BaselineFit` with known params for all 12 months.
    /// shape=2, scale=3 => mean=6, var=18.
    fn make_baseline() -> BaselineFit {
        let mut params: [Option<GammaParams>; 12] = [None; 12];
        for p in params.iter_mut() {
            *p = GammaParams::new(2.0, 3.0);
        }
        BaselineFit::new(params, vec![], 0)
    }

    /// Create a `BaselineFit` with month 6 (0-indexed 5) skipped.
    fn make_baseline_with_skip() -> BaselineFit {
        let mut params: [Option<GammaParams>; 12] = [None; 12];
        for (i, p) in params.iter_mut().enumerate() {
            if i != 5 {
                *p = GammaParams::new(2.0, 3.0);
            }
        }
        BaselineFit::new(params, vec![6], 0)
    }

    #[test]
    fn identity_factors() {
        let baseline = make_baseline();
        let n_years = 3;
        let mean_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
        let var_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
        let config = QmConfig::new();

        let targets = compute_target_params(&baseline, &mean_factors, &var_factors, &config);

        assert_eq!(targets.len(), n_years);
        for year_row in &targets {
            for (m, tp) in year_row.iter().enumerate() {
                let t = tp.expect("should have target params");
                let b = baseline.params_for_month((m + 1) as u8).unwrap();
                assert_relative_eq!(t.shape(), b.shape(), epsilon = 1e-10);
                assert_relative_eq!(t.scale(), b.scale(), epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn mean_scaling_preserves_cv() {
        let baseline = make_baseline();
        let n_years = 2;
        let mean_factors = ScenarioFactors::uniform(n_years, [2.0; 12]);
        let var_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
        let config = QmConfig::new().with_scale_var_with_mean(true);

        let targets = compute_target_params(&baseline, &mean_factors, &var_factors, &config);

        let b = baseline.params_for_month(1).unwrap();
        let cv_baseline = b.var().sqrt() / b.mean();

        for year_row in &targets {
            for tp in year_row {
                let t = tp.expect("should have target params");
                assert_relative_eq!(t.mean(), 12.0, epsilon = 1e-10);
                // var = 18 * (1.0 * 2.0^2) = 72.0
                assert_relative_eq!(t.var(), 72.0, epsilon = 1e-10);
                let cv_target = t.var().sqrt() / t.mean();
                assert_relative_eq!(cv_target, cv_baseline, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn mean_scaling_without_cv() {
        let baseline = make_baseline();
        let n_years = 2;
        let mean_factors = ScenarioFactors::uniform(n_years, [2.0; 12]);
        let var_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
        let config = QmConfig::new().with_scale_var_with_mean(false);

        let targets = compute_target_params(&baseline, &mean_factors, &var_factors, &config);

        let b = baseline.params_for_month(1).unwrap();
        let cv_baseline = b.var().sqrt() / b.mean();

        for year_row in &targets {
            for tp in year_row {
                let t = tp.expect("should have target params");
                assert_relative_eq!(t.mean(), 12.0, epsilon = 1e-10);
                // var unchanged: 18.0
                assert_relative_eq!(t.var(), 18.0, epsilon = 1e-10);
                // CV changes when variance is not scaled with mean.
                let cv_target = t.var().sqrt() / t.mean();
                assert!(
                    (cv_target - cv_baseline).abs() > 1e-6,
                    "CV should differ: target={cv_target}, baseline={cv_baseline}"
                );
            }
        }
    }

    #[test]
    fn variance_scaling() {
        let baseline = make_baseline();
        let n_years = 2;
        let mean_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
        let var_factors = ScenarioFactors::uniform(n_years, [2.0; 12]);
        let config = QmConfig::new().with_scale_var_with_mean(false);

        let targets = compute_target_params(&baseline, &mean_factors, &var_factors, &config);

        for year_row in &targets {
            for tp in year_row {
                let t = tp.expect("should have target params");
                assert_relative_eq!(t.mean(), 6.0, epsilon = 1e-10);
                assert_relative_eq!(t.var(), 36.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn combined_scaling() {
        let baseline = make_baseline();
        let n_years = 2;
        let mean_factors = ScenarioFactors::uniform(n_years, [1.5; 12]);
        let var_factors = ScenarioFactors::uniform(n_years, [1.3; 12]);
        let config = QmConfig::new().with_scale_var_with_mean(true);

        let targets = compute_target_params(&baseline, &mean_factors, &var_factors, &config);

        // target_mean = 6.0 * 1.5 = 9.0
        let expected_mean = 6.0 * 1.5;
        // effective_var_factor = 1.3 * 1.5^2 = 2.925
        // target_var = 18.0 * 2.925 = 52.65
        let expected_var = 18.0 * 1.3 * 1.5 * 1.5;

        for year_row in &targets {
            for tp in year_row {
                let t = tp.expect("should have target params");
                assert_relative_eq!(t.mean(), expected_mean, epsilon = 1e-10);
                assert_relative_eq!(t.var(), expected_var, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn scale_var_with_mean_comparison() {
        let baseline = make_baseline();
        let n_years = 2;
        let mean_factors = ScenarioFactors::uniform(n_years, [1.5; 12]);
        let var_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);

        let config_with = QmConfig::new().with_scale_var_with_mean(true);
        let config_without = QmConfig::new().with_scale_var_with_mean(false);

        let targets_with =
            compute_target_params(&baseline, &mean_factors, &var_factors, &config_with);
        let targets_without =
            compute_target_params(&baseline, &mean_factors, &var_factors, &config_without);

        for y in 0..n_years {
            for (tp_with, tp_without) in targets_with[y].iter().zip(targets_without[y].iter()) {
                let t_with = tp_with.expect("should have target params (with)");
                let t_without = tp_without.expect("should have target params (without)");
                assert!(
                    t_with.var() > t_without.var(),
                    "scale_var_with_mean=true should yield higher variance: {} vs {}",
                    t_with.var(),
                    t_without.var()
                );
            }
        }
    }

    #[test]
    fn skipped_months_propagate() {
        let baseline = make_baseline_with_skip();
        let n_years = 4;
        let mean_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
        let var_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
        let config = QmConfig::new();

        let targets = compute_target_params(&baseline, &mean_factors, &var_factors, &config);

        assert_eq!(targets.len(), n_years);
        for year_row in &targets {
            // Month 6 is 1-indexed => 0-indexed slot 5 should be None.
            assert!(
                year_row[5].is_none(),
                "Month 6 should be None in target (skipped in baseline)"
            );
            // Other months should be Some.
            for (m, tp) in year_row.iter().enumerate() {
                if m != 5 {
                    assert!(tp.is_some(), "Month {} should be Some", m + 1);
                }
            }
        }
    }

    #[test]
    fn all_target_params_positive() {
        let baseline = make_baseline();
        let n_years = 5;
        let mean_factors = ScenarioFactors::uniform(n_years, [1.2; 12]);
        let var_factors = ScenarioFactors::uniform(n_years, [0.8; 12]);
        let config = QmConfig::new();

        let targets = compute_target_params(&baseline, &mean_factors, &var_factors, &config);

        for year_row in &targets {
            for t in year_row.iter().flatten() {
                assert!(t.shape() > 0.0, "shape must be positive");
                assert!(t.scale() > 0.0, "scale must be positive");
                assert!(t.mean() > 0.0, "mean must be positive");
                assert!(t.var() > 0.0, "var must be positive");
            }
        }
    }
}
