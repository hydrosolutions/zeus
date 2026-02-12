//! Statistical significance testing for the Global Wavelet Spectrum.
//!
//! GWS values and thresholds are computed in original (pre-standardization) units
//! using sample variance (N-1 denominator). Tests GWS peaks against a red-noise
//! (AR(1)) or white-noise background using the chi-squared method from
//! Torrence & Compo (1998, Sections 4-5).

use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::f64::consts::PI;

use crate::cwt::CwtResult;
use crate::error::WaveletError;
use crate::mra::variance;
use crate::series::TimeSeries;

/// Background noise model for significance testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum NoiseModel {
    /// Red noise (AR(1) process).
    #[default]
    Red,
    /// White noise (uncorrelated).
    White,
}

/// Configuration for GWS significance testing.
///
/// Use the builder methods to customize the test parameters.
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::SignificanceConfig;
///
/// let config = SignificanceConfig::new()
///     .with_significance_level(0.95)
///     .with_noise_model(NoiseModel::White);
/// ```
#[derive(Clone, Debug)]
pub struct SignificanceConfig {
    /// Confidence level for the chi-squared test.
    significance_level: f64,
    /// Background noise model.
    noise_model: NoiseModel,
    /// Minimum degrees of freedom (2 for Morlet).
    dof_min: f64,
    /// Decorrelation factor (Torrence & Compo).
    gamma_fac: f64,
}

impl SignificanceConfig {
    /// Creates a new `SignificanceConfig` with default parameters.
    ///
    /// Defaults: `significance_level = 0.90`, `noise_model = Red`,
    /// `dof_min = 2.0`, `gamma_fac = 2.32`.
    pub fn new() -> Self {
        Self {
            significance_level: 0.90,
            noise_model: NoiseModel::Red,
            dof_min: 2.0,
            gamma_fac: 2.32,
        }
    }

    /// Sets the significance (confidence) level.
    pub fn with_significance_level(mut self, level: f64) -> Self {
        self.significance_level = level;
        self
    }

    /// Sets the background noise model.
    pub fn with_noise_model(mut self, model: NoiseModel) -> Self {
        self.noise_model = model;
        self
    }

    /// Sets the minimum degrees of freedom.
    pub fn with_dof_min(mut self, dof_min: f64) -> Self {
        self.dof_min = dof_min;
        self
    }

    /// Sets the decorrelation factor.
    pub fn with_gamma_fac(mut self, gamma_fac: f64) -> Self {
        self.gamma_fac = gamma_fac;
        self
    }

    /// Returns the significance level.
    pub fn significance_level(&self) -> f64 {
        self.significance_level
    }

    /// Returns the noise model.
    pub fn noise_model(&self) -> NoiseModel {
        self.noise_model
    }

    /// Returns the minimum degrees of freedom.
    pub fn dof_min(&self) -> f64 {
        self.dof_min
    }

    /// Returns the decorrelation factor.
    pub fn gamma_fac(&self) -> f64 {
        self.gamma_fac
    }
}

impl Default for SignificanceConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a GWS significance test.
///
/// All GWS values and thresholds are in original (pre-standardization) units.
/// See [`CwtResult::global_wavelet_spectrum`](crate::cwt::CwtResult::global_wavelet_spectrum)
/// for the standardized-unit counterpart.
#[derive(Clone, Debug)]
pub struct GwsResult {
    /// Periods (one per scale).
    periods: Vec<f64>,
    /// COI-masked Global Wavelet Spectrum (original units).
    gws_masked: Vec<f64>,
    /// Full (unmasked) Global Wavelet Spectrum (original units).
    gws_unmasked: Vec<f64>,
    /// Chi-squared significance threshold per scale (original units).
    significance_threshold: Vec<f64>,
    /// Whether each scale's GWS exceeds the threshold.
    significant: Vec<bool>,
    /// Estimated lag-1 autocorrelation of the input series.
    lag1: f64,
    /// Red-noise theoretical spectrum values.
    theoretical_spectrum: Vec<f64>,
    /// Effective degrees of freedom per scale.
    effective_dof: Vec<f64>,
    /// Number of reliable (inside COI) time points per scale.
    n_coi: Vec<usize>,
}

impl GwsResult {
    /// Creates a new `GwsResult` (crate-internal constructor).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        periods: Vec<f64>,
        gws_masked: Vec<f64>,
        gws_unmasked: Vec<f64>,
        significance_threshold: Vec<f64>,
        significant: Vec<bool>,
        lag1: f64,
        theoretical_spectrum: Vec<f64>,
        effective_dof: Vec<f64>,
        n_coi: Vec<usize>,
    ) -> Self {
        Self {
            periods,
            gws_masked,
            gws_unmasked,
            significance_threshold,
            significant,
            lag1,
            theoretical_spectrum,
            effective_dof,
            n_coi,
        }
    }

    /// Returns the periods (one per scale).
    pub fn periods(&self) -> &[f64] {
        &self.periods
    }

    /// Returns the COI-masked Global Wavelet Spectrum (original units, rescaled from standardized power by `variance`).
    pub fn gws_masked(&self) -> &[f64] {
        &self.gws_masked
    }

    /// Returns the full (unmasked) Global Wavelet Spectrum (original units, rescaled from standardized power by `variance`).
    pub fn gws_unmasked(&self) -> &[f64] {
        &self.gws_unmasked
    }

    /// Returns the significance threshold per scale (original units; `theoretical_spectrum * variance * chi2 / dof`).
    pub fn significance_threshold(&self) -> &[f64] {
        &self.significance_threshold
    }

    /// Returns whether each scale's GWS exceeds the threshold.
    pub fn significant(&self) -> &[bool] {
        &self.significant
    }

    /// Returns the estimated lag-1 autocorrelation.
    pub fn lag1(&self) -> f64 {
        self.lag1
    }

    /// Returns the red-noise theoretical spectrum values.
    pub fn theoretical_spectrum(&self) -> &[f64] {
        &self.theoretical_spectrum
    }

    /// Returns the effective degrees of freedom per scale.
    pub fn effective_dof(&self) -> &[f64] {
        &self.effective_dof
    }

    /// Returns the number of reliable time points per scale.
    pub fn n_coi(&self) -> &[usize] {
        &self.n_coi
    }

    /// Returns the number of scales.
    pub fn n_scales(&self) -> usize {
        self.periods.len()
    }

    /// Returns the indices of scales that are significant.
    pub fn significant_indices(&self) -> Vec<usize> {
        self.significant
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s)
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns the periods that are significant.
    pub fn significant_periods(&self) -> Vec<f64> {
        self.significant
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s)
            .map(|(i, _)| self.periods[i])
            .collect()
    }
}

/// Tests the Global Wavelet Spectrum for significance against a background noise model.
///
/// GWS values are rescaled from standardized CWT power to original units by multiplying
/// by the sample variance of the input data. The significance threshold is also in
/// original units, so the comparison is scale-invariant.
///
/// # Errors
///
/// | Variant | Trigger |
/// |---------|---------|
/// | [`WaveletError::SignificanceFailed`] | invalid config or chi-squared computation failure |
pub fn test_significance(
    series: &TimeSeries,
    cwt_result: &CwtResult,
    config: &SignificanceConfig,
) -> Result<GwsResult, WaveletError> {
    // 1. Validate config
    if config.significance_level() <= 0.0 || config.significance_level() >= 1.0 {
        return Err(WaveletError::SignificanceFailed(
            "significance_level must be in (0, 1)".to_string(),
        ));
    }

    // 2. Estimate lag-1
    let lag1 = match config.noise_model() {
        NoiseModel::Red => estimate_lag1(series.as_slice()),
        NoiseModel::White => 0.0,
    };

    // 3. Data variance
    let data_variance = variance(series.as_slice());

    // 4. Red-noise spectrum
    let dt = cwt_result.dt();
    let periods = cwt_result.periods();
    let scales = cwt_result.scales();
    let coi = cwt_result.coi();
    let theoretical_spectrum = red_noise_spectrum(lag1, dt, periods);

    // 5. COI mask
    let (mask, n_coi) = compute_coi_mask(periods, coi);

    // 6. Power
    let power = cwt_result.power();
    let n_times = cwt_result.n_times();
    let n_scales = cwt_result.n_scales();

    // 7. GWS (masked)
    let mut gws_masked = vec![0.0; n_scales];
    for j in 0..n_scales {
        let mut sum = 0.0;
        for t in 0..n_times {
            if mask[j][t] {
                sum += power[j][t];
            }
        }
        gws_masked[j] = data_variance * sum / n_coi[j].max(1) as f64;
    }

    // 8. GWS (unmasked)
    let mut gws_unmasked = vec![0.0; n_scales];
    for j in 0..n_scales {
        let sum: f64 = power[j].iter().sum();
        gws_unmasked[j] = data_variance * sum / n_times as f64;
    }

    // 9. Effective DOF
    let dof = effective_dof(&n_coi, dt, scales, config.gamma_fac(), config.dof_min());

    // 10. Chi-squared threshold
    let mut threshold = vec![0.0; n_scales];
    for j in 0..n_scales {
        let chisq_dist =
            ChiSquared::new(dof[j]).map_err(|e| WaveletError::SignificanceFailed(e.to_string()))?;
        let chisq_val = chisq_dist.inverse_cdf(config.significance_level()) / dof[j];

        // 11. Significance threshold
        threshold[j] = theoretical_spectrum[j] * data_variance * chisq_val;
    }

    // 12. Decision
    let significant: Vec<bool> = (0..n_scales)
        .map(|j| n_coi[j] > 0 && gws_masked[j] > threshold[j])
        .collect();

    Ok(GwsResult::new(
        periods.to_vec(),
        gws_masked,
        gws_unmasked,
        threshold,
        significant,
        lag1,
        theoretical_spectrum,
        dof,
        n_coi,
    ))
}

/// Estimates lag-1 autocorrelation via the Yule-Walker method.
///
/// Mean-centers the data, computes the ratio of autocovariance at lag 1
/// to autocovariance at lag 0, and clamps the result to `[0.0, 0.99]`.
fn estimate_lag1(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    let sum_sq: f64 = centered.iter().map(|&x| x * x).sum();
    if sum_sq < f64::EPSILON {
        return 0.0;
    }

    let sum_cross: f64 = centered.windows(2).map(|w| w[0] * w[1]).sum();
    let lag1 = sum_cross / sum_sq;
    lag1.clamp(0.0, 0.99)
}

/// Computes the red-noise theoretical spectrum (Torrence & Compo eq. 16).
fn red_noise_spectrum(lag1: f64, dt: f64, periods: &[f64]) -> Vec<f64> {
    periods
        .iter()
        .map(|&period| {
            let cos_term = (2.0 * PI * dt / period).cos();
            (1.0 - lag1 * lag1) / (1.0 - 2.0 * lag1 * cos_term + lag1 * lag1)
        })
        .collect()
}

/// Computes the COI mask and counts reliable time points per scale.
///
/// Returns `(mask, n_coi)` where `mask[j][t]` is `true` if `period[j] <= coi[t]`.
fn compute_coi_mask(periods: &[f64], coi: &[f64]) -> (Vec<Vec<bool>>, Vec<usize>) {
    let n_scales = periods.len();
    let mut mask = Vec::with_capacity(n_scales);
    let mut n_coi = Vec::with_capacity(n_scales);

    for &period in periods {
        let row: Vec<bool> = coi.iter().map(|&c| period <= c).collect();
        let count = row.iter().filter(|&&v| v).count();
        mask.push(row);
        n_coi.push(count);
    }

    (mask, n_coi)
}

/// Computes effective degrees of freedom per scale.
fn effective_dof(
    n_coi: &[usize],
    dt: f64,
    scales: &[f64],
    gamma_fac: f64,
    dof_min: f64,
) -> Vec<f64> {
    n_coi
        .iter()
        .zip(scales.iter())
        .map(|(&nc, &scale)| {
            let n_eff = (nc as f64 * dt) / (gamma_fac * scale);
            f64::max(dof_min, dof_min * n_eff)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CwtConfig, TimeSeries, cwt_morlet};

    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    #[test]
    fn lag1_known_ar1() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut x = vec![0.0_f64; 5000];
        for i in 1..5000 {
            x[i] = 0.7 * x[i - 1]
                + <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng);
        }
        let lag1 = estimate_lag1(&x);
        assert!(
            (lag1 - 0.7).abs() < 0.1,
            "estimated lag1 = {}, expected ~0.7",
            lag1
        );
    }

    #[test]
    fn lag1_white_noise_near_zero() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let x: Vec<f64> = (0..5000)
            .map(|_| <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng))
            .collect();
        let lag1 = estimate_lag1(&x);
        assert!(
            lag1 < 0.1,
            "estimated lag1 = {}, expected < 0.1 for white noise",
            lag1
        );
    }

    #[test]
    fn lag1_constant_signal() {
        let x = vec![42.0; 100];
        let lag1 = estimate_lag1(&x);
        assert!(
            (lag1 - 0.0).abs() < f64::EPSILON,
            "constant signal lag1 = {}, expected 0.0",
            lag1
        );
    }

    #[test]
    fn lag1_clamps_negative() {
        // Alternating +1/-1 has negative autocorrelation
        let x: Vec<f64> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let lag1 = estimate_lag1(&x);
        assert!(
            (lag1 - 0.0).abs() < f64::EPSILON,
            "alternating signal lag1 = {}, expected 0.0 (clamped)",
            lag1
        );
    }

    #[test]
    fn red_noise_white_is_flat() {
        let periods: Vec<f64> = (1..=20).map(|i| i as f64 * 2.0).collect();
        let spectrum = red_noise_spectrum(0.0, 1.0, &periods);
        for (i, &val) in spectrum.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-12,
                "spectrum[{}] = {}, expected 1.0 for white noise",
                i,
                val
            );
        }
    }

    #[test]
    fn red_noise_known_value() {
        let lag1 = 0.5;
        let dt = 1.0;
        let period = 4.0;
        let periods = vec![period];
        let spectrum = red_noise_spectrum(lag1, dt, &periods);

        // Manual computation: cos(2*pi*1.0/4.0) = cos(pi/2) = 0
        // P = (1 - 0.25) / (1 - 0 + 0.25) = 0.75 / 1.25 = 0.6
        let expected = 0.75 / 1.25;
        assert!(
            (spectrum[0] - expected).abs() < 1e-12,
            "spectrum = {}, expected {}",
            spectrum[0],
            expected
        );
    }

    #[test]
    fn red_noise_increases_with_period() {
        let lag1 = 0.7;
        let dt = 1.0;
        let periods: Vec<f64> = (1..=50).map(|i| i as f64 * 2.0).collect();
        let spectrum = red_noise_spectrum(lag1, dt, &periods);

        // Red noise has more power at low frequencies (long periods)
        assert!(
            spectrum.last().unwrap() > spectrum.first().unwrap(),
            "spectrum at longest period ({}) should exceed shortest ({})",
            spectrum.last().unwrap(),
            spectrum.first().unwrap()
        );
    }

    #[test]
    fn coi_mask_short_periods_all_inside() {
        // Very short periods should be inside COI for all time points
        let periods = vec![0.01];
        let coi = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let (mask, n_coi) = compute_coi_mask(&periods, &coi);

        assert!(mask[0].iter().all(|&v| v));
        assert_eq!(n_coi[0], 5);
    }

    #[test]
    fn coi_mask_long_periods_edges_excluded() {
        // Very long periods should exclude edge points where COI is small
        let periods = vec![100.0];
        let coi = vec![1.0, 10.0, 200.0, 10.0, 1.0];
        let (mask, n_coi) = compute_coi_mask(&periods, &coi);

        // Only t=2 (coi=200) should be inside
        assert!(!mask[0][0]);
        assert!(!mask[0][1]);
        assert!(mask[0][2]);
        assert!(!mask[0][3]);
        assert!(!mask[0][4]);
        assert_eq!(n_coi[0], 1);
    }

    #[test]
    fn gws_dimensions() {
        let n = 128;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let cwt_config = CwtConfig::new();
        let cwt_result = cwt_morlet(&ts, &cwt_config).unwrap();

        let sig_config = SignificanceConfig::new();
        let result = test_significance(&ts, &cwt_result, &sig_config).unwrap();

        assert_eq!(result.n_scales(), cwt_result.n_scales());
        assert_eq!(result.periods().len(), cwt_result.n_scales());
        assert_eq!(result.gws_masked().len(), cwt_result.n_scales());
        assert_eq!(result.gws_unmasked().len(), cwt_result.n_scales());
        assert_eq!(result.significance_threshold().len(), cwt_result.n_scales());
        assert_eq!(result.significant().len(), cwt_result.n_scales());
        assert_eq!(result.theoretical_spectrum().len(), cwt_result.n_scales());
        assert_eq!(result.effective_dof().len(), cwt_result.n_scales());
        assert_eq!(result.n_coi().len(), cwt_result.n_scales());
    }

    #[test]
    fn gws_constant_zero() {
        let data = vec![42.0; 128];
        let ts = TimeSeries::new(data).unwrap();
        let cwt_config = CwtConfig::new();
        let cwt_result = cwt_morlet(&ts, &cwt_config).unwrap();

        let sig_config = SignificanceConfig::new();
        let result = test_significance(&ts, &cwt_result, &sig_config).unwrap();

        for (j, &val) in result.gws_masked().iter().enumerate() {
            assert!(
                val < 1e-10,
                "gws_masked[{}] = {}, expected near zero for constant signal",
                j,
                val
            );
        }
    }

    #[test]
    fn significance_sine_detected() {
        let n = 512;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 32.0).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let cwt_config = CwtConfig::new().with_dj(0.125);
        let cwt_result = cwt_morlet(&ts, &cwt_config).unwrap();

        let sig_config = SignificanceConfig::new().with_significance_level(0.90);
        let result = test_significance(&ts, &cwt_result, &sig_config).unwrap();

        // Find if any significant period is near 32
        let sig_periods = result.significant_periods();
        let near_32 = sig_periods.iter().any(|&p| (p - 32.0).abs() / 32.0 < 0.2);
        assert!(
            near_32,
            "expected a significant period near 32, got significant periods: {:?}",
            sig_periods
        );
    }

    #[test]
    fn significance_white_noise_few_false_positives() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let n = 512;
        let data: Vec<f64> = (0..n)
            .map(|_| <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng))
            .collect();
        let ts = TimeSeries::new(data).unwrap();
        let cwt_config = CwtConfig::new();
        let cwt_result = cwt_morlet(&ts, &cwt_config).unwrap();

        let sig_config = SignificanceConfig::new()
            .with_significance_level(0.95)
            .with_noise_model(NoiseModel::White);
        let result = test_significance(&ts, &cwt_result, &sig_config).unwrap();

        let n_sig = result.significant_indices().len();
        let n_scales = result.n_scales();
        let false_positive_rate = n_sig as f64 / n_scales as f64;
        assert!(
            false_positive_rate < 0.20,
            "false positive rate = {:.2} ({}/{} scales), expected < 0.20",
            false_positive_rate,
            n_sig,
            n_scales
        );
    }

    #[test]
    fn config_defaults() {
        let config = SignificanceConfig::new();
        assert!(
            (config.significance_level() - 0.90).abs() < f64::EPSILON,
            "default significance_level = {}",
            config.significance_level()
        );
        assert_eq!(config.noise_model(), NoiseModel::Red);
        assert!(
            (config.dof_min() - 2.0).abs() < f64::EPSILON,
            "default dof_min = {}",
            config.dof_min()
        );
        assert!(
            (config.gamma_fac() - 2.32).abs() < f64::EPSILON,
            "default gamma_fac = {}",
            config.gamma_fac()
        );
    }

    #[test]
    fn config_validation() {
        let data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let cwt_config = CwtConfig::new();
        let cwt_result = cwt_morlet(&ts, &cwt_config).unwrap();

        // significance_level = 0.0 should fail
        let config_zero = SignificanceConfig::new().with_significance_level(0.0);
        let err = test_significance(&ts, &cwt_result, &config_zero).unwrap_err();
        assert!(matches!(err, WaveletError::SignificanceFailed(_)));

        // significance_level = 1.0 should fail
        let config_one = SignificanceConfig::new().with_significance_level(1.0);
        let err = test_significance(&ts, &cwt_result, &config_one).unwrap_err();
        assert!(matches!(err, WaveletError::SignificanceFailed(_)));
    }

    #[test]
    fn config_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<SignificanceConfig>();
    }

    #[test]
    fn gws_result_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<GwsResult>();
    }

    #[test]
    fn noise_model_is_copy() {
        fn assert_impl<T: Send + Sync + Copy>() {}
        assert_impl::<NoiseModel>();
    }

    #[test]
    fn gws_in_original_units() {
        // A sine with amplitude 10 should produce GWS values proportional to 10^2
        let n = 256;
        let amp = 10.0;
        let data: Vec<f64> = (0..n)
            .map(|i| amp * (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin())
            .collect();
        let ts = TimeSeries::new(data).unwrap();
        let cwt_config = CwtConfig::new();
        let cwt_result = cwt_morlet(&ts, &cwt_config).unwrap();

        let sig_config = SignificanceConfig::new();
        let result = test_significance(&ts, &cwt_result, &sig_config).unwrap();

        // Peak GWS should be on the order of amp^2 / 2 = 50
        let peak_gws = result
            .gws_unmasked()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            peak_gws > 10.0,
            "peak GWS = {peak_gws}, expected >> 1 for amplitude {amp}"
        );
    }

    #[test]
    fn significance_decision_scale_invariant() {
        // Significance decisions must not change when the signal is scaled by 100x
        let n = 256;
        let base: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin())
            .collect();
        let scaled: Vec<f64> = base.iter().map(|&x| x * 100.0).collect();

        let cwt_config = CwtConfig::new();
        let sig_config = SignificanceConfig::new();

        let ts_base = TimeSeries::new(base).unwrap();
        let cwt_base = cwt_morlet(&ts_base, &cwt_config).unwrap();
        let sig_base = test_significance(&ts_base, &cwt_base, &sig_config).unwrap();

        let ts_scaled = TimeSeries::new(scaled).unwrap();
        let cwt_scaled = cwt_morlet(&ts_scaled, &cwt_config).unwrap();
        let sig_scaled = test_significance(&ts_scaled, &cwt_scaled, &sig_config).unwrap();

        assert_eq!(
            sig_base.significant(),
            sig_scaled.significant(),
            "significance decisions changed when scaling input by 100x:\nbase significant indices: {:?}\nscaled significant indices: {:?}",
            sig_base.significant_indices(),
            sig_scaled.significant_indices()
        );
    }
}
