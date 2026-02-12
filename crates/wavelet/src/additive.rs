//! Combined additive wavelet decomposition with CWT significance testing.
//!
//! Bridges MRA decomposition with CWT-based significance to tag each
//! MRA detail level as statistically significant or not.

use crate::cwt::{CwtConfig, cwt_morlet};
use crate::error::WaveletError;
use crate::mra::{Mra, MraConfig, mra};
use crate::series::TimeSeries;
use crate::significance::{GwsResult, SignificanceConfig, test_significance};

/// Configuration for the combined additive wavelet decomposition.
///
/// Composes MRA, CWT, and significance testing configurations.
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::{WaveletFilter, MraConfig, AdditiveConfig};
///
/// let config = AdditiveConfig::new(MraConfig::new(WaveletFilter::La8));
/// ```
#[derive(Clone, Debug)]
pub struct AdditiveConfig {
    mra_config: MraConfig,
    cwt_config: CwtConfig,
    significance_config: SignificanceConfig,
}

impl AdditiveConfig {
    /// Creates a new `AdditiveConfig` with default CWT and significance configs.
    pub fn new(mra_config: MraConfig) -> Self {
        Self {
            mra_config,
            cwt_config: CwtConfig::new(),
            significance_config: SignificanceConfig::new(),
        }
    }

    /// Sets the CWT configuration.
    pub fn with_cwt(mut self, config: CwtConfig) -> Self {
        self.cwt_config = config;
        self
    }

    /// Sets the significance testing configuration.
    pub fn with_significance(mut self, config: SignificanceConfig) -> Self {
        self.significance_config = config;
        self
    }

    /// Returns the MRA configuration.
    pub fn mra_config(&self) -> &MraConfig {
        &self.mra_config
    }

    /// Returns the CWT configuration.
    pub fn cwt_config(&self) -> &CwtConfig {
        &self.cwt_config
    }

    /// Returns the significance testing configuration.
    pub fn significance_config(&self) -> &SignificanceConfig {
        &self.significance_config
    }
}

/// Combined result of MRA decomposition and CWT significance testing.
///
/// Tags each MRA detail level as statistically significant or not,
/// based on the Global Wavelet Spectrum significance test.
#[derive(Clone, Debug)]
pub struct WaveletAdditive {
    mra: Mra,
    significant_levels: Vec<bool>,
    smooth_significant: bool,
    gws_result: GwsResult,
    period_level_map: Vec<usize>,
}

impl WaveletAdditive {
    /// Creates a new `WaveletAdditive` (crate-internal constructor).
    pub(crate) fn new(
        mra: Mra,
        significant_levels: Vec<bool>,
        smooth_significant: bool,
        gws_result: GwsResult,
        period_level_map: Vec<usize>,
    ) -> Self {
        Self {
            mra,
            significant_levels,
            smooth_significant,
            gws_result,
            period_level_map,
        }
    }

    /// Returns the MRA decomposition result.
    pub fn mra(&self) -> &Mra {
        &self.mra
    }

    /// Returns the GWS significance test result.
    pub fn gws_result(&self) -> &GwsResult {
        &self.gws_result
    }

    /// Returns which MRA detail levels are statistically significant.
    pub fn significant_levels(&self) -> &[bool] {
        &self.significant_levels
    }

    /// Returns whether the smooth component is significant (always false).
    pub fn smooth_significant(&self) -> bool {
        self.smooth_significant
    }

    /// Returns the periods that are significant.
    pub fn significant_periods(&self) -> Vec<f64> {
        self.gws_result.significant_periods()
    }

    /// Returns the mapping of significant periods to 1-indexed MRA levels.
    pub fn period_level_map(&self) -> &[usize] {
        &self.period_level_map
    }

    /// Returns the number of significant detail levels.
    pub fn n_significant(&self) -> usize {
        self.significant_levels.iter().filter(|&&s| s).count()
    }
}

/// Performs a combined additive wavelet decomposition with significance testing.
///
/// Runs the CWT, significance test, and MRA, then maps significant CWT periods
/// to MRA detail levels.
///
/// # Errors
///
/// Returns errors from the underlying CWT, significance, or MRA computations.
pub fn analyze_additive(
    series: &TimeSeries,
    config: &AdditiveConfig,
) -> Result<WaveletAdditive, WaveletError> {
    let cwt_result = cwt_morlet(series, config.cwt_config())?;
    let gws_result = test_significance(series, &cwt_result, config.significance_config())?;
    let mra_result = mra(series, config.mra_config())?;

    let sig_periods = gws_result.significant_periods();
    let n_levels = mra_result.n_detail_levels();
    let (significant_levels, period_level_map) = build_significance_map(&sig_periods, n_levels);

    Ok(WaveletAdditive::new(
        mra_result,
        significant_levels,
        false,
        gws_result,
        period_level_map,
    ))
}

/// Maps a CWT period to a 1-indexed MRA level.
///
/// MRA level j captures periods in [2^j, 2^(j+1)). The mapping is
/// floor(log2(period)), clamped to [1, n_levels].
fn map_period_to_level(period: f64, n_levels: usize) -> usize {
    let level = period.log2().floor() as usize;
    level.clamp(1, n_levels)
}

/// Builds the significance map from significant periods.
///
/// Returns (significant_levels, period_level_map) where significant_levels
/// is a boolean vector indexed by 0-based MRA level, and period_level_map
/// is a vector of 1-indexed MRA levels for each significant period.
fn build_significance_map(sig_periods: &[f64], n_levels: usize) -> (Vec<bool>, Vec<usize>) {
    let mut significant = vec![false; n_levels];
    let mut period_map = Vec::with_capacity(sig_periods.len());
    for &period in sig_periods {
        let level = map_period_to_level(period, n_levels);
        significant[level - 1] = true; // convert 1-indexed to 0-indexed
        period_map.push(level);
    }
    (significant, period_map)
}

#[cfg(test)]
mod tests {
    use crate::{MraConfig, TimeSeries, WaveletFilter};

    use super::*;

    #[test]
    fn map_period_level_1() {
        let n_levels = 5;
        assert_eq!(map_period_to_level(2.0, n_levels), 1);
        assert_eq!(map_period_to_level(2.5, n_levels), 1);
        assert_eq!(map_period_to_level(3.9, n_levels), 1);
    }

    #[test]
    fn map_period_level_3() {
        let n_levels = 5;
        assert_eq!(map_period_to_level(8.0, n_levels), 3);
        assert_eq!(map_period_to_level(10.0, n_levels), 3);
        assert_eq!(map_period_to_level(15.0, n_levels), 3);
    }

    #[test]
    fn map_period_clamp_low() {
        let n_levels = 5;
        assert_eq!(map_period_to_level(0.5, n_levels), 1);
    }

    #[test]
    fn map_period_clamp_high() {
        let n_levels = 5;
        assert_eq!(map_period_to_level(1024.0, n_levels), 5);
    }

    #[test]
    fn map_period_exact_power_of_two() {
        let n_levels = 5;
        assert_eq!(map_period_to_level(4.0, n_levels), 2);
        assert_eq!(map_period_to_level(8.0, n_levels), 3);
    }

    #[test]
    fn significance_map_empty() {
        let (significant, period_map) = build_significance_map(&[], 5);
        assert_eq!(significant, vec![false; 5]);
        assert!(period_map.is_empty());
    }

    #[test]
    fn significance_map_single() {
        let (significant, period_map) = build_significance_map(&[4.0], 5);
        assert_eq!(significant, vec![false, true, false, false, false]);
        assert_eq!(period_map, vec![2]);
    }

    #[test]
    fn significance_map_multiple_same_level() {
        let (significant, period_map) = build_significance_map(&[4.0, 5.0], 5);
        // Both periods 4.0 and 5.0 map to level 2
        assert_eq!(significant, vec![false, true, false, false, false]);
        assert_eq!(period_map, vec![2, 2]);
    }

    #[test]
    fn significance_map_different_levels() {
        let (significant, period_map) = build_significance_map(&[4.0, 16.0], 5);
        // 4.0 -> level 2, 16.0 -> level 4
        assert_eq!(significant, vec![false, true, false, true, false]);
        assert_eq!(period_map, vec![2, 4]);
    }

    #[test]
    fn analyze_additive_smoke() {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        use std::f64::consts::PI;

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let data: Vec<f64> = (0..256)
            .map(|i| {
                let t = i as f64;
                (2.0 * PI * t / 8.0).sin()
                    + 0.5 * (2.0 * PI * t / 32.0).sin()
                    + 0.1 * <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
            })
            .collect();

        let ts = TimeSeries::new(data).unwrap();
        let config = AdditiveConfig::new(MraConfig::new(WaveletFilter::La8).with_levels(5));
        let result = analyze_additive(&ts, &config).unwrap();

        // mra() returns valid MRA
        assert!(result.mra().n_detail_levels() > 0);

        // significant_levels() length equals mra().n_detail_levels()
        assert_eq!(
            result.significant_levels().len(),
            result.mra().n_detail_levels()
        );

        // smooth_significant() is false
        assert!(!result.smooth_significant());

        // n_significant() >= 0 and <= n_detail_levels
        assert!(result.n_significant() <= result.mra().n_detail_levels());

        // period_level_map() length equals significant_periods() length
        assert_eq!(
            result.period_level_map().len(),
            result.significant_periods().len()
        );
    }

    #[test]
    fn additive_config_send_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<AdditiveConfig>();
    }

    #[test]
    fn wavelet_additive_send_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<WaveletAdditive>();
    }
}
