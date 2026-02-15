//! Multiresolution Analysis (MRA) via MODWT.

use crate::error::WaveletError;
use crate::filter::WaveletFilter;
use crate::modwt::{ModwtCoeffs, ModwtConfig, imodwt, max_modwt_level, modwt};
use crate::series::TimeSeries;
use tracing::warn;

/// Selects MRA decomposition levels based on the series length,
/// maximum period fraction, and wavelet filter feasibility.
///
/// Returns levels where the wavelet period `2^j` does not exceed
/// `max_period_frac * n` **and** `j` does not exceed
/// [`max_modwt_level(n, filter)`](crate::max_modwt_level).
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::{select_levels, WaveletFilter};
///
/// let levels = select_levels(365, 1.0 / 3.0, &WaveletFilter::La8);
/// assert_eq!(levels, vec![1, 2, 3, 4, 5]);
/// ```
pub fn select_levels(n: usize, max_period_frac: f64, filter: &WaveletFilter) -> Vec<usize> {
    let max_period = max_period_frac * n as f64;
    let max_level = max_modwt_level(n, filter);
    let mut levels = Vec::new();
    let mut j = 1;
    loop {
        if j > max_level {
            break;
        }
        let period = 2.0_f64.powi(j as i32);
        if period > max_period {
            break;
        }
        levels.push(j);
        j += 1;
    }
    levels
}

/// Configuration for a Multiresolution Analysis.
///
/// Use the builder methods to customize the analysis parameters.
/// When `n_levels` is omitted, auto-selection clamps to the filter's
/// maximum feasible MODWT level.
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::{WaveletFilter, MraConfig};
///
/// let config = MraConfig::new(WaveletFilter::La8)
///     .with_levels(4)
///     .with_include_smooth(false);
/// ```
#[derive(Clone, Debug)]
pub struct MraConfig {
    filter: WaveletFilter,
    n_levels: Option<usize>,
    max_period_frac: f64,
    include_smooth: bool,
}

impl MraConfig {
    /// Creates a new MRA configuration with defaults.
    ///
    /// Defaults: `n_levels = None` (auto), `max_period_frac = 1/3`,
    /// `include_smooth = true`.
    pub fn new(filter: WaveletFilter) -> Self {
        Self {
            filter,
            n_levels: None,
            max_period_frac: 1.0 / 3.0,
            include_smooth: true,
        }
    }

    /// Sets the number of decomposition levels explicitly.
    pub fn with_levels(mut self, n_levels: usize) -> Self {
        self.n_levels = Some(n_levels);
        self
    }

    /// Sets the maximum period fraction for automatic level selection.
    pub fn with_max_period_frac(mut self, frac: f64) -> Self {
        self.max_period_frac = frac;
        self
    }

    /// Sets whether to include the smooth component.
    pub fn with_include_smooth(mut self, include: bool) -> Self {
        self.include_smooth = include;
        self
    }

    /// Returns the wavelet filter.
    pub fn filter(&self) -> WaveletFilter {
        self.filter
    }

    /// Returns the explicit number of levels, if set.
    pub fn n_levels(&self) -> Option<usize> {
        self.n_levels
    }

    /// Returns the maximum period fraction.
    pub fn max_period_frac(&self) -> f64 {
        self.max_period_frac
    }

    /// Returns whether the smooth component is included.
    pub fn include_smooth(&self) -> bool {
        self.include_smooth
    }
}

/// Multiresolution Analysis (MRA) result.
///
/// Contains detail components, the smooth component, associated periods,
/// variance fractions, and reconstruction error.
#[derive(Clone, Debug)]
pub struct Mra {
    details: Vec<Vec<f64>>,
    smooth: Vec<f64>,
    periods: Vec<f64>,
    variance_fractions: Vec<f64>,
    reconstruction_error: f64,
}

impl Mra {
    /// Creates a new `Mra` (crate-internal constructor).
    #[allow(dead_code)]
    pub(crate) fn new(
        details: Vec<Vec<f64>>,
        smooth: Vec<f64>,
        periods: Vec<f64>,
        variance_fractions: Vec<f64>,
        reconstruction_error: f64,
    ) -> Self {
        Self {
            details,
            smooth,
            periods,
            variance_fractions,
            reconstruction_error,
        }
    }

    /// Returns the total number of components (details + smooth if present).
    pub fn n_components(&self) -> usize {
        self.details.len() + if self.smooth.is_empty() { 0 } else { 1 }
    }

    /// Returns the number of detail levels.
    pub fn n_detail_levels(&self) -> usize {
        self.details.len()
    }

    /// Returns the detail component at the given level (0-indexed).
    ///
    /// Returns `None` if the level is out of range.
    pub fn detail(&self, level: usize) -> Option<&[f64]> {
        self.details.get(level).map(|v| v.as_slice())
    }

    /// Returns the smooth component.
    pub fn smooth(&self) -> &[f64] {
        &self.smooth
    }

    /// Returns the periods associated with each component.
    pub fn periods(&self) -> &[f64] {
        &self.periods
    }

    /// Returns the variance fraction of each component (uses sample variance, N-1 denominator).
    pub fn variance_fractions(&self) -> &[f64] {
        &self.variance_fractions
    }

    /// Returns the reconstruction error.
    pub fn reconstruction_error(&self) -> f64 {
        self.reconstruction_error
    }

    /// Returns an iterator over all components (details then smooth).
    pub fn components(&self) -> impl Iterator<Item = &[f64]> {
        self.details
            .iter()
            .map(|v| v.as_slice())
            .chain(if self.smooth.is_empty() {
                None
            } else {
                Some(self.smooth.as_slice())
            })
    }

    /// Converts the MRA components to a column-major matrix representation.
    ///
    /// Each column corresponds to one component (details first, then smooth).
    pub fn to_matrix(&self) -> Vec<Vec<f64>> {
        self.components().map(|c| c.to_vec()).collect()
    }
}

/// Performs a Multiresolution Analysis on the given time series.
///
/// When `n_levels` is omitted from the config, levels are auto-selected
/// respecting both the period-fraction threshold and the filter's maximum
/// feasible MODWT level. [`WaveletError::LevelTooHigh`] only triggers for
/// explicitly requested levels that exceed the filter constraint.
///
/// # Errors
///
/// | Variant | Trigger |
/// |---------|---------|
/// | [`WaveletError::LevelTooHigh`] | explicitly requested levels exceed maximum |
/// | [`WaveletError::MraFailed`] | numerical failure during analysis |
pub fn mra(series: &TimeSeries, config: &MraConfig) -> Result<Mra, WaveletError> {
    let data = series.as_slice();
    let n = data.len();
    let filter = config.filter();

    // Determine number of levels
    let j = match config.n_levels() {
        Some(levels) => levels,
        None => {
            let levels = select_levels(n, config.max_period_frac(), &filter);
            let selected = *levels.last().ok_or_else(|| {
                WaveletError::MraFailed(
                    "no valid decomposition levels for this series length".into(),
                )
            })?;
            let max_level = max_modwt_level(n, &filter);
            let next_period = 2.0_f64.powi((selected + 1) as i32);
            let max_period = config.max_period_frac() * n as f64;
            if selected == max_level && next_period <= max_period {
                warn!(
                    filter_max = max_level,
                    series_len = n,
                    filter = ?filter,
                    "MRA auto-selected levels clamped by filter length constraint"
                );
            }
            selected
        }
    };

    // Validate level
    let max_level = max_modwt_level(n, &filter);
    if j > max_level {
        return Err(WaveletError::LevelTooHigh {
            requested: j,
            max: max_level,
            len: n,
        });
    }

    // Forward MODWT
    let modwt_config = ModwtConfig::new(filter, j);
    let coeffs = modwt(series, &modwt_config)?;

    // Extract detail components D[1..J] via iMODWT
    let mut details = Vec::with_capacity(j);
    for level in 0..j {
        // Build coeffs with only this level's detail non-zero
        let mut zero_details: Vec<Vec<f64>> = (0..j).map(|_| vec![0.0; n]).collect();
        zero_details[level] = coeffs
            .detail(level)
            .ok_or_else(|| WaveletError::MraFailed(format!("missing detail level {}", level)))?
            .to_vec();
        let detail_coeffs = ModwtCoeffs::new(zero_details, vec![0.0; n], filter);
        let component = imodwt(&detail_coeffs)?;
        details.push(component);
    }

    // Extract smooth component S[J]
    let smooth = if config.include_smooth() {
        let zero_details: Vec<Vec<f64>> = (0..j).map(|_| vec![0.0; n]).collect();
        let smooth_coeffs = ModwtCoeffs::new(zero_details, coeffs.smooth().to_vec(), filter);
        imodwt(&smooth_coeffs)?
    } else {
        vec![]
    };

    // Compute periods
    let mut periods = Vec::with_capacity(j + if config.include_smooth() { 1 } else { 0 });
    for level in 1..=j {
        // Detail j -> 2^j * sqrt(2) (geometric mean of band [2^j, 2^(j+1)])
        periods.push(2.0_f64.powi(level as i32) * std::f64::consts::SQRT_2);
    }
    if config.include_smooth() {
        // Smooth -> 2^(J+1)
        periods.push(2.0_f64.powi((j + 1) as i32));
    }

    // Compute variance fractions
    let data_var = variance(data);
    let mut variance_fractions =
        Vec::with_capacity(details.len() + if smooth.is_empty() { 0 } else { 1 });
    for detail in &details {
        variance_fractions.push(if data_var > 0.0 {
            variance(detail) / data_var
        } else {
            0.0
        });
    }
    if !smooth.is_empty() {
        variance_fractions.push(if data_var > 0.0 {
            variance(&smooth) / data_var
        } else {
            0.0
        });
    }

    // Compute reconstruction error
    let reconstruction_error = if config.include_smooth() {
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let sum: f64 = details.iter().map(|d| d[i]).sum::<f64>() + smooth[i];
            let err = (data[i] - sum).abs();
            if err > max_err {
                max_err = err;
            }
        }
        max_err
    } else {
        0.0
    };

    Ok(Mra::new(
        details,
        smooth,
        periods,
        variance_fractions,
        reconstruction_error,
    ))
}

/// Computes sample variance (N-1 denominator) of a data slice.
pub(crate) fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = data.iter().sum::<f64>() / nf;
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (nf - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_levels_365() {
        let levels = select_levels(365, 1.0 / 3.0, &WaveletFilter::La8);
        assert_eq!(levels, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn select_levels_small_n() {
        let levels = select_levels(8, 1.0 / 3.0, &WaveletFilter::La8);
        assert_eq!(levels, vec![1]);
    }

    #[test]
    fn select_levels_zero_frac() {
        let levels = select_levels(365, 0.0, &WaveletFilter::La8);
        assert!(levels.is_empty());
    }

    #[test]
    fn select_levels_n41_la8_clamped() {
        // Regression: N=41, La8 — period allows [1,2,3] but filter max is 2
        let levels = select_levels(41, 1.0 / 3.0, &WaveletFilter::La8);
        assert_eq!(levels, vec![1, 2]);
    }

    #[test]
    fn select_levels_filter_constraint_binding() {
        // N=41, La16 — max_modwt_level = 1, period allows [1,2,3]
        let levels = select_levels(41, 1.0 / 3.0, &WaveletFilter::La16);
        assert_eq!(levels, vec![1]);
    }

    #[test]
    fn select_levels_period_constraint_binding() {
        // N=41, Haar — max_modwt_level = 5, period allows [1,2,3], no clamping
        let levels = select_levels(41, 1.0 / 3.0, &WaveletFilter::Haar);
        assert_eq!(levels, vec![1, 2, 3]);
    }

    #[test]
    fn select_levels_max_level_zero() {
        // N=4, La8 — max_modwt_level = 0, no levels feasible
        let levels = select_levels(4, 1.0 / 3.0, &WaveletFilter::La8);
        assert!(levels.is_empty());
    }

    #[test]
    fn select_levels_large_frac_la8_clamped() {
        // N=41, La8, frac=1.0 — period allows many levels but filter max is 2
        let levels = select_levels(41, 1.0, &WaveletFilter::La8);
        assert_eq!(levels, vec![1, 2]);
    }

    #[test]
    fn select_levels_haar_large_frac() {
        // N=256, Haar, frac=1.0 — both constraints agree at level 8
        let levels = select_levels(256, 1.0, &WaveletFilter::Haar);
        assert_eq!(levels, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn mra_config_defaults() {
        let config = MraConfig::new(WaveletFilter::La8);
        assert_eq!(config.filter(), WaveletFilter::La8);
        assert_eq!(config.n_levels(), None);
        assert!((config.max_period_frac() - 1.0 / 3.0).abs() < f64::EPSILON);
        assert!(config.include_smooth());
    }

    #[test]
    fn mra_config_builder() {
        let config = MraConfig::new(WaveletFilter::D4)
            .with_levels(3)
            .with_max_period_frac(0.5)
            .with_include_smooth(false);

        assert_eq!(config.filter(), WaveletFilter::D4);
        assert_eq!(config.n_levels(), Some(3));
        assert!((config.max_period_frac() - 0.5).abs() < f64::EPSILON);
        assert!(!config.include_smooth());
    }

    #[test]
    fn mra_accessors() {
        let details = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let smooth = vec![5.0, 6.0];
        let periods = vec![4.0, 8.0, 16.0];
        let var_fracs = vec![0.3, 0.5, 0.2];
        let mra = Mra::new(details, smooth, periods, var_fracs, 0.001);

        assert_eq!(mra.n_components(), 3);
        assert_eq!(mra.n_detail_levels(), 2);
        assert_eq!(mra.detail(0), Some([1.0, 2.0].as_slice()));
        assert_eq!(mra.detail(1), Some([3.0, 4.0].as_slice()));
        assert_eq!(mra.detail(2), None);
        assert_eq!(mra.smooth(), &[5.0, 6.0]);
        assert_eq!(mra.periods(), &[4.0, 8.0, 16.0]);
        assert_eq!(mra.variance_fractions(), &[0.3, 0.5, 0.2]);
        assert!((mra.reconstruction_error() - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn mra_components_iterator() {
        let details = vec![vec![1.0], vec![2.0]];
        let smooth = vec![3.0];
        let mra = Mra::new(details, smooth, vec![], vec![], 0.0);

        let components: Vec<&[f64]> = mra.components().collect();
        assert_eq!(components.len(), 3);
        assert_eq!(components[0], &[1.0]);
        assert_eq!(components[1], &[2.0]);
        assert_eq!(components[2], &[3.0]);
    }

    #[test]
    fn mra_empty_smooth() {
        let mra = Mra::new(vec![vec![1.0]], vec![], vec![], vec![], 0.0);
        assert_eq!(mra.n_components(), 1);
        let components: Vec<&[f64]> = mra.components().collect();
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn mra_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<Mra>();
    }

    #[test]
    fn mra_config_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<MraConfig>();
    }

    #[test]
    fn mra_additive_reconstruction() {
        let n = 128;
        let data: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.1).sin() + 0.5 * (i as f64 * 0.3).cos())
            .collect();
        let ts = TimeSeries::new(data.clone()).unwrap();
        let config = MraConfig::new(WaveletFilter::La8).with_levels(3);
        let result = mra(&ts, &config).unwrap();

        assert!(
            result.reconstruction_error() < 1e-6,
            "reconstruction error too high: {}",
            result.reconstruction_error()
        );

        // Verify components actually sum to original
        for (i, &expected) in data.iter().enumerate() {
            let sum: f64 = (0..result.n_detail_levels())
                .map(|j| result.detail(j).unwrap()[i])
                .sum::<f64>()
                + result.smooth()[i];
            assert!(
                (expected - sum).abs() < 1e-6,
                "additive reconstruction failed at index {}",
                i
            );
        }
    }

    #[test]
    fn mra_component_lengths() {
        let n = 64;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = MraConfig::new(WaveletFilter::D4).with_levels(3);
        let result = mra(&ts, &config).unwrap();

        for level in 0..result.n_detail_levels() {
            assert_eq!(result.detail(level).unwrap().len(), n);
        }
        assert_eq!(result.smooth().len(), n);
    }

    #[test]
    fn mra_level_too_high() {
        let ts = TimeSeries::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let config = MraConfig::new(WaveletFilter::La8).with_levels(10);
        let err = mra(&ts, &config).unwrap_err();
        assert!(matches!(err, WaveletError::LevelTooHigh { .. }));
    }

    #[test]
    fn mra_exclude_smooth() {
        let n = 64;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = MraConfig::new(WaveletFilter::Haar)
            .with_levels(3)
            .with_include_smooth(false);
        let result = mra(&ts, &config).unwrap();

        assert!(result.smooth().is_empty());
        assert_eq!(result.n_detail_levels(), 3);
        // Periods should only have detail periods (no smooth period)
        assert_eq!(result.periods().len(), 3);
        assert_eq!(result.variance_fractions().len(), 3);
    }

    #[test]
    fn mra_periods_monotonic() {
        let n = 128;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = MraConfig::new(WaveletFilter::Haar).with_levels(4);
        let result = mra(&ts, &config).unwrap();

        let periods = result.periods();
        assert!(!periods.is_empty());
        // First period should be approximately 2*sqrt(2)
        assert!(
            (periods[0] - 2.0 * std::f64::consts::SQRT_2).abs() < 1e-10,
            "first period: {} (expected {})",
            periods[0],
            2.0 * std::f64::consts::SQRT_2
        );
        // Periods should be strictly increasing
        for i in 1..periods.len() {
            assert!(
                periods[i] > periods[i - 1],
                "periods not monotonic at index {}: {} <= {}",
                i,
                periods[i],
                periods[i - 1]
            );
        }
    }

    #[test]
    fn mra_constant_series() {
        let n = 32;
        let data = vec![42.0; n];
        let ts = TimeSeries::new(data).unwrap();
        let config = MraConfig::new(WaveletFilter::Haar).with_levels(3);
        let result = mra(&ts, &config).unwrap();

        // All variance fractions should be ~0 for constant input
        for (i, &vf) in result.variance_fractions().iter().enumerate() {
            assert!(
                vf.abs() < 1e-10 || vf.is_nan(),
                "variance fraction {} not ~0: {}",
                i,
                vf
            );
        }
    }

    #[test]
    fn mra_variance_fractions_reasonable() {
        let n = 256;
        let data: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.05).sin() + 0.5 * (i as f64 * 0.2).cos())
            .collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = MraConfig::new(WaveletFilter::La8).with_levels(4);
        let result = mra(&ts, &config).unwrap();

        let sum: f64 = result.variance_fractions().iter().sum();
        assert!(
            sum > 0.5 && sum < 2.0,
            "variance fractions sum out of range: {}",
            sum
        );
    }

    #[test]
    fn to_matrix_dimensions_with_smooth() {
        let n = 64;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = MraConfig::new(WaveletFilter::Haar).with_levels(3);
        let result = mra(&ts, &config).unwrap();

        let matrix = result.to_matrix();
        // 3 details + 1 smooth = 4 columns
        assert_eq!(matrix.len(), 4);
        for col in &matrix {
            assert_eq!(col.len(), n);
        }
    }

    #[test]
    fn to_matrix_dimensions_without_smooth() {
        let n = 64;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = MraConfig::new(WaveletFilter::Haar)
            .with_levels(3)
            .with_include_smooth(false);
        let result = mra(&ts, &config).unwrap();

        let matrix = result.to_matrix();
        // 3 details, no smooth
        assert_eq!(matrix.len(), 3);
        for col in &matrix {
            assert_eq!(col.len(), n);
        }
    }

    #[test]
    fn to_matrix_matches_components() {
        let n = 64;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.15).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = MraConfig::new(WaveletFilter::D4).with_levels(2);
        let result = mra(&ts, &config).unwrap();

        let matrix = result.to_matrix();
        let components: Vec<Vec<f64>> = result.components().map(|c| c.to_vec()).collect();

        assert_eq!(matrix.len(), components.len());
        for (i, (mat_col, comp)) in matrix.iter().zip(components.iter()).enumerate() {
            assert_eq!(mat_col, comp, "matrix column {} doesn't match component", i);
        }
    }

    #[test]
    fn variance_is_sample_variance() {
        // Known: sample variance of [2, 4, 4, 4, 5, 5, 7, 9] = 32/7 ≈ 4.571428...
        // Mean = 5.0, sum of squared deviations = 9+1+1+1+0+0+4+16 = 32
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let expected = 32.0 / 7.0; // sum of squared deviations = 32, N-1 = 7
        let v = variance(&data);
        assert!(
            (v - expected).abs() < 1e-10,
            "variance = {v}, expected {expected}"
        );
    }

    #[test]
    fn variance_single_element() {
        let v = variance(&[42.0]);
        assert!(
            v.abs() < f64::EPSILON,
            "single-element variance = {v}, expected 0.0"
        );
    }

    #[test]
    fn mra_auto_levels_short_series() {
        // N=41, La8, auto-select — must succeed (was the original bug)
        let data: Vec<f64> = (0..41).map(|i| (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = MraConfig::new(WaveletFilter::La8);
        let result = mra(&ts, &config).unwrap();
        assert_eq!(result.n_detail_levels(), 2);
        assert!(
            result.reconstruction_error() < 1e-6,
            "reconstruction error too high: {}",
            result.reconstruction_error()
        );
    }

    #[test]
    fn mra_auto_levels_all_filters() {
        // N=41, all 6 filters, auto-select — all must succeed
        let data: Vec<f64> = (0..41).map(|i| (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        for filter in [
            WaveletFilter::Haar,
            WaveletFilter::D4,
            WaveletFilter::D6,
            WaveletFilter::D8,
            WaveletFilter::La8,
            WaveletFilter::La16,
        ] {
            let config = MraConfig::new(filter);
            let result = mra(&ts, &config).unwrap_or_else(|e| {
                panic!("mra failed for {:?}: {}", filter, e);
            });
            let max_level = max_modwt_level(41, &filter);
            assert!(
                result.n_detail_levels() <= max_level,
                "{:?}: n_detail_levels {} > max_modwt_level {}",
                filter,
                result.n_detail_levels(),
                max_level
            );
        }
    }
}
