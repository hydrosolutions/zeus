//! Multiresolution Analysis (MRA) via MODWT.

use crate::error::WaveletError;
use crate::filter::WaveletFilter;
#[allow(unused_imports)]
use crate::modwt::ModwtCoeffs;
use crate::series::TimeSeries;

/// Selects MRA decomposition levels based on the series length and
/// maximum period fraction.
///
/// Returns levels where the wavelet period `2^j` does not exceed
/// `max_period_frac * n`.
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::select_levels;
///
/// let levels = select_levels(365, 1.0 / 3.0);
/// assert_eq!(levels, vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn select_levels(n: usize, max_period_frac: f64) -> Vec<usize> {
    let max_period = max_period_frac * n as f64;
    let mut levels = Vec::new();
    let mut j = 1;
    loop {
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

    /// Returns the variance fraction of each component.
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
        let _ = self;
        todo!()
    }
}

/// Performs a Multiresolution Analysis on the given time series.
///
/// # Errors
///
/// | Variant | Trigger |
/// |---------|---------|
/// | [`WaveletError::LevelTooHigh`] | requested levels exceed maximum |
/// | [`WaveletError::MraFailed`] | numerical failure during analysis |
pub fn mra(series: &TimeSeries, config: &MraConfig) -> Result<Mra, WaveletError> {
    let _ = (series, config);
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_levels_365() {
        let levels = select_levels(365, 1.0 / 3.0);
        assert_eq!(levels, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn select_levels_small_n() {
        let levels = select_levels(8, 1.0 / 3.0);
        assert_eq!(levels, vec![1]);
    }

    #[test]
    fn select_levels_zero_frac() {
        let levels = select_levels(365, 0.0);
        assert!(levels.is_empty());
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
}
