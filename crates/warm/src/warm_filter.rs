//! WARM pool filtering.

use crate::error::WarmError;
use crate::warm::WarmResult;

/// Tolerance bounds for filtering the WARM simulation pool.
///
/// Uses a builder pattern with sensible defaults.
///
/// # Example
///
/// ```ignore
/// use zeus_warm::FilterBounds;
///
/// let bounds = FilterBounds::default()
///     .with_mean_tol(0.05)
///     .with_sd_tol(0.10);
/// ```
#[derive(Clone, Debug)]
pub struct FilterBounds {
    mean_tol: f64,
    sd_tol: f64,
    tail_mass_tol: f64,
    spectral_corr_min: f64,
    peak_match_min: f64,
}

impl Default for FilterBounds {
    /// Returns default filter bounds.
    ///
    /// | Parameter | Default |
    /// |-----------|---------|
    /// | `mean_tol` | 0.1 |
    /// | `sd_tol` | 0.15 |
    /// | `tail_mass_tol` | 0.2 |
    /// | `spectral_corr_min` | 0.7 |
    /// | `peak_match_min` | 0.5 |
    fn default() -> Self {
        Self {
            mean_tol: 0.1,
            sd_tol: 0.15,
            tail_mass_tol: 0.2,
            spectral_corr_min: 0.7,
            peak_match_min: 0.5,
        }
    }
}

impl FilterBounds {
    /// Sets the mean tolerance (relative).
    pub fn with_mean_tol(mut self, tol: f64) -> Self {
        self.mean_tol = tol;
        self
    }

    /// Sets the standard deviation tolerance (relative).
    pub fn with_sd_tol(mut self, tol: f64) -> Self {
        self.sd_tol = tol;
        self
    }

    /// Sets the tail mass tolerance.
    pub fn with_tail_mass_tol(mut self, tol: f64) -> Self {
        self.tail_mass_tol = tol;
        self
    }

    /// Sets the minimum spectral correlation.
    pub fn with_spectral_corr_min(mut self, min: f64) -> Self {
        self.spectral_corr_min = min;
        self
    }

    /// Sets the minimum peak match score.
    pub fn with_peak_match_min(mut self, min: f64) -> Self {
        self.peak_match_min = min;
        self
    }

    /// Returns the mean tolerance.
    pub fn mean_tol(&self) -> f64 {
        self.mean_tol
    }

    /// Returns the standard deviation tolerance.
    pub fn sd_tol(&self) -> f64 {
        self.sd_tol
    }

    /// Returns the tail mass tolerance.
    pub fn tail_mass_tol(&self) -> f64 {
        self.tail_mass_tol
    }

    /// Returns the minimum spectral correlation.
    pub fn spectral_corr_min(&self) -> f64 {
        self.spectral_corr_min
    }

    /// Returns the minimum peak match score.
    pub fn peak_match_min(&self) -> f64 {
        self.peak_match_min
    }
}

/// Result of filtering a WARM simulation pool.
///
/// Contains indices of selected simulations and their corresponding
/// quality scores.
#[derive(Clone, Debug)]
pub struct FilteredPool {
    selected: Vec<usize>,
    scores: Vec<f64>,
}

impl FilteredPool {
    /// Creates a new `FilteredPool` (crate-internal constructor).
    #[allow(dead_code)]
    pub(crate) fn new(selected: Vec<usize>, scores: Vec<f64>) -> Self {
        Self { selected, scores }
    }

    /// Returns the indices of selected simulations.
    pub fn selected(&self) -> &[usize] {
        &self.selected
    }

    /// Returns the quality scores for selected simulations.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Returns the number of selected simulations.
    pub fn n_selected(&self) -> usize {
        self.selected.len()
    }
}

/// Filters a WARM simulation pool against observed data.
///
/// Evaluates each simulation against the observed data using the
/// specified tolerance bounds and returns the indices and scores
/// of simulations that pass all criteria.
///
/// # Errors
///
/// | Variant | Trigger |
/// |---------|---------|
/// | [`WarmError::InsufficientSimulations`] | pool is empty |
/// | [`WarmError::FilteringFailed`] | numerical failure during scoring |
pub fn filter_warm_pool(
    observed: &[f64],
    result: &WarmResult,
    bounds: &FilterBounds,
) -> Result<FilteredPool, WarmError> {
    let _ = (observed, result, bounds);
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_bounds_defaults() {
        let bounds = FilterBounds::default();
        assert!((bounds.mean_tol() - 0.1).abs() < f64::EPSILON);
        assert!((bounds.sd_tol() - 0.15).abs() < f64::EPSILON);
        assert!((bounds.tail_mass_tol() - 0.2).abs() < f64::EPSILON);
        assert!((bounds.spectral_corr_min() - 0.7).abs() < f64::EPSILON);
        assert!((bounds.peak_match_min() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn filter_bounds_builder() {
        let bounds = FilterBounds::default()
            .with_mean_tol(0.05)
            .with_sd_tol(0.10)
            .with_tail_mass_tol(0.15)
            .with_spectral_corr_min(0.8)
            .with_peak_match_min(0.6);

        assert!((bounds.mean_tol() - 0.05).abs() < f64::EPSILON);
        assert!((bounds.sd_tol() - 0.10).abs() < f64::EPSILON);
        assert!((bounds.tail_mass_tol() - 0.15).abs() < f64::EPSILON);
        assert!((bounds.spectral_corr_min() - 0.8).abs() < f64::EPSILON);
        assert!((bounds.peak_match_min() - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn filtered_pool_accessors() {
        let pool = FilteredPool::new(vec![0, 3, 7], vec![0.95, 0.88, 0.91]);
        assert_eq!(pool.selected(), &[0, 3, 7]);
        assert_eq!(pool.scores(), &[0.95, 0.88, 0.91]);
        assert_eq!(pool.n_selected(), 3);
    }

    #[test]
    fn filtered_pool_empty() {
        let pool = FilteredPool::new(vec![], vec![]);
        assert_eq!(pool.n_selected(), 0);
        assert!(pool.selected().is_empty());
        assert!(pool.scores().is_empty());
    }

    #[test]
    fn bounds_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<FilterBounds>();
    }

    #[test]
    fn pool_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<FilteredPool>();
    }
}
