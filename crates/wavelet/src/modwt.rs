//! Maximal Overlap Discrete Wavelet Transform (MODWT).

use crate::error::WaveletError;
use crate::filter::WaveletFilter;
use crate::series::TimeSeries;

/// Computes the maximum feasible MODWT decomposition level for a
/// given series length and filter.
///
/// The maximum level `J` satisfies `L_j = (2^J - 1)(L - 1) + 1 <= N`,
/// where `L` is the filter length and `N` is the series length.
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::{WaveletFilter, max_modwt_level};
///
/// let max_level = max_modwt_level(256, &WaveletFilter::La8);
/// assert_eq!(max_level, 5);
/// ```
pub fn max_modwt_level(n: usize, filter: &WaveletFilter) -> usize {
    let l = filter.length() as f64;
    let n = n as f64;
    // L_j = (2^J - 1)(L - 1) + 1 <= N
    // 2^J <= (N - 1) / (L - 1) + 1
    // J <= log2((N - 1) / (L - 1) + 1)
    if l <= 1.0 || n <= 1.0 {
        return 0;
    }
    let ratio = (n - 1.0) / (l - 1.0) + 1.0;
    if ratio <= 1.0 {
        return 0;
    }
    ratio.log2().floor() as usize
}

/// Configuration for a MODWT decomposition.
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::{WaveletFilter, ModwtConfig};
///
/// let config = ModwtConfig::new(WaveletFilter::La8, 4);
/// assert_eq!(config.n_levels(), 4);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModwtConfig {
    filter: WaveletFilter,
    n_levels: usize,
}

impl ModwtConfig {
    /// Creates a new MODWT configuration.
    pub fn new(filter: WaveletFilter, n_levels: usize) -> Self {
        Self { filter, n_levels }
    }

    /// Returns the wavelet filter.
    pub fn filter(&self) -> WaveletFilter {
        self.filter
    }

    /// Returns the number of decomposition levels.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }
}

/// MODWT decomposition coefficients.
///
/// Contains detail coefficients at each level and the final smooth
/// (scaling) coefficients.
#[derive(Clone, Debug)]
pub struct ModwtCoeffs {
    details: Vec<Vec<f64>>,
    smooth: Vec<f64>,
    filter: WaveletFilter,
}

impl ModwtCoeffs {
    /// Creates a new `ModwtCoeffs` (crate-internal constructor).
    #[allow(dead_code)]
    pub(crate) fn new(details: Vec<Vec<f64>>, smooth: Vec<f64>, filter: WaveletFilter) -> Self {
        Self {
            details,
            smooth,
            filter,
        }
    }

    /// Returns the number of decomposition levels.
    pub fn n_levels(&self) -> usize {
        self.details.len()
    }

    /// Returns the detail coefficients at the given level (0-indexed).
    ///
    /// Returns `None` if the level is out of range.
    pub fn detail(&self, level: usize) -> Option<&[f64]> {
        self.details.get(level).map(|v| v.as_slice())
    }

    /// Returns the smooth (scaling) coefficients.
    pub fn smooth(&self) -> &[f64] {
        &self.smooth
    }

    /// Returns the length of the original series.
    pub fn series_len(&self) -> usize {
        self.smooth.len()
    }

    /// Returns the wavelet filter used.
    pub fn filter(&self) -> WaveletFilter {
        self.filter
    }
}

/// Computes the MODWT decomposition of a time series.
///
/// # Errors
///
/// | Variant | Trigger |
/// |---------|---------|
/// | [`WaveletError::LevelTooHigh`] | `config.n_levels()` exceeds `max_modwt_level()` |
/// | [`WaveletError::ModwtFailed`] | numerical failure during transform |
pub fn modwt(series: &TimeSeries, config: &ModwtConfig) -> Result<ModwtCoeffs, WaveletError> {
    let _ = (series, config);
    todo!()
}

/// Reconstructs a time series from MODWT coefficients (inverse MODWT).
///
/// # Errors
///
/// Returns [`WaveletError::ModwtFailed`] if reconstruction fails.
pub fn imodwt(coeffs: &ModwtCoeffs) -> Result<Vec<f64>, WaveletError> {
    let _ = coeffs;
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_level_la8_256() {
        let level = max_modwt_level(256, &WaveletFilter::La8);
        assert_eq!(level, 5);
    }

    #[test]
    fn max_level_haar_256() {
        let level = max_modwt_level(256, &WaveletFilter::Haar);
        assert_eq!(level, 8);
    }

    #[test]
    fn max_level_short_series() {
        let level = max_modwt_level(4, &WaveletFilter::La8);
        assert_eq!(level, 0);
    }

    #[test]
    fn max_level_edge_case_length_one() {
        let level = max_modwt_level(1, &WaveletFilter::Haar);
        assert_eq!(level, 0);
    }

    #[test]
    fn config_accessors() {
        let config = ModwtConfig::new(WaveletFilter::D4, 3);
        assert_eq!(config.filter(), WaveletFilter::D4);
        assert_eq!(config.n_levels(), 3);
    }

    #[test]
    fn config_is_copy() {
        let a = ModwtConfig::new(WaveletFilter::La8, 4);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn coeffs_accessors() {
        let details = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let smooth = vec![5.0, 6.0];
        let coeffs = ModwtCoeffs::new(details, smooth, WaveletFilter::La8);

        assert_eq!(coeffs.n_levels(), 2);
        assert_eq!(coeffs.detail(0), Some([1.0, 2.0].as_slice()));
        assert_eq!(coeffs.detail(1), Some([3.0, 4.0].as_slice()));
        assert_eq!(coeffs.detail(2), None);
        assert_eq!(coeffs.smooth(), &[5.0, 6.0]);
        assert_eq!(coeffs.series_len(), 2);
        assert_eq!(coeffs.filter(), WaveletFilter::La8);
    }

    #[test]
    fn coeffs_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<ModwtCoeffs>();
    }
}
