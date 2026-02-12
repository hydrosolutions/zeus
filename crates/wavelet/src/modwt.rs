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
    let data = series.as_slice();
    let n = data.len();
    let filter = config.filter();
    let n_levels = config.n_levels();
    let max_level = max_modwt_level(n, &filter);

    if n_levels > max_level {
        return Err(WaveletError::LevelTooHigh {
            requested: n_levels,
            max: max_level,
            len: n,
        });
    }

    if n_levels == 0 {
        return Ok(ModwtCoeffs::new(vec![], data.to_vec(), filter));
    }

    // Normalize DWT filters to MODWT filters (divide by sqrt(2))
    let scaling = filter.scaling_coeffs();
    let wavelet = filter.wavelet_coeffs();
    let g_tilde: Vec<f64> = scaling
        .iter()
        .map(|&c| c * std::f64::consts::FRAC_1_SQRT_2)
        .collect();
    let h_tilde: Vec<f64> = wavelet
        .iter()
        .map(|&c| c * std::f64::consts::FRAC_1_SQRT_2)
        .collect();

    let mut details = Vec::with_capacity(n_levels);
    let mut input = data.to_vec();

    for j in 0..n_levels {
        let m = 1_usize << j;
        let (smooth_out, detail_out) = modwt_single_level(&input, &g_tilde, &h_tilde, m);
        details.push(detail_out);
        input = smooth_out;
    }

    Ok(ModwtCoeffs::new(details, input, filter))
}

/// Reconstructs a time series from MODWT coefficients (inverse MODWT).
///
/// # Errors
///
/// Returns [`WaveletError::ModwtFailed`] if reconstruction fails.
pub fn imodwt(coeffs: &ModwtCoeffs) -> Result<Vec<f64>, WaveletError> {
    let n_levels = coeffs.n_levels();

    if n_levels == 0 {
        return Ok(coeffs.smooth().to_vec());
    }

    let filter = coeffs.filter();
    let scaling = filter.scaling_coeffs();
    let wavelet = filter.wavelet_coeffs();
    let g_tilde: Vec<f64> = scaling
        .iter()
        .map(|&c| c * std::f64::consts::FRAC_1_SQRT_2)
        .collect();
    let h_tilde: Vec<f64> = wavelet
        .iter()
        .map(|&c| c * std::f64::consts::FRAC_1_SQRT_2)
        .collect();

    let mut current = coeffs.smooth().to_vec();

    for j_idx in (0..n_levels).rev() {
        let m = 1_usize << j_idx;
        let detail = coeffs
            .detail(j_idx)
            .ok_or_else(|| WaveletError::ModwtFailed(format!("missing detail level {}", j_idx)))?;
        current = imodwt_single_level(&current, detail, &g_tilde, &h_tilde, m);
    }

    Ok(current)
}

/// Performs a single level of the forward MODWT periodic convolution.
fn modwt_single_level(input: &[f64], g: &[f64], h: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    let n = input.len();
    let l = g.len();
    let mut smooth = vec![0.0; n];
    let mut detail = vec![0.0; n];

    for i in 0..n {
        let mut v = g[0] * input[i];
        let mut w = h[0] * input[i];
        for k in 1..l {
            let t = ((i as isize) - (k as isize) * (m as isize)).rem_euclid(n as isize) as usize;
            v += g[k] * input[t];
            w += h[k] * input[t];
        }
        smooth[i] = v;
        detail[i] = w;
    }

    (smooth, detail)
}

/// Performs a single level of the inverse MODWT periodic convolution.
fn imodwt_single_level(smooth: &[f64], detail: &[f64], g: &[f64], h: &[f64], m: usize) -> Vec<f64> {
    let n = smooth.len();
    let l = g.len();
    let mut output = vec![0.0; n];

    for (i, out) in output.iter_mut().enumerate() {
        let mut t = i;
        let mut x = g[0] * smooth[t] + h[0] * detail[t];
        for _k in 1..l {
            t = (t + m) % n;
            x += g[_k] * smooth[t] + h[_k] * detail[t];
        }
        *out = x;
    }

    output
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

    #[test]
    fn modwt_level_too_high() {
        let ts = TimeSeries::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let config = ModwtConfig::new(WaveletFilter::La8, 5);
        let err = modwt(&ts, &config).unwrap_err();
        assert!(matches!(err, WaveletError::LevelTooHigh { .. }));
    }

    #[test]
    fn modwt_zero_levels() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let ts = TimeSeries::new(data.clone()).unwrap();
        let config = ModwtConfig::new(WaveletFilter::Haar, 0);
        let coeffs = modwt(&ts, &config).unwrap();
        assert_eq!(coeffs.n_levels(), 0);
        assert_eq!(coeffs.smooth(), data.as_slice());
    }

    #[test]
    fn modwt_output_dimensions() {
        let n = 64;
        let data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = ModwtConfig::new(WaveletFilter::La8, 3);
        let coeffs = modwt(&ts, &config).unwrap();
        assert_eq!(coeffs.n_levels(), 3);
        assert_eq!(coeffs.smooth().len(), n);
        for level in 0..3 {
            assert_eq!(coeffs.detail(level).unwrap().len(), n);
        }
    }

    #[test]
    fn modwt_constant_signal() {
        let n = 32;
        let val = 5.0;
        let data = vec![val; n];
        let ts = TimeSeries::new(data).unwrap();
        let config = ModwtConfig::new(WaveletFilter::Haar, 3);
        let coeffs = modwt(&ts, &config).unwrap();
        // All detail coefficients should be approximately zero
        for level in 0..3 {
            let detail = coeffs.detail(level).unwrap();
            for &d in detail {
                assert!(d.abs() < 1e-10, "detail coefficient not zero: {}", d);
            }
        }
        // Smooth should be approximately constant
        for &s in coeffs.smooth() {
            assert!((s - val).abs() < 1e-10, "smooth not constant: {}", s);
        }
    }

    #[test]
    fn modwt_haar_known_values() {
        // Hand-computed MODWT with Haar at level 1 for [1, 2, 3, 4]
        // MODWT Haar scaling: g = [1/2, 1/2], wavelet: h = [1/2, -1/2]
        // Forward: v[i] = g[0]*x[i] + g[1]*x[(i-1)%n]
        //          w[i] = h[0]*x[i] + h[1]*x[(i-1)%n]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let ts = TimeSeries::new(data).unwrap();
        let config = ModwtConfig::new(WaveletFilter::Haar, 1);
        let coeffs = modwt(&ts, &config).unwrap();

        // v[0] = 0.5*1 + 0.5*4 = 2.5
        // v[1] = 0.5*2 + 0.5*1 = 1.5
        // v[2] = 0.5*3 + 0.5*2 = 2.5
        // v[3] = 0.5*4 + 0.5*3 = 3.5
        let expected_smooth = [2.5, 1.5, 2.5, 3.5];
        for (i, (&got, &exp)) in coeffs
            .smooth()
            .iter()
            .zip(expected_smooth.iter())
            .enumerate()
        {
            assert!(
                (got - exp).abs() < 1e-10,
                "smooth[{}]: got {}, expected {}",
                i,
                got,
                exp
            );
        }

        // w[0] = 0.5*1 - 0.5*4 = -1.5
        // w[1] = 0.5*2 - 0.5*1 = 0.5
        // w[2] = 0.5*3 - 0.5*2 = 0.5
        // w[3] = 0.5*4 - 0.5*3 = 0.5
        let detail = coeffs.detail(0).unwrap();
        let expected_detail = [-1.5, 0.5, 0.5, 0.5];
        for (i, (&got, &exp)) in detail.iter().zip(expected_detail.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "detail[{}]: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn modwt_energy_conservation() {
        let n = 128;
        let data: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.3).cos())
            .collect();
        let ts = TimeSeries::new(data.clone()).unwrap();
        let config = ModwtConfig::new(WaveletFilter::La8, 4);
        let coeffs = modwt(&ts, &config).unwrap();

        let input_energy: f64 = data.iter().map(|x| x * x).sum();
        let smooth_energy: f64 = coeffs.smooth().iter().map(|x| x * x).sum();
        let detail_energy: f64 = (0..4)
            .map(|j| coeffs.detail(j).unwrap().iter().map(|x| x * x).sum::<f64>())
            .sum();

        let total_energy = smooth_energy + detail_energy;
        assert!(
            (input_energy - total_energy).abs() / input_energy < 1e-10,
            "energy not conserved: input={}, total={}",
            input_energy,
            total_energy
        );
    }

    #[test]
    fn modwt_imodwt_roundtrip_haar() {
        let n = 64;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
        let ts = TimeSeries::new(data.clone()).unwrap();
        let config = ModwtConfig::new(WaveletFilter::Haar, 4);
        let coeffs = modwt(&ts, &config).unwrap();
        let reconstructed = imodwt(&coeffs).unwrap();

        assert_eq!(reconstructed.len(), n);
        for (i, (&orig, &recon)) in data.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-10,
                "roundtrip mismatch at {}: orig={}, recon={}",
                i,
                orig,
                recon
            );
        }
    }

    #[test]
    fn modwt_imodwt_roundtrip_la8() {
        let n = 256;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (t * 0.05).sin() + 0.5 * (t * 0.2).cos() + 0.3 * (t * 0.5).sin()
            })
            .collect();
        let ts = TimeSeries::new(data.clone()).unwrap();
        let config = ModwtConfig::new(WaveletFilter::La8, 4);
        let coeffs = modwt(&ts, &config).unwrap();
        let reconstructed = imodwt(&coeffs).unwrap();

        assert_eq!(reconstructed.len(), n);
        for (i, (&orig, &recon)) in data.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-10,
                "roundtrip mismatch at {}: orig={}, recon={}",
                i,
                orig,
                recon
            );
        }
    }

    #[test]
    fn modwt_imodwt_roundtrip_all_filters() {
        let n = 128;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.15).sin()).collect();
        let ts = TimeSeries::new(data.clone()).unwrap();

        for filter in [
            WaveletFilter::Haar,
            WaveletFilter::D4,
            WaveletFilter::D6,
            WaveletFilter::D8,
            WaveletFilter::La8,
            WaveletFilter::La16,
        ] {
            let max_level = max_modwt_level(n, &filter);
            let levels = max_level.min(3);
            if levels == 0 {
                continue;
            }
            let config = ModwtConfig::new(filter, levels);
            let coeffs = modwt(&ts, &config).unwrap();
            let reconstructed = imodwt(&coeffs).unwrap();

            for (i, (&orig, &recon)) in data.iter().zip(reconstructed.iter()).enumerate() {
                assert!(
                    (orig - recon).abs() < 1e-10,
                    "roundtrip mismatch for {:?} at {}: orig={}, recon={}",
                    filter,
                    i,
                    orig,
                    recon
                );
            }
        }
    }

    #[test]
    fn imodwt_preserves_length() {
        let n = 64;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = ModwtConfig::new(WaveletFilter::D4, 3);
        let coeffs = modwt(&ts, &config).unwrap();
        let reconstructed = imodwt(&coeffs).unwrap();
        assert_eq!(reconstructed.len(), n);
    }
}
