//! Continuous Wavelet Transform (CWT) with the Morlet wavelet.
//!
//! Computes the time-frequency power spectrum via FFT-based convolution,
//! following Torrence & Compo (1998).

use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

use crate::error::WaveletError;
use crate::series::TimeSeries;

/// Configuration for the Continuous Wavelet Transform.
///
/// Use the builder methods to customize the analysis parameters.
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::CwtConfig;
///
/// let config = CwtConfig::new()
///     .with_omega0(6.0)
///     .with_dt(1.0)
///     .with_dj(0.125);
/// ```
#[derive(Clone, Debug)]
pub struct CwtConfig {
    /// Morlet non-dimensional frequency.
    omega0: f64,
    /// Time step between observations.
    dt: f64,
    /// Fractional octave spacing.
    dj: f64,
    /// Smallest scale (None = auto = 2*dt).
    s0: Option<f64>,
    /// Number of scales (None = auto-computed).
    j_max: Option<usize>,
    /// Whether to subtract mean and divide by sample standard deviation (N-1 denominator).
    standardize: bool,
}

impl CwtConfig {
    /// Creates a new `CwtConfig` with default parameters.
    ///
    /// Defaults: `omega0 = 6.0`, `dt = 1.0`, `dj = 0.25`,
    /// `s0 = None` (auto), `j_max = None` (auto), `standardize = true`.
    pub fn new() -> Self {
        Self {
            omega0: 6.0,
            dt: 1.0,
            dj: 0.25,
            s0: None,
            j_max: None,
            standardize: true,
        }
    }

    /// Sets the Morlet non-dimensional frequency.
    pub fn with_omega0(mut self, omega0: f64) -> Self {
        self.omega0 = omega0;
        self
    }

    /// Sets the time step between observations.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Sets the fractional octave spacing.
    pub fn with_dj(mut self, dj: f64) -> Self {
        self.dj = dj;
        self
    }

    /// Sets the smallest scale.
    pub fn with_s0(mut self, s0: f64) -> Self {
        self.s0 = Some(s0);
        self
    }

    /// Sets the number of scales.
    pub fn with_j_max(mut self, j_max: usize) -> Self {
        self.j_max = Some(j_max);
        self
    }

    /// Sets whether to standardize the input signal.
    pub fn with_standardize(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }

    /// Returns the Morlet non-dimensional frequency.
    pub fn omega0(&self) -> f64 {
        self.omega0
    }

    /// Returns the time step.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Returns the fractional octave spacing.
    pub fn dj(&self) -> f64 {
        self.dj
    }

    /// Returns the smallest scale, if explicitly set.
    pub fn s0(&self) -> Option<f64> {
        self.s0
    }

    /// Returns the number of scales, if explicitly set.
    pub fn j_max(&self) -> Option<usize> {
        self.j_max
    }

    /// Returns whether standardization is enabled.
    pub fn standardize(&self) -> bool {
        self.standardize
    }
}

impl Default for CwtConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a Continuous Wavelet Transform.
///
/// Contains the complex wavelet coefficients (in standardized space when
/// `standardize = true`), scales, periods, cone of influence, and signal statistics.
#[derive(Clone, Debug)]
pub struct CwtResult {
    /// Complex wavelet coefficients `[n_scales][n_times]`.
    coefficients: Vec<Vec<Complex<f64>>>,
    /// Scale values.
    scales: Vec<f64>,
    /// Fourier-equivalent periods.
    periods: Vec<f64>,
    /// Cone of influence per time point.
    coi: Vec<f64>,
    /// Original signal mean.
    signal_mean: f64,
    /// Original signal sample standard deviation (N-1 denominator).
    signal_std: f64,
    /// Time step used.
    dt: f64,
    /// Scale spacing used.
    dj: f64,
}

impl CwtResult {
    /// Creates a new `CwtResult` (crate-internal constructor).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        coefficients: Vec<Vec<Complex<f64>>>,
        scales: Vec<f64>,
        periods: Vec<f64>,
        coi: Vec<f64>,
        signal_mean: f64,
        signal_std: f64,
        dt: f64,
        dj: f64,
    ) -> Self {
        Self {
            coefficients,
            scales,
            periods,
            coi,
            signal_mean,
            signal_std,
            dt,
            dj,
        }
    }

    /// Returns the complex wavelet coefficients `[n_scales][n_times]`.
    pub fn coefficients(&self) -> &[Vec<Complex<f64>>] {
        &self.coefficients
    }

    /// Returns the scale values.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Returns the Fourier-equivalent periods.
    pub fn periods(&self) -> &[f64] {
        &self.periods
    }

    /// Returns the cone of influence per time point.
    pub fn coi(&self) -> &[f64] {
        &self.coi
    }

    /// Returns the original signal mean.
    pub fn signal_mean(&self) -> f64 {
        self.signal_mean
    }

    /// Returns the original signal sample standard deviation (N-1 denominator).
    ///
    /// To rescale standardized power back to original units, multiply by `signal_std().powi(2)`.
    pub fn signal_std(&self) -> f64 {
        self.signal_std
    }

    /// Returns the time step used.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Returns the scale spacing used.
    pub fn dj(&self) -> f64 {
        self.dj
    }

    /// Returns the number of scales.
    pub fn n_scales(&self) -> usize {
        self.scales.len()
    }

    /// Returns the number of time points.
    pub fn n_times(&self) -> usize {
        self.coi.len()
    }

    /// Computes the wavelet power spectrum `|W(s,t)|^2` in standardized units.
    ///
    /// Returns a `[n_scales][n_times]` matrix of power values.
    pub fn power(&self) -> Vec<Vec<f64>> {
        self.coefficients
            .iter()
            .map(|row| row.iter().map(|c| c.norm_sqr()).collect())
            .collect()
    }

    /// Computes the global wavelet spectrum in standardized units.
    ///
    /// For each scale, returns the time-averaged power:
    /// `(1/N) * sum_t |W(s,t)|^2`.
    ///
    /// For original-unit GWS, see [`GwsResult::gws_unmasked`](crate::significance::GwsResult::gws_unmasked).
    pub fn global_wavelet_spectrum(&self) -> Vec<f64> {
        let n = self.n_times() as f64;
        self.coefficients
            .iter()
            .map(|row| {
                let sum: f64 = row.iter().map(|c| c.norm_sqr()).sum();
                sum / n
            })
            .collect()
    }
}

/// Computes the Continuous Wavelet Transform using the Morlet wavelet.
///
/// # Errors
///
/// | Variant | Trigger |
/// |---------|---------|
/// | [`WaveletError::InvalidConfig`] | invalid configuration parameters |
/// | [`WaveletError::SeriesTooShort`] | series has fewer than 4 observations |
/// | [`WaveletError::CwtFailed`] | numerical failure during transform |
pub fn cwt_morlet(series: &TimeSeries, config: &CwtConfig) -> Result<CwtResult, WaveletError> {
    validate_config(config)?;

    let data = series.as_slice();
    let n = data.len();

    if n < 4 {
        return Err(WaveletError::SeriesTooShort { len: n, min: 4 });
    }

    // Compute signal statistics
    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n as f64 - 1.0);
    let std_dev = variance.sqrt();
    let effective_std = if std_dev > f64::EPSILON { std_dev } else { 1.0 };

    // Standardize or copy
    let signal: Vec<f64> = if config.standardize {
        data.iter().map(|&x| (x - mean) / effective_std).collect()
    } else {
        data.to_vec()
    };

    // Zero-pad to next power of two
    let npad = n.next_power_of_two();
    let mut padded: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat_n(Complex::new(0.0, 0.0), npad - n))
        .collect();

    // Plan FFTs
    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(npad);
    let fft_inverse = planner.plan_fft_inverse(npad);

    // Forward FFT of padded signal
    fft_forward.process(&mut padded);
    let signal_fft = padded;

    // Build scales
    let s0_eff = config.s0.unwrap_or(2.0 * config.dt);
    let scales = build_scales(n, config.dt, s0_eff, config.dj, config.j_max);

    // Build wavenumbers
    let k = build_wavenumbers(npad, config.dt);

    let omega0 = config.omega0;

    // Per-scale loop
    let mut coefficients = Vec::with_capacity(scales.len());

    for &scale in &scales {
        // Build Morlet daughter in frequency domain
        let daughter = morlet_daughter(&k, scale, omega0, config.dt);

        // Pointwise multiply
        let mut product: Vec<Complex<f64>> = signal_fft
            .iter()
            .zip(daughter.iter())
            .map(|(&s, &d)| s * d)
            .collect();

        // Inverse FFT
        fft_inverse.process(&mut product);

        // Normalize (rustfft is unnormalized)
        let norm = 1.0 / npad as f64;

        // Extract first n points
        let row: Vec<Complex<f64>> = product[..n]
            .iter()
            .map(|&c| Complex::new(c.re * norm, c.im * norm))
            .collect();

        coefficients.push(row);
    }

    // Compute periods
    let periods: Vec<f64> = scales
        .iter()
        .map(|&s| s * 4.0 * PI / (omega0 + (2.0 + omega0 * omega0).sqrt()))
        .collect();

    // Compute COI
    let coi = compute_coi(n, config.dt, omega0);

    Ok(CwtResult::new(
        coefficients,
        scales,
        periods,
        coi,
        mean,
        std_dev,
        config.dt,
        config.dj,
    ))
}

/// Validates the CWT configuration parameters.
fn validate_config(config: &CwtConfig) -> Result<(), WaveletError> {
    if config.omega0 < 5.0 {
        return Err(WaveletError::InvalidConfig(
            "omega0 must be >= 5.0".to_string(),
        ));
    }
    if config.dt <= 0.0 {
        return Err(WaveletError::InvalidConfig("dt must be > 0".to_string()));
    }
    if config.dj <= 0.0 {
        return Err(WaveletError::InvalidConfig("dj must be > 0".to_string()));
    }
    if let Some(s0) = config.s0
        && s0 <= 0.0
    {
        return Err(WaveletError::InvalidConfig("s0 must be > 0".to_string()));
    }
    Ok(())
}

/// Builds the geometric scale array.
///
/// Scales follow `s0 * 2^(j * dj)` for `j = 0..=J`.
fn build_scales(n: usize, dt: f64, s0: f64, dj: f64, j_max: Option<usize>) -> Vec<f64> {
    let j_count = match j_max {
        Some(j) => j,
        None => {
            let val = (((n as f64) * dt / s0).ln() / (2.0_f64.ln())) / dj;
            val.floor() as usize
        }
    };
    (0..=j_count)
        .map(|j| s0 * 2.0_f64.powf(j as f64 * dj))
        .collect()
}

/// Builds the angular wavenumber array for the FFT grid.
///
/// Positive frequencies for `i = 0..=npad/2`, negative for `i = npad/2+1..npad`.
fn build_wavenumbers(npad: usize, dt: f64) -> Vec<f64> {
    let df = 2.0 * PI / (npad as f64 * dt);
    (0..npad)
        .map(|i| {
            if i == 0 {
                0.0
            } else if i <= npad / 2 {
                i as f64 * df
            } else {
                -((npad - i) as f64) * df
            }
        })
        .collect()
}

/// Builds the Morlet wavelet daughter in the frequency domain.
///
/// Applies the Heaviside function (zero for negative wavenumbers).
fn morlet_daughter(k: &[f64], scale: f64, omega0: f64, dt: f64) -> Vec<Complex<f64>> {
    let norm = (2.0 * PI * scale / dt).sqrt() * PI.powf(-0.25);
    k.iter()
        .map(|&ki| {
            if ki > 0.0 {
                let exponent = -0.5 * (scale * ki - omega0).powi(2);
                Complex::new(norm * exponent.exp(), 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect()
}

/// Computes the cone of influence for each time point.
fn compute_coi(n: usize, dt: f64, omega0: f64) -> Vec<f64> {
    let fourier_factor = 4.0 * PI / (omega0 + (2.0 + omega0 * omega0).sqrt());
    let coi_factor = fourier_factor / 2.0_f64.sqrt();
    (0..n)
        .map(|t| {
            let dist = t.min(n - 1 - t) as f64;
            f64::max(coi_factor * dt * dist, 0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = CwtConfig::new();
        assert!((config.omega0() - 6.0).abs() < f64::EPSILON);
        assert!((config.dt() - 1.0).abs() < f64::EPSILON);
        assert!((config.dj() - 0.25).abs() < f64::EPSILON);
        assert!(config.s0().is_none());
        assert!(config.j_max().is_none());
        assert!(config.standardize());
    }

    #[test]
    fn config_builder() {
        let config = CwtConfig::new()
            .with_omega0(8.0)
            .with_dt(0.5)
            .with_dj(0.125)
            .with_s0(1.0)
            .with_j_max(10)
            .with_standardize(false);

        assert!((config.omega0() - 8.0).abs() < f64::EPSILON);
        assert!((config.dt() - 0.5).abs() < f64::EPSILON);
        assert!((config.dj() - 0.125).abs() < f64::EPSILON);
        assert!((config.s0().unwrap() - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.j_max().unwrap(), 10);
        assert!(!config.standardize());
    }

    #[test]
    fn config_default_trait() {
        let from_new = CwtConfig::new();
        let from_default = CwtConfig::default();
        assert!((from_new.omega0() - from_default.omega0()).abs() < f64::EPSILON);
        assert!((from_new.dt() - from_default.dt()).abs() < f64::EPSILON);
        assert!((from_new.dj() - from_default.dj()).abs() < f64::EPSILON);
        assert_eq!(from_new.s0(), from_default.s0());
        assert_eq!(from_new.j_max(), from_default.j_max());
        assert_eq!(from_new.standardize(), from_default.standardize());
    }

    #[test]
    fn config_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<CwtConfig>();
    }

    #[test]
    fn invalid_omega0() {
        let config = CwtConfig::new().with_omega0(4.0);
        let data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let err = cwt_morlet(&ts, &config).unwrap_err();
        assert!(matches!(err, WaveletError::InvalidConfig(_)));
    }

    #[test]
    fn invalid_dt() {
        let config = CwtConfig::new().with_dt(0.0);
        let data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let err = cwt_morlet(&ts, &config).unwrap_err();
        assert!(matches!(err, WaveletError::InvalidConfig(_)));
    }

    #[test]
    fn invalid_dj() {
        let config = CwtConfig::new().with_dj(-0.1);
        let data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let err = cwt_morlet(&ts, &config).unwrap_err();
        assert!(matches!(err, WaveletError::InvalidConfig(_)));
    }

    #[test]
    fn series_too_short() {
        let config = CwtConfig::new();
        let ts = TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap();
        let err = cwt_morlet(&ts, &config).unwrap_err();
        assert!(matches!(
            err,
            WaveletError::SeriesTooShort { len: 3, min: 4 }
        ));
    }

    #[test]
    fn build_scales_geometric() {
        let s0 = 2.0;
        let dj = 0.25;
        let scales = build_scales(128, 1.0, s0, dj, Some(8));
        assert_eq!(scales.len(), 9); // 0..=8
        for (j, &scale) in scales.iter().enumerate() {
            let expected = s0 * 2.0_f64.powf(j as f64 * dj);
            assert!(
                (scale - expected).abs() < 1e-12,
                "scale[{}]: got {}, expected {}",
                j,
                scale,
                expected
            );
        }
    }

    #[test]
    fn build_scales_with_j_max() {
        let scales = build_scales(256, 1.0, 2.0, 0.25, Some(5));
        assert_eq!(scales.len(), 6); // j_max + 1
    }

    #[test]
    fn wavenumbers_ordering() {
        let npad = 16;
        let k = build_wavenumbers(npad, 1.0);
        assert_eq!(k.len(), npad);
        // k[0] should be 0
        assert!(k[0].abs() < f64::EPSILON);
        // First half (1..=npad/2) should be positive
        for (i, &ki) in k.iter().enumerate().take(npad / 2 + 1).skip(1) {
            assert!(ki > 0.0, "k[{}] = {} should be positive", i, ki);
        }
        // Second half (npad/2+1..npad) should be negative
        for (i, &ki) in k.iter().enumerate().take(npad).skip(npad / 2 + 1) {
            assert!(ki < 0.0, "k[{}] = {} should be negative", i, ki);
        }
    }

    #[test]
    fn morlet_daughter_heaviside() {
        let k = build_wavenumbers(16, 1.0);
        let daughter = morlet_daughter(&k, 4.0, 6.0, 1.0);
        // k[0] = 0 -> should be zero
        assert!(daughter[0].re.abs() < f64::EPSILON);
        assert!(daughter[0].im.abs() < f64::EPSILON);
        // Negative wavenumber entries should be zero
        for (i, d) in daughter.iter().enumerate().take(16).skip(16 / 2 + 1) {
            assert!(
                d.norm() < f64::EPSILON,
                "daughter[{}] should be zero for negative k",
                i
            );
        }
    }

    #[test]
    fn coi_symmetric() {
        let n = 64;
        let coi = compute_coi(n, 1.0, 6.0);
        assert_eq!(coi.len(), n);
        for t in 0..n {
            assert!(
                (coi[t] - coi[n - 1 - t]).abs() < 1e-12,
                "COI not symmetric: coi[{}]={} != coi[{}]={}",
                t,
                coi[t],
                n - 1 - t,
                coi[n - 1 - t]
            );
        }
    }

    #[test]
    fn output_dimensions() {
        let n = 128;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = CwtConfig::new();
        let result = cwt_morlet(&ts, &config).unwrap();

        assert_eq!(result.n_scales(), result.scales().len());
        assert_eq!(result.n_times(), n);
        assert_eq!(result.coefficients().len(), result.n_scales());
        for row in result.coefficients() {
            assert_eq!(row.len(), n);
        }
    }

    #[test]
    fn periods_monotonic() {
        let n = 128;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = CwtConfig::new();
        let result = cwt_morlet(&ts, &config).unwrap();

        let periods = result.periods();
        assert!(!periods.is_empty());
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
    fn power_nonnegative() {
        let n = 128;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = CwtConfig::new();
        let result = cwt_morlet(&ts, &config).unwrap();

        let power = result.power();
        for (s, row) in power.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                assert!(val >= 0.0, "power[{}][{}] = {} is negative", s, t, val);
            }
        }
    }

    #[test]
    fn gws_length() {
        let n = 128;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = CwtConfig::new();
        let result = cwt_morlet(&ts, &config).unwrap();

        let gws = result.global_wavelet_spectrum();
        assert_eq!(gws.len(), result.n_scales());
    }

    #[test]
    fn sine_peak_at_known_period() {
        let data: Vec<f64> = (0..512)
            .map(|i| (2.0 * PI * i as f64 / 32.0).sin())
            .collect();
        let ts = TimeSeries::new(data).unwrap();
        let config = CwtConfig::new().with_dj(0.125);
        let result = cwt_morlet(&ts, &config).unwrap();

        let gws = result.global_wavelet_spectrum();
        let periods = result.periods();

        // Find peak
        let (peak_idx, _) = gws
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_period = periods[peak_idx];
        let relative_error = ((peak_period - 32.0) / 32.0).abs();
        assert!(
            relative_error < 0.15,
            "peak period {} is not within 15% of 32 (error: {:.1}%)",
            peak_period,
            relative_error * 100.0
        );
    }

    #[test]
    fn constant_signal_near_zero_power() {
        let data = vec![42.0; 128];
        let ts = TimeSeries::new(data).unwrap();
        let config = CwtConfig::new();
        let result = cwt_morlet(&ts, &config).unwrap();

        let power = result.power();
        for (s, row) in power.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                assert!(
                    val < 1e-10,
                    "power[{}][{}] = {} should be near zero for constant signal",
                    s,
                    t,
                    val
                );
            }
        }
    }

    #[test]
    fn signal_stats_stored() {
        let data: Vec<f64> = (0..64).map(|i| i as f64 * 2.0 + 10.0).collect();
        let ts = TimeSeries::new(data.clone()).unwrap();
        let config = CwtConfig::new();
        let result = cwt_morlet(&ts, &config).unwrap();

        let n = data.len() as f64;
        let expected_mean = data.iter().sum::<f64>() / n;
        let expected_var = data
            .iter()
            .map(|&x| (x - expected_mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let expected_std = expected_var.sqrt();

        assert!(
            (result.signal_mean() - expected_mean).abs() < 1e-10,
            "mean: got {}, expected {}",
            result.signal_mean(),
            expected_mean
        );
        assert!(
            (result.signal_std() - expected_std).abs() < 1e-10,
            "std: got {}, expected {}",
            result.signal_std(),
            expected_std
        );
    }

    #[test]
    fn result_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<CwtResult>();
    }
}
