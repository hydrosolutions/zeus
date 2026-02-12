//! Spectral peak identification and matching for the WARM filter.

/// A significant spectral peak identified in the global wavelet spectrum.
#[derive(Clone, Debug)]
pub(crate) struct Peak {
    /// Index into the GWS/periods arrays.
    #[allow(dead_code)]
    pub index: usize,
    /// Period at this peak.
    pub period: f64,
    /// Power at this peak.
    pub power: f64,
    /// Signal-to-noise ratio (power / significance threshold).
    pub snr: f64,
}

/// Spectral quality metrics for a simulated series.
#[derive(Clone, Debug)]
pub(crate) struct SpectralMetrics {
    /// Pearson correlation of log-GWS (None if insufficient data).
    pub spectral_cor: Option<f64>,
    /// Fraction of observed peaks matched in the simulation.
    pub peak_match_frac: f64,
}

/// Identify significant peaks in the global wavelet spectrum.
///
/// Finds local maxima that exceed the significance threshold, ranks them by
/// signal-to-noise ratio, and returns the top `n_max` peaks.
pub(crate) fn identify_significant_peaks(
    gws: &[f64],
    signif: &[f64],
    periods: &[f64],
    n_max: usize,
) -> Vec<Peak> {
    let len = gws.len();
    if len < 3 {
        return Vec::new();
    }

    // 1. Find local maxima that exceed significance.
    let mut candidates: Vec<Peak> = Vec::new();
    for i in 1..len - 1 {
        if gws[i] > gws[i - 1] && gws[i] > gws[i + 1] && gws[i] > signif[i] {
            let snr = gws[i] / signif[i].max(1e-12);
            candidates.push(Peak {
                index: i,
                period: periods[i],
                power: gws[i],
                snr,
            });
        }
    }

    // 2. Sort by SNR descending, then by power descending as tiebreaker.
    candidates.sort_by(|a, b| {
        b.snr
            .partial_cmp(&a.snr)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.power
                    .partial_cmp(&a.power)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // 3. Take top n_max.
    candidates.truncate(n_max);
    candidates
}

/// Pearson correlation of log-transformed GWS vectors.
///
/// Returns `None` if the vectors differ in length, fewer than 3 finite pairs
/// remain after log-transformation, or the denominator is zero (constant
/// values).
pub(crate) fn spectral_correlation(gws_obs: &[f64], gws_sim: &[f64], eps: f64) -> Option<f64> {
    if gws_obs.len() != gws_sim.len() {
        return None;
    }

    // Log-transform with floor at eps.
    let log_obs: Vec<f64> = gws_obs.iter().map(|&v| v.max(eps).ln()).collect();
    let log_sim: Vec<f64> = gws_sim.iter().map(|&v| v.max(eps).ln()).collect();

    // Filter to finite pairs.
    let pairs: Vec<(f64, f64)> = log_obs
        .iter()
        .zip(log_sim.iter())
        .filter(|&(&a, &b)| a.is_finite() && b.is_finite())
        .map(|(&a, &b)| (a, b))
        .collect();

    if pairs.len() < 3 {
        return None;
    }

    // Inline Pearson correlation.
    let n = pairs.len() as f64;
    let mx: f64 = pairs.iter().map(|(xi, _)| xi).sum::<f64>() / n;
    let my: f64 = pairs.iter().map(|(_, yi)| yi).sum::<f64>() / n;

    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    for &(xi, yi) in &pairs {
        let dx = xi - mx;
        let dy = yi - my;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    let denom: f64 = (sum_xx * sum_yy).sqrt();
    if denom == 0.0 {
        return None;
    }

    Some(sum_xy / denom)
}

/// Fraction of observed peaks matched in a simulated GWS.
///
/// A peak is considered matched if there exists an index `j` in the simulated
/// spectrum whose period is within `period_tol` (on the log2 scale) and whose
/// log-power difference is within `mag_tol_log`.
///
/// Returns 1.0 if the peak list is empty.
pub(crate) fn peak_match_fraction(
    gws_sim: &[f64],
    periods: &[f64],
    peaks: &[Peak],
    period_tol: f64,
    mag_tol_log: f64,
    eps: f64,
) -> f64 {
    if peaks.is_empty() {
        return 1.0;
    }

    let mut matched = 0usize;

    for peak in peaks {
        let log2_peak_period = peak.period.log2();

        // Find the candidate with the highest simulated power within the
        // period tolerance window.
        let best = periods
            .iter()
            .enumerate()
            .filter(|&(j, &pj)| {
                j < gws_sim.len() && (pj.log2() - log2_peak_period).abs() <= period_tol
            })
            .max_by(|&(a_idx, _), &(b_idx, _)| {
                gws_sim[a_idx]
                    .partial_cmp(&gws_sim[b_idx])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some((j, _)) = best {
            let log_diff = ((gws_sim[j] + eps).ln() - (peak.power + eps).ln()).abs();
            if log_diff <= mag_tol_log {
                matched += 1;
            }
        }
    }

    matched as f64 / peaks.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identify_peaks_basic() {
        // Synthetic GWS with a clear peak at index 2.
        let gws = [1.0, 2.0, 10.0, 3.0, 1.0];
        let signif = [5.0, 5.0, 5.0, 5.0, 5.0];
        let periods = [2.0, 4.0, 8.0, 16.0, 32.0];

        let peaks = identify_significant_peaks(&gws, &signif, &periods, 10);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].index, 2);
        assert_relative_eq!(peaks[0].period, 8.0);
        assert_relative_eq!(peaks[0].power, 10.0);
        assert_relative_eq!(peaks[0].snr, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_identify_peaks_no_significant() {
        // GWS below significance everywhere.
        let gws = [1.0, 2.0, 3.0, 2.0, 1.0];
        let signif = [5.0, 5.0, 5.0, 5.0, 5.0];
        let periods = [2.0, 4.0, 8.0, 16.0, 32.0];

        let peaks = identify_significant_peaks(&gws, &signif, &periods, 10);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_identify_peaks_max_limit() {
        // 5 significant peaks, but n_max = 2.
        let gws = [0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 15.0, 0.0, 25.0, 0.0];
        let signif = [1.0; 11];
        let periods: Vec<f64> = (0..11).map(|i| 2.0_f64.powi(i)).collect();

        let peaks = identify_significant_peaks(&gws, &signif, &periods, 2);
        assert_eq!(peaks.len(), 2);
        // Highest SNR = 30/1 = 30, then 25/1 = 25.
        assert_eq!(peaks[0].index, 5);
        assert_eq!(peaks[1].index, 9);
    }

    #[test]
    fn test_identify_peaks_edge_at_boundary() {
        // "Peak" at index 0 (boundary) — should not be found.
        let gws = [100.0, 2.0, 1.0, 0.5, 0.1];
        let signif = [1.0; 5];
        let periods = [2.0, 4.0, 8.0, 16.0, 32.0];

        let peaks = identify_significant_peaks(&gws, &signif, &periods, 10);
        // Index 0 is not a local max (no left neighbour), last index similarly.
        assert!(peaks.is_empty());

        // "Peak" at last index.
        let gws2 = [0.1, 0.5, 1.0, 2.0, 100.0];
        let peaks2 = identify_significant_peaks(&gws2, &signif, &periods, 10);
        assert!(peaks2.is_empty());
    }

    #[test]
    fn test_spectral_correlation_identical() {
        let gws = [1.0, 4.0, 9.0, 16.0, 25.0];
        let r = spectral_correlation(&gws, &gws, 1e-12);
        assert_relative_eq!(r.expect("should be Some"), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_spectral_correlation_different_length() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0];
        assert!(spectral_correlation(&a, &b, 1e-12).is_none());
    }

    #[test]
    fn test_spectral_correlation_insufficient() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        assert!(spectral_correlation(&a, &b, 1e-12).is_none());
    }

    #[test]
    fn test_peak_match_fraction_exact() {
        // Simulated GWS has peaks at exactly the same periods with same magnitude.
        let gws_sim = [1.0, 10.0, 1.0, 20.0, 1.0];
        let periods = [2.0, 4.0, 8.0, 16.0, 32.0];
        let peaks = vec![
            Peak {
                index: 1,
                period: 4.0,
                power: 10.0,
                snr: 5.0,
            },
            Peak {
                index: 3,
                period: 16.0,
                power: 20.0,
                snr: 10.0,
            },
        ];

        let frac = peak_match_fraction(&gws_sim, &periods, &peaks, 0.5, 1.0, 1e-12);
        assert_relative_eq!(frac, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_peak_match_fraction_partial() {
        // 2 peaks: first matched, second not (magnitude way off).
        let gws_sim = [1.0, 10.0, 1.0, 0.001, 1.0];
        let periods = [2.0, 4.0, 8.0, 16.0, 32.0];
        let peaks = vec![
            Peak {
                index: 1,
                period: 4.0,
                power: 10.0,
                snr: 5.0,
            },
            Peak {
                index: 3,
                period: 16.0,
                power: 20.0,
                snr: 10.0,
            },
        ];

        let frac = peak_match_fraction(&gws_sim, &periods, &peaks, 0.5, 1.0, 1e-12);
        assert_relative_eq!(frac, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_peak_match_fraction_no_peaks() {
        let gws_sim = [1.0, 2.0, 3.0];
        let periods = [2.0, 4.0, 8.0];
        let peaks: Vec<Peak> = Vec::new();

        let frac = peak_match_fraction(&gws_sim, &periods, &peaks, 0.5, 1.0, 1e-12);
        assert_relative_eq!(frac, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_peak_match_fraction_period_shift() {
        // Peak period is shifted way beyond tolerance.
        let gws_sim = [1.0, 10.0, 1.0, 1.0, 1.0];
        let periods = [2.0, 4.0, 8.0, 16.0, 32.0];
        let peaks = vec![Peak {
            index: 3,
            period: 16.0,
            power: 10.0,
            snr: 5.0,
        }];

        // Tolerance is very tight (0.01 on log2 scale). The only candidate
        // with high power is at period=4, which is log2(4)=2 vs log2(16)=4,
        // difference=2, well beyond 0.01.
        let frac = peak_match_fraction(&gws_sim, &periods, &peaks, 0.01, 1.0, 1e-12);
        assert_relative_eq!(frac, 0.0, epsilon = 1e-6);
    }
}
