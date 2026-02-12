//! WARM pool filtering.

use rayon::prelude::*;
use zeus_wavelet::{CwtConfig, SignificanceConfig, TimeSeries, cwt_morlet, test_significance};

use crate::error::WarmError;
use crate::spectral::{
    Peak, SpectralMetrics, identify_significant_peaks, peak_match_fraction, spectral_correlation,
};
use crate::warm::WarmResult;

// Relaxation constants (matching R implementation)
const MAX_RELAXATION_ITER: usize = 20;
const MEAN_SD_RELAX_FACTOR: f64 = 1.25;
const MEAN_SD_RELAX_CEIL: f64 = 0.25;
const TAIL_LOG_RELAX_FACTOR: f64 = 1.25;
const TAIL_P_STEP: f64 = 0.02;
const TAIL_P_FLOOR: f64 = 0.40;
const SPECTRAL_CORR_STEP: f64 = 0.05;
const SPECTRAL_CORR_FLOOR: f64 = 0.30;
const PEAK_FRAC_STEP: f64 = 0.10;

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
    /// Relative tolerance on the mean (default 0.03).
    mean_tol: f64,
    /// Relative tolerance on the standard deviation (default 0.03).
    sd_tol: f64,
    /// Lower quantile probability for the tail filter (default 0.20).
    tail_low_p: f64,
    /// Upper quantile probability for the tail filter (default 0.80).
    tail_high_p: f64,
    /// Log-ratio tolerance for tail mass comparison (default ln(1.03)).
    tail_tol_log: f64,
    /// Small epsilon added before taking log in tail filter (default 1e-5).
    tail_eps: f64,
    /// Minimum spectral correlation for wavelet filter (default 0.60).
    spectral_corr_min: f64,
    /// Small epsilon for spectral computations (default 1e-10).
    spectral_eps: f64,
    /// Maximum number of significant peaks to match (default 2).
    n_sig_peaks_max: usize,
    /// Period tolerance for peak matching (default 0.50).
    peak_period_tol: f64,
    /// Log-ratio tolerance for peak magnitude comparison (default ln(1.5)).
    peak_mag_tol_log: f64,
    /// Minimum fraction of observed peaks that must be matched (default 1.0).
    peak_match_frac_min: f64,
    /// Number of simulations to select (default 5).
    n_select: usize,
    /// Configuration for the CWT analysis.
    cwt_config: CwtConfig,
    /// Configuration for GWS significance testing.
    significance_config: SignificanceConfig,
}

impl Default for FilterBounds {
    /// Returns default filter bounds.
    ///
    /// | Parameter | Default |
    /// |-----------|---------|
    /// | `mean_tol` | 0.03 |
    /// | `sd_tol` | 0.03 |
    /// | `tail_low_p` | 0.20 |
    /// | `tail_high_p` | 0.80 |
    /// | `tail_tol_log` | ln(1.03) |
    /// | `tail_eps` | 1e-5 |
    /// | `spectral_corr_min` | 0.60 |
    /// | `spectral_eps` | 1e-10 |
    /// | `n_sig_peaks_max` | 2 |
    /// | `peak_period_tol` | 0.50 |
    /// | `peak_mag_tol_log` | ln(1.5) |
    /// | `peak_match_frac_min` | 1.0 |
    /// | `n_select` | 5 |
    fn default() -> Self {
        Self {
            mean_tol: 0.03,
            sd_tol: 0.03,
            tail_low_p: 0.20,
            tail_high_p: 0.80,
            tail_tol_log: 1.03_f64.ln(),
            tail_eps: 1e-5,
            spectral_corr_min: 0.60,
            spectral_eps: 1e-10,
            n_sig_peaks_max: 2,
            peak_period_tol: 0.50,
            peak_mag_tol_log: 1.5_f64.ln(),
            peak_match_frac_min: 1.0,
            n_select: 5,
            cwt_config: CwtConfig::new(),
            significance_config: SignificanceConfig::new(),
        }
    }
}

impl FilterBounds {
    /// Sets the mean tolerance (relative).
    pub fn with_mean_tol(mut self, tol: f64) -> Self {
        self.mean_tol = tol;
        self
    }

    /// Returns the mean tolerance.
    pub fn mean_tol(&self) -> f64 {
        self.mean_tol
    }

    /// Sets the standard deviation tolerance (relative).
    pub fn with_sd_tol(mut self, tol: f64) -> Self {
        self.sd_tol = tol;
        self
    }

    /// Returns the standard deviation tolerance.
    pub fn sd_tol(&self) -> f64 {
        self.sd_tol
    }

    /// Sets the lower quantile probability for the tail filter.
    pub fn with_tail_low_p(mut self, p: f64) -> Self {
        self.tail_low_p = p;
        self
    }

    /// Returns the lower quantile probability for the tail filter.
    pub fn tail_low_p(&self) -> f64 {
        self.tail_low_p
    }

    /// Sets the upper quantile probability for the tail filter.
    pub fn with_tail_high_p(mut self, p: f64) -> Self {
        self.tail_high_p = p;
        self
    }

    /// Returns the upper quantile probability for the tail filter.
    pub fn tail_high_p(&self) -> f64 {
        self.tail_high_p
    }

    /// Sets the log-ratio tolerance for tail mass comparison.
    pub fn with_tail_tol_log(mut self, tol: f64) -> Self {
        self.tail_tol_log = tol;
        self
    }

    /// Returns the log-ratio tolerance for tail mass comparison.
    pub fn tail_tol_log(&self) -> f64 {
        self.tail_tol_log
    }

    /// Sets the epsilon added before log in the tail filter.
    pub fn with_tail_eps(mut self, eps: f64) -> Self {
        self.tail_eps = eps;
        self
    }

    /// Returns the epsilon added before log in the tail filter.
    pub fn tail_eps(&self) -> f64 {
        self.tail_eps
    }

    /// Sets the minimum spectral correlation for the wavelet filter.
    pub fn with_spectral_corr_min(mut self, min: f64) -> Self {
        self.spectral_corr_min = min;
        self
    }

    /// Returns the minimum spectral correlation.
    pub fn spectral_corr_min(&self) -> f64 {
        self.spectral_corr_min
    }

    /// Sets the epsilon for spectral computations.
    pub fn with_spectral_eps(mut self, eps: f64) -> Self {
        self.spectral_eps = eps;
        self
    }

    /// Returns the epsilon for spectral computations.
    pub fn spectral_eps(&self) -> f64 {
        self.spectral_eps
    }

    /// Sets the maximum number of significant peaks to match.
    pub fn with_n_sig_peaks_max(mut self, n: usize) -> Self {
        self.n_sig_peaks_max = n;
        self
    }

    /// Returns the maximum number of significant peaks to match.
    pub fn n_sig_peaks_max(&self) -> usize {
        self.n_sig_peaks_max
    }

    /// Sets the period tolerance for peak matching.
    pub fn with_peak_period_tol(mut self, tol: f64) -> Self {
        self.peak_period_tol = tol;
        self
    }

    /// Returns the period tolerance for peak matching.
    pub fn peak_period_tol(&self) -> f64 {
        self.peak_period_tol
    }

    /// Sets the log-ratio tolerance for peak magnitude comparison.
    pub fn with_peak_mag_tol_log(mut self, tol: f64) -> Self {
        self.peak_mag_tol_log = tol;
        self
    }

    /// Returns the log-ratio tolerance for peak magnitude comparison.
    pub fn peak_mag_tol_log(&self) -> f64 {
        self.peak_mag_tol_log
    }

    /// Sets the minimum fraction of observed peaks that must be matched.
    pub fn with_peak_match_frac_min(mut self, frac: f64) -> Self {
        self.peak_match_frac_min = frac;
        self
    }

    /// Returns the minimum fraction of observed peaks that must be matched.
    pub fn peak_match_frac_min(&self) -> f64 {
        self.peak_match_frac_min
    }

    /// Sets the number of simulations to select.
    pub fn with_n_select(mut self, n: usize) -> Self {
        self.n_select = n;
        self
    }

    /// Returns the number of simulations to select.
    pub fn n_select(&self) -> usize {
        self.n_select
    }

    /// Sets the CWT configuration for spectral analysis.
    pub fn with_cwt_config(mut self, config: CwtConfig) -> Self {
        self.cwt_config = config;
        self
    }

    /// Returns a reference to the CWT configuration.
    pub fn cwt_config(&self) -> &CwtConfig {
        &self.cwt_config
    }

    /// Sets the significance testing configuration.
    pub fn with_significance_config(mut self, config: SignificanceConfig) -> Self {
        self.significance_config = config;
        self
    }

    /// Returns a reference to the significance testing configuration.
    pub fn significance_config(&self) -> &SignificanceConfig {
        &self.significance_config
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

// ---------------------------------------------------------------------------
// Cheap filter helpers (private)
// ---------------------------------------------------------------------------

/// Filters simulations by relative mean closeness to the observed mean.
fn filter_mean(obs_mean: f64, sims: &[Vec<f64>], tol: f64) -> Vec<bool> {
    sims.iter()
        .map(|sim| {
            let sim_mean = crate::stats::mean(sim);
            if obs_mean.abs() < 1e-10 {
                (sim_mean - obs_mean).abs() <= tol
            } else {
                ((sim_mean - obs_mean) / obs_mean).abs() <= tol
            }
        })
        .collect()
}

/// Filters simulations by relative SD closeness to the observed SD.
fn filter_sd(obs_sd: f64, sims: &[Vec<f64>], tol: f64) -> Vec<bool> {
    sims.iter()
        .map(|sim| {
            let sim_sd = crate::stats::sd(sim);
            if obs_sd < 1e-10 {
                sim_sd < 1e-10
            } else {
                ((sim_sd - obs_sd) / obs_sd).abs() <= tol
            }
        })
        .collect()
}

/// Filters simulations by lower-tail deficit mass on a log scale.
fn filter_tail_low(
    observed: &[f64],
    sims: &[Vec<f64>],
    thr_low: f64,
    denom: f64,
    tail_eps: f64,
    tail_tol_log: f64,
) -> Vec<bool> {
    let obs_mass: f64 = observed
        .iter()
        .filter(|&&x| x < thr_low)
        .map(|&x| (thr_low - x).max(0.0))
        .sum::<f64>()
        / denom;

    sims.iter()
        .map(|sim| {
            let sim_mass: f64 = sim
                .iter()
                .filter(|&&x| x < thr_low)
                .map(|&x| (thr_low - x).max(0.0))
                .sum::<f64>()
                / denom;
            ((sim_mass + tail_eps).ln() - (obs_mass + tail_eps).ln()).abs() <= tail_tol_log
        })
        .collect()
}

/// Filters simulations by upper-tail excess mass on a log scale.
fn filter_tail_high(
    observed: &[f64],
    sims: &[Vec<f64>],
    thr_high: f64,
    denom: f64,
    tail_eps: f64,
    tail_tol_log: f64,
) -> Vec<bool> {
    let obs_mass: f64 = observed
        .iter()
        .filter(|&&x| x > thr_high)
        .map(|&x| (x - thr_high).max(0.0))
        .sum::<f64>()
        / denom;

    sims.iter()
        .map(|sim| {
            let sim_mass: f64 = sim
                .iter()
                .filter(|&&x| x > thr_high)
                .map(|&x| (x - thr_high).max(0.0))
                .sum::<f64>()
                / denom;
            ((sim_mass + tail_eps).ln() - (obs_mass + tail_eps).ln()).abs() <= tail_tol_log
        })
        .collect()
}

/// Computes spectral metrics for a single simulation against observed GWS.
#[allow(clippy::too_many_arguments)]
fn compute_spectral_metrics(
    sim: &[f64],
    gws_obs: &[f64],
    periods: &[f64],
    peaks: &[Peak],
    cwt_config: &CwtConfig,
    eps: f64,
    period_tol: f64,
    mag_tol_log: f64,
) -> Option<SpectralMetrics> {
    let ts = TimeSeries::new(sim.to_vec()).ok()?;
    let cwt_result = cwt_morlet(&ts, cwt_config).ok()?;
    let gws_sim = cwt_result.global_wavelet_spectrum();

    let spectral_cor = spectral_correlation(gws_obs, &gws_sim, eps);
    let peak_match_frac =
        peak_match_fraction(&gws_sim, periods, peaks, period_tol, mag_tol_log, eps);

    Some(SpectralMetrics {
        spectral_cor,
        peak_match_frac,
    })
}

/// Computes relative mean difference for sorting.
fn mean_rel_diff(obs_mean: f64, sim: &[f64]) -> f64 {
    let sim_mean = crate::stats::mean(sim);
    if obs_mean.abs() < 1e-10 {
        (sim_mean - obs_mean).abs()
    } else {
        ((sim_mean - obs_mean) / obs_mean).abs()
    }
}

/// Ranks candidates and selects top `n_select` based on spectral metrics.
fn rank_and_select(
    candidates: &[usize],
    spectral_metrics: &[Option<SpectralMetrics>],
    obs_peaks: &[Peak],
    n_select: usize,
) -> FilteredPool {
    let mut ranked: Vec<(usize, f64, f64)> = candidates
        .iter()
        .map(|&i| {
            let (cor, peak_frac) = spectral_metrics[i]
                .as_ref()
                .map(|m| {
                    (
                        m.spectral_cor.unwrap_or(f64::NEG_INFINITY),
                        m.peak_match_frac,
                    )
                })
                .unwrap_or((f64::NEG_INFINITY, 0.0));
            (i, cor, peak_frac)
        })
        .collect();

    if !obs_peaks.is_empty() {
        // Sort by peak_match_frac desc, then spectral_cor desc
        ranked.sort_by(|a, b| {
            let cmp_peak = b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal);
            cmp_peak.then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });
    } else {
        // Sort by spectral_cor desc only
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    ranked.truncate(n_select);
    let selected: Vec<usize> = ranked.iter().map(|&(i, _, _)| i).collect();
    let scores: Vec<f64> = ranked
        .iter()
        .map(|&(_, cor, _)| if cor.is_finite() { cor } else { 0.0 })
        .collect();

    FilteredPool::new(selected, scores)
}

/// Filters a WARM simulation pool against observed data.
///
/// Evaluates each simulation against the observed data using the
/// specified tolerance bounds and returns the indices and scores
/// of simulations that pass all criteria.
///
/// The implementation applies cheap statistical filters (mean, SD,
/// lower-tail mass, upper-tail mass), wavelet-based spectral filtering
/// (GWS correlation, peak matching), and adaptive relaxation when
/// insufficient candidates pass the initial thresholds.
///
/// # Errors
///
/// | Variant | Trigger |
/// |---------|---------|
/// | [`WarmError::InsufficientSimulations`] | pool is empty |
pub fn filter_warm_pool(
    observed: &[f64],
    result: &WarmResult,
    bounds: &FilterBounds,
) -> Result<FilteredPool, WarmError> {
    let sims = result.simulations();
    if sims.is_empty() {
        return Err(WarmError::InsufficientSimulations {
            requested: bounds.n_select(),
            available: 0,
        });
    }

    let n_sim = sims.len();
    let obs_mean = crate::stats::mean(observed);
    let obs_sd = crate::stats::sd(observed);

    // Compute observed quantiles for tail filters
    let mut sorted_obs = observed.to_vec();
    sorted_obs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let denom = crate::stats::robust_scale(observed);

    // Mutable bounds for relaxation
    let mut cur_mean_tol = bounds.mean_tol();
    let mut cur_sd_tol = bounds.sd_tol();
    let mut cur_tail_tol_log = bounds.tail_tol_log();
    let mut cur_tail_low_p = bounds.tail_low_p();
    let mut cur_tail_high_p = bounds.tail_high_p();
    let mut cur_spectral_corr_min = bounds.spectral_corr_min();
    let mut cur_peak_match_frac_min = bounds.peak_match_frac_min();
    let mut wavelet_enabled = true;
    let mut peak_matching_enabled = true;

    // --- Observed CWT + significance (compute once) ---
    let obs_cwt = TimeSeries::new(observed.to_vec())
        .ok()
        .and_then(|ts| cwt_morlet(&ts, bounds.cwt_config()).ok());

    let (gws_obs, obs_periods, obs_peaks) = if let Some(ref cwt_result) = obs_cwt {
        let gws = cwt_result.global_wavelet_spectrum();
        let periods = cwt_result.periods().to_vec();

        // Try significance testing for peak identification
        let peaks = if let Ok(ts) = TimeSeries::new(observed.to_vec()) {
            if let Ok(gws_result) = test_significance(&ts, cwt_result, bounds.significance_config())
            {
                identify_significant_peaks(
                    &gws,
                    gws_result.significance_threshold(),
                    &periods,
                    bounds.n_sig_peaks_max(),
                )
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        (gws, periods, peaks)
    } else {
        // CWT failed on observed data -- disable wavelet filter
        wavelet_enabled = false;
        (Vec::new(), Vec::new(), Vec::new())
    };

    // --- Compute spectral metrics for all sims in parallel ---
    let spectral_metrics: Vec<Option<SpectralMetrics>> = if wavelet_enabled {
        sims.par_iter()
            .map(|sim| {
                compute_spectral_metrics(
                    sim,
                    &gws_obs,
                    &obs_periods,
                    &obs_peaks,
                    bounds.cwt_config(),
                    bounds.spectral_eps(),
                    bounds.peak_period_tol(),
                    bounds.peak_mag_tol_log(),
                )
            })
            .collect()
    } else {
        vec![None; n_sim]
    };

    // --- Adaptive relaxation loop ---
    let mut thr_low = crate::stats::quantile_type7(&sorted_obs, cur_tail_low_p);
    let mut thr_high = crate::stats::quantile_type7(&sorted_obs, cur_tail_high_p);

    for _iter in 0..MAX_RELAXATION_ITER {
        let pass_mean = filter_mean(obs_mean, sims, cur_mean_tol);
        let pass_sd = filter_sd(obs_sd, sims, cur_sd_tol);
        let pass_tail_low = filter_tail_low(
            observed,
            sims,
            thr_low,
            denom,
            bounds.tail_eps(),
            cur_tail_tol_log,
        );
        let pass_tail_high = filter_tail_high(
            observed,
            sims,
            thr_high,
            denom,
            bounds.tail_eps(),
            cur_tail_tol_log,
        );

        let pass_wavelet: Vec<bool> = if wavelet_enabled {
            spectral_metrics
                .iter()
                .map(|m| match m {
                    Some(metrics) => {
                        let cor_ok = metrics
                            .spectral_cor
                            .is_some_and(|c| c >= cur_spectral_corr_min);
                        let peak_ok = if peak_matching_enabled {
                            metrics.peak_match_frac >= cur_peak_match_frac_min
                                || obs_peaks.is_empty()
                        } else {
                            true
                        };
                        cor_ok && peak_ok
                    }
                    None => false,
                })
                .collect()
        } else {
            vec![true; n_sim]
        };

        // AND all filters
        let candidates: Vec<usize> = (0..n_sim)
            .filter(|&i| {
                pass_mean[i]
                    && pass_sd[i]
                    && pass_tail_low[i]
                    && pass_tail_high[i]
                    && pass_wavelet[i]
            })
            .collect();

        if candidates.len() >= bounds.n_select() {
            // Success -- rank and return
            return Ok(rank_and_select(
                &candidates,
                &spectral_metrics,
                &obs_peaks,
                bounds.n_select(),
            ));
        }

        // Find the filter with lowest pass rate and relax it
        let counts = [
            pass_mean.iter().filter(|&&v| v).count(),
            pass_sd.iter().filter(|&&v| v).count(),
            pass_tail_low.iter().filter(|&&v| v).count(),
            pass_tail_high.iter().filter(|&&v| v).count(),
            pass_wavelet.iter().filter(|&&v| v).count(),
        ];

        // Find the minimum count (most restrictive filter)
        let min_idx = counts
            .iter()
            .enumerate()
            .min_by_key(|&(_, &c)| c)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let mut changed = false;
        match min_idx {
            0 => {
                // Relax mean
                let new_tol = (cur_mean_tol * MEAN_SD_RELAX_FACTOR).min(MEAN_SD_RELAX_CEIL);
                if (new_tol - cur_mean_tol).abs() > 1e-15 {
                    cur_mean_tol = new_tol;
                    changed = true;
                }
            }
            1 => {
                // Relax SD
                let new_tol = (cur_sd_tol * MEAN_SD_RELAX_FACTOR).min(MEAN_SD_RELAX_CEIL);
                if (new_tol - cur_sd_tol).abs() > 1e-15 {
                    cur_sd_tol = new_tol;
                    changed = true;
                }
            }
            2 => {
                // Relax tail low: Stage 1 - increase log tolerance, Stage 2 - shift quantile
                let ln2 = 2.0_f64.ln();
                let new_log = (cur_tail_tol_log * TAIL_LOG_RELAX_FACTOR).min(ln2);
                if (new_log - cur_tail_tol_log).abs() > 1e-15 {
                    cur_tail_tol_log = new_log;
                    changed = true;
                } else {
                    let new_p = (cur_tail_low_p + TAIL_P_STEP).min(TAIL_P_FLOOR);
                    if (new_p - cur_tail_low_p).abs() > 1e-15 {
                        cur_tail_low_p = new_p;
                        thr_low = crate::stats::quantile_type7(&sorted_obs, cur_tail_low_p);
                        changed = true;
                    }
                }
            }
            3 => {
                // Relax tail high: Stage 1 - increase log tolerance, Stage 2 - shift quantile
                let ln2 = 2.0_f64.ln();
                let new_log = (cur_tail_tol_log * TAIL_LOG_RELAX_FACTOR).min(ln2);
                if (new_log - cur_tail_tol_log).abs() > 1e-15 {
                    cur_tail_tol_log = new_log;
                    changed = true;
                } else {
                    let new_p = (cur_tail_high_p - TAIL_P_STEP).max(TAIL_P_FLOOR);
                    if (cur_tail_high_p - new_p).abs() > 1e-15 {
                        cur_tail_high_p = new_p;
                        thr_high = crate::stats::quantile_type7(&sorted_obs, cur_tail_high_p);
                        changed = true;
                    }
                }
            }
            4 => {
                // Relax wavelet: Stage 1 - lower corr min, Stage 2 - lower peak frac,
                // Stage 3 - disable peak matching, Stage 4 - disable wavelet entirely
                if cur_spectral_corr_min > SPECTRAL_CORR_FLOOR + 1e-15 {
                    cur_spectral_corr_min =
                        (cur_spectral_corr_min - SPECTRAL_CORR_STEP).max(SPECTRAL_CORR_FLOOR);
                    changed = true;
                } else if peak_matching_enabled && cur_peak_match_frac_min > 1e-15 {
                    cur_peak_match_frac_min = (cur_peak_match_frac_min - PEAK_FRAC_STEP).max(0.0);
                    changed = true;
                } else if peak_matching_enabled {
                    peak_matching_enabled = false;
                    changed = true;
                } else if wavelet_enabled {
                    wavelet_enabled = false;
                    changed = true;
                }
            }
            _ => {}
        }

        if !changed {
            break;
        }
    }

    // --- Exhausted relaxation: fallback ---
    // Sort ALL sims by |mean_rel_diff|, take top n_select
    let mut all_indices: Vec<usize> = (0..n_sim).collect();
    all_indices.sort_by(|&a, &b| {
        let diff_a = mean_rel_diff(obs_mean, &sims[a]);
        let diff_b = mean_rel_diff(obs_mean, &sims[b]);
        diff_a
            .partial_cmp(&diff_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_indices.truncate(bounds.n_select());

    let scores = all_indices
        .iter()
        .map(|&i| {
            spectral_metrics[i]
                .as_ref()
                .and_then(|m| m.spectral_cor)
                .unwrap_or(0.0)
        })
        .collect();

    Ok(FilteredPool::new(all_indices, scores))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn filter_bounds_defaults() {
        let bounds = FilterBounds::default();
        assert_relative_eq!(bounds.mean_tol(), 0.03, epsilon = 1e-6);
        assert_relative_eq!(bounds.sd_tol(), 0.03, epsilon = 1e-6);
        assert_relative_eq!(bounds.tail_low_p(), 0.20, epsilon = 1e-6);
        assert_relative_eq!(bounds.tail_high_p(), 0.80, epsilon = 1e-6);
        assert_relative_eq!(bounds.tail_tol_log(), 1.03_f64.ln(), epsilon = 1e-6);
        assert_relative_eq!(bounds.tail_eps(), 1e-5, epsilon = 1e-12);
        assert_relative_eq!(bounds.spectral_corr_min(), 0.60, epsilon = 1e-6);
        assert_relative_eq!(bounds.spectral_eps(), 1e-10, epsilon = 1e-18);
        assert_eq!(bounds.n_sig_peaks_max(), 2);
        assert_relative_eq!(bounds.peak_period_tol(), 0.50, epsilon = 1e-6);
        assert_relative_eq!(bounds.peak_mag_tol_log(), 1.5_f64.ln(), epsilon = 1e-6);
        assert_relative_eq!(bounds.peak_match_frac_min(), 1.0, epsilon = 1e-6);
        assert_eq!(bounds.n_select(), 5);
    }

    #[test]
    fn filter_bounds_builder() {
        let bounds = FilterBounds::default()
            .with_mean_tol(0.05)
            .with_sd_tol(0.10)
            .with_n_select(10);
        assert_relative_eq!(bounds.mean_tol(), 0.05, epsilon = 1e-6);
        assert_relative_eq!(bounds.sd_tol(), 0.10, epsilon = 1e-6);
        assert_eq!(bounds.n_select(), 10);
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

    #[test]
    fn filter_mean_pass() {
        let sims = vec![vec![10.0, 10.1, 9.9]]; // mean ~ 10.0
        let pass = filter_mean(10.0, &sims, 0.03);
        assert!(pass[0]);
    }

    #[test]
    fn filter_mean_fail() {
        let sims = vec![vec![20.0, 20.1, 19.9]]; // mean ~ 20.0, obs_mean = 10.0
        let pass = filter_mean(10.0, &sims, 0.03);
        assert!(!pass[0]);
    }

    #[test]
    fn filter_sd_pass() {
        let sims = vec![vec![8.0, 10.0, 12.0]]; // sd ~ 2.0
        let pass = filter_sd(2.0, &sims, 0.03);
        assert!(pass[0]);
    }

    #[test]
    fn filter_sd_fail() {
        let sims = vec![vec![1.0, 100.0, 200.0]]; // sd >> 2.0
        let pass = filter_sd(2.0, &sims, 0.03);
        assert!(!pass[0]);
    }

    #[test]
    fn filter_tail_low_known() {
        let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sims = vec![obs.clone()]; // identical
        let thr = 2.0;
        let denom = 1.0;
        let pass = filter_tail_low(&obs, &sims, thr, denom, 1e-5, 0.5);
        assert!(pass[0]);
    }

    #[test]
    fn filter_tail_high_known() {
        let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sims = vec![obs.clone()]; // identical
        let thr = 4.0;
        let denom = 1.0;
        let pass = filter_tail_high(&obs, &sims, thr, denom, 1e-5, 0.5);
        assert!(pass[0]);
    }

    #[test]
    fn filter_warm_pool_empty_errors() {
        let result = crate::warm::WarmResult::new(vec![], vec![], vec![]);
        let bounds = FilterBounds::default();
        let err = filter_warm_pool(&[1.0, 2.0, 3.0], &result, &bounds);
        assert!(err.is_err());
    }

    #[test]
    fn filter_warm_pool_with_sims() {
        // Create synthetic observed data and matching simulations
        let observed: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.1).sin() * 5.0 + 10.0)
            .collect();
        let sims: Vec<Vec<f64>> = (0..20)
            .map(|seed| {
                (0..50)
                    .map(|i| (i as f64 * 0.1 + seed as f64 * 0.01).sin() * 5.0 + 10.0)
                    .collect()
            })
            .collect();
        let result = crate::warm::WarmResult::new(sims, vec![(1, 0); 1], vec![false; 1]);
        let bounds = FilterBounds::default().with_n_select(3);
        let pool = filter_warm_pool(&observed, &result, &bounds).unwrap();
        assert!(pool.n_selected() <= 3);
        assert!(pool.n_selected() > 0);
    }

    #[test]
    fn filter_warm_pool_respects_n_select() {
        let observed: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let sims: Vec<Vec<f64>> = (0..50)
            .map(|seed| (0..30).map(|i| i as f64 + (seed as f64) * 0.001).collect())
            .collect();
        let result = crate::warm::WarmResult::new(sims, vec![(1, 0)], vec![false]);
        let bounds = FilterBounds::default().with_n_select(3);
        let pool = filter_warm_pool(&observed, &result, &bounds).unwrap();
        assert!(pool.n_selected() <= 3);
    }

    #[test]
    fn filter_warm_pool_relaxation_fallback() {
        // Extremely tight bounds that nothing can pass
        let observed: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let sims: Vec<Vec<f64>> = (0..10)
            .map(|seed| (0..30).map(|i| i as f64 * 2.0 + seed as f64).collect())
            .collect();
        let result = crate::warm::WarmResult::new(sims, vec![(1, 0)], vec![false]);
        let bounds = FilterBounds::default()
            .with_mean_tol(0.0001)
            .with_sd_tol(0.0001)
            .with_n_select(3);
        // Should still return results via fallback
        let pool = filter_warm_pool(&observed, &result, &bounds).unwrap();
        assert!(pool.n_selected() > 0);
    }
}
