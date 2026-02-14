use std::path::PathBuf;

use serde::Deserialize;

/// Top-level Zeus configuration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZeusConfig {
    /// Global RNG seed.
    #[serde(default)]
    pub seed: Option<u64>,

    /// I/O settings.
    #[serde(default)]
    pub io: IoConfig,

    /// WARM simulation settings.
    #[serde(default)]
    pub warm: WarmToml,

    /// Filter settings.
    #[serde(default)]
    pub filter: FilterToml,

    /// Resample settings.
    #[serde(default)]
    pub resample: ResampleToml,

    /// Markov chain settings.
    #[serde(default)]
    pub markov: MarkovToml,

    /// Evaluate settings.
    #[serde(default)]
    pub evaluate: EvaluateToml,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct IoConfig {
    pub input: Option<PathBuf>,
    pub output: Option<PathBuf>,
    #[serde(default = "default_precip_var")]
    pub precip_var: String,
    #[serde(default = "default_temp_max_var")]
    pub temp_max_var: Option<String>,
    #[serde(default = "default_temp_min_var")]
    pub temp_min_var: Option<String>,
    #[serde(default = "default_start_month")]
    pub start_month: u8,
    #[serde(default = "default_true")]
    pub trim_to_water_years: bool,
    #[serde(default = "default_compression")]
    pub compression: String,
    #[serde(default = "default_row_group_size")]
    pub row_group_size: usize,
}

fn default_precip_var() -> String {
    "pr".to_string()
}
fn default_temp_max_var() -> Option<String> {
    Some("tasmax".to_string())
}
fn default_temp_min_var() -> Option<String> {
    Some("tasmin".to_string())
}
fn default_start_month() -> u8 {
    10
}
fn default_true() -> bool {
    true
}
fn default_compression() -> String {
    "snappy".to_string()
}
fn default_row_group_size() -> usize {
    1_000_000
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarmToml {
    #[serde(default = "default_n_sim")]
    pub n_sim: usize,
    #[serde(default = "default_n_years")]
    pub n_years: usize,
    #[serde(default = "default_wavelet_filter")]
    pub wavelet_filter: String,
    #[serde(default)]
    pub mra_levels: Option<usize>,
    #[serde(default = "default_bypass_n")]
    pub bypass_n: usize,
    #[serde(default = "default_max_arma_order")]
    pub max_arma_order: [usize; 2],
    #[serde(default = "default_true")]
    pub match_variance: bool,
    #[serde(default = "default_var_tol")]
    pub var_tol: f64,
}

impl Default for WarmToml {
    fn default() -> Self {
        Self {
            n_sim: default_n_sim(),
            n_years: default_n_years(),
            wavelet_filter: default_wavelet_filter(),
            mra_levels: None,
            bypass_n: default_bypass_n(),
            max_arma_order: default_max_arma_order(),
            match_variance: true,
            var_tol: default_var_tol(),
        }
    }
}

fn default_n_sim() -> usize {
    1000
}
fn default_n_years() -> usize {
    50
}
fn default_wavelet_filter() -> String {
    "la8".to_string()
}
fn default_bypass_n() -> usize {
    30
}
fn default_max_arma_order() -> [usize; 2] {
    [5, 3]
}
fn default_var_tol() -> f64 {
    0.1
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilterToml {
    #[serde(default = "default_n_select")]
    pub n_select: usize,
    #[serde(default = "default_tol")]
    pub mean_tol: f64,
    #[serde(default = "default_tol")]
    pub sd_tol: f64,
    #[serde(default = "default_tail_low_p")]
    pub tail_low_p: f64,
    #[serde(default = "default_tail_high_p")]
    pub tail_high_p: f64,
    #[serde(default = "default_spectral_corr_min")]
    pub spectral_corr_min: f64,
    #[serde(default = "default_peak_match_frac_min")]
    pub peak_match_frac_min: f64,
    #[serde(default = "default_n_sig_peaks_max")]
    pub n_sig_peaks_max: usize,
}

impl Default for FilterToml {
    fn default() -> Self {
        Self {
            n_select: default_n_select(),
            mean_tol: default_tol(),
            sd_tol: default_tol(),
            tail_low_p: default_tail_low_p(),
            tail_high_p: default_tail_high_p(),
            spectral_corr_min: default_spectral_corr_min(),
            peak_match_frac_min: default_peak_match_frac_min(),
            n_sig_peaks_max: default_n_sig_peaks_max(),
        }
    }
}

fn default_n_select() -> usize {
    5
}
fn default_tol() -> f64 {
    0.03
}
fn default_tail_low_p() -> f64 {
    0.20
}
fn default_tail_high_p() -> f64 {
    0.80
}
fn default_spectral_corr_min() -> f64 {
    0.60
}
fn default_peak_match_frac_min() -> f64 {
    1.0
}
fn default_n_sig_peaks_max() -> usize {
    2
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResampleToml {
    #[serde(default = "default_annual_knn_n")]
    pub annual_knn_n: usize,
    #[serde(default = "default_precip_weight")]
    pub precip_weight: f64,
    #[serde(default = "default_temp_weight")]
    pub temp_weight: f64,
    #[serde(default = "default_sd_floor")]
    pub sd_floor: f64,
    #[serde(default = "default_narrow_window")]
    pub narrow_window: usize,
    #[serde(default = "default_wide_window")]
    pub wide_window: usize,
    #[serde(default = "default_sampling")]
    pub sampling: String,
}

impl Default for ResampleToml {
    fn default() -> Self {
        Self {
            annual_knn_n: default_annual_knn_n(),
            precip_weight: default_precip_weight(),
            temp_weight: default_temp_weight(),
            sd_floor: default_sd_floor(),
            narrow_window: default_narrow_window(),
            wide_window: default_wide_window(),
            sampling: default_sampling(),
        }
    }
}

fn default_annual_knn_n() -> usize {
    100
}
fn default_precip_weight() -> f64 {
    100.0
}
fn default_temp_weight() -> f64 {
    10.0
}
fn default_sd_floor() -> f64 {
    0.1
}
fn default_narrow_window() -> usize {
    3
}
fn default_wide_window() -> usize {
    30
}
fn default_sampling() -> String {
    "rank".to_string()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkovToml {
    #[serde(default = "default_dirichlet_alpha")]
    pub dirichlet_alpha: f64,
    #[serde(default)]
    pub dry_spell_factors: Option<[f64; 12]>,
    #[serde(default)]
    pub wet_spell_factors: Option<[f64; 12]>,
    #[serde(default)]
    pub wet_threshold: ThresholdToml,
    #[serde(default)]
    pub extreme_threshold: ThresholdToml,
}

impl Default for MarkovToml {
    fn default() -> Self {
        Self {
            dirichlet_alpha: default_dirichlet_alpha(),
            dry_spell_factors: None,
            wet_spell_factors: None,
            wet_threshold: ThresholdToml::default(),
            extreme_threshold: ThresholdToml {
                fixed: None,
                quantile: Some(0.8),
            },
        }
    }
}

fn default_dirichlet_alpha() -> f64 {
    1.0
}

/// Threshold specification — exactly one of `fixed` or `quantile` should be set.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdToml {
    pub fixed: Option<f64>,
    pub quantile: Option<f64>,
}

impl Default for ThresholdToml {
    fn default() -> Self {
        Self {
            fixed: Some(0.3),
            quantile: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerturbToml {
    #[serde(default)]
    pub precip_floor: f64,
    #[serde(default = "default_precip_cap")]
    pub precip_cap: f64,
    #[serde(default)]
    pub temperature: Option<TempPerturbToml>,
    #[serde(default)]
    pub quantile_map: Option<QmPerturbToml>,
    #[serde(default)]
    pub occurrence: Option<OccurrencePerturbToml>,
}

fn default_precip_cap() -> f64 {
    500.0
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TempPerturbToml {
    pub deltas: [f64; 12],
    #[serde(default)]
    pub transient: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QmPerturbToml {
    pub mean_factors: [f64; 12],
    pub var_factors: [f64; 12],
    #[serde(default)]
    pub intensity_threshold: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OccurrencePerturbToml {
    pub factors: [f64; 12],
    #[serde(default)]
    pub transient: bool,
    #[serde(default = "default_occurrence_intensity")]
    pub intensity_threshold: f64,
}

fn default_occurrence_intensity() -> f64 {
    0.3
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluateToml {
    #[serde(default = "default_precip_threshold")]
    pub precip_threshold: f64,
}

impl Default for EvaluateToml {
    fn default() -> Self {
        Self {
            precip_threshold: default_precip_threshold(),
        }
    }
}

fn default_precip_threshold() -> f64 {
    0.01
}
